from typing import Callable, List
from .config import Config
from .record import RunRecord
from .behavior import extract_behavior
from .perf import run_sweep


def branch_name(task_id: str, model_name: str) -> str:
    return f"foreman/bench-{task_id}-{model_name}"


def run_matrix(cfg: Config, cluster, harness_version: str, now: str,
               prompt_for: Callable[[str, int], str],
               endpoint: str, token: str,
               perf_call=None) -> List[RunRecord]:
    isvc_all = [m.inference_service for m in cfg.models]
    backend_for = {m.inference_service: m.inference_service for m in cfg.models}
    records: List[RunRecord] = []

    for model in cfg.models:
        # GPU serialization: this model alone on the accelerator
        cluster.serve_only(model.inference_service, isvc_all=isvc_all, backend_for=backend_for)
        cluster.wait_ready(model.inference_service)

        # perf sweep (single occupant)
        pf = cluster.endpoint_port_forward(model.inference_service, 18080)
        try:
            sweep = run_sweep(endpoint, model.name, token,
                              cfg.perf.concurrencies, cfg.perf.reqs_per_level,
                              cfg.perf.max_tokens, **({"call": perf_call} if perf_call else {}))
        finally:
            pf.terminate()
            try:
                pf.wait(timeout=5)
            except Exception:
                pass
        single_stream = next((s for s in sweep if s.concurrency == cfg.perf.concurrencies[0]), None)
        tok_s = single_stream.mean_per_req_tok_s if single_stream else 0.0

        # capability tasks, each on a distinct branch
        for task in cfg.corpus:
            br = branch_name(task.id, model.name)
            name = f"bench-{task.id}-{model.name}".replace(".", "-").lower()
            t = cluster.dispatch(agent=cfg.agent, repo=task.repo, issue=task.issue,
                                 branch=br, prompt=prompt_for(task.repo, task.issue), name=name)
            status = cluster.wait_terminal(t)
            beh = extract_behavior(cluster.transcript_of(t), status)
            records.append(RunRecord(
                model=model.name, quant=model.quant, params_b=model.params_b, arch=model.arch,
                hardware_tier=cfg.hardware.tier, harness_version=harness_version,
                task_id=task.id, repo=task.repo, issue=task.issue,
                filed_date=task.filed_date, model_cutoff=model.model_cutoff, run_at=now,
                verdict=beh.verdict, gate_verified=beh.gate_verified, branch=beh.branch,
                turns=beh.turns, gate_fix_attempts=beh.gate_fix_attempts, wall_clock_s=0.0,
                tok_per_s=tok_s, tool_calls=beh.tool_calls, files_edited=beh.files_edited,
                failure_mode=beh.failure_mode, levers_on=[],
            ))
    return records
