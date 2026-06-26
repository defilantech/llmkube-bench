from runner.config import Config, Hardware, Model, Task, Perf
from runner.cluster import FakeCluster
from runner.perf import RequestResult
from runner.orchestrate import run_matrix


def _cfg():
    return Config(
        hardware=Hardware("strix", "shadowstack", "default", "fleet-router", 1),
        agent="ornith-coder-lever",
        models=[Model("ornith-1.0-35b", "Q4_K_M", 35, "moe", "ornith-35b", "2026-04-01")],
        corpus=[Task("clean-433", "defilantech/LLMKube", 433, "2026-06-10", "issue-fix", "clean")],
        perf=Perf(concurrencies=[1], reqs_per_level=1, max_tokens=16),
    )


def test_run_matrix_emits_one_record_per_model_task_with_distinct_branch():
    fc = FakeCluster(
        terminal_status={"verdict": "GO", "result": {"summary": "ok",
            "extra": {"branch": "foreman/bench-clean-433-ornith-1.0-35b", "modelExtra": {}}}},
        transcript={"turnCount": 5, "messages": [
            {"role": "assistant", "tool_calls": [
                {"function": {"name": "bash", "arguments": "{}"}}]}]},
    )
    records = run_matrix(
        _cfg(), cluster=fc, harness_version="0.8.17", now="2026-06-26T07:00:00Z",
        prompt_for=lambda repo, issue: "fix it",
        perf_call=lambda *a, **k: RequestResult(ok=True, ttft=0.1, total=1.0, toks=50, tok_s=50.0),
        endpoint="http://localhost:18080", token="",
    )
    assert len(records) == 1
    r = records[0]
    assert r.model == "ornith-1.0-35b" and r.task_id == "clean-433"
    assert r.verdict == "GO" and r.branch == "foreman/bench-clean-433-ornith-1.0-35b"
    assert r.tok_per_s == 50.0 and r.contamination_free is True
    assert "foreman/bench-clean-433-ornith-1.0-35b" in fc.dispatched[0] or fc.dispatched
