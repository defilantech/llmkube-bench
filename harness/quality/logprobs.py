"""Perplexity and top-k KL divergence by teacher forcing through vLLM prompt_logprobs.

Both sides score the same token ids (the corpus stores ids, not text), so
there is no tokenizer or chat-template drift. vLLM returns None for position
0 and includes the actual token even when it is outside the top-k.

KL is computed over the reference's top-k set: p is the reference renormalized
over that set; q is the candidate's probability for each id, with ids missing
from the candidate's top-k given the candidate's smallest returned probability
(an upper bound), then renormalized over the same set. It is an approximation
of the full-vocabulary KL that is consistent across candidates, which is what
the gate needs.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import httpx


def _score(client, endpoint, model, ids, k):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": k}
    resp = client.post(endpoint.rstrip("/") + "/v1/completions", json=body, timeout=None)
    resp.raise_for_status()
    return resp.json()["choices"][0]["prompt_logprobs"]


def capture(client, endpoint, model, corpus_path, k=20) -> dict:
    positions = {}
    for line in Path(corpus_path).read_text().splitlines():
        rec = json.loads(line)
        ids = rec["token_ids"]
        entries = _score(client, endpoint, model, ids, k)
        out = []
        for pos in range(1, len(ids)):
            entry = entries[pos]
            actual = ids[pos]
            top = {tid: v["logprob"] for tid, v in entry.items() if v.get("rank", k + 1) <= k}
            out.append({"actual": actual, "actual_lp": entry[str(actual)]["logprob"], "top": top})
        positions[rec["id"]] = out
    return {"model": model, "endpoint": endpoint, "k": k, "positions": positions}


def _kl(p_lp: dict, q_lp: dict) -> float:
    support = list(p_lp)
    p = [math.exp(p_lp[t]) for t in support]
    floor = min(q_lp.values()) if q_lp else -30.0
    q = [math.exp(q_lp.get(t, floor)) for t in support]
    ps, qs = sum(p), sum(q)
    return sum((pi / ps) * math.log((pi / ps) / (qi / qs)) for pi, qi in zip(p, q) if pi > 0)


def compare(reference: dict, candidate: dict) -> dict:
    if set(reference["positions"]) != set(candidate["positions"]):
        raise ValueError("reference and candidate were captured on different corpus ids")
    kls, nll_r, nll_c = [], [], []
    for sid, ref_pos in reference["positions"].items():
        cand_pos = candidate["positions"][sid]
        if len(ref_pos) != len(cand_pos):
            raise ValueError(f"corpus item {sid} has different lengths")
        for r, c in zip(ref_pos, cand_pos):
            kls.append(_kl(r["top"], c["top"]))
            nll_r.append(-r["actual_lp"])
            nll_c.append(-c["actual_lp"])
    kls.sort()
    return {"n_positions": len(kls), "kld_mean": sum(kls) / len(kls), "kld_p99": kls[int(0.99 * (len(kls) - 1))],
            "ppl_ref": math.exp(sum(nll_r) / len(nll_r)), "ppl_cand": math.exp(sum(nll_c) / len(nll_c))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.quality.logprobs")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--endpoint", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--corpus", required=True, type=Path)
    c.add_argument("--k", type=int, default=20)
    c.add_argument("--output", required=True, type=Path)
    m = sub.add_parser("compare")
    m.add_argument("reference", type=Path)
    m.add_argument("candidate", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        with httpx.Client() as client:
            a.output.write_text(json.dumps(capture(client, a.endpoint, a.model, a.corpus, a.k)))
    else:
        print(json.dumps(compare(json.loads(a.reference.read_text()), json.loads(a.candidate.read_text())), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
