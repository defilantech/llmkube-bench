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

capture() writes newline-delimited JSON directly to disk, one item at a time,
so it never holds the whole run in memory and can be killed and resumed: the
first line is a header naming the model, endpoint, k, and the corpus file's
sha256; each following line is one corpus item's positions. Re-running
capture() against an output file with a matching header skips ids already
written and appends the rest. A mismatched header (different k or corpus) is
refused rather than silently overwritten.

compare() reads two such files in lockstep, one item at a time, rather than
loading either fully into memory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import httpx


def _score(client, endpoint, model, ids, k):
    body = {"model": model, "prompt": ids, "max_tokens": 1, "temperature": 0.0, "prompt_logprobs": k}
    resp = client.post(endpoint.rstrip("/") + "/v1/completions", json=body, timeout=None)
    resp.raise_for_status()
    return resp.json()["choices"][0]["prompt_logprobs"]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_header_line(fh, path) -> dict:
    line = fh.readline()
    if not line:
        raise ValueError(f"{path} is empty; not a logprobs capture file")
    header = json.loads(line)
    if header.get("kind") != "logprobs-capture":
        raise ValueError(f"{path} does not look like a logprobs capture file (missing/invalid 'kind')")
    return header


def _read_existing_capture(output_path: Path, k: int, corpus_sha256: str) -> set[str]:
    done_ids: set[str] = set()
    with output_path.open() as fh:
        header = _read_header_line(fh, output_path)
        if header.get("k") != k or header.get("corpus_sha256") != corpus_sha256:
            raise ValueError(
                f"{output_path} already exists with a mismatched header "
                f"(k={header.get('k')!r} vs {k!r}, corpus_sha256={header.get('corpus_sha256')!r} "
                f"vs {corpus_sha256!r}); refusing to overwrite"
            )
        for line in fh:
            line = line.strip()
            if line:
                done_ids.add(json.loads(line)["id"])
    return done_ids


def capture(client, endpoint, model, corpus_path, output_path, k=20) -> None:
    corpus_path, output_path = Path(corpus_path), Path(output_path)
    corpus_sha256 = _sha256_file(corpus_path)
    if output_path.exists() and output_path.stat().st_size > 0:
        done_ids = _read_existing_capture(output_path, k, corpus_sha256)
        out = output_path.open("a")
    else:
        done_ids = set()
        out = output_path.open("w")
        out.write(json.dumps({"kind": "logprobs-capture", "model": model, "endpoint": endpoint,
                               "k": k, "corpus_sha256": corpus_sha256}) + "\n")
        out.flush()
    try:
        with corpus_path.open() as cf:
            for line in cf:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec["id"] in done_ids:
                    continue
                ids = rec["token_ids"]
                entries = _score(client, endpoint, model, ids, k)
                positions = []
                for pos in range(1, len(ids)):
                    entry = entries[pos]
                    actual = ids[pos]
                    top = {tid: v["logprob"] for tid, v in entry.items() if v.get("rank", k + 1) <= k}
                    positions.append({"actual": actual, "actual_lp": entry[str(actual)]["logprob"], "top": top})
                out.write(json.dumps({"id": rec["id"], "positions": positions}) + "\n")
                out.flush()
    finally:
        out.close()


def _kl(p_lp: dict, q_lp: dict) -> float:
    support = list(p_lp)
    p = [math.exp(p_lp[t]) for t in support]
    floor = min(q_lp.values()) if q_lp else -30.0
    q = [math.exp(q_lp.get(t, floor)) for t in support]
    ps, qs = sum(p), sum(q)
    return sum((pi / ps) * math.log((pi / ps) / (qi / qs)) for pi, qi in zip(p, q) if pi > 0)


def compare(reference_path, candidate_path) -> dict:
    reference_path, candidate_path = Path(reference_path), Path(candidate_path)
    with reference_path.open() as rf, candidate_path.open() as cf:
        rh = _read_header_line(rf, reference_path)
        ch = _read_header_line(cf, candidate_path)
        if rh["k"] != ch["k"]:
            raise ValueError(f"reference and candidate were captured with different k ({rh['k']} vs {ch['k']})")
        if rh["corpus_sha256"] != ch["corpus_sha256"]:
            raise ValueError("reference and candidate were captured on different corpora (corpus_sha256 mismatch)")
        kls: list[float] = []
        nll_r: list[float] = []
        nll_c: list[float] = []
        for r_line, c_line in itertools.zip_longest(rf, cf):
            if r_line is None or c_line is None:
                raise ValueError("reference and candidate capture files have a different number of items")
            r_rec, c_rec = json.loads(r_line), json.loads(c_line)
            if r_rec["id"] != c_rec["id"]:
                raise ValueError(f"reference and candidate id order differs: {r_rec['id']!r} vs {c_rec['id']!r}")
            r_pos, c_pos = r_rec["positions"], c_rec["positions"]
            if len(r_pos) != len(c_pos):
                raise ValueError(f"corpus item {r_rec['id']} has a different number of positions in reference vs candidate")
            for i, (r, c) in enumerate(zip(r_pos, c_pos)):
                if r["actual"] != c["actual"]:
                    raise ValueError(
                        f"actual token mismatch for item {r_rec['id']} position {i}: "
                        f"reference={r['actual']} candidate={c['actual']}"
                    )
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
            capture(client, a.endpoint, a.model, a.corpus, a.output, a.k)
    else:
        print(json.dumps(compare(a.reference, a.candidate), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
