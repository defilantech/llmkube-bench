"""Greedy-decode agreement: how much two engines' output diverges on the same prefix, without
either one needing to expose prompt logprobs.

harness.quality.logprobs needs an OpenAI-compatible server that returns `prompt_logprobs` (vLLM
does; llama.cpp's `/v1/completions` does not), so it cannot compare llama.cpp against anything.
Agreement is the fallback that works against any server: send the same prefix to both engines at
temperature 0, and measure how long their greedy continuations stay byte-for-byte identical. It is
a cruder signal than KL divergence (it only sees the argmax token, not the distribution behind it),
but it needs nothing from the server beyond a completion.

Calibration. There is no absolute pass/fail threshold for agreement on its own: two different
engines running the *same* weights at the *same* precision still diverge eventually, because
floating point reduction order differs across kernels. The number that matters is comparative:
capture reference-vs-reference agreement first (e.g. two runs of today's deployed engine+quant, or
the same engine on two identical requests) to see the floor noise looks like, then judge a
candidate engine/quant by whether its agreement with the reference is close to that floor or far
below it. See METHOD.md for the worked calibration procedure.

Servers differ on whether a completions response includes token ids alongside the text (vLLM does
via `logprobs`/token round-tripping quirks that vary by version; llama.cpp's plain
`/v1/completions` does not). `capture` sidesteps this by re-tokenizing the returned *text* with the
one tokenizer supplied on the command line, so both sides are compared as token ids in the same
vocabulary regardless of what either server chose to report. This means a difference in
tokenization between the *serving* engine and the tokenizer used here would show up as spurious
disagreement; use the model's own tokenizer.json for both captures being compared.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import httpx

from harness.quality._resume import truncate_to


def first_divergence(a: list[int], b: list[int]) -> int:
    """Length of the common prefix of two token-id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def summarize(a: dict[str, list[int]], b: dict[str, list[int]]) -> dict:
    """Agreement stats over the ids present in both a and b (ids unique to one side are ignored)."""
    shared_ids = sorted(id_ for id_ in a if id_ in b)
    if not shared_ids:
        raise ValueError("a and b have no ids in common; nothing to compare")
    divergences: list[int] = []
    exact_matches = 0
    matched_positions = 0
    total_positions = 0
    for id_ in shared_ids:
        ta, tb = a[id_], b[id_]
        divergences.append(first_divergence(ta, tb))
        if ta == tb:
            exact_matches += 1
        n = min(len(ta), len(tb))
        total_positions += n
        matched_positions += sum(1 for i in range(n) if ta[i] == tb[i])
    items = len(shared_ids)
    return {
        "items": items,
        "exact_match_rate": exact_matches / items,
        "mean_first_divergence": sum(divergences) / items,
        "median_first_divergence": statistics.median(divergences),
        "token_agreement": matched_positions / total_positions,
    }


def _read_done_ids(output_path: Path) -> set[str]:
    """Reads ids already captured in output_path, truncating a crash-torn trailing line first.

    Mirrors harness.quality.logprobs._read_existing_capture: if the process was killed mid-write
    and the file ends in a truncated, unparseable line, that line is dropped (the file is
    rewritten via a temp file + os.replace, never truncated in place) and its item is treated as
    not done, so resume redoes it instead of crash-looping on the same bad line or, worse,
    concatenating the next write onto it. A malformed line anywhere but the end is not forgiven:
    that is data corruption, not a crash artifact, and raises ValueError naming the file and line.
    """
    if not output_path.exists() or output_path.stat().st_size == 0:
        return set()
    done_ids: set[str] = set()
    with output_path.open("rb") as fh:
        good_end = 0
        line_no = 0
        line_bytes = fh.readline()
        while line_bytes:
            line_no += 1
            stripped = line_bytes.strip()
            if not stripped:
                good_end = fh.tell()
                line_bytes = fh.readline()
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                remainder = fh.readline()
                if remainder.strip():
                    raise ValueError(f"{output_path}: invalid JSON at line {line_no}") from None
                # The broken line is the last thing in the file: a capture that was killed
                # mid-write leaves exactly this shape. Drop it and let its item be redone.
                truncate_to(output_path, good_end)
                return done_ids
            done_ids.add(rec["id"])
            good_end = fh.tell()
            line_bytes = fh.readline()
    return done_ids


def capture(client, endpoint, model, corpus_path, tokenizer, output_path, prefix_tokens: int,
           gen_tokens: int) -> None:
    """Resumable: an id already present in output_path is not re-requested."""
    corpus_path, output_path = Path(corpus_path), Path(output_path)
    done_ids = _read_done_ids(output_path)
    mode = "a" if output_path.exists() else "w"
    with output_path.open(mode) as out, corpus_path.open() as cf:
        for line in cf:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["id"] in done_ids:
                continue
            prefix_ids = rec["token_ids"][:prefix_tokens]
            prefix_text = tokenizer.decode(prefix_ids)
            prefix_sha = hashlib.sha256(prefix_text.encode("utf-8")).hexdigest()
            body = {"model": model, "prompt": prefix_text, "max_tokens": gen_tokens,
                   "temperature": 0.0, "top_k": 1}
            resp = client.post(endpoint.rstrip("/") + "/v1/completions", json=body, timeout=None)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["text"]
            tokens = tokenizer.encode(text).ids
            out.write(json.dumps({"id": rec["id"], "prefix_sha": prefix_sha, "tokens": tokens}) + "\n")
            out.flush()


def _read_capture(path: Path) -> tuple[dict[str, list[int]], dict[str, str]]:
    path = Path(path)
    tokens: dict[str, list[int]] = {}
    prefix_sha: dict[str, str] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}: invalid JSON at line {line_no}") from e
        tokens[rec["id"]] = rec["tokens"]
        prefix_sha[rec["id"]] = rec["prefix_sha"]
    return tokens, prefix_sha


def compare_files(a_path, b_path) -> dict:
    a_tokens, a_sha = _read_capture(a_path)
    b_tokens, b_sha = _read_capture(b_path)
    for id_ in a_tokens:
        if id_ in b_tokens and a_sha[id_] != b_sha[id_]:
            raise ValueError(
                f"prefix mismatch for item {id_!r}: the two captures were built from different "
                "prefixes (different corpus or --prefix-tokens); their generations are not "
                "comparable"
            )
    return summarize(a_tokens, b_tokens)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.quality.agreement")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("capture")
    c.add_argument("--endpoint", required=True)
    c.add_argument("--model", required=True)
    c.add_argument("--corpus", required=True, type=Path)
    c.add_argument("--tokenizer", required=True)
    c.add_argument("--prefix-tokens", type=int, default=1024)
    c.add_argument("--gen-tokens", type=int, default=64)
    c.add_argument("--output", required=True, type=Path)
    m = sub.add_parser("compare")
    m.add_argument("a", type=Path)
    m.add_argument("b", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "capture":
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(a.tokenizer)
        with httpx.Client() as client:
            capture(client, a.endpoint, a.model, a.corpus, tok, a.output, a.prefix_tokens, a.gen_tokens)
    else:
        print(json.dumps(compare_files(a.a, a.b), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
