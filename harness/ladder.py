"""Prefill ladder and low-concurrency decode for agentic single-stream serving.

Every rung and every decode run opens with a warmup request (some deployments
are slow on the first request after idle). Cold prompts start with a unique
nonce so they cannot hit the prefix cache; cached prompts send the same text
twice and time the second. Rates use the server's prompt_tokens.

A row whose server response carried no usage block (a dropped final frame, or
a proxy stripping include_usage) is never estimated from the requested size:
it is marked invalid instead, and its rate is None.

A decode stream in which no token was actually seen is invalid the same way:
completion_tokens under 2, or a TTFT that never came in before the stream
ended (ttft_s >= total_s), means there is nothing to measure a rate from, so
it is marked invalid and its rate is None rather than a divide-by-near-zero
number. Otherwise the rate is (completion_tokens - 1) / (total_s - ttft_s),
since the first token arrives at TTFT and only the remaining tokens land in
the decode window that follows it.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from harness import oai


def build_prompt(tokenizer, target_tokens: int, nonce: str, corpus: str) -> str:
    ids: list = list(tokenizer.encode(nonce).ids)
    corpus_ids = list(tokenizer.encode(corpus).ids)
    while len(ids) < target_tokens:
        if not corpus_ids:
            raise ValueError("corpus tokenizes to zero tokens; cannot pad prompt to target_tokens")
        ids.extend(corpus_ids[: target_tokens - len(ids)])
    return tokenizer.decode(ids[:target_tokens])


def _warmup(client, endpoint, model):
    oai.chat(client, endpoint, model, [{"role": "user", "content": f"warmup {uuid.uuid4().hex}"}], max_tokens=2)


def _prefill(client, endpoint, model, text):
    r = oai.chat(client, endpoint, model, [{"role": "user", "content": text}], max_tokens=1)
    return r


def _decode_row(r) -> tuple[bool, float | None]:
    """Returns (invalid, rate) for one decode stream's ChatResult.

    A stream in which no token was actually seen is invalid: completion_tokens under 2, or a
    TTFT that never arrived before the stream ended (ttft_s >= total_s). Otherwise the rate is
    (completion_tokens - 1) / (total_s - ttft_s), since the first token lands at TTFT and only
    the remaining tokens fall in the decode window that follows it.
    """
    if r.completion_tokens is None or r.completion_tokens < 2 or r.ttft_s >= r.total_s:
        return True, None
    return False, (r.completion_tokens - 1) / (r.total_s - r.ttft_s)


def _prefill_row(target: int, mode: str, repeat: int, r) -> dict:
    valid = r.prompt_tokens is not None
    tok_s = (r.prompt_tokens / r.ttft_s) if valid else None
    target_error = ((r.prompt_tokens - target) / target) if valid else None
    return {"target": target, "mode": mode, "repeat": repeat, "prompt_tokens": r.prompt_tokens,
            "ttft_s": r.ttft_s, "tok_s": tok_s, "valid": valid, "target_error": target_error}


def run_ladder(client, endpoint, model, tokenizer, corpus, sizes, repeats, decode_prompt_tokens,
               decode_max_tokens, concurrencies, on_progress=None) -> dict:
    prefill: list = []
    decode: list = []
    for size in sizes:
        _warmup(client, endpoint, model)
        for i in range(repeats):
            cold_text = build_prompt(tokenizer, size, f"cold-{uuid.uuid4().hex}", corpus)
            r = _prefill(client, endpoint, model, cold_text)
            prefill.append(_prefill_row(size, "cold", i, r))
            cached_text = build_prompt(tokenizer, size, f"cached-{uuid.uuid4().hex}", corpus)
            _prefill(client, endpoint, model, cached_text)
            r = _prefill(client, endpoint, model, cached_text)
            prefill.append(_prefill_row(size, "cached", i, r))
        if on_progress is not None:
            on_progress({"prefill": prefill, "decode": decode})
    for conc in concurrencies:
        _warmup(client, endpoint, model)
        texts = [build_prompt(tokenizer, decode_prompt_tokens, f"decode-{uuid.uuid4().hex}", corpus)
                 for _ in range(conc)]

        def one(text):
            return oai.chat(client, endpoint, model, [{"role": "user", "content": text}],
                            max_tokens=decode_max_tokens, extra={"ignore_eos": True})

        with ThreadPoolExecutor(max_workers=conc) as pool:
            results = list(pool.map(one, texts))
        rates: list = []
        completion_tokens: list = []
        invalid_streams = 0
        for r in results:
            completion_tokens.append(r.completion_tokens)
            invalid, rate = _decode_row(r)
            rates.append(rate)
            if invalid:
                invalid_streams += 1
        valid_rates = [rate for rate in rates if rate is not None]
        mean_rate = (sum(valid_rates) / len(valid_rates)) if valid_rates else None
        decode.append({"concurrency": conc, "per_stream_tok_s": rates, "completion_tokens": completion_tokens,
                       "invalid_streams": invalid_streams, "mean_per_stream_tok_s": mean_rate,
                       "ttft_s": [r.ttft_s for r in results]})
        if on_progress is not None:
            on_progress({"prefill": prefill, "decode": decode})
    return {"prefill": prefill, "decode": decode}


def _has_invalid(out: dict) -> bool:
    if any(not row["valid"] for row in out["prefill"]):
        return True
    return any(d["invalid_streams"] > 0 for d in out["decode"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.ladder")
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--label", required=True, help="e.g. exl3-3.5bpw-tp3-baseline-run1")
    ap.add_argument("--tokenizer", required=True, help="path to the model's tokenizer.json")
    ap.add_argument("--corpus", required=True, type=Path, help="text file used to fill prompts")
    ap.add_argument("--sizes", default="10000,40000,100000,300000")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--decode-prompt-tokens", type=int, default=4000)
    ap.add_argument("--decode-max-tokens", type=int, default=512)
    ap.add_argument("--concurrencies", default="1,2")
    ap.add_argument("--output", required=True, type=Path)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)

    meta = {"kind": "ladder", "label": a.label, "endpoint": a.endpoint, "model": a.model}

    def write(state: dict, complete: bool) -> None:
        payload = dict(meta)
        payload.update(state)
        payload["complete"] = complete
        payload["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        a.output.write_text(json.dumps(payload, indent=1))

    with httpx.Client() as client:
        out = run_ladder(client, a.endpoint, a.model, tok, a.corpus.read_text(),
                         [int(s) for s in a.sizes.split(",") if s], a.repeats, a.decode_prompt_tokens,
                         a.decode_max_tokens, [int(c) for c in a.concurrencies.split(",") if c],
                         on_progress=lambda state: write(state, complete=False))

    write(out, complete=True)

    if _has_invalid(out):
        print("ladder: one or more rows/streams had no usable server usage data (see 'valid' / "
              "'invalid_streams' in the output file)", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
