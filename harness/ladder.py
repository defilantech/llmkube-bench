"""Prefill ladder and low-concurrency decode for agentic single-stream serving.

Every rung and every decode run opens with a warmup request (this ring has a
hidden slow state after ~13 minutes idle). Cold prompts start with a unique
nonce so they cannot hit the prefix cache; cached prompts send the same text
twice and time the second. Rates use the server's prompt_tokens.
"""
from __future__ import annotations

import argparse
import json
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
        ids.extend(corpus_ids[: target_tokens - len(ids)])
    return tokenizer.decode(ids[:target_tokens])


def _warmup(client, endpoint, model):
    oai.chat(client, endpoint, model, [{"role": "user", "content": f"warmup {uuid.uuid4().hex}"}], max_tokens=2)


def _prefill(client, endpoint, model, text):
    r = oai.chat(client, endpoint, model, [{"role": "user", "content": text}], max_tokens=1)
    return r


def run_ladder(client, endpoint, model, tokenizer, corpus, sizes, repeats, decode_prompt_tokens,
               decode_max_tokens, concurrencies) -> dict:
    prefill = []
    for size in sizes:
        _warmup(client, endpoint, model)
        for i in range(repeats):
            cold_text = build_prompt(tokenizer, size, f"cold-{uuid.uuid4().hex}", corpus)
            r = _prefill(client, endpoint, model, cold_text)
            prefill.append({"target": size, "mode": "cold", "repeat": i, "prompt_tokens": r.prompt_tokens,
                            "ttft_s": r.ttft_s, "tok_s": (r.prompt_tokens or size) / r.ttft_s})
            cached_text = build_prompt(tokenizer, size, f"cached-{uuid.uuid4().hex}", corpus)
            _prefill(client, endpoint, model, cached_text)
            r = _prefill(client, endpoint, model, cached_text)
            prefill.append({"target": size, "mode": "cached", "repeat": i, "prompt_tokens": r.prompt_tokens,
                            "ttft_s": r.ttft_s, "tok_s": (r.prompt_tokens or size) / r.ttft_s})
    decode = []
    for conc in concurrencies:
        _warmup(client, endpoint, model)
        texts = [build_prompt(tokenizer, decode_prompt_tokens, f"decode-{uuid.uuid4().hex}", corpus)
                 for _ in range(conc)]

        def one(text):
            return oai.chat(client, endpoint, model, [{"role": "user", "content": text}],
                            max_tokens=decode_max_tokens, extra={"ignore_eos": True})

        with ThreadPoolExecutor(max_workers=conc) as pool:
            results = list(pool.map(one, texts))
        rates = [(r.completion_tokens or 0) / max(r.total_s - r.ttft_s, 1e-9) for r in results]
        decode.append({"concurrency": conc, "per_stream_tok_s": rates,
                       "mean_per_stream_tok_s": sum(rates) / len(rates),
                       "ttft_s": [r.ttft_s for r in results]})
    return {"prefill": prefill, "decode": decode}


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
    with httpx.Client() as client:
        out = run_ladder(client, a.endpoint, a.model, tok, a.corpus.read_text(),
                         [int(s) for s in a.sizes.split(",") if s], a.repeats, a.decode_prompt_tokens,
                         a.decode_max_tokens, [int(c) for c in a.concurrencies.split(",") if c])
    out.update({"kind": "ladder", "label": a.label, "endpoint": a.endpoint, "model": a.model,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
    a.output.write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
