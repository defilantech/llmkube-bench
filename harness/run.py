"""Async load generator for the LLMKube runtime bake-off.

Sends concurrent OpenAI-style streaming chat completions to an endpoint and
records per-request timing to JSONL. A separate summary pass produces p50 /
p95 / p99 TTFT and ITL plus aggregate throughput.

Usage:
    python -m harness.run \\
        --endpoint http://localhost:8080/v1/chat/completions \\
        --pattern chat \\
        --concurrency 16 \\
        --duration 5m \\
        --warmup 2m \\
        --output results/run/chat-c16.jsonl \\
        --runtime llamacpp

    python -m harness.run summarize results/run/chat-c16.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import httpx

from harness.patterns import PromptRecord, load_pattern

# --- Duration parsing -------------------------------------------------------

_DURATION_RE = re.compile(r"^(?P<num>\d+)(?P<unit>s|m|h)?$")


def parse_duration(s: str) -> float:
    m = _DURATION_RE.match(s.strip())
    if not m:
        raise argparse.ArgumentTypeError(f"invalid duration: {s!r}")
    n = int(m.group("num"))
    unit = m.group("unit") or "s"
    return n * {"s": 1, "m": 60, "h": 3600}[unit]


# --- Per-request record -----------------------------------------------------


@dataclass
class Sample:
    ts: float
    pattern: str
    concurrency: int
    runtime: str
    prompt_id: str
    prefix_id: str | None
    ok: bool
    prompt_tokens: int
    gen_tokens: int
    ttft_ms: float
    itl_ms: list[float]
    total_ms: float
    err: str | None = None


# --- Streaming request ------------------------------------------------------


async def one_request(
    client: httpx.AsyncClient,
    url: str,
    prompt: PromptRecord,
    pattern: str,
    concurrency: int,
    runtime: str,
) -> Sample:
    body = {
        "model": "bench",  # llama.cpp and vLLM both accept any string here
        "messages": prompt.messages,
        "max_tokens": prompt.max_tokens,
        "stream": True,
        "temperature": 0.0,
        "seed": 42,
        # vLLM honors this to return token usage in the final chunk:
        "stream_options": {"include_usage": True},
        # Qwen3.5 is a thinking model. Disable the reasoning phase so TTFT
        # measures time-to-first-output-token, not time-to-end-of-thinking.
        # Both runtimes forward chat_template_kwargs into the Qwen chat template.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.perf_counter()
    ttft_ms: float | None = None
    itl: list[float] = []
    last_token_ts: float | None = None
    prompt_tokens = 0
    gen_tokens = 0
    err: str | None = None

    try:
        async with client.stream("POST", url, json=body, timeout=httpx.Timeout(300.0)) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    obj = json.loads(payload)
                except json.JSONDecodeError:
                    continue

                usage = obj.get("usage")
                if usage:
                    prompt_tokens = usage.get("prompt_tokens", prompt_tokens)
                    gen_tokens = usage.get("completion_tokens", gen_tokens)

                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta") or {}
                # Count either a visible content token or a reasoning_content
                # token as "first output", so thinking models that stream
                # reasoning first don't skew TTFT. When chat_template_kwargs
                # disables thinking, reasoning_content never arrives and this
                # falls back to plain content.
                content = delta.get("content") or delta.get("reasoning_content")
                if content is None or content == "":
                    continue

                now = time.perf_counter()
                if ttft_ms is None:
                    ttft_ms = (now - t0) * 1000.0
                else:
                    itl.append((now - last_token_ts) * 1000.0)
                last_token_ts = now

    except Exception as exc:  # noqa: BLE001 — record and continue
        err = f"{type(exc).__name__}: {exc}"[:200]

    total_ms = (time.perf_counter() - t0) * 1000.0
    ok = err is None and ttft_ms is not None

    # Fallback: if the server didn't send a usage chunk, count from ITL length.
    if gen_tokens == 0 and itl:
        gen_tokens = len(itl) + 1  # +1 for the first token (no ITL for it)

    return Sample(
        ts=time.time(),
        pattern=pattern,
        concurrency=concurrency,
        runtime=runtime,
        prompt_id=prompt.id,
        prefix_id=prompt.prefix_id,
        ok=ok,
        prompt_tokens=prompt_tokens,
        gen_tokens=gen_tokens,
        ttft_ms=ttft_ms or -1.0,
        itl_ms=itl,
        total_ms=total_ms,
        err=err,
    )


# --- Orchestrator -----------------------------------------------------------


async def worker(
    client: httpx.AsyncClient,
    url: str,
    prompts: list[PromptRecord],
    pattern: str,
    concurrency: int,
    runtime: str,
    deadline: float,
    record: list[Sample],
    rng: random.Random,
) -> None:
    while time.perf_counter() < deadline:
        prompt = rng.choice(prompts)
        sample = await one_request(client, url, prompt, pattern, concurrency, runtime)
        record.append(sample)


async def run(
    endpoint: str,
    pattern: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
    runtime: str,
    output: Path,
) -> None:
    prompts = load_pattern(pattern)

    # No cert verification against the port-forwarded localhost; bench endpoint.
    limits = httpx.Limits(max_connections=concurrency + 8, max_keepalive_connections=concurrency + 8)
    async with httpx.AsyncClient(limits=limits) as client:
        rng = random.Random(42)

        # Warmup — discarded.
        if warmup_s > 0:
            print(f"[warmup] {warmup_s:.0f}s at concurrency={concurrency}", flush=True)
            throwaway: list[Sample] = []
            deadline = time.perf_counter() + warmup_s
            await asyncio.gather(*(
                worker(client, endpoint, prompts, pattern, concurrency, runtime, deadline, throwaway, rng)
                for _ in range(concurrency)
            ))

        # Measurement.
        print(f"[measure] {duration_s:.0f}s at concurrency={concurrency}", flush=True)
        samples: list[Sample] = []
        deadline = time.perf_counter() + duration_s
        await asyncio.gather(*(
            worker(client, endpoint, prompts, pattern, concurrency, runtime, deadline, samples, rng)
            for _ in range(concurrency)
        ))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for s in samples:
            f.write(json.dumps(asdict(s)) + "\n")

    print(f"[done] wrote {len(samples)} samples to {output}", flush=True)


# --- Summarize --------------------------------------------------------------


def _percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = min(f + 1, len(xs) - 1)
    if f == c:
        return xs[f]
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def summarize(path: Path) -> dict[str, float | str | int]:
    samples: list[Sample] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            samples.append(Sample(**obj))

    if not samples:
        return {"path": str(path), "note": "no samples"}

    ok = [s for s in samples if s.ok]
    total_gen = sum(s.gen_tokens for s in ok)
    total_prompt = sum(s.prompt_tokens for s in ok)

    if ok:
        ts_min = min(s.ts for s in ok)
        ts_max = max(s.ts for s in ok)
        wall_s = max(ts_max - ts_min, 1e-6)
    else:
        wall_s = 1e-6

    ttfts = [s.ttft_ms for s in ok if s.ttft_ms > 0]
    itls = [x for s in ok for x in s.itl_ms]

    return {
        "pattern": ok[0].pattern if ok else samples[0].pattern,
        "concurrency": ok[0].concurrency if ok else samples[0].concurrency,
        "runtime": ok[0].runtime if ok else samples[0].runtime,
        "requests_total": len(samples),
        "requests_ok": len(ok),
        "success_rate": len(ok) / len(samples) if samples else 0.0,
        "prompt_tokens_total": total_prompt,
        "gen_tokens_total": total_gen,
        "wall_s": wall_s,
        "throughput_gen_tps": total_gen / wall_s,
        "ttft_p50_ms": _percentile(ttfts, 0.50),
        "ttft_p95_ms": _percentile(ttfts, 0.95),
        "ttft_p99_ms": _percentile(ttfts, 0.99),
        "itl_p50_ms": _percentile(itls, 0.50),
        "itl_p95_ms": _percentile(itls, 0.95),
        "itl_mean_ms": statistics.fmean(itls) if itls else float("nan"),
    }


# --- CLI --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="run a measurement cell")
    p_run.add_argument("--endpoint", required=True)
    p_run.add_argument("--pattern", required=True)
    p_run.add_argument("--concurrency", type=int, required=True)
    p_run.add_argument("--duration", type=parse_duration, default=parse_duration("5m"))
    p_run.add_argument("--warmup", type=parse_duration, default=parse_duration("2m"))
    p_run.add_argument("--runtime", required=True, choices=["llamacpp", "vllm", "exllamav3"])
    p_run.add_argument("--output", type=Path, required=True)

    p_sum = sub.add_parser("summarize", help="print summary JSON for a JSONL file")
    p_sum.add_argument("path", type=Path)

    args = parser.parse_args(argv)

    if args.cmd == "run":
        asyncio.run(run(
            endpoint=args.endpoint,
            pattern=args.pattern,
            concurrency=args.concurrency,
            duration_s=args.duration,
            warmup_s=args.warmup,
            runtime=args.runtime,
            output=args.output,
        ))
        return 0

    if args.cmd == "summarize":
        result = summarize(args.path)
        print(json.dumps(result, indent=2))
        return 0

    return 2


if __name__ == "__main__":
    sys.exit(main())
