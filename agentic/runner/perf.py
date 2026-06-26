import json, time, urllib.request, statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional
from concurrent.futures import ThreadPoolExecutor

PROMPT = ("Write a Go function `mergeSorted(a, b []int) []int` that merges two sorted int "
          "slices into one sorted slice, with a short doc comment. Then explain the time "
          "complexity in two sentences.")


@dataclass
class RequestResult:
    ok: bool
    ttft: float = 0.0
    total: float = 0.0
    toks: int = 0
    tok_s: float = 0.0
    err: Optional[str] = None


@dataclass
class LevelStats:
    concurrency: int
    ok: int
    failed: int
    agg_tok_s: float
    mean_per_req_tok_s: float
    mean_ttft: float
    p50_latency: float
    p95_latency: float


def http_request(base: str, model: str, token: str, max_tokens: int) -> RequestResult:
    body = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": max_tokens, "temperature": 0.2,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = "Bearer " + token
    req = urllib.request.Request(base + "/v1/chat/completions", data=body, headers=headers)
    t0 = time.perf_counter()
    ttft, ctoks = None, 0
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except ValueError:
                    continue
                ch = obj.get("choices") or []
                if ch and ch[0].get("delta", {}).get("content") and ttft is None:
                    ttft = time.perf_counter() - t0
                if obj.get("usage"):
                    ctoks = obj["usage"].get("completion_tokens", 0)
    except Exception as e:                          # noqa: BLE001 - any transport error is a failed req
        return RequestResult(ok=False, err=str(e)[:80])
    total = time.perf_counter() - t0
    if ctoks == 0:
        return RequestResult(ok=False, err="no usage/tokens")
    return RequestResult(ok=True, ttft=ttft or total, total=total, toks=ctoks, tok_s=ctoks / total)


def _pct(sorted_xs: List[float], q: float) -> float:
    return sorted_xs[min(len(sorted_xs) - 1, int(len(sorted_xs) * q))]


def aggregate(results: List[RequestResult], wall: float) -> LevelStats:
    ok = [r for r in results if r.ok]
    failed = len(results) - len(ok)
    if not ok:
        return LevelStats(0, 0, failed, 0, 0, 0, 0, 0)
    lat = sorted(r.total for r in ok)
    toks = sum(r.toks for r in ok)
    return LevelStats(
        concurrency=0, ok=len(ok), failed=failed,
        agg_tok_s=toks / wall if wall else 0.0,
        mean_per_req_tok_s=statistics.mean(r.tok_s for r in ok),
        mean_ttft=statistics.mean(r.ttft for r in ok),
        p50_latency=_pct(lat, 0.5), p95_latency=_pct(lat, 0.95),
    )


def run_sweep(base: str, model: str, token: str, concurrencies: List[int],
              reqs_per_level: int, max_tokens: int,
              call: Callable[..., RequestResult] = http_request) -> List[LevelStats]:
    # warmup
    for _ in range(2):
        call(base, model, token, max_tokens)
    out: List[LevelStats] = []
    for c in concurrencies:
        with ThreadPoolExecutor(max_workers=c) as ex:
            t0 = time.perf_counter()
            results = list(ex.map(lambda _: call(base, model, token, max_tokens), range(reqs_per_level)))
            wall = time.perf_counter() - t0
        stats = aggregate(results, wall)
        stats.concurrency = c
        out.append(stats)
    return out
