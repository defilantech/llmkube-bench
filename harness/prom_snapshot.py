"""Capture Prometheus range snapshots to JSON for a measurement window.

Expects PROMETHEUS_URL env var (default http://localhost:9090). Queries a small
set of runtime + GPU metrics, emits one JSON blob per cell.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

QUERIES: dict[str, str] = {
    # GPU-side (DCGM exporter; on shadowstack this is scraped by microk8s/prometheus)
    "gpu_util": 'avg by (UUID) (DCGM_FI_DEV_GPU_UTIL)',
    "gpu_memory_used_mib": 'sum by (UUID) (DCGM_FI_DEV_FB_USED)',
    "gpu_power_w": 'sum(DCGM_FI_DEV_POWER_USAGE)',

    # vLLM-specific (emitted when runtime=vllm, ignored otherwise)
    "vllm_requests_running": 'sum(vllm:num_requests_running)',
    "vllm_requests_waiting": 'sum(vllm:num_requests_waiting)',
    "vllm_gpu_cache_usage_perc": 'avg(vllm:gpu_cache_usage_perc)',
    "vllm_prefix_cache_hit_rate_5m": (
        'sum(rate(vllm:prefix_cache_queries_total[5m])) / '
        'sum(rate(vllm:prefix_cache_hits_total[5m]))'
    ),

    # llama.cpp-specific
    "llamacpp_tokens_predicted_rate_5m": 'sum(rate(llamacpp:tokens_predicted_total[5m]))',
    "llamacpp_prompt_tokens_rate_5m": 'sum(rate(llamacpp:prompt_tokens_total[5m]))',
}


def snapshot(prom_url: str, start: float, end: float, step_s: int = 15) -> dict[str, Any]:
    out: dict[str, Any] = {
        "prom_url": prom_url,
        "start": start,
        "end": end,
        "step_s": step_s,
        "series": {},
    }
    with httpx.Client(timeout=30.0) as client:
        for name, query in QUERIES.items():
            try:
                resp = client.get(
                    f"{prom_url}/api/v1/query_range",
                    params={"query": query, "start": start, "end": end, "step": f"{step_s}s"},
                )
                resp.raise_for_status()
                data = resp.json().get("data", {})
                out["series"][name] = data.get("result", [])
            except Exception as exc:  # noqa: BLE001
                out["series"][name] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prometheus snapshot")
    parser.add_argument("--start", type=float, required=True, help="unix ts")
    parser.add_argument("--end", type=float, required=True, help="unix ts (default: now)")
    parser.add_argument("--step", type=int, default=15)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    prom_url = os.environ.get("PROMETHEUS_URL", "http://localhost:9090")
    end = args.end if args.end > 0 else time.time()
    data = snapshot(prom_url, args.start, end, args.step)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2))
    print(f"[prom] wrote {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
