"""Parse llama.cpp's `llama-perplexity --kl-divergence` tail into a small metrics dict.

llama-perplexity prints a run's PPL, KL-divergence-vs-a-reference, and top-p agreement as a block
of human-readable stats at the end of its run. This module regexes the four numbers a bakeoff
cares about out of that text, so they can sit next to the vLLM-side harness.quality.logprobs
numbers in the same summary. It does not run llama-perplexity itself: pipe its stdout in.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_PPL_RE = re.compile(r"Mean PPL\(Q\)\s*:\s*([0-9.]+)\s*\xb1\s*([0-9.]+)")
_KLD_MEAN_RE = re.compile(r"Mean\s+KLD:\s*(-?[0-9.]+)\s*\xb1\s*([0-9.]+)")
_KLD_P99_RE = re.compile(r"99\.0%\s+KLD:\s*(-?[0-9.]+)")
_SAME_TOP_P_RE = re.compile(r"Same top p:\s*([0-9.]+)\s*\xb1\s*([0-9.]+)\s*%")


def parse_llama_perplexity(text: str) -> dict:
    ppl_m = _PPL_RE.search(text)
    kld_mean_m = _KLD_MEAN_RE.search(text)
    kld_p99_m = _KLD_P99_RE.search(text)
    top_p_m = _SAME_TOP_P_RE.search(text)
    missing = [name for name, m in (("Mean PPL(Q)", ppl_m), ("Mean KLD", kld_mean_m),
                                    ("99.0% KLD", kld_p99_m), ("Same top p", top_p_m)) if m is None]
    if missing:
        raise ValueError(
            f"llama-perplexity output is missing expected line(s): {', '.join(missing)}; "
            "run with --kl-divergence against a reference logits file to get all four"
        )
    return {"ppl": float(ppl_m.group(1)), "ppl_err": float(ppl_m.group(2)),
            "kld_mean": float(kld_mean_m.group(1)), "kld_p99": float(kld_p99_m.group(1)),
            "same_top_p": float(top_p_m.group(1))}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.quality.kld")
    ap.add_argument("--file", type=Path, help="llama-perplexity output; omit to read stdin")
    a = ap.parse_args(argv)
    text = a.file.read_text() if a.file else sys.stdin.read()
    print(json.dumps(parse_llama_perplexity(text), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
