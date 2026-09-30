"""Run LiveCodeBench stdin/stdout solutions. Stdlib only, so it runs in a bare python image.

Each test runs in a fresh process group with a wall-clock timeout; on timeout the whole group is
killed. The sandbox pod (read-only root, no network, non-root, memory limit) is the real boundary;
this file only keeps one bad solution from stalling the run.

Standalone use: python lcb_exec.py < solutions.jsonl > results.jsonl
Each input line: {"question_id", "code", "tests": [{"input", "output"}]}.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile


def _norm(s: str) -> list[str]:
    return [" ".join(line.split()) for line in s.strip().splitlines() if line.strip()]


def run_solution(code: str, tests: list[dict], timeout_s: float) -> dict:
    results = []
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as fh:
            fh.write(code)
        for t in tests:
            proc = subprocess.Popen([sys.executable, "-I", path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                    stderr=subprocess.DEVNULL, cwd=tmp, start_new_session=True, text=True)
            try:
                out, _ = proc.communicate(t["input"], timeout=timeout_s)
                status = "ok" if _norm(out) == _norm(t["output"]) else "wrong"
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.communicate()
                status = "timeout"
            results.append({"status": status})
    return {"passed": all(r["status"] == "ok" for r in results) and bool(results), "results": results}


def main() -> int:
    for line in sys.stdin:
        rec = json.loads(line)
        res = run_solution(rec.get("code") or "", rec["tests"], float(os.environ.get("LCB_TIMEOUT_S", "6")))
        print(json.dumps({"question_id": rec["question_id"], **res}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
