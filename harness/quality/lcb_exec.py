"""Run LiveCodeBench stdin/stdout solutions. Stdlib only, so it runs in a bare python image.

Each test runs in a fresh session (its own process group) with a wall-clock timeout. On timeout
we kill that process group and reap only the direct child: stdin and stdout go through temp files,
not pipes, so nothing here does a blocking read that depends on every descendant closing its copy
of a pipe's write end. A solution that forks and calls os.setsid() can detach a grandchild into a
new session where killpg on the original group no longer reaches it; that grandchild is not our
problem to chase down (the sandbox pod's non-root uid, no network, read-only root and memory limit
are the real boundary), but it must never make this file hang waiting for it.

preexec_fn caps the child's own CPU time, address space, output file size, and (on Linux) process
count as a second line of defense; each rlimit is best-effort and skipped if the platform does not
support it. A solution that fills the pod's disk-backed /tmp is treated as a per-test failure, not
a fatal error: each test's files are written, run, read (capped at MAX_OUTPUT_BYTES) and unlinked
inside a try/except OSError, so a disk-full ENOSPC downgrades that one test to status "error"
instead of aborting the whole solution, and main() wraps each whole solution the same way so one
solution's unexpected exception never stops the rest of the batch.

Standalone use: python lcb_exec.py < solutions.jsonl > results.jsonl
Each input line: {"question_id", "code", "tests": [{"input", "output"}]}.
"""
from __future__ import annotations

import json
import math
import os
import signal
import subprocess
import sys
import tempfile

try:
    import resource
except ImportError:  # pragma: no cover - resource is POSIX-only
    resource = None

MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def _norm(s: str) -> list[str]:
    return [" ".join(line.split()) for line in s.strip().splitlines()]


def _limit_resources(timeout_s: float):
    """Returns a preexec_fn that best-effort caps the child's CPU time, address space, output
    file size, and (on Linux) process count. Each limit is set independently so one unsupported
    limit (RLIMIT_AS is not enforceable on macOS, for example) never blocks the others."""
    def _set() -> None:
        if resource is None:
            return
        cpu_s = math.ceil(timeout_s) + 1
        try:
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s))
        except (ValueError, OSError):
            pass
        if sys.platform.startswith("linux"):
            # Per-uid, not per-process-tree: the pod runs this as a dedicated uid (65534), so
            # capping it here caps only this sandbox's fork bombs, not anything else on the node.
            try:
                resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
            except (ValueError, OSError):
                pass
        try:
            one_gib = 1 << 30
            resource.setrlimit(resource.RLIMIT_AS, (one_gib, one_gib))
        except (ValueError, OSError):
            pass
        try:
            resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_OUTPUT_BYTES, MAX_OUTPUT_BYTES))
        except (ValueError, OSError):
            pass
    return _set


def run_solution(code: str, tests: list[dict], timeout_s: float) -> dict:
    results = []
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        path = os.path.join(tmp, "solution.py")
        with open(path, "w") as fh:
            fh.write(code)
        for i, t in enumerate(tests):
            in_path = os.path.join(tmp, f"in_{i}.txt")
            out_path = os.path.join(tmp, f"out_{i}.txt")
            try:
                with open(in_path, "w") as fh:
                    fh.write(t["input"])
                with open(in_path) as stdin_f, open(out_path, "w") as stdout_f:
                    proc = subprocess.Popen([sys.executable, "-I", path], stdin=stdin_f, stdout=stdout_f,
                                            stderr=subprocess.DEVNULL, cwd=tmp, start_new_session=True,
                                            preexec_fn=_limit_resources(timeout_s))
                    try:
                        proc.wait(timeout=timeout_s)
                        with open(out_path) as fh:
                            out = fh.read(MAX_OUTPUT_BYTES)
                        status = "ok" if _norm(out) == _norm(t["output"]) else "wrong"
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait()
                        status = "timeout"
            except OSError:
                status = "error"
            finally:
                for p in (in_path, out_path):
                    try:
                        os.unlink(p)
                    except OSError:
                        pass
            results.append({"status": status})
    return {"passed": all(r["status"] == "ok" for r in results) and bool(results), "results": results}


def main() -> int:
    for line in sys.stdin:
        qid = None
        try:
            rec = json.loads(line)
            qid = rec.get("question_id")
            res = run_solution(rec.get("code") or "", rec["tests"], float(os.environ.get("LCB_TIMEOUT_S", "6")))
            out = {"question_id": rec["question_id"], **res}
        except Exception as e:
            out = {"question_id": qid, "passed": False, "error": str(e)[:200]}
        print(json.dumps(out), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
