"""Shared helper for JSONL capture files that are appended to across resumable runs.

Both harness.quality.logprobs and harness.quality.agreement write one JSON record per line and
need to resume cleanly after a process is killed mid-write, which can leave a torn, unparseable
final line. Both use the same fix: atomically truncate the file back to the end of its last
complete line before resuming, so the torn item looks not-done and gets re-requested instead of
corrupting the next append or crash-looping on invalid JSON.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def truncate_to(path: Path, good_end: int) -> None:
    """Atomically drops everything in path after byte offset good_end.

    Rewrites to a temp file in the same directory and os.replace()s it in, rather than truncating
    the original file in place, so a crash partway through this rewrite can never leave a
    half-written file behind.
    """
    with path.open("rb") as src:
        good_bytes = src.read(good_end)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(good_bytes)
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.remove(tmp_name)
        except FileNotFoundError:
            pass
        raise
