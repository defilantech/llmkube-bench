"""Workload pattern loader.

Each pattern file is JSONL with one record per prompt:

    {"id": "chat-001", "messages": [{"role": "user", "content": "..."}], "max_tokens": 256}

The agentic pattern additionally carries a "prefix_id" field. Requests that share
a prefix_id share an identical leading system+user turn, which is what vLLM's
automatic prefix cache keys on. We emit the shared prefix verbatim on each call
(rather than linking by ID) so the comparison is fair to llama.cpp — both
runtimes see the exact same wire bytes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class PromptRecord:
    id: str
    messages: list[dict[str, Any]]
    max_tokens: int
    prefix_id: str | None = None


def load_pattern(name: str, root: Path | None = None) -> list[PromptRecord]:
    """Load a named pattern JSONL from the patterns directory."""
    base = root or Path(__file__).resolve().parent / "patterns"
    path = base / f"{name}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"pattern not found: {path}")
    records: list[PromptRecord] = []
    with path.open() as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            records.append(
                PromptRecord(
                    id=obj.get("id", f"{name}-{line_no:03d}"),
                    messages=obj["messages"],
                    max_tokens=int(obj.get("max_tokens", 256)),
                    prefix_id=obj.get("prefix_id"),
                )
            )
    if not records:
        raise ValueError(f"pattern {name} has no records")
    return records
