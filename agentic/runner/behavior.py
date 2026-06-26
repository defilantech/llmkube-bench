import json
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Behavior:
    turns: int
    tool_calls: Dict[str, int]
    files_edited: List[str]
    gate_fix_attempts: int
    failure_mode: Optional[str]
    branch: Optional[str]
    verdict: str = ""
    gate_verified: bool = False


EDIT_TOOLS = ("write_file", "str_replace")


def _classify_failure(status: dict) -> Optional[str]:
    if status.get("verdict") == "GO":
        return None
    extra = (status.get("result", {}).get("extra", {}) or {}).get("modelExtra", {}) or {}
    if extra.get("signal"):                       # stuck-loop detector
        return extra["signal"]                    # e.g. RepeatedToolCall
    gate = extra.get("gateOutput", "") or ""
    if "codegen drift" in gate.lower():
        return "codegen-drift"
    summary = (status.get("result", {}).get("summary") or "").lower()
    if "push to fork failed" in summary:
        return "push-failed"
    if "pre-existing env" in summary or "envtest" in summary:
        return "false-env-claim"
    if extra.get("gateAttempts"):
        return "gate-fail"
    return status.get("failureReason") or "unknown"


def extract_behavior(transcript: dict, status: dict) -> Behavior:
    tools: Counter = Counter()
    edited: List[str] = []
    for m in transcript.get("messages", []):
        if m.get("role") != "assistant":
            continue
        for tc in (m.get("tool_calls") or []):
            fn = tc["function"]["name"]
            tools[fn] += 1
            if fn in EDIT_TOOLS:
                try:
                    edited.append(json.loads(tc["function"].get("arguments", "{}")).get("path"))
                except (ValueError, TypeError):
                    pass
    extra = (status.get("result", {}).get("extra", {}) or {})
    model_extra = extra.get("modelExtra", {}) or {}
    return Behavior(
        turns=transcript.get("turnCount", 0),
        tool_calls=dict(tools),
        files_edited=[e for e in edited if e],
        gate_fix_attempts=int(model_extra.get("gateAttempts") or 0),
        failure_mode=_classify_failure(status),
        branch=extra.get("branch"),
        verdict=status.get("verdict", ""),
        gate_verified=status.get("verdict") == "GO",
    )
