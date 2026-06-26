import json
from dataclasses import dataclass, field, asdict
from datetime import date
from typing import Dict, List, Optional


def contamination_free(filed_date: str, model_cutoff: str) -> bool:
    """A run is contamination-free when the task was filed after the model's cutoff."""
    return date.fromisoformat(filed_date) > date.fromisoformat(model_cutoff)


@dataclass
class RunRecord:
    # identity
    model: str
    quant: str
    params_b: float
    arch: str
    hardware_tier: str
    harness_version: str
    task_id: str
    repo: str
    issue: int
    filed_date: str
    model_cutoff: str
    run_at: str
    # outcome
    verdict: str                 # GO | NO-GO | INCOMPLETE | ERROR
    gate_verified: bool
    branch: Optional[str]
    # effort
    turns: int
    gate_fix_attempts: int
    wall_clock_s: float
    tok_per_s: float
    # behavior
    tool_calls: Dict[str, int]
    files_edited: List[str]
    failure_mode: Optional[str]
    levers_on: List[str] = field(default_factory=list)
    # derived
    contamination_free: bool = field(default=False)

    def __post_init__(self):
        self.contamination_free = contamination_free(self.filed_date, self.model_cutoff)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, indent=2)

    @classmethod
    def from_json(cls, s: str) -> "RunRecord":
        d = json.loads(s)
        d.pop("contamination_free", None)  # recomputed in __post_init__
        return cls(**d)
