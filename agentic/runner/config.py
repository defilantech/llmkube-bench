from dataclasses import dataclass, field
from typing import List
import yaml


@dataclass
class Hardware:
    tier: str
    context: str
    namespace: str
    router: str
    gateway_backend_index: int


@dataclass
class Model:
    name: str
    quant: str
    params_b: float
    arch: str               # "moe" | "dense"
    inference_service: str  # InferenceService to scale up for this model
    model_cutoff: str       # YYYY-MM-DD, for the contamination flag


@dataclass
class Task:
    id: str
    repo: str
    issue: int
    filed_date: str         # YYYY-MM-DD
    kind: str               # "issue-fix"
    difficulty: str         # "clean" | "gotcha"


@dataclass
class Perf:
    concurrencies: List[int] = field(default_factory=lambda: [1, 4, 8])
    reqs_per_level: int = 8
    max_tokens: int = 160


@dataclass
class Config:
    hardware: Hardware
    agent: str
    models: List[Model]
    corpus: List[Task]
    perf: Perf


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    models = [Model(**m) for m in raw["models"]]
    names = [m.name for m in models]
    if len(names) != len(set(names)):
        raise ValueError("duplicate model name in config")
    return Config(
        hardware=Hardware(**raw["hardware"]),
        agent=raw["agent"],
        models=models,
        corpus=[Task(**t) for t in raw["corpus"]],
        perf=Perf(**raw.get("perf", {})),
    )
