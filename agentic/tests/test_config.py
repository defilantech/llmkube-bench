import textwrap
import pytest
from runner.config import load_config, Model, Task

YAML = textwrap.dedent("""
hardware:
  tier: strix-halo-gfx1151
  context: shadowstack
  namespace: default
  router: fleet-router
  gateway_backend_index: 1
agent: ornith-coder-lever
models:
  - name: ornith-1.0-35b
    quant: Q4_K_M
    params_b: 35
    arch: moe
    inference_service: ornith-35b
    model_cutoff: 2026-04-01
  - name: qwopus-3.6-27b
    quant: Q4_K_M
    params_b: 27
    arch: dense
    inference_service: strix-coder
    model_cutoff: 2026-03-01
corpus:
  - id: clean-433
    repo: defilantech/LLMKube
    issue: 433
    filed_date: 2026-06-10
    kind: issue-fix
    difficulty: clean
perf:
  concurrencies: [1, 4]
  reqs_per_level: 4
  max_tokens: 160
""")


def test_load_config_parses_models_tasks_and_perf(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML)
    cfg = load_config(str(p))
    assert cfg.hardware.tier == "strix-halo-gfx1151"
    assert cfg.agent == "ornith-coder-lever"
    assert [m.name for m in cfg.models] == ["ornith-1.0-35b", "qwopus-3.6-27b"]
    assert cfg.models[0].inference_service == "ornith-35b"
    assert cfg.corpus[0].issue == 433
    assert cfg.perf.concurrencies == [1, 4]


def test_load_config_rejects_duplicate_model_names(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(YAML.replace("qwopus-3.6-27b", "ornith-1.0-35b"))
    with pytest.raises(ValueError, match="duplicate model name"):
        load_config(str(p))
