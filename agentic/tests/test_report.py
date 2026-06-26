from runner.record import RunRecord
from runner.report import render_markdown


def _rec(model, arch, task_id, verdict, tok_s, turns):
    return RunRecord(model=model, quant="Q4", params_b=1, arch=arch,
        hardware_tier="strix", harness_version="0.8.17", task_id=task_id,
        repo="r", issue=1, filed_date="2026-06-10", model_cutoff="2026-04-01",
        run_at="t", verdict=verdict, gate_verified=verdict == "GO", branch=None,
        turns=turns, gate_fix_attempts=0, wall_clock_s=0.0, tok_per_s=tok_s,
        tool_calls={}, files_edited=[], failure_mode=None)


def test_render_has_perf_table_and_capability_matrix():
    md = render_markdown([
        _rec("ornith-35b", "moe", "clean-433", "GO", 57.0, 83),
        _rec("ornith-35b", "moe", "gotcha-813", "INCOMPLETE", 57.0, 71),
        _rec("qwopus-27b", "dense", "clean-433", "GO", 29.0, 31),
    ])
    assert "## Serving perf" in md and "## Capability" in md
    assert "ornith-35b" in md and "57" in md
    # capability matrix has a column per task
    assert "clean-433" in md and "gotcha-813" in md
    assert "GO" in md and "INCOMPLETE" in md
