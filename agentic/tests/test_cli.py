import json, pathlib
from runner.cli import cmd_report
from runner.record import RunRecord


def _write_rec(d, **kw):
    base = dict(model="m", quant="Q4", params_b=1, arch="moe", hardware_tier="strix",
        harness_version="0.8.17", task_id="clean-433", repo="r", issue=1,
        filed_date="2026-06-10", model_cutoff="2026-04-01", run_at="t", verdict="GO",
        gate_verified=True, branch=None, turns=10, gate_fix_attempts=0, wall_clock_s=0.0,
        tok_per_s=50.0, tool_calls={}, files_edited=[], failure_mode=None)
    base.update(kw)
    (d / f"{base['model']}-{base['task_id']}.json").write_text(RunRecord(**base).to_json())


def test_cmd_report_reads_records_and_writes_markdown(tmp_path):
    recs = tmp_path / "records"; recs.mkdir()
    out = tmp_path / "out.md"
    _write_rec(recs, model="ornith-35b")
    cmd_report(records_dir=str(recs), out=str(out))
    text = out.read_text()
    assert "## Serving perf" in text and "ornith-35b" in text
