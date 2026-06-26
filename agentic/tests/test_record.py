import json
from runner.record import RunRecord, contamination_free


def test_contamination_free_compares_dates():
    assert contamination_free(filed_date="2026-06-10", model_cutoff="2026-04-01") is True
    assert contamination_free(filed_date="2026-03-01", model_cutoff="2026-04-01") is False


def test_record_roundtrips_json():
    r = RunRecord(
        model="ornith-1.0-35b", quant="Q4_K_M", params_b=35, arch="moe",
        hardware_tier="strix-halo-gfx1151", harness_version="0.8.17",
        task_id="clean-433", repo="defilantech/LLMKube", issue=433,
        filed_date="2026-06-10", model_cutoff="2026-04-01", run_at="2026-06-26T07:00:00Z",
        verdict="GO", gate_verified=True, branch="foreman/bench-clean-433-ornith-1.0-35b",
        turns=83, gate_fix_attempts=0, wall_clock_s=900.0,
        tok_per_s=57.0, tool_calls={"read_file": 18, "bash": 47},
        files_edited=["internal/metrics/metrics.go"], failure_mode=None,
        levers_on=["recovery", "hermetic", "codegen"],
    )
    assert r.contamination_free is True            # derived in __post_init__
    blob = json.loads(r.to_json())
    assert blob["verdict"] == "GO"
    assert blob["contamination_free"] is True
    again = RunRecord.from_json(r.to_json())
    assert again == r
