import json, pathlib
from runner.behavior import extract_behavior

FX = pathlib.Path(__file__).parent / "fixtures"


def _load(name):
    return json.loads((FX / name).read_text())


def test_go_run_tool_mix_and_no_failure():
    b = extract_behavior(_load("transcript_go.json"), _load("status_go.json"))
    assert b.turns == 3
    assert b.tool_calls == {"read_file": 1, "grep": 1, "str_replace": 1, "submit_result": 1}
    assert b.files_edited == ["internal/router/proxy.go"]
    assert b.gate_fix_attempts == 0
    assert b.failure_mode is None
    assert b.branch == "foreman/bench-clean-433-x"


def test_stuck_loop_failure_mode():
    b = extract_behavior({"turnCount": 36, "messages": []}, _load("status_stuck.json"))
    assert b.failure_mode == "RepeatedToolCall"


def test_codegen_drift_failure_mode_and_attempts():
    b = extract_behavior({"turnCount": 71, "messages": []}, _load("status_codegen.json"))
    assert b.gate_fix_attempts == 3
    assert b.failure_mode == "codegen-drift"
