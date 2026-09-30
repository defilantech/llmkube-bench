from harness.quality import lcb, lcb_exec


def test_extract_code_takes_the_last_python_block():
    text = "first\n```python\nprint(1)\n```\nthen\n```python\nprint(2)\n```\n"
    assert lcb.extract_code(text).strip() == "print(2)"
    assert lcb.extract_code("no code here") is None


def test_select_problems_is_deterministic_and_stdin_only():
    rows = [{"question_id": str(i), "platform": "atcoder", "public_test_cases": '[{"input":"1","output":"1","testtype":"stdin"}]',
             "difficulty": ["easy", "medium", "hard"][i % 3]} for i in range(30)]
    rows.append({"question_id": "fn", "platform": "leetcode",
                 "public_test_cases": '[{"input":"1","output":"1","testtype":"functional"}]', "difficulty": "easy"})
    a = lcb.select_problems(rows, 10, seed=42)
    b = lcb.select_problems(rows, 10, seed=42)
    assert [r["question_id"] for r in a] == [r["question_id"] for r in b]
    assert "fn" not in {r["question_id"] for r in a} and len(a) == 10


def test_correct_solution_passes():
    r = lcb_exec.run_solution("print(int(input()) * 2)", [{"input": "21\n", "output": "42\n"}], timeout_s=5)
    assert r["passed"] is True


def test_wrong_solution_fails():
    r = lcb_exec.run_solution("print(0)", [{"input": "21\n", "output": "42\n"}], timeout_s=5)
    assert r["passed"] is False


def test_infinite_loop_times_out_and_is_killed():
    r = lcb_exec.run_solution("while True:\n    pass", [{"input": "", "output": ""}], timeout_s=1)
    assert r["passed"] is False and r["results"][0]["status"] == "timeout"


def test_bundle_runs_standalone_and_prints_results(tmp_path):
    import subprocess, sys, json as _j
    sols = _j.dumps({"question_id": "q1", "code": "print(int(input()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}]}) + "\n"
    src = open(lcb_exec.__file__).read()
    prog = lcb.bundle(sols, src)
    out = subprocess.run([sys.executable, "-"], input=prog, capture_output=True, text=True, timeout=30).stdout
    rec = _j.loads(out.strip().splitlines()[-1])
    assert rec["question_id"] == "q1" and rec["passed"] is True


def test_output_whitespace_is_normalized():
    r = lcb_exec.run_solution("print('a  b')\nprint()", [{"input": "", "output": "a b\n"}], timeout_s=5)
    assert r["passed"] is True
