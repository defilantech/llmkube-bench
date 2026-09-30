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


def test_select_problems_excludes_rows_with_no_public_tests():
    rows = [{"question_id": str(i), "platform": "atcoder", "public_test_cases": '[{"input":"1","output":"1","testtype":"stdin"}]',
             "difficulty": "easy"} for i in range(5)]
    rows.append({"question_id": "empty", "platform": "atcoder", "public_test_cases": "[]", "difficulty": "easy"})
    a = lcb.select_problems(rows, 10, seed=42)
    assert "empty" not in {r["question_id"] for r in a}


def test_correct_solution_passes():
    r = lcb_exec.run_solution("print(int(input()) * 2)", [{"input": "21\n", "output": "42\n"}], timeout_s=5)
    assert r["passed"] is True


def test_wrong_solution_fails():
    r = lcb_exec.run_solution("print(0)", [{"input": "21\n", "output": "42\n"}], timeout_s=5)
    assert r["passed"] is False


def test_infinite_loop_times_out_and_is_killed():
    r = lcb_exec.run_solution("while True:\n    pass", [{"input": "", "output": ""}], timeout_s=1)
    assert r["passed"] is False and r["results"][0]["status"] == "timeout"


def test_double_fork_escaper_does_not_block_on_a_detached_grandchild():
    import time

    code = (
        "import os, time\n"
        "pid = os.fork()\n"
        "if pid == 0:\n"
        "    os.setsid()\n"
        "    time.sleep(10)\n"
        "    os._exit(0)\n"
        "else:\n"
        "    time.sleep(5)\n"
    )
    start = time.monotonic()
    r = lcb_exec.run_solution(code, [{"input": "", "output": ""}], timeout_s=1)
    elapsed = time.monotonic() - start
    assert elapsed < 4
    assert r["passed"] is False and r["results"][0]["status"] == "timeout"


def test_bundle_runs_standalone_and_prints_results(tmp_path):
    import subprocess, sys, json as _j
    sols = _j.dumps({"question_id": "q1", "code": "print(int(input()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}]}) + "\n"
    src = open(lcb_exec.__file__).read()
    prog = lcb.bundle(sols, src)
    out = subprocess.run([sys.executable, "-"], input=prog, capture_output=True, text=True, timeout=30).stdout
    rec = _j.loads(out.strip().splitlines()[-1])
    assert rec["question_id"] == "q1" and rec["passed"] is True


def test_bundle_strips_the_main_guard_via_ast():
    src = open(lcb_exec.__file__).read()
    prog = lcb.bundle('{"question_id": "q", "code": "", "tests": []}\n', src)
    assert "__main__" not in prog


def test_output_whitespace_is_normalized():
    r = lcb_exec.run_solution("print('a  b')\nprint()", [{"input": "", "output": "a b\n"}], timeout_s=5)
    assert r["passed"] is True


def test_missing_interior_blank_line_fails():
    r = lcb_exec.run_solution("print('a')\nprint('b')", [{"input": "", "output": "a\n\nb\n"}], timeout_s=5)
    assert r["passed"] is False


def test_disk_hog_gets_a_non_ok_status_not_an_exception():
    code = (
        "with open('hog', 'wb') as f:\n"
        "    for _ in range(64):\n"
        "        f.write(b'0' * (1024 * 1024))\n"
        "        f.flush()\n"
        "print('done')\n"
    )
    r = lcb_exec.run_solution(code, [{"input": "", "output": "done\n"}], timeout_s=5)
    assert r["passed"] is False and r["results"][0]["status"] in ("error", "wrong")


def test_main_recovers_after_a_disk_hog_solution():
    import subprocess, sys, json as _j

    hog_code = (
        "with open('hog', 'wb') as f:\n"
        "    for _ in range(64):\n"
        "        f.write(b'0' * (1024 * 1024))\n"
        "        f.flush()\n"
    )
    lines = (
        _j.dumps({"question_id": "hog", "code": hog_code, "tests": [{"input": "", "output": "nope\n"}]}) + "\n"
        + _j.dumps({"question_id": "ok", "code": "print(int(input()) + 1)", "tests": [{"input": "1\n", "output": "2\n"}]}) + "\n"
    )
    out = subprocess.run([sys.executable, lcb_exec.__file__], input=lines, capture_output=True, text=True,
                         timeout=30).stdout
    recs = [_j.loads(l) for l in out.strip().splitlines()]
    assert recs[0]["question_id"] == "hog"
    assert recs[1]["question_id"] == "ok" and recs[1]["passed"] is True
