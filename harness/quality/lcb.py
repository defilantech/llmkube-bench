"""LiveCodeBench subset: select stdin problems deterministically, generate solutions, and score.

Only public test cases (plain JSON) are used; the private cases are pickled, and this harness
never unpickles downloaded data. Problems use stdin/stdout only, so one executor fits all.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import httpx

from harness import oai

PROMPT = ("Solve the following competitive programming problem in Python 3. Read from standard input "
          "and write to standard output. Reply with one ```python code block.\n\n{question}")


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```python\n(.*?)```", text, flags=re.S)
    return blocks[-1] if blocks else None


def select_problems(rows: list[dict], n: int, seed: int) -> list[dict]:
    stdin = [r for r in rows if all(t.get("testtype") == "stdin" for t in json.loads(r["public_test_cases"]))]
    stdin.sort(key=lambda r: r["question_id"])
    rng = random.Random(seed)
    rng.shuffle(stdin)
    return stdin[:n]


def generate(client, endpoint, model, problems, max_tokens, thinking: bool) -> list[dict]:
    out = []
    for p in problems:
        r = oai.chat(client, endpoint, model, [{"role": "user", "content": PROMPT.format(question=p["question_content"])}],
                     max_tokens=max_tokens, extra={"chat_template_kwargs": {"enable_thinking": thinking}})
        tests = [{"input": t["input"], "output": t["output"]} for t in json.loads(p["public_test_cases"])]
        out.append({"question_id": p["question_id"], "difficulty": p.get("difficulty"),
                    "code": extract_code(r.text), "tests": tests})
    return out


def bundle(solutions_jsonl: str, exec_source: str) -> str:
    """One program for `python -`: the executor's functions, then a driver fed from an embedded string."""
    body = exec_source.split("\nif __name__ ==")[0]
    driver = (
        "\nimport io as _io, sys as _sys\n"
        f"_sys.stdin = _io.StringIO({solutions_jsonl!r})\n"
        "raise SystemExit(main())\n"
    )
    return body + driver


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.quality.lcb")
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--endpoint", required=True)
    g.add_argument("--model", required=True)
    g.add_argument("--problems", required=True, type=Path, help="JSONL of LiveCodeBench rows")
    g.add_argument("--n", type=int, default=100)
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--max-tokens", type=int, default=8192)
    g.add_argument("--thinking", choices=["on", "off"], default="off")
    g.add_argument("--output", required=True, type=Path)
    b = sub.add_parser("bundle")
    b.add_argument("solutions", type=Path)
    b.add_argument("--output", required=True, type=Path)
    s = sub.add_parser("score")
    s.add_argument("results", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "bundle":
        exec_src = (Path(__file__).parent / "lcb_exec.py").read_text()
        a.output.write_text(bundle(a.solutions.read_text(), exec_src))
        return 0
    if a.cmd == "generate":
        rows = [json.loads(line) for line in a.problems.read_text().splitlines()]
        with httpx.Client() as client:
            sols = generate(client, a.endpoint, a.model, select_problems(rows, a.n, a.seed), a.max_tokens,
                            a.thinking == "on")
        a.output.write_text("".join(json.dumps(x) + "\n" for x in sols))
    else:
        res = [json.loads(line) for line in a.results.read_text().splitlines()]
        passed = sum(r["passed"] for r in res)
        print(json.dumps({"n": len(res), "passed": passed, "pass_rate": passed / len(res)}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
