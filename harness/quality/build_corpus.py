"""Build the fixed PPL/KL corpus from Apache-2.0 LLMKube sources at a pinned commit.

Usage: python -m harness.quality.build_corpus --repo <path to an LLMKube checkout> --commit <sha>
       --tokenizer <tokenizer.json> --output corpus.jsonl [--items 100] [--tokens 2048]
Code items come from *.go files, prose items from docs/**/*.md, both sorted by path and taken
deterministically, so anyone with the same commit and tokenizer gets the same corpus.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


def files_at(repo: Path, commit: str, suffix: str, prefix: str = "") -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit],
                         check=True, capture_output=True, text=True).stdout.split()
    return sorted(f for f in out if f.endswith(suffix) and f.startswith(prefix) and "vendor/" not in f)


def text_at(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def items(repo, commit, tok, paths, kind, n, tokens):
    out, buf, src = [], [], []
    for path in paths:
        buf.extend(tok.encode(text_at(repo, commit, path) + "\n").ids)
        src.append(path)
        while len(buf) >= tokens and len(out) < n:
            out.append({"id": f"{kind}-{len(out):03d}", "kind": kind, "token_ids": buf[:tokens], "sources": src[:]})
            buf, src = buf[tokens:], []
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} {kind} items available")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m harness.quality.build_corpus")
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--commit", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--items", type=int, default=100)
    ap.add_argument("--tokens", type=int, default=2048)
    a = ap.parse_args(argv)
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    code = items(a.repo, a.commit, tok, files_at(a.repo, a.commit, ".go"), "code", a.items, a.tokens)
    prose = items(a.repo, a.commit, tok, files_at(a.repo, a.commit, ".md", "docs/"), "prose", a.items, a.tokens)
    with a.output.open("w") as fh:
        for rec in code + prose:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(code) + len(prose)} items to {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
