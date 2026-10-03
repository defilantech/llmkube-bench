"""Build the fixed PPL/KL corpus from Apache-2.0 LLMKube sources at a pinned commit.

Usage: python -m harness.quality.build_corpus --repo <path to an LLMKube checkout> --commit <sha>
       --tokenizer <tokenizer.json> --output corpus.jsonl [--items 100] [--tokens 2048]
Code items come from *.go files, prose items from docs/**/*.md, both sorted by path and taken
deterministically, so anyone with the same commit and tokenizer gets the same corpus. Generated
Go (zz_generated*, *.pb.go, or a "Code generated ... DO NOT EDIT" header in the first 5 lines) is
excluded so the code corpus reflects hand-written code, not codegen boilerplate.
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import subprocess
from pathlib import Path


def files_at(repo: Path, commit: str, suffix: str, prefix: str = "") -> list[str]:
    out = subprocess.run(["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit],
                         check=True, capture_output=True, text=True).stdout
    names = [line for line in out.splitlines() if line]
    return sorted(f for f in names if f.endswith(suffix) and f.startswith(prefix) and "vendor/" not in f)


def text_at(repo: Path, commit: str, path: str) -> str:
    return subprocess.run(["git", "-C", str(repo), "show", f"{commit}:{path}"],
                          check=True, capture_output=True, text=True).stdout


def is_generated_go(path: str, text: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    if fnmatch.fnmatch(name, "zz_generated*") or path.endswith(".pb.go"):
        return True
    head = "\n".join(text.splitlines()[:5])
    return "Code generated" in head and "DO NOT EDIT" in head


def items(repo, commit, tok, paths, kind, n, tokens, skip_generated=False):
    out, buf, src = [], [], []
    skipped = 0
    for path in paths:
        text = text_at(repo, commit, path)
        if skip_generated and is_generated_go(path, text):
            skipped += 1
            continue
        buf.extend(tok.encode(text + "\n").ids)
        src.append(path)
        while len(buf) >= tokens and len(out) < n:
            out.append({"id": f"{kind}-{len(out):03d}", "kind": kind, "token_ids": buf[:tokens], "sources": src[:]})
            buf, src = buf[tokens:], []
        if len(out) == n:
            break
    if len(out) < n:
        raise SystemExit(f"only {len(out)} {kind} items available")
    return out, skipped


def _check_no_special_tokens(tok) -> None:
    with_special = tok.encode("hello").ids
    without_special = tok.encode("hello", add_special_tokens=False).ids
    if with_special != without_special:
        raise SystemExit(
            "tokenizer adds special tokens by default (tok.encode('hello').ids != "
            "tok.encode('hello', add_special_tokens=False).ids); the corpus must be built from "
            "raw token ids with none added, or positions will not line up with vLLM teacher forcing"
        )


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
    _check_no_special_tokens(tok)
    code, code_skipped = items(a.repo, a.commit, tok, files_at(a.repo, a.commit, ".go"), "code",
                                a.items, a.tokens, skip_generated=True)
    prose, _ = items(a.repo, a.commit, tok, files_at(a.repo, a.commit, ".md", "docs/"), "prose",
                      a.items, a.tokens)
    with a.output.open("w") as fh:
        for rec in code + prose:
            fh.write(json.dumps(rec) + "\n")
    print(f"wrote {len(code) + len(prose)} items to {a.output} ({code_skipped} generated Go files skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
