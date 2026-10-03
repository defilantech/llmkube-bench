# Third-party sources

## tonyd2wild_quality_battery.py

Vendored, unmodified, from
[`tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark`](https://github.com/tonyd2wild/GLM-5.3-Flash-EXL3-on-2x-NVIDIA-DGX-Spark)
`tools/quality_battery.py` at commit `dc91a125fc60349ce99498d65dac5bc772a43c54`.

License: MIT. Full text of the upstream `LICENSE` at that commit:

```
MIT License

Copyright (c) 2026 Tony DeAngelo (2Wild / @tonyd2wild)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## LiveCodeBench dataset (`livecodebench/code_generation_lite`)

Not vendored. `harness/quality/lcb.py generate` does not download anything: it
reads a local JSONL export you produce ahead of time with `--problems`. This
repo never redistributes the dataset.

License status, checked against the Hugging Face dataset card
(`https://huggingface.co/datasets/livecodebench/code_generation_lite`) on
2026-09-30: the card's `cardData` and YAML front matter list `license: cc`,
a generic Hugging Face Creative Commons tag with no variant (not CC-BY,
CC-BY-SA, CC-BY-NC or CC0). The dataset card does not otherwise state
redistribution or usage terms, and the underlying problems are sourced from
LeetCode, AtCoder and Codeforces, whose own terms of use are not addressed
by that tag. The upstream `LiveCodeBench/LiveCodeBench` GitHub repository
(the evaluation harness code, not the problem data) is MIT-licensed, which
is a separate matter from the dataset's rights.

**This is not a clearly permissive license for benchmark use, and we are not
guessing.** Treat the LiveCodeBench subset as "download and score locally,
do not redistribute problem text or committed solutions," and revisit before
publishing any derived artifact (generated solutions, scored transcripts,
etc.) that includes problem statements verbatim.

### Pinned release

Runs against this harness are pinned to:

- `version_tag`: `release_v6` (the latest release tag defined by the dataset's
  own loading script, `code_generation_lite.py`, as of 2026-09-30; it bundles
  `test.jsonl` through `test6.jsonl`)
- HF dataset revision: `0fe84c3912ea0c4d4a78037083943e8f0c4dd505`, obtained
  with:
  ```bash
  curl -s https://huggingface.co/api/datasets/livecodebench/code_generation_lite \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["sha"])'
  ```

Re-run that command before starting a new eval; if the sha has moved, decide
explicitly whether to re-pin rather than silently drifting onto a newer
revision.

### Producing the JSONL export

`harness/quality/lcb.py` only needs five fields per row: `question_id`,
`question_content`, `public_test_cases`, `difficulty`, `platform`. The
dataset's per-row files (`test.jsonl` ... `test6.jsonl`) are already
newline-delimited JSON text, not parquet and not pickled. Only the
`private_test_cases` field inside each row is a base64+zlib+pickle blob, and
we never read that field, so nothing here ever unpickles anything.

**Stdlib-only route** (no `datasets` library, downloads the raw JSONL files
directly at the pinned revision and keeps only the fields we use):

```bash
REV=0fe84c3912ea0c4d4a78037083943e8f0c4dd505
python3 - <<'PY'
import json, urllib.request

REV = "0fe84c3912ea0c4d4a78037083943e8f0c4dd505"
FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"]  # release_v6
FIELDS = ("question_id", "question_content", "public_test_cases", "difficulty", "platform")
base = f"https://huggingface.co/datasets/livecodebench/code_generation_lite/resolve/{REV}"

with open("lcb-release_v6.jsonl", "w") as out:
    for name in FILES:
        with urllib.request.urlopen(f"{base}/{name}") as resp:
            for line in resp:
                if not line.strip():
                    continue
                row = json.loads(line)
                out.write(json.dumps({k: row[k] for k in FIELDS}) + "\n")
PY
```

**`datasets` library route** (loads via the dataset's own loading script, so
it needs `trust_remote_code=True` on `datasets` versions that still support
loading scripts; recent `datasets` releases have deprecated and in some cases
removed script execution, so the stdlib route above is the more durable one):

```bash
python3 - <<'PY'
import json
from datasets import load_dataset

FIELDS = ("question_id", "question_content", "public_test_cases", "difficulty", "platform")
ds = load_dataset("livecodebench/code_generation_lite", version_tag="release_v6",
                  revision="0fe84c3912ea0c4d4a78037083943e8f0c4dd505", split="test",
                  trust_remote_code=True)
with open("lcb-release_v6.jsonl", "w") as out:
    for row in ds:
        out.write(json.dumps({k: row[k] for k in FIELDS}) + "\n")
PY
```

Pass the resulting file as `harness.quality.lcb generate --problems
lcb-release_v6.jsonl`.

### Thinking switch: `chat_template_kwargs` key for DeepSeek-V4.1-Flash

`harness/quality/lcb.py generate --thinking on` sends `chat_template_kwargs`
with a configurable key (`--thinking-kwarg`, default `enable_thinking`).

What the checkpoint ships: `deepseek-ai/DeepSeek-V4.1-Flash` has no
`chat_template` in `tokenizer_config.json` and no Jinja template file. Its own
reference encoder (`encoding/encoding.py`) takes `thinking_mode` as a string
(`"chat"` or `"thinking"`) plus a `reasoning_effort`.

What a live deployment honours: measured on 2026-09-30 against DeepSeek-V4.1-Flash
served by vLLM (the `deepseek_v4_1` model path), same prompt, temperature 0, with
each key tried in turn:

| `chat_template_kwargs` | reasoning returned |
|---|---|
| none | no (default is non-thinking) |
| `{"thinking_mode": "chat"}` / `{"thinking_mode": "thinking"}` | no (key ignored) |
| `{"enable_thinking": false}` / `{"thinking": false}` | no |
| `{"enable_thinking": true}` / `{"thinking": true}` | yes |

So on this serving path `enable_thinking` (or `thinking`) switches thinking and
`thinking_mode` does nothing; the vendored battery's hardcoded `enable_thinking`
works as-is. A deployment with its own `--chat-template` may read a different
key: repeat the check (send the same prompt with each candidate and compare the
reasoning length) before trusting `--thinking on`.
