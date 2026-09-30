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

Not vendored. `harness/quality/lcb.py` downloads this dataset at run time
(outside this repo, via the `datasets` library or a manual JSONL export) and
this repo never redistributes it.

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
