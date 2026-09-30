import json

import httpx

from harness.quality import agreement


class WordTok:
    """Fake tokenizer: token id i <-> the string f'wi'. encode/decode round-trip exactly, so a
    mocked server's returned text can be re-tokenized deterministically in tests."""

    def encode(self, text):
        class E:
            ids = [int(w[1:]) for w in text.split()]
        return E()

    def decode(self, ids):
        return " ".join(f"w{i}" for i in ids)


def write_corpus(base, items):
    p = base / "corpus.jsonl"
    p.write_text("".join(json.dumps(it) + "\n" for it in items))
    return p


def completions_server(reply_text, seen=None):
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if seen is not None:
            seen.append(body)
        return httpx.Response(200, json={"choices": [{"text": reply_text}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_first_divergence_counts_matching_prefix():
    assert agreement.first_divergence([1, 2, 3, 4], [1, 2, 9, 4]) == 2
    assert agreement.first_divergence([1, 2, 3], [1, 2, 3]) == 3
    assert agreement.first_divergence([], [5]) == 0


def test_summarize_reports_rates_over_shared_ids_only():
    a = {"x": [1, 2, 3, 4], "y": [5, 6, 7, 8], "only_a": [1]}
    b = {"x": [1, 2, 3, 4], "y": [5, 0, 7, 8]}
    s = agreement.summarize(a, b)
    assert s["items"] == 2
    assert s["exact_match_rate"] == 0.5
    assert s["mean_first_divergence"] == (4 + 1) / 2
    assert s["token_agreement"] == (4 + 3) / 8


def test_capture_builds_prefix_from_tokenizer_and_records_retokenized_reply(tmp_path):
    corpus = write_corpus(tmp_path, [{"id": "s0", "kind": "code", "token_ids": [1, 2, 3, 4, 5]}])
    out = tmp_path / "cap.jsonl"
    seen = []
    agreement.capture(completions_server("w6 w7", seen), "http://x", "m", corpus, WordTok(), out,
                      prefix_tokens=3, gen_tokens=2)
    body = seen[0]
    assert body["prompt"] == "w1 w2 w3"
    assert body["max_tokens"] == 2 and body["temperature"] == 0.0 and body["top_k"] == 1
    rec = json.loads(out.read_text().splitlines()[0])
    assert rec["id"] == "s0" and rec["tokens"] == [6, 7]
    assert "prefix_sha" in rec


def test_capture_skips_ids_already_in_the_output_file(tmp_path):
    corpus = write_corpus(tmp_path, [{"id": "s0", "kind": "code", "token_ids": [1, 2, 3]},
                                     {"id": "s1", "kind": "code", "token_ids": [4, 5, 6]}])
    out = tmp_path / "cap.jsonl"
    out.write_text(json.dumps({"id": "s0", "prefix_sha": "irrelevant", "tokens": [9]}) + "\n")
    seen = []
    agreement.capture(completions_server("w7", seen), "http://x", "m", corpus, WordTok(), out,
                      prefix_tokens=2, gen_tokens=1)
    assert len(seen) == 1  # only s1 was requested; s0 was already done
    ids = {json.loads(line)["id"] for line in out.read_text().splitlines()}
    assert ids == {"s0", "s1"}


def test_compare_files_rejects_mismatched_prefixes(tmp_path):
    # Different prefix_sha for the same id means the two captures were built from different
    # prefixes (different corpus or --prefix-tokens); comparing their generations would not be
    # apples-to-apples, so this must raise rather than silently report a number.
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a_path.write_text(json.dumps({"id": "s0", "prefix_sha": "aaa", "tokens": [1, 2]}) + "\n")
    b_path.write_text(json.dumps({"id": "s0", "prefix_sha": "bbb", "tokens": [1, 2]}) + "\n")
    try:
        agreement.compare_files(a_path, b_path)
    except ValueError as e:
        assert "prefix" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_compare_files_summarizes_matching_prefixes(tmp_path):
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a_path.write_text(json.dumps({"id": "s0", "prefix_sha": "same", "tokens": [1, 2, 3]}) + "\n")
    b_path.write_text(json.dumps({"id": "s0", "prefix_sha": "same", "tokens": [1, 2, 3]}) + "\n")
    r = agreement.compare_files(a_path, b_path)
    assert r["items"] == 1 and r["exact_match_rate"] == 1.0
