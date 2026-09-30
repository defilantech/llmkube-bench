import json
import math

import httpx

from harness.quality import logprobs


def completions_server(dist_by_pos):
    """Returns vLLM-shaped prompt_logprobs: first entry None, then {id: {logprob, rank, decoded_token}}."""
    def handler(request):
        body = json.loads(request.content)
        ids = body["prompt"]
        entries = [None]
        for pos in range(1, len(ids)):
            top = dist_by_pos(pos)
            entry = {str(t): {"logprob": lp, "rank": r + 1, "decoded_token": str(t)}
                     for r, (t, lp) in enumerate(sorted(top.items(), key=lambda kv: -kv[1]))}
            if str(ids[pos]) not in entry:
                entry[str(ids[pos])] = {"logprob": -9.0, "rank": 999, "decoded_token": str(ids[pos])}
            entries.append(entry)
        return httpx.Response(200, json={"choices": [{"text": "", "prompt_logprobs": entries}]})
    return httpx.Client(transport=httpx.MockTransport(handler))


def write_corpus(tmp_path, seqs):
    p = tmp_path / "corpus.jsonl"
    p.write_text("".join(json.dumps({"id": f"s{i}", "kind": "code", "token_ids": s}) + "\n"
                         for i, s in enumerate(seqs)))
    return p


def test_capture_skips_position_zero_and_keeps_actual_token(tmp_path):
    corpus = write_corpus(tmp_path, [[5, 6, 7]])
    dist = lambda pos: {6: math.log(0.5), 8: math.log(0.5)}
    cap = logprobs.capture(completions_server(dist), "http://x", "m", corpus, k=2)
    pos = cap["positions"]["s0"]
    assert len(pos) == 2
    assert pos[0]["actual"] == 6 and math.isclose(pos[0]["actual_lp"], math.log(0.5))
    assert pos[1]["actual"] == 7 and pos[1]["actual_lp"] == -9.0


def test_identical_models_have_zero_kl_and_equal_ppl(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3, 4]])
    dist = lambda pos: {2: math.log(0.7), 3: math.log(0.2), 4: math.log(0.1)}
    a = logprobs.capture(completions_server(dist), "http://x", "m", corpus, k=3)
    b = logprobs.capture(completions_server(dist), "http://x", "m", corpus, k=3)
    r = logprobs.compare(a, b)
    assert r["kld_mean"] < 1e-9 and math.isclose(r["ppl_ref"], r["ppl_cand"])


def test_shifted_model_has_positive_kl(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref = logprobs.capture(completions_server(lambda p: {2: math.log(0.9), 3: math.log(0.1)}),
                           "http://x", "m", corpus, k=2)
    cand = logprobs.capture(completions_server(lambda p: {2: math.log(0.5), 3: math.log(0.5)}),
                            "http://x", "m", corpus, k=2)
    r = logprobs.compare(ref, cand)
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert math.isclose(r["kld_mean"], expected, rel_tol=1e-6)


def test_compare_refuses_mismatched_corpora(tmp_path):
    a = {"positions": {"s0": [{"actual": 1, "actual_lp": -1.0, "top": {"1": -1.0}}]}}
    b = {"positions": {"s1": [{"actual": 1, "actual_lp": -1.0, "top": {"1": -1.0}}]}}
    try:
        logprobs.compare(a, b)
    except ValueError as e:
        assert "corpus" in str(e)
    else:
        raise AssertionError("expected ValueError")
