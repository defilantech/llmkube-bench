import json
import math
from pathlib import Path

import httpx

from harness.quality import logprobs


def completions_server(dist_by_pos, request_log=None, fail_after=None):
    """Returns vLLM-shaped prompt_logprobs: first entry None, then {id: {logprob, rank, decoded_token}}.

    request_log, if given, is appended with each request's prompt ids. fail_after, if given, makes
    the handler raise once more than fail_after requests have been seen, to simulate a crash
    partway through a capture run.
    """
    def handler(request):
        body = json.loads(request.content)
        ids = body["prompt"]
        if request_log is not None:
            request_log.append(ids)
        if fail_after is not None and len(request_log) > fail_after:
            raise RuntimeError("simulated interruption")
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


def write_corpus(base, seqs):
    p = base / "corpus.jsonl"
    p.write_text("".join(json.dumps({"id": f"s{i}", "kind": "code", "token_ids": s}) + "\n"
                         for i, s in enumerate(seqs)))
    return p


def read_capture(path):
    """Reads a capture JSONL file back into (header, {id: positions}) for easy assertions."""
    lines = Path(path).read_text().splitlines()
    header = json.loads(lines[0])
    positions = {}
    for line in lines[1:]:
        rec = json.loads(line)
        positions[rec["id"]] = rec["positions"]
    return header, positions


def test_capture_skips_position_zero_and_keeps_actual_token(tmp_path):
    corpus = write_corpus(tmp_path, [[5, 6, 7]])
    dist = lambda pos: {6: math.log(0.5), 8: math.log(0.5)}
    out = tmp_path / "cap.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, out, k=2)
    header, positions = read_capture(out)
    assert header["kind"] == "logprobs-capture" and header["k"] == 2
    pos = positions["s0"]
    assert len(pos) == 2
    assert pos[0]["actual"] == 6 and math.isclose(pos[0]["actual_lp"], math.log(0.5))
    assert pos[1]["actual"] == 7 and pos[1]["actual_lp"] == -9.0


def test_top_excludes_actual_token_when_rank_above_k(tmp_path):
    corpus = write_corpus(tmp_path, [[5, 6, 7]])
    dist = lambda pos: {6: math.log(0.5), 8: math.log(0.5)}
    out = tmp_path / "cap.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, out, k=2)
    _, positions = read_capture(out)
    # token 7 is the actual token at this position but its server-assigned rank (999) is above k=2,
    # so it must not appear in "top" even though it is the actual token.
    pos = positions["s0"][1]
    assert pos["actual"] == 7
    assert "7" not in pos["top"]
    assert set(pos["top"]) == {"6", "8"}


def test_identical_models_have_zero_kl_and_equal_ppl(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3, 4]])
    dist = lambda pos: {2: math.log(0.7), 3: math.log(0.2), 4: math.log(0.1)}
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, a_path, k=3)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, b_path, k=3)
    r = logprobs.compare(a_path, b_path)
    assert r["kld_mean"] < 1e-9 and math.isclose(r["ppl_ref"], r["ppl_cand"])


def test_shifted_model_has_positive_kl(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    logprobs.capture(completions_server(lambda p: {2: math.log(0.9), 3: math.log(0.1)}),
                     "http://x", "m", corpus, ref_path, k=2)
    logprobs.capture(completions_server(lambda p: {2: math.log(0.5), 3: math.log(0.5)}),
                     "http://x", "m", corpus, cand_path, k=2)
    r = logprobs.compare(ref_path, cand_path)
    expected = 0.9 * math.log(0.9 / 0.5) + 0.1 * math.log(0.1 / 0.5)
    assert math.isclose(r["kld_mean"], expected, rel_tol=1e-6)


def test_kl_uses_candidate_floor_for_missing_reference_id(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref_dist = lambda pos: {2: math.log(0.6), 3: math.log(0.3), 4: math.log(0.1)}
    cand_dist = lambda pos: {2: math.log(0.7), 3: math.log(0.3)}  # never returns id "4"
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    logprobs.capture(completions_server(ref_dist), "http://x", "m", corpus, ref_path, k=3)
    logprobs.capture(completions_server(cand_dist), "http://x", "m", corpus, cand_path, k=3)
    r = logprobs.compare(ref_path, cand_path)
    p = {"2": 0.6, "3": 0.3, "4": 0.1}
    floor = 0.3  # candidate's smallest returned probability stands in for the missing id "4"
    q = {"2": 0.7, "3": 0.3, "4": floor}
    qs = sum(q.values())
    expected = sum(pv * math.log(pv / (q[t] / qs)) for t, pv in p.items())
    assert math.isclose(r["kld_mean"], expected, rel_tol=1e-6)


def test_compare_raises_on_empty_top_instead_of_reporting_kl_zero(tmp_path):
    # dist returning {} means the server never ranked any other candidate at this position, so
    # the only entry is the actual token synthesized at rank 999 (above k) and "top" ends up {}.
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    empty_dist = lambda pos: {}
    normal_dist = lambda pos: {2: math.log(0.9), 3: math.log(0.1)}
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    logprobs.capture(completions_server(empty_dist), "http://x", "m", corpus, ref_path, k=2)
    logprobs.capture(completions_server(normal_dist), "http://x", "m", corpus, cand_path, k=2)
    _, ref_positions = read_capture(ref_path)
    assert ref_positions["s0"][0]["top"] == {}  # confirms the fixture actually produces an empty top
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        assert "empty top-k" in str(e) and "s0" in str(e) and "position 0" in str(e)
    else:
        raise AssertionError("expected ValueError instead of a silent KL of 0")


def test_compare_raises_on_actual_id_mismatch(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1)}
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, ref_path, k=2)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, cand_path, k=2)
    _, cand_positions = read_capture(cand_path)
    cand_positions["s0"][0]["actual"] = 999
    header = json.loads(cand_path.read_text().splitlines()[0])
    with cand_path.open("w") as fh:
        fh.write(json.dumps(header) + "\n")
        fh.write(json.dumps({"id": "s0", "positions": cand_positions["s0"]}) + "\n")
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        msg = str(e)
        assert "s0" in msg and "position 0" in msg
    else:
        raise AssertionError("expected ValueError")


def test_compare_raises_on_k_mismatch(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1)}
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, ref_path, k=2)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, cand_path, k=3)
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        assert "k" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_compare_raises_on_corpus_sha256_mismatch(tmp_path):
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1)}
    corpus_a = write_corpus(tmp_path, [[1, 2, 3]])
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    corpus_b = write_corpus(other_dir, [[1, 2, 4]])  # different content -> different sha256
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus_a, ref_path, k=2)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus_b, cand_path, k=2)
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        assert "corpus" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")


def test_compare_raises_on_id_order_mismatch(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3], [1, 5, 6]])
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1), 5: math.log(0.9), 6: math.log(0.1)}
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, ref_path, k=4)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, cand_path, k=4)
    lines = cand_path.read_text().splitlines()
    lines[1], lines[2] = lines[2], lines[1]
    cand_path.write_text("\n".join(lines) + "\n")
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        assert "s0" in str(e) or "s1" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_compare_refuses_mismatched_corpora(tmp_path):
    # Same shape as the corpus_sha256 case, phrased with hand-built minimal capture files to make
    # sure the message reads sensibly even without going through capture().
    a_path, b_path = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a_path.write_text(
        json.dumps({"kind": "logprobs-capture", "model": "m", "endpoint": "http://x", "k": 1, "corpus_sha256": "aaa"}) + "\n"
        + json.dumps({"id": "s0", "positions": [{"actual": 1, "actual_lp": -1.0, "top": {"1": -1.0}}]}) + "\n"
    )
    b_path.write_text(
        json.dumps({"kind": "logprobs-capture", "model": "m", "endpoint": "http://x", "k": 1, "corpus_sha256": "bbb"}) + "\n"
        + json.dumps({"id": "s0", "positions": [{"actual": 1, "actual_lp": -1.0, "top": {"1": -1.0}}]}) + "\n"
    )
    try:
        logprobs.compare(a_path, b_path)
    except ValueError as e:
        assert "corpus" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_capture_resumes_without_rerequesting_done_ids(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3], [1, 5, 6]])
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1), 5: math.log(0.9), 6: math.log(0.1)}
    out = tmp_path / "cap.jsonl"

    # First run "crashes" after the first item (s0) is written but before s1's request completes.
    first_requests = []
    flaky = completions_server(dist, request_log=first_requests, fail_after=1)
    try:
        logprobs.capture(flaky, "http://x", "m", corpus, out, k=4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the simulated interruption to raise")
    assert len(first_requests) == 2  # s0 succeeded, s1's request was the one that raised
    _, positions_after_crash = read_capture(out)
    assert set(positions_after_crash) == {"s0"}

    # Second run resumes: only s1 should be requested, s0 must not be re-requested.
    second_requests = []
    resumed = completions_server(dist, request_log=second_requests)
    logprobs.capture(resumed, "http://x", "m", corpus, out, k=4)
    assert len(second_requests) == 1
    assert second_requests[0] == [1, 5, 6]  # s1's token ids, not s0's
    header, positions = read_capture(out)
    assert set(positions) == {"s0", "s1"}
    assert header["k"] == 4


def test_capture_resumes_past_truncated_last_line(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3], [1, 5, 6]])
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1), 5: math.log(0.9), 6: math.log(0.1)}
    out = tmp_path / "cap.jsonl"

    # A clean crash after s0 is written, same as the interruption above, leaving a header plus
    # one complete item.
    first_requests = []
    flaky = completions_server(dist, request_log=first_requests, fail_after=1)
    try:
        logprobs.capture(flaky, "http://x", "m", corpus, out, k=4)
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected the simulated interruption to raise")

    # But this time the process was also killed mid-write of s1's line, leaving a partial,
    # unparseable trailing line with no closing brace or newline.
    with out.open("a") as fh:
        fh.write('{"id": "s1", "positions": [{"actual": 5, "actual_l')

    requests = []
    resumed = completions_server(dist, request_log=requests)
    logprobs.capture(resumed, "http://x", "m", corpus, out, k=4)  # must not raise

    assert len(requests) == 1
    assert requests[0] == [1, 5, 6]  # only the partial item (s1) was re-requested

    lines = out.read_text().splitlines()
    assert len(lines) == 3  # header plus 2 complete items
    header = json.loads(lines[0])
    assert header["kind"] == "logprobs-capture"
    rec0, rec1 = json.loads(lines[1]), json.loads(lines[2])  # both parse cleanly
    assert {rec0["id"], rec1["id"]} == {"s0", "s1"}


def test_compare_raises_on_no_positions(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    ref_path, cand_path = tmp_path / "ref.jsonl", tmp_path / "cand.jsonl"
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1)}
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, ref_path, k=2)
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, cand_path, k=2)
    # Header-only files: same header (so k/corpus_sha256 line up), zero item lines.
    ref_header = json.loads(ref_path.read_text().splitlines()[0])
    cand_header = json.loads(cand_path.read_text().splitlines()[0])
    ref_path.write_text(json.dumps(ref_header) + "\n")
    cand_path.write_text(json.dumps(cand_header) + "\n")
    try:
        logprobs.compare(ref_path, cand_path)
    except ValueError as e:
        assert str(e) == "no positions to compare"
    else:
        raise AssertionError("expected ValueError")


def test_capture_refuses_to_overwrite_mismatched_header(tmp_path):
    corpus = write_corpus(tmp_path, [[1, 2, 3]])
    dist = lambda p: {2: math.log(0.9), 3: math.log(0.1)}
    out = tmp_path / "cap.jsonl"
    logprobs.capture(completions_server(dist), "http://x", "m", corpus, out, k=2)
    before = out.read_text()
    try:
        logprobs.capture(completions_server(dist), "http://x", "m", corpus, out, k=3)
    except ValueError as e:
        assert "header" in str(e).lower()
    else:
        raise AssertionError("expected ValueError")
    assert out.read_text() == before  # never overwritten
