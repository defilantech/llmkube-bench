from runner.perf import aggregate, RequestResult


def test_aggregate_computes_throughput_and_percentiles():
    # 4 requests, 100 tokens each, total wall 8s -> agg 50 tok/s
    results = [RequestResult(ok=True, ttft=0.1, total=2.0, toks=100, tok_s=50.0) for _ in range(4)]
    agg = aggregate(results, wall=8.0)
    assert agg.ok == 4
    assert round(agg.agg_tok_s, 1) == 50.0
    assert round(agg.mean_ttft, 2) == 0.10
    assert round(agg.p50_latency, 1) == 2.0


def test_aggregate_ignores_failures_but_counts_them():
    results = [RequestResult(ok=True, ttft=0.1, total=2.0, toks=100, tok_s=50.0),
               RequestResult(ok=False, err="boom")]
    agg = aggregate(results, wall=2.0)
    assert agg.ok == 1 and agg.failed == 1
