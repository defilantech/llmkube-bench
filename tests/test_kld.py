from harness.quality.kld import parse_llama_perplexity

SAMPLE = """
====== Perplexity statistics ======
Mean PPL(Q)                   :   7.123456 ±   0.045678
====== KL divergence statistics ======
Mean    KLD:   0.012345 ±   0.000456
99.0%   KLD:   0.123456
====== Token probability statistics ======
Same top p: 93.456 ± 0.123 %
"""


def test_parse_llama_perplexity_reads_ppl_kld_and_top_p():
    r = parse_llama_perplexity(SAMPLE)
    assert r == {"ppl": 7.123456, "ppl_err": 0.045678, "kld_mean": 0.012345, "kld_p99": 0.123456,
                 "same_top_p": 93.456}


def test_a_reference_against_itself_can_print_a_tiny_negative_kld():
    # llama-perplexity's own output for a model compared with its own logits file: floating-point noise can make
    # the mean slightly negative. A noise-floor run must parse, not fail.
    text = SAMPLE.replace("0.012345 ±   0.000456", "-0.000006 ±   0.000000").replace("0.123456\n", "-0.000001\n")
    r = parse_llama_perplexity(text)
    assert r["kld_mean"] == -0.000006
    assert r["kld_p99"] == -0.000001
