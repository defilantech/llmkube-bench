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
