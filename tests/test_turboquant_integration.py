"""
Integration test : TurboQuantCache must work end-to-end inside model.generate().

This is the test that would have caught the two regressions the rebuild fixed :
  1. Cache API change in transformers >= 5.4 (state moved from key_cache to layers)
  2. TurboQuantMSE unit-norm assumption breaking reconstruction on real K/V

It runs against a tiny LlamaForCausalLM instantiated from a config so it has no
download cost and is fast enough for CI.

If this test ever fails on a transformers upgrade, you have a real regression.
"""

import pytest
import torch

from src.turboquant.polar_quant import TurboQuantCache, build_codebooks


@pytest.fixture(scope="module")
def tiny_llama():
    """A 2-layer Llama instantiated from config — no download, ~few hundred KB."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA, like Phi-4-mini
        head_dim=16,
        max_position_embeddings=256,
        rope_theta=10000.0,
        tie_word_embeddings=True,
    )
    model = LlamaForCausalLM(cfg).eval()
    return model, cfg


@pytest.fixture(scope="module")
def codebooks():
    return build_codebooks(max_bits=4)


def test_smoke_generate_runs_without_crashing(tiny_llama, codebooks):
    """The minimum bar : model.generate(past_key_values=TurboQuantCache(...)) returns."""
    model, cfg = tiny_llama
    cache = TurboQuantCache(dim=cfg.head_dim, bits=3, codebooks=codebooks)
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])

    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=10,
            do_sample=False,
            past_key_values=cache,
        )

    assert out.shape[1] == input_ids.shape[1] + 10, (
        f"expected {input_ids.shape[1] + 10} tokens, got {out.shape[1]}"
    )


def test_cache_seq_length_grows_correctly(tiny_llama, codebooks):
    """get_seq_length() must reflect what was actually stored."""
    model, cfg = tiny_llama
    cache = TurboQuantCache(dim=cfg.head_dim, bits=3, codebooks=codebooks)
    input_ids = torch.tensor([[10, 20, 30, 40]])

    with torch.no_grad():
        model.generate(
            input_ids=input_ids,
            max_new_tokens=5,
            do_sample=False,
            past_key_values=cache,
        )

    # After generating 5 new tokens on top of 4 prompt tokens, every layer
    # should hold the prompt + (max_new_tokens - 1) — the very last generated
    # token is sampled from the previous step's logits and never fed back
    # through the model, so it doesn't end up in the cache.
    expected = input_ids.shape[1] + 5 - 1
    for layer_idx, layer in enumerate(cache.layers):
        assert layer.get_seq_length() == expected, (
            f"layer {layer_idx} reports seq_length="
            f"{layer.get_seq_length()}, expected {expected}"
        )


def test_two_pass_forward_uses_cache(tiny_llama, codebooks):
    """Pass 1 populates the cache; pass 2 attends over it (the HellaSwag pattern)."""
    model, cfg = tiny_llama
    cache = TurboQuantCache(dim=cfg.head_dim, bits=4, codebooks=codebooks)
    ctx_ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    end_ids = torch.tensor([[9, 10, 11]])

    with torch.no_grad():
        out_ctx = model(input_ids=ctx_ids, past_key_values=cache, use_cache=True)
        cache_after_ctx = out_ctx.past_key_values
        # Sanity : cache holds the ctx tokens
        assert cache_after_ctx.layers[0].get_seq_length() == ctx_ids.shape[1]

        out_end = model(input_ids=end_ids, past_key_values=cache_after_ctx, use_cache=True)
        # After pass 2, cache holds ctx + ending
        assert cache_after_ctx.layers[0].get_seq_length() == ctx_ids.shape[1] + end_ids.shape[1]
        # Logits shape : (batch, ending_len, vocab)
        assert out_end.logits.shape == (1, end_ids.shape[1], cfg.vocab_size)


def test_compression_ratio_property(tiny_llama, codebooks):
    """The compression_ratio property must reflect the bit budget."""
    model, cfg = tiny_llama
    for bits in [2, 3, 4]:
        cache = TurboQuantCache(dim=cfg.head_dim, bits=bits, codebooks=codebooks)
        assert cache.compression_ratio == pytest.approx(16 / bits)


def test_output_is_not_garbage(tiny_llama, codebooks):
    """
    On a tiny untrained Llama we can't expect coherent text, but the TurboQuant
    output should at least be in the same statistical ballpark as the baseline:
    we check that the first generated logit's top-1 token is a valid vocab id
    (no NaN, no out-of-vocab) and that the cache reconstruction is not literally zero.
    """
    model, cfg = tiny_llama
    cache = TurboQuantCache(dim=cfg.head_dim, bits=4, codebooks=codebooks)
    input_ids = torch.tensor([[5, 6, 7, 8, 9]])

    with torch.no_grad():
        out = model(input_ids=input_ids, past_key_values=cache, use_cache=True)

    assert torch.isfinite(out.logits).all(), "logits contain NaN/Inf"
    assert (cache.layers[0].keys.abs().sum() > 0), "reconstructed K is all-zero"
    assert (cache.layers[0].values.abs().sum() > 0), "reconstructed V is all-zero"
