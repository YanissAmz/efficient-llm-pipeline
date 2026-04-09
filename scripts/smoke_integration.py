"""
Step 1 du rebuild integration-first : prouver que TurboQuantCache
fonctionne dans model.generate() sur le modèle cible.

Si ce script ne passe pas, NE PAS écrire train.py / eval.py.
Debug TurboQuantCache jusqu'à ce que ça passe.

Usage:
    python scripts/smoke_integration.py [--model microsoft/Phi-4-mini-instruct] [--bits 3]
"""

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.turboquant.polar_quant import TurboQuantCache, build_codebooks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--prompt", default="Hello, my name is")
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = p.parse_args()

    print(f"[smoke] loading {args.model} (attn={args.attn})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn,
    )
    model.eval()

    # Verify config is non-nested (no multimodal wrapper)
    assert hasattr(model.config, "hidden_size"), (
        f"model.config.hidden_size missing — multimodal wrapper detected. "
        f"Config type: {type(model.config).__name__}"
    )
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    print(
        f"[smoke] arch: hidden={model.config.hidden_size} "
        f"n_heads={model.config.num_attention_heads} "
        f"n_kv_heads={model.config.num_key_value_heads} "
        f"head_dim={head_dim} "
        f"n_layers={model.config.num_hidden_layers}"
    )

    # Build TurboQuantCache
    codebooks = build_codebooks(max_bits=4)
    cache = TurboQuantCache(dim=head_dim, bits=args.bits, codebooks=codebooks)

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.cuda()
    print(f"[smoke] prompt tokens: {tuple(input_ids.shape)}")

    # === Baseline run (no cache override) ===
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_baseline = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
        )
    t_baseline = time.perf_counter() - t0
    vram_baseline = torch.cuda.max_memory_allocated() / 1e9
    text_baseline = tok.decode(out_baseline[0], skip_special_tokens=True)

    # === TurboQuant run ===
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_tq = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            pad_token_id=tok.eos_token_id,
            past_key_values=cache,
        )
    t_tq = time.perf_counter() - t0
    vram_tq = torch.cuda.max_memory_allocated() / 1e9
    text_tq = tok.decode(out_tq[0], skip_special_tokens=True)

    # === Assertions ===
    assert out_tq.shape[1] > input_ids.shape[1], "TurboQuant generated 0 tokens"
    expected_len = input_ids.shape[1] + args.max_new_tokens
    assert out_tq.shape[1] == expected_len, (
        f"unexpected gen length {out_tq.shape[1]} (expected {expected_len})"
    )

    # === Report ===
    print("\n=== BASELINE ===")
    print(f"  text:    {text_baseline!r}")
    print(f"  latency: {t_baseline:.2f}s ({args.max_new_tokens / t_baseline:.1f} tok/s)")
    print(f"  vram:    {vram_baseline:.2f} GB")
    print("\n=== TURBOQUANT ===")
    print(f"  text:    {text_tq!r}")
    print(f"  latency: {t_tq:.2f}s ({args.max_new_tokens / t_tq:.1f} tok/s)")
    print(f"  vram:    {vram_tq:.2f} GB")
    print(f"  bits:    {args.bits}/16 (theoretical compression {16 / args.bits:.2f}x)")
    print(f"  cache.compression_ratio: {cache.compression_ratio:.2f}x")
    print(f"  cache.memory_saved (cumulative): {cache.memory_saved_mb:.2f} MB")

    print("\n[smoke] PASSED ✓")


if __name__ == "__main__":
    main()
