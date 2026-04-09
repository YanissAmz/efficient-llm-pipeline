"""
Diagnostic : mesurer l'erreur de reconstruction de TurboQuant sur de vraies K/V
extraites de Phi-4-mini, plutôt que sur du N(0,1) synthétique.

Hypothèse à valider/invalider : les unit tests passent sur du Gaussien isotrope,
mais les K/V d'un LLM ont une structure (axes principaux, scales par dim, etc.)
qui peut faire chuter la qualité de reconstruction de PolarQuant.

Usage: PYTHONPATH=. python scripts/diag_quantize_real_kv.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.turboquant.polar_quant import TurboQuantProd, build_codebooks

MODEL = "microsoft/Phi-4-mini-instruct"


def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(
        a.flatten(0, -2).float(), b.flatten(0, -2).float(), dim=-1
    ).mean().item()


def rel_l2(a, b):
    return ((a - b).norm() / a.norm()).item()


def main():
    print(f"[diag] loading {MODEL}")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="eager"
    )
    model.eval()

    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    print(f"[diag] head_dim={head_dim}")

    prompt = "The quick brown fox jumps over the lazy dog. It was a sunny afternoon and"
    input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()
    print(f"[diag] prompt tokens: {tuple(input_ids.shape)}")

    # Forward pass with real DynamicCache to capture K/V
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=True)
    cache = out.past_key_values
    n_layers = len(cache.layers)
    print(f"[diag] cache: {n_layers} layers")

    codebooks = build_codebooks(max_bits=4)

    # Test on layers 0, n//2, n-1
    sample_layers = [0, n_layers // 2, n_layers - 1]

    for bits in [2, 3, 4]:
        quant = TurboQuantProd(dim=head_dim, bits=bits, codebooks=codebooks).cuda()
        print(f"\n=== bits={bits} ===")
        for li in sample_layers:
            k = cache.layers[li].keys  # (batch, kv_heads, seq, head_dim)
            v = cache.layers[li].values
            print(
                f"  layer {li:2d}: k.shape={tuple(k.shape)} "
                f"k.std={k.float().std().item():.3f} "
                f"k.mean={k.float().mean().item():.3f} "
                f"k.abs.max={k.float().abs().max().item():.3f}"
            )

            # Quantize → dequantize via the cache's storage path
            k_idx, k_qjl, k_rnorm, k_mnorm, k_hat_inplace = quant.quantize(k.float())
            k_recon = quant.dequantize(k_idx, k_qjl, k_rnorm, k_mnorm)
            v_idx, v_qjl, v_rnorm, v_mnorm, _ = quant.quantize(v.float())
            v_recon = quant.dequantize(v_idx, v_qjl, v_rnorm, v_mnorm)

            # k_hat_inplace is what quantize() returned directly (without storage roundtrip)
            print(
                f"    K  rel_l2={rel_l2(k.float(), k_recon):.3f} "
                f"cos={cos_sim(k.float(), k_recon):.3f} "
                f"inplace_eq_recon={torch.allclose(k_hat_inplace, k_recon, atol=1e-4)}"
            )
            print(
                f"    V  rel_l2={rel_l2(v.float(), v_recon):.3f} "
                f"cos={cos_sim(v.float(), v_recon):.3f}"
            )

    # Reference : pure Gaussian
    print("\n=== reference: random N(0,1) of same shape as layer 0 K ===")
    k0 = cache.layers[0].keys
    x = torch.randn_like(k0.float())
    for bits in [2, 3, 4]:
        quant = TurboQuantProd(dim=head_dim, bits=bits, codebooks=codebooks).cuda()
        idx, qjl, rnorm, mnorm, _ = quant.quantize(x)
        recon = quant.dequantize(idx, qjl, rnorm, mnorm)
        print(f"  bits={bits} rel_l2={rel_l2(x, recon):.3f} cos={cos_sim(x, recon):.3f}")


if __name__ == "__main__":
    main()
