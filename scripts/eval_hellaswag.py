"""
Step 4 — HellaSwag loglikelihood eval with optional TurboQuantCache.

For each example we score the 4 endings by sum of log-probs of the ending tokens
conditioned on the context. To exercise the KV cache compression we use a
2-pass forward pattern :

  1) forward(ctx_ids, past_key_values=cache, use_cache=True)
     → cache stores ctx K/V (compressed if TurboQuant)
  2) forward(ending_ids, past_key_values=cache, use_cache=True)
     → ending attention reads the cached (decompressed) ctx K/V
     → returns logits for the ending positions

We then read pass1.logits[-1] (predicts ending[0]) and pass2.logits[:-1]
(predict ending[1..L-1]) to compute the full ending loglikelihood.

A single forward on the concatenated sequence would NOT exercise the cache :
inside one pass attention reads K/V freshly computed in the same pass, not
from the cache. The 2-pass pattern is what makes TurboQuant matter here.

Usage:
    PYTHONPATH=. python scripts/eval_hellaswag.py --samples 50
    PYTHONPATH=. python scripts/eval_hellaswag.py --samples 50 --turboquant --bits 4
"""

import argparse
import json
import os
import time

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.turboquant.polar_quant import TurboQuantCache, build_codebooks


def build_context(example) -> str:
    """HellaSwag context = activity_label + ctx (ctx_a + ctx_b joined)."""
    ctx = example["ctx_a"]
    if example.get("ctx_b"):
        ctx = ctx + " " + example["ctx_b"].capitalize()
    return f"{example['activity_label']}: {ctx}"


@torch.no_grad()
def score_ending(model, tok, ctx_ids, ending_text, cache):
    """Return sum of log-probs of `ending_text` tokens given `ctx_ids` and `cache`.

    Uses 2-pass forward so that the cache is populated by ctx and read by ending.
    """
    # Tokenize the ending with a leading space (HellaSwag endings are continuations)
    ending = " " + ending_text.strip() if not ending_text.startswith(" ") else ending_text
    end_ids = tok(ending, add_special_tokens=False, return_tensors="pt").input_ids.cuda()
    if end_ids.shape[1] == 0:
        return float("-inf"), 0

    # Pass 1: ctx forward, populates cache
    out_ctx = model(input_ids=ctx_ids, past_key_values=cache, use_cache=True)
    last_logits_ctx = out_ctx.logits[:, -1, :]  # (1, vocab) — predicts end_ids[0]
    # Capture the populated cache (in base case `cache` was None and the model
    # built a fresh DynamicCache internally; in TQ case it's the same instance)
    cache = out_ctx.past_key_values

    # Pass 2: ending forward, reads cache
    out_end = model(input_ids=end_ids, past_key_values=cache, use_cache=True)
    end_logits = out_end.logits[:, :-1, :]  # (1, L_end-1, vocab) — predict end_ids[1..]

    # log P(end_ids[0] | ctx)
    logp = F.log_softmax(last_logits_ctx, dim=-1)
    total_logp = logp[0, end_ids[0, 0]].item()

    # log P(end_ids[i] | ctx + end_ids[:i]) for i=1..L_end-1
    if end_ids.shape[1] > 1:
        logp_end = F.log_softmax(end_logits, dim=-1)  # (1, L_end-1, vocab)
        token_logp = logp_end[0, torch.arange(end_ids.shape[1] - 1), end_ids[0, 1:]]
        total_logp += token_logp.sum().item()

    return total_logp, end_ids.shape[1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--turboquant", action="store_true")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--length-norm", action="store_true", help="length-normalize logp")
    args = p.parse_args()

    print(f"[hs] loading {args.model} (turboquant={args.turboquant})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )

    print("[hs] loading HellaSwag validation split")
    ds = load_dataset("Rowan/hellaswag", split="validation")
    ds = ds.shuffle(seed=args.seed).select(range(args.samples))
    print(f"[hs] using {len(ds)} samples")

    codebooks = build_codebooks(max_bits=4) if args.turboquant else None

    n_correct = 0
    per_sample = []
    latencies = []
    vrams = []

    t_start = time.perf_counter()
    for i, ex in enumerate(ds):
        ctx_text = build_context(ex)
        ctx_ids = tok(ctx_text, return_tensors="pt").input_ids.cuda()

        scores = []
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for ending in ex["endings"]:
            cache = (
                TurboQuantCache(dim=head_dim, bits=args.bits, codebooks=codebooks)
                if args.turboquant
                else None
            )
            logp, n_tok = score_ending(model, tok, ctx_ids, ending, cache)
            score = logp / max(n_tok, 1) if args.length_norm else logp
            scores.append(score)
        t = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9

        pred = max(range(4), key=lambda k: scores[k])
        gold = int(ex["label"])
        ok = pred == gold
        n_correct += int(ok)
        per_sample.append({"pred": pred, "gold": gold, "scores": scores, "ok": ok})
        latencies.append(t)
        vrams.append(vram)

        if (i + 1) % 10 == 0 or i == 0:
            running_acc = n_correct / (i + 1)
            print(
                f"[hs] {i + 1:3d}/{len(ds)} : acc={running_acc:.3f} "
                f"({n_correct}/{i + 1}) lat={t:.2f}s vram={vram:.2f}GB"
            )

    total_time = time.perf_counter() - t_start
    accuracy = n_correct / len(ds)

    summary = {
        "model": args.model,
        "task": "hellaswag",
        "turboquant": args.turboquant,
        "bits": args.bits if args.turboquant else None,
        "samples": len(ds),
        "seed": args.seed,
        "length_norm": args.length_norm,
        "accuracy": accuracy,
        "correct": n_correct,
        "avg_latency_s": sum(latencies) / len(latencies),
        "total_latency_s": total_time,
        "avg_vram_gb": sum(vrams) / len(vrams),
        "max_vram_gb": max(vrams),
    }

    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = args.output
    if out_path is None:
        suffix = f"tq{args.bits}" if args.turboquant else "base"
        out_path = f"results/eval_phi4mini_hellaswag_{suffix}_n{args.samples}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    with open(out_path, "w") as f:
        json.dump({**summary, "per_sample": per_sample}, f, indent=2)
    print(f"\n[hs] saved to {out_path}")


if __name__ == "__main__":
    main()
