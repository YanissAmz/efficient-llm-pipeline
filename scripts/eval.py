"""
Step 2 — vanilla HF eval on GSM8K, optional TurboQuant cache.

No Unsloth, no TRL, no PEFT specific paths : pure transformers + datasets.
Same interface for base and TurboQuant runs (just toggle --turboquant).

Usage:
    PYTHONPATH=. python scripts/eval.py --samples 50
    PYTHONPATH=. python scripts/eval.py --samples 50 --turboquant --bits 3
    PYTHONPATH=. python scripts/eval.py --samples 50 --turboquant --bits 4 \\
        --output results/eval_phi4mini_tq4.json
"""

import argparse
import json
import os
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.evaluate.metrics import evaluate_batch, extract_answer
from src.turboquant.polar_quant import TurboQuantCache, build_codebooks


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning. End your final line with `#### N` where "
    "N is the integer or decimal final answer (no units, no extra text)."
)


def build_prompt(tok, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--samples", type=int, default=50)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--turboquant", action="store_true")
    p.add_argument("--bits", type=int, default=3)
    p.add_argument("--output", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--attn", default="sdpa", choices=["sdpa", "eager"])
    args = p.parse_args()

    print(f"[eval] loading {args.model} (attn={args.attn}, turboquant={args.turboquant})")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=args.attn,
    )
    model.eval()
    head_dim = getattr(
        model.config,
        "head_dim",
        model.config.hidden_size // model.config.num_attention_heads,
    )
    print(f"[eval] head_dim={head_dim} layers={model.config.num_hidden_layers}")

    print("[eval] loading GSM8K test split")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    ds = ds.shuffle(seed=args.seed).select(range(args.samples))
    print(f"[eval] using {len(ds)} samples (seed={args.seed})")

    codebooks = build_codebooks(max_bits=4) if args.turboquant else None

    responses: list[str] = []
    expected: list[str] = []
    latencies: list[float] = []
    vrams: list[float] = []
    new_token_counts: list[int] = []

    t_start = time.perf_counter()
    for i, ex in enumerate(ds):
        question = ex["question"]
        gold = ex["answer"]
        prompt = build_prompt(tok, question)
        input_ids = tok(prompt, return_tensors="pt").input_ids.cuda()

        # Fresh cache per example for fairness
        cache = None
        if args.turboquant:
            cache = TurboQuantCache(dim=head_dim, bits=args.bits, codebooks=codebooks)

        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        with torch.no_grad():
            gen_kwargs = dict(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
            if cache is not None:
                gen_kwargs["past_key_values"] = cache
            out = model.generate(**gen_kwargs)
        t = time.perf_counter() - t0
        vram = torch.cuda.max_memory_allocated() / 1e9

        gen = tok.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
        n_new = out.shape[1] - input_ids.shape[1]

        responses.append(gen)
        expected.append(gold)
        latencies.append(t)
        vrams.append(vram)
        new_token_counts.append(n_new)

        is_ok = (
            extract_answer(gen) is not None
            and extract_answer(gold) is not None
            and extract_answer(gen) == extract_answer(gold)
        )
        print(
            f"[eval] {i + 1:3d}/{len(ds)}: {'OK ' if is_ok else 'NO '} "
            f"{t:5.1f}s  {n_new:3d} tok  pred={extract_answer(gen)} gold={extract_answer(gold)}"
        )

    total_time = time.perf_counter() - t_start
    metrics = evaluate_batch(responses, expected)

    summary = {
        "model": args.model,
        "turboquant": args.turboquant,
        "bits": args.bits if args.turboquant else None,
        "samples": len(ds),
        "seed": args.seed,
        "accuracy": metrics["accuracy"],
        "correct": metrics["correct"],
        "no_answer": metrics["no_answer"],
        "avg_latency_s": sum(latencies) / len(latencies),
        "total_latency_s": total_time,
        "avg_new_tokens": sum(new_token_counts) / len(new_token_counts),
        "avg_throughput_tok_s": sum(new_token_counts) / sum(latencies),
        "avg_vram_gb": sum(vrams) / len(vrams),
        "max_vram_gb": max(vrams),
    }

    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    out_path = args.output
    if out_path is None:
        suffix = f"tq{args.bits}" if args.turboquant else "base"
        out_path = f"results/eval_phi4mini_{suffix}_n{args.samples}.json"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    full = {
        **summary,
        "responses": responses,
        "expected": expected,
        "latencies": latencies,
        "new_tokens": new_token_counts,
        "per_sample_results": metrics["results"],
    }
    with open(out_path, "w") as f:
        json.dump(full, f, indent=2)
    print(f"\n[eval] saved to {out_path}")


if __name__ == "__main__":
    main()
