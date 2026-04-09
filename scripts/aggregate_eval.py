"""
Aggregate eval JSONs into a single Markdown comparison table.

Usage:
    PYTHONPATH=. python scripts/aggregate_eval.py results/eval_phi4mini_*.json
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+", type=Path)
    p.add_argument("--output", type=Path, default=Path("results/comparison.md"))
    args = p.parse_args()

    rows = []
    for path in args.paths:
        if not path.exists():
            print(f"[warn] missing: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            data = json.load(f)
        # Build a label like "fp16 base" / "fp16 base + LoRA-v2" / "TQ 4-bit + LoRA-v1"
        has_lora = bool(data.get("lora_path"))
        has_tq = bool(data.get("turboquant"))
        if has_tq:
            base_label = f"TQ {data['bits']}-bit"
        else:
            base_label = "fp16 base"
        lora_tag = ""
        if has_lora:
            lp = data["lora_path"]
            if "v2" in lp:
                lora_tag = " + LoRA-v2"
            elif "v1" in lp:
                lora_tag = " + LoRA-v1"
            else:
                # Legacy v1 path (renamed after eval): treat as v1
                lora_tag = " + LoRA-v1"
        label = base_label + lora_tag
        rows.append(
            {
                "config": label,
                "has_lora": has_lora,
                "has_tq": has_tq,
                "bits": data.get("bits") or 16,
                "compression": "1.00x"
                if not has_tq
                else f"{16 / data['bits']:.2f}x",
                "samples": data["samples"],
                "accuracy_pct": data["accuracy"] * 100,
                "correct": data["correct"],
                "no_answer": data["no_answer"],
                "avg_lat_s": data["avg_latency_s"],
                "throughput_tok_s": data["avg_throughput_tok_s"],
                "avg_vram_gb": data["avg_vram_gb"],
                "avg_new_tokens": data["avg_new_tokens"],
            }
        )

    if not rows:
        print("[error] no rows", file=sys.stderr)
        sys.exit(1)

    # Sort: (no-lora → v1 → v2) × (high bits → low bits, fp16 first)
    def sort_key(r):
        lora_order = 0
        if "v1" in r["config"]:
            lora_order = 1
        elif "v2" in r["config"]:
            lora_order = 2
        elif r["has_lora"]:
            lora_order = 3
        return (lora_order, -r["bits"])

    rows.sort(key=sort_key)

    headers = [
        "Configuration",
        "KV compression",
        "Samples",
        "Accuracy",
        "No answer",
        "Avg latency",
        "Throughput",
        "Avg VRAM",
        "Avg new tokens",
    ]
    lines = []
    lines.append(f"# Phi-4-mini-instruct — GSM8K eval comparison\n")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append(
            "| {config} | {comp} | {n} | {acc:.1f}% ({correct}) | {na} | {lat:.2f}s | {tps:.1f} tok/s | {vram:.2f} GB | {nt:.0f} |".format(
                config=r["config"],
                comp=r["compression"],
                n=r["samples"],
                acc=r["accuracy_pct"],
                correct=r["correct"],
                na=r["no_answer"],
                lat=r["avg_lat_s"],
                tps=r["throughput_tok_s"],
                vram=r["avg_vram_gb"],
                nt=r["avg_new_tokens"],
            )
        )

    md = "\n".join(lines) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(md)
    print(md)
    print(f"\n[saved] {args.output}")


if __name__ == "__main__":
    main()
