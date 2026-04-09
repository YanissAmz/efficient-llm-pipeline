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
        label = "fp16 base" if not data.get("turboquant") else f"TQ {data['bits']}-bit"
        rows.append(
            {
                "config": label,
                "compression": "1.00x"
                if not data.get("turboquant")
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

    # Sort: base first then by bits descending (4, 3, 2)
    def sort_key(r):
        if r["config"] == "fp16 base":
            return (0, 0)
        bits = int(r["config"].split()[1].replace("-bit", ""))
        return (1, -bits)

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
