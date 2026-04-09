"""
Step 3 — vanilla LoRA fine-tuning on GSM8K (no Unsloth, no bnb 4-bit).

We deliberately skip QLoRA: Phi-4-mini-instruct in bf16 fits comfortably in
24 GB on the RTX 3090 (6.6 GB base + LoRA grads + activations + a bit of
optimizer state for the LoRA params), so the 4-bit base quantization adds
debug surface for zero practical gain. Pure transformers + peft + bf16.

Loss is masked on the prompt portion (system + user turns) so we only learn
from assistant tokens. The chat template is the same one used by eval.py for
end-to-end consistency.

Usage:
    PYTHONPATH=. python scripts/train.py             # full GSM8K train, 1 epoch
    PYTHONPATH=. python scripts/train.py --max-samples 50 --epochs 1  # smoke
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
    set_seed,
)


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning. End your final line with `#### N` where "
    "N is the integer or decimal final answer (no units, no extra text)."
)


def build_example(tok, question: str, answer: str, max_seq_length: int):
    """
    Build a chat-formatted example with labels masked on the prompt portion.

    Returns dict with input_ids, attention_mask, labels (lists of ints).
    """
    messages_prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]
    messages_full = messages_prompt + [{"role": "assistant", "content": answer}]

    prompt_text = tok.apply_chat_template(
        messages_prompt, tokenize=False, add_generation_prompt=True
    )
    full_text = tok.apply_chat_template(
        messages_full, tokenize=False, add_generation_prompt=False
    )

    prompt_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tok(full_text, add_special_tokens=False)["input_ids"]

    # Append eos if not already present
    if full_ids[-1] != tok.eos_token_id:
        full_ids = full_ids + [tok.eos_token_id]

    # Truncate to max_seq_length while preserving the assistant turn
    if len(full_ids) > max_seq_length:
        full_ids = full_ids[:max_seq_length]

    labels = list(full_ids)
    # Mask everything that belongs to the prompt
    n_prompt = min(len(prompt_ids), len(labels))
    for i in range(n_prompt):
        labels[i] = -100

    attention_mask = [1] * len(full_ids)

    return {
        "input_ids": full_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="microsoft/Phi-4-mini-instruct")
    p.add_argument("--output-dir", default="checkpoints/phi4mini-gsm8k-lora")
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--max-seq-length", type=int, default=1024)
    p.add_argument("--max-samples", type=int, default=None, help="cap dataset size (for smoke)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    args = p.parse_args()

    set_seed(args.seed)

    print(f"[train] loading {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False  # required for gradient checkpointing
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()  # needed when gradient checkpointing + LoRA on a frozen base

    print(f"[train] applying LoRA r={args.lora_r} alpha={args.lora_alpha}")
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    print("[train] loading GSM8K train split")
    ds = load_dataset("openai/gsm8k", "main", split="train")
    ds = ds.shuffle(seed=args.seed)
    if args.max_samples is not None:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    print(f"[train] dataset size: {len(ds)}")

    def preprocess(example):
        return build_example(tok, example["question"], example["answer"], args.max_seq_length)

    ds_tok = ds.map(preprocess, remove_columns=ds.column_names, desc="tokenizing")

    collator = DataCollatorForSeq2Seq(
        tokenizer=tok,
        padding=True,
        return_tensors="pt",
        label_pad_token_id=-100,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=args.logging_steps,
        save_strategy="no",
        report_to="none",
        seed=args.seed,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_tok,
        data_collator=collator,
    )

    print("[train] starting training")
    train_result = trainer.train()
    print(f"[train] done. metrics: {train_result.metrics}")

    print(f"[train] saving adapter to {output_dir}")
    model.save_pretrained(str(output_dir))
    tok.save_pretrained(str(output_dir))

    # Quick sanity report
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[train] trainable params: {n_trainable / 1e6:.2f}M")
    print(f"[train] peak vram: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
