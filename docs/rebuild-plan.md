# Rebuild Plan — Efficient LLM Pipeline

**Date** : 2026-04-09
**Session** : handoff from exploration session to clean rebuild
**Reason** : the current codebase was built for Colab T4 + Qwen3.5-2B fp16 vanilla. It does not cleanly integrate with the target environment (RTX 3090 + recent HuggingFace stack + Unsloth + modern transformers cache API). TurboQuant — the main differentiator of this portfolio project — is blocked on integration, not on correctness.

---

## 1. Honest diagnosis of the current state

### What works (keep as-is)
- `src/turboquant/polar_quant.py` — PolarQuant + QJL math is correct. Tests pass.
- `src/turboquant/qjl.py` — standalone 1-bit unbiased quantizer.
- `src/evaluate/metrics.py` — pure-Python GSM8K answer extraction.
- `tests/test_turboquant.py` + `tests/test_metrics.py` — **25/25 green**.
- `scripts/benchmark_v3.py` — Ollama LLM benchmark, unrelated to the pipeline but useful.
- Project scaffolding : `pyproject.toml`, `.pre-commit-config.yaml`, `Dockerfile`, `Makefile`, GitHub Actions CI, ruff config.

### What is broken / throw away
- `configs/default.yaml` — targets Qwen3-4B then Qwen3.5-4B, neither works end-to-end with TurboQuant.
- `scripts/train.py` — written in this session, works for Qwen3.5-4B smoke test (20 steps, 1.47→1.10 loss) but produces adapters in a multimodal-wrapper path (`base_model.model.model.language_model.layers.X.*`) that vanilla PEFT can't load.
- `scripts/eval.py` — written in this session, works for base + LoRA via Unsloth loader, but **crashes with TurboQuant** on `SDPA bias contiguous` error inside `unsloth_compiled_cache/unsloth_compiled_module_qwen3_5.py`.
- `src/serve/api.py` — untested against the rebuild stack.
- `src/turboquant/polar_quant.py::TurboQuantCache._init_recurrent_states` — GatedDeltaNet hook, only useful for Qwen3.5 hybrid. Strip it for pure-transformer target.
- `notebooks/01_finetune.ipynb` + `02_turboquant.ipynb` — Colab artifacts. Archive or delete, do not use as a source.

### The 4 integration blockers we hit
1. **Unsloth compiled patches** — `unsloth_compiled_cache/unsloth_compiled_module_qwen3_5.py` rewrites `Qwen3_5Attention_forward` to call a custom `attention_interface`. TurboQuantCache is not visible to this rewrite path.
2. **SDPA strict layout** — `torch.nn.functional.scaled_dot_product_attention` raises `(*bias): last dimension must be contiguous` when the cache returns dequantized K/V that don't match the pre-computed attention-mask layout. `.contiguous()` on K/V does not fix it.
3. **PEFT namespace path mismatch** — Unsloth saves LoRA keys as `base_model.model.model.language_model.layers.X.self_attn.q_proj.lora_A.weight`. Vanilla `PeftModel.from_pretrained` expects `base_model.model.model.layers.X.self_attn.q_proj.lora_A.default.weight`. Result: LoRA loads as "all missing", base model behavior dominates, 0% accuracy.
4. **Multimodal config nesting** — Qwen3.5 is tagged `any-to-any` on HF. `model.config.hidden_size` does not exist on the top-level config — it lives under `model.config.text_config`. Any code that hard-codes the non-nested path breaks.

### Wins from this session (keep for reference)
- **llama-server optimized** for deep-heretic on port 8090. Config:
  ```bash
  MAIN=/usr/share/ollama/.ollama/models/blobs/sha256-604d458ce8feb7d97cb9318878050c4c69485ef454164f6202eeeac684806ec9
  LLAMA_SET_ROWS=1 ~/llama.cpp-cuda/build/bin/llama-server \
    -m "$MAIN" -ngl 999 -c 4096 --flash-attn on \
    --cache-type-k q8_0 --cache-type-v q8_0 \
    --threads 4 --threads-batch 8 \
    --batch-size 2048 --ubatch-size 2048 \
    --spec-type ngram-mod --draft-max 64 --draft-min 0 \
    --port 8090 --host 127.0.0.1
  ```
  Measured : **27 tok/s → 35 tok/s baseline (+30%)**, **45-60 tok/s on repetitive/code content (+67 to +124%)** via n-gram speculative lookup.
- **Smoke training verified** : Qwen3.5-4B QLoRA via Unsloth, 20 steps, 159s, train loss 1.10, eval loss 0.86, VRAM peak 6.59GB. Adapters saved at `./checkpoints/qwen35-smoke/` (32.6MB, 64 keys across 8 full-attention layers 3/7/11/15/19/23/27/31).
- **uncensored-fast model is broken** : GLM-4.7-Flash Q5_K_M deepseek2 arch quant is unusable (0.1 tok/s, then timeouts). Replace or delete.

---

## 2. Rebuild strategy — integration-first, not infra-first

The current codebase made the classic mistake : scaffold → train → eval → **then** try to plug in the hard thing (TurboQuant). By the time we got there, the whole stack was wired in an incompatible way.

The rebuild does the opposite : **the first script is a 40-line smoke test that proves TurboQuantCache works in `model.generate()` on the target model**. Nothing else is written until that is green.

### The 4 design principles

**Principle 1 : Model choice is driven by integration stability, not by hype.**

Pick a pure-transformer GQA model. No hybrid (GatedDeltaNet), no sliding-window, no multimodal wrapper, no exotic arch. TurboQuant will plug into this cleanly because DynamicCache is well-defined on it.

Candidates, ranked :
1. **`meta-llama/Llama-3.2-3B-Instruct`** — 3.2B, pure transformer, canonical GQA, ubiquitous support. First choice.
2. **`Qwen/Qwen2.5-3B-Instruct`** — 3B, pure transformer, very stable, easy to swap to 7B later if VRAM allows.
3. **`microsoft/Phi-4-mini-instruct`** — 3.8B, recent (late 2025), pure transformer, good for portfolio novelty.

Do **NOT** use : Qwen3.5 (hybrid), Gemma4 (SWA+multimodal), Qwen3-4B (head_dim=128 inconsistency), anything tagged `image-text-to-text` or `any-to-any`.

**Principle 2 : Build in integration order, not in feature order.**

```
Step 1  (30 min)  — smoke_integration.py : load base model + 1 forward pass with TurboQuantCache
Step 2  (1 h)     — scripts/eval.py : base vs base+TurboQuant accuracy on GSM8K 50 samples
Step 3  (1 h)     — scripts/train.py : QLoRA fine-tuning, vanilla PEFT, no Unsloth
Step 4  (1 h)     — full benchmark : base / LoRA / LoRA+TurboQuant on 200 samples
Step 5  (30 min)  — README rewrite with real numbers
Step 6  (opt)     — src/serve/api.py update + Dockerfile smoke
```

If Step 1 fails, we stop and diagnose. We do NOT write Step 2 on a broken Step 1. The existing code is the proof that skipping this rule destroys the project.

**Principle 3 : No Unsloth if avoidable.**

Vanilla `transformers` + `peft` + `bitsandbytes`. We lose ~2x training speed (45 min → 90 min for full GSM8K QLoRA) but gain :
- Clean PEFT namespace (no `.default.` / `language_model.` confusion)
- Standard `AutoModelForCausalLM` path (no multimodal wrapper)
- No compiled cache patches that hijack the attention forward
- Portability (anyone can reproduce without installing xformers etc.)
- Much shorter debug cycle when things break

We can reconsider Unsloth later **after** the full pipeline is proven to work vanilla. Do not introduce Unsloth as an optimization until the baseline is green.

**Principle 4 : Integration test in CI, not just unit tests.**

Add `tests/test_turboquant_integration.py` :
```python
def test_turboquant_end_to_end_generate():
    """Ensures TurboQuantCache works with model.generate() on the target model."""
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    tok = AutoTokenizer.from_pretrained(MODEL)
    cache = TurboQuantCache(dim=model.config.head_dim, bits=4, codebooks=build_codebooks())
    input_ids = tok("Hello, my name is", return_tensors="pt").input_ids.cuda()
    out = model.generate(input_ids=input_ids, max_new_tokens=20, past_key_values=cache)
    text = tok.decode(out[0])
    assert "Hello" in text  # sanity
    assert len(text) > len("Hello, my name is")  # actually generated
```

If this test ever fails, CI stops and the PR doesn't merge. This would have caught the entire issue in this session on day 1.

---

## 3. Concrete next-session plan

### Before starting
- [ ] Verify GPU is free : `nvidia-smi`
- [ ] Stop any running llama-server : `pkill -9 -f "llama-server.*8090"`
- [ ] Confirm venv works : `~/projects/efficient-llm-pipeline/.venv/bin/python --version`
- [ ] Decide on model : default choice is `meta-llama/Llama-3.2-3B-Instruct` unless the user has a preference

### Step 0 : clean slate
1. Create a new branch : `git checkout -b rebuild-integration-first`
2. Delete failed artifacts : `rm -rf checkpoints/smoke-test checkpoints/qwen35-smoke results/eval_Qwen3*.json unsloth_compiled_cache/`
3. Strip `_init_recurrent_states` from `polar_quant.py::TurboQuantCache` (pure-transformer target doesn't need it)
4. Update `configs/default.yaml` to new model

### Step 1 : `scripts/smoke_integration.py` (MUST PASS BEFORE STEP 2)
- Load base model via `AutoModelForCausalLM`
- Instantiate `TurboQuantCache` with correct `head_dim`
- Run `model.generate(input_ids=..., max_new_tokens=50, past_key_values=cache)`
- Assert output is coherent (length > prompt, contains expected tokens)
- Print VRAM usage, latency, compression ratio
- **If this fails, debug until it passes. Do not proceed to Step 2.**

### Step 2 : `scripts/eval.py` (replaces current)
- `AutoModelForCausalLM` + optional `PeftModel.from_pretrained` (vanilla, no Unsloth)
- `--turboquant` flag that passes `past_key_values=TurboQuantCache(...)` to `generate`
- Eval on GSM8K test split, 50 samples for smoke, 200 for full
- Output JSON to `results/`

### Step 3 : `scripts/train.py` (replaces current)
- Vanilla `transformers.Trainer` + `peft.get_peft_model` + `BitsAndBytesConfig` for 4-bit
- No Unsloth, no TRL SFTTrainer specifics (use `Trainer` with `DataCollatorForLanguageModeling`)
- Saves adapters in standard PEFT namespace

### Step 4 : Full benchmark run (background, ~2 h)
- `python scripts/train.py` → 45-90 min
- `python scripts/eval.py` (base) → 15 min
- `python scripts/eval.py --lora ./checkpoints/lora` → 15 min
- `python scripts/eval.py --lora ./checkpoints/lora --turboquant --bits 3` → 15 min
- `python scripts/eval.py --lora ./checkpoints/lora --turboquant --bits 4` → 15 min

### Step 5 : README rewrite
Real numbers. Architecture diagram. Theoretical vs measured compression ratio. Accuracy delta table. VRAM saved. Latency impact. Honest discussion of where TurboQuant loses quality.

---

## 4. Open questions to resolve at the start of the next session

1. **Which model** ? Default : `meta-llama/Llama-3.2-3B-Instruct` (need HF gated access token). Fallback if no token : `Qwen/Qwen2.5-3B-Instruct` (ungated).
2. **Which bits for TurboQuant** ? Test 2, 3, 4. Paper says 3 is the sweet spot but we verify.
3. **Which dataset** ? GSM8K is good for math, but HellaSwag or ARC might be better for showing quality preservation under compression. Keep GSM8K for now, add one more eval if time permits.
4. **Keep `src/serve/api.py`** ? Only if Step 5 is done with time to spare. Not on the critical path.
5. **Benchmark against what** ? Compare TurboQuant to : fp16 baseline (upper bound), int8 KV cache (`bitsandbytes` equivalent), int4 KV cache. This gives a proper spectrum.

---

## 5. Success criteria for the rebuild

The rebuild is **done** when :
- [ ] `python scripts/smoke_integration.py` passes
- [ ] `pytest tests/` shows ≥ 25/25 passing (existing) + new integration test passing
- [ ] README has a real benchmark table with at least these columns :
  - Configuration (fp16 / int8 / int4 / TurboQuant-3bit / TurboQuant-4bit)
  - GSM8K accuracy (%)
  - VRAM used (GB)
  - Latency per sample (s)
  - KV cache size (MB) for a 1K-token context
- [ ] `results/` contains the raw JSON for each run
- [ ] Fine-tuned LoRA adapter is saved and loadable via vanilla PEFT
- [ ] CI is green

The rebuild is **portfolio-ready** when the above is true AND :
- [ ] TurboQuant shows a non-trivial compression ratio (≥ 3x) with < 5% accuracy drop
- [ ] README has a paragraph explaining the algorithm (PolarQuant + QJL)
- [ ] There is one diagram (architecture or accuracy-vs-compression trade-off)

---

## 6. If TurboQuant still does not work on the new model

This is the worst case. If even a pure-transformer GQA model rejects `TurboQuantCache`, the problem is in `polar_quant.py`, not in the stack. Debug path :
1. Print shapes and dtypes at each step of `TurboQuantCache.update()`
2. Compare the dequantized K/V against the originals : they should be close in L2 norm
3. Check if transformers' `Cache` API changed between the version the code was written for (pre-5.0) and 5.4. The new API has different method signatures for `update`, `get_seq_length`, `reorder_cache`. `TurboQuantCache` inherits from `DynamicCache` — verify that `DynamicCache.update` still has the same signature.
4. As a last resort : stop inheriting from `DynamicCache` and implement the new `Cache` base class directly. This is the "right" fix but more work.

---

*Handoff written by Claude (this session) for the rebuild session. Good luck, future me.*
