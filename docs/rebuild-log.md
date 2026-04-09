# Rebuild log — 2026-04-09

Chronological record of the integration-first rebuild session. Companion file
to `docs/rebuild-plan.md` (the *plan*) and `README.md` (the *outcome*). This
file is the *session log* : what happened, in which order, what we learned,
what's next.

---

## 0. Context going in

- Previous attempt was scaffolded for Colab T4 + Qwen3.5-2B fp16 vanilla, then
  ported onto RTX 3090 + modern HuggingFace stack. TurboQuant — the project's
  actual differentiator — was blocked on integration, not on math.
- Four documented integration blockers were known : Unsloth compiled patches,
  SDPA strict layout, PEFT namespace mismatch, multimodal config nesting.
- Decision : clean rebuild with **integration-first** ordering. The very first
  script would be a smoke test that exercises `model.generate(past_key_values=
  TurboQuantCache(...))` on the target model. No downstream code before that
  is green.
- Rebuild plan : `docs/rebuild-plan.md`.
- Target model after discussion : `microsoft/Phi-4-mini-instruct` (3.8B, pure
  transformer GQA, 32 layers, 8 KV heads, head_dim=128 — recent, portfolio-
  friendly, no SWA / multimodal / hybrid weirdness).

---

## 1. Session steps (chronological)

### Step 0+1 — clean slate + smoke + two hidden bugs (`38218a6`)

Clean slate :
- Branch `rebuild-integration-first` from `main`
- Discard the two previous-session uncommitted patches (`Qwen3-4B → Qwen3.5-4B`
  in `configs/default.yaml`, `.contiguous()` SDPA band-aid in `polar_quant.py`)
- Delete stale artefacts (`checkpoints/qwen35-smoke`, `unsloth_compiled_cache/`,
  `results/eval_Qwen3*.json`, broken `scripts/{train,eval,retest_t3}.py`)
- Strip `_init_recurrent_states` (GatedDeltaNet hooks for Qwen3.5 only) from
  `TurboQuantCache`
- Update `configs/default.yaml` to `microsoft/Phi-4-mini-instruct`

Smoke (`scripts/smoke_integration.py`, ~100 lines) :
- Loads Phi-4-mini in bf16 + `attn_implementation="sdpa"` explicit
- Asserts non-nested config (catches multimodal wrappers immediately)
- Builds `TurboQuantCache` with the model's `head_dim`
- Runs both a baseline and a TurboQuant `model.generate` and prints a side-by-
  side report

**Smoke "passed" on the first try — but the TurboQuant output was gibberish.**
Baseline produced *"Hello, my name is John Smith. I am a 35-year-old male
living in New York City..."*, TurboQuant at 3 bits produced *"Hello, my name
is near 201Dem.D *B]..."*. That triggered a deeper diagnosis.

#### Bug #1 — `DynamicCache` API change in `transformers ≥ 5.4`

Symptom : TurboQuantCache update returned the right shapes and the model
didn't crash, but output was fluent garbage.

Root cause : `DynamicCache` in 5.4 no longer exposes `self.key_cache` /
`self.value_cache` as list attributes. State now lives in
`self.layers: list[DynamicLayer]`, where each layer has its own `keys`,
`values`, `is_initialized` fields. The old `TurboQuantCache.__init__` did
`self.key_cache = []`, which was harmless (the attribute just sat there
unused by the parent) — but `Cache.get_seq_length()` and the attention
masking machinery now read from `self.layers[layer_idx]`, which was empty.
So the model was computing attention over an empty cached context while
our overridden `update()` silently wrote to a phantom attribute.

Fix : create a `TurboQuantLayer(DynamicLayer)` that stores the compressed
representation (`idx`, `qjl_bits`, `r_norm`, `mse_norm`) in-place and writes
the dequantized full `keys` / `values` back into the parent's attributes so
`get_seq_length()` and masking work. `TurboQuantCache.__init__` then swaps
`self.layer_class_to_replicate` to `partial(TurboQuantLayer, quantizer=self.
_quantizer)` so the parent's lazy layer creation uses our subclass.

After this fix the output was *less* random but still gibberish. That revealed
the second bug.

#### Bug #2 — `TurboQuantMSE` assumed unit-norm inputs

Symptom : after fixing bug #1, the cache was wired correctly but TurboQuant
reconstruction on real K/V was still unusable. Diagnostic script (`scripts/
diag_quantize_real_kv.py`) showed :

| bits | real K/V rel_l2 (before fix) | cos_sim (before fix) |
|---|---|---|
| 2 | 1.20 | 0.64 |
| 3 | 1.18 | 0.65 |
| 4 | 1.16 | 0.66 |

`rel_l2 > 1` means the reconstruction is *further from the input than zero*.
Barely any improvement from 2 to 4 bits — classic sign of *clipping*, not of
*precision*.

Root cause : the old `TurboQuantMSE.__init__` had `self.scale = 1.0 / (dim **
0.5)`, and `quantize()` did `y_norm = y / self.scale` (i.e. `y * sqrt(d)`).
This is only correct if the input vectors have unit norm (`‖x‖ = 1`). Real
Phi-4-mini K/V have `std ~ 2.7`, so `‖x‖ ~ 30`. The values fed to the
Lloyd-Max codebook (which was trained on `N(0,1)` percentiles 1-99, i.e.
`c ∈ [-2.3, 2.3]`) were in `[-30, 30]`, and every coordinate got clipped to
the extreme codebook entry.

The unit tests passed because `test_compression_reduces_error` only checked
`errors[4] < errors[2]` — a relation that holds even when every bit budget
produces useless reconstructions, because clipping is monotone in the number
of levels.

Fix : normalize each vector to unit norm in `TurboQuantMSE.quantize()`, store
`‖x‖` alongside the indices, and denormalize at dequantize time. The storage
tuple per vector went from 3 elements `(idx, qjl_bits, r_norm)` to 4 elements
`(idx, qjl_bits, r_norm, mse_norm)`. `TurboQuantProd`, `TurboQuantCache`, and
the existing tests were all updated to match.

After this fix :

| bits | real K/V rel_l2 (after fix) | cos_sim (after fix) |
|---|---|---|
| 2 | 0.73 | 0.81 |
| 3 | 0.42 | 0.92 |
| 4 | **0.22** | **0.98** |

Smoke output after both fixes, at 3 bits : *"Hello, my name is John, I am a
student at the University of Toronto. I am writing this post to say that I am
a student at the University of Toronto, and I am writing a review of the
article 'The Role of the Media in the 21st'"*. At 4 bits : identical to
baseline on the first tokens (*"Hello, my name is John Smith. I am a 35-year-
old male"*) then a small divergence.

Added two new anti-regression tests (`test_reconstruction_quality_high_dim`,
`test_unbiased_inner_product`) so this class of bug can't come back silently.
**Lesson : quantization unit tests must measure absolute quality at realistic
dimensions, not just relative monotonicity.**

### Step 2 — GSM8K eval matrix (`5f471eb`)

`scripts/eval.py` : vanilla `AutoModelForCausalLM` + optional TurboQuant via
`--turboquant --bits N` flag. GSM8K `test` split, chat-template prompt, greedy
decoding, per-sample result saved to JSON.

50-sample GSM8K results on Phi-4-mini :

| Config | Compression | Accuracy | Avg latency | Avg new tokens |
|---|---|---|---|---|
| fp16 base | 1.00× | 90.0% (45/50) | 2.75s | 192 |
| TQ 4-bit | 4.00× | 80.0% (40/50) | 5.47s | 184 |
| TQ 3-bit | 5.33× | 0.0% (0/50, 46 no-answer) | 7.86s | 274 |

**Finding #1 emerged** : TurboQuant 3-bit *collapses* on multi-step arithmetic
reasoning — the model loses the ability to close a chain-of-thought (46/50 fail
to produce a `####` final answer). TQ 4-bit is a clean –10 point drop.

### Step 3 — vanilla LoRA + 6-config matrix (`8ca045b`)

`scripts/train.py` : bf16 LoRA with `peft.get_peft_model` (r=16, alpha=32,
target `q/k/v/o_proj`), vanilla `transformers.Trainer`, no Unsloth, no bnb
4-bit (Phi-4-mini in bf16 fits in 24 GB with gradient checkpointing; there's
no practical gain from 4-bit quantization on this model, and the simpler path
avoids two classes of integration bugs).

LoRA-v1 : lr=2e-4, 1 epoch, train loss 0.71 → 0.45 (final), 17 min, peak 13 GB.

Eval on 50 GSM8K samples with LoRA-v1 :

| Config | Accuracy | Avg new tokens |
|---|---|---|
| fp16 base | 90.0% | 192 |
| fp16 base + LoRA-v1 | **76.0%** (–14) | **99** (–48%) |
| TQ 4-bit + LoRA-v1 | 54.0% (–36) | 116 |

**Finding #2 emerged** : naïve LoRA on a strong instruct model *hurts*
accuracy. The LoRA model produces much shorter answers (99 vs 192 tokens on
average) and arrives at wrong numbers *via the right method* — it adopts the
GSM8K surface format (`<<X*Y=Z>>` calculator annotations, `#### N` ending) but
loses some plain-English reasoning robustness.

Tested a gentler LoRA-v2 (lr=5e-5, 0.5 epoch) to rule out hyperparam artefacts :

| Config | Accuracy | Avg new tokens |
|---|---|---|
| fp16 base + LoRA-v2 | **76.0%** (same) | 103 (same) |
| TQ 4-bit + LoRA-v2 | 60.0% (slightly better than v1) | 117 |

v1 and v2 give **identical** base accuracy with identical output length, so
the degradation is structural. Kept both results honestly.

### Step 4 — HellaSwag loglikelihood eval (`3b94b84`)

Hypothesis driving this step : "TurboQuant 3-bit collapses on GSM8K because
multi-step math is exceptionally demanding. It might survive easier tasks."

`scripts/eval_hellaswag.py` : loglikelihood-based scoring of the 4 HellaSwag
endings. Uses a *2-pass forward* pattern to actually exercise the cache :

1. `model.forward(ctx, past_key_values=cache, use_cache=True)` — populates
   the cache with compressed ctx K/V
2. `model.forward(ending, past_key_values=cache_after_ctx, use_cache=True)` —
   ending attention reads the *cached* (dequantized-from-compressed) ctx

This is what makes TurboQuant matter on loglikelihood tasks. A single forward
on the concatenated `(ctx + ending)` would not : inside one pass, attention
reads freshly computed K/V, not the cache.

A subtle bug surfaced in the baseline run : when `cache=None`, the model
creates a fresh `DynamicCache` internally and returns it as `out.past_key_
values`, but my script was passing `cache=None` to both passes, so pass 2
saw no context and scored endings in isolation. Base accuracy was 38%.

Fix : `cache = out_ctx.past_key_values` between the two passes. Base jumped
to 62% (length-normalized).

Results on 50 HellaSwag samples :

| Config | Accuracy | vs base |
|---|---|---|
| fp16 base | 62.0% | — |
| TQ 4-bit | 60.0% | –2 |
| TQ 3-bit | 52.0% | –10 |

**Finding #3 emerged** : TurboQuant's break point is *task-dependent*, not
bit-budget-dependent. The same 3-bit setting collapses on GSM8K but is
usable on HellaSwag. The operating point depends on workload, not just on
the bit budget.

### Step 5 — README rewrite (`f192523`)

Replaced the aspirational Qwen3-4B / QLoRA / Unsloth README with an honest
current-state write-up : headline numbers, three findings, the two bugs the
rebuild fixed, a reproducibility section, and a limitations section. CLAUDE.md
updated too (gitignored, kept local).

### CI integration test (`2c4eb04`)

`tests/test_turboquant_integration.py` : five tests that instantiate a tiny
2-layer `LlamaForCausalLM` from a config (no model download, runs in CI in a
few seconds) and exercise the full `model.generate(past_key_values=Turbo
QuantCache(...))` path plus the 2-pass forward pattern. Specifically designed
so that each of the two bugs above would have tripped at least one of these
tests.

Suite is now **32/32** (13 metrics + 14 unit + 5 integration).

### CI all-branches + ruff clean (`32874fc`)

`.github/workflows/ci.yml` triggers on push to any branch (not just main) so
feature branches get validated *before* opening a PR. Fixed two B905 and one
RUF059 warning flagged by `ruff check`.

### Merge to main + PR #1 (`9e30961`)

Opened PR #1 with the full rebuild story in the description. PR CI ran
clean (lint + test, 4/4 checks SUCCESS) and was merged with `--merge`
(not squash) so the step-by-step history is preserved on `main`.

### Step 6 + cross-architecture validation (`a8ead28`)

**Cross-arch on Qwen2.5-3B-Instruct** (3B, **2 KV heads** vs Phi's 8, **36
layers** vs 32, head_dim=128). Ungated so no token required.

Quick diagnostic on real K/V reconstruction before running the matrix, to
know whether TurboQuant's per-vector math cares about the architecture :

| Model | bits | rel_l2 | cos_sim |
|---|---|---|---|
| Phi-4-mini layer 0 | 4 | 0.22 | 0.98 |
| Qwen2.5-3B layer 0 | 4 | 0.27 | 0.96 |

Nearly identical. Then the 50-sample GSM8K matrix :

| Config | accuracy | no_answer | avg new tokens |
|---|---|---|---|
| Qwen2.5-3B fp16 base | 70.0% (35/50) | 8 | 265 |
| Qwen2.5-3B + TQ 4-bit | **0.0% (0/50)** | **50** | 410 |
| Qwen2.5-3B + TQ 3-bit | 0.0% (0/50) | 50 | 512 (hard cap) |

**Finding #4 emerged** : TurboQuant's downstream task accuracy is *also*
architecture-dependent, and per-vector reconstruction metrics lie about it.
On Phi-4-mini TQ-4 costs 10 points; on Qwen2.5-3B TQ-4 is a total collapse.
Per-vector reconstruction cos_sim is essentially identical on both models,
so the gap is pure error compounding. Three architectural properties explain
the difference :

1. **2 KV heads vs 8 KV heads.** Qwen has 16 Q heads reading from only 2 KV
   heads, so every per-KV reconstruction error is amplified by 8× (vs ~3×
   on Phi). More Q heads share the same corrupted cache.
2. **36 layers vs 32.** Errors compound over 4 more residual-stream layers.
3. **Early-layer activation outliers.** On Qwen2.5-3B layer 0, `k.abs.max =
   92.5` with `k.std = 15.4`. On Phi-4-mini layer 0, `k.abs.max = 16.9`
   with `k.std = 2.7`. After per-vector normalization, the Lloyd-Max
   codebook — calibrated on `N(0,1)` percentiles 1–99, so `|c| ≤ 2.3` — is
   looking at values deeper in its tails on Qwen. The random rotation helps
   spread outliers but does not eliminate them.

**Per-vector L2 / cosine is a necessary but not sufficient metric for a
KV-cache quantizer.** This is the most non-obvious of the four findings, and
the one a casual reader would miss without the cross-arch comparison.

**FastAPI rewrite** (`src/serve/api.py`) : replaced the old Unsloth-based
version with a clean vanilla `AutoModelForCausalLM` + optional `PeftModel` +
optional `TurboQuantCache` server. Env-driven (`MODEL_NAME`, `LORA_PATH`,
`USE_TURBOQUANT`, `TURBOQUANT_BITS`). Per-request override of
`use_turboquant` and `bits` so a single running instance can A/B test
compression settings on the fly.

Live smoke test on Phi-4-mini + TQ4 at port 8765 :

- `GET /health` → `{"status": "ok", "model_loaded": true, "vram_used_gb": 7.67}`
- `GET /info` → full arch report (hidden_size, layers, KV heads, default bits)
- `POST /solve` (train speed question) →
  - baseline : `#### 60` in 962 ms, 66 tokens, `final_answer = "60"` ✓
  - TQ4 : `#### N1` in 2179 ms (2.3× slower), 61 tokens, `final_answer =
    "1"` — same failure mode as in the eval (correct free-text reasoning
    arrives at 60, but the `#### N` final format is corrupted)

This reproduces the –10 point GSM8K gap *live*, in the demo endpoint.

### Attempted scale-up (killed)

Launched a 150-sample Phi-4-mini run (GSM8K + HellaSwag × 3 configs) in
background and killed it ~13 min in. The scale-up would have produced
tighter error bars on numbers we already had, not any new finding. It was
the wrong use of time and was stopped deliberately.

---

## 2. Current state (snapshot at `a8ead28`)

### Repo layout

```
src/
  turboquant/
    polar_quant.py           PolarQuant + QJL + TurboQuantLayer(DynamicLayer) + TurboQuantCache(DynamicCache)
    qjl.py                   standalone 1-bit unbiased QJL quantizer
  evaluate/metrics.py        GSM8K answer extraction, exact-match, batch metrics
  serve/api.py               FastAPI server, env-driven, per-request TQ/LoRA override
scripts/
  smoke_integration.py       Step 1 — cache + generate smoke
  diag_quantize_real_kv.py   reconstruction-quality diagnostic on real K/V
  train.py                   Step 3 — vanilla LoRA bf16 training
  eval.py                    Step 2 — GSM8K eval, --lora-path and --turboquant flags
  eval_hellaswag.py          Step 4 — HellaSwag loglikelihood with 2-pass forward
  aggregate_eval.py          Markdown comparison-table builder
tests/
  test_metrics.py            13 tests
  test_turboquant.py         14 tests (12 original + 2 new anti-regression)
  test_turboquant_integration.py  5 integration tests (tiny Llama, no download)
configs/default.yaml
docs/
  rebuild-plan.md            the plan
  rebuild-log.md             THIS FILE (session log)
results/                     12 JSON eval outputs + comparison.md
```

### Test suite

**32/32 green** in CI (`lint` + `test` jobs). Entire suite runs in ~15 s.

### Eval results in `results/`

Phi-4-mini-instruct, 50 samples each :

- `eval_phi4mini_{base,tq4,tq3}_n50.json` — GSM8K base / TQ-4 / TQ-3
- `eval_phi4mini_{lora,lorav2}_{,tq4_,tq3_}n50.json` — LoRA v1 / v2, each × { base, TQ-4, TQ-3 }
- `eval_phi4mini_hellaswag_{base,tq4,tq3}_n50.json` — HellaSwag base / TQ-4 / TQ-3

Qwen2.5-3B-Instruct, 50 samples each :

- `eval_qwen25-3b_{base,tq4,tq3}_n50.json` — GSM8K

### Headline numbers

```
                                    GSM8K          HellaSwag
Phi-4-mini fp16 base                90.0%          62.0%
Phi-4-mini TurboQuant 4-bit         80.0% (-10)    60.0% (-2)
Phi-4-mini TurboQuant 3-bit          0.0% (-90)    52.0% (-10)
Phi-4-mini LoRA-v2 (gentle)         76.0% (-14)    —
Phi-4-mini LoRA-v2 + TQ-4           60.0% (-30)    —

Qwen2.5-3B fp16 base                70.0%          —
Qwen2.5-3B TurboQuant 4-bit          0.0% COLLAPSE —
Qwen2.5-3B TurboQuant 3-bit          0.0%          —
```

### Git + CI

- Branch `main` at `a8ead28`, in sync with `origin/main`
- 9 commits from the rebuild (squashed into the graph below), including the
  merge commit of PR #1
- CI passes on `main` (and on every branch since `32874fc`)

```
a8ead28  Step 6 + cross-arch: FastAPI serving + Qwen2.5-3B reveals arch-dependent collapse
9e30961  Merge pull request #1 from YanissAmz/rebuild-integration-first
32874fc  ci: run on all branches + appease ruff lint/format
2c4eb04  Add CI integration test for TurboQuantCache + model.generate
f192523  Step 5: rewrite README around the actual rebuild and findings
3b94b84  Step 4: HellaSwag loglikelihood eval reveals task-dependent compression limits
8ca045b  Step 3: vanilla LoRA training + 6-config eval matrix on GSM8K
5f471eb  Step 2: vanilla HF eval on GSM8K, base/TQ4/TQ3 50-sample comparison
38218a6  Step 0+1: clean slate + integration-first smoke on Phi-4-mini
```

---

## 3. The four findings in one place

1. **TurboQuant 4-bit is a viable operating point on the right architecture.**
   4× KV cache compression for a –2 to –10 accuracy point cost on Phi-4-mini.
2. **The break point of TurboQuant is task-dependent, not bit-budget-
   dependent.** TQ-3 collapses on multi-step arithmetic but survives single-
   step commonsense scoring. The right operating point depends on workload.
3. **TurboQuant is architecture-dependent, and per-vector L2 lies about it.**
   Identical reconstruction `cos_sim` on Phi-4-mini and Qwen2.5-3B, but TQ-4
   collapses on Qwen because of (a) aggressive GQA amplifying per-KV errors,
   (b) more residual layers compounding them, and (c) extreme early-layer
   activation outliers that live deep in the Lloyd-Max codebook tails.
4. **Naïve LoRA fine-tuning of a strong instruct model can hurt.** Two
   hyperparam regimes on Phi-4-mini both land at the same –14 pts on GSM8K
   by teaching the model the *surface format* of GSM8K answers while
   degrading the plain-English reasoning that made it work in the first place.

---

## 4. What's next — plan

Ordered by value-per-minute for the portfolio, not by how fun they are to
build.

### High value, small cost

- [ ] **Figure : accuracy vs compression for both models**, one plot per task,
      with a clear "collapse" marker where the curve falls off. This is the
      single thing that would make the four findings visible in under 10
      seconds. `results/` has all the data; one `matplotlib` script.
- [ ] **Dockerfile** for `src/serve/api.py`. Multi-stage, non-root user, bf16
      Phi-4-mini + optional LoRA. Probably 60-80 lines. Makes the project
      demo-able on any machine with an NVIDIA GPU and Docker.
- [ ] **`tests/test_turboquant_integration.py::test_qwen_collapses`**, an
      *intentional* failing-style test documenting the Qwen collapse, so
      anyone re-running the suite sees the finding actively. Needs a model
      download → guarded behind an env var (`RUN_SLOW_TESTS=1`) to keep CI
      fast.
- [ ] **`make demo`** target that starts the FastAPI server, waits for
      `/health`, then posts two `/solve` requests (one with TQ on, one off)
      and prints the diff. Ten-line make recipe, enormous onboarding win.

### Medium value, medium cost

- [ ] **Outlier-aware calibration** for the Lloyd-Max codebook. Fit a
      *per-layer* scale on a small calibration set so the codebook sees
      values inside its trained range on Qwen-style models. This is the
      natural fix for finding #3 and would be a meaningful extension of the
      paper, not just a reimplementation.
- [ ] **Custom CUDA kernel** for fused dequant + attention. This is where the
      *real* VRAM and latency wins live — currently the implementation stores
      both compressed indices and dequantized K/V, so VRAM is unchanged. The
      kernel would dequantize on the fly inside the attention softmax, turning
      the theoretical 4× compression into an actual 4× VRAM win.
- [ ] **Broader cross-arch sweep** : Llama-3.2-3B (gated), Mistral-7B-Instruct,
      Gemma-2-2B. One row per model with `{num_kv_heads, num_layers,
      early_abs_max, TQ4 GSM8K, TQ4 HellaSwag}`. Plot this as a scatter of
      "collapse risk" vs "KV head amplification factor".

### Nice to have, low priority

- [ ] Full GSM8K test (1319 samples) and full HellaSwag validation (10042
      samples) for tighter error bars. ~2 h of GPU per model.
- [ ] Streamlit demo wrapping the FastAPI server.
- [ ] Short blog post / README hero section with an architecture diagram and
      an animated chart.
- [ ] Re-enable Unsloth behind a flag, *after* the vanilla baseline stays
      green, purely as a speed optimization.

### Explicitly out of scope for this project

- A paper, a preprint, or any claim of novelty on the TurboQuant method
  itself. The contribution here is an honest *implementation + evaluation*
  of a published method, not a new method.

---

*Session run on 2026-04-09 on the tower with RTX 3090 24 GB, Python 3.12,
torch 2.10+cu128, transformers 5.4, peft 0.18, datasets 4.3.*
