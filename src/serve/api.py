"""
FastAPI serving — efficient-llm-pipeline

Vanilla transformers + optional PEFT adapter + optional TurboQuant KV cache
compression. No Unsloth, no bnb. The server configuration is driven by env vars:

    MODEL_NAME      : HF model id (default: microsoft/Phi-4-mini-instruct)
    LORA_PATH       : optional path to a saved PEFT adapter
    USE_TURBOQUANT  : "true" / "false" (default: "true")
    TURBOQUANT_BITS : 3 or 4 recommended (default: "4")
    MAX_NEW_TOKENS  : default cap on generation length (default: "512")

Launch:
    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

Endpoints:
    GET  /health — load state + VRAM usage
    GET  /info   — model + cache configuration
    POST /solve  — chain-of-thought math solver
"""

import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.evaluate.metrics import extract_answer
from src.turboquant.polar_quant import TurboQuantCache, build_codebooks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SYSTEM_PROMPT = (
    "You are a careful math tutor. Solve the problem step by step, "
    "showing your reasoning. End your final line with `#### N` where "
    "N is the integer or decimal final answer (no units, no extra text)."
)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class SolveRequest(BaseModel):
    question: str = Field(..., description="Math problem to solve", min_length=5)
    max_new_tokens: int = Field(512, ge=32, le=1024)
    use_turboquant: bool | None = Field(
        None, description="Override server default (None = use server config)"
    )
    bits: int | None = Field(None, ge=2, le=4, description="Override TurboQuant bit budget")


class SolveResponse(BaseModel):
    question: str
    answer: str
    final_answer: str | None
    latency_ms: float
    new_tokens: int
    use_turboquant: bool
    bits: int | None
    compression_ratio: float | None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    vram_used_gb: float | None


class InfoResponse(BaseModel):
    model_name: str
    lora_path: str | None
    head_dim: int
    num_hidden_layers: int
    num_key_value_heads: int
    default_use_turboquant: bool
    default_bits: int
    default_compression_ratio: float
    max_new_tokens: int


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------


class ModelState:
    model = None
    tokenizer = None
    codebooks = None
    head_dim: int = 0
    loaded: bool = False

    model_name: str = os.getenv("MODEL_NAME", "microsoft/Phi-4-mini-instruct")
    lora_path: str | None = os.getenv("LORA_PATH") or None
    use_tq: bool = os.getenv("USE_TURBOQUANT", "true").lower() == "true"
    tq_bits: int = int(os.getenv("TURBOQUANT_BITS", "4"))
    max_tokens: int = int(os.getenv("MAX_NEW_TOKENS", "512"))


state = ModelState()


# ---------------------------------------------------------------------------
# Lifespan: load model at startup, free VRAM at shutdown
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"loading {state.model_name} (lora={state.lora_path}, tq={state.use_tq})")
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        state.tokenizer = AutoTokenizer.from_pretrained(state.model_name)
        state.model = AutoModelForCausalLM.from_pretrained(
            state.model_name,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            attn_implementation="sdpa",
        )
        if state.lora_path:
            from peft import PeftModel

            logger.info(f"loading LoRA adapter from {state.lora_path}")
            state.model = PeftModel.from_pretrained(state.model, state.lora_path)
            state.model = state.model.merge_and_unload()

        state.model.eval()

        cfg = state.model.config
        state.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        if state.use_tq:
            state.codebooks = build_codebooks(max_bits=4)

        state.loaded = True
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0
        logger.info(
            f"model loaded | vram={vram:.2f} GB | head_dim={state.head_dim} | "
            f"tq={state.use_tq} bits={state.tq_bits}"
        )
    except Exception as e:
        logger.exception(f"failed to load model: {e}")

    yield

    if state.model is not None:
        del state.model
        del state.tokenizer
        state.model = None
        state.tokenizer = None
        state.loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("model freed")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Efficient LLM Pipeline",
    description="Phi-4-mini-instruct + TurboQuant KV cache compression demo",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
def health():
    vram = None
    if torch.cuda.is_available() and state.loaded:
        vram = round(torch.cuda.memory_allocated() / 1e9, 2)
    return HealthResponse(
        status="ok" if state.loaded else "model_not_loaded",
        model_loaded=state.loaded,
        device="cuda" if torch.cuda.is_available() else "cpu",
        vram_used_gb=vram,
    )


@app.get("/info", response_model=InfoResponse)
def info():
    if not state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")
    cfg = state.model.config
    return InfoResponse(
        model_name=state.model_name,
        lora_path=state.lora_path,
        head_dim=state.head_dim,
        num_hidden_layers=cfg.num_hidden_layers,
        num_key_value_heads=cfg.num_key_value_heads,
        default_use_turboquant=state.use_tq,
        default_bits=state.tq_bits,
        default_compression_ratio=round(16 / state.tq_bits, 2),
        max_new_tokens=state.max_tokens,
    )


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    if not state.loaded:
        raise HTTPException(status_code=503, detail="model not loaded")

    use_tq = req.use_turboquant if req.use_turboquant is not None else state.use_tq
    bits = req.bits if req.bits is not None else state.tq_bits

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": req.question},
    ]
    prompt = state.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    input_ids = state.tokenizer(prompt, return_tensors="pt").input_ids
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    gen_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=req.max_new_tokens,
        do_sample=False,
        pad_token_id=state.tokenizer.eos_token_id,
    )

    compression_ratio = None
    if use_tq:
        if state.codebooks is None:
            state.codebooks = build_codebooks(max_bits=4)
        cache = TurboQuantCache(dim=state.head_dim, bits=bits, codebooks=state.codebooks)
        gen_kwargs["past_key_values"] = cache
        compression_ratio = round(16 / bits, 2)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = state.model.generate(**gen_kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    n_new = int(out.shape[1] - input_ids.shape[1])
    response = state.tokenizer.decode(out[0, input_ids.shape[1] :], skip_special_tokens=True)
    final_answer = extract_answer(response)

    return SolveResponse(
        question=req.question,
        answer=response,
        final_answer=final_answer,
        latency_ms=round(latency_ms, 1),
        new_tokens=n_new,
        use_turboquant=use_tq,
        bits=bits if use_tq else None,
        compression_ratio=compression_ratio,
    )
