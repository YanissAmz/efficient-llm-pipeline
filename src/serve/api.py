"""
FastAPI serving — efficient-llm-pipeline

Endpoints :
    POST /solve     — résout un problème mathématique avec le modèle fine-tuné
    GET  /health    — statut de l'API et du modèle
    GET  /info      — informations sur le modèle et la compression

Lancement :
    uvicorn src.serve.api:app --host 0.0.0.0 --port 8000

Variables d'environnement :
    LORA_PATH      : chemin vers les adaptateurs LoRA (défaut: ./models/qwen-gsm8k-lora)
    USE_TURBOQUANT : activer la compression KV cache (défaut: true)
    TURBOQUANT_BITS: bits de compression (défaut: 3)
    MAX_NEW_TOKENS : tokens max générés (défaut: 512)
"""

import os
import time
import logging
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schémas Pydantic
# ---------------------------------------------------------------------------

class SolveRequest(BaseModel):
    question: str = Field(..., description="Problème mathématique à résoudre", min_length=5)
    max_new_tokens: int = Field(512, ge=64, le=1024, description="Tokens max à générer")
    use_turboquant: Optional[bool] = Field(None, description="Override compression (None = valeur serveur)")


class SolveResponse(BaseModel):
    question: str
    answer: str
    final_answer: Optional[str]   # valeur extraite après ####
    latency_ms: float
    use_turboquant: bool
    compression_ratio: Optional[float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    vram_used_gb: Optional[float]


class InfoResponse(BaseModel):
    model_name: str
    lora_path: str
    use_turboquant: bool
    turboquant_bits: int
    compression_ratio: float
    max_new_tokens: int


# ---------------------------------------------------------------------------
# État global du serveur
# ---------------------------------------------------------------------------

class ModelState:
    model       = None
    tokenizer   = None
    codebooks   = None
    loaded      = False
    model_name  = "Qwen/Qwen3.5-2B"
    lora_path   = os.getenv("LORA_PATH", "./models/qwen-gsm8k-lora")
    use_tq      = os.getenv("USE_TURBOQUANT", "true").lower() == "true"
    tq_bits     = int(os.getenv("TURBOQUANT_BITS", "3"))
    max_tokens  = int(os.getenv("MAX_NEW_TOKENS", "512"))


state = ModelState()

SYSTEM_PROMPT = (
    "Tu es un assistant mathématique expert. "
    "Décompose chaque problème étape par étape en montrant tous les calculs. "
    "Termine toujours par '#### <réponse>' sur la dernière ligne."
)


# ---------------------------------------------------------------------------
# Chargement du modèle (lifespan)
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage, libère la VRAM à l'arrêt."""
    logger.info("Chargement du modèle...")
    try:
        from unsloth import FastLanguageModel
        from src.turboquant.polar_quant import build_codebooks

        state.model, state.tokenizer = FastLanguageModel.from_pretrained(
            model_name=state.lora_path,
            max_seq_length=1024,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(state.model)

        if state.use_tq:
            state.codebooks = build_codebooks()

        state.loaded = True
        vram = torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0
        logger.info(f"Modèle chargé | VRAM : {vram:.2f} GB | TurboQuant : {state.use_tq}")

    except Exception as e:
        logger.error(f"Échec chargement modèle : {e}")

    yield

    # Nettoyage
    if state.model is not None:
        del state.model
        del state.tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Modèle libéré.")


# ---------------------------------------------------------------------------
# Application FastAPI
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Efficient LLM Pipeline",
    description="Qwen3.5-2B fine-tuné sur GSM8K avec compression TurboQuant",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse)
def health():
    """Statut de l'API et du modèle."""
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
    """Informations sur le modèle et la configuration de compression."""
    return InfoResponse(
        model_name=state.model_name,
        lora_path=state.lora_path,
        use_turboquant=state.use_tq,
        turboquant_bits=state.tq_bits,
        compression_ratio=round(16 / state.tq_bits, 1),
        max_new_tokens=state.max_tokens,
    )


@app.post("/solve", response_model=SolveResponse)
def solve(req: SolveRequest):
    """
    Résout un problème mathématique avec raisonnement chain-of-thought.

    Retourne la solution complète (CoT) et la réponse finale extraite.
    """
    if not state.loaded:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    use_tq = req.use_turboquant if req.use_turboquant is not None else state.use_tq

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user",   "content": [{"type": "text", "text": req.question}]},
    ]

    inputs = state.tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
        "input_ids"      : inputs,
        "max_new_tokens" : req.max_new_tokens,
        "do_sample"      : False,
    }

    compression_ratio = None
    if use_tq and state.codebooks is not None:
        from src.turboquant.polar_quant import TurboQuantCache
        cfg = getattr(state.model.config, 'text_config', state.model.config)
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        cache = TurboQuantCache(dim=head_dim, bits=state.tq_bits, codebooks=state.codebooks)
        kwargs["past_key_values"] = cache
        compression_ratio = round(16 / state.tq_bits, 1)

    t0 = time.perf_counter()
    with torch.no_grad():
        out = state.model.generate(**kwargs)
    latency_ms = (time.perf_counter() - t0) * 1000

    response = state.tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True)

    from src.evaluate.metrics import extract_answer
    final_answer = extract_answer(response)

    return SolveResponse(
        question=req.question,
        answer=response,
        final_answer=final_answer,
        latency_ms=round(latency_ms, 1),
        use_turboquant=use_tq,
        compression_ratio=compression_ratio,
    )
