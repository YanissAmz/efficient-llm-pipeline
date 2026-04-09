"""
TurboQuant — KV Cache compression
Implémentation de TurboQuantMSE et TurboQuantProd depuis le paper Google (arXiv:2504.19874).

Usage:
    from src.turboquant.polar_quant import TurboQuantProd, TurboQuantCache, build_codebooks

    codebooks = build_codebooks()
    quant = TurboQuantProd(dim=128, bits=3, codebooks=codebooks)
    cache = TurboQuantCache(dim=128, bits=3, codebooks=codebooks)
"""

import math
from functools import partial

import numpy as np
import torch
import torch.nn as nn
from transformers import DynamicCache
from transformers.cache_utils import DynamicLayer

# ---------------------------------------------------------------------------
# Codebooks Lloyd-Max
# ---------------------------------------------------------------------------


def build_codebooks(max_bits: int = 4, n_samples: int = 100_000) -> dict:
    """
    Précalcule les codebooks Lloyd-Max optimaux pour N(0,1), b=1..max_bits.

    Returns:
        dict {b: np.ndarray de shape (2^b,)}
    """
    codebooks = {}
    for b in range(1, max_bits + 1):
        n_levels = 2**b
        samples = np.random.randn(n_samples)
        centroids = np.linspace(
            np.percentile(samples, 1),
            np.percentile(samples, 99),
            n_levels,
        )
        for _ in range(200):
            assigned = np.argmin(np.abs(samples[:, None] - centroids[None, :]), axis=1)
            new_centroids = np.array(
                [
                    samples[assigned == k].mean() if (assigned == k).any() else centroids[k]
                    for k in range(n_levels)
                ]
            )
            if np.max(np.abs(new_centroids - centroids)) < 1e-6:
                break
            centroids = new_centroids
        codebooks[b] = np.sort(centroids)
    return codebooks


# ---------------------------------------------------------------------------
# TurboQuantMSE
# ---------------------------------------------------------------------------


class TurboQuantMSE(nn.Module):
    """
    Quantificateur MSE-optimal : normalisation + rotation aléatoire + Lloyd-Max.

    Étapes :
        1. n = ‖x‖₂           (norme par vecteur, stockée séparément)
        2. u = x / n          (vecteur unitaire)
        3. y = Π·u            (rotation orthogonale)  →  y_j ~ N(0, 1/d)
        4. ỹ = y · √d         →  ỹ_j ~ N(0, 1)
        5. idx = argmin_k |ỹ_j - c_k|  (codebook Lloyd-Max sur N(0,1))
        6. ŷ = c_{idx} / √d   (reconstruction unitaire)
        7. x̂ = n · (Πᵀ · ŷ)   (rescale par la norme stockée)

    Le scaling par ‖x‖ est crucial : sans lui le codebook clippe massivement
    pour des vecteurs non-unitaires (ex : K/V d'un LLM avec std ~2.7).
    """

    def __init__(self, dim: int, bits: int, codebooks: dict, seed: int = 42):
        super().__init__()
        self.dim = dim
        self.bits = bits
        self.sqrt_d = dim**0.5

        torch.manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.register_buffer("rotation", Q)

        cb = torch.tensor(codebooks[bits], dtype=torch.float32)
        self.register_buffer("codebook", cb)

    def quantize(self, x: torch.Tensor):
        """x : (..., dim)  →  (idx, x_norm, x_hat)"""
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        x_norm = x_flat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        u = x_flat / x_norm

        y = u @ self.rotation.T
        y_scaled = y * self.sqrt_d  # → coordinates ~ N(0, 1)

        dists = (y_scaled.unsqueeze(-1) - self.codebook).abs()
        idx = dists.argmin(dim=-1)

        y_hat_scaled = self.codebook[idx]
        y_hat = y_hat_scaled / self.sqrt_d
        u_hat = y_hat @ self.rotation
        x_hat = u_hat * x_norm

        return (
            idx.reshape(shape),
            x_norm.reshape(*shape[:-1], 1),
            x_hat.reshape(shape),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, x_hat = self.quantize(x)
        return x_hat


# ---------------------------------------------------------------------------
# TurboQuantProd
# ---------------------------------------------------------------------------


class TurboQuantProd(nn.Module):
    """
    Quantificateur inner-product unbiaisé : TurboQuantMSE(b-1) + QJL 1-bit.

    Garantie théorique : E[<y, x̃>] = <y, x>  pour tout y.

    Args:
        dim      : dimension des vecteurs (ex: head_dim du modèle)
        bits     : bits totaux par coordonnée (b-1 pour MSE + 1 pour QJL)
        codebooks: dict retourné par build_codebooks()
        seed     : graine pour la reproductibilité
    """

    def __init__(self, dim: int, bits: int, codebooks: dict, seed: int = 42):
        super().__init__()
        assert bits >= 2, "bits >= 2 requis (b-1 pour MSE + 1 pour QJL)"
        self.dim = dim
        self.bits = bits
        self.qjl_scale = math.sqrt(math.pi / 2) / dim

        self.mse = TurboQuantMSE(dim, bits - 1, codebooks, seed=seed)

        torch.manual_seed(seed + 1)
        self.register_buffer("S", torch.randn(dim, dim))

    def quantize(self, x: torch.Tensor):
        """
        x : (..., dim)
        →  (idx, qjl_bits, residual_norm, mse_norm, x_hat)
        """
        shape = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        idx, mse_norm, x_mse_flat = self.mse.quantize(x_flat)
        x_mse = x_mse_flat  # already (N, dim) since x_flat was

        r = x_flat - x_mse
        r_norm = r.norm(dim=-1, keepdim=True)

        qjl_bits = (r @ self.S.T).sign().to(torch.int8)
        correction = self.qjl_scale * r_norm * (qjl_bits.float() @ self.S)
        x_hat = x_mse + correction

        return (
            idx.reshape(shape),
            qjl_bits.reshape(shape),
            r_norm.reshape(*shape[:-1], 1),
            mse_norm.reshape(*shape[:-1], 1),
            x_hat.reshape(shape),
        )

    def dequantize(self, idx, qjl_bits, r_norm, mse_norm) -> torch.Tensor:
        """Reconstruction depuis les données compressées stockées."""
        shape = idx.shape
        idx_flat = idx.reshape(-1, self.dim)
        y_hat_scaled = self.mse.codebook[idx_flat]
        y_hat = y_hat_scaled / self.mse.sqrt_d
        u_hat = y_hat @ self.mse.rotation
        x_mse = u_hat * mse_norm.reshape(-1, 1)

        correction = (
            self.qjl_scale
            * r_norm.reshape(-1, 1)
            * (qjl_bits.reshape(-1, self.dim).float() @ self.S)
        )
        return (x_mse + correction).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *_, x_hat = self.quantize(x)
        return x_hat


# ---------------------------------------------------------------------------
# TurboQuantCache
# ---------------------------------------------------------------------------


class TurboQuantLayer(DynamicLayer):
    """
    Per-layer TurboQuant cache. Drop-in replacement for `DynamicLayer`.

    Compresses incoming K/V into a packed (idx, qjl_bits, r_norm) representation.
    On every update, materializes the dequantized full K/V tensors back into
    `self.keys` / `self.values` so the parent `Cache` machinery (get_seq_length,
    masking, position ids, ...) sees the correct shapes.
    """

    def __init__(self, quantizer: "TurboQuantProd"):
        super().__init__()
        self.quantizer = quantizer
        # Stored as 4-tuples: (idx, qjl_bits, r_norm, mse_norm)
        self._comp_k = None
        self._comp_v = None

    def get_seq_length(self, cache_position=None) -> int:
        if self._comp_k is None:
            return 0
        return self._comp_k[0].shape[-2]

    def update(self, key_states, value_states, *args, **kwargs):
        # Move quantizer to the right device on first call
        if any(b.device != key_states.device for b in self.quantizer.buffers()):
            self.quantizer = self.quantizer.to(key_states.device)

        dtype = key_states.dtype

        # Quantize new chunk
        k_idx, k_qjl, k_rnorm, k_mnorm, _ = self.quantizer.quantize(key_states)
        v_idx, v_qjl, v_rnorm, v_mnorm, _ = self.quantizer.quantize(value_states)

        new_k = (k_idx, k_qjl, k_rnorm, k_mnorm)
        new_v = (v_idx, v_qjl, v_rnorm, v_mnorm)

        if self._comp_k is None:
            self._comp_k = new_k
            self._comp_v = new_v
        else:
            self._comp_k = tuple(
                torch.cat([old, new], dim=-2) for old, new in zip(self._comp_k, new_k, strict=True)
            )
            self._comp_v = tuple(
                torch.cat([old, new], dim=-2) for old, new in zip(self._comp_v, new_v, strict=True)
            )

        # Dequantize FULL accumulated state
        full_k = self.quantizer.dequantize(*self._comp_k).to(dtype)
        full_v = self.quantizer.dequantize(*self._comp_v).to(dtype)

        # Populate parent attrs so get_seq_length / masking work
        self.keys = full_k
        self.values = full_v
        self.is_initialized = True

        return full_k, full_v


class TurboQuantCache(DynamicCache):
    """
    KV cache compressed with TurboQuant (PolarQuant + QJL).
    Drop-in replacement for HuggingFace `DynamicCache` (transformers >= 5.4 API).

    On every layer update : compresses K and V before storage, returns the
    dequantized versions for attention (unbiased in expectation).

    Args:
        dim      : head_dim of the model (hidden_size // num_attention_heads)
        bits     : compression bits (3 = good speed/quality tradeoff)
        codebooks: dict returned by build_codebooks()
    """

    def __init__(self, dim: int, bits: int, codebooks: dict):
        # Build the shared quantizer first — all layers reuse the same rotation/codebook/S
        self._quantizer = TurboQuantProd(dim, bits, codebooks)
        self.dim = dim
        self.bits = bits
        self.bits_original = 0

        # Initialize the parent with the default DynamicLayer factory, then swap
        super().__init__()
        self.layer_class_to_replicate = partial(TurboQuantLayer, quantizer=self._quantizer)

    def update(self, key_states, value_states, layer_idx, *args, **kwargs):
        # Track theoretical original bit count (fp16 baseline) for compression accounting
        self.bits_original += (key_states.numel() + value_states.numel()) * 16
        return super().update(key_states, value_states, layer_idx, *args, **kwargs)

    @property
    def compression_ratio(self) -> float:
        return 16 / self.bits

    @property
    def memory_saved_mb(self) -> float:
        return self.bits_original * (1 - self.bits / 16) / 8 / 1e6
