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
import numpy as np
import torch
import torch.nn as nn
from transformers import DynamicCache


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
        n_levels  = 2 ** b
        samples   = np.random.randn(n_samples)
        centroids = np.linspace(
            np.percentile(samples, 1),
            np.percentile(samples, 99),
            n_levels,
        )
        for _ in range(200):
            assigned      = np.argmin(np.abs(samples[:, None] - centroids[None, :]), axis=1)
            new_centroids = np.array([
                samples[assigned == k].mean() if (assigned == k).any() else centroids[k]
                for k in range(n_levels)
            ])
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
    Quantificateur MSE-optimal : rotation aléatoire + Lloyd-Max.

    Étapes :
        1. y = Π·x  (rotation orthogonale)  →  coords ~ N(0, 1/d)
        2. idx = argmin_k |y_j/scale - c_k|  (codebook Lloyd-Max)
        3. x̃ = Πᵀ · (c_{idx} * scale)       (reconstruction)
    """

    def __init__(self, dim: int, bits: int, codebooks: dict, seed: int = 42):
        super().__init__()
        self.dim   = dim
        self.bits  = bits
        self.scale = 1.0 / (dim ** 0.5)

        torch.manual_seed(seed)
        Q, _ = torch.linalg.qr(torch.randn(dim, dim))
        self.register_buffer('rotation', Q)

        cb = torch.tensor(codebooks[bits], dtype=torch.float32)
        self.register_buffer('codebook', cb)

    def quantize(self, x: torch.Tensor):
        """x : (..., dim)  →  (idx, x_hat)"""
        shape  = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        y      = x_flat @ self.rotation.T
        y_norm = y / self.scale

        dists = (y_norm.unsqueeze(-1) - self.codebook).abs()
        idx   = dists.argmin(dim=-1)

        y_hat = self.codebook[idx] * self.scale
        x_hat = y_hat @ self.rotation

        return idx.reshape(shape), x_hat.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, x_hat = self.quantize(x)
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
        assert bits >= 2, 'bits >= 2 requis (b-1 pour MSE + 1 pour QJL)'
        self.dim       = dim
        self.bits      = bits
        self.qjl_scale = math.sqrt(math.pi / 2) / dim

        self.mse = TurboQuantMSE(dim, bits - 1, codebooks, seed=seed)

        torch.manual_seed(seed + 1)
        self.register_buffer('S', torch.randn(dim, dim))

    def quantize(self, x: torch.Tensor):
        """
        x : (..., dim)
        →  (idx, qjl_bits, residual_norm, x_hat)
        """
        shape  = x.shape
        x_flat = x.reshape(-1, self.dim).float()

        idx, x_mse = self.mse.quantize(x_flat)

        r      = x_flat - x_mse
        r_norm = r.norm(dim=-1, keepdim=True)

        qjl_bits   = (r @ self.S.T).sign().to(torch.int8)
        correction = self.qjl_scale * r_norm * (qjl_bits.float() @ self.S)
        x_hat      = x_mse + correction

        return (
            idx.reshape(shape),
            qjl_bits.reshape(shape),
            r_norm.reshape(*shape[:-1], 1),
            x_hat.reshape(shape),
        )

    def dequantize(self, idx, qjl_bits, r_norm) -> torch.Tensor:
        """Reconstruction depuis les données compressées stockées."""
        shape  = idx.shape
        y_hat  = self.mse.codebook[idx.reshape(-1, self.dim)] * self.mse.scale
        x_mse  = y_hat @ self.mse.rotation
        correction = self.qjl_scale * r_norm.reshape(-1, 1) * (
            qjl_bits.reshape(-1, self.dim).float() @ self.S
        )
        return (x_mse + correction).reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, _, x_hat = self.quantize(x)
        return x_hat


# ---------------------------------------------------------------------------
# TurboQuantCache
# ---------------------------------------------------------------------------

class TurboQuantCache(DynamicCache):
    """
    Cache KV compressé avec TurboQuant.
    Drop-in replacement pour HuggingFace DynamicCache.

    À chaque update() : compresse K et V avant stockage.
    Retourne les versions décompressées pour l'attention (unbiaisées).

    Args:
        dim      : head_dim du modèle  (hidden_size // num_attention_heads)
        bits     : bits de compression (3 = bon compromis vitesse/qualité)
        codebooks: dict retourné par build_codebooks()
    """

    def __init__(self, dim: int, bits: int, codebooks: dict, model=None):
        super().__init__()
        self.quantizer     = TurboQuantProd(dim, bits, codebooks)
        self._comp_k       = []
        self._comp_v       = []
        self.bits          = bits
        self.bits_original = 0

        # Attributs requis par les couches GatedDeltaNet (linear attention) de Qwen3.5
        self.has_previous_state = False
        self.conv_states        = {}
        self.recurrent_states   = {}

        if model is not None:
            self._init_recurrent_states(model)

    def _init_recurrent_states(self, model):
        """Pré-alloue les états zéro pour les couches GatedDeltaNet de Qwen3.5."""
        # Naviguer jusqu'aux decoder layers
        lm = model
        for attr in ('model', 'language_model', 'model'):
            if hasattr(lm, attr):
                lm = getattr(lm, attr)
        if not hasattr(lm, 'layers'):
            return

        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        for layer in lm.layers:
            if not hasattr(layer, 'linear_attn'):
                continue
            la = layer.linear_attn
            idx = la.layer_idx

            # Conv state : (batch=1, conv_dim, d_conv)
            if hasattr(la, 'conv1d'):
                conv_dim = la.conv1d.weight.shape[0]
                d_conv   = la.conv1d.weight.shape[2]
                self.conv_states[idx] = torch.zeros(1, conv_dim, d_conv, device=device, dtype=dtype)

            # Recurrent state : (batch=1, num_heads, head_dim, head_dim)
            num_heads = getattr(la, 'num_heads', None)
            head_dim  = getattr(la, 'head_dim', None)
            if num_heads and head_dim:
                self.recurrent_states[idx] = torch.zeros(1, num_heads, head_dim, head_dim, device=device, dtype=dtype)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        device = key_states.device
        self.quantizer = self.quantizer.to(device)

        k_idx, k_qjl, k_norm, k_hat = self.quantizer.quantize(key_states)
        v_idx, v_qjl, v_norm, v_hat = self.quantizer.quantize(value_states)

        if layer_idx >= len(self._comp_k):
            self._comp_k.append((k_idx, k_qjl, k_norm))
            self._comp_v.append((v_idx, v_qjl, v_norm))
        else:
            ok = self._comp_k[layer_idx]
            ov = self._comp_v[layer_idx]
            self._comp_k[layer_idx] = (
                torch.cat([ok[0], k_idx],  dim=2),
                torch.cat([ok[1], k_qjl],  dim=2),
                torch.cat([ok[2], k_norm],  dim=2),
            )
            self._comp_v[layer_idx] = (
                torch.cat([ov[0], v_idx],  dim=2),
                torch.cat([ov[1], v_qjl],  dim=2),
                torch.cat([ov[2], v_norm],  dim=2),
            )

        self.bits_original += (key_states.numel() + value_states.numel()) * 16

        return k_hat.to(key_states.dtype), v_hat.to(value_states.dtype)

    @property
    def compression_ratio(self) -> float:
        return 16 / self.bits

    @property
    def memory_saved_mb(self) -> float:
        return self.bits_original * (1 - self.bits / 16) / 8 / 1e6
