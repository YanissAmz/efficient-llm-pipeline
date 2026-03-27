"""
QJL — Quantized Johnson-Lindenstrauss
Quantification 1-bit non-biaisée pour l'estimation de produits scalaires.

Référence : TurboQuant (arXiv:2504.19874), Section 3.
"""

import math
import torch
import torch.nn as nn


class QJL(nn.Module):
    """
    Quantificateur 1-bit non-biaisé basé sur Johnson-Lindenstrauss.

    Pour x ∈ R^d :
        Quant  : z = sign(S·x)          avec S ∈ R^(d×d), Sᵢⱼ ~ N(0,1)
        DeQuant: x̃ = (√(π/2) / d) · ‖x‖₂ · Sᵀ·z

    Garantie : E[<y, x̃>] = <y, x>  pour tout y  (non-biaisé)
    """

    def __init__(self, dim: int, seed: int = 43):
        super().__init__()
        self.dim   = dim
        self.scale = math.sqrt(math.pi / 2) / dim

        torch.manual_seed(seed)
        self.register_buffer('S', torch.randn(dim, dim))

    def quantize(self, x: torch.Tensor):
        """
        x : (..., dim)  →  (qjl_bits, x_norm)
        qjl_bits : (..., dim) int8  ∈ {-1, +1}
        x_norm   : (..., 1)  float  — ‖x‖₂ pour la reconstruction
        """
        shape    = x.shape
        x_flat   = x.reshape(-1, self.dim).float()
        x_norm   = x_flat.norm(dim=-1, keepdim=True)
        qjl_bits = (x_flat @ self.S.T).sign().to(torch.int8)
        return qjl_bits.reshape(shape), x_norm.reshape(*shape[:-1], 1)

    def dequantize(self, qjl_bits: torch.Tensor, x_norm: torch.Tensor) -> torch.Tensor:
        """Reconstruction non-biaisée depuis les bits QJL."""
        shape  = qjl_bits.shape
        bits_f = qjl_bits.reshape(-1, self.dim).float()
        norm_f = x_norm.reshape(-1, 1)
        x_hat  = self.scale * norm_f * (bits_f @ self.S)
        return x_hat.reshape(shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        qjl_bits, x_norm = self.quantize(x)
        return self.dequantize(qjl_bits, x_norm)
