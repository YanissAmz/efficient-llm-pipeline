import torch

from src.turboquant.polar_quant import (
    TurboQuantCache,
    TurboQuantMSE,
    TurboQuantProd,
    build_codebooks,
)
from src.turboquant.qjl import QJL


class TestBuildCodebooks:
    def test_returns_all_bits(self):
        cb = build_codebooks(max_bits=3)
        assert set(cb.keys()) == {1, 2, 3}

    def test_codebook_sizes(self):
        cb = build_codebooks(max_bits=3)
        for b in range(1, 4):
            assert len(cb[b]) == 2**b

    def test_sorted(self):
        cb = build_codebooks(max_bits=3)
        for _b, c in cb.items():
            assert all(c[i] <= c[i + 1] for i in range(len(c) - 1))


class TestQJL:
    def test_shape_preserved(self):
        qjl = QJL(dim=32)
        x = torch.randn(2, 4, 32)
        bits, norm = qjl.quantize(x)
        assert bits.shape == x.shape
        assert norm.shape == (2, 4, 1)

    def test_roundtrip(self):
        qjl = QJL(dim=64)
        x = torch.randn(8, 64)
        bits, norm = qjl.quantize(x)
        x_hat = qjl.dequantize(bits, norm)
        assert x_hat.shape == x.shape


class TestTurboQuantMSE:
    def test_shape_preserved(self):
        cb = build_codebooks(max_bits=3)
        quant = TurboQuantMSE(dim=32, bits=2, codebooks=cb)
        x = torch.randn(4, 32)
        idx, x_hat = quant.quantize(x)
        assert idx.shape == x.shape
        assert x_hat.shape == x.shape


class TestTurboQuantProd:
    def test_shape_preserved(self):
        cb = build_codebooks(max_bits=3)
        quant = TurboQuantProd(dim=32, bits=3, codebooks=cb)
        x = torch.randn(4, 32)
        _idx, _qjl_bits, _r_norm, x_hat = quant.quantize(x)
        assert x_hat.shape == x.shape

    def test_dequantize_matches(self):
        cb = build_codebooks(max_bits=3)
        quant = TurboQuantProd(dim=32, bits=3, codebooks=cb)
        x = torch.randn(4, 32)
        idx, qjl_bits, r_norm, x_hat = quant.quantize(x)
        x_recon = quant.dequantize(idx, qjl_bits, r_norm)
        assert torch.allclose(x_hat, x_recon, atol=1e-5)

    def test_compression_reduces_error(self):
        """More bits should reduce reconstruction error."""
        cb = build_codebooks(max_bits=4)
        x = torch.randn(16, 64)
        errors = {}
        for bits in [2, 3, 4]:
            quant = TurboQuantProd(dim=64, bits=bits, codebooks=cb)
            _, _, _, x_hat = quant.quantize(x)
            errors[bits] = (x - x_hat).norm().item()
        assert errors[4] < errors[2]


class TestTurboQuantCache:
    def test_update_returns_correct_shapes(self):
        cb = build_codebooks(max_bits=3)
        cache = TurboQuantCache(dim=32, bits=3, codebooks=cb)
        k = torch.randn(1, 4, 1, 32)  # batch, heads, seq, dim
        v = torch.randn(1, 4, 1, 32)
        k_out, v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape == k.shape
        assert v_out.shape == v.shape

    def test_cache_accumulates(self):
        cb = build_codebooks(max_bits=3)
        cache = TurboQuantCache(dim=32, bits=3, codebooks=cb)
        for _ in range(3):
            k = torch.randn(1, 4, 1, 32)
            v = torch.randn(1, 4, 1, 32)
            k_out, _v_out = cache.update(k, v, layer_idx=0)
        assert k_out.shape[2] == 3  # 3 tokens accumulated

    def test_compression_ratio(self):
        cb = build_codebooks(max_bits=3)
        cache = TurboQuantCache(dim=32, bits=3, codebooks=cb)
        assert cache.compression_ratio == 16 / 3
