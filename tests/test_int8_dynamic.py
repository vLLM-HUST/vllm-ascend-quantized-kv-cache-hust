"""Dynamic per-channel INT8 semantics math."""

import pytest

from vllm_ascend_quantized_kv_cache.methods.int8_dynamic.semantics import (
    Int8DynamicSemantics,
)

torch = pytest.importorskip(
    "torch",
    reason="CPU INT8 semantics tests require the optional torch host dependency",
)

HEAD_SIZE = 8
NUM_KV_HEADS = 2


def _config_like(num_kv_heads: int = NUM_KV_HEADS, head_size: int = HEAD_SIZE):
    from types import SimpleNamespace

    return SimpleNamespace(num_kv_heads=num_kv_heads, head_size=head_size)


def test_calc_scales_matches_legacy_math() -> None:
    torch.manual_seed(0)
    key = torch.randn(16, NUM_KV_HEADS, HEAD_SIZE)
    value = torch.randn(16, NUM_KV_HEADS, HEAD_SIZE)
    sem = Int8DynamicSemantics(_config_like())
    scales = sem.calc_scales(key, value)

    expected_k_max = key.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
    assert torch.allclose(scales.k_inv_scale, 127.0 / expected_k_max)
    assert torch.all(scales.k_offset == 0)
    # BNSD antiquant view: [1, H, 1, D]
    assert scales.k_aq_scale.shape == (1, NUM_KV_HEADS, 1, HEAD_SIZE)
    assert torch.allclose(
        scales.k_aq_scale.view(1, NUM_KV_HEADS, HEAD_SIZE),
        (1.0 / scales.k_inv_scale).view(1, NUM_KV_HEADS, HEAD_SIZE),
    )


def test_quantize_dequantize_roundtrip_within_resolution() -> None:
    x = torch.linspace(-1.0, 1.0, 32)
    inv_scale = torch.full_like(x, 127.0)
    offset = torch.zeros_like(x)
    q = Int8DynamicSemantics.quantize(x, inv_scale, offset)
    assert q.dtype == torch.int8
    restored = Int8DynamicSemantics.dequantize(q, inv_scale, offset, x.dtype)
    assert torch.max(torch.abs(restored - x)) <= 1.0 / 127.0 + 1e-6
