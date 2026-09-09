"""Dynamic per-channel INT8 solution: semantics math and store path."""

import pytest
import torch

from vllm_ascend_quantized_kv_cache.solutions.int8_dynamic import (
    attention_mixin as am,
)
from vllm_ascend_quantized_kv_cache.solutions.int8_dynamic.semantics import (
    Int8DynamicSemantics,
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


def test_mixin_state_init_accepts_legacy_and_contract_spellings() -> None:
    class Base:
        pass

    class Impl(am.Int8DynamicAttentionMixin, Base):
        pass

    for dtype, enabled in [
        ("int8", True),
        ("int8_per_token_head", True),
        ("int8_dynamic", True),
        ("auto", False),
        ("fp8", False),
    ]:
        impl = Impl()
        impl._init_int8_dynamic_state(dtype)
        assert impl.enable_int8 is enabled, dtype
        assert impl._int8_ready is False


def test_quantize_for_store_requires_scales_first() -> None:
    class Base:
        num_kv_heads = NUM_KV_HEADS
        head_size = HEAD_SIZE

    class Impl(am.Int8DynamicAttentionMixin, Base):
        pass

    impl = Impl()
    impl._init_int8_dynamic_state("int8")
    with pytest.raises(RuntimeError, match="scales are not initialised"):
        impl._int8_quantize_for_store(torch.zeros(1), torch.zeros(1))

    key = torch.randn(8, NUM_KV_HEADS, HEAD_SIZE)
    impl._calc_int8_scales(key, key)
    assert impl._int8_ready
    qk, qv = impl._int8_quantize_for_store(key * 4, key * -2)
    assert qk.dtype == torch.int8 and qv.dtype == torch.int8
    assert int(qk.abs().max()) <= 127


def test_dequant_paged_kv_to_dense_cpu_math() -> None:
    from vllm_ascend_quantized_kv_cache.ops.int8_ops import dequant_paged_kv_to_dense

    torch.manual_seed(1)
    num_blocks, block_size, hidden = 4, 4, NUM_KV_HEADS * HEAD_SIZE
    key_cache = torch.randint(-128, 127, (num_blocks, block_size, hidden)).to(
        torch.int8
    )
    value_cache = torch.randint(-128, 127, (num_blocks, block_size, hidden)).to(
        torch.int8
    )
    block_table = torch.tensor([[0, 1], [2, 3]])
    seq_lens = [6, 5]
    k_inv = torch.full((1, 1, 1), 0.5)
    k_off = torch.zeros_like(k_inv)

    dense_k, dense_v = dequant_paged_kv_to_dense(
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        torch.float32,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        k_inv_scale=k_inv,
        k_offset=k_off,
        v_inv_scale=k_inv,
        v_offset=k_off,
    )
    assert dense_k.shape == (11, NUM_KV_HEADS, HEAD_SIZE)
    assert dense_v.shape == (11, NUM_KV_HEADS, HEAD_SIZE)
    # dequant = (x - 0) * (1 / 0.5) = 2x. Request 0 covers slots 0-5 (block 0
    # then block 1), request 1 covers slots 6-10 (blocks 2 and 3).
    assert torch.equal(
        dense_k[4],
        key_cache[1, 0].view(NUM_KV_HEADS, HEAD_SIZE).to(torch.float32) * 2,
    )
    assert torch.equal(
        dense_k[6],
        key_cache[2, 0].view(NUM_KV_HEADS, HEAD_SIZE).to(torch.float32) * 2,
    )


def test_forward_int8_unsupported_state_fails_closed() -> None:
    class Base:
        num_kv_heads = NUM_KV_HEADS
        head_size = HEAD_SIZE
        scale = 0.1

        def reshape_and_cache(self, q, k, v, kv_cache, md, out):
            return q, k, v, out

    from types import SimpleNamespace

    class Impl(am.Int8DynamicAttentionMixin, Base):
        pass

    impl = Impl()
    impl._init_int8_dynamic_state("int8")
    impl._int8_ready = True
    md = SimpleNamespace(attn_state="SpecDecoding")  # duck-typed unknown enum
    output = torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE)
    with pytest.raises(RuntimeError, match="does not support attention state"):
        impl.forward_int8(
            torch.zeros(1, NUM_KV_HEADS, HEAD_SIZE),
            None,
            None,
            None,
            md,
            output,
        )
