# SPDX-License-Identifier: Apache-2.0
"""fp8_per_token_head 方法的契约 / 布局 / CPU 语义测试。

设备路径（triton-ascend store 内核 + FIA 读通路）尚未移植；这里的
语义往返就是将来内核的对拍基准，布局预算与宿主页大小口径逐位一致。
"""

from __future__ import annotations

import pytest
import torch

from vllm_ascend_quantized_kv_cache.dtypes import (
    KVCacheLayout,
    KVQuantMode,
    resolve_layout,
)
from vllm_ascend_quantized_kv_cache.methods.fp8_per_token_head.byte_cache import (
    Fp8PerTokenHeadLayout,
    fp8_caches_from_byte_tensors,
    fp8_per_token_head_layout,
)
from vllm_ascend_quantized_kv_cache.methods.fp8_per_token_head.semantics import (
    Fp8PerTokenHeadSemantics,
)
from vllm_ascend_quantized_kv_cache.methods.registry import get_method

HEAD = 128
KVH = 2
BLOCK = 128
NB = 4


# ---------------------------------------------------------------------------
# 契约层
# ---------------------------------------------------------------------------


def test_layout_contract() -> None:
    layout = resolve_layout("fp8_per_token_head", HEAD)
    assert isinstance(layout, KVCacheLayout)
    assert layout.storage_dtype == "uint8"  # 与宿主 STR_DTYPE_TO_TORCH_DTYPE 一致
    assert layout.packed_last_dim == HEAD  # E4M3 逐元素一字节，不打包
    assert layout.quant_mode is KVQuantMode.FP8_PER_TOKEN_HEAD


def test_layout_contract_rejects_unaligned_head_size() -> None:
    with pytest.raises(ValueError, match="divisible by 4"):
        resolve_layout("fp8_per_token_head", 126)


def test_method_discovery_and_fail_closed_adapter() -> None:
    method = get_method("fp8_per_token_head", head_size=HEAD, num_kv_heads=KVH)
    assert method.spec.dtype == "fp8_per_token_head"
    assert method.spec.requires_npu_kernels
    assert method.resolve_layout().quant_mode is KVQuantMode.FP8_PER_TOKEN_HEAD

    # 契约已立、适配器未接线：fail-closed 而不是静默假装可用。
    with pytest.raises(ValueError, match="no adapter wired"):
        method.host_adapter("vllm_ascend_hust")


# ---------------------------------------------------------------------------
# 布局：两张等大宿主缓冲 -> 四视图
# ---------------------------------------------------------------------------


def test_region_budget_matches_host_page_size() -> None:
    layout = Fp8PerTokenHeadLayout(
        num_blocks=NB, block_size=BLOCK, num_kv_heads=KVH, head_size=HEAD
    )
    assert layout.bytes_per_token_head == HEAD + 4
    # 宿主口径：数据 2*block*kvh*head*1 + scale 2*block*kvh*4。
    host_page = 2 * BLOCK * KVH * HEAD + 2 * BLOCK * KVH * 4
    assert 2 * layout.region_bytes == NB * host_page
    # head 128：256 B -> 132 B ≈ 1.94x（fp8 的价值在动态范围与免校准，
    # 不在压缩比——对比 int8_dynamic 同为 2x 档）。
    assert layout.compression_vs_fp16() == pytest.approx(256 / 132)


def test_views_alias_host_buffers() -> None:
    layout = Fp8PerTokenHeadLayout(
        num_blocks=NB, block_size=BLOCK, num_kv_heads=KVH, head_size=HEAD
    )
    key_buf = torch.zeros(
        NB, BLOCK, KVH, layout.bytes_per_token_head, dtype=torch.uint8
    )
    value_buf = torch.zeros_like(key_buf)
    assert key_buf.numel() == layout.region_bytes

    k_data, k_scale, v_data, v_scale = fp8_caches_from_byte_tensors(
        key_buf, value_buf, layout
    )
    assert k_data.shape == (NB, BLOCK, KVH, HEAD) and k_data.dtype is torch.uint8
    assert k_scale.shape == (NB, BLOCK, KVH, 1) and k_scale.dtype is torch.float32
    assert v_data.shape == k_data.shape and v_scale.shape == k_scale.shape

    # data 区与 scale 区在宿主缓冲里前后相接；写 fp32 读回 scale 视图。
    data_bytes = NB * BLOCK * KVH * HEAD
    key_buf.reshape(-1).view(torch.float32)[data_bytes // 4] = 2.5
    assert float(k_scale[0, 0, 0, 0]) == 2.5
    value_buf.reshape(-1).view(torch.float32)[data_bytes // 4] = -0.75
    assert float(v_scale[0, 0, 0, 0]) == -0.75


def test_layout_derivation_and_fail_closed() -> None:
    S = HEAD + 4
    key_buf = torch.zeros(NB, BLOCK, KVH, S, dtype=torch.uint8)
    value_buf = torch.zeros_like(key_buf)
    layout = fp8_per_token_head_layout(
        key_buf, value_buf, num_kv_heads=KVH, head_size=HEAD
    )
    assert (layout.num_blocks, layout.block_size) == (NB, BLOCK)

    with pytest.raises(RuntimeError, match="equal regions"):
        fp8_per_token_head_layout(
            key_buf,
            value_buf[:, :, :, : S - 8].contiguous(),
            num_kv_heads=KVH,
            head_size=HEAD,
        )
    short_key = key_buf[:, :, :, : S - 4].contiguous()
    with pytest.raises(RuntimeError, match="whole number of pages"):
        fp8_per_token_head_layout(
            short_key, short_key.clone(), num_kv_heads=KVH, head_size=HEAD
        )


# ---------------------------------------------------------------------------
# 语义：量化数学（将来 triton-ascend 内核的对拍基准）
# ---------------------------------------------------------------------------


def test_fake_quant_error_bounded_by_e4m3_precision() -> None:
    torch.manual_seed(5)
    sem = Fp8PerTokenHeadSemantics(type("Cfg", (), {"head_size": HEAD})())
    x = torch.randn(8, KVH, HEAD)
    scale = sem.compute_scale(x)
    assert scale.shape == (8, KVH, 1)

    fq = sem.fake_quant(x)
    # E4M3 3 位尾数：正规数半 ULP ≈ 2^-4 |x|；亚正规附近绝对误差不超过
    # scale 的最小粒度。逐元素上界两条都放宽一档。
    err = (fq.float() - x).abs()
    assert (err <= 0.07 * x.abs() + scale * 2**-8).all()


def test_per_token_head_scale_is_independent() -> None:
    """一个 (token, head) 里的离群值不毁掉其他槽位的分辨率。

    这是相对 int8_dynamic（scale 首个 prefill 定格、per-channel）的
    核心差异：这里每个 (token, head) 独立 amax，浮点编码天然吃动态范围。
    """
    torch.manual_seed(6)
    sem = Fp8PerTokenHeadSemantics(type("Cfg", (), {"head_size": HEAD})())
    x = torch.randn(2, KVH, HEAD)
    x[0, 0, 0] = 1.0e4  # 单点离群
    fq = sem.fake_quant(x)

    clean = fq[1, 0]
    rel = (clean - x[1, 0]).abs() / x[1, 0].abs().clamp(min=1e-3)
    assert rel.max() < 0.07  # 无离群槽位保持 E4M3 正常精度
    assert torch.isfinite(fq).all()

    scales = sem.compute_scale(x)
    assert scales[0, 0, 0] > scales[1, 0, 0] * 100  # 离群槽位自己的 scale


def test_zero_group_does_not_divide_by_zero() -> None:
    sem = Fp8PerTokenHeadSemantics(type("Cfg", (), {"head_size": HEAD})())
    x = torch.zeros(3, 1, HEAD)
    fq = sem.fake_quant(x)
    assert torch.isfinite(fq).all() and (fq == 0).all()


def test_byte_level_roundtrip_through_views() -> None:
    """量化 -> 写进宿主缓冲视图 -> 读回反量化，与 fake_quant 一致。

    数据按 E4M3 原始字节存（uint8 视图 <-> float8 视图），scale 走
    fp32 视图——这是将来内核读路径要复现的完整字节口径。
    """
    torch.manual_seed(7)
    layout = Fp8PerTokenHeadLayout(
        num_blocks=NB, block_size=BLOCK, num_kv_heads=KVH, head_size=HEAD
    )
    key_buf = torch.zeros(
        NB, BLOCK, KVH, layout.bytes_per_token_head, dtype=torch.uint8
    )
    value_buf = torch.zeros_like(key_buf)
    k_data, k_scale, _, _ = fp8_caches_from_byte_tensors(key_buf, value_buf, layout)

    sem = Fp8PerTokenHeadSemantics(type("Cfg", (), {"head_size": HEAD})())
    tokens = 32
    x = torch.randn(tokens, KVH, HEAD)
    packed, scale = sem.quantize(x)

    # token t 落在第 0 页的块内偏移 t（slot == t，块内前 32 个槽位）。
    k_data[0, :tokens] = packed.view(torch.uint8)
    k_scale[0, :tokens, :, 0] = scale[:, :, 0]

    roundtrip = sem.dequantize(
        k_data[0, :tokens].view(torch.float8_e4m3fn), k_scale[0, :tokens]
    )
    ref = sem.fake_quant(x)
    assert torch.allclose(roundtrip, ref, atol=1e-6)
