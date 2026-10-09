# SPDX-License-Identifier: Apache-2.0
"""KIVI 分页 int4 反量化 gather 的 torch 算子实现（当前路由的实现）。

这是 ascend#116 0003/0005 期的上游原版方案，也是 KIVI 方案实际走
的路径：triton-ascend 3.5 上融合 gather 内核会误编译（静默读到垃圾
数据、漏写输出，且两次运行表现不同——在 910B2 上实测发现），所以
``kivi_dequant_gather_cache`` 改用普通 torch 的 gather + 解包 + 反量化。
纯 torch：CPU 可测，不 import triton。
"""

from __future__ import annotations

import torch

from .kivi_layout import (
    _check_contiguous,
    _check_dequant_gather_metadata,
    _check_key_cache_layout,
    _check_same_device,
    _check_value_cache_layout,
)


def _unpack_int4_words(packed: torch.Tensor) -> torch.Tensor:
    """Unpack int32 words into 8 int4 lanes (new trailing dim), as floats.

    Implemented as eight scalar-shift extractions instead of a broadcast:
    the torch_npu aclnnRightShift adapter rejects broadcasting (and treats
    the self operand as the output shape).
    """
    packed_i32 = packed.to(torch.int32)
    lanes = [(packed_i32 >> (4 * lane)) & 0xF for lane in range(8)]
    return torch.stack(lanes, dim=-1).to(torch.float32)


def kivi_dequant_gather_cache(
    k_quant_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    k_mn_cache: torch.Tensor,
    v_quant_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    v_mn_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    target_dtype: torch.dtype,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把分页 int4 历史区中"活着的 token"gather 出来并反量化。

    实现步骤（纯 torch）：
      1. 用块表把每个请求引用的缓存块取出来（flat_ids 索引）；
      2. int4 word 解包成 8 lane，配合 scale/mn 反量化成块张量；
      3. 用 seq_lens 掩掉每块末尾的无效槽位，按请求顺序拼回稠密张量。
    """
    if block_table.shape[0] != seq_lens.numel():
        block_table = block_table[: seq_lens.numel()]

    _check_contiguous("k_quant_cache", k_quant_cache)
    _check_contiguous("k_scale_cache", k_scale_cache)
    _check_contiguous("k_mn_cache", k_mn_cache)
    _check_contiguous("v_quant_cache", v_quant_cache)
    _check_contiguous("v_scale_cache", v_scale_cache)
    _check_contiguous("v_mn_cache", v_mn_cache)
    _check_same_device("k_scale_cache", k_scale_cache, k_quant_cache)
    _check_same_device("k_mn_cache", k_mn_cache, k_quant_cache)
    _check_same_device("v_quant_cache", v_quant_cache, k_quant_cache)
    _check_same_device("v_scale_cache", v_scale_cache, k_quant_cache)
    _check_same_device("v_mn_cache", v_mn_cache, k_quant_cache)
    _check_same_device("block_table", block_table, k_quant_cache)
    _check_same_device("seq_lens", seq_lens, k_quant_cache)

    num_blocks, num_kv_heads, head_size, packed_block_size = k_quant_cache.shape
    block_size = packed_block_size * 8
    if group_size % 8 != 0:
        raise RuntimeError(
            "KIVI INT4 dequant requires group_size "
            f"({group_size}) to be divisible by 8."
        )
    _check_key_cache_layout(
        k_quant_cache,
        k_scale_cache,
        k_mn_cache,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        group_size=group_size,
    )
    _check_value_cache_layout(
        v_quant_cache,
        v_scale_cache,
        v_mn_cache,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        group_size=group_size,
    )
    if v_quant_cache.shape[0] != num_blocks or v_quant_cache.shape[1] != block_size:
        raise RuntimeError(
            "KIVI key/value cache block layout mismatch: "
            f"key num_blocks={num_blocks}, block_size={block_size}; "
            f"value shape={tuple(v_quant_cache.shape)}."
        )
    _, batch_size, total_tokens = _check_dequant_gather_metadata(
        block_table,
        seq_lens,
        num_blocks=num_blocks,
        block_size=block_size,
    )

    batch, max_blocks = block_table.shape
    flat_ids = block_table.reshape(-1).clamp_min(0).long()
    # keys group along block tokens, values group along head dims
    k_group_reps = block_size // group_size
    v_group_reps = head_size // group_size

    # keys: [NB, KVH, H, B/8] -> per requested block [batch, maxb, KVH, H, B]
    kq = _unpack_int4_words(k_quant_cache[flat_ids]).view(
        batch, max_blocks, num_kv_heads, head_size, block_size
    )
    ks = k_scale_cache[flat_ids].view(
        batch, max_blocks, num_kv_heads, head_size, k_group_reps
    )
    km = k_mn_cache[flat_ids].view(
        batch, max_blocks, num_kv_heads, head_size, k_group_reps
    )
    k_scale_full = ks.repeat_interleave(group_size, dim=-1)
    k_mn_full = km.repeat_interleave(group_size, dim=-1)
    k_blocks = (kq * k_scale_full + k_mn_full).permute(0, 1, 4, 2, 3)

    # values: [NB, B, KVH, H/8] -> [batch, maxb, B, KVH, H]
    vq = _unpack_int4_words(v_quant_cache[flat_ids]).view(
        batch, max_blocks, block_size, num_kv_heads, head_size
    )
    vs = v_scale_cache[flat_ids].view(
        batch, max_blocks, block_size, num_kv_heads, v_group_reps
    )
    vm = v_mn_cache[flat_ids].view(
        batch, max_blocks, block_size, num_kv_heads, v_group_reps
    )
    v_scale_full = vs.repeat_interleave(group_size, dim=-1)
    v_mn_full = vm.repeat_interleave(group_size, dim=-1)
    v_blocks = vq * v_scale_full + v_mn_full

    max_tokens = max_blocks * block_size
    seq_lens_t = seq_lens.to(torch.long)
    positions = torch.arange(max_tokens, device=block_table.device)
    valid = positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)

    dense_k = k_blocks.reshape(batch, max_tokens, num_kv_heads, head_size)[valid]
    dense_v = v_blocks.reshape(batch, max_tokens, num_kv_heads, head_size)[valid]
    return (
        dense_k.reshape(total_tokens, num_kv_heads, head_size)
        .to(target_dtype)
        .contiguous(),
        dense_v.reshape(total_tokens, num_kv_heads, head_size)
        .to(target_dtype)
        .contiguous(),
    )


__all__ = ["kivi_dequant_gather_cache"]
