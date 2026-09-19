# SPDX-License-Identifier: Apache-2.0
"""上游融合 dequant-gather 内核（triton-ascend）——实验性，当前不路由。

移植自 legacy ascend PR #116 提交 0007-0009。在 triton-ascend 3.5 上
误编译：静默读到垃圾数据、漏写输出，两次运行表现不同，16x16 与 32x32
tile 均复现（910B2 实测，scripts/npu_probe_kivi_dim.py 可复现）。

KIVI 方案实际路由的是 ``ops.kivi_gather`` 的纯 torch 实现（ascend#116
0003/0005 期上游方案，已逐位验证）。本模块保留用于未来 triton-ascend
修复后重验；请勿在生产代码中引用。
"""

try:
    from vllm.triton_utils import tl, triton
except ImportError:  # pragma: no cover - exercised only outside vllm hosts
    try:
        import triton
        import triton.language as tl
    except ImportError as _exc:
        raise ImportError(
            "KIVI triton kernels require triton (triton-ascend on Ascend "
            "NPU), importable directly or through vllm.triton_utils"
        ) from _exc

import torch


@triton.jit
def _kivi_dequant_gather_key_cache_kernel(
    k_quant_cache_ptr,
    k_scale_cache_ptr,
    k_mn_cache_ptr,
    token_block_ids_ptr,
    token_block_offsets_ptr,
    key_out_ptr,
    total_tokens,
    block_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    group_size: tl.constexpr,
    block_dim_size: tl.constexpr,
):
    token_start = tl.program_id(0) * block_dim_size
    head_idx = tl.program_id(1)
    dim_start = tl.program_id(2) * block_dim_size

    token_offsets = token_start + tl.arange(0, block_dim_size)
    dim_offsets = dim_start + tl.arange(0, block_dim_size)
    token_mask = token_offsets < total_tokens
    dim_mask = dim_offsets < head_size

    block_id = tl.load(
        token_block_ids_ptr + token_offsets, mask=token_mask, other=-1
    ).to(tl.int64)
    block_offset = tl.load(
        token_block_offsets_ptr + token_offsets, mask=token_mask, other=0
    ).to(tl.int64)
    live_mask = token_mask & (block_id >= 0)

    key_group_idx = block_offset // group_size
    key_pack_idx = block_offset // 8
    key_lane = block_offset - key_pack_idx * 8
    key_shift = key_lane[:, None] * 4
    num_key_groups = block_size // group_size
    key_quant_offsets = (
        block_id[:, None] * num_kv_heads * head_size * (block_size // 8)
        + head_idx * head_size * (block_size // 8)
        + dim_offsets[None, :] * (block_size // 8)
        + key_pack_idx[:, None]
    )
    key_scale_offsets = (
        block_id[:, None] * num_kv_heads * head_size * num_key_groups
        + head_idx * head_size * num_key_groups
        + dim_offsets[None, :] * num_key_groups
        + key_group_idx[:, None]
    )
    key_mask = live_mask[:, None] & dim_mask[None, :]
    key_packed = tl.load(k_quant_cache_ptr + key_quant_offsets, mask=key_mask, other=0)
    key_scale = tl.load(k_scale_cache_ptr + key_scale_offsets, mask=key_mask, other=0.0)
    key_mn = tl.load(k_mn_cache_ptr + key_scale_offsets, mask=key_mask, other=0.0)
    key_q = ((key_packed.to(tl.int32) >> key_shift.to(tl.int32)) & 0xF).to(tl.float32)
    key_deq = key_q * key_scale + key_mn
    out_offsets = (
        token_offsets[:, None] * num_kv_heads * head_size
        + head_idx * head_size
        + dim_offsets[None, :]
    )
    tl.store(
        key_out_ptr + out_offsets, key_deq, mask=live_mask[:, None] & dim_mask[None, :]
    )


@triton.jit
def _kivi_dequant_gather_value_cache_kernel(
    v_quant_cache_ptr,
    v_scale_cache_ptr,
    v_mn_cache_ptr,
    token_block_ids_ptr,
    token_block_offsets_ptr,
    value_out_ptr,
    total_tokens,
    block_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    group_size: tl.constexpr,
    block_dim_size: tl.constexpr,
):
    token_start = tl.program_id(0) * block_dim_size
    head_idx = tl.program_id(1)
    dim_start = tl.program_id(2) * block_dim_size

    token_offsets = token_start + tl.arange(0, block_dim_size)
    dim_offsets = dim_start + tl.arange(0, block_dim_size)
    token_mask = token_offsets < total_tokens
    dim_mask = dim_offsets < head_size

    block_id = tl.load(
        token_block_ids_ptr + token_offsets, mask=token_mask, other=-1
    ).to(tl.int64)
    block_offset = tl.load(
        token_block_offsets_ptr + token_offsets, mask=token_mask, other=0
    ).to(tl.int64)
    live_mask = token_mask & (block_id >= 0)

    value_group_idx = dim_offsets // group_size
    value_pack_idx = dim_offsets // 8
    value_lane = dim_offsets - value_pack_idx * 8
    value_shift = value_lane[None, :] * 4
    num_value_groups = head_size // group_size
    value_quant_offsets = (
        block_id[:, None] * block_size * num_kv_heads * (head_size // 8)
        + block_offset[:, None] * num_kv_heads * (head_size // 8)
        + head_idx * (head_size // 8)
        + value_pack_idx[None, :]
    )
    value_scale_offsets = (
        block_id[:, None] * block_size * num_kv_heads * num_value_groups
        + block_offset[:, None] * num_kv_heads * num_value_groups
        + head_idx * num_value_groups
        + value_group_idx[None, :]
    )
    value_mask = live_mask[:, None] & dim_mask[None, :]
    value_packed = tl.load(
        v_quant_cache_ptr + value_quant_offsets, mask=value_mask, other=0
    )
    value_scale = tl.load(
        v_scale_cache_ptr + value_scale_offsets, mask=value_mask, other=0.0
    )
    value_mn = tl.load(v_mn_cache_ptr + value_scale_offsets, mask=value_mask, other=0.0)
    value_q = ((value_packed.to(tl.int32) >> value_shift.to(tl.int32)) & 0xF).to(
        tl.float32
    )
    value_deq = value_q * value_scale + value_mn
    out_offsets = (
        token_offsets[:, None] * num_kv_heads * head_size
        + head_idx * head_size
        + dim_offsets[None, :]
    )
    tl.store(
        value_out_ptr + out_offsets,
        value_deq,
        mask=live_mask[:, None] & dim_mask[None, :],
    )


def _resolve_token_block_spans(
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    batch_size: int,
    total_tokens: int,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve per-token (block_id, block_offset) with plain torch ops.

    Upstream resolved these inside the gather kernels with a
    data-dependent scalar-load loop over ``cu_seq_lens``; that pattern
    miscompiles on current triton-ascend (silent garbage loads and dropped
    stores, varying between runs). Doing it on the host is both correct
    and cheaper: one searchsorted + gather per gather call.
    """
    device = block_table.device
    token_idx = torch.arange(total_tokens, device=device)
    req_of_token = torch.searchsorted(cu_seq_lens[1:], token_idx, right=True)
    local_pos = token_idx - cu_seq_lens[req_of_token]
    logical_block_idx = torch.div(local_pos, block_size, rounding_mode="floor")
    token_block_ids = block_table[req_of_token, logical_block_idx].contiguous()
    token_block_offsets = (local_pos - logical_block_idx * block_size).contiguous()
    return token_block_ids, token_block_offsets


def _launch_kivi_dequant_gather_cache_experimental(
    kernel,
    quant_cache: torch.Tensor,
    scale_cache: torch.Tensor,
    mn_cache: torch.Tensor,
    block_table: torch.Tensor,
    cu_seq_lens: torch.Tensor,
    *,
    batch_size: int,
    total_tokens: int,
    target_dtype: torch.dtype,
    group_size: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
) -> torch.Tensor:
    """实验路径：直接发射上游融合 gather 内核（当前不被路由）。

    在 triton-ascend 3.5 上会误编译（见模块头）；保留用于未来重验。
    """
    out = torch.empty(
        (total_tokens, num_kv_heads, head_size),
        dtype=target_dtype,
        device=quant_cache.device,
    )
    if total_tokens == 0:
        return out

    token_block_ids, token_block_offsets = _resolve_token_block_spans(
        block_table,
        cu_seq_lens,
        batch_size=batch_size,
        total_tokens=total_tokens,
        block_size=block_size,
    )

    # The gather kernels materialize token x dim tiles; 16x16 keeps the
    # tile inside the Ascend UB budget for every supported head size.
    block_dim_size = min(16, triton.next_power_of_2(head_size))
    grid = (
        triton.cdiv(total_tokens, block_dim_size),
        num_kv_heads,
        triton.cdiv(head_size, block_dim_size),
    )
    kernel[grid](
        quant_cache,
        scale_cache,
        mn_cache,
        token_block_ids,
        token_block_offsets,
        out,
        total_tokens,
        block_size,
        num_kv_heads,
        head_size,
        group_size,
        block_dim_size,
    )
    return out
