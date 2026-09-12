# SPDX-License-Identifier: Apache-2.0
"""KIVI int4 打包内核（triton-ascend，仅 Ascend NPU）——KIVI 在线路径。

移植自 legacy ascend PR #116 提交 0007 + 0008 + 0009 + 0013 的 pack 部分
（最终状态，逐字保留），并已在 910B2 上对照
``methods.kivi_int4.semantics`` 的纯 torch 参考逐位验证：
  * ``kivi_pack_key_cache``：键按 token 组量化打包（scale/mn 每组一份）；
  * ``kivi_pack_value_cache``：值按 head 维组逐 token 量化打包。

布局约定（与 ops/kivi_layout.py 的校验器一致）：
  键缓存 [NB, KVH, D, B/8]：同一缓存块内每 8 个连续 token 打进一个
  int32 word（lane = 块内偏移 % 8）；
  值缓存 [NB, B, KVH, H/8]：每个 token 一行，lane = head 维 % 8。

本模块只被 KIVI 方法惰性导入；依赖检查 fail-closed 并给出精确报错。
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

from ..kivi_layout import (
    _check_contiguous,
    _check_key_cache_layout,
    _check_same_device,
    _check_value_cache_layout,
)

_VECTORCORE_NUM = -1


def get_vectorcore_num() -> int:
    """本地移植自 vllm_ascend.ops.triton.triton_utils.get_vectorcore_num。"""
    global _VECTORCORE_NUM
    if _VECTORCORE_NUM < 0:
        properties = triton.runtime.driver.active.utils.get_device_properties(
            torch.npu.current_device()
        )
        count = int(properties.get("num_vectorcore", -1))
        if count <= 0:
            raise RuntimeError(
                "Failed to detect the Ascend NPU vectorcore count required "
                "by the KIVI pack kernels."
            )
        _VECTORCORE_NUM = count
    return _VECTORCORE_NUM


def _check_slot_mapping(slot_mapping: torch.Tensor, num_tokens: int) -> None:
    if slot_mapping.ndim != 1:
        raise RuntimeError(
            f"slot_mapping must be 1D, got shape={tuple(slot_mapping.shape)}."
        )
    if slot_mapping.numel() != num_tokens:
        raise RuntimeError(
            "slot_mapping length must match num_tokens, got "
            f"{slot_mapping.numel()} vs {num_tokens}."
        )
    if slot_mapping.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(
            f"slot_mapping must be int32/int64, got {slot_mapping.dtype}."
        )


def _check_key_slot_groups(
    slot_mapping: torch.Tensor,
    *,
    block_size: int,
    group_size: int,
) -> None:
    # Multiple token groups may be submitted together. We only require each
    # group to be contiguous, aligned, and fully contained in one cache block.
    if bool((slot_mapping < 0).any()):
        raise RuntimeError(
            "kivi_pack_key_cache requires all slot_mapping entries to be valid."
        )

    slot_groups = slot_mapping.view(-1, group_size)
    first_slots = slot_groups[:, :1]
    offsets = torch.arange(
        group_size,
        device=slot_mapping.device,
        dtype=slot_mapping.dtype,
    ).view(1, -1)
    if not bool((slot_groups == first_slots + offsets).all()):
        raise RuntimeError(
            "kivi_pack_key_cache requires each token group to map to contiguous slots."
        )
    if not bool(((first_slots % group_size) == 0).all()):
        raise RuntimeError(
            "kivi_pack_key_cache requires each token group to be group-size aligned."
        )
    if not bool(((slot_groups // block_size) == (first_slots // block_size)).all()):
        raise RuntimeError(
            "kivi_pack_key_cache requires each token group to stay "
            "within one cache block."
        )


@triton.jit
def _kivi_pack_value_cache_kernel(
    value_ptr,
    slot_mapping_ptr,
    v_quant_cache_ptr,
    v_scale_cache_ptr,
    v_mn_cache_ptr,
    num_tokens,
    block_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    group_size: tl.constexpr,
    num_groups: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    total_groups = num_tokens * num_kv_heads * num_groups

    group_offsets = tl.arange(0, group_size)
    lanes = tl.arange(0, 8)
    shifts = lanes * 4

    for linear_idx in tl.range(pid, total_groups, num_programs):
        group_idx = linear_idx % num_groups
        tmp = linear_idx // num_groups
        head_idx = tmp % num_kv_heads
        token_idx = tmp // num_kv_heads

        slot = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
        if slot >= 0:
            block_idx = slot // block_size
            block_offset = slot - block_idx * block_size
            head_start = group_idx * group_size
            value_base = (
                token_idx * num_kv_heads * head_size + head_idx * head_size + head_start
            )

            values = tl.load(value_ptr + value_base + group_offsets).to(tl.float32)
            mn = tl.min(values, axis=0)
            mx = tl.max(values, axis=0)
            scale = tl.maximum((mx - mn) / 15.0, 1.0e-6)

            scale_offset = (
                block_idx * block_size * num_kv_heads * num_groups
                + block_offset * num_kv_heads * num_groups
                + head_idx * num_groups
                + group_idx
            )
            tl.store(v_scale_cache_ptr + scale_offset, scale)
            tl.store(v_mn_cache_ptr + scale_offset, mn)

            for pack_base in tl.range(0, group_size, 8):
                pack_values = tl.load(value_ptr + value_base + pack_base + lanes).to(
                    tl.float32
                )
                quant = tl.minimum(
                    tl.maximum(tl.floor((pack_values - mn) / scale + 0.5), 0),
                    15,
                ).to(tl.int32)
                packed = tl.sum(
                    ((quant.to(tl.int64) & 0xF) << shifts.to(tl.int64)),
                    axis=0,
                ).to(tl.int32)
                pack_idx = (head_start + pack_base) // 8
                quant_offset = (
                    block_idx * block_size * num_kv_heads * (head_size // 8)
                    + block_offset * num_kv_heads * (head_size // 8)
                    + head_idx * (head_size // 8)
                    + pack_idx
                )
                tl.store(v_quant_cache_ptr + quant_offset, packed)


@triton.jit
def _kivi_pack_key_cache_kernel(
    key_ptr,
    slot_mapping_ptr,
    k_quant_cache_ptr,
    k_scale_cache_ptr,
    k_mn_cache_ptr,
    num_tokens,
    block_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    group_size: tl.constexpr,
    block_dim_size: tl.constexpr,
    num_dim_tiles: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    num_token_groups = num_tokens // group_size
    total_tiles = num_token_groups * num_kv_heads * num_dim_tiles

    dim_offsets = tl.arange(0, block_dim_size)
    dim_mask = dim_offsets < head_size
    token_offsets = tl.arange(0, group_size)
    lanes = tl.arange(0, 8)
    shifts = lanes[:, None] * 4

    for linear_idx in tl.range(pid, total_tiles, num_programs):
        dim_tile_idx = linear_idx % num_dim_tiles
        tmp = linear_idx // num_dim_tiles
        head_idx = tmp % num_kv_heads
        token_group_idx = tmp // num_kv_heads
        token_start = token_group_idx * group_size
        dim_start = dim_tile_idx * block_dim_size
        dims = dim_start + dim_offsets
        dim_mask = dims < head_size

        first_slot = tl.load(slot_mapping_ptr + token_start).to(tl.int64)
        if first_slot >= 0:
            block_idx = first_slot // block_size
            group_offset = first_slot - block_idx * block_size
            cache_group_idx = group_offset // group_size

            values = tl.load(
                key_ptr
                + (token_start + token_offsets[:, None]) * num_kv_heads * head_size
                + head_idx * head_size
                + dims[None, :],
                mask=dim_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            mn = tl.min(values, axis=0)
            mx = tl.max(values, axis=0)
            scale = tl.maximum((mx - mn) / 15.0, 1.0e-6)

            scale_offsets = (
                block_idx * num_kv_heads * head_size * (block_size // group_size)
                + head_idx * head_size * (block_size // group_size)
                + dims * (block_size // group_size)
                + cache_group_idx
            )
            tl.store(k_scale_cache_ptr + scale_offsets, scale, mask=dim_mask)
            tl.store(k_mn_cache_ptr + scale_offsets, mn, mask=dim_mask)

            for pack_base in tl.range(0, group_size, 8):
                pack_values = tl.load(
                    key_ptr
                    + (token_start + pack_base + lanes[:, None])
                    * num_kv_heads
                    * head_size
                    + head_idx * head_size
                    + dims[None, :],
                    mask=dim_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                quant = tl.minimum(
                    tl.maximum(
                        tl.floor((pack_values - mn[None, :]) / scale[None, :] + 0.5),
                        0,
                    ),
                    15,
                ).to(tl.int32)
                packed = tl.sum(
                    ((quant.to(tl.int64) & 0xF) << shifts.to(tl.int64)),
                    axis=0,
                ).to(tl.int32)
                pack_idx = (group_offset + pack_base) // 8
                quant_offsets = (
                    block_idx * num_kv_heads * head_size * (block_size // 8)
                    + head_idx * head_size * (block_size // 8)
                    + dims * (block_size // 8)
                    + pack_idx
                )
                tl.store(k_quant_cache_ptr + quant_offsets, packed, mask=dim_mask)


def kivi_pack_value_cache(
    value: torch.Tensor,  # 形状是 [num_tokens, num_kv_heads, head_size]
    slot_mapping: torch.Tensor,
    v_quant_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    v_mn_cache: torch.Tensor,
    group_size: int,
) -> None:
    """把一批全精度值打包写进分页 int4 值缓存（值按 head 维分组量化）。

    值布局 [NB, B, KVH, H/8]：每个 token 一行，scale/mn 每 head 维组一份；
    slot_mapping 必须落在同一缓存块内且对齐（内部校验器强制）。
    """
    num_tokens, num_kv_heads, head_size = value.shape
    if num_tokens == 0:
        return

    # Value packing already supports submitting multiple aligned cache blocks
    # in one launch as long as slot_mapping covers all tokens.

    assert group_size % 8 == 0
    assert head_size % group_size == 0
    assert head_size % 8 == 0

    _check_contiguous("value", value)
    _check_contiguous("slot_mapping", slot_mapping)
    _check_contiguous("v_quant_cache", v_quant_cache)
    _check_contiguous("v_scale_cache", v_scale_cache)
    _check_contiguous("v_mn_cache", v_mn_cache)
    _check_same_device("slot_mapping", slot_mapping, value)
    _check_same_device("v_quant_cache", v_quant_cache, value)
    _check_same_device("v_scale_cache", v_scale_cache, value)
    _check_same_device("v_mn_cache", v_mn_cache, value)
    _check_slot_mapping(slot_mapping, num_tokens)
    _check_value_cache_layout(
        v_quant_cache,
        v_scale_cache,
        v_mn_cache,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        group_size=group_size,
    )

    _kivi_pack_value_cache_kernel[(get_vectorcore_num(),)](
        value,
        slot_mapping,
        v_quant_cache,
        v_scale_cache,
        v_mn_cache,
        num_tokens,
        v_quant_cache.shape[1],
        num_kv_heads,
        head_size,
        group_size,
        head_size // group_size,
    )


def kivi_pack_key_cache(
    key: torch.Tensor,  # 形状是 [num_tokens, num_kv_heads, head_size]
    slot_mapping: torch.Tensor,
    k_quant_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    k_mn_cache: torch.Tensor,
    group_size: int,
) -> None:
    """把一批全精度键打包写进分页 int4 键缓存（键按 token 组量化）。

    键布局 [NB, KVH, H, B/8]：同一缓存块内每 8 个连续 token 的 int4 打进
    一个 int32 word（lane = 块内偏移 % 8），scale/mn 每组一份。
    """
    num_tokens, num_kv_heads, head_size = key.shape
    if num_tokens == 0:
        return

    # Key packing supports multiple token groups in one launch. Each group
    # must still map to contiguous aligned slots within a single cache block.

    assert group_size % 8 == 0
    assert num_tokens % group_size == 0
    assert head_size % 8 == 0

    _check_contiguous("key", key)
    _check_contiguous("slot_mapping", slot_mapping)
    _check_contiguous("k_quant_cache", k_quant_cache)
    _check_contiguous("k_scale_cache", k_scale_cache)
    _check_contiguous("k_mn_cache", k_mn_cache)
    _check_same_device("slot_mapping", slot_mapping, key)
    _check_same_device("k_quant_cache", k_quant_cache, key)
    _check_same_device("k_scale_cache", k_scale_cache, key)
    _check_same_device("k_mn_cache", k_mn_cache, key)
    _check_slot_mapping(slot_mapping, num_tokens)
    block_size = _check_key_cache_layout(  # blocksize=128
        k_quant_cache,  # k_quant_cache.shape == [num_blocks, num_kv_heads, head_size, block_size // 8]
        k_scale_cache,  # k_scale_cache.shape == [num_blocks, num_kv_heads, head_size, block_size // group_size]
        k_mn_cache,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        group_size=group_size,
    )
    _check_key_slot_groups(
        slot_mapping,
        block_size=block_size,
        group_size=group_size,
    )

    _kivi_pack_key_cache_kernel[(get_vectorcore_num(),)](
        key,
        slot_mapping,
        k_quant_cache,
        k_scale_cache,
        k_mn_cache,
        num_tokens,
        block_size,
        num_kv_heads,
        head_size,
        group_size,
        min(16, triton.next_power_of_2(head_size)),
        triton.cdiv(head_size, min(16, triton.next_power_of_2(head_size))),
    )
