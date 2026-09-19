# SPDX-License-Identifier: Apache-2.0
"""KIVI 缓存的共享布局校验器与解包辅助。

纯 torch，不依赖 triton 即可导入。triton 打包内核与 torch 版反量化
gather 共用这里的校验，保证两条路径对"合法布局"的认知一致。
"""

from __future__ import annotations

import torch


def _check_same_device(name: str, tensor: torch.Tensor, ref: torch.Tensor) -> None:
    if tensor.device != ref.device:
        raise RuntimeError(
            f"{name} must be on device {ref.device}, got {tensor.device}."
        )


def _check_contiguous(name: str, tensor: torch.Tensor) -> None:
    if not tensor.is_contiguous():
        raise RuntimeError(f"{name} must be contiguous.")


def _check_value_cache_layout(
    v_quant_cache: torch.Tensor,
    v_scale_cache: torch.Tensor,
    v_mn_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
    group_size: int,
) -> None:
    if v_quant_cache.ndim != 4:
        raise RuntimeError(
            f"v_quant_cache must be 4D, got shape={tuple(v_quant_cache.shape)}."
        )
    if v_scale_cache.ndim != 4 or v_mn_cache.ndim != 4:
        raise RuntimeError(
            "v_scale_cache and v_mn_cache must be 4D, got "
            f"{tuple(v_scale_cache.shape)} and {tuple(v_mn_cache.shape)}."
        )

    num_blocks, block_size, cache_heads, packed_head_size = v_quant_cache.shape
    expected_quant = (num_blocks, block_size, num_kv_heads, head_size // 8)
    expected_scale = (num_blocks, block_size, num_kv_heads, head_size // group_size)
    if tuple(v_quant_cache.shape) != expected_quant:
        raise RuntimeError(
            "v_quant_cache layout mismatch: got "
            f"{tuple(v_quant_cache.shape)}, expected {expected_quant}."
        )
    if tuple(v_scale_cache.shape) != expected_scale:
        raise RuntimeError(
            "v_scale_cache layout mismatch: got "
            f"{tuple(v_scale_cache.shape)}, expected {expected_scale}."
        )
    if tuple(v_mn_cache.shape) != expected_scale:
        raise RuntimeError(
            "v_mn_cache layout mismatch: got "
            f"{tuple(v_mn_cache.shape)}, expected {expected_scale}."
        )
    if packed_head_size != head_size // 8 or cache_heads != num_kv_heads:
        raise RuntimeError(
            "v_quant_cache head layout does not match input value tensor."
        )


def _check_key_cache_layout(
    k_quant_cache: torch.Tensor,
    k_scale_cache: torch.Tensor,
    k_mn_cache: torch.Tensor,
    *,
    num_kv_heads: int,
    head_size: int,
    group_size: int,
) -> int:
    if k_quant_cache.ndim != 4:
        raise RuntimeError(
            f"k_quant_cache must be 4D, got shape={tuple(k_quant_cache.shape)}."
        )
    if k_scale_cache.ndim != 4 or k_mn_cache.ndim != 4:
        raise RuntimeError(
            "k_scale_cache and k_mn_cache must be 4D, got "
            f"{tuple(k_scale_cache.shape)} and {tuple(k_mn_cache.shape)}."
        )

    num_blocks, cache_heads, cache_head_size, packed_block_size = k_quant_cache.shape
    block_size = packed_block_size * 8
    expected_quant = (num_blocks, num_kv_heads, head_size, block_size // 8)
    expected_scale = (num_blocks, num_kv_heads, head_size, block_size // group_size)
    if tuple(k_quant_cache.shape) != expected_quant:
        raise RuntimeError(
            "k_quant_cache layout mismatch: got "
            f"{tuple(k_quant_cache.shape)}, expected {expected_quant}."
        )
    if tuple(k_scale_cache.shape) != expected_scale:
        raise RuntimeError(
            "k_scale_cache layout mismatch: got "
            f"{tuple(k_scale_cache.shape)}, expected {expected_scale}."
        )
    if tuple(k_mn_cache.shape) != expected_scale:
        raise RuntimeError(
            "k_mn_cache layout mismatch: got "
            f"{tuple(k_mn_cache.shape)}, expected {expected_scale}."
        )
    if cache_heads != num_kv_heads or cache_head_size != head_size:
        raise RuntimeError("k_quant_cache head layout does not match input key tensor.")
    return block_size


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
    """键整组 flush 的内核侧闸门：连续、组对齐、且不跨缓存块。

    与 ``semantics.is_aligned_key_window`` 是同一条规则的两处实现（一处送
    内核前、一处状态机里），由 ``test_kivi_slot_group_guards_agree`` 钉住
    一致性。
    """
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


def _check_block_table(block_table: torch.Tensor) -> None:
    if block_table.ndim != 2:
        raise RuntimeError(
            f"block_table must be 2D, got shape={tuple(block_table.shape)}."
        )
    if block_table.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"block_table must be int32/int64, got {block_table.dtype}.")


def _check_seq_lens(seq_lens: torch.Tensor, batch_size: int) -> None:
    if seq_lens.ndim != 1:
        raise RuntimeError(f"seq_lens must be 1D, got shape={tuple(seq_lens.shape)}.")
    if seq_lens.numel() != batch_size:
        raise RuntimeError(
            "seq_lens length must match batch size, "
            f"got {seq_lens.numel()} vs {batch_size}.",
        )
    if seq_lens.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"seq_lens must be int32/int64, got {seq_lens.dtype}.")


def _check_dequant_gather_metadata(
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    num_blocks: int,
    block_size: int,
) -> tuple[torch.Tensor, int, int]:
    batch_size = int(seq_lens.numel())
    _check_block_table(block_table)
    _check_seq_lens(seq_lens, batch_size)
    _check_contiguous("block_table", block_table)
    _check_contiguous("seq_lens", seq_lens)

    if block_table.shape[0] != batch_size:
        raise RuntimeError(
            "block_table batch size must match seq_lens, "
            f"got {block_table.shape[0]} vs {batch_size}.",
        )
    max_seq_len = int(seq_lens.max().item()) if batch_size > 0 else 0
    if max_seq_len > block_table.shape[1] * block_size:
        raise RuntimeError(
            "block_table does not cover the longest KIVI sequence: "
            f"max_seq_len={max_seq_len}, max_blocks={block_table.shape[1]}, "
            f"block_size={block_size}."
        )

    block_counts = torch.div(
        seq_lens.to(torch.long) + block_size - 1,
        block_size,
        rounding_mode="floor",
    )
    block_positions = torch.arange(
        block_table.shape[1],
        dtype=torch.long,
        device=block_table.device,
    )
    live_block_mask = block_positions.unsqueeze(0) < block_counts.unsqueeze(1)
    live_block_ids = block_table[live_block_mask]
    if bool((live_block_ids < 0).any().item()):
        raise RuntimeError("KIVI block_table has invalid block ids for live tokens.")
    if bool((live_block_ids >= num_blocks).any().item()):
        raise RuntimeError("KIVI block_table references blocks outside the int4 cache.")

    total_tokens = int(seq_lens.sum().item())
    cu_seq_lens = torch.empty(
        (batch_size + 1,),
        dtype=torch.long,
        device=seq_lens.device,
    )
    cu_seq_lens[0] = 0
    cu_seq_lens[1:] = torch.cumsum(seq_lens.to(torch.long), dim=0)
    return cu_seq_lens, batch_size, total_tokens


__all__ = [
    "_check_block_table",
    "_check_contiguous",
    "_check_dequant_gather_metadata",
    "_check_key_cache_layout",
    "_check_key_slot_groups",
    "_check_same_device",
    "_check_seq_lens",
    "_check_slot_mapping",
    "_check_value_cache_layout",
]
