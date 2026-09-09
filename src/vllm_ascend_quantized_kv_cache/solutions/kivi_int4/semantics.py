# SPDX-License-Identifier: Apache-2.0
"""Pure semantics for the KIVI INT4 solution (CPU-testable).

Mined from legacy ascend PR #116 commits 0003-0009 (final state). KIVI
quantizes the *history* region of the KV cache per token-group for keys and
per head-dim-group for values, keeping the most recent ``residual_length``
tokens full precision in a residual window; once a key window fills up it
is flushed into the paged int4 cache as whole groups.

The device-side packing/gathering runs in triton-ascend kernels
(``ops.triton.kivi_cache``); everything in this module is plain torch and
runs on CPU so the numerics are unit-testable.
"""

from __future__ import annotations

from typing import Any

import torch


def validate_kivi_geometry(
    *,
    head_size: int,
    group_size: int,
    residual_length: int,
    block_size: int | None = None,
) -> None:
    """Mirror the layout invariants enforced by the legacy implementation.

    ``block_size`` is optional: impl objects derive it from the bound cache,
    and ``_write_kivi_key_quant_cache`` re-checks it at flush time.
    """
    if group_size <= 0 or group_size % 8:
        raise ValueError(
            f"kivi_int4 requires group_size divisible by 8, got {group_size}"
        )
    if residual_length % group_size:
        raise ValueError(
            f"kivi_int4 requires residual_length ({residual_length}) "
            f"to be divisible by group_size ({group_size})"
        )
    if head_size % 8:
        raise ValueError(
            f"kivi_int4 requires head_size divisible by 8, got {head_size}"
        )
    if head_size % group_size:
        raise ValueError(
            f"kivi_int4 requires head_size ({head_size}) to be "
            f"divisible by group_size ({group_size})"
        )
    if block_size is not None and block_size % group_size:
        raise ValueError(
            f"kivi_int4 requires block_size ({block_size}) to be "
            f"divisible by group_size ({group_size})"
        )


def validate_kivi_config(config: Any) -> None:
    """Accept either a SolutionConfig or a live impl (kivi_* attributes)."""
    group_size = getattr(config, "group_size", None)
    if group_size is None:
        group_size = getattr(config, "kivi_group_size", None)
    residual_length = getattr(config, "residual_length", None)
    if residual_length is None:
        residual_length = getattr(config, "kivi_residual_length", None)
    if group_size is None or residual_length is None:
        raise ValueError(
            "kivi_int4 config must define group_size/residual_length "
            "(or kivi_group_size/kivi_residual_length)"
        )
    validate_kivi_geometry(
        head_size=config.head_size,
        group_size=group_size,
        residual_length=residual_length,
        block_size=getattr(config, "block_size", None),
    )


def _normalize_config(config: Any) -> Any:
    """Expose a uniform group_size/residual_length view of *config*."""
    if hasattr(config, "group_size"):
        return config
    from types import SimpleNamespace

    return SimpleNamespace(
        head_size=config.head_size,
        group_size=config.kivi_group_size,
        residual_length=config.kivi_residual_length,
        block_size=getattr(config, "block_size", None),
    )


class KiviInt4Semantics:
    def __init__(self, config: Any) -> None:
        validate_kivi_config(config)
        self.config = _normalize_config(config)

    # -- config geometry ----------------------------------------------------

    @property
    def group_size(self) -> int:
        return self.config.group_size

    @property
    def bits(self) -> int:
        return 4

    def packed_key_last_dim(self) -> int:
        """int32 words per token row for keys: head_size / 8."""
        return self.config.head_size // 8

    # -- packing math -------------------------------------------------------

    @staticmethod
    def unpack_int4(packed: Any) -> Any:
        """Vectorized unpack of int32 into 8 int4 lanes (last dim)."""
        packed_i32 = packed.to(torch.int32)
        shifts = torch.arange(8, device=packed.device, dtype=torch.int32) * 4
        unpacked = (packed_i32.unsqueeze(-1) >> shifts) & 0xF
        return unpacked.to(torch.float32)

    @staticmethod
    def pack_int4(quant: Any) -> Any:
        quant = quant.to(torch.int32)
        packed = torch.zeros_like(quant[..., 0], dtype=torch.int32)
        for lane in range(8):
            packed = packed | (quant[..., lane] << (lane * 4))
        return packed

    def fake_quant_key(self, key: Any) -> Any:
        """Token-group fake quantization of keys [T, H, D]."""
        if key.numel() == 0:
            return key
        group_size = self.group_size
        num_tokens = key.shape[0]
        pad_tokens = (group_size - num_tokens % group_size) % group_size
        work = key.to(torch.float32)
        if pad_tokens:
            pad = work[-1:].expand(pad_tokens, -1, -1)
            work = torch.cat([work, pad], dim=0)
        grouped = work.view(-1, group_size, *key.shape[1:])
        mn = grouped.amin(dim=1, keepdim=True)
        mx = grouped.amax(dim=1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-6) / (2**self.bits - 1)
        quant = torch.clamp(torch.round((grouped - mn) / scale), 0, 2**self.bits - 1)
        dequant = (quant * scale + mn).view(-1, *key.shape[1:])
        return dequant[:num_tokens].to(key.dtype)

    def fake_quant_value(self, value: Any) -> Any:
        """Head-dim-group fake quantization of values [T, H, D]."""
        if value.numel() == 0:
            return value
        group_size = self.group_size
        head_size = value.shape[-1]
        pad_dim = (group_size - head_size % group_size) % group_size
        work = value.to(torch.float32)
        if pad_dim:
            pad = work[..., -1:].expand(*work.shape[:-1], pad_dim)
            work = torch.cat([work, pad], dim=-1)
        grouped = work.view(*work.shape[:-1], -1, group_size)
        mn = grouped.amin(dim=-1, keepdim=True)
        mx = grouped.amax(dim=-1, keepdim=True)
        scale = (mx - mn).clamp(min=1e-6) / (2**self.bits - 1)
        quant = torch.clamp(torch.round((grouped - mn) / scale), 0, 2**self.bits - 1)
        dequant = (quant * scale + mn).view(*work.shape)
        return dequant[..., :head_size].to(value.dtype)

    def dequant_key_blocks(
        self, k_quant: Any, k_scale: Any, k_mn: Any, target_dtype: Any
    ) -> Any:
        """Dequantize grouped key blocks -> [B, blocks, block_size, H, D]."""
        q = self.unpack_int4(k_quant).flatten(-2)
        block_size = q.shape[-1]
        scale = k_scale.repeat_interleave(self.group_size, dim=-1)[..., :block_size]
        mn = k_mn.repeat_interleave(self.group_size, dim=-1)[..., :block_size]
        deq = q.to(scale.dtype) * scale + mn
        return deq.permute(0, 1, 4, 2, 3).contiguous().to(target_dtype)

    def dequant_value_blocks(
        self, v_quant: Any, v_scale: Any, v_mn: Any, target_dtype: Any
    ) -> Any:
        """Dequantize value blocks -> [B, blocks, block_size, H, D]."""
        q = self.unpack_int4(v_quant).flatten(-2)
        head_size = q.shape[-1]
        scale = v_scale.repeat_interleave(self.group_size, dim=-1)[..., :head_size]
        mn = v_mn.repeat_interleave(self.group_size, dim=-1)[..., :head_size]
        deq = q.to(scale.dtype) * scale + mn
        return deq.contiguous().to(target_dtype)

    @staticmethod
    def normalize_block_layout(
        blocks: Any,
        batch_size: int,
        max_blocks: int,
        cache_block_size: int,
        name: str,
    ) -> Any:
        """Accept [B, blocks, block_size, ...] or the transposed layout."""
        if blocks.ndim != 5:
            raise RuntimeError(
                f"KIVI {name} cache must be 5D after dequant, got "
                f"shape={tuple(blocks.shape)}."
            )
        if blocks.shape[0] != batch_size:
            raise RuntimeError(
                f"KIVI {name} batch mismatch after dequant: expected "
                f"{batch_size}, got {blocks.shape[0]}."
            )
        if blocks.shape[1] == max_blocks and blocks.shape[2] == cache_block_size:
            return blocks
        if blocks.shape[1] == cache_block_size and blocks.shape[2] == max_blocks:
            return blocks.transpose(1, 2).contiguous()
        raise RuntimeError(
            f"KIVI {name} cache layout mismatch: shape={tuple(blocks.shape)}, "
            f"expected (*, {max_blocks}, {cache_block_size}, ...)."
        )

    # -- window bookkeeping (pure list logic) --------------------------------

    @staticmethod
    def cu_seqlens_to_seq_lens(cu_seqlens: Any) -> list[int]:
        if isinstance(cu_seqlens, torch.Tensor):
            cu_seqlens = cu_seqlens.tolist()
        prev = 0
        seq_lens = []
        for end in cu_seqlens:
            seq_lens.append(int(end) - prev)
            prev = int(end)
        return seq_lens

    @staticmethod
    def ordered_slots(
        block_table: Any, seq_lens: list[int], block_size: int
    ) -> list[list[int]]:
        """Flatten (block_id, offset) pairs into absolute slot ids per request."""
        block_table = block_table.to(torch.long)
        ordered_slots: list[list[int]] = []
        for req_idx, seq_len in enumerate(seq_lens):
            req_slots: list[int] = []
            for pos in range(int(seq_len)):
                block_pos = pos // block_size
                block_offset = pos % block_size
                block_id = int(block_table[req_idx, block_pos].item())
                if block_id >= 0:
                    req_slots.append(block_id * block_size + block_offset)
            ordered_slots.append(req_slots)
        return ordered_slots

    def is_aligned_key_window(self, window_slots: list[int], block_size: int) -> bool:
        """True when every group in *window_slots* is contiguous, aligned to
        its block, and fully contained in a single cache block."""
        if not window_slots:
            return False
        group_size = self.group_size
        if len(window_slots) % group_size:
            return False

        for start in range(0, len(window_slots), group_size):
            group_slots = window_slots[start : start + group_size]
            if len(group_slots) != group_size:
                return False

            first_slot = int(group_slots[0])
            last_slot = int(group_slots[-1])
            block_idx = first_slot // block_size
            block_offset = first_slot % block_size

            if block_offset % group_size != 0:
                return False
            if last_slot // block_size != block_idx:
                return False

            expected = list(range(first_slot, first_slot + group_size))
            if group_slots != expected:
                return False
        return True

    def build_causal_mask(
        self, q_len: int, kv_seq_len: int, dtype: Any, device: Any
    ) -> Any:
        """Additive causal mask for the pure-torch dense attention path."""
        q_pos = torch.arange(
            kv_seq_len - q_len, kv_seq_len, dtype=torch.long, device=device
        )
        kv_pos = torch.arange(kv_seq_len, dtype=torch.long, device=device)
        mask = kv_pos.unsqueeze(0) > q_pos.unsqueeze(1)
        neg_inf = torch.finfo(dtype).min
        return torch.where(
            mask,
            torch.full((), neg_inf, dtype=dtype, device=device),
            torch.zeros((), dtype=dtype, device=device),
        ).unsqueeze(0)

    def describe(self) -> dict[str, Any]:
        return {
            "scheme": "kivi_int4",
            "bits": self.bits,
            "group_size": self.group_size,
            "residual_length": self.config.residual_length,
            "key_grouping": "token groups (residual window flushed whole)",
            "value_grouping": "head-dim groups (per token)",
        }


__all__ = ["KiviInt4Semantics", "validate_kivi_config", "validate_kivi_geometry"]
