# SPDX-License-Identifier: Apache-2.0
"""NPU tensor helpers for the dynamic per-channel INT8 solution.

Ported from legacy ascend PR #116 commit 0001. The gather-and-dequant path
runs on plain torch ops, so it is exercised by the CPU unit tests; the NPU
fused-inference calls stay inside the attention mixin.
"""

from __future__ import annotations

from typing import Any

import torch


def dequant_paged_kv_to_dense(
    key: Any,
    value: Any,
    block_table: Any,
    seq_lens: list[int],
    target_dtype: Any,
    *,
    num_kv_heads: int,
    head_size: int,
    k_inv_scale: Any,
    k_offset: Any,
    v_inv_scale: Any,
    v_offset: Any,
) -> tuple[Any, Any]:
    """Gather paged INT8 KV blocks and dequantize to a dense target dtype.

    Mirrors ``AscendAttentionBackendImpl._dequant_paged_kv_to_dense``.
    """
    batch_size = block_table.shape[0]
    block_size = key.shape[1]
    hidden_size = key.shape[2]
    max_blocks_per_seq = block_table.shape[1]
    max_tokens_padded = max_blocks_per_seq * block_size

    flat_ids = block_table.reshape(-1)
    gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, hidden_size)
    gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, hidden_size)

    seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
    positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
    valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

    dense_k = gathered_k.view(-1, hidden_size)[valid_mask]
    dense_v = gathered_v.view(-1, hidden_size)[valid_mask]
    dense_k = dense_k.view(-1, num_kv_heads, head_size)
    dense_v = dense_v.view(-1, num_kv_heads, head_size)
    dense_k = (dense_k.to(target_dtype) - k_offset) * (1.0 / k_inv_scale)
    dense_v = (dense_v.to(target_dtype) - v_offset) * (1.0 / v_inv_scale)
    return dense_k, dense_v


__all__ = ["dequant_paged_kv_to_dense"]
