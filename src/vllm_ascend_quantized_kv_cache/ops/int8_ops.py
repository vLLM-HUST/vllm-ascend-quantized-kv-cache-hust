# SPDX-License-Identifier: Apache-2.0
"""动态 per-channel INT8 方案的 NPU 张量辅助算子。

移植自 legacy ascend PR #116 提交 0001。gather + 反量化路径是普通
torch 运算，CPU 单测直接覆盖；NPU fused-inference 调用则留在
attention mixin 里（依赖 torch_npu 与 attn_metadata）。
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
    """按块表 gather 分页 INT8 KV，并反量化成稠密目标 dtype。

    对应 legacy ``_dequant_paged_kv_to_dense``：把 [batch, max_blocks]
    个缓存块摊平 -> 用 seq_lens 掩掉无效槽位 -> 逐通道反量化。
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
