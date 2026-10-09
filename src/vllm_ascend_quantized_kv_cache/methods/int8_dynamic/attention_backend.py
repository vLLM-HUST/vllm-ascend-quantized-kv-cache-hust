# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plugin-owned port of ``AscendInt8AttentionBackendImpl``.

The implementation body is kept in this repository.  It is mixed over the
live host ``AscendAttentionBackendImpl`` by the adapter, so the plugin owns the
INT8 policy while reusing the host's cache writer, metadata and graph helpers.
"""

from __future__ import annotations

from typing import Any

import torch
import torch_npu
from vllm.v1.attention.backend import AttentionType
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.attention.attention_v1 import AscendAttentionState


class AscendInt8AttentionBackendMixin:
    """Runtime-calibrated INT8 KV cache implementation migrated from host."""

    def _nz_5d_view(self, cache: torch.Tensor, block_size: int) -> torch.Tensor:
        nz_fmt_last_dim = 32
        return cache.view(
            -1,
            self.num_kv_heads,
            self.head_size // nz_fmt_last_dim,
            block_size,
            nz_fmt_last_dim,
        )

    def forward(
        self,
        layer: Any,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor],
        attn_metadata: Any,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."
        if self.enable_hamming_sparse:
            self.layerIndex = int(layer.layer_name.split(".")[2])
        if self._use_layer_aware_fia_graph_replay:
            self._layer_name = layer.layer_name

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for "
                "AscendInt8AttentionBackendImpl"
            )

        assert layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0
        num_tokens = query.shape[0]
        if attn_metadata is None:
            return output.fill_(0)

        if (
            self.key_cache is None
            and kv_cache is not None
            and (
                isinstance(kv_cache, torch.Tensor)
                and kv_cache.dim() > 0
                and kv_cache.shape[0] == 2
                or isinstance(kv_cache, list | tuple)
                and len(kv_cache) >= 2
            )
        ):
            self.key_cache, self.value_cache = kv_cache[0], kv_cache[1]

        output_padded = None
        float_key, float_value = None, None
        if key is not None and value is not None:
            output_padded = output
            if attn_metadata.attn_state != AscendAttentionState.DecodeOnly:
                float_key, float_value = key, value
            self._prepare_int8_scales(layer, key, value)
            key, value = self._quantize_kv_to_runtime_int8(
                key, value, layer, attn_metadata.num_actual_tokens
            )
            query, key, value, output_padded = super().reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )

        if attn_metadata.model_runner_type == "pooling" and not attn_metadata.causal:
            attn_output = self._forward_encoder_attention(
                query, key, value, attn_metadata, output
            )
            output[:num_tokens] = attn_output[:num_tokens]
            return output

        if getattr(layer, "_int8_scales_ready", False):
            if attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                if _EXTRA_CTX.capturing:
                    attn_output, num_tokens = self.full_graph_fia(
                        query, key, value, attn_metadata, output, layer
                    )
                    output[:num_tokens] = attn_output[:num_tokens]
                    return output
                return self._forward_int8_decode(query, attn_metadata, output, layer)
            if attn_metadata.attn_state == AscendAttentionState.ChunkedPrefill:
                return self._forward_int8_chunked_prefill(
                    query, float_key, float_value, attn_metadata, output, layer
                )
            return self._forward_int8_prefill(
                query,
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
                layer,
            )

        if output_padded is not None:
            attn_output = self.forward_impl(
                query, key, value, kv_cache, attn_metadata, output_padded
            )
        else:
            attn_output = self.forward_impl(
                query, key, value, kv_cache, attn_metadata, output
            )
        output[:num_tokens] = attn_output[:num_tokens]
        return output

    def _prepare_int8_scales(
        self,
        layer: Any,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        if getattr(layer, "_int8_scales_ready", False):
            return
        k_max = key.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        v_max = value.abs().amax(dim=0, keepdim=True).clamp(min=1e-12)
        layer._int8_k_inv_scale = 127.0 / k_max
        layer._int8_v_inv_scale = 127.0 / v_max
        layer._int8_k_offset = torch.zeros_like(layer._int8_k_inv_scale)
        layer._int8_v_offset = torch.zeros_like(layer._int8_v_inv_scale)

        bnsd = (1, self.num_kv_heads, 1, self.head_size)
        layer._int8_k_aq_scale = (1.0 / layer._int8_k_inv_scale).view(bnsd).contiguous()
        layer._int8_v_aq_scale = (1.0 / layer._int8_v_inv_scale).view(bnsd).contiguous()
        layer._int8_k_aq_offset = layer._int8_k_offset.view(bnsd).contiguous()
        layer._int8_v_aq_offset = layer._int8_v_offset.view(bnsd).contiguous()
        layer._int8_scales_ready = True

    @staticmethod
    def _quantize_kv_to_runtime_int8(
        key: torch.Tensor,
        value: torch.Tensor,
        layer: Any,
        num_actual_tokens: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        actual_key = key[:num_actual_tokens]
        actual_value = value[:num_actual_tokens]
        k_int8 = torch.clamp(
            torch.round(actual_key * layer._int8_k_inv_scale + layer._int8_k_offset),
            -128,
            127,
        ).to(torch.int8)
        v_int8 = torch.clamp(
            torch.round(actual_value * layer._int8_v_inv_scale + layer._int8_v_offset),
            -128,
            127,
        ).to(torch.int8)
        return k_int8, v_int8

    def _forward_int8_decode(
        self,
        query: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            key_antiquant_scale=layer._int8_k_aq_scale,
            key_antiquant_offset=layer._int8_k_aq_offset,
            value_antiquant_scale=layer._int8_v_aq_scale,
            value_antiquant_offset=layer._int8_v_aq_offset,
            key_antiquant_mode=0,
            value_antiquant_mode=0,
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            sparse_mode=0,
            scale=self.scale,
            block_size=block_size,
        )
        output[:batch_size] = attn_output.squeeze(2)
        return output

    def _forward_int8_chunked_prefill(
        self,
        query: torch.Tensor,
        float_key: torch.Tensor | None,
        float_value: torch.Tensor | None,
        attn_metadata: Any,
        output: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])

        if num_decode_tokens > 0:
            num_block, block_size, _, _ = self.key_cache.shape
            kv_k = self.key_cache.view(num_block, block_size, -1)
            kv_v = self.value_cache.view(num_block, block_size, -1)
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode_tokens].unsqueeze(2),
                kv_k,
                kv_v,
                key_antiquant_scale=layer._int8_k_aq_scale,
                key_antiquant_offset=layer._int8_k_aq_offset,
                value_antiquant_scale=layer._int8_v_aq_scale,
                value_antiquant_offset=layer._int8_v_aq_offset,
                key_antiquant_mode=0,
                value_antiquant_mode=0,
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                sparse_mode=0,
                scale=self.scale,
                block_size=block_size,
            )
            output[:num_decode_tokens] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode_tokens:num_tokens]
            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode_tokens
                for i in range(num_decodes, len(actual_seq_qlen))
            ]
            all_new_prefill = True
            for i in range(num_decodes, len(attn_metadata.seq_lens_list)):
                q_start = actual_seq_qlen[i - 1] if i > 0 else 0
                qlen_i = actual_seq_qlen[i] - q_start
                if attn_metadata.seq_lens_list[i] > qlen_i:
                    all_new_prefill = False
                    break

            if all_new_prefill and float_key is not None and float_value is not None:
                prefill_k = float_key[num_decode_tokens:num_tokens]
                prefill_v = float_value[num_decode_tokens:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, block_size, _, _ = self.key_cache.shape
                paged_k = self.key_cache.view(num_block, block_size, -1)
                paged_v = self.value_cache.view(num_block, block_size, -1)
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_int8_paged_kv_to_dense(
                    paged_k,
                    paged_v,
                    attn_metadata.block_tables[num_decodes:],
                    prefill_sl,
                    query.dtype,
                    layer,
                )
                prefill_seq_kvlen = torch.tensor(prefill_sl, dtype=torch.int32).cumsum(
                    dim=0
                )

            cache_block_size = self.key_cache.shape[1]
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query=prefill_q,
                key=prefill_k,
                value=prefill_v,
                atten_mask=attn_metadata.attn_mask,
                block_table=None,
                input_layout="TND",
                sparse_mode=3,
                block_size=cache_block_size,
                actual_seq_lengths=prefill_seq_qlen,
                actual_seq_lengths_kv=prefill_seq_kvlen,
                num_key_value_heads=self.num_kv_heads,
                num_heads=self.num_heads,
                scale=self.scale,
            )
            n_prefill = num_tokens - num_decode_tokens
            output[num_decode_tokens:num_tokens] = attn_out.view(
                n_prefill, self.num_heads, self.head_size
            )[:n_prefill]

        return output

    def _forward_int8_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: Any,
        output: torch.Tensor,
        layer: Any,
    ) -> torch.Tensor:
        key, value, block_size, block_table, actual_seq_lengths_kv = (
            self._get_fia_params(key, value, attn_metadata)
        )
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        query = query[:num_tokens]

        if (
            attn_metadata.attn_state == AscendAttentionState.PrefillNoCache
            and self.attn_type != AttentionType.ENCODER_DECODER
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv
                    if isinstance(actual_seq_lengths_kv, list)
                    else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_int8_paged_kv_to_dense(
                    key, value, block_table, seq_lens, query.dtype, layer
                )
                block_table = None
                block_size = self.key_cache.shape[1]
                actual_seq_lengths_kv = torch.tensor(
                    seq_lens, dtype=torch.int32
                ).cumsum(dim=0)
            else:
                key = (key.to(query.dtype) - layer._int8_k_offset) * (
                    1.0 / layer._int8_k_inv_scale
                )
                value = (value.to(query.dtype) - layer._int8_v_offset) * (
                    1.0 / layer._int8_v_inv_scale
                )

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query=query,
            key=key,
            value=value,
            atten_mask=attn_metadata.attn_mask,
            block_table=block_table,
            input_layout="TND",
            sparse_mode=3,
            block_size=block_size,
            actual_seq_lengths=actual_seq_qlen,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            num_key_value_heads=self.num_kv_heads,
            num_heads=self.num_heads,
            scale=self.scale,
        )
        output[:num_tokens] = attn_output.view(
            num_tokens, self.num_heads, self.head_size
        )
        return output

    def _dequant_int8_paged_kv_to_dense(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: list[int],
        target_dtype: torch.dtype,
        layer: Any,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = block_table.shape[0]
        block_size = key.shape[1]
        hidden_size = key.shape[2]
        max_tokens_padded = block_table.shape[1] * block_size

        flat_ids = block_table.reshape(-1)
        gathered_k = key[flat_ids].view(batch_size, max_tokens_padded, hidden_size)
        gathered_v = value[flat_ids].view(batch_size, max_tokens_padded, hidden_size)

        seq_lens_t = torch.tensor(seq_lens, dtype=torch.long, device=key.device)
        positions = torch.arange(max_tokens_padded, dtype=torch.long, device=key.device)
        valid_mask = (positions.unsqueeze(0) < seq_lens_t.unsqueeze(1)).view(-1)

        dense_k = gathered_k.view(-1, hidden_size)[valid_mask].view(
            -1, self.num_kv_heads, self.head_size
        )
        dense_v = gathered_v.view(-1, hidden_size)[valid_mask].view(
            -1, self.num_kv_heads, self.head_size
        )
        dense_k = (dense_k.to(target_dtype) - layer._int8_k_offset) * (
            1.0 / layer._int8_k_inv_scale
        )
        dense_v = (dense_v.to(target_dtype) - layer._int8_v_offset) * (
            1.0 / layer._int8_v_inv_scale
        )
        return dense_k, dense_v


__all__ = ["AscendInt8AttentionBackendMixin"]
