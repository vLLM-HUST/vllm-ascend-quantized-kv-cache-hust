# SPDX-License-Identifier: Apache-2.0
"""动态 per-channel INT8 方案的 attention backend mixin。

移植自 legacy ascend PR #116 提交 0001（AscendAttentionBackendImpl 的
INT8 分支）；宿主基类在"构造子类"时才传入（见
adapters.vllm_ascend_hust.attention.build_impl_cls 的 bind 式工厂）。

三条 forward 分支（按注意力状态分派）：
  * decode：BNSD 布局直接对分页 int8 缓存做 fused attention + antiquant；
  * chunked prefill：decode 行走 int8 缓存，prefill 行走 fp16
    （新 token 直接用，命中缓存的先 gather 反量化）；
  * prefill：TND 布局 + fp16（缓存命中时手动反量化）。

本模块导入期不碰 torch_npu 和宿主：NPU 算子在方法内惰性解析，注意力
状态按枚举成员"名字"匹配（避免 import 宿主枚举），stub 测试可完整导入。
legacy 补丁里的调试 print 已剔除；量化数学委托给 semantics 模块。
"""

from __future__ import annotations

from typing import Any

from ...core.runtime import import_torch, import_torch_npu
from ...ops import int8_ops
from .semantics import Int8DynamicSemantics


def _attn_state_name(attn_metadata: Any) -> str:
    state = getattr(attn_metadata, "attn_state", None)
    if state is None:
        return "UNKNOWN"
    return getattr(state, "name", None) or str(state)


def _attn_type_name(attn_type: Any) -> str:
    return getattr(attn_type, "name", None) or str(attn_type)


class Int8DynamicAttentionMixin:
    """Dynamic per-channel INT8 KV cache for Ascend NPU attention backends.

    Expected to be combined with the host ``AscendAttentionBackendImpl`` (or
    a stub with the same surface in tests). ``_init_int8_dynamic_state`` must
    run after the base ``__init__``.
    """

    # -- state --------------------------------------------------------------

    def _init_int8_dynamic_state(
        self, kv_cache_dtype: str | None, vllm_config: Any = None
    ) -> None:
        """建立 INT8 方案的全部实例状态（构造或类手术后调用）。"""
        # Accept every dtype spelling that reaches the impl constructor: the
        # legacy patch keyed on "int8", the layout contract on
        # "int8_per_token_head", and host adapters may use the method name.
        self.enable_int8 = kv_cache_dtype in (
            "int8",
            "int8_per_token_head",
            "int8_dynamic",
        )
        self._int8_ready = False
        self._int8_scales = None
        self.key_cache = None
        self.value_cache = None

    @property
    def _sem(self) -> Int8DynamicSemantics:
        # self 自身就带 num_kv_heads/head_size 属性，可直接当 config 用
        return Int8DynamicSemantics(self)

    # -- store path ---------------------------------------------------------

    def _calc_int8_scales(self, key: Any, value: Any) -> None:
        """首次 prefill 时沿 token 维取一次 amax（之后一直复用）。"""
        self._int8_scales = self._sem.calc_scales(key, value)
        self._int8_ready = True

    def _int8_scales_or_raise(self):
        if self._int8_scales is None:
            raise RuntimeError(
                "INT8 dynamic per-channel scales are not initialised; "
                "_calc_int8_scales must run on the first prefill."
            )
        return self._int8_scales

    @staticmethod
    def _quantize_kv_to_int8(x: Any, inv_scale: Any, offset: Any) -> Any:
        return Int8DynamicSemantics.quantize(x, inv_scale, offset)

    def _int8_quantize_for_store(self, key: Any, value: Any) -> tuple[Any, Any]:
        """新 token 进分页缓存前的量化（K/V 各用各的 scale）。"""
        scales = self._int8_scales_or_raise()
        return (
            self._quantize_kv_to_int8(key, scales.k_inv_scale, scales.k_offset),
            self._quantize_kv_to_int8(value, scales.v_inv_scale, scales.v_offset),
        )

    def do_kv_cache_update(
        self, key: Any, value: Any, kv_cache: Any, attn_metadata: Any, layer: Any
    ) -> None:
        """覆写宿主的缓存写入：先量化 K/V，再委托宿主原始路径。

        首次调用时顺便完成 scale 计算（legacy 的"只算一次"语义）。
        """
        if getattr(self, "enable_int8", False):
            if not self._int8_ready:
                self._calc_int8_scales(key, value)
            key, value = self._int8_quantize_for_store(key, value)
        super().do_kv_cache_update(key, value, kv_cache, attn_metadata, layer)

    # -- compute path -------------------------------------------------------

    def _int8_aq_kwargs(self) -> dict[str, Any]:
        """NPU fused attention 的 antiquant 反量化参数包（BNSD 视图）。"""
        scales = self._int8_scales_or_raise()
        return {
            "key_antiquant_scale": scales.k_aq_scale,
            "key_antiquant_offset": scales.k_aq_offset,
            "value_antiquant_scale": scales.v_aq_scale,
            "value_antiquant_offset": scales.v_aq_offset,
            "key_antiquant_mode": 0,
            "value_antiquant_mode": 0,
        }

    def _dequant_paged_kv_to_dense(
        self,
        key: Any,
        value: Any,
        block_table: Any,
        seq_lens: list[int],
        target_dtype: Any,
    ) -> tuple[Any, Any]:
        scales = self._int8_scales_or_raise()
        # 分页 int8 -> 稠密 fp16（prefill 命中缓存时用）
        return int8_ops.dequant_paged_kv_to_dense(
            key,
            value,
            block_table,
            seq_lens,
            target_dtype,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            k_inv_scale=scales.k_inv_scale,
            k_offset=scales.k_offset,
            v_inv_scale=scales.v_inv_scale,
            v_offset=scales.v_offset,
        )

    def _forward_int8_decode(
        self,
        query: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        """INT8 decode：BNSD 布局直接吃分页 int8 缓存 + antiquant。

        decode 每 request 只有 1 个 query token，unsqueeze 成 BNSD 的 [B,1]。
        """
        torch_npu = import_torch_npu("_forward_int8_decode")
        num_block, block_size, _, _ = self.key_cache.shape
        key = self.key_cache.view(num_block, block_size, -1)
        value = self.value_cache.view(num_block, block_size, -1)
        batch_size = len(attn_metadata.seq_lens_list)

        attn_output, _ = torch_npu.npu_fused_infer_attention_score(
            query[:batch_size].unsqueeze(2),
            key,
            value,
            **self._int8_aq_kwargs(),
            block_table=attn_metadata.block_tables,
            actual_seq_lengths_kv=attn_metadata.seq_lens_list,
            num_heads=self.num_heads,
            num_key_value_heads=self.num_kv_heads,
            input_layout="BNSD",
            sparse_mode=0,
            scale=self.scale,
            block_size=block_size,
        )
        attn_output = attn_output.squeeze(2)
        output[:batch_size] = attn_output
        return output

    def _forward_int8_chunked_prefill(
        self,
        query: Any,
        float_key: Any,
        float_value: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        """INT8 chunked prefill：decode 行走 BNSD+int8，prefill 行走 TND+fp16。

        prefill 部分区分两种情况：全部是新请求 -> 直接用 fp16 新 K/V；
        承接已有缓存 -> 先 gather 分页 int8 再反量化成稠密 fp16。
        """
        torch = import_torch("_forward_int8_chunked_prefill")
        torch_npu = import_torch_npu("_forward_int8_chunked_prefill")
        num_decode = attn_metadata.num_decode_tokens
        num_decodes = attn_metadata.num_decodes
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])

        if num_decode > 0:
            num_block, block_size, _, _ = self.key_cache.shape
            kv_k = self.key_cache.view(num_block, block_size, -1)
            kv_v = self.value_cache.view(num_block, block_size, -1)
            attn_out, _ = torch_npu.npu_fused_infer_attention_score(
                query[:num_decode].unsqueeze(2),
                kv_k,
                kv_v,
                **self._int8_aq_kwargs(),
                block_table=attn_metadata.block_tables[:num_decodes],
                actual_seq_lengths_kv=attn_metadata.seq_lens_list[:num_decodes],
                num_heads=self.num_heads,
                num_key_value_heads=self.num_kv_heads,
                input_layout="BNSD",
                sparse_mode=0,
                scale=self.scale,
                block_size=block_size,
            )
            output[:num_decode] = attn_out.squeeze(2)

        if attn_metadata.num_prefills > 0:
            prefill_q = query[num_decode:num_tokens]
            prefill_seq_qlen = [
                actual_seq_qlen[i] - num_decode
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
                prefill_k = float_key[num_decode:num_tokens]
                prefill_v = float_value[num_decode:num_tokens]
                prefill_seq_kvlen = prefill_seq_qlen
            else:
                num_block, blk_size, _, _ = self.key_cache.shape
                paged_k = self.key_cache.view(num_block, blk_size, -1)
                paged_v = self.value_cache.view(num_block, blk_size, -1)
                prefill_bt = attn_metadata.block_tables[num_decodes:]
                prefill_sl = attn_metadata.seq_lens_list[num_decodes:]
                prefill_k, prefill_v = self._dequant_paged_kv_to_dense(
                    paged_k, paged_v, prefill_bt, prefill_sl, query.dtype
                )
                prefill_seq_kvlen = (
                    torch.tensor(prefill_sl, dtype=torch.int32).cumsum(dim=0).tolist()
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
            n_prefill = num_tokens - num_decode
            attn_out = attn_out.view(n_prefill, self.num_heads, self.head_size)
            output[num_decode:num_tokens] = attn_out[:n_prefill]

        return output

    def _forward_int8_prefill(
        self,
        query: Any,
        key: Any,
        value: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        """INT8 prefill：TND + fp16（新 K/V；缓存命中则先手动反量化）。"""
        torch = import_torch("_forward_int8_prefill")
        torch_npu = import_torch_npu("_forward_int8_prefill")
        key, value, block_size, block_table, actual_seq_lengths_kv = (
            self._get_fia_params(key, value, attn_metadata)
        )
        actual_seq_qlen = attn_metadata.actual_seq_lengths_q
        num_tokens = int(actual_seq_qlen[-1])
        query = query[:num_tokens]

        if (
            _attn_state_name(attn_metadata) == "PrefillNoCache"
            and _attn_type_name(self.attn_type) != "ENCODER_DECODER"
        ):
            key = key[:num_tokens]
            value = value[:num_tokens]

        # PrefillCacheHit: K/V read back from the cache are int8 -> dequant.
        if key.dtype == torch.int8:
            if block_table is not None:
                seq_lens = (
                    actual_seq_lengths_kv
                    if isinstance(actual_seq_lengths_kv, list)
                    else actual_seq_lengths_kv.tolist()
                )
                key, value = self._dequant_paged_kv_to_dense(
                    key, value, block_table, seq_lens, query.dtype
                )
                block_table = None
                block_size = self.key_cache.shape[1]
                actual_seq_lengths_kv = (
                    torch.tensor(seq_lens, dtype=torch.int32).cumsum(dim=0).tolist()
                )
            else:
                scales = self._int8_scales_or_raise()
                key = Int8DynamicSemantics.dequantize(
                    key, scales.k_inv_scale, scales.k_offset, query.dtype
                )
                value = Int8DynamicSemantics.dequantize(
                    value, scales.v_inv_scale, scales.v_offset, query.dtype
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
        attn_output = attn_output.view(num_tokens, self.num_heads, self.head_size)
        output[:num_tokens] = attn_output
        return output

    def forward_int8(
        self,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        attn_metadata: Any,
        output: Any,
    ) -> Any:
        """INT8 路径的完整 forward（量化写缓存 + 按状态分派计算）。

        与 legacy forward 的 INT8 分支一一对应：保存 fp16 副本 -> 量化
        K/V -> 写缓存 -> 按注意力状态分派。decode/chunked/prefill 之外
        的状态 fail-closed（不支持的状态静默走错路径比报错更危险）。
        """
        float_key, float_value = None, None
        if key is not None and value is not None:
            # decode 不需要 fp16 副本（直接吃 int8 缓存）；prefill 需要
            if _attn_state_name(attn_metadata) != "DecodeOnly":
                float_key, float_value = key, value
            if not self._int8_ready:
                self._calc_int8_scales(key, value)
            key, value = self._int8_quantize_for_store(key, value)
            query, key, value, output = self.reshape_and_cache(
                query, key, value, kv_cache, attn_metadata, output
            )

        state = _attn_state_name(attn_metadata)
        if state == "DecodeOnly":
            return self._forward_int8_decode(query, attn_metadata, output)
        if state == "ChunkedPrefill":
            return self._forward_int8_chunked_prefill(
                query, float_key, float_value, attn_metadata, output
            )
        if state in ("PrefillNoCache", "PrefillCacheHit"):
            return self._forward_int8_prefill(
                query,
                float_key if float_key is not None else key,
                float_value if float_value is not None else value,
                attn_metadata,
                output,
            )
        raise RuntimeError(
            f"int8_dynamic method does not support attention state {state!r}"
        )


__all__ = ["Int8DynamicAttentionMixin"]
