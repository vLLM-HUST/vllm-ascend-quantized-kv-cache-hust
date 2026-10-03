# SPDX-License-Identifier: Apache-2.0
"""FP8 per-token-head KV cache 量化方法（E4M3 + 每 (token, head) 动态 scale）。

出处与定位见调研文档 ``kvquant-schemes-beyond-int8-int4.md`` §9.3/§10.1：
宿主 vllm-hust 的 triton 后端（``triton_attn.py`` 的
``supported_kv_cache_dtypes``）原生声明 ``fp8_per_token_head``，是 NPU 上
不依赖厂商新算子的最低成本新 dtype；与本仓库 int8_dynamic 同一"在线
scale、免校准"定位，但粒度到 (token, head) 且带 E4M3 浮点动态范围。

当前状态（2026-10-03）：**契约 + 布局 + CPU 参考语义已立，设备路径未
移植**——``adapter_factories`` 刻意缺位，``host_adapter`` 会 fail-closed
报 "no adapter wired"；``bootstrap.REGISTERED_METHODS`` 也不含它，所以
``--kv-cache-dtype fp8_per_token_head`` 还不可选。后续按调研 §10.1 的
顺序推进：triton-ascend store 内核（对拍本目录 semantics）→ 形状包络
测量 → FIA 读通路 → 接线注册。

本 __init__ 只做"元数据注册"，绝不 import torch 以外的重模块。
"""

from __future__ import annotations

from ...core.hosts import VLLM_ASCEND_HUST
from ...dtypes import KVQuantMode
from ..base import MethodConfig, MethodSpec
from ..registry import register_method

PROVENANCE = "host-triton-per-token-head-contract (调研 §9.3/§10.1, 2026-10-03)"


def _validate_config(config: MethodConfig) -> None:
    """方案特有约束：head_size 是 4 的倍数（fp32 scale 对齐）。"""
    if config.head_size <= 0 or config.head_size % 4:
        raise ValueError(
            "fp8_per_token_head requires head_size positive and divisible "
            f"by 4 (fp32 scale alignment), got {config.head_size}"
        )


def _load_semantics(config):
    """惰性语义加载器：首次访问 method.semantics 时才 import。"""
    from .semantics import Fp8PerTokenHeadSemantics

    return Fp8PerTokenHeadSemantics(config)


def _make_spec() -> MethodSpec:
    return MethodSpec(
        name="fp8_per_token_head",
        # 宿主 triton 后端已声明的字面量；将来由 --kv-cache-dtype 直接选择。
        dtype="fp8_per_token_head",
        summary=(
            "Per-token-head dynamic-scale E4M3 storage: each (token, head) "
            "keeps head_size E4M3 bytes plus one fp32 scale computed online "
            "at write time (no calibration, no pre-quantized checkpoint). "
            "Contract, byte layout and CPU reference semantics are in place; "
            "the triton-ascend store kernel and the FIA read path are not "
            "ported yet, so no adapter is wired and the dtype is not "
            "selectable."
        ),
        provenance=PROVENANCE,
        quant_mode=KVQuantMode.FP8_PER_TOKEN_HEAD,
        supports=(VLLM_ASCEND_HUST,),
        requires_npu_kernels=True,
        config_validator=_validate_config,
        semantics_loader=_load_semantics,
        # adapter_factories 缺位 = fail-closed：host_adapter 抛
        # "no adapter wired for host ... yet"。
    )


# import 本模块即完成注册（注册 = 写一张"名片"进注册表，无任何副作用）
METHOD_SPEC = _make_spec()
register_method(METHOD_SPEC)

__all__ = ["METHOD_SPEC"]
