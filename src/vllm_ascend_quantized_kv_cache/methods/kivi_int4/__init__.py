# SPDX-License-Identifier: Apache-2.0
"""KIVI INT4 KV cache 量化方法（int4 历史区 + 全精度残差窗口）。

挖掘自 legacy ascend PR #116 提交 0003-0009（最终状态）。三块内容：
  - semantics.py         量化数学 + 残差窗口簿记（纯 torch，CPU 可测）
  - attention_backend.py NPU forward 状态机（残差管理 / flush / FIA 分派）
  - ops.triton.kivi_pack triton-ascend 打包内核；gather 走纯 torch
    （ops.kivi_gather，原因见该模块头注释）

KIVI 核心思想：每个请求最近 residual_length 个 token 保持全精度（残差
窗口），更早的 token 按 token 组（键）/ head 维组（值）量化成 int4 打进
分页历史缓存，注意力计算时 gather 出来反量化。

本 __init__ 只做"元数据注册"：构造 MethodSpec 并登记进注册表，
绝不 import torch 或任何设备模块——这是导入惰性约束的一部分。
"""

from __future__ import annotations

from ...core.hosts import VLLM_ASCEND_HUST
from ...dtypes import KVQuantMode
from ..base import MethodConfig, MethodSpec
from ..registry import register_method

PROVENANCE = "ascend-pr-116/0003-0009"


def _validate_config(config: MethodConfig) -> None:
    """方案特有布局约束（详见 geometry.validate_kivi_geometry）。

    校验器刻意放在零依赖模块里：方法发现与发布校验环境可能没有 torch。
    """
    from .geometry import validate_kivi_config

    validate_kivi_config(config)


def _load_semantics(config):
    """惰性语义加载器：首次访问 method.semantics 时才 import 本模块。"""
    from .semantics import KiviInt4Semantics

    return KiviInt4Semantics(config)


def _make_spec() -> MethodSpec:
    """构造方案 spec；适配器工厂在函数体内惰性 import 适配器模块。"""

    def ascend_adapter(method):
        from ...adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(method)

    return MethodSpec(
        name="kivi_int4",
        # 由 vLLM CLI 的 --kv-cache-dtype kivi_int4 直接选择，不依赖 checkpoint。
        dtype="kivi_int4",
        summary=(
            "KIVI INT4: the most recent residual window stays full precision "
            "per request; older token groups are packed into a paged int4 "
            "history cache by triton-ascend kernels and gathered + "
            "dequantized for TND fused-inference attention."
        ),
        provenance=PROVENANCE,
        quant_mode=KVQuantMode.KIVI_INT4,
        supports=(VLLM_ASCEND_HUST,),
        requires_npu_kernels=True,
        config_validator=_validate_config,
        semantics_loader=_load_semantics,
        adapter_factories={
            "vllm_ascend_hust": ascend_adapter,
        },
    )


# import 本模块即完成注册（注册 = 写一张"名片"进注册表，无任何副作用）
METHOD_SPEC = _make_spec()
register_method(METHOD_SPEC)

__all__ = ["METHOD_SPEC"]
