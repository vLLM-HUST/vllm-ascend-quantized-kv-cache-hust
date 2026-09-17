# SPDX-License-Identifier: Apache-2.0
"""动态 per-channel INT8 KV cache 方案。

挖掘自 legacy ascend PR #116 提交 0001。量化语义见 semantics.py
（token 维 amax 在线 scale + 对称量化），NPU forward 路径见
attention_mixin.py（decode / chunked-prefill / prefill 三条分支）。

本 __init__ 只做"元数据注册"：构造 MethodSpec 并登记进注册表，
绝不 import torch 或任何设备模块——这是导入惰性约束的一部分。
"""

from __future__ import annotations

from ...core.hosts import ALL_HOSTS
from ...dtypes import KVQuantMode
from ..base import MethodConfig, MethodSpec
from ..registry import register_method

PROVENANCE = "ascend-pr-116/0001"


def _validate_config(config: MethodConfig) -> None:
    """方法特有配置约束。

    动态 per-channel INT8 存的是满精度幅度的 int8 值，没有 4bit 打包
    约束；只要求 head_size 是 8 的倍数，以匹配 NPU fused attention 路径。
    """
    if config.head_size % 8:
        raise ValueError(
            "int8_dynamic expects head_size divisible by 8 for the NPU fused "
            f"attention path, got {config.head_size}"
        )


def _load_semantics(config):
    """惰性语义加载器：首次访问 sol.semantics 时才 import 本模块。"""
    from .semantics import Int8DynamicSemantics

    return Int8DynamicSemantics(config)


def _make_spec() -> MethodSpec:
    """构造方案 spec；适配器工厂在函数体内惰性 import 适配器模块。"""

    def ascend_adapter(method):
        from ...adapters.vllm_ascend_hust import AscendHustAdapter

        return AscendHustAdapter(method)

    def vllm_adapter(method):
        from ...adapters.vllm_hust import VllmHustAdapter

        return VllmHustAdapter(method)

    return MethodSpec(
        name="int8_dynamic",
        # legacy 补丁以 kv_cache_dtype == "int8" 作开关；布局契约经
        # "int8_per_token_head" 解析出同样的 int8 存储（带动态 scale 的
        # 最近似契约模式）。
        dtype="int8_per_token_head",
        summary=(
            "Dynamic per-channel INT8 KV cache: amax over the token dim on "
            "the first prefill, symmetric zero offset, online antiquant on "
            "the NPU fused-inference attention path."
        ),
        provenance=PROVENANCE,
        quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        supports=tuple(ALL_HOSTS),
        requires_npu_kernels=True,
        config_validator=_validate_config,
        semantics_loader=_load_semantics,
        adapter_factories={
            "vllm_ascend_hust": ascend_adapter,
            "vllm_hust": vllm_adapter,
        },
    )


# import 本模块即完成注册（注册 = 写一张"名片"进注册表，无任何副作用）
METHOD_SPEC = _make_spec()
register_method(METHOD_SPEC)

__all__ = ["METHOD_SPEC"]
