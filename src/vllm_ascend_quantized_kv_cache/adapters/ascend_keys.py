# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 宿主的 scheme 注册键 —— 唯一推导处。

键面是 HostContract 的一部分（docs/how-to-run.md §6.2 里 ``fa_quant_type``
的取值）：packed 系沿用格式语义类声明的 ``scheme_key``；有状态系用
``VLLM_HUST_KV_<方法名大写>`` 命名空间模式（与 scheme.py 动态生成 scheme
时用的模式一致）。

放在 adapters/ 根而不是 vllm_ascend_hust/ 子包里是刻意的：子包
``__init__`` 会连带导入 attention mixin（torch），而 checkpoint 注入
工具（tools/checkpoint.py）这类"只看元数据"的路径必须零 torch 可用。
本模块因此只允许依赖 methods.packed_base 的纯元数据表。

注意：``STATEFUL_IMPL_METHODS`` 是"元数据侧"的有状态方法清单；"设备侧"
的 mixin 接线表在 vllm_ascend_hust.attention._MIXINS，两侧一致由
tests/test_checkpoint_inject.py 钉死。
"""

from __future__ import annotations

from ..methods.packed_base import METHOD_NAME_TO_SEMANTICS

#: 有状态系方法（impl 类手术路径）的清单；与 attention._MIXINS 一一对应。
STATEFUL_IMPL_METHODS: tuple[str, ...] = ("int8_dynamic", "kivi_int4")


def ascend_scheme_key(method_name: str) -> str:
    """方法名 -> 宿主 @register_scheme 注册键；未知方法 fail-closed。"""
    semantics = METHOD_NAME_TO_SEMANTICS.get(method_name)
    if semantics is not None:
        return semantics.scheme_key
    if method_name in STATEFUL_IMPL_METHODS:
        return f"VLLM_HUST_KV_{method_name.upper()}"
    raise ValueError(f"unknown method for the Ascend adapter: {method_name!r}")


__all__ = ["STATEFUL_IMPL_METHODS", "ascend_scheme_key"]
