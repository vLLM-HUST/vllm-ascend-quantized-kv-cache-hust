# SPDX-License-Identifier: Apache-2.0
"""vllm-ascend-hust 宿主的注册入口。

零宿主改动的挂载路径：本包调用宿主的 @register_scheme 注册表，用一个
全新的 quant_type 键注册（宿主对重复键抛错，所以键必须带命名空间）。
scheme 之后经 checkpoint 的 fa_quant_type 键到达 attention 层，或经
general-plugins bootstrap 在每个进程里注册。
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter
from .attention import supported_impl_methods
from .scheme import build_scheme_cls, packed_semantics_for


class AscendHustAdapter(HostAdapter):
    host = "vllm_ascend_hust"
    host_module = "vllm_ascend"

    def register(
        self, *, quant_type: str | None = None, layer_type: str = "attention"
    ) -> dict[str, Any]:
        """把方法的 scheme 类注册进宿主注册表。

        幂等：重复注册同一方法时，宿主里已有的若是同名生成类（同一
        方法绑定的等价注册），视为 no-op 并在返回信息里标记
        ``already_registered=True``；键相同但类名不同（真冲突）仍抛错。
        """
        self.require_host()
        try:
            from vllm_ascend.quantization.methods.registry import register_scheme
        except ImportError as exc:
            raise RuntimeError(
                "vllm_ascend.quantization.methods.registry is unavailable; "
                "this adapter targets vllm-ascend-hust"
            ) from exc

        method_name = self.method.name
        try:
            from vllm_ascend.quantization.methods.base import (
                AscendAttentionScheme,
            )
        except ImportError as exc:
            raise RuntimeError(
                "vllm_ascend AscendAttentionScheme base class is unavailable"
            ) from exc

        scheme_cls = build_scheme_cls(method_name, AscendAttentionScheme)
        # 默认键名带 VLLM_HUST_ 命名空间，避免与宿主在树 scheme 撞键
        key = quant_type or self.default_quant_type(method_name)
        already_registered = False
        try:
            register_scheme(key, layer_type)(scheme_cls)
        except ValueError:
            # 重复键：build_scheme_cls 每次都生成新类对象，对象同一性
            # 永远不成立；按"生成类名等价"判断——同名类说明宿主里已是
            # 同一方法绑定的等价注册（例如本进程内第二次激活），视为
            # 幂等 no-op；名字都不同则是真冲突，fail-closed。
            from vllm_ascend.quantization.methods.registry import get_scheme_class

            existing = get_scheme_class(key, layer_type)
            if existing is None or existing.__name__ != scheme_cls.__name__:
                raise
            already_registered = True
        return {
            "host": self.host,
            "method": method_name,
            "quant_type": key,
            "layer_type": layer_type,
            "scheme_cls": scheme_cls,
            "already_registered": already_registered,
        }

    @staticmethod
    def default_quant_type(method_name: str) -> str:
        packed = packed_semantics_for(method_name)
        if packed is not None:
            return packed.scheme_key
        if method_name in supported_impl_methods():
            return f"VLLM_HUST_KV_{method_name.upper()}"
        raise ValueError(f"unknown method for the Ascend adapter: {method_name!r}")


__all__ = ["AscendHustAdapter"]
