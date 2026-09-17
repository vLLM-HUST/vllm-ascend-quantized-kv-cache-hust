# SPDX-License-Identifier: Apache-2.0
"""把 INT8 attention implementation 接入 vllm-ascend-hust。

接线只替换宿主 backend 的 INT8 impl 分支；是否启用完全由宿主
``--kv-cache-dtype int8`` 决定，
不读取 checkpoint 的 ``fa_quant_type``，也不修改模型量化 scheme。
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter

BACKEND_CLASS_PATH = (
    "vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend."
    "AscendInt8KvAttentionBackend"
)


class AscendHustAdapter(HostAdapter):
    host = "vllm_ascend_hust"
    host_module = "vllm_ascend"

    def register(self) -> dict[str, Any]:
        """安装宿主 impl 分派；运行时由 cache dtype ``int8`` 选择实现。"""
        self.require_host()
        from .backend import install_int8_impl_dispatch

        host_backend = install_int8_impl_dispatch()
        return {
            "host": self.host,
            "method": self.method.name,
            "attention_backend": f"{host_backend.__module__}.{host_backend.__name__}",
            "backend_class_path": BACKEND_CLASS_PATH,
            "integration": "host_get_impl_cls_dispatch",
            "cache_dtype_literal": "int8",
            "usage": "vllm serve MODEL --kv-cache-dtype int8",
        }


__all__ = ["BACKEND_CLASS_PATH", "AscendHustAdapter"]
