# SPDX-License-Identifier: Apache-2.0
"""把量化 KV attention implementation 接入 vllm-ascend-hust。

接线只替换宿主 backend 的量化 KV impl 分支；是否启用完全由宿主
``--kv-cache-dtype`` 的字面量决定（``int8`` -> int8_dynamic，
``kivi_int4`` -> KIVI INT4），不读取 checkpoint 的 ``fa_quant_type``，
也不修改模型量化 scheme。
"""

from __future__ import annotations

from typing import Any

from ..base import HostAdapter

_PACKAGE = "vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend"

#: 方法 dtype 字面量 -> 该 dtype 专属的插件 backend 类引用。
BACKEND_CLASS_PATH_BY_DTYPE = {
    "int8": f"{_PACKAGE}:AscendInt8KvAttentionBackend",
    "kivi_int4": f"{_PACKAGE}:AscendKiviInt4KvAttentionBackend",
}


class AscendHustAdapter(HostAdapter):
    host = "vllm_ascend_hust"
    host_module = "vllm_ascend"

    def register(self) -> dict[str, Any]:
        """安装宿主 impl 分派；运行时由 cache dtype 字面量选择实现。"""
        self.require_host()
        cache_dtype = self.method.spec.dtype
        backend_class_path = BACKEND_CLASS_PATH_BY_DTYPE.get(cache_dtype)
        if backend_class_path is None:
            raise ValueError(
                f"method {self.method.name!r} has dtype {cache_dtype!r}, which "
                f"the vllm-ascend-hust adapter cannot select; known dtypes: "
                f"{', '.join(sorted(BACKEND_CLASS_PATH_BY_DTYPE))}"
            )

        from .backend import install_kv_impl_dispatch

        host_backend = install_kv_impl_dispatch()
        return {
            "host": self.host,
            "method": self.method.name,
            "attention_backend": f"{host_backend.__module__}.{host_backend.__name__}",
            "backend_class_path": backend_class_path,
            "integration": "host_get_impl_cls_dispatch",
            "cache_dtype_literal": cache_dtype,
            "usage": f"vllm serve MODEL --kv-cache-dtype {cache_dtype}",
        }


__all__ = ["AscendHustAdapter", "BACKEND_CLASS_PATH_BY_DTYPE"]
