# SPDX-License-Identifier: Apache-2.0
"""Host-agnostic base for the packed KV-cache quantization handlers.

Mined from legacy ascend PR #160 commit 0001
(``vllm_ascend/quantization/methods/kv_*.py``). The four handlers are
*no-ops at the apply() level*: the actual quantization happens inside the
attention backend kernels. A handler only decides the storage dtype and
carries the per-layer scale parameters.

``torch`` is imported lazily inside methods so that registering the
solutions (metadata only) never imports heavy dependencies.
"""

from __future__ import annotations

from typing import Any

from ...core.runtime import import_torch


class PackedKvScheme:
    """Base class mirroring the legacy ``AscendAttentionScheme`` surface.

    Subclasses set the class-level metadata (``scheme_key``, ``cache_dtype``,
    ``storage_torch_dtype_name``, ``uses_scales``) and inherit the behaviour
    below unchanged.
    """

    #: Registry key used on the vllm-ascend-hust host, e.g. "VLLM_HUST_KV_INT4".
    scheme_key: str = ""
    #: ``--kv-cache-dtype`` style string this handler answers to.
    cache_dtype: str = ""
    #: ``torch.dtype`` name of the cache storage, e.g. "uint8".
    storage_torch_dtype_name: str = "uint8"
    #: Whether per-tensor k/v scale parameters are created on the layer.
    uses_scales: bool = False

    def __init__(
        self,
        quant_description: dict[str, Any] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""

    # -- scheme protocol ----------------------------------------------------

    def create_weights(self, layer: Any) -> None:
        torch = import_torch("PackedKvScheme.create_weights")
        layer.kv_cache_torch_dtype = getattr(torch, self.storage_torch_dtype_name)
        if self.uses_scales:
            dtype = torch.get_default_dtype()
            layer.k_cache_scale = torch.nn.Parameter(
                torch.ones(1, dtype=dtype), requires_grad=False
            )
            layer.v_cache_scale = torch.nn.Parameter(
                torch.ones(1, dtype=dtype), requires_grad=False
            )

    def process_weights_after_loading(self, layer: Any) -> None:
        if not self.uses_scales:
            return
        layer.k_cache_scale.data = layer.k_cache_scale.data.flatten()
        layer.v_cache_scale.data = layer.v_cache_scale.data.flatten()

    def apply(
        self,
        layer: Any,
        query: Any,
        key: Any,
        value: Any,
        kv_cache: Any,
        attn_metadata: Any,
        attn_type: Any,
        scale: Any,
        output: Any,
    ) -> Any:
        err_msg = (
            f"[vllm-hust/{self.scheme_key}] {type(self).__name__}.apply should "
            f"not be called. {self.cache_dtype} KV cache quantization is "
            "handled by the attention backend."
        )
        raise RuntimeError(err_msg)

    def describe(self) -> dict[str, Any]:
        return {
            "scheme_key": self.scheme_key,
            "cache_dtype": self.cache_dtype,
            "storage_torch_dtype": self.storage_torch_dtype_name,
            "uses_scales": self.uses_scales,
        }


__all__ = ["PackedKvScheme"]
