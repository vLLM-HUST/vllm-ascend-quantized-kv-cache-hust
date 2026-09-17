# SPDX-License-Identifier: Apache-2.0
"""packed 格式方法的共享语义基类与分发表。

挖掘自 legacy ascend PR #160 提交 0001（``kv_*.py`` handlers +
``kv_cache_utils.py``）。四个格式方法（int4_packed / fp4_e2m1 /
fp8_e4m3 / nvfp4）共用这里的行为基类；每个格式一个单文件方法
（methods/int4_packed.py 等），只声明四个类级元数据。

「packed 格式方法」在 apply() 层都是空操作——真正的量化发生在
attention backend 内核里；语义对象只负责两件事：
  1. 决定缓存存储 dtype（``kv_cache_torch_dtype``）；
  2. 携带 per-layer 的 scale 参数（需要的话）。

``torch`` 在方法内部惰性导入：注册方法（纯元数据）永不触发重导入。
"""

from __future__ import annotations

from typing import Any

from ..core.runtime import import_torch


class PackedFormatSemantics:
    """packed 格式方法的共享语义基类。

    与其他方法的 semantics 对象概念一致：方法的"纯语义"部分。
    子类声明四个类级元数据，行为全部继承自这里。
    """

    #: vllm-ascend-hust 宿主上的注册键，如 "VLLM_HUST_KV_INT4"
    #: （带命名空间，避免与宿主在树 scheme 撞键——宿主注册表重复键会抛错）
    scheme_key: str = ""
    #: 本方法对应的 ``--kv-cache-dtype`` 字符串（层 0 契约键）
    cache_dtype: str = ""
    #: 缓存存储的 torch.dtype 名，如 "uint8"（4bit 打包）或 "float8_e4m3fn"
    storage_torch_dtype_name: str = "uint8"
    #: 是否在 layer 上创建 per-tensor 的 k/v scale 参数
    uses_scales: bool = False

    def __init__(
        self,
        quant_description: dict[str, Any] | None = None,
        prefix: str | None = None,
    ) -> None:
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""

    # -- scheme 协议 ---------------------------------------------------------

    def create_weights(self, layer: Any) -> None:
        """在 attention layer 上挂存储 dtype 与（可选的）scale 参数。"""
        torch = import_torch("PackedFormatSemantics.create_weights")
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
        """权重加载后的整理：scale 压平（不需要则空操作）。"""
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
        """apply 永远抛错：packed 格式的量化在 attention backend 里做，
        这个方法被调用说明接线错了——fail-closed。"""
        err_msg = (
            f"[vllm-hust/{self.scheme_key}] {type(self).__name__}.apply should "
            f"not be called. {self.cache_dtype} KV cache quantization is "
            "handled by the attention backend."
        )
        raise RuntimeError(err_msg)

    def describe(self) -> dict[str, Any]:
        """语义元数据自描述。"""
        return {
            "scheme_key": self.scheme_key,
            "cache_dtype": self.cache_dtype,
            "storage_torch_dtype": self.storage_torch_dtype_name,
            "uses_scales": self.uses_scales,
        }


#: ``--kv-cache-dtype`` 字符串 -> 语义类（与 legacy 的分发表等价）
FORMAT_SEMANTICS: dict[str, type[PackedFormatSemantics]] = {}

#: 方法名 -> 语义类（适配器按方法名查用；与方法注册键一致，如 "int4_packed"）
METHOD_NAME_TO_SEMANTICS: dict[str, type[PackedFormatSemantics]] = {}

#: 非量化 dtype：绝不能分发到任何格式语义
_NON_QUANTIZED_DTYPES = frozenset({"", "auto", "float16", "bfloat16"})


def get_format_semantics(cache_dtype: str) -> PackedFormatSemantics | None:
    """按 ``--kv-cache-dtype`` 返回格式语义实例；未知/非量化返回 None。

    返回 None 的语义与 legacy 一致：由调用方决定是跳过还是 fail-closed。
    """
    semantics_cls = FORMAT_SEMANTICS.get(cache_dtype)
    if semantics_cls is None:
        return None
    return semantics_cls()


def setup_kv_cache_quant(layer: Any, cache_dtype: str) -> PackedFormatSemantics | None:
    """把格式语义的权重挂到 *layer* 上（legacy setup 等价）。

    *cache_dtype* 不是本库认识的量化 dtype 时为空操作并返回 None；
    成功时返回实际生效的语义对象。
    """
    if not cache_dtype or cache_dtype in _NON_QUANTIZED_DTYPES:
        return None

    semantics = get_format_semantics(cache_dtype)
    if semantics is None:
        return None

    # create_weights 内部惰性导入 torch
    semantics.create_weights(layer)
    return semantics


def storage_torch_dtype(cache_dtype: str) -> Any:
    """解析 *cache_dtype* 对应的 torch 存储 dtype（需要 torch 可用）。"""
    torch = import_torch("storage_torch_dtype")
    semantics = get_format_semantics(cache_dtype)
    if semantics is None:
        raise ValueError(f"unknown packed KV cache dtype: {cache_dtype!r}")
    return getattr(torch, semantics.storage_torch_dtype_name)
