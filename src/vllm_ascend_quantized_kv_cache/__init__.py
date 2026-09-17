# SPDX-License-Identifier: Apache-2.0
"""vLLM-HUST 量化 KV cache 方法库 —— 统一入口。

导入本包是"惰性"的：不加载 torch、vllm、任何设备模块，也不启动线程。
所有重模块都在真正使用方法的语义（semantics）或宿主适配器（adapter）
时才被加载。

快速上手::

    from vllm_ascend_quantized_kv_cache import kv_methods

    kv_methods.list()                       # 所有已注册方法
    kv_methods.list(host="vllm_ascend_hust")  # 按宿主过滤
    method = kv_methods.get("kivi_int4", head_size=128, block_size=128)
    method.resolve_layout()                   # -> KVCacheLayout 布局契约
    method.semantics                          # 纯语义数学（CPU 可测）
    kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")  # 插拔进宿主

层次与调用方（详见 docs/layers.md）::

    dtypes  布局契约     —— 任何进程可调（零依赖）
    core    宿主/激活/探测 —— 任何进程可调（零依赖）
    methods 模型+语义     —— 任何进程可调（语义需 torch；设备 mixin 仅 NPU）
    ops     设备内核      —— 仅库内方法路径与 NPU 诊断脚本调用
    adapters 每宿主一个入口 —— 按宿主栈可导入性 fail-closed（缺 vllm /
                             vllm_ascend 时 register 抛错）；双栈共存时
                             两者都可显式注册，默认宿主由 detect_host
                             决定（vllm_ascend 优先）
    bootstrap vLLM 钩子   —— vLLM 经 vllm.general_plugins entry point 调用

设备内核只在 Ascend NPU 上执行；任何设备路径在非 NPU 环境 fail-closed
并给出明确报错。分层设计见 ``docs/architecture.md``，发布流程见
``docs/packaging-and-release.md``。
"""

from ._version import __version__
from .core import VLLM_ASCEND_HUST, VLLM_HUST
from .dtypes import (
    KVCacheLayout,
    KVQuantMode,
    fp4_e2m1_packed_dim,
    get_kv_quant_mode,
    int4_packed_dim,
    is_quantized_kv_cache,
    nvfp4_packed_dim,
    resolve_layout,
)
from .methods.base import KvQuantMethod
from .methods.registry import get_spec as _get_spec
from .methods.registry import list_methods as _list_methods
from .methods.registry import reset_registry as _reset_registry


def _load_methods() -> None:
    """首次使用时注册全部内置方法（只写元数据，零重导入）。"""
    from . import methods  # noqa: F401  (注册副作用)


class _KvMethodsFacade:
    """统一的发现 / 配置 / 绑定 API（本库对外的唯一门面）。"""

    @staticmethod
    def get(name: str, **config) -> KvQuantMethod:
        """按名字取一个已配置的方法句柄。

        未知名（报错信息会列出全部已知方法）或非法配置都会抛
        ``ValueError``——与布局契约相同的 fail-closed 风格。
        """
        _load_methods()
        from .methods.registry import get_method

        return get_method(name, **config)

    @staticmethod
    def list(host: str | None = None) -> list[str]:
        """已注册方法名列表，可按宿主过滤。"""
        _load_methods()
        return list(_list_methods(host))

    @staticmethod
    def describe(name: str) -> dict:
        """单个方法的元数据字典（不触发任何重导入）。"""
        _load_methods()
        return _get_spec(name).describe()

    @staticmethod
    def describe_all(host: str | None = None) -> dict[str, dict]:
        """全部方法（可按宿主过滤）的元数据字典。"""
        _load_methods()
        return {name: _get_spec(name).describe() for name in _list_methods(host)}

    @staticmethod
    def reset() -> None:
        """仅供测试：清空全部注册。"""
        _reset_registry()

    @staticmethod
    def activate(name: str, *, host: str | None = None, **config) -> dict:
        """把方法 *name* 注册进宿主（统一激活管线，唯一入口）。

        *host* 为 None 时自动探测当前进程可导入的宿主栈
        （``vllm_ascend`` 优先，其次 ``vllm``）。显式传宿主名可在
        双宿主栈共存的环境里定点激活：
        ``activate("kivi_int4", host="vllm_ascend_hust"|"vllm_hust")``。
        未知名/非法配置/未知宿主/缺宿主栈/不支持组合全部 fail-closed。
        """
        _load_methods()
        from .core.activation import activate as _activate

        return _activate(name, host=host, **config)


# 统一门面实例：from vllm_ascend_quantized_kv_cache import kv_methods
kv_methods = _KvMethodsFacade()


class VllmAscendQuantizedKvCacheContractProposal:
    """仅元数据的提案占位类；不做任何运行时激活。"""


__all__ = [
    "KVCacheLayout",
    "KVQuantMode",
    "KvQuantMethod",
    "VLLM_ASCEND_HUST",
    "VLLM_HUST",
    "VllmAscendQuantizedKvCacheContractProposal",
    "__version__",
    "fp4_e2m1_packed_dim",
    "get_kv_quant_mode",
    "int4_packed_dim",
    "is_quantized_kv_cache",
    "kv_methods",
    "nvfp4_packed_dim",
    "resolve_layout",
]
