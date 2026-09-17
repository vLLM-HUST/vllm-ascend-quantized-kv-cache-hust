# SPDX-License-Identifier: Apache-2.0
"""量化方法模型：方法描述符、配置与统一句柄。

本模块刻意保持零依赖（不 import torch / vllm），保证在任意环境下
import 本包都是"惰性"的——这是整个库导入卫生（import hygiene）的
第一道防线，由 tests/test_facade.py 的子进程测试强制。
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

from ..core.hosts import ALL_HOSTS
from ..dtypes import KVCacheLayout, KVQuantMode, resolve_layout

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .registry import KvQuantMethod


@dataclass(frozen=True)
class MethodConfig:
    """所有方法共享的几何/量化配置（不可变）。

    字段是 INT8 布局计算与设备路径的通用输入。
    """

    head_size: int = 128  # 每个注意力头的维度 D
    num_kv_heads: int = 8  # KV 头数
    num_heads: int = 32  # Q 头数（num_heads = num_queries_per_kv * num_kv_heads）
    block_size: int = 128  # 分页缓存块大小（每块 token 数）

    def validated(self) -> MethodConfig:
        """基础合法性检查（正数性、头数关系）。

        方法特有的布局约束（如 group_size % 8 == 0）由各方案的
        config_validator 在 KvQuantMethod 构造时追加检查。
        """
        if self.head_size <= 0:
            raise ValueError(f"head_size must be positive, got {self.head_size}")
        if self.num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be positive, got {self.num_kv_heads}")
        if self.num_heads < self.num_kv_heads:
            raise ValueError(
                "num_heads must be >= num_kv_heads, got "
                f"{self.num_heads} < {self.num_kv_heads}"
            )
        if self.block_size <= 0:
            raise ValueError(f"block_size must be positive, got {self.block_size}")
        return self


@dataclass(frozen=True)
class MethodSpec:
    """量化方法的静态描述（注册表里的"名片"）。

    关键设计：spec 只携带元数据。构造或注册一个 spec 绝不会触发
    torch / vllm / 设备模块的导入——重模块全部通过惰性字段引用：
      - semantics_loader: 传入 config，返回纯语义对象（首次访问时才 import）
      - adapter_factories: 宿主名 -> 适配器工厂（工厂内部才 import 宿主）
    """

    name: str  # 方法名（注册表主键）
    dtype: str  # 契约层的 dtype 字符串（对应 dtypes.get_kv_quant_mode 的键）
    summary: str  # 一句话说明（量化语义、来源）
    provenance: str  # 出处：provenance/legacy-patches 下的补丁编号
    quant_mode: KVQuantMode  # 契约层的量化模式枚举
    supports: tuple[str, ...]  # 支持的宿主列表（core.hosts 中的名字）
    requires_npu_kernels: bool = True  # 设备执行是否依赖 Ascend NPU 内核
    config_validator: Callable[[MethodConfig], None] | None = None
    semantics_loader: Callable[[], Any] | None = None
    adapter_factories: Mapping[str, Callable[[KvQuantMethod], Any]] = field(
        default_factory=dict
    )

    def describe(self) -> dict[str, Any]:
        """导出为纯数据字典（供工具/manifest/日志使用，不触发任何重导入）。"""
        return {
            "name": self.name,
            "dtype": self.dtype,
            "quant_mode": int(self.quant_mode),
            "summary": self.summary,
            "provenance": self.provenance,
            "requires_npu_kernels": self.requires_npu_kernels,
            "supports": list(self.supports),
        }


class KvQuantMethod:
    """统一方法句柄 —— :func:`methods.registry.get_method` 的返回值。

    一个对象同时暴露三样东西，调用方无需关心方案挖掘自哪个 legacy 补丁：
      1. 契约：resolve_layout() 给出 KVCacheLayout（存储 dtype / packed 维度）
      2. 纯语义：semantics 属性（scale 计算、pack/unpack 等，CPU 可测）
      3. 宿主适配：host_adapter(host) 返回该宿主的集成对象（register() 等）
    """

    def __init__(self, spec: MethodSpec, config: MethodConfig) -> None:
        # 构造即校验：方法特有约束 + 基础约束，任何不合法立即抛 ValueError
        if spec.config_validator is not None:
            spec.config_validator(config)
        self.spec = spec
        self.config = config.validated()

    # -- 自省 ---------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def descriptor(self) -> dict[str, Any]:
        """方法元数据 + 当前配置的完整描述（README/文档中的 descriptor）。"""
        payload = self.spec.describe()
        payload["config"] = {
            f: getattr(self.config, f)
            for f in (
                "head_size",
                "num_kv_heads",
                "num_heads",
                "block_size",
            )
        }
        return payload

    def supports(self, host: str) -> bool:
        """该方法是否声明支持给定宿主。"""
        return host in self.spec.supports

    # -- 契约 ---------------------------------------------------------------

    def resolve_layout(self) -> KVCacheLayout:
        """解析 INT8 存储布局。"""
        return resolve_layout(self.spec.dtype, self.config.head_size)

    # -- 语义 ---------------------------------------------------------------

    @property
    def semantics(self) -> Any:
        """惰性加载纯语义对象（首次访问才 import 对应模块）。"""
        if self.spec.semantics_loader is None:
            raise ValueError(f"method {self.spec.name!r} exposes no semantics module")
        return self.spec.semantics_loader(self.config)

    # -- 宿主适配 -----------------------------------------------------------

    def host_adapter(self, host: str) -> Any:
        """返回宿主适配器（未知宿主/不支持/未接线均 fail-closed）。"""
        if host not in ALL_HOSTS:
            raise ValueError(
                f"unknown host {host!r}; known hosts: {', '.join(ALL_HOSTS)}"
            )
        if not self.supports(host):
            raise ValueError(
                f"method {self.spec.name!r} does not support host {host!r} "
                f"(supports: {', '.join(self.spec.supports) or 'none'})"
            )
        factory = self.spec.adapter_factories.get(host)
        if factory is None:
            raise ValueError(
                f"method {self.spec.name!r} has no adapter wired for host {host!r} yet"
            )
        return factory(self)

    def with_config(self, **overrides: Any) -> KvQuantMethod:
        """返回换了一组配置的新句柄（原句柄不变，config 不可变）。"""
        return KvQuantMethod(self.spec, replace(self.config, **overrides))


__all__ = ["KvQuantMethod", "MethodConfig", "MethodSpec"]
