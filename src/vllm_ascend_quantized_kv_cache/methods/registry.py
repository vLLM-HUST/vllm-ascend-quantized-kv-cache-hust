# SPDX-License-Identifier: Apache-2.0
"""方法注册表（fail-closed 风格）。

注册动作只写元数据，绝不导入任何重模块。查询未知名时抛 ValueError 并
列出全部已知方案——与 ``dtypes.resolve_layout`` 的 fail-closed 风格保持
一致：宁可显式失败，不做静默回退。
"""

from __future__ import annotations

from typing import Any

from .base import KvQuantMethod, MethodConfig, MethodSpec

# 模块级注册表：方法名 -> MethodSpec。
# 各方案子包在被 import 时调用 register_method 完成自注册
# （见 methods/__init__.py），门面 kv_methods 首次使用时触发。
_REGISTRY: dict[str, MethodSpec] = {}


def register_method(spec: MethodSpec) -> MethodSpec:
    """注册一个方法；同名重复注册抛 ValueError（防止意外覆盖）。"""
    if not isinstance(spec, MethodSpec):  # pragma: no cover - defensive
        raise TypeError(f"register_method expects a MethodSpec, got {type(spec)!r}")
    if spec.name in _REGISTRY:
        prev = _REGISTRY[spec.name].provenance
        raise ValueError(
            f"method {spec.name!r} is already registered (provenance {prev!r})"
        )
    _REGISTRY[spec.name] = spec
    return spec


def known_methods() -> tuple[str, ...]:
    """全部已注册方法名（排序后），如 ("int8_dynamic", "kivi_int4", ...)。"""
    return tuple(sorted(_REGISTRY))


def list_methods(host: str | None = None) -> tuple[str, ...]:
    """按宿主过滤方法名；host=None 返回全部。"""
    if host is None:
        return known_methods()
    return tuple(name for name in known_methods() if host in _REGISTRY[name].supports)


def get_method(name: str, **config_overrides: Any) -> KvQuantMethod:
    """按名取方法句柄，可携带几何/量化配置覆盖项。

    未知名抛 ValueError（附已知清单）；配置经 MethodConfig + 方案
    config_validator 双重校验后构造 KvQuantMethod。
    """
    spec = _REGISTRY.get(name)
    if spec is None:
        known = ", ".join(known_methods()) or "<none>"
        raise ValueError(f"unknown quantized KV method {name!r}; known: {known}")
    config = MethodConfig(**config_overrides) if config_overrides else MethodConfig()
    return KvQuantMethod(spec, config)


def get_spec(name: str) -> MethodSpec:
    """只取元数据 spec（不构造配置、不触发任何加载）。"""
    spec = _REGISTRY.get(name)
    if spec is None:
        known = ", ".join(known_methods()) or "<none>"
        raise ValueError(f"unknown quantized KV method {name!r}; known: {known}")
    return spec


def reset_registry() -> None:
    """仅供测试：清空全部注册。"""
    _REGISTRY.clear()


__all__ = [
    "get_method",
    "get_spec",
    "known_methods",
    "list_methods",
    "register_method",
    "reset_registry",
]
