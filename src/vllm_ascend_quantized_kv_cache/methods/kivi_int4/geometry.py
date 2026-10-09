# SPDX-License-Identifier: Apache-2.0
"""KIVI INT4 的布局不变量（纯 Python，零依赖）。

单独成模块是为了守住导入卫生：方法注册表的 config_validator 在
``kv_methods.get("kivi_int4")`` 时就会跑这些检查，而那条路径在没有
torch 的宿主/发布校验环境里也必须可用（torch 只属于 semantics/内核层）。

所有约束都是打包内核与 flush 状态机的前提条件，违反即 fail-closed：

* ``group_size % 8 == 0``：int32 打包 1 word = 8 个 int4 lane；
* ``residual_length % group_size == 0``：键窗口按整组 flush；
* ``head_size % 8 == 0``：同上，head 维要能按 word 切分；
* ``head_size % group_size == 0``：值按 head 维分组；
* ``block_size % group_size == 0``：flush 的块对齐检查。
"""

from __future__ import annotations

from typing import Any


def validate_kivi_geometry(
    *,
    head_size: int,
    group_size: int,
    residual_length: int,
    block_size: int | None = None,
) -> None:
    """复刻 legacy 实现强制执行的全部布局不变量。

    ``block_size`` 可选：impl 对象在绑定缓存后才能知道块大小，且
    ``_write_kivi_key_quant_cache`` 在 flush 时会再查一次
    （block_size % group_size）。
    """
    if group_size <= 0 or group_size % 8:
        # int32 打包要求组大小是 8 的倍数（1 word = 8 个 int4 lane）
        raise ValueError(
            f"kivi_int4 requires group_size divisible by 8, got {group_size}"
        )
    if residual_length % group_size:
        # 残差窗口按整组 flush，必须能被组大小整除
        raise ValueError(
            f"kivi_int4 requires residual_length ({residual_length}) "
            f"to be divisible by group_size ({group_size})"
        )
    if head_size % 8:
        raise ValueError(
            f"kivi_int4 requires head_size divisible by 8, got {head_size}"
        )
    if head_size % group_size:
        # 值按 head 维分组，head_dim 必须能被组大小整除
        raise ValueError(
            f"kivi_int4 requires head_size ({head_size}) to be "
            f"divisible by group_size ({group_size})"
        )
    if block_size is not None and block_size % group_size:
        raise ValueError(
            f"kivi_int4 requires block_size ({block_size}) to be "
            f"divisible by group_size ({group_size})"
        )


def validate_kivi_config(config: Any) -> None:
    """校验入口：接受 MethodConfig，也接受活的 impl 对象
    （后者的属性名带 kivi_ 前缀，如 kivi_group_size）。"""
    group_size = getattr(config, "group_size", None)
    if group_size is None:
        group_size = getattr(config, "kivi_group_size", None)
    residual_length = getattr(config, "residual_length", None)
    if residual_length is None:
        residual_length = getattr(config, "kivi_residual_length", None)
    if group_size is None or residual_length is None:
        raise ValueError(
            "kivi_int4 config must define group_size/residual_length "
            "(or kivi_group_size/kivi_residual_length)"
        )
    validate_kivi_geometry(
        head_size=config.head_size,
        group_size=group_size,
        residual_length=residual_length,
        block_size=getattr(config, "block_size", None),
    )


__all__ = ["validate_kivi_config", "validate_kivi_geometry"]
