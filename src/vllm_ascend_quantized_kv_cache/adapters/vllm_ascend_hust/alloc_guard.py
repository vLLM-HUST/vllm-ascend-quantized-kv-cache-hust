# SPDX-License-Identifier: Apache-2.0
"""KV 分配守卫（opt-in）：对宿主 fa_quant 分配切分做对称性调和。

为什么需要它（2026-09-17 在宿主源码 vllm-ascend-hust@b0613602f 上定位）：

宿主 ``AscendModelSlimConfig.get_kv_quant_split_factor``
（``vllm_ascend/quantization/modelslim_config.py``）对 K/V 同维的稠密层
把 V 的字节预算放大一倍（``v_quant_head_dim = dims[1] * 2``）——这是
legacy C8 "K 存 int8、V 留 fp16" 布局的遗留口径。而当前的存储路径
（``get_kv_quant_dtype`` 返回 (int8, int8)、``model_runner_v1`` 的
``_reshape_kv_cache_tensors`` 按满 head 同形视图）要求 K/V 字节比 1:1。
两处口径叠加的后果：

- ``--kv-cache-dtype`` 为浮点时：K/V 缓冲都偏大，**碰巧能跑**但浪费
  显存（V 区最多浪费 2/3）；
- 为 int8 量化存储（``int8_per_token_head``）时：K 缓冲只有重排所需
  的 2/3，初始化即失败——即 how-to-run.md §8.1 记录的阻塞
  （container-86，2026-09-12；机理分析见
  ``provenance/host-fixes/README.md`` 与
  ``docs/validation-int8-20260912.md``）。

守卫语义（**默认关闭**，``VLLM_HUST_KV_ALLOC_GUARD=1`` 显式开启）：
包装宿主 ``get_kv_quant_split_factor``——当且仅当 K/V 维度相等而宿主
给出的切分因子不对称时，改回对称切分（每份 ``total/d``）。K/V 维度
不等的 MLA 路径（如 kv_lora_rank≠qk_rope_head_dim，V 确实留 fp16）
一律不触碰；宿主未来修复（方法消失或行为已对称）时守卫自动退化成
no-op 并如实上报状态。上游修复提案见 ``provenance/host-fixes/``。

本模块不 import 宿主栈之外的重依赖；只有 ``install_alloc_guard`` 在
被请求时才 import ``vllm_ascend``。
"""

from __future__ import annotations

import functools
import os
from typing import Any

ENV_ALLOC_GUARD = "VLLM_HUST_KV_ALLOC_GUARD"
#: 打在包装函数上的幂等标记。
_GUARD_ATTR = "_vllm_hust_kv_alloc_guard"


def alloc_guard_requested() -> bool:
    """解析 opt-in 环境变量（1/true/yes/on）。"""
    return os.environ.get(ENV_ALLOC_GUARD, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _symmetric_factors(kv_head_dim_list: list[int]) -> list[float]:
    """K/V 同维时的正确切分：每份 ``total/d``（如 [d, d] -> [2.0, 2.0]）。

    与宿主 ``calc_split_factor`` 相同的除法，保证浮点逐位一致。
    """
    total = sum(kv_head_dim_list)
    return [total / dim for dim in kv_head_dim_list]


def install_alloc_guard() -> dict[str, Any]:
    """在当前进程宿主类上安装守卫；返回结构化状态供日志/info 透出。

    状态语义：
    - ``disabled``：环境变量未请求（默认；未 import 宿主）；
    - ``installed`` / ``already_installed``：包装完成 / 幂等重入；
    - ``surface_absent``：宿主类存在但方法已不在（宿主已重构/修复，
      守卫无事可做——不视为错误，但如实上报）；
    - ``host_missing``：请求了守卫但宿主栈不可导入（fail-closed）。
    """
    if not alloc_guard_requested():
        return {"status": "disabled"}

    try:
        from vllm_ascend.quantization.modelslim_config import (
            AscendModelSlimConfig,
        )
    except ImportError as exc:
        raise RuntimeError(
            f"{ENV_ALLOC_GUARD}=1 requests the KV allocation guard, but "
            "vllm_ascend.quantization.modelslim_config is unavailable; "
            "install vllm-ascend-hust first"
        ) from exc

    if not hasattr(AscendModelSlimConfig, "get_kv_quant_split_factor"):
        return {
            "status": "surface_absent",
            "detail": (
                "AscendModelSlimConfig.get_kv_quant_split_factor is gone; "
                "host has drifted (or the split issue is fixed) — nothing "
                "to guard"
            ),
        }

    original = AscendModelSlimConfig.get_kv_quant_split_factor
    if getattr(original, _GUARD_ATTR, False):
        return {"status": "already_installed"}

    @functools.wraps(original)
    def _guarded(self: Any, layer_name: str, kv_head_dim_list: list[int]):
        factors = original(self, layer_name, kv_head_dim_list)
        if len(kv_head_dim_list) == 2 and kv_head_dim_list[0] == kv_head_dim_list[1]:
            expected = _symmetric_factors(kv_head_dim_list)
            if list(factors) != expected:
                print(
                    "[vllm-hust-quantized-kv-cache] alloc guard: layer "
                    f"{layer_name!r} dims {kv_head_dim_list} are symmetric "
                    f"but host split factor {list(factors)} is not; "
                    f"overriding to {expected} (legacy K-int8/V-fp16 "
                    "budget vs current int8/int8 storage; see "
                    "provenance/host-fixes/README.md)",
                    flush=True,
                )
                return expected
        return factors

    setattr(_guarded, _GUARD_ATTR, True)
    AscendModelSlimConfig.get_kv_quant_split_factor = _guarded  # type: ignore[assignment]
    return {"status": "installed"}


__all__ = [
    "ENV_ALLOC_GUARD",
    "alloc_guard_requested",
    "install_alloc_guard",
]
