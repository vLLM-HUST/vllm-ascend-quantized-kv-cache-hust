# SPDX-License-Identifier: Apache-2.0
"""KV 分配守卫（adapters/vllm_ascend_hust/alloc_guard.py）的测试。

用 stub 复刻宿主 ``AscendModelSlimConfig.get_kv_quant_split_factor`` 的
legacy 行为（K/V 同维时 V 字节预算 ×2），验证守卫只在对称维度上把切分
调和回对称、MLA 式不等维路径不触碰、幂等、默认关闭、请求了但宿主缺失
时 fail-closed。全程 CPU，不 import 真宿主。
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust import alloc_guard
from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.alloc_guard import (
    ENV_ALLOC_GUARD,
    install_alloc_guard,
)


def _calc_split_factor(num_list: list[int]) -> list[float]:
    """宿主 vllm_ascend/utils.py calc_split_factor 的逐字复刻。"""
    total = sum(num_list)
    return [total / num for num in num_list]


class _StubModelSlimConfig:
    """复刻宿主 get_kv_quant_split_factor 的 legacy 行为（V ×2）。"""

    def get_kv_quant_split_factor(self, layer_name: str, kv_head_dim_list):
        return _legacy_split_factor(self, layer_name, kv_head_dim_list)

    @staticmethod
    def _fa_quant_layer(layer_name: str) -> bool:
        return True


def _legacy_split_factor(self, layer_name: str, kv_head_dim_list):
    if self._fa_quant_layer(layer_name):  # noqa: F841  (与宿主同形)
        k_quant_head_dim = kv_head_dim_list[0]
        v_quant_head_dim = kv_head_dim_list[1] * 2
        kv_head_dim_list = [k_quant_head_dim, v_quant_head_dim]
    return _calc_split_factor(kv_head_dim_list)


def _install_stub_host(monkeypatch: pytest.MonkeyPatch) -> None:
    # 恢复类上的原始方法（守卫包装是类级属性，会跨测试泄漏）
    _StubModelSlimConfig.get_kv_quant_split_factor = _legacy_split_factor
    modelslim = types.ModuleType("vllm_ascend.quantization.modelslim_config")
    modelslim.AscendModelSlimConfig = _StubModelSlimConfig
    monkeypatch.setitem(sys.modules, "vllm_ascend", types.ModuleType("vllm_ascend"))
    quant = types.ModuleType("vllm_ascend.quantization")
    quant.modelslim_config = modelslim
    monkeypatch.setitem(sys.modules, "vllm_ascend.quantization", quant)
    monkeypatch.setitem(
        sys.modules, "vllm_ascend.quantization.modelslim_config", modelslim
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv(ENV_ALLOC_GUARD, raising=False)
    yield


# -- 默认关闭 ------------------------------------------------------------------


def test_guard_is_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # 未请求守卫时不得触发宿主导入（装了假宿主也会被察觉）
    monkeypatch.setitem(sys.modules, "vllm_ascend", None)
    report = install_alloc_guard()
    assert report["status"] == "disabled"


def test_alloc_guard_requested_parses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for raw in ("1", "true", " yes ", "ON"):
        monkeypatch.setenv(ENV_ALLOC_GUARD, raw)
        assert alloc_guard.alloc_guard_requested()
    monkeypatch.setenv(ENV_ALLOC_GUARD, "0")
    assert not alloc_guard.alloc_guard_requested()


# -- 安装与调和语义 ------------------------------------------------------------


def test_guard_rebalances_symmetric_dims(
    monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    _install_stub_host(monkeypatch)
    monkeypatch.setenv(ENV_ALLOC_GUARD, "1")
    assert install_alloc_guard()["status"] == "installed"

    config = _StubModelSlimConfig()
    # 稠密层 [128, 128]：宿主 legacy 给 [3.0, 1.5]（V ×2），守卫必须改回 [2.0, 2.0]
    assert config.get_kv_quant_split_factor("model.layers.0", [128, 128]) == [2.0, 2.0]
    # 对称维度下守卫是恒等的
    assert config.get_kv_quant_split_factor("model.layers.1", [64, 64]) == [2.0, 2.0]
    assert "alloc guard" in capsys.readouterr().out


def test_guard_leaves_asymmetric_mla_dims_untouched(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_stub_host(monkeypatch)
    monkeypatch.setenv(ENV_ALLOC_GUARD, "1")
    install_alloc_guard()

    config = _StubModelSlimConfig()
    # MLA：kv_lora_rank=512 ≠ qk_rope_head_dim=64，V 确实占两倍字节——不触碰。
    # legacy 口径 [512, 64*2] → 总额 640 → [640/512, 640/128]
    assert config.get_kv_quant_split_factor("model.layers.0", [512, 64]) == [
        640 / 512,
        640 / 128,
    ]


def test_guard_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_stub_host(monkeypatch)
    monkeypatch.setenv(ENV_ALLOC_GUARD, "1")
    assert install_alloc_guard()["status"] == "installed"
    assert install_alloc_guard()["status"] == "already_installed"
    # 幂等重装后调和语义不变
    config = _StubModelSlimConfig()
    assert config.get_kv_quant_split_factor("layer", [128, 128]) == [2.0, 2.0]


# -- fail-closed 与表面漂移 ----------------------------------------------------


def test_guard_fails_closed_without_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(ENV_ALLOC_GUARD, "1")
    monkeypatch.setitem(sys.modules, "vllm_ascend", None)
    with pytest.raises(RuntimeError, match=ENV_ALLOC_GUARD):
        install_alloc_guard()


def test_guard_reports_absent_surface(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 宿主已重构/修复（方法不在）：守卫无事可做，如实上报而非报错
    _install_stub_host(monkeypatch)
    monkeypatch.setenv(ENV_ALLOC_GUARD, "1")
    del _StubModelSlimConfig.get_kv_quant_split_factor
    try:
        assert install_alloc_guard()["status"] == "surface_absent"
    finally:
        _StubModelSlimConfig.get_kv_quant_split_factor = (
            lambda self, layer_name, kv_head_dim_list: _calc_split_factor(
                kv_head_dim_list
            )
        )


def test_module_import_stays_inert(monkeypatch: pytest.MonkeyPatch) -> None:
    # 导入守卫模块本身不得拉起宿主栈
    monkeypatch.setitem(sys.modules, "vllm_ascend", None)
    importlib.reload(alloc_guard)
    assert alloc_guard.alloc_guard_requested() is False
