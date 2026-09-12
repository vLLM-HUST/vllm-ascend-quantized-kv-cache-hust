"""统一激活管线（core.activation / kv_methods.activate）的契约测试。

不依赖任何宿主栈：宿主探测替身打在 core.hosts._importable，适配器
替身换在真实 spec 的 adapter_factories 上（与 test_bootstrap 同风格）。
"""

import pytest

from vllm_ascend_quantized_kv_cache import bootstrap, kv_methods
from vllm_ascend_quantized_kv_cache.core import hosts as hosts_mod
from vllm_ascend_quantized_kv_cache.core.activation import activate


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(bootstrap.ENV_KV_METHODS, raising=False)
    yield


class _FakeAdapter:
    def __init__(self, method):
        self.method = method

    def register(self):
        return {"quant_type": "VLLM_HUST_KV_FAKE"}


def _install_fake_adapter(monkeypatch, host: str) -> None:
    """把指定宿主的适配器工厂临时换成替身（测试结束自动还原）。"""
    import vllm_ascend_quantized_kv_cache.methods.int8_dynamic as s8

    monkeypatch.setitem(s8.METHOD_SPEC.adapter_factories, host, _FakeAdapter)


def test_activate_unknown_method_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown quantized KV method"):
        activate("warp_drive", host="vllm_hust")


def test_activate_unknown_host_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown host"):
        activate("int8_dynamic", host="vllm_mystery")


def test_activate_invalid_config_fails_closed() -> None:
    # 配置校验发生在任何宿主交互之前
    with pytest.raises(ValueError, match="divisible by 8"):
        activate("kivi_int4", host="vllm_hust", head_size=100)


def test_activate_without_host_stack_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(hosts_mod, "_importable", lambda name: False)
    with pytest.raises(RuntimeError, match="neither vllm_ascend nor vllm"):
        activate("int8_dynamic")


def test_activate_explicit_host_registers(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, "vllm_hust")
    info = activate("int8_dynamic", host="vllm_hust")
    assert info["method"] == "int8_dynamic"
    assert info["host"] == "vllm_hust"
    assert info["quant_type"] == "VLLM_HUST_KV_FAKE"


def test_facade_activate_delegates(monkeypatch) -> None:
    _install_fake_adapter(monkeypatch, "vllm_ascend_hust")
    info = kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")
    assert info["method"] == "int8_dynamic"
    assert info["host"] == "vllm_ascend_hust"


def test_adapter_for_lazy_and_fail_closed() -> None:
    from vllm_ascend_quantized_kv_cache.adapters import adapter_for

    vllm_adapter = adapter_for("vllm_hust")
    assert vllm_adapter.host == "vllm_hust"
    with pytest.raises(ValueError, match="unknown host"):
        adapter_for("vllm_mystery")
