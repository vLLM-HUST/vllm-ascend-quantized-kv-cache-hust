"""Bootstrap hook: default no-op, explicit opt-in, fail-closed errors."""

import pytest

from vllm_ascend_quantized_kv_cache import bootstrap


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(bootstrap.ENV_KV_METHODS, raising=False)
    yield


def test_default_is_noop(monkeypatch) -> None:
    # even when a host would be detectable, no opt-in means no action
    monkeypatch.setattr(bootstrap, "detect_host", lambda: "vllm_ascend_hust")
    assert bootstrap.register_plugins() == []
    assert bootstrap.requested_methods() == []


def test_requested_methods_parse_and_dedupe(monkeypatch) -> None:
    monkeypatch.setenv(
        bootstrap.ENV_KV_METHODS, " int8_dynamic, kivi_int4 ,int8_dynamic,"
    )
    assert bootstrap.requested_methods() == ["int8_dynamic", "kivi_int4"]


def test_unknown_method_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_KV_METHODS, "warp_drive")
    with pytest.raises(ValueError, match="unknown quantized KV method"):
        bootstrap.register_plugins()


def test_missing_host_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_KV_METHODS, "int8_dynamic")
    monkeypatch.setattr(bootstrap, "detect_host", lambda: None)
    with pytest.raises(RuntimeError, match="neither vllm_ascend nor vllm"):
        bootstrap.register_plugins()


def test_activation_registers_into_detected_host(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_KV_METHODS, "int8_dynamic")

    registered = {}

    class _FakeAdapter:
        def __init__(self, method):
            self.method = method

        def register(self):
            registered["method"] = self.method.name
            return {"quant_type": "VLLM_HUST_KV_INT8_DYNAMIC"}

    monkeypatch.setattr(bootstrap, "detect_host", lambda: "vllm_ascend_hust")

    import vllm_ascend_quantized_kv_cache.methods.int8_dynamic as s8

    monkeypatch.setitem(
        s8.METHOD_SPEC.adapter_factories, "vllm_ascend_hust", _FakeAdapter
    )
    assert bootstrap.register_plugins() == ["int8_dynamic"]
    assert registered["method"] == "int8_dynamic"


def test_detect_host_prefers_ascend(monkeypatch) -> None:
    # detect_host 的规范出处是 core.hosts（bootstrap 只是再导出），
    # 替身打在规范位置上。
    from vllm_ascend_quantized_kv_cache.core import hosts as hosts_mod

    monkeypatch.setattr(hosts_mod, "_importable", lambda name: name == "vllm")
    assert bootstrap.detect_host() == "vllm_hust"
    monkeypatch.setattr(
        hosts_mod, "_importable", lambda name: name in ("vllm", "vllm_ascend")
    )
    assert bootstrap.detect_host() == "vllm_ascend_hust"
    monkeypatch.setattr(hosts_mod, "_importable", lambda name: False)
    assert bootstrap.detect_host() is None
