"""Bootstrap hook: default no-op, explicit opt-in, fail-closed errors."""

import pytest

from vllm_ascend_quantized_kv_cache import bootstrap


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    monkeypatch.delenv(bootstrap.ENV_SOLUTIONS, raising=False)
    yield


def test_default_is_noop(monkeypatch) -> None:
    # even when a host would be detectable, no opt-in means no action
    monkeypatch.setattr(bootstrap, "detect_host", lambda: "vllm_ascend_hust")
    assert bootstrap.register_plugins() == []
    assert bootstrap.requested_solutions() == []


def test_requested_solutions_parse_and_dedupe(monkeypatch) -> None:
    monkeypatch.setenv(
        bootstrap.ENV_SOLUTIONS, " int8_dynamic, kivi_int4 ,int8_dynamic,"
    )
    assert bootstrap.requested_solutions() == ["int8_dynamic", "kivi_int4"]


def test_unknown_solution_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_SOLUTIONS, "warp_drive")
    with pytest.raises(ValueError, match="unknown quantized KV solution"):
        bootstrap.register_plugins()


def test_missing_host_fails_closed(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_SOLUTIONS, "int8_dynamic")
    monkeypatch.setattr(bootstrap, "detect_host", lambda: None)
    with pytest.raises(RuntimeError, match="neither vllm_ascend nor vllm"):
        bootstrap.register_plugins()


def test_activation_registers_into_detected_host(monkeypatch) -> None:
    monkeypatch.setenv(bootstrap.ENV_SOLUTIONS, "int8_dynamic")

    registered = {}

    class _FakeAdapter:
        def __init__(self, solution):
            self.solution = solution

        def register(self):
            registered["solution"] = self.solution.name
            return {"quant_type": "VLLM_HUST_KV_INT8_DYNAMIC"}

    monkeypatch.setattr(bootstrap, "detect_host", lambda: "vllm_ascend_hust")

    import vllm_ascend_quantized_kv_cache.solutions.int8_dynamic as s8

    original_factories = dict(s8.SOLUTION_SPEC.adapter_factories)
    s8.SOLUTION_SPEC.adapter_factories["vllm_ascend_hust"] = _FakeAdapter
    try:
        assert bootstrap.register_plugins() == ["int8_dynamic"]
    finally:
        s8.SOLUTION_SPEC.adapter_factories.clear()
        s8.SOLUTION_SPEC.adapter_factories.update(original_factories)
    assert registered["solution"] == "int8_dynamic"


def test_detect_host_prefers_ascend(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "_importable", lambda name: name == "vllm")
    assert bootstrap.detect_host() == "vllm_hust"
    monkeypatch.setattr(
        bootstrap, "_importable", lambda name: name in ("vllm", "vllm_ascend")
    )
    assert bootstrap.detect_host() == "vllm_ascend_hust"
    monkeypatch.setattr(bootstrap, "_importable", lambda name: False)
    assert bootstrap.detect_host() is None
