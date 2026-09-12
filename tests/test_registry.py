"""Registry behaviour: registration, duplicate detection, host filtering."""

import pytest

from vllm_ascend_quantized_kv_cache.core.hosts import ALL_HOSTS
from vllm_ascend_quantized_kv_cache.dtypes import KVQuantMode
from vllm_ascend_quantized_kv_cache.methods import registry
from vllm_ascend_quantized_kv_cache.methods.base import MethodSpec
from vllm_ascend_quantized_kv_cache.methods.registry import (
    get_method,
    known_methods,
    register_method,
)


@pytest.fixture(autouse=True)
def _snapshot_registry():
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def _spec(name: str = "probe", supports: tuple[str, ...] = ALL_HOSTS) -> MethodSpec:
    return MethodSpec(
        name=name,
        dtype="int4",
        summary="probe",
        provenance="test",
        quant_mode=KVQuantMode.INT4,
        supports=supports,
    )


def test_register_and_get_roundtrip() -> None:
    register_method(_spec("probe_x"))
    assert known_methods() == ("probe_x",)
    method = get_method("probe_x", head_size=64)
    assert method.name == "probe_x"
    assert method.config.head_size == 64
    assert method.descriptor["quant_mode"] == int(KVQuantMode.INT4)


def test_duplicate_registration_raises() -> None:
    register_method(_spec("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_method(_spec("dup"))


def test_supports_filtering_and_fail_closed_adapter() -> None:
    register_method(_spec("solo", supports=("vllm_ascend_hust",)))
    method = get_method("solo")
    assert method.supports("vllm_ascend_hust")
    assert not method.supports("vllm_hust")
    with pytest.raises(ValueError, match="does not support host"):
        method.host_adapter("vllm_hust")


def test_method_without_adapter_factory_fails_closed() -> None:
    register_method(_spec("bare"))
    method = get_method("bare")
    with pytest.raises(ValueError, match="no adapter wired"):
        method.host_adapter("vllm_ascend_hust")


def test_with_config_returns_updated_handle() -> None:
    register_method(_spec("cfg"))
    method = get_method("cfg", head_size=128)
    other = method.with_config(head_size=64, num_kv_heads=4)
    assert other is not method
    assert other.config.head_size == 64
    assert method.config.head_size == 128
