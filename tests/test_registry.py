"""Registry behaviour: registration, duplicate detection, host filtering."""

import pytest

from vllm_ascend_quantized_kv_cache.core import registry
from vllm_ascend_quantized_kv_cache.core.hosts import ALL_HOSTS
from vllm_ascend_quantized_kv_cache.core.registry import (
    get_solution,
    known_solutions,
    register_solution,
)
from vllm_ascend_quantized_kv_cache.core.spec import SolutionSpec
from vllm_ascend_quantized_kv_cache.dtypes import KVQuantMode


@pytest.fixture(autouse=True)
def _snapshot_registry():
    saved = dict(registry._REGISTRY)
    registry._REGISTRY.clear()
    yield
    registry._REGISTRY.clear()
    registry._REGISTRY.update(saved)


def _spec(name: str = "probe", supports: tuple[str, ...] = ALL_HOSTS) -> SolutionSpec:
    return SolutionSpec(
        name=name,
        dtype="int4",
        summary="probe",
        provenance="test",
        quant_mode=KVQuantMode.INT4,
        supports=supports,
    )


def test_register_and_get_roundtrip() -> None:
    register_solution(_spec("probe_x"))
    assert known_solutions() == ("probe_x",)
    solution = get_solution("probe_x", head_size=64)
    assert solution.name == "probe_x"
    assert solution.config.head_size == 64
    assert solution.descriptor["quant_mode"] == int(KVQuantMode.INT4)


def test_duplicate_registration_raises() -> None:
    register_solution(_spec("dup"))
    with pytest.raises(ValueError, match="already registered"):
        register_solution(_spec("dup"))


def test_supports_filtering_and_fail_closed_adapter() -> None:
    register_solution(_spec("solo", supports=("vllm_ascend_hust",)))
    solution = get_solution("solo")
    assert solution.supports("vllm_ascend_hust")
    assert not solution.supports("vllm_hust")
    with pytest.raises(ValueError, match="does not support host"):
        solution.host_adapter("vllm_hust")


def test_solution_without_adapter_factory_fails_closed() -> None:
    register_solution(_spec("bare"))
    solution = get_solution("bare")
    with pytest.raises(ValueError, match="no adapter wired"):
        solution.host_adapter("vllm_ascend_hust")


def test_with_config_returns_updated_handle() -> None:
    register_solution(_spec("cfg"))
    solution = get_solution("cfg", head_size=128)
    other = solution.with_config(head_size=64, num_kv_heads=4)
    assert other is not solution
    assert other.config.head_size == 64
    assert solution.config.head_size == 128
