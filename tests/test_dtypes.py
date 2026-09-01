import pytest

from vllm_ascend_quantized_kv_cache import (
    KVQuantMode,
    fp4_e2m1_packed_dim,
    get_kv_quant_mode,
    int4_packed_dim,
    nvfp4_packed_dim,
    resolve_layout,
)


@pytest.mark.parametrize(
    ("dtype", "mode"),
    [
        ("int4", KVQuantMode.INT4),
        ("fp4_e2m1", KVQuantMode.FP4_E2M1),
        ("nvfp4", KVQuantMode.NVFP4),
        ("kivi_int4", KVQuantMode.KIVI_INT4),
    ],
)
def test_modes_are_stable(dtype: str, mode: KVQuantMode) -> None:
    assert get_kv_quant_mode(dtype) is mode


def test_packed_dimensions_match_legacy_contract() -> None:
    assert int4_packed_dim(128) == 64
    assert fp4_e2m1_packed_dim(128) == 72
    assert nvfp4_packed_dim(128) == 72


def test_unknown_and_misaligned_layouts_fail_closed() -> None:
    with pytest.raises(ValueError, match="not a registered"):
        resolve_layout("auto", 128)
    with pytest.raises(ValueError, match="divisible by 16"):
        resolve_layout("nvfp4", 130)


def test_layout_declares_storage_and_packed_size() -> None:
    layout = resolve_layout("fp4_e2m1", 128)
    assert layout.storage_dtype == "uint8"
    assert layout.packed_last_dim == 72
