"""Packed KV handlers: metadata dispatch, storage dtypes, layer setup."""

import pytest
import torch

from vllm_ascend_quantized_kv_cache.solutions.packed.fp4_e2m1 import (
    FP4E2M1PackedKvScheme,
)
from vllm_ascend_quantized_kv_cache.solutions.packed.fp8_e4m3 import (
    FP8E4M3PackedKvScheme,
)
from vllm_ascend_quantized_kv_cache.solutions.packed.int4 import Int4PackedKvScheme
from vllm_ascend_quantized_kv_cache.solutions.packed.nvfp4 import NVFP4PackedKvScheme
from vllm_ascend_quantized_kv_cache.solutions.packed.schemes import (
    CACHE_DTYPE_TO_SCHEME,
    get_packed_scheme,
    setup_kv_cache_quant,
)

EXPECTED = {
    "int4": (Int4PackedKvScheme, "uint8", True),
    "nvfp4": (NVFP4PackedKvScheme, "uint8", False),
    "fp8_e4m3": (FP8E4M3PackedKvScheme, "float8_e4m3fn", True),
    "fp4_e2m1": (FP4E2M1PackedKvScheme, "uint8", False),
}


@pytest.mark.parametrize("cache_dtype", sorted(EXPECTED))
def test_dispatch_returns_matching_handler(cache_dtype: str) -> None:
    scheme = get_packed_scheme(cache_dtype)
    expected_cls, storage, uses_scales = EXPECTED[cache_dtype]
    assert isinstance(scheme, expected_cls)
    assert scheme.storage_torch_dtype_name == storage
    assert scheme.uses_scales is uses_scales
    assert CACHE_DTYPE_TO_SCHEME[cache_dtype] is expected_cls


@pytest.mark.parametrize("unknown", ["auto", "float16", "bfloat16", "foo", ""])
def test_dispatch_returns_none_for_unknown(unknown: str) -> None:
    assert get_packed_scheme(unknown) is None


@pytest.mark.parametrize("cache_dtype", sorted(EXPECTED))
def test_create_weights_sets_layer_state(cache_dtype: str) -> None:
    scheme = get_packed_scheme(cache_dtype)
    layer = torch.nn.Module()
    scheme.create_weights(layer)
    assert layer.kv_cache_torch_dtype == getattr(torch, scheme.storage_torch_dtype_name)
    if scheme.uses_scales:
        assert isinstance(layer.k_cache_scale, torch.nn.Parameter)
        assert isinstance(layer.v_cache_scale, torch.nn.Parameter)
        # flatten round-trip
        layer.k_cache_scale.data = layer.k_cache_scale.data.view(1, 1)
        scheme.process_weights_after_loading(layer)
        assert layer.k_cache_scale.data.dim() == 1
    else:
        assert not hasattr(layer, "k_cache_scale")
        # no-op must not raise
        scheme.process_weights_after_loading(layer)


@pytest.mark.parametrize("cache_dtype", sorted(EXPECTED))
def test_apply_raises_with_scheme_key(cache_dtype: str) -> None:
    scheme = get_packed_scheme(cache_dtype)
    dummy = torch.zeros(1)
    with pytest.raises(RuntimeError) as excinfo:
        scheme.apply(
            layer=None,
            query=dummy,
            key=dummy,
            value=dummy,
            kv_cache=None,
            attn_metadata=None,
            attn_type=None,
            scale=None,
            output=None,
        )
    assert scheme.scheme_key in str(excinfo.value)


@pytest.mark.parametrize("noop", ["auto", "float16", "bfloat16", ""])
def test_setup_is_noop_for_non_quantized(noop: str) -> None:
    layer = torch.nn.Module()
    assert setup_kv_cache_quant(layer, noop) is None
    assert not hasattr(layer, "kv_cache_torch_dtype")


def test_setup_returns_applied_scheme() -> None:
    layer = torch.nn.Module()
    applied = setup_kv_cache_quant(layer, "int4")
    assert isinstance(applied, Int4PackedKvScheme)
    assert layer.kv_cache_torch_dtype == torch.uint8
