"""Adapter layer: impl/scheme class construction against stub base classes.

No vllm import happens here — the host base classes are stubs that mirror
the researched host surfaces.
"""

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.attention import (
    build_impl_cls,
    supported_impl_methods,
)
from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.scheme import (
    build_scheme_cls,
)
from vllm_ascend_quantized_kv_cache.adapters.vllm_hust.register import (
    DTYPE_LITERAL_MAP,
    map_cache_dtype,
)


class _StubAscendImpl:
    """Mirrors the AscendAttentionBackendImpl constructor surface."""

    def __init__(
        self,
        num_heads=32,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type="decoder",
        kv_sharing_target_layer_name=None,
        sinks=None,
        **kwargs,
    ):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.attn_type = attn_type
        self.vllm_config = SimpleNamespace(
            cache_config=SimpleNamespace(kivi_group_size=128, kivi_residual_length=128),
            scheduler_config=SimpleNamespace(max_num_seqs=16),
        )


def test_supported_impl_methods() -> None:
    assert set(supported_impl_methods()) == {"int8_dynamic", "kivi_int4"}


def test_build_impl_cls_unknown_method_fails_closed() -> None:
    with pytest.raises(ValueError, match="no Ascend impl mixin"):
        build_impl_cls("warp9_drive", _StubAscendImpl)


def test_build_impl_cls_int8_state_initialisation() -> None:
    impl_cls = build_impl_cls("int8_dynamic", _StubAscendImpl)
    impl = impl_cls(kv_cache_dtype="int8")
    assert impl.enable_int8 is True
    assert impl._int8_ready is False
    # positional form too
    impl2 = impl_cls(32, 128, 0.125, 8, None, None, "int8")
    assert impl2.enable_int8 is True


def test_build_impl_cls_kivi_state_initialisation() -> None:
    impl_cls = build_impl_cls("kivi_int4", _StubAscendImpl)
    impl = impl_cls(kv_cache_dtype="kivi_int4")
    assert impl.enable_kivi is True
    assert impl.kivi_group_size == 128
    assert impl.kivi_residual_length == 128
    assert impl.kivi_max_num_seqs == 16
    assert impl.k_quant_cache is None


class _StubAscendScheme:
    def __init__(self, quant_description=None, prefix=None):
        self.quant_description = quant_description or {}
        self.prefix = prefix or ""

    def create_weights(self, layer):
        self.create_weights_called = True

    def process_weights_after_loading(self, layer):
        pass

    def apply(self, *args):
        raise AssertionError("stub base apply must be overridden")


def test_build_packed_scheme_cls_matches_handler() -> None:
    scheme_cls = build_scheme_cls("int4_packed", _StubAscendScheme)
    scheme = scheme_cls()
    assert scheme.scheme_key == "VLLM_HUST_KV_INT4"
    layer = torch.nn.Module()
    scheme.create_weights(layer)
    assert getattr(scheme, "create_weights_called", False) is True
    assert layer.kv_cache_torch_dtype == torch.uint8
    assert isinstance(layer.k_cache_scale, torch.nn.Parameter)
    with pytest.raises(RuntimeError, match="VLLM_HUST_KV_INT4"):
        scheme.apply(None, None, None, None, None, None, None, None, None)


def test_build_stateful_scheme_cls_requires_impl_attribute(monkeypatch) -> None:
    import sys
    import types

    import vllm_ascend_quantized_kv_cache.core.runtime as runtime

    monkeypatch.setattr(runtime, "module_available", lambda name: True)
    fake_host = types.ModuleType("vllm_ascend.attention.attention_v1")

    class _FakeAscendImpl:
        head_size = 128
        kv_cache_dtype = "kivi_int4"

    fake_host.AscendAttentionBackendImpl = _FakeAscendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", fake_host)

    scheme_cls = build_scheme_cls("kivi_int4", _StubAscendScheme)
    scheme = scheme_cls()
    layer = SimpleNamespace()  # no .impl attribute
    with pytest.raises(RuntimeError, match="impl"):
        scheme.create_weights(layer)


def test_build_stateful_scheme_cls_performs_surgery(monkeypatch) -> None:
    import sys
    import types

    import vllm_ascend_quantized_kv_cache.core.runtime as runtime
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4 import (
        attention_mixin as kivi,
    )

    monkeypatch.setattr(runtime, "module_available", lambda name: True)
    fake_host = types.ModuleType("vllm_ascend.attention.attention_v1")

    class _FakeAscendImpl:
        head_size = 128
        num_kv_heads = 8
        kv_cache_dtype = "kivi_int4"
        vllm_config = None

    fake_host.AscendAttentionBackendImpl = _FakeAscendImpl
    monkeypatch.setitem(sys.modules, "vllm_ascend.attention.attention_v1", fake_host)

    scheme_cls = build_scheme_cls("kivi_int4", _StubAscendScheme)
    scheme = scheme_cls()

    class _Layer:
        pass

    layer = _Layer()
    layer.impl = _FakeAscendImpl()
    scheme.create_weights(layer)

    # the impl was swapped to the mixin-backed class and its state was
    # initialised despite the swap not re-running __init__
    assert isinstance(layer.impl, kivi.KiviInt4AttentionMixin)
    assert layer.impl.enable_kivi is True
    assert layer.impl.kivi_group_size == 128
    assert layer.impl.k_quant_cache is None
    # the base constructor surface is preserved
    assert layer.impl.kv_cache_dtype == "kivi_int4"


def test_dtype_literal_negotiation() -> None:
    assert map_cache_dtype("int8_dynamic") == "int8_per_token_head"
    assert map_cache_dtype("kivi_int4") == "int4_per_token_head"
    assert map_cache_dtype("nvfp4") == "nvfp4"
    assert set(DTYPE_LITERAL_MAP) == {
        "int8_dynamic",
        "kivi_int4",
        "int4_packed",
        "nvfp4",
        "fp8_e4m3",
    }
    with pytest.raises(ValueError, match="no vllm-hust CacheDType"):
        map_cache_dtype("fp4_e2m1")


def _install_fake_ascend_registry(monkeypatch) -> dict:
    """用桩模块顶替 vllm_ascend 的 scheme 注册表（CI 无宿主栈也可跑）。"""
    import sys
    import types

    class _FakeRegistry:
        def __init__(self):
            self.reg: dict = {}

        def register_scheme(self, key, layer_type):
            def deco(cls):
                k = (key, layer_type)
                if k in self.reg:
                    raise ValueError(
                        f"Scheme already registered for {key}/{layer_type}: "
                        f"{self.reg[k].__name__}"
                    )
                self.reg[k] = cls
                return cls

            return deco

        def get_scheme_class(self, key, layer_type):
            return self.reg.get((key, layer_type))

    fake = _FakeRegistry()
    fake_module = types.ModuleType("vllm_ascend.quantization.methods.registry")
    fake_module.register_scheme = fake.register_scheme
    fake_module.get_scheme_class = fake.get_scheme_class
    base_module = types.ModuleType("vllm_ascend.quantization.methods.base")

    class _AscendAttentionScheme:
        def __init__(self, quant_description=None, prefix=None):
            self.quant_description = quant_description or {}
            self.prefix = prefix or ""

    base_module.AscendAttentionScheme = _AscendAttentionScheme
    for name, mod in (
        ("vllm_ascend", types.ModuleType("vllm_ascend")),
        (
            "vllm_ascend.quantization",
            types.ModuleType("vllm_ascend.quantization"),
        ),
        (
            "vllm_ascend.quantization.methods",
            types.ModuleType("vllm_ascend.quantization.methods"),
        ),
        ("vllm_ascend.quantization.methods.registry", fake_module),
        ("vllm_ascend.quantization.methods.base", base_module),
    ):
        monkeypatch.setitem(sys.modules, name, mod)
    return fake.reg


def test_ascend_register_idempotent_and_conflict(monkeypatch) -> None:
    from vllm_ascend_quantized_kv_cache import kv_methods
    from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust import (
        AscendHustAdapter,
    )

    reg = _install_fake_ascend_registry(monkeypatch)
    method = kv_methods.get("int8_dynamic")
    adapter = AscendHustAdapter(method)
    monkeypatch.setattr(adapter, "require_host", lambda: None)

    first = adapter.register()
    assert first["quant_type"] == "VLLM_HUST_KV_INT8_DYNAMIC"
    assert first["already_registered"] is False

    # 第二次激活生成新的等价类（同名）：幂等 no-op
    second = adapter.register()
    assert second["already_registered"] is True

    # 键相同但类名不同（真冲突）：fail-closed
    reg[("VLLM_HUST_KV_FOREIGN", "attention")] = type("SomeoneElsesScheme", (), {})
    with pytest.raises(ValueError, match="Scheme already registered"):
        adapter.register(quant_type="VLLM_HUST_KV_FOREIGN")
