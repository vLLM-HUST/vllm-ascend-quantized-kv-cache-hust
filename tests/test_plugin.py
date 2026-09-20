"""Shipped plugin surface: method discovery, layout contract, host dispatch.

Every test here runs without vLLM / torch_npu installed; the ones that need
the host stack monkeypatch the host modules they touch.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vllm_ascend_quantized_kv_cache import (
    KVQuantMode,
    bootstrap,
    kv_methods,
    resolve_layout,
)
from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.register import (
    BACKEND_CLASS_PATH_BY_DTYPE,
)

PUBLISHED_METHODS = ["int8_dynamic", "kivi_int4"]


def test_published_methods_and_dtypes() -> None:
    assert kv_methods.list() == PUBLISHED_METHODS
    assert kv_methods.list(host="vllm_ascend_hust") == PUBLISHED_METHODS
    assert kv_methods.describe("int8_dynamic")["dtype"] == "int8"
    assert kv_methods.describe("kivi_int4")["dtype"] == "kivi_int4"
    with pytest.raises(ValueError, match="unknown quantized KV method"):
        kv_methods.get("fp8_e4m3")


def test_cli_int8_layout_contract() -> None:
    layout = resolve_layout("int8", 128)
    assert layout.storage_dtype == "int8"
    assert layout.packed_last_dim == 128
    assert layout.quant_mode is KVQuantMode.INT8_PER_TENSOR


def test_cli_kivi_int4_layout_contract() -> None:
    layout = resolve_layout("kivi_int4", 128)
    assert layout.storage_dtype == "uint8"
    # two int4 lanes per byte: the 4-bit history halves the head dim
    assert layout.packed_last_dim == 64
    assert layout.quant_mode is KVQuantMode.KIVI_INT4
    with pytest.raises(ValueError, match="positive even"):
        resolve_layout("kivi_int4", 127)


def test_unregistered_dtype_layout_still_fails_closed() -> None:
    with pytest.raises(ValueError, match="not a registered"):
        resolve_layout("int4", 128)


def test_kivi_method_config_validators_are_wired() -> None:
    method = kv_methods.get("kivi_int4", head_size=64, group_size=32)
    assert method.descriptor["config"]["residual_length"] == 128
    # head_size must be a whole number of int32 pack words
    with pytest.raises(ValueError, match="divisible by 8"):
        kv_methods.get("kivi_int4", head_size=12)
    # the residual window flushes whole groups only
    with pytest.raises(ValueError, match="residual_length"):
        kv_methods.get("kivi_int4", group_size=64, residual_length=32)


def test_bootstrap_is_noop_without_ascend(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "detect_host", lambda: None)
    assert bootstrap.register_plugins() == []


def test_bootstrap_registers_every_quantized_kv_backend(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(bootstrap, "detect_host", lambda: bootstrap.VLLM_ASCEND_HUST)
    import vllm_ascend_quantized_kv_cache.core.activation as activation

    monkeypatch.setattr(
        activation,
        "activate",
        lambda name, host: (
            calls.append((name, host)) or {"backend_class_path": "unused"}
        ),
    )
    assert bootstrap.register_plugins() == PUBLISHED_METHODS
    assert calls == [(name, "vllm_ascend_hust") for name in PUBLISHED_METHODS]


@pytest.mark.parametrize("method_name", PUBLISHED_METHODS)
def test_adapter_installs_host_impl_dispatch(method_name: str, monkeypatch) -> None:
    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    class HostBackend:
        pass

    installed = []
    monkeypatch.setattr(
        backend,
        "install_kv_impl_dispatch",
        lambda: installed.append(True) or HostBackend,
    )

    method = kv_methods.get(method_name)
    adapter = method.host_adapter("vllm_ascend_hust")
    monkeypatch.setattr(adapter, "require_host", lambda: None)
    info = adapter.register()
    assert installed == [True]
    assert info["integration"] == "host_get_impl_cls_dispatch"
    assert info["method"] == method_name
    cache_dtype = method.spec.dtype
    assert info["cache_dtype_literal"] == cache_dtype
    assert info["backend_class_path"] == BACKEND_CLASS_PATH_BY_DTYPE[cache_dtype]
    assert f"--kv-cache-dtype {cache_dtype}" in info["usage"]


@pytest.fixture
def host_stack(monkeypatch):
    """Install the three host surfaces the dispatcher touches, as stubs.

    Stubbed rather than skipped: the dispatch table is the plugin's only
    runtime switch, so it has to stay under test without a vLLM install.
    """
    import sys
    import types

    def stub(name: str) -> types.ModuleType:
        module = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    vllm = stub("vllm")
    vllm_config = stub("vllm.config")
    vllm.config = vllm_config

    stub("vllm_ascend")
    stub("vllm_ascend.attention")
    attention_v1 = stub("vllm_ascend.attention.attention_v1")
    utils = stub("vllm_ascend.attention.utils")

    class HostBackend:
        @staticmethod
        def get_impl_cls():
            return HostImpl

    class HostImpl:
        pass

    attention_v1.AscendAttentionBackend = HostBackend
    cache_config = SimpleNamespace(cache_dtype="auto")
    vllm_config.get_current_vllm_config = lambda: SimpleNamespace(
        cache_config=cache_config
    )
    utils.enable_cp = lambda: False
    return SimpleNamespace(
        backend=HostBackend,
        host_impl=HostImpl,
        cache_config=cache_config,
    )


def test_host_dispatch_selects_plugin_only_for_quantized_dtypes(
    monkeypatch, host_stack
) -> None:
    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    class Int8Impl:
        pass

    class KiviImpl:
        pass

    monkeypatch.setitem(backend._IMPL_BUILDERS, "int8", lambda: Int8Impl)
    monkeypatch.setitem(backend._IMPL_BUILDERS, "kivi_int4", lambda: KiviImpl)

    assert backend.install_kv_impl_dispatch() is host_stack.backend
    host_stack.cache_config.cache_dtype = "int8"
    assert host_stack.backend.get_impl_cls() is Int8Impl

    host_stack.cache_config.cache_dtype = "kivi_int4"
    assert host_stack.backend.get_impl_cls() is KiviImpl

    host_stack.cache_config.cache_dtype = "auto"
    assert host_stack.backend.get_impl_cls() is host_stack.host_impl

    # Repeated plugin loading must not stack wrappers.
    assert backend.install_kv_impl_dispatch() is host_stack.backend
    assert host_stack.backend.get_impl_cls() is host_stack.host_impl


def test_host_dispatch_rejects_context_parallel_for_quantized_dtypes(
    monkeypatch, host_stack
) -> None:
    import vllm_ascend.attention.utils as attention_utils

    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    host_stack.cache_config.cache_dtype = "kivi_int4"
    monkeypatch.setattr(attention_utils, "enable_cp", lambda: True)
    backend.install_kv_impl_dispatch()
    with pytest.raises(NotImplementedError, match="context parallel"):
        host_stack.backend.get_impl_cls()


def test_kivi_impl_composition_turns_on_kivi_state(monkeypatch, host_stack) -> None:
    """The dispatcher's INT4 impl must self-initialise from the host config."""
    import vllm_ascend.attention.attention_v1 as attention_v1

    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    class HostAttentionImpl:
        num_kv_heads = 8
        head_size = 128

        def __init__(self, *args, **kwargs):
            self.host_args = (args, kwargs)
            self.vllm_config = SimpleNamespace(
                cache_config=SimpleNamespace(
                    kivi_group_size=64, kivi_residual_length=64
                ),
                scheduler_config=SimpleNamespace(max_num_seqs=4),
            )

    monkeypatch.setattr(
        attention_v1, "AscendAttentionBackendImpl", HostAttentionImpl, raising=False
    )
    backend._build_kivi_impl_cls.cache_clear()
    try:
        impl = backend._build_kivi_impl_cls()()
        assert impl.enable_kivi is True
        assert impl.kivi_group_size == 64
        assert impl.kivi_residual_length == 64
        assert impl.kivi_max_num_seqs == 4
        assert impl.host_args == ((), {})
    finally:
        backend._build_kivi_impl_cls.cache_clear()


def test_bundle_v1_publishes_both_backends() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/manifests/vllm-hust-extension-v1.json"
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["host"] == {"provider": "vllm", "name": "vllm-ascend"}
    refs = {c["component_id"]: c["implementation_ref"] for c in payload["components"]}
    assert set(refs) == {"int8-kv-attention-backend", "kivi-int4-kv-attention-backend"}
    assert refs["int8-kv-attention-backend"].endswith(":AscendInt8KvAttentionBackend")
    assert refs["kivi-int4-kv-attention-backend"].endswith(
        ":AscendKiviInt4KvAttentionBackend"
    )


def test_runtime_backends_use_plugin_mixins_over_the_host_impl() -> None:
    backend_path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/adapters/vllm_ascend_hust/backend.py"
    )
    source = backend_path.read_text()
    assert "AscendInt8AttentionBackendMixin" in source
    assert "AscendKiviInt4AttentionBackendMixin" in source
    assert "Int8DynamicAttentionMixin" not in source
    assert "KiviInt4AttentionMixin" not in source


def _implementation_source(method_dir: str, filename: str) -> str:
    path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/methods"
        / method_dir
        / filename
    )
    return path.read_text()


def test_migrated_int8_backend_retains_host_feature_paths() -> None:
    source = _implementation_source("int8_dynamic", "attention_backend.py")
    for feature in (
        "_EXTRA_CTX.capturing",
        "self.full_graph_fia(",
        "layer._int8_scales_ready",
        'model_runner_type == "pooling"',
        "AscendAttentionState.ChunkedPrefill",
        "self._use_layer_aware_fia_graph_replay",
        "self.enable_hamming_sparse",
    ):
        assert feature in source


def test_kivi_backend_retains_legacy_attention_paths() -> None:
    source = _implementation_source("kivi_int4", "attention_backend.py")
    for feature in (
        "def forward(",
        "self._forward_kivi_attention(",
        "PrefillNoCache",
        "DecodeOnly",
        "ChunkedPrefill",
        "npu_fused_infer_attention_score",
        "kivi_pack_key_cache",
        "kivi_pack_value_cache",
        "kivi_dequant_gather_cache",
    ):
        assert feature in source


def test_kivi_lazy_kernel_imports_resolve() -> None:
    """Catch stale module paths inside the mixin's lazy imports.

    A typo in ``from ...ops.triton.<module>`` only surfaced on the NPU flush
    path, which is exactly where it must never happen.
    """
    import ast
    import importlib.util

    source = _implementation_source("kivi_int4", "attention_backend.py")
    package = "vllm_ascend_quantized_kv_cache"
    targets = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ImportFrom) and node.level == 3 and node.module:
            targets.add(f"{package}.{node.module}")
    assert targets, "expected the INT4 path to lazily import device modules"
    for module in sorted(targets):
        assert importlib.util.find_spec(module) is not None, module


def test_method_discovery_and_contracts_need_no_torch() -> None:
    """The release smoke venv has no torch; discovery must still work there."""
    import subprocess
    import sys

    program = """
import sys


class _BlockTorch:
    def find_spec(self, name, path=None, target=None):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("torch is blocked by this test")
        return None


sys.meta_path.insert(0, _BlockTorch())

from vllm_ascend_quantized_kv_cache import kv_methods

for name in ("int8_dynamic", "kivi_int4"):
    method = kv_methods.get(name, head_size=128, group_size=64)
    assert method.resolve_layout().packed_last_dim > 0
assert "torch" not in sys.modules, sorted(sys.modules)
"""
    src = Path(__file__).parents[1] / "src"
    env = {"PYTHONPATH": str(src), "PATH": "/usr/bin:/bin"}
    completed = subprocess.run(
        [sys.executable, "-c", program],
        env=env,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_shipped_dispatch_maps_each_dtype_to_its_own_mixin(monkeypatch, host_stack):
    """The builder table itself must map each dtype to that dtype's mixin.

    Per-dtype selection is checked with stub builders elsewhere; nothing else
    pins the shipped wiring, so swapping the two entries used to pass silently.
    Only the INT4 class is composed here: the INT8 mixin imports ``torch_npu``
    at module scope, so it is not importable off-device (INT4 imports it
    lazily, which is what lets these operator-boundary tests exist at all).
    """
    import vllm_ascend.attention.attention_v1 as attention_v1

    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend
    from vllm_ascend_quantized_kv_cache.methods.kivi_int4.attention_backend import (
        AscendKiviInt4AttentionBackendMixin,
    )

    assert {
        "int8": backend._build_int8_impl_cls,
        "kivi_int4": backend._build_kivi_impl_cls,
    } == backend._IMPL_BUILDERS

    class HostAttentionImpl:
        def forward(self, *args, **kwargs):  # the host's own entry
            raise AssertionError("host forward must not run")

        def __init__(self, *args, **kwargs):
            self.vllm_config = None

    monkeypatch.setattr(
        attention_v1, "AscendAttentionBackendImpl", HostAttentionImpl, raising=False
    )
    backend._build_kivi_impl_cls.cache_clear()
    try:
        kivi_cls = backend._build_kivi_impl_cls()
        assert issubclass(kivi_cls, AscendKiviInt4AttentionBackendMixin)
        # the plugin owns the INT4 forward entry, not the host base
        assert kivi_cls.forward is AscendKiviInt4AttentionBackendMixin.forward
    finally:
        backend._build_kivi_impl_cls.cache_clear()
