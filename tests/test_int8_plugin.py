from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import tomllib

import pytest

from vllm_ascend_quantized_kv_cache import (
    KVQuantMode,
    bootstrap,
    kv_methods,
    resolve_layout,
)
from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.register import (
    BACKEND_CLASS_PATH,
)


def test_only_int8_method_is_published() -> None:
    assert kv_methods.list() == ["int8_dynamic"]
    assert kv_methods.list(host="vllm_ascend_hust") == ["int8_dynamic"]
    assert kv_methods.describe("int8_dynamic")["dtype"] == "int8"
    with pytest.raises(ValueError, match="unknown quantized KV method"):
        kv_methods.get("kivi_int4")


def test_cli_int8_layout_contract() -> None:
    layout = resolve_layout("int8", 128)
    assert layout.storage_dtype == "int8"
    assert layout.packed_last_dim == 128
    assert layout.quant_mode is KVQuantMode.INT8_PER_TENSOR
    with pytest.raises(ValueError, match="not a registered"):
        resolve_layout("int4", 128)


def test_bootstrap_is_noop_without_ascend(monkeypatch) -> None:
    monkeypatch.setattr(bootstrap, "detect_host", lambda: None)
    assert bootstrap.register_plugins() == []


def test_bootstrap_registers_int8_backend(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(bootstrap, "detect_host", lambda: bootstrap.VLLM_ASCEND_HUST)
    import vllm_ascend_quantized_kv_cache.core.activation as activation

    monkeypatch.setattr(
        activation,
        "activate",
        lambda name, host: (
            calls.append((name, host)) or {"backend_class_path": BACKEND_CLASS_PATH}
        ),
    )
    assert bootstrap.register_plugins() == ["int8_dynamic"]
    assert calls == [("int8_dynamic", "vllm_ascend_hust")]


def test_adapter_installs_host_impl_dispatch(monkeypatch) -> None:
    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    class HostBackend:
        pass

    installed = []
    monkeypatch.setattr(
        backend,
        "install_int8_impl_dispatch",
        lambda: installed.append(True) or HostBackend,
    )

    method = kv_methods.get("int8_dynamic")
    adapter = method.host_adapter("vllm_ascend_hust")
    monkeypatch.setattr(adapter, "require_host", lambda: None)
    info = adapter.register()
    assert installed == [True]
    assert info["integration"] == "host_get_impl_cls_dispatch"
    assert info["cache_dtype_literal"] == "int8"
    assert "--kv-cache-dtype int8" in info["usage"]


def test_host_dispatch_selects_plugin_only_for_int8(monkeypatch) -> None:
    vllm_config = pytest.importorskip(
        "vllm.config", reason="host dispatch test requires the optional vLLM host"
    )
    attention_v1 = pytest.importorskip(
        "vllm_ascend.attention.attention_v1",
        reason="host dispatch test requires the optional vLLM Ascend host",
    )
    attention_utils = pytest.importorskip(
        "vllm_ascend.attention.utils",
        reason="host dispatch test requires the optional vLLM Ascend host",
    )

    import vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.backend as backend

    class HostImpl:
        pass

    class PluginImpl:
        pass

    class HostBackend:
        @staticmethod
        def get_impl_cls():
            return HostImpl

    config = SimpleNamespace(cache_config=SimpleNamespace(cache_dtype="int8"))
    monkeypatch.setattr(vllm_config, "get_current_vllm_config", lambda: config)
    monkeypatch.setattr(attention_utils, "enable_cp", lambda: False)
    monkeypatch.setattr(attention_v1, "AscendAttentionBackend", HostBackend)
    monkeypatch.setattr(backend, "_build_impl_cls", lambda: PluginImpl)

    assert backend.install_int8_impl_dispatch() is HostBackend
    assert HostBackend.get_impl_cls() is PluginImpl

    config.cache_config.cache_dtype = "auto"
    assert HostBackend.get_impl_cls() is HostImpl

    # Repeated plugin loading must not stack wrappers.
    assert backend.install_int8_impl_dispatch() is HostBackend
    assert HostBackend.get_impl_cls() is HostImpl


def test_bundle_v1_contains_only_int8_backend() -> None:
    manifest_path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/manifests/vllm-hust-extension-v1.json"
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "1.0"
    assert payload["host"] == {"provider": "vllm", "name": "vllm-ascend-hust"}
    assert len(payload["components"]) == 1
    component = payload["components"][0]
    assert component["component_id"] == "int8-kv-attention-backend"
    assert component["implementation_ref"].endswith(":AscendInt8KvAttentionBackend")


def test_extension_manager_bundle_is_static_and_preserves_runtime_hook() -> None:
    project_root = Path(__file__).parents[1]
    pyproject = tomllib.loads((project_root / "pyproject.toml").read_text())
    entry_points = pyproject["project"]["entry-points"]

    assert entry_points["vllm.general_plugins"] == {
        "vllm-ascend-int8-kv-cache": (
            "vllm_ascend_quantized_kv_cache.bootstrap:register_plugins"
        )
    }
    assert entry_points["vllm_hust.extension_bundles"] == {
        "org.vllm-hust.ascend-int8-kv-cache": (
            "vllm_ascend_quantized_kv_cache.extension_manager_manifest"
        )
    }

    manifest_path = (
        project_root
        / "src/vllm_ascend_quantized_kv_cache/extension_manager_manifest/"
        "vllm-hust-extension-v0.2.json"
    )
    payload = json.loads(manifest_path.read_text())
    assert payload["schema_version"] == "0.2-experimental"
    assert payload["extension_id"] == "org.vllm-hust.ascend-int8-kv-cache"
    assert payload["kind"] == "in_process_plugin"
    assert payload["activation"]["entry_points"] == [
        {
            "group": "vllm.general_plugins",
            "name": "vllm-ascend-int8-kv-cache",
        }
    ]
    assert payload["activation"]["additional_config"] == {}


def test_runtime_backend_uses_migrated_host_implementation() -> None:
    backend_path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/adapters/vllm_ascend_hust/backend.py"
    )
    source = backend_path.read_text()
    assert "AscendInt8AttentionBackendMixin" in source
    assert "Int8DynamicAttentionMixin" not in source


def test_migrated_backend_retains_host_feature_paths() -> None:
    implementation_path = (
        Path(__file__).parents[1]
        / "src/vllm_ascend_quantized_kv_cache/methods/int8_dynamic/attention_backend.py"
    )
    source = implementation_path.read_text()
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
