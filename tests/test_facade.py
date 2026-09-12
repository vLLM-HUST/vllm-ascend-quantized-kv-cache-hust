import importlib.util
import json
import sys
from pathlib import Path

import pytest

import vllm_ascend_quantized_kv_cache
from vllm_ascend_quantized_kv_cache import __version__, kv_methods

REPO_ROOT = Path(__file__).resolve().parent.parent


def test_import_is_inert_no_torch_no_vllm() -> None:
    """Importing the package must not import torch / vllm / device modules."""
    import subprocess

    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys\n"
            "sys.path.insert(0, 'src')\n"
            "import vllm_ascend_quantized_kv_cache\n"
            "print('\\n'.join(sorted(m for m in sys.modules if m.split('.')[0] in"
            " ('torch', 'vllm', 'triton', 'vllm_ascend'))))\n",
        ],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == "", f"leaked modules: {probe.stdout}"


def test_version_is_defined() -> None:
    assert __version__
    assert importlib.util.find_spec("vllm_ascend_quantized_kv_cache")


def test_all_builtin_methods_are_discoverable() -> None:
    names = kv_methods.list()
    assert set(names) == {
        "int8_dynamic",
        "kivi_int4",
        "int4_packed",
        "fp4_e2m1",
        "fp8_e4m3",
        "nvfp4",
    }


def test_list_by_host_is_stable() -> None:
    for host in ("vllm_hust", "vllm_ascend_hust"):
        assert set(kv_methods.list(host)) == set(kv_methods.list())


def test_get_unknown_method_fails_closed() -> None:
    with pytest.raises(ValueError, match="unknown quantized KV method"):
        kv_methods.get("does_not_exist")


def test_unknown_host_fails_closed() -> None:
    method = kv_methods.get("int4_packed")
    with pytest.raises(ValueError, match="unknown host"):
        method.host_adapter("gpu_supercluster")


def test_kivi_layout_contract() -> None:
    method = kv_methods.get("kivi_int4", head_size=128, block_size=128)
    layout = method.resolve_layout()
    assert layout.storage_dtype == "uint8"
    assert layout.packed_last_dim == 64
    assert int(layout.quant_mode) == 9


def test_descriptor_carries_provenance_and_config() -> None:
    payload = kv_methods.describe("kivi_int4")
    assert payload["provenance"] == "ascend-pr-116/0003-0013"
    assert payload["requires_npu_kernels"] is True
    assert "vllm_ascend_hust" in payload["supports"]
    method = kv_methods.get("kivi_int4", head_size=64, group_size=64)
    assert method.descriptor["config"]["head_size"] == 64


def test_invalid_config_fails_closed() -> None:
    with pytest.raises(ValueError, match="divisible"):
        kv_methods.get("kivi_int4", head_size=128, group_size=48)


def test_manifest_ships_consistent_version() -> None:
    manifest_path = (
        Path(vllm_ascend_quantized_kv_cache.__file__).parent
        / "manifests"
        / "vllm-hust-extension-v0.2.json"
    )
    manifest = json.loads(manifest_path.read_text())
    assert manifest["extension_version"] == __version__
    assert manifest["schema_version"] == "0.2-experimental"
    statuses = {impl["status"] for impl in manifest["implementation"]}
    assert statuses == {"import_only"}


def test_manifest_module_is_import_light() -> None:
    from vllm_ascend_quantized_kv_cache import manifests

    assert manifests.__doc__  # placeholder package, no heavy imports
