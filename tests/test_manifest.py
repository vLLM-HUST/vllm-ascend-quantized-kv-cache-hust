from pathlib import Path

import pytest

pytest.importorskip(
    "vllm_hust_ext",
    reason="vllm-hust-ext (extension manager) is installed in CI via the test extra",
)

from vllm_hust_ext.manifest import activation_blocker, load_manifest  # noqa: E402

import vllm_ascend_quantized_kv_cache  # noqa: E402
from vllm_ascend_quantized_kv_cache import __version__  # noqa: E402


def _manifest_path() -> Path:
    return (
        Path(vllm_ascend_quantized_kv_cache.__file__).parent
        / "manifests"
        / "vllm-hust-extension-v0.2.json"
    )


def test_descriptor_is_discoverable_but_not_activatable() -> None:
    manifest = load_manifest(_manifest_path())
    assert manifest.bundle_id == "org.vllm-hust.quantized-kv-cache"
    assert activation_blocker(manifest) is not None


def test_descriptor_version_matches_distribution() -> None:
    manifest = load_manifest(_manifest_path())
    assert manifest.extension_version == __version__
