from pathlib import Path

from vllm_hust_ext.manifest import activation_blocker, load_manifest

import vllm_ascend_quantized_kv_cache


def test_descriptor_is_discoverable_but_not_activatable() -> None:
    manifest = load_manifest(
        Path(vllm_ascend_quantized_kv_cache.__file__).with_name(
            "vllm-hust-extension-v0.2.json"
        )
    )
    assert manifest.bundle_id == "org.vllm-hust.quantized-kv-cache"
    assert activation_blocker(manifest) is not None
