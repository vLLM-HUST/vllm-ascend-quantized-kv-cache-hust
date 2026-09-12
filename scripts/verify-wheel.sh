#!/usr/bin/env bash
# Verify the contents of a built wheel, following section 8 of the
# vLLM-HUST packaging and release guide:
#   - the static manifest ships inside the wheel (discovery needs it)
#   - entry_points.txt carries the extension-bundle and general-plugins hooks
#   - the version recorded in metadata matches _version.py / the manifest
set -euo pipefail

WHEEL="${1:?usage: verify-wheel.sh <path-to-wheel>}"
EXPECTED_VERSION="$(python - <<'PY'
import re, pathlib
text = pathlib.Path("src/vllm_ascend_quantized_kv_cache/_version.py").read_text()
print(re.search(r'__version__\s*=\s*"([^"]+)"', text).group(1))
PY
)"

echo "wheel:      ${WHEEL}"
echo "version:    ${EXPECTED_VERSION}"
echo "--- contents (filtered) ---"
python -m zipfile -l "${WHEEL}" | awk '{print $1}' | grep -E \
  'manifests/vllm-hust-extension-v0.2.json|entry_points.txt|METADATA|_version.py|kivi_pack.py|bootstrap.py' ||
  { echo "FAIL: expected files missing from wheel"; exit 1; }

echo "--- entry points ---"
ENTRIES="$(python - <<PY
import zipfile
with zipfile.ZipFile("${WHEEL}") as zf:
    name = next(n for n in zf.namelist() if n.endswith("entry_points.txt"))
    print(zf.read(name).decode())
PY
)"
echo "${ENTRIES}"

echo "${ENTRIES}" | grep -q "vllm_hust.extension_bundles" ||
  { echo "FAIL: vllm_hust.extension_bundles entry point missing"; exit 1; }
echo "${ENTRIES}" | grep -q "org.vllm-hust.quantized-kv-cache" ||
  { echo "FAIL: extension id entry missing"; exit 1; }
echo "${ENTRIES}" | grep -q "vllm.general_plugins" ||
  { echo "FAIL: vllm.general_plugins entry point missing"; exit 1; }
echo "${ENTRIES}" | grep -q "bootstrap:register_plugins" ||
  { echo "FAIL: bootstrap hook missing"; exit 1; }

echo "--- manifest inside wheel ---"
python - <<PY
import json, zipfile
with zipfile.ZipFile("${WHEEL}") as zf:
    name = next(
        n for n in zf.namelist()
        if n.endswith("manifests/vllm-hust-extension-v0.2.json")
    )
    manifest = json.loads(zf.read(name))
assert manifest["extension_id"] == "org.vllm-hust.quantized-kv-cache"
assert manifest["extension_version"] == "${EXPECTED_VERSION}", manifest["extension_version"]
assert manifest["implementation"][0]["status"] == "import_only"
component_ids = [c["component_id"] for c in manifest["components"]]
assert "quantized-kv-layout" in component_ids, component_ids
print("manifest OK:", component_ids)
PY

echo "verify-wheel: OK"
