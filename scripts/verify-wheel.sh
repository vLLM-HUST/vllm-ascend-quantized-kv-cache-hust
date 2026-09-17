#!/usr/bin/env bash
set -euo pipefail

wheel=${1:?usage: scripts/verify-wheel.sh dist/package.whl}
python_bin=${PYTHON:-python3}
entries=$($python_bin -m zipfile -l "$wheel")

for required in \
  'vllm_ascend_quantized_kv_cache/_version.py' \
  'vllm_ascend_quantized_kv_cache/bootstrap.py' \
  'vllm_ascend_quantized_kv_cache/adapters/vllm_ascend_hust/backend.py' \
  'vllm_ascend_quantized_kv_cache/manifests/vllm-hust-extension-v1.json' \
  '.dist-info/entry_points.txt' \
  '.dist-info/METADATA'; do
  grep -q "$required" <<<"$entries" || {
    echo "FAIL: wheel is missing $required"
    exit 1
  }
done

for forbidden in 'kivi' 'fp4' 'nvfp4' 'BidKV打包与发布指南.md'; do
  if grep -qi "$forbidden" <<<"$entries"; then
    echo "FAIL: wheel contains removed/non-runtime content: $forbidden"
    exit 1
  fi
done

entry_file=$($python_bin - "$wheel" <<'PY'
import sys, zipfile
with zipfile.ZipFile(sys.argv[1]) as archive:
    name = next(n for n in archive.namelist() if n.endswith('.dist-info/entry_points.txt'))
    print(archive.read(name).decode())
PY
)
grep -q 'vllm.general_plugins' <<<"$entry_file"
grep -q 'bootstrap:register_plugins' <<<"$entry_file"
echo "PASS: INT8 KV plugin wheel verified"
