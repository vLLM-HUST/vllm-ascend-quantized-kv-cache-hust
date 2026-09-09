# Packaging and Release

Adapted from the vLLM-HUST packaging and release guide (bidkv reference,
`vllm-hust-docs/operations/bidkv-packaging-and-release-guide.md`) to this
repository. The Extension Manager (`vllm-hust-ext`) owns discovery,
validation, and activation intent; this document covers what we ship and
how.

## Version discipline

- The single version source is
  `src/vllm_ascend_quantized_kv_cache/_version.py`; hatchling reads it
  dynamically (`[tool.hatch.version]`).
- `manifests/vllm-hust-extension-v0.2.json` must carry the same value in
  `extension_version` (`tests/test_facade.py` enforces it).
- PyPI does not allow overwriting uploaded files: **any code change means a
  version bump.**

## What the wheel must contain

- `manifests/vllm-hust-extension-v0.2.json` — without it the manager cannot
  discover the bundle (guide §1: the manifest directory must exist and ship
  as package data).
- `entry_points.txt` with both hooks:
  - `vllm_hust.extension_bundles` → `org.vllm-hust.quantized-kv-cache` =
    `vllm_ascend_quantized_kv_cache.manifests`
  - `vllm.general_plugins` → `bootstrap:register_plugins`
- The triton kernel module (`ops/triton/kivi_cache.py`) ships inside the
  wheel; it is only imported on Ascend NPU paths.

`scripts/verify-wheel.sh dist/*.whl` checks all of the above plus the
manifest version/status invariants. CI runs it after every build.

## Build

Record the commit, clean `dist/`, then:

```bash
uv build --no-sources --out-dir dist
# PowerShell: uv build --no-sources --out-dir dist
```

Pure-Python package → expect a `py3-none-any` wheel and an sdist.

## Pre-flight checks (guide §8)

```bash
python -m zipfile -l dist/*.whl
bash scripts/verify-wheel.sh dist/*.whl

python -m venv .release-smoke
.release-smoke/bin/pip install --no-cache-dir dist/*.whl
.release-smoke/bin/python -c "
from vllm_ascend_quantized_kv_cache import kv_solutions, __version__
print(__version__, kv_solutions.list())
"
```

With the manager installed in the same environment:

```bash
vllm-hust-ext extension list
vllm-hust-ext extension inspect org.vllm-hust.quantized-kv-cache
```

Discovery must succeed with `activation_ready=false` (import_only by
design).

## Publish (tag-triggered)

`.github/release.yml` runs on `v*` tags:

1. `uv build --no-sources --out-dir dist` + wheel verification.
2. `uv publish --check-url ...` using a project-scoped PyPI API token for
   the `intellistream` org account (`__token__` user, `pypi-` prefix),
   stored only as the CI secret `UV_PUBLISH_TOKEN` — never in the repo.
3. No-cache smoke install **from PyPI** pinning the tagged version.

The **first published version requires manual maintainer approval**
(MAINTAINERS.md: release approval is owner-led), and per the guide no
alpha goes out before the end-to-end gate passes.

## Install and use (consumers)

```bash
pip install vllm-ascend-quantized-kv-cache
# activation is explicit:
VLLM_HUST_QUANT_KV_SOLUTIONS=int8_dynamic vllm serve MODEL ...
```

Installation alone must never change serving behaviour — that is what the
default-no-op bootstrap guarantees.

## Upgrade / rollback

- Stop serving processes before upgrading; the new version takes effect on
  restart.
- Roll back with `pip install vllm-ascend-quantized-kv-cache==<old>` —
  PyPI artifacts are immutable, which is why version bumps are mandatory.

## Release checklist

- [ ] `pytest -q` green; `ruff check . && ruff format --check .` clean
- [ ] version bumped in `_version.py` (and therefore manifest + wheel)
- [ ] `bash scripts/verify-wheel.sh` passes on the built wheel
- [ ] isolated smoke install verifies version + discovery
- [ ] `vllm-hust-ext extension inspect` discovers the bundle
- [ ] tag pushed; workflow published; PyPI no-cache smoke install OK
- [ ] maintainer approval recorded for the release
