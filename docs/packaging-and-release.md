# Packaging and release

版本唯一来源为 `_version.py`，并与 Bundle manifest 的 `bundle_version`
保持一致。manifest 必须包含在 wheel 内。

```bash
conda activate vllm-hust-dev
python -m pytest -q
python -m build
bash scripts/verify-wheel.sh dist/*.whl
```

发布前在隔离环境安装 wheel，检查：

```bash
python -m pip install --no-deps dist/*.whl
python -c 'from importlib.resources import files; print(files("vllm_ascend_quantized_kv_cache").joinpath("manifests", "vllm-hust-extension-v1.json"))'
```

部署时 Python 包必须安装到运行 `vllm` 的同一个环境。Bundle v1 静态准入
需要时，通过 `VLLM_EXTENSION_MANIFESTS` 显式传入 manifest 绝对路径；
不要依赖自定义 bundle entry point 扫描。
