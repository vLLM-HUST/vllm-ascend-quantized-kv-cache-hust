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

部署时 Python 包必须安装到运行 `vllm` 的同一个环境。直接启动时由
`vllm.general_plugins` 注册运行时后端。Extension Manager 部署时，wheel 的
`vllm_hust.extension_bundles` entry point 供 Manager 无导入地发现
`extension_manager_manifest/vllm-hust-extension-v0.2.json`；该发现入口不会
自行启用 INT8 后端。该目录与保留 v1 manifest 的 `manifests/` 分离，因为
Manager 要求每个发现入口只对应一份 manifest。

当前 Extension Manager 的 vLLM Provider 不会自动添加
`--kv-cache-dtype int8`，因此在 Provider 支持该选项前，Manager 启动命令仍须
显式包含该参数。

## 发布到 PyPI

PyPI 项目名为 `vllm-ascend-quantized-kv-cache`。团队发布使用仅授权该
项目的 PyPI API Token，并将完整的、以 `pypi-` 开头的 Token 保存为
GitHub Actions 的 `PYPI_TOKEN` Secret。Token 不写入仓库、项目元数据、
构建脚本或日志。

正式版本必须同时满足以下条件：

- `_version.py` 中的 `__version__` 与 Bundle manifest 的
  `bundle_version` 完全一致；
- Git tag 为同一版本加 `v` 前缀，例如版本 `0.2.0` 对应 tag `v0.2.0`；
- CI、单元测试、wheel 校验和宿主兼容性测试全部通过。

需要手工上传时，从安全位置把完整 Token 写入环境变量，并显式列出本次
发布的 wheel 和 sdist：

```bash
export UV_PUBLISH_TOKEN='<从 CI Secret 或密码库读取的完整 pypi-... Token>'

uv publish \
  --check-url https://pypi.org/simple \
  dist/vllm_ascend_quantized_kv_cache-0.2.0-py3-none-any.whl \
  dist/vllm_ascend_quantized_kv_cache-0.2.0.tar.gz

unset UV_PUBLISH_TOKEN
```

PowerShell 发布结束后使用：

```powershell
Remove-Item Env:UV_PUBLISH_TOKEN
```

仓库的 `.github/workflows/release.yml` 在推送 `v*` tag 后执行同一流程。
工作流从 tag 读取版本，检查包版本、manifest 版本和 tag 一致，再显式上传
对应的两个文件。PyPI 上已经存在且内容不同的同版本文件不能覆盖；这种
情况必须修复问题并发布新的版本号。
