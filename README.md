# Ascend INT8 KV Cache Plugin

面向 `vllm-ascend-hust` 的 INT8 KV-cache attention implementation 插件，
运行在由 `vllm-hust` 启动的推理进程中。仓库只提供一种量化方式：运行时
动态 per-channel INT8；它不是 CUDA、ROCm 或 CPU 可用的通用 vLLM 插件。

插件不读取模型 checkpoint 的 `fa_quant_type`，也不要求模型预量化。
唯一启用开关是 vLLM 命令行参数：

```bash
vllm serve MODEL --kv-cache-dtype int8
```

## 运行机制

安装 wheel 后，vLLM 通过 `vllm.general_plugins` 动态加载
`bootstrap.register_plugins`。入口为宿主 `AscendAttentionBackend` 安装
`get_impl_cls` 分派器：cache dtype 为 `int8` 时返回插件实现，其他 dtype
继续调用宿主原始分派逻辑。未传 `--kv-cache-dtype int8` 时不会执行 INT8
量化路径。

INT8 scale 在每层首次收到 K/V 时沿 token 维计算，K/V 分别使用动态对称
per-channel scale。decode 使用 Ascend fused attention 的在线 antiquant；
prefill 和 chunked prefill 在需要时 gather 并反量化分页缓存。

实际类组合为：

```text
AscendInt8KvAttentionImpl
  = plugin AscendInt8AttentionBackendMixin
  + host AscendAttentionBackendImpl
```

当 context parallel 开启时 INT8 KV cache 会明确拒绝启动，与当前宿主限制一致。

## 安装与运行

```bash
conda activate vllm-hust-dev
python -m pip install -e .

VLLM_LOGGING_LEVEL=INFO vllm serve MODEL \
  --kv-cache-dtype int8 \
  --max-model-len 8192
```

不需要 `VLLM_HUST_KV_METHODS`，不需要修改 checkpoint。

检查动态插件入口：

```bash
python -c "from importlib.metadata import entry_points; print([e for e in entry_points(group='vllm.general_plugins') if e.name == 'vllm-ascend-int8-kv-cache'])"
```

## Bundle manifest

wheel 内包含 Bundle v1 manifest：
`vllm_ascend_quantized_kv_cache/manifests/vllm-hust-extension-v1.json`。
其宿主声明为 `provider=vllm`、`name=vllm-ascend`。
需要静态准入的宿主可通过 `VLLM_EXTENSION_MANIFESTS` 显式传入该文件。
Manifest 的 `implementation_ref` 描述插件提供的 backend 组件；实际运行时
接入由 `vllm.general_plugins` 调用 `install_int8_impl_dispatch()` 完成。

## 验证

```bash
PYTHONPATH=src python -m pytest -q
python -m build
bash scripts/verify-wheel.sh dist/*.whl
```

设备执行仅支持 Ascend NPU。当前目标环境为 Ascend 910B +
`vllm-hust-dev`。

### 已验证环境

2026-09-16 完成了真实 NPU 端到端验证：

| 项目 | 已验证值 |
|---|---|
| 插件版本 | `0.2.0.dev0` |
| vLLM-HUST commit | `8a6655cf62` |
| vLLM-Ascend-HUST commit | `f4f49832` |
| vLLM 运行时版本 | `0.23.1.post1.dev498+g802ead286.dirty` |
| 设备 | Ascend 910B，单卡 |
| 模型 | Qwen2.5-14B-Instruct |
| 模型权重 | BF16 |
| KV cache | 动态 per-channel INT8 |
| 上下文长度 | 8192 |
| Prefill / decode | 通过 |
| ACL Graph capture / replay | 通过 |
| OpenAI Chat API | 3 次请求均返回 HTTP 200 |

该验证不代表已覆盖多卡、context parallel、所有模型或所有宿主版本。
正式发布前应使用目标 wheel 在每个声明支持的宿主版本上重复验证。
