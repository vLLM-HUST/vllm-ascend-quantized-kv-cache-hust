# How to run on Ascend 910B

```bash
conda activate vllm-hust-dev
cd /root/vllm-ascend-quantized-kv-cache-hust
python -m pip install -e .
```

确认 entry point 和 backend 注册：

```bash
python -c 'from vllm_ascend_quantized_kv_cache.bootstrap import register_plugins; print(register_plugins())'
```

先使用 eager 模式验证 INT8 prefill/decode：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve /path/to/model \
  --served-model-name int8-kv-model \
  --kv-cache-dtype int8 \
  --max-model-len 8192 \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-int8-kv-eager.log
```

请求成功后去掉 `--enforce-eager`，验证 ACL Graph capture/replay：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve /path/to/model \
  --served-model-name int8-kv-model \
  --kv-cache-dtype int8 \
  --max-model-len 8192 \
  2>&1 | tee /tmp/vllm-int8-kv.log
```

在另一终端发送请求：

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "int8-kv-model",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0,
    "max_tokens": 64
  }'
```

检查插件、INT8 配置、Graph 和请求结果：

```bash
rg -n \
  'vllm-ascend-int8-kv|kv_cache_dtype=int8|Graph capturing finished|Replaying aclgraph|200 OK|ERROR|Traceback' \
  /tmp/vllm-int8-kv.log
```

成功标准是 EngineCore 加载插件、配置显示 `kv_cache_dtype=int8`、
ACL Graph 完成 capture/replay、API 返回 HTTP 200，且无 FIA/KV cache 异常。

不设置量化 dtype 时按宿主默认路径运行：

```bash
vllm serve /path/to/model
```

本插件不读取或注入 `quantization_config.fa_quant_type`。模型 checkpoint
无需为 KV INT8 做任何修改。
日志中的 `No quantization signature detected` 只表示模型权重未量化，不表示
INT8 KV cache 未启用。
