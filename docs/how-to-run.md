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
  'vllm-ascend-quantized-kv|kv_cache_dtype=int8|Graph capturing finished|Replaying aclgraph|200 OK|ERROR|Traceback' \
  /tmp/vllm-int8-kv.log
```

成功标准是 EngineCore 加载插件、配置显示 `kv_cache_dtype=int8`、
ACL Graph 完成 capture/replay、API 返回 HTTP 200，且无 FIA/KV cache 异常。

## INT4（KIVI）

同上安装插件后，把 dtype 换成 `kivi_int4`：

```bash
VLLM_LOGGING_LEVEL=DEBUG vllm serve /path/to/model \
  --served-model-name kivi-int4-kv-model \
  --kv-cache-dtype kivi_int4 \
  --max-model-len 8192 \
  --enforce-eager \
  2>&1 | tee /tmp/vllm-kivi-int4-eager.log
```

量化组与残差窗口由宿主 `cache_config` 决定（默认 128/128）。先在 CPU 上
跑逐位对拍，再验证真机打包内核：

```bash
PYTHONPATH=src python -m pytest -q tests/test_kivi_int4.py
python scripts/check_int4_patch_parity.py  # 移植对账：补丁不变量是否仍在
python scripts/npu_probe_kivi_key.py   # 键打包/反量化对拍（需 NPU + triton）
python scripts/npu_smoke_kivi.py       # 端到端写入/注意力冒烟
python scripts/npu_probe_kivi_attention.py   # 打包+gather+fused attention 通路
KIVI_PROBE_HEAD=128 KIVI_PROBE_KV_HEADS=8 KIVI_PROBE_GROUP=128 \
KIVI_PROBE_BLOCK=128 KIVI_PROBE_RESIDUAL=128 \
  python scripts/npu_probe_kivi_attention.py   # 出厂默认几何
python scripts/npu_probe_kivi_dim.py   # 实验性融合 gather 探针（预期仍误编译）
```

四个设备探针在 2026-09-20 的 910B2 复验结果记录在
`validation-int4-20260920.md`。注意因果 prefill 的掩码必须用宿主
`AttentionMaskBuilder` 给的 `int8 [2048, 2048]` split-fuse 掩码，自己拼
`T×T` 加性掩码会被 aclnn 以 `561002` 拒掉。

宿主若未按 INT4 的字节预算分配（两张等大缓冲，单张
`num_blocks*block_size*num_kv_heads*(head_size/2 + 8*head_size/group_size)`
字节），插件在绑定缓存时抛 `KIVI key cache must hold ... bytes` 或
`equal regions` —— 这是 HOST_CONTRACT 的 fail-closed 口径，不是运行时 hack
的理由。宿主只需给两张普通缓冲，6 个视图由插件切。

INT4 路径没有 ACL Graph 捕获分支（legacy 状态即如此，graph 下的动态残差
窗口簿记未接入），先用 `--enforce-eager` 验证；去掉该参数属于本分支尚未
完成的真机工作项。

不设置量化 dtype 时按宿主默认路径运行：

```bash
vllm serve /path/to/model
```

本插件不读取或注入 `quantization_config.fa_quant_type`。模型 checkpoint
无需为 KV INT8 做任何修改。
日志中的 `No quantization signature detected` 只表示模型权重未量化，不表示
INT8 KV cache 未启用。
