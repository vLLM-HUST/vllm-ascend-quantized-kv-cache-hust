# Architecture

```text
vLLM runtime
  └─ vllm.general_plugins
       └─ bootstrap.register_plugins()
            └─ install quantized-KV get_impl_cls dispatch on host backend
                 ├─ cache_dtype int8       -> INT8 impl
                 └─ cache_dtype kivi_int4  -> INT4 (KIVI) impl

vllm serve MODEL --kv-cache-dtype {int8|kivi_int4}
  └─ host AscendAttentionBackend.get_impl_cls()
       ├─ plugin AscendInt8AttentionBackendMixin + host AscendAttentionBackendImpl
       │    ├─ dynamic per-channel quantize/store
       │    ├─ decode: fused attention + antiquant
       │    └─ prefill: dense or paged gather/dequant
       └─ plugin AscendKiviInt4AttentionBackendMixin + host AscendAttentionBackendImpl
            ├─ bind 2 host byte buffers -> 6 views (k/v × quant/scale/min)
            ├─ residual window per request -> whole-group key flush, slot-wise value flush
            ├─ triton-ascend int4 pack kernels (ops/triton/kivi_pack)
            └─ attention: torch dequant-gather (ops/kivi_gather) + TND fused attention
```

两条路径共享同一台分派器（`adapters/vllm_ascend_hust/backend.py`）：只有
量化 dtype 走插件实现，其他 dtype 委托宿主原逻辑；context parallel 一律
fail closed。

导入包根只加载纯 Python 契约和方法元数据。vLLM、vllm_ascend、torch、
torch_npu 与 triton 仅在插件注册、语义调用或设备路径实际使用时导入。

本插件的宿主是 `vllm-ascend-hust`；`vllm-hust` 负责启动进程、解析 CLI
和提供 attention registry。设备实现依赖 Ascend NPU，不支持 CUDA、ROCm
或 CPU 后端。

激活分为两步：插件被运行时加载时安装 impl 分派；CLI 的
`--kv-cache-dtype int8` / `--kv-cache-dtype kivi_int4` 才选择量化执行。
checkpoint 配置不参与决策。

INT4 的几何约束（`group_size % 8 == 0`、`residual_length % group_size == 0`、
`head_size % 8 == 0`、`head_size % group_size == 0`）在方法构造期校验；
量化数学与残差窗口簿记集中在 `methods/kivi_int4/semantics.py`，CPU 单测
即是 NPU 内核的数值参考。
