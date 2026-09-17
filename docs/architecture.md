# Architecture

```text
vLLM runtime
  └─ vllm.general_plugins
       └─ bootstrap.register_plugins()
            └─ install INT8-only get_impl_cls dispatch on host backend

vllm serve MODEL --kv-cache-dtype int8
  └─ host AscendAttentionBackend.get_impl_cls()
       └─ plugin AscendInt8AttentionBackendMixin + host AscendAttentionBackendImpl
            ├─ dynamic per-channel quantize/store
            ├─ decode: fused attention + antiquant
            └─ prefill: dense or paged gather/dequant
```

导入包根只加载纯 Python 契约和方法元数据。vLLM、vllm_ascend、torch_npu
仅在插件注册或设备路径实际使用时导入。

本插件的宿主是 `vllm-ascend-hust`；`vllm-hust` 负责启动进程、解析 CLI
和提供 attention registry。设备实现依赖 Ascend NPU，不支持 CUDA、ROCm
或 CPU 后端。

激活分为两步：插件被运行时加载时安装 impl 分派；CLI 的
`--kv-cache-dtype int8` 才选择量化执行。checkpoint 配置不参与决策。
