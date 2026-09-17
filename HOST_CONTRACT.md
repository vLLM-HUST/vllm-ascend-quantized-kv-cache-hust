# Host contract

本插件定位为 `vllm-ascend-hust` 设备后端扩展，运行在 `vllm-hust` 启动
的推理进程中。Manifest 宿主标识固定为：

```json
{"host": {"provider": "vllm", "name": "vllm-ascend"}}
```

它依赖三个现有宿主表面：

1. `vllm.general_plugins` 在运行时调用 `bootstrap.register_plugins`；
2. Ascend platform selector 返回
   `vllm_ascend.attention.attention_v1.AscendAttentionBackend`；
3. `AscendAttentionBackend.get_impl_cls()` 是可替换的静态 implementation
   分派点。

插件加载时为宿主 backend 安装幂等的 `get_impl_cls` 分派器。当
cache dtype 为 `int8` 时，返回由宿主 `AscendAttentionBackendImpl` 与本包
`AscendInt8AttentionBackendMixin` 组合的实现类；其他 dtype 始终委托给宿主
原始 `get_impl_cls`。宿主必须支持
`CacheConfig.cache_dtype == "int8"`，并为它分配未打包的 `torch.int8`
KV cache。

非 Ascend 宿主、缺少上述 API 或在 INT8 下开启 context parallel 时必须
fail closed。非 `int8` dtype 不应由插件拒绝，而应保持宿主原有行为。

当前已验证的宿主基线为 vLLM-HUST `8a6655cf62` 和
vLLM-Ascend-HUST `f4f49832`。对其他 commit 或发行版的兼容性不应仅根据
`host_api_range` 推断，必须重新运行集成测试。
