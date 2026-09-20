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

插件加载时为宿主 backend 安装幂等的 `get_impl_cls` 分派器。cache dtype 为
`int8` 时返回由宿主 `AscendAttentionBackendImpl` 与本包
`AscendInt8AttentionBackendMixin` 组合的实现类；为 `kivi_int4` 时返回
`AscendKiviInt4AttentionBackendMixin` 与同一宿主基类组合的实现类；其他
dtype 始终委托给宿主原始 `get_impl_cls`。

INT8 要求宿主支持 `CacheConfig.cache_dtype == "int8"`，并为它分配未打包的
`torch.int8` KV cache。

INT4（KIVI）额外要求宿主（逐步改动清单与已核实的宿主行号见
`docs/int4-host-integration.md`）：

1. 接受 CLI 字面量 `--kv-cache-dtype kivi_int4`；
2. 为该 dtype 每层分配**两张等大的字节缓冲**：键侧与值侧各
   `num_blocks * block_size * num_kv_heads * S` 字节，其中
   `S = head_size/2 + 8*head_size/group_size`（int4 数据 + 每组一份 fp32
   scale 与 min）。插件自己把这两张缓冲切成内核与 gather 需要的 6 个视图
   （`methods/kivi_int4/byte_cache.py`，视图不复制数据）；这一代 vLLM 没有
   "一层六张"的机制，所以字节区域是唯一的接合点。若宿主已在树里按 6 张
   分配（legacy 形态），绑定路径同样接受。
3. 暴露 `CacheConfig.kivi_group_size` / `kivi_residual_length` 两个旋钮
   （缺失时插件按 128/128 取值）。

插件不猜测缓存布局：字节数对不上区域预算、两张缓冲不等大、或除不出整数
页时都在绑定期抛错（`must hold N bytes` / `equal regions` / `whole number
of pages`）；残差窗口几何不满足 `validate_kivi_geometry` 时在方法构造期抛错。

INT4 的压缩比按 `KiviByteCacheLayout.compression_vs_fp16()` 计算，默认
head_size=128 / group_size=128 下约 **3.6x**（int4 数据本身是 4x，scale/min
开销吃掉一部分；group_size 越小开销越大），不要按"4x"报预算。

非 Ascend 宿主、缺少上述 API 或在量化 dtype 下开启 context parallel 时必须
fail closed。非量化 dtype 不应由插件拒绝，而应保持宿主原有行为。

当前已验证的宿主基线为 vLLM-HUST `8a6655cf62` 和
vLLM-Ascend-HUST `f4f49832`（覆盖 INT8 端到端）。INT4 的打包内核、gather 与
注意力通路已在 910B2 容器 `vllm-hust-cyj-21rc-cloud-container-86` 上逐位复验
通过（记录见 `docs/validation-int4-20260920.md`），但该容器上的宿主
`CacheDType` 既无 `kivi_int4` 也无 `int8`，因此 **INT4 端到端 serving 仍未
验证**，需先落地上面列出的宿主改动。对其他 commit 或发行版的兼容性不应仅根据
`host_api_range` 推断，必须重新运行集成测试。
