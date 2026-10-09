# Ascend Quantized KV Cache Plugin

面向 `vllm-ascend-hust`（由 `vllm-hust` 启动的推理进程）的量化 KV-cache
attention implementation 插件。提供两种量化方案：动态 per-channel **INT8**
（2x）与 **KIVI INT4**（历史区 int4 + 全精度残差窗口，约 3.6x）。不是
CUDA / ROCm / CPU 的通用 vLLM 插件；不读 checkpoint 的 `fa_quant_type`，
也不要求模型预量化——唯一开关是命令行 dtype：

```bash
vllm serve MODEL --kv-cache-dtype int8        # 动态 per-channel INT8
vllm serve MODEL --kv-cache-dtype kivi_int4   # KIVI INT4
```

## 安装与自检

```bash
conda activate vllm-hust-dev
python -m pip install -e .                       # 部署时必须装进运行 vllm 的同一环境
PYTHONPATH=src python -m pytest -q               # CPU 全量测试，不需要 NPU
python -m build && bash scripts/verify-wheel.sh dist/*.whl
```

插件经 `vllm.general_plugins` entry point 由 vLLM 自动加载，无需额外环境
变量。检查注册是否可见：

```bash
python -c "from importlib.metadata import entry_points; print([e for e in entry_points(group='vllm.general_plugins') if e.name == 'vllm-ascend-quantized-kv-cache'])"
```

## 运行机制

安装 wheel 后，`bootstrap.register_plugins()` 给宿主的
`AscendAttentionBackend` 装上 `get_impl_cls` 分派器：只有量化 dtype 走插件
实现，其他 dtype 原样委托宿主；context parallel 下两种量化 cache 一律
fail-closed（与当前宿主能力一致）。

```text
vllm serve MODEL --kv-cache-dtype {int8|kivi_int4}
  └─ host AscendAttentionBackend.get_impl_cls()
       ├─ plugin AscendInt8AttentionBackendMixin + host AscendAttentionBackendImpl
       │    ├─ 首个 K/V 到达时按 token 维算动态对称 per-channel scale
       │    ├─ decode：fused attention 的在线 antiquant
       │    └─ prefill / chunked prefill：需要时 gather 并反量化分页缓存
       └─ plugin AscendKiviInt4AttentionBackendMixin + host AscendAttentionBackendImpl
            ├─ 每请求最近 kivi_residual_length 个 token 保持全精度
            ├─ 更早的 token：键按 token 组、值按 head 维组做非对称 min-max
            ├─ 宿主只给两张 KV 缓冲 → 插件按字节预算切成 6 个视图（不复制数据）
            ├─ int4 打包：triton-ascend 内核（ops/triton/kivi_pack）
            └─ 注意力：gather + 反量化成稠密 TND 走 fused attention，再覆盖残差尾
```

INT4 的历史区是**两张等大字节缓冲**，每 token 每 head 单侧
`S = head_size/2 + 8*head_size/group_size` 字节（int4 数据 + 每组一份 fp32
scale 与 min）。默认几何（head 128 / group 128）相对 fp16 约 **3.6x**，
不是 4x——scale/min 的开销要算进预算，口径以
`KiviByteCacheLayout.compression_vs_fp16()` 为准。组大小与窗口长度读自宿主
`cache_config.kivi_group_size` / `kivi_residual_length`（默认 128/128）。

## 现状（feat/int4）

| 方案 | 代码 | 设备验证 | 端到端 serve | CLI 可选 |
|---|---|---|---|---|
| `int8_dynamic` | ✅ | ✅ 910B2 分派 + 前向 | ✅ 2026-10-03（[实测](docs/serving-verification-20261003.md)） | ✅ `int8` |
| `kivi_int4` | ✅ | ✅ 13 节设备记录（[验证记录](docs/validation-int4-20260920.md)） | ✅ 同上，稠密模型 3.56x 兑现；Qwen3.5-35B-A3B 也跑通 | ✅ `kivi_int4` |
| `fp8_per_token_head` | 写入内核有，读路径没有 | ❌ | ❌ | ❌ |
| 四个纯格式契约 | 只在 dev 分支 | ❌ | ❌ | ❌ |

端到端跑通依赖宿主那四处改动（CLI 字面量、存储 dtype、页大小、缓存形状）；
在**未打补丁**的宿主 revision 上选量化 dtype 会在构造 `CacheConfig` 时被
pydantic 的 `Literal` 拒绝，清单与已核行号见
[docs/int4-host-integration.md](docs/int4-host-integration.md)。模型级精度
评测（真实权重下的输出质量）尚未建立基线。

## 文档地图

按"要做什么"入口：

- **选方案** → [docs/schemes.md](docs/schemes.md)（一览 + 每方案一页 + 下一步）
- **改代码** → [docs/development.md](docs/development.md)（环境、分层纪律、如何加方法、测试策略）
  与 [docs/layers.md](docs/layers.md)（六层调用图、宿主可见性矩阵）
- **接宿主** → [docs/integration.md](docs/integration.md)（两条宿主链路、验收清单）
  与 [docs/int4-host-integration.md](docs/int4-host-integration.md)（INT4 字节预算）
- **上设备** → [docs/how-to-run.md](docs/how-to-run.md)（910B 运行与探针命令）
  与 [docs/npu-implementation.md](docs/npu-implementation.md)（内核清单、状态机、精度方法论）
- **查证据** → [docs/validation-int4-20260920.md](docs/validation-int4-20260920.md)（910B2 逐位记录）、
  [docs/serving-verification-20261003.md](docs/serving-verification-20261003.md)（serve 实测）、
  [docs/benchmark.md](docs/benchmark.md)（唯一保留的压测数据）
- **发布 / 打榜** → [docs/packaging-and-release.md](docs/packaging-and-release.md)、
  [docs/release-and-leaderboard.md](docs/release-and-leaderboard.md)、
  [docs/swe-prefix-benchmark-playbook.md](docs/swe-prefix-benchmark-playbook.md)
- **对外契约与出处** → [HOST_CONTRACT.md](HOST_CONTRACT.md)、
  [PROVENANCE.md](PROVENANCE.md)（legacy 移植范围与算术口径）
- **业界对照** → [docs/kvquant-survey.md](docs/kvquant-survey.md)（唯一一份调研）

## 支持范围

- 目标环境：Ascend 910B / 910B2 + `vllm-hust-dev`；已验证的宿主基线记在
  [HOST_CONTRACT.md](HOST_CONTRACT.md)。其他 commit 或发行版的兼容性不能只靠
  API 存在性推断，必须重跑集成测试。
- 非 NPU 环境、缺少契约要求的 API、或量化 dtype 下开启 context parallel 时
  一律 fail-closed 并报明确错误，不猜布局。
- 调研引用的外部数字（精度、吞吐、压缩比）全部来自论文或厂商口径，我们自己
  没复测过；对外引用前先跑自己的 benchmark。
