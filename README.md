# Ascend Quantized KV Cache Plugin

面向 `vllm-ascend-hust` 的量化 KV-cache attention implementation 插件，
运行在由 `vllm-hust` 启动的推理进程中。仓库提供两种量化方式：动态
per-channel INT8（2x）与 KIVI INT4（历史区 4x + 全精度残差窗口）；它不是
CUDA、ROCm 或 CPU 可用的通用 vLLM 插件。

插件不读取模型 checkpoint 的 `fa_quant_type`，也不要求模型预量化。
唯一启用开关是 vLLM 命令行参数：

```bash
vllm serve MODEL --kv-cache-dtype int8        # 动态 per-channel INT8
vllm serve MODEL --kv-cache-dtype kivi_int4   # KIVI INT4
```

## 运行机制

安装 wheel 后，vLLM 通过 `vllm.general_plugins` 动态加载
`bootstrap.register_plugins`。入口为宿主 `AscendAttentionBackend` 安装
`get_impl_cls` 分派器：cache dtype 为 `int8` / `kivi_int4` 时返回对应插件
实现，其他 dtype 继续调用宿主原始分派逻辑。未传量化 dtype 时不会执行任何
量化路径。

- **INT8**：scale 在每层首次收到 K/V 时沿 token 维计算，K/V 分别使用动态
  对称 per-channel scale。decode 使用 Ascend fused attention 的在线
  antiquant；prefill 和 chunked prefill 在需要时 gather 并反量化分页缓存。
- **INT4（KIVI）**：每请求最近 `kivi_residual_length` 个 token 保持全精度
  （残差窗口），更早的 token 按键的 token 组 / 值的 head 维组做非对称
  min-max 量化，经 triton-ascend 内核打包成分页 int4 历史区。vLLM 每层只
  会给两张 KV 缓冲，因此历史区是**两张等大字节缓冲**，插件按
  `S = head_size/2 + 8*head_size/group_size` 字节/token/head 的预算把它切成
  `(k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)` 六个视图（视图不复制
  数据）。注意力计算时把历史区 gather + 反量化成稠密张量走 TND fused
  attention，再覆盖全精度残差尾。组大小与窗口长度读自宿主
  `cache_config.kivi_group_size` / `kivi_residual_length`（默认 128/128）。
  默认几何（head 128、group 128）下相对 fp16 约 **3.6x**，不是 4x：scale
  与 min 的开销要算进预算（`KiviByteCacheLayout.compression_vs_fp16()`）。

实际类组合为：

```text
AscendInt8KvAttentionImpl
  = plugin AscendInt8AttentionBackendMixin
  + host AscendAttentionBackendImpl

AscendKiviInt4KvAttentionImpl
  = plugin AscendKiviInt4AttentionBackendMixin
  + host AscendAttentionBackendImpl
```

当 context parallel 开启时，两种量化 KV cache 都会明确拒绝启动，与当前
宿主限制一致。

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
python -c "from importlib.metadata import entry_points; print([e for e in entry_points(group='vllm.general_plugins') if e.name == 'vllm-ascend-quantized-kv-cache'])"
```

## Bundle manifest

wheel 内包含 Bundle v1 manifest：
`vllm_ascend_quantized_kv_cache/manifests/vllm-hust-extension-v1.json`。
其宿主声明为 `provider=vllm`、`name=vllm-ascend`。
需要静态准入的宿主可通过 `VLLM_EXTENSION_MANIFESTS` 显式传入该文件。
Manifest 的 `implementation_ref` 描述插件提供的 backend 组件；实际运行时
接入由 `vllm.general_plugins` 调用 `install_kv_impl_dispatch()` 完成。

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

### INT4 的验证状态

KIVI INT4 在 CPU 上覆盖了整条数据通路：语义数学、残差窗口状态机，以及用
CPU 参考打包器替换 triton 内核后跑通的 `forward()` 全链路（整组 flush →
分页 int4 历史区 → gather 反量化 → 残差尾覆盖 → 稠密注意力），输出与
独立写出的量化参考逐位对齐、并确认与全精度结果存在量化误差；语义侧的
分块反量化与在线 gather 实现互为对拍。三条 fused-attention 分支
（DecodeOnly / PrefillNoCache / ChunkedPrefill）用一个记录型 `torch_npu` 桩
在算子边界上验证：layout、sparse_mode、因果掩码、逐请求
`actual_seq_lengths`、跨批 cumsum 的 kv 长度、输入切片与输出写回区间都逐条
断言（对该区域做 5 处变异全部被抓）。移植自 legacy Ascend PR #116
0003-0009（该分支在 910B2 上逐位验证过打包与 gather 内核）。

**910B2 设备复验已完成（2026-09-20，HEAD `84e5ab3`，见
`docs/validation-int4-20260920.md`）**：打包内核与纯 torch dequant-gather 逐位
复现语义参考（value `EXACT`、key `max|diff|=0`），三条注意力分支（prefill /
decode / **chunked prefill**）在玩具几何与出厂默认几何（head 128 / kv 8 /
group 128 / block 128）下都与"对同一份 gather 结果直接调用 fused attention"
完全一致（差异 0，且把 chunked 的输出写回区间改错只有该探针能抓到）；多请求
批量 decode（历史长度互不相干、各跨 2~5 个 block、只靠
`actual_seq_lengths_kv` 前缀和描述）同样零差异，跨请求泄漏类变异（所有请求读
同一行残差、切片不偏移）只有 `scripts/npu_probe_kivi_batched.py` 能抓到，它同时
给出设备侧量化口径：int4 历史 vs fp16 缓存的注意力输出偏差 ≤0.068 倍 K/V rms、
余弦 ≥0.990。多步生成（64~67 个 decode step、跨多次整窗 flush、真 triton 打包）
也逐步对过：每一步的 gather 都符合 pinned 的 flush 调度，出厂几何上唯一的不同是
128 行里有 1 行的某个元素落在**恰好一半**的格点上（归一化值数学上是 7.5），内核的
fp32 中间结果是 7.49999973，于是比 `quantize_group` 低一档——这一条已写进
`quantize_group` 的 docstring，探针按"同档或平局相邻档"判定。两个探针还跑在 GQA 形状下（`KIVI_PROBE_GQA=7`，即 14Q/2KV 与
56Q/8KV）——这条形状此前完全没跑过，而把 `num_key_value_heads` 报错时 MHA 毫无
反应、GQA 立刻 NaN；纯 torch 兜底注意力（自己扩 q 头、自己拼掩码）也在设备上与
aclnn 对过，差异只有输出 rms 的 0.005。实验性融合 gather 仍误编译，保持不路由。
分派本身用 `scripts/probe_host_dispatch.py` 在**真实宿主类**上核对：
`auto`/`fp8`/`float16`
原样委托 `AscendAttentionBackendImpl`，`int8` / `kivi_int4` 各自返回插件组合的
实现类，用真实 `decode_context_parallel_size=2` 配置时抛
`NotImplementedError`。该脚本同时暴露并修掉了一处宿主漂移：新宿主已把
`enable_cp()` 换成 `enable_dcp()`/`enable_pcp()`，旧分派在真机上一选量化 dtype
就 `ImportError`。

**仍未验证的是端到端 serving**：该容器宿主的 `CacheDType` 是 pydantic 校验的
`Literal`，`kivi_int4`（和 `int8`）都不在其中，CLI 层面就会被拒；宿主还需按
上面的字节预算给每层分配两张等大缓冲，并暴露
`kivi_group_size` / `kivi_residual_length` 旋钮；对不上预算时插件在绑定期
fail-closed（见 `HOST_CONTRACT.md`、`docs/int4-host-integration.md`）。模型级
精度（真实权重下的输出质量）也要等端到端跑通后才能测。
