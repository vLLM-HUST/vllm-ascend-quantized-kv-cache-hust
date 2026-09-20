# INT4 (KIVI) 宿主接合清单

插件侧已完成（`feat/int4`）：`--kv-cache-dtype kivi_int4` 的 impl 分派、
INT4 语义与内核、以及把宿主给的两张缓冲切成 6 个视图。剩下的工作全在宿主：
这一代 vLLM 没有"一层六张 KV 缓存"的机制，所以宿主只需要按 INT4 的字节预算
给**两张等大通配缓冲**，其余由插件负责。

证据来自本地 checkout（非插件锁定基线，接合时需按目标基线复核）：
`vllm-hust@ba82f2122`、`vllm-ascend-hust@b0613602f`。910B2 容器上的宿主
（`vllm-hust@f18cf803c5`、`vllm-ascend-hust@17ed0571d`）另有一组行号，见下表
最后一列；两处结论一致，只是行号漂移。

| 接合点 | 本地 checkout | 容器 checkout |
|---|---|---|
| `CacheDType` 字面量 | `vllm/config/cache.py:19`（16 个取值） | `vllm/config/cache.py:39`（18 个取值）——两边都不含 `int8`/`kivi_int4` |
| `get_kv_cache_shape` | `vllm_ascend/attention/attention_v1.py:104` | 同文件 `:135`（两处都恒返回稠密形状、忽略 `cache_dtype_str`） |
| 未传 `cache_dtype_str` 的调用点 | `model_runner_v1.py:4556` | 同文件 `:4784`、`:4845` |
| CP 探测助手 | `enable_cp()`（`attention/utils.py:101`） | 只有 `enable_dcp():238` / `enable_pcp():243`，插件已同时支持（`fb046ec`） |

## 1. 宿主需要改的四处

1. **CLI 字面量**：`vllm/config/cache.py`（`CacheDType`）增 `"kivi_int4"`。
   该字段是 pydantic 校验的 `Literal`，所以容器宿主上连
   `--kv-cache-dtype int8` 都会在构造 `CacheConfig` 时报
   `ValidationError: Input should be 'auto', ...`（实测见
   `docs/validation-int4-20260920.md` 第 7 节）——不加字面量根本走不到插件。
   `_validate_cache_dtype` 只记日志、不校验额外配置，所以
   加字面量 + 下面两处映射即可，不需要运行时 hack。
2. **量化模式与页大小**：`vllm/v1/kv_cache_interface.py:33`（`KVQuantMode`）
   增 `KIVI_INT4`，`get_kv_quant_mode:62` 加字面量分支，
   `real_page_size_bytes:188` 按本仓 `KiviByteCacheLayout` 的公式给每 token
   字节数（现有 `INT4_PER_TOKEN_HEAD` 分支只减半宽，不含 scale/min，不能复用）。
3. **存储 dtype**：`vllm/utils/torch_utils.py:32` 的
   `STR_DTYPE_TO_TORCH_DTYPE` 加 `"kivi_int4": torch.uint8`（现有
   `int4_per_token_head` 条目在 :42）。
4. **缓存形状**：`vllm_ascend/attention/attention_v1.py:104` 的
   `get_kv_cache_shape` 目前忽略 `cache_dtype_str`、恒返回
   `(2, num_blocks, block_size, num_kv_heads, head_size)`；需要对
   `kivi_int4` 返回 `(2, num_blocks, block_size, num_kv_heads, S)`，
   `S` 见下式，取自 `cache_config.kivi_group_size`。
   同时 `vllm_ascend/worker/model_runner_v1.py:4556` 调用该函数时**没有传
   `cache_dtype_str`**，要把 spec/配置里的 dtype 传下去，否则分支永远不生效。

好消息是分配器已经天然给两张等大缓冲：`_reshape_kv_cache_tensors`
（`model_runner_v1.py:4562-4567`）用 `k_shape = kv_cache_shape[1:]`、
`v_shape = k_shape`，与 INT4 要求的"两侧字节数相等"完全一致。

## 2. 字节预算

每 token 每 head、单侧（K 或 V）字节数：

```text
S = head_size / 2 + 8 * head_size / group_size
    ^ int4 数据 0.5B/元素   ^ 每组一份 fp32 scale + fp32 min，摊到 group 个 token
region_bytes = num_blocks * block_size * num_kv_heads * S
```

默认几何（`head_size=128, group_size=128, block_size=128, num_kv_heads=8`）：
`S = 72`，单张 `region_bytes = num_blocks * 128 * 8 * 72`；相对 fp16 稠密 KV
压缩比 **3.56x**（int4 数据本身 4x，scale/min 吃掉一部分）。`group_size` 越
小、开销越大：`head=32, group=8` 时只有 1.33x。数值口径以
`KiviByteCacheLayout.compression_vs_fp16()` 为准，不要引用"4x"。

约束（插件在方法构造期 fail-closed，宿主分配必须满足）：
`group_size % 8 == 0`、`head_size % 8 == 0`、`head_size % group_size == 0`、
`block_size % group_size == 0`、`residual_length % group_size == 0`。
两张缓冲字节数不等、或对不上区域预算时，插件报
`KIVI key cache must hold N bytes` / `equal regions`，不会猜布局。

## 3. 接合后如何验证

```bash
# CPU：语义、状态机、两张缓冲->6 视图、FIA 参数逐条断言
PYTHONPATH=src python -m pytest -q
python scripts/check_int4_patch_parity.py     # 移植对账（补丁不变量）

# 宿主 venv：dtype 字面量 -> impl 类是否接到真实 AscendAttentionBackend
python scripts/probe_host_dispatch.py
# 安装态：wheel + entry point 经 vllm.plugins.load_general_plugins() 自动注册
python -m pip install --target /tmp/kivi-probe --no-deps dist/*.whl
PYTHONPATH=/tmp/kivi-probe python scripts/probe_installed_plugin.py

# 910B2：打包/gather 逐位 + 端到端
python scripts/npu_probe_kivi_key.py
python scripts/npu_probe_kivi_dim.py
python scripts/npu_smoke_kivi.py
python scripts/npu_probe_kivi_attention.py   # prefill / decode / chunked 三分支
KIVI_PROBE_SEQS=4 python scripts/npu_probe_kivi_batched.py  # 多请求 ragged 批量
KIVI_PROBE_STEPS=64 python scripts/npu_probe_kivi_generate.py  # 多步生成调度
vllm serve MODEL --kv-cache-dtype kivi_int4 --max-model-len 8192 --enforce-eager
```

## 4. 已知限制

- **无 ACL Graph 路径**：INT4 的残差窗口簿记含 Python 控制流，移植自 legacy
  的最终状态也没有 graph 捕获分支；先以 `--enforce-eager` 运行。
- **context parallel 明确拒绝**，与 INT8 同一口径（`HOST_CONTRACT.md`）。
- **gather 走纯 torch**：`ops/triton/kivi_gather_experimental.py` 在
  triton-ascend 3.5 上误编译（910B2 复现），保留但不路由。
- 宿主自带的 `int4_per_token_head` 是**另一种** INT4 格式（每 token-head 对称
  打包、RHT + 非对称 zp），与 KIVI 的分组非对称 + 残差窗口不是同一方案，
  不要混用字面量。
