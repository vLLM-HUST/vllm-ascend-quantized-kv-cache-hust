# NPU 实现要点（Ascend 910B）

本文记录设备层的实现决策、已验证结论与已知的 NPU 坑。方法语义与选型见
[schemes.md](schemes.md)；运行方法见 [how-to-run.md](how-to-run.md)。

## 1. 设备边界与 fail-closed 纪律

**内核层（triton-ascend / torch_npu）只在 Ascend NPU 上执行**，这是硬
边界，用三条机制守住：

1. **惰性导入**：`ops/` 与 attention mixin 里的 torch / torch_npu /
   triton 全部在**使用它的函数体内**导入（经 `core.runtime` 的
   `import_torch` / `import_torch_npu` / `import_triton`）。缺依赖时
   报一条带调用语境的精确错误，而不是 import 级联堆栈。
2. **统一守卫**：设备路径入口先过 `runtime.require_npu(what)`——
   `torch.npu.is_available()` 不成立就抛 `RuntimeError`，错误信息明确
   说明"内核层仅 NPU，纯语义 API 任何机器可用"。
3. **能力探测只查元数据**：`module_available()` 用 `find_spec` 判断
   可导入性，绝不真正 import（`bootstrap.detect_host` 判断
   `vllm_ascend` / `vllm` 同理）。

triton 的获取顺序：先试 `vllm.triton_utils`（vllm 系宿主环境），失败
直接 `import triton`（独立 triton-ascend 环境）——这是对 legacy 内核
模块仅做的导入适配。

## 2. 内核清单与验证状态

| 内核/路径 | 位置 | 设备 | 验证状态（910B2） |
|---|---|---|---|
| `kivi_pack_key_cache` / `kivi_pack_value_cache` | `ops/triton/kivi_pack.py` | triton-ascend | **逐位通过**（对照 `KiviInt4Semantics.fake_quant_*` CPU 参考） |
| `kivi_dequant_gather_cache`（实际路由） | `ops/kivi_gather.py` | 纯 torch 算子 | **逐位通过** |
| `_kivi_dequant_gather_*_kernel`（融合 gather，**不路由**） | `ops/triton/kivi_gather_experimental.py` | triton-ascend | 误编译，保留待重验（§3.3） |
| INT8 在线 antiquant（decode / chunked-prefill / prefill 三分支） | `methods/int8_dynamic/attention_mixin.py` | torch_npu `npu_fused_infer_attention_score` | 端口保真度；端到端待宿主集成 |
| KIVI TND 分派 + 稠密 fallback | `methods/kivi_int4/attention_mixin.py` | torch_npu + torch | 同上 |
| 统一激活管线（`kv_methods.activate` → 宿主 `register_scheme`） | `core/activation.py` + `adapters/vllm_ascend_hust/` | 宿主进程 | ✅ 真实 vllm-ascend-hust 宿主**进程内实测**（2026-09-11，910B2 容器：六方法注册可见、幂等再激活 OK；注意这只是注册链路，serving 未验证） |

vectorcore 数量探测（pack 内核的 grid 依赖）是从宿主
`get_vectorcore_num` 本地移植的：查不到 `num_vectorcore` 属性直接抛错。

## 3. 每个方法的 NPU 路径细节

### 3.1 `kivi_int4` 的存储布局

分页缓存是 6 元组 `(k_quant, k_scale, k_mn, v_quant, v_scale, v_mn)`：

```
k_quant: [num_blocks, num_kv_heads, head_size, block_size/8]   int32 word
k_scale/k_mn: [num_blocks, num_kv_heads, head_size, block_size/group_size]
v_quant: [num_blocks, block_size, num_kv_heads, head_size/8]   int32 word
v_scale/v_mn: [num_blocks, block_size, num_kv_heads, head_size/group_size]
```

键按 token 组量化（scale/mn 沿最后一维 = 每块内组数），值按 head 维组
量化（沿 head/组 维）。int32 word 里 lane L 在 bit 4L。

`ops/kivi_layout.py` 是两条路径（triton 打包与 torch gather）共享的
布局校验器——**"合法布局"的判断只有一份**，避免两条路径认知漂移。

### 3.2 `int8_dynamic` 的三条前向分支

全部走 `npu_fused_infer_attention_score`（fused inference 算子原生支持
antiquant 参数，形状要求 BNSD 视图 `[1, H, 1, D]`）：

- **DecodeOnly**：BNSD 布局直接读分页 int8 缓存 +
  `key/value_antiquant_scale/offset`，`block_table` 传块表。
- **ChunkedPrefill**：decode 行同上；prefill 行走 TND 布局——若是
  "全新 prefill"（`seq_lens == qlen`，缓存无历史）直接用浮点 K/V，
  否则把分页 int8 缓存 gather 反量化成稠密再算。
- **PrefillNoCache / PrefillCacheHit**：TND；缓存命中读回的是 int8，
  先 `_dequant_paged_kv_to_dense`（`ops/int8_ops.py`，按块表 gather +
  seq_lens 掩码 + 逐通道反量化）。

未知的 attention state 一律 `RuntimeError`（fail-closed）。

### 3.3 关键事故记录：triton-ascend 3.5 融合 gather 误编译

上游（ascend#116 0003/0005 期起）的融合 dequant-gather triton 内核在
triton-ascend 3.5 上**误编译**：静默读到垃圾数据、漏写输出、两次运行
表现不同；16×16 与 32×32 tile 均复现（910B2 实测）。

处置：

- `kivi_dequant_gather_cache` 改路由到 `ops/kivi_gather.py` 的**纯
  torch gather**（上游 0003/0005 期的原版方法），910B2 上逐位验证通过；
- 融合 triton gather 内核**保留但不引用**，等 triton-ascend 修复后用
  `scripts/npu_probe_kivi_dim.py` 重验再切回；
- 决不"跑着未验证的上游融合路径"。

配套诊断脚本（都在 Ascend 容器上跑，见 how-to-run.md §4）：

- `scripts/npu_probe_kivi_key.py`：定位 key gather NaN 来源；
- `scripts/npu_probe_kivi_dim.py`：哨兵预填（输出预填 777.0）区分
  "漏写 store"还是"算错"，并对比 16/32 两种 tile；
- `scripts/npu_smoke_kivi.py`：端到端 pack → gather → 对拍两个 CPU
  参考（fake_quant 语义参考 + 手工解包布局参考），并给出失败归因
  （pack 语义错 vs gather 错）。

### 3.4 已知的 torch_npu 适配坑

**aclnnRightShift 不支持广播**：int4 解包若写成广播移位
`packed >> shift_vector`，torch_npu 适配器会拒绝（且把 self 操作数当
输出形状）。因此 `KiviInt4Semantics.unpack_int4` 与
`kivi_gather._unpack_int4_words` 都写成**8 次标量移位 + stack**。
这是一处刻意保留的"非最优但正确"实现，改写前先在真机回归。

## 4. 有状态方法如何挂进宿主 impl：C8 式类手术

vllm-ascend-hust 宿主上，有状态方法（int8_dynamic / kivi_int4）的
挂载复用在树 C8 先例（`kv_c8.py`）：

1. `AscendHustAdapter.register()` 向宿主
   `vllm_ascend.quantization.methods.registry` 用 `@register_scheme`
   注册一个**生成的 scheme 类**（键 `VLLM_HUST_KV_*`，重复键在宿主
   会抛错，命名空间避免撞键）。
2. scheme 的 `create_weights` 执行类手术
   `layer.impl.__class__ = <生成的 impl 类>`。impl 类由
   `build_impl_cls(method_name, AscendAttentionBackendImpl)` 用
   `type()` 动态组合：`方法 mixin + 宿主 impl 基类`。
3. **关键细节**：换 `__class__` 不会重跑 `__init__`，所以手术後に必须
   显式调用 mixin 的状态初始化方法（`_state_init_method`：
   `_init_int8_dynamic_state` / `_init_kivi_state`），否则
   `enable_kivi`、残差窗口等属性缺失，forward 时才炸。
4. kivi 的配置旋钮从宿主 `vllm_config.cache_config` 的
   `kivi_group_size` / `kivi_residual_length` 读取（缺省 128/128），
   绑定后立刻按方法几何不变量校验。

## 5. KIVI 残差窗口状态机（设计要点）

- **每请求一行**：残差窗口是 ring buffer，行数按宿主
  `scheduler_config.max_num_seqs` 分配；`req_to_row` dict +
  `row_to_req` list + free list 管理行分配。
- **键整组 flush、值逐槽 flush**：键窗口写满 `group_size` 整组后
  调 triton pack 内核整组写入历史区；值逐 slot 写入。
- **flush 对齐不变量**（`is_aligned_key_window`）：每个组必须连续、
  组起点 block 内 group 对齐、**不跨缓存块**（pack 内核要求一个
  int32 word 的 8 个 lane 落在同一块）、槽位 id 连续递增。违反即
  拒绝 flush（fail-closed），绝不写坏历史区。
- **地址簿**：`ordered_slots` 把 (block_id, 块内偏移) 展开成每个请求
  的绝对槽位 id，是残差窗口与分页缓存之间的映射依据。
- **计算路径**：TND fused attention 作用于"gather 反量化的稠密历史 +
  全精度残差尾部"，另备纯 torch 稠密注意力 fallback（加性因果掩码
  `build_causal_mask`）。

## 6. 精度验证方法论

纯语义对象（`sol.semantics`）是设备内核的**数值参考实现**：

- `fake_quant_key` / `fake_quant_value` = 打包内核的量化语义参考；
- `unpack_int4` + `dequant_*_blocks` = 缓存布局数学参考。

NPU 冒烟（`npu_smoke_kivi.py`）同时对照两个参考：与 (a) 不符但与 (b)
一致 → pack 内核量化语义偏离；与两者都不符 → gather 侧问题。CPU 上
`tests/test_kivi_int4.py` 等测试用同一组参考对拍语义层本身。

当前结论（910B2）：**pack 与 gather 均逐位通过**；INT8 路径与端到端
serving 验证属于宿主集成阶段的工作。
