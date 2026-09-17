# int8_dynamic 验证记录（container-86，2026-09-12）

> 记录口径：本文件只描述当时环境里发生的事实与结果，事后不回写。
> 后续复验请新增带新日期的记录。

## 冻结输入

| 项 | 值 |
|---|---|
| 验证环境 | `container-86`（vllm-ascend-hust + vllm-hust 双栈，torch_npu，Ascend 910B2） |
| 模型 | Qwen3-0.6B（拷贝目录，config 注入量化描述） |
| 方法 | `int8_dynamic`（注册键 `VLLM_HUST_KV_INT8_DYNAMIC`） |
| 注入方式 | `vllm-hust-kv-inject`（完整 ModelSlim 形状：逐层 `fa_k/fa_v.scale`=FAQuant + 全线性层/embed_tokens=FLOAT） |
| 激活方式 | `VLLM_HUST_KV_METHODS=int8_dynamic` + 显式 `bootstrap.register_plugins()`（V1 in-proc 引擎在模型加载前不触发 `vllm.general_plugins`） |
| 引擎 | `LLM.generate`，`enforce_eager`，fp16，未设 `--kv-cache-dtype`（浮点 KV） |

## 结果（逐项）

| # | 验证项 | 结果 |
|---|---|---|
| 1 | scheme 注册进宿主 `@register_scheme` 注册表 | ✅ 幂等再激活 OK |
| 2 | `fa_quant_type` 分发：checkpoint 描述 → 每层 scheme 选择 | ✅ |
| 3 | `AscendKVCacheMethod.create_weights` 对 28/28 层完成 impl 类手术 | ✅（日志为证） |
| 4 | 引擎加载 + 浮点前向 `LLM.generate` 生成 | ✅ |
| 5 | **int8 存储路径（`--kv-cache-dtype int8_per_token_head`）** | ❌ **阻塞于宿主侧**，见下 |
| 6 | attention mixin 的 NPU int8 前向 | ⚠️ 未复验（依赖 #5 先通） |

## 已知阻塞（宿主侧，非本仓代码）

`kv_cache_dtype=int8_per_token_head` 时，宿主 `model_runner_v1.py` 的
KV cache 初始化器分配 head/2 打包布局（`[2314, 128, 8, 64]`），而
`_reshape_kv_cache_tensors` 按满 head int8 重排（要求
`[1122, 128, 8, 128]`）——两者不一致导致初始化失败。

- 探测手段：`python scripts/verify_host_sources.py --vllm-ascend-src <宿主checkout>`
  会点名该函数（`int8_kv_split_factor_asymmetry`，review 级）。
- 修复归属：**宿主仓**（vllm-ascend-hust）；修复后本仓只需复跑
  `npu_e2e` 门（how-to-run.md §6 配方 + §8 矩阵）。

## 后续行动（2026-09-17，本仓补齐，待真机闭环）

阻塞已从"现象记录"推进到"根因定位 + 可用修复"：

1. **根因定位**（宿主源码逐行算术，无需容器）：
   `AscendModelSlimConfig.get_kv_quant_split_factor` 对稠密 fa_quant 层
   采用 legacy"V×2"字节预算（切分 `[3.0, 1.5]`，K 只拿 1/3），而
   `get_kv_quant_dtype` 返回 (int8, int8)、重排按满头同形视图要求 K/V
   各占 1/2——K 缺 1/3 字节，int8 页大小下 `.view` 必然越界。浮点页
   大小时因缓冲偏大而"碰巧能跑"（V 区最多浪费 2/3），这正是本记录
   第 4 项能通过、第 5 项失败的原因。
2. **宿主补丁提案**：
   `provenance/host-fixes/vllm-ascend-hust-0001-kv-split-factor-symmetric-dense.patch`
   （对 `b0613602f` 工作树 `git apply --check` 通过；MLA 分支保留 V×2）。
3. **插件侧过渡守卫**：`VLLM_HUST_KV_ALLOC_GUARD=1`（how-to-run §6.5；
   `tests/test_alloc_guard.py` 以 stub 复刻宿主行为验证调和语义）。
4. **剩余动作**：容器上按 §6.5 配方复验 int8 端到端 → 结果以
   `vllm-hust-kv-evidence` 校验的新记录 + 新验证记录归档，本记录不再
   回写。

## 同批附带发现（已修复）

1. scheme 构造器向宿主基类 `AscendAttentionScheme` 传参在不同 fork 表面
   漂移下失败 → 已改参数化转发失败时无参回退（`tests/test_adapters.py`）。
2. 多继承动态 impl 类 `tp_base` 落在 mixin 上导致 `__class__` 赋值被拒 →
   `apply_impl_surgery` 增加整实例克隆替换回退（`tests/test_adapters.py`）。

## 遗留待办

1. 宿主侧 `_reshape_kv_cache_tensors` 修复后：int8 存储路径 `npu_e2e`
   复验（本记录门级推进的唯一阻塞）。
2. `vllm.general_plugins` 钩子在 V1 in-proc 引擎模型加载前未触发；
   `vllm serve` 完整入口的触发时机需宿主侧确认。
3. int8 路径的 PPL / ShareGPT 与 BF16 同协议对照（`matched_benchmark`
   门，尚未开始）。

## 原始日志

container-86 本地实验档案（未公开发布；对外声明 `npu_e2e` 前需按
acceptance-matrix 的证据纪律归档）。
