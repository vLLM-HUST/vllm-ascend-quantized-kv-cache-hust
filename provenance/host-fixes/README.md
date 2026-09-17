# 宿主修复提案：fa_quant KV 分配切分（vllm-ascend-hust）

状态：**提案，待 NPU 复验**（本仓环境无 NPU；补丁已对
vllm-ascend-hust@`b0613602f` 的工作树 `git apply --check` 通过）。
宿主侧合入前，serve 进程可先用插件侧的 opt-in 守卫
（`VLLM_HUST_KV_ALLOC_GUARD=1`，见下）获得同等效果。

## 补丁

- `vllm-ascend-hust-0001-kv-split-factor-symmetric-dense.patch`
- 基线：vllm-ascend-hust `b0613602f`（2026-08-13，即 container-86
  2026-09-12 验证时的宿主状态）
- 应用：`git apply vllm-ascend-hust-0001-kv-split-factor-symmetric-dense.patch`

## 根因（可从源码逐行复算，无需容器）

int8 存储路径（`--kv-cache-dtype int8_per_token_head`）初始化失败的
机理是宿主两处口径互相矛盾：

**分配侧**（`model_runner_v1.py::_allocate_kv_cache_tensors`）：

```
k_tensor_size = kv_cache_tensor.size // k_tensor_split_factor
```

`AscendModelSlimConfig.get_kv_quant_split_factor` 对 fa_quant 层把 V
的维度放大一倍（`v_quant_head_dim = dims[1] * 2`），于是 Qwen 稠密层
`[128, 128]` 得到切分 `[3.0, 1.5]`：**K 只分到总字节的 1/3，V 分到 2/3**。

这个"V×2"是 legacy C8 布局（K 存 int8、V 留 fp16，字节比 1:2）的遗留
口径；`get_kv_quant_dtype` 对稠密层如今返回 **(int8, int8)**。

**重排侧**（`model_runner_v1.py::_reshape_kv_cache_tensors`）：

```
num_blocks = (k.numel() + v.numel()) // spec.page_size_bytes
k_shape = attn_backend.get_kv_cache_shape(num_blocks, block, heads, head_size)
k_cache = raw_k_tensor.view(int8).view(k_shape)
```

按 spec 的 int8 页大小（`2×block×heads×head×1B`）计算，K 需要总字节
的 **1/2**。分配只给了 1/3 → `.view` 越界失败。

即 container-86 记录的现象（how-to-run.md §8.1：初始化
`[2314,128,8,64]` 打包布局 vs 重排要求 `[1122,128,8,128]` 满头布局）。
附带效应：浮点 `--kv-cache-dtype` 时因为页大小翻倍，K/V 缓冲都偏大
而**碰巧能跑**，但 V 区最多浪费 2/3 显存——2026-09-12 冒烟能过浮点
前向正是这个原因。

## 修法

对称维度（K/V 同 head_size 的稠密层）必须对称切分；"V×2" 只在 V 确实
留 2 字节 dtype 时成立（MLA：`get_kv_quant_dtype` 返回 `(quant, ori)`，
K=kv_lora_rank ≠ V=qk_rope_head_dim，维度本就不等）。补丁把 `use_mla`
传进 `get_kv_quant_split_factor` 并据此分支，两处调用/实现共 ~15 行。

## 验证状态

- ✅ 补丁对基线工作树可干净应用（`git apply --check`）；
- ✅ 算术在纯 CPU 侧可复算（本仓 `tests/test_alloc_guard.py` 用 stub
  钉住了"宿主 legacy 行为→守卫调和"的语义）；
- ⬜ NPU 端到端（int8 存储 `npu_e2e` 门）：容器上应用补丁 → 复跑
  how-to-run.md §6 配方（注入 + `VLLM_HUST_KV_METHODS=int8_dynamic`
  + `--kv-cache-dtype int8_per_token_head`）→ 结果写入新的
  validation 记录；
- ⬜ MLA 回归（DeepSeek 系）确认 `use_mla` 分支不影响既有 V-fp16 布局。

## 插件侧过渡方案（本仓已实现，默认关闭）

`adapters/vllm_ascend_hust/alloc_guard.py`：激活方法时若
`VLLM_HUST_KV_ALLOC_GUARD=1`，包装宿主
`get_kv_quant_split_factor`——K/V 维度相等而宿主给出不对称切分时改回
对称。MLA（维度不等）不触碰；宿主修复后守卫自动退化 no-op。完整配方
见 `docs/how-to-run.md` §6.5。
