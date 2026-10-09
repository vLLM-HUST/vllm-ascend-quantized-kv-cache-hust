# serve 实测：int8 与 kivi_int4 在 vllm-hust 里跑通（2026-10-03/04）

结论先行：**两个方案都能集成、都能 serve、输出正确**。kivi_int4 的
KV 容量收益精确兑现（稠密模型 3.56x）；**Qwen3.5-35B-A3B 也已跑通**
（2026-10-04 补齐，见 §4），输出连贯，decode 慢在模型层的纯 torch
GDN 兜底、与 KV 方案无关。

## 1. 验证了什么

机器：vllm-hust-cyj（2×910B2，CANN 9.1.0）。宿主：bench 树
vllm-hust@ba82f2122 + vllm-ascend-hust@b0613602f，装着本周的插件
wheel。模型：Qwen2.5-1.5B-Instruct（稠密；用它的原因见 §4）。
三组完全同参：`--enforce-eager --no-enable-chunked-prefill
--no-enable-prefix-caching --max-model-len 8192 --max-num-seqs 16
--tensor-parallel-size 2`，只差 `--kv-cache-dtype`。

真宿主分派核对（`scripts/probe_host_dispatch.py`）：PASS——
`int8 → AscendInt8KvAttentionImpl`、`kivi_int4 →
AscendKiviInt4KvAttentionImpl`、其余 dtype 原样走宿主、context
parallel 直接拒绝。

## 2. 稠密模型首跑（2026-10-03）

> 数据统一收在 [benchmark.md](benchmark.md)（项目只保留那一份）。
> 这里只留结论：Qwen2.5-1.5B-Instruct 上三组 serve 全部成功，页大小
> 数学被并发比精确验证（fp16→int8 恰好 2.0x、→kivi_int4 恰好
> 3.56x）；kivi 吞吐慢是 gather→反量化纯 torch 读路径的已知定位；
> 一个精度观察样本：int8 把"9.11 和 9.8 哪个大"答反，fp16 与
> kivi_int4 都正确。

## 3. 这轮在宿主上实测出的三处接合问题（已修）

前两处已回写进 `scripts/host_int4_patch.py`（4b、4c 两号编辑），
在新副本上一键可复现：

1. **KIVI 页大小缺分支**（vllm/v1/kv_cache_interface.py，
   `FullAttentionSpec.real_page_size_bytes` 覆盖版）：页按稠密
   256B/token 算，池子被对半切成稠密宽度的 K/V，插件的 S=72 视图
   直接 view 失败。补上 KIVI 分支（两侧各 S）后页变 18432，
   并发数立刻从 906x 涨到 1611x。
2. **head_size_v 劫持**（vllm_ascend/worker/model_runner_v1.py，
   `_reshape_kv_cache_tensors`）：这一代 `FullAttentionSpec` 天生带
   `head_size_v` 字段（默认 None），`hasattr` 恒真，V 的形状总被改回
   稠密头。kivi 时 V 必须与 K 同形。
3. **MoE 路由算子缺失**（vllm_ascend/device/device_op.py）：
   自研 `npu_moe_init_routing_custom` 底层调 CANN 的
   `aclnnMoeInitRoutingCustom`，而 CANN 9.1.0 的 libopapi.so 里没有
   这个符号。已改成等价的 `torch_npu.npu_moe_init_routing_v2`
   （参数族相同、四元组返回相同、scale None 进 None 出；空卡微测 +
   DeviceOperator 入口验证过）。这条与 KV 插件无关，是这台 CANN
   上跑任何 MoE 模型的公共修复。

服务器 bench 树已带以上全部修复（原始文件备份为 *.bak-qkv*），
另清掉了一个隐患：启动脚本的 `pkill` 杀不掉 setproctitle 改名的
Worker/EngineCore 孤儿进程，残留会占着 56GB×2 显存（现在的启动
脚本已加 `[V]LLM::` 清理）。顺带发现并修正：auto_bench.sh 的
free_mb 把 npu-smi"已用"当"空闲"，判断完全反向——之前 OFF/INT8
两臂的失败多半是被它带着在卡忙时硬启动。

## 4. Qwen3.5-35B-A3B：2026-10-04 已跑通

第一天失败后逐层排查，这台 9 月的宿主树对 Qwen3.5 的混合架构
（GDN 线性注意力 + 少量全注意力层）落后一整条链路，CANN 9.1.0 又
缺了一批新算子。逐个修掉后三组全部 serve 成功、输出连贯：

| # | 位置 | 问题 | 修法 |
|---|---|---|---|
| 1 | `device_op.py` MoE 路由 | CANN 9.1.0 缺 `aclnnMoeInitRoutingCustom` | 换 stock `npu_moe_init_routing_v2`（参数族/返回值相同） |
| 2 | `gdn.py` conv1d ×7 处 | 自定义算子绑定要 `Tensor?`，代码传 Python tuple | 新增 `_conv1d_opt`：tuple→NPU int64 张量，空→None |
| 3 | `gdn.py` conv1d eager（prefill+decode） | CANN 9.1.0 缺 `aclnnCausalConv1d` | 走 `_310p` 的纯 torch `causal_conv1d_fn`（同分页状态语义） |
| 4 | `device_op.py` GDN 门控 | 缺 `aclnnFusedGdnGating` | 走 `_310p` 的 `fused_gdn_gating_pytorch`（fp32 同数学） |
| 5 | `gdn.py` prefill chunk | 缺 `aclnnChunkGatedDeltaRuleFwdH` | 走 `_310p` 的 `chunk_gated_delta_rule_pytorch` |
| 6 | `gdn.py` decode 递推 | 预编译 wheel 根本没注册 `npu_recurrent_gated_delta_rule`（自研内核需 COMPILE_CUSTOM_KERNELS=1） | 复用第 5 条的 chunk 参考（T=1），状态转置进出同 prefill |

原始文件都备份为 `*.bak-qkv*`。中途还发现两个易踩的坑已记录：
启动脚本 `pkill -f vllm serve` 杀不掉 setproctitle 改名的
`VLLM::Worker` 孤儿进程（会占满双卡显存）；auto_bench 的 free_mb
把 npu-smi"已用"当"空闲"，判断反向。

**benchmark 数据**：统一见 [benchmark.md](benchmark.md)——三组全部
serve 成功、输出连贯；吞吐 ~2 tok/s 由第 3–6 条的纯 torch/Python
兜底支配（三组同瓶颈，与 KV 方案无关）；KV 可并发 fp16→量化组
+40%（90→126）。两个待查项也记在那里（int8 与 kivi 并发数相同；
GDN 兜底的性能恢复路径）。

## 5. 接下来

1. **kivi 融合读路径**（roadmap 第 4 步的另一半）：把 gather→反量化
   融进注意力内核，稠密模型上 4.3 → 接近 fp16 才算把 3.56x 的收益
   兑现成带宽收益。
2. **GDN 兜底的性能**：Qwen3.5 要出打榜级数字，需要把 §4 的 3–6 条
   换回真内核（升级宿主树）或写融合版，否则 ~2 tok/s 只是功能验证。
3. int8 的精度与并发数问题值得一次正经评测。
4. Qwen3.5 宿主升级后重跑本验证（同脚本、换 QKV_MODEL 即可）。
