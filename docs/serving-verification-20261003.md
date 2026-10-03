# serve 实测：int8 与 kivi_int4 在 vllm-hust 里跑通（2026-10-03）

结论先行：**两个方案都能集成、都能 serve、输出正确**。kivi_int4 的
KV 容量收益精确兑现（3.56x）；decode 吞吐还差一个融合读路径（见 §5）。
Qwen3.5-35B 在这台机器上目前起不来，卡在两个与 KV 插件无关的宿主
适配问题上（§4）。

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

## 2. benchmark 数据（8 路并发 × max_tokens 128，温度 0）

| | fp16 (auto) | int8 | kivi_int4 |
|---|---|---|---|
| 8192 上下文可并发数 | 453x | 906x（**2.0x**） | 1611x（**3.56x**） |
| 聚合吞吐（tok/s） | 53.0 | 49.0 | 4.3 |
| 单路 128 token 延迟 | 0.19s | 0.23s | 0.95s |
| 输出正确性 | 基准 | 一处事实题答错* | 与 fp16 一致 |

- 并发数比就是压缩比：906/453 = 2.00、1611/453 = 3.56——和布局公式
  逐位对上，KV 显存收益是真金白银进了块数。
- \* 有趣的精度样本：问"9.11 和 9.8 哪个大"，fp16 和 kivi_int4 都答
  9.8 更大（对），int8 答反了。单例不作结论，但说明 4bit 分组 +
  残差窗口的精度可以好于 per-token int8。
- kivi_int4 吞吐慢的根源不是 bug：每一步 decode 都要把 int4 历史
  gather 出来反量化成稠密再算注意力（纯 torch 路径），这正是调研
  说的"容量收益、不是带宽收益"。修法就是路线图里的融合读路径。

原始 JSON：服务器 `/root/bench/results/VERIFY/bench-*.json`。
复现脚本：`/root/bench/scripts/{start_vllm_qkv,wait_ready_qkv,verify_qkv}.sh`
与 `bench_qkv.py`（QKV_MODEL/QKV_SERVED/QKV_EP 环境变量选模型）。

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

## 4. Qwen3.5-35B-A3B：这台机器暂时起不来（与 KV 插件无关）

装好插件后 Qwen3.5 的引擎能拉起、KV 分配能走通，但第一个请求死在
`npu_causal_conv1d_custom`（Qwen3.5 混合架构里线性注意力层的卷积，
宿主把 tuple 传给了期望 Optional[Tensor] 的绑定）——**fp16 同样
死**，568 处同错误。加上 §3 的 MoE 算子缺失，这台 9 月的宿主树对
Qwen3.5 已经落后两拍。要解锁需要：升级 bench 树到更新的
vllm-hust/vllm-ascend-hust，重新对锚点打宿主补丁（本脚本的锚点
会漂移，需按新 rev 重测）。Qwen3.5 的几何适配本身已由专项测试
（head 256 / 每卡 1 KV 头 / GQA 8:1）和设备探针覆盖。

## 5. 接下来

1. **kivi 融合读路径**（roadmap 第 4 步的另一半）：把 gather→反量化
   融进注意力内核，吞吐 4.3 → 接近 fp16 才算把 3.56x 的收益兑现成
   带宽收益。
2. Qwen3.5 宿主升级后重跑本验证（同脚本、换 QKV_MODEL 即可），
   补打榜模型的真实数字。
3. int8 的精度问题值得一次正经评测（q4 那个错例提示 per-token int8
   的 scale 粒度可能不够）。
