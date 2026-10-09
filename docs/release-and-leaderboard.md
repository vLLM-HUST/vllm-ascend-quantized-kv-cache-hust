# 大包发布与打榜指南

整理日期：2026-10-02。依据 `vLLM-HUST/vllm-hust-website` main `5effa7b` 与
`vLLM-HUST/vllm-hust-benchmark` main 的最新文档梳理。

当前决策：

1. 本插件要发布**自己的 PyPI 大包**，发布前后完成测试；
2. 打榜场景**优先 swe-prefix**；
3. 沟通中提到的 VPN 限制忽略，不作为流程约束。

完整打榜操作记录（部署 → 启动 vLLM → swe-prefix 压测 → 提交排行榜 PR，
含实测踩坑）见 [swe-prefix-benchmark-playbook.md](swe-prefix-benchmark-playbook.md)。

## 1. 总览

发布大包与打榜是两条衔接的流水线：插件方把插件发布成 PyPI 包；压测方用
benchmark 仓跑压测，成绩自动 merge 进官网。对应关系：

| 环节 | 实际载体 | 仓库 |
|---|---|---|
| 大包（pip lib） | 插件 wheel + sdist | 本仓库 `vllm-ascend-quantized-kv-cache` |
| 压测软件 | `vllm-hust-benchmark`（编排层，底层 `vllm bench`） | [vllm-hust-benchmark](https://github.com/vLLM-HUST/vllm-hust-benchmark) |
| 成绩 merge 到 website | submissions → snapshots → HF 数据集 → 官网 | [vllm-hust-website](https://github.com/vLLM-HUST/vllm-hust-website) |
| 打榜场景 | swe-prefix（优先）；agentx（已退役，见 §4） | [swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse) |

注意：`BetterScale`（`vllm-betterscale==0.4.1`）是被测插件之一，不是压测
软件；压测编排层是 `vllm-hust-benchmark`。

## 2. 大包发布（PyPI）

### 2.1 规范文档

- [PLUGIN_STANDARD.md](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/PLUGIN_STANDARD.md)：
  插件标准。manifest `0.2-experimental`，经 `vllm_hust.extension_bundles`
  entry point 注册；alpha 公开发布被验收门卡住（BidKV 全生命周期、Mooncake
  真实数据路径、Production Stack K8s 演练需在 Ascend 环境留公开证据）。
- [PYPI_RELEASE_REMEDIATION.md](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/PYPI_RELEASE_REMEDIATION.md)：
  发布纪律。只发 vllm-hust 项目线自己的包；用 scoped token；不得拿旧 PyPI
  版本当当前版本。目前生态内仅 `vllm-hust==0.17.2.post1` 在 PyPI 上。
- [BidKV打包与发布指南](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/PLUGIN_STANDARD.md)
  （组织内规范原文；本仓库曾镜像到包目录，已随 main 的
  `fb83900` 删除，只留上游链接）：
  最完整的逐步操作手册与检查表。

### 2.2 发布流程

命令级的权威流程在 [packaging-and-release.md](packaging-and-release.md)
（定版本 → 测试 → 构建 → 查产物 → 隔离冒烟 → 上传 → 回装验证），本页
不再重复同一套命令。这里只留打榜场景额外要做的三件事：

1. **记录发布坐标**：`git rev-parse HEAD` 与工作树是否干净，写进发布说明；
2. **两个文件都要显式列出**（wheel + sdist），PyPI 不许覆盖同版本内容，
   所以改代码必须先升版本；
3. **website 同步**：更新 website 仓 `data/version_meta.json` 后，在**那个仓**
   里跑 `node --check assets/versions-page.js`、
   `python -m pytest tests/test_site_structure.py`、
   `bash scripts/check_stale_versions.sh`，review 后推 main。

### 2.3 本插件的差异点

- 加载机制走 `vllm.general_plugins` entry point（`bootstrap.register_plugins`），
  安装即被 vLLM 动态加载；**唯一启用开关是命令行参数**：
  ```bash
  vllm serve MODEL --kv-cache-dtype int8        # 动态 per-channel INT8
  vllm serve MODEL --kv-cache-dtype kivi_int4   # KIVI INT4
  ```
  未传 dtype 时不执行任何量化路径。
- 宿主约束记录在 `HOST_CONTRACT.md`：宿主声明为 `vllm-ascend-hust`（与
  manifest 的 `host.name` 一致），依赖
  `AscendAttentionBackend.get_impl_cls()` 分派点；context parallel 下拒绝启动。
- 仅 NPU 可用，不是通用 vLLM 插件；发布说明需写明验证过的宿主版本
  （vllm-ascend-hust commit、CANN、torch_npu）。

早先与打榜操作记录（vSpec 模板规范）比对出的**两个缺口，现已闭合**
（合并 main 的 `4aab295` / `d6a2e81`）：

- `pyproject.toml` 有 `vllm_hust.extension_bundles` entry point，指向
  `extension_manager_manifest` 包；**只保留一个同名表**——重复声明会让
  TOML 非法、整个构建失败（本次合并的文本 auto-merge 就产生了重复，
  `tests/test_plugin.py` 里的 entry point 断言会当场抓到）。
- 除 Bundle v1 之外，另有一份 v0.2 Manager manifest
  （`schema_version 0.2-experimental` / `extension_id` / `kind:
  in_process_plugin` / `implementation` / `activation`），发现入口不自行
  启用后端。两份 manifest 的版本与 `__version__` 由同一测试钉住，
  `extension_id` 与本 bundle 的 `bundle_id` 一致，`activation` 指向的
  entry point 名必须等于 wheel 实际发布的那个。

## 3. 打榜流程

### 3.1 压测怎么跑

```bash
python -m vllm_hust_benchmark.cli run-test serving_llama8B_tp1_sharegpt --execute
bash scripts/run-official-v0180-baselines.sh   # 官方基线一键入口（vllm-hust-benchmark 仓，不在本仓库）
```

硬件为 Ascend 910B2（benchmark 仓 `agent.md` 指定当前活跃机器，禁止无证据
引入 910B3 默认值）。

swe-prefix 压测的完整命令模板、OFF/ON 对比方法、数据口径（TPOT/E2E 要从
`requests.jsonl` 按 `end <= 900` 自算）、以及四件压测前必须对齐的事，见
[swe-prefix-benchmark-playbook.md](swe-prefix-benchmark-playbook.md) §10。

### 3.2 成绩 merge 到 website 的数据流

两条独立链路：

- **Runs 榜（自动）**：`export-leaderboard-artifact` 生成
  `run_leaderboard.json` + manifest → 写入 `submissions/<run-id>/`（含
  env-manifest、pip-packages、checksums、server 日志）→
  `publish-website --execute` 调 website 仓 `aggregate_results.py` 聚合成
  `leaderboard-data/snapshots/` → CI bot-commit 到 benchmark main →
  `push-to-hf.yml` 同步 HF 数据集 `intellistream/vllm-hust-benchmark-results`
  → 官网前端按 GitHub snapshots → HF → 本地优先级读取。**不是 PR**。
- **Benchmark settings / Frontier 页（人工）**：评审通过后直接替换 website 仓
  `data/leaderboard_frontier.json`，跑 website 仓的
  `scripts/curate_frontier_repeats.py`（不在本仓库），
  被替代点进 `archived_points`。

### 3.3 打榜红线

- 禁止伪造、重放、手改指标；必须保留完整 provenance（引擎与包版本、commit、
  workload、精度、芯片型号/数量）。
- 正式数据禁用 `--enforce-eager`：跑不起来就标 failed，不许 eager fallback 上榜。
- 官方基线 repeat 3 次取「最接近成功候选中位数」的一次，不取最优；
  Runs 页不做 fastest-repeat 挑选，禁止对百分位再平均。
- Frontier 用 whole-run best-of：只保留 output tokens/s/chip 最高的完整有效
  run，不许逐指标拼点。
- 插件榜排序 = 各并发档 `candidate/native` 吞吐比几何均值 − 1，跨档等权。

### 3.4 插件在压测中如何启用

压测环境 `pip install <插件 PyPI 包>`；`VLLM_PLUGINS` 环境变量控制插件开关
（跑 native baseline 时 `export VLLM_PLUGINS=""` 防污染）。本插件另由
`--kv-cache-dtype` 决定是否激活量化路径。每个上榜点的 configuration 记录
完整版本 pin（引擎、插件、CANN、torch_npu）。

## 4. 打榜场景

### 4.1 swe-prefix（优先）

- workload 仓：[swe-prefix-reuse](https://github.com/vLLM-HUST/swe-prefix-reuse)，
  回放 NVIDIA Open-SWE-Traces 的 8 条完整轨迹（360 个 assistant turn）。
- 客户端把下一段输入接在**实际已生成的 token IDs** 上（真实续写，不
  retokenize）；每次会话重放用唯一 cache salt + sticky 路由，测**会话内
  prefix 复用**；支持 session rotation depth D。
- 指标：Y = 900 秒窗口内实际收到输出 tokens / 900 / 全部芯片；X = P90 of
  `(output tokens - 1)/(last-token time - first-token time)`（仅窗口内完整
  结束的请求）。任何请求失败整跑作废；被打断的并发档必须单独重测。
- 详细协议：
  [FRONTIER-QWEN35-SWE-PREFIX.md](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/FRONTIER-QWEN35-SWE-PREFIX.md)。

与本插件的关系：swe-prefix 的长前缀复用正好压在 KV cache 容量与反量化开销
上，是量化 KV cache（INT8 / KIVI INT4）最能体现收益的场景，因此列为本插件
打榜的优先场景。

模型注意：现有参考实现是 Qwen3.5-35B-A3B（setting
`qwen35-35b-a3b-bf16-sweprefix-smoke-v1`，website PR #339）；我们的目标是
**Qwen3.5-30B**，workload 需用 30B tokenizer 重新准备 prepared 变体，且
排行榜要匹配或新建 30B 对应的 cohort/setting——workload sha256 必须登记在
cohort 契约的 `prepared_workload_variants` 里，tokenizer fingerprint 与
契约一致，否则仓库测试会红（等价性验证方法见操作记录 §11.3）。

### 4.2 agentx（已退役，仅作背景）

- workload 仓：[agentx-bench](https://github.com/vLLM-HUST/agentx-bench)，
  官方 256K corpus（393 会话 / 68,266 请求）DAG 延迟回放 + 每轮 prefix
  cache busting。
- **2026-09-27 起从活跃榜单退役**（cohort 带 `display_withdrawal`，见
  [BENCHMARK-SETTINGS.md](https://github.com/vLLM-HUST/vllm-hust-website/blob/main/docs/BENCHMARK-SETTINGS.md)）。
  不再作为打榜目标，仅在与外部对齐场景名词时说明，避免误跑。

## 5. 当前状态与行动清单

现状（2026-10-02，feat/int4 @ 当前 HEAD）：

- 版本 `0.2.0.dev0`；`dist/` 里有 9/21 本地构建的 wheel + sdist，**未发布 PyPI**
  （`pip index versions vllm-ascend-quantized-kv-cache` 查不到）。
  合并 main 后 `_version.py` 已随其 release 元数据升到 `0.2.0rc4`。
- `python -m pytest -q`：当时 90 passed；当前 HEAD 112 passed。
- manifest 已在 wheel 路径上，`scripts/verify-wheel.sh` 就绪。
- pyproject 已补 `vllm_hust.extension_bundles` entry point，v0.2 Manager
  manifest 也已就位（两个缺口都随合并 main 闭合）。

### 5.1 打榜服务器部署记录（vllm-hust-cyj，2026-10-02）

- 硬件/系统：2× Ascend 910B2（64 GiB HBM），CANN 9.1.0，torch_npu
  2.10.0.post4，Python 3.12.13，openEuler aarch64 容器。
- 部署目录：`/root/bench/`（`src/` 三个仓、`venv/`、`scripts/`、`logs/`、
  `results/`）；模型在 `/vllm-workspace/models/Qwen3.5-35B-A3B`（ModelScope
  官方快照，14/14 分片）。
- 栈组合：`vllm-hust@ba82f2122`（`VLLM_TARGET_DEVICE=empty` editable，避开
  arm_compute 的 GitHub 拉取——服务器 GitHub 直连/SSH 均不通）+
  `vllm-ascend-hust@b0613602f`（editable，`COMPILE_CUSTOM_KERNELS=0`——其
  vendored protobuf/自研 aclnn 算子在 CANN 9.1 下编译失败，待宿主侧修复）
  + 插件 wheel `0.2.0.dev0`。
- 宿主补丁：`scripts/host_int4_patch.py` 的锚点在该 pair 上有 3 处漂移，
  部署版脚本 `apply_host_edits_ba82f21.py`（在服务器 `/root/bench/`）按
  10 处锚点落地并做了幂等验证：CacheDType 同时补 `int8`/`kivi_int4`
  字面量、STR_DTYPE、is_quantized_kv_cache、KVQuantMode/KIVI_INT4、
  get_kv_quant_mode 分支、get_kv_cache_shape 的 KIVI 形状分支、runner
  两处通用 reshape 调用点传 dtype。
- 验证：`probe_host_dispatch.py` **RESULT: PASS**（auto/int8/kivi_int4
  分派与委托全对、CP 守卫 fail-closed）；探针顺带修了两处宿主 revision
  漂移（`enable_cp` 优先、`enable_cp` 也是 lru_cache 需 cache_clear）。
- workload：`swe-prefix-reuse prepare`（tokenizer 同模型）→
  `prepared/qwen35.json`，最长会话 141,269 token，与操作记录一致。
- 自动执行：bench 树上的 `scripts/auto_bench.sh`（**不在本仓库**，位于
  服务器 `/root/bench/scripts/`，与 `qkv35_run.sh` 同目录）由 setsid 拉起
  （断 SSH 不死），每
  60s 检查两卡 HBM 空闲 ≥52GB 且连续两拍，自动串行跑 OFF（fp16 基线）
  与 INT8 两组：serve → 20s 探测（valid/failed_requests 门槛）→ 900s
  正式窗口。进度看 `results/AUTO_STATE`，完成标记 `results/AUTO_DONE`。
- **已知风险（共享机器）**：卡上有同事的自动化测试循环（`/root/tk_dev`
  `pytest tests/quant`，单卡 ~25GB、循环执行），fp16 基线需要两卡各
  ≥52GB，无法共存；监督器只在两卡全空闲时启动。若该循环长期不间歇，
  需协调暂停（不可反复 kill 他人的任务）。
- 口径提醒：kivi_int4 目前需 `--enforce-eager`，违反打榜红线，本次打榜
  主力为 INT8；kivi_int4 待宿主 aclnn 算子修通后再评估。
- **容器重建事件（2026-10-06）**：pod 从 2×910B2 升到 4×910B2，`/vllm-workspace`
  被重建（5.9T 卷换 893G overlay），原 68GB 模型与 `/data` 只读共享盘一并
  消失；`/root`（98G 长期卷）上的全部部署物幸存，CANN 9.1.0 / torch_npu
  不变。模型重新下载到长期卷 `/root/bench/models/Qwen3.5-35B-A3B`，
  `start_vllm.sh` 与 `server-metadata.json` 已指向新路径。
- **10-03/10-04 的另一段工作**（本会话 fork 或本人）：用
  `start_vllm_qkv.sh`/`bench_qkv.py` 完成了三组**功能验证**（8 请求 ×
  1024 token、max-model-len 8192、INT4 走 eager）：kivi_int4 端到端 serve
  跑通（`READY after 250s`、`http_ok: true`）——验证记录里"端到端未跑通"
  的缺口已闭合；数字（2.0~2.3 tok/s，思考模式拖慢）仅证明功能，不是
  打榜口径。结果存 `/root/bench/results/VERIFY/`。
- **正式 swe-prefix 压测（2026-10-06 启动）**：`auto_bench2.sh` 串行跑
  int8 → kivi_int4 → off 三组，每组 20s 探测（`failed_requests=0` 门槛）
  + 900s 正式窗口；三组服务端配置唯一差异是 cache dtype 与（仅 INT4）
  `--enforce-eager`（无 ACL graph 路径，去 cudagraph 参数）。进度看
  `results/AUTO2_STATE`，完成标记 `results/AUTO2_DONE`，成绩在
  `results/c8-900s-{int8,kivi_int4,off}/`。INT4 组受 eager 拖累，数字
  只能作对照，不能作为正式榜点提交。
- **宿主 GDN chunked-prefill bug（2026-10-06 实测）**：Qwen3.5-35B-A3B 是
  GDN 混合架构；开启 `--enable-chunked-prefill` 后，GDN 投影自定义算子的
  decompose 在 profile 阶段按 `[tokens, 2048, -1, head_dim]` 重排
  （`vllm_ascend/ops/gdn.py:403` 经 `patch_qwen3_5.py:115`），输入缺一个
  head 维度直接 `RuntimeError: shape invalid`，**任何 cache dtype 都启动
  失败**（与 KV 插件无关，宿主问题，待报给宿主 owners）。绕过：关
  chunked prefill，`--max-num-batched-tokens 262144`（非 chunked 模式要求
  batched ≥ max_model_len；也 ≥ workload 最长会话 141,269 token）。此配置
  与操作记录（chunked ON + 8192）不同，属协议偏差，写榜时要声明。
- **监督器等卡逻辑反转 bug（已修）**：npu-smi HBM 行是"已用/65536"，
  旧脚本把已用当空闲，10-02 那轮"等到卡"实为等到卡被占满、随即启动失败。
  v2.1 起按 `65536 - used` 计算，阈值 53500MB（≥0.85×60.96GiB 硬要求），
  每组 arm 启动前复查，两拍稳定才启动。
- **2026-10-06 深夜环境战役（平台热升级撞上压测）**，按时间序：
  1. 平台把 pod 从 2 卡升到 4 卡并开始热升级 Ascend 栈
     （torch_npu post4→post6、triton 包被删得只剩无 `__init__` 的目录壳，
     旧目录改名 `triton_ascend_broken`）。
  2. **GDN 编译崩溃**：`--no-enforce-eager` 下 torch.compile 追踪
     Qwen3.5 GDN（`gdn.py:403` 经 dynamo decompose）必然 `shape invalid`；
     eager 不走 decompose，不受影响 → 三组 arm 全改 `--enforce-eager`
     （组间口径一致；代价：INT8 也上不了正式榜，本轮定位是三组 KV 方案
     对照）。
  3. **非 chunked 死锁**：关 chunked prefill 时 vLLM 强制
     `batched_tokens ≥ max_model_len`，262K-token 的 eager profile 前向又
     OOM（57.6/60.96GB）→ 只能 chunked ON + batched 8192 + eager（与操作
     记录同型，仅多 eager）。
  4. 23:32 出现一例 **aicore timeout**（sampler 阶段，retCode 0x25），
     随后 torch_npu 导入失败——均为平台升级中间态的 collateral，非我们
     代码问题；等升级收敛后 venv 导入链自愈，分派探针复验 PASS。
  5. venv 曾装 PyPI triton 3.5.0 试图遮蔽系统坏 triton——该轮子无 ascend
     backend（vllm 报 "0 active driver(s), Disabling Triton"，GDN gated
     layernorm 内核拿 stub 崩 `next_power_of_2`）→ 已卸载；**venv 不装
     triton**，透传系统版本。混搭内核拼修被否决（基准数字不可信）。
  6. 修复等待中：平台 triton 迁移完成后重跑监督器
     （`auto_bench2.sh`：int8 → kivi_int4 → off，chunked+eager+8192）。

行动清单：

- [ ] 版本转正：`_version.py` 与 manifest `bundle_version` 同步改为 `0.2.0`
- [x] 按 vSpec 模板补 `vllm_hust.extension_bundles` entry point，并对齐
      v0.2 manifest（合并 main 完成，见 §2.3）
- [ ] 重新 `python -m pytest -q` + `python -m build` + `verify-wheel.sh`
- [ ] 隔离环境 wheel 冒烟（含 manifest 定位检查）
- [ ] 申请/使用 scoped token 发布 PyPI，记录文件名与哈希
- [ ] 正式 PyPI 无缓存回装，`--kv-cache-dtype int8|kivi_int4` 各做一次启动冒烟
- [ ] swe-prefix 打榜：装 wheel → `--kv-cache-dtype` 激活 → benchmark 跑
      swe-prefix 场景 → `export-leaderboard-artifact` → merge 进 website
      （命令模板与排行榜 PR 流程照操作记录 §10–§11 执行）
- [ ] 发布后更新 website `data/version_meta.json` 并跑三项校验

## 5.2 正式压测结果（2026-10-07，swe-prefix C8×900s，2×910B2）

协议：`prepared/qwen35.json`（8 条 SWE 轨迹、最长会话 141,269 token），
C8、900s 窗口、seed 0、TP2+EP、chunked prefill ON + `--enforce-eager` +
batched 8192；三组仅 `--kv-cache-dtype` 不同。原始件在服务器
`/root/bench/results/c8-900s-{int8,off}/`，本仓留档见下表。

| 指标 | int8 | off (bf16 KV) |
|---|---:|---:|
| 窗口内完成 turn 数 | 13 | 13 |
| 窗口内输出 token | 1,936 | 1,888 |
| 输出吞吐（两卡合计） | 2.151 tok/s | 2.098 tok/s |
| 每卡吞吐 | 1.076 | 1.049 |
| decode P90（每用户） | 0.281 tok/s | 0.274 tok/s |
| TTFT P95 | 12.43 s | 13.05 s |
| TPOT 均值 / P95 | 3,597 / 3,693 ms | 3,697 / 3,774 ms |
| E2E P95 | 469 s | 481 s |
| 失败请求 | 0 | 0 |

结论与口径：

1. **int8 vs bf16**：吞吐 +2.5%、TTFT −5%、TPOT −2.7%（单次观测、无重复，
   差距在噪声范围内；方向上 INT8 无退化、略有正贡献）。**不能据此写
   "有提升"**。
2. **绝对速度的口径**：eager 模式下该引擎 pair 跑 Qwen3.5-35B（GDN 混合
   MoE）单流 ~1.5 tok/s、C8 每用户 ~0.28 tok/s——与 10-04 功能验证的
   warmup/单延迟数据一致，是这对引擎 + eager 的固有水平（对照：操作记录
   里 legacy 引擎 + FULL_DECODE_ONLY graph 模式 decode P90 ~40 tok/s）。
   打榜红线禁 eager，因此**今天两组数字都不能作为榜点提交**，价值是
   (a) int8 通路在真实 serving 下 900s 零失败的功能证明，(b) 与 fp16 的
   同栈对照。
3. **kivi_int4 被插件自己的对齐守卫拦截**（fail-closed 生效）：
   `KIVI key flush requires contiguous aligned token groups`——
   进程内第 1 个请求成功返回，第 2 个起触发；swe-prefix 探测 8/8 请求
   失败（探测门槛拦截，未进 900s）。10-04 同配置曾跑通 8×1024 token，
   今天唯一已知差异是 triton-ascend 换了安装来源（平台毁掉旧 overlay 后
   从华为云镜像装的 3.2.2），打包内核代码生成差异是头号嫌疑，插件逻辑
   bug 未排除。下一步：在 `_write_kivi_key_quant_cache` 守卫处加形状
   诊断，定位错位的 token groups 来自哪一步（宿主 slot 顺序 / 内核写出 /
   residual 复用）。
4. 环境备注：本次三组跑在平台 4 卡新 pod 上（torch_npu post6 系统 +
   venv post4 降级、triton-ascend 3.2.2 镜像重装）；10-06 深夜的升级
   战役详见 §5.1。

## 6. 参考

- 官网：https://vllm-hust.sage.org.ai/
- 打榜数据（HF）：https://huggingface.co/datasets/intellistream/vllm-hust-benchmark-results
- website docs（发布/打榜）：`PLUGIN_STANDARD.md`、`PYPI_RELEASE_REMEDIATION.md`、
  `BENCHMARK-SETTINGS.md`、`LEADERBOARD-RUNS-REVIEW.md`、
  `FRONTIER-REPEAT-SELECTION.md`、`FRONTIER-QWEN35-SWE-PREFIX.md`、
  `VERSION_METADATA.md`
- 本仓库：`HOST_CONTRACT.md`、`docs/packaging-and-release.md`、
  `docs/PLUGIN_STANDARD`（上游组织规范）
