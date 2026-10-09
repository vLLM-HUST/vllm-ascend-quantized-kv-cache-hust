# 以网络通信模块为例:从 PR 抽取到插件打包、部署、压测

> **收录说明**： teammates 提供的完整打榜操作记录（2026-10 收录，原文照存）。
> 示例包是 a2a-buffer-reuse / Qwen3.5-35B-A3B / 2×910B2；我们套用到
> `vllm-ascend-quantized-kv-cache` / Qwen3.5-30B 时，模型名、workload 变体与
> 排行榜 cohort 要相应替换，其余流程一致。与本项目现状的两处差异见
> [release-and-leaderboard.md](release-and-leaderboard.md) §2.3。

本文以 `vllm-hust-a2a-buffer-reuse`(下称"示例包")为参照,记录把 0.17 中某个网络通信相关优化抽取成独立 pypi 插件、部署到双卡容器并压测的完整流程。

> 示例包路径:`/Applications/hust-project/vllm-hust-a2a-buffer-reuse`
> 它是 `vllm-hust-legacy017-perf` 台账里 `#7` 候选(OProj / FlashComm2 all-to-all 复用接收 buffer)的单机制抽取。

---

## 0. 整体流程一览

| 步骤 | 做什么 | 产出 / 验证点 |
|---|---|---|
| 1 | 从 0.17 抽取网络相关 PR | 候选清单(含证据编号) |
| 2 | 以 vSpec 为模板打包成 pypi 结构 | 可 `pip install` 的包 + manifest |
| 3 | 启动 k8s 容器(双卡) | 容器内 `npu-smi info` 可见 2 张卡 |
| 4 | 把包传入容器 | 容器内能看到源码目录 |
| 5 | `pip install` | 安装无报错 |
| 6 | 验证打包成功 | `extension list` 显示 `disabled` |
| 7-8 | 启用并验证 | `extension list` 显示 `enabled` |
| 9 | 启动 vLLM | 服务 Ready、能正常生成 |
| 10 | 压测 | 对比 OFF/ON 吞吐、看证据事件 |

---

## 1. 抽取 0.17 中的网络相关 PR

参考 `legacy017-perf`《网络通信部分:现状与缺失清单》,逐条挑选候选。

**挑选原则(示例包的做法):**
- **单机制**:一个包只做一件事。示例包只做"all-to-all 接收 buffer 复用",不夹带其他优化。
- **不依赖台账仓库**:包不 import `vllm-hust-legacy017-perf`,只在 manifest 里引用它的证据编号(E038 / E039 / E049)。
- **记录来源 commit**:示例包 manifest 的 `additional_config` 记了 `source_core` 和 `source_ascend` 两个 commit,方便回溯到原始实现。
- **先判断能不能独立打包**:并不是所有 PR 都能做成插件。能打包的通常是"对宿主已有类做子类替换或 monkey patch"的改动;需要改 C++ 算子、改调度核心数据结构的,一般不适合。

**示例包的对应关系:**

| 项 | 内容 |
|---|---|
| 机制 | 用持续复用的 buffer 替换每次 forward 都执行的 `recv_buf = torch.empty(...)` |
| 作用对象 | vLLM-Ascend-HUST 的 `OProjRowParallelOp`、`Flashcomm2OProjRowParallelOp` |
| 台账证据 | E038、E039、E049 |

---

## 2. 以 vSpec 为模板打包成 pypi 结构

模板:<https://github.com/vLLM-HUST/vllm-hust-vSpec>。可以让 AI 按模板生成,但生成后要人工核对下面几项。

### 2.1 目录结构(以示例包为准)

```
vllm-hust-a2a-buffer-reuse/
├── pyproject.toml
├── README.md
├── src/vllm_hust_a2a_buffer_reuse/
│   ├── __init__.py          # 导出 __version__ 和 register
│   ├── _version.py          # 版本号唯一来源
│   ├── plugin.py            # register() 入口:开关判断 + 补丁安装
│   ├── buffer.py            # 可复用 buffer 的实现
│   └── manifests/
│       └── vllm-hust-extension-v0.2.json
└── tests/
    ├── test_activation.py   # 开关/默认关闭
    ├── test_a2a_guard.py    # 宿主代码不匹配时拒绝打补丁
    └── test_buffer.py       # buffer 复用逻辑
```

### 2.2 pyproject.toml 的关键点

```toml
[project]
name = "vllm-hust-a2a-buffer-reuse"
dynamic = ["version"]                      # 版本从 _version.py 读取
requires-python = ">=3.10"

# ① vLLM 自动发现插件的入口
[project.entry-points."vllm.general_plugins"]
a2a_buffer_reuse = "vllm_hust_a2a_buffer_reuse.plugin:register"

# ② vllm-hust-ext 命令发现扩展包的入口
[project.entry-points."vllm_hust.extension_bundles"]
"org.vllm-hust.a2a-buffer-reuse" = "vllm_hust_a2a_buffer_reuse.manifests"

[tool.setuptools.dynamic]
version = {attr = "vllm_hust_a2a_buffer_reuse._version.__version__"}

[tool.setuptools.package-data]
vllm_hust_a2a_buffer_reuse = ["manifests/*.json"]   # 不写这行,manifest 不会被打进包
```

**容易漏的地方:**
- 两个 entry-point 组缺一不可:缺 ① vLLM 不会加载插件;缺 ② `vllm-hust-ext extension list` 看不到它。
- `package-data` 必须包含 `manifests/*.json`,否则安装后 `extension list` 为空。
- `extension_id` 要和 entry-point 名字、manifest 里的 `extension_id` 三处一致:`org.vllm-hust.a2a-buffer-reuse`。

### 2.3 manifest 的关键字段

| 字段 | 示例包的值 | 说明 |
|---|---|---|
| `extension_id` | `org.vllm-hust.a2a-buffer-reuse` | 全局唯一 ID,`extension list` 显示的就是它 |
| `extension_version` | `0.1.0.dev0` | 与 `_version.py` 保持一致 |
| `kind` | `in_process_plugin` | 在 vLLM 进程内运行 |
| `host.version_range` | `>=0.23,<0.24` | 限定适用的宿主版本 |
| `runtime.process_scope` | `worker` | 补丁作用在 worker 进程 |
| `implementation` | `plugin:register` | 入口函数 |
| `activation.environment` | 三个 `VLLM_HUST_A2A_BUFFER_REUSE_*` 变量 | 默认 `ENABLE=0`,即默认关闭 |
| `additional_config` | 准入状态、来源 commit、证据编号 | 用于追溯,不影响运行 |

### 2.4 插件设计要点(示例包的做法,新包可直接沿用)

1. **默认关闭,显式开启**:`register()` 先检查开关,没开就直接返回,不 import 任何运行时模块。
2. **Kill switch 优先级最高**:`..._KILL_SWITCH=1` 时无论 ENABLE 是什么都不安装。
3. **拒绝猜测**:用 `inspect.getsource` 检查宿主类里是否存在预期的代码片段(`recv_buf = torch.empty`)。找不到就抛 `RuntimeError`,**绝不对不认识的实现打补丁**。
4. **幂等**:用模块级 marker(`__vllm_hust_a2a_buffer_reuse__`)防止重复安装。
5. **证据事件**:开启 `..._EVIDENCE=1` 后,首次观察到真正复用了 buffer 时输出一条 `LEGACY017_EVIDENCE runtime_effective ...` 到 stderr。这样"补丁装上了"和"补丁真的生效了"可以分开判断。
6. **测试不依赖真机**:所有测试 mock 宿主依赖,没有 torch / vLLM-Ascend 也能跑。

### 2.5 本地自测

```bash
cd /Applications/hust-project/vllm-hust-a2a-buffer-reuse
python -m pytest tests/
```

---

## 3. 启动 k8s 容器(双卡)

- 申请 **2 张 NPU**。本次环境是 Ascend 910B2,单卡 HBM 约 61 GiB。
- 进容器后先确认:

```bash
npu-smi info
```

**注意设备编号:** `npu-smi info` 里看到的 NPU 编号(例如 0 和 7)是**物理编号**;容器内给 vLLM 用的逻辑编号是 **0 和 1**。所以后面要用 `ASCEND_RT_VISIBLE_DEVICES=0,1`,写成 `0,7` 会报 `local rank 1 is out of bounds for 1 devices`。

---

## 4. 把项目传到容器

任选一种:

```bash
# 方式 A:scp(目录)
scp -r /Applications/hust-project/vllm-hust-a2a-buffer-reuse \
    <容器登录名>:~/workspace/

# 方式 B:打 zip 再传
cd /Applications/hust-project
zip -r vllm-hust-a2a-buffer-reuse.zip vllm-hust-a2a-buffer-reuse -x "*.DS_Store"
scp vllm-hust-a2a-buffer-reuse.zip <容器登录名>:~/workspace/
# 容器内:unzip vllm-hust-a2a-buffer-reuse.zip
```

传输前排除 `.DS_Store`(macOS 产生的垃圾文件)。

---

## 5. pip install

```bash
pip install -e ~/workspace/vllm-hust-a2a-buffer-reuse
```

- `-e`(可编辑安装)方便在容器里直接改代码调试;正式验证可以去掉 `-e`。
- 安装后 vLLM 会通过 `vllm.general_plugins` 入口自动发现它,但**默认什么都不做**(ENABLE 默认为 0)。

---

## 6. 验证打包成功

```bash
vllm-hust-ext extension list
```

应输出:

```
org.vllm-hust.a2a-buffer-reuse 0.1.0.dev0 disabled
```

**排错:**

| 现象 | 可能原因 |
|---|---|
| 列表里没有这个包 | 没有 `extension_bundles` 入口,或 `package-data` 漏了 manifest.json |
| ID 或版本不对 | manifest 与 `_version.py` / entry-point 名不一致 |
| 命令不存在 | 容器里没有装 `vllm-hust-ext` |

---

## 7. 启用扩展

```bash
vllm-hust-ext extension enable org.vllm-hust.a2a-buffer-reuse
```

---

## 8. 验证启用成功

```bash
vllm-hust-ext extension list
```

应输出:

```
org.vllm-hust.a2a-buffer-reuse 0.1.0.dev0 enabled
```

**注意(实测踩过的坑):** `enable` 只是把状态记成 `enabled`,**并不会让补丁装进 vLLM**。
- `vllm-hust-ext extension env` 的输出里 `VLLM_HUST_A2A_BUFFER_REUSE_ENABLE` 仍然是 `"0"`。
- 插件默认是 no-op,必须在启动 vLLM 的 shell 里 `export VLLM_HUST_A2A_BUFFER_REUSE_ENABLE=1`,补丁才会安装。
- 怎么确认装上了:服务日志里出现 `LEGACY017_EVIDENCE installed mechanism=a2a_receive_buffer_reuse classes=OProjRowParallelOp`(每个进程一条)。没有这行,就是没装上。
- 装上了也不等于复用路径在跑。要看 `runtime_effective` 事件(见第 10 步)。

---

## 9. 启动 vLLM

### 9.1 启动命令(已在 910B2 ×2 上验证可启动)

```bash
source /usr/local/Ascend/nnal/atb/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=0,1
export VLLM_HUST_A2A_BUFFER_REUSE_EVIDENCE=1   # 输出 installed / runtime_effective 证据事件
export VLLM_HUST_A2A_BUFFER_REUSE_ENABLE=1     # 仅 ON 组设置;OFF 组不要设
# (压测用 workload 最长会话约 141K token,所以 max-model-len 用 262144,见第 10 步)

vllm serve /models/Qwen3.5-35B-A3B \
  --served-model-name qwen3.5-35b-a3b \
  --host 127.0.0.1 --port 18180 \
  --dtype bfloat16 \
  --kv-cache-dtype auto --block-size 128 \
  --tensor-parallel-size 2 --enable-expert-parallel \
  --pipeline-parallel-size 1 --data-parallel-size 1 \
  --max-model-len 262144 --gpu-memory-utilization 0.85 \
  --max-num-seqs 16 --max-num-batched-tokens 8192 \
  --no-enable-prefix-caching --enable-chunked-prefill --no-enforce-eager \
  --seed 0 --scheduling-policy fcfs \
  --distributed-executor-backend mp --disable-custom-all-reduce \
  --no-trust-remote-code --load-format auto \
  --no-enable-log-requests --uvicorn-log-level info \
  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --cudagraph-capture-sizes 1 2 4 8 16
```

后台启动:

```bash
nohup setsid /root/start_qwen35.sh > /tmp/vllm_qwen35.log 2>&1 < /dev/null &
```

### 9.2 各关键参数为什么这样设

| 参数 | 原因 |
|---|---|
| `source .../atb/set_env.sh` | 否则 worker 加载不到 `libatb.so`,报 `OSError: libatb.so: cannot open shared object file` |
| `ASCEND_RT_VISIBLE_DEVICES=0,1` | 容器内逻辑卡号,见第 3 节 |
| `--tensor-parallel-size 2` | 模型权重约 68 GB,单卡约 61 GiB 放不下(单卡会在 `load_model` 阶段 OOM);TP=2 后每卡权重约 32.85 GiB |
| `--enable-expert-parallel` | 模型有 256 个专家,按专家切分更合适;若提示不支持,去掉即可 |
| `--dtype bfloat16` | 模型原生 bf16;改 fp16 不会减小权重体积,还有数值溢出风险 |
| `--served-model-name` | 要与实际模型相符,压测请求里的 `model` 字段必须与它一致 |

### 9.3 实测资源数据

- 每卡权重:约 32.85 GiB
- 可用 KV cache:约 16.73 GiB(每卡)
- `max-model-len 262144` 时 GPU KV cache 共 1,692,929 token,单请求 262144 token 时理论并发约 6.46 路
  (`max-model-len 32768` 时是 1,459,038 token;上下文越大,日志里的理论并发数越少)

### 9.4 验证服务可用

```bash
curl -s http://127.0.0.1:18180/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3.5-35b-a3b","messages":[{"role":"user","content":"用一句话介绍RDMA"}],"max_tokens":64}'
```

### 9.5 常见问题

| 现象 | 原因 / 处理 |
|---|---|
| `libatb.so` 找不到 | 没 source ATB 的 `set_env.sh` |
| `NPU out of memory` 发生在加载权重时 | 模型放不进单卡,用 TP=2 |
| `local rank 1 is out of bounds for 1 devices` | `ASCEND_RT_VISIBLE_DEVICES` 写成了物理编号 |
| 返回内容以 `Thinking Process:` 开头,`max_tokens` 被思考占满 | Qwen3.5 默认开启思考模式。请求里加 `"chat_template_kwargs":{"enable_thinking":false}`,或启动时加 `--reasoning-parser` |
| 看不到真实报错,只有外层 `Engine core initialization failed` | 根因在更前面的 `Worker` / `EngineCore` 日志里,用 `grep -nE "ERROR\|Error\|out of memory" 日志` 查 |
| 用 `pkill -f "vllm serve"` 把自己的 ssh 命令也杀了 | 改用 `pkill -f "[v]llm serve"`(方括号避免匹配到自身) |

---

## 10. 压测

工具:<https://github.com/vLLM-HUST/swe-prefix-reuse>

### 10.1 对比方法

OFF/ON 两组,除扩展开关外其余**完全相同**(同模型、同启动参数、同压测负载、同并发):

1. **OFF**:扩展为 `disabled`(或设置 `VLLM_HUST_A2A_BUFFER_REUSE_KILL_SWITCH=1`),启动 vLLM,压测,记录吞吐。
2. **ON**:扩展为 `enabled`,同样启动并压测,记录吞吐。
3. 每组多跑几轮取均值,避免单次抖动。

### 10.2 必须同时看"证据事件"

开启 `VLLM_HUST_A2A_BUFFER_REUSE_EVIDENCE=1`,启动后检查日志:

```bash
grep -E "LEGACY017_EVIDENCE|runtime_effective" /tmp/vllm_qwen35.log
```

| 结果 | 含义 |
|---|---|
| 有 `runtime_effective` 事件 | 复用路径确实被执行过 |
| 没有该事件,但补丁已安装 | 补丁装上了,但复用路径没被触发——此时吞吐差异不能归因于本优化 |

### 10.3 示例包已有的历史结论(作为对照基准)

来自台账 E038 / E039 / E049:

| 拓扑 | 结果 |
|---|---|
| TP2,非 PD 分离 | 补丁可安装,但 `runtime_effective` 命中 **0 次** |
| TP2 吞吐 | 原生 `207.65 tok/s`,开启后 `202.07 tok/s`——有界的**负结果** |
| TP4 | **尚未在真机验证**,是该机制真正可能有收益的场景 |

所以本流程(双卡、TP2)跑出来大概率仍是"无收益"。这不代表流程有问题,**压测的目的是产出可信的证据,而不是证明有提升**。要验证收益需要 4 卡 TP4。

### 10.4 压测前必须对齐的四件事(本次实测踩过)

1. **`--model` 用服务端注册的名字**,即 `--served-model-name`(`qwen3.5-35b-a3b`),不是模型路径。
2. **`--server-max-context` 必须等于服务端真实的 `--max-model-len`。** workload 的 8 个会话最长到 141269 token,所以服务端要 `max-model-len 262144`;32768 会让后面 5 个会话中途超长失败。
3. **`--server-metadata` 要单独写一个 JSON**,不能和 `--workload` 传同一个文件。至少含:引擎及版本、MOD 及版本、模型、精度、tokenizer、芯片型号与数量、脱敏后的启动命令。
4. **`swe-prefix-reuse` 命令要先装**:`cd swe-prefix-reuse && pip install -e .`。

### 10.5 本次压测流程与结果

先用 20 秒探测确认协议检查能过(`--duration 20`,看 `valid: true`、`failed_requests: 0`),再跑正式 900 秒。OFF 和 ON 两组用同一套参数,只差 `VLLM_HUST_A2A_BUFFER_REUSE_ENABLE`:

```bash
swe-prefix-reuse run \
  --workload prepared/qwen35.json \
  --endpoint http://127.0.0.1:18180/v1/completions \
  --model qwen3.5-35b-a3b \
  --server-max-context 262144 \
  --concurrency 8 --duration 900 --chips 2 \
  --server-metadata /root/workspace/server-metadata.json \
  --output results/c8-900s-off      # ON 组改成 c8-900s-on
```

每次改开关都要**重启 vLLM**(补丁是在进程启动时装的)。停服务用 `pkill -f "[v]llm serve"`。

**结果(2× 910B2,TP2+EP,C8,各一次 900 秒):**

| | OFF(扩展已装、补丁未启用) | ON(`ENABLE=1`) |
|---|---:|---:|
| 窗口内完成请求 | 314 | 318 |
| 输出吞吐(两卡合计) | 194.48 tok/s | 198.33 tok/s(+2.0%) |
| 每卡吞吐 | 97.24 | 99.17 |
| decode P90 | 40.72 tok/s | 40.71 tok/s |
| TTFT P50 / P95 | 1346 / 3093 ms | 1323 / 3089 ms |
| TPOT 均值 / P95 | 37.91 / 57.30 ms | 37.25 / 55.53 ms |
| E2E P95 | 73204 ms | 74158 ms |
| 失败请求 | 0 | 0 |
| `installed` 事件 | 0 | 4 |
| **`runtime_effective` 事件** | **0** | **0** |

**怎么解读:**
- ON 比 OFF 高 2.0%,但 decode P90 没变,且每组只有一次观测、没有重复、没有置信区间,这个差距在噪声范围内。
- ON 组补丁装上了,但**从没观察到复用路径真正执行**(`runtime_effective` 为 0),所以**不能把这 2% 归因于补丁**。这和示例包 README 里 TP2 的既有结论一致。
- 文档里不写"有提升";要验证收益得上 TP4。

**几个数据口径的坑:**
- `summary.json` 里**没有** TPOT 和 E2E。它们要从 `requests.jsonl` 里按窗口内请求自己算:`request.end` 是**相对**窗口起点的秒数(不是绝对时间),用 `end <= 900` 过滤,样本数应与 `summary.json` 的 `requests_completed_in_window` 一致(314 / 318)。
- TTFT P95 直接用 `summary.json` 里的 `ttft_seconds_p95`。
- 吞吐口径:窗口内实际收到的输出 token 数 ÷ 900 ÷ 所有占用的卡。排空阶段(窗口结束后仍在跑的请求)的 token 不计入。

---

## 11. 提交成绩到排行榜折线图

页面:<https://vllm-hust.sage.org.ai/leaderboard-runs.html#settings>(setting = `qwen35-35b-a3b-bf16-sweprefix-smoke-v1`)。

页面的数据来自 [vLLM-HUST/vllm-hust-website](https://github.com/vLLM-HUST/vllm-hust-website) 仓库。提交方式就是往这个仓库提 PR。本次 PR:<https://github.com/vLLM-HUST/vllm-hust-website/pull/339>。

### 11.1 要改哪些文件

| 文件 | 内容 |
|---|---|
| `data/leaderboard_frontier.json` | 在 `points` 里加 point;在对应 cohort 的 `contract.display_series_ids` 里登记新系列 |
| `data/leaderboard_frontier_swe_evidence.json` | 在 `runs` 里加对应的运行证据(summary、指标、`requests.jsonl` 的 sha256、client 配置) |
| `docs/FRONTIER-...md` | 证据文档,point 的 `evidence.url` 指向它 |
| `tests/leaderboard_frontier_model.test.cjs` | 这个测试里写死了点数等,加点后要同步更新 |

格式可直接参照 `scripts/import_qwen35_v018_native.py` 里的 `make_point` / `make_evidence_run`。

### 11.2 一个 point 必须有的内容

- `id`、`cohort_id`、`label`
- `configuration`:`engine`、`engine_version`、`mods`、`hardware`(型号 + 总卡数)、`context_capacity_tokens`、完整的 `parameters`(TP/EP、max-num-seqs、prefix caching、graph 模式、运行时版本、脱敏后的启动命令……)
- `load`:`concurrency`、`concurrency_series`、`presentation_group`、`session_rotation_depth`
- `metrics`:`output_tps`、`decode_p90_tps`、`ttft_p95_ms`、`tpot_ms`、`tpot_p95_ms`、`e2e_p95_ms`、`completed_requests`(没测到的不填,**不要填 0**)
- `evidence`:`status: "measured"`、HTTPS `url`、`run_ids`、`aggregation`、`benchmark_protocol`(含 workload sha256 和 tokenizer fingerprint)

### 11.3 仓库会强制检查的规则(实测被卡住过)

1. **workload sha256 必须在 cohort 契约的 `prepared_workload_variants` 里**,tokenizer fingerprint 必须与契约一致。不一致会让测试变红。
   - 仓库接受变体的前提是**证明等价**:把 `tokenizer.path`、`tokenizer.transformers`、`tokenizer.fingerprint` 替换成官方值后,重新序列化(`ensure_ascii=False`, `separators=(",",":")`)能还原出官方 sha。
   - 本次验证通过:换成 `/workspace/models/Qwen3.5-35B-A3B` / `5.17.0` / 官方 fingerprint 后,精确还原 `8044561f…`,说明会话内容字节级相同。
   - 但本机 tokenizer 的 fingerprint(`sha256(backend_tokenizer.to_str() + chat_template)`)确实不同(`tokenizer.json` 等在下载后被改过),所以**只能登记为"内容等价",不能说 tokenizer 一致**。
2. **一个窗口必须是 900 秒。** 60 秒的探测结果只能用于验证链路,不能当成绩提交。
3. **`mods` 里的 ID 应是工作坊目录(`data/ecosystem.json`)里的组件**,`mod_sources` 需要一个 GitHub 仓库地址。本次 a2a-buffer-reuse 还没发布成仓库、也不在目录里,所以 ON 点没带 `mod_sources`,并在 PR 里说明了。
4. **不要改 `default_groups`**:它决定所有访客的默认视图,只加到 `display_series_ids`,新组由用户在页面上勾选。
5. **单个点画不出线**:折线需要同一系列有多个并发点(C1/2/4/8/16)。本次每个系列只有 C8 一个点,页面上是两个点,不是折线。

### 11.4 本地校验(与 CI 一致)

```bash
node --test tests/leaderboard_runs_model.test.cjs tests/leaderboard_frontier_model.test.cjs \
  tests/leaderboard_tiering_evidence.test.cjs tests/leaderboard_mooncake_evidence.test.cjs \
  tests/plugin_performance.test.cjs tests/loader_publication_identity.test.cjs tests/snapshot_status.test.cjs
pytest tests/ -q
python scripts/check_engine_version_consistency.py
pre-commit run --files <改动的文件>     # 需要 Python >= 3.10
```

- macOS 自带 Python 是 3.9,`pre-commit` 的 `mdformat` 等钩子装不上。用 `uv venv --python 3.11` 建环境。
- `mdformat` 会自动重排 `.md` 文件,改完要 `git add` 后再跑一遍确认干净。
- `detect-secrets` 对哈希串比较敏感,已有写法是在行尾加 `# pragma: allowlist secret`。本次没触发。
- 本次 `pytest` 有 1 项失败(`test_site_structure.py::test_leaderboard_names_exact_settings_without_frontier_jargon`),在干净的 `main` 上也失败,与改动无关,在 PR 里已说明。

### 11.5 提交 PR

```bash
gh repo fork vLLM-HUST/vllm-hust-website --clone=false      # 没有权限直接推,先 fork
git clone https://github.com/vLLM-HUST/vllm-hust-website.git && cd vllm-hust-website
git remote add fork https://github.com/<你的账号>/vllm-hust-website.git
git checkout -b add-<标识>-sweprefix
# ...改文件、本地校验...
git commit -am "..." && git push -u fork HEAD
gh pr create --repo vLLM-HUST/vllm-hust-website --base main \
  --head <你的账号>:add-<标识>-sweprefix --title "..." --body "..."
```

### 11.6 PR 描述里必须如实写的

审核人靠这些判断数据能不能信,**不要只写好消息**:
- 每组只有一次观测、没有重复;
- 补丁是否真的生效(`runtime_effective` 事件数),没观察到就不要归因;
- OFF 的确切含义(本次是"扩展已装但补丁未启用");
- workload / tokenizer 与 cohort 不一致的地方,以及验证到了什么、没验证什么;
- 和 cohort 里其他点的配置差异(本次:前缀缓存关闭、无 MTP、运行时版本不同);
- 覆盖范围(最大 prompt 长度、完成的会话数);
- 哪些检查没跑(本次没跑 Playwright 浏览器校验)。

---

## 附:套用到新 PR 时的检查清单

- [ ] 这个 PR 是单机制、能靠子类替换/打补丁实现,而不是改核心算子?
- [ ] 包里是否**没有** import 台账仓库?
- [ ] 两个 entry-point 组(`vllm.general_plugins`、`vllm_hust.extension_bundles`)都配了?
- [ ] `package-data` 包含 manifest?
- [ ] `extension_id` / 版本号在 pyproject、`_version.py`、manifest 三处一致?
- [ ] 默认关闭?有 kill switch?
- [ ] 宿主代码不匹配时会报错拒绝,而不是猜着打补丁?
- [ ] 有证据事件区分"已安装"与"已生效"?
- [ ] 测试不依赖真机?
- [ ] manifest 的 `additional_config` 记了来源 commit、证据编号、准入状态?
- [ ] 在 README 里如实写明准入状态,没有把"能安装"写成"有收益"?
