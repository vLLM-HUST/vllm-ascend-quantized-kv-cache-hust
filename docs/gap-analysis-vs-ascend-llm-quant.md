# 差距分析：对照 Ascend-LLM-quant（2026-09-17）

> 对比对象：[jxd1111/Ascend-LLM-quant](https://github.com/jxd1111/Ascend-LLM-quant)
> （下称"参考项目"，快照：`docs/` 含 2026-09-16 的 v1-readiness 记录）。
> 本文档回答一个问题：**同为先 vLLM 插件，他们做了什么、我们没做**。
> 结论先行：工程护栏（诊断、契约、验证记录、基线锁定）曾是我们最大的
> 差距面，本文落地时已补齐可在本仓内完成的部分；**int8 端到端的宿主
> 阻塞已定位到行**（fa_quant 分配切分不对称），宿主补丁提案与插件侧
> 守卫均已就绪，剩余动作是在 NPU 容器上复验（§4）。

## 1. 定位差异（对比的前提）

两个项目都是通过 `vllm.general_plugins` 加载、默认 no-op、fail-closed
的 vLLM 插件，但量化对象不同，**不要**把参考项目当同类抄：

| | 参考项目 | 本项目 |
|---|---|---|
| 量化对象 | **线性层权重** W8A8（离线转换 + 运行时注册别名） | **KV cache**（int8/int4/fp8/fp4，运行时动态量化，无离线权重转换） |
| 包结构 | 双包：离线 Toolkit + 运行时 extension（各自独立 wheel） | 单包：方法库 + manifest bundle（Extension Manager 路线 `import_only`） |
| 宿主面 | vllm-hust v1 / vllm-ascend-hust **冻结 commit** | 两个宿主的**外部挂载**（不冻 commit，靠适配器镜像宿主表面） |
| 已验证深度 | W8A8 `npu_e2e`（Qwen2.5-14B，contract 1.1） | int8 分发/类手术/浮点前向真机通过；int8 存储路径阻塞于宿主 |
| KV cache | 明确声明"不是 KV-cache 插件" | 就是 KV-cache 插件（含 Adaptive Quantized KV 之外的独立路线） |

因此对比落在**插件工程化的公共面**：准入/诊断、工件契约、宿主基线、
证据纪律、发布工程。

## 2. 能力对照总表

图例：✅ 已有｜🆕 本次补齐｜⚠️ 部分具备｜❌ 缺失（含归属说明）。

| 能力 | 参考项目 | 本项目 | 状态 |
|---|---|---|---|
| 默认 no-op 的 `vllm.general_plugins` 钩子 | ✅ ENABLE+ARTIFACT 门 | ✅ `VLLM_HUST_KV_METHODS` 门 | ✅ |
| 幂等注册 / 冲突 fail-closed | ✅ | ✅（同名类等价 no-op，真冲突抛错） | ✅ |
| 环境诊断 doctor CLI | ✅ `ascend-quant-toolkit doctor` | ❌ | 🆕 `vllm-hust-kv-doctor` |
| serve 前 artifact/checkpoint 校验 CLI | ✅ `check --model`（契约+张量面） | ❌ | 🆕 `vllm-hust-kv-inject --check`（config 契约面） |
| 工件完整性契约（size/SHA-256 绑定、artifact_id、schema 版本化） | ✅ contract v1.1 + 2 份 JSON Schema | ❌（注入只留 .bak） | 🆕 `kv_inject_manifest.json` + `kv-evidence-v1.schema.json`（config 级最小闭环） |
| 回滚 / 恢复 | ✅ `restore-modelslim` | ✅ `--restore`（.bak） | ✅（🆕 回滚时清理清单） |
| 宿主基线冻结（精确 commit）+ 可执行锁定 | ✅ 冻结 v1/74f0c0a27 + `verify_host_sources.py` 入 CI | ⚠️ 只在文档记容器，无可执行锁定 | 🆕 `scripts/verify_host_sources.py`（静态表面核查 + 已知阻塞点名）；**commit 级锁定仍未做** |
| 验收矩阵 + 推广门 + 负向门 | ✅ 四级门 + 负向门清单 | ⚠️ 只有 how-to-run §8 口径矩阵 | 🆕 `docs/acceptance-matrix.md`（六级门 + 负向门逐条挂测试） |
| 带日期的验证记录 | ✅ 5 份 validation-*.md | ⚠️ 口径混在 how-to-run §8.1 | 🆕 `docs/validation-int8-20260912.md`（+2026-09-17 根因附录） |
| evidence JSON schema + 校验命令 | ✅ evidence-v1 + `validate-evidence` | ❌ | 🆕 `contracts/kv-evidence-v1.schema.json` + `vllm-hust-kv-evidence` |
| 发布工程（wheel 校验、隔离安装、发布流） | ✅ `verify_release.py` + OIDC publish | ✅ `verify-wheel.sh` + 隔离 smoke + release.yml | ✅ 基本对齐 |
| CHANGELOG / 发布清单 | ✅ | ❌ / ⚠️（打包指南非清单） | 🆕 CHANGELOG.md + `docs/release-checklist.md` |
| 设计文档（ADR / roadmap / 宿主现状分析） | ✅ ADR+roadmap+现状分析 | ⚠️ architecture/integration/survey 很全，缺 ADR | 🆕 `docs/adr/0001`（激活通路决策）；roadmap 仍缺（低优先） |
| 语义层 CPU 可测 | ✅（张量面校验） | ✅ 六方法语义层单测 | ✅ |
| 多方法/多 dtype 面积 | 单一 W8A8（W4 声明未验证） | 六方法（int8/int4/fp8/fp4） | ✅（面积优势） |

## 3. 本次补齐了什么（对应上面的 🆕）

1. **`vllm-hust-kv-doctor`**（`tools/doctor.py`）：对标 doctor/check 习惯
   的只读诊断。四层就绪门（semantics / npu_kernels / host_serving /
   model）对应 how-to-run 的 L2–L4 + serve 前最后一道；`--model` 复用
   注入契约校验；`--json` 供脚本消费。探测只用
   importlib.metadata/find_spec，绝不 import 重栈。
2. **注入契约清单 + `--check`**（`tools/checkpoint.py`）：注入落
   `kv_inject_manifest.json`（方法、fa_quant_type、层数、config.json
   size+SHA-256 绑定）；serve 前校验"清单在、哈希合、描述一致"三件事；
   `--restore` 一并清清单。这是参考项目 contract 思路在 KV 注入场景的
   最小完整闭环（config.json 是注入工具触碰的唯一文件）。
3. **`scripts/verify_host_sources.py`**：把"宿主表面存在"从文档口径变
   成可执行检查——registry/基类/ModelSlim 解析（fa_quant_type +
   kvcache_quant_layers）/general_plugins 加载器/CUSTOM 槽位/CacheDType
   字面量；并把 int8 已知阻塞（`_reshape_kv_cache_tensors`）作为
   review 级点名，确保不被遗忘。
4. **验收矩阵 + 验证记录**：`docs/acceptance-matrix.md`（六级推广门、
   方法状态矩阵、负向门逐条挂测试）与
   `docs/validation-int8-20260912.md`（container-86 真机记录，含冻结
   输入与剩余缺口）。
5. **CHANGELOG.md** + 一处测试加固：`test_unknown_method_fails_closed`
   与宿主环境解耦（原先依赖"机器上恰好装了 vllm"，顺序脆弱）。
6. **int8 阻塞定位到行并给出修复**（2026-09-17 追加）：
   - 根因：宿主 `AscendModelSlimConfig.get_kv_quant_split_factor` 对
     稠密层 V×2 的 legacy 字节预算 vs int8/int8 存储 + 满头重排
     （逐行算术见 `provenance/host-fixes/README.md`）；
   - 宿主补丁提案（`git apply --check` 通过）+ 插件侧 opt-in 守卫
     `VLLM_HUST_KV_ALLOC_GUARD`（`tests/test_alloc_guard.py` 验证调和
     语义），详见 how-to-run §6.5；
   - `verify_host_sources.py` 升级为全量命中、生产代码优先（真实
     checkout 上首命中常落在 tests/_310p，已修正误导）。
7. **evidence 体系**：`contracts/kv-evidence-v1.schema.json` + 零依赖
   校验器 `vllm-hust-kv-evidence`（未知字段拒绝、越级声明拒绝）+
   示例记录 `evidence/example-int8-v1.json`。
8. **发布清单** `docs/release-checklist.md` 与 **ADR 体例**
   `docs/adr/0001`（激活通路决策）。

## 4. 仍未做（诚实清单，按阻塞面分组）

### 本仓可做、待排期

- **commit 级宿主基线锁定**：本工具现在做"表面在不在"的静态核查；
  参考项目还锁精确 commit + 父链并入 CI。等宿主侧 int8 修复落地、
  出现第一个可冻结的宿主 commit 后再锁（现在锁会立刻过期）。
- **roadmap 文档**：HOST_CONTRACT + §5 路线已有内容，缺独立 roadmap
  页（低优先）。

### 阻塞在宿主/NPU 侧（本仓已把可做的做完）

- **int8 存储路径 `npu_e2e`**：根因已定位到宿主
  `get_kv_quant_split_factor` 的 V×2 遗留口径（逐行算术 +
  `git apply --check` 通过的宿主补丁提案 +
  CPU 单测通过的插件侧守卫，全见 `provenance/host-fixes/README.md`
  与 how-to-run §6.5）。**剩余动作是在 NPU 容器上复验**：宿主合补丁
  或 serve 进程 `VLLM_HUST_KV_ALLOC_GUARD=1`，按 §6 配方跑通后落
  新验证记录。没有 NPU，这一步在本仓环境无法完成。
- **`vllm.general_plugins` 触发时机**：V1 in-proc 引擎在模型加载前不
  触发钩子；`vllm serve` 完整入口需宿主侧确认（参考项目同样记录了
  多进程可重入约束，属于共同议题）。

### 流程性缺口（依赖真机数据）

- **`matched_benchmark` 门**：int8 的 PPL / ShareGPT / HBM 与 BF16
  同协议对照，当前零数据。参考项目自己也只到 `npu_e2e`（单剖面数据
  有、BF16 对照缺），我们不落后，但第一版本声明"功能跑通"时应避免
  任何性能口径。evidence schema 已就位，数据落地即可归档。

## 5. 建议的下一步（排序）

1. **NPU 复验**（第一版本最后一公里）：容器上先跑
   `VLLM_HUST_KV_ALLOC_GUARD=1` 的 serve 配方（无需改宿主）；同场
   对照宿主补丁版；结果写新验证记录 + evidence JSON。
2. 复验通过后：把宿主补丁提案提进宿主仓 → 冻结宿主 commit →
   `verify_host_sources` 并入 CI → 按 `docs/release-checklist.md`
   打 tag。
3. 随首份性能数据走 `vllm-hust-kv-evidence` 归档；`matched_benchmark`
   门推进。
4. `vllm serve` 入口的钩子触发时机与宿主确认（或把 opt-in 触发点
   前移的方案提交 HOST_CONTRACT 讨论）。

## 6. 参考项目值得继续借鉴的两个细节

- **发布前 readiness 清单里的"负向声明"习惯**：明确写"未验证的
  profile 不得宣称为硬件验证结果"——我们的验收矩阵已采纳同款纪律。
- **`VLLM_PLUGINS` 不要收窄的运维提示**：宿主 platform 插件走同一
  加载器，allowlist 会误伤——已补进 how-to-run §6.1 的提示行。
