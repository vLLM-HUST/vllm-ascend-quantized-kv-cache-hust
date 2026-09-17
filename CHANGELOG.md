# Changelog

本项目的可见变更都记录在这里。
格式遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)；
版本号遵循语义化版本（当前 0.x，开发期）。

## [Unreleased]

### Added

- `vllm-hust-kv-doctor`：只读环境诊断 CLI（torch / torch_npu / triton /
  vllm / vllm-ascend 探测、宿主检测、方法清单、磁盘、`--model` 注入契约
  校验、`--require` 就绪门与 `--json` 输出）——对标离线量化工具链的
  `doctor`/`check` 习惯（`src/vllm_ascend_quantized_kv_cache/tools/doctor.py`）。
- `vllm-hust-kv-inject --check` 与注入契约清单
  `kv_inject_manifest.json`：注入时把方法、fa_quant_type、层数与
  config.json 的 size/SHA-256 绑定落盘；serve 前校验 checkpoint 未漂移；
  `--restore` 一并清理清单。
- `scripts/verify_host_sources.py`：静态核查宿主 checkout 里插件注册
  所依赖的表面（register_scheme / AscendAttentionScheme / fa_quant_type
  解析 / general_plugins 加载器 / AttentionBackendEnum.CUSTOM /
  CacheDType 字面量），并点名 int8 存储路径的已知宿主阻塞；多命中
  全量报告、生产代码优先于 tests/_310p。
- **int8 存储路径修复包**（根因 + 双修复，均默认不改变行为）：
  - 根因定位：宿主 fa_quant 分配切分 V×2 遗留口径 vs int8/int8 存储
    （逐行算术见 `provenance/host-fixes/README.md`）；
  - 宿主补丁提案
    `provenance/host-fixes/vllm-ascend-hust-0001-kv-split-factor-symmetric-dense.patch`
    （对 b0613602f 工作树 `git apply --check` 通过）；
  - 插件侧守卫 `VLLM_HUST_KV_ALLOC_GUARD=1`（opt-in；对称维度强制
    对称切分，MLA 不触碰，宿主修复后自动 no-op；
    `tests/test_alloc_guard.py`）。
- evidence 体系：`contracts/kv-evidence-v1.schema.json` + 零依赖校验
  CLI `vllm-hust-kv-evidence`（未知字段拒绝、`matched_benchmark` 越级
  声明拒绝）+ 示例记录 `evidence/example-int8-v1.json`。
- `docs/acceptance-matrix.md`：验收与证据矩阵（推广门 + 负向门）。
- `docs/validation-int8-20260912.md`：int8 真机验证记录（container-86）
  + 2026-09-17 根因定位与修复附录。
- `docs/gap-analysis-vs-ascend-llm-quant.md`：对照
  [Ascend-LLM-quant](https://github.com/jxd1111/Ascend-LLM-quant)
  的差距分析。
- `docs/release-checklist.md` 发布清单与 `docs/adr/0001` 激活通路 ADR。
- `CHANGELOG.md`（本文件）。

### Fixed

- `tests/test_bootstrap.py::test_unknown_method_fails_closed` 与宿主环境
  解耦：无 vllm 栈的机器上也能走到"未知方法"分支（原先依赖
  `detect_host()` 恰好命中宿主，顺序脆弱）。

## [0.1.x] / 初始开发线（tag 前）

以 git 历史为准（`git log --oneline`）：

- `f9b2d5a` feat: init the plugins —— 六个量化方法（int8_dynamic /
  kivi_int4 / int4_packed / fp4_e2m1 / fp8_e4m3 / nvfp4）语义层、统一
  API、双宿主适配器、Extension Manager manifest。
- `eced7ac` feat!: rename solutions to methods and unify the activation
  pipeline —— 术语统一 + 统一激活管线（`kv_methods.activate`）。
- `8fe5fae` fix: harden host adapter surfaces verified by a real vLLM
  engine on 910B2 —— 真机（container-86）暴露的适配器表面加固。
- `17432de` feat: add vllm-hust-kv-inject and unify ascend scheme key
  derivation —— checkpoint 注入工具与注册键推导统一。
