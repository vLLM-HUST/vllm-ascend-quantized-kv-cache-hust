# 发布清单（quantized KV cache）

发版前逐项勾选；对齐 vLLM-HUST 打包指南与参考项目的发布纪律
（版本一致性、默认关闭、隔离安装、不夸大证据）。

## 版本与元数据

- [ ] `_version.py`、CHANGELOG、tag、release title 四处版本一致。
- [ ] Tag 形如 `v<version>`，指向 `main` 包含的提交。
- [ ] CHANGELOG 本版本条目齐全（Added/Changed/Fixed）。

## 行为与内容

- [ ] 安装后零行为改变：`vllm.general_plugins` 钩子默认 no-op（有测试）。
- [ ] 可选守卫默认关闭：`VLLM_HUST_KV_ALLOC_GUARD` 未设置时零效果。
- [ ] wheel 含静态 manifest（`org.vllm-hust.quantized-kv-cache`，`import_only`）。
- [ ] `bash scripts/verify-wheel.sh dist/*.whl` 通过。
- [ ] 隔离 venv 安装 + entry-point 发现 + 卸载干净（CI 已覆盖，本地复核）。
- [ ] `ruff check .` 与 `ruff format --check`（Python 文件）通过；全量
      `pytest -q` 绿（CPU 层基线）。

## 兼容与宿主

- [ ] 已验证的宿主版本/commit 记录在案（当前：真机 container-86 记录，
      见 `docs/validation-int8-20260912.md`；commit 级冻结待宿主 int8
      修复落地后建立）。
- [ ] `python scripts/verify_host_sources.py --vllm-ascend-src <checkout>
      --vllm-src <checkout>` 通过，known issues 逐条有结论。
- [ ] fail-closed 负向门全绿（docs/acceptance-matrix.md 负向门表）。

## 证据与口径

- [ ] 发布说明引用的能力都有对应门级证据；**未验证的 profile 不得写成
      硬件验证结果**。
- [ ] 新的真机结果已落 `vllm-hust-kv-evidence validate` 通过的记录 +
      带日期的 `docs/validation-*.md`。
- [ ] 性能数字只允许出现在 `matched_benchmark`（含 BF16 基线 run id）。

## 发布操作

- [ ] `python -m build`（或 uv build）产物完整；`twine check` 通过（如可用）。
- [ ] PyPI 发布走 tag 触发工作流（trusted publishing / 组织 token）。
- [ ] 发布后从 PyPI 冒烟安装并断言版本与方法清单（CI smoke 步骤）。
- [ ] PyPI 产物不可变：失败的发布以新版本修正或按策略 yank，禁止覆盖。
