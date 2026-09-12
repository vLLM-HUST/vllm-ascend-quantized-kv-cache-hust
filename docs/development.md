# 开发指南（Development）

本文面向要改这个库的贡献者：仓库结构、必须遵守的分层纪律、怎么加
新方法、测试策略、构建与 CI。发布流程见
[packaging-and-release.md](packaging-and-release.md)。

## 1. 环境与常用命令

```bash
python -m pip install -e ".[test]"     # pytest + ruff + vllm-hust-ext

pytest -q                              # 全量单测（CPU，无需 NPU）
ruff check .                           # lint
ruff format --check .                  # 格式
python -m build                        # 构建 wheel + sdist（纯 Python）
bash scripts/verify-wheel.sh dist/*.whl  # wheel 内容/清单/版本校验
```

CI（`.github/extension-ci.yml`，push main / PR 触发）：Python 3.10 /
3.12 / 3.14 矩阵，依次跑 `ruff check` → `ruff format --check` →
`pytest -q` → `python -m build` → `verify-wheel.sh` → 隔离 venv 冒烟
安装 → `vllm-hust-ext extension inspect` 发现性检查。**CI 没有 NPU
阶段**——设备验证在 910B2 容器上手工跑（见 §6）。

## 2. 仓库结构

```
src/vllm_ascend_quantized_kv_cache/
├── dtypes.py              # 层0 契约：dtype→mode→布局（全库"宪法"，零依赖）
├── core/
│   ├── hosts.py           # 宿主名单 + detect_host/require_host_stack
│   ├── activation.py      # 统一激活管线（kv_methods.activate 与 bootstrap 共用）
│   └── runtime.py         # NPU/导入能力探测 + 惰性 import 助手
├── methods/
│   ├── base.py            # MethodSpec/MethodConfig/KvQuantMethod（零依赖）
│   ├── registry.py        # fail-closed 注册表（只写元数据）
│   ├── int8_dynamic/      # __init__ 注册 + semantics.py + attention_mixin.py
│   ├── kivi_int4/         # 同上三件套
│   ├── packed_base.py     # PackedFormatSemantics 基类 + 分发表
│   └── int4_packed.py / fp4_e2m1.py / fp8_e4m3.py / nvfp4.py  # 格式方法
├── ops/
│   ├── int8_ops.py        # INT8 gather+反量化（纯 torch）
│   ├── kivi_layout.py     # KIVI 共享布局校验器（纯 torch）
│   ├── kivi_gather.py     # 实际路由的 torch gather
│   └── triton/
│       ├── kivi_pack.py                # triton-ascend pack（在线路径）
│       └── kivi_gather_experimental.py # 融合 gather（保留不路由）
├── adapters/
│   ├── base.py            # HostAdapter（宿主 import 只在方法内）
│   ├── vllm_ascend_hust/  # register / scheme / attention（impl 手术）
│   └── vllm_hust/         # register（dtype 协商）/ backend（惰性构建）
├── bootstrap.py           # vllm.general_plugins 钩子（默认 no-op）
├── manifests/             # Extension Manager 清单（import_only）
└── _version.py            # 唯一版本源
```

## 3. 分层纪律（改代码前必读）

五层严格按依赖序：`contracts → core → methods → ops → adapters →
bootstrap`。五条硬规则，前四条违反会被 `tests/test_facade.py` 的子进程
测试抓住：

1. **导入卫生**：import 本包不得拉起 torch / vllm / triton /
   vllm_ascend。所有重依赖在使用点函数体内惰性导入（经
   `core.runtime` 的 `import_*` 助手，错误信息带调用语境）。
2. **注册即元数据**：方法的 `__init__.py` 只构造 `MethodSpec` 并
   `register_method`；torch/vllm/宿主全部通过 *loader 字段* 引用
   （`semantics_loader`、`adapter_factories`），注册路径永不触发。
3. **设备代码只进 ops/ 与 mixin 方法体**：缺 NPU 栈时产生一条精确
   错误，不是 import 级联。
4. **适配器 bind 风格**：宿主基类作为参数传入
   （`build_impl_cls(name, base_cls)`）或在 `register()` 方法体内
   import；适配器模块顶层绝不 import 宿主。
5. **激活入口唯一**：把方法点亮进宿主的逻辑只存在于
   `core/activation.py::activate`；`kv_methods.activate` 与
   `bootstrap.register_plugins` 都只是它的薄封装。加新的激活形态
   （新的 CLI、新的宿主类型）时扩展管线或适配器，不要另起炉灶。

其余约定：

- **fail-closed 风格全库一致**：未知名/非法配置/不支持组合一律
  `ValueError`/`RuntimeError` 并给出清单或上下文，绝不静默回退。
  写新代码遇到"要不要猜一个默认"的时候，答案永远是抛错。
- **出处分明**：从 legacy 补丁移植的代码在模块 docstring 标注
  `ascend-pr-XXX/NNNN`；新逻辑不冒充 legacy。
- ruff 规则集 `B,E,F,I,SIM,UP`（`pyproject.toml`）；逐字保留的
  legacy triton 内核与 NPU 诊断脚本豁免 E501（见 per-file-ignores，
  别顺手"修"它们的行长）。

## 4. 如何新增一个方法

以假想的 `my_quant` 为例（现有例子对照
`methods/int8_dynamic/__init__.py`）：

1. **注册**：`methods/my_quant/__init__.py` 构造 `MethodSpec`
   （name、dtype 契约键、summary、provenance、quant_mode、支持宿主、
   config_validator、semantics_loader、adapter_factories），import
   即注册。
2. **契约**：若引入新 dtype 字符串，在 `dtypes.py` 的
   `get_kv_quant_mode` 精确表加键、`KVQuantMode` 加枚举、
   `resolve_layout` 加分支（storage dtype + 打包维度公式），并在
   `tests/test_dtypes.py` 补用例——包括非法 head_size 的抛错路径。
3. **语义**：`methods/my_quant/semantics.py` 纯 torch 数学 +
   `describe()`；在 `tests/` 加 CPU 对拍测试（参考 §5 的测试策略）。
4. **设备路径**（如需要）：`ops/` 加内核或写 attention mixin（状态
   初始化遵循 `_init_<name>_state(kv_cache_dtype, vllm_config)` 约定，
   并把它登记进 `adapters/vllm_ascend_hust/attention.py::_MIXINS`）。
   NPU 验证脚本按 `scripts/npu_smoke_*.py` 的模式写（对拍语义层参考，
   结尾打印 `RESULT: PASS/FAIL` 并以退出码区分）。
5. **宿主接线**：格式方法（packed）→ 在 `methods/packed_base.py` 的
   分发表（`FORMAT_SEMANTICS` / `METHOD_NAME_TO_SEMANTICS`）登记语义
   类（或仿 `PackedFormatSemantics` 写新基类）；vllm-hust 宿主 →
   在 `adapters/vllm_hust/register.py::DTYPE_LITERAL_MAP` 协商一个
   既有 CacheDType 字面量（没有合适字面量就保持 fail-closed）。
6. **记录出处**：spec 的 `provenance` 字段 + `PROVENANCE.md` 的
   mining map 各记一笔（源补丁、提取了什么、刻意留下什么）。
7. `pytest -q && ruff check . && ruff format --check .` 全绿后再提 PR。

## 5. 测试策略（没有 NPU 怎么开发）

- **语义层**：CPU 张量直接测量化数学；KIVI 语义同时是 NPU 内核的
  数值参考，所以 CPU 测试即内核对拍基准。
- **适配器层**：stub 基类镜像宿主表面（`AscendAttentionScheme` /
  `AttentionBackend` 等），内核 launch 钩子 stub 掉——验证逻辑
  （校验、登记、幂等、fail-closed）在无 triton/无宿主环境下全覆盖。
- **导入卫生**：`test_facade.py` 在干净子进程 import 本包并断言
  `sys.modules` 干净。
- **bootstrap / 激活管线**：`test_bootstrap.py` 覆盖默认 no-op、变量
  解析、宿主探测、未知名 fail-closed；`test_activation.py` 覆盖统一
  激活管线（`kv_methods.activate` / `core.activation.activate`）的
  fail-closed 矩阵与幂等语义，`test_adapters.py` 用桩注册表覆盖
  适配器幂等/冲突。
- **设备数值**：不属于 pytest——按 §1 的 NPU 脚本在 910B2 上跑，
  逐位对拍 CPU 参考（方法见
  [npu-implementation.md](npu-implementation.md) §6）。
  端到端 serving 属宿主集成 CI 阶段，本仓库 CI 不覆盖。

## 6. 版本与发布纪律（摘要）

- 唯一版本源 `src/vllm_ascend_quantized_kv_cache/_version.py`；
  hatchling 动态读取；manifest 的 `extension_version` 必须同值
  （测试强制）。**任何代码改动 = 版本号递增**（PyPI 不可覆盖）。
- 发布走 tag 触发的 `.github/release.yml`：build → verify-wheel →
  uv publish → PyPI 无缓存冒烟安装；首个发布版本需 maintainer 批准。
- 完整清单见 [packaging-and-release.md](packaging-and-release.md)。

## 7. 提交前自查

- [ ] 新代码遵守 §3 四条分层纪律（尤其：没有新的顶层重导入）
- [ ] `pytest -q` 全绿；`ruff check . && ruff format --check .` 干净
- [ ] 新方法：契约/语义/适配器/出处四项齐全（§4）
- [ ] 动过 legacy 逐字移植的文件（`ops/triton/kivi_pack.py`）？
      确认改动确有必要并在 PR 说明；否则保持逐字
- [ ] 涉及设备路径：910B2 上跑过对应 `scripts/npu_smoke_*.py` 并 PASS
- [ ] 代码改动已递增 `_version.py`
