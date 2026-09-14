# How to Run：从安装到端到端运行

本库有四个越来越"重"的运行层级，每层的前置条件与验证手段都写清楚：

| 层级 | 环境要求 | 能验证什么 | 文档章节 |
|---|---|---|---|
| L1 安装 + CPU 测试 | 任意机器，Python ≥ 3.10 | 包可导入、零重依赖、语义层/适配器全部单测 | §1–§3 |
| L2 Python API 使用 | 任意机器（语义层）；scale/量化数学需要 torch | 契约解析、语义数学、descriptor | §4 |
| L3 NPU 内核冒烟 | Ascend NPU 容器 + torch + torch_npu + triton-ascend | pack/gather 内核对拍 CPU 参考逐位通过 | §5 |
| L4 宿主 serving | L3 + vllm-ascend-hust 或 vllm-hust | 方案注册进宿主、引擎带量化缓存启动 | §6–§7 |

当前成熟度（0.2.0.dev0，诚实口径）：**L1/L2 全绿**；L3 的 KIVI
pack/gather 内核在 910B2 上**逐位验证通过**；L4 属宿主集成路线图——
注册与类路径钩子工作，完整 engine 路径未做端到端验证。见 §8。

---

## 1. 安装

### 1.1 使用方（只装包）

```bash
# 已发布时：从 PyPI
pip install vllm-ascend-quantized-kv-cache

# 或从仓库源码 / 本地 wheel
git clone https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust.git
pip install ./vllm-ascend-quantized-kv-cache-hust
# 或
pip install dist/vllm_ascend_quantized_kv_cache-*.whl
```

包是纯 Python、**零运行时依赖**。安装本身绝不改变任何 vLLM 行为
（`vllm.general_plugins` 钩子默认 no-op，激活见 §6/§7）。

### 1.2 开发者（可编辑安装 + 测试依赖）

```bash
git clone https://github.com/vLLM-HUST/vllm-ascend-quantized-kv-cache-hust.git
cd vllm-ascend-quantized-kv-cache-hust
python -m pip install -e ".[test]"   # pytest + ruff + vllm-hust-ext
```

## 2. 验证安装（30 秒冒烟）

```bash
python - <<'PY'
from vllm_ascend_quantized_kv_cache import kv_methods, __version__
print(__version__)                    # 0.2.0.dev0
print(kv_methods.list())            # ['fp4_e2m1', 'fp8_e4m3', 'int4_packed',
                                    #  'int8_dynamic', 'kivi_int4', 'nvfp4']
sol = kv_methods.get("kivi_int4", head_size=128, block_size=128)
print(sol.resolve_layout())           # storage uint8, packed 64
PY
```

也可以用 Extension Manager 检查 bundle 发现（可选，需先
`pip install "vllm-hust-ext @ git+https://github.com/vLLM-HUST/extension-manager.git@main"`）：

```bash
vllm-hust-ext extension inspect org.vllm-hust.quantized-kv-cache
# 期望：discovery 成功，activation_ready=false（import_only 是设计使然）
```

## 3. CPU 测试与静态检查（L1）

```bash
pytest -q                     # 当前基线：118 passed, 1 skipped
ruff check .
ruff format --check .
```

- skipped 的那一条是 `tests/test_manifest.py`：需要 `vllm-hust-ext`
  （装了 `.[test]` 或 CI 里才会跑）。
- `tests/test_facade.py` 会在干净子进程里断言**导入卫生**：import 本包
  后 `sys.modules` 里不得出现 torch / vllm / triton / vllm_ascend。
  如果你改出了顶层重导入，这条测试会抓住你。
- 语义层测试（`test_int8_dynamic.py` / `test_kivi_int4.py` /
  `test_format_methods.py`）在 CPU 张量上验证量化数学；适配器测试
  （`test_adapters.py`）用 stub 基类镜像宿主表面，内核 launch 钩子被
  stub 掉，不需要 NPU。

## 4. Python API 使用（L2）

统一入口是包根的 `kv_methods` 门面：

```python
from vllm_ascend_quantized_kv_cache import kv_methods

# 发现
kv_methods.list()                              # 全部方法
kv_methods.list(host="vllm_ascend_hust")       # 按宿主过滤
kv_methods.describe("kivi_int4")               # 元数据字典（不触发重导入）
kv_methods.describe_all()                      # 全部元数据

# 取一个已配置的方法句柄（构造即校验，非法配置当场抛 ValueError）
method = kv_methods.get(
    "kivi_int4",
    head_size=128, num_kv_heads=8, num_heads=32,
    block_size=128, group_size=128, residual_length=128,
)
method.descriptor        # 元数据 + 当前配置
method.resolve_layout()  # -> KVCacheLayout(dtype, storage_dtype, packed_last_dim, quant_mode)
method.supports("vllm_hust")

# 语义层：纯 torch 数学，CPU 可跑（需要 torch）
sem = method.semantics
sem.describe()          # 方法语义自描述
# 例：KIVI 键的 token 组假量化（NPU 打包内核的数值参考）
# import torch; k = torch.randn(256, 8, 128); dq = sem.fake_quant_key(k)

# 换配置（原句柄不变）
m2 = method.with_config(group_size=64, residual_length=256)

# 插拔进宿主：推荐走统一激活管线（自动探测/显式指定宿主均可）
kv_methods.activate("kivi_int4", host="vllm_ascend_hust")
kv_methods.activate("int8_dynamic")   # host=None → detect_host()（vllm_ascend 优先）

# 低层等价形式（需要逐项控制时）
ad = method.host_adapter("vllm_ascend_hust")
info = ad.register()    # {'quant_type': 'VLLM_HUST_KV_KIVI_INT4', ...}
```

fail-closed 行为一览（全部 `ValueError`/`RuntimeError`，无静默回退）：

| 操作 | 结果 |
|---|---|
| `get("不存在")` | ValueError，附全部已知方法名 |
| 违反几何约束（如 KIVI `group_size % 8`） | ValueError，指明违反的不变量 |
| `host_adapter("未知宿主")` / 不支持 / 未接线 | ValueError |
| 设备路径在无 NPU 环境被调用 | RuntimeError（`require_npu` 守卫） |
| `dtype` 字符串未注册 | `resolve_layout` 抛 ValueError |

## 5. NPU 内核冒烟与诊断（L3）

前置：Ascend 910B 容器，`torch` + `torch_npu` + `triton-ascend` 可用，
且本包已安装（或 `PYTHONPATH=src`）。三个脚本都从仓库根目录跑：

```bash
# 端到端冒烟：pack → gather → 对拍 CPU 参考，逐位校验
python scripts/npu_smoke_kivi.py
# 期望结尾：RESULT: PASS

# 融合 gather 内核重验探针（triton-ascend 修复后跑这个决定是否切回）
python scripts/npu_probe_kivi_dim.py
python scripts/npu_probe_kivi_key.py
```

背景与失败归因逻辑（pack 语义错 vs gather 错）见
[npu-implementation.md](npu-implementation.md) §3.3/§6。

## 6. 在 vllm-ascend-hust 上运行（L4，宿主 A）

**前提**：目标环境已装好 vllm-ascend-hust（可 `import vllm_ascend`），
且 NPU 可用。把本包 `pip install` 进**同一个环境**。

### 6.1 激活（每进程显式 opt-in）

```bash
# 在启动 serving 的同一 shell 里：
export VLLM_HUST_KV_METHODS=int8_dynamic   # 或 kivi_int4 / int4_packed / fp8_e4m3 / nvfp4 / fp4_e2m1
vllm serve <model> ...
```

启动日志里应看到：

```
[vllm-hust-quantized-kv-cache] activated method 'int8_dynamic' on host 'vllm_ascend_hust': VLLM_HUST_KV_INT8_DYNAMIC
```

这条路径的机制：`vllm.general_plugins` 入口钩子（`bootstrap.py`）读
环境变量 → 自动探测当前进程可导入的宿主（`vllm_ascend` 优先，其次
`vllm`）→ 把每个具名方案的 scheme 类注册进宿主
`@register_scheme` 注册表（键 `VLLM_HUST_KV_*`）。不设变量 =
完全 no-op。

### 6.2 让模型真正走量化 scheme（checkpoint 注入）

注册只解决"scheme 在注册表里"；注意力层选择哪个 scheme 由 checkpoint
的量化配置分发（ModelSlim 路径）。

**注意：只往 `config.json` 里加两个全局字段（`"quant_method": "ascend"`
+ `"fa_quant_type": "VLLM_HUST_KV_*"`）是不够的——一层都不会命中。**
宿主解析器（`AscendModelSlimConfig`）的 ModelSlim 契约是：

1. `fa_quant_type` 只是全局开关；
2. 生效层清单从逐层键 `<prefix>.layers.N.self_attn.fa_k.scale` 推导
   （`kvcache_quant_layers`）；没有逐层键就没有任何层走 FA 量化分支；
3. 完整的 ModelSlim 描述要求每个可量化模块显式声明类型，保持浮点的
   模块写 `"FLOAT"`（q/k/v/o_proj 与 mlp gate/up/down_proj 等）。

用工具一行生成完整描述（推荐）：

```bash
vllm-hust-kv-inject <model_dir> --method int8_dynamic           # 注入（自动备份）
vllm-hust-kv-inject <model_dir> --method kivi_int4 --dry-run    # 只打印不落盘
vllm-hust-kv-inject <model_dir> --restore                       # 回滚
```

等价的模块入口：`python -m
vllm_ascend_quantized_kv_cache.tools.checkpoint ...`。工具写入
`quantization_config`：`quant_method=ascend`、`fa_quant_type=<注册键>`、
每层 `fa_k.scale`/`fa_v.scale`（`FAQuant`）、全部线性层与 `embed_tokens`
的 `FLOAT`（`--include-lm-head` 可加 `lm_head`）。幂等可重跑；遇到外部
ModelSlim 描述（含本工具不生成的键）会拒绝覆盖，`--force` 才放行；
权重已量化的 checkpoint（非 ascend quant_method）一律拒绝。

手工等效配方（2 层示意，工具生成的就是这个形状）：

```json
"quantization_config": {
  "quant_method": "ascend",
  "version": "1.0.0",
  "fa_quant_type": "VLLM_HUST_KV_INT8_DYNAMIC",
  "model.embed_tokens.weight": "FLOAT",
  "model.layers.0.self_attn.fa_k.scale": "FAQuant",
  "model.layers.0.self_attn.fa_v.scale": "FAQuant",
  "model.layers.0.self_attn.q_proj.weight": "FLOAT",
  "model.layers.0.self_attn.k_proj.weight": "FLOAT",
  "model.layers.0.self_attn.v_proj.weight": "FLOAT",
  "model.layers.0.self_attn.o_proj.weight": "FLOAT",
  "model.layers.0.mlp.gate_proj.weight": "FLOAT",
  "model.layers.0.mlp.up_proj.weight": "FLOAT",
  "model.layers.0.mlp.down_proj.weight": "FLOAT"
}
```

（每层一组如上条目；`fa_quant_type` 取值即各方法的注册键：
`VLLM_HUST_KV_INT8_DYNAMIC` / `VLLM_HUST_KV_KIVI_INT4` /
`VLLM_HUST_KV_INT4` / `VLLM_HUST_KV_FP8_E4M3` / `VLLM_HUST_KV_NVFP4` /
`VLLM_HUST_KV_FP4_E2M1`。）

职责边界：checkpoint 注入只解决"分发"这一半（`fa_quant_type` →
scheme → `create_weights` 对有状态方案执行 C8 式
`layer.impl.__class__` 类手术，本库自动完成，见
[npu-implementation.md](npu-implementation.md) §4）；"注册"那一半
仍需 §6.1 的环境变量 opt-in；int8 存储路径另需 serve 期
`--kv-cache-dtype`（§6.4）。

### 6.3 KIVI 旋钮

KIVI 的组大小与残差窗口从宿主 `cache_config` 读取：
`kivi_group_size` / `kivi_residual_length`（缺省 128/128）。约束见
[schemes.md](schemes.md) §3.2，违反在 impl 初始化时即抛错。

### 6.4 impl 层的 enable 开关与 kv_cache_dtype 拼写

mixin 的 enable 标志按 impl 收到的 `kv_cache_dtype` 字符串打开：
`int8_dynamic` 接受 `int8` / `int8_per_token_head` / `int8_dynamic`
三种拼写，`kivi_int4` 接受 `kivi_int4` / `kivi`。按宿主
`--kv-cache-dtype` 支持的字面量集选择对应拼写；拼写不匹配时方案注册
成功但 enable 保持关闭（量化路径不生效），务必用 §6.1 的日志行 +
一次小流量请求验证确实走了量化路径。

## 7. 在 vllm-hust 上运行（L4，宿主 B）

**前提**：环境里可 `import vllm`（vllm-hust fork），NPU 可用，本包已
装入同一环境。

```bash
export VLLM_HUST_KV_METHODS=int8_dynamic        # 激活方法（同 §6.1）
export VLLM_HUST_KV_BACKEND_METHOD=int8_dynamic  # CUSTOM 后端服务哪个方法（缺省 int8_dynamic）

vllm serve <model> \
    --attention-backend CUSTOM \
    --kv-cache-dtype int8_per_token_head                # 方案协商出的 CacheDType 字面量
```

要点：

- 适配器把本包 backend 类路径注册到宿主
  `AttentionBackendEnum.CUSTOM` 槽位；引擎用 `--attention-backend
  CUSTOM` 选中后，宿主按字符串路径**惰性 import** 本包
  `adapters/vllm_hust/backend.py`，在那一刻才构建真正的 backend 类。
- 宿主 `CacheDType` 是封闭 Literal，方法必须协商一个既有字面量：

  | 方法 | `--kv-cache-dtype` 字面量 |
  |---|---|
  | int8_dynamic | `int8_per_token_head` |
  | kivi_int4 / int4_packed | `int4_per_token_head` |
  | fp8_e4m3 | `fp8_e4m3` |
  | nvfp4 | `nvfp4` |
  | fp4_e2m1 | **无**（fail-closed 拒绝，需宿主加字面量） |

- 设备执行仍走 Ascend NPU 内核；`get_impl_cls` 在非 NPU 环境拒绝启动。
- **成熟度提示**：此适配器是接口就绪的脚手架——注册与类路径钩子可用，
  完整 engine 路径（metadata builder → kernel → sampler 全链）属宿主
  集成路线图，本版本未验证。

## 8. 当前项目"能跑/不能跑"矩阵

| 能力 | 状态 |
|---|---|
| 安装后零行为改变 | ✅ 设计保证（bootstrap 默认 no-op，有测试） |
| CPU 语义层 / 契约 / 注册表 / 适配器逻辑 | ✅ 118 项单测全绿 |
| KIVI triton pack 内核 + torch gather（910B2） | ✅ 逐位验证通过 |
| 统一激活管线（`kv_methods.activate` → 宿主 `register_scheme`） | ✅ 910B2 容器真实 vllm-ascend-hust 宿主进程内实测通过（2026-09-11，`container-86`；六方法注册可见 + 幂等再激活 OK） |
| **fa_quant_type 分发 + impl 类手术（真实引擎）** | ✅ 2026-09-12 实测：Qwen3-0.6B + 注入 `fa_quant_type=VLLM_HUST_KV_INT8_DYNAMIC` 的 checkpoint，28/28 层经 `AscendKVCacheMethod.create_weights` 完成类手术（日志为证） |
| **真实 serving 冒烟（浮点前向路径）** | ✅ 同环境 `LLM.generate` 生成成功（enforce_eager，fp16，`kv-cache-dtype` 未设时走浮点 KV） |
| 真实 serving 冒烟（int8 存储路径） | ⚠️ 阻塞于**宿主侧**：`model_runner_v1._reshape_kv_cache_tensors` 对 `int8_per_token_head` 字面量的分配/重排不一致（初始化器分配 `[2314,128,8,64]` 打包布局，重排要求 `[1122,128,8,128]`）——见下"宿主联调工作项" |
| INT8 / KIVI attention mixin 的 NPU 前向 | ⚠️ 端口保真度；浮点路径已在真实引擎验证，int8 缓存路径待上述宿主问题解决后复验 |
| vllm-hust：CUSTOM backend 注册 + dtype 协商 | ⚠️ 脚手架就绪，端到端待宿主集成验证 |
| Extension Manager 激活（manager-mediated enable） | ❌ 设计上被阻塞（manifest `import_only`），等 HOST_CONTRACT 四协议落地，见 [integration.md](integration.md) §4 |

### 8.1 宿主联调工作项（2026-09-12 serving 冒烟结论）

在 `container-86`（vllm-ascend-hust + vllm-hust 双栈，torch_npu 环境）上的
复现配方：Qwen3-0.6B 拷贝 + `config.json` 注入
`quantization_config = {"quant_method": "ascend",
"fa_quant_type": "VLLM_HUST_KV_INT8_DYNAMIC", "model.layers.N.self_attn.fa_k.scale": …,
"<各线性层>.weight": "FLOAT"}`——即 §6.2 的完整 ModelSlim 形状，
今天直接用 `vllm-hust-kv-inject <model_dir> --method int8_dynamic` 生成；
`VLLM_HUST_KV_METHODS=int8_dynamic` + 显式
`bootstrap.register_plugins()`（V1 in-proc 引擎在模型加载前不触发
`vllm.general_plugins`，见下）。已验证通过：注册 → fa_quant_type 分发 →
scheme 实例化 → 28/28 层 impl 手术 → 引擎加载与浮点前向生成。
遗留阻塞：`kv_cache_dtype=int8_per_token_head` 时宿主
`model_runner_v1.py` 的 KV cache 初始化器分配 head/2 打包布局、
而 `_reshape_kv_cache_tensors` 按满 head int8 重排（两者不一致）。

两个附带发现（已修复或待办）：

1. 生成 scheme 构造器曾向宿主基类 `AscendAttentionScheme` 传参，而该基类
   继承 `object.__init__`（不同 fork 表面漂移）——已改为参数化转发失败时
   无参回退（`tests/test_adapters.py` 覆盖两种基类表面）。
2. 多继承动态 impl 类的 `tp_base` 落在 mixin 上，`__class__` 赋值被
   CPython 拒绝（`tp_base` 不一致）——`apply_impl_surgery` 已加
   "整实例克隆替换"回退（`tests/test_adapters.py` 覆盖）。
3. **待办**：`vllm.general_plugins` 钩子在 V1 in-proc 引擎的模型加载前
   未被触发；`vllm serve` 完整入口是否触发需在宿主侧确认，必要时把
   环境变量 opt-in 的触发点提前（宿主协议议题）。

## 9. 常见问题排查

| 症状 | 原因与处置 |
|---|---|
| `unknown quantized KV method 'xxx'; known: ...` | 方法名拼错；用报错里的清单或 `kv_methods.list()` |
| `VLLM_HUST_KV_METHODS=... requests quantized-KV methods, but neither vllm_ascend nor vllm is importable` | 环境变量设了但当前进程没有宿主栈；先装 vllm-ascend-hust / vllm-hust |
| `kivi_int4 requires group_size divisible by 8 ...` | 几何约束违反；按 [schemes.md](schemes.md) §3.2 调整 `kivi_group_size`/`kivi_residual_length`/`head_size`/`block_size` |
| `method 'fp4_e2m1' has no vllm-hust CacheDType literal mapping` | fp4_e2m1 在 vllm-hust 尚无字面量；换 nvfp4 或走 vllm-ascend-hust |
| `... requires an Ascend NPU device (torch_npu), but none is available` | 设备路径在非 NPU 环境被调用（设计行为）；语义 API 不受影响 |
| `kivi_int4 triton kernels require triton ...` | 容器里缺 triton / triton-ascend，或 vllm 的 triton shim 不可用 |
| `host adapter ... requires 'vllm_ascend', which is not importable` | 对应宿主栈不可导入（单栈环境的正常情况）；适配器守卫的是栈可导入性——双栈共存的机器上两者都可显式注册，默认宿主由 `detect_host()` 决定（vllm_ascend 优先） |
| 激活了方法但日志没有 `activated method` 行 | 环境变量没传进 serving 进程（systemd/k8s 需显式注入），或变量为空 |
