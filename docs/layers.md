# 调用层次与宿主可见性（vllm-hust / vllm-ascend-hust 各自能调什么）

本文是"**哪一层、谁可以调、调了会发生什么**"的权威速查。架构设计
背景见 [architecture.md](architecture.md)，方案语义见
[schemes.md](schemes.md)。

## 1. 六层一图

```
                    ┌─────────────────────────────────────────────┐
 vllm-hust 进程 ──▶ │ bootstrap (vllm.general_plugins 钩子)        │ ◀── vllm-ascend-hust 进程
                    │   VLLM_HUST_KV_METHODS=int8_dynamic,kivi…   │     （同一条钩子、同一环境变量）
                    └──────────────────┬──────────────────────────┘
                                       │ 统一激活管线 core.activation.activate
                    ┌──────────────────▼──────────────────────────┐
   任何进程 ──────▶ │ kv_methods 门面：list / get / describe /     │
                    │ activate / describe_all                      │
                    └─┬─────────────┬─────────────┬───────────────┘
                      │             │             │
       ┌──────────────▼──┐  ┌───────▼───────┐  ┌──▼──────────────────┐
       │ methods         │  │ core          │  │ adapters            │
       │ 模型+语义+mixin  │  │ hosts/runtime │  │ vllm_hust/          │
       │                 │  │ /activation   │  │ vllm_ascend_hust/   │
       └──────┬──────────┘  └───────────────┘  └──┬──────────────────┘
              │ 设备路径                           │ register() 时惰性 import 宿主
       ┌──────▼──────────┐                vllm-hust:  vllm.v1.attention…registry
       │ ops (torch /    │                vllm-ascend: vllm_ascend.quantization…registry
       │ triton-ascend)  │
       └─────────────────┘
   层0 dtypes.py（布局契约）被上面所有层引用；全库零依赖的"宪法"。
```

## 2. 宿主可见性矩阵

| 层 | 位置 | 任何进程（无宿主） | vllm-hust 进程 | vllm-ascend-hust 进程 | 说明 |
|---|---|---|---|---|---|
| 契约 | `dtypes.py` | ✅ | ✅ | ✅ | 零依赖；dtype→mode→布局唯一裁决 |
| 横切 | `core/`（hosts / runtime / activation） | ✅ | ✅ | ✅ | 宿主探测、NPU/导入守卫、统一激活管线 |
| 方法模型 | `methods/base.py`、`methods/registry.py` | ✅ | ✅ | ✅ | 元数据注册，零重导入 |
| 方法发现 | `kv_methods` 门面 | ✅ | ✅ | ✅ | `list/get/describe/activate` |
| 方法语义 | `methods/*/semantics.py` | 需 torch | ✅ | ✅ | 纯数学，CPU 可测，内核数值参考 |
| 设备 mixin | `methods/*/attention_mixin.py` | 需 torch；设备路径 fail-closed | ⚠️ 仅 CUSTOM 后端挂载后由引擎触发 | ✅ 由宿主 impl 触发（C8 类手术） | 只在 Ascend NPU 真正执行 |
| 设备内核 | `ops/`（torch）、`ops/triton/` | ❌ 不应直接调 | ❌ | ✅ 经 mixin 路径 | 库内专用 + `scripts/npu_*.py` 诊断 |
| Ascend 适配器 | `adapters/vllm_ascend_hust/` | import 安全；`register()` 抛"缺 vllm_ascend" | ❌ 单栈下 fail-closed* | ✅ `activate(..., host="vllm_ascend_hust")` | `@register_scheme` + C8 类手术 |
| vllm 适配器 | `adapters/vllm_hust/` | import 安全；`register()` 抛"缺 vllm" | ✅ `activate(..., host="vllm_hust")` | ❌ 单栈下 fail-closed* | CUSTOM 后端类路径 + CacheDType 协商 |
| 引导钩子 | `bootstrap.py` | ✅ no-op（未设环境变量时） | ✅ opt-in 后激活 | ✅ opt-in 后激活 | vLLM 经 entry point 调用的唯一钩子 |

✅ 可调　⚠️ 可导入但设备路径受限　❌ fail-closed（明确报错，不静默）

\* 适配器守卫的是**宿主栈可导入性**，不是平台二选一：双栈共存的环境
（NPU 开发机同时装了 vllm 与 vllm_ascend）里两个适配器都可显式注册，
两套注册表相互独立、互不干扰；默认宿主由 `detect_host()` 决定
（`vllm_ascend` 优先，与 NPU 部署一致）。

## 3. 两条宿主链路的调用序列

### 3.1 vllm-ascend-hust 进程（NPU 后端栈）

```
vllm serve 启动
  └─ vllm 调 vllm.general_plugins 钩子 → bootstrap.register_plugins()
       ├─ 读 VLLM_HUST_KV_METHODS（未设 → no-op 返回）
       ├─ core.hosts.detect_host() → "vllm_ascend_hust"（import vllm_ascend 可达）
       └─ 对每个方法名：core.activation.activate(name, host)
            ├─ methods.registry.get_method(name)        # 元数据+校验
            ├─ method.host_adapter("vllm_ascend_hust")  # AscendHustAdapter
            └─ adapter.register()
                 ├─ import vllm_ascend.quantization.methods.registry.register_scheme
                 ├─ build_scheme_cls(name, AscendAttentionScheme)  # 生成 scheme 类
                 └─ register_scheme("VLLM_HUST_KV_*")  # 宿主注册表
                      └─ 层命中 fa_quant_type 后：create_weights 做 C8 类手术
                         → mixin 支撑的 impl 在 NPU 上执行（ops/ 内核）
```

等价的显式编程入口（等幂）：`kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")`。

### 3.2 vllm-hust 进程（核心栈）

```
vllm serve 启动
  └─ 同一条 bootstrap 钩子 → detect_host() → "vllm_hust"（import vllm 可达）
       └─ core.activation.activate(name, host="vllm_hust")
            ├─ get_method(name)
            ├─ method.host_adapter("vllm_hust")         # VllmHustAdapter
            └─ adapter.register()
                 ├─ map_cache_dtype(name)               # CacheDType 字面量协商
                 └─ register_backend(CUSTOM, "…backend:HustQuantizedKvAttentionBackend")
                      └─ 引擎 --attention-backend CUSTOM 选中后按类路径惰性 import
                         backend.__getattr__ 构建 backend 类 → get_impl_cls 在 NPU 上执行
```

引擎侧配套：`--attention-backend CUSTOM --kv-cache-dtype <协商字面量>`；
`VLLM_HUST_KV_BACKEND_METHOD` 选 CUSTOM 后端服务的方法（缺省 int8_dynamic）。

## 4. 结构纪律（每层 `__init__` 里的契约）

每层包的 `__init__.py` docstring 写明"提供什么 + 谁可以调用"，要点：

1. **依赖只向下**：`dtypes → core → methods → ops → adapters → bootstrap`；
   上层可在函数体内延迟导入下层，绝不顶层反向依赖。
2. **重导入只在方法体内**：torch / vllm / triton / torch_npu 全部在使用
   点惰性导入；`import vllm_ascend_quantized_kv_cache` 本身零重依赖
   （`tests/test_facade.py` 子进程强制）。
3. **宿主入口唯一**：激活逻辑只存在于 `core.activation.activate`；
   `kv_methods.activate` 与 `bootstrap.register_plugins` 都是它的薄
   封装——改注册语义只改一处。
4. **适配器不互联**：`adapters/vllm_hust` 与 `adapters/vllm_ascend_hust`
   互不 import；选谁由 `core.hosts.detect_host()` 或调用方显式指定。
   `adapters.adapter_for(host)` 提供惰性查表（`vllm_ascend_hust`
   子包 import 即带 torch，故不允许被元数据路径急切导入）。

## 5. 常用调用速查

```python
from vllm_ascend_quantized_kv_cache import kv_methods, adapters

kv_methods.list()                                   # 六方法名（任何进程）
kv_methods.describe("kivi_int4")                    # 元数据（不触发重导入）
kv_methods.get("int8_dynamic", head_size=128)       # 句柄（构造即校验）

# 宿主激活（二选一：显式指定 or 自动探测）
kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")
kv_methods.activate("kivi_int4")                    # host=None → detect_host()

# 低层等价形式（适配器直接操作）
adapters.adapter_for("vllm_hust")                   # 惰性取适配器类
kv_methods.get("int8_dynamic").host_adapter("vllm_hust").register()
```

命令行（运维）等价物见 [how-to-run.md](how-to-run.md) §6–§7。
