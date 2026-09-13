# 集成指南：怎么接入 vllm-hust / vllm-ascend-hust

本文面向"把方法库接进宿主推理栈"的集成者。核心原则一句话：

> **安装 ≠ 激活**。安装本包绝不改变宿主行为；激活要么由运维按进程
> 显式 opt-in（今天可用），要么等宿主协议落地后由 Extension Manager
> 托管（路线图）。

两个宿主都支持**零宿主仓库改动**的外部挂载：

| 宿主 | 宿主仓库 | 挂载面 | 适配器 |
|---|---|---|---|
| vllm-ascend-hust | NPU 后端仓（`vllm_ascend`） | `@register_scheme` 量化注册表 + C8 式 impl 类手术 | `adapters/vllm_ascend_hust/` |
| vllm-hust | 核心 vllm fork（`vllm`） | attention registry 的 `AttentionBackendEnum.CUSTOM` 类路径 | `adapters/vllm_hust/` |

适配器铁律：对宿主的 import 只发生在方法内部，绝不出现在模块导入期；
宿主缺失时 fail-closed 并给出精确错误（`adapters/base.py`）。

## 1. 快速接入（现有路线，今天可用）

这是**推荐路线**：不改宿主仓库、不改 manifest 状态，靠 vLLM 原生的
`vllm.general_plugins` 插件钩子 + 环境变量。完整步骤、命令、故障排查
见 [how-to-run.md](how-to-run.md) §6/§7，这里只给集成视角的摘要：

```bash
pip install vllm-ascend-quantized-kv-cache        # 装进宿主所在环境

# vllm-ascend-hust（宿主 A）
export VLLM_HUST_KV_METHODS=int8_dynamic   # 逗号分隔可多个
vllm serve <model> ...
# scheme 经 checkpoint fa_quant_type=VLLM_HUST_KV_* 分发到注意力层

# vllm-hust（宿主 B）
export VLLM_HUST_KV_METHODS=int8_dynamic
vllm serve <model> --attention-backend CUSTOM --kv-cache-dtype int8_per_token_head

# 编程式等价入口（脚本/工具里用；与上面环境变量走同一条激活管线）
kv_methods.activate("int8_dynamic", host="vllm_ascend_hust")
kv_methods.activate("int8_dynamic")   # host=None → 自动探测宿主栈
```

机制细节：

- `bootstrap.register_plugins`（entry point
  `vllm.general_plugins` → `bootstrap:register_plugins`）在每次 vLLM
  进程启动时被宿主调用；环境变量未设时**立即返回**（这就是"安装不改
  变行为"的保证，`tests/test_bootstrap.py` 覆盖）。它和
  `kv_methods.activate` 共用同一条激活管线
  （`core/activation.py`，见 [layers.md](layers.md) §4）。
- 宿主探测顺序：可 `import vllm_ascend` → 走 Ascend 适配器；否则可
  `import vllm` → 走 vllm-hust 适配器；两者都没有 → RuntimeError。
- 未知名方法 / 不支持该宿主的方法 → fail-closed（ValueError），不会
  静默跳过。
- **已在真实宿主实测**：2026-09-11 在 910B2 容器
  （vllm-ascend-hust + torch_npu 环境）上，六方法注册可见，
  `activate("int8_dynamic")` / `activate("kivi_int4")` 注册进宿主
  scheme 注册表成功且幂等（进程内注册）。**2026-09-12 serving 冒烟
  推进**：Qwen3-0.6B checkpoint 注入 `fa_quant_type` 后，真实 vLLM
  引擎内 28/28 层经 `AscendKVCacheMethod` 完成 impl 类手术，浮点
  KV 路径 `LLM.generate` 生成成功；int8 存储路径当前阻塞于宿主
  `model_runner_v1` 对 `int8_per_token_head` 的分配/重排不一致
  （复现配方与证据见 [how-to-run.md](how-to-run.md) §8.1）。

## 2. 宿主 A：vllm-ascend-hust 挂载细节

`AscendHustAdapter.register()` 做三件事：

1. **注册 scheme 类**：调用宿主
   `vllm_ascend.quantization.methods.registry.register_scheme(key,
   layer_type)`，key 是命名空间化的 `VLLM_HUST_KV_<SOLUTION>`（打包
   方法直接用 handler 的 `scheme_key`）。宿主对重复注册键会抛错——
   适配器只在"重复键指向完全相同的类"时视为幂等 no-op，否则放行
   抛错。
2. **scheme 类怎么来的**：`build_scheme_cls(method_name,
   AscendAttentionScheme)` 用 `type()` 动态生成，把宿主基类与本方法
   行为组合（packed 方法继承 handler 行为；有状态方法的
   `create_weights` 承载 impl 手术）。
3. **分发到注意力层**：宿主按 checkpoint 量化配置的 `fa_quant_type`
   键（ModelSlim 路径）为每个 attention 层选 scheme；有状态方法的
   scheme 在 `create_weights` 里执行 C8 式 `layer.impl.__class__`
   类手术并显式重初始化 mixin 状态（细节见
   [npu-implementation.md](npu-implementation.md) §4）。

集成者需要做的：保证 checkpoint 的 `fa_quant_type` 值与注册键一致
（如 `VLLM_HUST_KV_KIVI_INT4`）；KIVI 另需在宿主 cache 配置里给
`kivi_group_size` / `kivi_residual_length`。

## 3. 宿主 B：vllm-hust 挂载细节

### 3.1 后端注册

宿主 attention registry 存的是**字符串类路径**，选中后端时才惰性
import。`VllmHustAdapter.register()` 把：

```
vllm_ascend_quantized_kv_cache.adapters.vllm_hust.backend:HustQuantizedKvAttentionBackend
```

注册到 `AttentionBackendEnum.CUSTOM` 槽位。本包侧用模块级
`__getattr__` 惰性构建该类——只有 vllm 进程真正解析这个路径的那一刻
才 import `vllm.*` 并组合 mixin，其它环境零依赖。

引擎侧：`--attention-backend CUSTOM` + `--kv-cache-dtype <字面量>`；
用 `VLLM_HUST_KV_BACKEND_METHOD` 选 CUSTOM 后端服务的方法（缺省
`int8_dynamic`）。

### 3.2 CacheDType 字面量协商（本宿主的主要约束）

vllm-hust 的 `CacheDType` 是封闭 Literal，在三层各自 fail-closed
（pydantic 配置、后端选择器、torch dtype 查表）。外部包**不能也不应**
在运行时扩字面量；适配器因此做"协商"：`DTYPE_LITERAL_MAP` 把方法映射
到最近的既有字面量：

| 方法 | 协商结果 | 说明 |
|---|---|---|
| `int8_dynamic` | `int8_per_token_head` | 布局契约经该键解析出同样的 int8 存储 |
| `kivi_int4` / `int4` | `int4_per_token_head` | uint8 打包存储 |
| `fp8_e4m3` | `fp8_e4m3` | 直接同名字面量 |
| `nvfp4` | `nvfp4` | 直接同名字面量 |
| `fp4_e2m1` | **无 → ValueError** | 加字面量是宿主路线图项，不是运行时 hack |

给宿主加字面量时：同步更新本文件表格、
`adapters/vllm_hust/register.py` 的 `DTYPE_LITERAL_MAP`，并在
`tests/test_adapters.py` 补对应断言。

## 4. Extension Manager 路线（设计上被阻塞，路线图）

本包同时是 `vllm-hust-ext` 的扩展 bundle：entry point
`vllm_hust.extension_bundles` → `org.vllm-hust.quantized-kv-cache`，
wheel 内带 Manifest 0.2 描述符
（`src/vllm_ascend_quantized_kv_cache/manifests/`）。

当前状态是**刻意** `import_only`：manager 能发现、能 inspect（
`vllm-hust-ext extension inspect org.vllm-hust.quantized-kv-cache`
应显示 `activation_ready=false`），但必须拒绝激活。

翻转为 `active` 的前提（见 [../HOST_CONTRACT.md](../HOST_CONTRACT.md)
与 [architecture.md](architecture.md)）：

1. 宿主实现四协议：
   `vllm.kv-cache.dtype-registry.v1`、`vllm.kv-cache.layout.v1`、
   `vllm.attention.quantized-kv.v1`、`vllm.kv-transfer.quantized-layout.v1`；
2. adapter 介导的激活取代环境变量 opt-in；
3. 按 packaging 指南的版本验证表记录兼容性证据（最低/最新宿主版本）。

协议未落地前的中间态就是 §1 的原生路径——它 fail-closed by
construction（未知名、缺宿主、缺内核、非 NPU 全部抛错），但启用意图
存在于运维环境变量而非 manager 目录，这是明确接受的折衷。

## 5. 集成验收清单

把本包接进一个宿主环境后，按序验证：

```bash
# 1. 安装无副作用
python -c "import vllm" && pip install vllm-ascend-quantized-kv-cache
python -c "import vllm"                      # 宿主照常导入
vllm serve <model>                           # 不设环境变量：行为与装包前一致

# 2. 激活可见
VLLM_HUST_KV_METHODS=<方法> vllm serve <model>
# 日志出现 [vllm-hust-quantized-kv-cache] activated method ...

# 3. 卸载/回滚
pip uninstall vllm-ascend-quantized-kv-cache
# entry point 随包消失，vllm 不再调用本钩子：
# 残留的环境变量成为无害 no-op，行为回到装包前。
# 反向要警惕的是"环境变量引用了已不存在的方法名"（升级改名时）：
# 那会 fail-closed，ValueError 附已知方法清单。
```

升级/回滚、PyPI 发布流程见
[packaging-and-release.md](packaging-and-release.md)。
