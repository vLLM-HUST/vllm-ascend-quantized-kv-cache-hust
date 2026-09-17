# 文档导航

`vllm-ascend-quantized-kv-cache` 是 vLLM-HUST 技术栈的**可插拔量化 KV cache
方法库**：把 provenance 保留的 legacy 实现（INT8 动态 per-channel、KIVI
INT4、int4 / fp4_e2m1 / fp8_e4m3 / nvfp4 packed handler）挖掘成**独立、
自注册的方法模块**，放在一个统一 API 后面，并为 **vllm-hust** 与
**vllm-ascend-hust** 两个宿主提供适配器。

30 秒了解本项目：

- **它解决什么问题**：推理时 KV cache 显存占用随上下文长度线性膨胀；
  本库把 K/V 量化成 int8 / int4 / fp8 / fp4 存进分页缓存，用 2x–4.5x
  的显存节省换取可控的精度损失。
- **它不是什么**：不是离线模型权重量化，也不是 Adaptive Quantized KV
  observer 项目；内核层（triton-ascend / torch_npu）**只在 Ascend NPU
  上执行**，纯语义层（数学）在任何机器上可跑可测，设备路径在非 NPU
  环境 fail-closed。
- **怎么用**：`pip install` 之后默认**零行为变化**；通过
  `VLLM_HUST_KV_METHODS` 环境变量按进程显式激活（见
  [how-to-run.md](how-to-run.md)）。

## 文档地图

| 文档 | 内容 | 适合谁 |
|---|---|---|
| [schemes.md](schemes.md) | 模块的作用与含义；六个量化方法的语义、布局、差异与选型建议 | 所有人，先读这篇 |
| [how-to-run.md](how-to-run.md) | **How to run**：CPU 测试、Python API 使用、NPU 冒烟、在两个宿主上跑起来、故障排查 | 想跑起来的人 |
| [acceptance-matrix.md](acceptance-matrix.md) | 验收与证据矩阵：六级推广门、方法状态、负向门（逐条挂测试） | 做验证/发版的人 |
| [validation-int8-20260912.md](validation-int8-20260912.md) | int8 真机验证记录（container-86，2026-09-12）：已证/阻塞/待办 | 做验证的人 |
| [gap-analysis-vs-ascend-llm-quant.md](gap-analysis-vs-ascend-llm-quant.md) | 对照 Ascend-LLM-quant 的差距分析：补了什么、还缺什么、下一步排序 | 所有人 |
| [release-checklist.md](release-checklist.md) | 发布清单：版本一致、默认关闭、隔离安装、证据口径 | 发版的人 |
| [adr/](adr/) | 架构决策记录（0001：中间态激活走环境变量 opt-in） | 架构参考 |
| [integration.md](integration.md) | 怎么集成进 vllm-hust / vllm-ascend-hust；Extension Manager 与 HOST_CONTRACT 路线 | 做集成/平台的人 |
| [layers.md](layers.md) | **调用层次与宿主可见性**：vllm-hust / vllm-ascend-hust 各自能调什么、两条激活链路 | 所有人 |
| [npu-implementation.md](npu-implementation.md) | NPU 实现要点：内核路由、fail-closed 守卫、C8 类手术、残差窗口状态机、已验证/已知问题 | 改内核或 attention 路径的人 |
| [development.md](development.md) | 开发指南：分层规则、新增方法步骤、测试策略、构建与发布纪律 | 贡献者 |
| [architecture.md](architecture.md) | 分层架构与"方法如何插入宿主"的设计（英文，设计基准文档） | 架构参考 |
| [packaging-and-release.md](packaging-and-release.md) | 打包、版本纪律、wheel 校验、PyPI 发布流程（英文） | 发版的人 |
| [../HOST_CONTRACT.md](../HOST_CONTRACT.md) | 宿主协议提案（dtype/layout/attention/kv-transfer 四协议） | 宿主侧路线图 |

## 最小可用示例

```python
from vllm_ascend_quantized_kv_cache import kv_methods

kv_methods.list()                            # ['fp4_e2m1', 'fp8_e4m3', 'int4_packed',
                                             #  'int8_dynamic', 'kivi_int4', 'nvfp4']
method = kv_methods.get("kivi_int4", head_size=128, block_size=128)
method.resolve_layout()   # KVCacheLayout(dtype="kivi_int4", storage_dtype="uint8",
                          #               packed_last_dim=64, quant_mode=KIVI_INT4)
method.semantics          # 纯 torch 语义对象（CPU 可跑）
kv_methods.activate("kivi_int4", host="vllm_ascend_hust")   # 插拔进宿主（统一激活管线）
```

任何未知名、非法配置、不支持的宿主/方法组合都会抛 `ValueError`
（fail-closed，绝不静默回退）。
