# Provenance

当前发行只保留动态 INT8 KV-cache 路径。

| 当前代码 | 来源 | 保留内容 |
|---|---|---|
| `methods/int8_dynamic/attention_backend.py` | vLLM-Ascend-HUST `f4f49832` 中的 `AscendInt8AttentionBackendImpl` | 动态 per-channel scale、对称 INT8 量化、decode/prefill/chunked-prefill、pooling 和 ACL Graph 路径 |
| `methods/int8_dynamic/semantics.py` | legacy vllm-ascend PR #116 patch 0001 | 可脱离 NPU 测试的量化语义 |
| `adapters/vllm_ascend_hust/` | 本仓库插件化适配 | 宿主 `get_impl_cls` 的 INT8-only 分派与非 INT8 委托 |

历史补丁原文继续保存在 `provenance/legacy-patches/`，但 KIVI INT4、packed
INT4、FP4、FP8 和 NVFP4 均不属于本分支的发行代码。

`attention_backend.py` 保留了上游 Huawei Technologies 版权声明，并继续
使用 Apache-2.0。本仓库根目录的 `LICENSE` 包含完整许可证文本。
