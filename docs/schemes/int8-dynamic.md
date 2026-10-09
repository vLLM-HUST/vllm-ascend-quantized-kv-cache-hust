# int8_dynamic — 每通道动态 INT8

一句话：K/V 存成 int8，省一半显存；scale 按通道算，第一次 prefill
算一次之后不变。

**状态（feat/int4）**：代码全有，`--kv-cache-dtype int8` 可用。
选实现这一步在真宿主上验过（validation-int4-20260920.md §11），
但没在引擎里端到端跑过。

## 怎么量化

- 第一次 prefill 沿 token 维取绝对值最大值，每个 (kv_head, head_dim)
  通道得到一个 scale，`inv_scale = 127/amax`，对称量化：
  `clamp(round(x·inv_scale), -128, 127)`。
- scale 之后固定不再更新——这是它最大的短板，服务时间长了
  分布漂移就吃亏。fp8_per_token_head 就是冲着这个短板来的。
- 不打包，真 int8 存储。

## 怎么算注意力

- decode：BNSD 布局直接读分页 int8 缓存，scale 交给
  `npu_fused_infer_attention_score` 在线 antiquant（算子边读边解压）。
- prefill / chunked-prefill：TND 布局；缓存命中的部分先 gather
  解压成稠密再算。

细节见 [npu-implementation.md](../npu-implementation.md) §3.2。

## 限制

- head_size 必须是 8 的倍数。

## 代码和测试

- 量化数学：`methods/int8_dynamic/semantics.py`
- 三条前向分支：`methods/int8_dynamic/attention_backend.py`
- 测试：`tests/test_int8_dynamic.py` + `test_plugin.py` 公共部分
- 出处：ascend#116/0001（见 [../PROVENANCE.md](../../PROVENANCE.md)）
