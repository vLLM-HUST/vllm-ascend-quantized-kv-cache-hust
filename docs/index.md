# Documentation

- [Architecture](architecture.md)

量化 KV cache 方案（一览：[schemes.md](schemes.md)）：

- [int8_dynamic — 每通道动态 INT8](schemes/int8-dynamic.md)
- [kivi_int4 — KIVI：近期保精度，历史压 int4](schemes/kivi-int4.md)
- [fp8_per_token_head — 每 (token, head) 动态 E4M3](schemes/fp8-per-token-head.md)
- [四个纯格式方案（dev 分支）](schemes/packed-formats.md)

- [行业 KV 量化方案调研](industry-survey.md)
- [除 INT8 / INT4 之外的 KV 量化方案调研](kvquant-schemes-beyond-int8-int4.md)
- [INT4 (KIVI) 宿主接合清单](int4-host-integration.md)
- [Ascend 910B 运行方法](how-to-run.md)
- [INT4 910B2 验证记录（2026-09-20）](validation-int4-20260920.md)
- [打包与发布](packaging-and-release.md)
- [大包发布与打榜](release-and-leaderboard.md)
- [打榜操作记录（swe-prefix 实测全流程）](swe-prefix-benchmark-playbook.md)
