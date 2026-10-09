# fp8_per_token_head — 每 (token, head) 动态 E4M3

一句话：每个 token 每个头存 head_size 字节 E4M3 + 4 字节 fp32
scale，scale 写入时现场算。head 128 时约 **1.94x**。

**状态（feat/int4，2026-10-03 起步）**：dtype 解析、字节布局、CPU
参考数学、写入内核都有了；读路径没写、没上机验过、命令行选不了
（选了报 "no adapter wired"）。

## 为什么做它

- 宿主 triton 后端本来就声明了这个 dtype，一个厂商新算子都不用等，
  是成本最低的新方案。
- scale 逐 token 更新，正好补 int8_dynamic "scale 用一次旧的"短板。
- 浮点格式天生不怕离群值（一个 (token, head) 里有离群值只影响
  自己那一份 scale）。

## 量化数学（CPU 参考，将来内核比对用）

```
scale = max(|x|) / 448，下限 1e-6（和宿主内核同一口径）
q     = e4m3(clamp(x / scale, -448, 448))
读回  = float(q) * scale
```

## 布局

和 KIVI 同一个接法：两张等大的宿主缓冲，插件各切两个视图——
每侧每 (token, head) 为 `[head_size 字节数据 | 4 字节 scale]`。
这个分法和宿主页大小公式逐位一致。head_size 要是 4 的倍数。

## 写入内核

`ops/triton/per_token_head_store.py`，照宿主的 CUDA triton 参考
一行行搬的：每 (token, head) 一个程序，absmax/448 得 scale，
clamp 后写入。int8 变体（`int8_per_token_head`）的分支也在里面，
以后顺手就能开。

上机要回答两件事：triton-ascend 支不支持往 fp8 指针写；这个内核
的形状编不编得过。

## 代码和测试

- `methods/fp8_per_token_head/{semantics,byte_cache}.py`
- `ops/triton/per_token_head_store.py`
- 测试：`tests/test_fp8_per_token_head.py`（15 个）

## 还差什么（按顺序）

读路径（注意力怎么带 per-token-head scale 读缓存）→ 910B2 上机
比对 → 接命令行。
