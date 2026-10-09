# kivi_int4 — KIVI：近期保精度，历史压 int4

一句话：每个请求最近 128 个 token 保持 fp16，更早的按组压成 int4，
整体约 3.6x。

**状态（feat/int4）**：做得最扎实的一个。910B2 上 13 节设备验证全部
通过（[validation-int4-20260920.md](../validation-int4-20260920.md)），
`--kv-cache-dtype kivi_int4` 可用；端到端 serve 还没跑过。

## 核心想法

键对位置敏感、值对通道敏感、新 token 更重要，所以每个请求的缓存
分两块：

- **残差窗口**：最近 `residual_length`（默认 128）个 token 保持 fp16，
  每请求一行环形缓冲；
- **历史区**：更早的 token 压成 int4——键按 token 分组（每 128 个
  一组，组内逐 (head, dim) 求 min/max，非对称量化），值按 head 维
  分组。键窗口写满一整组才写进历史区，值每挤掉一个最老槽位。

读的时候：历史区 gather 解压成稠密，拼上 fp16 残差尾，走 TND
fused attention。

## 几何要求（不满足直接报错）

```
group_size % 8 == 0          group_size 整除 residual_length
head_size % 8 == 0           group_size 整除 head_size
block_size % group_size == 0
```

默认 128/128/128 全满足；Qwen3.5-35B-A3B（head 256，每卡 1 个
KV 头）也满足，有专项测试锁着。

## 跑法约束

- 必须 `--enforce-eager`（没有 ACL Graph 捕获分支）。
- 不能开 chunked prefill：只支持"decode 行 + 全新 prompt"，带历史
  的续块会直接报错。
- 打包内核有可编译形状包络：group 128 + block 256 编不过，别这么配
  （实测表在 validation 记录 §10）。

## 压缩账（head 128 / group 128）

每 token 每头每侧 `S = head/2 + 8·head/group = 72` 字节，相对 fp16
约 **3.56x**——不是 4x，scale/min 吃掉一部分（head 256 时 S=144，
比例不变）。

## 代码和测试

- 数学与窗口记账：`methods/kivi_int4/{semantics,geometry,byte_cache}.py`
- 写入内核：`ops/triton/kivi_pack.py`（910B2 上和 CPU 参考逐位比对过）
- 读取：纯 torch 的 `ops/kivi_gather.py`（上游融合 gather 内核在
  triton-ascend 3.5 上误编译，故意不用）
- 前向状态机：`methods/kivi_int4/attention_backend.py`
- 测试：`tests/test_kivi_int4.py`（48 个）
- 宿主侧要改的四处：`scripts/host_int4_patch.py` 一键打/还原，
  说明在 [int4-host-integration.md](../int4-host-integration.md)
- 出处：ascend#116/0003–0013（见 [../PROVENANCE.md](../../PROVENANCE.md)）
