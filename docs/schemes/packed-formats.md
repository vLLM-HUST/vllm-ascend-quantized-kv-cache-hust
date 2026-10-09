# 四个纯格式方案（只在 dev 分支上）

`int4_packed` / `fp4_e2m1` / `fp8_e4m3` / `nvfp4`——只声明"存成什么
格式、scale 放哪"，量化计算由宿主 attention 内核做，插件自己不算。

> **注意**：这四个的代码在 **dev 分支**（那时包还叫 `sol.*`）。
> 现在 feat/int4 分支没有它们：`src/` 下搜不到
> `PackedFormatSemantics`，`dtypes.py` 也不认这四个 dtype。
> 这页是格式设计的备忘。

## 四个格式

| 方案 | 存储 | scale | 压缩 |
|---|---|---|---|
| `int4_packed` | uint8，2×int4/字节 | per-token-head，layer 上挂 k/v 参数 | 4x |
| `fp8_e4m3` | float8_e4m3fn | per-tensor，静态（checkpoint）或动态 | 2x |
| `fp4_e2m1`（MXFP4） | uint8，每 16 元素共享 1 个 fp8 指数 scale | 随数据走 | ~3.6x |
| `nvfp4` | uint8，fp4 数据 + 每 16 元素 1 个 fp8 scale | 随数据走 | ~3.6x |

`fp4_e2m1` 和 `nvfp4` 的区别：块 scale 的编码不同，打包布局也不同
——fp4_e2m1 每块 `8B 数据 + 1B scale` 交错；nvfp4 是数据一半 +
scale 一半分开。

## 共同结构

每个格式一个语义类：四个元数据（名字 / dtype / 存储 torch dtype /
带不带 scale）加三个行为，其中 `apply` **永远抛 RuntimeError**——
量化应该发生在 attention 内核里，这个方法被调到说明接线接错了。
注册键带 `VLLM_HUST_KV_` 前缀，防止和宿主自带的 scheme 撞名。

`fp4_e2m1` 在 vllm-hust 上没有对应的 dtype 字面量，选了直接被拒。

出处：ascend#160/0001（见 [../PROVENANCE.md](../../PROVENANCE.md)）。
