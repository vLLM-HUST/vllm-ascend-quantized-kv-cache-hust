# 0001：中间态激活走进程环境变量 opt-in

日期：2026-09-17 ｜ 状态：已接受（宿主四协议落地后复审）

## 背景

本库有两种可能的激活通路：

1. **Extension Manager enable**：manifest 在 manager 目录里声明组件，
   由 manager 决定哪个进程点亮哪个方法。这条路要求宿主先落地
   HOST_CONTRACT.md 的四个协议（dtype 注册表、layout 协商、attention
   能力查询、kv-transfer 布局握手）；
2. **进程环境变量 opt-in**：serve 进程设
   `VLLM_HUST_KV_METHODS=int8_dynamic,...`，`vllm.general_plugins`
   钩子把具名方法注册进"当前进程可导入的那个宿主"。

四协议尚未在 vllm-hust / vllm-ascend-hust 落地，而第一版本需要 int8
流程真实跑通。manifest 因此保持 `import_only`（manager 能 inspect、
必须拒绝 enable）。

## 决策

中间态采用**环境变量 opt-in**作为唯一激活通路：

- 安装绝不改变行为：钩子默认 no-op（有测试钉住）；
- 激活按进程显式声明，未知方法/缺宿主 fail-closed，绝不静默回退；
- 同一纪律延伸到可选的宿主行为调和：`VLLM_HUST_KV_ALLOC_GUARD`
  （int8 存储路径的分配守卫，见
  `provenance/host-fixes/README.md`）同样默认关闭、显式 opt-in。

## 后果

- ✅ 零宿主改动，今天就能在真实宿主上注册与分发（container-86 已验证）；
- ✅ 激活意图显式可审计（进程环境即声明），不会像 manager 静默 enable；
- ⚠️ 每进程都要显式传环境变量（systemd/k8s 需注入），没有"装了就能用"；
- ⚠️ `vllm.general_plugins` 在 V1 in-proc 引擎的模型加载前不触发——
  触发时机是宿主协议议题（how-to-run.md §8.1 待办 2）；
- 协议落地后复审：把 opt-in 换成 manager enable，本 ADR 由新决策
  supersede。
