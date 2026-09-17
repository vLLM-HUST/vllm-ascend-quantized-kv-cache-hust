# SPDX-License-Identifier: Apache-2.0
"""环境诊断 CLI（``vllm-hust-kv-doctor``）：一次回答"这台机器能跑到哪一层"。

对标离线量化工具链的 doctor/check 习惯：只读探测、JSON 可选输出、用退出
码当就绪门（ready → 0，not ready → 2）。探测分四层，与 how-to-run.md 的
L1–L4 运行层级一一对应：

===============  ==========================================  ==========
就绪门           含义                                        依赖
===============  ==========================================  ==========
semantics        语义层可用（CPU 上跑量化数学）               torch
npu_kernels      NPU 内核冒烟可跑（L3 前置）                  torch +
                                                             torch_npu +
                                                             triton
host_serving     宿主栈可导入，scheme 注册链路可走（L4 前置） vllm_ascend
                                                             或 vllm
model            ``--model`` 指向的 checkpoint 通过注入契约   inject 清单
                 校验（serve 前最后一道）                     + 哈希一致
===============  ==========================================  ==========

探测只用 importlib.metadata / find_spec / shutil 等标准库手段，**绝不真正
import torch / vllm / 宿主栈**——诊断必须能在任何残缺环境里安全执行。
int8 端到端还有一道已知宿主侧阻塞（how-to-run.md §8.1），doctor 探不到
源码层问题；配套的 ``scripts/verify_host_sources.py`` 静态核查宿主源码。
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from .checkpoint import check_checkpoint, known_method_names

#: 探测的包名 → 汇报名。宿主发行名带 -hust 别名，两个名字取先到者。
_PACKAGE_PROBES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("torch", ("torch",)),
    ("torch-npu", ("torch-npu",)),
    ("triton", ("triton", "pytorch-triton")),
    ("vllm", ("vllm-hust", "vllm")),
    ("vllm-ascend", ("vllm-ascend-hust", "vllm-ascend")),
)

#: 模块名 → 需要它的就绪门。find_spec 只查元数据，不执行模块代码。
_MODULE_PROBES: tuple[str, ...] = (
    "torch",
    "torch_npu",
    "triton",
    "vllm",
    "vllm_ascend",
)


def _package_version(names: tuple[str, ...]) -> str | None:
    for name in names:
        try:
            return version(name)
        except PackageNotFoundError:
            continue
    return None


def _module_present(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # pragma: no cover - 防御：残缺元数据
        return False


def collect_checks(
    model_path: str | Path | None = None,
    *,
    storage_path: str | Path | None = None,
) -> dict[str, Any]:
    """只读收集环境诊断报告（任何环境可安全调用，绝不 import 重栈）。"""
    modules = {name: _module_present(name) for name in _MODULE_PROBES}
    packages = {label: _package_version(names) for label, names in _PACKAGE_PROBES}

    host = None
    if modules["vllm_ascend"]:
        host = "vllm_ascend_hust"
    elif modules["vllm"]:
        host = "vllm_hust"

    target = Path(storage_path) if storage_path is not None else None
    if target is None:
        target = Path(model_path) if model_path is not None else Path.cwd()
    usage = shutil.disk_usage(target if target.exists() else target.parent)

    report: dict[str, Any] = {
        "python": sys.version.split()[0],
        "packages": packages,
        "modules": modules,
        "host": host,
        "methods": list(known_method_names()),
        "storage": {
            "path": str(target),
            "free_bytes": usage.free,
            "total_bytes": usage.total,
        },
    }

    if model_path is not None:
        report["model"] = check_checkpoint(model_path)

    report["ready"] = {
        # L2 语义层：CPU 量化数学。
        "semantics": modules["torch"],
        # L3 内核冒烟：triton-ascend 内核 launch 的前置栈。
        "npu_kernels": all(modules[n] for n in ("torch", "torch_npu", "triton")),
        # L4 宿主注册：scheme 能注册进当前进程可导入的宿主。
        "host_serving": host is not None,
    }
    if model_path is not None:
        # serve 前最后一道：宿主在 + 注入契约校验通过。
        report["ready"]["model"] = host is not None and report["model"]["valid"]
    return report


_REQUIRE_CHOICES = ("semantics", "npu_kernels", "host_serving", "model")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-kv-doctor",
        description=(
            "Read-only readiness diagnosis for the quantized-KV plugin: "
            "package/module presence, host stack, method inventory, disk, "
            "and (with --model) the injection-contract checkpoint check."
        ),
        epilog=(
            "int8 end-to-end additionally depends on a known host-side fix "
            "(docs/how-to-run.md §8.1); statically verify a host checkout "
            "with scripts/verify_host_sources.py."
        ),
    )
    parser.add_argument(
        "--model",
        type=Path,
        help="注入过的 checkpoint 目录：附带 kv-inject 契约校验报告",
    )
    parser.add_argument(
        "--path",
        type=Path,
        help="磁盘容量检查的目标路径（缺省为当前目录或 --model）",
    )
    parser.add_argument(
        "--require",
        choices=_REQUIRE_CHOICES,
        default="host_serving",
        help="退出码门：指定的就绪门不通过时退出 2（缺省 host_serving）",
    )
    parser.add_argument(
        "--json", action="store_true", help="只打印 JSON 报告（供脚本消费）"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.require == "model" and args.model is None:
        parser.error("--require model needs --model")

    report = collect_checks(args.model, storage_path=args.path)
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_human(report)

    gate = report["ready"][args.require]
    return 0 if gate else 2


def _print_human(report: dict[str, Any]) -> None:
    print(f"python {report['python']}")
    for label, ver in report["packages"].items():
        print(f"  {label:<12} {ver or 'not installed'}")
    host = report["host"]
    print(f"  host        {host or 'none (neither vllm_ascend nor vllm)'}")
    print(f"  methods     {', '.join(report['methods'])}")
    print(
        "  disk        free {:.1f} GiB at {}".format(
            report["storage"]["free_bytes"] / 2**30, report["storage"]["path"]
        )
    )
    for gate, ok in report["ready"].items():
        print(f"  ready[{gate:<12}] {'OK' if ok else 'NOT READY'}")
    model = report.get("model")
    if model is not None:
        verdict = "valid" if model["valid"] else "INVALID"
        print(f"  model       {verdict} ({model['model_dir']})")
        for issue in model["issues"]:
            print(f"    - {issue}")
    if not report["ready"]["host_serving"]:
        print(
            "  hint: set VLLM_HUST_KV_METHODS only inside a host "
            "(vllm_ascend/vllm) serving process; see docs/how-to-run.md §6"
        )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
