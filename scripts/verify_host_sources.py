#!/usr/bin/env python3
"""静态核查宿主源码树：本插件的注册表面是否真的存在（对标冻结基线的
可执行锁定，而非文档口径）。

适配器注册进宿主的那几个表面（registry、基类、ModelSlim 解析、
CUSTOM 后端槽位、CacheDType 字面量）都是**接口约定**——宿主侧重构、
fork 漂移时插件会 fail-closed，但那时已经是 serve 现场。本工具把失败
提前到联调前：给一个宿主 checkout 的路径，逐项报告表面在不在。

用法::

    python scripts/verify_host_sources.py \
        --vllm-ascend-src /path/to/vllm-ascend-hust \
        [--vllm-src /path/to/vllm-hust] [--json]

退出码：全部 required 表面命中 → 0；任何一项缺失 → 2。
known_issues 只报告不判失败：它们需要人工读代码确认修复状态
（静态扫描无法判断语义），但能确保问题不被遗忘。

只依赖标准库；在宿主进程外运行，不 import 任何宿主代码。
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

SKIP_DIRS = {".git", "__pycache__", ".venv", "build", "dist", ".eggs"}
MAX_FILE_BYTES = 2_000_000  # 超大文件跳过（生成的捆绑文件不是核查目标）

Issue = dict  # {"id","severity","detail","reference"}


def _iter_py_files(root: Path):
    for path in root.rglob("*.py"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        yield path


def _hit_rank(rel_path: str) -> tuple[int, int, str]:
    """生产代码优先于测试；主树优先于硬件变体子树（如 _310p）。"""
    parts = rel_path.split(os.sep)
    under_tests = "tests" in parts or parts[-1].startswith("test_")
    under_variant = any(
        part.startswith("_") and any(ch.isdigit() for ch in part) for part in parts
    )
    return (1 if under_tests else 0, 1 if under_variant else 0, rel_path)


def _scan(root: Path, patterns: dict[str, str]) -> dict[str, list[str]]:
    """在源码树里找每个正则的全部命中文件（生产代码优先排序）。

    只报首个命中会误导联调：真实宿主里同名表面散布在 tests/ 与硬件
    变体子树（_310p 等），第一处命中常常不是 serve 实际走的那份。
    """
    hits: dict[str, list[str]] = {name: [] for name in patterns}
    texts: list[tuple[Path, str]] = []
    for path in _iter_py_files(root):
        try:
            texts.append((path, path.read_text(encoding="utf-8", errors="replace")))
        except OSError:
            continue
    for name, pattern in patterns.items():
        rx = re.compile(pattern)
        for path, text in texts:
            if rx.search(text):
                hits[name].append(str(path.relative_to(root)))
        hits[name].sort(key=_hit_rank)
    return hits


def _check(root: Path, required: dict[str, str], known_issues: list[Issue]) -> dict:
    if not root.is_dir():
        return {"root": str(root), "present": False, "surfaces": {}, "known_issues": []}
    surfaces = _scan(root, required)
    known: list[Issue] = []
    for issue in known_issues:
        found_in = _scan(root, {issue["id"]: issue["pattern"]})[issue["id"]]
        if found_in:
            known.append({**issue, "found_in": found_in})
    return {
        "root": str(root),
        "present": True,
        "surfaces": surfaces,
        "missing": sorted(name for name, where in surfaces.items() if not where),
        "known_issues": known,
    }


# -- vllm-ascend-hust：register_scheme 注册链 + ModelSlim 分发链 --------------

ASCEND_REQUIRED = {
    # adapters/vllm_ascend_hust/register.py 的注册入口。
    "scheme_registry_register_scheme": r"def register_scheme\s*\(",
    "scheme_registry_get_scheme_class": r"def get_scheme_class\s*\(",
    # 生成 scheme 类继承的宿主基类。
    "ascend_attention_scheme_base": r"class AscendAttentionScheme\b",
    # checkpoint 分发链：fa_quant_type 全局开关 + 逐层键推导生效层清单
    # （tools/checkpoint.py 写出的描述由它消费，缺了就一层都不会命中）。
    "modelslim_fa_quant_type": r"fa_quant_type",
    "modelslim_kvcache_quant_layers": r"kvcache_quant_layers",
    # vllm.general_plugins 引导钩子的宿主侧加载器。
    "general_plugins_loader": r"vllm\.general_plugins",
}

ASCEND_KNOWN_ISSUES: list[Issue] = [
    {
        "id": "int8_kv_split_factor_asymmetry",
        "severity": "review",
        "detail": (
            "fa_quant KV 分配的 K/V 字节切分不对称（get_kv_quant_split_factor "
            "对稠密层 V×2）与 int8/int8 存储 + 满头重排矛盾，int8 存储路径 "
            "初始化即失败；根因分析与修复提案见 "
            "provenance/host-fixes/README.md，插件侧过渡守卫见 "
            "VLLM_HUST_KV_ALLOC_GUARD（how-to-run.md §6.5）。命中表示该 "
            "代码路径存在，需人工确认宿主是否已修复"
        ),
        "reference": "provenance/host-fixes/README.md",
        "pattern": r"get_kv_quant_split_factor|_reshape_kv_cache_tensors",
    }
]

# -- vllm-hust：CUSTOM 后端槽位 + CacheDType 字面量 ---------------------------

VLLM_REQUIRED = {
    "attention_backend_enum_custom": r"class AttentionBackendEnum\b",
    # 适配器协商的两个量化 dtype 字面量（int8_dynamic / kivi_int4 路径）。
    "cache_dtype_int8_per_token_head": r"int8_per_token_head",
    "cache_dtype_int4_per_token_head": r"int4_per_token_head",
}

VLLM_KNOWN_ISSUES: list[Issue] = []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="verify_host_sources.py",
        description=(
            "Statically verify that a host checkout actually contains the "
            "surfaces this plugin registers into (and flag known issues)."
        ),
    )
    parser.add_argument(
        "--vllm-ascend-src", type=Path, help="vllm-ascend-hust checkout 根"
    )
    parser.add_argument("--vllm-src", type=Path, help="vllm-hust checkout 根（可选）")
    parser.add_argument("--json", action="store_true", help="只打印 JSON 报告")
    args = parser.parse_args(argv)
    if args.vllm_ascend_src is None and args.vllm_src is None:
        parser.error("give at least one of --vllm-ascend-src / --vllm-src")

    reports = {}
    if args.vllm_ascend_src is not None:
        reports["vllm_ascend_hust"] = _check(
            args.vllm_ascend_src, ASCEND_REQUIRED, ASCEND_KNOWN_ISSUES
        )
    if args.vllm_src is not None:
        reports["vllm_hust"] = _check(args.vllm_src, VLLM_REQUIRED, VLLM_KNOWN_ISSUES)

    ok = all(
        report.get("present") and not report.get("missing")
        for report in reports.values()
    )
    payload = {"valid": ok, "hosts": reports}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        for host, report in reports.items():
            print(f"== {host}: {report['root']}")
            if not report.get("present"):
                print("   MISSING checkout directory")
                continue
            for name, where in report["surfaces"].items():
                if not where:
                    print(f"   [MISS] {name}")
                    continue
                extra = f" (+{len(where) - 1} more)" if len(where) > 1 else ""
                print(f"   [OK  ] {name}  ({where[0]}){extra}")
            for issue in report["known_issues"]:
                locs = ", ".join(issue["found_in"][:2])
                more = (
                    f" (+{len(issue['found_in']) - 2})"
                    if len(issue["found_in"]) > 2
                    else ""
                )
                print(f"   [REVIEW] {issue['id']}: {locs}{more} — {issue['reference']}")
        print(f"verify_host_sources: {'OK' if ok else 'FAILED'}")
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
