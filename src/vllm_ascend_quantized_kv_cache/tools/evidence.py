# SPDX-License-Identifier: Apache-2.0
"""evidence 记录校验器（``vllm-hust-kv-evidence``）：把验收矩阵变成机器可查。

对标离线量化工具链的 evidence 纪律（参考项目
``ascend-quant-evidence-v1`` + ``validate-evidence`` 命令）：验证运行的
结果落成**封闭、版本化的 JSON 记录**，由本工具校验后才能被验收矩阵
引用。schema 文件在 ``contracts/kv-evidence-v1.schema.json``；本模块用
内部 SPEC 实现其必需子集（本包零运行时依赖，不引入 jsonschema），
``tests/test_evidence.py`` 钉住两者的漂移。

纪律口径（与 docs/acceptance-matrix.md 一致）：
- **未知字段一律拒绝**——新增布局/指标必须走新 schema 版本，不允许
  静默改义；
- ``evidence_level`` 是六级推广门；声称 ``matched_benchmark`` 必须给出
  同协议 BF16 基线（``matched_baseline_run_id``）且 ``status=passed``；
- ``status=blocked`` 合法：阻塞记录同样是证据（它阻止"应该能跑"的
  口径混入发布说明）。
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCHEMA_ID = "vllm-hust-kv-evidence-v1"

GATES = (
    "schema_only",
    "cpu_semantics",
    "host_registration",
    "npu_kernel_bitexact",
    "npu_e2e",
    "matched_benchmark",
)
_STATUSES = ("passed", "failed", "blocked")
_HOSTS = ("vllm_ascend_hust", "vllm_hust")
_VERDICTS = _STATUSES
_DATE_RE = r"^\d{4}-\d{2}-\d{2}$"
_SHA256_RE = r"^[0-9a-f]{64}$"


def _str_spec(**kw: Any) -> dict:
    kw.setdefault("type", "str")
    return kw


#: 与 contracts/kv-evidence-v1.schema.json 保持同步（测试钉住漂移）。
SPEC: dict[str, dict[str, Any]] = {
    "schema": {"type": "const", "value": SCHEMA_ID, "required": True},
    "run_id": _str_spec(required=True),
    "date": _str_spec(required=True, pattern=_DATE_RE),
    "method": _str_spec(required=True),
    "profile": _str_spec(required=True),
    "evidence_level": _str_spec(required=True, enum=GATES),
    "status": _str_spec(required=True, enum=_STATUSES),
    "environment": {
        "type": "object",
        "required": True,
        "fields": {
            "host": _str_spec(required=True, enum=_HOSTS),
            "device": _str_spec(required=True),
            "container": _str_spec(),
            "torch": _str_spec(),
            "torch_npu": _str_spec(),
            "vllm": _str_spec(),
            "vllm_ascend": _str_spec(),
            "cann": _str_spec(),
            "tensor_parallel": {"type": "int"},
        },
    },
    "model": {
        "type": "object",
        "required": True,
        "fields": {
            "name": _str_spec(required=True),
            "num_layers": {"type": "int", "required": True},
            "dtype": _str_spec(),
        },
    },
    "recipe": {
        "type": "object",
        "required": True,
        "fields": {
            "fa_quant_type": _str_spec(),
            "kv_cache_dtype": _str_spec(),
            "env": {"type": "object"},
        },
    },
    "results": {
        "type": "list",
        "required": True,
        "items": {
            "type": "object",
            "fields": {
                "name": _str_spec(required=True),
                "verdict": _str_spec(required=True, enum=_VERDICTS),
                "detail": _str_spec(),
            },
        },
    },
    "artifacts": {
        "type": "list",
        "items": {
            "type": "object",
            "fields": {
                "path": _str_spec(required=True),
                "size": {"type": "int"},
                "sha256": _str_spec(pattern=_SHA256_RE),
            },
        },
    },
    "raw_logs": {"type": "list"},
    "notes": _str_spec(),
    "matched_baseline_run_id": _str_spec(),
}


def _check_value(
    value: Any, spec: dict[str, Any], path: str, issues: list[str]
) -> None:
    kind = spec.get("type", "str")
    if kind == "const":
        if value != spec["value"]:
            issues.append(f"{path}: must equal {spec['value']!r}, got {value!r}")
        return
    type_ok: dict[str, tuple[type, ...]] = {
        "str": (str,),
        "int": (int,),
        "float": (int, float),
        "bool": (bool,),
        "list": (list,),
        "object": (dict,),
    }
    if value is not None and not isinstance(value, type_ok.get(kind, (str,))):
        if kind == "int" and isinstance(value, bool):  # bool 是 int 子类
            issues.append(f"{path}: expected int, got bool")
            return
        issues.append(f"{path}: expected {kind}, got {type(value).__name__}")
        return
    if "enum" in spec and value is not None and value not in spec["enum"]:
        issues.append(f"{path}: {value!r} not in {list(spec['enum'])}")
        return
    pattern = spec.get("pattern")
    if pattern is not None and isinstance(value, str) and not re.match(pattern, value):
        issues.append(f"{path}: {value!r} does not match {pattern}")


def _check_object(obj: Any, spec: dict[str, Any], path: str, issues: list[str]) -> None:
    fields: dict[str, dict] = spec.get("fields", {})
    if not isinstance(obj, dict):
        issues.append(f"{path}: expected object, got {type(obj).__name__}")
        return
    if fields:
        # 只有声明了字段的封闭对象才拒绝未知键（recipe.env 这类自由
        # 键值表不在此列——与 JSON Schema 的 additionalProperties 对应）
        unknown = sorted(set(obj) - set(fields))
        if unknown:
            issues.append(
                f"{path}: unknown field(s) {unknown}; supported schema is "
                f"{SCHEMA_ID} (extend via a new schema revision, not silently)"
            )
    for name, field_spec in fields.items():
        if name not in obj:
            if field_spec.get("required"):
                issues.append(f"{path}.{name}: missing required field")
            continue
        child = f"{path}.{name}"
        value = obj[name]
        if field_spec["type"] == "object":
            _check_object(value, field_spec, child, issues)
        elif field_spec["type"] == "list":
            _check_list(value, field_spec, child, issues)
        else:
            _check_value(value, field_spec, child, issues)


def _check_list(value: Any, spec: dict[str, Any], path: str, issues: list[str]) -> None:
    if not isinstance(value, list):
        issues.append(f"{path}: expected list, got {type(value).__name__}")
        return
    items = spec.get("items")
    if items is None:
        return
    for idx, element in enumerate(value):
        child = f"{path}[{idx}]"
        if items["type"] == "object":
            _check_object(element, items, child, issues)
        else:
            _check_value(element, items, child, issues)


def validate_evidence(record: Any) -> dict[str, Any]:
    """校验一条 evidence 记录，返回结构化报告（``valid`` + ``issues``）。"""
    issues: list[str] = []
    _check_object(record, {"fields": SPEC, "type": "object"}, "record", issues)

    # 跨字段不变量（JSON Schema 表达不了的推广门语义）
    level = record.get("evidence_level") if isinstance(record, dict) else None
    if isinstance(record, dict):
        status = record.get("status")
        baseline = record.get("matched_baseline_run_id")
        if level == "matched_benchmark":
            if status != "passed":
                issues.append(
                    f"record: matched_benchmark requires status=passed (got {status!r})"
                )
            if not baseline:
                issues.append(
                    "record: matched_benchmark requires a non-null "
                    "matched_baseline_run_id (BF16 baseline under the "
                    "identical protocol)"
                )
        if level == "npu_e2e" and status == "passed":
            results = record.get("results") or []
            if not any(
                r.get("verdict") == "passed" for r in results if isinstance(r, dict)
            ):
                issues.append(
                    "record: npu_e2e with status=passed needs at least one "
                    "passed result entry"
                )

    run_id = record.get("run_id") if isinstance(record, dict) else None
    return {
        "valid": not issues,
        "issues": issues,
        "run_id": run_id,
        "evidence_level": level,
        "schema": SCHEMA_ID,
    }


def example_record() -> dict[str, Any]:
    """返回一份通过校验的示例记录（占位值，不构成任何真机声明）。"""
    return {
        "schema": SCHEMA_ID,
        "run_id": "example-int8-host-registration",
        "date": "2026-09-12",
        "method": "int8_dynamic",
        "profile": "int8_storage",
        "evidence_level": "host_registration",
        "status": "blocked",
        "environment": {
            "host": "vllm_ascend_hust",
            "device": "Ascend 910B2",
            "container": "container-86",
            "torch": None,
            "torch_npu": None,
            "vllm": None,
            "vllm_ascend": None,
            "cann": None,
            "tensor_parallel": 1,
        },
        "model": {"name": "Qwen3-0.6B", "num_layers": 28, "dtype": "float16"},
        "recipe": {
            "fa_quant_type": "VLLM_HUST_KV_INT8_DYNAMIC",
            "kv_cache_dtype": "int8_per_token_head",
            "env": {"VLLM_HUST_KV_METHODS": "int8_dynamic"},
        },
        "results": [
            {
                "name": "scheme registration + fa_quant dispatch + impl surgery",
                "verdict": "passed",
                "detail": (
                    "28/28 layers class-surgery OK "
                    "(see docs/validation-int8-20260912.md)"
                ),
            },
            {
                "name": "int8 storage allocation",
                "verdict": "blocked",
                "detail": (
                    "fa_quant split factor asymmetry; "
                    "see provenance/host-fixes/README.md"
                ),
            },
        ],
        "artifacts": [
            {
                "path": "config.json",
                "size": 0,
                # 空字符串的 SHA-256：众所周知的占位值，示例不冒充真机产物
                "sha256": "e3b0c44298fc1c149afbf4c8996fb924"
                "27ae41e4649b934ca495991b7852b855",
            }
        ],
        "raw_logs": ["container-86 local experiment archive"],
        "notes": (
            "EXAMPLE record with placeholder values; it documents the "
            "record shape, not a hardware claim. Produce real records per "
            "validation run (docs/acceptance-matrix.md)."
        ),
        "matched_baseline_run_id": None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-kv-evidence",
        description=(
            "Validate closed evidence records for quantized-KV validation "
            "runs (docs/acceptance-matrix.md discipline)."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    validate_parser = sub.add_parser("validate", help="校验一条 evidence 记录")
    validate_parser.add_argument("--file", type=Path, required=True)
    sub.add_parser("example", help="打印通过校验的示例记录（占位值）")
    args = parser.parse_args(argv)

    if args.command == "example":
        print(json.dumps(example_record(), ensure_ascii=False, indent=2))
        return 0

    try:
        record = json.loads(Path(args.file).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(json.dumps({"valid": False, "issues": [f"unreadable file: {exc}"]}))
        return 2
    report = validate_evidence(record)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["valid"] else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
