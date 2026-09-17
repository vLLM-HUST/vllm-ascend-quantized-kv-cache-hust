# SPDX-License-Identifier: Apache-2.0
"""evidence 记录校验器（tools/evidence.py）与契约 schema 一致性的测试。

覆盖：示例记录通过、各类负向记录 fail-closed（未知字段/错误枚举/坏
哈希/matched_benchmark 越级）、CLI 退出码、contracts/kv-evidence-v1
.schema.json 与校验器 SPEC 的漂移钉扎。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vllm_ascend_quantized_kv_cache.tools.evidence import (
    SPEC,
    example_record,
    main,
    validate_evidence,
)

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_FILE = ROOT / "evidence" / "example-int8-v1.json"
SCHEMA_FILE = ROOT / "contracts" / "kv-evidence-v1.schema.json"


# -- 正向 ----------------------------------------------------------------------


def test_example_record_passes() -> None:
    report = validate_evidence(example_record())
    assert report["valid"], report["issues"]


def test_shipped_example_file_passes() -> None:
    record = json.loads(EXAMPLE_FILE.read_text(encoding="utf-8"))
    assert validate_evidence(record)["valid"]


def test_shipped_example_round_trips_through_cli(tmp_path: Path) -> None:
    target = tmp_path / "record.json"
    target.write_text(EXAMPLE_FILE.read_text(encoding="utf-8"), encoding="utf-8")
    assert main(["validate", "--file", str(target)]) == 0


def test_example_subcommand_output_is_valid(capsys) -> None:
    assert main(["example"]) == 0
    record = json.loads(capsys.readouterr().out)
    assert validate_evidence(record)["valid"]


# -- 负向（fail-closed）---------------------------------------------------------


def _mutated(**changes) -> dict:
    record = example_record()
    for key, value in changes.items():
        if value is ...:
            record.pop(key, None)
        else:
            record[key] = value
    return record


def test_unknown_field_rejected() -> None:
    record = example_record()
    record["throughput_tok_s"] = 123.0  # 未知指标：必须走新 schema 版本
    report = validate_evidence(record)
    assert not report["valid"]
    assert any("unknown field" in i for i in report["issues"])


def test_unknown_gate_rejected() -> None:
    report = validate_evidence(_mutated(evidence_level="matched_pewpew"))
    assert not report["valid"]
    assert any("not in" in i for i in report["issues"])


def test_missing_required_field_rejected() -> None:
    report = validate_evidence(_mutated(run_id=...))
    assert not report["valid"]
    assert any("run_id" in i for i in report["issues"])


def test_bad_date_and_sha256_rejected() -> None:
    report = validate_evidence(_mutated(date="Sept 12"))
    assert not report["valid"]
    record = example_record()
    record["artifacts"][0]["sha256"] = "zz"
    assert not validate_evidence(record)["valid"]


def test_matched_benchmark_requires_baseline() -> None:
    record = _mutated(evidence_level="matched_benchmark", status="passed")
    report = validate_evidence(record)
    assert not report["valid"]
    assert any("matched_baseline_run_id" in i for i in report["issues"])
    record["matched_baseline_run_id"] = "bf16-run-001"
    assert validate_evidence(record)["valid"]
    record["status"] = "blocked"
    assert not validate_evidence(record)["valid"]


def test_blocked_status_is_valid_evidence() -> None:
    # 阻塞记录合法：它阻止"应该能跑"口径混入（示例即一条 blocked 记录）
    record = example_record()
    assert record["status"] == "blocked"
    assert validate_evidence(record)["valid"]


# -- schema 文件与校验器的一致性（防漂移）---------------------------------------


def test_schema_file_matches_validator_spec() -> None:
    schema = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    assert set(schema["properties"]) == set(SPEC)
    assert sorted(schema["required"]) == sorted(
        name for name, spec in SPEC.items() if spec.get("required")
    )
    assert schema["properties"]["evidence_level"]["enum"] == list(
        SPEC["evidence_level"]["enum"]
    )
    assert schema["additionalProperties"] is False


# -- CLI ----------------------------------------------------------------------


def test_cli_validate_exit_codes(tmp_path: Path, capsys) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{}", encoding="utf-8")
    assert main(["validate", "--file", str(bad)]) == 2
    assert '"valid": false' in capsys.readouterr().out
    missing = tmp_path / "nope.json"
    assert main(["validate", "--file", str(missing)]) == 2


def test_evidence_import_hygiene_in_clean_subprocess() -> None:
    code = (
        "import sys\n"
        "import vllm_ascend_quantized_kv_cache.tools.evidence\n"
        "heavy = {'torch', 'vllm', 'triton', 'vllm_ascend'}\n"
        "tops = (m.split('.')[0] for m in sys.modules)\n"
        "print('\\n'.join(sorted(t for t in tops if t in heavy)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == ""
