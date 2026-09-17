# SPDX-License-Identifier: Apache-2.0
"""环境诊断（tools/doctor.py）与宿主源静态核查（scripts/verify_host_sources.py）。

doctor 测试必须环境无关：断言报告结构与门推导规则，不断言"本机一定装了
torch/宿主"（CI 与 NPU 容器都要能跑）。脚本核查用合成源码树验证命中与
缺失两条路径。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from vllm_ascend_quantized_kv_cache.tools.doctor import (
    _REQUIRE_CHOICES,
    collect_checks,
    main,
)

# -- 报告结构（任何环境都成立）-------------------------------------------------


def test_collect_checks_structure_is_complete() -> None:
    report = collect_checks()
    assert set(report) >= {
        "python",
        "packages",
        "modules",
        "host",
        "methods",
        "storage",
        "ready",
    }
    # 探测面：五个包版本 + 五个模块存在性
    assert set(report["packages"]) == {
        "torch",
        "torch-npu",
        "triton",
        "vllm",
        "vllm-ascend",
    }
    assert set(report["modules"]) == {
        "torch",
        "torch_npu",
        "triton",
        "vllm",
        "vllm_ascend",
    }
    # 方法清单与注册表一致（int8 必在——第一版本的主角）
    assert "int8_dynamic" in report["methods"]
    # 宿主探测与模块存在性逻辑自洽
    if report["modules"]["vllm_ascend"]:
        assert report["host"] == "vllm_ascend_hust"
    elif report["modules"]["vllm"]:
        assert report["host"] == "vllm_hust"
    else:
        assert report["host"] is None


def test_ready_gates_derive_from_module_presence() -> None:
    report = collect_checks()
    ready = report["ready"]
    assert set(ready) >= {"semantics", "npu_kernels", "host_serving"}
    assert ready["semantics"] == report["modules"]["torch"]
    assert ready["npu_kernels"] == all(
        report["modules"][n] for n in ("torch", "torch_npu", "triton")
    )
    assert ready["host_serving"] == (report["host"] is not None)


def test_report_is_json_serializable(tmp_path: Path) -> None:
    payload = json.dumps(collect_checks(tmp_path), ensure_ascii=False)
    assert "storage" in payload


# -- --model 模式：注入契约校验接入 --------------------------------------------


def _make_model(tmp_path: Path) -> Path:
    from vllm_ascend_quantized_kv_cache.tools.checkpoint import inject

    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(
        json.dumps({"model_type": "qwen3", "num_hidden_layers": 2}),
        encoding="utf-8",
    )
    inject(model_dir, method="int8_dynamic")
    return model_dir


def test_model_mode_reports_injection_contract(tmp_path: Path) -> None:
    model_dir = _make_model(tmp_path)
    report = collect_checks(model_dir)
    assert report["model"]["valid"]
    assert report["ready"]["model"] == (report["host"] is not None)

    # 篡改 config 后 model 门必须翻转
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    tampered = collect_checks(model_dir)
    assert not tampered["model"]["valid"]
    assert tampered["ready"]["model"] is False


def test_cli_require_model_gate(tmp_path: Path, capsys) -> None:
    model_dir = _make_model(tmp_path)
    assert main(["--model", str(model_dir), "--require", "model", "--json"]) in (0, 2)
    # model 门 = host 在场 && 校验通过；解析 JSON 自洽校验
    report = json.loads(capsys.readouterr().out)
    expected = 0 if report["ready"]["model"] else 2
    assert main(["--model", str(model_dir), "--require", "model"]) == expected


def test_cli_require_model_without_model_errors(capsys) -> None:
    # 校验在 main() 里做（argparse 的 choices 之外的业务约束）
    try:
        main(["--require", "model"])
    except SystemExit as exc:
        assert exc.code == 2
    else:  # pragma: no cover
        raise AssertionError("--require model without --model must fail")
    assert "--require model needs --model" in capsys.readouterr().err


def test_require_choices_cover_the_documented_gates() -> None:
    assert set(_REQUIRE_CHOICES) >= {
        "semantics",
        "npu_kernels",
        "host_serving",
        "model",
    }


# -- 导入卫生 -----------------------------------------------------------------


def test_doctor_imports_are_inert_in_a_clean_subprocess() -> None:
    code = (
        "import sys\n"
        "import vllm_ascend_quantized_kv_cache.tools.doctor\n"
        "heavy = {'torch', 'vllm', 'triton', 'vllm_ascend'}\n"
        "tops = (m.split('.')[0] for m in sys.modules)\n"
        "print('\\n'.join(sorted(t for t in tops if t in heavy)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == ""


# -- scripts/verify_host_sources.py（子进程 + 合成源码树）-----------------------


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "verify_host_sources.py"


def _make_ascend_tree(tmp_path: Path, *, with_issue: bool = True) -> Path:
    root = tmp_path / "vllm-ascend-hust"
    pkg = root / "vllm_ascend" / "quantization" / "methods"
    pkg.mkdir(parents=True)
    (pkg / "registry.py").write_text(
        "def register_scheme(quant_type, layer_type):\n"
        "    ...\n"
        "def get_scheme_class(quant_type, layer_type):\n"
        "    ...\n",
        encoding="utf-8",
    )
    (pkg / "base.py").write_text(
        "class AscendAttentionScheme:\n    ...\n", encoding="utf-8"
    )
    slim = root / "vllm_ascend" / "quantization"
    (slim / "config.py").write_text(
        "# ModelSlim 解析\nfa_quant_type = None\nkvcache_quant_layers = []\n",
        encoding="utf-8",
    )
    loader = root / "vllm_ascend" / "platform"
    loader.mkdir()
    (loader / "plugins.py").write_text(
        'GROUP = "vllm.general_plugins"\n', encoding="utf-8"
    )
    if with_issue:
        mr = root / "vllm_ascend" / "v1" / "worker"
        mr.mkdir(parents=True)
        (mr / "model_runner_v1.py").write_text(
            "def _reshape_kv_cache_tensors():\n    ...\n", encoding="utf-8"
        )
    return root


def test_verify_host_sources_passes_on_synthetic_ascend_tree(
    tmp_path: Path, capsys
) -> None:
    root = _make_ascend_tree(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--vllm-ascend-src", str(root), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(proc.stdout)
    assert payload["valid"] is True
    ascend = payload["hosts"]["vllm_ascend_hust"]
    assert ascend["missing"] == []
    # 已知 int8 分配切分不对称问题被点名（review 不判失败，但必须被记住）
    issue = ascend["known_issues"][0]
    assert issue["id"] == "int8_kv_split_factor_asymmetry"
    assert issue["found_in"]  # 命中位置列表非空


def test_verify_host_sources_ranks_production_over_tests(
    tmp_path: Path,
) -> None:
    root = _make_ascend_tree(tmp_path)
    # 同名表面在测试目录里也出现：生产文件必须排在首位
    dup = root / "tests" / "ut" / "quantization"
    dup.mkdir(parents=True)
    (dup / "test_registry.py").write_text(
        "def register_scheme(x):\n    ...\n", encoding="utf-8"
    )
    # 硬件变体子树同样命中：排在主树之后、测试之前
    variant = root / "vllm_ascend" / "_310p" / "quantization" / "methods"
    variant.mkdir(parents=True)
    (variant / "registry.py").write_text(
        "def register_scheme(q, l):\n    ...\n", encoding="utf-8"
    )
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--vllm-ascend-src", str(root), "--json"],
        capture_output=True,
        text=True,
        check=True,
    )
    hits = json.loads(proc.stdout)["hosts"]["vllm_ascend_hust"]["surfaces"][
        "scheme_registry_register_scheme"
    ]
    assert len(hits) == 3
    assert not hits[0].startswith("tests")
    assert "_310p" not in hits[0]
    assert hits[-1].startswith("tests")


def test_verify_host_sources_fails_on_missing_surfaces(tmp_path: Path) -> None:
    root = tmp_path / "empty-host"
    root.mkdir()
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--vllm-ascend-src", str(root)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "MISS" in proc.stdout
