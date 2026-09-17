# SPDX-License-Identifier: Apache-2.0
"""checkpoint 注入工具（tools/checkpoint.py）与 ascend 注册键推导的测试。

覆盖三块：
1. 注册键推导（adapters/ascend_keys.py）：六个已知方法的键面、fail-closed、
   与设备侧 mixin 接线表的一致性（元数据侧 vs 设备侧防漂移）；
2. 注入语义：完整描述形状（逐层 fa_k/fa_v.scale + 全线性层 FLOAT）、
   幂等、备份/回滚、dry-run、对外部 ModelSlim 产物 fail-closed；
3. 导入卫生：本工具链在干净子进程里不得拉起 torch / vllm / 宿主。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_ascend_quantized_kv_cache.adapters.ascend_keys import (
    STATEFUL_IMPL_METHODS,
    ascend_scheme_key,
)
from vllm_ascend_quantized_kv_cache.tools.checkpoint import (
    DEFAULT_FLOAT_MODULES,
    MANIFEST_NAME,
    build_quant_description,
    check_checkpoint,
    inject,
    known_method_names,
    main,
    resolve_method_token,
    restore,
)

# 注册键面是 HostContract 的一部分（how-to-run.md §6.2 的 fa_quant_type 取值）。
EXPECTED_KEYS = {
    "int8_dynamic": "VLLM_HUST_KV_INT8_DYNAMIC",
    "kivi_int4": "VLLM_HUST_KV_KIVI_INT4",
    "int4_packed": "VLLM_HUST_KV_INT4",
    "fp8_e4m3": "VLLM_HUST_KV_FP8_E4M3",
    "nvfp4": "VLLM_HUST_KV_NVFP4",
    "fp4_e2m1": "VLLM_HUST_KV_FP4_E2M1",
}


# -- 注册键推导 ---------------------------------------------------------------


def test_ascend_keys_cover_the_documented_surface() -> None:
    for name, key in EXPECTED_KEYS.items():
        assert ascend_scheme_key(name) == key


def test_ascend_keys_fail_closed_on_unknown() -> None:
    with pytest.raises(ValueError, match="unknown method"):
        ascend_scheme_key("nope")


def test_ascend_keys_track_the_full_registry() -> None:
    for name in known_method_names():
        assert ascend_scheme_key(name) in EXPECTED_KEYS.values()


def test_stateful_metadata_list_matches_impl_mixins() -> None:
    # 元数据侧（ascend_keys）与设备侧（attention._MIXINS）必须一致。
    from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.attention import (
        supported_impl_methods,
    )

    assert set(STATEFUL_IMPL_METHODS) == set(supported_impl_methods())


def test_adapter_default_quant_type_delegates_to_ascend_keys() -> None:
    from vllm_ascend_quantized_kv_cache.adapters.vllm_ascend_hust.register import (
        AscendHustAdapter,
    )

    for name, key in EXPECTED_KEYS.items():
        assert AscendHustAdapter.default_quant_type(name) == key


# -- 方法名/注册键解析 --------------------------------------------------------


def test_resolve_method_token_accepts_name_and_key() -> None:
    assert resolve_method_token("int8_dynamic") == (
        "int8_dynamic",
        "VLLM_HUST_KV_INT8_DYNAMIC",
    )
    assert resolve_method_token("VLLM_HUST_KV_KIVI_INT4") == (
        "kivi_int4",
        "VLLM_HUST_KV_KIVI_INT4",
    )


def test_resolve_method_token_fail_closed() -> None:
    with pytest.raises(ValueError, match="known: "):
        resolve_method_token("nope")


# -- 描述形状 -----------------------------------------------------------------


def test_description_shape_is_complete_modelslim_artifact() -> None:
    desc = build_quant_description(2, "VLLM_HUST_KV_INT8_DYNAMIC")
    assert desc["fa_quant_type"] == "VLLM_HUST_KV_INT8_DYNAMIC"
    assert desc["model.embed_tokens.weight"] == "FLOAT"
    for i in range(2):
        # 每层：fa_k/fa_v.scale（推导生效层清单的键）+ 全部线性层 FLOAT
        assert desc[f"model.layers.{i}.self_attn.fa_k.scale"] == "FAQuant"
        assert desc[f"model.layers.{i}.self_attn.fa_v.scale"] == "FAQuant"
        for mod in DEFAULT_FLOAT_MODULES:
            assert desc[f"model.layers.{i}.{mod}.weight"] == "FLOAT"
    assert len(desc) == 2 + 1 + 2 * (2 + len(DEFAULT_FLOAT_MODULES))


def test_description_honors_prefix_and_extras() -> None:
    desc = build_quant_description(
        1,
        "VLLM_HUST_KV_KIVI_INT4",
        layer_prefix="transformer",
        include_lm_head=True,
        extra_float_modules=("mlp.shared_expert.gate_proj",),
    )
    assert "transformer.layers.0.self_attn.fa_k.scale" in desc
    assert desc["lm_head.weight"] == "FLOAT"
    assert desc["transformer.layers.0.mlp.shared_expert.gate_proj.weight"] == "FLOAT"
    assert "model.layers.0.self_attn.fa_k.scale" not in desc


# -- 注入语义 -----------------------------------------------------------------


@pytest.fixture()
def model_dir(tmp_path: Path) -> Path:
    config = {
        "model_type": "qwen3",
        "architectures": ["Qwen3ForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 1024,
    }
    (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def _read_config(model_dir: Path) -> dict:
    return json.loads((model_dir / "config.json").read_text(encoding="utf-8"))


def test_inject_writes_complete_config(model_dir: Path) -> None:
    info = inject(model_dir, method="int8_dynamic")
    assert info["fa_quant_type"] == "VLLM_HUST_KV_INT8_DYNAMIC"
    config = _read_config(model_dir)
    # 顶层键保留，只新增 quantization_config
    assert config["model_type"] == "qwen3"
    assert config["num_hidden_layers"] == 2
    qc = config["quantization_config"]
    assert qc["quant_method"] == "ascend"
    assert qc == {
        "quant_method": "ascend",
        **build_quant_description(2, "VLLM_HUST_KV_INT8_DYNAMIC"),
    }
    assert (model_dir / "config.json.bak-kvinject").is_file()


def test_inject_is_idempotent(model_dir: Path) -> None:
    inject(model_dir, method="int8_dynamic")
    first = _read_config(model_dir)
    inject(model_dir, method="VLLM_HUST_KV_INT8_DYNAMIC")  # 注册键拼写同样可重跑
    assert _read_config(model_dir) == first


def test_inject_dry_run_touches_nothing(model_dir: Path) -> None:
    before = (model_dir / "config.json").read_text(encoding="utf-8")
    info = inject(model_dir, method="int8_dynamic", dry_run=True)
    assert info["dry_run"] and info["backup"] is None
    assert (model_dir / "config.json").read_text(encoding="utf-8") == before
    assert not (model_dir / "config.json.bak-kvinject").exists()


def test_restore_round_trip(model_dir: Path) -> None:
    original = (model_dir / "config.json").read_text(encoding="utf-8")
    inject(model_dir, method="kivi_int4")
    assert "quantization_config" in _read_config(model_dir)
    restore(model_dir)
    assert (model_dir / "config.json").read_text(encoding="utf-8") == original
    assert not (model_dir / "config.json.bak-kvinject").exists()
    with pytest.raises(FileNotFoundError, match="no backup"):
        restore(model_dir)


def test_inject_refuses_foreign_ascend_description(model_dir: Path) -> None:
    config = _read_config(model_dir)
    config["quantization_config"] = {
        "quant_method": "ascend",
        "model_quant_type": "W8A8_DYNAMIC",
        "fa_quant_type": "FAKQuant",
        "model.layers.0.self_attn.q_proj.weight": "W8A8",
    }
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="foreign ModelSlim"):
        inject(model_dir, method="int8_dynamic")
    # --force 才放行，且外部键被整体替换掉
    inject(model_dir, method="int8_dynamic", force=True)
    qc = _read_config(model_dir)["quantization_config"]
    assert "model.layers.0.self_attn.q_proj.weight" in qc  # FLOAT 版
    assert qc["model.layers.0.self_attn.q_proj.weight"] == "FLOAT"
    assert "model_quant_type" not in qc


def test_inject_refuses_weight_quantized_checkpoint(model_dir: Path) -> None:
    config = _read_config(model_dir)
    config["quantization_config"] = {"quant_method": "w8a8"}
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="weight-quantized"):
        inject(model_dir, method="int8_dynamic", force=True)


def test_inject_requires_layer_count(model_dir: Path) -> None:
    config = _read_config(model_dir)
    del config["num_hidden_layers"]
    (model_dir / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match="num_hidden_layers"):
        inject(model_dir, method="int8_dynamic")
    inject(model_dir, method="int8_dynamic", num_layers=3)
    assert _read_config(model_dir)["quantization_config"] == {
        "quant_method": "ascend",
        **build_quant_description(3, "VLLM_HUST_KV_INT8_DYNAMIC"),
    }


def test_inject_missing_model_dir_fails(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        inject(tmp_path / "nope", method="int8_dynamic")


# -- CLI ----------------------------------------------------------------------


def test_cli_dry_run_then_inject_then_restore(
    model_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main([str(model_dir), "--method", "int8_dynamic", "--dry-run"]) == 0
    assert "quantization_config" not in _read_config(model_dir)
    assert main([str(model_dir), "--method", "int8_dynamic"]) == 0
    assert "quantization_config" in _read_config(model_dir)
    assert main([str(model_dir), "--restore"]) == 0
    assert "quantization_config" not in _read_config(model_dir)
    out = capsys.readouterr().out
    assert "fa_quant_type=VLLM_HUST_KV_INT8_DYNAMIC" in out


def test_cli_error_returns_exit_code_2(
    model_dir: Path, capsys: pytest.CaptureFixture
) -> None:
    assert main([str(model_dir), "--method", "nope"]) == 2
    assert "known: " in capsys.readouterr().err


def test_cli_lists_methods(capsys: pytest.CaptureFixture) -> None:
    assert main(["--list-methods"]) == 0
    assert "int8_dynamic" in capsys.readouterr().out


# -- 注入契约清单（manifest + --check）----------------------------------------


def test_inject_writes_manifest_and_check_passes(model_dir: Path) -> None:
    inject(model_dir, method="int8_dynamic")
    manifest_path = model_dir / MANIFEST_NAME
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"] == "vllm-hust-kv-inject-manifest-v1"
    assert manifest["method"] == "int8_dynamic"
    assert manifest["fa_quant_type"] == "VLLM_HUST_KV_INT8_DYNAMIC"
    assert manifest["num_layers"] == 2
    # config.json 的 size/SHA-256 绑定真实文件
    import hashlib

    payload = (model_dir / "config.json").read_bytes()
    assert manifest["config_json"] == {
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    # serve 前校验通过
    report = check_checkpoint(model_dir)
    assert report["valid"], report["issues"]
    assert report["description_matches"]


def test_check_detects_post_injection_drift(model_dir: Path) -> None:
    inject(model_dir, method="int8_dynamic")
    config_path = model_dir / "config.json"
    config = _read_config(model_dir)
    # 有人在注入后手改了 config.json：哈希失配必须抓住
    config["num_hidden_layers"] = 99
    config_path.write_text(json.dumps(config), encoding="utf-8")
    report = check_checkpoint(model_dir)
    assert not report["valid"]
    assert any("hash mismatch" in issue for issue in report["issues"])


def test_check_detects_missing_manifest(model_dir: Path) -> None:
    inject(model_dir, method="int8_dynamic")
    (model_dir / MANIFEST_NAME).unlink()
    report = check_checkpoint(model_dir)
    assert not report["valid"]
    assert any(MANIFEST_NAME in issue for issue in report["issues"])


def test_check_is_json_only_manifest_compatible(model_dir: Path) -> None:
    # 旧版注入（没有清单）不该被 --check 误判为有效
    inject(model_dir, method="int8_dynamic")
    (model_dir / MANIFEST_NAME).unlink()
    assert check_checkpoint(model_dir)["valid"] is False


def test_restore_removes_manifest(model_dir: Path) -> None:
    inject(model_dir, method="int8_dynamic")
    restore(model_dir)
    assert not (model_dir / MANIFEST_NAME).exists()
    # 回滚后残留的清单不能再让 --check 通过（防"陈旧契约"假绿）
    report = check_checkpoint(model_dir)
    assert not report["valid"]


def test_manifest_tracks_extra_options(model_dir: Path) -> None:
    inject(
        model_dir,
        method="kivi_int4",
        layer_prefix="transformer",
        include_lm_head=True,
        extra_float_modules=("mlp.shared_expert.gate_proj",),
    )
    report = check_checkpoint(model_dir)
    assert report["valid"], report["issues"]
    manifest = report["manifest"]
    assert manifest["layer_prefix"] == "transformer"
    assert manifest["include_lm_head"] is True
    assert manifest["extra_float_modules"] == ["mlp.shared_expert.gate_proj"]


def test_cli_check_exit_codes(model_dir: Path, capsys: pytest.CaptureFixture) -> None:
    assert main([str(model_dir), "--method", "int8_dynamic"]) == 0
    assert main([str(model_dir), "--check"]) == 0
    out = capsys.readouterr().out
    assert '"valid": true' in out
    # 篡改 config 后 --check 必须退出 2
    config_path = model_dir / "config.json"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace("qwen3", "qwenX"),
        encoding="utf-8",
    )
    assert main([str(model_dir), "--check"]) == 2
    assert '"valid": false' in capsys.readouterr().out
    assert main([str(model_dir), "--check", "--restore"]) == 0  # restore 优先无碍


# -- 导入卫生 -----------------------------------------------------------------


def test_tool_imports_are_inert_in_a_clean_subprocess() -> None:
    code = (
        "import sys\n"
        "import vllm_ascend_quantized_kv_cache.tools.checkpoint\n"
        "heavy = {'torch', 'vllm', 'triton', 'vllm_ascend'}\n"
        "tops = (m.split('.')[0] for m in sys.modules)\n"
        "print('\\n'.join(sorted(t for t in tops if t in heavy)))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert proc.stdout.strip() == ""
