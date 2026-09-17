# SPDX-License-Identifier: Apache-2.0
"""checkpoint 注入工具：把 ``fa_quant_type`` 分发配置写成合法的 ModelSlim 描述。

为什么需要这个工具（vllm-ascend-hust 宿主的 ModelSlim 契约）：宿主
``AscendModelSlimConfig`` 解析 checkpoint 的 ``quantization_config`` 时，
``fa_quant_type`` 只是全局开关；真正生效的层清单从逐层键
``<prefix>.layers.N.self_attn.fa_k.scale`` 推导（``kvcache_quant_layers``），
而且完整的 ModelSlim 描述要求每个可量化模块显式声明类型（保持浮点的
模块写 ``FLOAT``）。因此只加 ``{"quant_method": "ascend", "fa_quant_type":
"VLLM_HUST_KV_*"}`` 两个全局字段**不会命中任何 attention 层**——本工具
一行生成全量枚举的合法描述，免去手工编辑 checkpoint。

职责边界：注入只解决"分发"这一半（fa_quant_type -> scheme -> create_weights
类手术）；"注册"那一半仍需 ``VLLM_HUST_KV_METHODS`` 环境变量 opt-in
（bootstrap，见 docs/how-to-run.md §6.1）；int8 存储路径另需 serve 期
``--kv-cache-dtype``（§6.4）。

注入同时落一份契约清单 ``kv_inject_manifest.json``（方法、fa_quant_type、
层数与 config.json 的 size/SHA-256 绑定）：serve 前 ``--check`` 用它校验
checkpoint 未漂移；``--restore`` 回滚 config.json 时一并删除清单。

本模块只依赖标准库 + 本包轻量元数据：绝不 import torch / vllm / 宿主栈
（tests/test_checkpoint_inject.py 有子进程导入卫生测试钉住）。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

#: 保持浮点的线性模块后缀（相对 ``<prefix>.layers.N``）。
DEFAULT_FLOAT_MODULES: tuple[str, ...] = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)

#: ModelSlim 描述里"保持浮点"的取值（宿主解析为不量化）。
_FLOAT = "FLOAT"
#: 逐层 fa_*.scale 条目的类型值；与 ModelSlim FA 描述的生态约定一致。
#: 宿主解析只消费键的存在性（推导生效层清单），不读这个值。
_FA_QUANT = "FAQuant"
#: 描述 schema 版本（对齐 ModelSlim 产物先例；宿主解析不消费）。
_SCHEMA_VERSION = "1.0.0"
#: 注入前 config.json 的备份后缀。
_BACKUP_SUFFIX = ".bak-kvinject"
#: 注入契约清单文件名：把注入事实与 config.json 的 size/SHA-256 绑定，
#: 供 serve 前 ``--check`` 校验 checkpoint 未漂移（对标离线量化工件的
#: contract 思路，范围仅限本工具触碰的 config.json）。
MANIFEST_NAME = "kv_inject_manifest.json"
_MANIFEST_SCHEMA = "vllm-hust-kv-inject-manifest-v1"


def _file_digest(path: Path) -> dict[str, int | str]:
    """文件的 size + SHA-256 绑定记录（contract 的最小完整形状）。"""
    payload = path.read_bytes()
    return {"size": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def known_method_names() -> tuple[str, ...]:
    """本方法库支持 vllm-ascend-hust 宿主的全部方法名。"""
    from .. import methods  # noqa: F401  (注册副作用，纯元数据)
    from ..core.hosts import VLLM_ASCEND_HUST
    from ..methods.registry import list_methods

    return list_methods(host=VLLM_ASCEND_HUST)


def resolve_method_token(token: str) -> tuple[str, str]:
    """把 CLI 输入解析为 (方法名, fa_quant_type 注册键)。

    接受方法名（如 ``int8_dynamic``）或注册键本身（如
    ``VLLM_HUST_KV_INT8_DYNAMIC``——文档里 ``fa_quant_type`` 的取值）。
    未知输入 fail-closed 并列出全部已知方法。
    """
    from ..adapters.ascend_keys import ascend_scheme_key

    names = known_method_names()
    if token in names:
        return token, ascend_scheme_key(token)
    for name in sorted(names):
        if ascend_scheme_key(name) == token:
            return name, token
    raise ValueError(
        f"unknown quantized KV method {token!r}; known: {', '.join(sorted(names))}"
    )


def build_quant_description(
    num_layers: int,
    fa_quant_type: str,
    *,
    layer_prefix: str = "model",
    include_lm_head: bool = False,
    extra_float_modules: tuple[str, ...] = (),
) -> dict[str, str]:
    """构造全量枚举的 ModelSlim 风格量化描述。

    形状对齐宿主解析契约与 ModelSlim 产物先例：全局键（version /
    fa_quant_type）+ ``embed_tokens`` FLOAT + 每层 7 个线性模块 FLOAT +
    每层 ``fa_k.scale`` / ``fa_v.scale``（FAQuant）。刻意不写
    ``model_quant_type``：KV-only 量化没有对应的 ModelSlim 权重量化类型，
    宿主解析也不消费该键。

    键序确定（全局在前、模块键排序），保证幂等重跑的 diff 是干净的。
    """
    desc: dict[str, str] = {
        "version": _SCHEMA_VERSION,
        "fa_quant_type": fa_quant_type,
        f"{layer_prefix}.embed_tokens.weight": _FLOAT,
    }
    if include_lm_head:
        desc["lm_head.weight"] = _FLOAT
    modules = tuple(DEFAULT_FLOAT_MODULES) + tuple(extra_float_modules)
    for i in range(num_layers):
        base = f"{layer_prefix}.layers.{i}"
        for mod in modules:
            desc[f"{base}.{mod}.weight"] = _FLOAT
        desc[f"{base}.self_attn.fa_k.scale"] = _FA_QUANT
        desc[f"{base}.self_attn.fa_v.scale"] = _FA_QUANT
    ordered = {k: desc[k] for k in ("version", "fa_quant_type")}
    ordered.update({k: desc[k] for k in sorted(desc) if k not in ordered})
    return ordered


def inject(
    model_dir: str | Path,
    *,
    method: str,
    num_layers: int | None = None,
    layer_prefix: str = "model",
    include_lm_head: bool = False,
    extra_float_modules: tuple[str, ...] = (),
    dry_run: bool = False,
    force: bool = False,
) -> dict:
    """把量化描述注入 ``<model_dir>/config.json`` 的 quantization_config。

    安全策略（fail-closed）：
    - 已带非 ascend 量化配置（权重已量化的 checkpoint）一律拒绝——那不是
      本工具的目标输入，覆盖即破坏；
    - 已带 ascend 描述但包含本工具不会生成的键（外部 ModelSlim 产物）
      拒绝覆盖，除非 ``--force``；
    - 每次写入前先把原文件备份为 ``config.json.bak-kvinject``，
      ``restore`` 用它回滚。
    """
    method_name, fa_quant_type = resolve_method_token(method)
    model_path = Path(model_dir)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"no config.json under {model_path}; not a model dir?")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    layers = num_layers if num_layers is not None else config.get("num_hidden_layers")
    if not isinstance(layers, int) or layers <= 0:
        raise ValueError(
            "config.json has no usable num_hidden_layers "
            "(VLM/multimodal configs nest it); pass --num-layers explicitly"
        )
    fresh = build_quant_description(
        layers,
        fa_quant_type,
        layer_prefix=layer_prefix,
        include_lm_head=include_lm_head,
        extra_float_modules=extra_float_modules,
    )

    replaced_existing = False
    existing = config.get("quantization_config")
    if existing is not None:
        quant_method = existing.get("quant_method")
        if quant_method != "ascend":
            raise ValueError(
                f"checkpoint already carries quantization_config with "
                f"quant_method={quant_method!r} (a weight-quantized model?); "
                "this tool only injects into original float checkpoints"
            )
        stale = sorted(set(existing) - set(fresh) - {"quant_method"})
        if stale:
            if not force:
                raise ValueError(
                    "existing ascend quant description has keys this tool does "
                    f"not generate (a foreign ModelSlim artifact?): {stale[:4]}"
                    f" ... ({len(stale)} total); pass --force to overwrite"
                )
            replaced_existing = True

    if not dry_run:
        backup_path = config_path.with_name(config_path.name + _BACKUP_SUFFIX)
        backup_path.write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        config["quantization_config"] = {"quant_method": "ascend", **fresh}
        config_path.write_text(
            json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        manifest_path = model_path / MANIFEST_NAME
        manifest_path.write_text(
            json.dumps(
                {
                    "schema": _MANIFEST_SCHEMA,
                    "method": method_name,
                    "fa_quant_type": fa_quant_type,
                    "num_layers": layers,
                    "layer_prefix": layer_prefix,
                    "include_lm_head": include_lm_head,
                    "extra_float_modules": list(extra_float_modules),
                    "tool_version": _tool_version(),
                    "config_json": _file_digest(config_path),
                },
                indent=2,
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )

    return {
        "model_dir": str(model_path),
        "method": method_name,
        "fa_quant_type": fa_quant_type,
        "num_layers": layers,
        "dry_run": dry_run,
        "replaced_existing": replaced_existing,
        "backup": None
        if dry_run
        else str(config_path.with_name(config_path.name + _BACKUP_SUFFIX)),
        "manifest": None if dry_run else str(model_path / MANIFEST_NAME),
    }


def _tool_version() -> str:
    from .._version import __version__

    return __version__


def check_checkpoint(model_dir: str | Path) -> dict:
    """serve 前校验：注入清单存在、config.json 未漂移、描述与清单一致。

    校验三件事（任何一条不满足即 ``valid=False`` 并给出 issue）：
    1. ``kv_inject_manifest.json`` 存在且 schema 可识别；
    2. config.json 的 size/SHA-256 与清单绑定值一致（注入后没被人改过）；
    3. config.json 里的 ``quantization_config`` 与按清单参数重新生成的
       完整描述逐键一致（防"清单新、内容旧"的半截注入）。

    返回结构化报告 dict，``valid`` 为总体布尔；CLI ``--check`` 打印它。
    """
    model_path = Path(model_dir)
    issues: list[str] = []
    config_path = model_path / "config.json"
    manifest_path = model_path / MANIFEST_NAME

    manifest: dict | None = None
    if not manifest_path.is_file():
        issues.append(
            f"no {MANIFEST_NAME} under {model_path}; was the checkpoint "
            "injected by vllm-hust-kv-inject? (re-run inject to create it)"
        )
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"{MANIFEST_NAME} is not valid JSON: {exc}")
        else:
            if manifest.get("schema") != _MANIFEST_SCHEMA:
                issues.append(
                    f"unsupported manifest schema {manifest.get('schema')!r}; "
                    f"expected {_MANIFEST_SCHEMA!r}"
                )
                manifest = None

    digest = _file_digest(config_path) if config_path.is_file() else None
    if digest is None:
        issues.append(f"no config.json under {model_path}")
    elif manifest is not None and digest != manifest.get("config_json"):
        issues.append(
            "config.json hash mismatch: manifest binds "
            f"{manifest.get('config_json')}, found {digest} "
            "(checkpoint drifted after injection?)"
        )

    description_matches = False
    if manifest is not None and config_path.is_file():
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            issues.append(f"config.json is not valid JSON: {exc}")
        else:
            expected = build_quant_description(
                manifest["num_layers"],
                manifest["fa_quant_type"],
                layer_prefix=manifest.get("layer_prefix", "model"),
                include_lm_head=bool(manifest.get("include_lm_head")),
                extra_float_modules=tuple(manifest.get("extra_float_modules", ())),
            )
            actual = config.get("quantization_config")
            if actual == {"quant_method": "ascend", **expected}:
                description_matches = True
            else:
                issues.append(
                    "config.json quantization_config does not match the "
                    "description bound by the manifest"
                )

    return {
        "model_dir": str(model_path),
        "valid": not issues,
        "issues": issues,
        "manifest": manifest,
        "config_json": digest,
        "description_matches": description_matches,
    }


def restore(model_dir: str | Path) -> dict:
    """用注入时留下的备份回滚 config.json（备份用后即删）。"""
    config_path = Path(model_dir) / "config.json"
    backup_path = config_path.with_name(config_path.name + _BACKUP_SUFFIX)
    if not backup_path.is_file():
        raise FileNotFoundError(
            f"no backup at {backup_path}; nothing to restore "
            "(was the checkpoint injected by this tool?)"
        )
    config_path.write_text(backup_path.read_text(encoding="utf-8"), encoding="utf-8")
    backup_path.unlink()
    # 回滚后清单描述的不再是 checkpoint 的事实，一并删掉，避免留下
    # "看起来还能过 --check"的陈旧契约。
    manifest_path = config_path.with_name(MANIFEST_NAME)
    if manifest_path.is_file():
        manifest_path.unlink()
    return {"model_dir": str(config_path.parent), "restored": str(config_path)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="vllm-hust-kv-inject",
        description=(
            "Inject a complete ModelSlim-style quantization_config into a "
            "model checkpoint so the vllm-ascend-hust host dispatches the "
            "VLLM_HUST_KV_* scheme to every attention layer. See "
            "docs/how-to-run.md §6.2."
        ),
    )
    parser.add_argument(
        "model_dir",
        type=Path,
        nargs="?",
        help="原始浮点 checkpoint 目录（--list-methods 时可省略）",
    )
    parser.add_argument(
        "--method",
        help=(
            "方法名（如 int8_dynamic）或注册键（如 VLLM_HUST_KV_INT8_DYNAMIC）；"
            "--list-methods 查看全部"
        ),
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        help="覆盖层数（config.json 缺 num_hidden_layers 时必填，如 VLM）",
    )
    parser.add_argument(
        "--layer-prefix",
        default="model",
        help="层名前缀（缺省 model；transformer 系模型传 transformer）",
    )
    parser.add_argument(
        "--include-lm-head",
        action="store_true",
        help="额外写入 lm_head.weight=FLOAT（缺省不写）",
    )
    parser.add_argument(
        "--float-module",
        action="append",
        default=[],
        metavar="SUFFIX",
        help="额外的保持浮点模块后缀（可重复；如 mlp.shared_expert.gate_proj）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="覆盖已有的外部 ascend 量化描述（非 ascend 配置仍拒绝）",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="只打印将写入的描述，不落盘"
    )
    parser.add_argument("--restore", action="store_true", help="用备份回滚 config.json")
    parser.add_argument(
        "--check",
        action="store_true",
        help=(
            "serve 前校验：注入清单 + config.json 哈希 + 描述一致性；"
            "打印 JSON 报告，valid 时退出 0 否则 2"
        ),
    )
    parser.add_argument(
        "--list-methods", action="store_true", help="列出全部方法后退出"
    )
    args = parser.parse_args(argv)

    try:
        if args.list_methods:
            for name in known_method_names():
                print(name)
            return 0
        if args.restore:
            if args.model_dir is None:
                parser.error("model_dir is required for --restore")
            info = restore(args.model_dir)
            print(f"[vllm-hust-kv-inject] restored {info['restored']}")
            return 0
        if args.check:
            if args.model_dir is None:
                parser.error("model_dir is required for --check")
            report = check_checkpoint(args.model_dir)
            print(json.dumps(report, ensure_ascii=False, indent=2))
            return 0 if report["valid"] else 2
        if args.model_dir is None:
            parser.error("model_dir is required (or use --list-methods)")
        if not args.method:
            parser.error("--method is required (or use --list-methods / --restore)")
        info = inject(
            args.model_dir,
            method=args.method,
            num_layers=args.num_layers,
            layer_prefix=args.layer_prefix,
            include_lm_head=args.include_lm_head,
            extra_float_modules=tuple(args.float_module),
            dry_run=args.dry_run,
            force=args.force,
        )
    except (ValueError, RuntimeError, OSError, json.JSONDecodeError) as exc:
        print(f"[vllm-hust-kv-inject] error: {exc}", file=sys.stderr)
        return 2

    mode = "DRY-RUN（未落盘）" if info["dry_run"] else "已写入"
    print(
        f"[vllm-hust-kv-inject] {mode} config.json: "
        f"method={info['method']} fa_quant_type={info['fa_quant_type']} "
        f"layers={info['num_layers']}"
    )
    print(
        "  每层内容: fa_k/fa_v.scale (FAQuant) + "
        f"{len(DEFAULT_FLOAT_MODULES)} 个线性层 FLOAT + embed_tokens FLOAT"
    )
    if info["replaced_existing"]:
        print("  (--force 覆盖了已有的外部 ascend 量化描述)")
    if not info["dry_run"]:
        print(f"  备份: {info['backup']}（--restore 回滚）")
        print(f"  契约清单: {info['manifest']}（serve 前 --check 校验）")
    print(
        "下一步：注册 scheme 用 VLLM_HUST_KV_METHODS 环境变量（§6.1）；"
        "serve 前用 --check 复核 checkpoint；"
        "int8 存储另需 serve 期 --kv-cache-dtype（§6.4/§8.1）"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
