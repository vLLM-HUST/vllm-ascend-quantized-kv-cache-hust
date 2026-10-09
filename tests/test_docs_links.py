# SPDX-License-Identifier: Apache-2.0
"""Documentation links and paths must resolve.

The docs layer gets pruned and reorganised regularly, and a stale pointer is
invisible to every other gate: prose links are never executed, and a module
docstring that names a deleted file still imports fine.  This test walks every
markdown file and checks two things a reader actually relies on:

* every relative ``[text](target)`` link points at a file that exists;
* every backticked repository path (``docs/...``, ``scripts/...``, ``src/...``,
  ``tests/...``, ``provenance/...``) exists.

Anchors are followed too: ``file.md#section`` requires the target file *and* a
heading whose slug contains the anchor.

Backticked repository paths are checked only in the pages that hand the reader
a command to run (README, how-to-run, the host contract and the INT4
integration checklist), because those pages claim a file exists *here*. Design
prose elsewhere cites files by role (and by another branch's layout), so
checking it would report intent as breakage.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    "__pycache__",
    "node_modules",
    "dist",
    "build",
    ".tmp",
    ".venv",
    ".release-smoke",
}
MD_FILES = sorted(
    p
    for p in ROOT.rglob("*.md")
    if not any(part.startswith(".") or part in SKIP_DIRS for part in p.parts)
)

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
INLINE_PATH_RE = re.compile(
    r"`((?:docs|scripts|src|tests|provenance)/[^`\s]+?\.(?:md|py|sh|json|txt|yaml|yml|toml))`"
)
TOPDIR_PATH_RE = re.compile(r"`((?:README|HOST_CONTRACT|PROVENANCE|MAINTAINERS)\.md)`")
EXTERNAL = re.compile(r"^(https?://|mailto:|#)")
HEADING_RE = re.compile(r"^#{1,6}\s+(.+?)\s*$", re.M)
SLUG_STRIP_RE = re.compile(r"[^\w\u4e00-\u9fff-]+")


def _slug(heading_text: str) -> str:
    return SLUG_STRIP_RE.sub("-", heading_text.strip().lower()).strip("-")


def _anchors(md_text: str) -> set[str]:
    return {_slug(m.group(1)) for m in HEADING_RE.finditer(md_text)}


def test_every_markdown_file_is_checked() -> None:
    """Guard against the walk itself silently matching nothing."""
    names = {p.name for p in MD_FILES}
    assert {"README.md", "HOST_CONTRACT.md", "PROVENANCE.md"} <= names
    assert "kvquant-survey.md" in names


EXTERNAL_OWNER = (
    "不在本仓库",
    "dev 分支",
    "那个仓",
    "website",
    "vllm-hust-benchmark",
    "bench 树",
    "/root/",
    "服务器",
)

ANY_REPO_PATH_RE = re.compile(
    r"`((?:scripts|tests)/[^`\s]+?\.(?:py|sh))`",
)


def test_external_script_citations_say_who_owns_them() -> None:
    """A path that does not exist here must be labelled with its real repo.

    The benchmark and website tooling lives in other repositories, and docs
    cited it as bare ``scripts/foo.sh``.  A reader in this repo would run it,
    get "No such file", and have no way to tell whether the doc is stale or the
    tool is simply elsewhere.  So: either the path exists here, or the same
    line (or the one above it) names the owning repository.
    """
    unlabelled: list[str] = []
    for md in MD_FILES:
        lines = md.read_text(encoding="utf-8").splitlines()
        for n, line in enumerate(lines):
            for candidate in ANY_REPO_PATH_RE.findall(line):
                if "*" in candidate or "..." in candidate:
                    continue  # a glob a command expands
                if (ROOT / candidate).exists():
                    continue
                context = line + (lines[n - 1] if n else "")
                if not any(marker in context for marker in EXTERNAL_OWNER):
                    unlabelled.append(f"{md.relative_to(ROOT)}:{n + 1}: `{candidate}`")
    assert not unlabelled, (
        "citations of files that live in another repo, without saying so:\n"
        + "\n".join(sorted(unlabelled))
    )


# Package modules a doc may cite that legitimately do not live on this branch.
# Each entry states whose file it is, so a rename on *our* side still fails the
# check while an honest cross-repo reference does not.
EXTERNAL_MODULE_OWNERS = {
    "methods/my_quant/__init__.py": "template name for a method not written yet",
    "methods/my_quant/semantics.py": "template name for a method not written yet",
    "methods/packed_base.py": "dev branch",
    "adapters/vllm_hust/register.py": "dev branch",
    "ops/int4_per_token_head.py": "upstream vLLM",
    "ops/flydsl_turboquant_decode.py": "upstream vLLM (AMD FlyDSL)",
    "ops/triton/kivi_cache.py": "legacy vllm-ascend PR #116 layout",
    "ops/int8_ops.py": "legacy vllm-ascend PR #116 layout",
    "ops/triton_turboquant_store.py": "upstream vLLM",
    "ops/triton_turboquant_decode.py": "upstream vLLM",
}
PACKAGE_ROOT = (
    Path(__file__).resolve().parents[1] / "src/vllm_ascend_quantized_kv_cache"
)
BRACE_RE = re.compile(r"\{(\w[\w,]*)\}")


def _expand(path: str) -> list[str]:
    m = BRACE_RE.search(path)
    if not m:
        return [path]
    head, tail = path[: m.start()], path[m.end() :]
    out: list[str] = []
    for part in m.group(1).split(","):
        out.extend(_expand(f"{head}{part}{tail}"))
    return out


def test_cited_module_paths_exist_or_are_labelled() -> None:
    """A doc that names ``methods/...`` must point at a real module.

    The doc layer was written against another branch's layout (the int8/KIVI
    device mixins were called ``attention_mixin.py`` there, and
    ``ops/int8_ops.py`` is a legacy host path), so readers were told to open
    files this tree does not contain.  Brace forms like ``{register,backend}``
    are expanded and checked.
    """
    MODULE_RE = re.compile(r"`((?:adapters|core|methods|ops)/[A-Za-z0-9_./{},-]+\.py)`")
    wrong: list[str] = []
    for md in MD_FILES:
        text = md.read_text(encoding="utf-8")
        for n, line in enumerate(text.splitlines(), start=1):
            for candidate in MODULE_RE.findall(line):
                if "*" in candidate:
                    continue
                for expanded in _expand(candidate):
                    if (PACKAGE_ROOT / expanded).exists():
                        continue
                    if expanded in EXTERNAL_MODULE_OWNERS:
                        continue
                    wrong.append(f"{md.relative_to(ROOT)}:{n}: `{candidate}`")
    assert not wrong, (
        "cited modules that exist neither here nor in the labelled list:\n"
        + "\n".join(sorted(set(wrong)))
    )


def test_relative_markdown_links_resolve() -> None:
    broken: list[str] = []
    for md in MD_FILES:
        text = md.read_text(encoding="utf-8")
        anchors = _anchors(text)
        for raw in LINK_RE.findall(text):
            if EXTERNAL.match(raw):
                continue
            file_part, _, anchor = raw.partition("#")
            if not file_part:
                if anchor and anchor not in anchors:
                    broken.append(f"{md.relative_to(ROOT)}: in-page anchor #{anchor}")
                continue
            target = (md.parent / file_part).resolve()
            if not target.exists():
                broken.append(f"{md.relative_to(ROOT)}: -> {file_part}")
                continue
            if anchor and target.suffix == ".md":
                target_anchors = _anchors(target.read_text(encoding="utf-8"))
                if anchor and anchor not in target_anchors:
                    broken.append(
                        f"{md.relative_to(ROOT)}: #{anchor} is not a heading "
                        f"of {file_part}"
                    )
    assert not broken, "dangling markdown links:\n" + "\n".join(sorted(broken))


RUNBOOK_DOCS = (
    "README.md",
    "HOST_CONTRACT.md",
    "PROVENANCE.md",
    "docs/how-to-run.md",
    "docs/int4-host-integration.md",
)


def test_backticked_paths_in_runbooks_exist() -> None:
    missing: list[str] = []
    for name in RUNBOOK_DOCS:
        md = ROOT / name
        text = md.read_text(encoding="utf-8")
        for candidate in set(
            INLINE_PATH_RE.findall(text) + TOPDIR_PATH_RE.findall(text)
        ):
            if "*" in candidate or "..." in candidate:
                continue  # a glob a command expands, not a single file name
            if (ROOT / candidate).exists():
                continue
            missing.append(f"{name}: `{candidate}`")
    assert not missing, "runbook paths that do not exist:\n" + "\n".join(
        sorted(missing)
    )
