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

MD_FILES = sorted(
    p
    for p in ROOT.rglob("*.md")
    if ".git" not in p.parts and "__pycache__" not in p.parts
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
