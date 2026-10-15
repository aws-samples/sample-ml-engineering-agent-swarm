#!/usr/bin/env python3
"""Check a repository's DOX AGENTS.md tree (https://github.com/agent0ai/dox).

Rules:
  1. The root AGENTS.md exists, carries the DOX framework ("# DOX framework"), and has a
     "## Child DOX Index" section.
  2. Every durable boundary has an AGENTS.md: each top-level directory that holds files, and
     each directory with a package manifest (Cargo.toml, package.json, pyproject.toml, ...),
     except fixture repositories nested under the top-level data root fixtures/, which are test
     data owned by fixtures/AGENTS.md.
  3. Every child AGENTS.md is listed in the Child DOX Index of its nearest parent AGENTS.md.
  4. Every index entry points to an existing AGENTS.md.
  5. Child docs carry the DOX sections in order: Purpose, Ownership, Local Contracts,
     Work Guidance, Verification, Child DOX Index.

Generated and hidden directories are skipped. Prints one line per problem and a final
"DOX PASS" or "DOX FAIL". Exit code 0 on pass, 1 on fail.

Usage: dox_lint.py [REPO_ROOT] [--recursive]
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

DOC = "AGENTS.md"
FRAMEWORK_MARKER = "# DOX framework"
INDEX_HEADING = "## Child DOX Index"
SECTIONS = ["Purpose", "Ownership", "Local Contracts", "Work Guidance", "Verification", "Child DOX Index"]
MANIFESTS = {
    "Cargo.toml", "package.json", "pyproject.toml", "setup.py", "go.mod",
    "pom.xml", "build.gradle", "build.gradle.kts",
}
IGNORED = {
    "node_modules", "target", "dist", "build", "cdk.out", "venv", "__pycache__",
    "vendor", "coverage", "out",
}
# Top-level data roots: manifests below them belong to fixture repositories, not the project.
DATA_ROOTS = {"fixtures"}
ENTRY = re.compile(r"[\w.\-]+(?:/[\w.\-]+)*/AGENTS\.md")


@dataclass
class Result:
    ok: bool
    errors: list = field(default_factory=list)


def _skip(name: str) -> bool:
    return name.startswith(".") or name in IGNORED


def _index_section(text: str) -> str:
    start = text.find(INDEX_HEADING)
    if start == -1:
        return ""
    rest = text[start + len(INDEX_HEADING):]
    nxt = re.search(r"^## ", rest, re.M)
    return rest[: nxt.start()] if nxt else rest


def _walk(root: Path):
    for current, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not _skip(d))
        yield Path(current), files


def lint(root: Path | str, *, recursive: bool = False) -> Result:
    root = Path(root)
    errors: list = []
    root_doc = root / DOC
    if not root_doc.is_file():
        return Result(ok=False, errors=[f"missing root AGENTS.md at {root_doc}"])
    root_text = root_doc.read_text(encoding="utf-8")
    if FRAMEWORK_MARKER not in root_text:
        errors.append(f"root AGENTS.md lacks the DOX framework section ('{FRAMEWORK_MARKER}')")
    if INDEX_HEADING not in root_text:
        errors.append(f"root AGENTS.md lacks '{INDEX_HEADING}'")

    docs: list = []
    has_files: dict = {}
    durable: set[Path] = set()
    for current, files in _walk(root):
        rel = current.relative_to(root)
        if recursive and len(rel.parts) >= 2 and rel.parts[0] in DATA_ROOTS:
            if files:
                has_files[rel.parts[0]] = True
                durable.add(root / rel.parts[0])
            continue
        if DOC in files and current != root:
            docs.append(current / DOC)
        if rel.parts:
            top = rel.parts[0]
            if files:
                has_files[top] = True
        if recursive and files and rel.parts:
            ancestor = current
            while ancestor != root:
                relative = ancestor.relative_to(root)
                if not (len(relative.parts) >= 2 and relative.parts[0] in DATA_ROOTS):
                    durable.add(ancestor)
                ancestor = ancestor.parent
        manifest = MANIFESTS.intersection(files)
        in_data_root = len(rel.parts) >= 2 and rel.parts[0] in DATA_ROOTS
        if current != root and manifest and DOC not in files and not in_data_root:
            errors.append(f"{rel.as_posix()}: missing AGENTS.md (package manifest {sorted(manifest)[0]})")

    for top in sorted(has_files):
        if not (root / top / DOC).is_file():
            message = f"{top}: missing AGENTS.md (top-level directory)"
            if message.split(" (")[0] + ":" not in " ".join(errors):
                errors.append(message)

    if recursive:
        for directory in sorted(durable):
            relative = directory.relative_to(root).as_posix()
            if not (directory / DOC).is_file():
                errors.append(f"{relative}: missing AGENTS.md (recursive durable directory)")

    for doc in sorted(docs):
        rel = doc.relative_to(root).as_posix()
        text = doc.read_text(encoding="utf-8")
        positions = []
        for name in SECTIONS:
            match = re.search(rf"^## {re.escape(name)}\s*$", text, re.M)
            if match is None:
                errors.append(f"{rel}: missing section '## {name}'")
            else:
                positions.append(match.start())
        if positions != sorted(positions):
            errors.append(f"{rel}: DOX sections out of order (expected {', '.join(SECTIONS)})")

        parent = doc.parent.parent
        while parent != root and not (parent / DOC).is_file():
            parent = parent.parent
        parent_doc = parent / DOC
        index = _index_section(parent_doc.read_text(encoding="utf-8"))
        from_parent = doc.relative_to(parent).as_posix()
        indexed_paths = set(ENTRY.findall(index))
        if from_parent not in indexed_paths and rel not in indexed_paths:
            errors.append(f"{rel}: not listed in the Child DOX Index of {parent_doc.relative_to(root).as_posix()}")

    for doc in [root_doc, *sorted(docs)]:
        for entry in ENTRY.findall(_index_section(doc.read_text(encoding="utf-8"))):
            if not (doc.parent / entry).is_file() and not (root / entry).is_file():
                errors.append(f"{doc.relative_to(root).as_posix()}: index entry {entry} does not exist")

    return Result(ok=not errors, errors=errors)


def main(argv: list | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_root", nargs="?", type=Path, default=Path.cwd())
    parser.add_argument("--recursive", action="store_true", help="Require docs for all maintained directories and their parents")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    result = lint(args.repo_root, recursive=args.recursive)
    for error in result.errors:
        print(f"DOX FAIL: {error}")
    print("DOX PASS" if result.ok else "DOX FAIL")
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
