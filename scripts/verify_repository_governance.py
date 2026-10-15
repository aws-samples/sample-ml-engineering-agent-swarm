#!/usr/bin/env python3
"""Verify public-release safety and GitHub/GitLab main-tree equivalence."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple


class Finding(NamedTuple):
    path: str
    code: str
    detail: str


PUBLIC_AWS_ACCOUNT_IDS = {
    "000000000000",  # Deliberately invalid placeholder.
    "111122223333",  # AWS documentation example.
    "081040173940",
    "141502667606",
    "159807026194",
    "246618743249",
    "257758044811",
    "266458841044",
    "306415355426",
    "432418664414",
    "477331159723",
    "568282634449",
    "683313688378",
    "763104351884",
    # Regional AWS SageMaker publisher accounts used by the five image catalogs.
    # Verified against aws/sagemaker-python-sdk commit
    # 0495265923351eb4c0a3ceaa635acabe2892b050 and its image_uri_config JSON files.
    "001633400207",
    "007051062584",
    "017069133835",
    "048819808253",
    "094389454867",
    "098760798382",
    "104374241257",
    "106583098589",
    "108575199400",
    "121021644041",
    "122526803553",
    "122578899357",
    "124985052026",
    "126357580389",
    "131013547314",
    "156813124566",
    "190319476487",
    "204538143572",
    "205585389593",
    "211330385671",
    "217643126080",
    "237065988967",
    "245545462676",
    "263625296855",
    "272398656194",
    "281123927165",
    "303241398832",
    "341280168497",
    "341593696636",
    "354813040037",
    "364406365360",
    "366743142698",
    "377024640650",
    "380420809688",
    "411392592546",
    "414596584902",
    "437450045455",
    "440796970383",
    "442386744353",
    "446045086412",
    "450853457545",
    "451049120500",
    "452307495513",
    "453000072557",
    "453252182341",
    "454834333376",
    "457447274322",
    "467912361380",
    "468650794304",
    "492215442770",
    "503227376785",
    "510948584623",
    "520018980103",
    "536280801234",
    "539772159869",
    "550225433462",
    "563025443158",
    "574779866223",
    "579539705040",
    "588750061953",
    "590183813437",
    "590183933784",
    "598674086554",
    "607024016150",
    "626614931356",
    "628508329040",
    "633930458069",
    "637423239942",
    "638885417683",
    "651117190479",
    "659782779980",
    "662702820516",
    "669540362728",
    "675030665977",
    "680080141114",
    "680994064768",
    "692866216735",
    "705930551576",
    "709848358524",
    "720646828776",
    "727897471807",
    "730335477804",
    "737474898029",
    "740489534195",
    "746614075791",
    "749857270468",
    "763603941244",
    "764974769150",
    "772153158452",
    "777275614652",
    "780543022126",
    "783357654285",
    "801668240914",
    "811711786498",
    "833128469047",
    "834264404009",
    "835444307964",
    "843974653677",
    "853188333426",
    "867004704886",
    "871362719292",
    "875698925577",
    "886529160074",
    "890145073186",
    "895015795356",
    "898809789911",
    "907027046896",
    "912233562940",
    "914824155844",
    "933208885752",
    "951798379941",
    "975050140332",
    "978288397137",
    "990339680094",
}

# Keep organization-specific hosts, groups and identities in an external release denylist.
INTERNAL_PATTERNS = (
    (
        "internal-host",
        re.compile(r"(?i)\b(?:[a-z0-9-]+[.])*(?:corp|internal)[.][a-z0-9.-]+"),
    ),
    (
        "personal-path",
        re.compile(
            r"/" + r"Users/[^/\s]+|"
            r"(?<!~)/" + r"home/(?!user(?:/|\b)|appuser(?:/|\b))[^/\s]+"
        ),
    ),
)

SECRET_PATTERNS = (
    re.compile(
        r"(?:AKIA|ASIA|A3T[A-Z0-9]|AGPA|AIDA|AROA|AIPA|ANPA|ANVA)"
        r"[A-Z0-9]{12,}"
    ),
    re.compile(r"ghp_[A-Za-z0-9]{20,}"),
    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"glpat-[A-Za-z0-9_-]{12,}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"tvly-[A-Za-z0-9_-]{8,}"),
    re.compile(r"-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----"),
)

FORBIDDEN_TOP_LEVEL = {
    "artifacts",
    "checkpoints",
    "data",
    "datasets",
    "models",
    "papers",
}

FORBIDDEN_SUFFIXES = {
    ".arrow",
    ".ckpt",
    ".csv",
    ".db",
    ".docx",
    ".feather",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ipynb",
    ".joblib",
    ".jpeg",
    ".jpg",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".pmml",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
    ".tflite",
    ".tsv",
    ".webp",
    ".whl",
    ".xlsx",
    ".zip",
}

FORBIDDEN_PATH_PARTS = {
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "build",
    "cdk.out",
    "dist",
    "output",
    "runs",
    "tmp",
}

REQUIRED_PUBLIC_PATHS = {
    ".github/workflows/public-release.yml",
    ".gitlab-ci.yml",
    ".secrets.baseline",
    "AGENTS.md",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "scripts/verify_repository_governance.py",
    "tests/test_repository_governance.py",
    "tests/test_repository_policy.py",
}


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def scan_text(path: str, text: str, private_literals: Iterable[str] = ()) -> list[Finding]:
    findings: list[Finding] = []

    if any(value.casefold() in text.casefold() for value in private_literals if value):
        findings.append(Finding(path, "private-literal", "contains a private denylist value"))

    for code, pattern in INTERNAL_PATTERNS:
        if pattern.search(text):
            findings.append(Finding(path, code, "contains a private coordinate or identifier"))

    for pattern in SECRET_PATTERNS:
        for match in pattern.finditer(text):
            value = match.group(0).lower()
            if re.fullmatch(r"tvly-x{8,}", value):
                continue
            findings.append(Finding(path, "secret-token", "contains a credential-shaped value"))
            break

    # Only validated integrity-digest fields and public package hash paths are exempt.
    # Comments, URLs and all other lockfile fields still receive account-ID checks.
    account_text = re.sub(
        r'(\bhash\s*=\s*")(?:sha256:[a-f0-9]{64}|sha384:[a-f0-9]{96}|sha512:[a-f0-9]{128})(?=")',
        r"\1<integrity-digest>",
        text,
        flags=re.IGNORECASE,
    )
    account_text = re.sub(
        r"https://files[.]pythonhosted[.]org/packages/[a-f0-9]{2}/[a-f0-9]{2}/[a-f0-9]{60}/",
        "<public-package-hash>/",
        account_text,
        flags=re.IGNORECASE,
    )
    for account_id in re.findall(r"(?<!\d)\d{12}(?!\d)", account_text):
        if account_id not in PUBLIC_AWS_ACCOUNT_IDS:
            findings.append(Finding(path, "unknown-account-id", "contains an unreviewed 12-digit value"))

    return findings


def scan_path(path: str) -> list[Finding]:
    normalized = path.replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    parts = tuple(part for part in normalized.split("/") if part)
    findings: list[Finding] = []

    if normalized.casefold() in {".github/dependabot.yml", ".github/dependabot.yaml"}:
        findings.append(
            Finding(normalized, "public-write-automation", "automatic public PRs bypass approval")
        )

    if parts and parts[0].lower() in FORBIDDEN_TOP_LEVEL:
        findings.append(Finding(normalized, "internal-or-generated-path", "forbidden top-level path"))

    if any(part.lower() in FORBIDDEN_PATH_PARTS for part in parts):
        findings.append(Finding(normalized, "generated-artifact", "generated path is tracked"))

    if Path(normalized).suffix.lower() in FORBIDDEN_SUFFIXES:
        findings.append(Finding(normalized, "generated-artifact", "generated file type is tracked"))

    return findings


def missing_required_paths(repo: Path, required: Iterable[str]) -> list[str]:
    return sorted(path for path in required if not (repo / path).is_file())


def tracked_paths(repo: Path) -> list[str]:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return [item.decode("utf-8") for item in output.split(b"\0") if item]


def _candidate_entries(repo: Path, ref: str | None) -> list[tuple[str, str, str]]:
    if ref is None:
        args = ["ls-files", "--stage", "-z"]
    else:
        tree = _git(repo, "rev-parse", "--verify", f"{ref}^{{tree}}")
        args = ["ls-tree", "-r", "-z", tree]
    raw = subprocess.check_output(["git", *args], cwd=repo)
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, path = record.split(b"\t", 1)
        fields = metadata.decode("ascii").split()
        mode = fields[0]
        oid = fields[1] if ref is None else fields[2]
        if ref is None and fields[2] != "0":
            mode = "unmerged"
        entries.append((mode, oid, path.decode("utf-8")))
    return entries


def scan_repository(repo: Path, ref: str | None = None) -> list[Finding]:
    """Scan immutable index or commit blobs, never working-copy or symlink target bytes."""
    findings: list[Finding] = []
    denylist_path = os.environ.get("AMLAS_RELEASE_DENY_LITERALS_FILE")
    private_literals = []
    if denylist_path:
        private_literals = [
            line.strip()
            for line in Path(denylist_path).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
    entries = _candidate_entries(repo, ref)
    paths = {path for _, _, path in entries}
    for path in sorted(REQUIRED_PUBLIC_PATHS - paths):
        findings.append(Finding(path, "required-path-missing", "public governance file is absent"))
    for entry_number, (mode, oid, relative) in enumerate(entries, 1):
        path_findings = scan_text("", relative, private_literals)
        display_path = f"<redacted-path-{entry_number}>" if path_findings else relative
        findings.extend(
            Finding(display_path, item.code, "candidate path contains a prohibited identifier")
            for item in path_findings
        )
        findings.extend(item._replace(path=display_path) for item in scan_path(relative))
        exact_dox_alias = (
            mode == "120000"
            and relative == "CLAUDE.md"
            and any(target_mode == "100644" and target_path == "AGENTS.md" for target_mode, _, target_path in entries)
            and subprocess.check_output(["git", "cat-file", "blob", oid], cwd=repo) == b"AGENTS.md"
        )
        if mode not in {"100644", "100755"} and not exact_dox_alias:
            findings.append(
                Finding(display_path, "unsupported-git-mode", "symlink, submodule or unresolved entry")
            )
            continue
        size = int(_git(repo, "cat-file", "-s", oid))
        if size > 10 * 1024 * 1024:
            findings.append(Finding(display_path, "large-file", "tracked blob exceeds 10 MiB"))
            continue
        raw = subprocess.check_output(["git", "cat-file", "blob", oid], cwd=repo)
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(Finding(display_path, "binary-file", "tracked blob is not UTF-8 text"))
            continue
        findings.extend(scan_text(display_path, text, private_literals))
    return findings


def refs_have_equal_trees(repo: Path, left: str, right: str) -> bool:
    return _git(repo, "rev-parse", f"{left}^{{tree}}") == _git(repo, "rev-parse", f"{right}^{{tree}}")


def _print_findings(findings: Iterable[Finding]) -> None:
    for finding in findings:
        print(f"{finding.code}: {finding.path}: {finding.detail}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--scan-tree", action="store_true")
    parser.add_argument("--ref", help="Scan this immutable commit/tree; defaults to the staged index")
    parser.add_argument("--compare-refs", nargs=2, metavar=("GITLAB_REF", "GITHUB_REF"))
    args = parser.parse_args(argv)

    repo = args.repo.resolve()
    failed = False

    if args.scan_tree or not args.compare_refs:
        findings = scan_repository(repo, args.ref)
        if findings:
            _print_findings(findings)
            failed = True
        else:
            print(f"public_tree_scan=pass tracked_files={len(_candidate_entries(repo, args.ref))}")

    if args.compare_refs:
        left, right = args.compare_refs
        if refs_have_equal_trees(repo, left, right):
            tree = _git(repo, "rev-parse", f"{left}^{{tree}}")
            print(f"main_tree_sync=pass tree={tree}")
        else:
            left_tree = _git(repo, "rev-parse", f"{left}^{{tree}}")
            right_tree = _git(repo, "rev-parse", f"{right}^{{tree}}")
            print(
                f"main_tree_sync=fail {left}={left_tree} {right}={right_tree}",
                file=sys.stderr,
            )
            failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
