from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _version_tuple(value: str) -> tuple[int, ...]:
    match = re.match(r"^(\d+(?:\.\d+)*)", value)
    assert match is not None, f"unparseable version: {value}"
    return tuple(int(part) for part in match.group(1).split("."))


def test_repository_authority_policy_is_explicit() -> None:
    policy = (ROOT / "AGENTS.md").read_text(encoding="utf-8")
    required_statements = (
        "GitHub `main` is the authoritative public release tree.",
        "GitLab is the development system of record for branches and review.",
        "GitLab `main` must match the exact human-approved GitHub `main` tree.",
        "Preserve every other GitLab branch, tag, and historical commit.",
        "No change reaches GitHub until a human approves the sanitized public snapshot.",
    )
    for statement in required_statements:
        assert statement in policy


def test_anyio_security_floor_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    floor = next(
        (dependency for dependency in dependencies if dependency.startswith("anyio>=")),
        None,
    )
    assert floor is not None
    assert _version_tuple(floor.removeprefix("anyio>=")) >= (4, 14, 2)


def test_anyio_lock_is_patched() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    anyio = next(package for package in lock["package"] if package["name"] == "anyio")
    assert _version_tuple(anyio["version"]) >= (4, 14, 2)


def test_gitpython_security_floor_is_declared() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = pyproject["project"]["dependencies"]
    floor = next(
        (dependency for dependency in dependencies if dependency.startswith("gitpython>=")),
        None,
    )
    assert floor is not None
    assert _version_tuple(floor.removeprefix("gitpython>=")) >= (3, 1, 60)


def test_gitpython_lock_is_patched() -> None:
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    gitpython = next(package for package in lock["package"] if package["name"] == "gitpython")
    assert _version_tuple(gitpython["version"]) >= (3, 1, 60)


def test_native_verification_extras_and_configuration_are_preserved() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = project["project"]["optional-dependencies"]
    assert {"pytest>=8.4.2", "ruff>=0.6"}.issubset(set(extras.get("dev", [])))
    assert {
        "bandit>=1.7",
        "cyclonedx-bom>=4.0",
        "detect-secrets>=1.5",
        "pip-audit>=2.7",
    }.issubset(set(extras.get("security", [])))
    assert project["tool"]["bandit"]["exclude_dirs"] == [".venv", "build", "cdk.out", "dist"]
