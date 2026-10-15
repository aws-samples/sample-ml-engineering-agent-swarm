"""Tests for dox_lint.py: enforce a DOX AGENTS.md tree (https://github.com/agent0ai/dox).

Run: scripts/run_tests.sh
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import dox_lint as dl  # noqa: E402

CHILD = """# {name}

## Purpose
x
## Ownership
x
## Local Contracts
x
## Work Guidance

## Verification

## Child DOX Index
{index}
"""

ROOT = """# DOX framework

- DOX is highly performant AGENTS.md hierarchy installed here

## Child DOX Index

{index}
"""


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_repo(tmp_path: Path, root_index: str = "", with_child: bool = True) -> Path:
    repo = tmp_path / "repo"
    write(repo / "crates" / "agentd" / "Cargo.toml", "[package]\nname='agentd'\n")
    write(repo / "crates" / "agentd" / "src" / "main.rs", "fn main() {}\n")
    if with_child:
        write(repo / "crates" / "AGENTS.md", CHILD.format(name="crates", index="| `agentd/AGENTS.md` | supervisor |"))
        write(repo / "crates" / "agentd" / "AGENTS.md", CHILD.format(name="agentd", index=""))
    write(repo / "AGENTS.md", ROOT.format(index=root_index))
    return repo


def test_complete_tree_passes(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | Rust workspace |")
    result = dl.lint(repo)
    assert result.ok, result.errors


def test_missing_root_agents_md_fails(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    (repo / "AGENTS.md").unlink()
    result = dl.lint(repo)
    assert not result.ok
    assert any("root AGENTS.md" in e for e in result.errors)


def test_root_must_carry_the_dox_framework_and_index(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    write(repo / "AGENTS.md", "# Project\n\nno framework here\n")
    result = dl.lint(repo)
    assert not result.ok
    assert any("DOX framework" in e for e in result.errors)
    assert any("Child DOX Index" in e for e in result.errors)


def test_directory_with_manifest_requires_agents_md(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    (repo / "crates" / "agentd" / "AGENTS.md").unlink()
    write(repo / "crates" / "AGENTS.md", CHILD.format(name="crates", index=""))
    result = dl.lint(repo)
    assert not result.ok
    assert any("crates/agentd" in e and "missing AGENTS.md" in e for e in result.errors)


def test_top_level_directory_with_files_requires_agents_md(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    write(repo / "scripts" / "deploy.sh", "#!/bin/sh\n")
    result = dl.lint(repo)
    assert not result.ok
    assert any(e.startswith("scripts") or "scripts:" in e for e in result.errors)


def test_child_not_listed_in_parent_index_fails(tmp_path):
    repo = make_repo(tmp_path, root_index="")
    result = dl.lint(repo)
    assert not result.ok
    assert any("crates/AGENTS.md" in e and "not listed" in e for e in result.errors)


def test_index_entry_pointing_to_missing_file_fails(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |\n| `infra/AGENTS.md` | gone |")
    result = dl.lint(repo)
    assert not result.ok
    assert any("infra/AGENTS.md" in e and "does not exist" in e for e in result.errors)


def test_child_sections_must_all_exist_in_order(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    bad = CHILD.format(name="agentd", index="").replace("## Ownership\nx\n", "").replace(
        "## Purpose", "## Verification\n\n## Purpose", 1
    )
    write(repo / "crates" / "agentd" / "AGENTS.md", bad)
    result = dl.lint(repo)
    assert not result.ok
    joined = " ".join(result.errors)
    assert "Ownership" in joined
    assert "order" in joined


def test_generated_and_hidden_directories_are_ignored(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    for junk in ["target/debug/x.rs", "node_modules/pkg/package.json", ".git/config", "cdk.out/a.json", "dist/b.js"]:
        write(repo / junk, "x\n")
    result = dl.lint(repo)
    assert result.ok, result.errors


def test_cli_exit_codes_and_output(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    ok = subprocess.run([sys.executable, str(SCRIPTS / "dox_lint.py"), str(repo)], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert ok.stdout.strip().splitlines()[-1] == "DOX PASS"
    (repo / "crates" / "agentd" / "AGENTS.md").unlink()
    bad = subprocess.run([sys.executable, str(SCRIPTS / "dox_lint.py"), str(repo)], capture_output=True, text=True)
    assert bad.returncode == 1
    assert bad.stdout.strip().splitlines()[-1] == "DOX FAIL"


def test_a_fixture_repository_under_fixtures_needs_no_agents_md(tmp_path):
    # fixtures/ is test data owned by fixtures/AGENTS.md: a fixture repo's own manifest (for
    # example fixtures/python-calc/repo/pyproject.toml in cline-agentcore-harness) is not a
    # project package and must not demand its own AGENTS.md.
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |\n| `fixtures/AGENTS.md` | test data |")
    write(repo / "fixtures" / "AGENTS.md", CHILD.format(name="fixtures", index=""))
    write(repo / "fixtures" / "python-calc" / "repo" / "pyproject.toml", "[project]\nname='calc'\n")
    result = dl.lint(repo)
    assert result.ok, result.errors


def test_a_manifest_outside_fixtures_still_needs_agents_md(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    write(repo / "crates" / "fixtures-like" / "Cargo.toml", "[package]\nname='x'\n")
    result = dl.lint(repo)
    assert not result.ok
    assert any("crates/fixtures-like" in e and "missing AGENTS.md" in e for e in result.errors)


def test_recursive_mode_rejects_undocumented_source_directory(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    assert dl.lint(repo).ok
    result = dl.lint(repo, recursive=True)
    assert not result.ok
    assert any("crates/agentd/src" in e for e in result.errors)


def test_recursive_mode_covers_intermediate_maintained_directories(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    write(repo / "crates/agentd/src/AGENTS.md", CHILD.format(name="src", index=""))
    write(repo / "crates/agentd/AGENTS.md", CHILD.format(name="agentd", index="| `src/AGENTS.md` | source |"))
    write(repo / "crates/agentd/src/nested/leaf/code.py", "x = 1\n")
    result = dl.lint(repo, recursive=True)
    assert not result.ok
    assert any("src/nested:" in e for e in result.errors)
    assert any("src/nested/leaf:" in e for e in result.errors)


def test_recursive_mode_keeps_fixture_and_generated_exclusions(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |\n| `fixtures/AGENTS.md` | data |")
    write(repo / "crates/agentd/src/AGENTS.md", CHILD.format(name="src", index=""))
    write(repo / "crates/agentd/AGENTS.md", CHILD.format(name="agentd", index="| `src/AGENTS.md` | source |"))
    write(repo / "fixtures/AGENTS.md", CHILD.format(name="fixtures", index=""))
    write(repo / "fixtures/example/repo/package.json", "{}\n")
    write(repo / "build/generated/code.py", "x = 1\n")
    assert dl.lint(repo, recursive=True).ok


def test_recursive_cli_rejects_missing_nested_docs(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |")
    result = subprocess.run([sys.executable, str(SCRIPTS / "dox_lint.py"), str(repo), "--recursive"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "crates/agentd/src" in result.stdout


def test_child_index_rejects_substring_collision(tmp_path):
    repo = tmp_path / "collision"
    write(repo / "AGENTS.md", ROOT.format(index="| `not-foo/AGENTS.md` | listed |"))
    for name in ["foo", "not-foo"]:
        write(repo / name / "AGENTS.md", CHILD.format(name=name, index=""))
    assert not dl.lint(repo).ok
    assert any("foo/AGENTS.md: not listed" in x for x in dl.lint(repo).errors)


def test_child_index_must_be_in_nearest_parent(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |\n| `crates/agentd/AGENTS.md` | wrong parent |")
    write(repo / "crates/AGENTS.md", CHILD.format(name="crates", index=""))
    assert not dl.lint(repo).ok
    assert any("crates/agentd/AGENTS.md: not listed" in x for x in dl.lint(repo).errors)


def test_recursive_fixture_docs_are_opaque_data(tmp_path):
    repo = make_repo(tmp_path, root_index="| `crates/AGENTS.md` | x |\n| `fixtures/AGENTS.md` | data |")
    write(repo / "crates/agentd/src/AGENTS.md", CHILD.format(name="src", index=""))
    write(repo / "crates/agentd/AGENTS.md", CHILD.format(name="agentd", index="| `src/AGENTS.md` | source |"))
    write(repo / "fixtures/AGENTS.md", CHILD.format(name="fixtures", index=""))
    write(repo / "fixtures/repo/AGENTS.md", "fixture input; deliberately not a project document\n")
    assert dl.lint(repo, recursive=True).ok


def test_recursive_fixture_owner_cannot_disappear_with_its_index(tmp_path):
    repo = tmp_path / "fixture-owner"
    write(repo / "AGENTS.md", ROOT.format(index=""))
    write(repo / "fixtures/example/package.json", "{}\n")
    result = dl.lint(repo, recursive=True)
    assert not result.ok
    assert any("fixtures: missing AGENTS.md" in x for x in result.errors)
