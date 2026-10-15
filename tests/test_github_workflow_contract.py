"""Keep the release runner able to execute the same complete locked test cohort."""

import re
import shlex
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / ".github" / "workflows" / "public-release.yml"


def _verify_job():
    return yaml.safe_load(WORKFLOW.read_text())["jobs"]["verify"]


def _run_commands():
    return [step["run"] for step in _verify_job()["steps"] if "run" in step]


def test_release_provisions_the_declared_full_test_extras_and_node():
    """Native collection imports CDK; the full cohort includes managed CPU APIs."""
    commands = _run_commands()
    sync = next(command for command in commands if command.startswith("uv sync "))
    tokens = shlex.split(sync)
    extras = {tokens[i + 1] for i, token in enumerate(tokens) if token == "--extra"}
    assert {"dev", "security", "cdk", "agents", "runtime-cpu"} <= extras
    assert "--frozen" in tokens
    steps = _verify_job()["steps"]
    node = next((step for step in steps if step.get("uses", "").startswith("actions/setup-node@")), None)
    assert node is not None, "CDK/jsii must have an explicit supported Node prerequisite"
    assert re.fullmatch(r"actions/setup-node@[a-f0-9]{40}", node["uses"])
    assert str(node["with"]["node-version"]) == "22"
    assert steps.index(node) < next(i for i, step in enumerate(steps) if step.get("run") == sync)


def test_release_uses_a_scoped_runner_temp_directory_before_tests():
    """Ownership tests need TMPDIR; source checkout must stay separate from output."""
    job = _verify_job()
    value = job.get("env", {}).get("TMPDIR", "")
    # GitHub's context-availability table excludes runner from jobs.<job_id>.env.
    # https://docs.github.com/en/actions/reference/workflows-and-actions/contexts
    contexts = re.findall(r"\$\{\{\s*([a-z_]+)\.", value)
    assert contexts and set(contexts) <= {
        "github",
        "needs",
        "strategy",
        "matrix",
        "vars",
        "secrets",
        "inputs",
    }
    assert value == "${{ github.workspace }}/../amlas-public-release"
    checkout = Path("/workspace/repository")
    resolved = Path(value.replace("${{ github.workspace }}", str(checkout))).resolve()
    assert resolved != checkout and checkout not in resolved.parents
    commands = _run_commands()
    prepare = next((i for i, command in enumerate(commands) if 'mkdir -p "$TMPDIR"' in command), None)
    tests = next(i for i, command in enumerate(commands) if "pytest" in command)
    assert prepare is not None and prepare < tests


def test_release_executes_module_pytest_without_dropping_provisioned_extras():
    """Console pytest omitted repo-root imports and uv resync can discard CDK."""
    commands = _run_commands()
    tests = next(command for command in commands if "pytest" in command)
    tokens = shlex.split(tests)
    assert tokens[:6] == ["uv", "run", "--frozen", "--no-sync", "python", "-m"]
    assert tokens[6:] == ["pytest", "-q", "tests"]
    for command in commands:
        if "uv run " in command:
            assert "uv run --frozen --no-sync " in command


def test_release_keeps_required_security_checks_and_read_only_permissions():
    job = _verify_job()
    commands = "\n".join(_run_commands())
    for check in (
        "verify_repository_governance.py",
        "ruff check",
        "bandit",
        "pip-audit",
        "detect-secrets-hook",
        "cyclonedx-py",
    ):
        assert check in commands
    assert yaml.safe_load(WORKFLOW.read_text())["permissions"] == {"contents": "read"}
    assert not job.get("continue-on-error", False)
    assert all(not step.get("continue-on-error", False) for step in job["steps"])
