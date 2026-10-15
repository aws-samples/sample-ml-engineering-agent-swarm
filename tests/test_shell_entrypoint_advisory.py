"""Public-safe, closed-boundary regressions for the deployment entrypoints."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
ACCOUNT = "111122223333"  # AWS documentation account; never a deployment target.


@pytest.fixture
def shell_workspace(tmp_path: Path) -> tuple[Path, dict[str, str], Path]:
    repo = tmp_path / "project"
    repo.mkdir()
    shutil.copytree(ROOT / "scripts", repo / "scripts")
    shutil.copy2(ROOT / ".env.example", repo / ".env.example")
    (repo / ".env").write_text(f"AMLAS_AWS_ACCOUNT_ID={ACCOUNT}\nAWS_REGION=us-east-1\n")
    tools = tmp_path / "tools"
    tools.mkdir()
    runtime_tmp = tmp_path / "runtime-tmp"
    runtime_tmp.mkdir()
    runtime_cache = tmp_path / "runtime-cache"
    runtime_cache.mkdir()
    calls = tmp_path / "calls.jsonl"
    driver = """import json, os, sys
from pathlib import Path
name = Path(sys.argv[0]).name
args = sys.argv[1:]
raw_args = list(args)
if name == "aws":
 while args and args[0].startswith("--profile="): args = args[1:]
with open(os.environ["TEST_CALLS"], "a") as stream:
 stream.write(json.dumps({"tool": name, "args": args, "raw_args": raw_args, "profile": os.environ.get("AWS_PROFILE")}) + "\\n")
if name == "aws":
 if args[:1] == ["--version"]: print("aws fixture"); sys.exit(0)
 if args[:3] == ["configure", "get", "region"]: print("us-east-1"); sys.exit(0)
 if args[:2] == ["iam", "get-role"]: print("NoSuchEntity",file=sys.stderr);sys.exit(254)
 if args[:2] == ["sts", "get-caller-identity"]:
  output = args[args.index("--output") + 1] if "--output" in args else "text"
  print(json.dumps({"Account": os.environ["TEST_ACCOUNT"]}) if output == "json" else os.environ["TEST_ACCOUNT"]); sys.exit(0)
 if args[:2] == ["bedrock", "get-inference-profile"]: print("ACTIVE"); sys.exit(0)
 if args[:2] == ["cloudformation", "describe-type"]: print("FULLY_MUTABLE"); sys.exit(0)
 if args[:2] == ["bedrock-agentcore-control", "update-agent-runtime"] and "--generate-cli-skeleton" in args:
  print(json.dumps({"agentRuntimeId":""} if os.environ.get("TEST_CLI_MISSING") == "1" else {"agentRuntimeId":"", "metadataConfiguration":{"requireMMDSV2":False}, "platformVersion":""}));sys.exit(0)
 if args[:2] == ["bedrock-agentcore-control", "update-agent-runtime-endpoint"] and "--generate-cli-skeleton" in args:
  print('{"agentRuntimeId":"", "endpointName":"", "agentRuntimeVersion":""}');sys.exit(0)
 if args[:2] == ["bedrock-agentcore-control", "list-agent-runtimes"]: print('{"agentRuntimes": []}'); sys.exit(0)
 if args[:2] == ["cloudtrail", "list-trails"]: print('{"Trails": []}'); sys.exit(0)
 if args[:2] == ["cloudformation", "describe-stacks"]: print("CREATE_COMPLETE"); sys.exit(0)
 if args[:2] == ["sagemaker", "list-training-jobs"]: print('{"TrainingJobSummaries": []}'); sys.exit(0)
 if args[:2] == ["sagemaker", "list-processing-jobs"]: print('{"ProcessingJobSummaries": []}'); sys.exit(0)
 if args[:2] in (["s3api", "list-buckets"], ["kms", "list-keys"]): print(""); sys.exit(0)
 print("UNEXPECTED-AWS-CALL", file=sys.stderr); sys.exit(9)
if name == "uv":
 if args and args[0] == "--version": print("uv fixture"); sys.exit(0)
 if args and args[0] == "sync": sys.exit(0)
 if args[:2] == ["run", "--frozen"]: args = [args[0], *args[2:]]
 if args[:2] == ["run", "cdk"]: os.execvp("cdk", ["cdk", *args[2:]])
 if args[:2] == ["run", "python"]: os.execv(os.environ["TEST_PYTHON"], [os.environ["TEST_PYTHON"], *args[2:]])
 sys.exit(9)
if name == "cdk":
 if args[:1] in (["--version"], ["synth"]): print("cdk fixture"); sys.exit(0)
 sys.exit(9)
if name == "docker": print("Docker intentionally unavailable", file=sys.stderr); sys.exit(77)
sys.exit(9)
"""
    for name in ("aws", "uv", "cdk", "docker"):
        tool = tools / name
        tool.write_text(f"#!{sys.executable}\n" + driver)
        tool.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AWS_", "AMLAS_", "AGENTICML_", "CDK_"))}
    env.update(
        PATH=f"{tools}:{env['PATH']}",
        TMPDIR=str(runtime_tmp),
        XDG_CACHE_HOME=str(runtime_cache),
        TEST_CALLS=str(calls),
        TEST_ACCOUNT=ACCOUNT,
        TEST_PYTHON=sys.executable,
        AWS_EC2_METADATA_DISABLED="true",
        PYTHONPATH=f"{ROOT / 'src'}:{ROOT}",
    )
    return repo, env, calls


def invoke(
    workspace: tuple[Path, dict[str, str], Path], script: str, *args: str
) -> subprocess.CompletedProcess[str]:
    repo, env, _ = workspace
    return subprocess.run(
        ["bash", str(repo / "scripts" / script), *args],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )


def calls(workspace: tuple[Path, dict[str, str], Path]) -> list[dict]:
    path = workspace[2]
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def test_incompatible_read_only_modes_fail_before_any_tool(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = invoke(shell_workspace, "setup.sh", "--yes", "--synth-only", "--preflight-only")
    assert result.returncode == 2
    assert not calls(shell_workspace)


def test_default_credential_chain_setup_does_not_abort(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--synth-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert all(call["profile"] is None for call in calls(shell_workspace) if call["tool"] == "aws")


def test_default_credential_chain_teardown_preview_does_not_abort(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    repo, _, _ = shell_workspace
    manifest = repo / "output/deploy/deployment-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "deployment_id": "00000000-0000-0000-0000-aaaaaaaaaa10",
                "account": ACCOUNT,
                "region": "us-east-1",
                "prefix": "amlas",
                "profile": "",
                "selectors": {
                    "stacks": [],
                    "role_stack": "",
                    "role_name": "amlas-runtime-exec-us-east-1",
                    "secret_id": "amlas/tavily-api-key",
                    "runtime_name": "amlas",
                },
                "resources": {},
                "baseline": {},
                "sessions": [],
            }
        )
    )
    result = invoke(shell_workspace, "teardown.sh", "--yes", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    aws_calls = [call for call in calls(shell_workspace) if call["tool"] == "aws"]
    assert {tuple(call["args"][:2]) for call in aws_calls} == {
        ("sts", "get-caller-identity"),
        ("sagemaker", "list-training-jobs"),
        ("sagemaker", "list-processing-jobs"),
    }
    assert all(call["profile"] is None and "--profile" not in call["raw_args"] for call in aws_calls)


def test_teardown_uses_persisted_scoped_cli_without_installing(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    repo, env, _ = shell_workspace
    tools = Path(env["PATH"].split(":")[0])
    scoped = repo.parent / "setup-tool-bin"
    scoped.mkdir()
    (tools / "aws").rename(scoped / "aws")
    (tools / "aws").write_text("#!/usr/bin/env bash\necho 'wrong AWS CLI selected' >&2\nexit 78\n")
    (tools / "aws").chmod(0o755)
    with (repo / ".env").open("a") as stream:
        stream.write("AMLAS_DEPLOY_TOOL_BIN=" + str(scoped) + "\n")
    manifest = repo / "output/deploy/deployment-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "schema": 1,
                "deployment_id": "00000000-0000-0000-0000-aaaaaaaaaa10",
                "account": ACCOUNT,
                "region": "us-east-1",
                "prefix": "amlas",
                "profile": "",
                "tool_bin": str(scoped),
                "selectors": {
                    "stacks": [],
                    "role_stack": "",
                    "role_name": "amlas-runtime-exec-us-east-1",
                    "secret_id": "amlas/tavily-api-key",
                    "runtime_name": "amlas",
                    "ssm_prefix": "/amlas",
                    "model_group": "amlas-models",
                },
                "resources": {},
                "baseline": {},
                "sessions": [],
            }
        )
    )
    result = invoke(shell_workspace, "teardown.sh", "--yes", "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "wrong AWS CLI" not in result.stdout + result.stderr
    assert all(call["tool"] == "aws" for call in calls(shell_workspace))


def test_teardown_preview_rejects_missing_manifest_before_any_aws_call(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = invoke(shell_workspace, "teardown.sh", "--yes", "--dry-run")
    assert result.returncode != 0
    assert "deployment manifest is required" in (result.stdout + result.stderr).lower()
    assert not calls(shell_workspace)


def test_preflight_does_not_require_docker_and_still_checks_runtime(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    repo, _, _ = shell_workspace
    (repo / ".env").write_text(
        f"AMLAS_AWS_ACCOUNT_ID={ACCOUNT}\nAWS_PROFILE=example-profile\nAWS_REGION=us-east-1\n"
    )
    result = invoke(shell_workspace, "setup.sh", "--yes", "--preflight-only", "--skip-harness")
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any(call["tool"] == "docker" for call in calls(shell_workspace))
    assert any(
        call["args"][:2] == ["bedrock-agentcore-control", "list-agent-runtimes"]
        for call in calls(shell_workspace)
    )


def test_dotenv_rejects_entire_malformed_key_without_execution(
    shell_workspace: tuple[Path, dict[str, str], Path], tmp_path: Path
) -> None:
    repo, env, _ = shell_workspace
    sentinel = tmp_path / "harmless-sentinel"
    env["TEST_SENTINEL"] = str(sentinel)
    (repo / ".env").write_text("AA[$(touch${IFS}${TEST_SENTINEL})]=value\n")
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1"; load_env_file "$2"',
            "test",
            str(repo / "scripts/lib/deploy_common.sh"),
            str(repo),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not sentinel.exists()


def test_dotenv_accepts_single_character_key_and_literal_value(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    repo, env, _ = shell_workspace
    value = """literal "quote" 'quote' $(printf harmless) `printf harmless` \\ dollar$ comma, equals= end"""
    (repo / ".env").write_text("X='" + value + "'\n")
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1"; load_env_file "$2"; printf "VALUE=%s\\n" "${X:-}"',
            "test",
            str(repo / "scripts/lib/deploy_common.sh"),
            str(repo),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "VALUE=" + value in result.stdout


def test_preflight_rejects_missing_cli_features(shell_workspace: tuple[Path, dict[str, str], Path]) -> None:
    shell_workspace[1]["TEST_CLI_MISSING"] = "1"
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--preflight-only", "--skip-harness")
    assert result.returncode != 0
    assert "upgrade" in (result.stdout + result.stderr).lower()
    assert not any(
        call["args"][:2]
        in (
            ["secretsmanager", "create-secret"],
            ["secretsmanager", "put-secret-value"],
            ["ssm", "put-parameter"],
        )
        for call in calls(shell_workspace)
    )


def test_skip_runtime_omits_only_its_cli_feature_check(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    shell_workspace[1]["TEST_CLI_MISSING"] = "1"
    result = invoke(
        shell_workspace,
        "setup.sh",
        "--yes",
        "--skip-deps",
        "--preflight-only",
        "--skip-agent-runtime",
        "--skip-harness",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not any("--generate-cli-skeleton" in call["args"] for call in calls(shell_workspace))
    assert any(call["args"][:2] == ["bedrock", "get-inference-profile"] for call in calls(shell_workspace))


@pytest.mark.parametrize("key", ["KEY bogus", "KEY\tbogus", "KEY[0]", "KEY.bad", "0KEY"])
def test_dotenv_checks_entire_identifier(
    shell_workspace: tuple[Path, dict[str, str], Path], key: str
) -> None:
    repo, env, _ = shell_workspace
    (repo / ".env").write_text(key + "=unexpected\nGOOD=kept\n")
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1"; load_env_file "$2"; printf "KEY=%s GOOD=%s\\n" "${KEY:-}" "${GOOD:-}"',
            "test",
            str(repo / "scripts/lib/deploy_common.sh"),
            str(repo),
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "KEY= GOOD=kept" in result.stdout


@pytest.mark.parametrize("alias", ["AWS_DEFAULT_PROFILE", "AGENTICML_AWS_PROFILE"])
def test_supported_profile_alias_reaches_setup_and_persistence(
    shell_workspace: tuple[Path, dict[str, str], Path], alias: str
) -> None:
    shell_workspace[1][alias] = "example-profile"
    result = invoke(shell_workspace, "setup.sh", "--yes", "--skip-deps", "--synth-only")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "AWS_PROFILE=example-profile" in (shell_workspace[0] / ".env").read_text()
    assert all(
        call["profile"] == "example-profile" for call in calls(shell_workspace) if call["tool"] == "aws"
    )
