"""Closed-boundary acceptance for explicit regional runtime ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import calls, invoke
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace


def test_setup_accepts_runtime_and_role_selectors(shell_workspace: tuple[Path, dict[str, str], Path]) -> None:
    repo, env, _ = shell_workspace
    env.update(AGENTCORE_RUNTIME_NAME="environment_runtime", AGENTCORE_EXEC_ROLE_NAME="environment-role")
    result = invoke(
        shell_workspace,
        "setup.sh",
        "--yes",
        "--skip-deps",
        "--preflight-only",
        "--skip-harness",
        "--runtime-name",
        "selected_runtime",
        "--runtime-role-name",
        "selected-role",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    text = (repo / ".env").read_text()
    assert "AGENTCORE_RUNTIME_NAME=selected_runtime" in text
    assert "AGENTCORE_EXEC_ROLE_NAME=selected-role" in text


@pytest.mark.parametrize(
    "flag,value", [("--runtime-name", "invalid-name"), ("--runtime-role-name", "invalid role")]
)
def test_invalid_runtime_names_fail_before_tools(
    shell_workspace: tuple[Path, dict[str, str], Path], flag: str, value: str
) -> None:
    result = invoke(shell_workspace, "setup.sh", "--yes", "--preflight-only", flag, value)
    assert result.returncode == 2
    assert not calls(shell_workspace)


def test_conflicting_runtime_flags_fail_before_tools(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = invoke(
        shell_workspace,
        "setup.sh",
        "--yes",
        "--preflight-only",
        "--runtime-name",
        "one",
        "--runtime-name",
        "two",
    )
    assert result.returncode == 2
    assert not calls(shell_workspace)


def test_default_runtime_role_is_region_scoped(runtime_workspace: tuple[Path, dict[str, str], Path]) -> None:
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    selected = next(
        call for call in runtime_calls(runtime_workspace) if call["args"][:2] == ["iam", "get-role"]
    )
    assert selected["args"][selected["args"].index("--role-name") + 1] == "amlas-runtime-exec-us-east-1"


def test_existing_cross_region_role_is_not_overwritten(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    _repo, env, _ = runtime_workspace
    env.update(AWS_REGION="us-west-1", AGENTCORE_EXEC_ROLE_NAME="historical-role")
    env["TEST_EXISTING_ROLE"] = "west2"
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0
    assert "distinct" in (result.stdout + result.stderr).lower()
    forbidden = {
        ("iam", "update-assume-role-policy"),
        ("iam", "put-role-policy"),
        ("iam", "create-role"),
        ("bedrock-agentcore-control", "create-agent-runtime"),
        ("bedrock-agentcore-control", "update-agent-runtime"),
    }
    assert not any(
        tuple(call["args"][:2]) in forbidden and "--generate-cli-skeleton" not in call["args"]
        for call in runtime_calls(runtime_workspace)
    )


def test_compatible_existing_role_is_reused_without_iam_writes(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    first = run_runtime(runtime_workspace)
    assert first.returncode == 0, first.stdout + first.stderr
    runtime_workspace[2].write_text("")
    runtime_workspace[1]["TEST_EXISTING_ROLE"] = "captured"
    second = run_runtime(runtime_workspace)
    assert second.returncode == 0, second.stdout + second.stderr
    assert not any(
        call["args"][:2]
        in (["iam", "create-role"], ["iam", "put-role-policy"], ["iam", "update-assume-role-policy"])
        for call in runtime_calls(runtime_workspace)
    )


def test_main_setup_passes_selected_runtime_and_role_to_helper(
    shell_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    repo, env, _ = shell_workspace
    tools = Path(env["PATH"].split(":")[0])
    tool = tools / "aws"
    source = tool.read_text().replace(
        ' print("UNEXPECTED-AWS-CALL", file=sys.stderr); sys.exit(9)',
        ' if args[:2] == ["ssm", "describe-parameters"]: print("0");sys.exit(0)\n'
        ' if args[:2] == ["ssm", "put-parameter"]: assert "--tags" in args and "--overwrite" not in args;print(json.dumps({"Version":1}));sys.exit(0)\n'
        ' if args[:2] == ["ssm", "list-tags-for-resource"]: print(json.dumps({"TagList":[]}));sys.exit(0)\n'
        ' if args[0] == "secretsmanager": print("None");sys.exit(0)\n'
        ' print("UNEXPECTED-AWS-CALL", file=sys.stderr); sys.exit(9)',
    )
    tool.write_text(source)
    # These tests isolate selectors/key preservation/publication; ledger ownership
    # and foreign-resource refusal are verified in their native boundary suite.
    (repo / "scripts/lib/deployment_manifest.py").write_text(
        '#!/usr/bin/env python3\nimport json,sys\nif sys.argv[1]=="init": print(json.dumps({"deployment_id":"aaaaaaaa-aaaa-4aaa-8aaa-aaaaaaaaaaaa"}))\nelse: print(json.dumps({"resources":[],"refresh_errors":[],"remaining":[]}))\n'
    )
    (repo / "scripts/lib/build_runtime_image.sh").write_text("#!/usr/bin/env bash\nexit 0\n")
    (repo / "scripts/lib/deploy_agentcore_runtime.sh").write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        'printf "SELECTED=%s,%s\\n" "$AGENTCORE_RUNTIME_NAME" "$AGENTCORE_EXEC_ROLE_NAME"\n'
        'printf "AGENTCORE_RUNTIME_ARN=arn:aws:bedrock-agentcore:us-east-1:111122223333:runtime/example\\nFINAL_STATUS=READY\\nAGENTCORE_RUNTIME_DEPLOY_DONE\\n"\n'
    )
    env.update(AGENTCORE_RUNTIME_NAME="env_runtime", AGENTCORE_EXEC_ROLE_NAME="env-role")
    result = invoke(
        shell_workspace,
        "setup.sh",
        "--yes",
        "--skip-deps",
        "--agent-runtime-only",
        "--skip-harness",
        "--runtime-name",
        "selected_runtime",
        "--runtime-role-name",
        "selected-role",
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "SELECTED=selected_runtime,selected-role" in result.stdout
