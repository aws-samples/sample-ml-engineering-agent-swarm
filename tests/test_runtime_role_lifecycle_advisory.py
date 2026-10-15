"""Closed-boundary runtime-role lifecycle regressions against the real shell helper.

The fake enforces CloudFormation named-IAM and service-role request fields. No AWS
credentials or live services are used; service-side lifecycle acceptance belongs to E2E.
"""

from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
from test_runtime_deployment_advisory import (
    run_runtime,
    runtime_calls,
)
from test_runtime_deployment_advisory import (
    runtime_workspace as runtime_workspace,
)
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace


def mutations(workspace: tuple[Path, dict[str, str], Path]) -> list[list[str]]:
    return [
        call["args"][:2]
        for call in runtime_calls(workspace)
        if call["args"][:2]
        in (
            ["iam", "create-role"],
            ["iam", "put-role-policy"],
            ["iam", "update-assume-role-policy"],
            ["cloudformation", "create-stack"],
            ["cloudformation", "update-stack"],
            ["bedrock-agentcore-control", "create-agent-runtime"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
        )
        and "--generate-cli-skeleton" not in call["args"]
    ]


def test_new_role_is_owned_by_a_named_iam_stack_with_standard_service_issuer(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = runtime_calls(runtime_workspace)
    creates = [call["args"] for call in calls if call["args"][:2] == ["cloudformation", "create-stack"]]
    assert len(creates) == 1, "New runtime IAM role must be CloudFormation-owned"
    args = creates[0]
    assert args[args.index("--role-arn") + 1] == (
        f"arn:aws:iam::{ACCOUNT}:role/cdk-hnb659fds-cfn-exec-role-{ACCOUNT}-us-east-1"
    )
    assert "CAPABILITY_NAMED_IAM" in args
    template = json.loads(args[args.index("--template-body") + 1])
    role = template["Resources"]["RuntimeExecutionRole"]["Properties"]
    assert role["RoleName"] == "amlas-runtime-exec-us-east-1"
    assert role["Policies"][0]["PolicyDocument"]["Statement"]
    assert {tag["Key"]: tag["Value"] for tag in role["Tags"]}["amlas:deployment-prefix"] == "amlas"
    assert template["Outputs"]["RuntimeRoleArn"]["Value"] == {"Fn::GetAtt": ["RuntimeExecutionRole", "Arn"]}
    assert "RESOLVED_RUNTIME_ROLE_STACK_ARN=" in result.stdout
    assert not any(action[0] == "iam" for action in mutations(runtime_workspace))


@pytest.mark.parametrize(
    "failure",
    [
        "bootstrap-read",
        "bootstrap-missing",
        "bootstrap-trust",
        "bootstrap-account",
        "bootstrap-boundary",
        "stack-collision",
    ],
)
def test_unverified_standard_cleanup_route_fails_before_every_creation(
    runtime_workspace: tuple[Path, dict[str, str], Path], failure: str
) -> None:
    runtime_workspace[1]["TEST_ROUTE_FAILURE"] = failure
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not mutations(runtime_workspace)
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout


@pytest.mark.parametrize("failure", ["cfn-create", "cfn-wait", "cfn-readback", "cfn-role-readback"])
def test_failed_role_stack_never_reaches_managed_runtime_writes(
    runtime_workspace: tuple[Path, dict[str, str], Path], failure: str
) -> None:
    runtime_workspace[1]["TEST_FAILURE"] = failure
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not any(
        action[0] in ("iam", "bedrock-agentcore-control") for action in mutations(runtime_workspace)
    )
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout
    if failure != "cfn-create":
        assert "RESOLVED_RUNTIME_ROLE_STACK_ARN=" in result.stdout


def test_matching_existing_role_preserves_trust_grants_and_lifecycle_stack(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    first = run_runtime(runtime_workspace)
    assert first.returncode == 0, first.stdout + first.stderr
    state_file = Path(runtime_workspace[1]["TEST_STATE"])
    before = json.loads(state_file.read_text())
    prior_calls = len(runtime_calls(runtime_workspace))
    runtime_workspace[1]["TEST_EXISTING_ROLE"] = "captured"
    second = run_runtime(runtime_workspace)
    assert second.returncode == 0, second.stdout + second.stderr
    after = json.loads(state_file.read_text())
    assert before["trust"] == after["trust"]
    assert before["policy"] == after["policy"]
    later_calls = runtime_calls(runtime_workspace)[prior_calls:]
    assert not any(
        call["args"][:2]
        in (
            ["iam", "create-role"],
            ["iam", "put-role-policy"],
            ["cloudformation", "create-stack"],
            ["cloudformation", "update-stack"],
        )
        for call in later_calls
    )


def test_namespace_reaches_image_resource_grants_and_managed_configuration(
    runtime_workspace: tuple[Path, dict[str, str], Path],
) -> None:
    runtime_workspace[1].update(
        AMLAS_DEPLOYMENT_PREFIX="acceptance",
        AMLAS_SSM_PREFIX="/acceptance",
        AGENTCORE_RUNTIME_NAME="acceptance_runtime",
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    state = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())
    statements = {statement["Sid"]: statement for statement in state["policy"]["Statement"]}
    assert statements["EcrPullImage"]["Resource"].endswith("repository/acceptance-runtime")
    assert statements["PassSageMakerRole"]["Resource"].endswith("role/acceptance-sagemaker-exec")
    assert all("table/acceptance-training-jobs" in value for value in statements["DdbJobs"]["Resource"])
    assert statements["ReadHarnessIds"]["Resource"].endswith("parameter/acceptance/harness/*")
    assert (
        statements["TavilySecretRead"]["Resource"]
        == f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:acceptance/tavily-api-key-ABC123"
    )
    assert state["agentRuntimeArtifact"]["containerConfiguration"]["containerUri"].endswith(
        "/acceptance-runtime:v1"
    )
    assert state["environmentVariables"]["AMLAS_DEPLOYMENT_PREFIX"] == "acceptance"
    assert state["environmentVariables"]["AMLAS_SSM_PREFIX"] == "/acceptance"
    assert state["environmentVariables"]["AMLAS_JOBS_TABLE"] == "acceptance-training-jobs"
    assert state["environmentVariables"]["AMLAS_LAKE_BUCKET"] == f"acceptance-lake-{ACCOUNT}-us-east-1"
    calls = runtime_calls(runtime_workspace)
    assert any(
        call["args"][:2] == ["cloudformation", "describe-stacks"]
        and call["args"][call["args"].index("--stack-name") + 1] == "acceptance-AsyncTrainingStack"
        for call in calls
    )


def test_new_uuid_tags_are_atomic_on_role_stack_and_managed_runtime(runtime_workspace):
    deployment_id = str(uuid.uuid4())
    runtime_workspace[1].update(AMLAS_DEPLOYMENT_ID=deployment_id, TEST_CREATE="1")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = runtime_calls(runtime_workspace)
    stack_call = next(
        call["args"] for call in calls if call["args"][:2] == ["cloudformation", "create-stack"]
    )
    stack_tags = {tag["Key"]: tag["Value"] for tag in json.loads(stack_call[stack_call.index("--tags") + 1])}
    assert stack_tags.get("amlas:deployment-id") == deployment_id
    template = json.loads(stack_call[stack_call.index("--template-body") + 1])
    role_tags = {
        tag["Key"]: tag["Value"]
        for tag in template["Resources"]["RuntimeExecutionRole"]["Properties"]["Tags"]
    }
    assert role_tags.get("amlas:deployment-id") == deployment_id
    create_call = next(
        call["args"]
        for call in calls
        if call["args"][:2] == ["bedrock-agentcore-control", "create-agent-runtime"]
    )
    request = json.loads(create_call[create_call.index("--cli-input-json") + 1])
    assert request.get("tags") == {"amlas:deployment-prefix": "amlas", "amlas:deployment-id": deployment_id}
    assert request["environmentVariables"].get("AMLAS_DEPLOYMENT_ID") == deployment_id
    assert request["environmentVariables"].get("AMLAS_MODEL_PACKAGE_GROUP") == "amlas-models"
    assert request["environmentVariables"].get("AMLAS_FEATURE_STORE_DB") == "amlas_features"


def test_runtime_create_tags_are_atomic_with_a_reused_compatible_role(runtime_workspace):
    first = run_runtime(runtime_workspace)
    assert first.returncode == 0, first.stdout + first.stderr
    deployment_id = str(uuid.uuid4())
    runtime_workspace[1].update(
        AMLAS_DEPLOYMENT_ID=deployment_id, TEST_CREATE="1", TEST_EXISTING_ROLE="captured"
    )
    prior_calls = len(runtime_calls(runtime_workspace))
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    calls = runtime_calls(runtime_workspace)[prior_calls:]
    create_call = next(
        call["args"]
        for call in calls
        if call["args"][:2] == ["bedrock-agentcore-control", "create-agent-runtime"]
    )
    request = json.loads(create_call[create_call.index("--cli-input-json") + 1])
    assert request.get("tags") == {"amlas:deployment-prefix": "amlas", "amlas:deployment-id": deployment_id}
    assert not any(
        call["args"][:2]
        in (
            ["iam", "put-role-policy"],
            ["cloudformation", "create-stack"],
            ["cloudformation", "update-stack"],
        )
        for call in calls
    )


@pytest.mark.parametrize("prefix,request_uuid", [("amlas", True), ("acceptance", False)])
def test_foreign_runtime_uuid_is_refused_before_update(runtime_workspace, prefix, request_uuid):
    deployment_id = str(uuid.uuid4())
    runtime_workspace[1].update(
        AMLAS_DEPLOYMENT_PREFIX=prefix,
        AGENTCORE_RUNTIME_NAME="amlas",
        TEST_EXISTING_RUNTIME_TAGS=json.dumps(
            {"amlas:deployment-prefix": prefix, "amlas:deployment-id": str(uuid.uuid4())}
        ),
    )
    if request_uuid:
        runtime_workspace[1]["AMLAS_DEPLOYMENT_ID"] = deployment_id
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not any(
        call["args"][:2] == ["bedrock-agentcore-control", "update-agent-runtime"]
        and "--generate-cli-skeleton" not in call["args"]
        for call in runtime_calls(runtime_workspace)
    )
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout


def test_preflight_compatible_role_emits_native_reuse_identity_without_mutation(runtime_workspace):
    first = run_runtime(runtime_workspace)
    assert first.returncode == 0, first.stdout + first.stderr
    before = state_file = Path(runtime_workspace[1]["TEST_STATE"])
    original = json.loads(before.read_text())
    prior_calls = len(runtime_calls(runtime_workspace))
    runtime_workspace[1].update(TEST_EXISTING_ROLE="captured", AMLAS_RUNTIME_PREFLIGHT_ONLY="1")
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    assert (
        f"RESOLVED_REUSED_RUNTIME_ROLE_ARN=arn:aws:iam::{ACCOUNT}:role/amlas-runtime-exec-us-east-1"
        in result.stdout
    )
    assert "RESOLVED_REUSED_RUNTIME_ROLE_ID=" + "AROA" + "X" * 16 in result.stdout
    assert "RUNTIME-ROLE-PREFLIGHT PASS" in result.stdout
    assert json.loads(state_file.read_text())["trust"] == original["trust"]
    assert json.loads(state_file.read_text())["policy"] == original["policy"]
    later = runtime_calls(runtime_workspace)[prior_calls:]
    assert not any(
        call["args"][:2]
        in (
            ["cloudformation", "create-stack"],
            ["cloudformation", "update-stack"],
            ["iam", "put-role-policy"],
            ["iam", "update-assume-role-policy"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
        )
        and "--generate-cli-skeleton" not in call["args"]
        for call in later
    )


def test_explicit_tavily_keyless_opt_out_reaches_generated_runtime_environment(runtime_workspace):
    runtime_workspace[1]["AMLAS_TAVILY_KEYLESS"] = "false"
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = json.loads(Path(runtime_workspace[1]["TEST_STATE"]).read_text())
    assert actual["environmentVariables"].get("AMLAS_TAVILY_KEYLESS") == "false"
