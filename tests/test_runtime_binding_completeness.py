"""Generated runtime bindings must resolve without host dotenv or runtime SSM reads."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from test_runtime_deployment_advisory import run_runtime, runtime_calls
from test_runtime_deployment_advisory import runtime_workspace as runtime_workspace
from test_shell_entrypoint_advisory import ACCOUNT
from test_shell_entrypoint_advisory import shell_workspace as shell_workspace

from amlas.config.externalized import resolve


def state(workspace):
    return json.loads(Path(workspace[1]["TEST_STATE"]).read_text())


def test_generated_runtime_has_verified_lake_guardrail_and_owned_job_tag_grants(
    runtime_workspace, monkeypatch
):
    runtime_workspace[1].update(
        AMLAS_DEPLOYMENT_PREFIX="binding-blue",
        AMLAS_SSM_PREFIX="/reader-blue",
        AGENTCORE_RUNTIME_NAME="binding_blue",
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = state(runtime_workspace)
    environment = actual["environmentVariables"]
    expected = {
        "lake_bucket": f"binding-blue-lake-{ACCOUNT}-us-east-1",
        "lake_silver_prefix": f"s3://binding-blue-lake-{ACCOUNT}-us-east-1/silver/",
        "lake_gold_prefix": f"s3://binding-blue-lake-{ACCOUNT}-us-east-1/gold/",
        "guardrail_id": "guardrailex01",
        "guardrail_version": "DRAFT",
        "feature_store_db": "binding_blue_features",
    }
    for key in tuple(os.environ):
        if key.startswith(("AMLAS_", "AGENTICML_", "AWS_")):
            monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)

    def forbidden_ssm(*args):
        raise AssertionError("Generated environment omitted a required stack binding")

    for key, value in expected.items():
        assert resolve(key, ssm_getter=forbidden_ssm) == value
    statements = {item["Sid"]: item for item in actual["policy"]["Statement"]}
    assert statements["ApplySelectedGuardrail"]["Action"] == ["bedrock:ApplyGuardrail"]
    assert (
        statements["ApplySelectedGuardrail"]["Resource"]
        == f"arn:aws:bedrock:us-east-1:{ACCOUNT}:guardrail/guardrailex01"
    )
    assert statements["SageMakerJobTags"]["Action"] == ["sagemaker:AddTags"]
    assert statements["SageMakerJobTags"]["Resource"] == [
        f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:training-job/binding-blue-*",
        f"arn:aws:sagemaker:us-east-1:{ACCOUNT}:processing-job/binding-blue-*",
    ]
    assert statements["ReadHarnessIds"]["Resource"].endswith("parameter/reader-blue/harness/*")


def test_custom_secret_and_model_group_are_selected_in_generated_environment_and_exact_policy(
    runtime_workspace,
):
    runtime_workspace[1].update(
        TAVILY_SECRET_ID="amlas/research-key",
        AMLAS_MODEL_PACKAGE_GROUP="chosen-models",
        AMLAS_FEATURE_STORE_DB="chosen_features",
    )
    result = run_runtime(runtime_workspace)
    assert result.returncode == 0, result.stdout + result.stderr
    actual = state(runtime_workspace)
    environment = actual["environmentVariables"]
    assert environment.get("AMLAS_TAVILY_SECRET_ID") == "amlas/research-key"
    assert environment["AMLAS_MODEL_PACKAGE_GROUP"] == "chosen-models"
    assert environment["AMLAS_FEATURE_STORE_DB"] == "chosen_features"
    secret = next(item for item in actual["policy"]["Statement"] if item["Sid"] == "TavilySecretRead")
    assert (
        secret["Resource"] == f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:amlas/research-key-ABC123"
    )
    assert secret["Action"] == ["secretsmanager:GetSecretValue", "secretsmanager:DescribeSecret"]


@pytest.mark.parametrize(
    "missing",
    ["lake-silver", "lake-gold", "guardrail-id", "guardrail-version", "guardrail-arn", "secret-read"],
)
def test_intended_unverified_binding_refuses_role_and_runtime_creation(runtime_workspace, missing):
    runtime_workspace[1]["TEST_BINDING_FAILURE"] = missing
    result = run_runtime(runtime_workspace)
    assert result.returncode != 0, result.stdout + result.stderr
    assert not any(
        call["args"][:2]
        in (
            ["cloudformation", "create-stack"],
            ["iam", "create-role"],
            ["iam", "put-role-policy"],
            ["bedrock-agentcore-control", "create-agent-runtime"],
            ["bedrock-agentcore-control", "update-agent-runtime"],
        )
        and "--generate-cli-skeleton" not in call["args"]
        for call in runtime_calls(runtime_workspace)
    )
    assert "AGENTCORE_RUNTIME_DEPLOY_DONE" not in result.stdout
