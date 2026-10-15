"""Synthesize the real CDK app to prove deployment isolation without AWS calls."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENT_ID = str(uuid.uuid5(uuid.NAMESPACE_DNS, "deployment.example.invalid"))
STACKS = {
    "AgentCoreRuntimeStack",
    "LakeStack",
    "ModelRegistryStack",
    "SecretsStack",
    "BackendIamStack",
    "NetworkStack",
    "DetectionsStack",
    "GuardrailStack",
    "HarnessStack",
    "RuntimeStack",
    "AsyncTrainingStack",
}


@pytest.fixture(scope="module")
def assemblies(tmp_path_factory):
    results = {}
    for prefix in ("amlas", "trial-a", "trial-b", "a" * 24, "trial-custom", "trial-selected"):
        output = tmp_path_factory.mktemp("infra-" + prefix)
        env = dict(os.environ)
        for key in tuple(env):
            if key.startswith("AWS_") or key.startswith("AMLAS_") or key.startswith("AGENTICML_"):
                env.pop(key)
        env.update(
            AMLAS_DEPLOYMENT_PREFIX=prefix,
            AMLAS_SSM_PREFIX="",
            AMLAS_DEPLOYMENT_ID="",
            TAVILY_SECRET_ID="",
            AMLAS_MODEL_PACKAGE_GROUP="",
            CDK_DEFAULT_ACCOUNT="111122223333",
            CDK_DEFAULT_REGION="us-west-2",
            CDK_OUTDIR=str(output),
            AWS_EC2_METADATA_DISABLED="true",
            CDK_CONTEXT_JSON=json.dumps(
                {
                    "availability-zones:account=111122223333:region=us-west-2": ["us-west-2a", "us-west-2b"],
                }
            ),
        )
        if prefix == "trial-a":
            env["AMLAS_DEPLOYMENT_ID"] = DEPLOYMENT_ID
        if prefix == "trial-custom":
            env["AMLAS_SSM_PREFIX"] = "/custom-reader/"
        if prefix == "trial-selected":
            env["TAVILY_SECRET_ID"] = "trial-selected/research-key"
            env["AMLAS_MODEL_PACKAGE_GROUP"] = "chosen-models"
        completed = subprocess.run(
            [sys.executable, "-m", "infra.app"],
            cwd=ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        manifest = json.loads((output / "manifest.json").read_text())
        stacks = {
            key: value
            for key, value in manifest["artifacts"].items()
            if value["type"] == "aws:cloudformation:stack"
        }
        templates = {
            key: json.loads((output / value["properties"]["templateFile"]).read_text())
            for key, value in stacks.items()
        }
        results[prefix] = stacks, templates
    return results


def _resources(templates, resource_type):
    return [
        resource["Properties"]
        for template in templates.values()
        for resource in template["Resources"].values()
        if resource["Type"] == resource_type
    ]


def test_default_stack_selection_and_public_names_remain_compatible(assemblies):
    stacks, templates = assemblies["amlas"]
    assert set(stacks) == STACKS
    assert all(item["properties"].get("stackName", name) == name for name, item in stacks.items())
    assert _resources(templates, "AWS::ECR::Repository")[0]["RepositoryName"] == "amlas-runtime"
    assert _resources(templates, "AWS::Glue::Database")[0]["DatabaseInput"]["Name"] == "amlas_lake"
    assert (
        _resources(templates, "AWS::SageMaker::ModelPackageGroup")[0]["ModelPackageGroupName"]
        == "amlas-models"
    )
    assert all(item["Name"].startswith("/amlas/") for item in _resources(templates, "AWS::SSM::Parameter"))
    assert {item["HarnessName"] for item in _resources(templates, "AWS::BedrockAgentCore::Harness")} == {
        "research_lead",
        "fe_lead",
        "ml_execution_lead",
        "data_engineering_lead",
        "deploy_serving_lead",
        "mlops_lead",
        "monitoring_lead",
    }


@pytest.mark.parametrize("prefix", ["trial-a", "trial-b"])
def test_same_logical_stack_selection_has_distinct_physical_namespace(assemblies, prefix):
    stacks, templates = assemblies[prefix]
    assert set(stacks) == STACKS
    assert all(
        item["properties"].get("stackName", name) == prefix + "-" + name for name, item in stacks.items()
    )
    assert all(
        item["Name"].startswith("/" + prefix + "/") for item in _resources(templates, "AWS::SSM::Parameter")
    )
    assert (
        _resources(templates, "AWS::Glue::Database")[0]["DatabaseInput"]["Name"]
        == prefix.replace("-", "_") + "_lake"
    )
    assert (
        _resources(templates, "AWS::SageMaker::ModelPackageGroup")[0]["ModelPackageGroupName"]
        == prefix + "-models"
    )
    secret_ids = [
        item["Value"]
        for item in _resources(templates, "AWS::SSM::Parameter")
        if item["Name"].endswith("/tavily-secret-id")
    ]
    assert secret_ids == [prefix + "/tavily-api-key"]
    assert all(
        item["HarnessName"].startswith(prefix.replace("-", "_") + "_")
        for item in _resources(templates, "AWS::BedrockAgentCore::Harness")
    )
    exports = [
        item["Export"]["Name"]
        for template in templates.values()
        for item in template.get("Outputs", {}).values()
        if "Export" in item
    ]
    assert exports and all(name.startswith(prefix + "-") for name in exports)


def _named_resources(templates):
    fields = {
        "AWS::ECR::Repository": "RepositoryName",
        "AWS::S3::Bucket": "BucketName",
        "AWS::IAM::Role": "RoleName",
        "AWS::KMS::Alias": "AliasName",
        "AWS::SSM::Parameter": "Name",
        "AWS::Logs::LogGroup": "LogGroupName",
        "AWS::DynamoDB::Table": "TableName",
        "AWS::Lambda::Function": "FunctionName",
        "AWS::SNS::Topic": "TopicName",
        "AWS::CloudTrail::Trail": "TrailName",
        "AWS::CloudWatch::Alarm": "AlarmName",
        "AWS::Bedrock::Guardrail": "Name",
        "AWS::Events::Rule": "Name",
        "AWS::SageMaker::ModelPackageGroup": "ModelPackageGroupName",
        "AWS::BedrockAgentCore::Harness": "HarnessName",
        "AWS::BedrockAgentCore::Runtime": "AgentRuntimeName",
        "AWS::BedrockAgentCore::Memory": "Name",
        "AWS::BedrockAgentCore::CodeInterpreterCustom": "Name",
    }
    return {
        (resource["Type"], json.dumps(resource["Properties"][fields[resource["Type"]]], sort_keys=True))
        for template in templates.values()
        for resource in template["Resources"].values()
        if resource["Type"] in fields and fields[resource["Type"]] in resource.get("Properties", {})
    }


def test_two_deployments_do_not_share_explicit_resource_or_export_names(assemblies):
    _, first = assemblies["trial-a"]
    _, second = assemblies["trial-b"]
    assert _named_resources(first).isdisjoint(_named_resources(second))
    first_exports = {
        item["Export"]["Name"]
        for template in first.values()
        for item in template.get("Outputs", {}).values()
        if "Export" in item
    }
    second_exports = {
        item["Export"]["Name"]
        for template in second.values()
        for item in template.get("Outputs", {}).values()
        if "Export" in item
    }
    assert first_exports.isdisjoint(second_exports)
    first_metrics = _resources(first, "AWS::Logs::MetricFilter")
    second_metrics = _resources(second, "AWS::Logs::MetricFilter")
    assert {item["MetricTransformations"][0]["MetricNamespace"] for item in first_metrics}.isdisjoint(
        {item["MetricTransformations"][0]["MetricNamespace"] for item in second_metrics}
    )


def test_runtime_policy_and_backend_config_match_the_selected_resources(assemblies):
    _, templates = assemblies["trial-a"]
    runtime = _resources({"RuntimeStack": templates["RuntimeStack"]}, "AWS::BedrockAgentCore::Runtime")[0]
    env = runtime["EnvironmentVariables"]
    assert env.get("AMLAS_DEPLOYMENT_PREFIX") == "trial-a"
    assert env["AGENTICML_MODEL_PACKAGE_GROUP"] == "trial-a-models"
    assert env["AGENTICML_FEATURE_STORE_DB"] == "trial_a_features"
    assert env["TAVILY_SECRET_ID"] == "trial-a/tavily-api-key"
    policies = json.dumps(_resources(templates, "AWS::IAM::Policy"), sort_keys=True)
    assert "training-job/trial-a-*" in policies
    assert "secret:trial-a/tavily-api-key-*" in policies
    assert "training-job/amlas-*" not in policies
    assert "secret:amlas/tavily-api-key-*" not in policies


def test_namespaced_resources_carry_native_deployment_ownership(assemblies):
    _, templates = assemblies["trial-a"]
    taggable = [
        resource["Properties"]
        for template in templates.values()
        for resource in template["Resources"].values()
        if "Tags" in resource.get("Properties", {})
    ]
    assert len(taggable) >= 20
    for resource in taggable:
        tags = resource["Tags"]
        assert isinstance(tags, (dict, list))
        tag_map = tags if isinstance(tags, dict) else {tag["Key"]: tag["Value"] for tag in tags}
        assert tag_map.get("amlas:deployment-prefix") == "trial-a"
    # Native schemas use tag maps on these generic CfnResource types, so
    # app-level TagManager propagation must not silently omit their ownership.
    for resource_type in (
        "AWS::BedrockAgentCore::Runtime",
        "AWS::BedrockAgentCore::Memory",
        "AWS::BedrockAgentCore::CodeInterpreterCustom",
    ):
        for resource in _resources(templates, resource_type):
            assert resource.get("Tags", {}).get("amlas:deployment-prefix") == "trial-a"


def test_harness_wire_names_remain_valid_at_the_prefix_boundary(assemblies):
    prefix = "a" * 24
    _, templates = assemblies[prefix]
    harnesses = _resources(templates, "AWS::BedrockAgentCore::Harness")
    names = [item["HarnessName"] for item in harnesses]
    assert len(names) == len(set(names)) == 7
    assert all(name.startswith(prefix + "_") for name in names)
    # Live CloudFormation AWS::BedrockAgentCore::Harness schema, 2026-10-13:
    # properties/HarnessName uses ^[a-zA-Z][a-zA-Z0-9_]{0,39}$.
    assert all(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,39}", name) for name in names)
    assert all(
        {tag["Key"]: tag["Value"] for tag in item.get("Tags", [])}.get("amlas:deployment-prefix") == prefix
        for item in harnesses
    )


def test_shortened_harness_names_preserve_distinct_logical_leads():
    from infra.cdk_stacks import harness_stack

    name_for = getattr(harness_stack, "_harness_name", None)
    assert name_for is not None, "missing bounded harness naming behavior"
    prefix = "a" * 24
    # These collide under simple truncation, so the discriminator must retain
    # logical identity while respecting the native service's 40-character bound.
    names = [name_for(prefix, key) for key in ("data_engineering_lead", "data_engineering_other")]
    assert names[0] != names[1]
    assert all(name.startswith(prefix + "_") for name in names)
    assert all(re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_]{0,39}", name) for name in names)
    assert name_for("amlas", "data_engineering_lead") == "data_engineering_lead"


def test_explicit_ssm_namespace_matches_runtime_discovery(assemblies):
    _, templates = assemblies["trial-custom"]
    parameters = _resources(templates, "AWS::SSM::Parameter")
    assert parameters
    assert all(item["Name"].startswith("/custom-reader/") for item in parameters)


def test_supplied_deployment_identity_reaches_native_resource_tags(assemblies):
    _, templates = assemblies["trial-a"]
    resources = [resource for template in templates.values() for resource in template["Resources"].values()]
    for resource in resources:
        if "Tags" in resource.get("Properties", {}):
            tags = resource["Properties"]["Tags"]
            tag_map = tags if isinstance(tags, dict) else {tag["Key"]: tag["Value"] for tag in tags}
            assert tag_map.get("amlas:deployment-id") == DEPLOYMENT_ID
    for resource_type in (
        "AWS::BedrockAgentCore::Harness",
        "AWS::BedrockAgentCore::Runtime",
        "AWS::BedrockAgentCore::Memory",
        "AWS::BedrockAgentCore::CodeInterpreterCustom",
    ):
        for resource in _resources(templates, resource_type):
            tags = resource.get("Tags", {})
            tag_map = tags if isinstance(tags, dict) else {tag["Key"]: tag["Value"] for tag in tags}
            assert tag_map.get("amlas:deployment-id") == DEPLOYMENT_ID


def test_gold_table_cannot_be_created_before_its_glue_database(assemblies):
    _, templates = assemblies["amlas"]
    resources = templates["LakeStack"]["Resources"]
    databases = [key for key, item in resources.items() if item["Type"] == "AWS::Glue::Database"]
    tables = [item for item in resources.values() if item["Type"] == "AWS::Glue::Table"]
    assert len(databases) == len(tables) == 1
    database_id = databases[0]
    table = tables[0]
    dependency = table.get("DependsOn", [])
    dependency = [dependency] if isinstance(dependency, str) else dependency
    # CloudFormation schedules by Ref/GetAtt/DependsOn edges, not Python order.
    assert table["Properties"]["DatabaseName"] == {"Ref": database_id} or database_id in dependency


def test_selected_resource_identities_match_stack_publication_and_runtime(assemblies):
    _, templates = assemblies["trial-selected"]
    groups = _resources(templates, "AWS::SageMaker::ModelPackageGroup")
    assert len(groups) == 1
    assert groups[0]["ModelPackageGroupName"] == "chosen-models"
    parameters = {item["Name"]: item["Value"] for item in _resources(templates, "AWS::SSM::Parameter")}
    assert parameters["/trial-selected/model-package-group"] == "chosen-models"
    assert parameters["/trial-selected/tavily-secret-id"] == "trial-selected/research-key"
    runtime = _resources({"RuntimeStack": templates["RuntimeStack"]}, "AWS::BedrockAgentCore::Runtime")[0]
    assert runtime["EnvironmentVariables"]["AGENTICML_MODEL_PACKAGE_GROUP"] == "chosen-models"
    assert runtime["EnvironmentVariables"]["TAVILY_SECRET_ID"] == "trial-selected/research-key"
    policies = json.dumps(_resources({"RuntimeStack": templates["RuntimeStack"]}, "AWS::IAM::Policy"))
    assert "secret:trial-selected/research-key-*" in policies
    assert "secret:trial-selected/tavily-api-key-*" not in policies
