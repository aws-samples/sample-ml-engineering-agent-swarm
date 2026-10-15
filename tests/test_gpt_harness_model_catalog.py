"""Harness reviewers share model admission without gaining compute access."""

from __future__ import annotations

from pathlib import Path

from aws_cdk import App, Environment
from aws_cdk.assertions import Template

from amlas.config.runtime import bedrock_invoke_resources
from infra.cdk_stacks.harness_stack import FORBIDDEN_ACTIONS, HarnessStack


def test_harness_invocation_policy_uses_shared_catalog_and_preserves_guardrail(tmp_path: Path) -> None:
    app = App(outdir=str(tmp_path / "harness-assembly"))
    stack = HarnessStack(
        app, "HarnessModelAdmission", env=Environment(account="111122223333", region="us-west-2")
    )
    template = Template.from_stack(stack).to_json()
    statements = [
        statement
        for resource in template["Resources"].values()
        if resource["Type"] == "AWS::IAM::Policy"
        for statement in resource["Properties"]["PolicyDocument"]["Statement"]
    ]
    invoke = next(item for item in statements if item.get("Sid") == "InvokeModelAndApplyGuardrail")
    expected = [
        *bedrock_invoke_resources("111122223333", "us-west-2"),
        "arn:aws:bedrock:us-west-2:111122223333:guardrail/*",
    ]
    assert set(invoke["Resource"]) == set(expected)
    assert "arn:aws:bedrock:*::foundation-model/openai.gpt-6.1-sol" in invoke["Resource"]
    assert "bedrock:ApplyGuardrail" in invoke["Action"]
    actions = {
        action
        for statement in statements
        for action in (
            statement["Action"] if isinstance(statement["Action"], list) else [statement["Action"]]
        )
    }
    assert not actions.intersection(FORBIDDEN_ACTIONS)
    assert not any("CodeInterpreter" in action for action in actions)
