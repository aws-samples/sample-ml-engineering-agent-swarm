"""HarnessStack (ADR-0007 phase 1, T-HARN-13) — one managed AgentCore harness per leaf Lead.

Provisions an ``AWS::BedrockAgentCore::Harness`` for each harness-hosted Lead from its
:class:`~amlas.harness.spec.HarnessSpec`, and publishes each harness id to SSM under
``/amlas/`` so the runtime resolves it the same way it resolves the guardrail id — no
out-of-band ``create_harness`` calls.

Two invariants are enforced *by the IAM policy this stack writes*, not by convention:

* **ADR-0002 — a harness is NEVER a compute path.** The execution role grants no
  ``bedrock-agentcore:InvokeAgentRuntimeCommand`` (T-HARN-05.1/.6) and no
  ``*CodeInterpreter*`` (T-HARN-13.5). The first matters most: `allowedTools` only scopes LLM
  tool *selection*, and ``InvokeAgentRuntimeCommand`` bypasses the LLM entirely, so IAM is the
  only thing that closes it. Denying it in IAM holds regardless of prompt or model behaviour.
* **WA-02 guardrail parity.** The role grants ``bedrock:ApplyGuardrail`` so a bound guardrail
  cannot fail for want of a permission — a denial there would be indistinguishable from a
  guardrail that was never bound.

Uses the L1 ``CfnResource`` seam, matching how ``RuntimeStack`` declares its AgentCore
primitives: version-portable, and there is no L2 construct for a harness.

Deletion is ASYNCHRONOUS (>5 min, cascading to the managed Memory) — see T-HARN-12 and the
destroy-order note in ``scripts/lib/deploy_common.sh``.
"""

from __future__ import annotations

import hashlib
from typing import Any

from aws_cdk import CfnOutput, CfnResource, Stack
from aws_cdk import aws_iam as iam
from aws_cdk import aws_ssm as ssm
from constructs import Construct

from amlas.config.runtime import bedrock_invoke_resources

from ._tags import deployment_tags

#: Actions that would make a harness a compute path. Never granted, and asserted absent by
#: ``validation coverage`` so a future policy edit cannot quietly add one.
FORBIDDEN_ACTIONS: tuple[str, ...] = (
    "bedrock-agentcore:InvokeAgentRuntimeCommand",
    "bedrock-agentcore:CreateCodeInterpreter",
    "bedrock-agentcore:StartCodeInterpreterSession",
    "bedrock-agentcore:InvokeCodeInterpreter",
)


class HarnessStack(Stack):
    """One harness per harness-hosted Lead, plus the shared execution role."""

    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"

        self.execution_role = self._build_execution_role()
        self.harnesses: dict[str, CfnResource] = {}

        for lead_name, properties in self._harness_properties().items():
            harness = CfnResource(
                self,
                f"Harness{_pascal(lead_name)}",
                type="AWS::BedrockAgentCore::Harness",
                properties=properties,
            )
            harness.node.add_dependency(self.execution_role)
            self.harnesses[lead_name] = harness
            self._publish(lead_name, harness)

    # ------------------------------------------------------------------
    # specs → CFN properties
    # ------------------------------------------------------------------
    def _harness_properties(self) -> dict[str, dict[str, Any]]:
        """Build every lead's L1 properties from its HarnessSpec.

        Imported lazily so the module is importable in a CDK context that has not installed the
        agents extra; the spec itself makes no AWS call and needs no credentials.
        """
        from amlas.harness.registry import HARNESS_HOSTED_LEADS
        from amlas.harness.spec import harness_spec_for_lead

        out: dict[str, dict[str, Any]] = {}
        for lead_name in HARNESS_HOSTED_LEADS:
            spec = harness_spec_for_lead(lead_name)
            properties = spec.to_cfn_properties(execution_role_arn=self.execution_role.role_arn)
            properties["HarnessName"] = _harness_name(self.hyphen_project, lead_name)
            # Generic CfnResource has no TagManager, so publish supported native tags.
            properties["Tags"] = [
                {"Key": key, "Value": value} for key, value in deployment_tags(self.hyphen_project).items()
            ]
            out[lead_name] = properties
        return out

    # ------------------------------------------------------------------
    # IAM — where ADR-0002 is actually enforced
    # ------------------------------------------------------------------
    def _build_execution_role(self) -> iam.Role:
        role = iam.Role(
            self,
            "HarnessExecutionRole",
            role_name=f"{self.hyphen_project}-harness-execution-role",
            assumed_by=iam.ServicePrincipal(
                "bedrock-agentcore.amazonaws.com",
                conditions={
                    "StringEquals": {"aws:SourceAccount": self.account},
                    "ArnLike": {
                        "aws:SourceArn": (f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:*")
                    },
                },
            ),
            # ASCII ONLY. IAM validates `description` against
            # [\u0009\u000A\u000D\u0020-\u007E\u00A1-\u00FF], which EXCLUDES the em dash
            # (U+2014) this repo's prose uses everywhere else. `cdk synth` does not check it
            # (an L1 CfnResource's properties are not validated at all), so the first
            # HarnessStack deploy failed at CREATE_FAILED on this string alone. Keep it
            # ASCII; the Harness resource's own prompt fields are a different service with a
            # different charset and are left as authored.
            description=(
                "AMLAS harness execution role - model invocation + guardrail application "
                "only. Deliberately grants NO InvokeAgentRuntimeCommand and no "
                "code-interpreter actions (ADR-0002: a harness is never a compute path)."
            ),
        )

        # Model invocation, and the guardrail the model config binds (WA-02).
        role.add_to_policy(
            iam.PolicyStatement(
                sid="InvokeModelAndApplyGuardrail",
                actions=[
                    "bedrock:InvokeModel",
                    "bedrock:InvokeModelWithResponseStream",
                    "bedrock:ApplyGuardrail",
                ],
                resources=[
                    *bedrock_invoke_resources(self.account, self.region),
                    f"arn:aws:bedrock:{self.region}:{self.account}:guardrail/*",
                ],
            )
        )
        # Managed memory events — the harness owns the memory resource it creates.
        role.add_to_policy(
            iam.PolicyStatement(
                sid="ManagedMemoryEvents",
                actions=[
                    "bedrock-agentcore:CreateEvent",
                    "bedrock-agentcore:GetEvent",
                    "bedrock-agentcore:ListEvents",
                    "bedrock-agentcore:PutMemoryRecord",
                    "bedrock-agentcore:RetrieveMemoryRecords",
                ],
                resources=[f"arn:aws:bedrock-agentcore:{self.region}:{self.account}:memory/*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="Observability",
                actions=[
                    "logs:CreateLogGroup",
                    "logs:CreateLogStream",
                    "logs:PutLogEvents",
                ],
                resources=[f"arn:aws:logs:{self.region}:{self.account}:log-group:*"],
            )
        )
        role.add_to_policy(
            iam.PolicyStatement(
                sid="WorkloadIdentity",
                actions=[
                    "bedrock-agentcore:GetWorkloadAccessToken",
                    "bedrock-agentcore:GetWorkloadAccessTokenForJWT",
                ],
                resources=[
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}"  # nosemgrep: string-concat-in-list
                    ":workload-identity-directory/default",
                    f"arn:aws:bedrock-agentcore:{self.region}:{self.account}"  # nosemgrep: string-concat-in-list
                    ":workload-identity-directory/default/workload-identity/*",
                ],
            )
        )
        return role

    # ------------------------------------------------------------------
    # SSM + outputs
    # ------------------------------------------------------------------
    def _publish(self, lead_name: str, harness: CfnResource) -> None:
        """Publish the harness id to SSM, mirroring the guardrail-id pattern."""
        ssm.StringParameter(
            self,
            f"HarnessId{_pascal(lead_name)}Param",
            parameter_name=f"{self.ssm_prefix}/harness/{lead_name.replace('_', '-')}-id",
            string_value=harness.get_att("HarnessId").to_string(),
            description=f"AgentCore harness id for the {lead_name} Lead (ADR-0007 phase 1)",
        )
        CfnOutput(
            self,
            f"HarnessArn{_pascal(lead_name)}Out",
            value=harness.get_att("Arn").to_string(),
            export_name=f"{self.hyphen_project}-harness-{lead_name.replace('_', '-')}-arn",
        )


def _pascal(name: str) -> str:
    """``research_lead`` → ``ResearchLead`` for CDK logical ids."""
    return "".join(part.capitalize() for part in name.split("_"))


def _harness_name(prefix: str, lead_name: str) -> str:
    """Preserve legacy lead names; keep namespaced names within the service bound."""
    if prefix == "amlas":
        return lead_name
    namespace = prefix.replace("-", "_")
    name = f"{namespace}_{lead_name}"
    if len(name) <= 40:
        return name
    # Keep the full namespace and distinguish leads that share a truncated stem.
    digest = hashlib.sha256(lead_name.encode("utf-8")).hexdigest()[:8]
    available = 40 - len(namespace) - len(digest) - 2
    if available < 1:
        raise ValueError("deployment prefix leaves no room for a valid harness name")
    return f"{namespace}_{lead_name[:available]}_{digest}"
