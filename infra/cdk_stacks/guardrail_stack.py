"""GuardrailStack (NEW, WA-02 / AGENTSEC04+AGENTSEC08) — Bedrock Guardrail as IaC.

The audit's #2 High finding: a real ``BedrockGuardrailBackend`` existed but **no CDK stack
provisioned a Bedrock Guardrail**, and the live ``BedrockModel`` attached no
``guardrailConfig`` — so harmful-output / prompt-injection / PII filtering was *available*
but not *enforced in-band*. This stack provisions the missing Bedrock Guardrail resource and
publishes its id + version to SSM so the runtime binds it on every model call (via
``config.guardrails`` → ``_build.py`` / TUI ``converse``).

Responsible-AI defaults (conservative; the audit §8 open decision #7 — guardrail strictness
is a policy choice, so every threshold is a CfnParameter with a secure default):
    * Content-policy filters on hate / insults / sexual / violence / misconduct / prompt-attack
      at ``ContentFilterStrength`` (default ``HIGH``) for BOTH input and output.
    * Sensitive-information (PII) policy ANONYMIZE-ing common PII entities by default
      (``PiiDefaultAction``), so prompts/responses are redacted in-band.
    * Blocked-input / blocked-output messaging is parameterized.

SEC-01: no account/org-specific values hardcoded — strictness + messaging are CfnParameters;
the guardrail name is the ``hyphen_project`` convention.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    CfnParameter,
    Stack,
)
from aws_cdk import (
    aws_bedrock as bedrock,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct

# The content-filter categories the guardrail applies (input + output). PROMPT_ATTACK only
# applies to INPUT (AWS rejects an output strength on it), handled below.
_CONTENT_FILTER_TYPES = ("SEXUAL", "VIOLENCE", "HATE", "INSULTS", "MISCONDUCT")

# Common PII entities anonymized by default (privacy — WA-04 prompt/response side).
_PII_ENTITIES = (
    "EMAIL",
    "PHONE",
    "NAME",
    "US_SOCIAL_SECURITY_NUMBER",
    "CREDIT_DEBIT_CARD_NUMBER",
    "US_BANK_ACCOUNT_NUMBER",
    "IP_ADDRESS",
)


class GuardrailStack(Stack):
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

        # ── §8.7 responsible-AI policy CfnParameters (secure defaults; deployer can tune) ──
        self.strength_param = CfnParameter(
            self,
            "ContentFilterStrength",
            type="String",
            default="HIGH",
            allowed_values=["NONE", "LOW", "MEDIUM", "HIGH"],
            description="Bedrock Guardrail content-filter strength for hate/violence/sexual/"
            "insults/misconduct/prompt-attack (the WA-02 responsible-AI policy decision). "
            "Default HIGH (most conservative).",
        )
        self.pii_action_param = CfnParameter(
            self,
            "PiiDefaultAction",
            type="String",
            default="ANONYMIZE",
            allowed_values=["ANONYMIZE", "BLOCK"],
            description="Default action for detected PII entities in prompts/responses. "
            "Default ANONYMIZE (redact in-band).",
        )
        self.blocked_input_msg = CfnParameter(
            self,
            "BlockedInputMessage",
            type="String",
            default="This request was blocked by the content guardrail.",
            description="Message returned when an input is blocked by the guardrail.",
        )
        self.blocked_output_msg = CfnParameter(
            self,
            "BlockedOutputMessage",
            type="String",
            default="The response was blocked by the content guardrail.",
            description="Message returned when an output is blocked by the guardrail.",
        )

        self.guardrail = self._build_guardrail()
        self._write_ssm_params()
        self._emit_outputs()

    def _build_guardrail(self) -> bedrock.CfnGuardrail:
        strength = self.strength_param.value_as_string
        filters = [
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type=ftype,
                input_strength=strength,
                output_strength=strength,
            )
            for ftype in _CONTENT_FILTER_TYPES
        ]
        # PROMPT_ATTACK: input-only filtering (AWS requires output_strength=NONE).
        filters.append(
            bedrock.CfnGuardrail.ContentFilterConfigProperty(
                type="PROMPT_ATTACK",
                input_strength=strength,
                output_strength="NONE",
            )
        )

        pii_entities = [
            bedrock.CfnGuardrail.PiiEntityConfigProperty(
                type=entity,
                action=self.pii_action_param.value_as_string,
            )
            for entity in _PII_ENTITIES
        ]

        return bedrock.CfnGuardrail(
            self,
            "Guardrail",
            name=f"{self.hyphen_project}-guardrail",
            description="Amlas in-band content + PII guardrail (WA-02; AGENTSEC04/08). "
            "Bound on every Bedrock model turn via config.guardrails.",
            blocked_input_messaging=self.blocked_input_msg.value_as_string,
            blocked_outputs_messaging=self.blocked_output_msg.value_as_string,
            content_policy_config=bedrock.CfnGuardrail.ContentPolicyConfigProperty(
                filters_config=filters,
            ),
            sensitive_information_policy_config=(
                bedrock.CfnGuardrail.SensitiveInformationPolicyConfigProperty(
                    pii_entities_config=pii_entities,
                )
            ),
        )

    def _write_ssm_params(self) -> None:
        # Publish the id + version so the runtime reads them into AMLAS_GUARDRAIL_ID/VERSION
        # (config.guardrails) and binds the guardrail in-band on every Bedrock call.
        ssm.StringParameter(
            self,
            "GuardrailIdParam",
            parameter_name=f"{self.ssm_prefix}/guardrail-id",
            string_value=self.guardrail.attr_guardrail_id,
            description="Bedrock Guardrail id — bound in-band on every model turn (WA-02)",
        )
        ssm.StringParameter(
            self,
            "GuardrailVersionParam",
            parameter_name=f"{self.ssm_prefix}/guardrail-version",
            string_value=self.guardrail.attr_version,
            description="Bedrock Guardrail version (WA-02)",
        )

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "GuardrailIdOut",
            value=self.guardrail.attr_guardrail_id,
            export_name=f"{self.hyphen_project}-guardrail-id",
        )
        CfnOutput(
            self,
            "GuardrailArnOut",
            value=self.guardrail.attr_guardrail_arn,
            export_name=f"{self.hyphen_project}-guardrail-arn",
        )
        CfnOutput(
            self,
            "GuardrailVersionOut",
            value=self.guardrail.attr_version,
            export_name=f"{self.hyphen_project}-guardrail-version",
        )
