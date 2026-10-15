"""SecretsStack (NEW) — Tavily API key secret (imported, not recreated).

Per design Part C.2.2: the runtime resolves the Tavily key from Secrets Manager
secret id `amlas/tavily-api-key` (matches
[`secrets.py`](src/amlas/cloud/secrets.py:30)).

IMPORTANT — first-principles reconciliation with live state:
A LIVE read of default showed this secret ALREADY EXISTS out-of-band
(created 2026-05-29, ARN suffix `-5maU8d`, description "Phase 14c sets value").
Creating a same-named `aws_secretsmanager.Secret` in CDK would fail CloudFormation
with "secret already exists". Therefore this stack IMPORTS the existing secret via
`Secret.from_secret_name_v2` (a real reference to a real resource), publishes its
id/ARN to SSM for the config feedback loop, and exposes it so consuming stacks can
grant read. The real `tvly-…` key value is set out-of-band via:

    AWS_PROFILE=default aws secretsmanager put-secret-value \
        --secret-id amlas/tavily-api-key --secret-string "$TAVILY_KEY"

SEC-01: NO secret material in code — only the secret id/ARN (non-sensitive).
The secret arn is resolved from CDK tokens via the import, never hardcoded.
"""

from __future__ import annotations

from typing import Any

from aws_cdk import (
    CfnOutput,
    Stack,
)
from aws_cdk import (
    aws_secretsmanager as secretsmanager,
)
from aws_cdk import (
    aws_ssm as ssm,
)
from constructs import Construct

DEFAULT_TAVILY_SECRET_ID = "amlas/tavily-api-key"  # nosec B105 — Secrets Manager secret *name* (resource id), not a credential


class SecretsStack(Stack):
    def __init__(
        self,
        scope: Construct,
        cid: str,
        *,
        hyphen_project: str = "amlas",
        ssm_prefix: str | None = None,
        tavily_secret_id: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(scope, cid, **kwargs)
        self.hyphen_project = hyphen_project
        self.ssm_prefix = ssm_prefix if ssm_prefix is not None else f"/{hyphen_project}"
        self.tavily_secret_id = (
            tavily_secret_id if tavily_secret_id is not None else f"{hyphen_project}/tavily-api-key"
        )

        # Import the pre-existing secret (created out-of-band). This is a real
        # reference, not a recreation — avoids the CFN "already exists" collision.
        self.tavily_secret = secretsmanager.Secret.from_secret_name_v2(
            self,
            "TavilySecret",
            secret_name=self.tavily_secret_id,
        )

        ssm.StringParameter(
            self,
            "TavilySecretIdParam",
            parameter_name=f"{self.ssm_prefix}/tavily-secret-id",
            string_value=self.tavily_secret_id,
            description="Secrets Manager secret id holding the Tavily API key (value set out-of-band)",
        )

        self._emit_outputs()

    def grant_read(self, grantee: Any) -> None:
        """Grant a principal/role read on the Tavily secret (used by other stacks)."""
        self.tavily_secret.grant_read(grantee)

    def _emit_outputs(self) -> None:
        CfnOutput(
            self,
            "TavilySecretId",
            value=self.tavily_secret_id,
            export_name=f"{self.hyphen_project}-tavily-secret-id",
        )
        CfnOutput(
            self,
            "TavilySecretArn",
            value=self.tavily_secret.secret_arn,
            export_name=f"{self.hyphen_project}-tavily-secret-arn",
        )
