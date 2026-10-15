"""AgentCore harness hosting for the leaf Leads (ADR-0007 phase 1).

Public surface:

* :mod:`amlas.harness.spec` — :class:`~amlas.harness.spec.HarnessSpec`, the pure offline
  configuration object every assertion and every AWS payload derives from.
* :mod:`amlas.harness.registry` — which Leads are harness-hosted, and their local peers.
* :mod:`amlas.harness.client` — the ``InvokeHarness`` event-stream client and the
  :class:`~amlas.harness.client.HarnessSessions` reuse ledger.
* :mod:`amlas.harness.provisioner` — the programmatic lifecycle (create / update / replace /
  delete) and the constraints CloudFormation cannot express.
* :mod:`amlas.harness.approvals` — the approval gate on paused ``inline_function`` calls, which
  is what stops a server-side loop self-approving spend or an irreversible action.

Importing this package calls no AWS. The client builds a boto3 client only when actually
invoked without one injected, so tests drive it with a fake stream — which is what lets the
credential-free suite cover the full configuration AND the stream contract. Provisioning
lives in CDK (``infra/cdk_stacks/harness_stack.py``).
"""

from __future__ import annotations

from .approvals import ApprovalDeniedError, build_resume_messages, resolve_tool_calls
from .client import (
    HARNESS_BOTOCORE_FLOOR,
    SESSION_ID_MIN_LENGTH,
    HarnessInvokeError,
    HarnessSdkTooOldError,
    HarnessSessions,
    HarnessTerminalEventMissingError,
    HarnessToolUse,
    HarnessTurn,
    invoke_harness,
    make_harness_session_id,
)
from .leads import (
    DEFAULT_MAX_TOOL_HOPS,
    HARNESS_LEADS_ENV,
    HarnessLeadError,
    HarnessToolRefusedError,
    LeadRun,
    build_local_lead,
    default_tool_executor,
    harness_arn_env_var,
    harness_arn_for,
    harness_enabled,
    harness_hosted,
    harness_ssm_key,
    run_lead_turn,
)
from .provisioner import (
    DELETE_MINIMUM_SECONDS,
    DeleteOutcome,
    HarnessDeletingError,
    HarnessProvisionerError,
    HarnessState,
    delete_harness,
    find_harness,
    provision_harness,
)
from .registry import (
    HARNESS_HOSTED_LEADS,
    HARNESS_HOSTED_SUBAGENTS,
    harness_declaration,
    local_peer_builder,
    subagent_harness_declaration,
    subagent_local_peer_builder,
)
from .spec import (
    ALLOWED_TOOL_PREFIX,
    DEFAULT_MEMORY_STRATEGIES,
    SUMMARIZATION_PROMPT_SOURCE,
    HarnessableDeclaration,
    HarnessSpec,
    HarnessSpecError,
    SummarizationConfig,
    harness_spec_for_lead,
    harness_spec_for_subagent,
    harness_spec_from_declaration,
)

__all__ = [
    "ALLOWED_TOOL_PREFIX",
    "DEFAULT_MAX_TOOL_HOPS",
    "DEFAULT_MEMORY_STRATEGIES",
    "DELETE_MINIMUM_SECONDS",
    "HARNESS_BOTOCORE_FLOOR",
    "HARNESS_HOSTED_LEADS",
    "HARNESS_HOSTED_SUBAGENTS",
    "HARNESS_LEADS_ENV",
    "SESSION_ID_MIN_LENGTH",
    "SUMMARIZATION_PROMPT_SOURCE",
    "ApprovalDeniedError",
    "DeleteOutcome",
    "HarnessDeletingError",
    "HarnessInvokeError",
    "HarnessLeadError",
    "HarnessProvisionerError",
    "HarnessSdkTooOldError",
    "HarnessSessions",
    "HarnessSpec",
    "HarnessSpecError",
    "HarnessState",
    "HarnessTerminalEventMissingError",
    "HarnessToolRefusedError",
    "HarnessToolUse",
    "HarnessTurn",
    "HarnessableDeclaration",
    "LeadRun",
    "SummarizationConfig",
    "build_local_lead",
    "build_resume_messages",
    "default_tool_executor",
    "delete_harness",
    "find_harness",
    "harness_arn_env_var",
    "harness_arn_for",
    "harness_declaration",
    "harness_enabled",
    "harness_hosted",
    "harness_spec_for_lead",
    "harness_spec_for_subagent",
    "harness_spec_from_declaration",
    "harness_ssm_key",
    "invoke_harness",
    "local_peer_builder",
    "make_harness_session_id",
    "provision_harness",
    "resolve_tool_calls",
    "run_lead_turn",
    "subagent_harness_declaration",
    "subagent_local_peer_builder",
]
