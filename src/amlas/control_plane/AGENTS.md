# src/amlas/control_plane

## Purpose

Own capability dispatch, approval/plan policy, cost boundaries, project memory, sessions and integration hooks.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Separate planned, approved and executed results; Plan Mode must not invoke the executor.
- Spend/irreversible boundaries retain the existing approval and budget checks; each caller must use the documented capability path.
- Default AWS mutation scope follows the selected deployment prefix at call time.
  Explicit owner resource-prefix overrides retain precedence.
- `aws_resource_policy.py:resolve_control_region` — Resolve the same selected region used by every AWS control tool.
- `aws_resource_policy.py:ResourceControlPolicy` — The configured guard (prefix + region pin + allowlist override).
- `capability.py:register_capability_executor` — Register an explicit executor callable for ``flow_id`` (last write wins).
- `capability.py:resolve_executor` — Resolve the executor callable for ``flow_id`` (registry → override → convention).
- Owned files: `__init__.py`, `approvals.py`, `aws_resource_policy.py`, `capability.py`, `cost_confidence.py`, `cost_governance.py`, `custom_commands.py`, `guardrails.py`, `mcp_bridge.py`, `mention.py`, `plan_mode.py`, `project_memory.py`, `runtime_session.py`, `token_cost.py`, `traceability.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
