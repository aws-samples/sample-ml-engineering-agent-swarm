# scripts/lib

## Purpose

Own the maintained scripts/lib boundary and its checked-in inputs and outputs.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Selected model profiles and Bedrock invocation resources come from the shared
  configuration catalog. Unadmitted profiles fail preflight; selected reused-role
  compatibility remains fail-closed after catalog changes.
- `agentcore_runtime_control.py:RuntimeDeployError` — A required control operation or readback failed.
- `deployment_manifest.py:LedgerError` — An identity or ownership gate failed; cleanup must stop.
- `write_research_secret.py:write_secret` — Use the selected profile or default credential chain and explicit region.
- Owned files: `agentcore_runtime_control.py`, `build_runtime_image.sh`, `deploy_agentcore_runtime.sh`, `deploy_common.sh`, `deployment_manifest.py`, `prepare_host.sh`, `write_research_secret.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
