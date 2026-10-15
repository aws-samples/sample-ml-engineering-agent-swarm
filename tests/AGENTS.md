# tests

## Purpose

Own the shipped automated regressions and publication/governance oracles.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Maintain the interfaces and file formats owned by the modules listed below; parent contracts apply.
- `test_agent_deployment_context.py` covers configured references and selected mutation scope.
- `test_generation_task_contract.py` covers generation routing/spec round-trips and tabular controls.
- `test_sonnet_defaults.py` and `test_sonnet_imputation.py` cover shared model
  defaults, direct consumers, explicit overrides and native parameter compatibility.
- `test_prompt_policy.py` covers normal prompt delegation and caller authority
  transport/snapshot with malformed model-authored policy controls.
- GPT reviewer and Harness catalog tests cover actual model capabilities, strict
  cache/sampling fields and review-model invocation grants.
- `test_sdk_prompt_policy.py` exercises the installed SDK concurrent tool path
  and real approval gate, including spoof denial, snapshot/reuse and overlapping
  session isolation. It also records the capability-only direct-tool boundary.
- Owned files: `runtime_image_acceptance.py`, `test_astra_safeguards.py`, `test_cli_persisted_discovery.py`, `test_client_deployment_namespace.py`, `test_cloudtrail_capacity_preflight.py`, `test_deployment_manifest_cleanup.py`, `test_github_workflow_contract.py`, `test_infra_deployment_namespace.py`, `test_managed_job_ownership.py`, `test_manifest_cli_credential_chain.py`, `test_pipeline_dag_compilation.py`, `test_prepare_host.py`, `test_region_client_metadata.py`, `test_repository_governance.py`, `test_repository_policy.py`, `test_research_public_fallback.py`, `test_runtime_binding_completeness.py`, `test_runtime_default_endpoint.py`, `test_runtime_deployment_advisory.py`, `test_runtime_ownership_controller.py`, `test_runtime_role_lifecycle_advisory.py`, `test_runtime_role_portability.py`, `test_setup_deployment_namespace.py`, `test_setup_owned_publication.py`, `test_setup_reuse_witness.py`, `test_shell_entrypoint_advisory.py`, `test_tavily_setup_choice.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
