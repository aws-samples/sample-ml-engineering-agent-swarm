# infra/cdk_stacks

## Purpose

Own the maintained infra/cdk_stacks boundary and its checked-in inputs and outputs.

## Ownership

Inherits [../AGENTS.md](../AGENTS.md); this document owns the files directly in this directory.

## Local Contracts

- Runtime and Harness Bedrock invocation grants use the same admitted model
  catalog, including reviewer models. Keep Harness guardrails and forbidden
  compute actions unchanged when adding model support.
- `_iam.py:cfn_getatt_arn` — Return cdk-nag's canonical ``<LogicalId.Arn>`` rendering for an L2.
- `_iam.py:acknowledge_iam5_resources` — Acknowledge exact residual IAM5 resource patterns with review evidence.
- `_kms.py:make_data_key` — Return a customer-managed CMK for a data stack — create by default, import if BYO.
- `_s3.py:make_access_log_bucket` — Create a retained, private sink for S3 server-access logs.
- `_tags.py:cost_tag_values` — Resolve the cost-allocation tag values from env (safe defaults; no org-specific value).
- `_tags.py:apply_cost_tags` — Stamp project/environment/owner cost-allocation tags on every taggable resource (WA-12).
- Owned files: `__init__.py`, `_iam.py`, `_kms.py`, `_s3.py`, `_tags.py`, `agentcore_runtime_stack.py`, `async_training_stack.py`, `backend_iam_stack.py`, `detections_stack.py`, `guardrail_stack.py`, `harness_stack.py`, `lake_stack.py`, `model_registry_stack.py`, `network_stack.py`, `research_scan_stack.py`, `runtime_stack.py`, `secrets_stack.py`.

## Work Guidance

Update this contract when owned interfaces, configuration, side effects or boundaries change. Read indexed child contracts before editing their files.

## Verification

Use the root verification contract and the owning regressions for changed behavior.

## Child DOX Index

None.
