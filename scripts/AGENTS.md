# Deployment Script Instructions

## Purpose

Validate, deploy, and clean up the portable application through explicit cloud selectors.

## Ownership

This directory owns shell entrypoints and the dependency-free deployment ownership ledger.
The root instructions own repository authority and the full verification contract.

## Local Contracts

- `dox_lint.py` is byte-vendored from the canonical DOX lint. `--recursive` enforces maintained nested-directory coverage; it emits `DOX PASS`/`DOX FAIL` and exits0/1. CI invokes it before dependency preparation.

- `lib/deployment_manifest.py` runs with standard-library Python before project dependency
  installation. Its commands are `init`, `refresh`, `plan`, `delete`, and `record-session`.
  Successful commands emit JSON on stdout; failures emit sanitized diagnostics on stderr.
- `init --file PATH --account ACCOUNT --region REGION --prefix PREFIX --stacks NAME...`
  also receives `--secret-id`, `--runtime-name`, and optional `--role-stack`, `--role-name`
  and `--profile`, plus immutable `--ssm-prefix` and `--model-group` selectors.
  `--tool-bin` records the setup-selected CLI directory for noninstalling cleanup lookup. It records native identities before writes and returns the stable UUID
  `deployment_id`. Reruns with the same selectors reuse that manifest and UUID.
- Setup initializes `output/deploy/deployment-manifest.json` after preflight and before
  cloud writes. New resources carry `amlas:deployment-id` and `amlas:deployment-prefix`.
  An occupied mutable target is rejected for every prefix unless the existing manifest proves ownership. Resources
  present before setup are never adopted. Explicit immutable execution-role reuse requires
  the preflight-verified native RoleARN/RoleId witness. Successful compatibility
  preflight emits `RESOLVED_REUSED_RUNTIME_ROLE_ARN` and
  `RESOLVED_REUSED_RUNTIME_ROLE_ID`; setup forwards this exact pair to ledger init.
  Missing, duplicate or wrong-account witness fields fail before initialization. An old schema may upgrade with the same
  UUID only when resources, sessions, stack inventory are empty and all baseline values are null.
- Setup refreshes the ledger after write phases and from its exit trap. Refresh retains
  partial native handles when a later metadata read fails. Missing or foreign ownership
  tags on expected new resources, changed identities, or incomplete metadata verification
  block cleanup until a successful refresh resolves the uncertainty.
- `record-session --file PATH --runtime-id ID --session-id ID [--qualifier DEFAULT]`
  records only sessions of an owned runtime. AgentCore session IDs have 33–256 characters.
  Record the session before invocation and refresh after invocation to capture generated logs.
- Teardown requires `--manifest PATH` or the default setup manifest. `--dry-run` observes
  native metadata and emits the guarded plan without AWS mutations. Before the first
  delete, every owned target must match the account, region, native identity, original
  preexisting flag, and ownership proof. Each mutation rechecks its own target.
- Stop recorded sessions and verify runtime absence. Inventory standalone training and
  processing jobs by atomic UUID/prefix/session creation tags, stop owned active jobs, and
  verify terminal states before supporting object, role, stack, or key cleanup. Native terminal
  job history remains explicitly reported. Prefix alone never owns a managed job. Remove its auxiliary role
  stack and the selected application stacks in reverse order using the recorded native
  StackId and the same verified CloudFormation service RoleARN and RoleId.
  Cleanup never directly creates or deletes IAM roles.
- Inventory retained ECR, S3, DynamoDB, KMS, model groups, SSM parameters and log groups
  through exact CloudFormation/native handles. Empty all S3 versions and delete markers;
  enforce service batch limits. Generated runtime and Lambda logs bind to a recorded
  runtime ID or exact native FunctionPhysicalId in an owned StackId. An explicit foreign
  ownership UUID overrides that derived proof and protects the log group. Snapshot future
  Lambda log namespaces before execution and require creation-incarnation evidence; older
  untagged groups remain protected even when a new function reuses their fixed names.
- Keep encrypted data usable until it has been removed. Schedule owned customer KMS keys
  with the seven-day minimum window and verify `PendingDeletion`. Pending keys remain
  explicitly reported with their actual deletion date; they are not reported as absent.
- Shared bootstrap resources and reused roles remain outside the product cleanup ledger.
  `--nuke-bootstrap` is rejected because those resources are not owned by this manifest.
- AWS execution uses argv and the existing credential chain or selected local profile.
  Do not retrieve secret values or SSM values for ownership discovery, log credentials,
  install dependencies during teardown, or perform account-wide prefix deletion sweeps.

- `lib/write_research_secret.py create|put` consumes JSON stdin and calls the
  native Secrets Manager SDK through the declared project interpreter. It uses
  the exact selected local profile or default chain and an explicit Region.
  Create preserves ownership tags; put preserves the intended secret value.
  Success emits no payload or response; errors return one with a fixed sanitized
  diagnostic. Keys never enter argv, temporary files, dotenv or helper/uv
  environment. Real SDK/profile/stdin checks supplement fake transport tests.

- `preflight_cloudtrail_capacity()` uses the selected account/Region and normal
  AWS CLI wrapper. ListTrails includes shadow trails and must be readable and
  structurally valid. The fixed five-trail limit is checked before new secret or
  infrastructure writes when DetectionsStack is selected. Exact selected-trail
  presence exempts only capacity; ownership remains the ledger's independent
  guard. No automatic trail deletion, quota request or new authentication path.
  Operator metadata permissions include `cloudtrail:ListTrails`.

- `lib/agentcore_runtime_control.py` updates the runtime and verifies its exact
  image, role, environment, network/protocol, platform, version, ownership and MMDSv2.
  DEFAULT is managed by runtime updates; never call UpdateAgentRuntimeEndpoint for
  DEFAULT. Poll its native READY status and exact liveVersion. targetVersion may be
  absent or null for DEFAULT; a present target must still match. Missing/null/wrong
  liveVersion never passes. This exception does not change named-endpoint contracts.

## Work Guidance

Read the root instructions before editing. Reproduce lifecycle failures with closed native
AWS state before changing behavior. Keep resource identity and ownership uncertainty visible.
Maintain runtime, CFN actor, retained-resource and partial-failure regression coverage.

## Verification

```bash
python3 -I tests/test_deployment_manifest_cleanup.py
bash -n scripts/teardown.sh
uv run --frozen --extra dev ruff check scripts/lib/deployment_manifest.py tests/test_deployment_manifest_cleanup.py
```

Run the root's full locked suite and source review before live execution. Fixture success
proves the guards against the fake native state; it does not prove live AWS cleanup.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `lib/AGENTS.md` | Maintained lib boundary |
