# Infrastructure Instructions

## Purpose

Synthesize the deployment stacks and their resource bindings from the selected account,
region and optional deployment prefix.

## Ownership

This subtree owns CDK construction, physical names, discovery parameters, IAM grants,
exports and supported native resource tags. Shell deployment and cleanup remain in scripts.

## Local Contracts

- Resolve AMLAS_DEPLOYMENT_PREFIX at app execution through the shared configuration helper.
  Default amlas retains legacy physical stack and resource names. A distinct prefix assigns
  prefix-LogicalStackID physical names while retaining logical IDs and CLI selections.
- Pass the selected prefix to every stack. SSM paths, roles, logs, KMS aliases, lake/database,
  registry, secret identifiers, runtime settings and IAM scopes must refer to that deployment.
  Explicit constructor overrides remain supported. Publish the resolved AMLAS_SSM_PREFIX
  consistently when a custom discovery root is selected.
- Bind supported TAVILY_SECRET_ID and AMLAS_MODEL_PACKAGE_GROUP selections to the
  resource, published discovery value, runtime environment and exact IAM scope.
  Keep prefix-derived legacy defaults when no override is selected.
- The gold Glue table must reference or explicitly depend on its stack-owned database;
  Python construction order does not establish CloudFormation scheduling.
- Tag each stack and supported resources with amlas:deployment-prefix. When
  AMLAS_DEPLOYMENT_ID is supplied, propagate it as amlas:deployment-id on every
  supported resource and stack; omit the identity tag when absent. Generic harness
  resources publish native Tags explicitly because they do not implement TagManager.
- HarnessName is at most 40 characters and uses letters, digits and underscores. Preserve
  default bare Lead names; for longer namespaced names preserve the namespace and retain a
  stable identity discriminator. Never rename the logical registry keys or SSM suffixes.
- Keep RuntimeStack and BackendIamStack synth-only in the setup workflow. Never expand the
  explicit nine-stack deployment to --all; RuntimeStack shares backend names with AsyncTrainingStack.
- Preserve default monitoring namespaces. Separate security metric namespaces for isolated
  deployments so alarms do not aggregate events from another deployment.
- Synthesis is offline evidence. It does not prove successful deployment or managed invocation.

## Work Guidance

Read the root instructions, reproduce defects with synthesized-resource regressions first,
and keep account credentials and private environment identifiers outside source.

## Verification

Run tests/test_infra_deployment_namespace.py using the frozen cdk and dev extras on the
approved remote development host. Verify the default names, two disjoint namespaces,
resource/export/SSM identities, runtime policy and configuration alignment, supported native
ownership tags, and harness length/collision boundaries. Run the repository checks and the
actual fresh setup deployment and invocation before reporting live acceptance.

## Child DOX Index

| Path | Covers |
| --- | --- |
| `cdk_stacks/AGENTS.md` | Maintained cdk_stacks boundary |
