#!/usr/bin/env bash
# Create or update the AgentCore runtime for the freshly pushed arm64 image,
# then poll it to READY. Called by scripts/setup.sh.
#
# ============================================================================
# FIX-11 (durable) — kill the H8 stale-runtime-id + update-only path.
# ----------------------------------------------------------------------------
# WHAT WAS WRONG (H8): this script hardcoded a now-DELETED runtime id
#   (amlas-i2usHYDNZE) and called `update-agent-runtime` on it. After the
#   account was torn down (Phase 8) that runtime no longer exists, so every
#   invocation ResourceNotFound-failed. A net-new deploy needs a CREATE, not an
#   UPDATE against a dead id.
#
# WHAT THIS DOES NOW (create-or-discover, idempotent):
#   1. Ensure the selected execution role (default `amlas-runtime-exec-<region>`) exists. In the
#      normal single-account design RuntimeStack owns this role — but in THIS
#      split-stack account RuntimeStack is intentionally EXCLUDED from the
#      9-stack set (deploy_common.sh) because its fixed names collide with the
#      already-deployed AsyncTrainingStack (amlas-sagemaker-exec, DDB
#      amlas-training-jobs, callback lambda). So we provision the runtime-exec
#      role here through a CloudFormation-owned lifecycle stack when absent, or
#      reusing it only if the trust and inline policy match. An incompatible existing role fails without changing its grants.
#      The desired policy MIRRORS
#      runtime_stack._build_runtime_role (Bedrock InvokeModel on admitted inference
#      profiles + their FM ARNs, S3 RW on the session + lake buckets, CW logs,
#      SageMaker create/describe/stop + PassRole→sagemaker-exec, Secrets read).
#   2. DISCOVER the runtime by NAME (list-agent-runtimes | match agentRuntimeName).
#      If ABSENT → `create-agent-runtime` (fresh id/ARN). If PRESENT → repoint the
#      container image via `update-agent-runtime` (so re-runs are idempotent).
#   3. Poll `get-agent-runtime` until READY / *_FAILED.
#   4. Print the ARN/id (the caller publishes it to SSM).
#
# SEC-01 / R-001: account id + all ARNs are resolved from the authenticated
# profile (or an AWS_ACCOUNT_ID override) at runtime — never hardcoded.
# ============================================================================
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Honor an explicitly selected profile; otherwise use the standard AWS
# credential chain.
if [ -n "${AWS_PROFILE:-}" ]; then
  export AWS_PROFILE
fi
# FIX-10: a stale inherited Bedrock API key would break nothing here (control
# plane is SigV4), but keep the session consistent with the rest of Phase 5.
unset AWS_BEARER_TOKEN_BEDROCK 2>/dev/null || true

. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common.sh"
NON_INTERACTIVE=1
resolve_aws_region
REGION="$AWS_REGION"
ACCT=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
if [ -z "${ACCT}" ] || [ "${ACCT}" = "None" ]; then
  echo "ERROR: could not resolve AWS account id (set AWS_ACCOUNT_ID or a valid AWS_PROFILE)" >&2
  exit 1
fi

DEPLOYMENT_PREFIX=${AMLAS_DEPLOYMENT_PREFIX:-amlas}
SSM_PREFIX=${AMLAS_SSM_PREFIX:-/${DEPLOYMENT_PREFIX}}
DEPLOYMENT_ID=${AMLAS_DEPLOYMENT_ID:-}
if [ -n "$DEPLOYMENT_ID" ]; then
  [[ "$DEPLOYMENT_ID" =~ ^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$ ]] || die "Invalid deployment UUID."
fi
[[ "$DEPLOYMENT_PREFIX" =~ ^[a-z][a-z0-9-]{0,31}$ ]] || die "Invalid deployment prefix."
[[ "$SSM_PREFIX" =~ ^/[A-Za-z0-9_.-]+(/[A-Za-z0-9_.-]+)*$ ]] || die "Invalid SSM prefix."
STACK_PREFIX=""
[ "$DEPLOYMENT_PREFIX" = "amlas" ] || STACK_PREFIX="${DEPLOYMENT_PREFIX}-"
ROLE_POLICY_NAME="${DEPLOYMENT_PREFIX}-runtime-exec-policy"
ROLE_STACK_NAME="${DEPLOYMENT_PREFIX}-runtime-role"
RUNTIME_NAME=${AGENTCORE_RUNTIME_NAME:-${DEPLOYMENT_PREFIX//-/_}}
EXEC_ROLE_NAME=${AGENTCORE_EXEC_ROLE_NAME:-${DEPLOYMENT_PREFIX}-runtime-exec-${REGION}}
IMAGE_TAG=${IMAGE_TAG:-v1}
IMAGE_URI=${ACCT}.dkr.ecr.${REGION}.amazonaws.com/${DEPLOYMENT_PREFIX}-runtime:${IMAGE_TAG}
NETWORK_MODE=${AGENTCORE_NETWORK_MODE:-PUBLIC}
SERVER_PROTOCOL=${AGENTCORE_SERVER_PROTOCOL:-HTTP}
[[ "$RUNTIME_NAME" =~ ^[A-Za-z][A-Za-z0-9_]{0,47}$ ]] || die "Invalid runtime name."
[[ "$EXEC_ROLE_NAME" =~ ^[A-Za-z0-9_+=,.@-]{1,64}$ ]] || die "Invalid execution-role name."

# Preflight existing selections before unrelated setup writes. A new role needs no stack outputs yet.
if [ "${AMLAS_RUNTIME_PREFLIGHT_ONLY:-0}" = "1" ]; then
  if ! ROLE_PREFLIGHT_DOCUMENT="$(aws iam get-role --role-name "$EXEC_ROLE_NAME" --output json 2>&1)"; then
    case "$ROLE_PREFLIGHT_DOCUMENT" in
      *NoSuchEntity*) echo "RUNTIME-ROLE-PREFLIGHT PASS — new selected role"; exit 0 ;;
      *) die "Selected runtime role could not be verified; no setup resources were changed." ;;
    esac
  fi
fi

# ADR-0002: the runtime's compute subagents DISPATCH work to managed services. Pin the backend
# selectors + the SageMaker exec-role in the runtime Environment so run_processing_job /
# submit_tabular_training actually submit real SageMaker Processing/Training jobs (NOT the offline
# pandas fallback, which cannot run inside the container). Overridable via env; sane defaults:
#   ETL_BACKEND=processing  DQ_BACKEND=processing  → dispatch, not offline.
#   SAGEMAKER_EXECUTION_ROLE_ARN → the account's SageMaker exec role (the data-plane job role).
#
# FIX-33 (live-discovered): the env SAGEMAKER_EXECUTION_ROLE_ARN (the role the tool PASSES to
# CreateTrainingJob) MUST equal the role the PassSageMakerRole IAM statement is scoped to
# (SAGEMAKER_ROLE_ARN below = amlas-sagemaker-exec). A prior default of a DIFFERENT role
# (AmazonSageMaker-ExecutionRole-*) meant CreateTrainingJob tried to pass a role the runtime-exec
# role had NO iam:PassRole grant for → AccessDeniedException, blocking every dispatch. Default to
# the canonical amlas-sagemaker-exec, which (a) exists and (b) is what AsyncTrainingStack exports
# (SageMakerExecutionRoleArnOut). Still env-overridable, but the override must match a PassRole grant.
_SM_ROLE_ARN=${SAGEMAKER_EXECUTION_ROLE_ARN:-${AMLAS_SAGEMAKER_ROLE_ARN:-arn:aws:iam::${ACCT}:role/${DEPLOYMENT_PREFIX}-sagemaker-exec}}
# BUGFIX (cross-account deploy): reject a stale env SageMaker-role ARN that points at a DIFFERENT
# account than the live one — a .env carried over from another account (e.g. account 000000000000)
# would otherwise pass a foreign role the runtime-exec role has no PassRole for. If the resolved
# ARN's embedded account != ${ACCT}, fall back to this account's canonical amlas-sagemaker-exec.
_SM_ROLE_ACCT=$(printf '%s' "$_SM_ROLE_ARN" | sed -n 's#^arn:aws:iam::\([0-9]\{12\}\):.*#\1#p')
if [ -n "${_SM_ROLE_ACCT}" ] && [ "${_SM_ROLE_ACCT}" != "${ACCT}" ]; then
  echo "WARN: SAGEMAKER role ARN ${_SM_ROLE_ARN} is for account ${_SM_ROLE_ACCT}, not ${ACCT} — overriding with this account's amlas-sagemaker-exec."
  _SM_ROLE_ARN="arn:aws:iam::${ACCT}:role/${DEPLOYMENT_PREFIX}-sagemaker-exec"
fi
_ETL_BACKEND=${ETL_BACKEND:-processing}
_DQ_BACKEND=${DQ_BACKEND:-processing}
# CLI shorthand: comma-separated key=value pairs.
RUNTIME_ENV="ETL_BACKEND=${_ETL_BACKEND},DQ_BACKEND=${_DQ_BACKEND},SAGEMAKER_EXECUTION_ROLE_ARN=${_SM_ROLE_ARN},AMLAS_AWS_REGION=${REGION}"
# Local named profiles are never injected into the managed runtime: it uses its
# execution role. Preserve only explicitly selected model identifiers.
normalize_model_env || exit $?
for model_key in AMLAS_CHAT_MODEL AMLAS_MODEL_PLANNER AMLAS_MODEL_REVIEWER AMLAS_MODEL_SPECIALIST AMLAS_MODEL_BOUNDED AMLAS_MODEL_FINETUNE; do
  model_value="${!model_key:-}"
  if [ -n "$model_value" ]; then RUNTIME_ENV="${RUNTIME_ENV},${model_key}=${model_value}"; fi
done

# ADR-0007 harness routing. BOTH of these are required for a Lead to reach its harness, and
# BOTH fail SILENTLY when absent, so they are set here rather than left to the operator:
#
#   AMLAS_AWS_ACCOUNT_ID — ``harness_arn_for`` composes the harness ARN from the account id,
#     and ``aws_account_id`` is env-ONLY (ssm=None, required=True, no default). Unset, it
#     raises MissingRequiredConfigError, which the routing predicate CATCHES by design — so
#     the Lead quietly runs locally instead of failing. The runtime has no other source for
#     it: the role carries no sts:GetCallerIdentity and the account is never hardcoded.
#
#   AMLAS_HARNESS_LEADS — the opt-in. Unset means every Lead builds in-process, which is the
#     correct default (HarnessStack existing must never reroute a Lead by itself). Set it to
#     a comma-separated subset, or ``all``, to cut over. Left EMPTY here unless the caller
#     exports it, so a redeploy never silently changes where production runs.
RUNTIME_ENV="${RUNTIME_ENV},AMLAS_AWS_ACCOUNT_ID=${ACCT}"
if [ -n "${AMLAS_HARNESS_LEADS:-}" ]; then
  RUNTIME_ENV="${RUNTIME_ENV},AMLAS_HARNESS_LEADS=${AMLAS_HARNESS_LEADS}"
  echo "HARNESS ROUTING: AMLAS_HARNESS_LEADS=${AMLAS_HARNESS_LEADS} (these Leads will run on their managed harness)"
else
  echo "HARNESS ROUTING: AMLAS_HARNESS_LEADS unset — every Lead builds in-process (default)"
fi
echo "RUNTIME_ENV=${RUNTIME_ENV}"
require_runtime_cli_capabilities || exit $?
account_safety_gate "$ACCT"

# Session bucket the runtime-exec role must read/write (staging).
# BUGFIX (net-new / cross-account deploy): the comment always said "prefer the CFN export, then
# SSM, then env" but the code was env-FIRST — so a stale ``AMLAS_SESSION_BUCKET`` carried over in
# .env from a DIFFERENT account (e.g. account 000000000000) shadowed the freshly-deployed bucket,
# baking the WRONG account's bucket into the runtime env + SSM. The full-AWS dispatch then failed
# ``kms:GenerateDataKey ... resource does not exist`` uploading to the foreign bucket. Now truly
# CFN-FIRST (this deploy's AsyncTrainingStack export is authoritative), then SSM, then env as a
# last resort — and a stale/foreign env pin can no longer win.
SESSION_BUCKET=$(aws cloudformation describe-stacks --stack-name "${STACK_PREFIX}AsyncTrainingStack" --region "$REGION" \
  --query "Stacks[0].Outputs[?ExportName=='${DEPLOYMENT_PREFIX}-async-session-bucket'].OutputValue | [0]" --output text 2>/dev/null)
if [ -z "${SESSION_BUCKET}" ] || [ "${SESSION_BUCKET}" = "None" ]; then
  SESSION_BUCKET=$(aws ssm get-parameter --name "${SSM_PREFIX}/session-bucket" --region "$REGION" \
    --query 'Parameter.Value' --output text 2>/dev/null || echo "")
fi
if [ -z "${SESSION_BUCKET}" ] || [ "${SESSION_BUCKET}" = "None" ]; then
  # Neither CFN nor SSM resolved — fall back to the env pin only as a last resort.
  SESSION_BUCKET="${AMLAS_SESSION_BUCKET:-${SESSION_BUCKET:-}}"
  [ -n "${SESSION_BUCKET}" ] && [ "${SESSION_BUCKET}" != "None" ] && \
    echo "WARN: session bucket resolved from env (AMLAS_SESSION_BUCKET=${SESSION_BUCKET}) — CFN/SSM both empty; verify it belongs to account ${ACCT}."
fi
LAKE_BUCKET=${AMLAS_LAKE_BUCKET:-${DEPLOYMENT_PREFIX}-lake-${ACCT}-${REGION}}
# Same cross-account guard: an env AMLAS_LAKE_BUCKET embedding a different account id is stale —
# fall back to this account's canonical lake bucket name.
case "$LAKE_BUCKET" in
  amlas-lake-*-*)
    _lake_acct=$(printf '%s' "$LAKE_BUCKET" | sed -n 's#^amlas-lake-\([0-9]\{12\}\)-.*#\1#p')
    if [ -n "${_lake_acct}" ] && [ "${_lake_acct}" != "${ACCT}" ]; then
      echo "WARN: AMLAS_LAKE_BUCKET ${LAKE_BUCKET} is for account ${_lake_acct}, not ${ACCT} — overriding."
      LAKE_BUCKET="${DEPLOYMENT_PREFIX}-lake-${ACCT}-${REGION}"
    fi ;;
esac
# Full deployment publishes lake and guardrail settings. Materialize verified values
# so the managed role needs no broad SSM read, and a missing guardrail cannot become NO-OP.
LAKE_SETTINGS_DOCUMENT="$(aws cloudformation describe-stacks --stack-name "${STACK_PREFIX}LakeStack" --region "$REGION" --output json)" \
  || die "Selected lake settings could not be verified; no execution role was changed."
GUARDRAIL_SETTINGS_DOCUMENT="$(aws cloudformation describe-stacks --stack-name "${STACK_PREFIX}GuardrailStack" --region "$REGION" --output json)" \
  || die "Selected guardrail settings could not be verified; no execution role was changed."
VERIFIED_BINDINGS_JSON="$(LAKE_SETTINGS="$LAKE_SETTINGS_DOCUMENT" GUARDRAIL_SETTINGS="$GUARDRAIL_SETTINGS_DOCUMENT" \
  SELECTED_LAKE="$LAKE_BUCKET" ACCOUNT_VALUE="$ACCT" REGION_VALUE="$REGION" python3 - <<'BINDINGS'
import json, os, re, sys

def outputs(raw):
    stacks = json.loads(raw)["Stacks"]
    if len(stacks) != 1 or stacks[0].get("StackStatus") not in ("CREATE_COMPLETE", "UPDATE_COMPLETE"):
        raise ValueError("Selected stack is not verified complete.")
    return {item["OutputKey"]: item["OutputValue"] for item in stacks[0].get("Outputs", [])}

try:
    lake = outputs(os.environ["LAKE_SETTINGS"])
    guardrail = outputs(os.environ["GUARDRAIL_SETTINGS"])
    bucket = lake["LakeBucketName"]
    silver = lake["LakeSilverPrefix"]
    gold = lake["LakeGoldPrefix"]
    guardrail_id = guardrail["GuardrailIdOut"]
    guardrail_version = guardrail["GuardrailVersionOut"]
    guardrail_arn = guardrail["GuardrailArnOut"]
    if bucket != os.environ["SELECTED_LAKE"]:
        raise ValueError("Selected lake differs from its published binding.")
    base = "s3://" + bucket + "/"
    if (not all(isinstance(value, str) and value.startswith(base) and len(value) > len(base)
                and value.endswith("/") for value in (silver, gold)) or silver == gold):
        raise ValueError("Lake tier bindings are missing or do not belong to the selected bucket.")
    expected_arn = "arn:aws:bedrock:" + os.environ["REGION_VALUE"] + ":" + os.environ["ACCOUNT_VALUE"] + ":guardrail/" + guardrail_id
    if not re.fullmatch(r"[a-z0-9]+", guardrail_id) or guardrail_arn != expected_arn:
        raise ValueError("Guardrail identity differs from the selected account/Region binding.")
    if not re.fullmatch(r"DRAFT|[1-9][0-9]{0,4}", guardrail_version):
        raise ValueError("Guardrail version is missing or invalid.")
    print(json.dumps({"lake_silver_prefix": silver, "lake_gold_prefix": gold,
                      "guardrail_id": guardrail_id, "guardrail_version": guardrail_version,
                      "guardrail_arn": guardrail_arn}))
except (KeyError, TypeError, ValueError):
    print("Required selected lake/guardrail bindings are missing, incompatible or unverifiable; no execution role was changed.", file=sys.stderr)
    sys.exit(1)
BINDINGS
)" || exit $?
_binding_value() {
  BINDING_KEY="$1" BINDINGS_DOCUMENT="$VERIFIED_BINDINGS_JSON" python3 - <<'BINDING_VALUE'
import json, os
print(json.loads(os.environ["BINDINGS_DOCUMENT"])[os.environ["BINDING_KEY"]])
BINDING_VALUE
}
LAKE_SILVER_PREFIX=$(_binding_value lake_silver_prefix) || exit $?
LAKE_GOLD_PREFIX=$(_binding_value lake_gold_prefix) || exit $?
GUARDRAIL_ID=$(_binding_value guardrail_id) || exit $?
GUARDRAIL_VERSION=$(_binding_value guardrail_version) || exit $?
GUARDRAIL_ARN=$(_binding_value guardrail_arn) || exit $?
TAVILY_SECRET_SELECTION=${TAVILY_SECRET_ID:-${AMLAS_TAVILY_SECRET_ID:-${DEPLOYMENT_PREFIX}/tavily-api-key}}
TAVILY_SECRET_DOCUMENT="$(aws secretsmanager describe-secret --secret-id "$TAVILY_SECRET_SELECTION" --region "$REGION" --output json)" \
  || die "Selected Tavily secret metadata could not be verified; no execution role was changed."
TAVILY_SECRET_ARN="$(SECRET_DOCUMENT="$TAVILY_SECRET_DOCUMENT" SECRET_SELECTION="$TAVILY_SECRET_SELECTION" \
  ACCOUNT_VALUE="$ACCT" REGION_VALUE="$REGION" python3 - <<'SECRET_METADATA'
import json, os, sys
try:
    secret = json.loads(os.environ["SECRET_DOCUMENT"])
    arn = secret["ARN"]
    selected = os.environ["SECRET_SELECTION"]
    prefix = "arn:aws:secretsmanager:" + os.environ["REGION_VALUE"] + ":" + os.environ["ACCOUNT_VALUE"] + ":secret:"
    if not isinstance(arn, str) or not arn.startswith(prefix):
        raise ValueError("Secret belongs to a different account/Region.")
    if (selected.startswith("arn:") and arn != selected) or (not selected.startswith("arn:") and secret.get("Name") != selected):
        raise ValueError("Secret identity differs from the selected secret.")
    print(arn)
except (KeyError, TypeError, ValueError):
    print("Selected Tavily secret identity is incompatible or unverifiable; no execution role was changed.", file=sys.stderr)
    sys.exit(1)
SECRET_METADATA
)" || exit $?

# FIX-33: the iam:PassRole grant MUST cover exactly the role the tool passes (_SM_ROLE_ARN),
# so derive it from that single source of truth — an env override of the passed role then
# automatically gets its matching PassRole grant (no silent drift → no AccessDenied).
SAGEMAKER_ROLE_ARN="${_SM_ROLE_ARN}"

# FIX-11: the session + lake buckets are SSE-KMS encrypted with per-stack CMKs
# (AsyncTrainingStack `alias/amlas-async-training`, LakeStack `alias/amlas-lake`).
# The runtime's GetObject on the staged dataset needs kms:Decrypt on the bucket's
# CMK — S3 RW alone is NOT enough (CW: "not authorized to perform kms:Decrypt").
# Discover each bucket's CMK ARN from its encryption config so the exec-role KMS
# statement is scoped to exactly those keys (no wildcard). Resolve to full ARNs.
_bucket_kms_arn() {
  local b="$1"
  local kid
  kid=$(aws s3api get-bucket-encryption --bucket "$b" --region "$REGION" \
    --query 'ServerSideEncryptionConfiguration.Rules[0].ApplyServerSideEncryptionByDefault.KMSMasterKeyID' \
    --output text) || return $?
  [ -z "$kid" ] || [ "$kid" = "None" ] && { echo ""; return; }
  case "$kid" in
    arn:aws:kms:*) echo "$kid" ;;                                    # already an ARN
    alias/*) aws kms describe-key --key-id "$kid" --region "$REGION" --query 'KeyMetadata.Arn' --output text 2>/dev/null ;;
    *) echo "arn:aws:kms:${REGION}:${ACCT}:key/${kid}" ;;            # bare key id
  esac
}
SESSION_KMS_ARN=$(_bucket_kms_arn "$SESSION_BUCKET") || die "Session-bucket encryption could not be verified; no execution role was changed."
LAKE_KMS_ARN=$(_bucket_kms_arn "$LAKE_BUCKET") || die "Lake-bucket encryption could not be verified; no execution role was changed."
# Build a JSON array of the discovered (non-empty) CMK ARNs for the KMS statement.
KMS_RESOURCES=$(python3 -c "import json,sys; arns=[a for a in ['${SESSION_KMS_ARN}','${LAKE_KMS_ARN}'] if a and a!='None']; print(json.dumps(arns))") || die "Could not serialize verified bucket key resources."
[ -z "$KMS_RESOURCES" ] && KMS_RESOURCES='[]'

# ADR-0002 dispatch blocker (live-discovered 2026-07-21): submit_sagemaker_job writes job state
# to the DynamoDB table `amlas-training-jobs`, which AsyncTrainingStack encrypts with a per-stack
# CMK. DynamoDB SSE-with-CMK requires the CALLER (this exec-role) to hold kms:Decrypt +
# kms:GenerateDataKey on that key — the SseKmsForBuckets statement is S3-scoped (kms:ViaService
# s3.*) and does NOT cover it, so PutItem failed KMS-AccessDenied and the trainer could not
# dispatch (CW: "the role ... lacks kms:Decrypt ... KMS error on DynamoDB put_item"). Discover the
# table's CMK ARN dynamically (SEC-01: no hardcoded ARNs) and scope a dynamodb.* KMS grant to it.
DDB_JOBS_TABLE=${AMLAS_JOBS_TABLE:-${DEPLOYMENT_PREFIX}-training-jobs}
DDB_KMS_ARN=$(aws dynamodb describe-table --table-name "$DDB_JOBS_TABLE" --region "$REGION" \
  --query 'Table.SSEDescription.KMSMasterKeyArn' --output text) || die "Training-table encryption could not be verified; no execution role was changed."
[ "$DDB_KMS_ARN" = "None" ] && DDB_KMS_ARN=""
DDB_KMS_RESOURCES=$(python3 -c "import json; arn='${DDB_KMS_ARN}'; print(json.dumps([arn] if arn else []))") || die "Could not serialize verified table key resources."
[ -z "$DDB_KMS_RESOURCES" ] && DDB_KMS_RESOURCES='[]'

echo "ACCT=${ACCT} REGION=${REGION}"
echo "RUNTIME_NAME=${RUNTIME_NAME} IMAGE_URI=${IMAGE_URI}"
echo "EXEC_ROLE_NAME=${EXEC_ROLE_NAME}"
echo "SESSION_BUCKET=${SESSION_BUCKET:-<unresolved>} LAKE_BUCKET=${LAKE_BUCKET}"

# FAIL LOUD on an unresolved session bucket instead of baking an EMPTY ARN into the policy.
# With SESSION_BUCKET empty, the SessionBucketRW statement below renders as
#   "Resource": ["arn:aws:s3:::", "arn:aws:s3:::/*"]
# which IAM ACCEPTS and which grants nothing. The runtime then reaches READY, streams
# `spec_loaded`, and dies on the first staging read with a bare
#   An error occurred (403) when calling the HeadObject operation: Forbidden
# — no mention of a bucket, a policy, or this script. Measured: that is exactly how the first
# live E2E failed, because the role was provisioned before AsyncTrainingStack existed.
# An unresolved bucket means the deploy is not ready; say so here, where the fix is obvious.
if [ -z "${SESSION_BUCKET}" ] || [ "${SESSION_BUCKET}" = "None" ]; then
  echo "ERROR: session bucket unresolved (CFN AsyncTrainingStack → SSM /amlas/session-bucket → AMLAS_SESSION_BUCKET all empty)." >&2
  echo "       Refusing to grant 'arn:aws:s3:::' — deploy AsyncTrainingStack first, or export AMLAS_SESSION_BUCKET." >&2
  exit 1
fi
# ADR-0002: submit_tabular_training stages train/test to the SESSION bucket before dispatching to
# SageMaker, so the runtime must carry the canonical session-bucket setting.
# SESSION_BUCKET is only resolved above (CFN→SSM→env), so rebuild RUNTIME_ENV here to include it.
if [ -n "${SESSION_BUCKET}" ] && [ "${SESSION_BUCKET}" != "None" ]; then
  RUNTIME_ENV="${RUNTIME_ENV},AMLAS_SESSION_BUCKET=${SESSION_BUCKET}"
fi
echo "RUNTIME_ENV(final)=${RUNTIME_ENV}"
echo "SESSION_KMS_ARN=${SESSION_KMS_ARN:-<none>} LAKE_KMS_ARN=${LAKE_KMS_ARN:-<none>}"
echo "KMS_RESOURCES=${KMS_RESOURCES}"
echo "DDB_JOBS_TABLE=${DDB_JOBS_TABLE} DDB_KMS_ARN=${DDB_KMS_ARN:-<none>}"
echo "DDB_KMS_RESOURCES=${DDB_KMS_RESOURCES}"

# ---------------------------------------------------------------------------
# STEP 1 — ensure the runtime execution role (create-or-discover).
# ---------------------------------------------------------------------------
TRUST_JSON=$(cat <<JSON
{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"bedrock-agentcore.amazonaws.com"},"Action":"sts:AssumeRole","Condition":{"StringEquals":{"aws:SourceAccount":"${ACCT}"},"ArnLike":{"aws:SourceArn":"arn:aws:bedrock-agentcore:${REGION}:${ACCT}:*"}}}]}
JSON
)

# Shared with CDK and metadata preflight; no separately copied model allowlist.
BEDROCK_RESOURCES="$(uv run --frozen python - "$ACCT" "$REGION" <<'MODELS'
import json
import sys
from amlas.config.runtime import bedrock_invoke_resources
print(json.dumps(bedrock_invoke_resources(sys.argv[1], sys.argv[2])))
MODELS
)" || die "Could not resolve the reviewed Bedrock invocation catalog."

# NOTE (AgentCorePrimitives Resource:*): code-interpreter/memory session ids are
# service-generated at runtime (no stable ARN pattern to pre-scope), and the
# account/role boundary already confines usage. IAM rejects arbitrary keys inside
# a Statement element (MalformedPolicyDocument), so this rationale lives here as a
# bash comment rather than an inline "_comment_*" JSON key (FIX: that inline key
# was the root cause of the Phase-5 MalformedPolicyDocument failure).
#
# SECURITY POSTURE (IAM review):
# Bedrock AgentCore does NOT yet expose resource ARNs or condition keys for
# per-session ownership on CreateCodeInterpreter/InvokeCodeInterpreter/CreateMemory
# etc., so ``Resource:"*"`` is the ONLY grantable form for these actions today. Risk
# is bounded by (a) the single-account, single-runtime-role boundary and (b) these
# being control-plane session primitives (not data exfiltration). CUSTOMERS ADOPTING
# THIS: re-scope to session ARNs / add ownership Condition keys the moment AgentCore
# supports them; until then this is the least-privilege the service permits.
#
# ADR-0007 harness routing needs exactly two grants, and unlike AgentCorePrimitives above BOTH
# are resource-scopable, so neither uses "*":
#
#   InvokeOwnHarnesses — the Service Authorization Reference lists InvokeHarness against the
#     required resource types ``harness`` (arn:...:harness/${HarnessId}) and ``harness-endpoint``
#     (arn:...:harness/${HarnessId}/harness-endpoint/${Name}). One ``harness/*`` covers both,
#     confined to this account and region. This is the RUNTIME's role — do not confuse it with
#     the harnesses' own execution role in HarnessStack, which must OMIT InvokeAgentRuntimeCommand.
#
#     ⚠️ ``InvokeHarness`` ALONE IS NOT ENOUGH, and the reference does not say so. MEASURED
#     against the live service: with only ``bedrock-agentcore:InvokeHarness`` granted, the call
#     fails with
#       AccessDeniedException: ... is not authorized to perform:
#       bedrock-agentcore:InvokeAgentRuntime on resource: .../harness/research_lead-...
#     so the InvokeHarness API authorizes against the AgentRuntime action on the harness
#     resource as well. The same doubled-up shape is visible in the reference for DeleteHarness,
#     which lists BOTH ``DeleteAgentRuntime`` and ``DeleteHarness``. Grant both; dropping
#     ``InvokeAgentRuntime`` re-breaks every harness-hosted Lead at the first turn. Note the
#     resource stays the HARNESS ARN — this is not a grant over agent runtimes.
#
#   ReadHarnessIds — HarnessStack publishes each harness id at
#     ``/amlas/harness/<lead-name>-id`` and ``harness_arn_for`` composes the ARN from it.
#     Scoped to that prefix only: the runtime has no reason to read the rest of ``/amlas``.
#     Without this the read returns "" and the Lead SILENTLY runs locally, because the routing
#     predicate is deliberately non-raising — a denied read is indistinguishable from "not
#     deployed". That is the failure mode a live cutover cannot see, hence the tight grant plus
#     the AMLAS_AWS_ACCOUNT_ID env var set above.
# Preserve historical default-role grants; isolated deployments read/stop only their own jobs.
JOB_READ_SCOPE="*"
[ "$DEPLOYMENT_PREFIX" = "amlas" ] || JOB_READ_SCOPE="${DEPLOYMENT_PREFIX}-*"
POLICY_JSON=$(cat <<JSON
{
  "Version": "2012-10-17",
  "Statement": [
    {"Sid":"BedrockInvoke","Effect":"Allow","Action":["bedrock:InvokeModel","bedrock:InvokeModelWithResponseStream"],"Resource":${BEDROCK_RESOURCES}},
    {"Sid":"ApplySelectedGuardrail","Effect":"Allow","Action":["bedrock:ApplyGuardrail"],"Resource":"${GUARDRAIL_ARN}"},
    {"Sid":"EcrPullAuth","Effect":"Allow","Action":["ecr:GetAuthorizationToken"],"Resource":"*"},
    {"Sid":"EcrPullImage","Effect":"Allow","Action":["ecr:BatchCheckLayerAvailability","ecr:GetDownloadUrlForLayer","ecr:BatchGetImage"],"Resource":"arn:aws:ecr:${REGION}:${ACCT}:repository/${DEPLOYMENT_PREFIX}-runtime"},
    {"Sid":"SessionBucketRW","Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],"Resource":["arn:aws:s3:::${SESSION_BUCKET}","arn:aws:s3:::${SESSION_BUCKET}/*"]},
    {"Sid":"LakeBucketRW","Effect":"Allow","Action":["s3:GetObject","s3:PutObject","s3:DeleteObject","s3:ListBucket"],"Resource":["arn:aws:s3:::${LAKE_BUCKET}","arn:aws:s3:::${LAKE_BUCKET}/*"]},
    {"Sid":"CloudWatchLogs","Effect":"Allow","Action":["logs:CreateLogGroup","logs:CreateLogStream","logs:PutLogEvents","logs:DescribeLogStreams"],"Resource":"arn:aws:logs:${REGION}:${ACCT}:log-group:/aws/bedrock-agentcore/*"},
    {"Sid":"SageMakerCreateScoped","Effect":"Allow","Action":["sagemaker:CreateTrainingJob","sagemaker:CreateProcessingJob"],"Resource":["arn:aws:sagemaker:${REGION}:${ACCT}:training-job/${DEPLOYMENT_PREFIX}-*","arn:aws:sagemaker:${REGION}:${ACCT}:processing-job/${DEPLOYMENT_PREFIX}-*"]},
    {"Sid":"SageMakerJobTags","Effect":"Allow","Action":["sagemaker:AddTags"],"Resource":["arn:aws:sagemaker:${REGION}:${ACCT}:training-job/${DEPLOYMENT_PREFIX}-*","arn:aws:sagemaker:${REGION}:${ACCT}:processing-job/${DEPLOYMENT_PREFIX}-*"]},
    {"Sid":"SageMakerList","Effect":"Allow","Action":["sagemaker:ListTrainingJobs","sagemaker:ListProcessingJobs"],"Resource":"*"},
    {"Sid":"SageMakerScoped","Effect":"Allow","Action":["sagemaker:DescribeTrainingJob","sagemaker:StopTrainingJob","sagemaker:DescribeProcessingJob","sagemaker:StopProcessingJob"],"Resource":["arn:aws:sagemaker:${REGION}:${ACCT}:training-job/${JOB_READ_SCOPE}","arn:aws:sagemaker:${REGION}:${ACCT}:processing-job/${JOB_READ_SCOPE}"]},
    {"Sid":"PassSageMakerRole","Effect":"Allow","Action":["iam:PassRole"],"Resource":"${SAGEMAKER_ROLE_ARN}","Condition":{"StringEquals":{"iam:PassedToService":"sagemaker.amazonaws.com"}}},
    {"Sid":"DdbJobs","Effect":"Allow","Action":["dynamodb:GetItem","dynamodb:PutItem","dynamodb:UpdateItem","dynamodb:Query"],"Resource":["arn:aws:dynamodb:${REGION}:${ACCT}:table/${DDB_JOBS_TABLE}","arn:aws:dynamodb:${REGION}:${ACCT}:table/${DDB_JOBS_TABLE}/index/*"]},
    {"Sid":"EventsPut","Effect":"Allow","Action":["events:PutEvents"],"Resource":"arn:aws:events:${REGION}:${ACCT}:event-bus/default"},
    {"Sid":"AgentCorePrimitives","Effect":"Allow","Action":["bedrock-agentcore:CreateCodeInterpreter","bedrock-agentcore:StartCodeInterpreterSession","bedrock-agentcore:InvokeCodeInterpreter","bedrock-agentcore:StopCodeInterpreterSession","bedrock-agentcore:ListCodeInterpreters","bedrock-agentcore:GetCodeInterpreter","bedrock-agentcore:CreateMemory","bedrock-agentcore:GetMemory","bedrock-agentcore:ListEvents","bedrock-agentcore:CreateEvent","bedrock-agentcore:ListSessions"],"Resource":"*"},
    {"Sid":"InvokeOwnHarnesses","Effect":"Allow","Action":["bedrock-agentcore:InvokeHarness","bedrock-agentcore:InvokeAgentRuntime"],"Resource":"arn:aws:bedrock-agentcore:${REGION}:${ACCT}:harness/*"},
    {"Sid":"ReadHarnessIds","Effect":"Allow","Action":["ssm:GetParameter"],"Resource":"arn:aws:ssm:${REGION}:${ACCT}:parameter${SSM_PREFIX}/harness/*"},
    {"Sid":"TavilySecretRead","Effect":"Allow","Action":["secretsmanager:GetSecretValue","secretsmanager:DescribeSecret"],"Resource":"${TAVILY_SECRET_ARN}"},
    {"Sid":"SseKmsForBuckets","Effect":"Allow","Action":["kms:Decrypt","kms:GenerateDataKey","kms:DescribeKey"],"Resource":${KMS_RESOURCES},"Condition":{"StringEquals":{"kms:ViaService":"s3.${REGION}.amazonaws.com"}}},
    {"Sid":"SseKmsForDdbJobs","Effect":"Allow","Action":["kms:Decrypt","kms:GenerateDataKey","kms:DescribeKey"],"Resource":${DDB_KMS_RESOURCES},"Condition":{"StringEquals":{"kms:ViaService":"dynamodb.${REGION}.amazonaws.com"}}}
  ]
}
JSON
)
# FIX-11: if CMK discovery yielded no keys (e.g. buckets are SSE-S3 not SSE-KMS),
# drop the affected statement entirely — an empty "Resource":[] is an invalid IAM
# policy. python re-emits the doc with the empty-resource statement(s) filtered out.
# This covers BOTH the S3-bucket CMK statement and the DDB-jobs-table CMK statement
# (the DDB table may be SSE-S3/AWS-owned-key in some accounts, in which case no
# caller-side KMS grant is needed and the statement must be omitted).
_DROP_SIDS=""
[ "$KMS_RESOURCES" = "[]" ] && _DROP_SIDS="${_DROP_SIDS} SseKmsForBuckets"
[ "$DDB_KMS_RESOURCES" = "[]" ] && _DROP_SIDS="${_DROP_SIDS} SseKmsForDdbJobs"
if [ -n "${_DROP_SIDS// /}" ]; then
  POLICY_JSON=$(printf '%s' "$POLICY_JSON" | DROP_SIDS="$_DROP_SIDS" python3 -c 'import sys,json,os
drop=set(os.environ.get("DROP_SIDS","").split())
d=json.load(sys.stdin)
d["Statement"]=[s for s in d["Statement"] if s.get("Sid") not in drop]
print(json.dumps(d))')
  echo "NOTE: omitted empty-resource KMS statement(s):${_DROP_SIDS} (SSE-S3 / AWS-owned key)"
fi

# A compatible selected role remains immutable. New roles belong to an auxiliary
# stack whose recorded standard service role performs rollback and deletion too.
verify_role_documents() {
  DESIRED_TRUST="$TRUST_JSON" DESIRED_POLICY="$POLICY_JSON" \
    EXISTING_ROLE="$ROLE_DOCUMENT" EXISTING_POLICY="$POLICY_DOCUMENT" python3 - <<'COMPARE'
import json, os, sys
from urllib.parse import unquote

def policy(value):
    if isinstance(value, str):
        value = json.loads(unquote(value))
    return value

def normalized(value):
    if isinstance(value, dict):
        return {key: normalized(item) for key, item in value.items()}
    if isinstance(value, list):
        return sorted((normalized(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    return value

try:
    role = json.loads(os.environ["EXISTING_ROLE"])["Role"]
    existing_policy = json.loads(os.environ["EXISTING_POLICY"])["PolicyDocument"]
    compatible = (normalized(policy(role["AssumeRolePolicyDocument"])) == normalized(json.loads(os.environ["DESIRED_TRUST"])) and
                  normalized(policy(existing_policy)) == normalized(json.loads(os.environ["DESIRED_POLICY"])))
    if not compatible:
        raise ValueError("Existing role trust or inline policy differs from this deployment.")
    print(role["Arn"])
except (ValueError, TypeError, KeyError):
    print("Existing execution role is incompatible or unverifiable; choose a distinct --runtime-role-name. No role policy was changed.", file=sys.stderr)
    sys.exit(1)
COMPARE
}
create_runtime_role_stack() {
  local bootstrap_resources bootstrap_role_name bootstrap_role_document cfn_role_arn
  local existing_stack role_template stack_tags stack_document
  bootstrap_resources="$(aws cloudformation list-stack-resources --stack-name CDKToolkit --region "$REGION" --output json)" \
    || die "Could not verify the existing CDK CloudFormation cleanup route; no role was created."
  bootstrap_role_name="$(BOOTSTRAP_RESOURCES="$bootstrap_resources" python3 - <<'BOOTSTRAP'
import json, os, sys
try:
    resources = json.loads(os.environ["BOOTSTRAP_RESOURCES"])["StackResourceSummaries"]
    matches = [item for item in resources if item.get("LogicalResourceId") == "CloudFormationExecutionRole"
               and item.get("ResourceType") == "AWS::IAM::Role"
               and item.get("ResourceStatus") in ("CREATE_COMPLETE", "UPDATE_COMPLETE")]
    if len(matches) != 1 or not matches[0].get("PhysicalResourceId"):
        raise ValueError("Missing or ambiguous standard CloudFormation execution role.")
    print(matches[0]["PhysicalResourceId"])
except (KeyError, TypeError, ValueError):
    print("Existing CDK CloudFormation cleanup route is unverifiable; no role was created.", file=sys.stderr)
    sys.exit(1)
BOOTSTRAP
)" || return $?
  bootstrap_role_document="$(aws iam get-role --role-name "$bootstrap_role_name" --output json)" \
    || die "The existing CloudFormation execution role is unreadable; no role was created."
  cfn_role_arn="$(BOOTSTRAP_ROLE="$bootstrap_role_document" BOOTSTRAP_NAME="$bootstrap_role_name" ACCOUNT_VALUE="$ACCT" python3 - <<'ISSUER'
import json, os, sys
try:
    role = json.loads(os.environ["BOOTSTRAP_ROLE"])["Role"]
    expected = "arn:aws:iam::" + os.environ["ACCOUNT_VALUE"] + ":role/" + os.environ["BOOTSTRAP_NAME"]
    statements = role["AssumeRolePolicyDocument"]["Statement"]
    if role["Arn"] != expected or role.get("PermissionsBoundary"):
        raise ValueError("Execution role identity or boundary differs from the standard route.")
    if statements != [{"Effect": "Allow", "Principal": {"Service": "cloudformation.amazonaws.com"}, "Action": "sts:AssumeRole"}]:
        raise ValueError("Execution role trust differs from the standard service issuer.")
    print(role["Arn"])
except (KeyError, TypeError, ValueError):
    print("Existing standard CloudFormation service issuer is incompatible or unverifiable; no role was created.", file=sys.stderr)
    sys.exit(1)
ISSUER
)" || return $?
  if existing_stack="$(aws cloudformation describe-stacks --stack-name "$ROLE_STACK_NAME" --region "$REGION" --output json 2>&1)"; then
    die "The selected runtime-role lifecycle stack already exists while its role is absent; refusing to change it."
  else
    case "$existing_stack" in
      *ValidationError*does\ not\ exist*) ;;
      *) die "Runtime-role stack absence could not be verified; no role was created." ;;
    esac
  fi
  role_template="$(DESIRED_TRUST="$TRUST_JSON" DESIRED_POLICY="$POLICY_JSON" \
    ROLE_NAME_VALUE="$EXEC_ROLE_NAME" POLICY_NAME_VALUE="$ROLE_POLICY_NAME" PREFIX_VALUE="$DEPLOYMENT_PREFIX" DEPLOYMENT_ID_VALUE="$DEPLOYMENT_ID" python3 - <<'TEMPLATE'
import json, os
prefix = os.environ["PREFIX_VALUE"]
tags = [{"Key": "amlas:deployment-prefix", "Value": prefix},
        {"Key": "amlas:component", "Value": "runtime-role"}]
if os.environ["DEPLOYMENT_ID_VALUE"]:
    tags.append({"Key": "amlas:deployment-id", "Value": os.environ["DEPLOYMENT_ID_VALUE"]})
print(json.dumps({"AWSTemplateFormatVersion": "2010-09-09", "Resources": {
    "RuntimeExecutionRole": {"Type": "AWS::IAM::Role", "Properties": {
        "RoleName": os.environ["ROLE_NAME_VALUE"],
        "Description": "AgentCore runtime execution role managed by its deployment lifecycle stack",
        "AssumeRolePolicyDocument": json.loads(os.environ["DESIRED_TRUST"]),
        "Policies": [{"PolicyName": os.environ["POLICY_NAME_VALUE"],
                      "PolicyDocument": json.loads(os.environ["DESIRED_POLICY"])}],
        "Tags": tags}}}, "Outputs": {"RuntimeRoleArn": {
            "Value": {"Fn::GetAtt": ["RuntimeExecutionRole", "Arn"]}}}}))
TEMPLATE
)" || die "Could not serialize the runtime-role lifecycle template."
  stack_tags="$(PREFIX_VALUE="$DEPLOYMENT_PREFIX" DEPLOYMENT_ID_VALUE="$DEPLOYMENT_ID" python3 - <<'TAGS'
import json, os
tags = [{"Key": "amlas:deployment-prefix", "Value": os.environ["PREFIX_VALUE"]},
        {"Key": "amlas:component", "Value": "runtime-role"}]
if os.environ["DEPLOYMENT_ID_VALUE"]:
    tags.append({"Key": "amlas:deployment-id", "Value": os.environ["DEPLOYMENT_ID_VALUE"]})
print(json.dumps(tags))
TAGS
)" || die "Could not serialize the runtime-role ownership tags."
  aws cloudformation validate-template --template-body "$role_template" --region "$REGION" >/dev/null \
    || die "Runtime-role lifecycle template validation failed; no role was created."
  echo "Creating CloudFormation-owned runtime-role stack ${ROLE_STACK_NAME}."
  ROLE_STACK_ARN="$(aws cloudformation create-stack --stack-name "$ROLE_STACK_NAME" --region "$REGION" \
    --role-arn "$cfn_role_arn" --capabilities CAPABILITY_NAMED_IAM --template-body "$role_template" \
    --tags "$stack_tags" --timeout-in-minutes 15 --query StackId --output text)" || return $?
  case "$ROLE_STACK_ARN" in
    arn:aws:cloudformation:"$REGION":"$ACCT":stack/"$ROLE_STACK_NAME"/*) ;;
    *) die "CloudFormation returned an invalid runtime-role cleanup handle." ;;
  esac
  # Emit immediately: a later waiter/readback/runtime failure still has a cleanup handle.
  printf 'RESOLVED_RUNTIME_ROLE_STACK_ARN=%s\n' "$ROLE_STACK_ARN"
  aws cloudformation wait stack-create-complete --stack-name "$ROLE_STACK_ARN" --region "$REGION" \
    || die "Runtime-role stack creation did not complete; retain its recorded cleanup handle."
  stack_document="$(aws cloudformation describe-stacks --stack-name "$ROLE_STACK_ARN" --region "$REGION" --output json)" \
    || die "Created runtime-role stack could not be verified; retain its recorded cleanup handle."
  STACK_DOCUMENT="$stack_document" STACK_ARN_VALUE="$ROLE_STACK_ARN" CFN_ROLE_VALUE="$cfn_role_arn" \
    ROLE_ARN_VALUE="arn:aws:iam::${ACCT}:role/${EXEC_ROLE_NAME}" python3 - <<'STACK'
import json, os, sys
try:
    stack = json.loads(os.environ["STACK_DOCUMENT"])["Stacks"][0]
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack.get("Outputs", [])}
    if (stack["StackId"] != os.environ["STACK_ARN_VALUE"] or stack["StackStatus"] != "CREATE_COMPLETE"
            or stack.get("RoleARN") != os.environ["CFN_ROLE_VALUE"]
            or outputs.get("RuntimeRoleArn") != os.environ["ROLE_ARN_VALUE"]):
        raise ValueError("Stack state, service issuer or role output mismatches the request.")
except (IndexError, KeyError, TypeError, ValueError):
    print("Created runtime-role lifecycle stack failed verification; managed-runtime writes were not attempted.", file=sys.stderr)
    sys.exit(1)
STACK
  [ "$?" = "0" ] || return 1
  ROLE_DOCUMENT="$(aws iam get-role --role-name "$EXEC_ROLE_NAME" --output json)" \
    || die "CloudFormation-created role could not be read back."
  POLICY_DOCUMENT="$(aws iam get-role-policy --role-name "$EXEC_ROLE_NAME" --policy-name "$ROLE_POLICY_NAME" --output json)" \
    || die "CloudFormation-created role policy could not be read back."
  ROLE_ARN="$(verify_role_documents)" || return $?
}
if ROLE_DOCUMENT="$(aws iam get-role --role-name "$EXEC_ROLE_NAME" --output json 2>&1)"; then
  POLICY_DOCUMENT="$(aws iam get-role-policy --role-name "$EXEC_ROLE_NAME" --policy-name "$ROLE_POLICY_NAME" --output json 2>&1)" || die "Existing role policy is unverifiable; choose a distinct --runtime-role-name."
  ROLE_ARN="$(verify_role_documents)" || exit $?
  case "$ROLE_ARN" in arn:*:iam::"$ACCT":role/*) ;; *) die "IAM returned an invalid execution-role ARN." ;; esac
  ROLE_ID="$(ROLE_DOCUMENT_VALUE="$ROLE_DOCUMENT" python3 - <<'ROLE_IDENTITY'
import json, os, re, sys
try:
    role_id = json.loads(os.environ["ROLE_DOCUMENT_VALUE"])["Role"]["RoleId"]
    if not isinstance(role_id, str) or not re.fullmatch(r"AROA[A-Z0-9]+", role_id):
        raise ValueError("Invalid native role identity")
except (KeyError, TypeError, ValueError):
    print("Compatible execution role has no verifiable native RoleId.", file=sys.stderr)
    sys.exit(1)
print(role_id)
ROLE_IDENTITY
)" || exit $?
  printf 'RESOLVED_REUSED_RUNTIME_ROLE_ARN=%s\n' "$ROLE_ARN"
  printf 'RESOLVED_REUSED_RUNTIME_ROLE_ID=%s\n' "$ROLE_ID"
  echo "Reusing compatible execution role without changing its trust or grants."
else
  [ "${AMLAS_RUNTIME_PREFLIGHT_ONLY:-0}" != "1" ] || die "Selected existing role changed during preflight; retry before writing resources."
  case "$ROLE_DOCUMENT" in
    *NoSuchEntity*) create_runtime_role_stack || exit $? ;;
    *) die "Execution role could not be read; refusing to create it after an unverifiable IAM response." ;;
  esac
fi
if [ "${AMLAS_RUNTIME_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "RUNTIME-ROLE-PREFLIGHT PASS — compatible existing role"
  exit 0
fi
printf 'RESOLVED_RUNTIME_ROLE_ARN=%s\n' "$ROLE_ARN"
sleep 20
MODEL_PACKAGE_GROUP_VALUE=${AMLAS_MODEL_PACKAGE_GROUP:-${DEPLOYMENT_PREFIX}-models}
FEATURE_STORE_DB_VALUE=${AMLAS_FEATURE_STORE_DB:-${DEPLOYMENT_PREFIX//-/_}_features}
RUNTIME_TAGS_JSON="$(PREFIX_VALUE="$DEPLOYMENT_PREFIX" DEPLOYMENT_ID_VALUE="$DEPLOYMENT_ID" python3 - <<'RUNTIME_TAGS'
import json, os
tags = {"amlas:deployment-prefix": os.environ["PREFIX_VALUE"]}
if os.environ["DEPLOYMENT_ID_VALUE"]:
    tags["amlas:deployment-id"] = os.environ["DEPLOYMENT_ID_VALUE"]
print(json.dumps(tags))
RUNTIME_TAGS
)" || die "Could not serialize managed-runtime ownership tags."

# A JSON map preserves comma-containing values such as AMLAS_HARNESS_LEADS.
RUNTIME_ENV_JSON="$(ETL_VALUE="$_ETL_BACKEND" DQ_VALUE="$_DQ_BACKEND" ROLE_VALUE="$_SM_ROLE_ARN" \
  REGION_VALUE="$REGION" ACCOUNT_VALUE="$ACCT" SESSION_VALUE="$SESSION_BUCKET" \
  PREFIX_VALUE="$DEPLOYMENT_PREFIX" SSM_PREFIX_VALUE="$SSM_PREFIX" JOBS_VALUE="$DDB_JOBS_TABLE" LAKE_VALUE="$LAKE_BUCKET" \
  DEPLOYMENT_ID_VALUE="$DEPLOYMENT_ID" MODEL_GROUP_VALUE="$MODEL_PACKAGE_GROUP_VALUE" FEATURE_DB_VALUE="$FEATURE_STORE_DB_VALUE"   SILVER_VALUE="$LAKE_SILVER_PREFIX" GOLD_VALUE="$LAKE_GOLD_PREFIX" GUARDRAIL_ID_VALUE="$GUARDRAIL_ID"   GUARDRAIL_VERSION_VALUE="$GUARDRAIL_VERSION" SECRET_ID_VALUE="$TAVILY_SECRET_SELECTION" python3 - <<'ENVIRONMENT'
import json, os
result = {"ETL_BACKEND": os.environ["ETL_VALUE"], "DQ_BACKEND": os.environ["DQ_VALUE"],
          "SAGEMAKER_EXECUTION_ROLE_ARN": os.environ["ROLE_VALUE"],
          "AMLAS_AWS_REGION": os.environ["REGION_VALUE"], "AMLAS_AWS_ACCOUNT_ID": os.environ["ACCOUNT_VALUE"],
          "AMLAS_SESSION_BUCKET": os.environ["SESSION_VALUE"],
          "AMLAS_DEPLOYMENT_PREFIX": os.environ["PREFIX_VALUE"], "AMLAS_SSM_PREFIX": os.environ["SSM_PREFIX_VALUE"],
          "AMLAS_JOBS_TABLE": os.environ["JOBS_VALUE"], "AMLAS_LAKE_BUCKET": os.environ["LAKE_VALUE"],
          "AMLAS_MODEL_PACKAGE_GROUP": os.environ["MODEL_GROUP_VALUE"], "AMLAS_FEATURE_STORE_DB": os.environ["FEATURE_DB_VALUE"],
          "AMLAS_LAKE_SILVER_PREFIX": os.environ["SILVER_VALUE"], "AMLAS_LAKE_GOLD_PREFIX": os.environ["GOLD_VALUE"],
          "AMLAS_GUARDRAIL_ID": os.environ["GUARDRAIL_ID_VALUE"], "AMLAS_GUARDRAIL_VERSION": os.environ["GUARDRAIL_VERSION_VALUE"],
          "AMLAS_TAVILY_SECRET_ID": os.environ["SECRET_ID_VALUE"]}
if os.environ["DEPLOYMENT_ID_VALUE"]:
    result["AMLAS_DEPLOYMENT_ID"] = os.environ["DEPLOYMENT_ID_VALUE"]
for key in ("AMLAS_CHAT_MODEL", "AMLAS_MODEL_PLANNER", "AMLAS_MODEL_REVIEWER", "AMLAS_MODEL_SPECIALIST",
            "AMLAS_MODEL_BOUNDED", "AMLAS_MODEL_FINETUNE", "AMLAS_HARNESS_LEADS", "AMLAS_TAVILY_KEYLESS"):
    if key in os.environ:
        result[key] = os.environ[key]
print(json.dumps(result))
ENVIRONMENT
)" || exit $?
if python3 "$(dirname "${BASH_SOURCE[0]}")/agentcore_runtime_control.py" --region "$REGION" \
  --name "$RUNTIME_NAME" --image "$IMAGE_URI" --role "$ROLE_ARN" --network-mode "$NETWORK_MODE" \
  --protocol "$SERVER_PROTOCOL" --environment-json "$RUNTIME_ENV_JSON" --tags-json "$RUNTIME_TAGS_JSON" ; then
  printf 'RESOLVED_SESSION_BUCKET=%s\n' "$SESSION_BUCKET"
else
  exit $?
fi
