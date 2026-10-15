#!/usr/bin/env bash
# =============================================================================
# scripts/lib/deploy_common.sh
# -----------------------------------------------------------------------------
# Shared helper library sourced by BOTH scripts/setup.sh (one-click deploy) and
# scripts/teardown.sh (one-click destroy). Centralizes the pieces that must be
# IDENTICAL across both entrypoints so they can never drift:
#
#   * repo-root resolution (location-independent)
#   * .env discovery + create-from-template + sourcing
#   * config resolution with the CONFIG-FIRST / PROMPT-SECOND model
#   * AWS/CDK environment export (the exact vars the proven manual run used)
#   * the ACCOUNT SAFETY GATE (sts get-caller-identity MUST equal the configured
#     account before ANY mutating call — non-negotiable, prevents F2 wrong-account)
#   * the canonical 9-stack deploy list + its reverse (destroy) order
#   * plain-text logging helpers (no ANSI, so tee'd log files stay clean)
#
# Compatibility: written for bash 3.2 (the macOS system bash) — NO associative
# arrays, NO ${var,,} lowercasing, NO mapfile. Portable to Linux bash 4/5 too.
#
# This file is a LIBRARY: it defines functions + constants and is meant to be
# `source`d. It intentionally does NOT set -euo pipefail (the sourcing entrypoint
# owns that) and does NOT execute any side effects on its own.
# =============================================================================

# -----------------------------------------------------------------------------
# CONSTANTS — defaults used by the public setup and teardown entry points.
# Every value is overridable via .env or an interactive prompt.
# -----------------------------------------------------------------------------

# Empty means the standard AWS credential chain. Operators may select any
# configured AWS CLI profile through AWS_PROFILE or the setup prompt.
DEFAULT_AWS_PROFILE=""

# Default target account — INTENTIONALLY EMPTY (SEC-A01 repo-hygiene: no private
# 12-digit account id is hardcoded on any shipped/operational surface). The real
# account id is supplied entirely by the operator's environment via
# AMLAS_AWS_ACCOUNT_ID (root .env, gitignored) or the interactive prompt. setup.sh
# fails loud (F4) if it resolves empty, and the account_safety_gate then asserts the
# live STS identity EXACTLY equals the resolved value before any mutation — so the
# F2 wrong-account guard is fully preserved without shipping the literal id.
DEFAULT_AWS_ACCOUNT_ID=""

# Deployments require a selected region; there is no implicit deployment target.
DEFAULT_AWS_REGION=""

# Secrets Manager secret ID that SecretsStack IMPORTS (does not create). Must
# exist before/at deploy or runtime GetSecretValue fails (H7 / F11 / FIX-05).
DEFAULT_TAVILY_SECRET_ID="amlas/tavily-api-key"

# The documented DUMMY value used when no real Tavily key is supplied. A real
# tvly-... key is set out-of-band later via put-secret-value.
TAVILY_PLACEHOLDER_VALUE="tvly-xxxxxxxxxxxxxxxx"

# --- Full agent runtime (AgentCore) — now part of the default setup.sh run -----
# The managed AgentCore runtime is created OUT-OF-BAND by the AWS CLI path
# (scripts/lib/deploy_agentcore_runtime.sh), NOT by a CDK stack: RuntimeStack is intentionally
# excluded from the 9-stack set because its fixed names collide with the already-
# deployed AsyncTrainingStack (amlas-sagemaker-exec role, amlas-training-jobs DDB,
# callback lambda). setup.sh builds and pushes
# the arm64 runtime image to the `amlas-runtime` ECR repo (created by
# AgentCoreRuntimeStack, which IS in the 9-stack set), then invokes that script and
# publishes the resulting runtime ARN to SSM so the TUI/CLI resolve it.
DEFAULT_AGENTCORE_RUNTIME_NAME="amlas"
DEFAULT_AGENTCORE_IMAGE_TAG="v1"
# SSM parameter the TUI/CLI/config resolve the runtime ARN from when .env is blank.
AMLAS_AGENTCORE_SSM_PARAM="/amlas/agentcore-runtime-arn"
# SSM parameter the CLI/runtime resolve the session (staging) bucket from when .env
# is blank (F2). setup.sh STEP 10.5d publishes it alongside the runtime ARN so a
# clean run leaves the account fully wired; mirrors deploy_agentcore_runtime.sh's resolution
# of the AsyncTrainingStack `amlas-async-session-bucket` export.
AMLAS_SESSION_BUCKET_SSM_PARAM="/amlas/session-bucket"

# The canonical 9-stack deploy set, in dependency order (space-separated
# so it works on bash 3.2). NEVER `cdk deploy --all` — RuntimeStack collides with
# AsyncTrainingStack and BackendIamStack is synth-only. HarnessStack reads no
# other stack's outputs, so it
# goes last on deploy and FIRST on destroy.
#
# HarnessStack destroy is SLOW and ASYNCHRONOUS: deleting a harness takes >5 minutes and
# cascades to the managed Memory it created, and a harness in DELETING state is neither
# updatable nor deletable (T-HARN-12). Expect the teardown to wait on this stack.
AMLAS_STACKS_DEPLOY_ORDER="AgentCoreRuntimeStack LakeStack ModelRegistryStack SecretsStack AsyncTrainingStack NetworkStack DetectionsStack GuardrailStack HarnessStack"

# The same 9 stacks in reverse order, used by teardown so dependent/soft-ordered
# resources (Lake, AgentCoreRuntime ECR) are destroyed last.
AMLAS_STACKS_DESTROY_ORDER="HarnessStack GuardrailStack DetectionsStack NetworkStack AsyncTrainingStack SecretsStack ModelRegistryStack LakeStack AgentCoreRuntimeStack"

# The CDK app entry (matches infra/cdk.json). Used with --app for every cdk call.
AMLAS_CDK_APP="python -m infra.app"

# -----------------------------------------------------------------------------
# LOGGING HELPERS — plain text with prefixes (no ANSI codes → clean tee'd logs).
# All log lines go to stdout; the entrypoint tees stdout+stderr to a log file.
# -----------------------------------------------------------------------------
log_info()  { printf '%s [INFO]  %s\n'  "$(date -u +%FT%TZ)" "$*"; }
log_warn()  { printf '%s [WARN]  %s\n'  "$(date -u +%FT%TZ)" "$*"; }
log_error() { printf '%s [ERROR] %s\n'  "$(date -u +%FT%TZ)" "$*" >&2; }
log_step()  { printf '\n%s ===== %s =====\n' "$(date -u +%FT%TZ)" "$*"; }

# die MESSAGE — log an error and exit non-zero (used for fatal precondition fails).
die() { log_error "$*"; exit 1; }

# -----------------------------------------------------------------------------
# resolve_repo_root SCRIPT_SOURCE
#   Echoes the absolute repo root = parent of the scripts/ dir that contains the
#   calling entrypoint. Location-independent: safe to run from anywhere. This is
#   the durable fix for H2/F3 (the old scripts hardcoded a wrong absolute cd).
#   Arg: the entrypoint's "${BASH_SOURCE[0]}".
# -----------------------------------------------------------------------------
resolve_repo_root() {
  local script_source="$1"
  local script_dir
  # scripts/ dir = dir of the entrypoint; repo root = its parent.
  script_dir="$(cd "$(dirname "$script_source")" && pwd)"
  # If sourced from scripts/lib/, callers pass their own path; entrypoints live
  # directly in scripts/, so the repo root is one level up from scripts/.
  ( cd "$script_dir/.." && pwd )
}

# -----------------------------------------------------------------------------
# ensure_env_file REPO_ROOT
#   Guarantees a .env exists at the repo root. If missing, creates it from
#   .env.example (the documented `cp .env.example .env` step). Idempotent.
# -----------------------------------------------------------------------------
ensure_env_file() {
  local repo_root="$1"
  if [ -f "$repo_root/.env" ]; then
    log_info ".env found at repo root — will load it."
  elif [ -f "$repo_root/.env.example" ]; then
    cp "$repo_root/.env.example" "$repo_root/.env"
    log_warn ".env was missing — created it from .env.example. Review before real use."
  else
    die "Neither .env nor .env.example exists at $repo_root — cannot continue."
  fi
}

# -----------------------------------------------------------------------------
# load_env_file REPO_ROOT
#   Loads the repo-root .env into the environment WITHOUT bash-sourcing it.
#
#   WHY NOT `. .env`? A .env is dotenv-format, NOT a shell script. Its values
#   legitimately contain shell metacharacters — e.g. AMLAS_LAKE_BUCKET=<LAKE_BUCKET>
#   or s3://<LAKE_BUCKET>/silver/ — and `. .env` makes bash interpret `<`/`>` as
#   redirection ("syntax error near unexpected token `newline'"). The proven
#   manual deploy never bash-sourced .env (Python's dotenv read it). So we parse
#   it safely line-by-line and export each KEY=VALUE literally, so no value is
#   ever re-interpreted by the shell.
#
#   Handles: blank lines, full-line `#` comments, an optional `export ` prefix,
#   surrounding single/double quotes, and trailing ` #` inline comments on
#   UNquoted values (matching python-dotenv semantics). Only well-formed
#   identifier keys are exported.
# -----------------------------------------------------------------------------
valid_env_key() {
  case "$1" in
    ""|[0-9]*|*[!A-Za-z0-9_]*) return 1 ;;
    *) return 0 ;;
  esac
}

decode_single_quoted_env_literal() {
  local text="$1" result="" char next index=0 length
  length=${#text}
  while [ "$index" -lt "$length" ]; do
    char="${text:$index:1}"
    next="${text:$((index + 1)):1}"
    if [ "$char" = '\' ] && { [ "$next" = '\' ] || [ "$next" = "'" ]; }; then
      result="$result$next"
      index=$((index + 2))
    else
      result="$result$char"
      index=$((index + 1))
    fi
  done
  DECODED_ENV_LITERAL="$result"
}

load_env_file() {
  local repo_root="$1"
  local env_file="$repo_root/.env"
  [ -f "$env_file" ] || return 0
  local line key val
  while IFS= read -r line || [ -n "$line" ]; do
    # Strip leading whitespace.
    line="${line#"${line%%[![:space:]]*}"}"
    # Skip blank lines and full-line comments.
    [ -z "$line" ] && continue
    case "$line" in \#*) continue ;; esac
    # Must be KEY=VALUE.
    case "$line" in *=*) ;; *) continue ;; esac
    key="${line%%=*}"
    val="${line#*=}"
    # Drop an optional leading "export ".
    key="${key#export }"
    # Trim any trailing whitespace on the key (bash 3.2 safe).
    key="${key%"${key##*[![:space:]]}"}"
    # Only accept valid shell identifiers as keys.
    valid_env_key "$key" || continue
    # Value handling: strip surrounding quotes if present; otherwise strip a
    # trailing " #inline comment" and any trailing whitespace.
    case "$val" in
      \"*\") val="${val#\"}"; val="${val%\"}" ;;
      \'*\') val="${val#\'}"; val="${val%\'}"; decode_single_quoted_env_literal "$val"; val="$DECODED_ENV_LITERAL" ;;
      *)
        case "$val" in *" #"*) val="${val%% #*}" ;; esac
        val="${val%"${val##*[![:space:]]}"}"
        ;;
    esac
    # BUGFIX (net-new-user): do NOT clobber a value the operator already exported in the
    # shell, and never load an unresolved ``<PLACEHOLDER>`` from a freshly-created .env
    # (.env.example ships ``AMLAS_AWS_ACCOUNT_ID=<AWS_ACCOUNT_ID>`` etc.). Previously this
    # blindly overwrote the shell env with the placeholder, so a user who exported
    # AMLAS_AWS_ACCOUNT_ID still got ``<AWS_ACCOUNT_ID>`` and the account-safety gate failed.
    # Precedence now: shell export > real .env value > (placeholder ignored).
    case "$val" in
      "<"*">") continue ;;   # unresolved placeholder → treat as unset
    esac
    # If the shell already has a non-empty value for this key, the shell wins.
    if [ -n "${!key:-}" ]; then
      continue
    fi
    # Export the literal value (no shell re-interpretation of its contents).
    export "$key=$val"
  done < "$env_file"
  log_info "Loaded configuration from .env (safe parser)"
}

# -----------------------------------------------------------------------------
# persist_env_value ENV_FILE KEY VALUE
#   Atomically upserts one resolved, non-secret setting in the repo-root .env.
#
#   setup.sh runs in a child process, so exporting AWS_PROFILE there does not
#   configure a later TUI/CLI process launched from the user's shell. Persisting
#   the selected profile/region after the account safety gate validates them keeps
#   local Bedrock inference configured even when a later CDK or AgentCore phase fails.
#
#   The replacement preserves unrelated lines and collapses duplicate definitions
#   of KEY to one canonical KEY=VALUE line. The temporary file is mode 0600 because
#   the same .env may contain secrets managed by the operator.
# -----------------------------------------------------------------------------
persist_env_value() {
  local env_file="$1"
  local key="$2"
  local value="$3"
  local tmp_file serialized

  valid_env_key "$key" || die "Refusing to persist an invalid .env key."
  [ -f "$env_file" ] || die "Cannot persist $key: .env file not found at $env_file."

  case "$value" in *$'\n'*|*$'\r'*) die "Refusing to persist a selector containing a line break." ;; esac
  # Single-quoted dotenv literals preserve hashes/spaces and escape only quote/backslash.
  serialized="${value//\\/\\\\}"
  serialized="${serialized//\'/\\\'}"
  if [[ "$value" =~ ^[A-Za-z0-9_./:@+=,-]*$ ]]; then
    serialized="$value"
  else
    serialized="'$serialized'"
  fi
  tmp_file="$(mktemp "${env_file}.tmp.XXXXXX")" || die "Could not create a temporary .env file."
  if ! AMLAS_PERSIST_LITERAL="$serialized" awk -v key="$key" '
    {
      candidate = $0
      sub(/^[[:space:]]*/, "", candidate)
      sub(/^export[[:space:]]+/, "", candidate)
      if (candidate ~ ("^" key "[[:space:]]*=")) {
        if (!written) {
          print key "=" ENVIRON["AMLAS_PERSIST_LITERAL"]
          written = 1
        }
        next
      }
      print
    }
    END {
      if (!written) {
        print key "=" ENVIRON["AMLAS_PERSIST_LITERAL"]
      }
    }
  ' "$env_file" > "$tmp_file"; then
    rm -f "$tmp_file"
    die "Could not persist $key to $env_file."
  fi

  chmod 600 "$tmp_file"
  mv "$tmp_file" "$env_file"
  log_info "Persisted $key to the repo .env for future TUI/CLI runs."
}

# -----------------------------------------------------------------------------
# prompt_with_default VAR_NAME "Prompt text" "default_value"
#   CONFIG-FIRST / PROMPT-SECOND resolver. Resolution order:
#     1. If the named variable is already set+non-empty (from .env/shell) -> keep.
#     2. Else if NON_INTERACTIVE=1 -> use the provided default (no prompt).
#     3. Else -> interactively prompt, showing [default]; empty input -> default.
#   The resolved value is assigned back to VAR_NAME and exported.
#   Relies on the global NON_INTERACTIVE (0/1) set by the entrypoint's arg parse.
# -----------------------------------------------------------------------------
prompt_with_default() {
  local var_name="$1"
  local prompt_text="$2"
  local default_value="$3"
  local current_value
  # Indirect read of the current value of $var_name (bash 3.2 safe).
  valid_env_key "$var_name" || die "Invalid configuration variable name."
  current_value="${!var_name:-}"

  if [ -n "$current_value" ]; then
    # Already configured (via .env or shell export) — respect it.
    export "$var_name=$current_value"
    return 0
  fi

  if [ "${NON_INTERACTIVE:-0}" = "1" ]; then
    export "$var_name=$default_value"
    return 0
  fi

  # Interactive prompt with default shown in brackets.
  local reply
  printf '%s [%s]: ' "$prompt_text" "$default_value" > /dev/tty
  # Read from the controlling terminal so this works even while stdout is tee'd.
  if [ -r /dev/tty ]; then
    read -r reply < /dev/tty
  else
    reply=""
  fi
  if [ -z "$reply" ]; then
    reply="$default_value"
  fi
  export "$var_name=$reply"
}

# -----------------------------------------------------------------------------
# resolve_aws_region — explicit AWS_REGION, application/legacy aliases,
# AWS_DEFAULT_REGION, selected profile metadata, then a required prompt.
# Reading profile config does not contact AWS or resolve credentials.
resolve_aws_region() {
  local profile_region=""
  if [ -z "${AWS_REGION:-}" ]; then
    AWS_REGION="${AMLAS_AWS_REGION:-${AGENTICML_AWS_REGION:-${AWS_DEFAULT_REGION:-}}}"
  fi
  if [ -z "${AWS_REGION:-}" ] && command -v aws >/dev/null 2>&1; then
    if [ -n "${AWS_PROFILE:-}" ]; then
      profile_region="$(aws configure get region --profile "${AWS_PROFILE:-<default-credential-chain>}" 2>/dev/null || true)"
    else
      profile_region="$(aws configure get region 2>/dev/null || true)"
    fi
  fi
  prompt_with_default AWS_REGION "AWS region (required)" "${profile_region:-$DEFAULT_AWS_REGION}"
  [ -n "$AWS_REGION" ] || die "An AWS region is required. Use --region, AWS_REGION, or the selected profile's region."
}


# Select one deployment namespace before deriving resource names. Legacy names
# remain unchanged by default. No cloud or dependency operation occurs here.
configure_deployment_namespace() {
  local prefix="${AMLAS_DEPLOYMENT_PREFIX:-amlas}" ssm_prefix
  [[ "$prefix" =~ ^[a-z][a-z0-9]*(-[a-z0-9]+)*$ ]] && [ "${#prefix}" -le 24 ] || die "Invalid deployment prefix: lowercase letters/digits with single hyphens, max 24 characters."
  ssm_prefix="${AMLAS_SSM_PREFIX:-/$prefix}"
  [[ "$ssm_prefix" =~ ^/[A-Za-z0-9_-]+(/[A-Za-z0-9_-]+)*$ ]] || die "Invalid SSM namespace."
  if [ "$prefix" != "amlas" ] && [ "$ssm_prefix" = "/amlas" ]; then
    die "The selected deployment prefix cannot publish to the legacy /amlas SSM namespace."
  fi
  export AMLAS_DEPLOYMENT_PREFIX="$prefix" AMLAS_SSM_PREFIX="$ssm_prefix"
  DEFAULT_TAVILY_SECRET_ID="$prefix/tavily-api-key"
  DEFAULT_AGENTCORE_RUNTIME_NAME="${prefix//-/_}"
  AMLAS_AGENTCORE_SSM_PARAM="$ssm_prefix/agentcore-runtime-arn"
  AMLAS_SESSION_BUCKET_SSM_PARAM="$ssm_prefix/session-bucket"
  if [ "$prefix" != "amlas" ]; then
    case "${TAVILY_SECRET_ID:-}" in ""|amlas/tavily-api-key) export TAVILY_SECRET_ID="$DEFAULT_TAVILY_SECRET_ID" ;; esac
    case "${AMLAS_MODEL_PACKAGE_GROUP:-}" in ""|amlas-models) export AMLAS_MODEL_PACKAGE_GROUP="$prefix-models" ;; esac
    case "${AMLAS_FEATURE_STORE_DB:-}" in ""|amlas_features) export AMLAS_FEATURE_STORE_DB="${prefix//-/_}_features" ;; esac
  fi
  export AMLAS_JOBS_TABLE="${AMLAS_JOBS_TABLE:-$prefix-training-jobs}"
}

deployment_stack_name() {
  local logical_id="$1" prefix="${2:-${AMLAS_DEPLOYMENT_PREFIX:-amlas}}"
  if [ "$prefix" = "amlas" ]; then printf '%s\n' "$logical_id"; else printf '%s-%s\n' "$prefix" "$logical_id"; fi
}

# filter_harness_stacks — omit only the explicitly disabled optional feature.
filter_harness_stacks() {
  local name deploy="" destroy=""
  for name in $AMLAS_STACKS_DEPLOY_ORDER; do
    [ "$name" = "HarnessStack" ] || deploy="${deploy:+$deploy }$name"
  done
  for name in $AMLAS_STACKS_DESTROY_ORDER; do
    [ "$name" = "HarnessStack" ] || destroy="${destroy:+$destroy }$name"
  done
  AMLAS_STACKS_DEPLOY_ORDER="$deploy"
  AMLAS_STACKS_DESTROY_ORDER="$destroy"
}

# _metadata_read — read-only AWS call, preserving the distinction between
# a missing resource, denied permission, and an unverifiable request.
_metadata_read() {
  local context="$1" advice="$2" err output rc
  shift 2
  err="$(mktemp)" || die "Could not prepare metadata diagnostics."
  output="$("$@" 2>"$err")"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    case "$(cat "$err")" in
      *AccessDenied*|*Unauthorized*) log_error "$context metadata permission denied; access was not verified." ;;
      *ResourceNotFound*|*TypeNotFound*) log_error "$context was not found in the selected region/account. $advice" ;;
      *) log_error "$context could not be verified; check credentials, connectivity and service availability. $advice" ;;
    esac
    cat "$err" >&2
    rm -f "$err"
    return "$rc"
  fi
  rm -f "$err"
  printf '%s' "$output"
}

# resolve_model_profiles — retain the sample's existing runtime IAM contract.
# Overrides outside this allowlist need an explicit IAM/source review.
resolve_model_profiles() {
  uv run --frozen python - <<'MODELS'
import os
import sys
from amlas.config.chat_model import chat_model_id
from amlas.config.runtime import BEDROCK_INVOKE_PROFILES, load_config
config = load_config()
profiles = {config.models.planner, config.models.reviewer, config.models.specialist,
            config.models.bounded, config.models.finetune,
            chat_model_id()}
unsupported = profiles - set(BEDROCK_INVOKE_PROFILES)
if unsupported:
    print("Configured model profiles are outside the runtime IAM allowlist: " + ", ".join(sorted(unsupported)), file=sys.stderr)
    print("Choose an existing BEDROCK_INVOKE_PROFILES entry or review the runtime IAM allowlist before deployment.", file=sys.stderr)
    sys.exit(2)
print(",".join(sorted(profiles)))
MODELS
}

# Normalize model overrides through the existing one-version compatibility seam.
normalize_model_env() {
  local model_lines model_key model_value
  model_lines="$(uv run --frozen python - <<'MODELS'
from amlas.config._env_compat import env_with_compat
keys = ("AMLAS_CHAT_MODEL", "AMLAS_MODEL_PLANNER", "AMLAS_MODEL_REVIEWER",
        "AMLAS_MODEL_SPECIALIST", "AMLAS_MODEL_BOUNDED", "AMLAS_MODEL_FINETUNE")
for key in keys:
    value = env_with_compat(key)
    if value is not None:
        if "\n" in value or "\r" in value:
            raise ValueError("Model identifiers cannot contain line breaks.")
        print(key + "=" + value)
MODELS
)" || return $?
  while IFS='=' read -r model_key model_value; do
    [ -n "$model_key" ] || continue
    valid_env_key "$model_key" || die "Invalid model configuration variable."
    export "$model_key=$model_value"
  done <<< "$model_lines"
}

# Input skeletons are local API-shape checks; they do not make AWS requests.
require_runtime_cli_capabilities() {
  local update_shape endpoint_shape
  update_shape="$(aws bedrock-agentcore-control update-agent-runtime --generate-cli-skeleton input --region "$AWS_REGION" 2>/dev/null)" || die "Upgrade the AWS CLI: selected runtime requires MMDSv2 and platform-version API fields."
  endpoint_shape="$(aws bedrock-agentcore-control update-agent-runtime-endpoint --generate-cli-skeleton input --region "$AWS_REGION" 2>/dev/null)" || die "Upgrade the AWS CLI: selected runtime requires version-aware DEFAULT endpoint updates."
  if ! UPDATE_SHAPE="$update_shape" ENDPOINT_SHAPE="$endpoint_shape" python3 - <<'SHAPES'
import json, os, sys
try:
    update = json.loads(os.environ["UPDATE_SHAPE"])
    endpoint = json.loads(os.environ["ENDPOINT_SHAPE"])
    supported = ("platformVersion" in update and
                 "requireMMDSV2" in update.get("metadataConfiguration", {}) and
                 "agentRuntimeVersion" in endpoint)
except (ValueError, TypeError):
    supported = False
sys.exit(0 if supported else 1)
SHAPES
  then
    die "Upgrade the AWS CLI: its API model lacks required MMDSv2, platform-version or endpoint-version fields."
  fi
}

# CloudTrail has five trails per Region, including shadow trails; not adjustable.
# https://docs.aws.amazon.com/awscloudtrail/latest/userguide/WhatIsCloudTrail-Limits.html
# ListTrails is auto-paginated by the native CLI. Presence is not ownership.
preflight_cloudtrail_capacity() {
  local metadata result count existing
  [ -n "${AWS_REGION:-}" ] && [ -n "${AMLAS_AWS_ACCOUNT_ID:-}" ] && [ -n "${AMLAS_DEPLOYMENT_PREFIX:-}" ] || die "CloudTrail capacity requires resolved account, Region and deployment prefix."
  metadata="$(_metadata_read "CloudTrail trail capacity" "Verify cloudtrail:ListTrails permission in the selected profile and Region." \
    aws cloudtrail list-trails --region "$AWS_REGION" --output json)" || {
    log_error "Verify cloudtrail:ListTrails access before retrying deployment; CloudTrail capacity was not verified."
    return 1
  }
  if ! result="$(CLOUDTRAIL_METADATA="$metadata" python3 - <<'TRAILS'
import json
import os
import re
import sys

try:
    data = json.loads(os.environ["CLOUDTRAIL_METADATA"])
    if not isinstance(data, dict) or not isinstance(data.get("Trails"), list):
        raise ValueError("Unverifiable trail list")
    if data.get("NextToken") not in (None, ""):
        raise ValueError("Incomplete pagination")
    account = os.environ["AMLAS_AWS_ACCOUNT_ID"]
    region = os.environ["AWS_REGION"]
    selected = os.environ["AMLAS_DEPLOYMENT_PREFIX"] + "-security-trail"
    seen = set()
    existing = False
    for trail in data["Trails"]:
        if not isinstance(trail, dict) or any(
            not isinstance(trail.get(key), str) or not trail[key]
            for key in ("Name", "HomeRegion", "TrailARN")
        ):
            raise ValueError("Unverifiable trail identity")
        arn = trail["TrailARN"]
        match = re.fullmatch(r"arn:([a-z0-9-]+):cloudtrail:([a-z0-9-]+):([0-9]{12}):trail/(.+)", arn)
        if not match or match[2] != trail["HomeRegion"] or match[4] != trail["Name"] or arn in seen:
            raise ValueError("Inconsistent or duplicate trail identity")
        seen.add(arn)
        existing = existing or (match[3] == account and match[2] == region and match[4] == selected)
    print(str(len(data["Trails"])) + " " + str(int(existing)))
except (ValueError, TypeError, KeyError):
    sys.exit(1)
TRAILS
)"; then
    die "CloudTrail trail metadata could not be verified; verify ListTrails access and complete valid metadata before deployment."
  fi
  read -r count existing <<< "$result"
  if [ "$count" -ge 5 ] && [ "$existing" != 1 ]; then
    die "CloudTrail trail quota is full ($count/5) in the selected Region. This limit cannot be increased. Choose another Region or have an operator review and remove only an explicitly approved obsolete trail before retrying. Preserve unrelated trails."
  fi
  log_info "CloudTrail trail capacity verified: $count/5. Selected trail presence does not establish ownership; the deployment manifest remains required."
}

# preflight_deployment_services MODEL_PROFILES_CSV — metadata checks only.
# Success does not prove InvokeModel permission, capacity, or a live deployment.
preflight_deployment_services() {
  local profiles="$1" model status provisioning old_ifs="$IFS"
  if [ "${SKIP_AGENT_RUNTIME:-0}" != "1" ]; then require_runtime_cli_capabilities || return $?; fi
  IFS=,
  for model in $profiles; do
    IFS="$old_ifs"
    status="$(_metadata_read "Configured model profile $model" "Choose an available compatible profile through AMLAS_MODEL_* / AMLAS_CHAT_MODEL, or another region."       aws bedrock get-inference-profile --inference-profile-identifier "$model" --region "$AWS_REGION" --query status --output text)" || return $?
    [ "$status" = "ACTIVE" ] || die "Configured model profile $model is not ready in the selected region."
    IFS=,
  done
  IFS="$old_ifs"
  if [ "${SKIP_HARNESS:-0}" != "1" ]; then
    provisioning="$(_metadata_read "Harness CloudFormation resource type" "Use --skip-harness to omit that feature, or choose a region where the type is available."       aws cloudformation describe-type --type RESOURCE --type-name AWS::BedrockAgentCore::Harness --region "$AWS_REGION" --query ProvisioningType --output text)" || return $?
    case "$provisioning" in FULLY_MUTABLE|IMMUTABLE) ;; *) die "Harness resource type is not provisionable; use --skip-harness or choose another region." ;; esac
  fi
  if [ "${SKIP_AGENT_RUNTIME:-0}" != "1" ]; then
    _metadata_read "AgentCore runtime control plane" "Use --skip-agent-runtime to omit the managed runtime, or verify access in the selected region."       aws bedrock-agentcore-control list-agent-runtimes --max-results 1 --region "$AWS_REGION" >/dev/null || return $?
  fi
  log_info "Metadata preflight passed. Runtime invocation permission/capacity still requires live verification."
}

# -----------------------------------------------------------------------------
# prompt_secret_with_default VAR_NAME "Prompt text" "default_value"
#   Like prompt_with_default but silences echo while typing (for the Tavily key).
#   Empty input -> default. Respects NON_INTERACTIVE and pre-set values.
# -----------------------------------------------------------------------------
prompt_secret_with_default() {
  local var_name="$1"
  local prompt_text="$2"
  local default_value="$3"
  local current_value
  valid_env_key "$var_name" || die "Invalid configuration variable name."
  current_value="${!var_name:-}"

  if [ -n "$current_value" ]; then
    export "$var_name=$current_value"
    return 0
  fi
  if [ "${NON_INTERACTIVE:-0}" = "1" ]; then
    export "$var_name=$default_value"
    return 0
  fi

  local reply
  printf '%s [optional; leave blank for no-key research]: ' "$prompt_text" > /dev/tty
  if [ -r /dev/tty ]; then
    read -r -s reply < /dev/tty
    printf '\n' > /dev/tty
  else
    reply=""
  fi
  if [ -z "$reply" ]; then
    reply="$default_value"
  fi
  export "$var_name=$reply"
}

# -----------------------------------------------------------------------------
# export_aws_env PROFILE ACCOUNT REGION
#   Exports the EXACT environment the proven manual run used (deploy-env.sh):
#   profile, region (three var names CDK/boto3 look at), CDK account/region, and
#   the JSII node-version warning silence (F12: node v25 is "untested by JSII").
# -----------------------------------------------------------------------------
export_aws_env() {
  local profile="$1"
  local account="$2"
  local region="$3"
  if [ -n "$profile" ]; then
    export AWS_PROFILE="$profile"
    export AWS_DEFAULT_PROFILE="$profile"
    export AMLAS_AWS_PROFILE="$profile"
  else
    unset AWS_PROFILE AWS_DEFAULT_PROFILE AMLAS_AWS_PROFILE AGENTICML_AWS_PROFILE
  fi
  export AWS_REGION="$region"
  export AMLAS_AWS_REGION="$region"
  export AWS_DEFAULT_REGION="$region"
  export CDK_DEFAULT_REGION="$region"
  export CDK_DEFAULT_ACCOUNT="$account"
  export JSII_SILENCE_WARNING_UNTESTED_NODE_VERSION=1
  log_info "AWS env exported: profile=${profile:-<default-credential-chain>} account=$account region=$region"
}

# -----------------------------------------------------------------------------
# account_safety_gate EXPECTED_ACCOUNT
#   THE non-negotiable guard. Calls `aws sts get-caller-identity` and asserts the
#   live account EXACTLY equals EXPECTED_ACCOUNT. Aborts loudly on mismatch or on
#   any failure to resolve identity. MUST run before any mutating aws/cdk call.
#   (Prevents F2 — deploying/destroying the wrong account.)
# -----------------------------------------------------------------------------
# Explicit CLI profile selection has the same provider semantics as Session(profile_name=...).
aws() {
  if [ -n "${AWS_PROFILE:-}" ]; then
    command aws "--profile=$AWS_PROFILE" "$@"
  else
    command aws "$@"
  fi
}

account_safety_gate() {
  local expected="$1"
  log_step "ACCOUNT SAFETY GATE"
  local actual
  actual="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
  if [ -z "$actual" ] || [ "$actual" = "None" ]; then
    log_error "Could not resolve AWS identity with profile '${AWS_PROFILE:-<unset>}'."
    log_error "Check credentials (e.g. 'aws sts get-caller-identity --profile ${AWS_PROFILE:-<default-credential-chain>}')."
    die "Safety gate FAILED — no identity."
  fi
  if [ "$actual" != "$expected" ]; then
    log_error "ACCOUNT MISMATCH — refusing to proceed."
    log_error "  expected account: $expected"
    log_error "  actual account:   $actual  (profile=${AWS_PROFILE:-<default-credential-chain>})"
    die "Safety gate FAILED — wrong account. NOTHING was mutated."
  fi
  log_info "Safety gate PASSED — live account $actual == configured $expected (profile=${AWS_PROFILE:-<default-credential-chain>})."
}

# -----------------------------------------------------------------------------
# aws_cli_available  — returns 0 if the aws CLI is on PATH, else non-zero.
# -----------------------------------------------------------------------------
aws_cli_available() { type -P aws >/dev/null 2>&1; }

# Required before profile metadata or identity validation, even with --skip-deps.
require_aws_cli() {
  aws_cli_available || die "aws CLI not found on PATH. Install it (https://aws.amazon.com/cli/) and re-run."
}


# Create discovery parameters with atomic ownership. Updating an existing value
# requires its current UUID/prefix witness; retagging is never adoption.
publish_owned_parameter() {
  local name="$1" value="$2" count tags
  [ -n "${AMLAS_DEPLOYMENT_ID:-}" ] || die "A deployment ownership identifier is required for publication."
  count="$(aws ssm describe-parameters --parameter-filters "Key=Name,Option=Equals,Values=$name" --query 'length(Parameters)' --output text --region "$AWS_REGION")" || die "Parameter metadata could not be verified."
  case "$count" in
    0)
      aws ssm put-parameter --name "$name" --value "$value" --type String \
        --tags "Key=amlas:deployment-prefix,Value=$AMLAS_DEPLOYMENT_PREFIX" "Key=amlas:deployment-id,Value=$AMLAS_DEPLOYMENT_ID" \
        --region "$AWS_REGION" >/dev/null || die "Owned parameter creation failed."
      ;;
    1)
      tags="$(aws ssm list-tags-for-resource --resource-type Parameter --resource-id "$name" --output json --region "$AWS_REGION")" || die "Parameter ownership could not be verified."
      if ! printf '%s' "$tags" | python3 -c 'import json,os,sys; t={x["Key"]:x["Value"] for x in json.load(sys.stdin).get("TagList",[])}; sys.exit(0 if t.get("amlas:deployment-id")==os.environ.get("AMLAS_DEPLOYMENT_ID") and t.get("amlas:deployment-prefix")==os.environ.get("AMLAS_DEPLOYMENT_PREFIX") else 1)'; then
        die "Existing discovery parameter is not owned by this deployment."
      fi
      aws ssm put-parameter --name "$name" --value "$value" --type String --overwrite --region "$AWS_REGION" >/dev/null || die "Owned parameter update failed."
      ;;
    *) die "Parameter metadata returned an unexpected count." ;;
  esac
}

verify_deployed_stacks() {
  local logical physical status failed=0
  for logical in $AMLAS_STACKS_DEPLOY_ORDER CDKToolkit; do
    if [ "$logical" = "CDKToolkit" ]; then physical="$logical"; else physical="$(deployment_stack_name "$logical")"; fi
    status="$(aws cloudformation describe-stacks --stack-name "$physical" --region "$AWS_REGION" --query 'Stacks[0].StackStatus' --output text)" || status="NOT_FOUND"
    printf '    %-48s %s\n' "$physical" "$status"
    case "$status" in CREATE_COMPLETE|UPDATE_COMPLETE) ;; *) failed=1 ;; esac
  done
  [ "$failed" = "0" ] || die "Selected physical stacks are not all successfully deployed."
}


# Machine API keys stay in memory/stdin and the existing platform secret store.
# Never pass a key in argv, write it to dotenv, or emit the returned secret body.
write_tavily_secret() {
  local operation="$1" secret_id="$2" helper_path
  helper_path="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/write_research_secret.py"
  case "$operation" in create|put) ;; *) die "Unsupported secret operation." ;; esac
  if [ "$operation" = "create" ]; then
    AMLAS_SECRET_TARGET="$secret_id" python3 -c 'import json,os; print(json.dumps({"Name":os.environ["AMLAS_SECRET_TARGET"],"SecretString":os.environ["AMLAS_TAVILY_API_KEY"],"Tags":[{"Key":"amlas:deployment-prefix","Value":os.environ["AMLAS_DEPLOYMENT_PREFIX"]},{"Key":"amlas:deployment-id","Value":os.environ["AMLAS_DEPLOYMENT_ID"]}]}))' |
      AWS_PROFILE="${AWS_PROFILE:-}" AWS_REGION="$AWS_REGION" env -u AMLAS_TAVILY_API_KEY uv run --frozen --no-sync python "$helper_path" create >/dev/null || die "Secret creation failed; no key was logged."
  else
    AMLAS_SECRET_TARGET="$secret_id" python3 -c 'import json,os; print(json.dumps({"SecretId":os.environ["AMLAS_SECRET_TARGET"],"SecretString":os.environ["AMLAS_TAVILY_API_KEY"]}))' |
      AWS_PROFILE="${AWS_PROFILE:-}" AWS_REGION="$AWS_REGION" env -u AMLAS_TAVILY_API_KEY uv run --frozen --no-sync python "$helper_path" put >/dev/null || die "Secret update failed; no key was logged."
  fi
}


# Only the successful native compatibility preflight may claim immutable role
# reuse. Reject ambiguous/partial markers before ledger initialization.
collect_runtime_role_reuse_witness() {
  local line arn_count=0 id_count=0
  REUSED_RUNTIME_ROLE_ARN=""
  REUSED_RUNTIME_ROLE_ID=""
  while IFS= read -r line; do
    case "$line" in
      RESOLVED_REUSED_RUNTIME_ROLE_ARN=*) REUSED_RUNTIME_ROLE_ARN="${line#*=}"; arn_count=$((arn_count+1)) ;;
      RESOLVED_REUSED_RUNTIME_ROLE_ID=*) REUSED_RUNTIME_ROLE_ID="${line#*=}"; id_count=$((id_count+1)) ;;
    esac
  done <<< "$1"
  if [ "$arn_count" = "0" ] && [ "$id_count" = "0" ]; then return 0; fi
  [ "$arn_count" = "1" ] && [ "$id_count" = "1" ] && [ -n "$REUSED_RUNTIME_ROLE_ARN" ] && [ -n "$REUSED_RUNTIME_ROLE_ID" ] || die "Runtime role reuse witness is incomplete or ambiguous."
  case "$REUSED_RUNTIME_ROLE_ARN" in
    arn:*:iam::"$AMLAS_AWS_ACCOUNT_ID":role/*) ;;
    *) die "Runtime role reuse witness has an unexpected account or ARN." ;;
  esac
  [[ "$REUSED_RUNTIME_ROLE_ID" =~ ^[A-Z0-9]+$ ]] || die "Runtime role reuse witness has an invalid native role identity."
}

append_runtime_role_reuse_args() {
  if [ -n "${REUSED_RUNTIME_ROLE_ARN:-}" ] || [ -n "${REUSED_RUNTIME_ROLE_ID:-}" ]; then
    [ -n "${REUSED_RUNTIME_ROLE_ARN:-}" ] && [ -n "${REUSED_RUNTIME_ROLE_ID:-}" ] || die "Runtime role reuse witness is incomplete."
    MANIFEST_INIT_ARGS+=(--reused-role-arn "$REUSED_RUNTIME_ROLE_ARN" --reused-role-id "$REUSED_RUNTIME_ROLE_ID")
  fi
}
