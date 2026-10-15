#!/usr/bin/env bash
# =============================================================================
# scripts/setup.sh — ONE-CLICK SETUP for the AMLAS CDK stacks
# -----------------------------------------------------------------------------
# Encodes the PROVEN manual deploy sequence (docs/deploy/01-manual-walkthrough-run.md)
# into a single robust, idempotent, re-runnable entrypoint that a brand-new user
# can run from anywhere. It:
#
#   1. Resolves the repo root from its own location (no hardcoded paths — H2/F3).
#   2. Loads .env (creating it from .env.example if missing).
#   3. Resolves required config CONFIG-FIRST / PROMPT-SECOND (profile, account,
#      region, Tavily key) — .env value wins; else prompt; --yes uses defaults.
#   4. Requires AWS CLI; verifies/installs uv, cdk CLI (Node), and Python 3.11
#      (via uv). Docker is a SOFT/optional check (only needed for Phase-6 image).
#   5. ACCOUNT SAFETY GATE: asserts live STS account == configured account BEFORE
#      any mutating call. Aborts loudly on mismatch (F2).
#   6. Runs the proven sequence: prepare dependencies -> ensure .env (KMS empty, F15) ->
#      create/update Tavily secret -> cdk bootstrap (skip if already present) ->
#      cdk synth -> cdk deploy the explicit 9-stack set in the documented order.
#   7. Tees ALL output to output/deploy/logs/setup-<timestamp>.log and prints a
#      final summary (stacks + status, key outputs, log path).
#
# Idempotent: re-running against an already-deployed account no-ops the stacks
# (cdk deploy is naturally idempotent), skips bootstrap, and create-or-updates
# the secret.
#
# Usage:
#   scripts/setup.sh [--yes|--non-interactive] [--skip-deps] [--skip-bootstrap]
#                    [--synth-only] [--skip-agent-runtime] [--agent-runtime-only]
#                    [--help]
#
# Flags:
#   --deployment-prefix NAME Isolate resource names (default amlas; lowercase, max 24).
#   --ask-tavily-key         Ask for an optional hidden key, including with --yes.
#   --profile NAME           Select a local AWS CLI profile; overrides .env.
#   --region REGION          Select the target region; overrides .env/profile.
#   --account-id ACCOUNT     Required 12-digit target for the identity safety gate.
#   --runtime-name NAME      Select the runtime (letter, digits/underscores; max 48).
#   --runtime-role-name NAME Select its IAM execution role; overrides .env.
#   --preflight-only         Check identity and service/model metadata; no cloud writes.
#   --skip-harness           Omit HarnessStack and its metadata checks.
#
# Model choices: AMLAS_MODEL_* and AMLAS_CHAT_MODEL use the existing runtime IAM
# allowlist. Metadata checks do not prove live invocation permission or capacity.
#   --yes, --non-interactive  Skip all prompts; use .env values or built-in
#                             defaults. (Needed for the automated 3x validation.)
#   --skip-deps               Skip prereq install/verify (assume tools present).
#   --skip-bootstrap          Never run cdk bootstrap (assume already bootstrapped).
#   --synth-only              Stop after `cdk synth` (no deploy) — safe dry run.
#   --skip-agent-runtime      Deploy the 9 CDK stacks only; SKIP building/pushing the
#                             AgentCore runtime image + creating the managed runtime.
#                             (The local TUI/CLI work without it; the managed
#                             InvokeAgentRuntime endpoint won't be provisioned.)
#   --agent-runtime-only      SKIP the CDK deploy; ONLY build+push the runtime image,
#                             create-or-update the AgentCore runtime, and publish its
#                             ARN to SSM. Use to (re)deploy the agent on an account
#                             whose 9 stacks are already up.
#   --help, -h                Print this help and exit 0.
#
# FULL AGENT STACK (default): after the 9-stack CDK deploy, setup.sh also builds +
# pushes the arm64 runtime image to the `amlas-runtime` ECR repo, create-or-discovers
# the managed AgentCore runtime (scripts/lib/deploy_agentcore_runtime.sh), polls it to READY, and
# publishes its ARN to SSM (/amlas/agentcore-runtime-arn) so the TUI/CLI resolve it.
# This phase REQUIRES Docker + buildx; disable it with --skip-agent-runtime.
#
# Bash 3.2 compatible (macOS system bash). Requires: bash, aws, cdk, uv, node.
# =============================================================================
set -euo pipefail

# --- Resolve repo root + source the shared library (location-independent) -----
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
# shellcheck source=scripts/lib/deploy_common.sh
. "$SCRIPT_DIR/lib/deploy_common.sh"
# Definitions only; host changes run after the authenticated account gate.
. "$SCRIPT_DIR/lib/prepare_host.sh"
REPO_ROOT="$(resolve_repo_root "$SCRIPT_SOURCE")"
cd "$REPO_ROOT"

# --- Defaults for flags -------------------------------------------------------
NON_INTERACTIVE=0     # consumed by prompt_* helpers in the shared library
SKIP_DEPS=0
SKIP_BOOTSTRAP=0
SYNTH_ONLY=0
SKIP_AGENT_RUNTIME=0  # default: DO deploy the full agent runtime (build+push+create+SSM)
PREFLIGHT_ONLY=0
SKIP_HARNESS=0
ASK_TAVILY_KEY=0
CLI_DEPLOYMENT_PREFIX=""
CLI_RUNTIME_NAME=""
CLI_RUNTIME_ROLE=""
AGENT_RUNTIME_ONLY=0  # only the agent runtime phase; skip the 9-stack CDK deploy

# --- Argument parsing ---------------------------------------------------------
print_help() {
  # Print the usage block from the header comments (lines starting with '# ').
  awk 'NR == 1 {next} /^set -/ {exit} {sub(/^# ?/, ""); print}' "$SCRIPT_SOURCE"
}
while [ "$#" -gt 0 ]; do
  case "$1" in
    --yes|--non-interactive) NON_INTERACTIVE=1 ;;
    --ask-tavily-key) ASK_TAVILY_KEY=1 ;;
    --deployment-prefix)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { printf '%s\n' 'Missing deployment prefix.' >&2; exit 2; }
      [[ "$2" =~ ^[a-z][a-z0-9]*(-[a-z0-9]+)*$ ]] && [ "${#2}" -le 24 ] || { printf '%s\n' 'Invalid deployment prefix.' >&2; exit 2; }
      [ -z "$CLI_DEPLOYMENT_PREFIX" ] || [ "$CLI_DEPLOYMENT_PREFIX" = "$2" ] || { printf '%s\n' 'Conflicting deployment prefixes.' >&2; exit 2; }
      CLI_DEPLOYMENT_PREFIX="$2"; export AMLAS_DEPLOYMENT_PREFIX="$2" AMLAS_SSM_PREFIX="/$2"
      shift 2; continue ;;
    --runtime-name|--runtime-role-name)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { printf '%s\n' 'Missing runtime selector value.' >&2; exit 2; }
      case "$1" in
        --runtime-name)
          [[ "$2" =~ ^[A-Za-z][A-Za-z0-9_]{0,47}$ ]] || { printf '%s\n' 'Invalid runtime name: use a letter followed by letters, digits or underscores (max 48).' >&2; exit 2; }
          [ -z "$CLI_RUNTIME_NAME" ] || [ "$CLI_RUNTIME_NAME" = "$2" ] || { printf '%s\n' 'Conflicting runtime names.' >&2; exit 2; }
          CLI_RUNTIME_NAME="$2"; export AGENTCORE_RUNTIME_NAME="$2" ;;
        --runtime-role-name)
          [[ "$2" =~ ^[A-Za-z0-9_+=,.@-]{1,64}$ ]] || { printf '%s\n' 'Invalid runtime role name (IAM role-name characters, max 64).' >&2; exit 2; }
          [ -z "$CLI_RUNTIME_ROLE" ] || [ "$CLI_RUNTIME_ROLE" = "$2" ] || { printf '%s\n' 'Conflicting runtime role names.' >&2; exit 2; }
          CLI_RUNTIME_ROLE="$2"; export AGENTCORE_EXEC_ROLE_NAME="$2" ;;
      esac
      shift 2; continue ;;
    --profile|--region|--account-id)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "$1 requires a value" >&2; exit 2; }
      case "$1" in --profile) export AWS_PROFILE="$2" ;; --region) export AWS_REGION="$2" ;; --account-id) export AMLAS_AWS_ACCOUNT_ID="$2" ;; esac
      shift 2; continue ;;
    --preflight-only)        PREFLIGHT_ONLY=1 ;;
    --skip-harness)          SKIP_HARNESS=1; export AMLAS_SKIP_HARNESS_STACK=1 ;;
    --skip-deps)             SKIP_DEPS=1 ;;
    --skip-bootstrap)        SKIP_BOOTSTRAP=1 ;;
    --synth-only)            SYNTH_ONLY=1 ;;
    --skip-agent-runtime)    SKIP_AGENT_RUNTIME=1 ;;
    --agent-runtime-only)    AGENT_RUNTIME_ONLY=1 ;;
    --help|-h)               print_help; exit 0 ;;
    *) echo "Unknown argument: $1 (use --help)"; exit 2 ;;
  esac
  shift
done

if [ "$SYNTH_ONLY" = "1" ] && [ "$PREFLIGHT_ONLY" = "1" ]; then
  printf '%s\n' 'ERROR: --synth-only and --preflight-only are mutually exclusive.' >&2
  exit 2
fi
if [ "$AGENT_RUNTIME_ONLY" = "1" ] && { [ "$SYNTH_ONLY" = "1" ] || [ "$PREFLIGHT_ONLY" = "1" ]; }; then
  echo "--agent-runtime-only cannot be combined with --synth-only/--preflight-only" >&2
  exit 2
fi

# --synth-only is a non-mutating CDK dry run — it implies no agent runtime phase.
if [ "$SYNTH_ONLY" = "1" ]; then SKIP_AGENT_RUNTIME=1; fi

# --- Set up tee logging BEFORE any real work ---------------------------------
# All stdout+stderr go both to the terminal and to a timestamped log file so the
# whole run is auditable (global rule: tee long output to logs and read them).
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$REPO_ROOT/output/deploy/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup-$TS.log"
# Redirect the rest of the script through tee via a process substitution.
exec > >(tee -a "$LOG_FILE") 2>&1

log_step "AMLAS ONE-CLICK SETUP START ($TS)"
log_info "Repo root: $REPO_ROOT"
log_info "Log file:  $LOG_FILE"
log_info "Flags: non_interactive=$NON_INTERACTIVE skip_deps=$SKIP_DEPS skip_bootstrap=$SKIP_BOOTSTRAP synth_only=$SYNTH_ONLY skip_agent_runtime=$SKIP_AGENT_RUNTIME agent_runtime_only=$AGENT_RUNTIME_ONLY"

# =============================================================================
# STEP 1 — .env: ensure present, then load it (config source of truth)
# =============================================================================
log_step "STEP 1 — Configuration (.env)"
ensure_env_file "$REPO_ROOT"
load_env_file "$REPO_ROOT"
configure_deployment_namespace
case "${AMLAS_SKIP_HARNESS_STACK:-}" in 1|true|yes|on) SKIP_HARNESS=1 ;; esac
if [ "$SKIP_HARNESS" = "1" ]; then export AMLAS_SKIP_HARNESS_STACK=1; filter_harness_stacks; fi

# =============================================================================
# STEP 2 — Resolve required settings (CONFIG-FIRST, PROMPT-SECOND)
# For each: use the .env/shell value if set; else prompt (default in brackets);
# --yes uses the default. New keys AMLAS_AWS_PROFILE / AMLAS_TAVILY_API_KEY are
# read here in addition to the AWS-standard names.
# =============================================================================
log_step "STEP 2 — Resolve deploy settings"

# Profile: prefer AWS_PROFILE, then AMLAS_AWS_PROFILE, else prompt/default.
if [ -z "${AWS_PROFILE:-}" ]; then
  AWS_PROFILE="${AMLAS_AWS_PROFILE:-${AGENTICML_AWS_PROFILE:-${AWS_DEFAULT_PROFILE:-}}}"
fi
prompt_with_default AWS_PROFILE "AWS CLI profile (headless, maps to target account)" "$DEFAULT_AWS_PROFILE"

# Account id: prefer AMLAS_AWS_ACCOUNT_ID (the REQUIRED config key), else prompt/default.
prompt_with_default AMLAS_AWS_ACCOUNT_ID "Target AWS account id (safe-to-wipe test account)" "$DEFAULT_AWS_ACCOUNT_ID"

# The AWS CLI prerequisite must fail clearly before profile metadata/identity,
# while the account guard still runs before long dependency installs or cloud writes.
if [ "$SKIP_DEPS" = "0" ]; then ensure_aws_cli_prerequisite; fi
require_aws_cli

# Region: explicit settings, selected profile metadata, then a required prompt.
resolve_aws_region
AGENTCORE_RUNTIME_NAME="${AGENTCORE_RUNTIME_NAME:-$DEFAULT_AGENTCORE_RUNTIME_NAME}"
AGENTCORE_EXEC_ROLE_NAME="${AGENTCORE_EXEC_ROLE_NAME:-$AMLAS_DEPLOYMENT_PREFIX-runtime-exec-$AWS_REGION}"
[[ "$AGENTCORE_RUNTIME_NAME" =~ ^[A-Za-z][A-Za-z0-9_]{0,47}$ ]] || die "Invalid configured runtime name."
[[ "$AGENTCORE_EXEC_ROLE_NAME" =~ ^[A-Za-z0-9_+=,.@-]{1,64}$ ]] || die "Invalid configured runtime role name."
export AGENTCORE_RUNTIME_NAME AGENTCORE_EXEC_ROLE_NAME

# Tavily secret id (the id, never the key value) + the key value itself (optional).
prompt_with_default TAVILY_SECRET_ID "Tavily Secrets Manager secret ID" "$DEFAULT_TAVILY_SECRET_ID"
# AMLAS_TAVILY_API_KEY: the actual tvly-... value (optional). Blank -> placeholder.
if [ "$ASK_TAVILY_KEY" = "1" ]; then
  if ! ( : </dev/tty ) 2>/dev/null; then die "--ask-tavily-key needs an interactive terminal. Omit it for no-key research."; fi
  saved_non_interactive="$NON_INTERACTIVE"
  NON_INTERACTIVE=0
  prompt_secret_with_default AMLAS_TAVILY_API_KEY "Tavily API key (tvly-...; optional)" "$TAVILY_PLACEHOLDER_VALUE"
  NON_INTERACTIVE="$saved_non_interactive"
else
  prompt_secret_with_default AMLAS_TAVILY_API_KEY "Tavily API key (tvly-...; optional)" "$TAVILY_PLACEHOLDER_VALUE"
fi

log_info "Resolved: profile=${AWS_PROFILE:-} account=$AMLAS_AWS_ACCOUNT_ID region=$AWS_REGION secret_id=$TAVILY_SECRET_ID"
case "$AMLAS_AWS_ACCOUNT_ID" in ""|*[!0-9]*) die "A 12-digit target AWS account id is required; use --account-id or AMLAS_AWS_ACCOUNT_ID." ;; esac
[ "${#AMLAS_AWS_ACCOUNT_ID}" -eq 12 ] || die "The target AWS account id must contain 12 digits."
export_aws_env "${AWS_PROFILE:-}" "$AMLAS_AWS_ACCOUNT_ID" "$AWS_REGION"
account_safety_gate "$AMLAS_AWS_ACCOUNT_ID"

# =============================================================================
# STEP 3 — Prerequisite verification / installation (idempotent, mac+linux)
# =============================================================================
if [ "$SKIP_DEPS" = "1" ]; then
  log_step "STEP 3 — Prerequisites (SKIPPED via --skip-deps)"
else
  log_step "STEP 3 — Prerequisites"
  # Metadata-only and synthesis paths retain their Docker-free contract.
  # Complete host preparation is needed for the default runtime image phase.
  if [ "$PREFLIGHT_ONLY" = "0" ] && [ "$SYNTH_ONLY" = "0" ] && [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
    prepare_host_prerequisites
  fi

  # 3a. AWS CLI was required before identity validation; report its version here.
  log_info "aws CLI present: $(aws --version 2>&1 | head -1)"

  # 3b. uv — REQUIRED; auto-install via the official installer if missing.
  if command -v uv >/dev/null 2>&1; then
    log_info "uv present: $(uv --version 2>&1)"
  else
    log_warn "uv not found — installing via official installer (astral.sh)."
    # Security review: two-step install (download → execute) instead of
    # curl-pipe-sh so the script is inspectable on disk before it runs.
    UV_INSTALLER="$(mktemp -t uv-install.XXXXXX.sh)"
    curl -LsSf https://astral.sh/uv/install.sh -o "${UV_INSTALLER}"
    UV_INSTALL_DIR="$(_amlas_host_root)/bin" UV_NO_MODIFY_PATH=1 sh "${UV_INSTALLER}"
    rm -f "${UV_INSTALLER}"
    # Re-source common install locations so `uv` is on PATH for the rest of the run.
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    command -v uv >/dev/null 2>&1 || die "uv install appears to have failed (not on PATH). Re-open shell and re-run."
    log_info "uv installed: $(uv --version 2>&1)"
  fi

  # 3c. Node.js + CDK CLI — the cdk CLI is a NODE tool, NOT a Python dep (study §1.4).
  if command -v cdk >/dev/null 2>&1; then
    log_info "cdk CLI present: $(cdk --version 2>&1)"
  else
    log_warn "cdk CLI not found."
    if command -v npm >/dev/null 2>&1; then
      log_warn "Installing AWS CDK CLI globally via npm (npm install -g aws-cdk)."
      npm install -g aws-cdk@2  # Security review: pin to the v2 major line
      command -v cdk >/dev/null 2>&1 || die "cdk install via npm failed (not on PATH)."
      log_info "cdk installed: $(cdk --version 2>&1)"
    else
      die "cdk CLI missing AND npm is not installed. Install Node.js+npm (or 'brew install aws-cdk'), then re-run."
    fi
  fi
  # Node presence is informational (cdk depends on it; if cdk ran above, node exists).
  if command -v node >/dev/null 2>&1; then
    log_info "node present: $(node --version 2>&1)"
  fi

  # 3d. Docker + buildx — REQUIRED for the full agent-runtime phase (arm64 image
  # build/push), OPTIONAL otherwise. When the agent runtime is enabled (the
  # default), a missing/stopped Docker or missing buildx is a hard failure (fail
  # fast, before any mutation); with --skip-agent-runtime it stays a soft warn.
  if [ "$PREFLIGHT_ONLY" = "1" ] || [ "$SKIP_AGENT_RUNTIME" = "1" ]; then
    log_info "Docker build prerequisites skipped: this mode does not build a runtime image."
  elif command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    log_info "docker present and daemon reachable."
    if docker buildx version >/dev/null 2>&1; then
      log_info "docker buildx present: $(docker buildx version 2>&1 | head -1)"
    elif [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
      die "docker buildx not available but the agent runtime phase needs it. Install buildx (Docker Desktop bundles it) or re-run with --skip-agent-runtime."
    else
      log_warn "docker buildx missing — OK with --skip-agent-runtime (not building the runtime image)."
    fi
  elif [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
    die "docker not available/running but the FULL AGENT RUNTIME phase needs it to build+push the arm64 image. Start Docker, or re-run with --skip-agent-runtime (deploys the 9 CDK stacks only)."
  else
    log_warn "docker not available/running — OK: skipping the agent runtime phase (--skip-agent-runtime)."
  fi
fi

# =============================================================================
# STEP 4 — Install Python deps (provides `python -m infra.app` importability).
# The data extra is included so this single setup entry point prepares the
# advertised data-quality, imputation, synthetic-data, and Parquet capabilities.
# uv also provisions Python 3.11 per .python-version.
# =============================================================================
log_step "STEP 4 — prepare project dependencies"
if [ "$SYNTH_ONLY" = "1" ] || [ "$PREFLIGHT_ONLY" = "1" ]; then
  uv sync --frozen --extra cdk --extra agents
else
  uv sync --frozen --extra cdk --extra agents --extra data
fi
# Verify Python 3.11 is what uv resolved (informational; uv manages this).
log_info "uv Python: $(uv run python --version 2>&1)"

# =============================================================================
# STEP 5 — Export the proven AWS/CDK environment, THEN run the safety gate.
# The gate MUST pass before ANY mutating call below (secret/bootstrap/deploy).
# =============================================================================
log_step "STEP 5 — AWS environment + safety gate"
log_info "AWS identity and region were validated before dependency installation."

# Persist the validated, non-secret runtime selectors before any CDK or AgentCore
# phase can fail. setup.sh cannot export into its parent shell, but the TUI/CLI
# load the repo .env on their next run, so local Bedrock inference keeps working
# with the selected profile even after a partial backend deployment.
persist_env_value "$REPO_ROOT/.env" AWS_PROFILE "${AWS_PROFILE:-}"
persist_env_value "$REPO_ROOT/.env" AWS_REGION "$AWS_REGION"
persist_env_value "$REPO_ROOT/.env" AWS_DEFAULT_REGION "$AWS_REGION"
persist_env_value "$REPO_ROOT/.env" AMLAS_AWS_REGION "$AWS_REGION"
persist_env_value "$REPO_ROOT/.env" AMLAS_AWS_PROFILE "${AWS_PROFILE:-}"
persist_env_value "$REPO_ROOT/.env" AMLAS_SKIP_HARNESS_STACK "$SKIP_HARNESS"
persist_env_value "$REPO_ROOT/.env" AGENTCORE_RUNTIME_NAME "$AGENTCORE_RUNTIME_NAME"
persist_env_value "$REPO_ROOT/.env" AGENTCORE_EXEC_ROLE_NAME "$AGENTCORE_EXEC_ROLE_NAME"
persist_env_value "$REPO_ROOT/.env" AMLAS_DEPLOYMENT_PREFIX "$AMLAS_DEPLOYMENT_PREFIX"
persist_env_value "$REPO_ROOT/.env" AMLAS_SSM_PREFIX "$AMLAS_SSM_PREFIX"
persist_env_value "$REPO_ROOT/.env" TAVILY_SECRET_ID "$TAVILY_SECRET_ID"
persist_env_value "$REPO_ROOT/.env" AMLAS_MODEL_PACKAGE_GROUP "${AMLAS_MODEL_PACKAGE_GROUP:-$AMLAS_DEPLOYMENT_PREFIX-models}"
persist_env_value "$REPO_ROOT/.env" AMLAS_JOBS_TABLE "$AMLAS_JOBS_TABLE"

# =============================================================================
# STEP 6 — Reconcile .env deploy-critical keys.
#   * AMLAS_AWS_ACCOUNT_ID must be set (REQUIRED/fail-loud config key F4).
#   * AMLAS_KMS_KEY_ARN must be EMPTY so CDK mints per-stack CMKs (F15). We assert
#     it is empty in the loaded .env; app.py also guards, but we keep .env honest.
# We do NOT rewrite the user's .env destructively; we only warn if KMS is non-empty
# and not a real arn:aws:kms: value (the app.py guard will neutralize it anyway).
# =============================================================================
log_step "STEP 6 — .env reconciliation (account set, KMS empty per F15)"
if [ -z "${AMLAS_AWS_ACCOUNT_ID:-}" ]; then
  die "AMLAS_AWS_ACCOUNT_ID is empty — REQUIRED (F4). Set it in .env and re-run."
fi
if [ -n "${AMLAS_KMS_KEY_ARN:-}" ]; then
  case "$AMLAS_KMS_KEY_ARN" in
    arn:aws:kms:*)
      log_info "AMLAS_KMS_KEY_ARN is a real KMS ARN — will import it as a BYO shared CMK." ;;
    *)
      log_warn "AMLAS_KMS_KEY_ARN='$AMLAS_KMS_KEY_ARN' is NOT a valid arn:aws:kms: value."
      log_warn "app.py guard (F15) will treat it as empty and mint per-stack CMKs (secure default)." ;;
  esac
else
  log_info "AMLAS_KMS_KEY_ARN is empty — CDK will mint per-stack rotation-enabled CMKs (F15 secure default)."
fi

# =============================================================================
# STEP 7 — Create-or-update the Tavily secret (SecretsStack IMPORTS it; H7/F11).
# Idempotent: create if absent, else put a new value. Uses the resolved key value
# (real tvly-... if provided, otherwise the documented placeholder).
# =============================================================================
REUSED_RUNTIME_ROLE_ARN=""
REUSED_RUNTIME_ROLE_ID=""
if [ "$SYNTH_ONLY" != "1" ]; then
  normalize_model_env
  MODEL_PROFILES="$(resolve_model_profiles)"
  preflight_deployment_services "$MODEL_PROFILES"
  if [ "$AGENT_RUNTIME_ONLY" = "0" ]; then preflight_cloudtrail_capacity; fi
  if [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
    RUNTIME_PREFLIGHT_OUT="$(AMLAS_RUNTIME_PREFLIGHT_ONLY=1 AWS_ACCOUNT_ID="$AMLAS_AWS_ACCOUNT_ID" AMLAS_AWS_REGION="$AWS_REGION" bash "$SCRIPT_DIR/lib/deploy_agentcore_runtime.sh")" || die "Runtime role compatibility preflight failed."
    printf '%s\n' "$RUNTIME_PREFLIGHT_OUT"
    collect_runtime_role_reuse_witness "$RUNTIME_PREFLIGHT_OUT"
  fi
fi
if [ "$PREFLIGHT_ONLY" = "1" ]; then
  log_info "PREFLIGHT-PASS — no cloud resources changed."
  exit 0
fi


# Capture native baseline before the first cloud write. Preserve partial handles
# on any later error so cleanup never relies on an account-wide prefix sweep.
DEPLOYMENT_MANIFEST="$REPO_ROOT/output/deploy/deployment-manifest.json"
if [ "$SYNTH_ONLY" != "1" ]; then
  PHYSICAL_STACKS=()
  for logical_stack in $AMLAS_STACKS_DEPLOY_ORDER; do
    PHYSICAL_STACKS+=("$(deployment_stack_name "$logical_stack")")
  done
  MANIFEST_INIT_ARGS=(init --file "$DEPLOYMENT_MANIFEST" --account "$AMLAS_AWS_ACCOUNT_ID" --region "$AWS_REGION" --prefix "$AMLAS_DEPLOYMENT_PREFIX" --ssm-prefix "$AMLAS_SSM_PREFIX" --model-group "${AMLAS_MODEL_PACKAGE_GROUP:-$AMLAS_DEPLOYMENT_PREFIX-models}" --tool-bin "$(_amlas_host_root)/bin" --stacks "${PHYSICAL_STACKS[@]}" --secret-id "$TAVILY_SECRET_ID" --runtime-name "$AGENTCORE_RUNTIME_NAME" --role-stack "$AMLAS_DEPLOYMENT_PREFIX-runtime-role" --role-name "$AGENTCORE_EXEC_ROLE_NAME")
  if [ -n "${AWS_PROFILE:-}" ]; then MANIFEST_INIT_ARGS+=(--profile "$AWS_PROFILE"); fi
  append_runtime_role_reuse_args
  MANIFEST_INIT_OUT="$(python3 -I "$SCRIPT_DIR/lib/deployment_manifest.py" "${MANIFEST_INIT_ARGS[@]}")"
  AMLAS_DEPLOYMENT_ID="$(printf '%s' "$MANIFEST_INIT_OUT" | python3 -c 'import json,sys; print(json.load(sys.stdin)["deployment_id"])')"
  export AMLAS_DEPLOYMENT_ID
  persist_env_value "$REPO_ROOT/.env" AMLAS_DEPLOYMENT_ID "$AMLAS_DEPLOYMENT_ID"
  AMLAS_DEPLOY_TOOL_BIN="$(_amlas_host_root)/bin"
  export AMLAS_DEPLOY_TOOL_BIN
  persist_env_value "$REPO_ROOT/.env" AMLAS_DEPLOY_TOOL_BIN "$AMLAS_DEPLOY_TOOL_BIN"
  refresh_deployment_manifest() {
    python3 -I "$SCRIPT_DIR/lib/deployment_manifest.py" refresh --file "$DEPLOYMENT_MANIFEST"
  }
  preserve_deployment_exit() {
    local setup_rc=$?
    trap - EXIT
    if ! refresh_deployment_manifest; then
      log_error "Deployment handles were retained, but native inventory verification is incomplete."
      if [ "$setup_rc" = "0" ]; then setup_rc=1; fi
    fi
    exit "$setup_rc"
  }
  trap preserve_deployment_exit EXIT
fi

if [ "$SYNTH_ONLY" = "1" ]; then
  log_info "Tavily secret writes SKIPPED in synth-only mode."
else
log_step "STEP 7 — Tavily secret (create-or-update)"
if aws secretsmanager describe-secret --secret-id "$TAVILY_SECRET_ID" --region "$AWS_REGION" >/dev/null 2>&1; then
  if [ "$AMLAS_TAVILY_API_KEY" = "$TAVILY_PLACEHOLDER_VALUE" ]; then
    log_info "Existing Tavily secret preserved: no replacement key was supplied."
  else
    write_tavily_secret put "$TAVILY_SECRET_ID"
    log_info "Secret '$TAVILY_SECRET_ID' already existed — updated its value."
  fi
else
  write_tavily_secret create "$TAVILY_SECRET_ID"
  log_info "Secret '$TAVILY_SECRET_ID' created."
fi
if [ "$AMLAS_TAVILY_API_KEY" = "$TAVILY_PLACEHOLDER_VALUE" ]; then
  log_info "No Tavily key supplied. No-key research and supplied-URL retrieval remain available; rerun with --ask-tavily-key to configure a key."
fi

fi

# Path the CDK deploy writes stack outputs to (also referenced by the final
# summary + teardown). Defined BEFORE the CDK guard so it is always set even when
# --agent-runtime-only skips the deploy block (set -u safety).
mkdir -p "$REPO_ROOT/output/live-results"
OUTPUTS_FILE="$REPO_ROOT/output/live-results/cdk-outputs.json"

# =============================================================================
# STEPS 8-10 — the CDK deploy (bootstrap → synth → deploy the 9-stack set).
# Skipped entirely by --agent-runtime-only (the 9 stacks are assumed already up).
# =============================================================================
if [ "$AGENT_RUNTIME_ONLY" = "1" ]; then
  log_step "STEPS 8-10 — CDK deploy SKIPPED (per --agent-runtime-only)"
else

# =============================================================================
# STEP 8 — cdk bootstrap (idempotent: skip if BootstrapVersion SSM param exists).
# The fresh-account first-deploy prereq the README omits (F1 / FIX-06).
# =============================================================================
log_step "STEP 8 — cdk bootstrap"
if [ "$SYNTH_ONLY" = "1" ]; then
  log_info "Bootstrap SKIPPED in synth-only mode."
elif [ "$SKIP_BOOTSTRAP" = "1" ]; then
  log_info "Bootstrap SKIPPED via --skip-bootstrap."
elif aws ssm get-parameter --name /cdk-bootstrap/hnb659fds/version --region "$AWS_REGION" >/dev/null 2>&1; then
  BOOTSTRAP_VER="$(aws ssm get-parameter --name /cdk-bootstrap/hnb659fds/version --region "$AWS_REGION" --query Parameter.Value --output text 2>/dev/null || echo '?')"
  log_info "CDK already bootstrapped (BootstrapVersion=$BOOTSTRAP_VER) — skipping bootstrap."
else
  log_info "Bootstrapping aws://$AMLAS_AWS_ACCOUNT_ID/$AWS_REGION ..."
  cdk bootstrap "aws://$AMLAS_AWS_ACCOUNT_ID/$AWS_REGION"
  log_info "Bootstrap complete."
fi

# =============================================================================
# STEP 9 — cdk synth (dry run; provisions nothing; validates the app imports).
# =============================================================================
log_step "STEP 9 — cdk synth"
uv run cdk synth --app "$AMLAS_CDK_APP" >/dev/null
log_info "cdk synth OK (templates written to cdk.out/)."

if [ "$SYNTH_ONLY" = "1" ]; then
  log_step "SYNTH-ONLY MODE — stopping before deploy (per --synth-only)."
  log_info "Log file: $LOG_FILE"
  exit 0
fi

# =============================================================================
# STEP 10 — Deploy the explicit 9-stack set in the proven order (NEVER --all).
# Writes outputs to output/live-results/cdk-outputs.json (used by teardown too).
# =============================================================================
log_step "STEP 10 — cdk deploy (9-stack set, proven order)"
log_info "Deploying stacks: $AMLAS_STACKS_DEPLOY_ORDER"
read -r -a DEPLOY_STACKS <<< "$AMLAS_STACKS_DEPLOY_ORDER"
uv run cdk deploy "${DEPLOY_STACKS[@]}" \
  --app "$AMLAS_CDK_APP" \
  --require-approval never \
  --region "$AWS_REGION" \
  --outputs-file "$OUTPUTS_FILE" \
  --progress events
log_info "cdk deploy returned success."

fi  # end of the CDK-deploy block (skipped by --agent-runtime-only)

# =============================================================================
# STEP 10.5 — FULL AGENT RUNTIME (build+push arm64 image → create/discover the
# managed AgentCore runtime → publish its ARN to SSM). This is what makes the
# managed InvokeAgentRuntime endpoint real (the local TUI/CLI already work off
# the 9 stacks). RuntimeStack is intentionally NOT a CDK stack here (fixed-name
# collision with AsyncTrainingStack — F13); the runtime is created via the CLI
# path in scripts/lib/deploy_agentcore_runtime.sh. Skipped by --skip-agent-runtime.
# =============================================================================
if [ "$SKIP_AGENT_RUNTIME" = "1" ]; then
  log_step "STEP 10.5 — FULL AGENT RUNTIME (SKIPPED via --skip-agent-runtime)"
  log_warn "Managed AgentCore runtime NOT deployed. The local TUI/CLI still work; the"
  log_warn "managed InvokeAgentRuntime endpoint will be absent until you re-run with the"
  log_warn "agent-runtime phase enabled (default) or 'scripts/setup.sh --agent-runtime-only'."
else
  log_step "STEP 10.5 — FULL AGENT RUNTIME (build+push image → create runtime → publish SSM)"

  # 10.5a — build + push the arm64 runtime image to the amlas-runtime ECR repo
  # (created by AgentCoreRuntimeStack). The helper resolves the account from
  # the authenticated profile and logs into ECR itself.
  log_info "Building + pushing the arm64 runtime image (tag=$DEFAULT_AGENTCORE_IMAGE_TAG) ..."
  AWS_ACCOUNT_ID="$AMLAS_AWS_ACCOUNT_ID" \
    AMLAS_AWS_REGION="$AWS_REGION" \
    IMAGE_TAG="$DEFAULT_AGENTCORE_IMAGE_TAG" \
    bash "$SCRIPT_DIR/lib/build_runtime_image.sh"
  log_info "Runtime image build+push complete."

  # 10.5b — create-or-discover the AgentCore runtime + its exec role, poll to READY.
  # Capture the output so we can extract the ARN it prints (AGENTCORE_RUNTIME_ARN=...).
  log_info "Creating/discovering the AgentCore runtime '$AGENTCORE_RUNTIME_NAME' (polls to READY; can take minutes) ..."
  if RUNTIME_DEPLOY_OUT="$(AWS_ACCOUNT_ID="$AMLAS_AWS_ACCOUNT_ID" \
    AMLAS_AWS_REGION="$AWS_REGION" \
    AGENTCORE_RUNTIME_NAME="$AGENTCORE_RUNTIME_NAME" \
    IMAGE_TAG="$DEFAULT_AGENTCORE_IMAGE_TAG" \
    bash "$SCRIPT_DIR/lib/deploy_agentcore_runtime.sh" 2>&1)"; then
    printf '%s\n' "$RUNTIME_DEPLOY_OUT"
  else
    runtime_rc=$?
    printf '%s\n' "$RUNTIME_DEPLOY_OUT" >&2
    log_error "Managed runtime deployment failed; runtime ARN was not published."
    exit "$runtime_rc"
  fi

  # Extract the ARN + final status the script emitted (last match wins).
  RUNTIME_ARN="$(printf '%s\n' "$RUNTIME_DEPLOY_OUT" | sed -n 's/^AGENTCORE_RUNTIME_ARN=//p' | tail -1)"
  RUNTIME_STATUS="$(printf '%s\n' "$RUNTIME_DEPLOY_OUT" | sed -n 's/^FINAL_STATUS=//p' | tail -1)"

  # 10.5c — publish the ARN to SSM so the TUI/CLI/dispatch resolve it (the loop
  # deploy_agentcore_runtime.sh does not publish SSM itself; it prints the ARN.
  # from AMLAS_AGENTCORE_SSM_PARAM when AGENTCORE_RUNTIME_ARN is unset (FIX-12).
  if [ "$RUNTIME_STATUS" != "READY" ] || [ -z "$RUNTIME_ARN" ] || [ "$RUNTIME_ARN" = "None" ] || ! printf '%s\n' "$RUNTIME_DEPLOY_OUT" | grep -qx 'AGENTCORE_RUNTIME_DEPLOY_DONE'; then
    die "Runtime output did not verify a usable READY ARN; SSM was not updated."
  fi
  publish_owned_parameter "$AMLAS_AGENTCORE_SSM_PARAM" "$RUNTIME_ARN"
  log_info "Published verified READY runtime ARN to SSM $AMLAS_AGENTCORE_SSM_PARAM=$RUNTIME_ARN"

  # 10.5d — publish the session (staging) bucket to SSM so the CLI/runtime resolve it
  # when .env is blank (F2). STEP 10.5 previously wrote ONLY the runtime-ARN param, so
  # a clean run left the account half-wired: in Subtask 2 the /amlas/session-bucket
  # param had to be published BY HAND. Resolve the value DYNAMICALLY, mirroring
  # deploy_agentcore_runtime.sh uses the same precedence: exported AMLAS_SESSION_BUCKET, then
  # the AsyncTrainingStack CFN export 'amlas-async-session-bucket'
  # (e.g. asynctrainingstack-sessionbucket9c3e1938-b8mfmm7oj42u), then any existing SSM
  # value — never hardcoded. Idempotent via --overwrite.
  SESSION_BUCKET="$(printf '%s\n' "$RUNTIME_DEPLOY_OUT" | sed -n 's/^RESOLVED_SESSION_BUCKET=//p' | tail -1)"
  if [ -n "$SESSION_BUCKET" ] && [ "$SESSION_BUCKET" != "None" ]; then
    publish_owned_parameter "$AMLAS_SESSION_BUCKET_SSM_PARAM" "$SESSION_BUCKET"
    log_info "Published session bucket to SSM $AMLAS_SESSION_BUCKET_SSM_PARAM=$SESSION_BUCKET"
  else
    log_warn "Could not resolve the session bucket (AMLAS_SESSION_BUCKET / AsyncTrainingStack"
    log_warn "export 'amlas-async-session-bucket' / SSM all empty) — $AMLAS_SESSION_BUCKET_SSM_PARAM NOT updated."
    log_warn "Re-run 'scripts/setup.sh --agent-runtime-only' after confirming AsyncTrainingStack is deployed."
  fi
fi

# =============================================================================
# STEP 11 — Post-deploy verification + final summary.
# =============================================================================
log_step "STEP 11 — Verification + summary"

# List the CREATE/UPDATE_COMPLETE stacks matching our set (evidence, not assumption).
log_info "CloudFormation stack statuses (amlas 9-set + CDKToolkit):"
verify_deployed_stacks

# Show the /amlas/* SSM params the deploy wrote (config feedback loop intact).
log_info "$AMLAS_SSM_PREFIX/* SSM parameters:"
aws ssm get-parameters-by-path --path "$AMLAS_SSM_PREFIX" --recursive --region "$AWS_REGION" \
  --query "Parameters[].[Name,Value]" --output text 2>/dev/null | sed 's/^/    /' || \
  log_warn "Could not list /amlas SSM params (non-fatal)."

# Show the resolved AgentCore runtime ARN (if the agent phase ran + published it).
if [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
  RT_ARN_NOW="$(aws ssm get-parameter --name "$AMLAS_AGENTCORE_SSM_PARAM" --region "$AWS_REGION" \
    --query 'Parameter.Value' --output text 2>/dev/null || echo '')"
  if [ -n "$RT_ARN_NOW" ] && [ "$RT_ARN_NOW" != "None" ]; then
    log_info "AgentCore runtime ARN ($AMLAS_AGENTCORE_SSM_PARAM): $RT_ARN_NOW"
  else
    log_warn "AgentCore runtime ARN not present in SSM — see STEP 10.5 output above."
  fi
fi

log_step "AMLAS ONE-CLICK SETUP COMPLETE"
log_info "Outputs file: $OUTPUTS_FILE"
log_info "Log file:     $LOG_FILE"
if [ "$SKIP_AGENT_RUNTIME" = "0" ]; then
  log_info "FULL stack deployed (9 CDK stacks + managed AgentCore runtime). Launch: ./bin/amlas-ui"
else
  log_info "9 CDK stacks deployed; managed AgentCore runtime SKIPPED. Local TUI still works: ./bin/amlas-ui"
  log_info "To add the managed runtime later: scripts/setup.sh --agent-runtime-only"
fi
