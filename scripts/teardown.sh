#!/usr/bin/env bash
# Teardown only identities owned by a verified setup deployment manifest.
set -euo pipefail
SCRIPT_SOURCE="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_SOURCE")" && pwd)"
# shellcheck source=scripts/lib/deploy_common.sh
. "$SCRIPT_DIR/lib/deploy_common.sh"
REPO_ROOT="$(resolve_repo_root "$SCRIPT_SOURCE")"
cd "$REPO_ROOT"

NON_INTERACTIVE=0
DRY_RUN=0
MANIFEST_SELECTOR=""
PROFILE_SELECTOR=""
ACCOUNT_SELECTOR=""
REGION_SELECTOR=""
PREFIX_SELECTOR=""
PROFILE_SELECTED=0

print_help() {
  cat <<'HELP'
Usage: scripts/teardown.sh [--manifest PATH] [--yes] [--dry-run]
                          [--profile NAME] [--account-id ACCOUNT]
                          [--region REGION] [--deployment-prefix PREFIX]

The default manifest is output/deploy/deployment-manifest.json. Setup records
pre-existing resources as protected and new native identities with ownership tags.
Dry-run validates the exact recorded identities and performs no AWS mutations.
Cleanup stops recorded runtime sessions, removes its runtime and auxiliary role
stack, deletes recorded stacks in reverse order, and cleans exact owned retained
resources. KMS keys remain explicitly pending deletion for AWS's minimum 7 days.
Shared bootstrap resources are not owned; --nuke-bootstrap is rejected.
HELP
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --manifest|--profile|--account-id|--region|--deployment-prefix)
      [ "$#" -ge 2 ] && [ -n "$2" ] || { echo "$1 requires a value" >&2; exit 2; }
      case "$1" in
        --manifest) MANIFEST_SELECTOR="$2" ;;
        --profile) PROFILE_SELECTOR="$2"; PROFILE_SELECTED=1 ;;
        --account-id) ACCOUNT_SELECTOR="$2" ;;
        --region) REGION_SELECTOR="$2" ;;
        --deployment-prefix) PREFIX_SELECTOR="$2" ;;
      esac
      shift 2; continue ;;
    --yes|--non-interactive) NON_INTERACTIVE=1 ;;
    --dry-run) DRY_RUN=1 ;;
    --keep-bootstrap|--skip-harness) : ;; # The manifest is the authoritative selected set.
    --nuke-bootstrap) echo "Shared bootstrap resources are not owned by this deployment manifest; refusing --nuke-bootstrap." >&2; exit 2 ;;
    --help|-h) print_help; exit 0 ;;
    *) echo "Unknown argument: $1 (use --help)" >&2; exit 2 ;;
  esac
  shift
done

if [ -f "$REPO_ROOT/.env" ]; then load_env_file "$REPO_ROOT"; fi
MANIFEST_FILE="${MANIFEST_SELECTOR:-${AMLAS_DEPLOYMENT_MANIFEST:-$REPO_ROOT/output/deploy/deployment-manifest.json}}"
[ -f "$MANIFEST_FILE" ] || die "A deployment manifest is required: $MANIFEST_FILE. Existing resources cannot be adopted by prefix."
[ -f "$SCRIPT_DIR/lib/deployment_manifest.py" ] || die "Deployment manifest controller is missing."
command -v python3 >/dev/null 2>&1 || die "python3 is required for dependency-free manifest cleanup."
# Deterministic lookup only: use the setup-persisted path, then its ledger copy.
TOOL_BIN="${AMLAS_DEPLOY_TOOL_BIN:-}"
if [ -z "$TOOL_BIN" ]; then
  TOOL_BIN="$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("tool_bin", ""))' "$MANIFEST_FILE")"
fi
if [ -n "$TOOL_BIN" ] && [ -x "$TOOL_BIN/aws" ]; then
  export PATH="$TOOL_BIN:$PATH"
  hash -r
fi
command -v aws >/dev/null 2>&1 || die "The setup-selected AWS CLI is unavailable; no cleanup dependencies are installed."


if [ -n "$PREFIX_SELECTOR" ]; then
  RECORDED_PREFIX="$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["prefix"])' "$MANIFEST_FILE")"
  [ "$PREFIX_SELECTOR" = "$RECORDED_PREFIX" ] || die "Selected deployment prefix does not match the manifest."
fi

TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$REPO_ROOT/output/deploy/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/teardown-$TS.log"
exec > >(tee -a "$LOG_FILE") 2>&1
log_step "AMLAS EXACT-IDENTITY TEARDOWN ($TS)"
log_info "Manifest: $MANIFEST_FILE"

# The manifest supplies the target when no explicit selector is given. Config
# account/region selectors remain additional equality checks against that target.
ACCOUNT_SELECTOR="${ACCOUNT_SELECTOR:-${AMLAS_AWS_ACCOUNT_ID:-}}"
REGION_SELECTOR="${REGION_SELECTOR:-${AWS_REGION:-${AMLAS_AWS_REGION:-}}}"
MANIFEST_ARGS=(--file "$MANIFEST_FILE")
if [ -n "$ACCOUNT_SELECTOR" ]; then MANIFEST_ARGS+=(--account "$ACCOUNT_SELECTOR"); fi
if [ -n "$REGION_SELECTOR" ]; then MANIFEST_ARGS+=(--region "$REGION_SELECTOR"); fi
if [ "$PROFILE_SELECTED" = "1" ]; then MANIFEST_ARGS+=(--profile "$PROFILE_SELECTOR"); fi

log_step "Validate account, region, native identities and ownership; show plan"
python3 -I "$SCRIPT_DIR/lib/deployment_manifest.py" plan "${MANIFEST_ARGS[@]}"
if [ "$DRY_RUN" = "1" ]; then
  log_info "DRY-RUN: native identities observed; no AWS mutations performed."
  exit 0
fi

if [ "$NON_INTERACTIVE" != "1" ]; then
  log_warn "This permanently deletes the exact owned resources in the verified manifest plan."
  printf 'Type "destroy" to proceed: ' >/dev/tty
  CONFIRM=""
  if [ -r /dev/tty ]; then read -r CONFIRM </dev/tty; fi
  [ "$CONFIRM" = "destroy" ] || die "Confirmation not received; no AWS mutations performed."
fi

log_step "Guarded native cleanup"
python3 -I "$SCRIPT_DIR/lib/deployment_manifest.py" delete "${MANIFEST_ARGS[@]}" \
  --poll-attempts "${AMLAS_MANIFEST_POLL_ATTEMPTS:-240}" \
  --poll-seconds "${AMLAS_MANIFEST_POLL_SECONDS:-5}"
log_info "Cleanup verified. Reused resources and shared bootstrap resources were preserved; pending KMS keys are reported separately."
