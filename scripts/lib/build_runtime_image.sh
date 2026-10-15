#!/usr/bin/env bash
# Build and push the AgentCore runtime image (linux/arm64) to the ECR
# repository created by AgentCoreRuntimeStack. Called by scripts/setup.sh.
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Honor an explicitly selected profile; otherwise use the standard AWS
# credential chain. Account and region are always resolved at runtime.
if [ -n "${AWS_PROFILE:-}" ]; then
  export AWS_PROFILE
fi
. "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/deploy_common.sh"
NON_INTERACTIVE=1
configure_deployment_namespace
resolve_aws_region
REGION="$AWS_REGION"
ACCT=${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}
if [ -z "${ACCT}" ] || [ "${ACCT}" = "None" ]; then
  echo "ERROR: could not resolve AWS account id (set AWS_ACCOUNT_ID or a valid AWS_PROFILE)" >&2
  exit 1
fi
REPO="$AMLAS_DEPLOYMENT_PREFIX-runtime"
REGISTRY="${ACCT}.dkr.ecr.${REGION}.amazonaws.com"
IMAGE_URI="${REGISTRY}/${REPO}"
# The image tag is parameterized so a rebuild can publish a new version without
# editing this helper. The deploy helper reads the same IMAGE_TAG value.
TAG="${IMAGE_TAG:-v1}"

START=$(date +%s)

echo "=== ECR LOGIN ==="
aws ecr get-login-password --region "${REGION}" \
  | docker login --username AWS --password-stdin "${REGISTRY}" 2>&1 || exit $?

echo "=== BUILDX BUILD + PUSH (linux/arm64) ==="
# Single-arch arm64 (AgentCore requirement). --provenance=false keeps a plain
# image manifest (not an OCI index) so describe-images/inspect report arm64.
docker buildx build \
  --platform linux/arm64 \
  --provenance=false \
  -t "${IMAGE_URI}:${TAG}" \
  -t "${IMAGE_URI}:latest" \
  --push \
  -f Dockerfile . 2>&1

BUILD_RC=$?
END=$(date +%s)
echo "=== BUILD_RC=${BUILD_RC} DURATION_SEC=$((END-START)) ==="
if [ "$BUILD_RC" -ne 0 ]; then
  echo "ERROR: runtime image build failed (exit $BUILD_RC)." >&2
  exit "$BUILD_RC"
fi
echo "IMAGE_URI=${IMAGE_URI}:${TAG}"
echo "RUNTIME_IMAGE_PUSH_DONE"
