#!/usr/bin/env bash
# Sourceable Linux host preparation. No authentication or AWS resource writes.
# setup calls ensure_aws_cli_prerequisite before identity and full prep after its account gate.

_amlas_host_error() { printf 'Host prerequisite error: %s\n' "$*" >&2; return 1; }

_amlas_host_root() {
  if [ -n "${TMPDIR:-}" ]; then
    printf '%s/amlas-host-tools\n' "${TMPDIR%/}"
  elif [ -n "${XDG_CACHE_HOME:-}" ]; then
    printf '%s/amlas-host-tools\n' "${XDG_CACHE_HOME%/}"
  else
    local repo
    repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)" || return 1
    printf '%s/../.amlas-host-tools\n' "$repo"
  fi
}

_amlas_host_platform() {
  local os_id version
  os_id="$(awk -F= '$1 == "ID" {gsub(/"/, "", $2); print $2}' /etc/os-release)" || return 1
  version="$(awk -F= '$1 == "VERSION_ID" {gsub(/"/, "", $2); print $2}' /etc/os-release)" || return 1
  printf '%s:%s\n' "$os_id" "$version"
}

_amlas_host_node_ready() {
  local version major
  version="$(node --version 2>/dev/null)" || return 1
  version="${version#v}"; major="${version%%.*}"
  case "$major" in ''|*[!0-9]*) return 1 ;; esac
  [ "$major" -ge 22 ]
}

_amlas_host_tools_ready() {
  local missing=0 tool
  if ! _amlas_host_node_ready; then printf 'Missing prerequisite: node >=22\n' >&2; missing=1; fi
  for tool in npm cmake git gcc g++ make python3.11; do
    if ! "$tool" --version >/dev/null 2>&1; then printf 'Missing prerequisite: %s\n' "$tool" >&2; missing=1; fi
  done
  if ! zip -v >/dev/null 2>&1; then printf 'Missing prerequisite: zip\n' >&2; missing=1; fi
  if ! docker --version >/dev/null 2>&1; then printf 'Missing prerequisite: docker\n' >&2; missing=1; fi
  [ "$missing" -eq 0 ]
}

_amlas_host_check() {
  local missing=0
  if ! aws --version >/dev/null 2>&1; then printf 'Missing prerequisite: AWS CLI\n' >&2; missing=1; fi
  _amlas_host_tools_ready || missing=1
  if ! docker info >/dev/null 2>&1; then printf 'Missing prerequisite: usable Docker daemon access\n' >&2; missing=1; fi
  if ! docker buildx version >/dev/null 2>&1; then printf 'Missing prerequisite: Docker buildx\n' >&2; missing=1; fi
  [ "$missing" -eq 0 ]
}

ensure_aws_cli_prerequisite() {
  local tool_root installer
  tool_root="$(_amlas_host_root)" || return 1
  export PATH="$tool_root/bin:$PATH"
  # A supported CLI stays untouched; the setup CLI-shape gate verifies AgentCore features.
  if command -v aws >/dev/null 2>&1 && aws --version >/dev/null 2>&1; then return 0; fi
  command -v curl >/dev/null 2>&1 || { _amlas_host_error 'Install curl using approved host administration before AWS CLI bootstrap.'; return 1; }
  command -v sha256sum >/dev/null 2>&1 || { _amlas_host_error 'sha256sum is required to verify the official AWS CLI installer.'; return 1; }
  mkdir -p "$tool_root/bin" "$tool_root/data" || return 1
  installer="$tool_root/aws-cli-install.sh"
  curl --proto '=https' --tlsv1.2 -fLsS --retry 2 https://awscli.amazonaws.com/v2/install.sh -o "$installer" || return 1
  # Inspected official installer; it verifies Linux packages with AWS's public signing key.
  # Fail closed if this script changes instead of executing an unreviewed replacement.
  if ! printf '%s  %s\n' '80f4b79b35bb5c427261e8e9f0fd6959ad3513a697f6e37d823959a72cb654dc' "$installer" | sha256sum -c -; then  # pragma: allowlist secret -- public AWS installer SHA256, verified from its official HTTPS source.
    _amlas_host_error 'AWS CLI installer checksum mismatch; refresh the reviewed official installer before retrying.'
    return 1
  fi
  XDG_DATA_HOME="$tool_root/data" XDG_BIN_HOME="$tool_root/bin" bash "$installer" --quiet || return 1
  aws --version >/dev/null 2>&1 || { _amlas_host_error 'AWS CLI bootstrap did not produce a usable CLI.'; return 1; }
}

_amlas_host_configure_paths() {
  local tool_root
  tool_root="$(_amlas_host_root)" || return 1
  export PATH="$tool_root/bin:$tool_root/npm/bin:$PATH"
  export npm_config_prefix="$tool_root/npm"
  export DOCKER_CONFIG="${DOCKER_CONFIG:-$tool_root/docker-config}"
  mkdir -p "$tool_root/bin" "$npm_config_prefix" "$DOCKER_CONFIG" || return 1
  # AL2023's versioned binaries may not have unversioned alternatives selected.
  if ! _amlas_host_node_ready && command -v node-22 >/dev/null 2>&1; then
    ln -sf "$(command -v node-22)" "$tool_root/bin/node" || return 1
  fi
  if ! npm --version >/dev/null 2>&1 && command -v npm-22 >/dev/null 2>&1; then
    ln -sf "$(command -v npm-22)" "$tool_root/bin/npm" || return 1
  fi
  hash -r
}

_amlas_host_configure_docker() {
  local tool_root real_docker quoted_binary quoted_config
  tool_root="$(_amlas_host_root)" || return 1
  real_docker="$(command -v docker)" || { _amlas_host_error 'Docker installation did not produce a CLI.'; return 1; }
  if docker info >/dev/null 2>&1; then return 0; fi
  if ! sudo -n true >/dev/null 2>&1; then
    _amlas_host_error 'Docker needs authorized noninteractive sudo or existing daemon access; no login or group change is performed.'
    return 1
  fi
  if ! sudo -n "$real_docker" info >/dev/null 2>&1; then
    command -v systemctl >/dev/null 2>&1 || { _amlas_host_error 'Start the local Docker daemon through approved host administration.'; return 1; }
    sudo -n systemctl start docker || return 1
    sudo -n "$real_docker" info >/dev/null 2>&1 || { _amlas_host_error 'The Docker daemon remains unavailable.'; return 1; }
  fi
  # Use one scoped client wrapper. Never change socket modes, enable TCP, or require a new login.
  printf -v quoted_binary '%q' "$real_docker"
  printf -v quoted_config '%q' "$DOCKER_CONFIG"
  {
    printf '#!/usr/bin/env bash\n'
    printf 'default_config=%s\n' "$quoted_config"
    printf 'exec sudo -n /usr/bin/env "DOCKER_CONFIG=${DOCKER_CONFIG:-$default_config}" %s "$@"\n' "$quoted_binary"
  } > "$tool_root/bin/docker" || return 1
  chmod 700 "$tool_root/bin/docker" || return 1
  hash -r
  docker info >/dev/null 2>&1 || { _amlas_host_error 'The scoped Docker client wrapper could not access the daemon.'; return 1; }
}

_amlas_host_ensure_buildx() {
  local arch digest binary download
  if docker buildx version >/dev/null 2>&1; then return 0; fi
  case "$(uname -m)" in
    aarch64|arm64) arch=arm64; digest=38e890e1a162bfdbf32fbe91404991dc4fc28e55b0de901b34374a061bab4d2c ;;
    x86_64|amd64) arch=amd64; digest=4fe4cc38adf48169132749b6ca22a990928db0118e3407584ee553723115d287 ;;
    *) _amlas_host_error 'Docker buildx automation supports Linux ARM64 and AMD64.'; return 1 ;;
  esac
  binary="$DOCKER_CONFIG/cli-plugins/docker-buildx"
  if [ -e "$binary" ]; then
    _amlas_host_error 'An existing buildx plugin is unusable; have its owner reconcile it before retrying.'
    return 1
  fi
  mkdir -p "$DOCKER_CONFIG/cli-plugins" || return 1
  download="$DOCKER_CONFIG/cli-plugins/buildx.download"
  # Official immutable release assets and API-published SHA-256 digests.
  curl --proto '=https' --tlsv1.2 -fLsS --retry 2 "https://github.com/docker/buildx/releases/download/v0.38.0/buildx-v0.38.0.linux-$arch" -o "$download" || return 1
  if ! printf '%s  %s\n' "$digest" "$download" | sha256sum -c -; then
    _amlas_host_error 'Docker buildx checksum verification failed.'
    return 1
  fi
  chmod 700 "$download" && mv "$download" "$binary" || return 1
  docker buildx version >/dev/null 2>&1 || { _amlas_host_error 'The verified buildx plugin is not usable with this Docker CLI.'; return 1; }
}

prepare_host_prerequisites() {
  local platform
  if ! _amlas_host_tools_ready >/dev/null 2>&1; then
    platform="$(_amlas_host_platform)" || return 1
    case "$platform" in
      amzn:2023) ;;
      amzn:2*) _amlas_host_error 'Amazon Linux 2 cannot run the required native Node toolchain; use the documented Cloud Desktop container route or an AL2023 EC2 host.'; return 1 ;;
      *) _amlas_host_error 'Automatic package preparation supports Amazon Linux 2023. On Ubuntu/Debian or another Linux host, install Node >=22, npm, Docker/buildx, CMake, Git, build tools, zip and Python 3.11 through approved host administration, then retry.'; return 1 ;;
    esac
    sudo -n true >/dev/null 2>&1 || { _amlas_host_error 'Authorized noninteractive sudo is required for AL2023 package preparation.'; return 1; }
    sudo -n dnf install -y docker nodejs22 nodejs22-npm cmake git python3.11 python3.11-devel gcc gcc-c++ make zip unzip tar gzip || return 1
  fi
  _amlas_host_configure_paths || return 1
  _amlas_host_configure_docker || return 1
  _amlas_host_ensure_buildx || return 1
  _amlas_host_check || { _amlas_host_error 'Host preparation is incomplete; review the missing prerequisites above.'; return 1; }
  printf 'Agentic ML Engineering host prerequisites ready.\n'
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
  set -euo pipefail
  [ "$#" -eq 1 ] || { printf 'Usage: %s --check | --install\n' "$0" >&2; exit 2; }
  case "$1" in
    --check) _amlas_host_check ;;
    --install) ensure_aws_cli_prerequisite && prepare_host_prerequisites ;;
    *) printf 'Usage: %s --check | --install\n' "$0" >&2; exit 2 ;;
  esac
fi
