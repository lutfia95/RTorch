#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

LIBTORCH_VERSION="${LIBTORCH_VERSION:-2.7.0}"
LIBTORCH_VARIANT="${LIBTORCH_VARIANT:-cpu}"
LIBTORCH_DIR="${LIBTORCH_DIR:-$REPO_ROOT/third_party/libtorch}"
LIBTORCH_CXX11_ABI="${LIBTORCH_CXX11_ABI:-1}"
FORCE=0

usage() {
  cat <<USAGE
Download the LibTorch C++ distribution used by RTorch.

Usage:
  scripts/download-libtorch.sh [--force]

Environment:
  LIBTORCH_VERSION    PyTorch/LibTorch version to download. Default: $LIBTORCH_VERSION
  LIBTORCH_VARIANT    Build variant: cpu, cu118, cu126, cu128, rocm6.3. Default: $LIBTORCH_VARIANT
  LIBTORCH_DIR        Install directory. Default: $LIBTORCH_DIR
  LIBTORCH_CXX11_ABI  Linux ABI package selector: 1 for cxx11 ABI, 0 for pre-cxx11 ABI.
                      Default: $LIBTORCH_CXX11_ABI
  LIBTORCH_URL        Full archive URL override. Skips automatic URL construction.

Examples:
  scripts/download-libtorch.sh
  LIBTORCH_VARIANT=cu128 scripts/download-libtorch.sh
  LIBTORCH_VERSION=2.7.0 LIBTORCH_VARIANT=cpu scripts/download-libtorch.sh --force
USAGE
}

while (($#)); do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Required command not found: $1" >&2
    exit 1
  fi
}

libtorch_url() {
  if [[ -n "${LIBTORCH_URL:-}" ]]; then
    printf '%s\n' "$LIBTORCH_URL"
    return
  fi

  local os
  os="$(uname -s)"
  case "$os" in
    Linux)
      local package_prefix
      if [[ "$LIBTORCH_CXX11_ABI" == "1" ]]; then
        package_prefix="libtorch-cxx11-abi-shared-with-deps"
      elif [[ "$LIBTORCH_CXX11_ABI" == "0" ]]; then
        package_prefix="libtorch-shared-with-deps"
      else
        echo "LIBTORCH_CXX11_ABI must be 1 or 0." >&2
        exit 2
      fi

      printf 'https://download.pytorch.org/libtorch/%s/%s-%s%%2B%s.zip\n' \
        "$LIBTORCH_VARIANT" "$package_prefix" "$LIBTORCH_VERSION" "$LIBTORCH_VARIANT"
      ;;
    *)
      echo "Automatic LibTorch URL construction is only configured for Linux." >&2
      echo "Set LIBTORCH_URL to an official LibTorch archive URL for this platform." >&2
      exit 2
      ;;
  esac
}

print_env() {
  local loader_path_var="LD_LIBRARY_PATH"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    loader_path_var="DYLD_LIBRARY_PATH"
  fi
  cat <<EOF

LibTorch is installed at:
  $LIBTORCH_DIR

Use it with:
  export LIBTORCH="$LIBTORCH_DIR"
  export $loader_path_var="\$LIBTORCH/lib:\${$loader_path_var:-}"
  cargo build
EOF
}

require_command curl
require_command unzip

if [[ -e "$LIBTORCH_DIR" && "$FORCE" != "1" ]]; then
  echo "LibTorch directory already exists: $LIBTORCH_DIR"
  echo "Pass --force to replace it."
  print_env
  exit 0
fi

URL="$(libtorch_url)"
TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

echo "Downloading LibTorch $LIBTORCH_VERSION ($LIBTORCH_VARIANT)"
echo "URL: $URL"
curl -fL --retry 3 --retry-delay 2 -o "$TMP_DIR/libtorch.zip" "$URL"

echo "Extracting archive"
unzip -q "$TMP_DIR/libtorch.zip" -d "$TMP_DIR"

if [[ ! -d "$TMP_DIR/libtorch/include" || ! -d "$TMP_DIR/libtorch/lib" ]]; then
  echo "Downloaded archive did not contain the expected libtorch/include and libtorch/lib directories." >&2
  exit 1
fi

mkdir -p "$(dirname -- "$LIBTORCH_DIR")"
if [[ -e "$LIBTORCH_DIR" ]]; then
  rm -rf "$LIBTORCH_DIR"
fi
mv "$TMP_DIR/libtorch" "$LIBTORCH_DIR"

print_env
