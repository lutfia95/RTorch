#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd)"

HDF5_DIR="${HDF5_DIR:-$REPO_ROOT/third_party/hdf5}"
HDF5_VERSION="${HDF5_VERSION:-1.14}"
CONDA_CHANNEL="${CONDA_CHANNEL:-conda-forge}"
FORCE=0

usage() {
  cat <<USAGE
Download a local HDF5 runtime/development install for EquiLoad.

This uses conda, mamba, or micromamba to install HDF5 and libaec into a local
prefix. HDF5 does not provide a LibTorch-style universal zip archive.

Usage:
  scripts/download-hdf5.sh [--force]

Environment:
  HDF5_DIR       Install prefix. Default: $HDF5_DIR
  HDF5_VERSION   HDF5 version constraint. Default: $HDF5_VERSION
  CONDA_CHANNEL  Conda channel. Default: $CONDA_CHANNEL
  CONDA_EXE      Explicit conda/mamba/micromamba executable.

Examples:
  scripts/download-hdf5.sh
  HDF5_VERSION=1.14 scripts/download-hdf5.sh --force
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

find_conda() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    printf '%s\n' "$CONDA_EXE"
    return
  fi

  for candidate in micromamba mamba conda; do
    if command -v "$candidate" >/dev/null 2>&1; then
      command -v "$candidate"
      return
    fi
  done

  echo "Could not find conda, mamba, or micromamba." >&2
  echo "Install one of them, or install HDF5 with your system package manager." >&2
  exit 1
}

print_env() {
  local loader_path_var="LD_LIBRARY_PATH"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    loader_path_var="DYLD_LIBRARY_PATH"
  fi
  cat <<EOF

HDF5 is installed at:
  $HDF5_DIR

Use it with one command:
  source "$HDF5_DIR/equiload-env.sh"

Or set the variables manually:
  export HDF5_ROOT="$HDF5_DIR"
  export $loader_path_var="\$HDF5_ROOT/lib:\${$loader_path_var:-}"
  cargo build --features hdf5

If you also use the local LibTorch helper:
  export HDF5_ROOT="$HDF5_DIR"
  export LIBTORCH="$REPO_ROOT/third_party/libtorch"
  export $loader_path_var="\$LIBTORCH/lib:\$HDF5_ROOT/lib:\${$loader_path_var:-}"
EOF
}

write_env_file() {
  local loader_path_var="LD_LIBRARY_PATH"
  if [[ "$(uname -s)" == "Darwin" ]]; then
    loader_path_var="DYLD_LIBRARY_PATH"
  fi
  cat >"$HDF5_DIR/equiload-env.sh" <<EOF
# Source this file before building or running EquiLoad with HDF5 support:
#   source "$HDF5_DIR/equiload-env.sh"
export HDF5_ROOT="$HDF5_DIR"
export LIBTORCH="$REPO_ROOT/third_party/libtorch"
export $loader_path_var="\$LIBTORCH/lib:\$HDF5_ROOT/lib:\${$loader_path_var:-}"
EOF
}

CONDA_BIN="$(find_conda)"

if [[ -e "$HDF5_DIR" && "$FORCE" != "1" ]]; then
  echo "HDF5 directory already exists: $HDF5_DIR"
  echo "Pass --force to replace it."
  write_env_file
  print_env
  exit 0
fi

if [[ -e "$HDF5_DIR" ]]; then
  rm -rf "$HDF5_DIR"
fi

mkdir -p "$(dirname -- "$HDF5_DIR")"

case "$(basename -- "$CONDA_BIN")" in
  micromamba)
    "$CONDA_BIN" create -y -p "$HDF5_DIR" -c "$CONDA_CHANNEL" "hdf5=$HDF5_VERSION" libaec
    ;;
  *)
    "$CONDA_BIN" create -y -p "$HDF5_DIR" -c "$CONDA_CHANNEL" "hdf5=$HDF5_VERSION" libaec
    ;;
esac

if [[ ! -f "$HDF5_DIR/include/hdf5.h" || ! -d "$HDF5_DIR/lib" ]]; then
  echo "Installed prefix does not contain include/hdf5.h and lib/." >&2
  exit 1
fi

write_env_file
print_env
