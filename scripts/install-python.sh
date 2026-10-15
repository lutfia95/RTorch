#!/usr/bin/env bash
# Build and install EquiLoad into the checkout's active/discovered virtual environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# `--venv` selects the environment to build into; all remaining options are
# passed through to maturin. Keeping this wrapper-level option separate avoids
# accidentally forwarding it to `maturin develop`.
setup_args=()
maturin_args=()
while (($#)); do
    case "$1" in
        --venv)
            (($# >= 2)) || { echo '--venv requires a path' >&2; exit 2; }
            setup_args+=(--venv "$2")
            shift 2
            ;;
        *)
            maturin_args+=("$1")
            shift
            ;;
    esac
done

source "$repo_root/scripts/setup-equiload-env.sh" "${setup_args[@]}"
source "$repo_root/scripts/internal/ensure-maturin.sh"
"${VIRTUAL_ENV}/bin/python" -m maturin develop --release --features hdf5 "${maturin_args[@]}"
