#!/usr/bin/env bash
# Build and install EquiLoad into the checkout's active/discovered virtual environment.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/setup-equiload-env.sh"
source "$repo_root/scripts/ensure-build-tools.sh"
"${VIRTUAL_ENV}/bin/python" -m maturin develop --release --features hdf5 "$@"
