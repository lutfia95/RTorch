#!/usr/bin/env bash
# Build a release extension against the active Python's PyTorch and make it
# available to scripts in this checkout, without changing the Python environment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/setup-equiload-env.sh"
source "$repo_root/scripts/internal/ensure-build-tools.sh"
equiload_python="${EQUILOAD_PYTHON:-$VIRTUAL_ENV/bin/python}"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBTORCH="${LIBTORCH:-$("$equiload_python" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')}"
export EQUILOAD_CPU_BRIDGE="${EQUILOAD_CPU_BRIDGE:-1}"
export EQUILOAD_CXX11_ABI="${EQUILOAD_CXX11_ABI:-$("$equiload_python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')}"
# CMake discovers a system HDF5 installation automatically. Set HDF5_ROOT only
# when using a custom prefix.
"$equiload_python" -m maturin build --release --features hdf5 --skip-auditwheel --interpreter "$equiload_python" \
    --out "$repo_root/dist" "$@"
find "$repo_root/dist" -maxdepth 1 -type f -name "equiload-*.whl" -exec \
    "$equiload_python" "$repo_root/scripts/internal/relocate-local-wheel.py" {} \;
