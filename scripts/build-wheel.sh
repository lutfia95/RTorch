#!/usr/bin/env bash
# Build a release extension against the active Python's PyTorch and make it
# available to scripts in this checkout, without changing the Python environment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
source "$repo_root/scripts/setup-equiload-env.sh"
equiload_python="${EQUILOAD_PYTHON:-$VIRTUAL_ENV/bin/python}"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBTORCH="${LIBTORCH:-$("$equiload_python" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')}"
export EQUILOAD_CPU_BRIDGE="${EQUILOAD_CPU_BRIDGE:-1}"
export EQUILOAD_CXX11_ABI="${EQUILOAD_CXX11_ABI:-$("$equiload_python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')}"
if [[ -z "${HDF5_ROOT:-}" ]]; then
    for candidate in "$repo_root/third_party/hdf5" "$HOME/.local/equiload-hdf5"; do
        if [[ -s "$candidate/lib/libhdf5.so" && -f "$candidate/include/hdf5.h" ]]; then
            export HDF5_ROOT="$candidate"
            break
        fi
    done
fi
if [[ -z "${HDF5_ROOT:-}" ]]; then
    echo 'Set HDF5_ROOT to a usable HDF5 installation with include/ and lib/.' >&2
    exit 1
fi
"$equiload_python" -m maturin build --release --features hdf5 --skip-auditwheel --interpreter "$equiload_python" \
    --out "$repo_root/dist" "$@"
equiload_target="${CARGO_TARGET_DIR:-$repo_root/target}"
case "$(uname -s)" in
    Linux) cp "$equiload_target/release/libequiload.so" "$repo_root/equiload.so" ;;
    *) echo 'Install the generated wheel to use the extension on this platform.' ;;
esac
