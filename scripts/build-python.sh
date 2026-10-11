#!/usr/bin/env bash
# Build a release extension against the active Python's PyTorch and make it
# available to scripts in this checkout, without changing the Python environment.
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
rtorch_python="${RTORCH_PYTHON:-python3}"
export PATH="$HOME/.cargo/bin:$PATH"
export LIBTORCH="${LIBTORCH:-$("$rtorch_python" -c 'import pathlib, torch; print(pathlib.Path(torch.__file__).parent)')}"
export RTORCH_CPU_BRIDGE="${RTORCH_CPU_BRIDGE:-1}"
export RTORCH_CXX11_ABI="${RTORCH_CXX11_ABI:-$("$rtorch_python" -c 'import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))')}"
if [[ -z "${HDF5_ROOT:-}" ]]; then
    for candidate in "$repo_root/third_party/hdf5" "$HOME/.local/rtorch-hdf5"; do
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
"$rtorch_python" -m maturin build --release --features hdf5 --skip-auditwheel --interpreter "$rtorch_python" \
    --out "$repo_root/optimization_results/wheels" "$@"
rtorch_target="${CARGO_TARGET_DIR:-$repo_root/target}"
case "$(uname -s)" in
    Linux) cp "$rtorch_target/release/librtorch.so" "$repo_root/rtorch.so" ;;
    *) echo 'Install the generated wheel to use the extension on this platform.' ;;
esac
