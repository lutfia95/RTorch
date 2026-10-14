#!/usr/bin/env bash
# Ensure Maturin and patchelf are available in EquiLoad's selected environment.
set -euo pipefail

python="${VIRTUAL_ENV:?source scripts/setup-equiload-env.sh first}/bin/python"
if "$python" -c 'import maturin' >/dev/null 2>&1 && command -v patchelf >/dev/null 2>&1; then
    return 0 2>/dev/null || exit 0
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" 'maturin[patchelf]>=1.5,<2'
else
    "$python" -m pip install 'maturin[patchelf]>=1.5,<2'
fi
