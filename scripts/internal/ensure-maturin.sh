#!/usr/bin/env bash
# Ensure Maturin is available in EquiLoad's selected environment.
set -euo pipefail

python="${VIRTUAL_ENV:?source scripts/setup-equiload-env.sh first}/bin/python"
if "$python" -c 'import maturin' >/dev/null 2>&1; then
    return 0 2>/dev/null || exit 0
fi

if command -v uv >/dev/null 2>&1; then
    uv pip install --python "$python" 'maturin>=1.5,<2'
else
    "$python" -m pip install 'maturin>=1.5,<2'
fi
