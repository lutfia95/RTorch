#!/usr/bin/env bash
# Configure the versioned EquiLoad Git hooks for this checkout.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

venv_dir="${VIRTUAL_ENV:-$repo_root/.venv}"
python="$venv_dir/bin/python"
if [[ ! -x "$python" ]]; then
    printf 'No virtual environment found at %s.\n' "$venv_dir" >&2
    printf 'Create one and install EquiLoad first; see README.md.\n' >&2
    exit 1
fi

if ! "$python" -m ruff --version >/dev/null 2>&1; then
    if command -v uv >/dev/null 2>&1; then
        uv pip install --python "$python" "ruff>=0.15,<1"
    elif "$python" -m pip --version >/dev/null 2>&1; then
        "$python" -m pip install "ruff>=0.15,<1"
    else
        printf 'Install uv or pip in %s, then rerun this script.\n' "$venv_dir" >&2
        exit 1
    fi
fi

git config --local core.hooksPath .githooks
git config --local equiload.hookPython "$python"
printf 'EquiLoad pre-commit hook installed (Python: %s).\n' "$python"
