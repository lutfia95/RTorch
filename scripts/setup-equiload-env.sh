#!/usr/bin/env bash
# EquiLoad / LibTorch environment for Linux + Bash + an existing uv venv.
#
#   source ./setup-equiload-env.sh
#   source ./setup-equiload-env.sh --venv /path/to/.venv --check
#   source ./setup-equiload-env.sh --cuda-root /usr/local/cuda-12.8
#
# This file must be SOURCED to change the calling shell. It does not build or
# clean your project. Run it again after moving the project or running uv sync.
# Missing Python packages are installed with uv pip, without editing uv.lock or
# pyproject.toml. Install your project's pinned dependencies with uv sync first.
# Existing torch is retained; absent torch is installed using uv's configured
# indexes. CUDA compiler packages follow torch.version.cuda (not nvidia-smi).
# Automatic pip-only CUDA setup requires CUDA 13+; older CUDA needs --cuda-root.
# No GPU is needed for --check. A system C/C++ compiler is still required.
#
# References:
# https://docs.astral.sh/uv/pip/environments/
# https://pypi.org/project/cuda-toolkit/
# https://docs.nvidia.com/cuda/cuda-installation-guide-linux/

if [ -z "${BASH_VERSION:-}" ]; then
    printf '%s\n' 'This script must be sourced from Bash.' >&2
    return 2 2>/dev/null || exit 2
fi
if [[ "${BASH_SOURCE[0]}" == "$0" && "${1-}" != "--help" ]]; then
    printf 'Use: source %q [--venv DIR] [--check]\n' "$0" >&2
    exit 2
fi

_equiload_setup_main() {
    local venv='' cuda_root='' install=1 check=0 cursor base uv python exports
    local script_dir
    script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" || return 1
    while (( $# )); do
        case "$1" in
            --venv|--cuda-root)
                if (( $# < 2 )) || [[ -z "$2" ]]; then
                    printf 'Missing value for %s\n' "$1" >&2; return 2
                fi
                if [[ "$1" == --venv ]]; then venv=$2; else cuda_root=$2; fi
                shift 2 ;;
            --no-install) install=0; shift ;;
            --check) check=1; shift ;;
            --help)
                cat <<'HELP'
Usage: source setup-equiload-env.sh [options]
  --venv DIR       Select an existing uv virtual environment explicitly.
  --cuda-root DIR  Use a full external toolkit matching PyTorch's CUDA series.
  --no-install     Do not install packages; still repair local linker aliases.
  --check          Compile, link and run a small CUDA/NVRTC probe; no GPU needed.
  --help           Show this help.

Discovery: --venv, then VIRTUAL_ENV, then .venv in the current directory or
its parents, then .venv beside this script or in its parents.
Exports include LIBTORCH, Torch_DIR, CUDA_HOME, CUDACXX and the build/library
search paths. Existing TORCH_CUDA_ARCH_LIST is preserved. Re-sourcing is safe.
Linux/Bash only. uv must be installed. CPU-only torch skips CUDA setup.
CUDA 13+ compiler components can be installed via pip; older CUDA requires
an existing full toolkit passed with --cuda-root. Project files are not edited.
HELP
                return 0 ;;
            *) printf 'Unknown option: %s (see --help)\n' "$1" >&2; return 2 ;;
        esac
    done
    if [[ -z "$venv" && -n "${VIRTUAL_ENV:-}" ]]; then venv=$VIRTUAL_ENV; fi
    if [[ -z "$venv" ]]; then
        for base in "$PWD" "$script_dir"; do
            cursor=$base
            while :; do
                if [[ -f "$cursor/.venv/pyvenv.cfg" ]]; then
                    venv="$cursor/.venv"; break
                fi
                [[ "$cursor" == / ]] && break
                cursor=$(dirname -- "$cursor")
            done
            [[ -n "$venv" ]] && break
        done
    fi
    if [[ -z "$venv" || ! -f "$venv/pyvenv.cfg" || ! -x "$venv/bin/python" ]]; then
        printf '%s\n' 'No usable venv found. Run uv sync, or pass --venv /path/to/.venv.' >&2
        return 1
    fi
    venv="$(cd -- "$venv" && pwd -P)" || return 1
    python="$venv/bin/python"
    uv="$(type -P uv)" || { printf '%s\n' 'uv is required but was not found on PATH.' >&2; return 1; }
    if ! exports="$("$python" - "$uv" "$install" "$check" "$cuda_root" <<'__EQUILOAD_PYTHON__'
import ast
import contextlib
import importlib
import importlib.metadata as md
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile

UV, INSTALL, CHECK, EXTERNAL = sys.argv[1:]
INSTALL, CHECK = INSTALL == '1', CHECK == '1'
VENV = Path(sys.prefix).absolute()
PYTHON = str(VENV / 'bin/python')
SITES = list(dict.fromkeys(Path(sysconfig.get_path(k)) for k in ('purelib', 'platlib')))

def fail(message):
    raise RuntimeError(message)

def note(message):
    print('[equiload-env] ' + message, file=sys.stderr)

def version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None

def run(argv, **kwargs):
    return subprocess.run([str(x) for x in argv], check=True, **kwargs)

def install(*requirements):
    if not INSTALL:
        fail('Missing packages; run again without --no-install: ' + ' '.join(requirements))
    note('Ensuring packages: ' + ' '.join(requirements))
    run([UV, 'pip', 'install', '--python', PYTHON, *requirements], stdout=sys.stderr)
    importlib.invalidate_caches()

def inside_venv(path):
    try:
        path.resolve().relative_to(VENV.resolve())
        return True
    except ValueError:
        return False

def alias(link, target):
    # Only create aliases in this venv, never in a system toolkit or uv cache.
    if link.exists():
        return
    if not inside_venv(link.parent):
        fail(f'Cannot add linker alias outside the venv: {link}. Use uv copy/hardlink mode.')
    if link.is_symlink():
        old = os.readlink(link)
        if '/' in old or not re.fullmatch(re.escape(link.name) + r'\.\d+(?:\.\d+)*', old):
            fail(f'Refusing to replace an unfamiliar broken symlink: {link} -> {old}')
        link.unlink()
    link.symlink_to(target)
    note(f'Link: {link.name} -> {target}')

def repair_library_names(directory):
    groups = {}
    for file in directory.glob('*.so.*'):
        match = re.fullmatch(r'(.+\.so)\.(\d+(?:\.\d+)*)', file.name)
        if match and file.is_file():
            groups.setdefault(match[1], []).append((file, tuple(map(int, match[2].split('.')))))
    for name, files in groups.items():
        if (directory / name).exists():
            continue
        if len({v[0] for _, v in files}) != 1:
            fail(f'Multiple ABI majors for {directory / name}; select the correct linker alias manually.')
        # Prefer the soname (e.g. .so.13); otherwise the newest full version.
        target = sorted(files, key=lambda item: (len(item[1]), tuple(-n for n in item[1])))[0][0]
        alias(directory / name, target.name)

def wheel_root(major):
    # Locate the actual installed compiler; do not hardcode Python/CUDA paths.
    candidates = [p / 'nvidia' / f'cu{major}' for p in SITES]
    try:
        dist = md.distribution('nvidia-cuda-nvcc')
        candidates += [Path(dist.locate_file(f)).absolute().parent.parent
                       for f in dist.files or [] if str(f).endswith('/bin/nvcc')]
    except md.PackageNotFoundError:
        pass
    return next((p for p in candidates if (p / 'bin/nvcc').is_file()), None)

def library_dirs(major):
    result = []
    for dist in md.distributions():
        name = (dist.metadata.get('Name') or '').lower().replace('_', '-')
        if not name.startswith('nvidia-'):
            continue
        suffix = re.search(r'-cu(\d+)$', name)
        if suffix and int(suffix[1]) != major:
            continue
        for file in dist.files or []:
            if '.so' not in Path(file).name:
                continue
            if any(re.fullmatch(r'cu\d+', part) and part != f'cu{major}' for part in Path(file).parts):
                continue
            path = Path(dist.locate_file(file)).absolute().parent
            if 'stubs' not in path.parts and path.is_dir() and path not in result:
                result.append(path)
    return result

def cmake_version(path):
    result = subprocess.run([str(path), '--version'], check=False,
                            capture_output=True, text=True)
    match = re.search(r'cmake version (\d+)\.(\d+)\.(\d+)', result.stdout)
    if result.returncode or not match:
        return None
    return tuple(map(int, match.groups()))

def main():
    if sys.platform != 'linux' or sys.prefix == sys.base_prefix:
        fail('This script requires Linux and a Python virtual environment.')
    if not version('torch'):
        install('torch')
    torch_version = md.version('torch')
    torch_root = Path(md.distribution('torch').locate_file('torch')).absolute()
    torch_config = torch_root / 'share/cmake/Torch'
    if not (torch_config / 'TorchConfig.cmake').is_file():
        fail(f'No Torch CMake configuration at {torch_config}. Check the torch installation.')
    # Reading the generated version file avoids importing torch before its
    # possibly missing CUDA runtime dependencies have been installed.
    series = 'unknown'
    for node in ast.parse((torch_root / 'version.py').read_text()).body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == 'cuda' for t in targets):
            series = ast.literal_eval(node.value)
            break
    if series == 'unknown':
        fail('Cannot read the CUDA version from torch/version.py.')
    note(f'Venv: {VENV}; PyTorch: {torch_version}; CUDA: {series or "CPU only"}')

    # Remove only search-path entries contributed by a previous successful run.
    env = dict(os.environ)
    old = json.loads(env.get('EQUILOAD_ENV_STATE', '{}'))
    for key, entries in old.get('paths', {}).items():
        env[key] = ':'.join(p for p in env.get(key, '').split(':') if p and p not in entries)
    unsets = []
    for key, value in old.get('values', {}).items():
        if env.get(key) == value:
            env.pop(key, None)
            unsets.append(key)
    additions, values = {}, {'VIRTUAL_ENV': str(VENV), 'LIBTORCH': str(torch_root), 'Torch_DIR': str(torch_config)}

    def prepend(key, paths):
        new = list(dict.fromkeys(str(p) for p in paths if p))
        previous = [p for p in env.get(key, '').split(':') if p and p not in new]
        env[key] = ':'.join(new + previous)
        additions[key] = new

    root = None
    libs = [torch_root / 'lib']
    bins = [VENV / 'bin']
    if series:
        if not re.fullmatch(r'\d+\.\d+', series):
            fail(f'Unrecognized torch.version.cuda: {series!r}')
        major = int(series.split('.')[0])
        if EXTERNAL:
            root = Path(EXTERNAL).expanduser().absolute()
        elif major < 13:
            fail(f'CUDA {series} pip wheels may not provide nvcc. Pass --cuda-root to a full matching toolkit.')
        else:
            packages = ('nvidia-cuda-nvcc', 'nvidia-cuda-cccl', 'nvidia-cuda-nvrtc',
                        'nvidia-cuda-runtime', 'nvidia-cuda-crt', 'nvidia-nvvm')
            root = wheel_root(major)
            missing = any(not version(p) for p in packages)
            skew = any(version(p) and '.'.join(version(p).split('.')[:2]) != series
                       for p in packages if p != 'nvidia-cuda-cccl')
            if not root or missing or skew:
                # Including the installed torch pins it and lets uv resolve its
                # exact NVIDIA dependencies together with the toolkit extras.
                install(f'torch=={torch_version}', f'cuda-toolkit[nvcc,cccl,nvrtc,cudart]=={series}.*')
                root = wheel_root(major)
            if root is None:
                fail('The installed compiler package did not provide bin/nvcc. Pass --cuda-root to a full toolkit.')

        nvcc = root / 'bin/nvcc'
        if not nvcc.is_file() or not os.access(nvcc, os.X_OK):
            fail(f'CUDA compiler is missing or not executable: {nvcc}')
        output = run([nvcc, '--version'], capture_output=True, text=True).stdout
        match = re.search(r'release\s+(\d+\.\d+)', output)
        if not match or match[1] != series:
            fail(f'nvcc must match PyTorch CUDA {series}; compiler reports: {output.strip()}')
        if not (root / 'include/cuda_runtime.h').is_file():
            fail(f'CUDA headers missing under {root / "include"}')

        if not EXTERNAL and (root / 'lib').is_dir() and not (root / 'lib64').exists():
            alias(root / 'lib64', 'lib')
        cuda_libs = list(dict.fromkeys(p for p in (root / 'lib64', root / 'lib', *library_dirs(major)) if p.is_dir()))
        repaired = set()
        for directory in cuda_libs:
            actual = directory.resolve()
            if inside_venv(directory) and actual not in repaired:
                repair_library_names(directory)
                repaired.add(actual)
        for name in ('libcudadevrt.a', 'libcudart_static.a', 'libcudart.so', 'libnvrtc.so'):
            if not any((p / name).is_file() for p in cuda_libs):
                fail(f'Missing {name}. Ensure the CUDA {series} runtime/NVRTC packages or full toolkit are installed.')
        libs += cuda_libs
        bins.insert(0, root / 'bin')
        values.update({key: str(root) for key in ('CUDA_HOME', 'CUDA_PATH', 'CUDA_BIN_PATH', 'CUDAToolkit_ROOT')})
        values.update(CUDACXX=str(nvcc), CUDA_LIB_PATH=str(root / 'lib64' if (root / 'lib64').is_dir() else root / 'lib'))
        note(f'CUDA toolkit: {root}')
    elif EXTERNAL:
        fail('--cuda-root was supplied, but this PyTorch installation is CPU-only.')

    prepend('PATH', bins)
    for key in ('LD_LIBRARY_PATH', 'LIBRARY_PATH', 'CMAKE_LIBRARY_PATH'):
        prepend(key, libs)
    prepend('CMAKE_PREFIX_PATH', [torch_root] + ([root] if root else []))
    cmake = shutil.which('cmake', path=env['PATH'])
    minimum_cmake = (3, 31, 9)
    if not cmake or cmake_version(cmake) is None or cmake_version(cmake) < minimum_cmake:
        install('cmake>=3.31.9,<4')
        cmake = shutil.which('cmake', path=env['PATH'])
    if not cmake or cmake_version(cmake) is None or cmake_version(cmake) < minimum_cmake:
        fail('cmake >= 3.31.9 is required but was not found after installation.')
    values['CMAKE'] = cmake
    env.update(values)
    if CHECK and root:
        note('Checking CUDA compilation, static runtime linking and NVRTC loading (no GPU required)...')
        with tempfile.TemporaryDirectory(prefix='equiload-cuda-check-') as tmp:
            source = Path(tmp) / 'check.cu'
            binary = Path(tmp) / 'check'
            source.write_text('#include <cuda_runtime.h>\n#include <nvrtc.h>\nint main() { int a=0,b=0; return int(nvrtcVersion(&a,&b)); }\n')
            run([root / 'bin/nvcc', source, '-lnvrtc', '-o', binary], env=env, stdout=sys.stderr)
            run([binary], env=env, stdout=sys.stderr)
        note('CUDA/NVRTC check passed.')
    state = json.dumps({'paths': additions, 'values': values}, separators=(',', ':'))
    lines = [f'unset {key}' for key in unsets if key not in values]
    for key in set(old.get('paths', {})) | set(additions):
        lines.append(f'export {key}={shlex.quote(env[key])}')
    lines += [f'export {key}={shlex.quote(value)}' for key, value in values.items()]
    lines.append('export EQUILOAD_ENV_STATE=' + shlex.quote(state))
    note('Ready. Run cargo build; if CMake cached an old toolkit, clean only equiload-sys once.')
    return '\n'.join(lines)

try:
    # Third-party imports must not contaminate the shell code on stdout.
    with contextlib.redirect_stdout(sys.stderr):
        commands = main()
    print(commands)
except (RuntimeError, OSError, ValueError, subprocess.CalledProcessError, ImportError) as exc:
    note(f'ERROR: {exc}')
    sys.exit(1)
__EQUILOAD_PYTHON__
    )"; then
        return 1
    fi
    eval "$exports" || return 1
    hash -r
}

if _equiload_setup_main "$@"; then
    unset -f _equiload_setup_main
    return 0 2>/dev/null || exit 0
else
    unset -f _equiload_setup_main
    return 1 2>/dev/null || exit 1
fi
