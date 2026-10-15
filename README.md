# EquiLoad

EquiLoad is an optional native HDF5 graph loader for [Equitrain](https://github.com/lutfia95/equitrain) and PyTorch Geometric. It reads Equitrain trajectory files and returns ordinary PyG batches. EquiLoad accelerates CPU-side HDF5 reads and graph construction; models and GPU transfer remain PyTorch's responsibility.

## Install

EquiLoad is a CPU-side native extension: it loads HDF5 data and builds CPU PyG batches. Install your preferred PyTorch build first (including CUDA when Equitrain trains on a GPU), then install EquiLoad. The loader itself needs no CUDA toolkit or GPU.

A source installation is required: EquiLoad's native extension links against
PyTorch's C++ ABI, which is not stable across Torch releases. It must be built
against the exact Torch installation Equitrain will use. EquiLoad is therefore
distributed from its source repository, not as a generic PyPI wheel or sdist.
A source install needs Linux, Rust, a C++20 compiler, and HDF5 development
files. On Ubuntu:

```bash
sudo apt-get install build-essential libhdf5-dev
python -m pip install 'equitrain[torch]'
git clone https://github.com/lutfia95/RTorch.git equiload
cd equiload
bash scripts/install-python.sh --venv /path/to/equitrain/.venv
```

When EquiLoad lives beside the active virtual environment, the installer can
discover it automatically. It builds against the environment's exact Torch
ABI:

```bash
bash scripts/install-python.sh
```

For a custom HDF5 installation, set its prefix before installing:

```bash
export HDF5_ROOT=/path/to/hdf5
bash scripts/install-python.sh
```

### Local development wheel

```bash
bash scripts/build-wheel.sh
python -m pip install --force-reinstall --no-deps dist/equiload-*.whl
```

This produces a **local compatibility wheel**, tagged `linux_x86_64`; do not
upload or share it. It works only with the exact Torch ABI used to build it.
EquiLoad does not publish generic binary wheels because such a wheel can fail
to import with another supported Torch release. EquiLoad remains CPU-side;
PyTorch moves returned batches to the GPU.

## Use with Equitrain

After Equitrain’s optional integration is installed, opt in explicitly:

```bash
equitrain --backend torch --torch-loader equiload ...
```

The Python loader remains Equitrain’s default, so ordinary Equitrain users do not need Rust or EquiLoad.

## Python API

```python
import equiload

loader = equiload.HDF5GraphLoader.from_hdf5_file(
    "train.h5",
    r_max=5.0,
    atomic_numbers=[1, 6, 8],
    batch_size=32,
)

for batches in loader:
    for batch in batches:
        batch = batch.to("cuda")  # optional: PyTorch owns device transfer
        # train on batch
```

The supported public API consists only of `HDF5GraphDataset`, `HDF5GraphDatasetCollection`, and `HDF5GraphLoader`.

## Development

```bash
source scripts/setup-equiload-env.sh
cargo test --workspace --all-features
python -m pytest crates/equiload-python/tests -q
```

Install the versioned pre-commit checks once per clone:

```bash
rustup component add rustfmt clippy
bash scripts/install-git-hooks.sh
```

The hook runs Ruff formatting and linting for Python, then `cargo fmt` and strict Clippy for Rust. The installer records the active virtual environment, or the checkout's `.venv` when none is active.

`equiload-core` and `equiload-sys` are internal implementation crates and are not published. Development benchmarks live in [benchmarks](benchmarks/README.md).

## License

MIT. See [LICENSE](LICENSE).
