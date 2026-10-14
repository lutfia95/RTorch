# EquiLoad

EquiLoad is an optional native HDF5 graph loader for [Equitrain](https://github.com/lutfia95/equitrain) and PyTorch Geometric. It reads Equitrain trajectory files and produces ordinary PyG batches, while keeping graph construction and HDF5 reads in native code.

It is deliberately small: EquiLoad accelerates loading. Model execution and GPU transfer remain PyTorch's responsibility.

## Install from this checkout

EquiLoad is not published to PyPI yet. A Rust compiler is needed only to build it; users of a future wheel will not need Rust.

On Linux, create and activate a Python environment with PyTorch, PyG, and Maturin:

```bash
uv venv
source .venv/bin/activate
uv pip install torch torch-geometric maturin
```

Build and install EquiLoad into that same environment:

```bash
bash scripts/install-python.sh
```

The installer sources the version-controlled setup helper automatically. You can also source `scripts/setup-equiload-env.sh` directly before running Cargo commands. The helper is safe to source repeatedly. It discovers the active `.venv`, locates the installed PyTorch/LibTorch files, installs a compatible CMake when required, and exports the necessary linker paths. For a CUDA PyTorch build it also prepares the matching CUDA compiler components. It must be sourced because it changes the current shell environment.

EquiLoad needs an HDF5 development installation at build time. On Ubuntu:

```bash
sudo apt-get install libhdf5-dev
```

If HDF5 is installed in a nonstandard prefix, set `HDF5_ROOT` before installing:

```bash
export HDF5_ROOT=/path/to/hdf5
bash scripts/install-python.sh
```

To create a redistributable wheel instead of installing the checkout:

```bash
bash scripts/build-wheel.sh
python -m pip install dist/equiload-*.whl
```

## Use with Equitrain

After Equitrain's optional EquiLoad loader integration is installed, select it explicitly:

```bash
equitrain --backend torch --torch-loader equiload ...
```

The normal Python loader remains Equitrain's default, so no Rust toolchain or native extension is required for regular Equitrain installations.

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

`HDF5GraphLoader` produces the same list-of-PyG-`Batch` structure used by Equitrain. It is CPU-side loading and graph construction; `batch.to("cuda")` measures or performs host-to-device transfer separately.

## Development

```bash
source scripts/setup-equiload-env.sh
cargo test --workspace --all-features
python -m pytest crates/equiload-python/tests -q
```

The Rust crates are `equiload`, `equiload-sys`, and `equiload-python`. The public Python package and wheel are both named `equiload`.

## License

MIT. See [LICENSE](LICENSE).
