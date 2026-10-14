# EquiLoad

EquiLoad is an optional native HDF5 graph loader for [Equitrain](https://github.com/lutfia95/equitrain) and PyTorch Geometric. It reads Equitrain trajectory files and returns ordinary PyG batches. EquiLoad accelerates CPU-side HDF5 reads and graph construction; models and GPU transfer remain PyTorch's responsibility.

## Install from source

EquiLoad is currently distributed from source. You need Linux, Rust, a C++20 compiler, and HDF5 development files. On Ubuntu:

```bash
sudo apt-get install build-essential libhdf5-dev
python -m venv .venv
source .venv/bin/activate
python -m pip install torch torch-geometric
bash scripts/install-python.sh
```

The installer works from any current directory. It discovers the checkout’s virtual environment, configures the matching LibTorch/CUDA build paths, and installs EquiLoad into that environment. It uses `uv` when available and otherwise uses `pip` for any missing build dependencies.

For a custom HDF5 installation, set its prefix before installing:

```bash
export HDF5_ROOT=/path/to/hdf5
bash scripts/install-python.sh
```

## Local wheel

```bash
bash scripts/build-wheel.sh
python -m pip install --force-reinstall --no-deps dist/equiload-*.whl
```

This produces a **local compatibility wheel**, not a universal binary release: build and install it against the same Python, PyTorch ABI, CUDA variant, and HDF5 ABI. A future published release must provide a tested wheel matrix for those combinations.

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

`equiload-core` and `equiload-sys` are internal implementation crates and are not published. Development benchmarks live in [benchmarks](benchmarks/README.md); helper scripts live in [tools](tools/README.md).

## License

MIT. See [LICENSE](LICENSE).
