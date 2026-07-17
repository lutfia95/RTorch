# RTorch

[![RTorch](https://github.com/lutfia95/RTorch/actions/workflows/rtorch.yml/badge.svg?branch=main)](https://github.com/lutfia95/RTorch/actions/workflows/rtorch.yml)

RTorch (`rtorch`) is a Rust-facing tensor loading and preprocessing crate backed
by LibTorch, PyTorch's C++ distribution. It provides a small safe Rust API over a
C ABI and C++ LibTorch backend.

RTorch is useful when you want a PyTorch-like dataset and dataloader workflow in
Rust without starting Python at runtime.

## What Is Included

| RTorch feature | Python/PyTorch equivalent | Status |
| --- | --- | --- |
| `TensorDataset::from_class_folders(...)` | custom `torch.utils.data.Dataset` scanning class folders | Included |
| `.pt` single-tensor loading | `torch.load(path)` | Included for Python and LibTorch archives |
| `.h5` / `.hdf5` sample loading | `h5py.File(...)` plus tensor conversion | Included with the `hdf5` feature |
| `TensorDataset::len()` | `len(dataset)` | Included |
| `TensorDataset::get(index)` | `dataset[index]` / `__getitem__` | Included |
| `TensorDataset::class_names()` | `dataset.classes` / `ImageFolder.classes` | Included |
| `DataLoader::new(...)` | `torch.utils.data.DataLoader(...)` | Included |
| `DataLoaderConfig::drop_last` | `drop_last=True` | Included |
| `DataLoader::len()` | `len(loader)` | Included |
| `DataLoader` iterator | `for batch in loader:` | Included |
| `TransformPipeline` | `torchvision.transforms.Compose(...)` style preprocessing | Included |
| `to_float` | `tensor.float()` | Included |
| `resize` | `torch.nn.functional.interpolate(...)` | Included |
| `center_crop` | center crop transform | Included |
| `normalize` | `(x - mean) / std` | Included |
| `Batch::data()` | tensor values copied to host data | Lazy `Result<&[f32]>` view |
| `Batch::tensor()` | batch tensor | Included |
| `statistics(&[T])` | tensor reductions such as mean/std/min/max | Included |
| `Tensor` autograd basics | `requires_grad`, `backward`, `grad` | Included |

## Not Included Yet

| Feature | Current status |
| --- | --- |
| JPEG/PNG/WebP image decoding | Not included |
| Python bindings | Not included |
| multiprocessing workers / `num_workers` | Not included |
| prefetch queues | Not included |
| pinned memory | Not included |
| custom sampler API | Not included |
| custom collate function | Not included |
| augmentation-heavy transforms such as random crop/flip/color jitter | Not included |
| neural network modules, optimizers, training loops | Not included |
| TorchScript model APIs | Not included |

RTorch is currently a LibTorch-backed tensor dataset loader and small tensor API,
not a full PyTorch replacement.

## Repository Layout

```text
Cargo.toml
README.md
scripts/
  download-libtorch.sh
  download-hdf5.sh
examples/
  python/
  rust/
crates/
  rtorch/
    src/lib.rs
    examples/
      hdf5.rs
      statistics.rs
      tensor_folder.rs
  rtorch-sys/
    build.rs
    src/lib.rs
    ffi/
      CMakeLists.txt
      include/rtorch.h
      src/rtorch.cpp
```

The main safe Rust API lives in
[`crates/rtorch/src/lib.rs`](crates/rtorch/src/lib.rs). The C++ backend lives in
[`crates/rtorch-sys/ffi/src/rtorch.cpp`](crates/rtorch-sys/ffi/src/rtorch.cpp).

## Installation And Build

### Requirements

You need:

- Rust toolchain with Cargo
- CMake
- a C++17 compiler
- LibTorch
- optional HDF5 development/runtime libraries for `.h5` and `.hdf5`

### Install LibTorch

The helper script downloads the Linux CPU LibTorch C++ package into
`third_party/libtorch`:

```bash
bash scripts/download-libtorch.sh
```

For CUDA builds, choose the matching PyTorch LibTorch variant:

```bash
LIBTORCH_VARIANT=cu128 bash scripts/download-libtorch.sh
```

Use it with:

```bash
export LIBTORCH="$PWD/third_party/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo build
```

On macOS, use `DYLD_LIBRARY_PATH` instead:

```bash
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:${DYLD_LIBRARY_PATH:-}"
```

`LIBTORCH` must point to the LibTorch root directory containing `include/` and
`lib/`.

### Install HDF5

HDF5 support is explicit and reproducible: enable the Cargo `hdf5` feature when
building. The helper script installs HDF5 and `libaec` into
`third_party/hdf5` using conda, mamba, or micromamba:

```bash
bash scripts/download-hdf5.sh
```

Then source the generated environment file:

```bash
source third_party/hdf5/rtorch-env.sh
```

That sets:

```bash
export HDF5_ROOT="$PWD/third_party/hdf5"
export LIBTORCH="$PWD/third_party/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
```

On macOS the generated file sets `DYLD_LIBRARY_PATH` instead.

If you see:

```text
libhdf5.so.310: cannot open shared object file
```

then the binary was built with HDF5, but the dynamic loader cannot find the HDF5
shared libraries at runtime. Run:

```bash
source third_party/hdf5/rtorch-env.sh
```

or manually add `third_party/hdf5/lib` to `LD_LIBRARY_PATH` on Linux or
`DYLD_LIBRARY_PATH` on macOS.

On Ubuntu you can also install HDF5 from the system package manager:

```bash
sudo apt-get update
sudo apt-get install -y libhdf5-dev
```

### Build And Test

With LibTorch only:

```bash
export LIBTORCH="$PWD/third_party/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo build
cargo test
```

With LibTorch and local HDF5:

```bash
source third_party/hdf5/rtorch-env.sh
cargo build --features hdf5
cargo test --features hdf5
```

Run examples:

```bash
cargo run -p rtorch --example statistics
cargo run -p rtorch --features hdf5 --example hdf5 -- test_data/train.h5 positions
cargo run -p rtorch --example tensor_folder
```

## Add RTorch To Another Rust Project

Until RTorch is published on crates.io, add it as a path dependency:

```toml
[dependencies]
rtorch = { path = "/path/to/RTorch/crates/rtorch" }
```

Then build or run with LibTorch available:

```bash
export LIBTORCH=/path/to/libtorch
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo run
```

If you use HDF5:

```toml
[dependencies]
rtorch = { path = "/path/to/RTorch/crates/rtorch", features = ["hdf5"] }
```

```bash
export HDF5_ROOT=/path/to/hdf5
export LD_LIBRARY_PATH="$LIBTORCH/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
cargo run
```

## Dataset Formats

### Class Folder `.pt` Layout

RTorch can scan a class-folder dataset:

```text
dataset/
  cats/
    cat_001.pt
    cat_002.pt
  dogs/
    dog_001.pt
    nested/
      dog_002.pt
```

Rules:

- every top-level directory is one class
- class names come from directory names
- labels are assigned by sorted class directory order
- `.pt`, `.h5`, and `.hdf5` files are discovered recursively
- `.pt` files must contain a single tensor; model checkpoints, dictionaries,
  custom Python objects, and TorchScript modules are not tensor samples
- image-like transforms expect `C x H x W`

Example labels:

| Folder | Label |
| --- | --- |
| `cats` | `0` |
| `dogs` | `1` |

### HDF5 Layout

RTorch can load an HDF5 file directly:

```rust
let dataset = TensorDataset::from_hdf5_file("test_data/train.h5", "positions")?;
```

Build the consuming crate with the `hdf5` feature before using this API.

For HDF5 files in class folders, RTorch looks for a numeric dataset named:

1. `tensor`
2. `data`
3. `positions`

For atomistic-style HDF5 files, if a `structures` dataset exists with `offset`
and `length` fields, RTorch exposes each structure as one sample by slicing the
selected numeric dataset along the first dimension.

For example, a file with:

```text
positions:       [858, 3]
forces:          [858, 3]
atomic_numbers:  [858]
structures:      fields offset, length, cell, pbc, energy, ...
```

can be read as:

```rust
let positions = TensorDataset::from_hdf5_file("test_data/train.h5", "positions")?;
```

Each sample is then a variable-length slice such as `[16, 3]`, `[24, 3]`, etc.
Use `batch_size = 1` unless all samples in a batch have matching shapes.

## Python To Rust Examples

More side-by-side examples are in [`examples/`](examples/).

### Class Folder `.pt`

Python:

```python
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader

class TensorFolderDataset(Dataset):
    def __init__(self, root):
        self.classes = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
        self.samples = []
        for label, class_name in enumerate(self.classes):
            for path in sorted((Path(root) / class_name).rglob("*.pt")):
                self.samples.append((path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        return torch.load(path), label

dataset = TensorFolderDataset("dataset")
loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)

for data, labels in loader:
    print(data.shape, labels)
```

Rust:

```rust
use rtorch::{DataLoader, DataLoaderConfig, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;
    println!("classes: {:?}", dataset.class_names()?);

    let mut loader = DataLoader::with_config(
        &dataset,
        None,
        DataLoaderConfig {
            batch_size: 8,
            shuffle: true,
            drop_last: true,
        },
    )?;

    for batch in &mut loader {
        let batch = batch?;
        println!("{:?} {:?}", batch.shape(), batch.labels());
    }

    Ok(())
}
```

### Direct HDF5 File

Python:

```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class H5Structures(Dataset):
    def __init__(self, path, key):
        self.path = path
        self.key = key
        with h5py.File(path, "r") as f:
            self.spans = [(int(s["offset"]), int(s["length"])) for s in f["structures"]]

    def __len__(self):
        return len(self.spans)

    def __getitem__(self, index):
        offset, length = self.spans[index]
        with h5py.File(self.path, "r") as f:
            values = f[self.key][offset:offset + length]
        return torch.as_tensor(values, dtype=torch.float32), 0

dataset = H5Structures("test_data/train.h5", "positions")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

for positions, labels in loader:
    print(positions.shape, labels)
```

Rust:

```rust
use rtorch::{DataLoader, DataLoaderConfig, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_hdf5_file("test_data/train.h5", "positions")?;
    let sample = dataset.get(0)?;

    println!("sample shape: {:?}", sample.shape());
    println!("sample label: {}", sample.label());

    let mut loader = DataLoader::with_config(
        &dataset,
        None,
        DataLoaderConfig {
            batch_size: 1,
            shuffle: false,
            drop_last: false,
        },
    )?;

    for batch in &mut loader {
        let batch = batch?;
        println!("{:?} {:?}", batch.shape(), batch.labels());
    }

    Ok(())
}
```

## API Reference

### Dataset API

| Rust | Python/PyTorch equivalent | Meaning |
| --- | --- | --- |
| `TensorDataset::from_class_folders(root)` | custom folder `Dataset(root)` | Scan class folders and index tensor files |
| `TensorDataset::from_hdf5_file(path, key)` | custom `h5py` dataset | Index one HDF5 file by dataset key |
| `dataset.len()` | `len(dataset)` | Number of samples |
| `dataset.is_empty()` | `len(dataset) == 0` | Empty check |
| `dataset.class_names()` | `dataset.classes` | Class names in label order |
| `dataset.get(index)` | `dataset[index]` | Load one sample |
| `dataset.get_with_transform(index, transform)` | `transform(dataset[index])` | Load one transformed sample |

### Sample And Batch API

| Rust | Python/PyTorch equivalent | Meaning |
| --- | --- | --- |
| `sample.data()?` | `sample_tensor.float().cpu().flatten()` | Lazily materialized flat `&[f32]` data |
| `sample.shape()` | `sample_tensor.shape` | Sample tensor shape |
| `sample.label()` | label from `dataset[index]` | Sample label |
| `sample.tensor()` | sample tensor | Owned LibTorch-backed tensor |
| `batch.data()?` | `batch_tensor.float().cpu().flatten()` | Lazily materialized flat `&[f32]` data |
| `batch.shape()` | `batch_tensor.shape` | Batch tensor shape |
| `batch.labels()` | batch labels tensor/list | Labels for each sample |
| `batch.tensor()` | batch tensor | Owned LibTorch-backed tensor |
| `batch.statistics()` | `batch.mean()`, `batch.std()`, etc. | Aggregate statistics |

### DataLoader API

| Rust | Python/PyTorch equivalent | Meaning |
| --- | --- | --- |
| `DataLoader::new(dataset, transforms, batch_size, shuffle)` | `DataLoader(dataset, batch_size, shuffle)` | Basic loader |
| `DataLoader::with_config(...)` | `DataLoader(..., drop_last=...)` | Loader with config |
| `DataLoaderConfig::batch_size` | `batch_size` | Samples per batch |
| `DataLoaderConfig::shuffle` | `shuffle` | Shuffle order on reset/start |
| `DataLoaderConfig::drop_last` | `drop_last` | Drop incomplete final batch |
| `loader.len()` | `len(loader)` | Number of batches |
| `loader.reset()` | new epoch / recreate iterator | Reset cursor and reshuffle if enabled |
| `loader.next_batch()` | `next(iter(loader))` | Pull one batch |
| `for batch in &mut loader` | `for batch in loader` | Iterate batches |

### Transform API

| Rust | Python/PyTorch equivalent | Meaning |
| --- | --- | --- |
| `TransformConfig::to_float` | `x.float()` | Convert tensor to float32 |
| `TransformConfig::resize` | `F.interpolate(...)` | Resize `C x H x W` tensor |
| `TransformConfig::center_crop` | center crop transform | Crop spatial center |
| `TransformConfig::normalize_mean/std` | `(x - mean) / std` | Channel-wise normalization |
| `TransformPipeline::new(config)` | `Compose([...])` | Create transform pipeline |

## Supported Scalar Types

These types are supported by `statistics(&[T])` and tensor construction where
applicable.

| Rust type | LibTorch dtype |
| --- | --- |
| `f16` | `torch.float16` |
| `bf16` | `torch.bfloat16` |
| `f32` | `torch.float32` |
| `f64` | `torch.float64` |
| `i8` | `torch.int8` |
| `i16` | `torch.int16` |
| `i32` | `torch.int32` |
| `i64` | `torch.int64` |
| `u8` | `torch.uint8` |
| `u16` | `torch.uint16` when supported by the detected LibTorch |
| `u32` | `torch.uint32` when supported by the detected LibTorch |
| `u64` | `torch.uint64` when supported by the detected LibTorch |
| `isize` | platform-width signed integer dtype |
| `usize` | platform-width unsigned integer dtype when supported |
| `bool` | `torch.bool` |

All fields in `Statistics` are returned as `f64`.

Example:

```rust
use rtorch::statistics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let stats = statistics(&[1.0_f32, 2.0, 3.0, 4.0])?;
    println!("mean={} std={}", stats.mean, stats.population_std);
    Ok(())
}
```

## Tensor API

`Tensor` is an owned LibTorch tensor handle. It supports:

| Rust | Python/PyTorch equivalent |
| --- | --- |
| `Tensor::from_slice(values, shape, requires_grad)` | `torch.tensor(..., requires_grad=...)` |
| `Tensor::zeros(shape, dtype, requires_grad)` | `torch.zeros(...)` |
| `Tensor::ones(shape, dtype, requires_grad)` | `torch.ones(...)` |
| `tensor.shape()` | `tensor.shape` |
| `tensor.scalar_type()` | `tensor.dtype` |
| `tensor.requires_grad()` | `tensor.requires_grad` |
| `tensor.set_requires_grad(...)` | `tensor.requires_grad_(...)` |
| `tensor.add(&other)` | `tensor + other` |
| `tensor.matmul(&other)` | `tensor.matmul(other)` |
| `tensor.reshape(shape)` | `tensor.reshape(...)` |
| `tensor.sum()` | `tensor.sum()` |
| `tensor.backward()` | `tensor.backward()` |
| `tensor.grad()` | `tensor.grad` |
| `tensor.to_vec_f32()` | `tensor.cpu().float().flatten().tolist()` |

Autograd example:

```rust
use rtorch::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], true)?;
    let doubled = input.add(&input)?;
    let mut total = doubled.sum()?;
    total.backward()?;

    assert_eq!(input.grad()?.to_vec_f32()?, vec![2.0, 2.0, 2.0]);
    Ok(())
}
```

## Architecture

The stack is:

```text
safe Rust API -> rtorch-sys -> C ABI -> C++ LibTorch backend
```

Rust cannot directly call arbitrary LibTorch C++ APIs as if they were Rust
functions. C++ has no stable portable ABI for templates, overloaded methods,
exceptions, and objects such as `torch::Tensor`. RTorch therefore exposes a
small C ABI and wraps it in safe Rust types.

## Error Behavior

Most operations return `Result<T, rtorch::Error>`.

Common failure cases:

- `LIBTORCH` is not set
- runtime linker cannot find LibTorch or HDF5 shared libraries
- dataset path does not exist
- dataset root is not a directory
- no `.pt`, `.h5`, or `.hdf5` files were found
- HDF5 support was not enabled at build time
- requested HDF5 dataset key does not exist
- tensor shapes in a batch are incompatible
- transform shape requirements are not met
- `batch_size` is zero


