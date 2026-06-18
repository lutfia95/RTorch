# RTorch

[![RTorch](https://github.com/lutfia95/RTorch/actions/workflows/rtorch.yml/badge.svg?branch=main)](https://github.com/lutfia95/RTorch/actions/workflows/rtorch.yml)

RTorch (`rtorch`) is a Rust-facing data loading and preprocessing crate built on top of LibTorch.

The current goal of this repository is narrow and practical:

- load tensor samples stored as `.pt` files from class-labeled folders
- expose dataset metadata to Rust
- batch and iterate samples from Rust
- apply common tensor preprocessing through LibTorch before batching

RTorch does not start Python or require the Python `torch` package. It does require
LibTorch, which is PyTorch's C++ distribution. It is currently a small Rust API
over a C ABI and C++ LibTorch backend.

## What Is Included

The repository currently includes:

- a safe Rust crate: `crates/rtorch`
- a low-level FFI crate: `crates/rtorch-sys`
- a C++17 LibTorch backend compiled through CMake
- an owned `Tensor` handle that preserves LibTorch tensor and autograd state
- basic tensor construction, inspection, addition, matrix multiplication,
  reshape, sum, backward, and gradient access
- a dataset scanner for class-folder `.pt` tensor datasets
- HDF5 dataset loading for `.h5` and `.hdf5` tensor data when RTorch is built
  with native HDF5 development files available
- a transform pipeline with these operations:
  - `to_float`
  - `resize`
  - `center_crop`
  - `normalize`
- a batch loader with:
  - configurable `batch_size`
  - optional shuffling
  - optional `drop_last`
  - batch count through `len()`
  - Rust `Iterator` support
  - iterative `next_batch()` access
- dataset metadata APIs:
  - `len()`
  - `is_empty()`
  - `class_names()`
- direct dataset sample access with `get(index)`
- LibTorch-computed batch statistics:
  - element count and sum
  - mean
  - population and sample standard deviation
  - minimum and maximum
- standalone Torch-backed statistics over common Rust numeric slices

## What Is Not Included Yet

The crate does not currently include:

- image decoding for JPEG, PNG, WebP, or similar formats
- Python `Dataset` or Python `DataLoader` bindings
- multiprocessing or worker threads
- prefetch queues
- pinned-memory support
- random crop, horizontal flip, color jitter, or augmentation-heavy transforms
- generic custom dataset traits on the Rust side
- direct zero-copy exposure of LibTorch tensors to Rust
- neural-network module, optimizer, scheduler, CUDA, distributed-training, or
  TorchScript model APIs

That means the current project is best understood as a first LibTorch-backed tensor dataset loader for Rust, not a full training input pipeline framework.

## Architecture

The current stack is:

`Rust safe API -> rtorch-sys -> C ABI -> C++ LibTorch shim`

Why it is structured this way:

- Rust gets a small, ergonomic API
- the `-sys` crate owns the raw FFI boundary
- the C ABI hides C++ exceptions, templates, and LibTorch internals
- the C++ layer handles actual tensor loading and preprocessing using LibTorch

This design keeps Rust code simpler and avoids binding the full LibTorch C++ surface directly.

### Why There Is Still A Small Binding

Rust cannot directly call arbitrary LibTorch C++ APIs as if they were Rust
functions. C++ has no stable, portable ABI for templates, overloaded methods,
exceptions, and objects such as `torch::Tensor`.

RTorch therefore uses a deliberately small C ABI. Only the operations exposed by
RTorch need wrapper functions; it does not generate or maintain bindings for all
of LibTorch. Alternatives such as `cxx`, `autocxx`, or an existing crate still
create an interop layer, even if some of it is generated.

## Dataset Format

The loader expects a directory tree like this:

```text
dataset/
  class_a/
    sample1.pt
    sample2.pt
  class_b/
    sample3.pt
    nested/
      sample4.pt
```

Rules:

- every top-level directory under `dataset/` is treated as one class
- class names are taken from the directory names
- class labels are assigned by sorted directory order
- `.pt` files are discovered recursively inside each class directory
- files are processed in sorted path order within each class

Example label mapping:

```text
dataset/
  cats/
  dogs/
```

Produces:

- `cats -> 0`
- `dogs -> 1`

Each `.pt` file should contain a single tensor. The expected common shape is `C x H x W`.

## Build Requirements

You need:

- Rust toolchain with Cargo
- CMake
- a C++17 compiler such as Clang
- a local LibTorch installation
- optional HDF5 development files for `.h5` and `.hdf5` loading

RTorch expects PyTorch's LibTorch C++ distribution. The helper script downloads
the Linux CPU C++ package into `third_party/libtorch` by default:

```bash
scripts/download-libtorch.sh
```

For a CUDA build, choose the matching PyTorch LibTorch variant:

```bash
LIBTORCH_VARIANT=cu128 scripts/download-libtorch.sh
```

Then set the LibTorch path before building:

```bash
export LIBTORCH="$PWD/third_party/libtorch"
export LD_LIBRARY_PATH="$LIBTORCH/lib:${LD_LIBRARY_PATH:-}"
cargo build
```

`LIBTORCH` must point to the LibTorch root directory containing `include/` and `lib/`.

HDF5 support is enabled automatically when CMake can find HDF5. If HDF5 is
installed outside the system search path, pass its root while building.

The helper script installs HDF5 and its `libaec` runtime dependency into
`third_party/hdf5` using conda, mamba, or micromamba:

```bash
scripts/download-hdf5.sh
```

Then build or run with both native library paths available:

```bash
source third_party/hdf5/rtorch-env.sh
cargo build
```

Or set the variables manually:

```bash
export HDF5_ROOT="$PWD/third_party/hdf5"
export LD_LIBRARY_PATH="$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
cargo build
```

With the local LibTorch helper too:

```bash
export LIBTORCH="$PWD/third_party/libtorch"
export HDF5_ROOT="$PWD/third_party/hdf5"
export LD_LIBRARY_PATH="$LIBTORCH/lib:$HDF5_ROOT/lib:${LD_LIBRARY_PATH:-}"
cargo run -p rtorch --example hdf5 -- test_data/train.h5 positions
```

If you see an error such as `libhdf5.so.310: cannot open shared object file`,
the binary was built with HDF5 but the dynamic loader cannot find the HDF5
shared library at runtime. Set `HDF5_ROOT` and `LD_LIBRARY_PATH` as shown above,
or run `source third_party/hdf5/rtorch-env.sh`. Activating the conda prefix alone
does not set `HDF5_ROOT` for this build. You can also install HDF5 through your
system package manager, for example `sudo apt install libhdf5-dev`.

## Add RTorch To A Rust Project

Until RTorch is published on crates.io, add it as a path dependency:

```toml
[dependencies]
rtorch = { path = "/path/to/RTorch/crates/rtorch" }
```

Then build or run with LibTorch available:

```bash
export LIBTORCH=/path/to/libtorch
cargo run
```

On Linux, the LibTorch shared-library directory may also need to be added to
`LD_LIBRARY_PATH`:

```bash
export LD_LIBRARY_PATH="$LIBTORCH/lib:$LD_LIBRARY_PATH"
```

On macOS, use `DYLD_LIBRARY_PATH` when the LibTorch package does not provide an
install-time runtime path:

```bash
export DYLD_LIBRARY_PATH="$LIBTORCH/lib:$DYLD_LIBRARY_PATH"
```

## Usage Examples

### Compute Statistics From Rust Values

This calls LibTorch reductions without creating a dataset:

```rust
use rtorch::statistics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let values = [1.0_f32, 2.0, 3.0, 4.0];
    let stats = statistics(&values)?;

    println!("count: {}", stats.count);
    println!("sum: {}", stats.sum);
    println!("mean: {}", stats.mean);
    println!("population std: {}", stats.population_std);
    println!("sample std: {}", stats.sample_std);
    println!("min: {}", stats.min);
    println!("max: {}", stats.max);

    Ok(())
}
```

The same function accepts different scalar types:

```rust
use rtorch::statistics;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let float32 = statistics(&[1.0_f32, 2.0, 3.0])?;
    let float64 = statistics(&[1.0_f64, 2.0, 3.0])?;
    let signed = statistics(&[-10_i64, 0, 10])?;
    let unsigned = statistics(&[10_u16, 20, 30])?;
    let booleans = statistics(&[true, false, true])?;

    println!("f32 mean: {}", float32.mean);
    println!("f64 mean: {}", float64.mean);
    println!("i64 range: {}..={}", signed.min, signed.max);
    println!("u16 sum: {}", unsigned.sum);
    println!("true fraction: {}", booleans.mean);

    Ok(())
}
```

### Use `f16` And `bf16`

Rust does not provide built-in `f16` or `bf16` primitive types. RTorch re-exports
both types from the `half` crate:

```rust
use rtorch::{bf16, f16, statistics};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let float16_values = [
        f16::from_f32(1.0),
        f16::from_f32(2.0),
        f16::from_f32(3.0),
    ];
    let bfloat16_values = [
        bf16::from_f32(2.0),
        bf16::from_f32(4.0),
        bf16::from_f32(6.0),
    ];

    println!("f16: {:?}", statistics(&float16_values)?);
    println!("bf16: {:?}", statistics(&bfloat16_values)?);

    Ok(())
}
```

### Supported Statistics Types

| Rust type | LibTorch dtype |
| --- | --- |
| `f16` | `torch.float16` |
| `bf16` | `torch.bfloat16` |
| `f32` | `torch.float32` |
| `f64` | `torch.float64` |
| `i8`, `i16`, `i32`, `i64` | Matching signed integer dtype |
| `u8`, `u16`, `u32`, `u64` | Matching unsigned integer dtype |
| `isize`, `usize` | Matching 32-bit or 64-bit integer dtype |
| `bool` | `torch.bool` |

All fields in the returned `Statistics` value are `f64`, regardless of input
type. Integer values larger than the exact range of `f64` may therefore lose
precision in `sum`, `mean`, `min`, or `max`.

Run the repository's multi-type example with:

```bash
cargo run -p rtorch --example statistics
```

These types currently apply to the standalone `statistics(&[T])` API. The
dataset loader converts returned batch storage to `f32`, so `Batch::data()`
remains `&[f32]` and `Batch::statistics()` describes those returned `f32`
values.

### Inspect A Dataset

The dataset root must contain one directory per class:

```rust
use rtorch::TensorDataset;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;

    println!("samples: {}", dataset.len());
    println!("classes: {:?}", dataset.class_names()?);

    Ok(())
}
```

For this directory:

```text
dataset/
  cats/
    cat_001.pt
  dogs/
    dog_001.pt
```

`class_names()` returns `["cats", "dogs"]`, and their labels are `0` and `1`.

Class-folder datasets may contain `.pt`, `.h5`, or `.hdf5` files. For HDF5 files
RTorch looks for a numeric dataset named `tensor`, `data`, or `positions`.

### Load An HDF5 File

An HDF5 file can also be loaded directly by selecting the numeric dataset to
read:

```rust
use rtorch::{DataLoader, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_hdf5_file("test_data/train.h5", "positions")?;
    let sample = dataset.get(0)?;

    println!("sample shape: {:?}", sample.shape());
    println!("sample label: {}", sample.label());

    let mut loader = DataLoader::new(&dataset, None, 1, false)?;

    for batch in &mut loader {
        let batch = batch?;
        println!("shape: {:?}", batch.shape());
        println!("values: {}", batch.data().len());
    }

    Ok(())
}
```

If the HDF5 file contains a `structures` dataset with `offset` and `length`
fields, each structure is exposed as one sample by slicing the selected numeric
dataset along its first dimension. Otherwise, the selected dataset is returned
as one sample. Batching still requires compatible sample shapes, so variable
length HDF5 structures should be loaded with `batch_size = 1` unless the samples
have matching shapes.

Run the HDF5 example with:

```bash
cargo run -p rtorch --example hdf5 -- test_data/train.h5 positions
```

### Load Batches Without Transforms

```rust
use rtorch::{DataLoader, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;
    let mut loader = DataLoader::new(&dataset, None, 16, false)?;

    while let Some(batch) = loader.next_batch()? {
        println!("shape: {:?}", batch.shape());
        println!("labels: {:?}", batch.labels());
        println!("elements: {}", batch.data().len());
    }

    Ok(())
}
```

All tensors in a batch must have compatible shapes because LibTorch stacks them
into one batch tensor.

### Load, Transform, And Analyze Batches

```rust
use rtorch::{DataLoader, TensorDataset, TransformConfig, TransformPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;

    let transforms = TransformPipeline::new(&TransformConfig {
        to_float: true,
        resize: Some((256, 256)),
        center_crop: Some((224, 224)),
        normalize_mean: Some(vec![0.485, 0.456, 0.406]),
        normalize_std: Some(vec![0.229, 0.224, 0.225]),
    })?;

    let mut loader = DataLoader::new(&dataset, Some(&transforms), 8, true)?;

    while let Some(batch) = loader.next_batch()? {
        let stats = batch.statistics()?;

        println!("shape: {:?}", batch.shape());
        println!("labels: {:?}", batch.labels());
        println!("mean: {:.6}", stats.mean);
        println!("population std: {:.6}", stats.population_std);
        println!("range: [{:.6}, {:.6}]", stats.min, stats.max);
    }

    Ok(())
}
```

Batch statistics cover every tensor element in the batch, not the labels.

### Iterate For Multiple Epochs

```rust
use rtorch::{DataLoader, TensorDataset};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;
    let mut loader = DataLoader::new(&dataset, None, 32, true)?;

    for epoch in 0..3 {
        loader.reset()?;

        while let Some(batch) = loader.next_batch()? {
            println!("epoch={epoch} shape={:?}", batch.shape());
        }
    }

    Ok(())
}
```

When shuffling is enabled, `reset()` creates a new shuffled sample order.

### Read Values From A Batch

`Batch::data()` is a flat `f32` slice in row-major tensor order:

```rust
while let Some(batch) = loader.next_batch()? {
    let shape = batch.shape();
    let values = batch.data();

    println!("shape: {shape:?}");
    println!("first value: {:?}", values.first());
}
```

For shape `[8, 3, 224, 224]`, the flat slice contains
`8 * 3 * 224 * 224` values.

## API Overview

The main Rust types are:

- `Tensor`
- `ScalarType`
- `TensorDataset`
- `Sample`
- `TransformConfig`
- `TransformPipeline`
- `DataLoaderConfig`
- `DataLoader`
- `Batch`
- `Statistics`
- `StatisticsElement`
- `statistics(&[T])`

Typical flow:

1. Create a dataset from a root folder.
2. Optionally create a transform pipeline.
3. Create a loader from the dataset and optional transforms.
4. Pull batches in a loop with `next_batch()` or by iterating the loader.

Statistics can also be computed without a dataset:

```rust
let stats = rtorch::statistics(&[1.0, 2.0, 3.0, 4.0])?;
println!("mean={} std={}", stats.mean, stats.population_std);
```

### Owned Tensors And Autograd

`Tensor` keeps its underlying LibTorch tensor alive behind the Rust API:

```rust
use rtorch::Tensor;

let input = Tensor::from_slice(&[1.0_f32, 2.0, 3.0], &[3], true)?;
let doubled = input.add(&input)?;
let mut total = doubled.sum()?;
total.backward()?;

assert_eq!(input.grad()?.to_vec_f32()?, vec![2.0, 2.0, 2.0]);
# Ok::<(), rtorch::Error>(())
```

This is the foundation for broader model and training support. It is not yet a
complete replacement for Python PyTorch.

## Full Example

```rust
use rtorch::{DataLoader, TensorDataset, TransformConfig, TransformPipeline};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dataset = TensorDataset::from_class_folders("dataset")?;

    println!("sample count: {}", dataset.len());
    println!("is empty: {}", dataset.is_empty());
    println!("classes: {:?}", dataset.class_names()?);

    let transforms = TransformPipeline::new(&TransformConfig {
        to_float: true,
        resize: Some((256, 256)),
        center_crop: Some((224, 224)),
        normalize_mean: Some(vec![0.485, 0.456, 0.406]),
        normalize_std: Some(vec![0.229, 0.224, 0.225]),
    })?;

    let mut loader = DataLoader::new(&dataset, Some(&transforms), 8, true)?;

    while let Some(batch) = loader.next_batch()? {
        println!("batch shape: {:?}", batch.shape());
        println!("batch labels: {:?}", batch.labels());
        println!("flat element count: {}", batch.data().len());
        println!("batch statistics: {:?}", batch.statistics()?);
    }

    Ok(())
}
```

## How To Use The Dataset API

Create a dataset:

```rust
let dataset = TensorDataset::from_class_folders("dataset")?;
```

What this does:

- validates that the root exists
- validates that the root is a directory
- scans class subdirectories
- recursively finds `.pt` files
- records class names and integer labels

Useful methods:

```rust
let count = dataset.len();
let empty = dataset.is_empty();
let classes = dataset.class_names()?;
let sample = dataset.get(0)?;
```

What each method means:

- `len()` returns the total number of discovered `.pt` samples across all classes
- `is_empty()` is a convenience check based on `len()`
- `class_names()` returns class names in the same sorted order used for label assignment
- `get(index)` loads one sample and returns its data, shape, label, statistics,
  and owned tensor

## How To Use The Transform Pipeline

The transform pipeline is configured through `TransformConfig` and created with `TransformPipeline::new(...)`.

Example:

```rust
let transforms = TransformPipeline::new(&TransformConfig {
    to_float: true,
    resize: Some((256, 256)),
    center_crop: Some((224, 224)),
    normalize_mean: Some(vec![0.485, 0.456, 0.406]),
    normalize_std: Some(vec![0.229, 0.224, 0.225]),
})?;
```

### `to_float`

```rust
to_float: true
```

Behavior:

- converts input tensors to `float32` if they are not already `float32`

Use it when:

- your stored `.pt` tensors are integer or another numeric type
- you want to normalize or resize consistently in float space

### `resize`

```rust
resize: Some((256, 256))
```

Behavior:

- resizes a `C x H x W` tensor to the target height and width
- uses LibTorch interpolation internally

Important:

- the tensor must be 3D
- resize happens before center crop if both are enabled

### `center_crop`

```rust
center_crop: Some((224, 224))
```

Behavior:

- crops the spatial center of a `C x H x W` tensor

Important:

- the tensor must be 3D
- crop dimensions must not exceed the current tensor height and width

### `normalize_mean` and `normalize_std`

```rust
normalize_mean: Some(vec![0.485, 0.456, 0.406]),
normalize_std: Some(vec![0.229, 0.224, 0.225]),
```

Behavior:

- performs channel-wise normalization
- the mean and std vectors must have the same length
- their length must match the channel count `C`

The operation is effectively:

```text
output = (input - mean) / std
```

per channel.

### Transform Order

The current backend applies transforms in this order:

1. `to_float`
2. `resize`
3. `center_crop`
4. `normalize`

That order matters. For example, normalization happens after resizing and cropping.

## How To Use The DataLoader

Create a loader like this:

```rust
let mut loader = DataLoader::new(&dataset, Some(&transforms), 8, true)?;
```

Arguments:

- `&dataset`: the dataset to read from
- `Some(&transforms)`: optional preprocessing pipeline
- `8`: batch size
- `true`: whether to shuffle sample order on reset/start

If you do not want transforms:

```rust
let mut loader = DataLoader::new(&dataset, None, 8, false)?;
```

For Python-style `drop_last` behavior, use `DataLoaderConfig`:

```rust
let mut loader = DataLoader::with_config(
    &dataset,
    None,
    DataLoaderConfig {
        batch_size: 8,
        shuffle: true,
        drop_last: true,
    },
)?;
```

Useful methods:

```rust
let batches = loader.len();
let empty = loader.is_empty();
```

### Iterating Batches

Call `next_batch()` until it returns `Ok(None)`:

```rust
while let Some(batch) = loader.next_batch()? {
    println!("{:?}", batch.shape());
}
```

`DataLoader` also implements `Iterator<Item = Result<Batch, Error>>`:

```rust
for batch in &mut loader {
    let batch = batch?;
    println!("{:?}", batch.shape());
}
```

Return meaning:

- `Ok(Some(batch))`: a batch was produced
- `Ok(None)`: end of dataset
- `Err(...)`: loading, preprocessing, or backend failure

### Resetting The Loader

To iterate again:

```rust
loader.reset()?;
```

If `shuffle` was enabled, reset reshuffles the sample order.

## How To Use `Batch`

Each `Batch` exposes data, shape, labels, and aggregate statistics:

```rust
let shape = batch.shape();
let labels = batch.labels();
let data = batch.data();
let statistics = batch.statistics()?;
```

What they mean:

- `shape()` returns the batch tensor shape as a Rust slice of `usize`
- `labels()` returns the class label for each sample in the batch
- `data()` returns the flattened tensor data as `&[f32]`
- `statistics()` returns statistics computed by LibTorch across all elements in
  the returned batch

The `Statistics` value contains `count`, `sum`, `mean`, `population_std`,
`sample_std`, `min`, and `max`.

For a batch of image-like tensors, a shape might look like:

```text
[8, 3, 224, 224]
```

Meaning:

- batch size `8`
- channels `3`
- height `224`
- width `224`

`data()` is flat memory, so the caller is responsible for interpreting it according to `shape()`.

## Error Behavior

Operations return Rust `Result` values.

Typical failure cases include:

- dataset path does not exist
- dataset path is not a directory
- no `.pt` files were found
- tensor shape is incompatible with resize, crop, or normalize
- normalize mean/std lengths do not match
- normalize channel count does not match the tensor channel count
- LibTorch fails to load a `.pt` file
- batch size is zero

## Current Performance Characteristics

The current implementation is simple and correct-first.

Important current behavior:

- samples are loaded on demand
- loading is synchronous
- there are no worker threads
- there is no prefetch queue
- data is copied from the stacked LibTorch tensor into owned batch storage before Rust reads it

So this is appropriate for early experiments and API development, but not yet optimized for high-throughput training workloads.

## Repository Layout

```text
Cargo.toml
README.md
crates/
  rtorch/
    src/lib.rs
    examples/statistics.rs
    examples/tensor_folder.rs
  rtorch-sys/
    src/lib.rs
    build.rs
    ffi/
      include/rtorch.h
      src/rtorch.cpp
      CMakeLists.txt
```

Meaning:

- `rtorch`: safe Rust API
- `rtorch-sys`: raw FFI declarations and build integration
- `ffi/include`: C ABI header
- `ffi/src`: C++ LibTorch implementation

## Example Run Flow

A typical run looks like this:

1. You point `TensorDataset` at a dataset root.
2. The backend scans class directories and indexes `.pt` files.
3. You build an optional transform pipeline.
4. You create a `DataLoader`.
5. Each `next_batch()` call:
   loads sample tensors from disk,
   applies transforms,
   stacks the tensors,
   computes aggregate statistics with LibTorch,
   returns batch data, labels, and statistics to Rust.

## Limitations

Be aware of these limitations before building on top of this crate:

- input format is restricted to `.pt` tensor files
- tensors are expected to work with image-like `C x H x W` transforms
- no generic augmentation framework exists yet
- no Rust-native tensor abstraction is exposed
- no training-loop integration helpers are included

## Future Direction

The next logical improvements are:

- broader dataset support
- richer metadata and sample inspection APIs
- worker threads and prefetching
- more transforms
- cleaner tensor ownership across the FFI boundary
- a clearer split between LibTorch-backed mode and any future native Rust mode
