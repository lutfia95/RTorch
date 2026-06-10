# RTorch

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
- a dataset scanner for class-folder `.pt` tensor datasets
- a transform pipeline with these operations:
  - `to_float`
  - `resize`
  - `center_crop`
  - `normalize`
- a batch loader with:
  - configurable `batch_size`
  - optional shuffling
  - iterative `next_batch()` access
- dataset metadata APIs:
  - `len()`
  - `is_empty()`
  - `class_names()`
- LibTorch-computed batch statistics:
  - element count and sum
  - mean
  - population and sample standard deviation
  - minimum and maximum
- standalone Torch-backed statistics over Rust `f32` slices

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

Set the LibTorch path before building:

```bash
export LIBTORCH=/path/to/libtorch
cargo build
```

`LIBTORCH` must point to the LibTorch root directory containing `include/` and `lib/`.

## API Overview

The main Rust types are:

- `TensorDataset`
- `TransformConfig`
- `TransformPipeline`
- `DataLoader`
- `Batch`
- `Statistics`
- `statistics(&[f32])`

Typical flow:

1. Create a dataset from a root folder.
2. Optionally create a transform pipeline.
3. Create a loader from the dataset and optional transforms.
4. Pull batches in a loop with `next_batch()`.

Statistics can also be computed without a dataset:

```rust
let stats = rtorch::statistics(&[1.0, 2.0, 3.0, 4.0])?;
println!("mean={} std={}", stats.mean, stats.population_std);
```

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
```

What each method means:

- `len()` returns the total number of discovered `.pt` samples across all classes
- `is_empty()` is a convenience check based on `len()`
- `class_names()` returns class names in the same sorted order used for label assignment

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

### Iterating Batches

Call `next_batch()` until it returns `Ok(None)`:

```rust
while let Some(batch) = loader.next_batch()? {
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
