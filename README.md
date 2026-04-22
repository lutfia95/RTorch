# pytorch_loader

Rust bindings for a narrow LibTorch data-loading and preprocessing surface.

This first scaffold focuses on:

- loading tensor samples stored as `.pt` files from class-labeled folders
- batching and iterating them from Rust
- applying common tensor preprocessing in C++/LibTorch:
  - `to_float`
  - `resize`
  - `center_crop`
  - `normalize`

## Current shape

The binding architecture is:

`Rust safe API -> pytorch-loader-sys -> C ABI -> C++ LibTorch shim`

The C++ shim intentionally hides templates, exceptions, and ownership from Rust.

## Dataset format

The initial dataset loader expects:

```text
dataset/
  class_a/
    sample1.pt
    sample2.pt
  class_b/
    sample3.pt
```

Each sample file should contain a single tensor, typically shaped `C x H x W`.

## Example

```rust
use pytorch_loader::{DataLoader, TensorDataset, TransformConfig, TransformPipeline};

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
        println!("shape={:?} labels={:?}", batch.shape, batch.labels);
        println!("elements={}", batch.data.len());
    }

    Ok(())
}
```

## Build prerequisites

- Rust toolchain with Cargo
- CMake
- Clang or another C++17 compiler
- LibTorch installed locally

Set:

```bash
export LIBTORCH=/path/to/libtorch
cargo build
```

`LIBTORCH` should point at the LibTorch root containing `include/` and `lib/`.

## Notes

- This does not yet bind Python's `torch.utils.data.DataLoader`.
- This also does not yet decode image files like JPEG or PNG.
- The preprocessing path is tensor-based today; image decoding can be added behind a second backend.
