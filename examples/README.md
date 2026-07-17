# RTorch Examples

This folder shows Python/PyTorch-style dataset code beside equivalent RTorch
Rust code.

Runnable Rust examples live in `crates/rtorch/examples/` and can be launched with
Cargo:

```bash
cargo run -p rtorch --example complete_example
cargo run -p rtorch --example complete_example -- path/to/sample.pt
cargo run -p rtorch --example statistics
cargo run -p rtorch --features hdf5 --example hdf5 -- test_data/train.h5 positions
cargo run -p rtorch --example tensor_folder
```

The HDF5 example requires the `hdf5` Cargo feature and an HDF5 development
installation. Class-folder `.pt` examples accept files containing one tensor
written by either Python `torch.save` or LibTorch C++ serialization.
The complete Rust walkthrough is
[`crates/rtorch/examples/complete_example.rs`](../crates/rtorch/examples/complete_example.rs).

The files here are short reference snippets:

| Use case | Python | Rust |
| --- | --- | --- |
| Class-folder `.pt` loader | `python/pt_class_folder_loader.py` | `rust/pt_class_folder_loader.rs` |
| HDF5 structure loader | `python/hdf5_loader.py` | `rust/hdf5_loader.rs` |
