# EquiLoad Examples

This folder contains the runnable EquiLoad examples and short Python/PyTorch-style
dataset snippets beside equivalent EquiLoad Rust code.

Runnable Rust examples in this folder can be launched with Cargo:

```bash
cargo run -p equiload --example complete_example
cargo run -p equiload --example complete_example -- path/to/sample.pt
cargo run -p equiload --example statistics
cargo run -p equiload --features hdf5 --example hdf5 -- test_data/train.h5 positions
cargo run -p equiload --example tensor_folder
```

The HDF5 example requires the `hdf5` Cargo feature and an HDF5 development
installation. Class-folder `.pt` examples accept tensor payloads written by
Python `torch.save` or LibTorch C++ serialization.
The complete Rust walkthrough is [`complete_example.rs`](complete_example.rs).

The files here are short reference snippets:

| Use case | Python | Rust |
| --- | --- | --- |
| Class-folder `.pt` loader | `python/pt_class_folder_loader.py` | `rust/pt_class_folder_loader.rs` |
| HDF5 structure loader | `python/hdf5_loader.py` | `rust/hdf5_loader.rs` |
