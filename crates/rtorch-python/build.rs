use std::{env, path::PathBuf};

fn main() {
    // Link arguments on an rlib do not propagate to its final extension-module
    // consumer. Put runtime paths on the shared object itself.
    for key in ["LIBTORCH", "HDF5_ROOT"] {
        println!("cargo:rerun-if-env-changed={key}");
        if let Ok(root) = env::var(key) {
            if matches!(
                env::var("CARGO_CFG_TARGET_OS").as_deref(),
                Ok("linux" | "macos")
            ) {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath,{}",
                    PathBuf::from(root).join("lib").display()
                );
            }
        }
    }
}
