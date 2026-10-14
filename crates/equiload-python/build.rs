fn main() {
    // The extension and Python's torch package are siblings in site-packages.
    // Keep this relative so a wheel never records the build machine's path.
    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../torch/lib"),
        Ok("macos") => println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../torch/lib"),
        _ => {}
    }
}
