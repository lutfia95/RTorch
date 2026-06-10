use std::env;
use std::path::PathBuf;

fn main() {
    let libtorch = env::var("LIBTORCH").expect("LIBTORCH must point to a local LibTorch installation");
    let libtorch_path = PathBuf::from(&libtorch);

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-changed=ffi");

    let dst = cmake::Config::new("ffi")
        .define("CMAKE_PREFIX_PATH", &libtorch_path)
        .define("CMAKE_CXX_STANDARD", "17")
        .build();

    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=rtorch_ffi");
    println!("cargo:rustc-link-search=native={}", libtorch_path.join("lib").display());
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
}
