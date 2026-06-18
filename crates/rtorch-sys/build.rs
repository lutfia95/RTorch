use std::env;
use std::fs;
use std::path::PathBuf;

fn main() {
    let libtorch =
        env::var("LIBTORCH").expect("LIBTORCH must point to a local LibTorch installation");
    let libtorch_path = PathBuf::from(&libtorch);

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=HDF5_ROOT");
    println!("cargo:rerun-if-env-changed=HDF5_DIR");
    println!("cargo:rerun-if-changed=ffi");

    let mut cmake = cmake::Config::new("ffi");
    cmake
        .define("CMAKE_PREFIX_PATH", &libtorch_path)
        .define("CMAKE_CXX_STANDARD", "17");
    if let Ok(hdf5_root) = env::var("HDF5_ROOT") {
        cmake.define("HDF5_ROOT", hdf5_root);
    }
    if let Ok(hdf5_dir) = env::var("HDF5_DIR") {
        cmake.define("HDF5_DIR", hdf5_dir);
    }

    let dst = cmake.build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=rtorch_ffi");
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    emit_hdf5_link_flags(dst.join("lib").join("rtorch-hdf5-link-libs.txt"));

    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("macos") => println!("cargo:rustc-link-lib=dylib=c++"),
        Ok("linux") => println!("cargo:rustc-link-lib=dylib=stdc++"),
        _ => {}
    }
}

fn emit_hdf5_link_flags(link_file: PathBuf) {
    let Ok(contents) = fs::read_to_string(link_file) else {
        return;
    };

    for line in contents
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let path = PathBuf::from(line);
        if path.is_absolute() {
            if let Some(parent) = path.parent() {
                println!("cargo:rustc-link-search=native={}", parent.display());
                emit_runtime_search_path(parent);
            }
            if let Some(name) = library_name(&path) {
                println!("cargo:rustc-link-lib=dylib={name}");
            }
        } else {
            println!("cargo:rustc-link-lib=dylib={line}");
        }
    }
}

fn library_name(path: &std::path::Path) -> Option<String> {
    let file_name = path.file_name()?.to_str()?;
    let name = file_name.strip_prefix("lib").unwrap_or(file_name);
    if let Some((library, _)) = name.split_once(".so.") {
        return Some(library.to_string());
    }
    for suffix in [".so", ".dylib", ".a"] {
        if let Some(stripped) = name.strip_suffix(suffix) {
            return Some(stripped.to_string());
        }
    }

    Some(name.to_string())
}

fn emit_runtime_search_path(path: &std::path::Path) {
    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") | Ok("macos") => {
            println!("cargo:rustc-link-arg=-Wl,-rpath,{}", path.display());
        }
        _ => {}
    }
}
