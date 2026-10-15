use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let (libtorch_path, detected_abi) = discover_libtorch();

    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=HDF5_ROOT");
    println!("cargo:rerun-if-env-changed=HDF5_DIR");
    println!("cargo:rerun-if-env-changed=EQUILOAD_CPU_BRIDGE");
    println!("cargo:rerun-if-env-changed=EQUILOAD_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=PYO3_PYTHON");
    println!("cargo:rerun-if-env-changed=PYTHON_SYS_EXECUTABLE");
    println!("cargo:rerun-if-env-changed=VIRTUAL_ENV");
    println!("cargo:rerun-if-changed=ffi");

    let use_cpu_bridge = env::var("EQUILOAD_CPU_BRIDGE").as_deref() != Ok("0");
    let cxx11_abi = env::var("EQUILOAD_CXX11_ABI").ok().or(detected_abi);

    let mut cmake = cmake::Config::new("ffi");
    cmake
        .define("CMAKE_PREFIX_PATH", &libtorch_path)
        .define("Torch_DIR", libtorch_path.join("share/cmake/Torch"))
        .define("CMAKE_CXX_STANDARD", "20")
        .define(
            "EQUILOAD_CPU_BRIDGE",
            if use_cpu_bridge { "ON" } else { "OFF" },
        )
        .env("LIBTORCH", &libtorch_path)
        .env(
            "EQUILOAD_CPU_BRIDGE",
            if use_cpu_bridge { "1" } else { "0" },
        );
    if let Some(abi) = &cxx11_abi {
        cmake.env("EQUILOAD_CXX11_ABI", abi);
    }

    let hdf5_enabled = env::var_os("CARGO_FEATURE_HDF5").is_some();
    cmake.define(
        "EQUILOAD_ENABLE_HDF5",
        if hdf5_enabled { "ON" } else { "OFF" },
    );
    if let Ok(hdf5_root) = env::var("HDF5_ROOT") {
        cmake.define("HDF5_ROOT", &hdf5_root);
        let root = PathBuf::from(&hdf5_root);
        let library = root.join("lib/libhdf5.so");
        if library.is_file() && root.join("include/hdf5.h").is_file() {
            cmake.define("HDF5_hdf5_LIBRARY_RELEASE", library);
            cmake.define("HDF5_C_INCLUDE_DIR", root.join("include"));
        }
    }
    if let Ok(hdf5_dir) = env::var("HDF5_DIR") {
        cmake.define("HDF5_DIR", hdf5_dir);
    }

    let dst = cmake.build();

    println!(
        "cargo:rustc-link-search=native={}",
        dst.join("lib").display()
    );
    println!("cargo:rustc-link-lib=static=equiload_ffi");
    println!(
        "cargo:rustc-link-search=native={}",
        libtorch_path.join("lib").display()
    );
    println!("cargo:rustc-link-lib=dylib=torch");
    println!("cargo:rustc-link-lib=dylib=torch_cpu");
    println!("cargo:rustc-link-lib=dylib=c10");
    if hdf5_enabled {
        emit_hdf5_link_flags(dst.join("lib").join("equiload-hdf5-link-libs.txt"));
    }

    match env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => {
            println!("cargo:rustc-link-lib=dylib=stdc++");
            println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN/../torch/lib");
        }
        Ok("macos") => {
            println!("cargo:rustc-link-lib=dylib=c++");
            println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path/../torch/lib");
        }
        _ => {}
    }
}

fn discover_libtorch() -> (PathBuf, Option<String>) {
    if let Ok(libtorch) = env::var("LIBTORCH") {
        let abi = env::var("EQUILOAD_CXX11_ABI")
            .ok()
            .or_else(|| discover_python_torch().map(|(_, abi)| abi));
        return (PathBuf::from(libtorch), abi);
    }

    discover_python_torch()
        .map(|(path, abi)| (path, Some(abi)))
        .unwrap_or_else(|| {
            panic!(
                "could not find LibTorch: install Python package 'torch' in the build environment or set LIBTORCH explicitly"
            )
        })
}

fn discover_python_torch() -> Option<(PathBuf, String)> {
    for python in python_candidates() {
        let output = Command::new(&python)
            .args([
                "-c",
                "import pathlib, torch; root = pathlib.Path(torch.__file__).resolve().parent; print(root); print(int(torch._C._GLIBCXX_USE_CXX11_ABI))",
            ])
            .output();
        let Ok(output) = output else {
            continue;
        };
        if !output.status.success() {
            continue;
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let mut lines = stdout.lines();
        let Some(root) = lines.next() else {
            continue;
        };
        let Some(abi) = lines.next() else {
            continue;
        };
        let root = PathBuf::from(root);
        if root.join("include").is_dir() && root.join("lib").is_dir() {
            return Some((root, abi.to_owned()));
        }
    }

    None
}

fn python_candidates() -> Vec<PathBuf> {
    let mut candidates = Vec::new();
    for variable in ["PYO3_PYTHON", "PYTHON_SYS_EXECUTABLE"] {
        if let Some(value) = env::var_os(variable) {
            candidates.push(PathBuf::from(value));
        }
    }
    if let Some(venv) = env::var_os("VIRTUAL_ENV") {
        candidates.push(PathBuf::from(venv).join("bin/python"));
    }

    if let Some(manifest_dir) = env::var_os("CARGO_MANIFEST_DIR") {
        let mut directory = PathBuf::from(manifest_dir);
        while directory.pop() {
            candidates.push(directory.join(".venv/bin/python"));
        }
    }

    candidates.extend([PathBuf::from("python3"), PathBuf::from("python")]);
    candidates
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
            }
            if let Some(name) = library_name(&path) {
                println!("cargo:rustc-link-lib=dylib={name}");
            }
        } else {
            println!("cargo:rustc-link-lib=dylib={line}");
        }
    }
}

fn library_name(path: &Path) -> Option<String> {
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
