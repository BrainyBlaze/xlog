//! Build script for xlog-cuda: compiles CUDA kernels to PTX

use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels");

    // List of CUDA kernels to compile
    let kernels = [
        "join",
        "dedup",
        "groupby",
        "scan",
        "sort",
        "filter",
        "pack",
        "set_ops",
        "circuit",
        "mc_sample",
        "arith",
        "sat",
        "d4",
        "neural",
    ];

    for kernel in &kernels {
        let cu_path = kernels_dir.join(format!("{}.cu", kernel));
        let ptx_path = kernels_dir.join(format!("{}.ptx", kernel));

        // Only recompile if the .cu file is newer than the .ptx file
        let needs_compile = if ptx_path.exists() {
            let cu_meta = std::fs::metadata(&cu_path).ok();
            let ptx_meta = std::fs::metadata(&ptx_path).ok();
            match (cu_meta, ptx_meta) {
                (Some(cu), Some(ptx)) => cu.modified().ok() > ptx.modified().ok(),
                _ => true,
            }
        } else {
            true
        };

        if needs_compile {
            println!("cargo:warning=Compiling {} to PTX", kernel);

            let status = Command::new("nvcc")
                .args([
                    "-ptx",
                    "-arch=sm_70",
                    "-Wno-deprecated-gpu-targets",
                    "-O3",
                    "-o",
                    ptx_path.to_str().unwrap(),
                    cu_path.to_str().unwrap(),
                ])
                .status()
                .expect("Failed to execute nvcc. Is CUDA toolkit installed?");

            if !status.success() {
                panic!("nvcc failed to compile {}.cu", kernel);
            }
        }

        // Tell Cargo to rerun this script if the .cu file changes
        println!("cargo:rerun-if-changed={}", cu_path.display());
    }
}
