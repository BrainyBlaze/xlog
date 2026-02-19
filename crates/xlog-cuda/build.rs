//! Build script for xlog-cuda: compiles CUDA kernels to cubin + portable PTX.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

// Import the canonical kernel list from the shared manifest.
include!("src/kernel_manifest_data.rs");

fn find_nvcc() -> PathBuf {
    // Check NVCC_PATH env var first, then fall back to PATH lookup.
    if let Ok(p) = env::var("NVCC_PATH") {
        let path = PathBuf::from(p);
        if path.exists() {
            return path;
        }
    }
    // Verify nvcc is on PATH.
    match Command::new("nvcc").arg("--version").output() {
        Ok(output) if output.status.success() => PathBuf::from("nvcc"),
        _ => {
            panic!(
                "nvcc not found — required to generate portable PTX/cubin build \
                 artifacts. Install the CUDA toolkit and ensure nvcc is on PATH."
            );
        }
    }
}

fn main() {
    let nvcc = find_nvcc();

    let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let kernels_dir = Path::new(&manifest_dir)
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .join("kernels");
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Environment knobs.
    let no_cubin = env::var("XLOG_NO_CUBIN").map(|v| v == "1").unwrap_or(false);
    let cubin_archs: Vec<String> = env::var("XLOG_CUBIN_ARCHS")
        .unwrap_or_else(|_| "sm_120".to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    // Rerun triggers.
    println!("cargo:rerun-if-env-changed=XLOG_NO_CUBIN");
    println!("cargo:rerun-if-env-changed=XLOG_CUBIN_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC_PATH");

    // Emit kernel-dir metadata for packaging scripts that parse build output
    // (e.g. maturin, pyxlog wheel builds) to stage artifacts into the
    // installed layout (<exe_dir>/kernels/).
    // Note: this is NOT propagated as DEP_* to downstream crates (no `links`
    // key in Cargo.toml). Scripts should grep `cargo:kernel-dir=` from
    // `cargo build -vv` output or read OUT_DIR directly.
    println!("cargo:kernel-dir={}", out_dir.display());

    for name in KERNEL_CU_NAMES {
        let cu_path = kernels_dir.join(format!("{name}.cu"));
        println!("cargo:rerun-if-changed={}", cu_path.display());

        // (a) Generate cubin per arch (unless XLOG_NO_CUBIN=1).
        if !no_cubin {
            for arch in &cubin_archs {
                let cubin_path = out_dir.join(format!("{name}.{arch}.cubin"));
                let status = Command::new(&nvcc)
                    .args([
                        "--cubin",
                        &format!("-arch={arch}"),
                        "-O3",
                        "-o",
                        cubin_path.to_str().unwrap(),
                        cu_path.to_str().unwrap(),
                    ])
                    .status()
                    .unwrap_or_else(|e| panic!("failed to run nvcc for {name}.{arch}.cubin: {e}"));

                if !status.success() {
                    panic!("nvcc failed to compile {name}.cu to cubin for {arch}");
                }
            }
        }

        // (b) Always generate portable PTX (sm_75 baseline — lowest arch in CUDA 13+).
        let ptx_path = out_dir.join(format!("{name}.portable.ptx"));
        let status = Command::new(&nvcc)
            .args([
                "--ptx",
                "-arch=sm_75",
                "-O3",
                "-o",
                ptx_path.to_str().unwrap(),
                cu_path.to_str().unwrap(),
            ])
            .status()
            .unwrap_or_else(|e| panic!("failed to run nvcc for {name}.portable.ptx: {e}"));

        if !status.success() {
            panic!("nvcc failed to compile {name}.cu to portable PTX");
        }
    }
}
