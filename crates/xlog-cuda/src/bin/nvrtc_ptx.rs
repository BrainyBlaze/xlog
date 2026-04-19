use std::fs;
use std::path::PathBuf;

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

fn kernels_dir() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let packaged_dir = manifest_dir.join("kernels");
    if packaged_dir.is_dir() {
        return packaged_dir;
    }

    manifest_dir
        .parent()
        .expect("crates/")
        .parent()
        .expect("repo root")
        .join("kernels")
}

fn main() {
    let kernel = std::env::args().nth(1).unwrap_or_else(|| "sat".to_string());
    let kernels_dir = kernels_dir();
    let cu_path = kernels_dir.join(format!("{}.cu", kernel));
    let ptx_path = kernels_dir.join(format!("{}.ptx", kernel));

    let src = fs::read_to_string(&cu_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", cu_path.display(), e));

    // Keep options minimal so this works in environments that only have NVRTC available.
    // Default sm_75 is the lowest arch supported by CUDA 13+.
    // Leak is fine — this is a one-shot CLI tool.
    let arch: &'static str = Box::leak(
        std::env::var("XLOG_NVRTC_ARCH")
            .unwrap_or_else(|_| "sm_75".to_string())
            .into_boxed_str(),
    );
    let opts = CompileOptions {
        arch: Some(arch),
        ..Default::default()
    };

    let ptx =
        compile_ptx_with_opts(&src, opts).unwrap_or_else(|e| panic!("NVRTC compile failed: {e}"));

    fs::write(&ptx_path, ptx.to_src())
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", ptx_path.display(), e));

    println!("wrote {}", ptx_path.display());
}
