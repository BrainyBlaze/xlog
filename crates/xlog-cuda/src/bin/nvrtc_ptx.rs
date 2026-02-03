use std::fs;
use std::path::PathBuf;

use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

fn kernels_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
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
    let opts = CompileOptions {
        // NVRTC expects an `sm_XX` architecture string for `--gpu-architecture`.
        arch: Some("sm_70"),
        ..Default::default()
    };

    let ptx =
        compile_ptx_with_opts(&src, opts).unwrap_or_else(|e| panic!("NVRTC compile failed: {e}"));

    fs::write(&ptx_path, ptx.to_src())
        .unwrap_or_else(|e| panic!("Failed to write {}: {}", ptx_path.display(), e));

    println!("wrote {}", ptx_path.display());
}
