// crates/xlog-cuda/tests/test_w33_hg_source_audit.rs
//! W3.3 source-audit certs for HG block-slice WCOJ kernels.

use std::fs;
use std::path::PathBuf;

fn wcoj_cu_source() -> String {
    let mut p = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    p.push("kernels/wcoj.cu");
    fs::read_to_string(&p).unwrap_or_else(|e| panic!("failed to read {}: {}", p.display(), e))
}

fn extract_extern_c_global_body(src: &str, name: &str) -> Option<String> {
    let needle = format!("__global__ void {}", name);
    let start = src.find(&needle)?;
    let after_name = start + needle.len();
    let mut depth_paren = 0i32;
    let mut saw_paren = false;
    let mut after_args = None;
    for (offset, b) in src[after_name..].bytes().enumerate() {
        if b == b'(' {
            depth_paren += 1;
            saw_paren = true;
        } else if b == b')' {
            depth_paren -= 1;
            if saw_paren && depth_paren == 0 {
                after_args = Some(after_name + offset + 1);
                break;
            }
        }
    }
    let mut depth = 0i32;
    let mut body_start = None;
    let mut body_end = None;
    let bytes = src.as_bytes();
    let mut i = after_args?;
    while i < bytes.len() {
        if bytes[i] == b'{' {
            if body_start.is_none() {
                body_start = Some(i + 1);
            }
            depth += 1;
        } else if bytes[i] == b'}' {
            depth -= 1;
            if depth == 0 {
                body_end = Some(i);
                break;
            }
        }
        i += 1;
    }
    Some(src[body_start?..body_end?].to_string())
}

#[test]
fn triangle_u32_materialize_uses_hg_block_slice_surface() {
    let src = wcoj_cu_source();
    assert!(
        !src.contains("__global__ void wcoj_triangle_materialize("),
        "u32 triangle materialize must use the W3.3 HG block-slice symbol"
    );

    let materialize = extract_extern_c_global_body(&src, "wcoj_triangle_materialize_hg_cached_u32")
        .expect("wcoj_triangle_materialize_hg_cached_u32 must exist");

    assert!(
        materialize.contains("blockIdx.x * block_work_unit"),
        "u32 triangle materialize must derive its block slice from block_work_unit"
    );
}

#[test]
fn triangle_u64_count_and_materialize_are_hg_block_slice() {
    let src = wcoj_cu_source();
    assert!(
        !src.contains("__global__ void wcoj_triangle_count_u64("),
        "u64 triangle count must use the W3.3 HG block-slice symbol"
    );
    assert!(
        !src.contains("__global__ void wcoj_triangle_materialize_u64("),
        "u64 triangle materialize must use the W3.3 HG block-slice symbol"
    );

    let count = extract_extern_c_global_body(&src, "wcoj_triangle_count_hg_u64")
        .expect("wcoj_triangle_count_hg_u64 must exist");
    let materialize = extract_extern_c_global_body(&src, "wcoj_triangle_materialize_hg_u64")
        .expect("wcoj_triangle_materialize_hg_u64 must exist");

    for (name, body) in [
        ("wcoj_triangle_count_hg_u64", count),
        ("wcoj_triangle_materialize_hg_u64", materialize),
    ] {
        assert!(
            body.contains("blockIdx.x * block_work_unit"),
            "{name} must derive its block slice from block_work_unit"
        );
        assert!(
            body.contains("upper_bound_u32(xy_work_prefix"),
            "{name} must recover the root row from the prefix-summed work plan"
        );
    }
}

#[test]
fn four_cycle_u32_count_and_materialize_are_hg_block_slice() {
    let src = wcoj_cu_source();
    assert!(
        !src.contains("__global__ void wcoj_4cycle_count("),
        "u32 4-cycle count must use the W3.3 HG block-slice symbol"
    );
    assert!(
        !src.contains("__global__ void wcoj_4cycle_materialize("),
        "u32 4-cycle materialize must use the W3.3 HG block-slice symbol"
    );

    let count = extract_extern_c_global_body(&src, "wcoj_4cycle_count_hg_u32")
        .expect("wcoj_4cycle_count_hg_u32 must exist");
    let materialize = extract_extern_c_global_body(&src, "wcoj_4cycle_materialize_hg_u32")
        .expect("wcoj_4cycle_materialize_hg_u32 must exist");

    for (name, body) in [
        ("wcoj_4cycle_count_hg_u32", count),
        ("wcoj_4cycle_materialize_hg_u32", materialize),
    ] {
        assert!(
            body.contains("blockIdx.x * block_work_unit"),
            "{name} must derive its block slice from block_work_unit"
        );
        assert!(
            body.contains("upper_bound_u32(e1_work_prefix"),
            "{name} must recover the root row from the prefix-summed work plan"
        );
    }
}
