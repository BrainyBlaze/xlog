use std::fs;
use std::path::Path;

// Guardrail: compile path code must not perform device->host reads outside of
// allowed cold-path functions. The `compile_gpu_d4_and_verify_cached` function
// and disk cache helpers contain intentional D→H transfers that run once per
// compilation, not per query.
#[test]
fn no_device_to_host_reads_in_cache_compile_path() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // gpu_cache.rs is checked by no_dtoh_in_gpu_cache.rs with its own allowlist.
    // Here we check mod.rs with cold-path allowlist.
    let mod_path = Path::new(manifest_dir)
        .join("src")
        .join("compilation")
        .join("mod.rs");
    let text = fs::read_to_string(&mod_path).expect("read source");

    // Strip bodies of allowed cold-path functions (run once per compilation, not per query).
    let allowed_fns = &[
        "fn compile_gpu_d4_and_verify_cached(",
        "fn hash_random_vars(",
        "fn detect_compute_capability(",
        "fn build_artifact_from_device(",
    ];
    let filtered = strip_fn_bodies(&text, allowed_fns);

    assert!(
        !filtered.contains("dtoh"),
        "mod.rs hot path must not reference dtoh transfers (found outside allowed cold-path fns): {}",
        mod_path.display()
    );
    assert!(
        !filtered.contains("copy_to_host"),
        "mod.rs hot path must not use copy_to_host: {}",
        mod_path.display()
    );
}

/// Strip function bodies for `allowed` function signatures, returning only
/// lines outside those bodies for assertion.
fn strip_fn_bodies(text: &str, allowed: &[&str]) -> String {
    let mut inside_allowed = false;
    let mut brace_depth: i32 = 0;
    let mut lines = Vec::new();
    for line in text.lines() {
        if !inside_allowed && allowed.iter().any(|m| line.contains(m)) {
            inside_allowed = true;
            brace_depth = 0;
        }
        if inside_allowed {
            for ch in line.chars() {
                if ch == '{' {
                    brace_depth += 1;
                } else if ch == '}' {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        inside_allowed = false;
                    }
                }
            }
        } else {
            lines.push(line);
        }
    }
    lines.join("\n")
}
