use std::path::PathBuf;

// Guardrail: GPU cache eval/grad hot-path code must not perform device->host reads.
// Allowed exceptions (off hot path, run at most once per compilation):
//   - `into_handle()` — lookup-time slot index caching
//   - `build_artifact_from_device()` — cold-path D→H copy for disk cache serialization
//   - `store_from_xgcf()` — cold-path D→H copy for actual node/edge counts
#[test]
fn no_device_to_host_reads_in_gpu_cache() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("gpu_cache.rs");

    let text = std::fs::read_to_string(&path).expect("read source");

    // Strip lines inside allowed methods (off hot path).
    let allowed_methods = &["fn into_handle(", "fn build_artifact_from_device(", "fn store_from_xgcf("];
    let filtered: String = {
        let mut inside_allowed = false;
        let mut brace_depth: i32 = 0;
        let mut lines = Vec::new();
        for line in text.lines() {
            if !inside_allowed && allowed_methods.iter().any(|m| line.contains(m)) {
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
    };

    assert!(
        !filtered.contains("dtoh_sync_copy_into"),
        "GPU hot path must not call dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !filtered.contains("copy_to_host"),
        "GPU hot path must not call copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !filtered.contains("dtoh"),
        "GPU hot path must not reference dtoh transfers (found in {})",
        path.display()
    );
}
