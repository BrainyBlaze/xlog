use std::path::PathBuf;

// Guardrail: GPU cache eval/grad hot-path code must not perform device->host reads.
// The one allowed exception is `into_handle()`, which runs at lookup time (once per
// compilation) to cache the slot index on the host — this avoids D→H syncs on every
// subsequent eval call.
#[test]
fn no_device_to_host_reads_in_gpu_cache() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("compilation");
    path.push("gpu_cache.rs");

    let text = std::fs::read_to_string(&path).expect("read source");

    // Strip lines inside the into_handle() method (lookup-time, off hot path).
    let filtered: String = {
        let mut inside_into_handle = false;
        let mut brace_depth: i32 = 0;
        let mut lines = Vec::new();
        for line in text.lines() {
            if line.contains("fn into_handle(") {
                inside_into_handle = true;
                brace_depth = 0;
            }
            if inside_into_handle {
                for ch in line.chars() {
                    if ch == '{' {
                        brace_depth += 1;
                    } else if ch == '}' {
                        brace_depth -= 1;
                        if brace_depth == 0 {
                            inside_into_handle = false;
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
