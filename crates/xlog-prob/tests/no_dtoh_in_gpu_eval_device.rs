use std::path::PathBuf;

#[path = "utils/scan.rs"]
mod scan_utils;

// Guardrail: device-only eval APIs must not perform device->host reads.
#[test]
fn no_device_to_host_reads_in_gpu_eval_device() {
    let mut path = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    path.push("src");
    path.push("gpu.rs");

    let text = std::fs::read_to_string(&path).expect("read gpu.rs");

    let fn_names = [
        "eval_log_wmc_device_inplace",
        "eval_log_wmc_device_into",
        "eval_log_wmc_device",
    ];
    for name in fn_names {
        let body = scan_utils::extract_fn_body(&text, name)
            .unwrap_or_else(|| panic!("missing function {}", name));
        assert!(
            !body.contains("dtoh_sync_copy_into"),
            "device-only {} must not call dtoh_sync_copy_into (found in {})",
            name,
            path.display()
        );
        assert!(
            !body.contains("copy_to_host"),
            "device-only {} must not call copy_to_host (found in {})",
            name,
            path.display()
        );
        assert!(
            !body.contains("dtoh"),
            "device-only {} must not reference dtoh transfers (found in {})",
            name,
            path.display()
        );
    }
}
