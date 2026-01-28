use std::fs;
use std::path::Path;

#[test]
fn no_device_to_host_reads_in_cache_compile_path() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let paths = [
        Path::new(manifest_dir)
            .join("src")
            .join("compilation")
            .join("mod.rs"),
        Path::new(manifest_dir)
            .join("src")
            .join("compilation")
            .join("gpu_cache.rs"),
    ];
    for path in paths {
        let text = fs::read_to_string(&path).expect("read source");
        assert!(
            !text.contains("dtoh_sync_copy_into"),
            "cache compile path must not use dtoh_sync_copy_into: {}",
            path.display()
        );
        assert!(
            !text.contains("copy_to_host"),
            "cache compile path must not use copy_to_host: {}",
            path.display()
        );
    }
}
