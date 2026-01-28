use std::path::Path;

pub fn assert_no_dtoh_in_file(path: &Path) {
    let text = std::fs::read_to_string(path).expect("read source");
    assert!(
        !text.contains("dtoh_sync_copy_into"),
        "GPU path must not call dtoh_sync_copy_into (found in {})",
        path.display()
    );
    assert!(
        !text.contains("copy_to_host"),
        "GPU path must not call copy_to_host (found in {})",
        path.display()
    );
    assert!(
        !text.contains("dtoh"),
        "GPU path must not reference dtoh transfers (found in {})",
        path.display()
    );
}
