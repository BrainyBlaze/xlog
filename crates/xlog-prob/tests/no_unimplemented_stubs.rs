use std::path::{Path, PathBuf};

fn visit_rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for ent in entries.flatten() {
        let path = ent.path();
        if path.is_dir() {
            visit_rs_files(&path, out);
            continue;
        }
        if path.extension().and_then(|e| e.to_str()) == Some("rs") {
            out.push(path);
        }
    }
}

#[test]
fn compilation_module_contains_no_unimplemented_stubs() {
    let base = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src/compilation");

    let mut files = Vec::new();
    visit_rs_files(&base, &mut files);

    // If the compilation module directory disappears entirely, that's fine for this check.
    for path in files {
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));
        assert!(
            !text.contains("unimplemented!("),
            "Found unimplemented! stub in {}",
            path.display()
        );
    }
}
