use std::collections::HashSet;
use std::fs;
use std::path::PathBuf;

use cudarc::nvrtc::Ptx;
use xlog_cuda::{provider::kernel_paths::KernelArtifactLocator, CudaDevice};

fn extract_ptx_directive(ptx: &str, directive: &str) -> Option<String> {
    for line in ptx.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix(directive) {
            let rest = rest.trim();
            if rest.is_empty() {
                continue;
            }
            return rest.split_whitespace().next().map(|s| s.to_string());
        }
    }
    None
}

fn extract_entry_names(ptx: &str) -> Vec<String> {
    let mut entries = Vec::new();
    for line in ptx.lines() {
        let line = line.trim();
        let line = if let Some(rest) = line.strip_prefix(".visible .entry ") {
            rest
        } else if let Some(rest) = line.strip_prefix(".entry ") {
            rest
        } else {
            continue;
        };

        if let Some((name, _)) = line.split_once('(') {
            let name = name.trim();
            if !name.is_empty() {
                entries.push(name.to_string());
            }
        }
    }
    entries
}

fn kernel_locator() -> KernelArtifactLocator {
    KernelArtifactLocator::new(
        std::env::var_os("XLOG_CUBIN_DIR").map(PathBuf::from),
        None,
        Some(PathBuf::from(env!("OUT_DIR"))),
    )
}

#[test]
fn validate_all_ptx_files_load_and_resolve_all_entry_points() {
    let device = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Skipping PTX validation: CUDA runtime unavailable: {}", e);
            return;
        }
    };

    let locator = kernel_locator();

    for spec in xlog_cuda::kernel_manifest_data::KERNEL_MODULES {
        let (path, is_cubin) = locator
            .resolve_module_path(spec.cu_name, 999)
            .unwrap_or_else(|| {
                panic!(
                    "{}: no portable PTX found in XLOG_CUBIN_DIR or the test build output",
                    spec.cu_name
                )
            });
        assert!(
            !is_cubin,
            "{}: expected portable PTX fallback when using dummy cc",
            spec.cu_name
        );

        let filename = path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("<unknown>");
        let module_name = format!("validate_{}", spec.cu_name);

        let ptx = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("Failed to read {}: {}", path.display(), e));

        let version = extract_ptx_directive(&ptx, ".version")
            .unwrap_or_else(|| panic!("{}: missing .version directive", filename));
        let target = extract_ptx_directive(&ptx, ".target")
            .unwrap_or_else(|| panic!("{}: missing .target directive", filename));
        let address_size = extract_ptx_directive(&ptx, ".address_size")
            .unwrap_or_else(|| panic!("{}: missing .address_size directive", filename));

        assert_eq!(
            address_size, "64",
            "{}: expected .address_size 64, got {}",
            filename, address_size
        );

        let entries = extract_entry_names(&ptx);
        assert!(
            !entries.is_empty(),
            "{}: no .entry kernels found (ptx version {}, target {})",
            filename,
            version,
            target
        );

        for declared_kernel in spec.kernels {
            assert!(
                entries.iter().any(|entry| entry == declared_kernel),
                "{}: manifest declares missing PTX entry {}",
                filename,
                declared_kernel
            );
        }

        let mut seen = HashSet::new();
        for e in &entries {
            assert!(
                seen.insert(e.as_str()),
                "{}: duplicate .entry name {}",
                filename,
                e
            );
        }

        // cudarc::load_ptx requires function names with 'static lifetime.
        // Leak small strings here since this is a one-shot test process.
        let entry_refs: Vec<&'static str> = entries
            .iter()
            .cloned()
            .map(|s| Box::leak(s.into_boxed_str()) as &'static str)
            .collect();

        device
            .inner()
            .load_ptx(Ptx::from_src(&ptx), &module_name, &entry_refs)
            .unwrap_or_else(|e| {
                panic!(
                    "{}: failed to load PTX (version {}, target {}, entries {}): {}",
                    filename,
                    version,
                    target,
                    entries.len(),
                    e
                )
            });

        for entry in &entries {
            assert!(
                device.inner().get_func(&module_name, entry).is_some(),
                "{}: loaded module but could not resolve function {}",
                filename,
                entry
            );
        }
    }
}
