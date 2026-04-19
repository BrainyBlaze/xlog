use std::collections::HashSet;
use std::fs;

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
    KernelArtifactLocator::from_env()
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
                    "{}: no portable PTX found in XLOG_CUBIN_DIR, package kernels/, or OUT_DIR",
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

#[test]
fn validate_kernel_build_flags_suppress_deprecated_gpu_targets() {
    let cmake_path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../kernels/CMakeLists.txt");
    let cmake = fs::read_to_string(&cmake_path)
        .unwrap_or_else(|e| panic!("Failed to read {}: {}", cmake_path.display(), e));
    assert!(
        cmake.contains("-Wno-deprecated-gpu-targets"),
        "kernels/CMakeLists.txt must include -Wno-deprecated-gpu-targets to silence nvcc warnings for sm_70"
    );
}

#[test]
fn validate_arith_ptx_contains_expected_kernels() {
    let ptx_path = kernel_locator()
        .resolve_module_path("arith", 999)
        .map(|(path, _)| path)
        .expect("arith portable PTX should exist in generated or staged artifacts");
    let ptx = fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("arith.ptx should exist after build: {}", e));
    assert!(ptx.contains("arith_binary_i64"));
    assert!(ptx.contains("arith_binary_f64"));
    assert!(ptx.contains("arith_abs_i64"));
    assert!(ptx.contains("arith_cast"));
    assert!(ptx.contains("arith_fill_const_u32"));
}

#[test]
fn validate_neural_ptx_contains_expected_kernels() {
    let ptx_path = kernel_locator()
        .resolve_module_path("neural", 999)
        .map(|(path, _)| path)
        .expect("neural portable PTX should exist in generated or staged artifacts");
    let ptx = fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("neural.ptx should exist after build: {}", e));
    assert!(ptx.contains("neural_fill_ad_chain_f32"));
    assert!(ptx.contains("neural_scatter_ad_chain_grads_f32"));
}

#[test]
fn validate_d4_ptx_contains_expected_kernels() {
    let ptx_path = kernel_locator()
        .resolve_module_path("d4", 999)
        .map(|(path, _)| path)
        .expect("d4 portable PTX should exist in generated or staged artifacts");
    let ptx = fs::read_to_string(&ptx_path)
        .unwrap_or_else(|e| panic!("d4.ptx should exist after build: {}", e));

    // D4 module entrypoints.
    // Phase 1 will add compilation kernels, but we start with invariant validation + levelization.
    assert!(ptx.contains("d4_validate_cnf"));
    assert!(ptx.contains("d4_levelize_counts"));
    assert!(ptx.contains("d4_levelize_emit"));

    // BFS frontier expansion (Task 4).
    assert!(ptx.contains("d4_frontier_prepare"));
    assert!(ptx.contains("d4_frontier_expand"));
    assert!(ptx.contains("d4_frontier_prepare_dense"));
    assert!(ptx.contains("d4_frontier_expand_dense"));

    // GPU-only assertion helpers (used for tests/invariants without device->host reads).
    assert!(ptx.contains("d4_assert_u32_eq"));
    assert!(ptx.contains("d4_assert_bitset_var"));
    assert!(ptx.contains("d4_assert_dense_var"));
}
