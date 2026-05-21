use xlog_logic::{
    diagnose_module_boundaries, CandidateSourceKind, ModuleBoundaryInput, ModuleDeclaration,
    ModuleManifest, ModuleRole, ModuleViolationKind,
};

#[test]
fn adapter_only_fact_modules_pass_boundary_diagnostics() {
    let report = diagnose_module_boundaries(ModuleBoundaryInput {
        modules: vec![
            ModuleManifest::new("bfo", ModuleRole::FrozenKernel, ["entity", "root_cause"]),
            ModuleManifest::new("cyber", ModuleRole::AdapterOnly, ["cyber_fact"])
                .with_held_out(false),
        ],
        declarations: vec![
            ModuleDeclaration::fact("cyber", "cyber_fact"),
            ModuleDeclaration::candidate_source(
                "cyber",
                "cyber_fact",
                CandidateSourceKind::TrainingEvidence,
            ),
        ],
    });

    assert!(report.passed());
    assert!(report.violations.is_empty());
}

#[test]
fn diagnostics_fail_when_adapter_mutates_frozen_kernel_or_leaks_held_out_labels() {
    let report = diagnose_module_boundaries(ModuleBoundaryInput {
        modules: vec![
            ModuleManifest::new("bfo", ModuleRole::FrozenKernel, ["entity", "root_cause"]),
            ModuleManifest::new("heldout", ModuleRole::AdapterOnly, ["heldout_fact"])
                .with_held_out(true),
        ],
        declarations: vec![
            ModuleDeclaration::rule("heldout", "root_cause"),
            ModuleDeclaration::candidate_source(
                "heldout",
                "root_cause",
                CandidateSourceKind::HeldOutLabel,
            ),
        ],
    });

    assert!(!report.passed());
    assert!(report
        .violations
        .iter()
        .any(|v| v.kind == ModuleViolationKind::FrozenKernelMutation));
    assert!(report
        .violations
        .iter()
        .any(|v| v.kind == ModuleViolationKind::HeldOutLeakage));
}
