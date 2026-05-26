//! Reusable diagnostics for XLOG module boundary audits.

use std::collections::{BTreeMap, BTreeSet};

/// Role assigned to a module in a boundary audit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleRole {
    /// Read-only kernel module whose predicates cannot be mutated by adapters.
    FrozenKernel,
    /// Adapter module that may add facts but not learned/core rules.
    AdapterOnly,
    /// Unrestricted module.
    Regular,
}

/// Candidate provenance kind for leakage diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CandidateSourceKind {
    /// Candidate came from training evidence.
    TrainingEvidence,
    /// Candidate came from a held-out label and must not drive induction.
    HeldOutLabel,
    /// Candidate was generated without label leakage.
    GeneratedCandidate,
}

/// Module manifest used by diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleManifest {
    /// Module name.
    pub name: String,
    /// Boundary role.
    pub role: ModuleRole,
    /// Predicates owned by this module.
    pub predicates: BTreeSet<String>,
    /// Whether this module represents held-out data.
    pub held_out: bool,
}

impl ModuleManifest {
    /// Construct a module manifest.
    pub fn new<I, S>(name: impl Into<String>, role: ModuleRole, predicates: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            name: name.into(),
            role,
            predicates: predicates.into_iter().map(Into::into).collect(),
            held_out: false,
        }
    }

    /// Mark whether the module contains held-out data.
    pub fn with_held_out(mut self, held_out: bool) -> Self {
        self.held_out = held_out;
        self
    }
}

/// Kind of module declaration.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleDeclarationKind {
    /// Fact-only declaration.
    Fact,
    /// Rule declaration.
    Rule,
    /// Candidate provenance declaration.
    CandidateSource(CandidateSourceKind),
}

/// Declaration observed during module-boundary analysis.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleDeclaration {
    /// Module containing the declaration.
    pub module: String,
    /// Predicate named by the declaration.
    pub predicate: String,
    /// Declaration kind.
    pub kind: ModuleDeclarationKind,
}

impl ModuleDeclaration {
    /// Fact declaration.
    pub fn fact(module: impl Into<String>, predicate: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            predicate: predicate.into(),
            kind: ModuleDeclarationKind::Fact,
        }
    }

    /// Rule declaration.
    pub fn rule(module: impl Into<String>, predicate: impl Into<String>) -> Self {
        Self {
            module: module.into(),
            predicate: predicate.into(),
            kind: ModuleDeclarationKind::Rule,
        }
    }

    /// Candidate provenance declaration.
    pub fn candidate_source(
        module: impl Into<String>,
        predicate: impl Into<String>,
        source: CandidateSourceKind,
    ) -> Self {
        Self {
            module: module.into(),
            predicate: predicate.into(),
            kind: ModuleDeclarationKind::CandidateSource(source),
        }
    }
}

/// Input to the module boundary diagnostic pass.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleBoundaryInput {
    /// Module manifests.
    pub modules: Vec<ModuleManifest>,
    /// Declarations to audit.
    pub declarations: Vec<ModuleDeclaration>,
}

/// Violation kind emitted by module boundary diagnostics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ModuleViolationKind {
    /// A non-kernel module attempted to define a frozen kernel predicate.
    FrozenKernelMutation,
    /// An adapter-only module declared a rule.
    AdapterRuleDefinition,
    /// Held-out labels were used as candidate provenance.
    HeldOutLeakage,
    /// A declaration referenced an unknown module.
    UnknownModule,
}

/// One module diagnostic violation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ModuleViolation {
    /// Violation kind.
    pub kind: ModuleViolationKind,
    /// Module where the violation occurred.
    pub module: String,
    /// Predicate involved in the violation.
    pub predicate: String,
    /// Human-readable diagnostic detail.
    pub detail: String,
}

/// Module boundary diagnostic report.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ModuleBoundaryReport {
    /// Violations found by the audit.
    pub violations: Vec<ModuleViolation>,
}

impl ModuleBoundaryReport {
    /// Whether the audited boundary passed.
    pub fn passed(&self) -> bool {
        self.violations.is_empty()
    }
}

/// Diagnose frozen-kernel, adapter-only, and held-out leakage boundaries.
pub fn diagnose_module_boundaries(input: ModuleBoundaryInput) -> ModuleBoundaryReport {
    let modules: BTreeMap<String, ModuleManifest> = input
        .modules
        .into_iter()
        .map(|module| (module.name.clone(), module))
        .collect();
    let frozen_predicates: BTreeSet<String> = modules
        .values()
        .filter(|module| module.role == ModuleRole::FrozenKernel)
        .flat_map(|module| module.predicates.iter().cloned())
        .collect();

    let mut violations = Vec::new();
    for declaration in input.declarations {
        let Some(module) = modules.get(&declaration.module) else {
            violations.push(ModuleViolation {
                kind: ModuleViolationKind::UnknownModule,
                module: declaration.module,
                predicate: declaration.predicate,
                detail: "declaration references an unknown module".to_string(),
            });
            continue;
        };

        if module.role != ModuleRole::FrozenKernel
            && frozen_predicates.contains(&declaration.predicate)
        {
            violations.push(ModuleViolation {
                kind: ModuleViolationKind::FrozenKernelMutation,
                module: declaration.module.clone(),
                predicate: declaration.predicate.clone(),
                detail: "adapter attempted to define a frozen kernel predicate".to_string(),
            });
        }

        if module.role == ModuleRole::AdapterOnly
            && matches!(declaration.kind, ModuleDeclarationKind::Rule)
        {
            violations.push(ModuleViolation {
                kind: ModuleViolationKind::AdapterRuleDefinition,
                module: declaration.module.clone(),
                predicate: declaration.predicate.clone(),
                detail: "adapter-only module declared a rule".to_string(),
            });
        }

        if module.held_out
            && matches!(
                declaration.kind,
                ModuleDeclarationKind::CandidateSource(CandidateSourceKind::HeldOutLabel)
            )
        {
            violations.push(ModuleViolation {
                kind: ModuleViolationKind::HeldOutLeakage,
                module: declaration.module,
                predicate: declaration.predicate,
                detail: "held-out label used as candidate provenance".to_string(),
            });
        }
    }

    ModuleBoundaryReport { violations }
}
