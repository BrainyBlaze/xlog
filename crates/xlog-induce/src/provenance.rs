//! Compatibility aliases for generated induction provenance.
//!
//! v0.8.7 introduced `InductionProvenanceRegistry`; v0.8.8 generalized the
//! same surface as `InducedRuleRegistry`. Keep the older name as an alias so
//! Project 1 reproducers and consumers do not lose their validated API.

pub use crate::types::{
    InducedRuleProvenance, InductionAlternative, InductionSupportRow, RuleSourceKind,
};

/// Backwards-compatible registry name for induced rule provenance.
pub type InductionProvenanceRegistry = crate::types::InducedRuleRegistry;
