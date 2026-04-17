//! Deterministic per-topology top-K reduction and tie diagnostics.
//!
//! Task 3 (M8 Phase 1) reduces the device-produced score table to the
//! ordered result list using the lexicographic comparator:
//!   1. higher positive coverage
//!   2. lower negative coverage
//!   3. lower left relation index
//!   4. lower right relation index
//!
//! Populates `local_rank`, `next_positives_covered`, `next_negatives_covered`,
//! and `tie_class_size` on each `ScoredCandidate`.
