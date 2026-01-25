//! GPU-native knowledge compilation.
//!
//! This module is the home of GPU-native compilation + verification utilities.
//!
//! Production correctness requires the GPU CDCL equivalence verifier (see `validation`).

pub mod sparse_matrix;
pub mod validation;

pub use sparse_matrix::GpuCsrCnf;
pub use validation::{check_equivalence_gpu, validate_equivalence_gpu, GpuEquivalenceConfig};
