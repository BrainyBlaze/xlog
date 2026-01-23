//! GPU-native knowledge compilation.
//!
//! This module implements hybrid GPU compilation paths:
//! - Tensor Path: Sparse matrix operations for small CNFs
//! - GPU D4 Path: BFS-parallelized tree search (future)

pub mod sparse_matrix;
pub mod tensor_path;
pub mod validation;

pub use sparse_matrix::GpuCsrCnf;
pub use tensor_path::compile_tensor_path;
pub use validation::validate_against_cpu_evaluator;
