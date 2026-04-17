//! Batched topology scoring orchestration over request-local indexes.
//!
//! Task 3 (M8 Phase 1) drives the `kernels/ilp_exact.cu` kernels that score
//! all `(left, right)` pairs for all four topologies in a single batched
//! GPU pass, avoiding per-candidate host transfers.
