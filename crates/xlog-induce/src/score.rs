//! Batched topology scoring orchestration over request-local indexes.
//!
//! The native bounded exact-induction path drives the `kernels/ilp_exact.cu`
//! kernels that score all `(left, right)` pairs for all four topologies in a
//! single batched GPU pass, avoiding per-candidate host transfers.
