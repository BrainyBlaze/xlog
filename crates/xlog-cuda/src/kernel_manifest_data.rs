// Single source of truth for CUDA kernel modules.
//
// This file is consumed by both `build.rs` (via include!()) and `provider.rs`
// (via mod) to avoid the kernel list being duplicated in two places.
// NOTE: Use regular comments (//), NOT inner doc comments (//!), because
// include!() in build.rs would interpret //! as documenting the wrong item.

/// Module names matching the .cu filenames (without extension).
/// Order matches provider.rs load order. All 21 modules listed.
pub const KERNEL_CU_NAMES: &[&str] = &[
    "join",
    "dedup",
    "groupby",
    "scan",
    "sort",
    "filter",
    "set_ops",
    "pack",
    "pir",
    "cnf",
    "cache",
    "weights",
    "circuit",
    "mc_sample",
    "mc_eval",
    "arith",
    "sat",
    "d4",
    "neural",
    "ilp",
    "ilp_credit",
];
