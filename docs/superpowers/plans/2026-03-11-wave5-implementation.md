# Wave 5: Probabilistic Backend Decomposition + Coherence/Polish — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Decompose the two largest remaining domain backend files (gpu_d4.rs, mc.rs) and complete the final coherence/polish sweep across the workspace.

**Architecture:** Three sub-waves: 5a splits gpu_d4.rs (3,669 → ~3,500 LOC in 3 submodules), 5b splits mc.rs (3,399 → ~2,800 LOC in 5 submodules), 5c applies 10 coherence items across the workspace. Each sub-wave has its own gate.

**Tech Stack:** Rust workspace, CUDA kernels (unchanged), PyO3 (pyxlog, unchanged)

**Spec:** `docs/superpowers/specs/2026-03-10-wave5-prob-backends-coherence-design.md`

---

## Chunk 1: Sub-wave 5a — gpu_d4.rs Decomposition

### Spec Deviation

The spec proposes 4 submodules (mod.rs, frontier.rs, smoothing.rs, build.rs). Code analysis shows that **smoothing does not live in gpu_d4.rs** — it's in `GpuXgcf::smooth_random_vars_device` (called from `compilation/mod.rs:337`). The `compile_gpu_d4_with_gate` function handles: validate → frontier → scratch → count/emit/meta → levelize → build. There is no separate smoothing phase within gpu_d4.rs.

**Adapted split: 3 submodules** (mod.rs, frontier.rs, build.rs). This matches the actual code structure while preserving the spec's intent.

### File Map

```
crates/xlog-prob/src/compilation/
├── gpu_d4.rs (DELETE — 3,669 lines)
├── gpu_d4/
│   ├── mod.rs    (NEW — entry points, config, validation, free-var, shared helpers, ~550 LOC)
│   ├── frontier.rs (NEW — frontier structs, builders, D4WorkItem, ~1,400 LOC incl. tests)
│   └── build.rs    (NEW — compile_gpu_d4_with_gate, circuit construction, ~1,600 LOC incl. tests)
├── mod.rs          (MODIFY — `pub mod gpu_d4;` → `pub mod gpu_d4;` unchanged, re-exports unchanged)
```

### Module Responsibilities

| Module | Production Content | Test Content |
|--------|-------------------|-------------|
| `mod.rs` | `GpuCompileConfig`, `validate_cnf_gpu`, `compute_free_var_mask_gpu[_gated]`, `compile_gpu_d4` + `_gated` (thin wrappers), `alloc_compile_gate`, `memset_u8_sync`, `exclusive_scan_u32_inplace` (pub(crate)), `bitset_words_per_item`, `checked_pool_len_*` | `gpu_d4_compile_config_requires_smoothing_caps` |
| `frontier.rs` | `D4WorkItem`, `GpuFrontierBitset` + methods, `GpuFrontierDense` + methods, `build_frontier_bitset`, `build_frontier_dense` | 4 frontier tests |
| `build.rs` | `compile_gpu_d4_with_gate`, `alloc_component_scratch` | 5 compilation tests |

### Cross-module dependencies

- `frontier.rs` uses: `super::{exclusive_scan_u32_inplace, checked_pool_len_usize, alloc_compile_gate}` from mod.rs, `D4_MODULE`/`SCAN_MODULE`/kernel consts from xlog_cuda
- `build.rs` production uses: `super::{validate_cnf_gpu, exclusive_scan_u32_inplace, checked_pool_len_usize, memset_u8_sync}` from mod.rs, `super::frontier::build_frontier_bitset` from frontier.rs. Tests additionally use `super::alloc_compile_gate`.
- `build.rs` accesses frontier struct fields: `GpuFrontierBitset.items` and frontier accessor methods. Frontier struct fields that are directly accessed by `compile_gpu_d4_with_gate` must be `pub(super)` (see Task 2 Step 2).
- Test modules use: `super::*` + shared `try_provider()` and `alloc_component_scratch` helpers
- `exclusive_scan_u32_inplace` is `pub(crate)` — used by `xlog-prob/src/gpu.rs`. Stays accessible via `crate::compilation::gpu_d4::exclusive_scan_u32_inplace`.

---

### Task 1: Create gpu_d4/ directory and scaffold mod.rs

**Files:**
- Create: `crates/xlog-prob/src/compilation/gpu_d4/mod.rs`
- Create: `crates/xlog-prob/src/compilation/gpu_d4/frontier.rs` (empty placeholder)
- Create: `crates/xlog-prob/src/compilation/gpu_d4/build.rs` (empty placeholder)
- Delete: `crates/xlog-prob/src/compilation/gpu_d4.rs` (after extraction)

- [ ] **Step 1: Create gpu_d4 directory**

```bash
mkdir -p crates/xlog-prob/src/compilation/gpu_d4
```

- [ ] **Step 2: Copy gpu_d4.rs → gpu_d4/mod.rs**

```bash
cp crates/xlog-prob/src/compilation/gpu_d4.rs crates/xlog-prob/src/compilation/gpu_d4/mod.rs
```

- [ ] **Step 3: Add submodule declarations to mod.rs**

At the top of `gpu_d4/mod.rs`, after the module doc comment and `use` block, add:

```rust
pub(crate) mod build;
pub(crate) mod frontier;
```

- [ ] **Step 4: Create empty placeholder files**

`frontier.rs`:
```rust
//! Frontier expansion for GPU D4 compilation.
```

`build.rs`:
```rust
//! XGCF circuit construction from frontier items.
```

- [ ] **Step 5: Delete the old gpu_d4.rs**

```bash
rm crates/xlog-prob/src/compilation/gpu_d4.rs
```

- [ ] **Step 6: Verify compilation**

```bash
cargo check -p xlog-prob
```

Expected: compiles clean (mod.rs is the full original file).

- [ ] **Step 7: Commit**

```bash
git add crates/xlog-prob/src/compilation/gpu_d4/
git add -u crates/xlog-prob/src/compilation/gpu_d4.rs
git commit -m "refactor(xlog-prob): convert gpu_d4.rs to directory module"
```

---

### Task 2: Extract frontier.rs

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4/frontier.rs`

Move these items from `mod.rs` → `frontier.rs`:
- `D4WorkItem` struct (line 283)
- `GpuFrontierBitset` struct + `impl` (lines 395–426)
- `GpuFrontierDense` struct + `impl` (lines 431–457)
- `build_frontier_bitset` function (lines 464–638)
- `build_frontier_dense` function (lines 639–802)
- Related test functions: `gpu_d4_frontier_prepare_and_expand_*`, `gpu_d4_build_frontier_dense_*`, `gpu_d4_build_frontier_bitset_*` (4 tests)

- [ ] **Step 1: Promote shared helpers in mod.rs**

Make these functions accessible to frontier.rs:

```rust
// These need to be pub(super) or pub(crate) for frontier.rs access:
pub(super) fn alloc_compile_gate(...) // was fn (private)
pub(super) fn bitset_words_per_item(...) // was fn (private)
pub(super) fn checked_pool_len_usize(...) // was fn (private)
pub(super) fn checked_pool_len_u32(...) // was fn (private)
// exclusive_scan_u32_inplace is already pub(crate) — no change needed
```

- [ ] **Step 2: Move structs and functions to frontier.rs**

Promote `GpuFrontierBitset` and `GpuFrontierDense` fields that are directly accessed by `compile_gpu_d4_with_gate` (in build.rs) to `pub(super)`. In particular, `GpuFrontierBitset.items` and `GpuFrontierDense.items` (and any other fields read by build.rs) must be `pub(super)`.

Move the listed items. Add imports at top of frontier.rs:

```rust
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use super::{
    alloc_compile_gate, bitset_words_per_item, checked_pool_len_u32,
    checked_pool_len_usize, exclusive_scan_u32_inplace, GpuCompileConfig,
};
```

- [ ] **Step 3: Re-export frontier types from mod.rs**

In `gpu_d4/mod.rs`:

```rust
pub use frontier::{build_frontier_bitset, build_frontier_dense, GpuFrontierBitset, GpuFrontierDense};
pub(crate) use frontier::D4WorkItem;
```

- [ ] **Step 4: Move related test functions**

Move the 4 frontier test functions from `mod.rs` test module into a `#[cfg(test)] mod tests` block in frontier.rs. The test helpers `try_provider`, `words_per_item`, `alloc_component_scratch` stay in mod.rs tests (shared). If frontier tests need them, either:
- Duplicate `try_provider()` in frontier.rs tests (preferred — it's ~15 lines)
- Or make it `pub(super)` in mod.rs tests

- [ ] **Step 5: Verify compilation**

```bash
cargo check -p xlog-prob
```

- [ ] **Step 6: Run tests**

```bash
cargo test -p xlog-prob --release -- gpu_d4
```

All gpu_d4 tests should pass.

- [ ] **Step 7: Commit**

```bash
git add -u crates/xlog-prob/src/compilation/gpu_d4/
git commit -m "refactor(xlog-prob): extract frontier.rs from gpu_d4"
```

---

### Task 3: Extract build.rs

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4/mod.rs`
- Modify: `crates/xlog-prob/src/compilation/gpu_d4/build.rs`

Move these items from `mod.rs` → `build.rs`:
- `compile_gpu_d4_with_gate` function (lines 856–1245)
- `alloc_component_scratch` function (lines 803–835)
- Related test functions: all 5 `gpu_d4_compile_phase1_*` tests

- [ ] **Step 1: Promote remaining shared helpers**

Make these accessible to build.rs:

```rust
pub(super) fn memset_u8_sync(...) // was fn (private)
// alloc_compile_gate already pub(super) from Task 2
// checked_pool_len_usize already pub(super) from Task 2
// exclusive_scan_u32_inplace already pub(crate)
```

- [ ] **Step 2: Move functions to build.rs**

Add imports at top of `build.rs`:

```rust
use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DevicePtr, DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{d4_kernels, scan_kernels, D4_MODULE, SCAN_MODULE};
use xlog_cuda::CudaKernelProvider;
use xlog_solve::GpuCnf;

use crate::gpu::{GpuCircuitBuilder, GpuCircuitLayout, GpuXgcf};
use super::{
    checked_pool_len_usize, exclusive_scan_u32_inplace, memset_u8_sync,
    validate_cnf_gpu, GpuCompileConfig,
};
use super::frontier::build_frontier_bitset;
```

- [ ] **Step 3: Update compile_gpu_d4[_gated] in mod.rs**

These are thin wrappers. Update them to call `build::compile_gpu_d4_with_gate`:

```rust
pub fn compile_gpu_d4(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
) -> Result<GpuXgcf> {
    let compile_needed = alloc_compile_gate(provider, 1)?;
    build::compile_gpu_d4_with_gate(cnf, provider, config, &compile_needed)
}

pub fn compile_gpu_d4_gated(
    cnf: &GpuCnf,
    provider: &Arc<CudaKernelProvider>,
    config: &GpuCompileConfig,
    compile_needed: &TrackedCudaSlice<u32>,
) -> Result<GpuXgcf> {
    build::compile_gpu_d4_with_gate(cnf, provider, config, compile_needed)
}
```

Change `compile_gpu_d4_with_gate` to `pub(super)` in build.rs.

- [ ] **Step 4: Move 5 compilation test functions to build.rs**

Duplicate `try_provider()` in build.rs tests (or import from a shared test helper). Also duplicate `alloc_component_scratch` in the build.rs test module if compilation tests call it — since it just moved to build.rs production code, tests in the same file can call it directly via `super::alloc_component_scratch`.

- [ ] **Step 5: Verify compilation**

```bash
cargo check -p xlog-prob
```

- [ ] **Step 6: Run all tests**

```bash
cargo test -p xlog-prob --release -- gpu_d4
```

All gpu_d4 tests should pass.

- [ ] **Step 7: Commit**

```bash
git add -u crates/xlog-prob/src/compilation/gpu_d4/
git commit -m "refactor(xlog-prob): extract build.rs from gpu_d4"
```

---

### Task 4: Gate sub-wave 5a

**Files:** None modified

- [ ] **Step 1: Run workspace tests**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
```

- [ ] **Step 2: Run CUDA certification**

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

Expected: 206/206.

- [ ] **Step 3: Verify external callers still work**

Check that `crate::compilation::gpu_d4::exclusive_scan_u32_inplace` is still accessible from `xlog-prob/src/gpu.rs`:

```bash
cargo check -p xlog-prob
```

- [ ] **Step 4: Document final line counts**

```bash
wc -l crates/xlog-prob/src/compilation/gpu_d4/*.rs
```

Record in commit message.

---

## Chunk 2: Sub-wave 5b — mc.rs Decomposition

### File Map

```
crates/xlog-prob/src/
├── mc.rs (DELETE — 3,399 lines)
├── mc/
│   ├── mod.rs      (NEW — McProgram struct, types, public evaluate*, compile, ~800 LOC)
│   ├── evidence.rs (NEW — EvidenceForcing, compile_evidence_forcing, resolve_sampling_method, ~200 LOC)
│   ├── buffers.rs  (NEW — GPU buffer allocation, sample/prob tables, reset plan, ~600 LOC)
│   ├── sampling.rs (NEW — GPU eval loop, nonmonotone SCC handling, state management, ~500 LOC)
│   └── results.rs  (NEW — binomial_estimate, normal_quantile, CPU eval path (host-io), ~800 LOC)
```

### Module Responsibilities

| Module | Content |
|--------|---------|
| `mod.rs` | McSamplingMethod, McCountStrategy, McTimingBreakdown, McEvalConfig, McQueryEstimate, McResult, McDeviceResult. Internal structs: ProbFactSpec, AdSpec, GpuMcPlan, ProbTableDevice, AdDecisionDevice, AdTableDevice, EvalStats, Relation, SccKind, SccPlan. McProgram struct + pub methods (compile_source*, evaluate*, num_vars). Private: compile_program, provider, build_gpu_plan, evaluate_gpu_counts_with. Re-exports from submodules. |
| `evidence.rs` | ForceabilityReason enum, EvidenceForcing struct, `impl McProgram { compile_evidence_forcing, resolve_sampling_method }` |
| `buffers.rs` | upload_slice, load_deterministic_facts, McSampleResetPlan, build_sample_reset_plan, clone_buffer_device, build_zero_arity_buffer, dedup_relation, build_prob_tables_device, build_sample_buffers, build_buffer_from_rows, push_value_bytes |
| `sampling.rs` | evaluate_program_gpu, execute_nonmonotone_scc_gpu, snapshot_scc_state, clone_state, apply_state_to_store_move, state_signature, states_equal, buffers_equal, intersect_states_device, buffer_intersection |
| `results.rs` | binomial_estimate, normal_quantile, CPU host-io path: build_scc_plans, is_recursive_scc, evaluate_program_inplace, eval_monotone_recursive_scc, eval_nonmonotone_scc, intersect_states, hash_scc_state, eval_rule, select_relation, negated_atom_holds, atom_matches_bound, materialize_head_non_aggregate, AggState, value_le, eval_aggregate_head, atom_holds, evidence_satisfied. Also: compile_sampling_plan, extend_prob_facts_with_coin, ensure_predicate_decls, augment_schemas_for_program, apply_sample_facts |

### Key Design Notes

- `impl McProgram` blocks can be distributed across modules within the same crate. `evidence.rs` adds an `impl McProgram` block for evidence methods.
- Internal structs (ProbFactSpec, AdSpec, etc.) stay in mod.rs and are `pub(super)` for submodule access.
- Feature-gated `#[cfg(feature = "host-io")]` code moves intact with its gate annotations.
- `NONMONOTONE_SEMANTICS` const stays in mod.rs (public API).

---

### Task 5: Create mc/ directory and scaffold mod.rs

**Files:**
- Create: `crates/xlog-prob/src/mc/mod.rs`
- Create: `crates/xlog-prob/src/mc/evidence.rs` (placeholder)
- Create: `crates/xlog-prob/src/mc/buffers.rs` (placeholder)
- Create: `crates/xlog-prob/src/mc/sampling.rs` (placeholder)
- Create: `crates/xlog-prob/src/mc/results.rs` (placeholder)
- Delete: `crates/xlog-prob/src/mc.rs`

- [ ] **Step 1: Create mc directory**

```bash
mkdir -p crates/xlog-prob/src/mc
```

- [ ] **Step 2: Copy mc.rs → mc/mod.rs**

```bash
cp crates/xlog-prob/src/mc.rs crates/xlog-prob/src/mc/mod.rs
```

- [ ] **Step 3: Add submodule declarations**

At the top of `mc/mod.rs`, after the module doc comment and before `use`:

```rust
mod buffers;
mod evidence;
mod results;
mod sampling;
```

- [ ] **Step 4: Create empty placeholder files**

```rust
// evidence.rs
//! Evidence forcing for Monte Carlo sampling.

// buffers.rs
//! GPU buffer allocation and sample management.

// sampling.rs
//! GPU evaluation loop and nonmonotone SCC handling.

// results.rs
//! Result aggregation, statistics, and CPU evaluation path.
```

- [ ] **Step 5: Delete old mc.rs**

```bash
rm crates/xlog-prob/src/mc.rs
```

- [ ] **Step 6: Verify compilation**

```bash
cargo check -p xlog-prob
```

- [ ] **Step 7: Commit**

```bash
git add crates/xlog-prob/src/mc/
git add -u crates/xlog-prob/src/mc.rs
git commit -m "refactor(xlog-prob): convert mc.rs to directory module"
```

---

### Task 6: Extract evidence.rs

**Files:**
- Modify: `crates/xlog-prob/src/mc/mod.rs`
- Modify: `crates/xlog-prob/src/mc/evidence.rs`

Move from `mod.rs` → `evidence.rs`:
- `ForceabilityReason` enum (line 2463)
- `EvidenceForcing` struct (line 2472)
- `impl McProgram { compile_evidence_forcing }` (lines 1164–1239)
- `impl McProgram { resolve_sampling_method }` (lines 294–321)

- [ ] **Step 1: Move ForceabilityReason and EvidenceForcing**

Add to `evidence.rs`:

```rust
use xlog_core::{Result, XlogError};
use super::{McProgram, McSamplingMethod};
```

Move the enum and struct. Re-export from mod.rs:

```rust
pub use evidence::{EvidenceForcing, ForceabilityReason};
```

- [ ] **Step 2: Move compile_evidence_forcing and resolve_sampling_method**

These are `impl McProgram` methods. In `evidence.rs`:

```rust
impl McProgram {
    pub fn compile_evidence_forcing(&self) -> Result<EvidenceForcing> {
        // ... (moved from mod.rs)
    }

    pub(crate) fn resolve_sampling_method(
        &self,
        requested: Option<McSamplingMethod>,
    ) -> Result<(McSamplingMethod, EvidenceForcing)> {
        // ... (moved from mod.rs)
    }
}
```

The `impl McProgram` block needs access to `self.prob_facts`, `self.annotated_disjunctions`, `self.evidence`, `self.bernoulli_probs`. These fields need to be `pub(super)` in the McProgram struct definition.

- [ ] **Step 3: Promote McProgram fields to pub(super)**

In `mod.rs`, change all McProgram fields from private to `pub(super)`:

```rust
pub struct McProgram {
    pub(super) gpu_config: GpuConfig,
    pub(super) program: Program,
    #[cfg(feature = "host-io")]
    pub(super) base_store: HashMap<String, Relation>,
    #[cfg(feature = "host-io")]
    pub(super) scc_plans: Vec<SccPlan>,
    pub(super) queries: Vec<GroundAtom>,
    pub(super) evidence: Vec<(GroundAtom, bool)>,
    pub(super) bernoulli_probs: Vec<f32>,
    pub(super) prob_facts: Vec<ProbFactSpec>,
    pub(super) annotated_disjunctions: Vec<AdSpec>,
}
```

- [ ] **Step 4: Verify compilation and run tests**

```bash
cargo check -p xlog-prob && cargo test -p xlog-prob --release -- mc
```

- [ ] **Step 5: Commit**

```bash
git add -u crates/xlog-prob/src/mc/
git commit -m "refactor(xlog-prob): extract evidence.rs from mc"
```

---

### Task 7: Extract buffers.rs

**Files:**
- Modify: `crates/xlog-prob/src/mc/mod.rs`
- Modify: `crates/xlog-prob/src/mc/buffers.rs`

Move from `mod.rs` → `buffers.rs`:
- `upload_slice` (line 99)
- `load_deterministic_facts` (line 1242)
- `McSampleResetPlan` struct (line 1283)
- `build_sample_reset_plan` (line 1303)
- `clone_buffer_device` (line 1383)
- `build_zero_arity_buffer` (line 1429)
- `dedup_relation` (line 1448)
- `build_prob_tables_device` (line 1457)
- `build_sample_buffers` (line 1585)
- `build_buffer_from_rows` (line 1977)
- `push_value_bytes` (line 2265)
- `infer_schema_from_values` (line 2181)
- `scalar_type_from_value` (line 2221)
- `infer_term_scalar_type` (line 2160)
- `ensure_schema_for_atom` (line 2146)

- [ ] **Step 1: Move functions to buffers.rs**

Add imports:

```rust
use std::collections::HashMap;
use std::sync::Arc;

use cudarc::driver::{CudaView, DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{mc_eval_kernels, MC_EVAL_MODULE};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_logic::ast::{Atom, Program, Term};
use xlog_runtime::Executor;

use crate::provenance::{atom_key_from_ground_atom, Value};
use super::{
    AdDecisionDevice, AdSpec, AdTableDevice, GpuMcPlan, McProgram,
    ProbFactSpec, ProbTableDevice,
};
```

Internal structs referenced by buffers.rs (ProbTableDevice, AdTableDevice, etc.) need `pub(super)` visibility in mod.rs.

- [ ] **Step 2: Promote internal struct visibility**

In mod.rs, make these structs `pub(super)`:

```rust
pub(super) struct ProbFactSpec { ... }
pub(super) struct AdSpec { ... }
pub(super) struct GpuMcPlan { ... }
pub(super) struct ProbTableDevice { ... }
pub(super) struct AdDecisionDevice { ... }
pub(super) struct AdTableDevice { ... }
pub(super) struct McSampleResetPlan { ... }  // (in buffers.rs after move)
```

And their fields need `pub(super)` too.

- [ ] **Step 3: Update call sites in mod.rs**

`evaluate_gpu_counts_with` calls: `load_deterministic_facts`, `build_prob_tables_device`, `build_sample_reset_plan`, `clone_buffer_device`, `build_sample_buffers`, `dedup_relation`. Update to `buffers::function_name(...)`.

- [ ] **Step 4: Verify compilation and run tests**

```bash
cargo check -p xlog-prob && cargo test -p xlog-prob --release -- mc
```

- [ ] **Step 5: Commit**

```bash
git add -u crates/xlog-prob/src/mc/
git commit -m "refactor(xlog-prob): extract buffers.rs from mc"
```

---

### Task 8: Extract sampling.rs

**Files:**
- Modify: `crates/xlog-prob/src/mc/mod.rs`
- Modify: `crates/xlog-prob/src/mc/sampling.rs`

Move from `mod.rs` → `sampling.rs`:
- `evaluate_program_gpu` (line 1691)
- `execute_nonmonotone_scc_gpu` (line 1767)
- `snapshot_scc_state` (line 1836)
- `clone_state` (line 1857)
- `apply_state_to_store_move` (line 1873)
- `state_signature` (line 1879)
- `states_equal` (line 1888)
- `buffers_equal` (line 1908)
- `intersect_states_device` (line 1928)
- `buffer_intersection` (line 1962)

- [ ] **Step 1: Move functions to sampling.rs**

Add imports:

```rust
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use xlog_core::{RelId, Result, XlogError};
use xlog_cuda::{CudaBuffer, CudaKernelProvider};
use xlog_ir::RirNode;
use xlog_runtime::Executor;

use super::buffers::{clone_buffer_device, dedup_relation};
use super::{EvalStats, SccKind, SccPlan};
```

**Note:** `evaluate_program_gpu` and `execute_nonmonotone_scc_gpu` use `Executor` methods (`execute_recursive_scc`, `execute_non_recursive_scc`, `execute_node`), `RirNode` for rule dispatch, and internal types `SccKind`/`SccPlan`/`EvalStats` for plan iteration and statistics tracking.

- [ ] **Step 2: Make moved functions pub(super) as needed**

`evaluate_program_gpu` and `execute_nonmonotone_scc_gpu` are called from `evaluate_gpu_counts_with` in mod.rs. Make them `pub(super)`.

- [ ] **Step 3: Update call sites in mod.rs**

In `evaluate_gpu_counts_with`, update: `evaluate_program_gpu(...)` → `sampling::evaluate_program_gpu(...)`.

- [ ] **Step 4: Verify compilation and run tests**

```bash
cargo check -p xlog-prob && cargo test -p xlog-prob --release -- mc
```

- [ ] **Step 5: Commit**

```bash
git add -u crates/xlog-prob/src/mc/
git commit -m "refactor(xlog-prob): extract sampling.rs from mc"
```

---

### Task 9: Extract results.rs and gate sub-wave 5b

**Files:**
- Modify: `crates/xlog-prob/src/mc/mod.rs`
- Modify: `crates/xlog-prob/src/mc/results.rs`

Move from `mod.rs` → `results.rs`:
- `binomial_estimate` (line 3320, cfg host-io)
- `normal_quantile` (line 3346, cfg host-io)
- `build_scc_plans` (line 2480, cfg host-io)
- `is_recursive_scc` (line 2535, cfg host-io)
- `evaluate_program_inplace` (line 2555, cfg host-io)
- `eval_monotone_recursive_scc` (line 2598, cfg host-io)
- `eval_nonmonotone_scc` (line 2695, cfg host-io)
- `intersect_states` (line 2761, cfg host-io)
- `hash_scc_state` (line 2789, cfg host-io)
- `eval_rule` (line 2817, cfg host-io)
- `select_relation` (line 2888, cfg host-io)
- `negated_atom_holds` (line 2914, cfg host-io)
- `atom_matches_bound` (line 2937, cfg host-io)
- `materialize_head_non_aggregate` (line 2977, cfg host-io)
- `AggState` struct + impl (line 3025, cfg host-io)
- `value_le` (line 3187, cfg host-io)
- `eval_aggregate_head` (line 3200, cfg host-io)
- `atom_holds` (line 3301, cfg host-io)
- `evidence_satisfied` (line 3309, cfg host-io)
- `compile_sampling_plan` (line 2386)
- `extend_prob_facts_with_coin` (line 2235)
- `ensure_predicate_decls` (line 2070)
- `augment_schemas_for_program` (line 2035)
- `apply_sample_facts` impl method (line 1117, cfg host-io)

- [ ] **Step 1: Move all functions to results.rs**

Note: Most of these are `#[cfg(feature = "host-io")]`. Preserve all feature gates.

Add imports:

```rust
use std::collections::{HashMap, HashSet};
#[cfg(feature = "host-io")]
use std::hash::{Hash, Hasher};

use xlog_core::{Result, ScalarType, Schema, XlogError};
#[cfg(feature = "host-io")]
use xlog_logic::ast::{AggExpr, AggOp, BodyLiteral, GroundAtom, Rule};
use xlog_logic::ast::{
    AnnotatedDisjunction, Atom, Evidence, PredDecl, ProbFact, Program, Term,
};
#[cfg(feature = "host-io")]
use xlog_logic::stratify::{build_dependency_graph, find_sccs_for_lowering};

use crate::provenance::{atom_key_from_ground_atom, validate_prob, GroundAtom, Value};
#[cfg(feature = "host-io")]
use crate::provenance::{eval_arith_expr, eval_comparison, unify_atom, value_from_term};
use super::{AdSpec, EvalStats, McProgram, ProbFactSpec};
#[cfg(feature = "host-io")]
use super::{Relation, SccKind, SccPlan};
```

- [ ] **Step 2: Make functions pub(super) as needed**

- `compile_sampling_plan` → `pub(super)` (called from compile_program in mod.rs)
- `extend_prob_facts_with_coin` → `pub(super)` (called from compile_program)
- `ensure_predicate_decls` → `pub(super)` (called from build_gpu_plan)
- `augment_schemas_for_program` → `pub(super)` (called from build_gpu_plan)
- `build_scc_plans` → `pub(super)` (called from compile_program)
- `binomial_estimate`, `normal_quantile` → `pub(super)` (called from evaluate_cpu)
- `evaluate_program_inplace` → `pub(super)` (called from evaluate_cpu)
- `evidence_satisfied`, `atom_holds` → `pub(super)` (called from evaluate_cpu)
- `apply_sample_facts` → `impl McProgram` method, keep in results.rs

- [ ] **Step 3: Update call sites in mod.rs**

In `compile_program`: `results::compile_sampling_plan(...)`, `results::build_scc_plans(...)`, etc.
In `evaluate_cpu`: `results::evaluate_program_inplace(...)`, `results::binomial_estimate(...)`, etc.
In `build_gpu_plan`: `results::ensure_predicate_decls(...)`, `results::augment_schemas_for_program(...)`.

- [ ] **Step 4: Verify compilation and run tests**

```bash
cargo check -p xlog-prob && cargo test -p xlog-prob --release -- mc
```

- [ ] **Step 5: Run 5b gate**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
cargo test -p xlog-cuda-tests --test certification_suite --release
```

Expected: workspace green, 206/206 cert.

- [ ] **Step 5a: Targeted MC supplemental tests**

```bash
cargo test -p xlog-prob --release -- mc
.venv/bin/python -m pytest python/tests/test_mc_device_results.py -x --timeout=120
```

These targeted MC tests (Rust `mc` tests + Python `test_mc_device_results`) provide additional confidence that the mc.rs decomposition preserved sampling behavior. (Per spec: "targeted MC supplemental sub-wave coverage")

- [ ] **Step 6: Document final line counts**

```bash
wc -l crates/xlog-prob/src/mc/*.rs
```

- [ ] **Step 7: Commit**

```bash
git add -u crates/xlog-prob/src/mc/
git commit -m "refactor(xlog-prob): extract results.rs from mc, complete 5b decomposition"
```

---

## Chunk 3: Sub-wave 5c — Coherence/Polish

### 5c.1 Naming — NO-OP

Research found **zero** `fetch_*` functions across the entire workspace. The codebase already uses a consistent naming convention: `get_*` for retrieval, `create_*` for allocation, `build_*` for construction. No renames needed.

---

### Task 10: Config coherence (5c.2)

**Files:**
- Modify: `crates/xlog-prob/src/compilation/gpu_d4/mod.rs` — `GpuCompileConfig` (add Default)
- Modify: `crates/xlog-prob/src/compilation/gpu_cache.rs` — `GpuCircuitCacheConfig` (add Default, doc comments)
- Modify: `crates/xlog-prob/src/compilation/validation.rs` — `GpuEquivalenceConfig` (add doc comments)
- Modify: `crates/xlog-solve/src/gpu_cdcl.rs` — `GpuCdclConfig` (add doc comments)
- Modify: All 14 config structs across workspace (add `#[non_exhaustive]`)

- [ ] **Step 1: Add Default impl for GpuCompileConfig**

In `gpu_d4/mod.rs`:

```rust
impl Default for GpuCompileConfig {
    fn default() -> Self {
        Self {
            frontier_depth: 4,
            max_frontier_items: 65536,
            max_depth: 128,
            smooth_node_cap: 1 << 20,
            smooth_edge_cap: 1 << 21,
            cdcl_restart_interval: 100,
            cdcl_learned_bytes: 64 * 1024 * 1024,
            cdcl_conflict_budget: None,
            incremental_verify: true,
        }
    }
}
```

Verify default values match what callers currently construct. Check `compilation/mod.rs`, `exact.rs`, `exact_gpu.rs` for existing GpuCompileConfig constructions to confirm reasonable defaults.

- [ ] **Step 2: Add Default impl and doc comments for GpuCircuitCacheConfig**

In `gpu_cache.rs`, read the struct fields first and add appropriate defaults.

- [ ] **Step 3: Add doc comments for GpuEquivalenceConfig and GpuCdclConfig**

In `validation.rs` and `gpu_cdcl.rs`, add `///` doc comments explaining purpose and when to customize.

- [ ] **Step 4: Add #[non_exhaustive] to public config structs**

For each public config struct, add `#[non_exhaustive]`:

```rust
#[non_exhaustive]
#[derive(Debug, Clone, Copy)]
pub struct GpuCompileConfig { ... }
```

Affects: `MemoryBudget`, `RuntimeConfig`, `McEvalConfig`, `WfsConfig`, `NeuralFastPathConfig`, `GpuConfig`, `GpuCircuitCacheConfig`, `GpuCompileConfig`, `GpuEquivalenceConfig`, `NetworkConfig`, `SolverConfig`, `GpuCdclConfig`, `OptimizerConfig`.

**Migration:** `#[non_exhaustive]` breaks struct literal construction from external crates. Within the workspace, internal crates (xlog-cli, xlog-prob, xlog-runtime, etc.) can still use struct literals since they're in the same workspace. However, pyxlog is a `cdylib` and compiles as a separate crate — check its config struct construction sites:

```bash
rg 'GpuCompileConfig|McEvalConfig|GpuConfig|WfsConfig|SolverConfig|GpuCdclConfig|NetworkConfig|OptimizerConfig|MemoryBudget|RuntimeConfig' crates/pyxlog/src/ --type rust -n
```

For each config struct literal found in pyxlog: if a `Default` impl exists, convert to `Config { field: value, ..Default::default() }`. If no `Default` exists yet, add one first (or use a builder/constructor if one exists).

For xlog-cli:
```bash
rg 'GpuCompileConfig|McEvalConfig|SolverConfig|GpuCdclConfig' crates/xlog-cli/src/ --type rust -n
```

Same migration: convert struct literals to use `..Default::default()`.

- [ ] **Step 4a: Add Default impl for NetworkConfig**

In `crates/xlog-neural/src/lib.rs` (or wherever `NetworkConfig` is defined), add a `Default` impl if one doesn't exist. Check current field values used by callers to determine sensible defaults.

- [ ] **Step 5: Verify compilation**

```bash
cargo check --workspace --exclude pyxlog
cargo check -p pyxlog
```

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: add Default impls, #[non_exhaustive], doc comments to config structs (5c.2)"
```

---

### Task 11: Test harness consolidation (5c.3)

**Files:**
- Create: `crates/xlog-cuda/tests/common/mod.rs`
- Modify: ~31 test files in `crates/xlog-cuda/tests/`
- Modify: `crates/xlog-prob/tests/gpu_mc_device_counts.rs`
- Modify: `crates/xlog-prob/tests/mc_gpu_native.rs`
- Modify: `crates/xlog-runtime/src/relation.rs` (test module)

- [ ] **Step 1: Create canonical setup_provider in xlog-cuda tests**

Create `crates/xlog-cuda/tests/common/mod.rs`:

```rust
use std::sync::Arc;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

/// Canonical CUDA provider for tests. Returns None if CUDA is unavailable.
pub fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let memory = Arc::new(GpuMemoryManager::new(
        device.clone(),
        MemoryBudget::with_limit(1024 * 1024 * 1024),
    ));
    CudaKernelProvider::new(device, memory)
        .ok()
        .map(Arc::new)
}
```

- [ ] **Step 2: Update xlog-cuda test files**

For each of the ~31 test files in `crates/xlog-cuda/tests/`:

Replace the local `setup_provider` function with:

```rust
mod common;
use common::setup_provider;
```

Note: Some tests return `Option<CudaKernelProvider>` (non-Arc) vs `Option<Arc<CudaKernelProvider>>`. Standardize to `Arc<CudaKernelProvider>`. Callers that unwrap to non-Arc should add `.as_ref()` or `&*provider`.

- [ ] **Step 3: Create canonical setup_provider in xlog-prob tests**

Create `crates/xlog-prob/tests/common/mod.rs` with the same pattern.

Update `gpu_mc_device_counts.rs` and `mc_gpu_native.rs` to use it.

- [ ] **Step 4: Update relation.rs test module**

The `setup_provider` in `crates/xlog-runtime/src/relation.rs` is inline in a `#[cfg(test)]` module. Leave as-is (it's within a source file, not a separate test file — different pattern).

- [ ] **Step 5: Verify compilation and run tests**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
```

- [ ] **Step 6: Commit**

```bash
git add crates/xlog-cuda/tests/common/
git add crates/xlog-prob/tests/common/
git add -u
git commit -m "refactor: consolidate setup_provider() test helpers (5c.3)"
```

---

### Task 12: xlog-prob exports + WFS consolidation (5c.4 + 5c.5)

**Files:**
- Modify: `crates/xlog-prob/src/lib.rs`
- Modify: `crates/xlog-prob/src/wfs.rs`

- [ ] **Step 1: Add top-level re-exports to lib.rs**

```rust
// Primary entry points (convenience re-exports)
pub use compilation::{
    compile_gpu_d4_and_verify, compile_gpu_d4_and_verify_cached,
    GpuCompileConfig, CircuitCompileProfile,
};
pub use exact::{ExactDdnnfProgram, ExactResult, GpuConfig};
pub use mc::{
    McEvalConfig, McProgram, McSamplingMethod, McCountStrategy,
    McResult, McDeviceResult, EvidenceForcing, ForceabilityReason,
};
pub use wfs::{
    WfsConfig, WfsResult, TruthValue, WfsAtom, WfsRule, WfsLiteral,
    evaluate_wfs_rules, evaluate_wfs_with_rules,
};
```

- [ ] **Step 2: Audit WFS entry point callers**

Search for calls to `evaluate_wfs_scc`, `evaluate_wfs_scc_with_config`, `evaluate_wfs_with_rules_config` across the workspace:

```bash
rg "evaluate_wfs_scc|evaluate_wfs_with_rules_config" --type rust
```

For any callers outside xlog-prob tests:
- If only used in tests → downgrade to `pub(crate)`
- If used by other crates → keep `pub`

- [ ] **Step 3: Downgrade legacy WFS functions**

Based on caller audit:
- `evaluate_wfs_scc` → `pub(crate)` if test-only (doc says "Legacy interface")
- `evaluate_wfs_scc_with_config` → `pub(crate)` if test-only
- `evaluate_wfs_with_rules_config` → keep `pub` (takes config, unlike the default wrapper)

- [ ] **Step 4: Verify compilation**

```bash
cargo check --workspace --exclude pyxlog
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor(xlog-prob): add top-level re-exports, consolidate WFS entry points (5c.4+5c.5)"
```

---

### Task 13: Visibility tightening (5c.7)

**Files:** Multiple files across xlog-prob, xlog-solve, xlog-logic

- [ ] **Step 1: Audit xlog-prob internal helpers**

Focus on the highest-density files:
- `compilation/gpu_cache.rs` (~8–12 items)
- `gpu.rs` (~5–8 items)

For each `pub fn`, search for external callers (outside the crate):

```bash
# For each function, e.g., some_helper:
rg "some_helper" --type rust --glob '!crates/xlog-prob/**'
```

If no external callers: change `pub fn` → `pub(crate) fn`.

- [ ] **Step 2: Audit xlog-solve internals**

Focus on:
- `instance.rs` (~10–15 items)
- `proof.rs` (~6–8 items)
- `gpu_cdcl.rs` (~8–10 items)

Same method: search for external callers, downgrade where safe.

- [ ] **Step 3: Audit xlog-logic internals**

Focus on AST manipulation helpers in `ast.rs`, `expand.rs`, `lower.rs`.

- [ ] **Step 4: Verify compilation**

```bash
cargo check --workspace --exclude pyxlog
cargo check -p pyxlog
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: tighten visibility pub -> pub(crate) across xlog-prob, xlog-solve, xlog-logic (5c.7)"
```

---

### Task 14: Unwrap/expect policy + stale comments (5c.6 + 5c.9)

**Files:** Multiple across workspace

- [ ] **Step 1: Opportunistic unwrap fixes in production code**

Focus on the highest-risk files:
- `xlog-prob/src/compilation/gpu_d4/build.rs` (check for bare `.unwrap()` in production paths)
- `xlog-cuda/src/provider/mod.rs` (kernel launches, device ops)
- `xlog-runtime/src/executor/mod.rs`

For each bare `.unwrap()` in production code:
- If guarding an internal invariant → change to `.expect("invariant: <description>")`
- If guarding user-facing input → change to `.ok_or_else(|| XlogError::...)?`
- If in test code → leave as-is

**Do NOT attempt to fix all 930 unwrap/expect calls.** Focus on the top 3 files and production-path unwraps only.

- [ ] **Step 2: Build script unwrap fixes**

Fix the 8 bare `.unwrap()` calls in build scripts (if any). Change to `.expect("descriptive message")`.

- [ ] **Step 3: Count and document final tally**

```bash
# Production unwrap/expect count (excluding test files):
rg '\.unwrap\(\)|\.expect\(' --type rust --glob '!**/tests/**' --glob '!**/test_*' -c
```

Record in commit message.

- [ ] **Step 4: Remove stale comments**

Search for outdated references:
- Comments referencing old file names (`download_column_u32`, `download_column_f32`, etc.)
- Comments referencing pre-split line numbers
- Comments referencing pre-refactoring file structure

```bash
rg 'download_column_u32|download_column_f32|download_column_i32|download_column_i64' --type rust -l
rg 'lib\.rs:\d{4}|executor\.rs:\d{4}|provider\.rs:\d{4}' --type rust -l
```

- [ ] **Step 5: Verify compilation**

```bash
cargo check --workspace --exclude pyxlog
```

- [ ] **Step 6: Commit**

```bash
git add -u
git commit -m "refactor: opportunistic unwrap->expect fixes, stale comment cleanup (5c.6+5c.9)"
```

---

### Task 15: Clone audit + RIR visitor decision (5c.8 + 5c.10)

**Files:** Documentation + possible small changes

- [ ] **Step 1: Profile-guided clone audit**

This is an *assessment*, not a mass removal. Check the specific areas noted in the spec:

1. **xlog-prob sample store**: Look at `clone_state` in `mc/sampling.rs` — is it called per-sample? If so, document cost.
2. **xlog-logic AST lowering**: Check `.clone()` calls in `lower.rs` during rule compilation.
3. **Schema cloning post-provider split**: Check if schemas are cloned unnecessarily in provider submodules.

For each:
- If the clone is necessary (ownership transfer, borrow checker) → add brief comment `// Deliberate: <reason>`
- If the clone can be eliminated cheaply → fix it
- If the clone matters for performance but needs deeper refactoring → document, do not fix

- [ ] **Step 2: RIR visitor trait decision**

Audit RirNode traversal patterns:

```bash
rg 'RirNode' --type rust -l
```

Count distinct match-dispatch patterns across:
- `executor/` (Wave 3 split)
- `xlog-prob/` (MC, exact eval)
- `xlog-logic/` (lowering, optimization)

**If 3+ distinct traversal patterns exist:** Design the visitor trait in a follow-up issue. Do NOT implement in this wave.

**If only executor uses match dispatch:** Document: "RIR visitor trait not warranted. Match dispatch in executor is explicit and greppable. Revisit if new consumers emerge."

- [ ] **Step 3: Document decision**

Add a brief note to the commit message recording the clone audit results and RIR visitor decision.

- [ ] **Step 4: Verify compilation**

```bash
cargo check --workspace --exclude pyxlog
```

- [ ] **Step 5: Commit**

```bash
git add -u
git commit -m "refactor: clone audit + RIR visitor trait decision (5c.8+5c.10)"
```

---

### Task 16: Final gate (5c gate — Wave 5 completion)

**Files:** None

- [ ] **Step 1: Rust workspace tests**

```bash
cargo test --workspace --all-targets --exclude pyxlog --release
```

- [ ] **Step 2: CUDA certification 206/206**

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

- [ ] **Step 3: Non-slow batch**

```bash
.venv/bin/python -m pytest python/tests/ -x --timeout=120 -k "not slow" --ignore=python/tests/test_ilp_ga_reliability.py --ignore=python/tests/test_ilp_reliability.py --ignore=python/tests/test_ilp_sparse.py --ignore=python/tests/test_ilp_performance.py
```

- [ ] **Step 4: ILP reliability 20/20**

```bash
.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -x --timeout=600
```

- [ ] **Step 5: ILP sparse 8/8**

```bash
.venv/bin/python -m pytest python/tests/test_ilp_sparse.py -x --timeout=120
```

- [ ] **Step 6: GA reliability**

```bash
.venv/bin/python -m pytest python/tests/test_ilp_ga_reliability.py -x --timeout=600
```

- [ ] **Step 7: ILP performance**

```bash
.venv/bin/python -m pytest python/tests/test_ilp_performance.py -x --timeout=600
```

- [ ] **Step 8: pyxlog compile**

```bash
cargo check -p pyxlog
```

- [ ] **Step 9: Python wheel build**

```bash
maturin develop --release -m crates/pyxlog/Cargo.toml
```

- [ ] **Step 10: Document final metrics**

Record:
- Final line counts for gpu_d4/*.rs and mc/*.rs
- Net LOC reduction across the wave
- Visibility tightening count (pub → pub(crate))
- Unwrap/expect final count in production code
- Clone audit results
- RIR visitor decision
