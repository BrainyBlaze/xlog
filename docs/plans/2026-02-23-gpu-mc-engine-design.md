# GPU-Complete Monte Carlo Engine Design

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:writing-plans to produce
> an implementation plan from this design.

**Goal:** Replace the per-sample host loop in the MC engine with batched
world-tagged GPU execution, achieving ≥10x throughput improvement and removing
all hot-path host-device round-trips.

**Architecture:** Add a hidden `world_id` column (appended as last column) to
MC-path buffers. Run the Executor once per batch of worlds with world-isolated
operator semantics. Accumulate query/evidence counts on GPU with a single
host read at the API boundary.

**Branch:** `v0.4.0-alpha-integrated` (continuing current worktree)

---

## Current State (Problem)

The MC engine (`crates/xlog-prob/src/mc.rs`) is GPU-assisted but not
GPU-complete. The main bottleneck is the per-sample host loop at line 741:

```rust
for sample_idx in 0..cfg.samples {
    executor.reset_for_mc();                  // host HashMap clear
    restore_store(&provider, &base_store, ..); // device clone per relation
    build_sample_buffers(..);                  // kernel + DTOH per table
    evaluate_program_gpu(..);                  // GPU joins/unions
    on_sample(..);                             // GPU truth check + accumulate
}
```

Per sample this does:
- O(relations) device-to-device copies (restore_store)
- O(prob_tables) kernel launches + DTOH row count checks (build_sample_buffers)
- Full program evaluation (GPU, but dispatched per-world from host)
- Query/evidence truth check + accumulate (GPU kernels)

For 50k samples this is 50k host-device round-trip cycles. The actual GPU
computation per sample is fast; the overhead is host orchestration and
device memory management churn.

Additional hot-path issues:
- `compact_buffer_by_device_mask_device_count` at `provider.rs:5142` has a
  `device.synchronize()` that blocks the host on every compaction call
- `device_row_count_u32` at `mc.rs:1065` does a DTOH copy per call site
- `CudaBuffer::is_empty()` checks `row_cap` (capacity), not the device-resident
  actual row count — after counted compaction, `row_cap > 0` while actual rows
  may be 0

## Phase 0: Immediate Stabilization

Quick wins with no architectural changes. Estimated effort: 1 day.

### 0.1 Mark heavy MC tests `#[ignore]`

Tests that hang or take >10s move behind `#[ignore]`:
- `crates/xlog-prob/tests/mc.rs` — all tests (50k-80k samples)
- `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs` — calls `evaluate_cpu()` explicitly

Keep `cdcl_q2_status.rs` as separate slow suite (not an MC issue).

### 0.2 Add fast MC smoke tests

New tests with 100-500 samples exercising the GPU path:
- Probabilistic fact marginal
- Evidence conditioning
- Annotated disjunction
- Non-monotone program

Target: each test completes in <2s.

### 0.3 Remove hot-path synchronize in compaction

Remove `self.device.synchronize()?` at `provider.rs:5142` in
`compact_buffer_by_device_mask_device_count`. Same-stream kernel ordering
guarantees correctness without an explicit sync.

### 0.4 Replace per-sample DTOH with device-side zero-row handling

The `device_row_count_u32` calls in `build_sample_buffers` (mc.rs:1266,
mc.rs:1318) do a DTOH per table per sample. Replace with device-side
handling:

- Launch a device kernel that checks the device-resident `d_num_rows` and
  writes a device-side skip flag
- Or: remove the zero-row early-exit and let `dedup_relation` handle empty
  inputs (it already does at mc.rs:1077)

**Important:** Do NOT blindly skip the zero-check. `CudaBuffer::is_empty()`
checks `row_cap` (capacity), not the actual device-resident row count. After
counted compaction, `row_cap` can be nonzero while actual rows are 0. The
replacement must use device-side count, not `is_empty()`.

### 0.5 Add RNG sample offset

Add `sample_offset: u64` parameter to `mc_sample_bernoulli` kernel:

```c
extern "C" __global__ void mc_sample_bernoulli(
    uint8_t* out,
    const float* probs,
    uint32_t num_vars,
    uint32_t num_samples,
    uint64_t seed,
    uint64_t sample_offset  // NEW
) {
    // ...
    // Replace: const uint64_t x = splitmix64(seed ^ (tid * ...));
    // With: global_idx = sample_offset * num_vars + tid;
    //       x = splitmix64(seed ^ (global_idx * 0x9e3779b97f4a7c15ULL));
}
```

And update `sample_bernoulli_matrix_device` in `provider.rs:1227` to accept
and pass through the offset. This enables per-batch sampling with
deterministic global indexing.

**Alternative:** Generate the full sample matrix once (all samples) and slice
per batch. Simpler but uses O(samples × num_vars) memory upfront. Use this
approach if total matrix fits in budget; otherwise use offset.

## Phase 1: World-Tagged Executor

The core architectural change. Replace per-sample loop with single-pass
batched execution.

### 1.1 World ID column placement

**world_id is appended as the last column**, not column 0. Rationale:
compiled column indices in Executor operators (`executor.rs:623`) reference
positions 0..arity-1. Prepending world_id at index 0 would shift every
existing index and break semantics. Appending as last column leaves existing
indices unchanged.

The world_id column:
- Type: `u32`
- Position: `schema.arity()` (last column after append)
- Present only in MC-path buffers; normal execution has no world_id column

### 1.2 Execution context

Add to `crates/xlog-runtime/src/executor.rs`:

```rust
pub enum ExecutionContext {
    Normal,
    MonteCarlo { world_col: usize },
}
```

The Executor gains a `context: ExecutionContext` field. When
`MonteCarlo { world_col }`, operators automatically include `world_col` in
their key/isolation logic.

### 1.3 World-aware operator semantics

When `context` is `MonteCarlo { world_col }`:

| Operator | Behavior |
|----------|----------|
| **Join** | Prepend `world_col` to join keys. Both sides must have matching world_id. |
| **GroupBy** | Prepend `world_col` to grouping keys. Aggregates are per-world. |
| **Distinct/Dedup** | Include `world_col` in key set. |
| **Diff/Semi/Anti** | World-scoped comparisons (same world_id on both sides). |
| **Scan** | Return the world-tagged relation as-is. |
| **Project** | Preserve `world_col` through projection (always include it). |
| **Union** | Concatenate (world_id preserved naturally). |

### 1.4 Deterministic fact broadcast

Deterministic EDB facts are loaded once without world_id. When a world-tagged
probabilistic relation joins against a deterministic EDB, the EDB side has no
world_id column.

Strategy: **Implicit broadcast in join operator.** When `MonteCarlo` context
is active and only one side of a join has the world_id column:

- The side without world_id is treated as broadcast (matches all world_ids)
- Join output inherits world_id from the tagged side
- Implementation: the join kernel already matches on key columns. By not
  including world_id in the untagged side's keys, each EDB row matches every
  world_id. The output world_id is projected from the tagged side.

This avoids physically replicating EDB data across worlds.

### 1.5 Batch materialization

Replace per-sample `build_sample_buffers` with batched materialization:

For each probabilistic table with N rows and W worlds in the batch:
1. Launch `mc_materialize_batch(sample_matrix, var_indices, N, W,
   sample_offset, world_id_out, mask_out)` — produces N×W candidate rows,
   each tagged with `world_id`, masked by that world's sample bits
2. Compact once to get active rows across all worlds
3. Append world_id column to the compacted buffer

Same for annotated disjunction tables with the AD mask variant.

New kernel in `kernels/mc_eval.cu`:
```c
extern "C" __global__ void mc_materialize_batch(
    const uint8_t* sample_matrix, // [num_samples × num_vars]
    const uint32_t* var_indices,  // [N] — which var each row depends on
    uint32_t N,                   // rows in base table
    uint32_t W,                   // worlds in this batch
    uint32_t sample_offset,       // global sample index offset
    uint32_t* world_id_out,       // [N*W] — output world_id per candidate row
    uint8_t* mask_out             // [N*W] — 1 if row is active in this world
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= N * W) return;
    uint32_t row = gid % N;
    uint32_t world = gid / N;
    uint32_t global_sample = sample_offset + world;
    uint32_t var = var_indices[row];
    uint32_t bit_idx = global_sample * num_vars + var; // needs num_vars param
    mask_out[gid] = sample_matrix[bit_idx] ? 1u : 0u;
    world_id_out[gid] = world;
}
```

### 1.6 Batch loop

```rust
let batch_size = cfg.world_batch_size; // default 1024
for batch_start in (0..cfg.samples).step_by(batch_size) {
    let W = min(batch_size, cfg.samples - batch_start);

    // 1. Materialize all prob/AD tables for worlds [batch_start..batch_start+W]
    let world_tagged_tables = materialize_batch(
        &provider, &sample_matrix, &prob_tables, &ad_tables,
        batch_start, W,
    )?;

    // 2. Set up world-tagged store
    executor.set_context(ExecutionContext::MonteCarlo {
        world_col: schema.arity(), // last column
    });
    for (pred, buf) in world_tagged_tables {
        // Merge with broadcast EDB (no world_id) already in store
        executor.put_relation(&pred, buf);
    }

    // 3. Execute program ONCE for all W worlds
    evaluate_program_gpu(&provider, &mut executor, &plan, ..)?;

    // 4. Accumulate query/evidence counts on GPU
    accumulate_batch_counts(
        &provider, &executor, &plan,
        &mut d_query_counts, &mut d_evidence_count,
        W,
    )?;

    // 5. Reset for next batch
    executor.reset_for_mc();
    restore_deterministic_store(&provider, &base_store, &mut executor)?;
}

// Single DTOH at the end
provider.device().synchronize()?;
let query_counts = provider.device().inner()
    .dtoh_sync_copy(&d_query_counts)?;
let evidence_count = provider.device().inner()
    .dtoh_sync_copy(&d_evidence_count)?;
```

### 1.7 McEvalConfig changes

Add to `McEvalConfig`:
```rust
pub world_batch_size: usize, // default 1024
```

RNG determinism: sample matrix indexed by global `batch_start + local_idx`,
so results are identical regardless of `world_batch_size`.

## Phase 2: Query/Evidence Accumulation

After program evaluation, query relations have rows tagged with `world_id`.

### 2.1 World-partitioned count kernel

New kernel `mc_count_query_worlds`:
```c
extern "C" __global__ void mc_count_query_worlds(
    const uint32_t* world_ids,    // world_id column of query relation
    uint32_t num_rows,            // rows in query relation
    uint32_t num_worlds,          // W for this batch
    uint8_t* out_present          // [num_worlds] — 1 if query holds in world
) {
    uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= num_rows) return;
    uint32_t wid = world_ids[gid];
    if (wid < num_worlds) {
        out_present[wid] = 1u;
    }
}
```

### 2.2 Evidence evaluation

Same approach — for each evidence atom, check if its relation has ≥1 row per
world. Compare against expected truth value. A world's evidence is satisfied
only if all evidence atoms match.

### 2.3 Accumulation

Per batch, produce a `[W]` bitmap of evidence-ok worlds and `[num_queries × W]`
bitmap of query-true worlds. Launch `mc_accumulate_counts` to add to running
device-side counters. Single DTOH at the end.

## Phase 3: Non-Monotone SCC Support

### 3.1 Fallback first

Initially, keep the per-sample fallback for programs with non-monotone SCCs
(`mc.rs:1405`). The batched path activates only for monotone programs.

Detection: `nonmonotone_sccs.is_empty()` already computed in `build_gpu_plan`.

### 3.2 Future: world-tagged fixpoint

When extending to non-monotone:
- State snapshots are world-tagged (all worlds' state in one buffer)
- Fixpoint detection per world: `state_signature` keyed by world_id
- Cycle/intersection semantics apply per-world independently
- A world that converges can be marked done while others iterate

This is deferred to a follow-up design.

## Acceptance Gates

1. No `for sample_idx in 0..cfg.samples` in GPU MC path (monotone programs)
2. No hot-path DTOH/synchronize in MC evaluator internals
3. Default MC tests finish in <5s and are deterministic
4. GPU MC throughput improves by ≥10x on 50k samples (measured on RTX PRO 3000)
5. CPU reference (`evaluate_cpu`) only runs with `#[ignore]` or explicit feature flag
6. Results are bit-identical regardless of `world_batch_size` (RNG determinism)
7. No cross-world bleed in any operator (regression tests)

## Files Modified

| File | Change |
|------|--------|
| `crates/xlog-prob/src/mc.rs` | Replace per-sample loop with batched world-tagged execution |
| `crates/xlog-runtime/src/executor.rs` | Add `ExecutionContext`, world-aware operator dispatch |
| `crates/xlog-cuda/src/provider.rs` | Remove hot-path sync; add `sample_offset` to sampler; batch materialization |
| `kernels/mc_sample.cu` | Add `sample_offset` parameter |
| `kernels/mc_eval.cu` | New kernels: `mc_materialize_batch`, `mc_count_query_worlds` |
| `crates/xlog-prob/tests/mc.rs` | Mark heavy tests `#[ignore]`, add fast smoke tests |
| `crates/xlog-prob/tests/gpu_mc_vs_cpu.rs` | Mark `#[ignore]` |
| `crates/xlog-prob/tests/mc_world_isolation.rs` | New: cross-world bleed regression tests |
| `crates/xlog-prob/tests/mc_batch_determinism.rs` | New: verify batch_size doesn't affect results |

## Rollout Order

1. **Phase 0** — stabilize tests, remove hot-path sync, add RNG offset
2. **Phase 1** — world-aware operators for monotone programs
3. **Phase 2** — world-partitioned accumulation
4. **Phase 3** — non-monotone SCC (deferred, fallback for now)
