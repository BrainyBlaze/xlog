# GPU Provenance + GPU PIR + GPU CNF (Phase 3) Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement full GPU-native provenance extraction, device-resident PIR interning, and GPU CNF encoding with deterministic, memory-bounded behavior and zero host data-plane transfers.

**Architecture:** Extend `xlog-runtime` with a provenance-aware execution mode that carries `prov_id` in device buffers and uses a GPU PIR interner. Produce a canonical `GpuPirGraph` on device and encode CNF with GPU Tseitin kernels, returning a `GpuCnfEncoding` suitable for GPU D4/CDCL. All steps are deterministic and fail-fast on overflow.

**Tech Stack:** Rust (xlog-runtime, xlog-prob, xlog-cuda, xlog-solve), CUDA C++ kernels (new `pir.cu`, `cnf.cu`), existing GPU sort/dedup/scan kernels.

---

## Worktree
- Branch: `feature/gpu-pir-cnf-phase3`
- Worktree: `/home/dev/xlog/.worktrees/feature/gpu-pir-cnf-phase3`

Baseline verification (already run in worktree):
- `cargo build -q` => PASS
- `cargo test -q` => PASS

## Implementation Status (2026-01-28)
- **Completed:** Task 1 (GPU PIR layout + tests), Task 2 (PIR kernel module plumbing), Task 3 (GPU PIR interner),
  Task 6 (GPU CNF encoder + tests).
- **Pending:** Task 4 (provenance-aware runtime execution mode) and Task 5 (GPU provenance extraction path).

---

## Task 1: Add GPU PIR data structures + tests

**Files:**
- Create: `crates/xlog-prob/src/compilation/gpu_pir.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Test: `crates/xlog-prob/tests/gpu_pir_layout.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn gpu_pir_layout_matches_cpu_nodes() {
    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.neg_lit(LeafId::new(1));
    let root = pir.and(vec![a, b]);

    let Some(provider) = test_utils::try_provider() else { return; };
    let gpu = GpuPirGraph::from_host(&pir, &provider).expect("from_host");

    let node_type = provider.dtoh_copy(&gpu.node_type).unwrap();
    assert_eq!(node_type.len(), 3);
    assert_eq!(node_type[0], PIR_LIT);
    assert_eq!(node_type[1], PIR_NEG_LIT);
    assert_eq!(node_type[2], PIR_AND);
}
```

**Step 2: Run test to verify it fails**
Run: `cargo test -p xlog-prob gpu_pir_layout_matches_cpu_nodes -q`
Expected: FAIL (missing `GpuPirGraph` / `from_host`).

**Step 3: Implement GPU PIR types**
Implement `GpuPirGraph` and `GpuPirRoots` exactly per design (SoA + CSR), constants `PIR_*`. Add `from_host` for tests only (upload). Ensure invariants enforced and sizes validated.

**Step 4: Run test to verify it passes**
Run: `cargo test -p xlog-prob gpu_pir_layout_matches_cpu_nodes -q`
Expected: PASS.

**Step 5: Commit**
```
git add crates/xlog-prob/src/compilation/gpu_pir.rs crates/xlog-prob/src/compilation/mod.rs crates/xlog-prob/tests/gpu_pir_layout.rs
git commit -m "feat(prob): add gpu pir graph layout"
```

---

## Task 2: Add GPU PIR kernel module plumbing

**Files:**
- Create: `kernels/pir.cu`
- Create: `kernels/pir.ptx`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Test: `crates/xlog-cuda/tests/pir_provider_tests.rs`

**Step 1: Write failing provider test**

```rust
#[test]
fn test_provider_loads_pir_module_entrypoints() {
    let Some(provider) = setup_provider() else { return; };
    let device = provider.device().inner();
    assert!(device.get_func(PIR_MODULE, pir_kernels::PIR_PACK_KEYS).is_some());
    assert!(device.get_func(PIR_MODULE, pir_kernels::PIR_HASH_KEYS).is_some());
    assert!(device.get_func(PIR_MODULE, pir_kernels::PIR_MARK_UNIQUE).is_some());
}
```

**Step 2: Implement kernel stubs (real entrypoints)**
Create `kernels/pir.cu` with real kernels (no stubs) for:
- `pir_pack_keys`
- `pir_hash_keys`
- `pir_mark_unique`

Keep signatures stable and usable by interner. Build to `pir.ptx` via build.rs.

**Step 3: Wire provider**
Load PTX and expose kernel names in `xlog-cuda` provider. Add test module.

**Step 4: Run test**
`cargo test -p xlog-cuda pir_provider_tests -q`

**Step 5: Commit**
```
git add kernels/pir.cu kernels/pir.ptx crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider/mod.rs crates/xlog-cuda/tests/pir_provider_tests.rs
git commit -m "feat(cuda): add pir kernel module plumbing"
```

---

## Task 3: Implement GPU PIR interner (deterministic, memory-bounded)

**Files:**
- Create: `crates/xlog-prob/src/compilation/gpu_pir_intern.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs` (helpers for stable sort + dedup on packed keys)
- Test: `crates/xlog-prob/tests/gpu_pir_intern.rs`

**Step 1: Write failing test**

```rust
#[test]
fn gpu_pir_intern_is_deterministic() {
    let Some(provider) = test_utils::try_provider() else { return; };
    let mut interner = GpuPirInterner::new(&provider, 1024, 4096).unwrap();

    let batch = PirBatch::and_or_batch(vec![vec![1, 2], vec![2, 1]]); // same set
    let ids_a = interner.intern_batch(&batch).unwrap();
    let ids_b = interner.intern_batch(&batch).unwrap();

    let a = provider.dtoh_copy(&ids_a).unwrap();
    let b = provider.dtoh_copy(&ids_b).unwrap();
    assert_eq!(a, b);
    assert_eq!(a[0], a[1]); // canonicalized
}
```

**Step 2: Run test to verify it fails**
`cargo test -p xlog-prob gpu_pir_intern_is_deterministic -q`
Expected: FAIL (missing interner).

**Step 3: Implement interner**
- Represent PIR node keys in SoA (tag, payload, child_range)
- Sort children by `(parent_id, child_id)` using existing radix sort
- Hash per-node keys on GPU (`pir_hash_keys`)
- Stable sort candidate nodes by `(hash, tag, payload, len)`
- Use `pir_mark_unique` to compare keys + children for equality
- Exclusive scan to assign IDs
- Append to `GpuPirGraph` pools with capacity checks
- Return `DeviceSlice<u32>` of interned ids

No atomics for ID assignment; overflow => device trap.

**Step 4: Run test to verify it passes**
`cargo test -p xlog-prob gpu_pir_intern_is_deterministic -q`

**Step 5: Commit**
```
git add crates/xlog-prob/src/compilation/gpu_pir_intern.rs crates/xlog-prob/src/compilation/mod.rs crates/xlog-prob/tests/gpu_pir_intern.rs crates/xlog-cuda/src/provider/mod.rs
git commit -m "feat(prob): add gpu pir interner"
```

---

## Task 4: Add provenance-aware runtime execution mode

**Files:**
- Modify: `crates/xlog-runtime/src/executor.rs`
- Modify: `crates/xlog-runtime/src/lib.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `kernels/dedup.cu` (dedup with OR-reduce)
- Test: `crates/xlog-runtime/tests/provenance_runtime_tests.rs`

**Step 1: Write failing test**

```rust
#[test]
fn provenance_join_emits_and() {
    let Some((mut exec, provider)) = create_executor_with_provenance() else { return; };
    // two relations with prov_id
    let a = create_buf_with_prov(&provider, &[(1, 10, 3), (2, 20, 7)]); // (x, y, prov)
    let b = create_buf_with_prov(&provider, &[(1, 99, 5)]);
    exec.store_mut().put("a", a);
    exec.store_mut().put("b", b);

    let plan = build_join_plan();
    exec.execute_plan_with_provenance(&plan).unwrap();

    let out = exec.store_mut().remove("__xlog_query_0").unwrap();
    let prov = provider.download_column_u32(&out, out.arity() - 1).unwrap();
    assert_eq!(prov.len(), 1);
}
```

**Step 2: Run test to verify it fails**
`cargo test -p xlog-runtime provenance_join_emits_and -q`
Expected: FAIL (missing provenance mode).

**Step 3: Implement provenance mode**
- Add `ExecutionMode { normal, provenance }` or `Executor::execute_plan_with_provenance`
- Extend buffers with `prov_id` column (u32) in provenance mode
- Join: compute AND of prov_id via new kernel `prov_and` (or reuse decision builder)
- Union/Dedup: add `dedup_reduce_or` that ORs prov_id for duplicate keys
- Diff: preserve left provenance
- Project/Filter: pass through prov_id

**Step 4: Run test to verify it passes**
`cargo test -p xlog-runtime provenance_join_emits_and -q`

**Step 5: Commit**
```
git add crates/xlog-runtime/src/executor.rs crates/xlog-runtime/src/lib.rs crates/xlog-cuda/src/provider/mod.rs kernels/dedup.cu crates/xlog-runtime/tests/provenance_runtime_tests.rs
git commit -m "feat(runtime): add gpu provenance execution mode"
```

---

## Task 5: GPU WFS for non-monotone SCCs

**Files:**
- Create: `crates/xlog-prob/src/wfs_gpu.rs`
- Modify: `crates/xlog-prob/src/lib.rs`
- Modify: `crates/xlog-runtime/src/executor.rs` (WFS hooks)
- Test: `crates/xlog-prob/tests/gpu_wfs_tests.rs`

**Step 1: Write failing test**

```rust
#[test]
fn gpu_wfs_handles_negation_cycle() {
    let Some(provider) = test_utils::try_provider() else { return; };
    let program = "p() :- not q(). q() :- not p().";
    let result = extract_gpu_provenance(program, provider).unwrap();
    assert!(result.queries.is_empty() || result.pir_roots.len() > 0);
}
```

**Step 2: Run test to verify it fails**
`cargo test -p xlog-prob gpu_wfs_handles_negation_cycle -q`

**Step 3: Implement GPU WFS**
- Port `wfs.rs` logic to GPU buffers
- Use runtime executor to evaluate rule bodies with current true/false sets
- Iterate to fixpoint, deterministic ordering
- Emit provenance PIR nodes on GPU for true atoms

**Step 4: Run test to verify it passes**
`cargo test -p xlog-prob gpu_wfs_handles_negation_cycle -q`

**Step 5: Commit**
```
git add crates/xlog-prob/src/wfs_gpu.rs crates/xlog-prob/src/lib.rs crates/xlog-runtime/src/executor.rs crates/xlog-prob/tests/gpu_wfs_tests.rs
git commit -m "feat(prob): add gpu wfs provenance"
```

---

## Task 6: Implement GPU CNF encoder

**Files:**
- Create: `kernels/cnf.cu`
- Create: `kernels/cnf.ptx`
- Modify: `crates/xlog-cuda/build.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Create: `crates/xlog-prob/src/compilation/gpu_cnf.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Test: `crates/xlog-prob/tests/gpu_cnf_encode.rs`

**Step 1: Write failing test**

```rust
#[test]
fn gpu_cnf_matches_cpu_encode() {
    let Some(provider) = test_utils::try_provider() else { return; };
    let (pir, roots) = sample_pir();
    let cpu = encode_cnf(&pir, &roots).unwrap();
    let gpu = encode_cnf_gpu(&GpuPirGraph::from_host(&pir, &provider).unwrap(), &roots, &provider).unwrap();

    assert_eq!(cpu.cnf.num_vars(), provider.dtoh_copy(&gpu.cnf.num_vars).unwrap()[0]);
}
```

**Step 2: Run test to verify it fails**
`cargo test -p xlog-prob gpu_cnf_matches_cpu_encode -q`

**Step 3: Implement GPU CNF encoder**
- Kernels K0–K3 per design: assign vars, count clauses/lits, prefix-sum, emit
- CSR sizing with scan kernels
- Return `GpuCnfEncoding` with var tables on device

**Step 4: Run test to verify it passes**
`cargo test -p xlog-prob gpu_cnf_matches_cpu_encode -q`

**Step 5: Commit**
```
git add kernels/cnf.cu kernels/cnf.ptx crates/xlog-cuda/build.rs crates/xlog-cuda/src/provider/mod.rs crates/xlog-prob/src/compilation/gpu_cnf.rs crates/xlog-prob/src/compilation/mod.rs crates/xlog-prob/tests/gpu_cnf_encode.rs
git commit -m "feat(prob): add gpu cnf encoder"
```

---

## Task 7: End-to-end GPU provenance -> PIR -> CNF API

**Files:**
- Create: `crates/xlog-prob/src/gpu_provenance.rs`
- Modify: `crates/xlog-prob/src/lib.rs`
- Test: `crates/xlog-prob/tests/gpu_provenance_to_cnf.rs`

**Step 1: Write failing test**

```rust
#[test]
fn gpu_provenance_produces_cnf() {
    let Some(provider) = test_utils::try_provider() else { return; };
    let program = "0.6::a(). b() :- a().";
    let out = extract_gpu_provenance_cnf(program, provider).unwrap();
    assert!(out.cnf.var_cap > 0);
}
```

**Step 2: Run test to verify it fails**
`cargo test -p xlog-prob gpu_provenance_produces_cnf -q`

**Step 3: Implement GPU provenance pipeline API**
- Execute program in provenance mode using runtime
- Build `GpuPirGraph` and `GpuPirRoots` on device
- Encode CNF on device
- Return `GpuCnfEncoding`

**Step 4: Run test to verify it passes**
`cargo test -p xlog-prob gpu_provenance_produces_cnf -q`

**Step 5: Commit**
```
git add crates/xlog-prob/src/gpu_provenance.rs crates/xlog-prob/src/lib.rs crates/xlog-prob/tests/gpu_provenance_to_cnf.rs
git commit -m "feat(prob): add gpu provenance to cnf pipeline"
```

---

## Task 8: CUDA certification + docs

**Files:**
- Modify: `crates/xlog-cuda-tests/src/` (new certification cases)
- Modify: `crates/xlog-gpu/tests/v032_gpu_certification.rs`
- Modify: `docs/certification/*.md`
- Modify: `docs/ROADMAP.md`
- Modify: `CHANGELOG.md`

**Step 1: Add certification tests**
- PIR interning determinism
- GPU CNF encoding correctness
- GPU WFS non-monotone correctness
- Zero host reads in GPU provenance/CNF

**Step 2: Run certification subset**
`cargo test -p xlog-cuda-tests -q`
`cargo test -p xlog-gpu v032_gpu_certification -q`

**Step 3: Update docs**
Update certification report, roadmap, changelog.

**Step 4: Commit**
```
git add crates/xlog-cuda-tests crates/xlog-gpu/tests docs/certification docs/ROADMAP.md CHANGELOG.md
git commit -m "test(cuda): certify gpu provenance and cnf"
```

---

## Final Verification
Run:
```
cargo fmt --all --check
cargo test --workspace -q
cargo test --workspace --release -q
cargo build -p pyxlog -q
cd target/debug && ln -sf libpyxlog.so pyxlog.so && cd -
PYTHONPATH=target/debug python3 -m pytest python/tests -q
```

---

**Plan complete and saved to** `docs/plans/2026-01-27-gpu-pir-cnf-phase3.md`.

Two execution options:

1. Subagent-Driven (this session)
2. Parallel Session (separate)

Which approach?
