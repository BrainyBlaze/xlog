# GPU CNF Builder Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement GPU-native Tseitin CNF encoding from GPU PIR (root-reachable subgraph only) with deterministic variable numbering and fully device-resident CSR output.

**Architecture:** Use a device-side reachability worklist from roots, then assign leaf/choice/node variables on GPU via prefix scans, count clauses/literals per node, prefix-sum into CSR offsets, and emit CNF clauses on GPU. Host only allocates buffers and launches kernels; CNF data never leaves device.

**Tech Stack:** Rust (xlog-prob/xlog-solve), CUDA kernels in `kernels/cnf.cu`, cudarc, existing scan kernels in `kernels/scan.cu`.

**Status (2026-01-28): Implemented** — `encode_cnf_gpu` in `crates/xlog-prob/src/compilation/gpu_cnf.rs`, kernels in
`kernels/cnf.cu` (PTX module `xlog_cnf`), device-resident counts and CSR emission, reachability worklist hardened with
queue-ready + in-flight guards, tests in `crates/xlog-prob/tests/gpu_cnf.rs`.

---

### Task 1: GPU CNF tests (CPU vs GPU equivalence)

**Files:**
- Create: `crates/xlog-prob/tests/gpu_cnf.rs`

**Step 1: Write the failing tests**

```rust
use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::MemoryBudget;
use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};
use xlog_prob::cnf::encode_cnf;
use xlog_prob::compilation::{encode_cnf_gpu, GpuPirGraph, GpuPirRoots};
use xlog_prob::pir::{ChoiceVarId, LeafId, PirGraph};

fn try_provider() -> Option<Arc<CudaKernelProvider>> {
    let device = match CudaDevice::new(0) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            eprintln!("Skipping test: CUDA runtime unavailable: {}", e);
            return None;
        }
    };
    let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
    let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
    match CudaKernelProvider::new(device, memory) {
        Ok(p) => Some(Arc::new(p)),
        Err(e) => {
            eprintln!("Skipping test: failed to create provider: {}", e);
            None
        }
    }
}

fn canonicalize(clauses: Vec<Vec<i32>>) -> Vec<Vec<i32>> {
    let mut out: Vec<Vec<i32>> = clauses
        .into_iter()
        .map(|mut c| {
            c.sort();
            c
        })
        .collect();
    out.sort();
    out
}

fn gpu_cnf_to_host(
    provider: &Arc<CudaKernelProvider>,
    cnf: &xlog_solve::GpuCnf,
) -> (u32, Vec<Vec<i32>>) {
    let device = provider.device().inner();
    let mut num_vars = [0u32; 1];
    let mut num_clauses = [0u32; 1];
    let mut num_lits = [0u32; 1];
    device.dtoh_sync_copy_into(&cnf.num_vars, &mut num_vars).unwrap();
    device.dtoh_sync_copy_into(&cnf.num_clauses, &mut num_clauses).unwrap();
    device.dtoh_sync_copy_into(&cnf.num_lits, &mut num_lits).unwrap();

    let clauses_len = num_clauses[0] as usize;
    let lits_len = num_lits[0] as usize;
    let mut offsets = vec![0u32; clauses_len + 1];
    let mut lits = vec![0i32; lits_len];

    let offsets_view = cnf.clause_offsets.slice(0..(clauses_len + 1));
    let lits_view = cnf.literals.slice(0..lits_len);
    device.dtoh_sync_copy_into(&offsets_view, &mut offsets).unwrap();
    device.dtoh_sync_copy_into(&lits_view, &mut lits).unwrap();

    let mut clauses = Vec::with_capacity(clauses_len);
    for i in 0..clauses_len {
        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        clauses.push(lits[start..end].to_vec());
    }

    (num_vars[0], clauses)
}

#[test]
fn gpu_cnf_matches_cpu_encoding_simple() {
    let Some(provider) = try_provider() else { return; };

    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.neg_lit(LeafId::new(1));
    let and = pir.and(vec![a, b]);
    let t = pir.const_true();
    let f = pir.const_false();
    let dec = pir.decision(ChoiceVarId::new(0), f, t);
    let root = pir.or(vec![and, dec]);

    let cpu = encode_cnf(&pir, &[root]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[root], &provider).unwrap();
    let gpu = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let (gpu_vars, gpu_clauses) = gpu_cnf_to_host(&provider, &gpu.cnf);

    assert_eq!(gpu_vars, cpu.cnf.num_vars());
    assert_eq!(
        canonicalize(gpu_clauses),
        canonicalize(cpu.cnf.clauses().to_vec())
    );
}

#[test]
fn gpu_cnf_prunes_unreachable_nodes() {
    let Some(provider) = try_provider() else { return; };

    let mut pir = PirGraph::new();
    let a = pir.lit(LeafId::new(0));
    let b = pir.lit(LeafId::new(1));
    let r1 = pir.and(vec![a]);
    let _r2 = pir.or(vec![b]);

    let cpu = encode_cnf(&pir, &[r1]).unwrap();

    let gpu_pir = GpuPirGraph::from_host(&pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[r1], &provider).unwrap();
    let gpu = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let (gpu_vars, gpu_clauses) = gpu_cnf_to_host(&provider, &gpu.cnf);
    assert_eq!(gpu_vars, cpu.cnf.num_vars());
    assert_eq!(
        canonicalize(gpu_clauses),
        canonicalize(cpu.cnf.clauses().to_vec())
    );
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test -p xlog-prob --test gpu_cnf gpu_cnf_matches_cpu_encoding_simple -q`

Expected: FAIL (missing `encode_cnf_gpu` or not implemented).

**Step 3: Commit test**

```bash
git add crates/xlog-prob/tests/gpu_cnf.rs
git commit -m "test(gpu-cnf): add GPU vs CPU CNF equivalence tests"
```

---

### Task 2: Implement GPU CNF encoding (reachability + vars + clauses)

**Files:**
- Create: `crates/xlog-prob/src/compilation/gpu_cnf.rs`
- Modify: `crates/xlog-prob/src/compilation/mod.rs`
- Create: `kernels/cnf.cu`
- Modify: `kernels/CMakeLists.txt`
- Generate: `kernels/cnf.ptx`
- Modify: `crates/xlog-cuda/src/provider.rs`

**Step 1: Write minimal public API in `gpu_cnf.rs`**

```rust
pub struct GpuCnfVarTables {
    pub node_var: TrackedCudaSlice<u32>,
    pub leaf_var: TrackedCudaSlice<u32>,
    pub choice_var: TrackedCudaSlice<u32>,
    pub max_var: u32,
}

pub struct GpuCnfEncoding {
    pub cnf: xlog_solve::GpuCnf,
    pub vars: GpuCnfVarTables,
}

pub fn encode_cnf_gpu(
    pir: &GpuPirGraph,
    roots: &GpuPirRoots,
    provider: &Arc<CudaKernelProvider>,
) -> Result<GpuCnfEncoding> { /* implemented in Step 3 */ }
```

**Step 2: Add CNF kernel module plumbing in provider**

- Add `const CNF_PTX` include + `CNF_MODULE` name.
- Register kernel names for:
  - `cnf_reachability_init`
  - `cnf_reachability_bfs`
  - `cnf_mark_leaf_choice`
  - `cnf_assign_leaf_var`
  - `cnf_assign_choice_var`
  - `cnf_mark_node_vars`
  - `cnf_assign_node_var`
  - `cnf_count_clauses`
  - `cnf_capture_last_counts`
  - `cnf_compute_leaf_choice_totals`
  - `cnf_compute_totals`
  - `cnf_emit_clauses`
  - `cnf_set_clause_end`

**Step 3: Implement kernels in `kernels/cnf.cu`**

- **Reachability worklist**:
  - `cnf_reachability_init(roots, num_roots, reachable, queue, head, tail)`
  - `cnf_reachability_bfs(node_type, child_offsets, children, dec_f, dec_t, num_nodes, reachable, queue, head, tail)`
  - Worklist logic: pop from queue using atomic head; push unseen children (atomicExch on reachable). Trap if tail > num_nodes.

- **Leaf/choice marking**:
  - `cnf_mark_leaf_choice(node_type, leaf_id, decision_var, reachable, num_nodes, leaf_used, choice_used)`

- **Assign leaf/choice vars**:
  - Use `scan_u8_mask_device` to compute prefix for `leaf_used` and `choice_used`.
  - Use kernels to write `leaf_var[id] = prefix[id] + 1` and `choice_var[id] = prefix[id] + (1 + num_leaf)`.

- **Assign node vars**:
  - `cnf_mark_node_vars(node_type, reachable, num_nodes, node_needs_var)` (true for reachable and non‑LIT).
  - Prefix scan to assign `node_var[node] = base_node + prefix[node]`.
  - For LIT nodes, set `node_var[node] = leaf_var[leaf_id]`.

- **Count clauses/lits**:
  - `cnf_count_clauses(node_type, child_offsets, reachable, num_nodes, clause_counts, lit_counts)` implementing design formulas (deg+1, etc.).
  - `cnf_sum_counts` (atomic sum of counts) to compute totals.

- **Emit clauses**:
  - `cnf_emit_clauses(..., clause_base, lit_base, clause_offsets, literals)` writes per‑node clauses at precomputed offsets.
  - `cnf_set_clause_end(num_clauses, num_lits, clause_offsets)` sets final CSR offset.

**Step 4: Implement `encode_cnf_gpu` in Rust**

- Validate `roots.len() > 0`.
- Allocate device buffers: `reachable`, `queue`, `head`, `tail`, `leaf_used`, `choice_used`, `node_needs_var`, `node_var`, `leaf_var`, `choice_var`, `clause_counts`, `lit_counts`, `clause_base`, `lit_base`.
- Launch reachability init + BFS.
- Assign leaf/choice vars via scans and kernels; compute `num_leaf`/`num_choice` via count mask, then base offsets for node vars.
- Assign node vars.
- Count clauses/lits, scan to compute bases, compute totals on device.
- Allocate `GpuCnf` buffers to host-known capacities; totals remain device-resident (no host reads).
- Emit clauses and set `clause_offsets[num_clauses]=num_lits`.
- Fill `cnf.num_vars/num_clauses/num_lits` device scalars.
- Return `GpuCnfEncoding` with `max_var = var_cap` (actual counts stay device-resident in `cnf.num_vars`).

**Step 5: Run tests**

Run: `cargo test -p xlog-prob --test gpu_cnf gpu_cnf_matches_cpu_encoding_simple -q`
Expected: PASS.

**Step 6: Commit**

```bash
git add crates/xlog-prob/src/compilation/gpu_cnf.rs crates/xlog-prob/src/compilation/mod.rs kernels/cnf.cu kernels/cnf.ptx kernels/CMakeLists.txt crates/xlog-cuda/src/provider.rs
git commit -m "feat(gpu-cnf): add device-side Tseitin encoder"
```

---

### Task 3: Expand tests + verification

**Files:**
- Modify: `crates/xlog-prob/tests/gpu_cnf.rs`

**Step 1: Add a NegLit/Decision/empty AND/OR edge-case test**

```rust
#[test]
fn gpu_cnf_handles_empty_and_or_and_neglit() {
    let Some(provider) = try_provider() else { return; };

    let mut pir = PirGraph::new();
    let a = pir.neg_lit(LeafId::new(0));
    let and0 = pir.and(vec![]);
    let or0 = pir.or(vec![]);
    let root = pir.or(vec![a, and0, or0]);

    let cpu = encode_cnf(&pir, &[root]).unwrap();
    let gpu_pir = GpuPirGraph::from_host(&pir, &provider).unwrap();
    let roots = GpuPirRoots::from_host(&[root], &provider).unwrap();
    let gpu = encode_cnf_gpu(&gpu_pir, &roots, &provider).unwrap();

    let (gpu_vars, gpu_clauses) = gpu_cnf_to_host(&provider, &gpu.cnf);
    assert_eq!(gpu_vars, cpu.cnf.num_vars());
    assert_eq!(
        canonicalize(gpu_clauses),
        canonicalize(cpu.cnf.clauses().to_vec())
    );
}
```

**Step 2: Run tests**

Run: `cargo test -p xlog-prob --test gpu_cnf -q`
Expected: PASS.

**Step 3: Commit**

```bash
git add crates/xlog-prob/tests/gpu_cnf.rs
git commit -m "test(gpu-cnf): cover NegLit and empty And/Or cases"
```

---

**Plan complete and saved to `docs/plans/2026-01-28-gpu-cnf-builder.md`. Two execution options:**

**1. Subagent-Driven (this session)** — I dispatch fresh subagents per task, review between tasks, fast iteration.

**2. Parallel Session (separate)** — Open a new session with executing-plans, batch execution with checkpoints.

**Which approach?**
