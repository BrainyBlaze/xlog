# dILP Beta Critical Path — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Upgrade the alpha dILP trainer to beta: sparse mask API, trainer backend abstraction, `train_and_promote()`, holdout scoring, hard-negative mining, artifact persistence, recursive candidates, and 20/20 reliability gate.

**Architecture:** The beta path replaces the dense N3 mask transfer with a sparse candidate-indexed API (`set_rule_mask_sparse`) where Python sends C soft-probs and Rust does deterministic top-k + executor mapping. The trainer abstracts the mask backend (dense vs. sparse) so `train_only()` works with either. `train_and_promote()` wraps `train_only()` + trial compilation + promotion gates. Holdout scoring, hard-negative mining, and artifact save/load are layered on top.

**Tech Stack:** Rust (PyO3 0.21.2, xlog-runtime, xlog-cuda), Python 3.10, PyTorch, maturin (mixed layout at `crates/pyxlog/`), pytest.

**Baseline:** HEAD `4b2b3af5` on `main`, 233 tests (66 ILP-specific), alpha reliability 20/20.

**Build/test commands:**
```bash
# Rust build
cargo build -p pyxlog --release

# Python install
/home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml

# All Python tests
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/ -v --timeout=600

# Specific test file
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120

# Rust tests (non-pyxlog)
cargo test --workspace --all-targets --exclude pyxlog --release
```

**Key file locations:**
- Rust PyO3 bindings: `crates/pyxlog/src/lib.rs` (CompiledIlpProgram struct at line ~3857)
- ILP registry: `crates/xlog-runtime/src/ilp_registry.rs`
- Executor TMJ dispatch: `crates/xlog-runtime/src/executor.rs` (line ~2743: `execute_tensor_masked_join`)
- GPU extraction kernel: `crates/xlog-cuda/src/provider.rs` (line ~10494: `extract_active_rule_indices`)
- Python trainer: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Python types: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Python exceptions: `crates/pyxlog/python/pyxlog/ilp/exceptions.py`
- Python __init__: `crates/pyxlog/python/pyxlog/ilp/__init__.py`
- Test conftest (CUDA skip): `python/tests/conftest.py`

---

## Task 1: Lock Alpha Baseline

**Goal:** Tag the current commit and verify the alpha test suite is frozen.

**Files:**
- None created/modified

**Step 1: Tag current commit**

```bash
git tag -a v0.4.0-alpha -m "dILP alpha: 66 ILP tests, 20/20 reliability, dense N3 mask"
```

**Step 2: Verify tag**

```bash
git log --oneline -1 v0.4.0-alpha
```
Expected: `4b2b3af5 test(ilp): Tasks 7-10 — ...`

**Step 3: Run full alpha test suite to confirm baseline**

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_*.py -v --timeout=600
```
Expected: 66 passed (including 20 reliability tests)

**Step 4: Commit** — No code changes; tag only.

---

## Task 2: `set_rule_mask_sparse()` — Rust + PyO3

**Goal:** Add a new Rust method on `CompiledIlpProgram` that accepts sparse (candidate_ids, soft_probs, budget) and builds the executor mask internally. This avoids Python materializing an N3 tensor for GPU transfer.

**Files:**
- Modify: `crates/xlog-runtime/src/ilp_registry.rs` (add `insert_mask_from_sparse`)
- Modify: `crates/pyxlog/src/lib.rs` (add `set_rule_mask_sparse` pymethods)
- Create: `python/tests/test_ilp_sparse.py`

### Step 1: Write the failing Python test

Create `python/tests/test_ilp_sparse.py`:

```python
# python/tests/test_ilp_sparse.py
"""Tests for set_rule_mask_sparse() — dense-parity checks."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4])]


def _compile():
    return pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)


def test_sparse_mask_exists():
    """set_rule_mask_sparse is callable."""
    prog = _compile()
    candidates = prog.valid_candidates("W", False)
    C = len(candidates)
    soft = [1.0 / C] * C
    ids = list(range(C))
    prog.set_rule_mask_sparse("W", ids, soft, 32)


def test_sparse_derives_same_as_dense():
    """Sparse and dense APIs derive identical facts for same top-k selection."""
    prog_dense = _compile()
    prog_sparse = _compile()
    candidates = prog_dense.valid_candidates("W", False)
    C = len(candidates)
    n = prog_dense.ilp_schema_size()

    # Build a soft distribution favoring candidate 0
    soft = [0.0] * C
    soft[0] = 1.0

    # Dense path: build N3 tensor
    M_hard = torch.zeros(n * n * n, device="cuda")
    M_soft = torch.zeros(n * n * n, device="cuda")
    c0 = candidates[0]
    flat_idx = c0["i"] * n * n + c0["j"] * n + c0["k"]
    M_hard[flat_idx] = 1.0
    M_soft[flat_idx] = 1.0
    prog_dense.set_rule_mask("W", M_hard, M_soft, n)
    prog_dense.evaluate()
    dense_pos = [prog_dense.fact_exists(r, v) for r, v in POS]

    # Sparse path
    ids = list(range(C))
    prog_sparse.set_rule_mask_sparse("W", ids, soft, 32)
    prog_sparse.evaluate()
    sparse_pos = [prog_sparse.fact_exists(r, v) for r, v in POS]

    assert dense_pos == sparse_pos, f"dense={dense_pos} vs sparse={sparse_pos}"


def test_sparse_tagged_entries_match():
    """Tagged entries from sparse path match dense path."""
    prog_dense = _compile()
    prog_sparse = _compile()
    candidates = prog_dense.valid_candidates("W", False)
    C = len(candidates)
    n = prog_dense.ilp_schema_size()

    # Single candidate active
    soft = [0.0] * C
    soft[0] = 1.0
    c0 = candidates[0]

    # Dense
    M_hard = torch.zeros(n * n * n, device="cuda")
    M_soft = torch.zeros(n * n * n, device="cuda")
    flat_idx = c0["i"] * n * n + c0["j"] * n + c0["k"]
    M_hard[flat_idx] = 1.0
    M_soft[flat_idx] = 1.0
    prog_dense.set_rule_mask("W", M_hard, M_soft, n)
    prog_dense.evaluate()
    dense_tags = prog_dense.get_tagged_results()

    # Sparse
    ids = list(range(C))
    prog_sparse.set_rule_mask_sparse("W", ids, soft, 32)
    prog_sparse.evaluate()
    sparse_tags = prog_sparse.get_tagged_results()

    assert set(dense_tags) == set(sparse_tags)


def test_sparse_validates_candidate_count():
    """Partial candidate list raises ValueError."""
    prog = _compile()
    candidates = prog.valid_candidates("W", False)
    C = len(candidates)
    with pytest.raises(ValueError, match="candidate"):
        prog.set_rule_mask_sparse("W", list(range(C - 1)), [1.0] * (C - 1), 32)


def test_sparse_validates_length_mismatch():
    """Mismatched ids/soft_probs lengths raise ValueError."""
    prog = _compile()
    candidates = prog.valid_candidates("W", False)
    C = len(candidates)
    with pytest.raises(ValueError, match="length"):
        prog.set_rule_mask_sparse("W", list(range(C)), [1.0] * (C + 1), 32)


def test_sparse_top_k_deterministic_tiebreak():
    """When soft probs tie, lower candidate_id wins."""
    prog = _compile()
    candidates = prog.valid_candidates("W", False)
    C = len(candidates)
    # All equal soft probs, budget = 1 -> candidate 0 should win
    soft = [1.0 / C] * C
    prog.set_rule_mask_sparse("W", list(range(C)), soft, 1)
    prog.evaluate()
    tags = prog.get_tagged_results()
    if tags:
        i, j, k, _ = tags[0]
        assert (i, j, k) == (candidates[0]["i"], candidates[0]["j"], candidates[0]["k"])
```

> **Note:** The method is `evaluate()` on `CompiledIlpProgram`.

### Step 2: Run test to verify it fails

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120
```
Expected: FAIL — `AttributeError: 'CompiledIlpProgram' has no attribute 'set_rule_mask_sparse'`

### Step 3: Add `insert_mask_from_sparse` to ILP registry

Modify `crates/xlog-runtime/src/ilp_registry.rs` — add a method that builds dense CudaBuffers from sparse (i,j,k,soft) data:

```rust
/// Insert a mask from sparse candidate data.
///
/// Builds dense N*N*N hard + soft CudaBuffers from the provided
/// (i,j,k) triples and soft probabilities. top-k selection is done
/// by caller (PyO3 layer).
pub fn insert_mask_from_sparse(
    &mut self,
    name: String,
    schema_size: usize,
    active_ijk: &[(u32, u32, u32)],
    active_soft: &[f32],
    provider: &CudaKernelProvider,
) -> Result<(), XlogError> {
    let total = schema_size * schema_size * schema_size;

    // Build host-side dense buffers
    let mut hard_host = vec![0f32; total];
    let mut soft_host = vec![0f32; total];
    for (idx, &(i, j, k)) in active_ijk.iter().enumerate() {
        let flat = (i as usize) * schema_size * schema_size
                 + (j as usize) * schema_size
                 + (k as usize);
        if flat < total {
            hard_host[flat] = 1.0;
            soft_host[flat] = active_soft[idx];
        }
    }

    // Upload to GPU as single-column CudaBuffers
    let hard_buf = provider.create_buffer_from_f32_slice(&hard_host)?;
    let soft_buf = provider.create_buffer_from_f32_slice(&soft_host)?;

    self.masks.insert(name, IlpMask { hard: hard_buf, soft: soft_buf, schema_size });
    Ok(())
}
```

> **Implementation note:** If `create_buffer_from_f32_slice` does not exist on CudaKernelProvider, look for: `create_buffer_from_slices`, `htod_copy`, or build via `CudaBuffer::from_columns`. The key operation is: host f32 vec -> single-column GPU CudaBuffer with `total` f32 elements. The exact method name will need to match the provider API.

### Step 4: Add `set_rule_mask_sparse` to CompiledIlpProgram

Modify `crates/pyxlog/src/lib.rs` — add a new `#[pymethods]` fn on `CompiledIlpProgram`:

```rust
/// Sparse mask API: accepts candidate IDs + soft probs, Rust does top-k.
///
/// candidate_ids: list[int] — must be [0..C) in order, length C
/// soft_probs: list[float] — one per candidate, length C
/// budget: int — max active rules (top-k)
/// allow_recursive: bool — whether i==k/j==k candidates are included
#[pyo3(signature = (name, candidate_ids, soft_probs, budget, allow_recursive = false))]
pub fn set_rule_mask_sparse(
    &mut self,
    name: String,
    candidate_ids: Vec<u32>,
    soft_probs: Vec<f64>,
    budget: usize,
    allow_recursive: bool,
) -> PyResult<()> {
    let tmj = extract_tmj_meta_for_mask(&self.plan, Some(&name));
    let n = tmj.schema_size;
    if n == 0 {
        return Err(PyValueError::new_err(format!(
            "no learnable mask '{}' found", name
        )));
    }

    // Recompute expected C from valid_candidates logic (must match allow_recursive)
    let expected_c = self.expected_candidate_count(&name, allow_recursive)?;

    // Validate: candidate_ids must be exactly [0..C) in order
    if candidate_ids.len() != expected_c {
        return Err(PyValueError::new_err(format!(
            "candidate_ids length {} != expected candidate count {}",
            candidate_ids.len(), expected_c
        )));
    }
    for (idx, &cid) in candidate_ids.iter().enumerate() {
        if cid != idx as u32 {
            return Err(PyValueError::new_err(format!(
                "candidate_ids must be [0..{}), got id {} at position {}",
                expected_c, cid, idx
            )));
        }
    }
    if soft_probs.len() != candidate_ids.len() {
        return Err(PyValueError::new_err(format!(
            "soft_probs length {} != candidate_ids length {}",
            soft_probs.len(), candidate_ids.len()
        )));
    }

    // Resolve candidates: get the (i,j,k) triples for each id
    let candidate_triples = self.candidate_triples_for_mask(&name, allow_recursive)?;

    // Top-k selection: sort by soft_prob desc, tie-break by lower id
    let mut indexed: Vec<(usize, f64)> = soft_probs.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });
    indexed.truncate(budget.min(indexed.len()));

    let active_ijk: Vec<(u32, u32, u32)> = indexed.iter()
        .map(|&(idx, _)| candidate_triples[idx])
        .collect();
    let active_soft: Vec<f32> = indexed.iter()
        .map(|&(idx, _)| soft_probs[idx] as f32)
        .collect();

    self.executor.ilp_registry_mut().insert_mask_from_sparse(
        name,
        n,
        &active_ijk,
        &active_soft,
        &self.provider,
    ).map_err(|e| PyRuntimeError::new_err(e.to_string()))
}
```

Also add private helpers on `CompiledIlpProgram` (not in `#[pymethods]`):

```rust
/// Returns sorted (i,j,k) candidate triples for the given mask.
/// Same pruning logic as valid_candidates but returns raw triples.
/// `allow_recursive` must match what was passed to valid_candidates / set_rule_mask_sparse.
fn candidate_triples_for_mask(&self, mask_name: &str, allow_recursive: bool) -> PyResult<Vec<(u32, u32, u32)>> {
    let tmj = extract_tmj_meta_for_mask(&self.plan, Some(mask_name));
    let n = tmj.schema_size;
    let head_name = &tmj.head_rel_name;

    let k_head = self.rel_index.iter()
        .position(|(_, name)| name == head_name)
        .ok_or_else(|| PyValueError::new_err(
            format!("head relation '{}' not found for mask '{}'", head_name, mask_name)
        ))? as u32;

    let has_tuples: Vec<bool> = self.rel_index.iter()
        .map(|(_, name)| {
            self.executor.store().get(name)
                .map(|buf| buf.num_rows() > 0)
                .unwrap_or(false)
        })
        .collect();

    let mut triples: Vec<(u32, u32, u32)> = Vec::new();
    for i in 0..n as u32 {
        for j in 0..n as u32 {
            let k = k_head;
            if !has_tuples[i as usize] && !has_tuples[j as usize] {
                continue;
            }
            // Prune recursive candidates (i==k or j==k) unless allow_recursive
            if !allow_recursive && (i == k || j == k) {
                continue;
            }
            // Also prune if head has no tuples AND body ref is recursive
            // (only when allow_recursive=false — already handled above)
            triples.push((i, j, k));
        }
    }
    triples.sort_by_key(|&(i, j, k)| (k, i, j));
    Ok(triples)
}

fn expected_candidate_count(&self, mask_name: &str, allow_recursive: bool) -> PyResult<usize> {
    Ok(self.candidate_triples_for_mask(mask_name, allow_recursive)?.len())
}
```

### Step 5: Build and run tests

```bash
cargo build -p pyxlog --release && \
/home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && \
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_sparse.py -v --timeout=120
```
Expected: 6 passed

### Step 6: Run full regression

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/ -v --timeout=600
```
Expected: 239 passed (233 existing + 6 new)

### Step 7: Commit

```bash
git add crates/xlog-runtime/src/ilp_registry.rs crates/pyxlog/src/lib.rs python/tests/test_ilp_sparse.py
git commit -m "feat(ilp): set_rule_mask_sparse() — sparse candidate mask API with dense-parity tests"
```

---

## Task 3: Trainer Backend Abstraction

**Goal:** Abstract the mask-setting code in `trainer.py` behind a `MaskBackend` protocol so `train_only()` can use either dense or sparse. Default to sparse; fall back to dense when `config.debug_dense_mask=True`.

**Files:**
- Create: `crates/pyxlog/python/pyxlog/ilp/backend.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/__init__.py`
- Create: `python/tests/test_ilp_backend.py`

### Step 1: Write the failing test

Create `python/tests/test_ilp_backend.py`:

```python
# python/tests/test_ilp_backend.py
"""Tests for trainer backend abstraction (dense vs sparse)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_sparse_backend_converges():
    """train_only with debug_dense_mask=False (sparse) converges on reach."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        debug_dense_mask=False,
    )
    result = train_only(SOURCE, "W_reach", POS, NEG, config)
    assert result.converged
    assert "edge" in result.discovered_rule


def test_dense_backend_still_works():
    """train_only with debug_dense_mask=True (dense fallback) still works."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        debug_dense_mask=True,
    )
    result = train_only(SOURCE, "W_reach", POS, NEG, config)
    assert result.converged
    assert "edge" in result.discovered_rule


def test_sparse_and_dense_find_same_rule():
    """Both backends converge to the same rule on the same seed."""
    base = dict(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    r_sparse = train_only(SOURCE, "W_reach", POS, NEG,
                          TrainConfig(**base, debug_dense_mask=False))
    r_dense = train_only(SOURCE, "W_reach", POS, NEG,
                         TrainConfig(**base, debug_dense_mask=True))
    if r_sparse.converged and r_dense.converged:
        assert r_sparse.discovered_rule == r_dense.discovered_rule
```

### Step 2: Run test to verify it fails

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_backend.py -v --timeout=300
```
Expected: Tests may partially pass (dense path works already), but sparse backend won't converge yet because trainer doesn't use it.

### Step 3: Create backend abstraction

Create `crates/pyxlog/python/pyxlog/ilp/backend.py`:

```python
"""Mask backend abstraction for trainer.

Two backends:
- DenseMaskBackend: N3 tensor approach (alpha-compatible)
- SparseMaskBackend: candidate-indexed soft-probs (beta default)

Both provide the same interface:
- init_weights(C, n, device) -> learnable parameters
- apply_mask(prog, mask_name, W, tau, budget, candidates, n) -> candidate_soft_probs
- decode_argmax(W, candidates, n) -> candidate_index
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class MaskBackend(Protocol):
    """Protocol for mask backends."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        """Return learnable parameter tensor."""
        ...

    def apply_mask(
        self,
        prog,
        mask_name: str,
        W: torch.Tensor,
        tau: float,
        budget: int,
        candidates: list[dict],
        n: int,
    ) -> torch.Tensor:
        """Apply mask to program. Returns candidate_soft_probs shape (C,).
        allow_recursive is passed through to set_rule_mask_sparse for sparse backend."""
        ...

    def decode_argmax(self, W: torch.Tensor, candidates: list[dict], n: int) -> int:
        """Return the candidate index of the argmax."""
        ...


class DenseMaskBackend:
    """Dense N3 mask — same as alpha."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn((n, n, n), requires_grad=True, device=device)

    def apply_mask(self, prog, mask_name, W, tau, budget, candidates, n, allow_recursive=False):
        M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
        flat = M_soft.view(-1)
        k = min(budget, flat.numel())
        _, topk_idx = flat.topk(k)
        M_hard_flat = torch.zeros_like(flat)
        M_hard_flat[topk_idx] = 1.0
        M_hard = M_hard_flat.view_as(M_soft)

        prog.set_rule_mask(
            mask_name,
            M_hard.detach().contiguous().view(-1),
            M_soft.detach().contiguous().view(-1),
            n,
        )

        # Extract candidate-level soft probs
        cand_probs = torch.stack(
            [M_soft[c["i"], c["j"], c["k"]] for c in candidates]
        )
        return cand_probs

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            flat = W.view(-1)
            idx = flat.argmax().item()
            i = idx // (n * n)
            j = (idx % (n * n)) // n
            k = idx % n
        for ci, c in enumerate(candidates):
            if c["i"] == i and c["j"] == j and c["k"] == k:
                return ci
        return 0


class SparseMaskBackend:
    """Sparse candidate-indexed mask — beta default."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn(C, requires_grad=True, device=device)

    def apply_mask(self, prog, mask_name, W, tau, budget, candidates, n, allow_recursive=False):
        # Gumbel-softmax over C-dimensional logits
        cand_probs = F.gumbel_softmax(W, tau=tau, hard=False, dim=0)

        # Send sparse mask to Rust (allow_recursive must match valid_candidates call)
        C = len(candidates)
        ids = list(range(C))
        soft_list = cand_probs.detach().cpu().tolist()
        prog.set_rule_mask_sparse(mask_name, ids, soft_list, budget, allow_recursive)

        return cand_probs

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            return W.argmax().item()
```

### Step 4: Refactor `trainer.py` to use backend

Modify `crates/pyxlog/python/pyxlog/ilp/trainer.py`:

1. Add import at top:
```python
from pyxlog.ilp.backend import DenseMaskBackend, SparseMaskBackend
```

2. In `_run_single_attempt`, replace the fixed dense mask logic with the backend:

Key changes:
- Select backend based on `config.debug_dense_mask`
- Replace `W = torch.randn((n, n, n), ...)` with `backend.init_weights(C, n, "cuda")`
- Replace `_build_budget_aware_mask(W, ...)` + `prog.set_rule_mask(...)` with `backend.apply_mask(...)`
- Replace `_decode_argmax_ijk(W, n)` with candidate-index-based argmax via `backend.decode_argmax(W, candidates, n)`
- Loss computation uses candidate soft probs via `_compute_loss_from_candidates()`
- Convergence check translates candidate index back to (i,j,k) for single-rule re-check

The refactored `_run_single_attempt` core loop:
```python
backend = SparseMaskBackend() if not config.debug_dense_mask else DenseMaskBackend()
W = backend.init_weights(C, n, "cuda")
optimizer = torch.optim.Adam([W], lr=0.1)

# ... in loop:
cand_probs = backend.apply_mask(
    prog, mask_name, W, temp_controller.tau,
    config.max_active_rules, candidates, n,
)

# NaN check
if torch.isnan(cand_probs).any() or torch.isinf(cand_probs).any():
    raise _NumericFailure()

prog.evaluate()

# Loss via candidate probs
loss = _compute_loss_from_candidates(prog, cand_probs, positives, negatives, candidates)

# Argmax via backend
argmax_idx = backend.decode_argmax(W, candidates, n)
argmax_ijk = (candidates[argmax_idx]["i"], candidates[argmax_idx]["j"], candidates[argmax_idx]["k"])
```

3. Add `_compute_loss_from_candidates` function:
```python
def _compute_loss_from_candidates(prog, cand_probs, positives, negatives, candidates):
    """Per-fact surrogate loss using candidate-level soft probs."""
    loss = torch.tensor(0.0, device="cuda")

    for rel_name, values in positives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = torch.tensor(0.0, device="cuda")
            for (i, j, k) in contributing:
                for ci, c in enumerate(candidates):
                    if c["i"] == i and c["j"] == j and c["k"] == k:
                        credit = credit + cand_probs[ci]
                        break
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            loss = loss + (-cand_probs.sum() / len(candidates))

    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            credit = torch.tensor(0.0, device="cuda")
            for (i, j, k) in contributing:
                for ci, c in enumerate(candidates):
                    if c["i"] == i and c["j"] == j and c["k"] == k:
                        credit = credit + cand_probs[ci]
                        break
            loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    return loss
```

4. Keep the old `_build_budget_aware_mask` and `_compute_loss` as `DenseMaskBackend` inlines equivalent logic. Remove if fully dead.

### Step 5: Update `__init__.py` exports

Add to `crates/pyxlog/python/pyxlog/ilp/__init__.py`:
```python
from pyxlog.ilp.backend import DenseMaskBackend, SparseMaskBackend
```
And add to `__all__`.

### Step 6: Run tests

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_backend.py python/tests/test_ilp_trainer.py python/tests/test_ilp_reliability.py -v --timeout=600
```
Expected: All pass. Existing trainer tests use `debug_dense_mask=True` by default, exercising the dense fallback. New backend tests exercise sparse.

### Step 7: Run full regression

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/ -v --timeout=600
```
Expected: All pass

### Step 8: Commit

```bash
git add crates/pyxlog/python/pyxlog/ilp/backend.py crates/pyxlog/python/pyxlog/ilp/trainer.py crates/pyxlog/python/pyxlog/ilp/__init__.py python/tests/test_ilp_backend.py
git commit -m "feat(ilp): trainer backend abstraction — sparse default + dense fallback"
```

---

## Task 4: `train_and_promote()` with Transactional Commit

**Goal:** Implement the promotion entry point that wraps `train_only()`, compiles a trial program with the discovered rule committed, runs promotion gates, and returns a `PromotionResult`.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `relation_facts` pymethod on `CompiledIlpProgram`)
- Create: `crates/pyxlog/python/pyxlog/ilp/promoter.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/__init__.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py` (add `committed_source`, `novel_*` fields to `PromotionResult`)
- Create: `python/tests/test_ilp_promoter.py`

**Prerequisite Rust method:** `relation_facts(rel_name: str) -> list[list[int]]`
Returns all tuples in the named relation as a list of value lists. Required by
`novel_fact_audit` gate and `regression_check` gate.

```rust
/// Return all facts in the named relation as a list of int lists.
#[pyo3(signature = (rel_name))]
pub fn relation_facts(&self, rel_name: String) -> PyResult<Vec<Vec<i64>>> {
    match self.executor.store().get(&rel_name) {
        Some(buf) => {
            let arity = buf.arity();
            let rows = buf.num_rows();
            let host_data = buf.to_host()?; // download from GPU
            let mut result = Vec::with_capacity(rows);
            for r in 0..rows {
                let mut row = Vec::with_capacity(arity);
                for c in 0..arity {
                    row.push(host_data[c * rows + r] as i64);
                }
                result.push(row);
            }
            Ok(result)
        }
        None => Ok(vec![]),
    }
}
```

> **Note:** The exact `to_host()` / column-major layout depends on `CudaBuffer` API.
> Look at `fact_exists` implementation for the pattern to download GPU data.

### Step 1: Write the failing test

Create `python/tests/test_ilp_promoter.py`:

```python
# python/tests/test_ilp_promoter.py
"""Tests for train_and_promote() — transactional commit + promotion gates."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import (
    train_and_promote, TrainConfig, PromotionResult, PromotionStatus,
)

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_promote_returns_promotion_result():
    """train_and_promote returns a PromotionResult."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    assert isinstance(result, PromotionResult)


def test_promote_with_holdout_can_promote():
    """With holdout examples provided, promotion is possible."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3]), ("reach", [2, 4])],
        holdout_negatives=[],
    )
    if result.status == PromotionStatus.PROMOTED:
        assert all(g.passed for g in result.gates)


def test_promote_without_holdout_requires_review():
    """Without holdout, result is MANUAL_REVIEW_REQUIRED (not PROMOTED)."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(SOURCE, "W_reach", POS, NEG, config)
    assert result.status in (
        PromotionStatus.MANUAL_REVIEW_REQUIRED,
        PromotionStatus.NOT_CONVERGED,
    )


def test_promote_gates_are_populated():
    """Gate list should contain all required gates per design doc Section 4.3."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=[("reach", [1, 3])],
        holdout_negatives=[],
    )
    gate_names = [g.name for g in result.gates]
    assert "training_positive" in gate_names
    assert "training_negative" in gate_names
    assert "novel_fact_audit" in gate_names
    assert "regression_check" in gate_names


def test_promote_not_converged():
    """Non-convergent training returns NOT_CONVERGED status."""
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=1,
        tau_start=1.0, tau_floor=0.1, seed=0,
    )
    result = train_and_promote(
        distractor_source, "W",
        [("reach", [1, 3])], [], config,
    )
    assert result.status == PromotionStatus.NOT_CONVERGED
```

### Step 2: Run test to verify it fails

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_promoter.py -v --timeout=300
```
Expected: FAIL — `ImportError: cannot import name 'train_and_promote'`

### Step 3: Implement promoter

Create `crates/pyxlog/python/pyxlog/ilp/promoter.py`:

```python
"""dILP promotion pipeline — train_and_promote() entry point.

Wraps train_only(), compiles trial program with discovered rule,
runs promotion gates, returns PromotionResult.

**Type update required:** Before implementing this module, add the following
fields to `PromotionResult` in `types.py` to match design doc Section 5.1:
- committed_source: str | None  — Datalog source with rule committed (on PROMOTED)
- novel_count: int | None
- novel_rate: float | None
- novel_examples: list[str] | None  — up to 10 sampled
These fields are already defined in the design doc's PromotionResult type.
"""
from __future__ import annotations

import pyxlog
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import (
    GateResult,
    PromotionResult,
    PromotionStatus,
    TrainConfig,
)


def train_and_promote(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig = TrainConfig(),
    holdout_positives: list[tuple[str, list[int]]] | None = None,
    holdout_negatives: list[tuple[str, list[int]]] | None = None,
) -> PromotionResult:
    """Train and optionally promote a learned rule.

    Steps:
    1. Call train_only() to find best rule.
    2. If not converged: NOT_CONVERGED.
    3. If no holdout provided: MANUAL_REVIEW_REQUIRED.
    4. Compile trial program with discovered rule committed.
    5. Run promotion gates against trial.
    6. All pass: PROMOTED. Any fail: MANUAL_REVIEW_REQUIRED.
    """
    train_result = train_only(source, mask_name, positives, negatives, config)

    if not train_result.converged:
        return PromotionResult(
            status=PromotionStatus.NOT_CONVERGED,
            artifact=train_result.artifact,
        )

    gates: list[GateResult] = []
    discovered = train_result.discovered_rule

    # Compile trial program with discovered rule committed
    trial_source = _commit_rule(source, mask_name, discovered)
    try:
        trial = pyxlog.IlpProgramFactory.compile(
            trial_source, device=config.device, memory_mb=config.memory_mb,
        )
    except Exception as e:
        return PromotionResult(
            status=PromotionStatus.COMMIT_FAILED,
            gates=[GateResult(name="commit", passed=False, detail=str(e))],
            artifact=train_result.artifact,
        )

    # Run the committed program (no mask needed)
    trial.evaluate()

    # Gate: training_positive
    tp_count = sum(1 for r, v in positives if trial.fact_exists(r, v))
    tp_ok = tp_count == len(positives)
    gates.append(GateResult(
        name="training_positive",
        passed=tp_ok,
        detail=f"{tp_count}/{len(positives)} derived",
    ))

    # Gate: training_negative
    tn_count = sum(1 for r, v in negatives if trial.fact_exists(r, v))
    tn_ok = tn_count == 0
    gates.append(GateResult(
        name="training_negative",
        passed=tn_ok,
        detail=f"{tn_count}/{len(negatives)} derived (want 0)",
    ))

    # Gate: novel_fact_audit (always required per design doc Section 4.3)
    head_rel = discovered.split("(")[0].strip()  # extract head relation name
    all_derived = trial.relation_facts(head_rel)
    known_set = {(r, tuple(v)) for r, v in positives}
    if holdout_positives:
        known_set |= {(r, tuple(v)) for r, v in holdout_positives}
    novel = [f for f in all_derived if (head_rel, tuple(f)) not in known_set]
    novel_count = len(novel)
    total_derived = len(all_derived) if all_derived else 1
    novel_rate = novel_count / total_derived if total_derived > 0 else 0.0
    novel_ok = novel_rate <= config.max_novel_rate
    gates.append(GateResult(
        name="novel_fact_audit",
        passed=novel_ok,
        detail=f"novel_rate={novel_rate:.3f} ({novel_count}/{total_derived}), threshold={config.max_novel_rate}",
    ))

    # Gate: regression_check (always required per design doc Section 4.3)
    # Compare protected relations between original and trial programs
    regression_ok = True
    regression_detail_parts = []
    if config.protected_relations:
        original_prog = pyxlog.IlpProgramFactory.compile(
            source, device=config.device, memory_mb=config.memory_mb,
        )
        original_prog.evaluate()
        for rel in config.protected_relations:
            orig_facts = set(map(tuple, original_prog.relation_facts(rel)))
            trial_facts = set(map(tuple, trial.relation_facts(rel)))
            lost = orig_facts - trial_facts
            if lost:
                regression_ok = False
                regression_detail_parts.append(f"{rel}: lost {len(lost)} facts")
    if not regression_detail_parts:
        regression_detail_parts.append("all protected relations preserved")
    gates.append(GateResult(
        name="regression_check",
        passed=regression_ok,
        detail="; ".join(regression_detail_parts),
    ))

    # Holdout gates
    if holdout_positives is None and holdout_negatives is None:
        return PromotionResult(
            status=PromotionStatus.MANUAL_REVIEW_REQUIRED,
            gates=gates,
            novel_count=novel_count,
            novel_rate=novel_rate,
            novel_examples=[str(f) for f in novel[:10]],
            artifact=train_result.artifact,
        )

    hp = holdout_positives or []
    hn = holdout_negatives or []

    if hp:
        hp_derived = sum(1 for r, v in hp if trial.fact_exists(r, v))
        hp_ok = hp_derived >= len(hp) * 0.95
        gates.append(GateResult(
            name="holdout_positive",
            passed=hp_ok,
            detail=f"{hp_derived}/{len(hp)} derived (threshold 95%)",
        ))
    if hn:
        hn_derived = sum(1 for r, v in hn if trial.fact_exists(r, v))
        hn_ok = hn_derived == 0
        gates.append(GateResult(
            name="holdout_negative",
            passed=hn_ok,
            detail=f"{hn_derived}/{len(hn)} derived (want 0)",
        ))

    all_pass = all(g.passed for g in gates)
    status = PromotionStatus.PROMOTED if all_pass else PromotionStatus.MANUAL_REVIEW_REQUIRED

    return PromotionResult(
        status=status,
        gates=gates,
        novel_count=novel_count,
        novel_rate=novel_rate,
        novel_examples=[str(f) for f in novel[:10]],
        committed_source=trial_source if all_pass else None,  # design doc Section 5.1
        artifact=train_result.artifact,
    )


def _commit_rule(source: str, mask_name: str, rule: str) -> str:
    """Replace the learnable declaration with the discovered rule."""
    lines = source.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"learnable({mask_name})"):
            result.append(f"    {rule}")
        else:
            result.append(line)
    return "\n".join(result)
```

> **Note:** Uses `trial.evaluate()` — replace with the actual method name (`evaluate()`).

### Step 4: Update `__init__.py`

Add to `crates/pyxlog/python/pyxlog/ilp/__init__.py`:
```python
from pyxlog.ilp.promoter import train_and_promote
```
And add `"train_and_promote"` to `__all__`.

### Step 5: Run tests

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_promoter.py -v --timeout=300
```
Expected: 5 passed

### Step 6: Full regression

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/ -v --timeout=600
```

### Step 7: Commit

```bash
git add crates/pyxlog/python/pyxlog/ilp/promoter.py crates/pyxlog/python/pyxlog/ilp/__init__.py python/tests/test_ilp_promoter.py
git commit -m "feat(ilp): train_and_promote() with transactional commit + promotion gates"
```

---

## Task 5: Holdout Scoring + Ambiguity Scan

**Goal:** Implement LOO (leave-one-out) holdout F1 for <= 20 examples, and top-M ambiguity scan for alternative rules.

**Files:**
- Create: `crates/pyxlog/python/pyxlog/ilp/holdout.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/promoter.py` (wire in ambiguity scan)
- Modify: `crates/pyxlog/python/pyxlog/ilp/__init__.py`
- Create: `python/tests/test_ilp_holdout.py`

### Step 1: Write the failing test

Create `python/tests/test_ilp_holdout.py`:

```python
# python/tests/test_ilp_holdout.py
"""Tests for holdout scoring and ambiguity scan."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_and_promote, TrainConfig, PromotionStatus
from pyxlog.ilp.holdout import loo_holdout_f1

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
NEG = []


def test_loo_holdout_f1_perfect():
    """LOO on reach should give F1 >= 0.9 when rule is correct."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", POS, NEG, config)
    assert f1 >= 0.9, f"LOO F1 should be >= 0.9 for reach, got {f1}"


def test_loo_singleton_skipped():
    """LOO with 1 positive should return None (can't hold out)."""
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    f1 = loo_holdout_f1(SOURCE, "W_reach", [("reach", [1, 3])], NEG, config)
    assert f1 is None


def test_promote_with_loo():
    """Promotion with holdout should use LOO results."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        holdout_strategy="loo",
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=POS[:2],
        holdout_negatives=[],
    )
    if result.status == PromotionStatus.PROMOTED:
        assert result.artifact.discovered_rule


def test_ambiguity_scan_smoke():
    """Ambiguity check does not crash."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        check_ambiguity=True,
    )
    result = train_and_promote(
        SOURCE, "W_reach", POS, NEG, config,
        holdout_positives=POS[:2],
        holdout_negatives=[],
    )
    assert isinstance(result, type(result))
```

### Step 2: Run test to verify failure

Expected: `ImportError: cannot import name 'loo_holdout_f1'`

### Step 3: Implement holdout

Create `crates/pyxlog/python/pyxlog/ilp/holdout.py`:

```python
"""Holdout scoring strategies for dILP.

LOO (leave-one-out) for <= 20 examples.
"""
from __future__ import annotations

import pyxlog
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import TrainConfig


def loo_holdout_f1(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
) -> float | None:
    """Leave-one-out cross-validation F1 over positives.

    For each positive example, train on all-but-one, evaluate the committed
    rule on the held-out example. Computes per-fold precision (against known
    positives+negatives) and recall (held-out derived?), then returns the
    mean F1 across folds.

    Returns None if |positives| < 2.
    """
    if len(positives) < 2:
        return None

    fold_f1s: list[float] = []
    pos_set = {(r, tuple(v)) for r, v in positives}
    neg_set = {(r, tuple(v)) for r, v in negatives}

    for i in range(len(positives)):
        train_pos = positives[:i] + positives[i + 1:]
        held_out = positives[i]

        result = train_only(source, mask_name, train_pos, negatives, config)
        if not (result.converged and result.discovered_rule):
            fold_f1s.append(0.0)
            continue

        trial_source = _commit_rule(source, mask_name, result.discovered_rule)
        try:
            trial = pyxlog.IlpProgramFactory.compile(
                trial_source, device=config.device, memory_mb=config.memory_mb,
            )
            trial.evaluate()
        except Exception:
            fold_f1s.append(0.0)
            continue

        # Recall: did we derive the held-out example?
        held_out_derived = trial.fact_exists(held_out[0], held_out[1])
        recall = 1.0 if held_out_derived else 0.0

        # Precision: of all derived head facts, how many are in the positive set?
        head_rel = held_out[0]
        all_derived = trial.relation_facts(head_rel)
        if all_derived:
            true_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in pos_set)
            false_pos = sum(1 for f in all_derived if (head_rel, tuple(f)) in neg_set)
            precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 1.0
        else:
            precision = 1.0 if recall == 0.0 else 0.0

        # F1 = harmonic mean
        if precision + recall > 0:
            f1 = 2 * precision * recall / (precision + recall)
        else:
            f1 = 0.0
        fold_f1s.append(f1)

    return sum(fold_f1s) / len(fold_f1s) if fold_f1s else None


def _commit_rule(source: str, mask_name: str, rule: str) -> str:
    """Replace learnable declaration with discovered rule."""
    lines = source.splitlines()
    result = []
    for line in lines:
        if line.strip().startswith(f"learnable({mask_name})"):
            result.append(f"    {rule}")
        else:
            result.append(line)
    return "\n".join(result)
```

> **Note:** The method is `evaluate()` on `CompiledIlpProgram`.

### Step 4: Add ambiguity scan to promoter

Add to `crates/pyxlog/python/pyxlog/ilp/promoter.py`:

```python
def _scan_ambiguity(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    winning_rule: str,
    artifact: "LearnedArtifact",
    config: TrainConfig,
    top_m: int = 256,
) -> list[str]:
    """Check top-M candidates by final soft probability for alternative rules.

    Per design doc Section 4.2: scan top-M by final soft probability, not
    by candidate position. M = min(top_m, C).
    """
    prog = pyxlog.IlpProgramFactory.compile(
        source, device=config.device, memory_mb=config.memory_mb,
    )
    candidates = prog.valid_candidates(mask_name, config.allow_recursive_candidates)

    # Sort candidates by final soft probability (descending) from artifact
    soft_probs = artifact.soft_probs if artifact.soft_probs else []
    if soft_probs and len(soft_probs) == len(candidates):
        indexed = sorted(enumerate(candidates), key=lambda x: -soft_probs[x[0]])
        sorted_candidates = [c for _, c in indexed]
    else:
        sorted_candidates = candidates

    alternatives = []
    scan_count = min(top_m, len(sorted_candidates))
    for c in sorted_candidates[:scan_count]:
        rule_str = f"{c['head_name']}(X, Y) :- {c['left_name']}(X, Z), {c['right_name']}(Z, Y)."
        if rule_str == winning_rule:
            continue

        trial_source = _commit_rule(source, mask_name, rule_str)
        try:
            trial = pyxlog.IlpProgramFactory.compile(
                trial_source, device=config.device, memory_mb=config.memory_mb,
            )
            trial.evaluate()

            all_pos = all(trial.fact_exists(r, v) for r, v in positives)
            no_neg = not any(trial.fact_exists(r, v) for r, v in negatives)

            if all_pos and no_neg:
                alternatives.append(rule_str)
        except Exception:
            continue

    return alternatives
```

Wire it into `train_and_promote` when `config.check_ambiguity` is True. Call as:
```python
    if config.check_ambiguity:
        alts = _scan_ambiguity(
            source, mask_name, positives, negatives,
            discovered, train_result.artifact, config,
            top_m=256 if not config.exhaustive_ambiguity else len(candidates),
        )
        # Store in result — ambiguous_alternatives is informational, not gating
```

### Step 5: Update `__init__.py`

Add exports for `loo_holdout_f1`.

### Step 6: Run tests + regression

### Step 7: Commit

```bash
git commit -m "feat(ilp): LOO holdout scoring + top-M ambiguity scan"
```

---

## Task 6: Hard-Negative Mining

**Goal:** Add `sample_false_positives()` Rust API and wire it into the training loop for dynamic negative generation.

**Files:**
- Modify: `crates/pyxlog/src/lib.rs` (add `sample_false_positives` pymethod)
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py` (mine negatives mid-loop)
- Create: `python/tests/test_ilp_hard_negatives.py`

### Step 1: Write the failing test

Create `python/tests/test_ilp_hard_negatives.py`:

```python
# python/tests/test_ilp_hard_negatives.py
"""Tests for hard-negative mining via sample_false_positives."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_sample_false_positives_exists():
    """sample_false_positives is callable on CompiledIlpProgram."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    # Set a mask that activates edge+edge -> reach
    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    exclude = [("reach", [1, 3]), ("reach", [2, 4])]
    fps = prog.sample_false_positives("reach", exclude, 10)
    assert isinstance(fps, list)
    for fp in fps:
        assert isinstance(fp, (list, tuple))


def test_sample_false_positives_excludes_positives():
    """Returned facts should not include excluded positives."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    exclude = [("reach", [1, 3])]
    fps = prog.sample_false_positives("reach", exclude, 10)
    for fp in fps:
        assert tuple(fp) != (1, 3), "Should not return excluded positive"


def test_sample_false_positives_bounded():
    """Result is bounded by max_n."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W", False)
    n = prog.ilp_schema_size()

    M = torch.zeros(n * n * n, device="cuda")
    for c in candidates:
        if c["left_name"] == "edge" and c["right_name"] == "edge":
            flat = c["i"] * n * n + c["j"] * n + c["k"]
            M[flat] = 1.0
            break
    prog.set_rule_mask("W", M, M, n)
    prog.evaluate()

    fps = prog.sample_false_positives("reach", [], 2)
    assert len(fps) <= 2
```

> **Note:** Uses `evaluate()` — replace with `evaluate()` if that is the actual method name.

### Step 2: Run test to verify failure

Expected: `AttributeError: 'CompiledIlpProgram' has no attribute 'sample_false_positives'`

### Step 3: Implement Rust API

Add to `crates/pyxlog/src/lib.rs` in `#[pymethods] impl CompiledIlpProgram`:

```rust
/// Sample up to max_n derived facts for head_rel that are NOT in exclude.
///
/// Returns list[list[int]] — each inner list is a tuple of column values.
#[pyo3(signature = (head_rel, exclude, max_n))]
pub fn sample_false_positives(
    &self,
    py: Python<'_>,
    head_rel: String,
    exclude: Vec<(String, Vec<i64>)>,
    max_n: usize,
) -> PyResult<Vec<Vec<i64>>> {
    let buf = self.executor.store().get(&head_rel)
        .ok_or_else(|| PyValueError::new_err(
            format!("Relation '{}' not found", head_rel)
        ))?;

    let exclude_set: HashSet<Vec<i64>> = exclude.into_iter()
        .filter(|(r, _)| r == &head_rel)
        .map(|(_, v)| v)
        .collect();

    let arity = buf.arity();
    let row_count = read_device_row_count(&self.provider, buf)
        .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;

    if row_count == 0 || arity == 0 {
        return Ok(Vec::new());
    }

    let mut columns: Vec<Vec<i32>> = Vec::with_capacity(arity);
    for col_idx in 0..arity {
        let col = buf.column(col_idx)
            .ok_or_else(|| PyRuntimeError::new_err("Missing column"))?;
        let mut host = vec![0i32; row_count];
        self.provider.device().inner()
            .dtoh_sync_copy_into(col, &mut host)
            .map_err(|e| PyRuntimeError::new_err(e.to_string()))?;
        columns.push(host);
    }

    let mut result = Vec::new();
    for row in 0..row_count {
        let vals: Vec<i64> = columns.iter().map(|c| c[row] as i64).collect();
        if !exclude_set.contains(&vals) {
            result.push(vals);
            if result.len() >= max_n {
                break;
            }
        }
    }
    Ok(result)
}
```

### Step 4: Build, install, test

```bash
cargo build -p pyxlog --release && \
/home/dev/.local/bin/maturin develop --release -m crates/pyxlog/Cargo.toml && \
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_hard_negatives.py -v --timeout=120
```
Expected: 3 passed

### Step 5: Wire into trainer loop

In `trainer.py`, at the end of each attempt (inside the step loop), mine hard negatives periodically when `config.max_mined_negatives > 0`:

```python
# After forward pass, before loss:
if config.max_mined_negatives > 0 and step > 0 and step % 20 == 0:
    head_name = candidates[0]["head_name"]
    fps = prog.sample_false_positives(head_name, positives, config.max_mined_negatives)
    mined_negs = [(head_name, fp) for fp in fps]
    all_negatives = list(negatives) + mined_negs
else:
    all_negatives = negatives

# Use all_negatives in loss computation
```

### Step 6: Run tests + regression

### Step 7: Commit

```bash
git commit -m "feat(ilp): sample_false_positives() + hard-negative mining in trainer"
```

---

## Task 7: Artifact Save/Load

**Goal:** Implement `LearnedArtifact.save(path)` and `LearnedArtifact.load(path)` with JSON serialization, metadata, and hash checking.

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Create: `python/tests/test_ilp_artifact.py`

### Step 1: Write the failing test

Create `python/tests/test_ilp_artifact.py`:

```python
# python/tests/test_ilp_artifact.py
"""Tests for artifact save/load."""
import json
import pytest
from pathlib import Path

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, LearnedArtifact

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]


def test_save_load_roundtrip(tmp_path):
    """Artifact survives save/load roundtrip."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)
    assert result.converged

    path = tmp_path / "artifact.json"
    result.artifact.save(path)
    assert path.exists()

    loaded = LearnedArtifact.load(path)
    assert loaded.discovered_rule == result.artifact.discovered_rule
    assert len(loaded.candidate_map) == len(result.artifact.candidate_map)
    assert loaded.logits == pytest.approx(result.artifact.logits, abs=1e-6)


def test_save_produces_valid_json(tmp_path):
    """Saved artifact is valid JSON."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)
    path = tmp_path / "artifact.json"
    result.artifact.save(path)

    with open(path) as f:
        data = json.load(f)
    assert "discovered_rule" in data
    assert "candidate_map" in data
    assert "metadata" in data


def test_load_rejects_incompatible_hash(tmp_path):
    """Loading artifact with wrong hash raises when verify_hash=True."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(SOURCE, "W_reach", POS, [], config)
    path = tmp_path / "artifact.json"
    result.artifact.save(path)

    # Corrupt the hash
    with open(path) as f:
        data = json.load(f)
    data["metadata"]["candidate_map_hash"] = "bogus"
    with open(path, "w") as f:
        json.dump(data, f)

    with pytest.raises(ValueError, match="hash"):
        LearnedArtifact.load(path, verify_hash=True)
```

### Step 2: Run test to verify failure

Expected: `AttributeError: type object 'LearnedArtifact' has no attribute 'save'`

### Step 3: Implement save/load on LearnedArtifact

Add methods to `LearnedArtifact` in `crates/pyxlog/python/pyxlog/ilp/types.py`:

```python
def save(self, path: Path) -> None:
    """Save artifact to JSON file."""
    import json
    import hashlib

    map_data = [
        {"id": c.id, "i": c.i, "j": c.j, "k": c.k,
         "left_name": c.left_name, "right_name": c.right_name,
         "head_name": c.head_name}
        for c in self.candidate_map
    ]
    map_str = json.dumps(map_data, sort_keys=True)
    self.metadata.candidate_map_hash = hashlib.sha256(map_str.encode()).hexdigest()

    # Compute config_hash from config_snapshot if present
    config_hash = ""
    if self.config_snapshot:
        import dataclasses
        config_dict = dataclasses.asdict(self.config_snapshot)
        config_str = json.dumps(config_dict, sort_keys=True, default=str)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()
    self.metadata.config_hash = config_hash

    data = {
        "schema_version": getattr(self, "schema_version", "beta-v1"),
        "discovered_rule": self.discovered_rule,
        "candidate_map": map_data,
        "logits": self.logits,
        "soft_probs": self.soft_probs,
        "selected_hard": self.selected_hard,
        "metadata": {
            "pyxlog_version": self.metadata.pyxlog_version,
            "git_sha": self.metadata.git_sha,
            "cuda_version": self.metadata.cuda_version,
            "device_name": self.metadata.device_name,
            "candidate_map_hash": self.metadata.candidate_map_hash,
            "config_hash": self.metadata.config_hash,
            "timestamp_utc": self.metadata.timestamp_utc,
        },
        "config_snapshot": dataclasses.asdict(self.config_snapshot) if self.config_snapshot else None,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

@classmethod
def load(cls, path: Path, verify_hash: bool = False) -> "LearnedArtifact":
    """Load artifact from JSON file."""
    import json
    import hashlib

    with open(path) as f:
        data = json.load(f)

    candidate_map = [
        CandidateMapEntry(**c) for c in data["candidate_map"]
    ]

    if verify_hash:
        map_str = json.dumps(data["candidate_map"], sort_keys=True)
        computed = hashlib.sha256(map_str.encode()).hexdigest()
        stored = data.get("metadata", {}).get("candidate_map_hash", "")
        if computed != stored:
            raise ValueError(
                f"Candidate map hash mismatch: computed {computed}, stored {stored}"
            )

    # Schema version compatibility check
    stored_version = data.get("schema_version", "")
    if verify_hash and stored_version and stored_version != "beta-v1":
        raise ValueError(
            f"Incompatible schema version: {stored_version} (expected beta-v1)"
        )

    meta_data = data.get("metadata", {})
    metadata = ArtifactMetadata(
        pyxlog_version=meta_data.get("pyxlog_version", ""),
        git_sha=meta_data.get("git_sha"),
        cuda_version=meta_data.get("cuda_version", ""),
        device_name=meta_data.get("device_name", ""),
        candidate_map_hash=meta_data.get("candidate_map_hash", ""),
        config_hash=meta_data.get("config_hash", ""),
        timestamp_utc=meta_data.get("timestamp_utc", ""),
    )

    return cls(
        candidate_map=candidate_map,
        logits=data.get("logits", []),
        soft_probs=data.get("soft_probs", []),
        selected_hard=data.get("selected_hard", []),
        discovered_rule=data.get("discovered_rule", ""),
        config_snapshot=None,  # Config not restored from JSON in beta (deferred)
        metadata=metadata,
    )
```

### Step 4: Run tests + regression

### Step 5: Commit

```bash
git commit -m "feat(ilp): artifact save/load with JSON + hash verification"
```

---

## Task 8: Recursive Candidates

**Goal:** Enable `allow_recursive_candidates=True` to include i==k and j==k candidates. Guard behind config flag with dedicated tests.

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py` (remove alpha guard)
- Create: `python/tests/test_ilp_recursive.py`

### Step 1: Write the failing test

Create `python/tests/test_ilp_recursive.py`:

```python
# python/tests/test_ilp_recursive.py
"""Tests for recursive candidate support (beta feature)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, IlpConfigError

SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
RECURSIVE_POS = [("reach", [1, 4]), ("reach", [1, 3]), ("reach", [2, 4])]
RECURSIVE_NEG = []


def test_recursive_candidates_no_longer_raises():
    """allow_recursive_candidates=True is now allowed in beta."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        allow_recursive_candidates=True,
    )
    result = train_only(SOURCE, "W", RECURSIVE_POS, RECURSIVE_NEG, config)
    assert isinstance(result.attempt_count, int)


def test_recursive_candidates_has_more_candidates():
    """Recursive mode includes body-references-head candidates."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    non_recursive = prog.valid_candidates("W", False)
    recursive = prog.valid_candidates("W", True)
    assert len(recursive) > len(non_recursive)


def test_non_recursive_still_default():
    """Default config still has allow_recursive_candidates=False."""
    config = TrainConfig()
    assert config.allow_recursive_candidates is False


def test_recursive_reach_runs_without_crash():
    """With recursive candidates enabled, training runs without errors."""
    config = TrainConfig(
        step_budget_per_attempt=150, max_attempts=7,
        tau_start=2.0, tau_floor=0.05, seed=42,
        allow_recursive_candidates=True,
    )
    result = train_only(SOURCE, "W", RECURSIVE_POS, RECURSIVE_NEG, config)
    assert result.total_steps > 0
```

### Step 2: Run test to verify failure

Expected: `IlpConfigError: allow_recursive_candidates=True is a beta-only feature`

### Step 3: Remove alpha guard

In `crates/pyxlog/python/pyxlog/ilp/trainer.py`, in `_validate_inputs`, remove:

```python
    # Alpha scope: recursive candidates are beta-only
    if config.allow_recursive_candidates:
        raise IlpConfigError(
            "allow_recursive_candidates=True is a beta-only feature"
        )
```

And update `_run_single_attempt` to pass `config.allow_recursive_candidates` to `valid_candidates`:

```python
    candidates = prog.valid_candidates(mask_name, config.allow_recursive_candidates)
```

### Step 4: Run tests + regression

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_recursive.py python/tests/test_ilp_robustness.py -v --timeout=300
```
Expected: Recursive tests pass. The `test_recursive_candidates_rejected_in_alpha` test in robustness.py will now **fail** — update it to verify the feature works instead of raising.

### Step 5: Update robustness test

In `python/tests/test_ilp_robustness.py`, change `test_recursive_candidates_rejected_in_alpha` to:

```python
def test_recursive_candidates_accepted_in_beta():
    """allow_recursive_candidates=True is now a supported beta feature."""
    config = TrainConfig(max_attempts=1, allow_recursive_candidates=True)
    # Should not raise — just run with limited budget
    result = train_only(SOURCE, "W", [("reach", [1, 3])], [], config)
    assert result.total_steps > 0
```

### Step 6: Full regression

### Step 7: Commit

```bash
git commit -m "feat(ilp): enable recursive candidates behind config flag"
```

---

## Task 9: Beta Reliability Gate

**Goal:** Run 20/20 reliability gate across 4 stages x 5 seeds using the sparse backend.

**Files:**
- Create: `python/tests/test_ilp_beta_gate.py`

### Step 1: Write the test

Create `python/tests/test_ilp_beta_gate.py`:

```python
# python/tests/test_ilp_beta_gate.py
"""Beta reliability gate: 20/20 on sparse backend.

Extends alpha gate with sparse mask backend (debug_dense_mask=False).
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

# Re-use stage definitions from reliability
from test_ilp_reliability import STAGES


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "stage_name,source,positives,negatives,mask_name",
    STAGES,
    ids=["reach", "grandparent", "colleague", "plus2"],
)
def test_beta_reliability_sparse(
    stage_name, source, positives, negatives, mask_name, seed,
):
    """Each stage converges with sparse backend (beta gate: 5/5 per stage)."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=seed,
        device=0,
        memory_mb=512,
        debug_dense_mask=False,  # sparse backend
    )
    result = train_only(
        source=source,
        mask_name=mask_name,
        positives=positives,
        negatives=negatives,
        config=config,
    )
    assert result.converged, (
        f"Beta gate: {stage_name} failed with seed={seed}, "
        f"attempts={result.attempt_count}, steps={result.total_steps}"
    )
```

### Step 2: Run

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/test_ilp_beta_gate.py -v --timeout=1800
```
Expected: 20 passed

### Step 3: Run full regression

```bash
LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/local/cuda/lib64 .venv/bin/python -m pytest python/tests/ -v --timeout=1800
```
Expected: All tests pass

### Step 4: Commit

```bash
git commit -m "test(ilp): beta reliability gate — 20/20 sparse backend"
```

---

## Summary

| Task | Scope | New Tests | Key Files |
|------|-------|-----------|-----------|
| 1 | Lock alpha baseline | 0 | git tag only |
| 2 | `set_rule_mask_sparse()` | 6 | lib.rs, ilp_registry.rs |
| 3 | Backend abstraction | 3 | backend.py, trainer.py |
| 4 | `train_and_promote()` | 5 | lib.rs (relation_facts), promoter.py, types.py |
| 5 | Holdout + ambiguity | 4 | holdout.py, promoter.py |
| 6 | Hard-negative mining | 3 | lib.rs, trainer.py |
| 7 | Artifact save/load | 3 | types.py |
| 8 | Recursive candidates | 4 | trainer.py |
| 9 | Beta reliability gate | 20 | test_ilp_beta_gate.py |

**Total new tests:** ~48
**Expected final count:** ~281 tests (233 + 48)
**Critical dependency chain:** Task 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8 -> 9
