# dILP Hardening Implementation Plan (Alpha Milestone)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the alpha milestone of the dILP hardening design — Python types, exception taxonomy, Rust `valid_candidates()` API, adaptive temperature, entropy regularization, structured multi-start, `train_only()` entry point, and 5/5 reliability gate.

**Architecture:** Dual-track (contract-first). Track A adds Rust primitives (`valid_candidates`, configurable `max_active_rules`). Track B builds the Python trainer module (`pyxlog.ilp`) on top of those primitives. The existing `ilp_showcase.py` remains untouched as a reference; the new trainer replaces its ad-hoc training loop with structured, testable code.

**Tech Stack:** Rust (PyO3 bindings in `crates/pyxlog`), Python 3.10+ (dataclasses, enum, logging), PyTorch (Gumbel-Softmax), maturin (build).

**Design doc:** `docs/plans/2026-02-26-dilp-hardening-design.md`

**Key syntax references (from existing codebase):**
- Learnable rule: `learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).`
- Compile: `pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)` (static method)
- CUDA gate: `torch = pytest.importorskip("torch")` + `if not torch.cuda.is_available(): pytest.skip(...)`
- Template names: `bL`, `bR` (or `b1`, `b2`) — identified by the compiler, not by prefix

**Alpha scope (what's IN):**
- Python types (TrainConfig, TrainResult, LearnedArtifact, exceptions)
- Rust `valid_candidates(mask_name, allow_recursive)` API with candidate pruning
  (alpha always passes `allow_recursive=False`; the param exists for API completeness)
- Configurable `max_active_rules` (no longer hardcoded 32)
- Adaptive temperature controller (COOLING/PLATEAU/WARMUP)
- Entropy regularization (normalized)
- Structured multi-start (best-of-K with random reserve)
- `train_only()` entry point with per-attempt isolation (dense mask, Phase 1)
- NaN/Inf detection and numeric failure policy
- Telemetry level 0 (summary) and level 1 (per-step)
- 5/5 reliability gate on showcase suite

**Alpha scope (what's OUT — deferred):**
- `set_rule_mask_sparse()` (beta — Phase 2 sparse API)
- Factorized scoring over C-vector (RC — requires Phase 2 sparse mask)
- `train_and_promote()` + promotion gates (beta)
- Ambiguity detection (beta)
- CUDA event profiling counters (beta)
- `sample_false_positives()` hard-negative mining (beta)
- Recursive candidate policy — trainer using `allow_recursive=True` (beta)
- Holdout LOO/k-fold (beta)
- Artifact save/load persistence (beta)
- Typed schemas (GA)
- Telemetry level 2 streaming (GA)

**Alpha training loop uses dense N³ mask** (same as current showcase). The
`valid_candidates()` API provides the candidate list for Python-side bookkeeping
(argmax decoding, confidence metrics, rule formatting), but the actual mask
passed to `set_rule_mask()` is still the full N³ tensor. Factorized scoring
(operating only on C logits) is deferred to RC when `set_rule_mask_sparse()`
is available.

---

## Track A: Rust Primitives

### Task 1: Add `valid_candidates()` to CompiledIlpProgram

**Files:**
- Modify: `crates/pyxlog/src/lib.rs:3828-4085` (CompiledIlpProgram impl)
- Test: `python/tests/test_ilp_candidates.py` (create)

**Step 1: Write the failing Python test**

```python
# python/tests/test_ilp_candidates.py
"""Tests for valid_candidates() Rust API."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP tests", allow_module_level=True)

SOURCE = """
    edge(1, 2).
    edge(2, 3).
    edge(3, 4).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_valid_candidates_returns_candidates():
    """valid_candidates returns non-empty list of CandidateInfo dicts."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    # Each candidate has required keys
    c = candidates[0]
    assert set(c.keys()) == {"id", "i", "j", "k", "left_name", "right_name", "head_name"}


def test_valid_candidates_deterministic_ids():
    """Same program produces same candidate IDs."""
    prog1 = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    prog2 = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    c1 = prog1.valid_candidates("W")
    c2 = prog2.valid_candidates("W")
    assert [c["id"] for c in c1] == [c["id"] for c in c2]


def test_valid_candidates_prunes_template_plus_template():
    """Candidates where both body atoms are template variables are pruned."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    # Get the actual template names from relation list
    names = prog.ilp_relation_names()
    templates = {n for n in names if n.startswith("bL") or n.startswith("bR")
                 or n.startswith("b1") or n.startswith("b2")}
    for c in candidates:
        assert not (c["left_name"] in templates and c["right_name"] in templates), \
            f"Template+template candidate not pruned: {c}"


def test_valid_candidates_head_is_target():
    """All candidates have head_name == the learnable target relation."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    for c in candidates:
        assert c["head_name"] == "reach", f"Wrong head: {c}"


def test_valid_candidates_nonrecursive_prunes_head_in_body():
    """Default (allow_recursive=False): candidates where body==head are pruned."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")  # default: allow_recursive=False
    for c in candidates:
        # reach has no base facts, so body referencing reach should be pruned
        assert c["left_name"] != "reach" and c["right_name"] != "reach", \
            f"Recursive candidate not pruned: {c}"


def test_valid_candidates_ids_contiguous():
    """IDs are 0..C-1 with no gaps."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    candidates = prog.valid_candidates("W")
    ids = sorted(c["id"] for c in candidates)
    assert ids == list(range(len(candidates)))
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: FAIL with `AttributeError: 'CompiledIlpProgram' has no attribute 'valid_candidates'`

**Step 3: Implement valid_candidates in Rust**

In `crates/pyxlog/src/lib.rs`, add to the `#[pymethods]` impl block for
`CompiledIlpProgram` (after `ilp_relation_names` around line 3984):

```rust
/// Return the set of valid (i,j,k) candidates for the given learnable mask.
///
/// Pruning rules:
/// - k must be the head relation for this mask
/// - At least one of (i,j) must be a base relation with nonzero tuples
/// - Template+template body pairs are pruned
/// - If allow_recursive is false: i==k_head or j==k_head are pruned
///
/// Returns list of dicts: [{id, i, j, k, left_name, right_name, head_name}]
/// IDs assigned 0..C-1 after sorting by (k, i, j) ascending.
#[pyo3(signature = (mask_name, allow_recursive=false))]
fn valid_candidates(
    &self,
    py: Python<'_>,
    mask_name: String,
    allow_recursive: bool,
) -> PyResult<Vec<HashMap<String, PyObject>>> {
    let n = self.compiled_schema_size;
    let head_name = &self.head_rel_name;  // need to add this field

    // Find head index in rel_index
    let k_head = self.rel_index.iter()
        .position(|(_, name)| name == head_name)
        .ok_or_else(|| PyValueError::new_err(
            format!("head relation '{}' not in rel_index", head_name)
        ))? as u32;

    // Identify which relations have nonzero tuples in the store
    let has_tuples: Vec<bool> = self.rel_index.iter()
        .map(|(_, name)| {
            self.executor.store.get(name)
                .map(|buf| buf.num_rows().unwrap_or(0) > 0)
                .unwrap_or(false)
        })
        .collect();

    // Identify template variables (bL, bR, b1, b2 etc — no base facts)
    // Templates are relations in rel_index that have zero tuples
    // and are body-variable placeholders from lower_learnable_rule

    let mut candidates: Vec<(u32, u32, u32)> = Vec::new();

    for i in 0..n as u32 {
        for j in 0..n as u32 {
            let k = k_head;  // only head relation is valid target

            // Prune: both body atoms must not both be template (no tuples)
            if !has_tuples[i as usize] && !has_tuples[j as usize] {
                continue;
            }

            // Prune: recursive (body references head) unless allowed
            if !allow_recursive && (i == k || j == k) {
                // Allow if head already has base facts
                if !has_tuples[k as usize] {
                    continue;
                }
            }

            candidates.push((i, j, k));
        }
    }

    // Sort by (k, i, j) and assign stable IDs
    candidates.sort();

    let result: Vec<HashMap<String, PyObject>> = candidates.iter()
        .enumerate()
        .map(|(id, &(i, j, k))| {
            let mut d = HashMap::new();
            d.insert("id".into(), id.to_object(py));
            d.insert("i".into(), i.to_object(py));
            d.insert("j".into(), j.to_object(py));
            d.insert("k".into(), k.to_object(py));
            d.insert("left_name".into(),
                     self.rel_index[i as usize].1.to_object(py));
            d.insert("right_name".into(),
                     self.rel_index[j as usize].1.to_object(py));
            d.insert("head_name".into(),
                     self.rel_index[k as usize].1.to_object(py));
            d
        })
        .collect();

    Ok(result)
}
```

Also add `head_rel_name: String` field to `CompiledIlpProgram` struct
(line ~3830) and populate it in the `compile()` method by extracting the
head relation name from the learnable rule's AST.

**Step 4: Rebuild and run tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: All 6 tests PASS

**Step 5: Run existing ILP tests for regression**

Run: `.venv/bin/python -m pytest python/tests/test_ilp.py -v`
Expected: All 9 tests PASS

**Step 6: Commit**

```bash
git add crates/pyxlog/src/lib.rs python/tests/test_ilp_candidates.py
git commit -m "feat(ilp): add valid_candidates() API with candidate pruning"
```

---

### Task 2: Make max_active_rules configurable

**Files:**
- Modify: `crates/xlog-logic/src/lower.rs:524` (hardcoded 32)
- Modify: `crates/xlog-ir/src/rir.rs:181-199` (TensorMaskedJoin)
- Modify: `crates/pyxlog/src/lib.rs` (compile method to accept max_active)
- Test: `python/tests/test_ilp_candidates.py` (extend)

**Step 1: Write the failing test**

Add to `python/tests/test_ilp_candidates.py`:

```python
def test_max_active_rules_configurable():
    """Compilation accepts max_active_rules parameter."""
    prog = pyxlog.IlpProgramFactory.compile(
        SOURCE, device=0, memory_mb=512, max_active_rules=64
    )
    assert prog.ilp_schema_size() > 0


def test_max_active_rules_default_is_32():
    """Default max_active_rules is 32 (backward compatible)."""
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    # No way to query it directly, but compilation succeeds with default
    assert prog.ilp_schema_size() > 0


def test_max_active_rules_rejects_out_of_range():
    """Values outside 16-128 are rejected."""
    with pytest.raises(ValueError):
        pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512, max_active_rules=5)
    with pytest.raises(ValueError):
        pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512, max_active_rules=500)
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_candidates.py::test_max_active_rules_configurable -v`
Expected: FAIL with `TypeError: compile() got an unexpected keyword argument 'max_active_rules'`

**Step 3: Implement configurable max_active_rules**

1. In `crates/pyxlog/src/lib.rs`, `IlpProgramFactory::compile()` (~line 3767):
   add `max_active_rules: Option<usize>` param with `#[pyo3(signature = (source, device=0, memory_mb=512, max_active_rules=None))]`.
   Validate: if provided, must be 16..=128; if None, default to 32.
   Pass through to the compiler/lowering phase.

2. In `crates/xlog-logic/src/lower.rs:524`: accept `max_active_rules` as a
   parameter to `lower_learnable_rule()` instead of hardcoding 32.

3. Threading: `compile()` → `compiler.compile_program()` needs to carry the
   value. Either add it to `Compiler` config or pass to `lower_learnable_rule()`.

**Step 4: Rebuild and run tests**

Run: `.venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: All tests PASS (including new ones)

**Step 5: Regression check**

Run: `.venv/bin/python -m pytest python/tests/test_ilp.py -v`
Expected: All 9 tests PASS (default 32 preserves behavior)

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/lower.rs crates/pyxlog/src/lib.rs python/tests/test_ilp_candidates.py
git commit -m "feat(ilp): make max_active_rules configurable (16-128, default 32)"
```

---

## Track B: Python Trainer Module

### Task 3: Create pyxlog.ilp module with types and exceptions

**Files:**
- Modify: `crates/pyxlog/pyproject.toml` (add `python-source`, rename module)
- Create: `python/pyxlog/__init__.py` (re-export native module)
- Create: `python/pyxlog/ilp/__init__.py`
- Create: `python/pyxlog/ilp/types.py`
- Create: `python/pyxlog/ilp/exceptions.py`
- Test: `python/tests/test_ilp_types.py` (create)

**Step 1: Set up maturin mixed Python/Rust layout**

Current state: pyxlog is a pure cdylib. Maturin auto-generates `__init__.py`
at install time. To add pure-Python sub-packages (`pyxlog.ilp`), switch to
maturin's mixed layout:

1. Add `python-source` and rename native module in `crates/pyxlog/pyproject.toml`:

```toml
[tool.maturin]
python-source = "python"
module-name = "pyxlog._native"
features = ["pyo3/extension-module", "host-io"]
```

2. Create `python/pyxlog/__init__.py` (replaces maturin's auto-generated one):

```python
# Re-export everything from the native Rust module
from pyxlog._native import *  # noqa: F401,F403
from pyxlog._native import __doc__

if hasattr(_native, "__all__"):
    __all__ = _native.__all__
```

Note: The `python/` directory here is relative to `crates/pyxlog/` (where
pyproject.toml lives), so the actual path is `crates/pyxlog/python/pyxlog/`.
Alternatively, use an absolute path. Check maturin docs — `python-source`
can also be `../../python` to point to the repo-level python directory.
Test with `maturin develop --release` to confirm imports work.

3. Verify: `python -c "import pyxlog; print(pyxlog.IlpProgramFactory)"` must still work.
4. Verify: `python -c "from pyxlog.ilp import TrainConfig"` must work after creating types.

**Step 2: Write type definitions**

`python/pyxlog/ilp/types.py`:

```python
"""dILP trainer types — see docs/plans/2026-02-26-dilp-hardening-design.md"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


@dataclass(frozen=True)
class TrainConfig:
    """Optimizer configuration for dILP training. Frozen after creation."""

    # Budget
    global_step_limit: int = 1000
    step_budget_per_attempt: int = 150
    max_attempts: int = 5

    # Adaptive temperature
    tau_start: float = 2.0
    tau_floor: float = 0.05
    plateau_window: int = 10
    plateau_threshold: float = 0.01
    warmup_increment: float = 0.3
    trap_disc_threshold: float = 0.85
    trap_progress_window: int = 15

    # Entropy regularization
    entropy_weight_start: float = 0.1
    entropy_weight_end: float = 0.0

    # Multi-start
    random_reserve_fraction: float = 0.2
    narrowing_top_fraction: float = 0.5

    # Hard-negative mining
    max_mined_negatives: int = 10

    # Scaling
    max_active_rules: int = 32
    debug_dense_mask: bool = True   # alpha: dense; beta: sparse default

    # Recursion
    allow_recursive_candidates: bool = False

    # Guarantees
    check_ambiguity: bool = False
    exhaustive_ambiguity: bool = False
    max_novel_rate: float = 0.0
    protected_relations: tuple[str, ...] = ()
    holdout_strategy: str = "loo"
    waiver_untyped: bool = False

    # Reproducibility
    seed: int | None = None
    deterministic: bool = False

    # Compilation
    device: int = 0
    memory_mb: int = 512

    # Observability
    telemetry_level: int = 0
    max_telemetry_steps: int = 1000
    telemetry_sink: Path | None = None

    # Numeric stability
    max_numeric_failures: int = 3


class PromotionStatus(Enum):
    PROMOTED = "promoted"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    COMMIT_FAILED = "commit_failed"
    NOT_CONVERGED = "not_converged"
    GATE_FAILED = "gate_failed"


@dataclass
class CandidateMapEntry:
    id: int
    i: int
    j: int
    k: int
    left_name: str
    right_name: str
    head_name: str


@dataclass
class StepRecord:
    step: int
    loss: float
    argmax_rule: str
    discreteness: float
    temperature: float
    entropy: float
    stable_count: int
    forward_p95_us: float = 0.0
    active_candidates: int = 0


@dataclass
class TrainTelemetry:
    steps: list[StepRecord] = field(default_factory=list)
    step_timings: dict | None = None


@dataclass
class ArtifactMetadata:
    pyxlog_version: str = ""
    git_sha: str | None = None
    cuda_version: str = ""
    device_name: str = ""
    candidate_map_hash: str = ""
    config_hash: str = ""
    timestamp_utc: str = ""


@dataclass
class LearnedArtifact:
    candidate_map: list[CandidateMapEntry] = field(default_factory=list)
    logits: list[float] = field(default_factory=list)
    soft_probs: list[float] = field(default_factory=list)
    selected_hard: list[int] = field(default_factory=list)
    discovered_rule: str = ""
    config_snapshot: TrainConfig | None = None
    telemetry: TrainTelemetry = field(default_factory=TrainTelemetry)
    metadata: ArtifactMetadata = field(default_factory=ArtifactMetadata)


@dataclass
class TrainResult:
    converged: bool = False
    discovered_rule: str | None = None
    attempt_count: int = 0
    total_steps: int = 0
    precision: float = 0.0
    recall: float = 0.0
    holdout_f1: float | None = None
    confidence_margin: float = 0.0
    top_k_concentration: float = 0.0
    rule_frequency: float = 0.0
    holdout_variance: float = 0.0
    single_attempt: bool = True
    ambiguous_alternatives: list[str] | None = None
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)


@dataclass
class GateResult:
    name: str = ""
    passed: bool = False
    detail: str = ""


@dataclass
class PromotionResult:
    status: PromotionStatus = PromotionStatus.NOT_CONVERGED
    gates: list[GateResult] = field(default_factory=list)
    novel_count: int | None = None
    novel_rate: float | None = None
    novel_examples: list[str] | None = None
    artifact: LearnedArtifact = field(default_factory=LearnedArtifact)
```

`python/pyxlog/ilp/exceptions.py`:

```python
"""dILP exception taxonomy."""


class IlpConfigError(ValueError):
    """Invalid TrainConfig or empty examples. Raised before GPU work."""


class IlpCandidateError(ValueError):
    """No valid candidates for the given mask. Cannot train."""


class IlpTrainingError(RuntimeError):
    """CUDA or numerical failure during training.

    Attributes:
        context: dict with keys: attempt, step, C, k, device_name,
                 allocated_bytes, terminal_reason.
    """

    def __init__(self, message: str, context: dict | None = None):
        super().__init__(message)
        self.context = context or {}
```

`python/pyxlog/ilp/__init__.py`:

```python
"""dILP trainer module."""
from pyxlog.ilp.exceptions import IlpCandidateError, IlpConfigError, IlpTrainingError
from pyxlog.ilp.types import (
    ArtifactMetadata,
    CandidateMapEntry,
    GateResult,
    LearnedArtifact,
    PromotionResult,
    PromotionStatus,
    StepRecord,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)

__all__ = [
    "IlpConfigError",
    "IlpCandidateError",
    "IlpTrainingError",
    "TrainConfig",
    "TrainResult",
    "TrainTelemetry",
    "StepRecord",
    "LearnedArtifact",
    "ArtifactMetadata",
    "CandidateMapEntry",
    "PromotionStatus",
    "PromotionResult",
    "GateResult",
]
```

**Step 3: Write tests**

```python
# python/tests/test_ilp_types.py
"""Tests for dILP type definitions and exceptions."""
from pyxlog.ilp import (
    TrainConfig,
    TrainResult,
    PromotionStatus,
    IlpConfigError,
    IlpCandidateError,
    IlpTrainingError,
)


def test_train_config_frozen():
    cfg = TrainConfig()
    try:
        cfg.tau_start = 99.0  # type: ignore
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.global_step_limit == 1000
    assert cfg.max_attempts == 5
    assert cfg.tau_floor == 0.05
    assert cfg.max_active_rules == 32
    assert cfg.allow_recursive_candidates is False
    assert cfg.max_numeric_failures == 3
    assert cfg.device == 0
    assert cfg.memory_mb == 512


def test_train_config_custom():
    cfg = TrainConfig(tau_start=3.0, max_attempts=10, seed=42)
    assert cfg.tau_start == 3.0
    assert cfg.max_attempts == 10
    assert cfg.seed == 42


def test_promotion_status_values():
    assert PromotionStatus.PROMOTED.value == "promoted"
    assert PromotionStatus.MANUAL_REVIEW_REQUIRED.value == "manual_review_required"
    assert PromotionStatus.COMMIT_FAILED.value == "commit_failed"
    assert PromotionStatus.NOT_CONVERGED.value == "not_converged"
    assert PromotionStatus.GATE_FAILED.value == "gate_failed"


def test_train_result_defaults():
    result = TrainResult()
    assert result.converged is False
    assert result.discovered_rule is None
    assert result.attempt_count == 0


def test_ilp_config_error_is_value_error():
    assert issubclass(IlpConfigError, ValueError)


def test_ilp_candidate_error_is_value_error():
    assert issubclass(IlpCandidateError, ValueError)


def test_ilp_training_error_has_context():
    err = IlpTrainingError("CUDA OOM", {"attempt": 2, "step": 50, "C": 100})
    assert err.context["attempt"] == 2
    assert err.context["step"] == 50
    assert "CUDA OOM" in str(err)


def test_ilp_training_error_default_context():
    err = IlpTrainingError("generic failure")
    assert err.context == {}
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_types.py -v`
Expected: All tests PASS (pure Python, no CUDA required)

**Step 5: Commit**

```bash
git add python/pyxlog/ilp/ python/tests/test_ilp_types.py
git commit -m "feat(ilp): add pyxlog.ilp types and exception taxonomy"
```

---

### Task 4: Implement adaptive temperature controller

**Files:**
- Create: `python/pyxlog/ilp/temperature.py`
- Test: `python/tests/test_ilp_temperature.py` (create)

**Step 1: Write the failing tests**

```python
# python/tests/test_ilp_temperature.py
"""Tests for adaptive temperature controller."""
from pyxlog.ilp.temperature import AdaptiveTempController, TempMode


def test_initial_mode_is_cooling():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    assert controller.mode == TempMode.COOLING
    assert controller.tau == 2.0


def test_cooling_decreases_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    tau0 = controller.tau
    controller.step(loss=5.0, disc=0.3, witness_coverage=0.0)
    assert controller.tau < tau0


def test_tau_never_below_floor():
    controller = AdaptiveTempController(
        tau_start=0.1, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=10,
    )
    for _ in range(20):
        controller.step(loss=0.1, disc=0.3, witness_coverage=1.0)
    assert controller.tau >= 0.05


def test_plateau_holds_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=3,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    for _ in range(10):
        controller.step(loss=1.0, disc=0.5, witness_coverage=0.5)
    tau_at_plateau = controller.tau
    controller.step(loss=1.0, disc=0.5, witness_coverage=0.5)
    assert controller.mode == TempMode.PLATEAU
    assert controller.tau == tau_at_plateau


def test_trap_warms_up_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=3,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=3,
        total_steps=100,
    )
    for _ in range(5):
        controller.step(loss=0.5, disc=0.95, witness_coverage=0.0)
    assert controller.mode == TempMode.WARMUP
    assert controller.tau > 0.05
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_temperature.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement the controller**

Create `python/pyxlog/ilp/temperature.py` implementing `AdaptiveTempController`
with three modes (COOLING, PLATEAU, WARMUP). See design Section 2.1.
Key logic:
- COOLING: linear decrease from tau_start toward tau_floor
- PLATEAU: EMA loss flat over window → hold tau constant
- WARMUP: disc > threshold AND no witness coverage progress → bump tau

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_temperature.py -v`
Expected: All 5 tests PASS

**Step 5: Commit**

```bash
git add python/pyxlog/ilp/temperature.py python/tests/test_ilp_temperature.py
git commit -m "feat(ilp): adaptive temperature controller (COOLING/PLATEAU/WARMUP)"
```

---

### Task 5: Implement entropy regularization

**Files:**
- Create: `python/pyxlog/ilp/entropy.py`
- Test: `python/tests/test_ilp_entropy.py` (create)

**Step 1: Write the failing tests**

```python
# python/tests/test_ilp_entropy.py
"""Tests for entropy regularization."""
import pytest

torch = pytest.importorskip("torch")


def test_normalized_entropy_uniform():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.zeros(C)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert abs(h.item() - 1.0) < 1e-5


def test_normalized_entropy_one_hot():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.full((C,), -100.0)
    logits[0] = 100.0
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert h.item() < 0.01


def test_normalized_entropy_gradient_flows():
    from pyxlog.ilp.entropy import normalized_entropy
    C = 10
    logits = torch.randn(C, requires_grad=True)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    h.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_entropy_weight_decay():
    from pyxlog.ilp.entropy import entropy_weight_at_step
    w0 = entropy_weight_at_step(0, 100, start=0.1, end=0.0)
    w50 = entropy_weight_at_step(50, 100, start=0.1, end=0.0)
    w100 = entropy_weight_at_step(100, 100, start=0.1, end=0.0)
    assert abs(w0 - 0.1) < 1e-6
    assert abs(w50 - 0.05) < 1e-6
    assert abs(w100 - 0.0) < 1e-6
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_entropy.py -v`
Expected: FAIL with `ModuleNotFoundError`

**Step 3: Implement**

Create `python/pyxlog/ilp/entropy.py`:
- `normalized_entropy(probs, C)` → `H / log(C)`, numerically stable via clamped log
- `entropy_weight_at_step(step, total, start, end)` → linear decay

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_entropy.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

```bash
git add python/pyxlog/ilp/entropy.py python/tests/test_ilp_entropy.py
git commit -m "feat(ilp): normalized entropy regularization with linear weight decay"
```

---

### Task 6: Implement train_only() core training loop

**Files:**
- Create: `python/pyxlog/ilp/trainer.py`
- Modify: `python/pyxlog/ilp/__init__.py` (export train_only)
- Test: `python/tests/test_ilp_trainer.py` (create)

This is the largest task. It integrates Tasks 1-5 into a working training loop.

**Alpha training loop uses the dense N³ mask** (same path as current showcase).
The `valid_candidates()` output is used for:
- Determining the candidate count C for confidence/coverage metrics
- Decoding the argmax (i,j,k) triple into a readable rule string
- Building the candidate map for the artifact

The mask itself is still built as a full N³ tensor via `build_budget_aware_mask()`
and passed to `set_rule_mask()` via DLPack. Factorized scoring (only C logits)
is deferred to RC when the sparse mask API is ready.

**Step 1: Write the failing integration tests**

```python
# python/tests/test_ilp_trainer.py
"""Integration tests for train_only()."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP trainer tests", allow_module_level=True)

from pyxlog.ilp import train_only, TrainConfig, TrainResult, IlpConfigError

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
REACH_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
REACH_NEG = []


def test_train_only_converges_on_reach():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert isinstance(result, TrainResult)
    assert result.converged
    assert "edge" in result.discovered_rule
    assert "reach" in result.discovered_rule
    assert result.attempt_count >= 1
    assert result.total_steps > 0


def test_train_only_returns_telemetry_level_1():
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert len(result.artifact.telemetry.steps) > 0
    step0 = result.artifact.telemetry.steps[0]
    assert step0.step == 0
    assert isinstance(step0.loss, float)
    assert isinstance(step0.temperature, float)


def test_train_only_empty_positives_raises():
    config = TrainConfig(max_attempts=1)
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=[], negatives=[], config=config,
        )


def test_train_only_contradictory_examples_raises():
    config = TrainConfig(max_attempts=1)
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradict"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=pos, negatives=neg, config=config,
        )


def test_train_only_precision_recall():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert result.precision > 0.0
        assert result.recall > 0.0


def test_train_only_confidence_metrics():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert 0.0 <= result.confidence_margin <= 1.0
        assert 0.0 <= result.top_k_concentration <= 1.0
        assert 0.0 <= result.rule_frequency <= 1.0


def test_train_only_global_step_limit():
    config = TrainConfig(
        global_step_limit=50, step_budget_per_attempt=30,
        max_attempts=10, tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert result.total_steps <= 50
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_only'`

**Step 3: Implement train_only**

Create `python/pyxlog/ilp/trainer.py` (~300 lines). Structure:

```python
def train_only(source, mask_name, positives, negatives, config, ...) -> TrainResult:
    _validate_inputs(...)
    attempts = []
    for attempt_idx in range(config.max_attempts):
        # Fresh compile per attempt (isolation)
        prog = pyxlog.IlpProgramFactory.compile(
            source, device=config.device, memory_mb=config.memory_mb,
            max_active_rules=config.max_active_rules,
        )
        # Get candidates for bookkeeping
        candidates = prog.valid_candidates(mask_name, config.allow_recursive_candidates)
        # Training loop with dense N³ mask
        result = _run_single_attempt(prog, mask_name, candidates, ...)
        attempts.append(result)
    return _select_winner(attempts, config)
```

Key internals (adapted from `ilp_showcase.py` patterns):
- `_run_single_attempt()`: builds N³ weight tensor, training loop with
  `build_budget_aware_mask()` → `set_rule_mask()` → `evaluate()` →
  `compute_loss()` → optimizer step. Uses AdaptiveTempController and entropy.
- `_build_budget_aware_mask()`: ST-Gumbel-Softmax over N³ tensor (same as showcase)
- `_compute_loss()`: per-fact surrogate credit + missed-positive + negative penalty + entropy
- `_check_convergence()`: stable argmax + all_derived + argmax-only validation
- `_select_winner()`: deterministic tie-break chain across attempts

**Step 4: Run tests**

Run: `.venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v --timeout=300`
Expected: All 7 tests PASS

**Step 5: Full regression**

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=600`
Expected: All existing + new tests pass

**Step 6: Commit**

```bash
git add python/pyxlog/ilp/trainer.py python/pyxlog/ilp/__init__.py python/tests/test_ilp_trainer.py
git commit -m "feat(ilp): train_only() with adaptive temp + entropy + multi-start"
```

---

### Task 7: Multi-start integration tests with distractors

**Files:**
- Create: `python/tests/test_ilp_multistart.py`

**Step 1: Write tests**

```python
# python/tests/test_ilp_multistart.py
"""Integration tests for structured multi-start with distractor relations."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP tests", allow_module_level=True)

from pyxlog.ilp import train_only, TrainConfig

GRANDPARENT_SOURCE = """
    parent(1, 2). parent(2, 3). parent(3, 4). parent(4, 5).
    gender(1, 0). gender(2, 1). gender(3, 0). gender(4, 1).
    sibling(1, 3). sibling(3, 1).
    learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""
GRANDPARENT_POS = [("grandparent", [1, 3]), ("grandparent", [2, 4]), ("grandparent", [3, 5])]
GRANDPARENT_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]


def test_multistart_converges_with_distractors():
    config = TrainConfig(
        step_budget_per_attempt=120, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE, mask_name="W_gp",
        positives=GRANDPARENT_POS, negatives=GRANDPARENT_NEG, config=config,
    )
    assert result.converged
    assert "parent" in result.discovered_rule


def test_multistart_reports_attempt_count():
    config = TrainConfig(
        step_budget_per_attempt=120, max_attempts=7,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE, mask_name="W_gp",
        positives=GRANDPARENT_POS, negatives=GRANDPARENT_NEG, config=config,
    )
    assert 1 <= result.attempt_count <= 7
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_multistart.py -v --timeout=300`
Expected: PASS

**Step 3: Commit**

```bash
git add python/tests/test_ilp_multistart.py
git commit -m "test(ilp): multi-start integration tests with distractor relations"
```

---

### Task 8: Robustness tests including NaN/Inf behavior

**Files:**
- Create: `python/tests/test_ilp_robustness.py`

**Step 1: Write tests**

```python
# python/tests/test_ilp_robustness.py
"""Robustness tests: input validation, NaN/Inf detection, edge cases."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP tests", allow_module_level=True)

from pyxlog.ilp import train_only, TrainConfig, IlpConfigError, IlpTrainingError

SOURCE = """
    edge(1, 2). edge(2, 3).
    learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_empty_positives_raises_config_error():
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(SOURCE, "W", [], [], TrainConfig(max_attempts=1))


def test_contradictory_examples_raises_config_error():
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradict"):
        train_only(SOURCE, "W", pos, neg, TrainConfig(max_attempts=1))


def test_all_distractors_returns_not_converged():
    """When no useful relations exist, training should not converge but not crash."""
    # Only distractor relations, no useful base for reach
    distractor_source = """
        color(1, 0). color(2, 1).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=1.0, tau_floor=0.1, seed=0,
    )
    result = train_only(
        source=distractor_source, mask_name="W",
        positives=[("reach", [1, 3])], negatives=[], config=config,
    )
    # May or may not converge, but must not crash
    assert isinstance(result.converged, bool)
    assert result.total_steps > 0


def test_nan_inf_detection_raises_after_limit():
    """NaN/Inf in logits triggers IlpTrainingError after max_numeric_failures.

    Strategy: monkey-patch the trainer's internal _build_budget_aware_mask to
    inject NaN into the weight tensor, then verify the error path fires.
    """
    from unittest.mock import patch
    import pyxlog.ilp.trainer as trainer_mod

    original_fn = trainer_mod._build_budget_aware_mask

    call_count = [0]

    def _inject_nan(*args, **kwargs):
        call_count[0] += 1
        result = original_fn(*args, **kwargs)
        # Corrupt the soft mask to force NaN in loss computation
        result[1].fill_(float("nan"))  # result is (M_hard, M_soft) or similar
        return result

    config = TrainConfig(
        step_budget_per_attempt=10, max_attempts=5,
        max_numeric_failures=2, seed=0,
    )

    with patch.object(trainer_mod, "_build_budget_aware_mask", _inject_nan):
        with pytest.raises(IlpTrainingError, match="numeric_instability"):
            train_only(
                source=SOURCE, mask_name="W",
                positives=[("reach", [1, 3])], negatives=[], config=config,
            )
    # Should have tried exactly max_numeric_failures attempts before raising
    assert call_count[0] >= 2
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_robustness.py -v --timeout=120`
Expected: PASS

**Step 3: Commit**

```bash
git add python/tests/test_ilp_robustness.py
git commit -m "test(ilp): robustness tests for validation, distractors, numeric policy"
```

---

### Task 9: Confidence metrics tests

**Files:**
- Extend: `python/tests/test_ilp_trainer.py` (already has confidence test from Task 6)

This task verifies that `rule_frequency` works across multi-attempt runs.

**Step 1: Add test to `test_ilp_trainer.py`**

```python
def test_train_only_rule_frequency_multi_attempt():
    """rule_frequency reflects how many attempts found the winning rule."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        # At least the winning attempt found this rule
        assert result.rule_frequency >= 1.0 / result.attempt_count
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_trainer.py::test_train_only_rule_frequency_multi_attempt -v`
Expected: PASS

**Step 3: Commit**

```bash
git add python/tests/test_ilp_trainer.py
git commit -m "test(ilp): rule_frequency metric assertion for multi-attempt runs"
```

---

### Task 10: Alpha reliability gate — 5/5 consecutive passes

**Files:**
- Create: `python/tests/test_ilp_reliability.py`

This is the alpha exit gate: all 4 showcase stages converge with 5 different seeds.

**Step 1: Write the reliability test**

```python
# python/tests/test_ilp_reliability.py
"""Alpha reliability gate: 5/5 consecutive passes on showcase suite.

See design doc Section 6.1. Validates train_only() converges on all 4 stages
with seeds 0..4.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP reliability tests", allow_module_level=True)

from pyxlog.ilp import train_only, TrainConfig

# --- Stage definitions (matching ilp_showcase.py domains) ---

STAGE_1_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_1_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
STAGE_1_NEG = []

STAGE_2_SOURCE = """
    parent(1, 2). parent(2, 3). parent(2, 4). parent(3, 5). parent(4, 6).
    gender(1, 0). gender(2, 1). gender(3, 1). gender(4, 0).
    sibling(2, 7). sibling(7, 2).
    learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_2_POS = [("grandparent", [1, 3]), ("grandparent", [1, 4]),
               ("grandparent", [2, 5]), ("grandparent", [2, 6])]
STAGE_2_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]

STAGE_3_SOURCE = """
    worksAt(1, 100). worksAt(7, 100). worksAt(2, 200). worksAt(4, 200).
    livesIn(1, 201). livesIn(2, 201). livesIn(7, 202). livesIn(4, 202).
    learnable(W_col) :: colleague(X, Y) :- bL(X, Z), bR(Y, Z).
"""
STAGE_3_POS = [("colleague", [1, 7]), ("colleague", [7, 1]),
               ("colleague", [2, 4]), ("colleague", [4, 2])]
STAGE_3_NEG = [("colleague", [1, 2]), ("colleague", [3, 4])]

STAGE_4_SOURCE = """
    succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4). succ(4, 5).
    pred(1, 0). pred(2, 1). pred(3, 2). pred(4, 3). pred(5, 4).
    learnable(W_p2) :: plus2(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_4_POS = [("plus2", [0, 2]), ("plus2", [1, 3]), ("plus2", [2, 4]), ("plus2", [3, 5])]
STAGE_4_NEG = [("plus2", [0, 1]), ("plus2", [5, 0])]

STAGES = [
    ("reach", STAGE_1_SOURCE, STAGE_1_POS, STAGE_1_NEG, "W_reach"),
    ("grandparent", STAGE_2_SOURCE, STAGE_2_POS, STAGE_2_NEG, "W_gp"),
    ("colleague", STAGE_3_SOURCE, STAGE_3_POS, STAGE_3_NEG, "W_col"),
    ("plus2", STAGE_4_SOURCE, STAGE_4_POS, STAGE_4_NEG, "W_p2"),
]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "stage_name,source,positives,negatives,mask_name",
    STAGES,
    ids=["reach", "grandparent", "colleague", "plus2"],
)
def test_alpha_reliability(stage_name, source, positives, negatives, mask_name, seed):
    """Each stage converges with each seed (alpha gate: 5/5)."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=seed,
        device=0,
        memory_mb=512,
    )
    result = train_only(
        source=source,
        mask_name=mask_name,
        positives=positives,
        negatives=negatives,
        config=config,
    )
    assert result.converged, (
        f"Stage {stage_name} failed with seed={seed}: "
        f"attempts={result.attempt_count}, steps={result.total_steps}, "
        f"best_rule={result.discovered_rule}"
    )
```

**Step 2: Run the reliability gate**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600`
Expected: All 20 tests PASS (4 stages × 5 seeds)

**Step 3: Commit**

```bash
git add python/tests/test_ilp_reliability.py
git commit -m "test(ilp): alpha reliability gate — 5/5 on all 4 showcase stages"
```

---

## Task Dependencies

```
Task 1 (valid_candidates Rust) ──────────┐
Task 2 (configurable max_active_rules) ──┤
                                         ├─→ Task 6 (train_only core) ─→ Task 7 (multi-start tests)
Task 3 (Python types + exceptions) ──────┤                             ├─→ Task 8 (robustness tests)
Task 4 (adaptive temperature) ───────────┤                             ├─→ Task 9 (confidence metrics)
Task 5 (entropy regularization) ─────────┘                             └─→ Task 10 (reliability gate)
```

Tasks 1-5 can be implemented in parallel (no interdependencies).
Task 6 depends on all of 1-5.
Tasks 7-10 depend on Task 6 and can run in parallel.

## Build & Test Commands

```bash
# Rebuild Rust bindings (needed after Task 1, 2)
cd /home/dev/projects/xlog && .venv/bin/maturin develop --release

# Run Python tests (no CUDA needed)
.venv/bin/python -m pytest python/tests/test_ilp_types.py -v

# Run Python tests (CUDA required)
.venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v
.venv/bin/python -m pytest python/tests/test_ilp_temperature.py -v
.venv/bin/python -m pytest python/tests/test_ilp_entropy.py -v
.venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v --timeout=300
.venv/bin/python -m pytest python/tests/test_ilp_multistart.py -v --timeout=300
.venv/bin/python -m pytest python/tests/test_ilp_robustness.py -v --timeout=120
.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600

# Run Rust tests (regression)
cargo test --workspace --all-targets --exclude pyxlog --release

# Run full Python test suite
.venv/bin/python -m pytest python/tests/ -v --timeout=600
```

## Beta/RC/GA Deferred Tasks (not in this plan)

- **Beta B1:** `set_rule_mask_sparse()` Rust API (Phase 2 sparse path)
- **Beta B2:** `train_and_promote()` + promotion gates
- **Beta B3:** `sample_false_positives()` Rust API + hard-negative mining
- **Beta B4:** Holdout LOO/k-fold cross-validation
- **Beta B5:** Ambiguity detection (top-256 scan)
- **Beta B6:** Recursive candidates (`allow_recursive_candidates = True`)
- **Beta B7:** Artifact save/load (JSON)
- **Beta B8:** 20/20 reliability gate
- **RC R1:** CUDA event profiling counters (4 sub-timers)
- **RC R2:** Factorized scoring over C-vector (replacing N³ in training loop)
- **GA G1:** Typed schemas + waiver policy
- **GA G2:** Telemetry level 2 streaming to sink
- **GA G3:** 50/50 reliability gate + Clopper-Pearson CI
- **GA G4:** Deterministic mode + reproducibility gate
- **GA G5:** Performance SLO benchmarks (N=20/50/150)
- **GA G6:** Robustness scenario suite (all 6 adversarial inputs)
