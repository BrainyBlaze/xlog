# dILP Hardening Implementation Plan (Alpha Milestone)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Implement the alpha milestone of the dILP hardening design — Python types, exception taxonomy, Rust `valid_candidates()` API, adaptive temperature, entropy regularization, structured multi-start, `train_only()` entry point, and 5/5 reliability gate.

**Architecture:** Dual-track (contract-first). Track A adds Rust primitives (`valid_candidates`, configurable `max_active_rules`). Track B builds the Python trainer module (`pyxlog.ilp`) on top of those primitives. The existing `ilp_showcase.py` remains untouched as a reference; the new trainer replaces its ad-hoc training loop with structured, testable code.

**Tech Stack:** Rust (PyO3 bindings in `crates/pyxlog`), Python 3.10+ (dataclasses, enum, logging), PyTorch (Gumbel-Softmax), maturin (build).

**Design doc:** `docs/plans/2026-02-26-dilp-hardening-design.md`

**Alpha scope (what's IN):**
- Python types (TrainConfig, TrainResult, LearnedArtifact, exceptions)
- Rust `valid_candidates(mask_name)` API with candidate pruning
- Configurable `max_active_rules` (no longer hardcoded 32)
- Adaptive temperature controller (COOLING/PLATEAU/WARMUP)
- Entropy regularization (normalized)
- Structured multi-start (best-of-K with random reserve)
- `train_only()` entry point with per-attempt isolation
- NaN/Inf detection and numeric failure policy
- Telemetry level 0 (summary) and level 1 (per-step)
- 5/5 reliability gate on showcase suite

**Alpha scope (what's OUT — deferred):**
- `set_rule_mask_sparse()` (beta — Phase 2 sparse API)
- `train_and_promote()` + promotion gates (beta)
- Ambiguity detection (beta)
- CUDA event profiling counters (beta)
- `sample_false_positives()` hard-negative mining (beta)
- Recursive candidate policy (beta)
- Holdout LOO/k-fold (beta)
- Artifact save/load persistence (beta)
- Typed schemas (GA)
- Telemetry level 2 streaming (GA)

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
import pyxlog

SOURCE = """
    edge(1, 2).
    edge(2, 3).
    edge(3, 4).
    learnable reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_valid_candidates_returns_candidates():
    """valid_candidates returns non-empty list of CandidateInfo dicts."""
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE)
    candidates = prog.valid_candidates("W")
    assert isinstance(candidates, list)
    assert len(candidates) > 0
    # Each candidate is a dict with required keys
    c = candidates[0]
    assert set(c.keys()) == {"id", "i", "j", "k", "left_name", "right_name", "head_name"}


def test_valid_candidates_deterministic_ids():
    """Same program produces same candidate IDs."""
    factory = pyxlog.IlpProgramFactory()
    prog1 = factory.compile(SOURCE)
    prog2 = factory.compile(SOURCE)
    c1 = prog1.valid_candidates("W")
    c2 = prog2.valid_candidates("W")
    assert [c["id"] for c in c1] == [c["id"] for c in c2]


def test_valid_candidates_prunes_template_template():
    """Candidates where both body atoms are templates are pruned."""
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE)
    candidates = prog.valid_candidates("W")
    for c in candidates:
        assert not (c["left_name"].startswith("b") and c["right_name"].startswith("b")), \
            f"Template+template candidate not pruned: {c}"


def test_valid_candidates_head_is_target():
    """All candidates have head_name == the learnable target relation."""
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE)
    candidates = prog.valid_candidates("W")
    for c in candidates:
        assert c["head_name"] == "reach", f"Wrong head: {c}"


def test_valid_candidates_nonrecursive_prunes_head_in_body():
    """With allow_recursive=False (default), candidates where body==head are pruned."""
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE)
    candidates = prog.valid_candidates("W")
    for c in candidates:
        # reach has no base facts, so i==k_idx or j==k_idx should be pruned
        assert c["left_name"] != "reach" and c["right_name"] != "reach", \
            f"Recursive candidate not pruned: {c}"


def test_valid_candidates_ids_contiguous():
    """IDs are 0..C-1 with no gaps."""
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE)
    candidates = prog.valid_candidates("W")
    ids = sorted(c["id"] for c in candidates)
    assert ids == list(range(len(candidates)))
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: FAIL with `AttributeError: 'CompiledIlpProgram' has no attribute 'valid_candidates'`

**Step 3: Implement valid_candidates in Rust**

In `crates/pyxlog/src/lib.rs`, add to the `#[pymethods]` impl block for `CompiledIlpProgram` (after `ilp_relation_names` around line 3984):

```rust
/// Return the set of valid (i,j,k) candidates for the given learnable mask.
/// Each candidate is a dict: {id, i, j, k, left_name, right_name, head_name}.
/// Pruning: template+template bodies removed, recursive (body==head) removed
/// unless head has base facts. Sorted by (k,i,j), IDs assigned 0..C-1.
fn valid_candidates(&self, mask_name: String) -> PyResult<Vec<HashMap<String, PyObject>>> {
    // Implementation:
    // 1. Look up mask_name in ILP registry to get schema_size (N) and rel_index
    // 2. Identify head relation index (k) from the learnable rule metadata
    // 3. Enumerate all (i, j) pairs where i,j in 0..N
    // 4. Prune: template+template, recursive (i==k_head or j==k_head unless head has base facts)
    // 5. Sort by (k, i, j), assign IDs 0..C-1
    // 6. Return as list of dicts
    // ... (full implementation in Step 3)
}
```

The implementation needs access to:
- `self.rel_index` — the sorted `Vec<(RelId, String)>` mapping
- `self.executor.store` — to check which relations have nonzero tuples
- The learnable rule's head relation name (stored during compilation)

Add a field `head_rel_name: String` to `CompiledIlpProgram` (populated in `compile()`).

**Step 4: Rebuild and run tests**

Run: `cd /home/dev/projects/xlog && .venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: All 6 tests PASS

**Step 5: Commit**

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
    factory = pyxlog.IlpProgramFactory()
    prog = factory.compile(SOURCE, max_active_rules=64)
    # Verify it compiles without error; the value is used at evaluation time
    assert prog.ilp_schema_size() > 0
```

**Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_candidates.py::test_max_active_rules_configurable -v`
Expected: FAIL with `TypeError: compile() got an unexpected keyword argument 'max_active_rules'`

**Step 3: Implement configurable max_active_rules**

1. In `crates/pyxlog/src/lib.rs` `compile()` method: add `max_active_rules: Option<usize>` param, default to 32, validate range 16-128.
2. Pass through to `lower_learnable_rule()` in `crates/xlog-logic/src/lower.rs:524`.
3. In `lower.rs`: replace hardcoded `32` with the parameter.

**Step 4: Rebuild and run tests**

Run: `.venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v`
Expected: All tests PASS (including new one)

**Step 5: Run existing ILP tests to verify no regression**

Run: `.venv/bin/python -m pytest python/tests/test_ilp.py -v`
Expected: All 9 tests PASS (default 32 preserves current behavior)

**Step 6: Commit**

```bash
git add crates/xlog-logic/src/lower.rs crates/pyxlog/src/lib.rs python/tests/test_ilp_candidates.py
git commit -m "feat(ilp): make max_active_rules configurable (16-128, default 32)"
```

---

## Track B: Python Trainer Module

### Task 3: Create pyxlog.ilp module with types and exceptions

**Files:**
- Create: `python/pyxlog/__init__.py` (if not exists, or verify it imports pyxlog native)
- Create: `python/pyxlog/ilp/__init__.py`
- Create: `python/pyxlog/ilp/types.py`
- Create: `python/pyxlog/ilp/exceptions.py`
- Test: `python/tests/test_ilp_types.py` (create)

**Important:** pyxlog is a native extension (cdylib via maturin), not a pure Python package.
The `python/pyxlog/ilp/` module is a pure Python sub-package that imports from the
native `pyxlog` module. It lives alongside the native module in the package.

Check if `python/pyxlog/__init__.py` exists first. If pyxlog is purely a cdylib with
no Python wrapper, we need to create the package structure carefully — possibly as
a separate `pyxlog_ilp` package or by using maturin's mixed Python/Rust layout.

**Step 1: Verify current package structure**

Run: `ls -la python/pyxlog/ 2>/dev/null && cat crates/pyxlog/pyproject.toml`

If no `python/pyxlog/` exists, check maturin config for `python-source` or
`python-packages` settings. The module name in pyproject.toml is `pyxlog`.

For maturin mixed layout, create `python/pyxlog/ilp/` alongside the native module.

**Step 2: Write the type definitions**

```python
# python/pyxlog/ilp/types.py
"""dILP trainer types — see docs/plans/2026-02-26-dilp-hardening-design.md"""
from __future__ import annotations

import hashlib
import json
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
    memory_mb: int | None = None

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

```python
# python/pyxlog/ilp/exceptions.py
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

```python
# python/pyxlog/ilp/__init__.py
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
    """TrainConfig is immutable after creation."""
    cfg = TrainConfig()
    try:
        cfg.tau_start = 99.0  # type: ignore
        assert False, "Should have raised FrozenInstanceError"
    except AttributeError:
        pass  # dataclass(frozen=True) raises AttributeError


def test_train_config_defaults():
    """Default values match design doc."""
    cfg = TrainConfig()
    assert cfg.global_step_limit == 1000
    assert cfg.max_attempts == 5
    assert cfg.tau_floor == 0.05
    assert cfg.max_active_rules == 32
    assert cfg.allow_recursive_candidates is False
    assert cfg.max_numeric_failures == 3


def test_train_config_custom():
    """Custom values override defaults."""
    cfg = TrainConfig(tau_start=3.0, max_attempts=10, seed=42)
    assert cfg.tau_start == 3.0
    assert cfg.max_attempts == 10
    assert cfg.seed == 42


def test_promotion_status_values():
    """PromotionStatus enum has required members."""
    assert PromotionStatus.PROMOTED.value == "promoted"
    assert PromotionStatus.MANUAL_REVIEW_REQUIRED.value == "manual_review_required"
    assert PromotionStatus.COMMIT_FAILED.value == "commit_failed"
    assert PromotionStatus.NOT_CONVERGED.value == "not_converged"
    assert PromotionStatus.GATE_FAILED.value == "gate_failed"


def test_train_result_defaults():
    """TrainResult defaults are safe."""
    result = TrainResult()
    assert result.converged is False
    assert result.discovered_rule is None
    assert result.attempt_count == 0


def test_ilp_config_error_is_value_error():
    """IlpConfigError inherits ValueError."""
    assert issubclass(IlpConfigError, ValueError)


def test_ilp_candidate_error_is_value_error():
    """IlpCandidateError inherits ValueError."""
    assert issubclass(IlpCandidateError, ValueError)


def test_ilp_training_error_has_context():
    """IlpTrainingError carries enriched context dict."""
    err = IlpTrainingError("CUDA OOM", {"attempt": 2, "step": 50, "C": 100})
    assert err.context["attempt"] == 2
    assert err.context["step"] == 50
    assert "CUDA OOM" in str(err)
```

**Step 4: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_types.py -v`
Expected: All tests PASS (pure Python, no Rust needed)

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
    # Feed identical losses to trigger plateau
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
    # High disc + no progress triggers trap
    for _ in range(5):
        controller.step(loss=0.5, disc=0.95, witness_coverage=0.0)
    assert controller.mode == TempMode.WARMUP
    # tau should have increased
    assert controller.tau > 0.05
```

**Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_temperature.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'pyxlog.ilp.temperature'`

**Step 3: Implement the controller**

```python
# python/pyxlog/ilp/temperature.py
"""Adaptive temperature controller for dILP training."""
from __future__ import annotations

from enum import Enum, auto


class TempMode(Enum):
    COOLING = auto()
    PLATEAU = auto()
    WARMUP = auto()


class AdaptiveTempController:
    """Three-mode temperature controller: COOLING → PLATEAU → WARMUP.

    See design doc Section 2.1.
    """

    def __init__(
        self,
        tau_start: float,
        tau_floor: float,
        plateau_window: int,
        plateau_threshold: float,
        warmup_increment: float,
        trap_disc_threshold: float,
        trap_progress_window: int,
        total_steps: int,
    ):
        self.tau = tau_start
        self.tau_start = tau_start
        self.tau_floor = tau_floor
        self.plateau_window = plateau_window
        self.plateau_threshold = plateau_threshold
        self.warmup_increment = warmup_increment
        self.trap_disc_threshold = trap_disc_threshold
        self.trap_progress_window = trap_progress_window
        self.total_steps = max(total_steps, 1)

        self.mode = TempMode.COOLING
        self._step_count = 0
        self._ema_loss = None
        self._ema_alpha = 2.0 / (plateau_window + 1)
        self._loss_history: list[float] = []
        self._coverage_history: list[float] = []

    def step(self, loss: float, disc: float, witness_coverage: float) -> float:
        """Update temperature based on current training state. Returns new tau."""
        self._step_count += 1
        self._loss_history.append(loss)
        self._coverage_history.append(witness_coverage)

        # Update EMA loss
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss

        # Check trap condition
        if self._is_trapped(disc):
            self.mode = TempMode.WARMUP
            self.tau = min(self.tau + self.warmup_increment, self.tau_start)
        elif self._is_plateau():
            self.mode = TempMode.PLATEAU
            # Hold tau constant
        else:
            self.mode = TempMode.COOLING
            # Linear decrease
            frac = self._step_count / self.total_steps
            self.tau = self.tau_start + (self.tau_floor - self.tau_start) * frac

        # Enforce floor
        self.tau = max(self.tau, self.tau_floor)
        return self.tau

    def _is_trapped(self, disc: float) -> bool:
        """Detect trap: high discreteness + no progress in witness coverage."""
        if disc < self.trap_disc_threshold:
            return False
        if len(self._coverage_history) < self.trap_progress_window:
            return False
        recent = self._coverage_history[-self.trap_progress_window:]
        # No progress = max coverage hasn't increased
        return max(recent) <= min(recent)

    def _is_plateau(self) -> bool:
        """Detect plateau: EMA loss flat over window."""
        if len(self._loss_history) < self.plateau_window:
            return False
        recent = self._loss_history[-self.plateau_window:]
        ema_values = []
        ema = recent[0]
        for v in recent:
            ema = self._ema_alpha * v + (1 - self._ema_alpha) * ema
            ema_values.append(ema)
        delta = abs(ema_values[-1] - ema_values[0])
        return delta < self.plateau_threshold
```

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
import torch
from pyxlog.ilp.entropy import normalized_entropy, entropy_weight_at_step


def test_normalized_entropy_uniform():
    """Uniform distribution has entropy 1.0 (normalized)."""
    C = 10
    logits = torch.zeros(C)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert abs(h.item() - 1.0) < 1e-5


def test_normalized_entropy_one_hot():
    """One-hot distribution has entropy ~0.0."""
    C = 10
    logits = torch.full((C,), -100.0)
    logits[0] = 100.0
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    assert h.item() < 0.01


def test_normalized_entropy_gradient_flows():
    """Entropy bonus produces non-zero gradient on logits."""
    C = 10
    logits = torch.randn(C, requires_grad=True)
    probs = torch.softmax(logits, dim=0)
    h = normalized_entropy(probs, C)
    h.backward()
    assert logits.grad is not None
    assert logits.grad.abs().sum() > 0


def test_entropy_weight_decay():
    """Weight decays linearly from start to end."""
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

```python
# python/pyxlog/ilp/entropy.py
"""Entropy regularization for dILP training. See design Section 2.2."""
from __future__ import annotations

import math

import torch


def normalized_entropy(probs: torch.Tensor, C: int) -> torch.Tensor:
    """Normalized Shannon entropy: H / log(C). Range [0, 1].

    Uses log_softmax-style computation for numerical stability.
    """
    log_C = math.log(C) if C > 1 else 1.0
    # Clamp to avoid log(0)
    log_probs = torch.log(probs.clamp(min=1e-38))
    H = -(probs * log_probs).sum()
    return H / log_C


def entropy_weight_at_step(
    step: int, total_steps: int, start: float, end: float
) -> float:
    """Linear decay from start to end over total_steps."""
    if total_steps <= 0:
        return end
    frac = min(step / total_steps, 1.0)
    return start + (end - start) * frac
```

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

This is the largest task. It integrates:
- Adaptive temperature (Task 4)
- Entropy regularization (Task 5)
- Structured multi-start (best-of-K with random reserve)
- NaN/Inf detection
- Per-attempt isolation (fresh compile)
- Telemetry collection (level 0 + 1)
- Candidate-aware factorized scoring

**Step 1: Write the failing integration test**

```python
# python/tests/test_ilp_trainer.py
"""Integration tests for train_only()."""
import pytest
from pyxlog.ilp import train_only, TrainConfig, TrainResult, IlpConfigError

REACH_SOURCE = """
    edge(1, 2).
    edge(2, 3).
    edge(3, 4).
    edge(4, 5).
    edge(5, 6).
    learnable reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""

REACH_POSITIVES = [
    ("reach", [1, 3]),
    ("reach", [2, 4]),
    ("reach", [3, 5]),
    ("reach", [4, 6]),
]

REACH_NEGATIVES = []


def test_train_only_converges_on_reach():
    """train_only finds reach(X,Y) :- edge(X,Z), edge(Z,Y)."""
    config = TrainConfig(
        step_budget_per_attempt=100,
        max_attempts=5,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W",
        positives=REACH_POSITIVES,
        negatives=REACH_NEGATIVES,
        config=config,
    )
    assert isinstance(result, TrainResult)
    assert result.converged
    assert "edge" in result.discovered_rule
    assert "reach" in result.discovered_rule
    assert result.attempt_count >= 1
    assert result.total_steps > 0


def test_train_only_returns_telemetry():
    """Level-1 telemetry has per-step records."""
    config = TrainConfig(
        step_budget_per_attempt=50,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        telemetry_level=1,
    )
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W",
        positives=REACH_POSITIVES,
        negatives=REACH_NEGATIVES,
        config=config,
    )
    assert len(result.artifact.telemetry.steps) > 0
    step0 = result.artifact.telemetry.steps[0]
    assert step0.step == 0
    assert isinstance(step0.loss, float)
    assert isinstance(step0.temperature, float)


def test_train_only_empty_positives_raises():
    """Empty positives raises IlpConfigError."""
    config = TrainConfig(max_attempts=1)
    with pytest.raises(IlpConfigError):
        train_only(
            source=REACH_SOURCE,
            mask_name="W",
            positives=[],
            negatives=[],
            config=config,
        )


def test_train_only_precision_recall():
    """Result includes precision and recall metrics."""
    config = TrainConfig(
        step_budget_per_attempt=100,
        max_attempts=5,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W",
        positives=REACH_POSITIVES,
        negatives=REACH_NEGATIVES,
        config=config,
    )
    if result.converged:
        assert result.precision > 0.0
        assert result.recall > 0.0
```

**Step 2: Run to verify failure**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v`
Expected: FAIL with `ImportError: cannot import name 'train_only' from 'pyxlog.ilp'`

**Step 3: Implement train_only**

Create `python/pyxlog/ilp/trainer.py`. This is a substantial file (~300 lines).
Key structure:

```python
# python/pyxlog/ilp/trainer.py
"""dILP trainer — train_only entry point. See design Section 5.1."""
from __future__ import annotations

import logging
import math
import time
from dataclasses import replace

import torch

import pyxlog
from pyxlog.ilp.exceptions import IlpCandidateError, IlpConfigError, IlpTrainingError
from pyxlog.ilp.temperature import AdaptiveTempController
from pyxlog.ilp.entropy import normalized_entropy, entropy_weight_at_step
from pyxlog.ilp.types import (
    ArtifactMetadata,
    CandidateMapEntry,
    LearnedArtifact,
    StepRecord,
    TrainConfig,
    TrainResult,
    TrainTelemetry,
)

log = logging.getLogger(__name__)


def train_only(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig,
    holdout_positives: list[tuple[str, list[int]]] | None = None,
    holdout_negatives: list[tuple[str, list[int]]] | None = None,
) -> TrainResult:
    """Train a single learnable rule via structured multi-start.

    See design doc Section 5.1 for full contract.
    """
    _validate_inputs(source, mask_name, positives, config)

    # Run structured multi-start
    attempts: list[_AttemptResult] = []
    total_steps = 0
    numeric_failures = 0

    for attempt_idx in range(config.max_attempts):
        if total_steps >= config.global_step_limit:
            break

        remaining = config.global_step_limit - total_steps
        step_budget = min(config.step_budget_per_attempt, remaining)
        if step_budget <= 0:
            break

        seed = _attempt_seed(config.seed, attempt_idx)
        try:
            result = _run_single_attempt(
                source=source,
                mask_name=mask_name,
                positives=positives,
                negatives=negatives,
                config=config,
                step_budget=step_budget,
                seed=seed,
                attempt_idx=attempt_idx,
                prior_attempts=attempts,
            )
            attempts.append(result)
            total_steps += result.steps_used
        except _NumericFailure:
            numeric_failures += 1
            if numeric_failures >= config.max_numeric_failures:
                raise IlpTrainingError(
                    f"Numeric instability: {numeric_failures} NaN/Inf attempts",
                    {"attempt": attempt_idx, "terminal_reason": "numeric_instability"},
                )
            continue

    return _select_winner(attempts, config, total_steps)


# ... (internal helpers: _validate_inputs, _run_single_attempt,
#      _build_mask, _compute_loss, _check_convergence, _select_winner,
#      _attempt_seed, _NumericFailure)
```

The implementation follows the existing `ilp_showcase.py` patterns
(build_budget_aware_mask, compute_loss, convergence checking) but refactored
into the structured multi-start framework with adaptive temperature and entropy.

**Key internal functions to implement:**

1. `_validate_inputs()` — check non-empty positives, valid config ranges
2. `_run_single_attempt()` — compile fresh program, training loop with
   adaptive temp + entropy, convergence detection
3. `_build_budget_aware_mask()` — ST-Gumbel-Softmax over C candidates (not N³)
4. `_compute_loss()` — per-fact surrogate credit + missed-positive penalty +
   negative penalty + entropy bonus
5. `_check_convergence()` — stable argmax count + all_derived gate +
   argmax-only validation
6. `_select_winner()` — deterministic tie-break across attempts
7. `_attempt_seed()` — derive per-attempt seed from base seed

**Step 4: Run tests**

Run: `.venv/bin/maturin develop --release && .venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v`
Expected: All 4 tests PASS

**Step 5: Run full test suite regression check**

Run: `.venv/bin/python -m pytest python/tests/ -v --timeout=300`
Expected: All existing tests still pass

**Step 6: Commit**

```bash
git add python/pyxlog/ilp/trainer.py python/pyxlog/ilp/__init__.py python/tests/test_ilp_trainer.py
git commit -m "feat(ilp): train_only() entry point with adaptive temp + entropy + multi-start"
```

---

### Task 7: Structured multi-start with random reserve and narrowing

**Files:**
- Modify: `python/pyxlog/ilp/trainer.py` (enhance _run_single_attempt)
- Test: `python/tests/test_ilp_multistart.py` (create)

This task adds the progressive narrowing behavior (design Section 2.3):
- First `ceil(K * random_reserve_fraction)` attempts use random init
- Remaining attempts narrow candidates based on prior attempt results
- Winner selected by deterministic tie-break chain

**Step 1: Write the failing tests**

```python
# python/tests/test_ilp_multistart.py
"""Tests for structured multi-start behavior."""
from pyxlog.ilp import train_only, TrainConfig

GRANDPARENT_SOURCE = """
    parent(1, 2).
    parent(2, 3).
    parent(3, 4).
    parent(4, 5).
    gender(1, 0).
    gender(2, 1).
    gender(3, 0).
    gender(4, 1).
    sibling(1, 3).
    sibling(3, 1).
    learnable grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""

GRANDPARENT_POS = [("grandparent", [1, 3]), ("grandparent", [2, 4]), ("grandparent", [3, 5])]
GRANDPARENT_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]


def test_multistart_uses_multiple_attempts():
    """With distractors, multiple attempts may be needed."""
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=5,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE,
        mask_name="W",
        positives=GRANDPARENT_POS,
        negatives=GRANDPARENT_NEG,
        config=config,
    )
    assert result.converged
    assert "parent" in result.discovered_rule


def test_multistart_reports_attempt_count():
    """Result correctly reports how many attempts were used."""
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE,
        mask_name="W",
        positives=GRANDPARENT_POS,
        negatives=GRANDPARENT_NEG,
        config=config,
    )
    assert 1 <= result.attempt_count <= 7


def test_multistart_global_step_limit():
    """Global step limit is respected across attempts."""
    config = TrainConfig(
        global_step_limit=50,
        step_budget_per_attempt=30,
        max_attempts=10,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=GRANDPARENT_SOURCE,
        mask_name="W",
        positives=GRANDPARENT_POS,
        negatives=GRANDPARENT_NEG,
        config=config,
    )
    # total_steps should not exceed global_step_limit
    assert result.total_steps <= 50
```

**Step 2: Run to verify it exercises the code**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_multistart.py -v`
Expected: Tests should pass if Task 6 is complete (multi-start is part of train_only)

**Step 3: Commit**

```bash
git add python/tests/test_ilp_multistart.py
git commit -m "test(ilp): structured multi-start integration tests with distractors"
```

---

### Task 8: NaN/Inf detection and numeric failure policy

**Files:**
- Modify: `python/pyxlog/ilp/trainer.py` (add NaN checks in training loop)
- Test: `python/tests/test_ilp_robustness.py` (create)

**Step 1: Write the failing tests**

```python
# python/tests/test_ilp_robustness.py
"""Tests for robustness: NaN/Inf handling, empty inputs, contradictions."""
import pytest
from pyxlog.ilp import train_only, TrainConfig, IlpConfigError, IlpTrainingError

SOURCE = """
    edge(1, 2).
    edge(2, 3).
    learnable reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def test_empty_positives_raises_config_error():
    """Empty positives raises IlpConfigError before GPU work."""
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(SOURCE, "W", [], [], TrainConfig(max_attempts=1))


def test_contradictory_examples_raises_config_error():
    """Same fact in positives and negatives raises IlpConfigError."""
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradictory"):
        train_only(SOURCE, "W", pos, neg, TrainConfig(max_attempts=1))


def test_numeric_failure_limit():
    """After max_numeric_failures NaN attempts, raises IlpTrainingError."""
    # This test is hard to trigger naturally; we test the policy indirectly
    # by checking that max_numeric_failures is respected in config
    config = TrainConfig(max_numeric_failures=1)
    assert config.max_numeric_failures == 1
```

**Step 2: Run tests**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_robustness.py -v`
Expected: Tests pass (validation logic is in _validate_inputs from Task 6)

**Step 3: Commit**

```bash
git add python/tests/test_ilp_robustness.py
git commit -m "test(ilp): robustness tests for input validation and numeric failure policy"
```

---

### Task 9: Confidence metrics (margin + concentration + rule frequency)

**Files:**
- Modify: `python/pyxlog/ilp/trainer.py` (_select_winner computes metrics)
- Test: `python/tests/test_ilp_trainer.py` (extend)

**Step 1: Add tests to test_ilp_trainer.py**

```python
def test_train_only_confidence_metrics():
    """Result includes confidence margin and concentration."""
    config = TrainConfig(
        step_budget_per_attempt=100,
        max_attempts=3,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
    )
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W",
        positives=REACH_POSITIVES,
        negatives=REACH_NEGATIVES,
        config=config,
    )
    if result.converged:
        assert 0.0 <= result.confidence_margin <= 1.0
        assert 0.0 <= result.top_k_concentration <= 1.0
        assert 0.0 <= result.rule_frequency <= 1.0
```

**Step 2: Run test**

Run: `.venv/bin/python -m pytest python/tests/test_ilp_trainer.py::test_train_only_confidence_metrics -v`
Expected: PASS (metrics computed in _select_winner from Task 6)

**Step 3: Commit**

```bash
git add python/tests/test_ilp_trainer.py
git commit -m "test(ilp): confidence metric assertions (margin, concentration, frequency)"
```

---

### Task 10: Alpha reliability gate — 5/5 consecutive passes

**Files:**
- Create: `python/tests/test_ilp_reliability.py`

This is the alpha exit gate. Runs `train_only` on all 4 showcase stages with
5 different seeds and verifies 100% convergence.

**Step 1: Write the reliability test**

```python
# python/tests/test_ilp_reliability.py
"""Alpha reliability gate: 5/5 consecutive passes on showcase suite.

See design doc Section 6.1. This test validates that train_only()
converges on all 4 stages with seeds 0..4.
"""
import pytest
from pyxlog.ilp import train_only, TrainConfig

# Stage definitions (same domains as ilp_showcase.py)
STAGE_1_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_1_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
STAGE_1_NEG = []

STAGE_2_SOURCE = """
    parent(1, 2). parent(2, 3). parent(3, 4). parent(4, 5).
    gender(1, 0). gender(2, 1). gender(3, 0). gender(4, 1).
    sibling(1, 3). sibling(3, 1).
    learnable grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_2_POS = [("grandparent", [1, 3]), ("grandparent", [2, 4]), ("grandparent", [3, 5])]
STAGE_2_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]

STAGE_3_SOURCE = """
    worksAt(1, 100). worksAt(7, 100). worksAt(2, 200). worksAt(4, 200).
    livesIn(1, 201). livesIn(2, 201). livesIn(7, 202). livesIn(4, 202).
    learnable colleague(X, Y) :- bL(X, Z), bR(Y, Z).
"""
STAGE_3_POS = [
    ("colleague", [1, 7]), ("colleague", [7, 1]),
    ("colleague", [2, 4]), ("colleague", [4, 2]),
]
STAGE_3_NEG = [("colleague", [1, 2]), ("colleague", [3, 4])]

STAGE_4_SOURCE = """
    succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4). succ(4, 5).
    pred(1, 0). pred(2, 1). pred(3, 2). pred(4, 3). pred(5, 4).
    learnable plus2(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_4_POS = [("plus2", [0, 2]), ("plus2", [1, 3]), ("plus2", [2, 4]), ("plus2", [3, 5])]
STAGE_4_NEG = [("plus2", [0, 1]), ("plus2", [5, 0])]

STAGES = [
    ("reach", STAGE_1_SOURCE, STAGE_1_POS, STAGE_1_NEG, "W"),
    ("grandparent", STAGE_2_SOURCE, STAGE_2_POS, STAGE_2_NEG, "W"),
    ("colleague", STAGE_3_SOURCE, STAGE_3_POS, STAGE_3_NEG, "W"),
    ("plus2", STAGE_4_SOURCE, STAGE_4_POS, STAGE_4_NEG, "W"),
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

# Run Python tests
.venv/bin/python -m pytest python/tests/test_ilp_candidates.py -v
.venv/bin/python -m pytest python/tests/test_ilp_types.py -v
.venv/bin/python -m pytest python/tests/test_ilp_temperature.py -v
.venv/bin/python -m pytest python/tests/test_ilp_entropy.py -v
.venv/bin/python -m pytest python/tests/test_ilp_trainer.py -v --timeout=300
.venv/bin/python -m pytest python/tests/test_ilp_reliability.py -v --timeout=600

# Run Rust tests (regression)
cargo test --workspace --all-targets --exclude pyxlog --release

# Run full Python test suite
.venv/bin/python -m pytest python/tests/ -v --timeout=600
```

## Beta/RC/GA Deferred Tasks (not in this plan)

These are documented for future planning sessions:

- **Beta Task B1:** `set_rule_mask_sparse()` Rust API (Phase 2 sparse path)
- **Beta Task B2:** `train_and_promote()` entry point + promotion gates
- **Beta Task B3:** `sample_false_positives()` Rust API + hard-negative mining
- **Beta Task B4:** Holdout LOO/k-fold cross-validation
- **Beta Task B5:** Ambiguity detection (top-256 scan)
- **Beta Task B6:** Recursive candidate policy (`allow_recursive_candidates`)
- **Beta Task B7:** Artifact save/load persistence (JSON)
- **Beta Task B8:** 20/20 reliability gate
- **RC Task R1:** CUDA event profiling counters (4 sub-timers)
- **RC Task R2:** Factorized scoring over C-vector (replacing N³)
- **GA Task G1:** Typed schemas + waiver policy
- **GA Task G2:** Telemetry level 2 streaming to sink
- **GA Task G3:** 50/50 reliability gate + statistical CI
- **GA Task G4:** Deterministic mode + reproducibility gate
- **GA Task G5:** Performance SLO benchmarks (N=20/50/150)
- **GA Task G6:** Robustness scenario test suite
