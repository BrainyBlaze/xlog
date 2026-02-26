# dILP Hardening — Design Document

> **Date:** 2026-02-26
> **Status:** Approved (pending implementation plan)
> **Approach:** Dual-Track with shared contract (Contract first, parallel Rust + Python work)
> **Scope:** All 6 hardening areas

---

## Table of Contents

1. [API Contract](#1-api-contract)
2. [Robust Optimization](#2-robust-optimization)
3. [Scaling](#3-scaling)
4. [Theoretical and Semantic Guarantees](#4-theoretical-and-semantic-guarantees)
5. [Runtime Productionization](#5-runtime-productionization)
6. [GA Exit Criteria](#6-ga-exit-criteria)

---

## 1. API Contract

### Rust Additions

**`valid_candidates(mask_name) → Vec<CandidateInfo>`**

Returns the deterministic set of valid candidates for a given learnable mask.
Each candidate has a stable integer ID that persists across recompilations
(as long as the relation set is unchanged).

```rust
struct CandidateInfo {
    id: u32,           // stable, deterministic
    i: u32,            // body-left index into rel_index
    j: u32,            // body-right index into rel_index
    k: u32,            // head index into rel_index
    left_name: String,
    right_name: String,
    head_name: String,
}
```

Deterministic tie-break for ID assignment: sort by `(k, i, j)` ascending,
assign IDs `0..C-1`. Ties on `(k, i, j)` are impossible (unique triples).

**`set_rule_mask_sparse(mask_name, candidate_ids, soft_probs, budget)`**

Accepts only valid candidate IDs + their soft probabilities. Rust performs
deterministic top-k hardening:
1. Sort candidates by `soft_prob` descending.
2. Tie-break: lower `candidate_id` wins.
3. Take top `min(budget, len(candidates))`.
4. Build sparse hard mask for executor.

Strict input validation: candidate_ids must be a subset of valid_candidates().
Unknown IDs raise ValueError. Duplicate IDs raise ValueError.

**Profiling counters** (4 sub-counters, exposed via Python bindings):
- `ilp.extract_us` — GPU kernel time (CUDA events)
- `ilp.d2h_us` — device-to-host transfer time (CUDA events)
- `ilp.sort_us` — host-side top-k sorting (host Instant)
- `ilp.join_us` — sum of hash_join_v2 calls (CUDA events)

### Python Types

```python
@dataclass(frozen=True)
class TrainConfig:
    # Budget
    global_step_limit: int          # hard cap across ALL attempts
    step_budget_per_attempt: int    # max steps per single attempt
    max_attempts: int               # structured multi-start attempts

    # Adaptive temperature (Section 2.1)
    tau_start: float
    tau_floor: float                # minimum tau (never anneals below this)
    plateau_window: int             # EMA loss window for plateau detection
    plateau_threshold: float        # EMA loss change threshold
    warmup_increment: float         # tau bump on trap detection
    trap_disc_threshold: float      # disc above which trap check activates
    trap_progress_window: int       # steps to check witness coverage progress

    # Entropy regularization (Section 2.2)
    entropy_weight_start: float
    entropy_weight_end: float       # linearly decayed

    # Multi-start (Section 2.3)
    random_reserve_fraction: float  # fraction of attempts kept fully random (default 0.2)
    narrowing_top_fraction: float   # fraction of candidates kept in narrowed attempts

    # Hard-negative mining (Section 2.4)
    max_mined_negatives: int        # cap for sample_false_positives()

    # Scaling (Section 3)
    max_active_rules: int           # default: min(C, 32), range 16-128
    debug_dense_mask: bool          # False = sparse API, True = dense fallback

    # Guarantees (Section 4)
    check_ambiguity: bool           # default False, recommended True for promotion
    exhaustive_ambiguity: bool      # default False, full C scan vs top-256
    max_novel_rate: float           # default 0.0, threshold for novel_fact_audit
    protected_relations: list[str]  # relations checked by regression gate
    holdout_strategy: str           # "loo" (≤20 examples), "kfold" (>20), "fixed"
    waiver_untyped: bool            # True = allow promotion without typed schemas

    # Reproducibility
    seed: int | None                # None = random per attempt
    deterministic: bool             # enables torch deterministic algorithms

    # Observability (Section 5)
    telemetry_level: int            # 0=summary, 1=per-step, 2=debug trace
    max_telemetry_steps: int        # default 1000, hard cap 10000
    telemetry_sink: Path | None     # file path for level-2 streaming

    # Numeric stability (Section 6)
    max_numeric_failures: int       # default 3, NaN/Inf attempts before terminal error

    # Metadata
    schema_version: str             # compatibility hash, auto-computed


class PromotionStatus(Enum):
    PROMOTED = "promoted"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    COMMIT_FAILED = "commit_failed"
    NOT_CONVERGED = "not_converged"
    GATE_FAILED = "gate_failed"


@dataclass
class ArtifactMetadata:
    pyxlog_version: str
    git_sha: str | None
    cuda_version: str
    device_name: str
    candidate_set_hash: str     # SHA-256 of sorted candidate_ids
    config_hash: str            # SHA-256 of TrainConfig JSON
    timestamp_utc: str          # ISO 8601


@dataclass
class LearnedArtifact:
    candidate_ids: list[int]
    logits: list[float]
    soft_probs: list[float]
    selected_hard: list[int]
    discovered_rule: str
    config_snapshot: TrainConfig
    telemetry: TrainTelemetry
    metadata: ArtifactMetadata

    def save(self, path: Path) -> None: ...

    @classmethod
    def load(cls, path: Path) -> LearnedArtifact: ...


@dataclass
class StepRecord:
    step: int
    loss: float
    argmax_rule: str
    discreteness: float
    temperature: float
    entropy: float
    stable_count: int
    forward_p95_us: float
    active_candidates: int


@dataclass
class TrainTelemetry:
    steps: list[StepRecord]       # bounded by max_telemetry_steps
    step_timings: dict | None     # CUDA event timings (level ≥ 1)


@dataclass
class TrainResult:
    converged: bool
    discovered_rule: str | None
    attempt_count: int
    total_steps: int
    precision: float
    recall: float
    holdout_f1: float | None
    confidence_margin: float
    top_k_concentration: float
    rule_frequency: float           # fraction of attempts finding this rule
    holdout_variance: float         # variance of holdout F1 across winning attempts
    single_attempt: bool            # True if only 1 attempt ran
    ambiguous_alternatives: list[str] | None
    artifact: LearnedArtifact


@dataclass
class GateResult:
    name: str
    passed: bool
    detail: str


@dataclass
class PromotionResult:
    status: PromotionStatus
    gates: list[GateResult]
    novel_count: int | None
    novel_rate: float | None
    novel_examples: list[str] | None  # up to 10 sampled
    artifact: LearnedArtifact
```

### Evaluation Set Definitions

- **Precision:** fraction of derived head-relation facts that are in the positive set.
- **Recall:** fraction of positive-set facts that are derived.
- **Holdout F1:** harmonic mean of precision and recall on held-out examples only.

### Schema Version

`schema_version` is a SHA-256 hash of the sorted relation names + arities.
Used to reject incompatible artifacts at load time.

---

## 2. Robust Optimization

### 2.1 Adaptive Temperature Controller

Three-mode controller replacing linear annealing:

| Mode | Condition | Action |
|------|-----------|--------|
| COOLING | EMA loss decreasing, disc < threshold | Decrease tau by linear schedule |
| PLATEAU | EMA loss flat (Δ < plateau_threshold over window) | Hold tau constant |
| WARMUP | Trap detected | Increase tau by warmup_increment |

**Trap detection:** `disc > trap_disc_threshold` AND no progress in witness
coverage OR holdout score over `trap_progress_window` steps. Uses EMA loss
(not raw loss) for plateau detection to avoid reacting to single-step noise.

**Temperature floor:** `tau` never drops below `tau_floor`. This prevents
the mask from reaching disc=1.0 on the wrong cell, preserving gradient flow
for exploration.

### 2.2 Entropy Regularization

Normalized Shannon entropy bonus over valid candidate probabilities:

```
H_norm = -sum(p_i * log(p_i)) / log(C)
loss_total = loss_task - entropy_weight * H_norm
```

- Normalized by `log(C)` for consistency across candidate-set sizes.
- `entropy_weight` decays linearly from `entropy_weight_start` to
  `entropy_weight_end` over the step budget.
- Numerically stable: use `log_softmax` to avoid log(0).

### 2.3 Structured Multi-Start

Best-of-K with progressive narrowing:

1. Run K attempts total (K = `max_attempts`).
2. First `ceil(K * random_reserve_fraction)` attempts use fully random init.
3. Remaining attempts narrow the candidate set: keep only the top
   `narrowing_top_fraction` of candidates (by best soft_prob from prior
   attempts), zero-init the rest.
4. Select winner by deterministic tie-break chain:
   holdout F1 → witness coverage → fewer false positives → fewer steps → lower candidate_id.

The 20% random reserve prevents pruning the true rule early when early
attempts converge to a local minimum that looks globally best.

### 2.4 Hard-Negative Mining

Bounded `sample_false_positives(head_rel, exclude, max_n) → list[tuple]`
Rust API:

- Returns up to `max_n` facts derived for `head_rel` that are NOT in `exclude`.
- Used to dynamically generate negative examples during training.
- Caller passes the positive set as `exclude`.
- Bounded output prevents memory explosion from large derived sets.

---

## 3. Scaling

### Current Bottleneck

Dense N³ mask storage + O(N³) extraction kernel + fixed max_active_rules=32.

| N | Dense mask memory | Sparse memory (C candidates, k active) |
|---|------------------|---------------------------------------|
| 6 | 3.4 KB | ~1 KB |
| 50 | 2 MB | ~10 KB (C≈2400, k=32) |
| 100 | 8 MB | ~40 KB (C≈10000, k=128) |
| 150 | 27 MB | ~90 KB (C≈22500, k=128) |

### 3.1 Sparse Candidate Pipeline

**Phase 1 (alpha):** `valid_candidates()` prunes impossible candidates. Python
trainer materializes scores only for valid candidates. Dense mask still used
for GPU transfer (backward compatible).

**Phase 2 (beta/RC):** `set_rule_mask_sparse()` replaces dense mask. Python
computes `logits → gumbel_softmax → soft_probs[C]` (gradient path stays in
PyTorch), sends sparse data to Rust. Rust does deterministic top-k + executor
mapping. No Rust-side Gumbel or softmax.

**Candidate pruning rules** (per mask_name / head relation):

A candidate `(i, j, k)` is valid iff:
- `k` is the head relation for this mask_name.
- At least one of `(i, j)` is a base relation with nonzero tuples at compile time.
- Template placeholders (bL/bR) are allowed only if the other body atom is a
  base relation that can populate the join. Template+template is always pruned.
- Recursive candidates where `i == k` or `j == k` are pruned unless the head
  already has base facts.

### 3.2 Factorized Scoring

Trainer maintains a flat `logits` tensor of shape `(C,)`. Temperature scaling,
Gumbel noise, and softmax computed over the C-vector directly:

```python
# Instead of: M_soft = gumbel_softmax(W.view(-1), tau)  # N³ elements
# Do:         M_soft = gumbel_softmax(logits, tau)        # C elements
```

Reduces forward/backward pass from O(N³) to O(C).

### 3.3 Memory Formulas

Let C = |valid_candidates|, k = min(budget, C):

| Buffer | Dense (Phase 1) | Sparse (Phase 2) |
|--------|-----------------|-------------------|
| Logits (Python) | N³ × 4B | C × 4B |
| Soft probs (Python) | N³ × 4B | C × 4B |
| Hard mask (GPU) | N³ × 4B | k × 12B (i,j,k indices) |
| Extraction temp | 5 × N³ × 4B | 0 (no scan needed) |
| **Total** | **9 × N³ × 4B** | **2C × 4B + k × 12B** |

Coverage: k/C (candidates explored per forward pass).

### 3.4 Profiling Infrastructure

Four sub-counters using CUDA event timing for GPU operations, host Instant
for CPU-side sort. Exposed in `TrainTelemetry.step_timings`.

**Soft SLO:** Rolling p95 over last 20 steps. When `ilp.join_us + ilp.extract_us`
p95 exceeds 500ms, telemetry emits a warning. Informational, not gating.

### 3.5 Max-Active-Rules Scaling

Configurable via `TrainConfig.max_active_rules`:
- Default: `min(C, 32)` (adaptive from candidate count)
- Range: 16–128 (hard bounds enforced by Rust API)
- Higher values let the optimizer "see" more candidates per step, reducing
  multi-start restarts.

### 3.6 Scaling Milestones

| Milestone | N target | Mechanism | Memory budget |
|-----------|---------|-----------|---------------|
| Alpha | N ≤ 20 | Dense mask, current kernel | < 64 KB |
| Beta | N ≤ 50 | Sparse candidate API (Phase 1), configurable max_active | < 10 MB |
| RC | N ≤ 100 | Sparse mask API (Phase 2), factorized scoring | < 50 MB |
| GA | N ≤ 150 | Phase 2 + profiling-guided budget tuning | < 100 MB |

---

## 4. Theoretical and Semantic Guarantees

### 4.1 Rule Fragment Typing

Optional type annotations on relations, checked at candidate pruning time:
- Relations declare column domains: `parent: (Person, Person)`, `gender: (Person, GenderCode)`
- `valid_candidates()` prunes `(i, j, k)` where join-column domains are incompatible.

**Milestone policy:**
- Alpha/beta: opt-in. Prune candidates when present, ignore when absent.
- GA promotion: typed schemas required. If absent, promotion blocked unless
  `TrainConfig.waiver_untyped = True`, which forces `manual_review_required`
  (no auto-promotion without types).

### 4.2 Ambiguity Detection

Post-convergence check for alternative rules that also satisfy all examples.

**Two-tier scan:**
- **Default (promotion path, on-by-default):** Scan top-M candidates by final
  soft probability (M = min(256, C)). Report any that independently derive all
  positives and no negatives.
- **Exhaustive (release validation):** Scan all C candidates. Gated behind
  `TrainConfig.exhaustive_ambiguity = False`.

Results reported in `TrainResult.ambiguous_alternatives`.

### 4.3 Promotion Soundness Checks

**Learned fragment definition:** positive, non-negated, non-aggregated, no cuts.
Within this fragment, monotonicity holds by construction (adding definite clauses
can only derive more facts).

**Promotion gates:**

| Gate | What it checks | Required |
|------|---------------|----------|
| `training_positive` | All training positives derived | Always |
| `training_negative` | No training negatives derived | Always |
| `holdout_positive` | Holdout positives ≥95% derived | If holdout provided |
| `holdout_negative` | No holdout negatives derived | If holdout provided |
| `novel_fact_audit` | `novel_rate ≤ max_novel_rate` | Always |
| `regression_check` | All `protected_relations` facts preserved | Always |

**Novel-fact audit details:**
- Returns `novel_count`, `novel_rate` (novel / total derived), `novel_examples`
  (up to 10 sampled facts for inspection).
- `max_novel_rate = 0.0` by default (zero tolerance).
- For `train_and_promote`: novel_rate > threshold → `manual_review_required`.

**Holdout requirement for auto-promotion:**
- `train_and_promote` with empty holdout → `MANUAL_REVIEW_REQUIRED`.
- `train_only` with empty holdout → holdout gates pass vacuously.

### 4.4 Calibration and Confidence

**Margin-based:**
- `confidence_margin`: top-1 minus top-2 soft probability (normalized over valid candidates). Range [0, 1].
- `top_k_concentration`: fraction of total soft mass in top-k selected candidates.

**Stability-based (multi-attempt):**
- `rule_frequency`: fraction of attempts converging to the same rule.
- `holdout_variance`: variance of holdout F1 across attempts that found the winning rule.
- `single_attempt = True` flag when only 1 attempt ran (metrics not meaningful).

Informational — not gating.

---

## 5. Runtime Productionization

### 5.1 Trainer Entry Points

**`train_only(program, config) → TrainResult`**
- Runs structured multi-start. Returns best learned rule + telemetry.
- Does NOT mutate the caller's program. Each attempt compiles a fresh
  `CompiledIlpProgram` internally (clean store + ILP registry per attempt).
- Pure function over program state — safe to call repeatedly.

**`train_and_promote(program, config) → PromotionResult`**
- Calls `train_only` internally.
- Compiles a *trial* program with the discovered rule committed.
- Runs all promotion gates against the trial.
- If all gates pass: recompiles the caller's program with the committed rule
  (atomic compile+swap). Returns `PROMOTED`.
- If any gate fails or holdout empty: caller's program unchanged. Returns
  `MANUAL_REVIEW_REQUIRED` with per-gate detail + artifact.
- If commit fails recompilation: returns `COMMIT_FAILED`.

**Transactional guarantee:** The caller's program is never modified until all
gates pass on the trial compilation. Failed commits leave the program
byte-for-byte unchanged.

### 5.2 Artifact Persistence

`LearnedArtifact` is CPU-serializable:
- `save(path)` / `load(path)` using JSON (alpha) or MessagePack (GA).
- Includes `schema_version` hash for compatibility checking.
- Includes `ArtifactMetadata` with pyxlog_version, git_sha, cuda_version,
  device_name, candidate_set_hash, config_hash, timestamp_utc.
- No GPU tensors, no program objects, no CUDA state.

### 5.3 Observability

Three telemetry levels:

| Level | Content | Storage |
|-------|---------|---------|
| 0 | TrainResult summary only | In-memory |
| 1 | Per-step StepRecord list | In-memory (bounded by max_telemetry_steps) |
| 2 | Per-candidate per-step traces | Streamed to telemetry_sink file (NDJSON) |

Level 2 with no `telemetry_sink` set → downgrade to level 1 with warning.

**Logging:** Structured log records via Python `logging` module at INFO (level 0),
DEBUG (level 1+). No print statements.

### 5.4 Exception Taxonomy

```python
class IlpConfigError(ValueError):
    """Invalid TrainConfig or empty examples. Raised before GPU work."""

class IlpCandidateError(ValueError):
    """No valid candidates for the given mask. Cannot train."""

class IlpTrainingError(RuntimeError):
    """CUDA or numerical failure during training."""
    context: dict  # attempt, step, C, k, device_name, allocated_bytes, terminal_reason

class IlpCommitError(RuntimeError):
    """Rule commit / recompilation failed."""
```

CUDA errors are caught, enriched with training context (attempt index, step,
C, k, device name, allocated bytes), wrapped in `IlpTrainingError`, re-raised.
Caller decides retry policy based on `error.context`.

### 5.5 NaN/Inf Policy

- Per-attempt: NaN/Inf in logits or loss → abandon attempt immediately.
- Separate `numeric_failure_count` counter (not counted against convergence budget).
- After `max_numeric_failures` (default 3) NaN/Inf attempts, raise
  `IlpTrainingError` with `terminal_reason = "numeric_instability"`.

---

## 6. GA Exit Criteria

### 6.1 Reliability Gate

Statistical criterion using Clopper-Pearson 95% CI lower bound on success rate:

| Milestone | Runs | Required | 95% CI lower bound |
|-----------|------|----------|-------------------|
| Alpha | 5 | 5/5 | ≥ 0.478 |
| Beta | 20 | 20/20 | ≥ 0.831 |
| GA | 50 | 50/50 | ≥ 0.929 |

If any GA run fails: extend to 100 runs, require ≥98/100 (CI ≥ 0.932).
Tested on the showcase suite (4 stages) with seeds `0..N-1`.

### 6.2 Correctness Gate

| Check | Condition |
|-------|-----------|
| Training positives | 100% derived |
| Training negatives | 0% derived |
| Holdout (LOO/k-fold) | ≥95% accuracy |
| Novel-fact audit | novel_rate ≤ max_novel_rate |
| Regression check | Protected-relation facts preserved |
| Ambiguity scan | Top-256 default; exhaustive for GA release |

**Holdout strategy:** LOO for ≤20 examples, 5-fold stratified CV for >20.

### 6.3 Performance Gate

**Forward-pass SLO:**

| N range | Forward-pass p95 | Memory ceiling |
|---------|-----------------|----------------|
| N ≤ 20 | ≤ 50ms | ≤ 10 MB |
| N ≤ 50 | ≤ 200ms | ≤ 50 MB |
| N ≤ 150 | ≤ 500ms | ≤ 100 MB |

**End-to-end training SLO (informational for beta, hard gate for GA):**

| N range | Time-to-converge p95 | Step budget p95 |
|---------|---------------------|-----------------|
| N ≤ 20 | ≤ 30s | ≤ 200 steps total |
| N ≤ 50 | ≤ 2min | ≤ 500 steps total |
| N ≤ 150 | ≤ 10min | ≤ 2000 steps total |

"Steps total" = sum across all attempts including failed ones.
Measured on showcase benchmark suite with synthetic N-scaling benchmarks.

### 6.4 Robustness Gate

| Scenario | Expected behavior |
|----------|------------------|
| All distractors, no useful base relations | `converged = False`, no crash |
| Empty positive examples | `IlpConfigError` before training |
| Contradictory examples (pos ∩ neg ≠ ∅) | `IlpConfigError` before training |
| N=1 (degenerate) | `IlpCandidateError` |
| CUDA OOM at large N | `IlpTrainingError` with enriched context |
| NaN/Inf in logits | Attempt abandoned, terminal after max_numeric_failures |

Dedicated robustness test suite with one test per scenario.

### 6.5 Reproducibility Gate

**Deterministic mode** (`TrainConfig.deterministic = True`):
- `torch.use_deterministic_algorithms(True)`
- Disables cuDNN benchmarking
- CUDA deterministic reductions

**GA requirement:** Same seed + deterministic mode produces:
- Same `discovered_rule` string
- Same `selected_hard` candidate IDs
- Soft probabilities within `rtol=1e-5, atol=1e-6`

Scoped to same GPU model + CUDA version + pyxlog version.
Cross-hardware determinism is explicitly out of scope.

### 6.6 Gate Summary

| Gate | Alpha | Beta | GA |
|------|-------|------|----|
| Reliability | 5/5 | 20/20 (CI≥0.831) | 50/50 (CI≥0.929) |
| Correctness | Training only | LOO/k-fold ≥95% | ≥95% + ambiguity scan |
| Forward-pass perf | No SLO | N≤50 ≤200ms | N≤150 ≤500ms |
| End-to-end perf | No SLO | Informational | Hard gate |
| Robustness | Crash-free | Exception taxonomy | Full scenario suite |
| Reproducibility | Not required | Same rule (det mode) | Rule + candidates + tolerance |
| Typed schemas | Optional | Optional | Required (or waiver) |
