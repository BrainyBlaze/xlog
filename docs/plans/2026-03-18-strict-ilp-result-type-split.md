# Strict ILP Result Type Split Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace the current mixed strict/compat ILP result contract with a clean split: `train_only(..., strict_gpu_native=True)` returns a dedicated strict result/artifact pair with no implicit host materialization, while compatibility mode continues to return the existing host-shaped `TrainResult` / `LearnedArtifact`.

**Architecture:** Keep the compatibility contract intact for `strict_gpu_native=False`. Introduce `StrictTrainResult` and `StrictLearnedArtifact` for strict mode and make them intentionally non-host-shaped: they carry only host-known metadata plus private device-resident export state for one explicit one-way export into compatibility objects. Remove the current strict finalization DTOH helpers entirely. Strict winner selection must use device-native tensors only and retain just one winning device payload instead of cloning every attempt.

**Tech Stack:** Python dataclasses (`pyxlog.ilp.types`), PyTorch CUDA tensors / DLPack, Rust PyO3 bindings (`crates/pyxlog/src/ilp.rs`), `pytest`, `maturin`.

## Requirements Reconfirmed From Spec + Review

1. `train_only(..., strict_gpu_native=True)` must return a strict type, not `TrainResult`.
2. The strict type must not auto-populate Python `bool` / `float` / `list` summary fields from device state.
3. Human-readable rule strings, selected candidate IDs, logits, soft probabilities, and selected-hard lists may only appear through explicit `export_compat_result()` / `export_compat_artifact()`.
4. Strict finalization must not call:
   - `read_device_*`
   - `set_rule_mask_sparse_selected(...)`
   - host-shaped export helpers
5. Public ungated DTOH helpers must not remain reachable for strict callers.
6. `TrainResult` + `LearnedArtifact` remain compatibility-only and preserve current behavior for `strict_gpu_native=False`.
7. Best-attempt selection must not retain one cloned GPU payload per attempt if only one winner/export path is needed.

## Recommended Design

### Public result split

- Keep:
  - `TrainResult`
  - `LearnedArtifact`
- Add:
  - `StrictTrainResult`
  - `StrictLearnedArtifact`

`TrainResult` remains the host-facing compatibility object with:
- `converged`
- `precision`
- `recall`
- `rule_frequency`
- `discovered_rule`
- host-shaped artifact data

`StrictTrainResult` becomes the strict default object with only host-known metadata:
- `attempt_count: int`
- `total_steps: int`
- `single_attempt: bool`
- `artifact: StrictLearnedArtifact`
- `strict_gpu_native: bool = True`
- `compat_materialized: bool = False`
- `export_compat_result() -> TrainResult`

`StrictLearnedArtifact` becomes the strict artifact with only host-known metadata:
- `candidate_map`
- `config_snapshot`
- `telemetry`
- `strict_gpu_native: bool = True`
- `compat_materialized: bool = False`
- `export_compat_artifact() -> LearnedArtifact`

No strict public fields should expose host-derived convergence booleans, Python float metrics, decoded rule strings, candidate ID lists, logits, or probabilities.

### Strict winner selection

Do **not** compute convergence / precision / recall in strict finalization. That is the host-materialization trap.

Instead:
- keep the strict hot loop device-native
- capture per-attempt final device tensors only:
  - final `W`
  - final `cand_probs`
  - final device loss scalar (`credit_loss.detach()`)
- choose the winning attempt with a device-native score and retain only one winner state

Recommended score:
1. lowest final device loss
2. tie-break: higher top-1 candidate probability
3. equality keeps the earlier retained winner by using strict `<` comparisons only

This keeps strict selection deterministic without downloading summaries.

### Explicit compatibility export

All host materialization moves into the one-way explicit export path:
- `StrictTrainResult.export_compat_result()`
- `StrictLearnedArtifact.export_compat_artifact()`

That export path is allowed to:
- decode argmax
- read candidate IDs to host
- call compatibility sparse setters
- call `evaluate()`
- compute `converged`, `precision`, `recall`, `rule_frequency`
- build `discovered_rule`
- fill `logits`, `soft_probs`, `selected_hard`, `confidence_margin`, `top_k_concentration`

Strict mode itself must not do any of that.

## Task 1: Write the failing strict type-split tests first

**Files:**
- Modify: `python/tests/test_ilp_trainer.py`
- Modify: `python/tests/test_ilp_types.py`
- Modify: `python/tests/test_ilp_d2h_gate.py`

**Step 1: Add strict return-type RED tests**

Add tests like:

```python
def test_train_only_strict_gpu_native_returns_strict_result_type():
    result = train_only(... strict config ...)
    assert isinstance(result, StrictTrainResult)
    assert not isinstance(result, TrainResult)
    assert isinstance(result.artifact, StrictLearnedArtifact)
```

```python
def test_strict_result_has_no_host_summary_fields():
    result = train_only(... strict config ...)
    assert not hasattr(result, "converged")
    assert not hasattr(result, "precision")
    assert not hasattr(result, "recall")
    assert not hasattr(result, "rule_frequency")
    assert not hasattr(result, "discovered_rule")
```

```python
def test_strict_artifact_has_no_host_materialized_payload_fields():
    result = train_only(... strict config ...)
    art = result.artifact
    assert not hasattr(art, "logits")
    assert not hasattr(art, "soft_probs")
    assert not hasattr(art, "selected_hard")
    assert not hasattr(art, "discovered_rule")
```

**Step 2: Add explicit export RED tests**

Add tests like:

```python
def test_strict_result_export_compat_result_returns_train_result():
    strict = train_only(... strict config ...)
    compat = strict.export_compat_result()
    assert isinstance(compat, TrainResult)
    assert compat.converged
    assert compat.discovered_rule
```

```python
def test_strict_artifact_export_compat_artifact_returns_learned_artifact():
    strict = train_only(... strict config ...)
    compat_art = strict.artifact.export_compat_artifact()
    assert isinstance(compat_art, LearnedArtifact)
    assert compat_art.discovered_rule
```

**Step 3: Add public DTOH helper regression tests**

The bad patch introduced `read_device_i64_scalar`, `read_device_bool_scalar`, and `read_device_i64_list` as public host-download APIs. Add tests that assert they are no longer part of the public strict surface:

```python
def test_compiled_ilp_program_has_no_public_read_device_download_helpers():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    assert not hasattr(prog, "read_device_i64_scalar")
    assert not hasattr(prog, "read_device_bool_scalar")
    assert not hasattr(prog, "read_device_i64_list")
```

**Step 4: Add strict no-host-finalization behavioral tests**

Add a behavioral test that traps Python sync methods and verifies strict training still succeeds **without** accessing host summaries:

```python
def test_train_only_strict_gpu_native_passes_with_python_host_sync_traps(monkeypatch):
    ...
    result = train_only(... strict config ...)
    assert isinstance(result, StrictTrainResult)
```

**Step 5: Run the targeted RED tests**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q -k "strict or export_compat"`
- `python -m pytest python/tests/test_ilp_types.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q -k "strict or read_device"`

Expected:
- at least one new strict type-split test fails before implementation

## Task 2: Introduce `StrictTrainResult` and `StrictLearnedArtifact`

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/__init__.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`

**Step 1: Add the new public strict dataclasses**

In `types.py`, add:

```python
@dataclass
class StrictLearnedArtifact:
    candidate_map: list[CandidateMapEntry] = field(default_factory=list)
    config_snapshot: TrainConfig | None = None
    telemetry: TrainTelemetry = field(default_factory=TrainTelemetry)
    strict_gpu_native: bool = True
    compat_materialized: bool = False
    _compat_exporter: Callable[[], LearnedArtifact] | None = field(...)

    def export_compat_artifact(self) -> LearnedArtifact:
        ...


@dataclass
class StrictTrainResult:
    attempt_count: int = 0
    total_steps: int = 0
    single_attempt: bool = True
    artifact: StrictLearnedArtifact = field(default_factory=StrictLearnedArtifact)
    strict_gpu_native: bool = True
    compat_materialized: bool = False
    _compat_exporter: Callable[[], TrainResult] | None = field(...)

    def export_compat_result(self) -> TrainResult:
        ...
```

Do **not** copy compatibility summary fields into these classes.

**Step 2: Export the new types**

Update `crates/pyxlog/python/pyxlog/ilp/__init__.py` so users can import:
- `StrictTrainResult`
- `StrictLearnedArtifact`

**Step 3: Update trainer type hints**

Change:
- `train_only(...) -> TrainResult | StrictTrainResult`
- `_train_on_compiled(...) -> TrainResult | StrictTrainResult`

Also update trainer docstrings so they no longer falsely claim strict mode returns `TrainResult`.

**Step 4: Run focused tests**

Run:
- `python -m pytest python/tests/test_ilp_types.py -q`

Expected:
- the type surface tests pass or fail only on the still-unimplemented strict trainer path

## Task 3: Refactor strict attempt state so finalization is fully DTOH-free

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`

**Step 1: Add an internal strict attempt state**

Introduce a private container for strict attempts, for example:

```python
class _StrictAttemptState:
    __slots__ = (
        "candidate_map",
        "telemetry_steps",
        "telemetry_timings",
        "steps_used",
        "W_device",
        "cand_probs_device",
        "final_loss_device",
    )
```

Only store:
- host-known metadata (`candidate_map`, timings, step count)
- winner-export tensors (`W_device`, `cand_probs_device`, `final_loss_device`)

Do **not** store host summaries.

**Step 2: Remove strict finalization DTOH logic**

Delete strict-path dependence on:
- `_read_device_i64_scalar`
- `_read_device_bool_scalar`
- `_read_device_i64_list`
- `_strict_witness_coverage`
- `_check_convergence_strict`
- the current `_finalize_strict_attempt(...)` host-summary logic

Strict finalization should just package the attempt-local device state.

**Step 3: Stop re-entering compatibility sparse setters in strict finalization**

Strict finalization must not call:
- `set_rule_mask_sparse_selected(...)`
- `evaluate()`
- `decode_argmax_*`

The current post-loop path that rebuilds selected IDs and runs compatibility evaluation must be removed from strict mode entirely.

**Step 4: Capture a device-native per-attempt score**

At the last strict step, retain:
- `last_logits = W.detach().clone()`
- `last_cand_probs = masked_cand_probs.detach().clone()` or the exact final candidate distribution intended for export
- `last_loss = credit_loss.detach().clone()`

No host sync is needed here.

**Step 5: Implement device-native winner selection that retains only one attempt**

Replace “store all attempts then pick one” with “keep current best strict attempt only”.

Recommended helper:

```python
def _pick_better_strict_attempt(best, candidate):
    # compare candidate.final_loss_device vs best.final_loss_device on CUDA
    # tie-break with top1 candidate mass
    # update retained tensors with torch.where(...)
```

Important constraints:
- do not branch on CUDA tensors with `.item()`
- do not keep per-attempt cloned payloads once a new best is known
- host-known metadata that is identical across attempts (`candidate_map`) can be reused directly

**Step 6: Run focused trainer tests**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q -k "strict and not export_compat"`

Expected:
- strict finalization tests pass without any host-summary fields

## Task 4: Move all host materialization into explicit compatibility export only

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/trainer.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/types.py`
- Modify: `crates/pyxlog/python/pyxlog/ilp/backend.py`
- Modify: `crates/pyxlog/src/ilp.rs`

**Step 1: Keep compatibility export as the only host-shaped path**

`_export_compat_attempt(...)` becomes the only place allowed to:
- decode argmax
- call `.cpu()`, `.item()`, `.tolist()`, `.numpy()`
- call `set_rule_mask_sparse_selected(...)`
- call `evaluate()`
- build rule strings and host metrics

This is acceptable because it is explicit, named compatibility export.

**Step 2: Remove the public `read_device_*` Rust methods**

Delete from `crates/pyxlog/src/ilp.rs`:
- `read_device_i64_scalar(...)`
- `read_device_i64_list(...)`
- `read_device_bool_scalar(...)`

Also delete their internal wrappers if no longer used.

Do **not** replace them with another public DTOH helper under a new name.

**Step 3: If a private Rust host download wrapper remains, keep it private and compatibility-only**

Prefer not to need one at all. The explicit compatibility export path can usually materialize via PyTorch from already-retained device tensors.

If a Rust helper truly remains necessary:
- keep it private
- keep it out of the public PyO3 surface
- do not call it from strict finalization

**Step 4: Rebuild strict result construction**

Add a dedicated `_build_strict_train_result(...) -> StrictTrainResult` that:
- takes the one retained winning strict attempt state
- builds `StrictLearnedArtifact`
- wires `export_compat_result()` / `export_compat_artifact()`
- does not instantiate `TrainResult`

**Step 5: Re-run focused export tests**

Run:
- `python -m pytest python/tests/test_ilp_trainer.py -q -k "export_compat or strict"`
- `python -m pytest python/tests/test_ilp_types.py -q`

Expected:
- strict path returns strict types
- explicit export returns compatibility types
- no public `read_device_*` helpers remain

## Task 5: Migrate compatibility-style callers and tests without polluting the strict API

**Files:**
- Modify: `python/tests/test_ilp_backend.py`
- Modify: `python/tests/test_ilp_artifact.py`
- Modify: `python/tests/test_ilp_reset.py`
- Modify: `python/tests/test_ilp_robustness.py`
- Modify: `python/tests/test_ilp_multistart.py`
- Modify: `python/tests/test_ilp_performance.py`
- Modify: `python/tests/test_ilp_reliability.py` if touched
- Modify: other `train_only()` callers that still assume default strict returns `TrainResult`

**Step 1: Audit every default-configuration `train_only()` caller**

Classify each caller:

- **Compatibility-contract tests**
  - set `strict_gpu_native=False`
  - continue asserting `TrainResult` / `LearnedArtifact` fields directly

- **Strict-contract tests**
  - keep `strict_gpu_native=True` or default
  - assert `StrictTrainResult` / `StrictLearnedArtifact`
  - call explicit export only when host-shaped assertions are intended

**Step 2: Update the obvious compatibility suites**

The following files currently assume default strict returns host summaries and will need explicit migration:
- `python/tests/test_ilp_reset.py`
- `python/tests/test_ilp_multistart.py`
- `python/tests/test_ilp_robustness.py`
- `python/tests/test_ilp_performance.py`
- `python/tests/test_ilp_reliability.py` if it still relies on default strict host fields

Preferred migration:
- set `strict_gpu_native=False` where the test is clearly about legacy/compat behavior
- keep strict tests separate and explicit

**Step 3: Preserve already-correct explicit compat tests**

Keep and adapt the explicit-export tests already added in:
- `python/tests/test_ilp_backend.py`
- `python/tests/test_ilp_artifact.py`

These should continue to validate the one-way export boundary.

**Step 4: Run the touched caller/test suites**

Run:
- `python -m pytest python/tests/test_ilp_backend.py python/tests/test_ilp_artifact.py python/tests/test_ilp_reset.py python/tests/test_ilp_robustness.py python/tests/test_ilp_multistart.py -q`
- `python -m pytest python/tests/test_ilp_performance.py -q -k "not slo_scaling"`

If `python/tests/test_ilp_reliability.py` is touched, run its exact targeted command and record the result.

## Task 6: Update public docs to document the split cleanly

**Files:**
- Modify: `docs/architecture/python-bindings.md`
- Modify: `docs/architecture/dilp-training.md`

**Step 1: Document the new return contract**

Document that:
- `train_only(..., strict_gpu_native=False)` returns `TrainResult`
- `train_only(..., strict_gpu_native=True)` returns `StrictTrainResult`
- explicit `export_compat_result()` / `export_compat_artifact()` performs the host materialization boundary

**Step 2: Remove misleading strict-summary language**

Delete or revise any wording that suggests strict mode returns host summaries by default.

**Step 3: Document explicit DTOH boundaries**

State clearly:
- strict path = no implicit host materialization in finalization
- compatibility export = explicit, named host materialization boundary

**Step 4: Run doc-adjacent tests if any**

No dedicated doc tests are required; rely on the runtime/test matrix in Task 7.

## Task 7: Rebuild, run the verification matrix, and update evidence

**Files:**
- Modify: `docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md`

**Step 1: Rebuild native bindings**

Run:
- `maturin develop --release -m crates/pyxlog/Cargo.toml`

Expected:
- successful rebuild and editable install

**Step 2: Run the required matrix**

Run:
- `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
- `python -m pytest python/tests/test_ilp_device_queries.py -q -x`
- `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
- `python -m pytest python/tests/test_ilp_holdout.py -q -x`
- `python -m pytest python/tests/test_ilp_promoter.py -q -x`
- `python -m pytest python/tests/test_ilp_trainer.py -q`

**Step 3: Run the extra API migration suites**

Run:
- `python -m pytest python/tests/test_ilp_backend.py -q`
- `python -m pytest python/tests/test_ilp_artifact.py -q`
- `python -m pytest python/tests/test_ilp_reset.py python/tests/test_ilp_robustness.py python/tests/test_ilp_multistart.py -q`
- `python -m pytest python/tests/test_ilp_performance.py -q -k "not slo_scaling"`

If a Rust test is added:
- run the exact targeted `cargo test` command for that module

**Step 4: Update evidence**

Replace the incorrect “strict summary semantics fixed” claim. The updated evidence doc must say:
- the prior follow-up reopened hidden DTOH through public `read_device_*` helpers
- the strict/compat split removed that escape
- strict mode now returns dedicated strict types
- compatibility export is the only host-materializing path
- verification commands and exact fresh results

**Step 5: Commit**

```bash
git add crates/pyxlog/python/pyxlog/ilp/types.py \
        crates/pyxlog/python/pyxlog/ilp/__init__.py \
        crates/pyxlog/python/pyxlog/ilp/trainer.py \
        crates/pyxlog/python/pyxlog/ilp/backend.py \
        crates/pyxlog/src/ilp.rs \
        python/tests/test_ilp_trainer.py \
        python/tests/test_ilp_types.py \
        python/tests/test_ilp_d2h_gate.py \
        python/tests/test_ilp_backend.py \
        python/tests/test_ilp_artifact.py \
        python/tests/test_ilp_reset.py \
        python/tests/test_ilp_robustness.py \
        python/tests/test_ilp_multistart.py \
        python/tests/test_ilp_performance.py \
        docs/architecture/python-bindings.md \
        docs/architecture/dilp-training.md \
        docs/evidence/2026-03-17-strict-ilp-gpu-native-audit.md \
        docs/plans/2026-03-18-strict-ilp-result-type-split.md
git commit -m "fix(ilp): split strict and compatibility training results"
```

## Explicit Non-Goals

- Do not reintroduce any public `read_device_*` DTOH helper under a new name.
- Do not mix `strict_state` into `TrainResult`.
- Do not auto-populate `converged`, `precision`, `recall`, `rule_frequency`, or `discovered_rule` on the strict type.
- Do not make strict finalization call `set_rule_mask_sparse_selected(...)` or other compatibility-only setters.
- Do not weaken the explicit compatibility export boundary.

## Handoff Summary

If implemented correctly, the final state is:

- `strict_gpu_native=True` returns `StrictTrainResult` / `StrictLearnedArtifact`
- `strict_gpu_native=False` returns `TrainResult` / `LearnedArtifact`
- strict finalization performs zero implicit host materialization
- no public ungated DTOH helper remains for strict callers
- only one winner device payload is retained across attempts
- compatibility export remains explicit and one-way
