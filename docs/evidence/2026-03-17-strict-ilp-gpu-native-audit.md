# Strict ILP GPU-Native Audit Evidence

Updated: 2026-03-18T01:51:12Z

## Scope

This update closes the two remaining strict-path gaps called out in `docs/plans/2026-03-17-strict-gpu-native-substrate-closure.md`:

1. strict sparse selected-device mask updates
2. strict trainer/result finalization and artifact export

The final shape is:

- strict mode returns `StrictTrainResult` / `StrictLearnedArtifact`
- compatibility mode returns `TrainResult` / `LearnedArtifact`
- compatibility export is one-way and explicit only

## Review-driven fix rounds

- **Round 1:** close the original two closure-plan gaps
  - device-backed strict sparse selected-ID substrate in xlog runtime
  - strict result/artifact split with explicit compatibility export
- **Round 2:** repair regressions uncovered in review
  - restore strict result truthfulness without implicit DTOH
  - restore public malformed-ID validation
  - hard-gate public generic strict runtime escape hatches
- **Round 3:** repair follow-up regressions from the strict split
  - fix multi-attempt winner metadata/argmax export correctness
  - remove the public trusted strict setter entry point
  - hard-gate host-shaped sparse-selected APIs under `strict_zero_dtoh`

## Issue 1: strict sparse selected-device mask updates and strict-mode API escape hatches

- **Issue:** strict sparse selected-device updates needed a real device-backed substrate with no hidden DTOH in the strict selected-ID path, while still preserving the public malformed-ID validation contract and closing strict-mode escape hatches on the Python surface.
- **Root cause:** the earlier follow-up mixed three different concerns:
  - validated public selected-ID input handling
  - strict hot-loop device updates
  - compatibility-only host-shaped sparse setters
  That left an unvalidated public strict setter name reachable and left `set_rule_mask_sparse_selected(...)` callable under `strict_zero_dtoh=True`.
- **Fix:**
  - `crates/pyxlog/src/ilp.rs:1077` hard-gates `set_rule_mask_sparse_selected(...)` when `strict_zero_dtoh=True`.
  - `crates/pyxlog/src/ilp.rs:1154` keeps only the validated public device setter; the public `set_rule_mask_sparse_selected_device_trusted(...)` entry point was removed.
  - `crates/pyxlog/src/ilp.rs:1170` keeps public `evaluate()` hard-gated for strict callers when a `SparseDevice` mask is present.
  - `crates/pyxlog/src/ilp.rs:657` and `crates/pyxlog/src/ilp.rs:750` make the strict loss/grad path own the internal reevaluation it needs, without reopening a public helper.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:925` applies per-step strict sparse selected-device updates directly from CUDA tensors and never calls the host-shaped sparse setter.
  - `crates/pyxlog/python/pyxlog/ilp/backend.py:168` hard-gates `SparseMaskBackend.apply_mask(..., strict_gpu_native=True)` as an unsupported public entry point so strict callers cannot bypass `train_only(...)`.
- **Strict vs compatibility classification:**
  - public selected-device setter: strict-compatible public API, validated
  - host-shaped selected sparse setter: compatibility-only, hard-gated out of strict mode
  - generic public `SparseDevice evaluate()`: compatibility-only, hard-gated out of strict mode
  - strict hot-loop reevaluation: internal loss/grad machinery only, not a public export/helper boundary

## Issue 2: strict trainer finalization still rebuilt host-shaped results implicitly

- **Issue:** the previous follow-up either returned fake zeroed summaries or restored implicit DTOH through helper APIs. Both violated the closure spec.
- **Root cause:** the old API forced strict runs into the compatibility-shaped `TrainResult` contract. That made truthful strict results depend on implicit host reconstruction of:
  - `converged`
  - `precision`
  - `recall`
  - `rule_frequency`
  - decoded rule strings and artifact payloads
- **Fix:**
  - `crates/pyxlog/python/pyxlog/ilp/types.py:305` introduces `StrictLearnedArtifact`.
  - `crates/pyxlog/python/pyxlog/ilp/types.py:376` introduces `StrictTrainResult`.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:399` adds a dedicated strict training path returning the strict type.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:925` keeps the strict attempt loop DTOH-free.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1090` finalizes strict attempts by packaging device state only; it does not call `read_device_*`, `set_rule_mask_sparse_selected(...)`, or compatibility decode helpers.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1598` moves compatibility materialization into explicit export helpers only.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1728` computes compatibility `rule_frequency` during explicit export using only one retained winner payload plus small per-attempt selected/argmax device tensors.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1768` builds `StrictTrainResult` instead of `TrainResult` for strict runs.
  - `crates/pyxlog/src/ilp.rs` no longer exposes public `read_device_i64_scalar`, `read_device_i64_list`, or `read_device_bool_scalar` methods on `CompiledIlpProgram`.
- **Strict vs compatibility classification:**
  - strict return path: device-backed state only, no implicit host summaries
  - compatibility export: explicit one-way host materialization via `export_compat_result()` / `export_compat_artifact()`
  - public DTOH helpers for strict callers: removed

## Issue 3: strict winner-state correctness during explicit compatibility export

- **Issue:** multi-attempt strict winner selection could keep stale `argmax_candidate_id`, `attempt` identity, and attempt-local telemetry/step metadata, so explicit compatibility export could decode the wrong rule or report the wrong attempt metadata.
- **Root cause:** `_select_better_strict_attempt(...)` only swapped a subset of device fields, while explicit export later mixed the merged device state with stale attempt-local Python metadata.
- **Fix:**
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:496` now swaps `argmax_candidate_id_device` and `attempt_id_device` along with the winner tensors.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:416` records host metadata by real attempt index.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1591` materializes the winning attempt index explicitly during export.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1598` uses the winning attempt’s stored host metadata instead of stale merged-state metadata.
  - `crates/pyxlog/python/pyxlog/ilp/trainer.py:1792` resolves the winning attempt ID first, then exports the matching compatibility result/artifact.
- **Classification:** strict-result correctness fix only; no hidden host export was reintroduced into strict finalization.

## Issue 4: strict/compatibility caller fallout

- **Issue:** older tests and helper callers still assumed default `train_only()` returned host-shaped `TrainResult`.
- **Root cause:** the new strict split is intentionally incompatible with the older mixed default contract.
- **Fix:** compatibility-style suites now opt into `strict_gpu_native=False` when they assert host-shaped fields directly, while strict suites assert `StrictTrainResult` / `StrictLearnedArtifact` and call explicit export only when they intentionally need compatibility artifacts.
- **Classification:** compatibility-suite migration only; no strict-path fallback or silent routing was added.

## Behavioral DoD status

- **strict sparse selected-device mask updates perform zero host materialization**
  - satisfied on the validated strict selected-device path used by the strict trainer hot loop
- **no provider download helper is reachable from the strict sparse selected-device path**
  - satisfied
- **strict sparse masks are not represented only as host `Vec<(u32,u32,u32)>`**
  - satisfied via runtime `SparseDevice` substrate in `crates/xlog-runtime/src/ilp_registry.rs:31`
- **strict trainer success/finalization path does not call `.cpu()`, `.item()`, `.tolist()`, or `.numpy()`**
  - satisfied in the strict attempt/finalization/build path
- **strict mode does not auto-populate `discovered_rule`, `logits`, `soft_probs`, `selected_hard`, `confidence_margin`, or `top_k_concentration`**
  - satisfied via strict result/artifact type split
- **compatibility artifact export is explicit and named**
  - satisfied via `export_compat_result()` / `export_compat_artifact()`
- **strict mode does not silently route through compatibility code**
  - satisfied for the strict result/export boundary and strict Python API surface; compatibility-only host-shaped sparse setters remain hard-gated

## Verification

- `maturin develop --release -m crates/pyxlog/Cargo.toml`
  - Result: succeeded, editable `pyxlog-0.4.0` installed
- `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q`
  - Result: `33 passed in 1.83s`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
  - Result: `19 passed in 23.27s`
- `python -m pytest python/tests/test_ilp_device_queries.py -q -x`
  - Result: `2 passed in 0.22s`
- `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
  - Result: `17 passed in 1.43s`
- `python -m pytest python/tests/test_ilp_holdout.py -q -x`
  - Result: `8 passed in 133.71s (0:02:13)`
- `python -m pytest python/tests/test_ilp_promoter.py -q -x`
  - Result: `10 passed in 54.28s`
- `python -m pytest python/tests/test_ilp_trainer.py -q`
  - Result: `28 passed in 97.42s (0:01:37)`
- `python -m pytest python/tests/test_ilp_backend.py python/tests/test_ilp_artifact.py python/tests/test_ilp_reset.py python/tests/test_ilp_multistart.py python/tests/test_ilp_robustness.py -q`
  - Result: `29 passed in 211.11s (0:03:31)`
- Rust runtime tests
  - Result: not applicable for this change set; no new Rust runtime unit tests were added

## Residual risks

- No remaining hidden host-materialization escape was identified in the audited strict Python ILP training/finalization/export path.
- Generic `SparseDevice` runtime execution is still not a genuinely sparse executor. The underlying sparsity/performance limitation remains in:
  - `crates/xlog-runtime/src/executor/node_dispatch.rs:545`
  - `crates/xlog-runtime/src/ilp_registry.rs:31`
  - public generic `evaluate()` remains hard-gated at `crates/pyxlog/src/ilp.rs:1170`
  - strict loss/grad reevaluation currently drives that executor internally from `crates/pyxlog/src/ilp.rs:750`
- This is no longer a host-materialization blocker, but it remains the exact residual non-DTOH limitation in the current strict trainer runtime substrate.
