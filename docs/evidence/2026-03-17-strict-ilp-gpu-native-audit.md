# Strict ILP GPU-Native Audit Evidence

## Issue 1: strict trainer hot loop still executed hidden Python-side host syncs

- **Issue:** strict `train_only()` still used Python CUDA scalar/materialization paths inside the step loop.
- **Root cause:** the existing sparse trainer path read CUDA scalars with `.cpu()` / `.item()`, synchronized for forward timing, and kept compatibility-only telemetry/adaptive control in the same loop.
- **Fix:** split the trainer into compatibility and strict paths. The strict path is `_run_single_attempt_strict()` in `crates/pyxlog/python/pyxlog/ilp/trainer.py:667`, uses a simple step-based temperature schedule, skips per-step telemetry/timing, and keeps the hot loop free of Python `.cpu()` / `.item()` / `torch.cuda.synchronize()` calls. Strict-incompatible config knobs are rejected in `crates/pyxlog/python/pyxlog/ilp/trainer.py:183-197`.
- **Classification:** strict path fixed; compatibility telemetry/dense-mask behavior remains available only with `strict_gpu_native=False`.

## Issue 2: strict sparse mask setup still escaped to host-selected IDs

- **Issue:** sparse strict training still materialized selected candidate IDs on the Python side before calling into Rust.
- **Root cause:** `SparseMaskBackend.apply_mask()` converted `torch.argsort(...)` results with `.cpu().tolist()` in the active training loop.
- **Fix:** add a strict sparse helper in `crates/pyxlog/python/pyxlog/ilp/backend.py:162-180` and a corresponding native entry point `set_rule_mask_sparse_selected_device(...)` in `crates/pyxlog/src/ilp.rs:1129-1268`. Candidate ordering is now stored on the compiled program in `crates/pyxlog/src/lib.rs:462-485` and uploaded by `set_candidate_map(...)` in `crates/pyxlog/src/ilp.rs:599-607`.
- **Classification:** strict path fixed on the Python side; compatibility host-selected sparse APIs remain available behind non-strict mode.

## Issue 3: host semantic runtime APIs remained callable under strict mode

- **Issue:** strict ILP runtime still allowed host-materializing semantic helpers such as `fact_exists()`, `relation_facts()`, `sample_false_positives()`, `batch_fact_membership()`, and `batch_tagged_credit()`.
- **Root cause:** only the legacy `set_rule_mask_sparse()` path was guarded by `strict_zero_dtoh`; the host semantic query surface was still open.
- **Fix:** add `ensure_host_semantic_compat(...)` in `crates/pyxlog/src/ilp.rs:2110` and gate host semantic APIs at:
  - `crates/pyxlog/src/ilp.rs:1322-1347`
  - `crates/pyxlog/src/ilp.rs:1350-1527`
  - `crates/pyxlog/src/ilp.rs:1529-1604`
  - `crates/pyxlog/src/ilp.rs:1774-1816`
  - `crates/pyxlog/src/ilp.rs:1818-1904`
- **Classification:** compatibility-only APIs are now explicitly unavailable in strict mode.

## Issue 4: strict training still leaked into compatibility-only holdout/promotion flows

- **Issue:** `train_only()` in strict mode still computed host holdout scores after convergence, and `train_and_promote()` / holdout helpers were still callable under strict mode.
- **Root cause:** holdout and promotion code paths use host semantic APIs (`fact_exists()`, `relation_facts()`) by design but were not hard-gated.
- **Fix:** skip automatic holdout scoring for strict training in `crates/pyxlog/python/pyxlog/ilp/trainer.py:1225-1228`, hard-gate holdout scoring in `crates/pyxlog/python/pyxlog/ilp/holdout.py:17-35`, and hard-gate promotion in `crates/pyxlog/python/pyxlog/ilp/promoter.py:21-46`.
- **Classification:** compatibility-only and explicitly unavailable in strict mode.

## Verification

- `maturin develop --release -m crates/pyxlog/Cargo.toml`
  - Result: build succeeded; editable `pyxlog-0.4.0` installed.
- `python -m pytest python/tests/test_ilp_sparse.py python/tests/test_ilp_sparse_guard.py python/tests/test_ilp_types.py -q`
  - Result: `25 passed in 3.04s`
- `python -m pytest python/tests/test_ilp_d2h_gate.py -q`
  - Result: `17 passed in 57.08s`
- `python -m pytest python/tests/test_ilp_device_queries.py -q -x`
  - Result: `2 passed in 0.68s`
- `python -m pytest python/tests/test_ilp_credit_gpu.py -q -x`
  - Result: `17 passed in 6.13s`
- `python -m pytest python/tests/test_ilp_holdout.py -q -x`
  - Result: `8 passed in 319.71s`
- `python -m pytest python/tests/test_ilp_promoter.py -q -x`
  - Result: `10 passed in 224.35s`
- `python -m pytest python/tests/test_ilp_trainer.py -q`
  - Result: `22 passed in 174.54s`

## Residual risks

- `crates/pyxlog/src/ilp.rs:1189-1267`
  - `set_rule_mask_sparse_selected_device(...)` still resolves selected candidate IDs by downloading the selected-ID tensor to host with `download_column_untracked::<...>()` before mapping into host `active_entries`. This removes the Python-side `.cpu().tolist()` escape, but the native strict sparse path still contains a control-plane device-to-host read.
- `crates/pyxlog/python/pyxlog/ilp/trainer.py:830-905`
  - strict attempt finalization still materializes summary metrics (`selected_hard`, witness coverage, final result fields) on the host after the step loop. The hot loop is clean, but the final result packaging is not fully device-only.
