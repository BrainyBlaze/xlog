# G39 W65 Authorized DTS Source Fix

Date: 2026-05-18.
xlog branch: `feat/w65-sort-label-propagation-g39`.
xlog pre-closure HEAD: `a7a14048`.
DTS-DLM branch: `fix/w65-support-body-len-guard`.
DTS-DLM HEAD: `1c2e9ed`.
DTS-DLM behavior fix commit: `e0324fa`.

## Authorization

The W65 lock-conflict escalation at
`docs/plans/2026-05-18-g39-w65-lock-conflict-escalation.md` is superseded by the
user authorization to mutate DTS-DLM for the full xlog + DTS production-grade
goal. The earlier escalation and RCA are preserved as evidence of the
pre-authorization xlog-only state; they are not deleted.

The superseded conflict was:

- Goal-039 lock 14 made `Sort enrichment: N sort-map misses` an error condition.
- M_W65.1 required zero diagnostics on m37c-prime replay.
- The RCA showed the remaining warning source was DTS-DLM support source
  generation, not xlog sort-label propagation.
- The original plan's DTS mutation lock prevented the required source fix.

The authorized DTS scope is exactly the W65 source-generation defect:

- generated `support_N` clauses now include `wmir_len_N(RId)` guards;
- static support metadata upload now includes `wmir_len_N` relations;
- frozen replay snapshots include the same `wmir_len_N` uploads;
- recursive sort enrichment now resolves support rows to a fixpoint independent
  of xlog row order;
- probe call sites use the shared static-relation upload helper.

## DTS Red Tests

The body-length guard test failed before the DTS fix:

```text
PYTHONPATH=src pytest -q \
  src/tests/propagate/test_xlog_executor.py::test_mixed_unary_binary_support_1_excludes_binary_rule_ids

assert [20] == []
```

The recursive sort-enrichment fixpoint test failed before the fixpoint change:

```text
PYTHONPATH=src pytest -q \
  src/tests/propagate/test_xlog_executor.py::test_enrich_sorts_resolves_recursive_support_until_fixpoint

p4 usable pred sort: 1
expected: IdSort.LEARNED_PRED
captured warning: Sort enrichment: 1 sort-map misses
```

## DTS Green Validation

Focused propagation and pilot regression:

```text
PYTHONPATH=src pytest -q \
  src/tests/propagate/test_xlog_executor.py \
  src/tests/propagate/test_m29_evaluate_capture.py \
  src/tests/propagate/test_m31_wmir_committed_capture.py \
  src/tests/propagate/test_m32_compile_source_capture.py \
  src/tests/propagate/test_m33_wmir_body_capture.py \
  src/tests/learn/test_m17_t3_doc.py \
  src/tests/pilots/test_m30_strata_selector.py \
  src/tests/pilots/test_m34_strata_selector.py

75 passed, 14 warnings
```

Touched-file lint and syntax:

```text
ruff check \
  src/dts_dlm/propagate/xlog_executor.py \
  src/dts_dlm/learn/t3_doc.py \
  src/dts_dlm/pilots/m16_xlog_credit_probe.py \
  src/dts_dlm/pilots/m17_neural_supervision_probe.py \
  src/tests/propagate/test_xlog_executor.py

All checks passed!
```

```text
PYTHONPATH=src python -m compileall -q \
  src/dts_dlm/propagate/xlog_executor.py \
  src/dts_dlm/learn/t3_doc.py \
  src/dts_dlm/pilots/m16_xlog_credit_probe.py \
  src/dts_dlm/pilots/m17_neural_supervision_probe.py \
  src/tests/propagate/test_xlog_executor.py

exit 0
```

## xlog W65 Certs

```text
cargo test -p xlog-integration --test test_w65_sort_label -- --nocapture

test result: ok. 5 passed; 0 failed
```

```text
cargo check -p pyxlog -p xlog-integration --tests

Finished `dev` profile
```

## m37c-prime Replay

Environment:

```text
PYTHONPATH=/tmp/g39-pyxlog:/home/dev/projects/dts-dlm/.worktrees/w65-support-body-len-guard/src
XLOG_CUBIN_DIR=/home/dev/projects/xlog/.worktrees/g39-w65-sort-label/target/release/build/xlog-cuda-43b482a33001fc07/out
```

Five-doc bounded replay:

```text
run_id=g39-w65-dtsfix-5doc-20260518-r3
eval_path=/tmp/g39-w65-corpus-5.jsonl
out_root=/tmp/g39-w65-dtsfix-m37c-prime
log=/tmp/g39-w65-dtsfix-5doc-r3.log

RC=0
SORT_WARNINGS=0
ROWS=5
fails=0
wall=38.6s
```

Fifty-doc replay:

```text
run_id=g39-w65-dtsfix-50doc-20260518-r2
eval_path=/tmp/g39-pre-corpus-50.jsonl
out_root=/tmp/g39-w65-dtsfix-m37c-prime
log=/tmp/g39-w65-dtsfix-50doc-r2.log
manifest=/tmp/g39-w65-dtsfix-m37c-prime/g39-w65-dtsfix-50doc-20260518-r2/eval/manifest.json

RC=0
SORT_WARNINGS=0
ROWS=50
fails=0
wall=1370.5s
```

This clears the W65 warning gate after the authorized DTS source-generation fix.

## M_W65 Status

| Metric | Status | Evidence |
|---|---:|---|
| M_W65.1 sort-map diagnostic count on m37c-prime 50-doc replay | PASS | `g39-w65-dtsfix-50doc-20260518-r2`: `RC=0`, `ROWS=50`, `SORT_WARNINGS=0`. |
| M_W65.2 Schema-API regression | PASS | Current rerun: `cargo test -p xlog-integration --test test_w65_sort_label -- --nocapture` passed 5/5; `cargo check -p pyxlog -p xlog-integration --tests` passed. |
| M_W65.3 every output relation column has non-default sort label | PASS | Current rerun covers compile-time schema, runtime query result, PyO3 packing, and `_native.pyi`. |
| M_W65.4 DTS-DLM unchanged check | SUPERSEDED | The unchanged check remains true for the pre-authorization RCA state; post-authorization DTS source mutation is recorded at `e0324fa` and `1c2e9ed`. |
| M_W65.5 RCA documented | PASS | `docs/evidence/2026-05-14-g39-w65-sort-label-rca.md` plus this supersession evidence. |

## Boundaries

No push, tag, board close, or merge is included in this W65 evidence update.
The DTS-DLM ignored fixture symlinks used for validation remain untracked.
