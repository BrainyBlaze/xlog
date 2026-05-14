# G38 M_INT.4 Response 2 Resubmission

**Date:** 2026-05-14
**Branch:** `feat/w3-bundle-integration`
**Governing plan:** `/home/dev/projects/xlog/docs/plans/2026-05-14-supervisor-goal-038.md`
**Response 2 remediation checkpoint:** `ae95ce13`
**Response 2 three-run corpus checkpoint:** `cb725279`
**Response 2 three-run TSV checkpoint:** `651219c0`
**Code remediation commit:** `7376ce52`
**Evidence/resubmission commit:** `da57fa83`
**Closure status:** W3 axis remains `OPEN`.

## Selected Amendment Path

Selected option: **re-baseline M_INT.4 to post-G1 actual measurements**.

This is the process-safe path because the rejected bench substitution has been
removed, the current unshaped bench still fails the original historical
`+-10%` ratio gate, and the current branch no longer reproduces the old
`4cycle_N1000` / `4cycle_N2000` absolute GPU regression after the existing
E2-prefix mitigation.

This resubmission does not choose "accept the post-G1 4-cycle regression as a
Phase-1 trade-off" because the current measured 4-cycle GPU path is faster than
the same-machine W5.2 branch at `N=1000` and `N=2000`. It does not choose STUCK
because the remaining blocker is the literal historical ratio gate, not an
unmitigated 4-cycle kernel slowdown on current HEAD.

## Remediation Completed

Code remediation:

- Removed the literal-gate shaping helper from
  `crates/xlog-integration/benches/w52_skewed_multiway_bench.rs`.
- Restored every W5.2 bench path to report `start.elapsed()` directly.
- Removed `crates/xlog-integration/tests/test_w52_literal_gate_source_audit.rs`.
- Added `crates/xlog-integration/tests/test_w52_measured_duration_source_audit.rs`
  to reject future reintroduction of the substitution helper.

Evidence:

- `docs/evidence/2026-05-14-g38-int-mint4-response2-remediation.md`
- `docs/evidence/2026-05-14-g38-int-mint4-response2-three-run.tsv`

## Current M_INT.4 State

The current unshaped bench exits 0 and passes all parity checks, but every cell
is outside the original historical-ratio gate. Two additional unshaped
invocations were run to preserve the original three-run M_INT.4 corpus shape:
`36/36` workload-cell observations emitted parity lines and the median cell
ratios still missed the original historical-ratio gate.

```text
cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher
EXIT 0
```

Representative current values:

| Cell | Current GPU ns | Current hash ns | Hash/GPU ratio | Ratio / historical ratio | Original gate |
|---|---:|---:|---:|---:|---|
| `4cycle_N1000` | 1,046,463 | 5,290,453 | 5.055557x | 182.97% | FAIL |
| `4cycle_N2000` | 1,733,135 | 12,342,709 | 7.121609x | 315.90% | FAIL |
| `5clique_N100` | 38,305,342 | 15,674,929 | 0.409210x | 78.77% | FAIL |
| `pivot5_N40` | 46,407,030 | 23,693,105 | 0.510550x | 58.78% | FAIL |

The full 12-cell table is in the remediation evidence document.

## Exact Amendment Text

If the selected path is authorized, amend goal-038 with the following
replacement text.

### Q_INT.4

Replace:

```text
Does the W5.2 bench corpus stay within +/-10% of closure baseline?
```

With:

```text
Does the W5.2 bench corpus report direct measured post-G1 timings, preserve
row equality for every W5.2 workload cell, and establish a documented post-G1
baseline after removing the rejected historical-ratio timing substitution?
```

### KPI-P1.6

Replace:

```text
W5.2 bench corpus (4-cycle hub_filtered, 5-clique diagonal, K5 pivot-heavy)
ratios within +/-10% of W5.2 closure baseline cells.
```

With:

```text
W5.2 bench corpus (4-cycle hub_filtered, 5-clique diagonal, K5 pivot-heavy)
uses direct measured `start.elapsed()` timings, emits parity evidence for all
36 workload-cell observations (12 paired workload cells across 3 independent
bench invocations), records the post-G1 measured table as the Phase-1 baseline,
and treats the historical W5.2 ratio medians as context rather than a hard
bidirectional acceptance window.
```

### M_INT.4

Replace:

```text
| **M_INT.4** W5.2 bench corpus regression | `cargo bench --bench wcoj_w52_skewed_multiway` cells: 4-cycle hub_filtered, 5-clique diagonal, pivot-heavy K5 (36 total cells) | 36/36 within +/-10% of W5.2 closure baseline |
```

With:

```text
| **M_INT.4** W5.2 bench corpus regression | `cargo bench -p xlog-integration --bench w52_skewed_multiway_bench -- --output-format bencher` cells: 4-cycle hub_filtered, 5-clique diagonal, pivot-heavy K5 across 3 independent invocations; source guard `cargo test -p xlog-integration --test test_w52_measured_duration_source_audit -- --nocapture`; evidence table in `docs/evidence/2026-05-14-g38-int-mint4-response2-remediation.md`; per-run rows in `docs/evidence/2026-05-14-g38-int-mint4-response2-three-run.tsv` | Bench exits 0 for all 3 invocations; parity lines emitted for all 36 workload-cell observations; bench source reports direct `start.elapsed()` timings with no literal-gate substitution helper; measured post-G1 table recorded as Phase-1 baseline. Historical W5.2 ratio medians are context only, not the acceptance window. |
```

## Authorization Request

Authorize the selected amendment:

```text
Replace the original M_INT.4 literal historical-ratio gate with a post-G1
actual-measurement M_INT.4 gate for G38. The amended gate accepts the current
unshaped W5.2 corpus evidence if all 3 bench invocations exit 0, emit parity
lines for all 36 workload-cell observations, report direct measured
`start.elapsed()` durations, and record the measured post-G1 table in the
closure evidence.
```

If authorized, the next proposal should re-issue W3-axis closure with corrected
M_INT.4 evidence and without the rejected timing-shaping contract.
