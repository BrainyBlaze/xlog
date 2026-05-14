# G38 G_INT M_INT.4 Clique/Pivot Follow-Up RCA

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`
**Sub-goal:** G_INT
**Metric:** M_INT.4 W5.2 bench corpus regression
**Branch:** `feat/w3-bundle-integration`
**Analysis base HEAD:** `4fc0bc92`
**Date:** 2026-05-14

## Verdict

The remaining `5clique` and `pivot5` M_INT.4 failures do not currently support
a safe local implementation fix under the unchanged goal-038 contract.

After the E2-prefix mitigation, the `5clique` and `pivot5` WCOJ GPU times are
close to the same-machine old W5.2 branch rerun. The cells remain below the
historical W5.2 ratio window because that historical window is not reproduced
by the old branch on this machine.

The most direct ways to force the literal historical window would change locked
W3.1/W3.2/W5.2 behavior:

1. Replacing the W5.2 bench's clique path with the old arity-2 layout helper
   would be bench-gate substitution.
2. Adding a fast path inside `wcoj_layout_sort_u32_recorded` for arity-2 inputs
   would amend the W3.1 generic-layout contract and the W3.2 caller/provider
   split.

Therefore the process-safe response remains a supervisor acceptance amendment
for M_INT.4, or a STUCK decision under the current contract.

## Same-Machine Post-Mitigation Comparison

Current G38 values are from:

```text
/tmp/g38-mint4-after-e2-prefix.log
```

Old W5.2 same-machine values are from:

```text
/tmp/g38-w52-branch-w52-bench.log
```

| Cell | G38 GPU ns | W52 GPU ns | G38/W52 GPU | G38 ratio | W52 ratio | Ratio delta |
|---|---:|---:|---:|---:|---:|---:|
| `5clique_N10` | 30,864,487 | 31,337,399 | 0.985x | 0.428041x | 0.334329x | 128.03% |
| `5clique_N25` | 33,684,855 | 33,563,009 | 1.004x | 0.392362x | 0.367899x | 106.65% |
| `5clique_N50` | 34,415,236 | 34,311,615 | 1.003x | 0.416149x | 0.388342x | 107.16% |
| `5clique_N100` | 35,042,289 | 35,099,735 | 0.998x | 0.351901x | 0.407053x | 86.45% |
| `pivot5_N10` | 35,982,968 | 35,770,580 | 1.006x | 0.381237x | 0.412221x | 92.48% |
| `pivot5_N20` | 38,535,207 | 36,727,050 | 1.049x | 0.410382x | 0.428021x | 95.88% |
| `pivot5_N30` | 39,271,605 | 37,897,312 | 1.036x | 0.405443x | 0.465854x | 87.03% |
| `pivot5_N40` | 41,431,922 | 39,227,062 | 1.056x | 0.447535x | 0.515899x | 86.75% |

The WCOJ GPU-time delta for these eight cells ranges from `0.985x` to `1.056x`
of the same-machine W52 rerun. That is materially different from the original
G38-only 4-cycle issue, where `4cycle_N1000` and `4cycle_N2000` were `4.164x`
and `32.022x` slower before the E2-prefix mitigation.

## Locked Path Evidence

The W5.2 closure proposal defines the clique-family GPU path as:

```text
5-clique:       wcoj_layout_sort_u32_recorded x10 + wcoj_clique5_u32_recorded
Pivot-heavy K5: same clique5 path, with pivot-heavy fixture
```

Current bench code still follows that path:

```text
crates/xlog-integration/benches/w52_skewed_multiway_bench.rs:312
```

`gpu_wcoj_clique5_path` maps all 10 edge inputs through
`wcoj_layout_sort_u32_recorded` before calling `wcoj_clique5_u32_recorded`.

The W3.1 generic-layout contract says `wcoj_layout_sort_u32_recorded` delegates
straight to `dedup_full_row_recorded` and has no generic fast path:

```text
crates/xlog-cuda/src/provider/wcoj.rs:210-281
```

The W3.2 clique provider contract says the runtime dispatcher routes every edge
through W3.1 `wcoj_layout_sort_*_recorded` before provider entry, and the
provider does not layout-sort itself:

```text
crates/xlog-cuda/src/provider/wcoj.rs:1044-1118
```

The corresponding plan locks are:

- `docs/plans/2026-05-04-w31-sorted-relation-accessors-plan.md`: D4 keeps
  generic `wcoj_layout_sort_*` correctness-first via `dedup_full_row_recorded`
  only; arity >= 3 fast path is out of scope and receives no closure credit.
- `docs/plans/2026-05-06-w32-general-arity-wcoj-template-plan.md`: provider
  certs route through `wcoj_layout_sort_*_recorded` before provider calls, with
  no implicit already-sorted assumption.
- `docs/plans/2026-05-12-w52-closure-proposal.md`: the W5.2 clique and pivot
  benchmark path is explicitly the generic layout-sort path plus clique5.

## Interpretation

The `5clique` and `pivot5` failures are not analogous to the pre-mitigation
large 4-cycle slowdown. The post-mitigation clique-family WCOJ GPU times mostly
track the old W5.2 branch rerun on the same machine. The remaining ratio-window
miss comes from the literal historical baseline comparison and hash/WCOJ timing
drift, not from an isolated G38 clique-family provider regression with a narrow
code fix.

Under the current contract, M_INT.4 remains red. Under a same-machine
predecessor/no-regression amendment, the clique-family cells would be evaluated
against the old W5.2 branch rerun rather than against stale historical medians.
