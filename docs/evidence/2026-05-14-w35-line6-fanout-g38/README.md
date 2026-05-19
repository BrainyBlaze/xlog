# W3.5 Line-6 Fanout Spike Evidence

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`  
**Sub-goal:** G_W35 / W3.5 shared-memory narrowing  
**Branch:** `bench-spike/w35-line6-fanout-g38`  
**Base:** `feat/w33-hg-block-slice-prod @ 035b0713`

## Scope

This spike reopens W3.5 on the paper-class fixture required by G38, then closes it via the S_W35.5 graceful-close path after the redesigned spike failed parity below the required threshold:

- `triangle-line6-fanout-512`: 512 root keys, 1 `xy` row per root, 512 `z` fanout per root, expected 262,144 output rows.
- `triangle-small-inner-4K`: inherited low-fanout parity geometry, p50 work per `xy` row = 2.

The measured experiment added a line-6 shared-memory candidate and a benchmark harness:

| Surface | File | Behavior |
|---|---|---|
| CUDA kernel experiment | `crates/xlog-cuda/kernels/wcoj.cu` | Tested a line-6 shared-memory count candidate plus a high-fanout aligned materializer. |
| Provider experiment | `crates/xlog-cuda/src/provider/wcoj_metadata.rs` | Tested the line-6 provider entry, high-fanout aligned path, and low-fanout guard. |
| Bench experiment | `crates/xlog-integration/benches/wcoj_w35_line6_fanout.rs` | Generated paper-class and parity fixtures, checked row equality, reported direct paired timing, Criterion medians, occupancy, sync diagnostics, and peak VRAM. |

The experimental production code was reverted after the RED result. Final branch state keeps the single global HG path and adds the required kernel-header citation:

```text
// Paper §5 Algorithm 2 lines 1,3,4,5,7,9,10,12 preserved; lines 6 + per-warp narrowing dropped per Phase-1 §2.2 A5 hardware constraint.
```

## Commands

```text
cargo fmt --all
EXIT 0

cargo test -p xlog-integration --bench wcoj_w35_line6_fanout --no-run
EXIT 0

cargo bench -p xlog-integration --bench wcoj_w35_line6_fanout -- --output-format bencher
EXIT 0
```

Final graceful-close verification after reverting the experimental production code:

```text
cargo fmt --check --all
EXIT 0

cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run
EXIT 0
```

## Measurements

Final measurements are in `measurements.tsv`.

| Cell | Direct speedup | Direct p vs baseline | Gate p | Criterion speedup | Row equality | Static bytes | Peak VRAM | Verdict |
|---|---:|---:|---:|---:|---|---:|---:|---|
| `triangle-line6-fanout-512` | 1.432992x | 0.001953 | 0.539062 | 1.450661x | PASS, 262,144 rows | 2,092 | 17,848,360 | RED: M_W35.1 requires >= 1.5x |
| `triangle-small-inner-4K` | 1.024675x | 0.187500 | 0.065430 | 0.936408x | PASS, 1,219 rows | 4,096 | 17,848,360 | RED: parity did not clear the statistical/criterion guard |

## RCA

The paper-class fixture validates the G37 fixture-amendment premise: p50 work per root is 512, and the aligned path reduces line-6 lookup work enough to beat the cached HG path directionally.

The remaining miss is pipeline-level:

- M_W35.1 still misses the required 1.5x gate after the high-fanout aligned path; the final direct median is 1.432992x and Criterion is 1.450661x.
- The aligned path has low shared-memory footprint (2,092 bytes) and only 2 block syncs, so the miss is not explained by shared-memory occupancy.
- The low-fanout parity cell is protected by the fanout guard, but the final Criterion parity ratio is 0.936408x, below the 0.95x guard.
- Peak VRAM remains far below the 38 GB gate in both cells.

## Graceful-Close Verdict

G_W35 is closed-as-graceful per S_W35.5, not GREEN. The measured line-6 shared-memory design did not satisfy M_W35.1 and regressed the parity Criterion below 0.95x after redesign, so the final branch reverts to the single global HG path and documents M_W35.1 as unreachable under the current hardware-Amdahl constraint.

Do not cut `feat/w35-line6-fanout-prod` from the failed experimental code. Downstream G_W36 must use the G_W35 graceful-close state, not a passing shared-memory predecessor.
