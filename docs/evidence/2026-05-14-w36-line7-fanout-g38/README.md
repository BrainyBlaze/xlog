# W3.6 Line-7 Warp Prefix Graceful Close

**Goal document:** `docs/plans/2026-05-14-supervisor-goal-038.md`  
**Sub-goal:** G_W36 / W3.6 warp-level primitives  
**Branch context:** `bench-spike/w35-line6-fanout-g38` graceful-close state  
**Predecessor:** G_W35 closed-as-graceful in `docs/evidence/2026-05-14-w35-line6-fanout-g38/`

## Scope

G_W36 is sequenced after G_W35 and measures cooperative warp `Prefix(x_j)` against the G_W35 shared-memory baseline. G_W35 did not produce an accepted shared-memory baseline:

- `triangle-line6-fanout-512`: RED, final Criterion speedup 1.450661x against the 1.5x gate.
- `triangle-small-inner-4K`: RED, final Criterion speedup 0.936408x against the 0.95x parity guard.
- G_W35 final state reverted to the single global HG path.

Because the required G_W35 shared-memory predecessor is closed-as-graceful, a line-7 warp-prefix speedup branch would have no accepted G_W35-only baseline to compare against. G_W36 therefore closes via S_W36.3 rather than introducing another production dispatch path.

## Paper-Citation Evidence

The kernel header in `crates/xlog-cuda/kernels/wcoj.cu` carries the required paper-alignment citation:

```text
// Paper §5 Algorithm 2 lines 1,3,4,5,7,9,10,12 preserved; lines 6 + per-warp narrowing dropped per Phase-1 §2.2 A5 hardware constraint.
```

For G_W36, `per-warp narrowing dropped` is the operative line-7 closure: the HG kernel keeps the paper §5 outer block-sliced shape and does not add cooperative warp `Prefix(x_j)`.

## Verification

No additional production code was added for G_W36. The graceful-close state is covered by the G_W35 post-revert verification:

```text
cargo fmt --check --all
EXIT 0

cargo test -p xlog-integration --bench wcoj_w33_superhub --no-run
EXIT 0
```

## Verdict

G_W36 is closed-as-graceful per S_W36.3, not GREEN. W3.6 should be represented downstream as a graceful flag, not as a passing warp-prefix implementation.
