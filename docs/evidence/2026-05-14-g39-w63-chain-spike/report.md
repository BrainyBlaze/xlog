# Goal-039 G_W63_CHAIN Bench Spike

Date: 2026-05-17.
Branch: `bench-spike/w63-chain-promoter-g39`.
Base: `feat/g39-pre-profiler-trace` at `d7a82eca3fbfee09976559cb4354c1f5e8804621`.

## Scope

This is the G_W63 bench spike, not the production G_W63 closure branch.
G_PRE measured `evaluate_pct = 0.9661778834300995`, so G_W63_CHAIN priority is
HIGH. The spike tests whether 2-atom chain bodies can be detected and routed
through an optimized path while preserving the existing binary fallback identity.

The plan names a production `ChainJoin` RIR node. This spike deliberately uses
the existing `RirNode::MultiWayJoin` fallback wrapper to avoid widening the RIR
enum across the full workspace before timing evidence exists.

## Implemented Spike Surface

- `crates/xlog-logic/src/promote.rs`: detects `Project(Join(Scan, Scan))` with
  one inner key and emits a two-input `MultiWayJoin` carrying the original
  binary plan as `fallback`.
- `crates/xlog-runtime/src/executor/wcoj_dispatch.rs`: adds a chain matcher and
  default-on `XLOG_WCOJ_W63_CHAIN_ENABLE` gate. `0` / `false` disables the route
  for A/B timing.
- Runtime chain route:
  - sorted eligible U32/Symbol inputs: W4.3 `sort_merge_join_v2_inner_u32_1key`;
  - threshold eligible unsorted U32/Symbol inputs: W4.2
    `nested_loop_join_v2_inner_u32_1key`;
  - otherwise: existing `hash_join_v2`.
- `crates/xlog-runtime/src/executor/recursive.rs`: attempts chain dispatch before
  triangle / 4-cycle / K-clique dispatch in non-recursive and recursive
  `MultiWayJoin` paths.
- `crates/xlog-integration/tests/test_w63_chain_promoter_spike.rs`: end-to-end
  fallback parity cert and ignored timing smoke.

## Timing Smoke

Command:

```bash
cargo test -p xlog-integration --test test_w63_chain_promoter_spike -- --ignored --nocapture
```

Raw output:

```text
W63_CHAIN_TIMING sorted_threshold n=2000 iterations=20 fallback_ms=189.714 chain_ms=32.720 ratio=5.798042 fallback_dispatches=0 chain_dispatches=20
test chain_dispatch_timing_smoke_sorted_threshold_cell ... ok
```

Interpretation:

- Synthetic sorted threshold cell: `a(X,Z), b(Z,Y)` with 2,000 rows per relation.
- Cartesian product is 4,000,000, exactly the W4.2/W4.3 threshold cell.
- Fallback route: env-disabled chain path, embedded binary plan.
- Chain route: default-on chain path, 20/20 dispatches.
- Ratio: `fallback_ms / chain_ms = 5.798042x`.

This timing smoke is evidence that the route is worth productionizing. It is not
the final M_W63.1 / M_W63.2 acceptance evidence because it is not a m37c-prime
trace replay subset and not the plan's 977K synthetic scale gate.

## Focused Verification

```bash
cargo test -p xlog-logic chain
cargo test -p xlog-runtime chain
cargo test -p xlog-integration --test test_w63_chain_promoter_spike -- --nocapture
```

Results:

- `xlog-logic chain`: PASS. Chain promoter positive and rejection tests pass.
- `xlog-runtime chain`: PASS. Chain matcher, env gate, and existing chain-named
  runtime tests pass.
- `test_w63_chain_promoter_spike`: PASS. Default-on chain dispatch matches
  env-disabled fallback rows; ignored timing smoke is excluded from normal test.

## Status Against G_W63 Metrics

- M_W63.1: NOT YET RUN. Requires m37c-prime trace replay subset with at least
  100 chain invocations.
- M_W63.2: PARTIAL SPIKE ONLY. Synthetic threshold cell shows 5.798042x, but the
  977K scale gate is not yet run.
- M_W63.3: PARTIAL. Row equality cert passes on the spike chain fixture.
- M_W63.4-M_W63.8: NOT YET RUN.

Next action: decide whether to productionize this MultiWayJoin-based spike or
replace it with the plan's named `ChainJoin` RIR node before running full
M_W63.1-M_W63.8 acceptance.
