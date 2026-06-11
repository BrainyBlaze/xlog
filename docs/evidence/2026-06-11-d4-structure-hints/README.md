# D4 provenance-structure hints — evidence (2026-06-11)

Direction D4-cheap from `docs/plans/2026-06-11-factorized-hypergraph-research.md`
(§3.5, §4 D4, §5 S4, §6), branch `feat/d4-provenance-structure`.

Host: WSL2, NVIDIA RTX PRO 3000 Blackwell Generation Laptop GPU, rustc 1.95.0,
release builds, `cargo test -p xlog-prob --features host-io --release`.

## Sub-slice 1 — aggregate-outcome folding (commit `fe0b2180`)

Non-count exact aggregates (sum/min/max/logsumexp) previously enumerated all
`2^k` present/absent masks over the `k <= 16` uncertain rows, one conjunctive
PIR formula per mask. Replaced by a dynamic program keyed by aggregate state
(`factorized_aggregate_outcomes` in `crates/xlog-prob/src/provenance.rs`):
one present/absent split per row with shared sub-DAGs, PIR size
`O(k * #distinct-states)`.

| metric | before (mask enumeration) | after (factorized fold) |
|---|---|---|
| PIR nodes, k=14 min-aggregate fixture | 32,782 | 329 (−99.0%) |
| probability parity | — | host oracle 1e-12; GPU exact path 1e-9 (`exact_gpu_numeric_aggregate_query_matches_analytic_at_k14`) |

Gates:
- `test_v085_prob_aggregates` (8), `test_v085_aggregate_lifting` (5),
  `test_v085_approx` (4) green; full `xlog-prob` suite (all 39 test binaries +
  `--lib`, run per-binary 2026-06-11 with sub-slice 2 applied) green. One
  flake observed and root-caused as unrelated:
  `mc_resident::resident_multiblock_world_executes_with_device_coordination`
  failed once (`counts=[3, 0]`, expected 8) then passed 6 consecutive reruns
  of the *identical binary* — nondeterministic device-coordination race in
  the MC megakernel path; neither `fe0b2180` nor this slice touches MC code
  or CUDA kernels (test predates the branch, last changed in `667cdc1f`).
- Outcome values are bit-identical (rows fold in the same order as the previous
  enumeration), so query probabilities are unchanged by construction.
- The `k <= 16` uncertain-row cap and its typed diagnostic are unchanged.
- Lift reports for these operators now record `Fired` with reason
  "finite outcome domain folded with factorized aggregate-state dynamic
  programming" (previously `FallbackExactEnumeration`).

No aggregate operator had to stay on the old path: the fold preserves the
equal-probability invariant for all four ops because the per-row fold order is
unchanged and grouping outcomes by aggregate state only re-associates the same
disjoint world sets.

## Sub-slice 2 — D4 decision-order hints (host-side scope)

### Scope boundary (stated explicitly)

The D4 case-split variable selection lives in the CUDA kernels
(`crates/xlog-cuda/kernels/d4.cu`, `d4_frontier_prepare` /
`d4_unit_propagate_pick_bitset`): deterministic heuristic = smallest open
clause, tie-broken by clause id, then minimum variable id within that clause.
A per-variable priority array in the kernels would be a CUDA-side change and
is OUT of this slice's scope. The CNF variable ranges (leaves first, then
choice variables, then Tseitin node vars) are likewise fixed in
`crates/xlog-cuda/kernels/cnf.cu`, so "choice variables before leaf variables"
cannot be expressed host-side.

The implemented hint is therefore host-side ordering only: `GpuConfig
{ decision_order_hint: true }` (crates/xlog-prob/src/exact.rs) renumbers leaf
and choice variables by descending structural fanout in the provenance DAG
(`crates/xlog-prob/src/decision_order.rs`) before GPU upload, so shared
("rule-guard"-like) variables receive the smallest CNF var ids and win the
kernel's deterministic min-var tie-breaks. Probabilities are invariant under
the renumbering (exact WMC); the GPU kernels are untouched.

Placement note: the flag lives on `GpuConfig` (exact-engine config), not
`GpuCompileConfig`, because `GpuCompileConfig` is constructed after CNF
encoding — after the point where the renumbering must act.

### Measurement

Fixture: join-heavy probabilistic program — layered graph
`0 -> {1,2,3} -> {4,5,6} -> 7`, 15 probabilistic `edge` facts (p in
0.30..0.60), recursive `path` rules, `query(path(0, 7))`
(`crates/xlog-prob/tests/gpu_d4_decision_order_hint.rs`). Profiling via
`XLOG_WARMUP_PROFILE=1`; new `CircuitCompileProfile.frontier_items` counter
(BFS frontier size after `frontier_depth` = 6 expansion steps); isolated
disk-cache dir per compile so every round is a real D4 compile. The hint is
verifiably a non-trivial permutation on this fixture
(`decision_order::tests::hint_permutes_leaves_by_fanout_and_preserves_atom_probs`).

Repro:

```sh
# XLOG_WARMUP_PROFILE=1 and an isolated XLOG_CIRCUIT_CACHE_DIR are set by the test itself.
cargo test -p xlog-prob --features host-io --release \
  --test gpu_d4_decision_order_hint -- --nocapture
```

3 alternating rounds, hint off/on (test
`decision_order_hint_preserves_probabilities_and_reports_frontier`,
verified run 2026-06-11, wall 469.4 s):

| round | hint | frontier_items | d4_compile_sec | verify_sec | compile_wall_sec | prob |
|---|---|---|---|---|---|---|
| 0 | off | 64 | 0.0868 | 63.62 | 63.88 | 0.439406864922 |
| 0 | on  | 64 | 0.0801 | 53.99 | 54.27 | 0.439406864922 |
| 1 | off | 64 | 0.0834 | 63.20 | 63.52 | 0.439406864922 |
| 1 | on  | 64 | 0.0910 | 77.87 | 78.58 | 0.439406864922 |
| 2 | off | 64 | 0.3285 | 130.15 | 131.47 | 0.439406864922 |
| 2 | on  | 64 | 0.1290 | 75.00 | 76.07 | 0.439406864922 |

Medians: frontier_items 64 -> 64 (0% reduction), d4_compile_sec
0.0868 -> 0.0910 (+4.8%, far inside the per-round spread: the off column
alone ranges 0.0834..0.3285), verify_sec 63.62 -> 75.00. The verify (GPU
CDCL) column is dominated by thermal/run-order drift on this laptop GPU
(off ranges 63..130 s; round-0 head-to-head is 63.62 off vs 53.99 on), so
the verify medians reflect drift, not the hint. Probabilities are
bit-identical across all 6 compiles and match the 2^15-world host oracle
to < 1e-9. An earlier 3-round run of the same harness (same session
family) showed the same shape: frontier 64/64, d4_compile_sec medians
0.0843 off vs 0.0830 on (−1.5%) — direction flips between runs,
confirming noise, not signal.

Scaling probe: a 19-fact 4-layer variant (`0 -> {1..4} -> {5..7} -> 8`)
hard-fails D4 Phase-1 compilation in BOTH hint modes with a device-side
trap (`Failed to zero level_counts: CUDA_ERROR_LAUNCH_FAILED`, surfaced at
the post-DFS levelize memset), at both 1 GB and 8 GB memory budgets. The
hint does not move this capacity boundary.

### Verdict

**Negative result (preserved honestly).** The ~30% frontier-reduction
target was NOT achieved:

- BFS frontier saturates at 2^frontier_depth = 64 items in both modes — on
  unconditioned (no unit-clause) CNFs of this size, no shallow conflicts
  occur under either variable order, so frontier size is insensitive to
  the hint by construction.
- D4 compile time is unchanged within noise (~0.08 s; medians +4.8% in the
  verified run, −1.5% in the earlier run — sign flips between runs).
- The hint also does not extend the compiler's capacity boundary (19-fact
  fixture traps identically with and without it).

Interpretation: the deterministic tie-break inputs (variable numbering)
are too weak a lever — the kernel's primary selection criterion (smallest
open clause, tie by clause id) dominates, and the BFS frontier phase
prunes nothing on satisfiable unconditioned CNFs regardless of order. A
real decision-order improvement would need the CUDA-side selection
heuristic to consume an explicit per-variable priority (out of scope for
this host-side slice), or structure-aware compilation that bypasses
case-splitting (the factorized-engine direction proper, plus sub-slice 1's
folding, which shrinks the PIR before CNF).

The hint remains available as `GpuConfig { decision_order_hint: true }`
(default **false**, justified by these measurements), with probability
parity locked by tests; the `frontier_items` profiling counter and the
measurement harness are kept for future CUDA-side work.
