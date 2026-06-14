# WCOJ Architecture Guide

Status: WCOJ architecture guide.
Audience: xlog maintainers, WCOJ implementers, and downstream integrators.
Primary paper reference: Sun et al., "Scaling Worst-Case Optimal Datalog to GPUs",
arXiv:2604.20073v2, https://arxiv.org/html/2604.20073v2.

This guide describes the xlog WCOJ subsystem as of the WCOJ bundle handoff plus
the follow-on integration surfaces that must compose with it. It is intentionally
implementation-facing: it names the RIR nodes, promoter gates, runtime hooks,
CUDA provider entries, cost-model data flow, and recursive fixpoint contracts
that protect row-set determinism when a rule is routed away from the ordinary
binary-join path.

## Scope Map

The current production WCOJ path is centered on `RirNode::MultiWayJoin`. The
node is emitted by `xlog-logic::promote::promote_multiway`, consumed by
`crates/xlog-runtime/src/executor/wcoj_dispatch.rs`, and backed by recorded
CUDA provider entries in `crates/xlog-cuda/src/provider/wcoj.rs` plus
`crates/xlog-cuda/kernels/wcoj.cu`.

Current or already-completed WCOJ surfaces:

- Triangle WCOJ for `U32`, `Symbol`, and `U64`.
- 4-cycle WCOJ for `U32`, `Symbol`, and `U64`.
- K=5 and K=6 clique WCOJ with K-clique variable-order plans.
- Runtime histogram metadata for the leader edge on K-clique dispatch.
- K-clique helper-split specs carried from the planner into the helper pass.
- Recursive SCC integration through body-keyed dispatch during seed and
  per-variant semi-naive evaluation.

Follow-on integration surfaces covered here:

- `ChainJoin` and `try_promote_chain`.
- K=7/K=8 templates.
- Sort-label propagation.
- CUDA Graphs capture.
- Per-stream pool sizing for the integrated bundle.

This document does not mark closure-board state. The closure board remains the
source of truth for approval state.

## Paper Alignment

The paper's section 4 describes the high-level SRDatalog iteration: compile
rules to GPU phases, execute histogram/count/materialize, compute delta, build
or update flat indexes, and merge new tuples into recursive state. xlog maps
that to RIR lowering, promotion, runtime dispatch, provider recorded launches,
and the recursive SCC executor.

The paper's section 5 is the load-balancing anchor for HG-WCOJ. Algorithm 1
defines the three-phase shape: histogram, count, and materialize. Algorithm 2
defines the HG-WCOJ kernel body over prefix-summed root-key work units. xlog's
HG block-slice implementation should cite section 5 Algorithm 1/2, not section
4, because the load-balanced `unique_keys`, `fan_out`, `prefix_sum`, and
`total` launch parameters are the Algorithm 1/2 scheduling surface.

The paper's section 5 also introduces helper-relation splitting around Figure
3. xlog's helper-split path should cite section 5 Figure 3 or
"section 5 helper-relation splitting", because this is the paper mechanism for
surfacing buried inner-variable skew as a top-level histogram-balanced key.

The paper's section 6 is the stream-mux anchor. It describes phase-aligned
rule parallelism over CUDA streams, where independent rules overlap Count,
Scan/Resize, and Materialize style phases without violating monotonic Datalog
convergence. xlog stream-mux docs should cite section 6, not section 5.

Citation audit checklist:

- Literal paper labels used by the WCOJ bundle: `§5 Algorithm 1/2`, `§5 Figure 3`,
  `§6 stream-mux`, and `§4 semi-naive iteration`.
- HG block-slice: cite arXiv:2604.20073v2 section 5 Algorithm 1/2.
- Helper-split: cite arXiv:2604.20073v2 section 5 Figure 3.
- Stream-mux: cite arXiv:2604.20073v2 section 6.
- Semi-naive merge/update context: section 4 is acceptable, but it is not the
  block-slice citation.
- High-arity motivation for K=7/K=8: section 3 discusses real program-analysis
  rules with six to eight body clauses; use that as motivation, while the code
  still needs normal certification before claiming production support.

## End-to-End Data Flow

The WCOJ path starts after ordinary Datalog lowering has produced relational
IR. The compiler then runs optimizer passes, selectivity and helper passes,
and finally `promote_multiway`. The promoter recognizes certified shapes and
wraps them in a physical-plan node that preserves the original binary tree as
fallback.

The essential invariant is:

```text
execute(MultiWayJoin.dispatch) == execute(MultiWayJoin.fallback)
```

The equality is row-set equality, not necessarily row-order equality. When an
integration test compares outputs it should canonicalize rows unless a specific
sort-label contract is under test.

The runtime then uses the following decision order:

1. Match the first-class RIR node and validate its shape.
2. Check configured gates and cost-model decisions.
3. Resolve relation names to live CUDA buffers.
4. Build or reuse recorded launch-stream layouts and metadata.
5. Launch count, scan, total, and materialize phases.
6. Install the result only after the full provider pipeline succeeds.
7. On any decline or provider error, execute the embedded fallback.

The provider entries are deliberately recorded-launch paths. They must preserve
the allocator's stream-order contract: every temporary buffer touched by a
kernel is tracked against the launch stream before it can be dropped or reused.

## RIR Surface: MultiWayJoin and ChainJoin

`RirNode::MultiWayJoin` is the WCOJ node for triangle, 4-cycle, and K-clique
families. The node carries:

- `inputs`: physical input scans in kernel slot order.
- `slot_vars`: per-slot variable-class ids.
- `output_columns`: head projection used by fallback and default dispatch.
- `fallback`: the IR-equivalent binary-join tree captured before promotion.
- `plan`: optional structured K-clique route, including planned hash routes.
- `var_order`: optional `VariableOrder`, including K-clique variable plans.

Triangle and 4-cycle use the legacy `VariableOrder` form. K-clique uses
`VariableOrder::kclique`, which carries a `KCliqueVariableOrder` with:

- `k`, currently K=5 or K=6 on the production path in this branch.
- `variable_positions`, the chosen binding order.
- `edge_permutation`, the physical edge order.
- `column_swaps`, for oriented edge layouts after permutation.
- `sorted_layout_requirements`, currently focused on the leader edge.
- `helper_split_specs`, planner-produced specs for buried skew.
- `stream_group`, the hook for stream-mux grouping.

Generic walkers must treat `MultiWayJoin` as a structured subtree, not a leaf.
They must walk `inputs` and `fallback` when a transform is semantic, but they
must not mutate the dispatch shape independently from fallback. Recursive
`rewrite_scan_nth` is the canonical example: it must update the same occurrence
in both dispatch inputs and fallback so the delta-substitution occurrence identity is
preserved.

`ChainJoin` is a follow-on integration node added for the chain-promoter route
at commit `41f1447f`. It is not a paper
WCOJ shape. It is a first-class xlog route for hot two-atom chains discovered
by the profiler trace. The production node carries:

- `left` and `right`, normally scans.
- `left_key` and `right_key`.
- `output_columns`.
- `fallback`, again the IR-equivalent binary join.

The architectural reason to use `ChainJoin` rather than overloading
`MultiWayJoin` is auditability. A two-atom chain is not worst-case optimal in
the paper sense, so it needs a separate matcher, dispatch counter, env gate,
and fallback semantics.

## Promoter Surface

`crates/xlog-logic/src/promote.rs` is the main physical promotion boundary.
It is intentionally conservative. A rejected shape stays as ordinary relational
IR and therefore uses the established binary fallback path.

`promote_multiway` runs per SCC and currently checks these families:

- `try_promote_triangle`
- `try_promote_4cycle`
- `try_promote_clique_k` for K=5 and K=6
- `try_promote_chain` on the chain-promoter production branch

Triangle promotion accepts a canonical or normalized three-scan body. It infers
which relation is `e_xy`, `e_yz`, and `e_xz`, builds canonical slot variables,
copies the original projection, and captures the original node as fallback.
If `CompilerConfig::wcoj_variable_ordering` is disabled, the node keeps
`var_order = None` so slice-1 behavior remains bit-identical.

4-cycle promotion performs the same role for `[e_wx, e_xy, e_yz, e_zw]`.
The variable-ordering path is rotation-only for 4-cycle, which avoids the
triangle column-swap cases.

K-clique promotion flattens the body, verifies exactly `C(K, 2)` scan atoms,
derives head-variable classes through union-find, rejects reversed or duplicate
edges, reorders edges to canonical lexicographic order, and then asks the
K-clique planner for a full variable-order plan. A recognized K-clique emits a
positive route:

- `MultiwayPlan::WcojWithPlan` when the cost gate predicts WCOJ should win.
- `MultiwayPlan::PlannedHashRoute` when stats are incomplete or hash is
  predicted to win.

A planned hash route is still a successful recognition. It is not the same as
"the promoter missed the shape". This distinction matters for evidence and
for future cost-model debugging.

The helper-split implementation removed the old always-empty helper spec path. The K-clique promoter
now copies `plan.helper_split_specs` into `KCliqueVariableOrder`. The compiler
then invokes `helper_split_pass::run_kclique_specs`, which can materialize a
`__w37_helper_*` relation and feed that helper relation back into promotion.

`try_promote_chain` recognizes exactly a two-scan inner
join with one join key and wraps it as `ChainJoin`. It should run before
triangle and 4-cycle matching so helper-split output that naturally becomes a
two-atom chain gets the chain route instead of being left as a generic binary
join. It must reject non-inner joins and multi-key joins.

## Dispatch Surface: wcoj_dispatch.rs

`crates/xlog-runtime/src/executor/wcoj_dispatch.rs` owns runtime admission.
The file has two kinds of entry points:

- Rule-keyed wrappers, such as `try_dispatch_wcoj_triangle(rule)`.
- Body-keyed entries, such as `try_dispatch_wcoj_triangle_on_body(body)`.

Body-keyed entries are required for recursive SCCs because the recursive
engine rewrites one scan occurrence to a delta relation per variant. The
rewritten body is no longer the original `CompiledRule` body, so dispatch must
key off the rewritten `RirNode`.

Dispatch is fail-closed. Typical decline cases are:

- Gate disabled.
- Node is not the expected `MultiWayJoin` or `ChainJoin`.
- Inputs are not scans.
- Slot variables or projection do not match the certified shape.
- A referenced relation name or store buffer is missing.
- Scalar widths are mixed or unsupported.
- Runtime or stream cannot be resolved.
- Provider launch returns an error.

For triangle and 4-cycle, dispatch validates width and then builds sorted
layouts through `wcoj_layout_*_recorded`. It optionally rotates leader inputs
using `prepare_leader_inputs` when `var_order` is present. Successful dispatch
records observed selectivity back into `StatsManager` through
`record_wcoj_feedback`.

For K-clique, dispatch validates the K-clique plan, builds oriented and sorted
edge layouts according to `KCliqueVariableOrder`, builds leader-edge metadata,
and calls one of:

- `wcoj_clique5_u32_recorded_planned`
- `wcoj_clique5_u64_recorded_planned`
- `wcoj_clique6_u32_recorded_planned`
- `wcoj_clique6_u64_recorded_planned`

These provider entries build runtime `WcojRelationMetadata` before launch.
The metadata contains `unique_keys`, `fan_out`, `prefix_sum`, and `total`.
Those parameters drive the HG block-slice at the leader edge instead of using
only a static `leader_count`.

For `ChainJoin`, dispatch routes through `try_dispatch_w63_chain_on_body`. The
route order is:

1. Sort-merge if both inputs are sorted and width-eligible.
2. Bounded sort-merge for large sorted one-to-one cells.
3. Nested-loop for threshold-eligible small cells.
4. Hash join otherwise.

The chain dispatcher updates `StatsManager::record_join_result`, increments
`w63_chain_dispatch_count`, and uses `output_columns` for the final projection.

### Aggregate-Fused Group-By-Root Count

Origin: the factorized-hypergraph research plan's aggregate-fusion design;
gate evidence: aggregate-fused WCOJ evidence (6.05x /
5.37x vs materialize+groupby on hub fixtures, 2.49x on small uniform).

`deg(X, count(V)) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` never materializes
the triangle rows:

1. **Promoter** (`try_promote_triangle_inside_aggregate`,
   `xlog-logic/src/promote.rs`): aggregate rules lower to
   `Project{final} -> GroupBy -> Project{group} -> <join tree>`; the
   descent promotes the inner triangle to `MultiWayJoin` and remaps the
   group projection from join-output space into the (X, Y, Z) output space.
2. **Executor hook** (`try_dispatch_wcoj_groupby_root_agg`,
   `executor/wcoj_dispatch.rs`): fires on `GroupBy { Project { MultiWayJoin
   (triangle) }, key_cols == [0], aggs == [(_, Count | Sum | Min | Max)] }`.
   The group key being column 0 = the variable-order root X is the
   soundness condition for one-pass aggregate propagation. For Sum/Min/Max
   the aggregate value column must map to a triangle output variable (Y =
   col 1 or Z = col 2) with plain U32 type, so the kernel can read it
   during traversal (4-byte keys only). Count also admits uniform U64-key
   triangles. Counter: `wcoj_groupby_fusion_dispatch_count`. Kill switch:
   `XLOG_DISABLE_WCOJ_GROUPBY_FUSION=1`. Structural mismatches decline
   silently to materialize+groupby; pipeline errors go through
   `wcoj_decline_on_error` (`XLOG_WCOJ_STRICT` honored).
3. **Provider** (`xlog-cuda/src/provider/wcoj_metadata.rs`):
   * `wcoj_triangle_groupby_root_count_u32_recorded`
     (`wcoj_triangle_groupby_root_count_hg_u32` kernel): the count-phase
     traversal accumulates per-e_xy-row match counts (integer atomicAdd —
     order-insensitive, deterministic values), then compacts count>0 roots
     and reduces per X with the recorded groupby Sum. Output schema
     (X: U32/Symbol, count: U64) matches the unfused baseline.
   * `wcoj_triangle_groupby_root_agg_u32_recorded` (sum/min/max widening;
     `wcoj_triangle_groupby_root_{sum,min,max}_hg_u32` kernels): same
     traversal, but each match also folds its value (Y or Z, selected by
     `WcojRootAggValue`) into a per-row accumulator — u64 atomicAdd for
     sum (a per-row partial can exceed `u32::MAX`), u32 atomicMin/atomicMax
     for min/max (min identity `u32::MAX`, max identity 0). The 3-column
     (X, count, agg) staging buffer is compacted to count>0 and reduced per
     X with the recorded groupby using the same AggOp (the recorded groupby
     Sum was widened to accept U64 value columns via `groupby_sum_u64`).
     Output schema: (X, U64) for sum, (X, U32) for min/max — matching the
     unfused baseline. Bag semantics: every (Y, Z) completion contributes
     its value, exactly like aggregating the materialized projection.
   * `wcoj_triangle_groupby_root_count_u64_recorded` (u64 count widening;
     `wcoj_triangle_groupby_root_count_hg_u64` kernel): u64-key count
     sibling. The recorded groupby is U32/Symbol-key only, so the per-X
     reduction reuses the WCOJ relation metadata (one unique root per
     group, e_xy lex-sorted) plus `wcoj_groupby_root_segment_sum_counts_u32`
     (per-row atomicAdd into per-unique-root u64 totals), then compacts
     totals>0. Output schema (X: U64, count: U64).

   * `wcoj_triangle_groupby_root_agg_u64_recorded` (u64 sum/min/max widening;
     `wcoj_triangle_groupby_root_{sum,min,max}_hg_u64` kernels): u64-key
     sum/min/max sibling. Per-row u64 partials (sum wraps like
     `groupby_sum_u64`; min identity `u64::MAX`, max 0) reduce per unique
     root through the WCOJ relation metadata plus
     `wcoj_groupby_root_segment_{sum,min,max}_values_u64`, then count>0
     groups are compacted. The unfused baseline exists because the legacy
     groupby was widened to u64-value sum/min/max (`groupby_min_u64` /
     `groupby_max_u64`; min/max result schema preserves the value width).
   * `wcoj_4cycle_groupby_root_count_u32_recorded` (4-cycle count widening;
     `wcoj_4cycle_groupby_root_count_hg_u32` kernel): 4-cycle count
     fusion for `deg(W, count(V)) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)`.
     `try_promote_4cycle_inside_aggregate` descends the aggregate wrapper
     (output position 0 = the variable-order root W in both certified
     `output_columns` layouts). Gating decision: the fused path mirrors
     the triangle fusion (default-on behind
     `XLOG_DISABLE_WCOJ_GROUPBY_FUSION`); the opt-in
     `XLOG_USE_WCOJ_4CYCLE*` gates keep governing only the non-aggregate
     materialize dispatch, which is exactly what a declined or
     kill-switched fusion falls back to.
   * `wcoj_4cycle_groupby_root_agg_u32_recorded` (4-cycle sum/min/max widening;
     `wcoj_4cycle_groupby_root_{sum,min,max}_hg_u32` kernels): 4-cycle
     sum/min/max sibling, same per-row accumulator design as the triangle
     agg entry (u64 atomicAdd sum partials, u32 atomicMin/atomicMax with
     `u32::MAX`/0 identities; (W, count, agg) staging compacted to
     count>0, recorded groupby reduce). The aggregate value must be a
     4-cycle output variable the kernel reads during traversal —
     `Wcoj4CycleRootAggValue::{X, Y, Z}`, i.e. output cols 1/2/3 sourced
     from e1.col1 / e2.col1 / e3.col1 (the same columns
     `build_4cycle_head_schema` types the materialized baseline from) —
     with plain U32 type. 4-byte keys only.
   * `wcoj_4cycle_groupby_root_count_u64_recorded` (u64-key 4-cycle count widening;
     `wcoj_4cycle_groupby_root_count_hg_u64` kernel): u64-key 4-cycle
     count sibling, reducing per-row match counts per unique root through
     the WCOJ relation metadata plus
     `wcoj_groupby_root_segment_sum_counts_u32` (the recorded groupby is
     U32/Symbol-key only), then compacting totals>0. Output schema
     (W: U64, count: U64).
   * `wcoj_clique{5,6}_groupby_root_count_u32_recorded_planned` (K-clique count fusion;
     `wcoj_clique{5,6}_groupby_root_count_hg_u32` kernels): K-clique count
     fusion for `q(R, count(*)) :- <complete K_5 / K_6 body>` grouped by
     the plan's root variable, at the 4-byte width-class only. The fused
     kernel reuses the planned clique count traversal
     (`wcoj_clique_template_count_t` + leader metadata row resolution)
     but accumulates each leader-edge row's completion count into a
     per-row counter array via atomicAdd; the row's group key is the
     oriented leader edge's col0 (the kernel's binding[0] root). Unlike
     triangle/4-cycle, the root under `KCliqueVariableOrder` is
     PLAN-DEPENDENT: `variable_order[0]` plus leader-edge
     orientation/column swaps determine the physical root column, so the
     executor branch (`try_dispatch_wcoj_groupby_root_count_clique`)
     fuses only when the GroupBy key column maps to the head variable
     with `variable_positions[r] == 0`, and declines everything else
     silently (non-root keys, K=7/8, u64/mixed widths, planned-hash
     routes). `try_promote_clique_inside_aggregate` descends the
     aggregate wrapper by synthesizing the canonical k-variable head
     projection from the unique topological order of the variable-class
     tournament (first-appearance slot order is unusable — the bushy DP
     plan reorders scan leaves) and remaps the group projection into
     clique head-variable space. Edge orientation + layout is shared
     with the unfused clique dispatch
     (`orient_and_layout_kclique_edges`); the provider entry
     additionally layout-normalizes per dispatch (31b0ccf0 contract).
     Same kill switch and counter as the other fusions.

   Symbol semantics (locked by tests): count over
   Symbol-keyed/valued bodies fuses — Symbol is u32-physical and count
   never reads values — and the output preserves the Symbol key type.
   Sum/min/max over Symbol VALUES is not meaningful data arithmetic: the
   fused hook declines (triangle and 4-cycle alike) and the unfused
   groupby rejects with the same value-type error, so fused and
   kill-switch runs fail identically.

   Recorded-groupby value widths: `groupby_multi_agg_recorded`
   accepts U64 value columns for Sum and Min/Max via
   `groupby_min_u64`/`groupby_max_u64`; the result schema preserves the value
   width under its U32/Symbol-key constraint, matching the legacy
   groupby's u64-value semantics bit for bit.

   All reduction work is O(n_xy) — input-sized, never join-output-sized.

Gate evidence covers the sum/min/max and u64-count widening, 4-cycle count,
u64 sum/min/max, Symbol locks, 4-cycle sum/min/max, u64-key 4-cycle count,
recorded u64-value min/max, and K-clique count fusion (>= 3x vs unfused on the
skewed K=5 hub fixture).

**Recursive-stratum inputs (covered, no code change needed).** A
non-recursive aggregate rule in a later stratum whose triangle body reads
predicates computed by an earlier recursive stratum (e.g.
`deg(X, count(Z)) :- tc(X,Y), tc(Y,Z), q(X,Z)` after a `tc` fixpoint)
dispatches through the same fused path: the promoter's inside-aggregate
descent is shape-only (no recursive-RelId gating), and the recursive
engine's merged relations are produced by `union_gpu`/`diff_gpu`, whose
outputs are lex-sorted + deduped — exactly the layout the fused provider
entry's binary-search work plan assumes (the unfused triangle dispatch
re-sorts per dispatch via `wcoj_layout_*_recorded`; the fused entry relies
on this store invariant instead). Verified empirically and locked by
`crates/xlog-integration/tests/test_wcoj_groupby_fusion_recursive.rs`
(count + sum, mixed tc/tc/q body, all-recursive tc self-join body; fused
counter == 1, kill-switch row parity, host-oracle parity). Aggregates
*inside* recursive rules remain stratification-rejected at compile time —
out of scope by language contract, not by this dispatch surface.

Deferred (stated explicitly): u64-key 4-cycle sum/min/max fusion (the
count path has its u64 sibling; the agg path declines 8-byte keys),
u64-key k-clique count fusion, K=7/K=8 clique count fusion, clique
sum/min/max fusion, LogSumExp/float aggregates.

**Design decision — float/LogSumExp fused aggregates are deferred, not
just unimplemented.** The fused kernels' correctness argument rests
on integer atomics being associative, commutative AND exact: any
interleaving of `atomicAdd`/`atomicMin`/`atomicMax` on integers yields
bit-identical accumulators, which is what lets the per-row partials be
deterministic values under the GPU contract. Float `atomicAdd` is
commutative but NOT associative in IEEE-754 — accumulator bits depend on
warp scheduling, so a float-summing fused kernel would return
run-to-run-different values and break the deterministic-values contract
(the same reason `groupby_multi_agg_recorded` rejects LogSumExp today:
its max → sumexp → final chain needs deterministic intermediate sums).
Candidate designs, in preference order, for whoever picks this up:

1. *Per-block deterministic tree reduction:* keep the work plan's
   fixed-shape block slicing, reduce each block's float partials in a
   shared-memory tree with a fixed combine order, then combine the
   per-block partials in block-index order (one ordered pass, no float
   atomics anywhere). Cost: an extra block_partials buffer sized
   grid * n_roots_touched and a second ordered-combine kernel.
2. *Fixed-point encoding:* scale f64 values into i64/u128 fixed-point,
   reuse the existing exact integer atomics, convert back after the
   recorded reduce. Cost: range/precision analysis per aggregate, and
   LogSumExp still needs its max pass first.

Either way the unfused baseline must come first (the legacy and recorded
groupbys reject float values for sum/min/max today), mirroring how the symbol
and u64 widenings landed: widen the baseline, lock parity, then fuse.

## Cost Model and Variable Ordering

There are two related cost surfaces.

The runtime WCOJ dispatch model lives in
`crates/xlog-runtime/src/executor/wcoj_cost_model.rs`. It builds a
`WcojCostModel` from `RuntimeConfig::resolved_wcoj_cost_model()`. The
cardinality-aware model reads relation stats and estimates the first binary
intermediate through `StatsManager::estimate_join_cardinality`. If the
estimate is large enough, runtime dispatch is allowed to fire.

The compile-time variable-ordering model lives in
`crates/xlog-logic/src/wcoj_var_ordering.rs`. The trait is named
`WcojVariableOrderingModel`. The cardinality leader-selection work introduced `LeaderCardinalityModel`, which
picks a non-default leader only when relation cardinality evidence clears the
configured ratio threshold. The heat-aware ordering work adds `HeatAwareLeaderModel`, which folds in
relation heat and observed selectivity.

`StatsManager` is the shared evidence store:

- Relation cardinality comes from `update_cardinality`.
- Relation heat comes from access recording.
- Join selectivity comes from `record_join_result` or explicit snapshots.
- Snapshots allow compile-time planning to consume runtime evidence without
  directly borrowing the executor.

For K-clique, `plan_kclique_var_order` is the full-variable planner. It uses
cardinality, NDV, selectivity, prefix-degree, key heat, and skew statistics to
produce:

- `variable_order`
- `edge_permutation`
- `variable_share_allocation`
- `cost_prediction`
- `predicted_winner`
- `helper_split_specs`

The K-clique cost gate is one-sided: WCOJ routes only when estimated WCOJ cost
is no greater than the configured hash-chain cost ceiling. Missing or invalid
stats decline to planned hash rather than guessing.

## Recursive Integration

Recursive WCOJ integration lives in
`crates/xlog-runtime/src/executor/recursive.rs`. The core helper is
`execute_wcoj_or_fallback_node`. It tries triangle, 4-cycle, clique5, and
clique6 body-keyed dispatch before falling back to `execute_node`.

The helper is used at two sites:

- Seed pass: every rule is evaluated once against the current store before the
  first delta is computed.
- Per-variant loop: every recursive scan occurrence with a non-empty delta is
  rewritten to that delta relation and evaluated as a variant.

The multi-recursive integration removed the old "recursive scan count greater than one" exclusion.
Multi-recursive bodies can reach a `MultiWayJoin`; the per-variant loop builds
one variant per recursive occurrence and lets dispatch decide whether the
rewritten body is eligible.

The occurrence-identity contract says `rewrite_scan_nth` must rewrite
the selected occurrence and no other occurrence. For `MultiWayJoin`, that means
the dispatch `inputs` and `fallback` must stay semantically aligned. For
`ChainJoin`, the same rule applies to `left`, `right`, and fallback.

The recursive K-clique refresh work extends recursive K-clique handling. After the merge phase
updates a recursive predicate, `refresh_kclique_edge_metadata_after_merge`
records affected K-clique rules so the next K-clique dispatch rebuilds
leader-edge metadata from the current store state. The actual provider
metadata remains launch-local; the refresh counter is the recursive accounting
surface.

## WCOJ Bundle Architecture

### HG block-slice

HG block-slice is the xlog implementation of the paper section 5 Algorithm
1/2 scheduling pattern. The important inputs are:

- `unique_keys`: root-key values.
- `fan_out`: work units owned by each root key.
- `prefix_sum`: prefix over `fan_out`.
- `total`: number of unique root keys in the metadata array.

For triangle and 4-cycle, the code uses shape-specific work-prefix builders.
For K-clique, `wcoj_clique_template_count_hg_grid_t<K, T>` and
`wcoj_clique_template_materialize_hg_grid_t<K, T>` accept the metadata launch
params directly. The kernel computes a block's slice over flattened work, maps
work indexes back to leader keys using `prefix_sum`, and then runs the inner
WCOJ traversal under the chosen variable order.

The count pass writes per-block and per-thread counts. The provider uses
`multiblock_scan_u32_inplace_on_stream` to prefix-sum counts on the launch
stream, then `wcoj_compute_total` to get the final output row count. The
materialize pass reruns traversal and writes to deterministic offsets. This is
why WCOJ output allocation is deterministic and does not require runtime
atomics.

### helper-split

The helper-split pass is the xlog implementation surface for paper section 5
Figure 3. Its purpose is not to binary-factor arbitrary deep joins. Its purpose
is to expose a buried skew variable as a top-level key only when stats show a
large enough heat/skew ratio.

The AST/RIR helper-split pass is in `crates/xlog-logic/src/optimizer.rs` under
`helper_split_pass`. The K-clique-specific integration path is:

1. `plan_kclique_var_order` detects buried skew.
2. The promoter copies `helper_split_specs` into `KCliqueVariableOrder`.
3. The compiler runs `helper_split_pass::run_kclique_specs`.
4. The helper relation is promoted and gets fresh metadata on dispatch.

Uniform distributions must produce no helper relation. Helper/direct row-set
equality is a required regression gate.

### stream-mux

Stream-mux is the stream-aligned multiplexing path anchored in paper
section 6. It is represented in the WCOJ plan surface by `StreamGroupId` and
the `stream_group` field on `KCliqueVariableOrder`.

The current dispatch code already preserves launch-stream discipline through
`wcoj_dispatch_stream_or_init`, `LaunchRecorder`, and recorded provider
entries. Full phase-aligned rule multiplexing is integration-level work:
independent rules in the same monotonic stratum can run count/materialize
phases on separate streams, but every stream must still obey allocator and
drop-safety ordering.

For follow-on integration, pool sizing is explicit. The default planned knob is
`XLOG_WCOJ_POOL_MB_PER_STREAM=256`. The validation contract is a
4-arm by 4-stream worst-case enumeration:

```text
4 arms * 4 streams * 256 MB = 4096 MB of per-stream pool budget
```

The integration gate owns the measured headroom proof, including the planned
3.2 GB headroom record. This guide records the sizing contract so future stream
work does not add unbounded pools or hidden env vars.

## Follow-On Integration Architecture

### chain-promoter

The chain-promoter is an xlog-original performance route motivated by
the profiler trace, not by the WCOJ paper. Its architectural contract
is still the WCOJ-style physical-node contract:

- first-class RIR node (`ChainJoin`);
- captured fallback;
- conservative shape matcher;
- dispatch counter;
- row-set equality against fallback;
- recursive `rewrite_scan_nth` support;
- stats feedback.

This lets chain routing compose with helper-split output while avoiding a
false paper claim.

### K=7/K=8 templates

The K=7/K=8 clique template work uses the existing K-clique planning structures
already define `K_CLIQUE_MAX_K = 8` and `K_CLIQUE_MAX_EDGES = 28`, and
`plan_kclique_var_order` accepts variable counts up to eight. Production
support still requires:

- ABI wrappers for K=7 and K=8.
- Promoter admission for K=7 and K=8 while rejecting K=9.
- Runtime dispatch counters.
- Row equality against hash-join fallback.
- Register-footprint certification at K=8.

Do not claim K=7/K=8 dispatch merely because the generic structs can represent
the plan. The templates close only after their metrics pass.

### CUDA Graphs

The CUDA Graphs work builds on the current recorded-launch discipline, which is graph-ready
in the sense that WCOJ launches are grouped, stream-resolved, and strict about
allocator lifetime. That is not the same as graph capture.

The CUDA Graphs architecture should capture stable Stage-4 hot-loop launch
sequences only after the involved kernels, allocations, and stream fences have
fixed shapes for a replay window. The graph path must preserve the same
fallback behavior: if a shape is not capturable or a graph update fails, the
ordinary recorded-launch path remains authoritative.

Graph capture must not weaken the existing counters. Dispatch counters measure
logical route success, not whether a launch came from a graph replay or an
ordinary kernel call.

### sort-label propagation

Authoritative sort-label propagation matters for WCOJ because
layout and chain routes consume sortedness evidence. A missing sort label must
not silently assert sorted input. A stale sort label can be worse than no label
because it may route to sort-merge or layout fast paths with an invalid
precondition.

The follow-on integration rule is simple: metadata may enable a fast path only when the
producer can prove the ordering after projection, padding, helper materialize,
or fallback execution.

## Operational Invariants

The following invariants are load-bearing:

- `fallback` is always executable and semantically equivalent.
- Dispatch declines silently and safely; it does not mutate the store.
- Successful dispatch writes into the store only through the caller's normal
  installation path.
- Width classes are homogeneous per WCOJ dispatch.
- Mixed `Symbol`/`U32` is only valid through the 4-byte key path when the
  provider explicitly supports it.
- Runtime metadata is launch-local and refreshed from live buffers.
- Recursive variants use body-keyed dispatch after delta rewriting.
- Helper-split is stats-gated; uniform input stays unsplit.
- Cost gates choose planned hash rather than unstructured decline when the
  shape is recognized but WCOJ is predicted not to win.
- Stream-mux and CUDA Graphs are scheduling layers; they cannot change row-set
  semantics.

## Testing and Evidence Guide

Use focused tests for the layer that owns the behavior:

- RIR walker tests for `MultiWayJoin` and `ChainJoin` traversal.
- Promoter tests for accepted and rejected shapes.
- Runtime matcher tests for malformed nodes and route counters.
- Provider tests for kernel row equality and metadata launch parameters.
- Integration tests for fallback parity, recursive variants, helper-split
  composition, and widened-frontier replay.
- Bench tests only after row-set equality and route counters are green.

When a test forces fallback, prefer replacing `MultiWayJoin` with its embedded
`fallback` in a test helper over adding production kill switches. When an env
knob is required by plan, keep it named and scoped; do not add legacy or hidden
fallback knobs.

## Source Map

| Surface | Primary files |
|---|---|
| RIR nodes | `crates/xlog-ir/src/rir.rs` |
| Promoter | `crates/xlog-logic/src/promote.rs` |
| Helper-split | `crates/xlog-logic/src/optimizer.rs` |
| Variable ordering | `crates/xlog-logic/src/wcoj_var_ordering.rs` |
| K-clique full planner | `crates/xlog-logic/src/hypergraph/var_order.rs` |
| Runtime dispatch | `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` |
| Runtime cost model | `crates/xlog-runtime/src/executor/wcoj_cost_model.rs` |
| Recursive SCC integration | `crates/xlog-runtime/src/executor/recursive.rs` |
| Rewrite occurrence identity | `crates/xlog-runtime/src/executor/rewrite.rs` |
| CUDA provider WCOJ entries | `crates/xlog-cuda/src/provider/wcoj.rs` |
| CUDA kernels | `crates/xlog-cuda/kernels/wcoj.cu` |
| Closure board | `docs/v065-closure-board.md` |

## Glossary

`MultiWayJoin`: First-class RIR node for WCOJ-capable multiway shapes.

`ChainJoin`: First-class RIR node for a two-atom chain route.

`slot_vars`: Per-input variable-class ids used to validate a physical WCOJ
shape.

`fallback`: Captured binary plan that must produce the same row set as the
specialized route.

`WcojVariableOrderingModel`: Compile-time leader-selection trait for triangle
and 4-cycle variable ordering.

`KCliqueVariableOrder`: Full K-clique physical plan carried through RIR.

`WcojRelationMetadata`: Runtime leader-edge histogram metadata for HG
block-slice dispatch.

`helper_split_specs`: Planner-produced helper-split requests for buried skew.

`stream-mux`: Phase-aligned CUDA stream multiplexing for independent monotonic
rules.

`CUDA Graphs`: Launch-overhead amortization layer; graph capture is not
equivalent to recorded launch discipline.
