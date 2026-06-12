# D3 — Factorized Recursive Deltas: S3 Spike Design

Date: 2026-06-12
Branch: `feat/d3-factorized-delta` (worktree `.worktrees/d3-factorized-delta`, from main `d0c1da7c`)
Upstream direction: `docs/plans/2026-06-11-factorized-hypergraph-research.md` §4 D3, §5 S3
Prerequisites satisfied: D1 (aggregate-fused WCOJ, merged) and D2 (GPU Free Join, merged
`d0c1da7c`) — the research report parked D3 "until D1+D2 evidence exists".

## 1. Gate (verbatim from the research report, §5 S3)

| Spike | Workload | Gate |
|---|---|---|
| S3 factorized delta | transitive closure on dense block graph (delta blowup) | **≥5× peak-memory reduction** at **wall-clock ≤1.2×**; **deterministic row-set parity** with current engine |

Per the perf discipline: spike before any plan; a failed gate parks the direction with
evidence. Per the RunPod rule: the memory/wall-clock measurement runs on a minimal RunPod
instance (per-run authorization); parity tests run locally.

## 2. Current engine memory anatomy (measured surface, per fixpoint iteration)

Production path for TC (`path(X,Z) :- path(X,Y), edge(Y,Z)` + base rule) is
`execute_recursive_scc` (`crates/xlog-runtime/src/executor/recursive.rs`):

1. Variant body (Scan swapped to delta) → `execute_wcoj_or_fallback_node` → 2-atom body
   falls back to the binary walker → **`hash_join_v2`** materializes the raw join: one flat
   row per **derivation witness** (x,y,z), then projection to (x,z). Raw size =
   Σ_x Σ_{y∈δ[x]} deg_edge(y) — the C1/C3 blowup term.
2. `diff_gpu(delta_raw, full)` → `diff_via_deterministic_set`:
   `dedup_full_row_deterministic(raw)` (sorts the entire raw buffer; sort scratch),
   `dedup_full_row_deterministic(R)` (**re-sorts the entire stable relation every
   iteration**), typed sorted diff-probe kernel + mask compaction.
3. `union_gpu(full, delta_new)`: concat (|R|+|δ|) + full sort + dedup → fresh R copy.

Peak ≈ |raw| × (1 + sort-scratch factor) + 2–3 × |R|. On a delta-blowup workload |raw|
dominates: almost all raw rows are either **rediscoveries** (already in R) or **duplicate
witnesses** of the same novel tuple, and the engine materializes all of them before diff
throws them away. (Note: `diff_gpu`'s doc comment says "host set fallback" — stale;
`diff_via_deterministic_set` is fully GPU. No host-transfer claim is part of this spike.)

## 3. D3 thesis applied (factorized delta, minimum viable)

Keep the **delta candidates as a d-representation** and flatten only at the novel
boundary (output-linear, per the verified constant-delay enumeration result):

- `edge` is layout-normalized once, outside the loop, via `wcoj_layout_u32_recorded`
  (lex-sorted, full-row-deduped (y,z)) — the D2 flat sorted-range trie. For a frontier row
  (x,y), the candidate z-set is the trie range `edge[y] = [lo,hi)` — a product node
  (x × range), never expanded.
- The per-x novel set is `(∪_{y∈δ[x]} edge[y]) \ R[x]` — a union of trie ranges minus a
  sorted segment. The spike evaluates this union–diff over a **dense-domain characteristic
  bitvector**: one n-bit row per source x (n = node-domain size), n²/8 bytes total —
  **64× smaller than flat R** (8 bytes/pair vs 1/8 byte/pair) and independent of witness
  multiplicity. Justification: the gate workload is *dense* TC (R approaches n²), exactly
  where a dense characteristic vector is the canonical d-rep evaluation; sparse-domain
  generalization is a Phase B decision (§8), not silently claimed.
- Fused mark → subtract → emit (three kernels, no atomically-grown buffers, recorded-launch
  compatible two-phase sizing):
  1. **mark**: grid-stride over candidate work items (delta row i, offset k) — work prefix
     from range lengths, same pattern as `fj_expand_work_prefix_u32` — `atomicOr` bit
     (x, z) into the bitmap. Rediscovery and witness duplicates collapse in the bitmap;
     nothing is materialized.
  2. **subtract**: grid-stride over R rows, clear bit (x,z). After this pass the bitmap is
     exactly the novel set. (Clearing non-candidate bits is a no-op.)
  3. **emit**: popcount per 32-bit word → `exclusive_scan_u32_inplace` → emit (x,z) pairs
     at scanned offsets. Output is **lex-sorted and deduped by construction** (word order =
     (x, z) order) — it is simultaneously `delta_new` for the next iteration (no sort, no
     dedup, no diff) and the union input.
- R stays a flat lex-sorted (x,z) buffer (canonical form preserved by `union_gpu`, which
  returns sorted+deduped). Stable-relation factorization and merge-path union are explicitly
  **out of spike scope** (§8) — both engines pay the same `union_gpu` cost, isolating the
  measured difference to the delta pipeline (join+diff+dedup), which is where the research
  report located C1+C3.

Peak for the spike loop ≈ |R| terms (shared with baseline via `union_gpu`) + n²/8 bitmap +
|novel| emit — the |raw| term and both per-iteration full sorts are gone.

Per-iteration transfers: one `dtoh_scalar_untracked` u32 read of the novel count (fixpoint
termination test) — metadata-only, consistent with the zero-host data-plane contract
(bounded `num_rows` reads are exempt by design).

## 4. Spike scope and deliberate non-goals

In scope (u32/Symbol width only):
- `GpuMemoryManager` peak-bytes high-water mark (`fetch_max` at the two reservation
  funnels: `alloc`, `alloc_raw`; `peak_bytes()` + `reset_peak()`); needed to *measure* the
  gate on both engines. Additive, zero behavioral change.
- 3 CUDA kernels in `wcoj.cu` (`fj_delta_*`), manifest registration, name consts.
- Provider entry `fj_delta_novel_u32_recorded(delta, edge_norm, full_r, domain, stream)`
  in a new `provider/fj_delta.rs`, plus a thin TC driver loop in the spike test.
- Domain bound: node ids must satisfy `max_id < domain ≤ 2^26` (bitmap ≤ 512 MB at the
  extreme; gate fixtures use ≤ 2^13). Checked, fail-closed with a typed error.
- Parity tests (local): CPU-oracle TC on irregular fixtures (cycles, diamonds, skewed
  degrees, non-block structure — the bitmap path is shape-agnostic) + cross-parity vs the
  production executor row set.
- Gate measurement (RunPod): baseline = the **actual production executor** running the TC
  program (strongest evidence; `xlog-cuda-tests` already drives the executor for the cert
  suite) vs the spike loop, both instrumented via `reset_peak`/`peak_bytes`, LP-MULTI-RUN
  style: isolated serial runs, 3 runs × median-of-3 reps, clocks/temp recorded.

Non-goals (explicitly deferred, not debt — each is gated future work listed in §8):
- u64 width, sparse-domain novel evaluation, factorized stable relation, merge-path union,
  production promoter/dispatch integration, multi-predicate SCCs, stratified negation
  interaction. The spike proves or refutes the memory physics; Phase B happens only on a
  PASS gate with its own plan.

## 5. Gate fixture — dense block graph

Deterministic block-cycle digraph: k blocks of b nodes (n = k·b), complete bipartite
edges B_i → B_{(i+1) mod k} (|E| = n·b). TC = n² pairs, converging in k iterations with a
final all-rediscovery iteration (cycle wrap). Properties targeted by the gate phrase
"delta blowup":
- every novel pair has exactly b duplicate witnesses (raw = b × novel each iteration);
- the final iteration is 100% rediscovery (raw = n·b², novel = 0);
- |raw| / |R| = b²/n = b/k — choose b ≫ k so the delta term dominates peak.

Gate scale (primary): k=4, b=256 → n=1024, |E|=262,144, R=1,048,576 pairs (8 MB flat;
bitmap 128 KB), per-iteration raw = n·b² = 67.1M rows (~0.5–1.5 GB through
hash_join+diff on the baseline). Secondary scale: k=4, b=384 (n=1536, raw=226M) to show
the ratio trend. Both fit a 16 GB pod with headroom; work items stay far below the u32
work-index budget. Local parity scale: k=3, b=8 plus irregular non-block fixtures.

Expected (to be *measured*, not assumed): baseline peak carries the ~0.5–1.5 GB raw+sort
terms; spike peak carries ~10–40 MB (R terms + bitmap + novel). Headroom over 5× is large;
wall-clock is the real risk (§7).

## 6. Implementation plan (Phase A)

1. `feat(cuda): peak-bytes high-water mark on GpuMemoryManager` — counter + accessors + unit test.
2. `feat(cuda): fj_delta bitmap novel-set kernels` — `fj_delta_mark_u32`,
   `fj_delta_subtract_u32`, `fj_delta_emit_u32` (+ popcount/scan plumbing), manifest
   registration, provider entry with validation (sorted-delta precondition via layout
   normalization, domain bound, schema checks).
3. `test(cuda-tests): factorized delta spike parity + gate bench` —
   `test_factorized_delta_spike.rs`: oracle parity (irregular + block fixtures),
   executor cross-parity, `#[ignore]` gate measurement test
   `s3_measurement_factorized_delta_gate` (baseline executor loop vs spike loop).
4. Local: full parity suite + targeted regressions (`set_ops_tests`, `device_row_counts`,
   memory unit tests). No local perf measurement (standing rule).
5. RunPod (user authorization required): minimal instance, run gate, pull log, delete pod,
   confirm deletion. Evidence under `docs/evidence/2026-06-12-s3-factorized-delta/`.
6. Gate decision recorded in this doc + memory. PASS → propose Phase B plan; FAIL → park
   with evidence, branch preserved unmerged.

## 7. Risks

- **Wall-clock ≤1.2×**: the spike replaces a sort-heavy pipeline with linear passes, but
  `atomicOr` contention on hot bitmap words (complete-bipartite blocks hit the same z-words
  from many threads) and the per-iteration |R| subtract walk could bite. Mitigations if
  measured hot: per-block staging of bitmap words in shared memory; subtract only the R
  x-segments active in the frontier (lower_bound on delta's x set).
- **Baseline fairness**: baseline is the unmodified production executor — no strawman. The
  spike shares `union_gpu` so the delta in peak is attributable to the factorized pipeline.
- **u32 work-index budget**: mark-phase work items = raw candidate count (67M / 226M at the
  two scales) — within u32; checked fail-closed like the FJ frontier budget.
- **Bitmap domain**: dense-domain assumption is honest for the gate workload but not
  general; Phase B must add a stats-gated route (extend the W2.5 cardinality cost-model
  discipline, not a new oracle) and a sparse strategy before any production claim.
- **SRDatalog P1–P5 alignment**: the spike changes the *representation* of the delta
  pipeline, not semi-naive semantics (same delta_d sets, same fixpoint); no claim conflicts
  with the paper's recursion contract. Re-verify at Phase B design.

## 8. Deferred (gated future work, recorded so nothing is silently dropped)

u64/width-parameterized kernels (FjWidth pattern exists); sparse-domain novel evaluation
(sort-based emit with within-iteration witness dups, or per-x k-way range merge);
factorized stable relation (d-rep R with canonical-form maintenance under union — the open
research question; bitmap spike deliberately keeps R flat); merge-path union of sorted
R ∪ novel (W4.3 sort-merge kernel assets are the natural substrate); production
integration (promoter recognition of linear recursive rules, kill switch
`XLOG_DISABLE_FACTORIZED_DELTA`, dispatch counter, epistemic preflight bucket per the D2
`free_join_route_count` precedent); multi-predicate SCC and mutual recursion; negation
strata interaction.

## 9. Gate evidence

(to be filled after the RunPod run)
