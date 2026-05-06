# W3.2 Evidence — General-Arity WCOJ Kernel Template (k = 5 and k = 6)

**Closes board item: W3.2 only.**
**Date:** 2026-05-06.
**Branch:** `feat/w32-general-arity-wcoj-template`.
**Base:** `main` at `d5073bdb` (W3.1 closure commit).
**Plan:** `docs/plans/2026-05-06-w32-general-arity-wcoj-template-plan.md`
(approved iteration 4).
**Head:** branch tip — `git log d5073bdb..HEAD --oneline` lists
the full commit sequence.

## Summary

`crates/xlog-cuda/kernels/wcoj.cu` gains a single C++ template
covering K-clique enumeration at K ∈ {5, 6}. Eight `extern "C"`
ABI wrappers (k=5/k=6 × count/materialize × u32/u64) call into
the shared template; **k=6 wrapper bodies contain only template
calls — no hand-written algorithm**. Tier-1 + Tier-2 source-audit
certs lock this contract literally.

`crates/xlog-cuda/src/provider/wcoj.rs` gains four public
methods on `CudaKernelProvider`:
* `wcoj_clique5_u32_recorded(edges: &[&CudaBuffer; 10], stream)`
* `wcoj_clique5_u64_recorded(edges: &[&CudaBuffer; 10], stream)`
* `wcoj_clique6_u32_recorded(edges: &[&CudaBuffer; 15], stream)`
* `wcoj_clique6_u64_recorded(edges: &[&CudaBuffer; 15], stream)`

All four delegate to one generic `wcoj_clique_recorded_inner`
helper. Width-class (4-byte = U32 + Symbol mixable; 8-byte = U64)
and K (5 or 6) drive kernel-name selection; otherwise the
orchestration is identical: validate → upload edge-pointer
arrays → count → device scan → total → materialize → output.

`crates/xlog-logic/src/promote.rs` gains `try_promote_clique_k`
for k ∈ {5, 6}. Tree-flatten + complete-K_k validation. Robust
to left-deep / right-deep / bushy lowered trees. Rejects:
filter wrappers, reversed atoms, self-edges, constants in atom
positions, recursive scan bodies, and atom multisets that don't
form the complete K_k edge set.

`crates/xlog-runtime/src/executor/wcoj_dispatch.rs` gains
public counters + dispatcher entries. The runtime layout-sorts
each edge through W3.1's `wcoj_layout_sort_*_recorded`
unconditionally before invoking the provider; silent fallback
to `MultiWayJoin.fallback` on dispatcher decline / kernel error.
**No** force / kill / adaptive knobs (per W3.2 D8 lock).

The existing slice 1 / slice 2 (triangle / 4-cycle) +
slice 4 (linear-recursive) + W3.1 (sort accessors) hot paths
are **bit-identical pre-W3.2** — the W3.2 promoter and
dispatcher only fire on the new shapes (10 / 15 atoms).

## Acceptance Properties (33 tests)

| Part | # tests | Location | What it locks |
|------|---------|----------|---------------|
| Provider × width-class — k=5 | 3 | `crates/xlog-cuda/tests/test_wcoj_clique5.rs` | u32 / u64 / Symbol round-trip vs. `cpu_clique_reference<T, 5>` brute-force oracle. |
| Provider × width-class — k=6 | 3 | `crates/xlog-cuda/tests/test_wcoj_clique6.rs` | Same shape at K=6. |
| Runtime dispatch — counter advance | 2 | `crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs` | k=5 + k=6 K-clique rule executes via the new dispatcher; `executor.wcoj_clique{5,6}_dispatch_count() >= 1` AND dispatch row set equals the row set produced by a `replace_multiway_with_fallback` rewrite of the same plan (test-only RIR helper that substitutes `MultiWayJoin` nodes with their `fallback` field — no new force/kill/adaptive knobs). |
| Runtime dispatch — fallback cert | 2 | same file | k=5 + k=6 dispatcher decline engineered by uploading ONE edge with `ScalarType::I64` schema (outside both 4-byte and 8-byte width-classes); promoter still emits `MultiWayJoin`, layout-sort rejects the I64 column, dispatcher returns `Ok(None)`. Asserts: (a) compiled plan contains a `MultiWayJoin` with `inputs.len() == C(k, 2)` (catches promotion regression); (b) counter == 0 observably; (c) decline row set equals the `MultiWayJoin.fallback` reference row set under the same malformed-schema fixture (both empty under the type-mismatch). |
| Promoter positive shape | 6 | `crates/xlog-logic/tests/test_w32_clique_promoter.rs` | left-deep / right-deep / bushy × k=5/k=6 all promote correctly. Tree-flatten + complete-K_k validation accepts all shapes. |
| Promoter negative shape | 8 | same file | missing-edge (9 atoms), self-edge, cycle-5 (pentagon), disconnected, filter-wrapper, reversed-atom (broken keys), filter-wrapped (named cell), linear-recursive (recursive-scan-count > 0). All 8 must NOT promote. |
| Promoter k=7 unsupported sentinel | 1 | same file | 21-atom K_7 body rejects (W3.2 only handles k ∈ {5, 6}). |
| k=6 source-audit Tier 1 | 4 | `crates/xlog-cuda/tests/test_w32_kernel_source_audit.rs` | wrapper bodies template-call-only (u32/u64 × count/materialize): exact statement count + template-call presence + no for/while/do/switch. |
| k=6 source-audit Tier 2 | 4 | same file | file-wide: no `template <>` specialization for `<5>` / `<6>`; no `if constexpr (K == 6)` / `K == 5` branch; no `clique6` helper body outside the 4 ABI wrappers; no isolated `5` / `6` literal in the template body (static-assert + template-default contexts whitelisted). |

**W3.2 acceptance total: 33 tests, 33/33 PASS.**

## Cert Test Results

```
cargo test -p xlog-cuda --release --test test_wcoj_clique5
running 3 tests
test clique5_u32_round_trips_against_cpu_oracle ... ok
test clique5_symbol_round_trips_against_cpu_oracle ... ok
test clique5_u64_round_trips_against_cpu_oracle ... ok
test result: ok. 3 passed; 0 failed; 0 ignored

cargo test -p xlog-cuda --release --test test_wcoj_clique6
running 3 tests
test clique6_u32_round_trips_against_cpu_oracle ... ok
test clique6_symbol_round_trips_against_cpu_oracle ... ok
test clique6_u64_round_trips_against_cpu_oracle ... ok
test result: ok. 3 passed; 0 failed; 0 ignored

cargo test -p xlog-cuda --release --test test_w32_kernel_source_audit
running 8 tests
test k6_count_u32_wrapper_is_template_call_only ... ok
test k6_count_u64_wrapper_is_template_call_only ... ok
test k6_materialize_u32_wrapper_is_template_call_only ... ok
test k6_materialize_u64_wrapper_is_template_call_only ... ok
test no_explicit_k6_template_specialization ... ok
test no_if_constexpr_k_equals_6_branch ... ok
test no_clique6_helper_function_body ... ok
test no_six_literal_in_template_body ... ok
test result: ok. 8 passed; 0 failed; 0 ignored

cargo test -p xlog-logic --release --test test_w32_clique_promoter
running 15 tests
test clique5_left_deep_promotes ... ok
test clique5_right_deep_promotes ... ok
test clique5_bushy_promotes ... ok
test clique6_left_deep_promotes ... ok
test clique6_right_deep_promotes ... ok
test clique6_bushy_promotes ... ok
test non_clique_5_atoms_with_missing_edge_does_not_promote ... ok
test clique5_with_self_edge_rejected ... ok
test cycle_5_does_not_promote ... ok
test disconnected_subcomponents_do_not_promote ... ok
test clique_with_constant_in_atom_does_not_promote ... ok
test clique5_with_reversed_atom_rejected ... ok
test clique5_with_filter_wrapper_rejected ... ok
test linear_recursive_clique5_does_not_promote ... ok
test clique7_does_not_promote ... ok
test result: ok. 15 passed; 0 failed; 0 ignored

cargo test -p xlog-integration --release --test test_wcoj_clique_dispatch
running 4 tests
test clique5_dispatch_counter_advances_and_row_set_matches_fallback_body ... ok
test clique6_dispatch_counter_advances_and_row_set_matches_fallback_body ... ok
test clique5_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback ... ok
test clique6_dispatcher_decline_does_not_advance_counter_and_row_set_matches_fallback ... ok
test result: ok. 4 passed; 0 failed; 0 ignored
```

**Total: 6 + 4 + 15 + 8 = 33 new W3.2 tests, 33/33 PASS.**

## Workspace Tally

| Suite | PASS | FAIL | IGN | Pre-W3.2 baseline |
|-------|------|------|-----|-------------------|
| Workspace tests (default features, lib + integration) — `cargo test --workspace --release --exclude pyxlog --exclude xlog-cuda-tests` | **1990** | 0 | 17 | 1957 |
| W3.2 provider × width-class (k=5 + k=6) | 6 | 0 | 0 | (new) |
| W3.2 runtime dispatch + decline | 4 | 0 | 0 | (new) |
| W3.2 promoter shape | 15 | 0 | 0 | (new) |
| W3.2 source-audit Tier 1 + Tier 2 | 8 | 0 | 0 | (new) |
| W3.1 priors (layout u32 / u64 / sort u32 / u64 / roundtrip) | 9 + 6 + 5 + 5 + 72 | 0 | 0 | unchanged |
| W2.1 cert | 11 | 0 | 0 | unchanged |
| W2.3 trace cert (`recursive-stats-trace`) | 10 | 0 | 0 | unchanged |
| W2.4 cert | 3 | 0 | 0 | unchanged |
| W2.6 cert | 7 | 0 | 0 | unchanged |
| Slice-4 cert | 6 | 0 | 0 | unchanged |
| CUDA certification suite (`run_full_certification` meta-cert, 206 sub-tests) | 1 | 0 | 0 | unchanged |
| `cargo fmt --check --all` | clean | — | — | clean |

**Workspace pass-count delta: +33** (1957 → 1990) — exactly
matches the locked plan's `+33` acceptance grid.

## Code-Level Changes

| File | Change |
|------|--------|
| `crates/xlog-cuda/kernels/wcoj.cu` | Templated K-clique kernel: per-thread `wcoj_clique_template_count_t<K, T>` + `wcoj_clique_template_emit_t<K, T>` + `clique_recurse_t<K, Level, T, Out>` (level-keyed `if constexpr (Level >= K_VAL)` recursion). Grid-level templates `wcoj_clique_template_count_grid_t<K, T>` + `wcoj_clique_template_materialize_grid_t<K, T>` absorb thread-idx + bounds checks. 8 `extern "C" __global__` ABI wrappers (k=5/k=6 × count/materialize × u32/u64), **each body is exactly ONE statement that calls the grid-level template** — no conditionals, no loops, no thread-idx computation in the wrapper itself. K=6 wrappers call `<6, T>` instantiation, K=5 calls `<5, T>`; algorithm body is K-uniform with no K-specific code anywhere in the source. |
| `crates/xlog-cuda/src/kernel_manifest_data.rs` | +8 manifest entries for the new kernel symbols. |
| `crates/xlog-cuda/src/provider/mod.rs` | +8 `pub const` kernel-name strings in `wcoj_kernels`. |
| `crates/xlog-cuda/src/provider/wcoj.rs` | +549 lines: 4 public `wcoj_clique{5,6}_{u32,u64}_recorded` methods + 1 generic `wcoj_clique_recorded_inner` helper handling validation, edge-pointer-array upload, count/scan/total phase, materialize phase. |
| `crates/xlog-logic/src/promote.rs` | +307 lines: `try_promote_clique_k(body, k)` for k ∈ {5, 6}. Tree-flatten + complete-K_k UF validation; rejects filter/reversed/self-edge/constant/recursive bodies. Wired into `promote_multiway` dispatch chain after triangle / 4-cycle, gated on `recursive_scan_count == 0`. |
| `crates/xlog-runtime/src/executor/mod.rs` | +4 lines: 2 new `pub(super) u64` counter fields on `Executor` initialized to 0. |
| `crates/xlog-runtime/src/executor/recursive.rs` | Clique5 + clique6 dispatch entries wired into the **non-recursive SCC** dispatch chain only. The recursive WCOJ helper (`execute_wcoj_or_fallback_node`) is **NOT extended** for clique-keyed dispatch (per W3.2 plan §177); recursive clique bodies are rejected at the promoter level by the `recursive_scan_count == 0` gate in `promote_multiway`. |
| `crates/xlog-runtime/src/executor/wcoj_dispatch.rs` | +185 lines: public counter accessors + 4 `try_dispatch_wcoj_clique{5,6}_{,_on_body}` entries delegating to one generic `try_dispatch_wcoj_clique_k_on_body(body, k)` that does shape match → relation resolve → W3.1 layout-sort each edge → provider call → counter increment. Silent fallback on decline. |
| `crates/xlog-cuda/tests/test_wcoj_clique5.rs` | NEW. 3 tests + shared CPU oracle + fixtures. |
| `crates/xlog-cuda/tests/test_wcoj_clique6.rs` | NEW. 3 tests (k=6 mirror). |
| `crates/xlog-cuda/tests/test_w32_kernel_source_audit.rs` | NEW. 8 tests (Tier 1 + Tier 2). |
| `crates/xlog-logic/tests/test_w32_clique_promoter.rs` | NEW. 15 tests (6 positive + 8 negative + 1 k=7 sentinel). |
| `crates/xlog-integration/tests/test_wcoj_clique_dispatch.rs` | NEW. 4 tests (counter advance × 2 + dispatcher decline × 2). |

**No changes** to:
* CUDA `.cu` source for triangle / 4-cycle / sort / dedup / project / ... — only `wcoj.cu` is appended to.
* W3.1's `wcoj_layout_sort_u32_recorded` / `_u64_recorded` (consumed unchanged by the new dispatcher).
* W3.1's `wcoj_layout_u32_recorded` / `_u64_recorded` arity-2 hot path.
* The slice 1 / slice 2 (triangle / 4-cycle) provider entries or kernels.

## Decision Mapping (D1–D8 from approved plan iteration 4)

| Decision | Locked answer | Implementation |
|----------|---------------|----------------|
| D1 | C++/CUDA `template <int K>`. k=6 = ABI wrapper + explicit instantiation; no hand-written body. | `wcoj_clique_template_count_t<K_VAL, T>` + `wcoj_clique_template_emit_t<K_VAL, T>` in `wcoj.cu`. K=6 wrappers call `<6, T>(...)`; bodies are template-call-only (Tier-1 audit). |
| D2 | Test-only `cpu_clique_reference<T, const K>` brute-force oracle, runtime length-assert, no const-generic length expressions. | `crates/xlog-cuda/tests/test_wcoj_clique5.rs:cpu_clique_reference<T, K>` with first-line `assert_eq!(edges.len(), K * (K - 1) / 2)`. |
| D3 | u32 + u64 + Symbol; Symbol gets its own cert at both k. | 6 provider tests = 3 (u32/u64/Symbol) × 2 (k=5/k=6). |
| D4 | Runtime integration required. Default-dispatch on shape match; silent fallback. No force/kill/adaptive knobs. | `try_dispatch_wcoj_clique_k_on_body` in `wcoj_dispatch.rs`. Silent fallback on decline (counter doesn't advance). 4 runtime certs cover counter-advance + decline. |
| D5 | Tree-flatten + complete-K_k validation. Robust to left-deep / right-deep / bushy. Rejects filter/reversed/self-edge/recursive. | `flatten_clique_body` + `try_promote_clique_k` in `promote.rs`. 15 promoter certs cover positive + negative shape contracts. |
| D6 | Lex `(i, j)` for `i < j` canonical edge order. | `clique_edge_idx(i, j, k) = i * (k - 1) - i * (i - 1) / 2 + (j - i - 1)`. Promoter emits inputs in this order; provider upload preserves. |
| D7 | 33 tests across all 5 cert dimensions. No arity-7 positive coverage. | Acceptance grid above. |
| D8 | No skew classifier / env knobs / W3.3+ work. No push, no tag, no self-mark DONE. Branch from `d5073bdb`; plan as branch commit #1. | Implemented as locked. |

## K=6 Compile-Budget Gate (Step 12 hard stop)

The plan's Step 12 first bullet pinned `cargo build -p xlog-cuda
--release` and the k=6 provider cert as the hard-stop on the
template strategy. **Both passed cleanly:**

* `cargo build -p xlog-cuda --release` — 27.34s clean (initial
  compile after appending W3.2 template + 8 wrappers).
* PTX inspection: all 8 `wcoj_clique{5,6}_*` entries present in
  the compiled PTX (`grep ".entry wcoj_clique"` returned 8).
* No register-pressure failure or compile-time blow-up at K=6.
* `test_wcoj_clique6` runs cleanly — k=6 kernel produces correct
  CPU-oracle row sets at u32 / u64 / Symbol.

The template strategy survives the K=6 budget without
modification — locked.

## Process Rule Compliance

* Process rule #1: this slice does **not** self-mark W3.2 DONE.
* Process rule #2: every commit references W3.2.
* Process rule #3: plan opens with "Closes W3.2 only."
* Process rule #5: no release-train references; no scope-punting
  wording — out-of-scope items (k=7 support, recursive clique
  helper, skew classifier for cliques) are rejected in W3.2 with
  no closure credit.
* Process rule #6: no push, no tag.

## Closure Board Update Proposal

After explicit user "mark W3.2 DONE" approval, a follow-up
commit applies:

* `docs/v065-closure-board.md` — W3.2 status `OPEN → DONE`,
  status tally `DONE: 6 → 7; OPEN: 13 → 12` (verify counts at
  apply-time).
* `docs/v065-closure-board.md` "Completed" section gets a W3.2
  entry referencing the branch commits (full sequence via
  `git log d5073bdb..HEAD --oneline`).
* FF-merge `feat/w32-general-arity-wcoj-template` into local
  `main`. No tag, no push.
