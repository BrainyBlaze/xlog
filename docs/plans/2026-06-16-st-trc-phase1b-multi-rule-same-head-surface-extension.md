# ST-TRC Phase-1b — Multi-rule same-head trainable-rule surface extension (Path B)

**Status:** scoped, implementation pending @human go. Owner: @xlog-claude (training-surface lane). Engine prerequisite for the faithful ST-TRC joint soft-mixture, parallel to @dts-dlm-main's top-K candidate preservation (the Phase-1b mask-surface prereq).

## Problem (found by the Phase-1a capability pre-check)

The trainable-rule surface restricts a query head to **exactly one defining rule**:

```
ValueError: Query predicate 'target' has 3 defining rules; expected exactly 1 matching rule
```

Source: `CompiledProgram::find_query_rule` (`crates/pyxlog/src/neural.rs:2290`) errors when `matches > 1`. `get_or_build_query_signature` builds the circuit template from a **single** rule's body (its neural groups, guard, hard filters), so N same-head candidate rules cannot be represented in one `QuerySignature`.

The faithful ST-TRC Phase-1a/b mechanism is exactly this topology: **N candidate rules deriving one head** (e.g. `belnap_both(F) :- belnap_supported(F), belnap_refuted(F)` as the correct candidate vs same-head distractors), each soft-gated by its own `nsr_guard`, the head's valuation an OR-amalgamation over candidates, ST-Gumbel selecting the winner. The current surface cannot express the joint mixture.

## Verified workaround (Path A — runs on the existing surface today)

Train **each candidate as its own single-rule program** against the same engine-labelled supervision and rank by loss/guard. Capability toy (positives `{0,2}` = `supp ∩ refut`), default Adam, 300 steps:

| candidate | final loss | guard weight (sigmoid) |
|---|---|---|
| **correct** `supp(C), refut(C)` | **0.0018** | **0.9963** |
| distractor `only_a(C)` | 22.35 | 0.333 |
| distractor `only_b(C)` | 22.35 | 0.333 |

Path A is a **per-candidate necessary-condition** check (does the correct join fit the supervision where distractors can't?), not the joint competition. It is sufficient for Phase-1a-lite signal-convergence and needs **zero** surface work. Path B below is the faithful joint soft-mixture.

## The extension (Path B)

1. **Signature:** generalize `find_query_rule` → `find_query_rules` returning *all* N rules for `(predicate, arity)`; build a **union signature** = `Vec<QuerySignature>` (one sub-signature per rule, each carrying its own groups + guard + hard_filters). No longer an error on `matches > 1`.
2. **Circuit:** the GPU engine already OR-amalgamates multi-rule predicates natively (standard Datalog → D4/XGCF). The trainable layer compiles/caches the N per-rule templates (or one combined circuit) and threads each rule's guard.
3. **Forward:** evaluate each candidate's gated contribution (`guard_k × template_k valuation`), OR-combine (prob-sum / max amalgamation, as the circuit already does), BCE on the head valuation.
4. **Backward:** gradient routes to each candidate's guard (and any nn) through the **existing** XGCF log-space autodiff (the P2.1 / finite-diff-verified custom-autograd path) — unchanged; only the forward gains the union/OR-amalgamation.
5. **ST-Gumbel (Phase-1b):** replace the N independent sigmoid guards with a learnable score + temperature-annealed Gumbel-softmax over the candidate set (the torch-side `(i)` piece, shared seam with @dts-dlm-claude's temperature schedule). The OR-amalgamated forward from (3) is the engine substrate it gates.

## Reuse vs net-new

- **Reuse:** the engine's native multi-rule OR-amalgamation (D4/XGCF); the per-guard XGCF autodiff (P2.1); `forward_backward_grouped` batching; the guard module (`nsr_guard` = the single-candidate special case).
- **Net-new (this slice):** `find_query_rules` / union-signature builder; the N-template OR-amalgamated forward in the complex-query path (`forward_backward_complex_tensor` / `_batch_`); per-candidate guard gradient bookkeeping across the union.

## Open question (separate small check)

**Trainable-body negation.** The belnap near-miss distractor `belnap_true(F) :- belnap_supported(F), NOT belnap_refuted(F)` carries a negated literal. Whether the trainable surface admits a `NOT` literal in a candidate body is **not yet verified** — Phase-1a uses positive distractors first; this is a separate small surface check before the negated near-miss is admitted.

## Sequence

`@dts-dlm-main top-K (landed)` → `@dts-dlm-claude Phase-1a-lite harness on Path A (in flight)` → **this slice (Path B), implement on @human go** → `@xlog-claude-2 fused candidate-join super-step` → ST-Gumbel mask (shared torch seam). Phase-1a-lite separation result is confirmatory; it does not change this surface ask.
