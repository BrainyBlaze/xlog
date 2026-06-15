# Neuro-Symbolic Mixed Bodies Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow a `trainable_rule` body to mix `nn/4` neural predicates with ordinary (EDB/derived) relations, so the differentiable joint-training path stops fail-closing on the most common neuro-symbolic pattern (a learned neural score *gated by* symbolic facts).

**Architecture:** The neural training path compiles a single placeholder-grounded d-DNNF/XGCF circuit once and hot-swaps per-example neural leaf weights (`crates/pyxlog/src/neural.rs` → `crates/xlog-prob/src/exact.rs`). We extend this by representing each ordinary body relation as an additional **fixed (non-gradient) circuit leaf** whose per-example weight is the boolean truth of that relation evaluated at the example's key. The deterministic relational machinery in `xlog-prob` already conjoins such leaves; we are wiring a new leaf *kind* into the existing template + weight-swap loop, not building a new circuit engine.

**Tech Stack:** Rust (`pyxlog` PyO3 bindings, `xlog-prob` exact circuit, `xlog-logic` AST/proof-trace), Python (`pyxlog.ilp.neurosymbolic`), PyTorch (autograd via DLPack), CUDA. Build: `maturin`/`cargo`; tests: `pytest` (CUDA-gated) + `cargo test`.

---

## How to read this plan (honesty note)

This plan is **hybrid-fidelity by design**, per the repo rule against fake precision:

- **Python, tests, and the exact guard site** are given as complete, runnable code — these are verified against the current source.
- **Rust circuit internals** are specified at the *function + algorithm + data-structure* level with exact file:line targets, **not** as fabricated full listings. The injection kernel and slot layout in `xlog-prob` must be finalized against a compiler and a running GPU; inventing line-perfect Rust here would be guesswork. **Task 1 is a spike** whose explicit job is to lock those details before the bite-sized Rust tasks are executed. Treat Rust code blocks below as the intended shape, to be reconciled with spike findings.

If you are executing this plan and a Rust block does not compile as written, that is expected: the spike output (recorded in Task 1) is the source of truth for exact signatures.

---

## Progress

- **2026-06-12 — CPU slice of Task 3 landed & validated.** The pure gate-vs-unbound-join classifier was extracted into a CPU crate so it could be validated without a GPU on this machine (no CUDA here). New module `crates/xlog-logic/src/trainable_body.rs` exposes `classify_trainable_body_literal(literal, bound_vars, is_neural) -> TrainableBodyClass { Neural | Builtin | Negated | Epistemic | Gate | UnboundJoin{var} }`, re-exported from `xlog_logic`. Tests `crates/xlog-logic/tests/trainable_body.rs` — **9/9 pass via `cargo test -p xlog-logic --test trainable_body`** (CPU). Branch `feat/neurosymbolic-mixed-bodies` (worktree `../xlog-mixed-bodies`), commit `550e18aa`.
  - **Gate rule (decided):** a positive non-neural relation is a Stage-A gate iff every *named* variable is in `bound_vars` (= head vars ∪ neural-input vars). Anonymous `_` is allowed (existential projection). A named var not bound — even if used nowhere else — is an `UnboundJoin` (fail closed to Stage B); rename to `_` for an explicit existence check.
  - **Remaining for Task 3 (GPU session):** in `crates/pyxlog/src/neural.rs::build_query_signature` (line ~1796), compute `bound_vars`, call `xlog_logic::classify_trainable_body_literal`, push `Gate` literals to a new `gates` field, map `Builtin`→continue, and keep typed errors for `Negated`/`Epistemic`/`UnboundJoin` (the last now names the unbound variable). Then Tasks 4-8 (GPU).

---

## Spike findings (Task 1) — recorded 2026-06-15

**Environment.** Validated on an RTX 4090 (`sm_89`), CUDA 12.6 (`nvcc` 12.6.20), Rust
1.96.0 MSVC, VS2022 (MSVC 14.42). The earlier "no CUDA on this machine" premise no
longer holds — Task 1 ran locally on the GPU.

**Build gotchas locked** (for anyone running the GPU tasks on Windows):
- `.cargo/config.toml` pins a Linux-only `rustc-wrapper = scripts/rustc-wrapper.sh`
  → on Windows it fails with os error 193. Override per-invocation with
  `cargo --config <file>` where the file sets `build.rustc-wrapper = ""`.
- `crates/xlog-cuda/build.rs` defaults `XLOG_CUBIN_ARCHS=sm_120` (Blackwell), which
  **nvcc 12.6 cannot compile**. Set `XLOG_CUBIN_ARCHS=sm_89` for Ada/4090 (or
  `XLOG_NO_CUBIN=1` to rely on the sm_75 portable-PTX JIT fallback). Driver 591.44
  is newer than the toolkit, so no `XLOG_PTX_MAX_VERSION` downgrade is needed.
- Build/test inside a `vcvars64` environment so `nvcc` finds `cl.exe`.

**Step 1 — gate semantics: VALIDATED on GPU.** Throwaway test
`crates/xlog-prob/tests/spike_mixed_body_gate.rs` (deleted after recording, per Step 5)
compiled, via the non-template `ExactDdnnfProgram::compile_source` + `evaluate()` on
the GPU D4→XGCF path:

```
0.6::p_net(0).  0.6::p_net(2).  0.8::guard().  allowed(0).
root_case(C) :- p_net(C), guard(), allowed(C).
query(root_case(0)).  query(root_case(2)).
```

Result: `P(root_case(0)) = 0.48` (= 0.6·0.8) and `P(root_case(2)) = 0` **exactly**.
→ A *defined* deterministic relation in a mixed body is a per-key 0/1 multiplier in
the WMC; the conjunction `prob ∧ prob ∧ gate` already compiles correctly. The
gated-out probability is exactly 0 in the **compile** path — the `min_p`≈ε concern
is specific to the **fast-path fixed-leaf injection**, not the compiler.

**Step 2 — fixed-leaf injection point: LOCKED from source.**
`neural_backward_nll_buffers_inner` (`exact.rs:931`) has three per-group loops under
the hard invariant `probs.len() == out_grads.len() == slots.num_groups_usize()`
(`exact.rs:954,961`):
1. **fill** (`exact.rs:1005-1059`): `neural_fill_ad_chain_f32(prob_col, labels,
   slot_vars, eps, min_p, var_log_true, var_log_false)` via `cache.var_log_weights_mut()`.
2. **base scatter** (`exact.rs:1069-1137`): `neural_scatter_ad_chain_grads_f32(...,
   grad_true, grad_false, 0u8, out_col)` → writes `dlogZ_base/dp` into `out_grads[g]`.
3. **query scatter** (`exact.rs:1179-1221`): same kernel, phase `1u8`,
   `out -= dlogZ_query/dp`. `out_grads[g]` is exactly what flows back to torch.

A gate = a 1-label leaf = a `GpuWeightSlots` group (`neural_fast_path.rs:38`) with
**one** slot CNF var. It must be **filled (loop 1) but excluded from scatter (2 & 3)**.

**Decision: parallel fixed-slot set (not an `is_fixed` mask).** Extend
`neural_backward_nll_buffers_inner` with `fixed_slots: &GpuWeightSlots` +
`fixed_weights: &[CudaBuffer]` (1 row, value 1.0/0.0). Fill them in loop 1 (same
kernel, `labels = 1`), allocate **no** `out_grads` buffer for them, and never enter
loops 2 & 3. The existing `probs.len() == num_groups` invariant then applies to the
**neural** groups only. → **Gradient isolation is structural**: a gate has no grad
sink, so the scatter loops physically cannot write into it. This is strictly safer
than a shared-index `is_fixed` mask, which keeps the gate in `probs`/`out_grads` and
risks the documented grad-leak (spec §12). `GpuWeightSlots::upload(groups: &[Vec<u32>])`
builds the fixed set directly from the gate CNF vars.

**Step 3 — gradient isolation:** under the parallel-set design, zero gate gradient is
*structural* (no `out_grads` entry; fill is forward-only). Empirical confirmation is
Task 5/6 (needs the injection actually wired), not the spike.

**Bonus — hard-zero gate without ε:** the codebase already has
`force_query_var_false`/`force_query_var_true` (`exact.rs:1148-1158`), which force a
CNF var's weight to a hard 0/1 around the query run. A gate-*false* leaf can reuse the
same forcing on the gate's CNF var to get **exactly 0** instead of `min_p`≈ε —
resolving the spec §12 risk with **no new kernel**. Prefer this to the `abs=1e-6`
assertion when a hard zero is wanted.

**Net:** the fixed-leaf approach is sound; no architectural surprise. Tasks 3–8 should
treat the parallel fixed-slot signature above as the source of truth, superseding the
"shape-only" Rust blocks below.

### Implementation note (Tasks 4-5, recorded 2026-06-15)

Stage A shipped GREEN (`test_gate_relation_zeroes_ineligible_examples` passes; full
`test_nn4_dilp_training_surface.py` = 10/10 on RTX 4090). The implementation chose the
*pragmatic* variant over the spike's recommended parallel `fixed_slots` set, to avoid
re-cutting the `exact.rs` kernel signature on the working neural path:

- A gate is emitted as a **single-choice annotated disjunction** in `generate_template_ast`,
  grounded at the same head placeholder as the query, AFTER the neural disjunctions.
- `compile_circuit_for_template` appends **one 1-var slot per gate** to the existing
  `GpuWeightSlots` (gates are extra "groups", so `probs.len() == num_groups` still holds).
- `forward_backward_complex_tensor` appends a fixed 1.0/0.0 prob buffer per gate (truth
  from `evaluate_gate_truths` — substitutes the query's ground terms into the rule head and
  checks EDB-fact membership). Gate buffers are **filled but never backpropagated** (no
  network owns them), so the gate stays constant and receives no gradient — Task 6's
  grad-isolation assertions hold because gate predicates never enter the Python grad dicts.

Consequences vs. the parallel-set design:
- Gated-out probability is `~min_p` (≈1e-12), not a hard zero — covered by the test's
  `abs=1e-6`. If a hard zero is ever required, switch to the `force_query_var_false` path
  noted above.
- A gate gradient is computed then discarded (negligible cost) rather than structurally
  skipped. Isolation is behavioural (no grad sink consumed), not structural.
- Scope handled: head-bound EDB-fact gates (the flagship Stage-A pattern). Out of scope:
  gates over derived relations, anonymous-`_` gates, and the batched forward path
  (`forward_backward_batch_complex_tensor`) — gated batch queries fail closed on the
  `num_groups` invariant rather than silently miscomputing.

---

## Background: the exact failing point (verified)

1. `train_neurosymbolic_program` (`crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py`) desugars each `trainable_rule(id) :: head :- body.` into a real rule guarded by a synthetic 1-param neural predicate `nsr_guard_<id>`, compiles the whole source with the **real** parser (`pyxlog.Program.compile`), and trains via `program.forward_backward(query, target)`.

2. `forward_backward` → `forward_backward_complex_tensor` (`neural.rs:1112`) builds a circuit template via `build_query_signature` (`neural.rs:1767`) and `generate_template_ast` (`neural.rs:1953`), compiles it to a d-DNNF program once, caches it, and per example injects `network(inputs[i])` probabilities as circuit leaf weights, accumulating gradients back through DLPack (`exact.rs:931` `neural_backward_nll_buffers_inner`).

3. **The restriction** lives at `neural.rs:1796-1805`:

   ```rust
   if self.neural_registry.get(&body_atom.predicate).is_none() {
       return Err(PyValueError::new_err(format!(
           "Query rule for '{}' references relation '{}/{}' which is not an nn/4 \
            neural predicate; the neural query template path grounds only neural \
            predicates, so this body literal would evaluate over an empty relation",
           pred_name, body_atom.predicate, body_atom.terms.len()
       )));
   }
   ```

   The reason it must fail-closed today: `generate_template_ast` (`neural.rs:1974-2022`) adds leaves **only** for neural groups, then `template_program.rules.push(template_rule.clone())`. So an ordinary relation in the rule body is referenced but never populated → empty relation → query probability silently collapses to 0. The guard turns that silent zero into a typed error. Current behavior is asserted by `python/tests/test_nn4_dilp_training_surface.py::test_unsupported_body_relation_fails_closed`.

4. **What already works** (so we reuse, not rebuild): `xlog-prob` compiles deterministic facts to `const_true` provenance nodes and joins them uniformly with probabilistic facts (`crates/xlog-prob/src/provenance.rs:383,1160`); the training fast-path reuses that same cached circuit (`exact.rs:1004-1068`). Builtins (`is`, comparisons) in trainable bodies already flow (`neural.rs:1791-1793`, `_ => continue`). We are extending the *template population + leaf injection*, which is the only gap.

## Scope decision: two stages

| | Body shape | Example | Circuit impact | This plan |
|---|---|---|---|---|
| **Stage A — gates** | ordinary relation shares only **already-bound** variables (head/neural-input vars); introduces no new join variable | `root_case(Case) :- neural_root(Case, positive), allowed(Case).` | one extra **fixed** leaf per ordinary literal; per-example boolean weight | **Full implementation** |
| **Stage B — joins** | ordinary relation introduces a **new** existential variable joined to the neural predicate over a shared domain | `plastic(Edge, strengthen) :- saliency(Event, strengthen), pre_before_post(Event, Edge).` | neural predicate must be grounded over the **real** join domain (N leaves/example), head aggregates via OR; placeholder grounding no longer sufficient | **Scoped only** (Task 9) — likely its own spec |

Stage A unblocks the common "neural score gated by symbolic eligibility" pattern and is the right first ship. **The flagship plasticity rule is Stage B** (it introduces `Event`); Stage A is the prerequisite that validates the fixed-leaf injection mechanism Stage B also needs.

## File structure

- `crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py` — desugaring already preserves body literals; extend to (a) collect ordinary-relation literals per rule, (b) materialize their per-example truth from the compiled program, (c) pass them to the engine. Add result field for gate diagnostics.
- `crates/pyxlog/src/neural.rs` — `build_query_signature` (relax guard, classify gate literals), `generate_template_ast` (emit fixed leaf + keep literal), `forward_backward_complex_tensor` (build fixed-leaf weight buffers per example, no grad), supporting structs (`NeuralGroup`/new `GateGroup`, `QuerySignature`).
- `crates/xlog-prob/src/exact.rs` — `neural_backward_nll_buffers_inner` accepts fixed (no-grad) leaf buffers alongside neural prob buffers; fill their `var_log_weights` but skip gradient scatter.
- `crates/xlog-logic/src/proof_trace.rs` — already stores `body_literals`; ensure ordinary literals appear in the trace map body (Python passes them).
- `python/tests/test_nn4_dilp_training_surface.py` — flip the fail-closed test to a positive gate test; add gate-zeroing and gradient-isolation tests.
- `crates/pyxlog/tests/` or `crates/xlog-prob/tests/` — Rust unit test for fixed-leaf conjunction semantics.

---

## Task 1: Spike — validate the fixed-leaf gate mechanism (throwaway)

**Goal:** Empirically confirm that (a) a deterministic relation can be evaluated per example key `i` from the compiled program, and (b) injecting its truth as a fixed circuit leaf yields `P(head(i)) = p_net(i) * sigmoid(w) * gate(i)` with `gate(i) ∈ {0,1}` and **no** gradient to the gate. Lock the exact `exact.rs` slot/buffer layout the real tasks depend on.

**Files:**
- Scratch: `crates/pyxlog/examples/spike_mixed_body.rs` (or a `#[ignore]` test) — delete after.
- Read for reference: `crates/xlog-prob/src/exact.rs:931-1130`, `crates/pyxlog/src/neural.rs:1112-1373`, `crates/xlog-prob/src/provenance.rs:360-400`.

- [ ] **Step 1: Manually construct the target circuit.** Hand-write a small `.xlog` equivalent to the desugared mixed-body program *with the gate relation included as a deterministic fact at the example key*, e.g.:

  ```
  nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
  allowed(0). allowed(1).
  nsr_guard_rule_mixed(0).            // guard leaf, weight = sigmoid(w)
  root_case(Case) :- neural_root(Case, positive), nsr_guard_rule_mixed(0), allowed(Case).
  ```
  Compile it via the **non-template** exact path (`ExactDdnnfProgram::compile_source`) for `query(root_case(2))` and `query(root_case(0))`. Confirm `P(root_case(2)) = 0` (gated out) and `P(root_case(0)) = p_net*sigmoid(w)`.

- [ ] **Step 2: Determine how to inject `allowed(i)` per example into the cached neural circuit.** Inspect `GpuWeightSlots` and `var_log_weights_mut()` (`exact.rs:1004-1057`). Decide the concrete representation of a *fixed* leaf: most likely a probabilistic leaf set to weight `1.0`/`0.0` (log `0` / `-inf` clamped by `cfg.min_p`) that is **excluded from the gradient scatter loop** (`exact.rs:1069-1130`). Record the exact struct fields and kernel-launch shape needed.

- [ ] **Step 3: Confirm gradient isolation.** Verify that a leaf injected as a fixed weight and omitted from `neural_scatter_ad_chain_grads_f32` produces zero gradient for that leaf while neural leaves still receive correct gradients.

- [ ] **Step 4: Write the spike findings into this plan file** under a new "## Spike findings" section: exact signature of the function that injects fixed leaves, the slot struct changes, and whether the gate must be a `ProbFact` or a new leaf kind. **The Rust tasks below consume this.**

- [ ] **Step 5: Delete the scratch file.** Do not commit the spike. `git status` clean of `spike_*`.

---

## Task 2: Failing test — gate relation is honored, not rejected (Python)

**Files:**
- Test: `python/tests/test_nn4_dilp_training_surface.py` (replace `test_unsupported_body_relation_fails_closed`).

- [ ] **Step 1: Replace the fail-closed test with a positive gate test.**

  ```python
  def test_gate_relation_zeroes_ineligible_examples() -> None:
      """A deterministic relation sharing only the head variable acts as a per-example
      gate: P(head(i)) = p_net(i)[positive] * sigmoid(w) for eligible i, exactly 0 otherwise."""
      network = _root_net()
      w0 = 0.7
      source = """
          nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
          allowed(0).
          allowed(1).
          trainable_rule(rule_mixed, weight=0.7) :: root_case(Case) :-
              neural_root(Case, positive), allowed(Case).
          train(root_case, binary_cross_entropy).
      """
      result = train_neurosymbolic_program(
          source,
          networks={"root_net": network},
          examples=_examples(),                      # rows 0..3; allowed only for 0,1
          config=NeuroSymbolicTrainingConfig(steps=1, learning_rate=0.0),
      )

      inputs = _examples()[0]["inputs"]
      with torch.no_grad():
          p_pos = network(inputs.cuda().reshape(-1, 1))[:, 1].cpu()
      p_guard = 1.0 / (1.0 + math.exp(-w0))

      assert result.query_probabilities[0] == pytest.approx(float(p_pos[0]) * p_guard, abs=1e-5)
      assert result.query_probabilities[1] == pytest.approx(float(p_pos[1]) * p_guard, abs=1e-5)
      assert result.query_probabilities[2] == pytest.approx(0.0, abs=1e-6)
      assert result.query_probabilities[3] == pytest.approx(0.0, abs=1e-6)
  ```

- [ ] **Step 2: Run it; expect failure with the current typed engine error.**

  Run: `pytest python/tests/test_nn4_dilp_training_surface.py::test_gate_relation_zeroes_ineligible_examples -v`
  Expected: FAIL — error message contains "is not an nn/4 neural predicate" (the guard at `neural.rs:1796`).

- [ ] **Step 3: Commit the failing test.**

  ```bash
  git add python/tests/test_nn4_dilp_training_surface.py
  git commit -m "test(neurosymbolic): mixed-body gate relation should zero ineligible examples"
  ```

---

## Task 3: Classify gate literals and relax the guard (Rust)

**Files:**
- Modify: `crates/pyxlog/src/neural.rs:1767-1906` (`build_query_signature`), and the `QuerySignature`/`NeuralGroup` definitions (search `enum QuerySignature`, `struct NeuralGroup` in the same file).

- [ ] **Step 1: Add a gate representation to the signature.** Introduce `struct GateLiteral { predicate: String, arg_var: String /* the shared, head-bound variable */ }` and add `gates: Vec<GateLiteral>` to both `QuerySignature::Boolean` and `QuerySignature::Targeted` (or hang gates off a shared field). A literal is a **Stage-A gate** iff: it is `BodyLiteral::Positive`, its predicate is **not** in `neural_registry`, **and** every term is either a constant or a variable that already appears in the rule head or in a neural input position (i.e. introduces no new variable). If a non-neural literal introduces a fresh variable → it is Stage B → keep the existing typed error (now reworded to "introduces an unbound join variable; Stage B not yet supported").

- [ ] **Step 2: Replace the unconditional rejection at `neural.rs:1796-1805`** with the classification:

  ```rust
  if self.neural_registry.get(&body_atom.predicate).is_none() {
      // Stage A: ordinary relation that only re-binds existing variables is a gate.
      match self.classify_gate_literal(body_atom, &rule.head, &groups)? {
          GateOrUnsupported::Gate(gate) => { gates.push(gate); continue; }
          GateOrUnsupported::UnboundJoin(var) => {
              return Err(PyValueError::new_err(format!(
                  "Query rule for '{}' relation '{}/{}' introduces unbound join variable '{}'; \
                   mixed-body joins (Stage B) are not yet supported",
                  pred_name, body_atom.predicate, body_atom.terms.len(), var
              )));
          }
      }
  }
  ```
  Implement `classify_gate_literal` next to `build_query_signature`.

- [ ] **Step 2b: Verify the gate predicate is actually defined** in `self.ast` (as a fact or rule head). If not, keep a typed error ("references undefined relation 'X/n'") — this preserves fail-closed for genuine typos and is distinct from the Stage-B message.

- [ ] **Step 3: Build + run the Python test; expect a DIFFERENT failure.**

  Run: `maturin develop --release` (or the repo's build step) then `pytest ...::test_gate_relation_zeroes_ineligible_examples -v`
  Expected: FAIL, but **no longer** the "not an nn/4" error — now it fails later (template lacks the gate leaf, so probability is wrong/zero everywhere, or a missing-relation panic). This proves the guard is relaxed. Record the new failure mode.

- [ ] **Step 4: Commit.**

  ```bash
  git add crates/pyxlog/src/neural.rs
  git commit -m "feat(neurosymbolic): classify Stage-A gate literals, relax neural-only guard"
  ```

---

## Task 4: Emit gate leaves into the template circuit (Rust)

**Files:**
- Modify: `crates/pyxlog/src/neural.rs:1953-2094` (`generate_template_ast`).

- [ ] **Step 1: For each `GateLiteral`, add a deterministic backing so the relation is non-empty at the placeholder grounding.** The gate's shared variable resolves to a head placeholder constant (the same canonical placeholder logic at `neural.rs:1964-1972`). Emit the gate atom as a **probabilistic leaf** (a 1-choice `AnnotatedDisjunction` or `ProbFact`) at that placeholder constant, weight to be injected per example — mirroring how neural groups are emitted at `neural.rs:2008-2019`. Keep the gate literal in `template_rule` (it is already there via `template_rule.clone()` at `neural.rs:2022`), so the compiled circuit conjoins `neural ∧ guard ∧ gate`.

- [ ] **Step 2: Record gate→slot mapping** in the returned signature/cache so Task 5 can inject per-example weights into the correct circuit variable, exactly parallel to neural group slots (`neural.rs:2329` `GpuWeightSlots` construction — extend it to carry gate slots).

- [ ] **Step 3: Build; run the Python test.** Expected: still FAIL on values (no per-example injection yet → gate weight is whatever the placeholder prob is), but the circuit now **compiles with the gate variable present**. Confirm via a temporary `eprintln!` of circuit var count, or that the error is now "values differ" rather than "compile/empty relation".

- [ ] **Step 4: Commit.**

  ```bash
  git add crates/pyxlog/src/neural.rs
  git commit -m "feat(neurosymbolic): emit Stage-A gate leaves into circuit template"
  ```

---

## Task 5: Evaluate gate truth per example and inject as a fixed leaf (Python + Rust)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py` (collect gate predicates from desugared rules; evaluate per example key; pass to engine).
- Modify: `crates/pyxlog/src/neural.rs:1112-1373` (`forward_backward_complex_tensor`) — accept gate weights, build fixed buffers.
- Modify: `crates/xlog-prob/src/exact.rs:931-1130` (`neural_backward_nll_buffers_inner`) — fill gate leaf weights, **exclude from gradient scatter** (per Task 1 spike).

- [ ] **Step 1 (Python): collect gate literals and evaluate their extension once.** After `program = pyxlog.Program.compile(...)`, for each trainable rule, identify body literals whose predicate is neither a neural predicate nor a guard nor a builtin. Query the compiled program for that relation's extension (use the existing deterministic query surface — confirm the method name, e.g. `program.query_relation("allowed")` or evaluating `?- allowed(X).`; if no such API exists, this is a 1-method addition to `pyxlog` and becomes Task 5a). Build, per gate, a boolean tensor `gate_truth[i] = (i ∈ extension)` aligned to the example rows (key = row index `i`, matching the `train_head(i)` query convention).

- [ ] **Step 2 (Python): pass gate truth into `forward_backward`.** Extend the per-example call so the engine receives `{gate_predicate: gate_truth[i]}` for query `train_head(i)`. Keep gradients off (these are constants). Plumb via a new optional arg on the native `forward_backward` or a pre-step `program.set_gate_weights(...)`.

- [ ] **Step 3 (Rust): inject gate weight as a fixed leaf.** In `forward_backward_complex_tensor`, build a buffer for each gate leaf set to the example's `gate_truth` (1.0 → log 0; 0.0 → log(min_p) clamped, then the circuit AND collapses to ~0). Pass alongside `prob_bufs` to `neural_backward_nll_buffers_with_device_loss`, tagged as no-grad. In `neural_backward_nll_buffers_inner`, fill these into `var_log_weights` in the same loop as neural probs (`exact.rs:1004-1057`) but **do not** add them to the gradient scatter set (`exact.rs:1069-1130`).

- [ ] **Step 4: Run the Stage-A test; expect PASS.**

  Run: `pytest python/tests/test_nn4_dilp_training_surface.py::test_gate_relation_zeroes_ineligible_examples -v`
  Expected: PASS — rows 0,1 = `p_net*sigmoid(w)`, rows 2,3 = `0.0`.

- [ ] **Step 5: Commit.**

  ```bash
  git add crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py crates/pyxlog/src/neural.rs crates/xlog-prob/src/exact.rs
  git commit -m "feat(neurosymbolic): inject per-example gate truth as fixed circuit leaf"
  ```

---

## Task 6: Gradient isolation test — gate leaf receives no gradient (Python)

**Files:**
- Test: `python/tests/test_nn4_dilp_training_surface.py`.

- [ ] **Step 1: Write the test.**

  ```python
  def test_gate_relation_receives_no_gradient() -> None:
      """The gate is a ground fact: training must flow gradients to the net and the
      rule weight, but never attempt to differentiate the gate relation."""
      network = _root_net()
      source = """
          nn(root_net, [Case], Label, [negative, positive]) :: neural_root(Case, Label).
          allowed(0).
          allowed(1).
          trainable_rule(rule_mixed, weight=0.0) :: root_case(Case) :-
              neural_root(Case, positive), allowed(Case).
          train(root_case, binary_cross_entropy).
      """
      result = train_neurosymbolic_program(
          source, networks={"root_net": network}, examples=_examples(),
          config=NeuroSymbolicTrainingConfig(steps=4, learning_rate=0.2),
      )
      # Neural + rule weight still learn; no "gate" entry pollutes the grad dicts.
      assert result.neural_parameter_grads["root_net"] > 0.0
      assert result.symbolic_weight_grads["rule_mixed"] > 0.0
      assert "allowed" not in result.symbolic_weight_grads
      assert "allowed" not in result.neural_parameter_grads
  ```

- [ ] **Step 2: Run; expect PASS** (asserts behavior already implemented in Task 5). If it fails because `allowed` leaks into a grad dict, fix the dict construction in `neurosymbolic.py` to exclude gate predicates.

  Run: `pytest python/tests/test_nn4_dilp_training_surface.py::test_gate_relation_receives_no_gradient -v`

- [ ] **Step 3: Commit.**

  ```bash
  git add python/tests/test_nn4_dilp_training_surface.py crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py
  git commit -m "test(neurosymbolic): gate relation is non-differentiable"
  ```

---

## Task 7: Rust unit test for fixed-leaf conjunction semantics (Rust)

**Files:**
- Test: `crates/xlog-prob/tests/mixed_body_gate.rs` (new), modeled on `crates/xlog-prob/tests/exact_ddnnf.rs:27-62`.

- [ ] **Step 1: Write a CUDA-gated Rust test** that compiles the manual program from Task 1 Step 1 and asserts `P(root_case(2)) == 0` and `P(root_case(0)) == p_net*sigmoid(w)` directly through `ExactDdnnfProgram`, independent of the Python layer. This locks the engine semantics so a future Python refactor can't hide a regression.

  Run: `cargo test -p xlog-prob --test mixed_body_gate -- --nocapture`
  Expected: PASS (skip-gated if no CUDA, matching existing test conventions).

- [ ] **Step 2: Commit.**

  ```bash
  git add crates/xlog-prob/tests/mixed_body_gate.rs
  git commit -m "test(xlog-prob): fixed-leaf gate conjunction semantics"
  ```

---

## Task 8: Proof-trace + inventory carry gate literals; docs (Python + docs)

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py` (`_build_proof_trace_map` — include gate literals in `body_literals`).
- Modify: `docs/architecture/dilp-training.md` and `docs/architecture/ucr-xlog-diagnostics.md` (replace the "must consist of nn/4 + builtins" limitation text with the Stage-A capability + the remaining Stage-B limitation).

- [ ] **Step 1: Include gate literals in the proof trace body.** Confirm `result.proof_trace_map.traces()` body for a mixed rule lists both the neural and the gate literal. Add an assertion to `test_proof_trace_credit_assignment` (or a sibling test) that the gate literal appears in the trace's body for a mixed rule.

  Run: `pytest python/tests/test_nn4_dilp_training_surface.py -k proof_trace -v`
  Expected: PASS.

- [ ] **Step 2: Update the architecture docs** to state: Stage-A gate relations (no new join variable) are supported and non-differentiable; Stage-B join relations remain unsupported with a typed error naming the unbound variable.

- [ ] **Step 3: Full suite + commit.**

  Run: `pytest python/tests/test_nn4_dilp_training_surface.py -v` (all pass)
  ```bash
  git add crates/pyxlog/python/pyxlog/ilp/neurosymbolic.py docs/architecture/dilp-training.md docs/architecture/ucr-xlog-diagnostics.md
  git commit -m "docs(neurosymbolic): document Stage-A mixed-body gates; trace carries gate literals"
  ```

---

## Task 9: Stage B (joins) — scope only, do NOT implement here

**Files:**
- Create (spec, untracked per repo convention): `docs/superpowers/specs/2026-06-12-neurosymbolic-mixed-body-joins.md`.

- [ ] **Step 1: Write a short spec capturing the Stage-B problem and the key design decision**, so the flagship plasticity rule (`plastic(Edge, L) :- saliency(Event, L), pre_before_post(Event, Edge).`) has a home. Required content:
  - Why placeholder grounding is insufficient: the neural predicate must be grounded over the **real** join domain (one leaf per `Event`), and the head aggregates contributions via OR (`xlog-prob` provenance already does OR-aggregation).
  - The enabling assumption to keep the single-cached-circuit performance contract: the **relational structure is example-independent** (graph topology fixed; only neural saliency varies per example). Under that assumption, compile once over the real domain and swap N neural weights/example.
  - The slot/tensor-source change: map each example's neural outputs to N domain-keyed leaves instead of 1 placeholder leaf.
  - Open question to resolve in that spec: examples whose *structure* varies per example (per-example facts) — requires either per-structure circuit caching or per-example recompile; measure the cost before committing.

- [ ] **Step 2: Do not commit the spec** (repo convention: specs/plans stay untracked). Leave it in the working tree for review.

---

## Risks & mitigations

- **Spike invalidates the fixed-leaf approach** (e.g. the weight-swap path can't cleanly host a no-grad leaf). Mitigation: Task 1 is explicitly throwaway and gates the rest; if it fails, the fallback is per-example recompile of the full program slice (slower, but correct) — re-scope before Task 3.
- **No deterministic-relation query API in `pyxlog`.** Mitigation: Task 5 Step 1 flags this; adding one read-only relation-extension accessor is small and independently useful.
- **`min_p` clamping makes gated-out probability ~ε, not exactly 0.** Mitigation: assert with `abs=1e-6` (Task 2) and, if needed, special-case weight `0.0` → hard circuit `false` rather than `log(min_p)`.
- **Grad-dict leakage** of gate predicate names. Covered by Task 6.
- **Scope creep into Stage B.** Mitigation: the classifier (Task 3) hard-stops Stage-B shapes with a distinct typed error; Stage B is spec-only here.

## Self-review (completed)

- **Spec coverage:** the requirement "trainable body mixes nn/4 + ordinary relations" is met for the gate class (Tasks 2-7) and explicitly deferred-with-a-home for the join class (Task 9). The original fail-closed test is intentionally replaced (Task 2), not silently dropped.
- **Placeholder scan:** Rust internals are deliberately spike-driven (declared up front in "How to read this plan"), not TODO-elided; Python/tests are complete.
- **Type consistency:** `GateLiteral`/`GateOrUnsupported`/`gates` introduced in Task 3 are consumed in Tasks 4-5; gate weights flow Python→`forward_backward_complex_tensor`→`neural_backward_nll_buffers_inner` consistently; grad dicts keyed by rule id (`rule_mixed`) match the desugared guard id, and gate predicates are excluded from them (Task 6).
