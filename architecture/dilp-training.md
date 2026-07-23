# dILP Training

Differentiable ILP on the tensorized super-graph: candidate masks, tagged credit, the neurosymbolic joint mixture, and the neural join-body path with its measured limits.

This document describes the dILP training subsystem: a GPU-accelerated differentiable Inductive Logic
Programming engine that learns Datalog rules from positive/negative examples via gradient descent.

## Design Goals

1. **Learn rules, not weights** — discover symbolic Datalog clauses (e.g., `reach(X,Y) :- edge(X,Z), edge(Z,Y).`) from data
2. **GPU-resident hot loop** — no semantic column downloads in the training step loop
3. **Sparse by default** — candidate-indexed soft-probs instead of materializing N³ tensors
4. **Transactional promotion** — learned rules pass gate checks before entering the knowledge base
5. **Auditable transfer evidence** — learned rules carry fold, held-out-domain, gate, and base-kernel checksum metadata

## Core Idea: Tensorized Super-Graph Masking

Traditional ILP systems compile candidate rules into executable programs — impossible at millisecond
timescales. XLOG's approach pre-compiles a "super-graph" of all candidate rules and activates them
via continuous mask tensors optimized with Gumbel-Softmax:

```
Candidate rules          Logit tensor W (C floats)
  ┌──────────┐              ┌───────────┐
  │ r1: A←B,C│              │ w1  w2 .. │ ─── Gumbel-Softmax(τ) ──►  soft mask
  │ r2: A←B,D│              │           │                               │
  │ ...       │              └───────────┘                               ▼
  └──────────┘                                              set_rule_mask_sparse()
                                                                        │
                                                                        ▼
                                                            xlog evaluate (GPU)
                                                                        │
                                                                        ▼
                                                              BCE loss + ∇W
```

At convergence, `argmax(W)` picks the winning rule. Temperature annealing (τ → τ_floor) drives the
soft mask toward a one-hot selection.

For the full mathematical treatment and the resolved design decisions that led
to the shipped trainer, see `rfc-tensorized-ilp` (internal RFC, retired from the docs site).

## Architecture Overview

```
                           Python (pyxlog.ilp)
     ┌─────────────────────────────────────────────────────┐
     │  train_only()                                       │
     │    ├─ valid_candidates()  → candidate map           │
     │    ├─ MaskBackend.init_weights()                    │
     │    ├─ AdaptiveTempController                        │
     │    └─ step loop ──────────────────────────┐         │
     │         ├─ MaskBackend.apply_mask()       │         │
     │         ├─ program.evaluate_device()      │ GPU     │
     │         ├─ BCE loss (torch)               │ only    │
     │         ├─ loss.backward()                │         │
     │         └─ optimizer.step()               │         │
     │                                           │         │
     │  train_and_promote()                      │         │
     │    ├─ train_only()                        │         │
     │    ├─ trial compile (Rust)                │         │
     │    └─ promotion gates                     │         │
     └───────────────────────────────────────────┘         │
                        │                                   │
                        ▼                                   │
     ┌────────────────────────────────────┐                │
     │  Rust (xlog-runtime, xlog-cuda)   │                │
     │    ├─ set_rule_mask_sparse()       │◄───────────────┘
     │    ├─ IlpRegistry (mask storage)  │
     │    ├─ TensorMaskedJoin (executor) │
     │    ├─ batch_fact_membership()     │
     │    ├─ batch_fact_membership_device() │
     │    ├─ batch_tagged_credit()       │
     │    └─ batch_tagged_credit_device()│
     └────────────────────────────────────┘
                        │
                        ▼
     ┌────────────────────────────────────┐
     │  CUDA (kernels/ilp.cu)            │
     │    └─ extract_nonzero_indices()   │
     └────────────────────────────────────┘
```

## Key Entry Points

### Python (pyxlog.ilp)

| File | Purpose |
|------|---------|
| `pyxlog/ilp/trainer.py` | `train_only()` — multi-start training loop |
| `pyxlog/ilp/promoter.py` | `train_and_promote()` — training + gate pipeline |
| `pyxlog/ilp/neurosymbolic.py` | `train_neurosymbolic_program()` — joint `nn/4` and symbolic rule-weight training |
| `pyxlog/ilp/inventory.py` | `build_rule_inventory()` — selected/rejected clause inventory with transfer metadata |
| `pyxlog/ilp/backend.py` | `MaskBackend` protocol, `SparseMaskBackend`, `DenseMaskBackend` |
| `pyxlog/ilp/temperature.py` | `AdaptiveTempController` — cosine-annealed τ schedule |
| `pyxlog/ilp/entropy.py` | Entropy regularization helpers |
| `pyxlog/ilp/holdout.py` | `holdout_f1_and_variance()` — LOO (`<=20`) and k-fold (`>20`) F1 scoring |
| `pyxlog/ilp/types.py` | `TrainConfig`, `TrainResult`, `PromotionResult`, `LearnedArtifact`, etc. |
| `pyxlog/ilp/exceptions.py` | `IlpConfigError`, `IlpCandidateError`, `IlpTrainingError` |

### Rust (xlog-runtime, xlog-cuda)

| File | Purpose |
|------|---------|
| `crates/xlog-runtime/src/ilp_registry.rs` | `IlpRegistry` — mask storage, `IlpTaggedResult` metadata |
| `crates/xlog-runtime/tests/ilp_integration_tests.rs` | Rust-side integration tests for mask round-trips |
| `crates/xlog-cuda/tests/ilp_kernel_tests.rs` | CUDA kernel unit tests (extract_nonzero_indices) |

### CUDA Kernels

| File | Purpose |
|------|---------|
| `kernels/ilp.cu` | `extract_nonzero_indices()` — N³ mask → sparse index extraction |

## Mask Backends

The `MaskBackend` protocol abstracts how the learnable tensor W is applied to the XLOG executor:

### SparseMaskBackend (default)

```
W (C logits) → Gumbel-Softmax(τ) → candidate_soft_probs (C,)
                                          │
                    argsort/top-k on CUDA in Python/Torch
                                          │
                    set_rule_mask_sparse_selected(selected_ids, selected_soft_probs)
                                          │
                                          ▼
                              Rust builds executor mask internally
                              (no N³ tensor materialized, no full soft-vector device-to-host transfer)
```

- Learnable params: `C` floats (one per candidate rule)
- Memory: O(C) — typically C < 100
- Preferred hot-loop path calls `set_rule_mask_sparse_selected()` on the compiled program
- Legacy compatibility path `set_rule_mask_sparse()` remains available when Rust-side ranking is desired

### DenseMaskBackend (alpha-compatible, debug)

```
W (N×N×N logits) → Gumbel-Softmax(τ) → 3D soft mask
                                              │
                           flatten → DLPack → set_rule_mask()
                                              │
                                              ▼
                                    Rust imports N³ tensor
```

- Learnable params: N³ floats (N = schema size)
- Memory: O(N³) — expensive for large schemas
- Enabled via `TrainConfig(debug_dense_mask=True)` for parity testing

## Training Pipeline

### train_only()

1. **Candidate enumeration** — `valid_candidates(source, mask_name)` returns all syntactically legal body-pair assignments
2. **Multi-start** — up to `max_attempts` independent restarts with fresh logits
3. **Step loop** (per attempt, up to `step_budget_per_attempt`):
   - Apply mask via backend
   - Forward pass: `program.evaluate_device()` (GPU-only, no host reads)
   - BCE loss between predicted and target fact membership
   - Backward pass: `loss.backward()` through PyTorch autograd
   - Optimizer step on W
   - Temperature anneal: τ_start → τ_floor (cosine schedule)
   - Optional deterministic controls (`deterministic=True`) for reproducible attempt seeding
   - Early stopping: when argmax is stable and loss < threshold
4. **Decode** — `argmax(W)` maps to winning candidate → discovered rule string

### train_and_promote()

1. Call `train_only()` — get `TrainResult`
2. If not converged → `PromotionStatus.NOT_CONVERGED`
3. **Trial compile** — substitute discovered rule into source, compile via Rust
4. **Promotion gates** (all must pass for `PROMOTED`):
   - **Convergence gate** — training converged (already checked)
   - **Novel-rate gate** — fraction of non-example derivations ≤ `max_novel_rate`
   - **Protected-relation gate** — no unwanted relation side-effects
   - **Holdout F1 gate** — F1 on held-out examples ≥ threshold
   - **Ambiguity gate** — top-M scan (or exhaustive mode) detects no alternative winning candidates
   - **Typed-schema gate** — optional hard gate requiring relation type metadata (or waiver-driven manual review)
5. All pass → `PromotionStatus.PROMOTED` with `committed_source`

### External Consumer Training Surface

External consumer validation work added a higher-level training entry point for
sources that mix neural predicates and trainable symbolic clauses:

```python
from pyxlog.ilp.neurosymbolic import NeuroSymbolicTrainingConfig, train_neurosymbolic_program

result = train_neurosymbolic_program(
    source,
    networks={"score": torch_module},
    examples=training_rows,
    config=NeuroSymbolicTrainingConfig(steps=16, learning_rate=0.05),
)
```

The source owns declarative `nn(...)`, `trainable_rule(...)`, and `train(...)`
declarations. The result reports neural gradient norms, symbolic gradients,
final symbolic weights, and a `RuleInventory` suitable for transfer audits.

#### Existential-join trainable bodies (Stage B)

A `trainable_rule` body may join a neural predicate to an ordinary relation on an
**existential** (non-head) variable — the neural predicate is grounded over the
**real join domain inside the circuit** and OR-aggregated at the head:

```
plastic(Edge) :- saliency(Event, strengthen), pre_before_post(Event, Edge).
```

Here `Event` appears only in the body. The engine materializes the join domain
from `pre_before_post`'s ground facts, emits one neural leaf per joined event, and
the differentiable provenance OR-aggregates the per-event contributions per head
binding, yielding `P(plastic(Edge)) = σ(w) · (1 − ∏_{e : pre_before_post(e,Edge)} (1 − p_saliency(e)))`.
Gradient flows into the neural predicate (all joined events) and the rule guard,
but never into the deterministic join relation. The per-event features arrive
through a `domain_inputs={"net": features}` channel, with a companion
`domain_ids={"net": ids}` naming the domain constant each row holds (see the
`domain_ids` contract below), and `examples` carry only per-head-binding
`targets`. Because `saliency` is learned as a function of the event feature (not an
id lookup), the trained predicate generalizes to unseen events.

Constraints: the join domain must be ground facts (a derived relation is rejected,
since its extension is not materialized); head-binding ids must be `0..N-1`
row-aligned with `targets`; a single join network is supported; and the exact
d-DNNF compiler builds one circuit over all head-binding queries, so the planted
graph must stay within the compiler's fixed buffer (empirically ~6–7 events).
Worked example + CUDA-gated recovery test:
[`examples/plasticity_incircuit/`](https://github.com/BrainyBlaze/xlog/tree/main/examples/plasticity_incircuit) and
`python/tests/test_plasticity_incircuit.py`. Head-variable ("hard filter") joins
remain supported as pre-filters; only the existential-join case is new.

#### Neural join bodies in the joint mixture

A rule that puts a neural predicate on an **existential join variable** is no longer
confined to being the program's only trainable rule: it now competes as **one of
several same-head candidates** in the joint mixture. Give the mixture a relation
vocabulary and it will find which relation the rule joins on, while learning the
neural predicate from scratch:

```
trainable_rule(cand_pre_before_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
trainable_rule(cand_post_before_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
trainable_rule(cand_co_occurs,       weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), co_occurs(Ev, E).
```

Each candidate's mask is the OR, over the join extension **read from the engine**
(`relation_facts` — never a caller-supplied binding→constants hint), of the network's
**per-constant** probability, computed in log space:

```
mask_k[h] = 1 − ∏_{e ∈ ext_k(h)} (1 − p_net(e))
```

This is the engine proposal's long-deferred **Change 2** ("multi-rule same-head with
neural bodies"), delivered by **route (A)** — extend the mixture to carry neural join
masks. Route (B) — un-fusing the circuit backward so the mixture could be pushed back
into the compiled program — is **not needed for this goal** and is not pursued.

**The `domain_ids` contract.** `domain_ids` is the **one** map from a domain constant
to its feature row, and **both** engines — the exact d-DNNF circuit and the torch-side
mixture — resolve rows through it:

```python
train_neurosymbolic_program(
    source,
    networks={"sal_net": net},
    domain_inputs={"sal_net": features},      # [D, k]: one row per join-domain constant
    domain_ids={"sal_net": [0, 2, 4, ...]},   # which CONSTANT each row holds
    examples=[{"targets": targets}],
    config=NeuroSymbolicTrainingConfig(steps=1500, learning_rate=0.05),
)
```

The ids must be **distinct**; they may be in **any order** and need not be contiguous
(a joined constant absent from `domain_ids` is refused, not silently mis-indexed).
Omitting `domain_ids` defaults it to `[0 … D-1]`. There is no rank-indexing fallback:
the map is the only path from a constant to a feature row.

**Semantics anchor.** With a *single* candidate, the torch-side OR reproduces the exact
d-DNNF circuit to ~2e-07 (tolerance 1e-4) on **four** domain layouts — dense, sparse,
superset (rows for constants that are never joined) and shuffled (`domain_ids` in
non-sorted order). See `python/tests/test_join_semantics_anchor.py`.

**This is candidate SELECTION, not rule induction.** `build_join_candidates` fills the
single free slot of a fixed body template once per relation name **supplied by the
caller** — `|R|` candidates, no conjunctions, no chaining through an intermediate
variable, no recursion, no negation. It is a different and *narrower* search than the
engine-side dILP enumerator (`valid_candidates`, `|R|²` chained candidates with
recursion), and it does **not** call it: the two induction paths remain disjoint. What is
new here is the neural predicate on an existential variable, trained *through* the logic
— not an enlargement of the hypothesis space.

Worked example + CUDA-gated tests:
[`examples/neural_join_discovery/`](https://github.com/BrainyBlaze/xlog/tree/main/examples/neural_join_discovery),
`python/tests/test_join_discovery.py` and `python/tests/test_join_identifiability.py`.

**Limits — stated plainly:**

1. **One join network per program.** `domain_inputs` currently supports a single join
   network.
2. **Head arity must be 1.** The multi-outcome form
   `plastic(Edge, L) :- saliency(Event, L), pre_before_post(Event, Edge).` **does not
   compile** — the mixture's eligibility call is fixed at arity 1. Multi-outcome
   plasticity (a learned label on the head) is **not** supported and must not be
   claimed.
3. **The inter-candidate noisy-OR is a modelling choice, not compiled semantics.** The
   anchor above pins the **per-candidate** mask against the exact circuit. The rule that
   **combines** several candidates into one head probability has no exact-circuit
   counterpart and cannot have one: declaring more than one `trainable_rule` is
   precisely what routes execution away from the circuit and into the torch-side
   mixture.
4. **Saturation.** The noisy-OR saturates as the number of joined constants per head
   binding grows: at the default init (`p ≈ 0.5`) a binding with `k` joined constants
   starts at `1 − (1−p)^k ≈ 1`, the gradient to the detector vanishes, and the optimizer
   lands in a degenerate *inverted* minimum that more steps do not escape (seed 0,
   `k = 6`, bare: loss pinned at the base-rate entropy 0.640 at 1500/3000/6000/12000
   steps, with the wrong candidate hardened to 1.0). Shifting the detector's initial
   logit for the positive label by `−2.0` (a "quiet prior": an *initialization* encoding
   the prior that constants are mostly negative) removes the basin. Measured over 5
   seeds at `n_edges = 40` — seeds discovering the rule / mean accuracy:

   | joined constants per binding | bare      | with quiet prior |
   | ---------------------------- | --------- | ---------------- |
   | 1                            | 5/5 1.000 | 5/5 1.000        |
   | 2                            | 5/5 1.000 | 5/5 1.000        |
   | 4                            | 4/5 0.915 | 5/5 1.000        |
   | 6                            | 3/5 0.840 | 5/5 1.000        |
   | 8                            | 3/5 0.860 | 4/5 0.930        |
   | 16                           | 4/5 0.835 | 5/5 0.920        |

   Beyond roughly 4–6 joined constants per head binding the detector stops converging
   reliably without a sparsity prior. Saturation hits the **detector** before it hits
   the **selection**: at `k = 16` with the prior all 5 seeds still pick the correct
   relation, but one never converges its detector (accuracy 0.600).

5. **Identifiability — the mixture cannot rank relations it cannot distinguish.** The
   inter-candidate noisy-OR is **monotone** and the objective carries **no sparsity term
   at all** (no L1, no `weight_decay`, no simplex over candidate weights). Two candidates
   with the same extension are therefore exactly degenerate: `1 − (1−w₁m)(1−w₂m)` is
   reachable with the mass **split**, so the loss is flat between them. Measured
   (`python/tests/test_join_identifiability.py`):

   | vocabulary contains…                        | outcome                                        |
   | ------------------------------------------- | ---------------------------------------------- |
   | label-independent distractors (the demo)     | correct relation wins by **3333×**             |
   | a distractor sharing 5 of 6 of the edge's own events | correct relation still wins by **971×** |
   | a nested superset (same salient events)      | margin collapses to **1.003×** → reported as a **tie** |
   | an exact extensional duplicate               | weights equal to **12 decimals** → a **tie**   |
   | a trivially-true relation                    | **1 of 2 seeds** lands in a degenerate minimum and selects the **wrong** relation *confidently* (0.955, accuracy 0.500 — far below the head-gate baseline); the other seed recovered when the distractor became exactly class-independent |

   Three consequences, all load-bearing. **(a) `argmax` over candidate weights is not a
   selection signal.** Python's `max` returns the *first* key holding the maximum, so on
   indistinguishable relations it reports whichever the caller listed first — a confident
   wrong answer. Use `discovery.select_rule`, which claims a rule only when one candidate
   is both *believed* (weight ≥ 0.5) and *alone* at the top (runner-up > 0.01 behind),
   and abstains otherwise. **(b) Accuracy is not evidence that the relation was
   identified** — it is 1.000 in every tied case above. **(c) Weight alone catches
   ambiguity, not degeneracy** — in the trivially-true world the wrong candidate can come
   back *believed and alone* on weight, so the weight/tie gates pass it through. **The
   fit gate ships in `select_rule(fits=...)`**: the joint mixture
   (`neurosymbolic._train_joint_mixture`) exports `candidate_train_fit`, each
   candidate's TRAIN-set agreement `mean((mask ≥ 0.5) == targets)` off the final step's
   own mask, independent of its guard weight or its rank among the others. Passing it
   through (`select_rule(weights, fits=candidate_train_fit, min_fit=0.75)`) drops any
   candidate below `min_fit` *before* ranking; if none survive, the run abstains and
   names the fit gate in `reason`. Measured on the trivially-true world: the recovered
   seed's winner fits at ~1.0 and is selected exactly as before; the seed that used to
   derail selects `co_occurs` at weight 0.955 but fit 0.500 — a coin flip — and is now
   caught and abstained on rather than confidently misreported
   (`python/tests/test_join_identifiability.py`, formerly the last `xfail(strict)`).

`train_and_promote(...)` also accepts `training_fold`, `held_out_domains`,
`base_kernel_checksum_before`, and `base_kernel_checksum_after`. These fields are
recorded on `PromotionResult.rule_inventory`, along with selected and rejected
candidate clauses and gate outcomes.

#### Toward a neural body literal in the engine's own candidate space (spike finding)

The mixture above is candidate *selection* over a caller-supplied vocabulary; it does not
touch the engine's dILP enumerator, so the two induction paths stay disjoint. A natural
next step is to let a neural predicate ride in the enumerator's *own* candidate space, so
that `valid_candidates` — with its chain template `head(X, Y) :- L(X, Z), R(Z, Y)`, its
`|R|²` breadth and its recursion — searches neural-bodied rules directly. The following is
**measured by a spike, not implemented in this branch**; it records what is already true
and what the one remaining obstacle is, so the work is not restarted from scratch.

**Already true — no enumerator change needed.** The chain template's join variable `Z` is
existential (it is absent from `head_projection`). A relation with **no tuples** is still a
legal body slot: `candidate_triples_for_mask` prunes a pair only when *both* slots are
empty, and `rel_index` is keyed by relation *name*, not extension. So a candidate whose
body puts a neural predicate on `Z` is *already enumerated*. Given the neural predicate its
domain as ground tuples, the engine also *derives* it: activated alone, the triple
`(has_event, sal, plastic)` produced its head facts with the expected labels and every
query fact was covered. The structural half of the bridge exists today.

**The one obstacle — the credit is linear coverage.** `compute_ilp_loss_grad_gpu` builds a
**binary** coverage matrix (`ilp_credit.cu`: `credit[f] = Σ_c A[f,c]·p_c`, `A ∈ {0,1}`,
CSR with column indices and no value array), and its gradient is hand-injected into torch
(`cand_probs.detach()` in, `cand_probs.backward(credit_grad)` out). To carry a per-event
neural probability the credit must become real-valued —
`s_c(f) = 1 − Π_{z ∈ ext_c(f)} (1 − p_net(z))` for a neural candidate, `A[f,c]` otherwise —
where `ext_c(f)` is read from the engine exactly as Stage B already does
(`join_bodies.read_join_extension`). A host-side torch reimplementation of that credit
trains end to end. But because the combination is a **sum**, a crisp relational rule
(`s ∈ {0,1}`) dominates a soft neural rule (`s < 1`) whenever both explain the data: in the
spike, with the true join isolated the neural rule won (weight 0.9995, detector separation
0.81), but adding one perfect relational competitor flipped selection to it (0.997 vs
0.0007) even though the detector still learned. **The open design question — make the
neural score enter as a product, or calibrate the credit so a soft-but-correct rule is not
dominated by a crisp coincidental one — must be settled before any Rust credit-value
kernel work.** Crossing onto this path also inherits the enumerator path's entropy
regularization, temperature schedule and holdout arbiter — the Occam pressure the mixture
lacks (see limit 5).

## Artifact Persistence

`LearnedArtifact` captures the full training result for reproducibility:

```python
artifact.save("artifact.json")   # JSON with SHA-256 candidate-map hash
loaded = LearnedArtifact.load("artifact.json", verify_hash=True)
```

Schema version: `beta-v1`. Fields: discovered rule, logits, candidate map, config, telemetry,
precision/recall, metadata (timestamp, schema version, candidate map hash).

## GPU Contract

The training step loop obeys XLOG's GPU-resident contract:

- `evaluate_device()` — no host reads for semantic results
- `batch_fact_membership_device()` — returns a CUDA bool mask via DLPack with zero semantic-loop device-to-host transfer
- `batch_tagged_credit_device()` — returns CSR-style CUDA credit data via DLPack with zero semantic-loop device-to-host transfer
- `batch_fact_membership()` / `batch_tagged_credit()` remain available when host materialization is desired
- `AtomicU64` device-to-host counter on `CudaKernelProvider` — hard gate raises if `download_column_*` is observed during step loop
- `host_transfer_stats()` / `reset_host_transfer_stats()` expose broader host transfer accounting for profiling
- Legacy `set_rule_mask_sparse()` still performs a control-plane soft-probability download; the selected-candidate sparse path avoids it

## Testing

- **86+ static test functions** across ILP Python test files (expanded by parametrized GA/beta gates)
- **Reliability gate**: 20 consecutive `train_only()` runs must all converge (20/20 pass)
- **GA reliability gate**: default 50-seed statistical run (`test_ilp_ga_reliability.py`)
- **GA performance/transfer tests**: `forward_p95_us` + host transfer accounting (`test_ilp_performance.py`)
- **Dense/sparse parity**: every sparse-path test has a `debug_dense_mask=True` variant
- Rust-side: `ilp_integration_tests.rs`, `ilp_kernel_tests.rs`
- CUDA certification: `extract_nonzero_indices` covered by kernel test suite

## Design Documents

The dILP design went through many internal iterations before converging on the
RFC. The live references are:

| Document | Content |
|----------|---------|
| `rfc-tensorized-ilp` (internal RFC, retired from the docs site) | Full RFC: mathematical foundation, hardware rationale, implementation map, and resolved design decisions |
| `dilp-showcase-report` (internal validation report, retired) | Validation: four-stage showcase run analysis |
| `external-consumer-diagnostics` (retired) | External-consumer issue resolutions and reusable diagnostics |

## See Also

- [Python API reference](/reference/python) — user-facing API reference
- [Diagnostics guide](/guides/diagnostics) — proof traces, rule inventories, runtime audits, and transfer metrics
- [GPU Execution](/architecture/gpu-execution) — mask DAG evaluation, stream compaction
- [Probabilistic reasoning](/probabilistic/engines) — XGCF circuits, provenance (shared infrastructure)
- [Interop guide](/guides/interop) — DLPack details
