# Neural join bodies in the joint mixture — selecting the rule's join relation

`examples/plasticity_incircuit/` recovers a **hand-written** existential-join rule.
This one does not get the rule. It gets a relation **vocabulary**, and has to find
which relation the rule joins on — while learning the neural predicate from scratch.

> **Read this before quoting the numbers.** This is candidate **selection**, not rule
> induction: the body template is fixed and has exactly one free slot (the relation
> name), and the vocabulary is supplied by the caller. The search is `|R|` candidates
> wide — no conjunctions, no chaining, no recursion, no negation — and it does **not**
> use the engine's dILP enumerator. The genuinely new thing is the **neural predicate on
> an existential variable, trained through the logic**. See limits **5** and **6** below.

A vocabulary goes in:

```
pre_before_post   post_before_pre   co_occurs
```

One same-head Stage-B candidate per relation comes out (`build_join_candidates`),
and they all compete in **one joint mixture**:

```
trainable_rule(cand_pre_before_post, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
trainable_rule(cand_post_before_pre, weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
trainable_rule(cand_co_occurs,       weight=0.0) :: plastic(E) :- saliency(Ev, strengthen), co_occurs(Ev, E).
```

**No candidate is hand-written.** Each candidate's mask is the OR, over the join
extension **read from the engine** (`relation_facts` — the edge→events map is never
handed to the trainer), of the network's **per-event** probability:

```
mask_k[edge] = 1 − ∏_{e : r_k(e, edge)} (1 − p_saliency(e))
```

The mixture has to put its mass on the relation whose join extension actually carries
the planted signal, and the detector has to be learned per **event**, not per edge.

## Planted world

An edge is plastic iff **some** of its pre→post events is salient (feature > 0.5). A
positive edge carries **exactly one** salient event out of six. That is deliberately
the shape a head-bound gate on a *pooled* per-edge feature cannot recover — the lone
salient event dilutes — and it is why the accuracy below is printed against a
**0.847** baseline ceiling.

The two distractor relations are **equal-cardinality**: each hands every edge the same
number of events as `pre_before_post`, drawn from *other* edges. So their noisy-OR is
exactly as sharp (no structural advantage) and they carry no label information. There
is nothing to win on but the truth.

## Run

Requires a CUDA build of `pyxlog`.

```
python run_demo.py
```

## Output (verified, RTX 3090)

```
Planted rule : plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
               (an event is salient iff its feature > 0.5; a positive edge has exactly ONE
                salient event out of 6)
Vocabulary   : ['pre_before_post', 'post_before_pre', 'co_occurs']
World        : 40 edges, 240 events, 15 of 40 edges plastic

candidates: NONE hand-written — generated from the vocabulary
  cand_pre_before_post     plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
  cand_post_before_pre     plastic(E) :- saliency(Ev, strengthen), post_before_pre(Ev, E).
  cand_co_occurs           plastic(E) :- saliency(Ev, strengthen), co_occurs(Ev, E).

The distractors are equal-cardinality (the same number of events per edge as
the correct relation, drawn from OTHER edges): their OR is exactly as sharp and
they carry no label information. There is nothing to win on but the truth.

loss 0.8353 -> 0.0049

Candidate weights in the joint mixture:
  cand_pre_before_post     weight=0.99975  <-- DISCOVERED
  cand_post_before_pre     weight=0.00030
  cand_co_occurs           weight=0.00028

Discovered rule: plastic(E) :- saliency(Ev, strengthen), pre_before_post(Ev, E).
  planted rule joined on 'pre_before_post' -> CORRECT

Learned per-event P(saliency=strengthen) — one detector, 240 events:
  planted-salient events (feature > 0.5), n=15  mean P=0.9945  min P=0.9649
  planted-quiet   events (feature <= 0.5), n=225 mean P=0.0006  max P=0.0077
  first eight events:
    event 0   feature=0.306           P(strengthen)=0.0003
    event 1   feature=0.174           P(strengthen)=0.0000
    event 2   feature=0.111           P(strengthen)=0.0000
    event 3   feature=0.209           P(strengthen)=0.0000
    event 4   feature=0.168           P(strengthen)=0.0000
    event 5   feature=0.316           P(strengthen)=0.0004
    event 6   feature=0.786  salient  P(strengthen)=0.9997
    event 7   feature=0.238           P(strengthen)=0.0000

Generalization — the detector is a FUNCTION of the feature, not an id lookup.
  the world's features span [0.010, 0.940]; 0.95 and 0.005 lie strictly
  OUTSIDE that support, so classifying them is extrapolation, not interpolation:
    feature=0.95   P(strengthen)=1.0000  (expect > 0.5)
    feature=0.02   P(strengthen)=0.0000  (expect < 0.5)
    feature=0.005  P(strengthen)=0.0000  (expect < 0.5)

Accuracy vs the planted labels: 1.000 (40/40 edges)
  head-gate baseline ceiling on this world shape: 0.847
  (ONE gate on a pooled per-edge feature cannot beat 0.847 here; a per-event
   detector under the join's OR has no such ceiling)

RULE DISCOVERED ✓
```

Discovery is stable across seeds: 5/5 seeds pick `cand_pre_before_post`. The contract
is pinned by `python/tests/test_join_discovery.py` (CUDA-gated).

## The `domain_ids` contract

`domain_ids` is the **one** map from a domain constant to its feature row, and **both**
engines — the exact d-DNNF circuit and the torch-side mixture — resolve rows through
it:

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

The ids must be distinct; they may be in any order and need not be contiguous. Omitting
`domain_ids` defaults it to `[0 … D-1]` — which is what this demo's dense world happens
to be. It is passed explicitly anyway: the row↔constant correspondence is the caller's
to declare, never something the trainer should infer from an ordering.

## HONEST SCOPE — what this does NOT do

**1. One join network per program.** `domain_inputs` currently supports a single join
network. A program with two neural predicates on join variables is not supported.

**2. Head arity must be 1.** The multi-outcome form of the rule —

```
plastic(Edge, L) :- saliency(Event, L), pre_before_post(Event, Edge).
```

— **does not compile.** The mixture's eligibility call is fixed at arity 1. The head is
a single-outcome predicate over one binding variable. **Multi-outcome plasticity
(strengthen/weaken as a learned label on the head) is not claimed and does not work.**

**3. The inter-candidate noisy-OR is a MODELLING CHOICE, not compiled semantics.** The
semantics anchor pins each candidate's **per-candidate** mask against the exact d-DNNF
circuit — the torch-side OR reproduces the circuit to ~2e-07 (tolerance 1e-4) on four
domain layouts: dense, sparse, superset and shuffled. But the rule that **combines**
several candidates into one head probability has **no exact-circuit counterpart, and
cannot have one**: declaring more than one `trainable_rule` is precisely what routes
execution away from the circuit and into the torch-side mixture. The single-candidate
case is anchored. The multi-candidate combination is a model we chose.

**4. Saturation limit.** The noisy-OR saturates as the number of joined constants per
head binding grows: with `k` events at per-event probability `p` the mask is
`1 − (1−p)^k`, so at the default init (`p ≈ 0.5`) every binding starts at ~1, the
gradient to the detector vanishes, and the optimizer's cheapest descent is to kill the
detector outright. The mitigation is a **quiet prior** — the detector's initial logit
for the positive label shifted by `−2.0`, i.e. the prior that events are mostly quiet,
which this world satisfies by construction. It is an *initialization*, not an
assertion, and it is **load-bearing**. Measured over 5 seeds at `n_edges=40`
(seeds-discovering-the-rule / mean accuracy):

| events/edge | bare            | with quiet prior |
| ----------- | --------------- | ---------------- |
| 1           | 5/5 &nbsp;1.000 | 5/5 &nbsp;1.000  |
| 2           | 5/5 &nbsp;1.000 | 5/5 &nbsp;1.000  |
| 4           | 4/5 &nbsp;0.915 | 5/5 &nbsp;1.000  |
| 6           | 3/5 &nbsp;0.840 | 5/5 &nbsp;1.000  |
| 8           | 3/5 &nbsp;0.860 | 4/5 &nbsp;0.930  |
| 16          | 4/5 &nbsp;0.835 | 5/5 &nbsp;0.920  |

Beyond roughly 4–6 joined constants per head binding the detector stops converging
reliably **without** the prior. Without it the failure is a genuine degenerate,
*inverted* minimum, not slow convergence: at seed 0, `k=6`, bare, the loss sits at
0.640 — the base-rate entropy — at 1500, 3000, 6000 **and** 12000 steps, with the
*wrong* candidate hardened to weight 1.0. More steps do not rescue it.

And saturation hits the **detector** before it hits the **selection**: at `k=16` with
the prior, all 5 seeds still pick the correct *relation*, but one never converges its
detector (accuracy 0.600). This example runs at `k=6` with the prior — inside the
regime we can stand behind.

**5. Identifiability — the 3333× margin is a property of THIS WORLD.** The distractors
here are built to carry **zero** label information (equal cardinality, drawn from other
edges' events), so exactly one candidate is fittable at all, and the correct relation has
perfect precision by construction while each distractor false-fires on ~41 % of edges.
That gap is what `0.99975 vs 0.0003` measures. The mechanism has no Occam term to fall
back on: the inter-candidate noisy-OR is **monotone** and there is **no sparsity
pressure anywhere** — no L1, no `weight_decay`, no simplex over the weights. Measured
(`python/tests/test_join_identifiability.py`):

| the vocabulary also contains…                            | outcome                                          |
| -------------------------------------------------------- | ------------------------------------------------ |
| nothing (the demo)                                        | correct relation wins by **3333×**               |
| a distractor holding 5 of 6 of the edge's own events      | correct relation **still wins by 971×**          |
| a nested superset (same salient events + quiet ones)      | margin collapses to **1.003×** → reported a **tie** |
| an exact extensional duplicate                            | weights agree to **12 decimals** → a **tie**     |
| a trivially-true relation                                 | **1 of 2 seeds derails** into a degenerate minimum: the **wrong** relation is selected, *confidently* (0.955), at accuracy **0.500** — far below the 0.847 head-gate baseline. The other seed recovered when the distractor composition became exactly class-independent |

Partial overlap is survivable, and that is a real result. Exact and near-duplicates are
**not identifiable at all** — and there `select_rule` abstains, which is the right
answer. The trivially-true case is worse and is **measured, not fixed** (`xfail`,
`strict`): the mixture comes back **believing the wrong rule**, so the tie and
min-weight gates pass it straight through. **Degeneracy is not ambiguity.** Closing it
needs two things this branch does not have: an **Occam/sparsity term** on the candidate
weights, so the objective prefers the *minimal* separating relation, and a **fit gate**,
so a rule the model cannot actually fit the data with is not reported as a rule.

Two rules follow:

- **`argmax` over the candidate weights is not a selection signal.** Python's `max`
  returns the *first* key holding the maximum, so on two indistinguishable relations it
  reports whichever the caller **typed first** — reverse the vocabulary and the
  "discovered rule" reverses with it. Use `discovery.select_rule`: it names a rule only
  when one candidate is both *believed* (weight ≥ 0.5) and *alone* at the top (runner-up
  more than 0.01 behind), and **abstains** otherwise.
- **Accuracy is not evidence of selection.** It is **1.000** in every tied row above.
  The demo prints the two side by side; they are independent claims.

**6. This is selection, not induction.** `|R|` candidates from one fixed template with
one free slot, over a caller-supplied vocabulary. No conjunctions, no chaining through
an intermediate variable, no recursion, no negation, head arity 1. The engine's own dILP
enumerator (`valid_candidates`: `|R|²` chained candidates, with recursion) is **not on
this code path** — the two induction paths in xlog remain disjoint. Widening this by one
honest notch means admitting head-bound filter conjuncts (`…, f(E).`), which the OR math
already supports; a real bridge means teaching `valid_candidates` to enumerate
neural-bodied candidates, which this work does not do.
