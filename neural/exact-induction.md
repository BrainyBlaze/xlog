# Bounded exact induction

A non-gradient, GPU-native rule miner — it enumerates every candidate 2-body Datalog rule across four fixed topologies and returns the top-K per topology, deterministically.

Bounded exact induction finds the best simple rules that explain your data,
without any training loop and without randomness. You give it a target relation,
a list of candidate relations, and some positive and negative example pairs. It
scores every possible two-part rule on the GPU and hands back the highest-scoring
rules for each rule shape.

The payoff is repeatability. Because it *enumerates* every rule rather than
*searching* with gradients, the same inputs always produce the exact same ranked
output — no seeds, no temperature, no run-to-run drift.

This is the exhaustive counterpart to
[differentiable ILP](/neural/rule-learning) (inductive logic programming — learning
logical rules from labelled examples). That page learns a clause with gradients;
this one enumerates the whole space instead.

## When to use this

Reach for bounded exact induction when you want to mine short rules from example
pairs and you need the answer to be exactly reproducible.

It fits when your rules have the shape `H(X, Y) :- <two body atoms>` — a head
relation `H` explained by exactly two candidate relations. If you need longer rules,
soft/approximate matches, or a trained clause, use
[differentiable ILP](/neural/rule-learning) instead.

## The smallest example

`induce_exact` takes a compiled program, the head and candidate relation names, and
the positive (and optional negative) example pairs as device tensors:

```python
from pyxlog.ilp import induce_exact

result = induce_exact(
    prog,                        # CompiledIlpProgram
    head_relation="p_A",
    candidate_relations=["p_B", "p_C", "p_D"],
    positive_arg0=pos_a0,        # 1-D device torch tensors
    positive_arg1=pos_a1,
    negative_arg0=neg_a0,        # optional
    negative_arg1=neg_a1,
    k_per_topology=2,
    backend="native",
)

for cand in result.candidates:
    print(cand.topology, cand.left_rel_idx, cand.right_rel_idx,
          cand.positives_covered, cand.negatives_covered)
```

Each printed line is one ranked rule: its shape, the ids of the two body relations,
and how many positive and negative examples it covered. The exact counts depend on
your data, so the output looks like this (illustrative):

```
chain 0 1 14 0
star 2 2 9 1
```

`induce_exact` returns an `ExactInductionResult` whose `candidates` list holds the
ranked `ScoredCandidate` records. Each record carries its topology, the left and
right relation ids, positive and negative coverage counts, and its local rank.

Two edge cases are handled for you. When you pass no negatives, the engine
synthesizes an empty negative buffer so the kernel signature stays uniform. When
there are no candidates or no positives, it returns an all-zero result rather than
launching.

## Confirm it worked

You get a `ScoredCandidate` for the top rules of each shape, ordered by rank. A
rule that covers many positives and few negatives is a good rule; the ranking puts
those first.

To check that the run stayed on its efficient single-transfer path, read
`prog.d2h_transfer_count()` — see [One counted transfer per call](#one-counted-transfer-per-call).

## Four fixed topologies

A "2-body" rule has a head `H(X, Y)` and two body atoms drawn from candidate
relations `L` (left) and `R` (right). A **topology** is simply the way those two
atoms share variables. The engine considers exactly four:

| Topology | Rule shape | A pair `(x, y)` is covered when |
|---|---|---|
| `chain` | `H(X,Y) :- L(X,Z), R(Z,Y)` | some `z` has `(x, z)` in `L` and `(z, y)` in `R` |
| `star` | `H(X,Y) :- L(X,Y), R(X,Y)` | `(x, y)` is in both `L` and `R` |
| `fanout` | `H(X,Y) :- L(X,Z), R(X,Y)` | `(x, y)` is in `R` and `x` has some outgoing `L` edge |
| `fanin` | `H(X,Y) :- L(X,Y), R(Z,Y)` | `(x, y)` is in `L` and `y` has some incoming `R` edge |

Each `(topology, L, R)` combination is scored in isolation against its own template.
The kernel checks these four coverage rules directly on the candidate row sets; it
does not route through general rule evaluation. As a result, no candidate's score can
leak into another's, by construction.

## How a call runs

One CUDA launch covers the whole sweep. The grid is `(C, C, 4)` blocks — one block
per `(left, right, topology)` triple over the `C` candidate relations.

Each block scans all positive and negative query pairs for its triple and writes
exactly one coverage slot. Because every block owns a distinct output slot, the
scoring path never has two threads updating the same location (no cross-block
atomics). That is what makes the result bit-for-bit identical on every run.

After scoring, the engine selects the top-`K` per topology on the device. The host
then applies a fixed dictionary-order (lexicographic) sort to break ties, so equal
scores always resolve the same way.

### Column types

The kernel picks its code path by column type. `u64` columns use one kernel; `u32`
and `symbol` columns use another. Symbol columns keep their logical schema and are
never silently narrowed to a smaller integer type.

A request that mixes `u32` and `symbol` candidate types is rejected with a typed
error rather than coerced into one type.

## One counted transfer per call

The scoring sweep does no host round-trips — nothing is copied back from the GPU to
the CPU while scoring runs.

The setup uploads (a small candidate-offset array, and a device-to-device column
concatenation) are constant-size, and they stay on the device — they are not
device-to-host (GPU-to-CPU) transfers.

The production `induce_exact` call copies results back from the GPU exactly
**once**: a single compact export of the selected top-`K` rows, no matter how many
candidates, queries, or topologies are involved. You can verify this with
`prog.d2h_transfer_count()`, which counts those GPU-to-CPU transfers.

<Note>
You may see "two count-array transfers" mentioned for the parity and
chain-shared-memory tests. That is a separate diagnostic path that reads back the raw
per-slot count arrays to compare against a reference implementation; it is test-only
accounting. The production `induce_exact(backend="native")` path counts exactly one
device-to-host transfer.
</Note>

## Availability

<Note>
Bounded exact induction ships in the `pyxlog` PyPI wheel that tracks the `v0.9.2`
release, alongside the rest of the neural-symbolic surface. Its CUDA kernel is not yet
in the formal certification registry, because its compiled GPU code (PTX) is not
committed — see [CUDA certification](/architecture/certification).
</Note>

## See also

<Card title="Rule learning (differentiable ILP)" icon="brain" href="/neural/rule-learning">
  The gradient-trained counterpart — learn a clause with Gumbel-Softmax masking and
  promote it through reliability gates.
</Card>

<Card title="Diagnostics and provenance" icon="clipboard-list" href="/guides/diagnostics">
  Audit records for mined rules — support rows, rejected alternatives, and selection
  traces.
</Card>
