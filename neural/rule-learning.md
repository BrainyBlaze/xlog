# Rule learning (differentiable ILP)

Train neural predicates and learn Datalog rules end-to-end — one differentiable path from PyTorch tensors through the GPU circuit to a promoted symbolic clause.

You can teach an xlog program two ways at once: write the rules you already know,
and let the engine discover the rules you don't. A loss you compute on a symbolic
query flows all the way back into your PyTorch network's weights, so logic and
learning train together instead of in separate stages.

Two features make this work, and they share one machinery:

- A **neural predicate** turns a PyTorch network into a logical relation. Its facts
  carry probabilities that the network produces.
- **Differentiable ILP** (Inductive Logic Programming — learning a rule from
  examples) searches candidate Datalog clauses with gradient descent and promotes
  the winner into your program.

Both compile down to the same circuit and use the same gradient bridge. That is why
a single loss can reach from a symbolic answer back to network weights.

<Frame caption="A loss on a symbolic query backpropagates through the cached circuit into the network's weights; only weights change between iterations.">
  <img className="block dark:hidden" src="/assets/diagrams/neural-symbolic-loop-light.svg" alt="Neural-symbolic training loop: a PyTorch network produces predicate probabilities, the cached XGCF circuit evaluates the query probability, a loss is computed against the target, and gradients flow back through the circuit to the network weights." />
  <img className="hidden dark:block" src="/assets/diagrams/neural-symbolic-loop-dark.svg" alt="Neural-symbolic training loop: a PyTorch network produces predicate probabilities, the cached XGCF circuit evaluates the query probability, a loss is computed against the target, and gradients flow back through the circuit to the network weights." />
</Frame>

## When to use this

Reach for a **neural predicate** when you already know the rule but a fact's truth
depends on raw data — an image, an embedding, a sensor reading — and you want a
network to supply that truth.

Reach for **differentiable ILP** when you have examples of what should be true and
want the engine to find the clause that explains them, rather than writing it by
hand.

<Note>
This whole surface — neural predicates, differentiable ILP, and exact induction —
ships only through the `pyxlog` PyPI wheel, which tracks the `v0.10.0` release. It
is a Python-first API. The crates published on crates.io do not expose it, so
install `pyxlog` to use anything on this page.
</Note>

## Neural predicates

A neural predicate declares that a relation's truth is decided by a network rather
than by stated facts. You write the declaration in the program with `::`, binding a
registered network name to a predicate head:

```xlog
nn(coin_net, [X], Y, [heads, tails]) :: coin(X, Y).
```

This is the four-argument form, `nn/4`. It reads: for input `X`, the network
`coin_net` produces a distribution over the labels `[heads, tails]`, and that
distribution becomes the probability of `coin(X, heads)` and `coin(X, tails)`. The
label list makes this a **classification** predicate — the network output is a
distribution over categorical labels.

Drop the label list and you get the three-argument form, `nn/3`, an **embedding**
predicate:

```xlog
nn(embed_net, [X], E) :: node_embedding(X, E).
```

Here the network maps an input to a learned vector `E` instead of a label
distribution. A given network may be declared in one form or the other, but not
both. Declaring the same name as both classification and embedding is rejected at
compile time.

<Note>
Declarations are validated when the program compiles: input and output variables
must be disjoint, every argument must appear in the bound predicate, and anonymous
or aggregate variables are not allowed in a neural declaration.
</Note>

## Registering networks

The declaration only names a network. You supply the actual `torch.nn.Module` from
Python at registration, which is where xlog wires the network into PyTorch's
automatic-differentiation (autograd) machinery so gradients can flow through it.

- `register_network(name, module, optimizer, scheduler=None, ...)` binds a
  classifier, or `nn/4` head. You pass the module and its optimizer, and optionally a
  learning-rate scheduler. The bridge holds them and drives the forward and backward
  passes through the circuit. Batching, top-`k` truncation, determinism, and a call
  cache are configurable here.
- `register_embedding(name, module_or_tensor, trainable=True)` binds an `nn/3`
  embedding — either a module or a plain tensor of vectors. Set `trainable=False` to
  freeze it.

Once a network is registered, evaluating a query that touches the predicate does
three things. It runs the network. It converts the output into probabilities on the
circuit's input nodes. And during training, it pushes gradients back through those
nodes into the module's parameters.

The bridge also exposes `forward_backward(query, expected=True)`, several
differentiable losses (`belnap_loss`, `semantic_loss_tensor`, `mse_loss_tensor`),
and epoch drivers (`train_epoch`, `train_model`) with optional validation-set early
stopping.

## Differentiable ILP

Writing a neural predicate assumes you already know the rule. Differentiable ILP is
for the opposite case: you have examples of what should be true and want the engine
to **discover the clause**. xlog frames this as learning which body to attach to a
head.

You mark a clause as learnable with a mask annotation:

```xlog
learnable(mask) :: reach(X, Y) :- edge(X, Z), edge(Z, Y).
```

Compiling one candidate rule at a time is impossible at training timescales. So xlog
instead pre-compiles a single tensorized graph — call it a **super-graph** — that
holds every syntactically legal candidate body at once. A continuous **mask** tensor
then decides which candidates are active.

Each candidate gets a score (a logit). A **Gumbel-Softmax** relaxation — a
differentiable way to make a soft, random pick among discrete options, controlled by
a temperature `τ` — turns those scores into a soft selection over candidates. The
circuit evaluates under that soft mask, a loss compares the derived facts against
your examples, and the gradient updates the scores.

As training proceeds, the temperature `τ` anneals toward a floor on a cosine
schedule. Lower temperature sharpens the soft mask toward a single hard choice
(one-hot). At convergence the engine takes `argmax` over the scores — it keeps the
single highest-scoring candidate. That winner is decoded back into a concrete Datalog
clause: a discrete rule string, not a soft mixture.

### Sparse and dense mask backends

How the mask reaches the executor is chosen by a backend:

| Backend | Parameters | When to use |
|---|---|---|
| `SparseMaskBackend` (default) | `O(C)` — one logit per candidate | Production. Ranks candidates on the device and pushes only the selected sparse set; no cubic tensor is materialized. |
| `DenseMaskBackend` | `O(N³)` — full schema cube | Debugging and parity checks. Enable with `debug_dense_mask=True`; it materializes the whole mask, so it is expensive on large schemas. |

The sparse backend is what keeps training inside the GPU-resident hot loop: candidate
probabilities are ranked and applied on the device, without downloading a full mask
vector to the host CPU.

## The Python training API

Two entry points drive learning, both in `pyxlog.ilp`.

`train_only(...)` runs the search. It enumerates the legal candidates and launches
several independent restarts from fresh scores. Each restart iterates the step loop:
apply the mask, evaluate on the device, compute the loss, back-propagate, step the
optimizer, anneal `τ`, and stop early once the `argmax` winner is stable and the loss
is below threshold. It returns the discovered rule and its training telemetry.

`train_and_promote(...)` wraps `train_only` with a set of **promotion gates** —
checks a discovered rule must pass before xlog commits it into your program. A rule
is written into your committed source only if it clears every gate:

| Gate | What it checks |
|---|---|
| Convergence | Training actually reached a stable `argmax` winner. |
| Novel-rate | The rule does not over-derive facts outside the examples. |
| Protected-relation | It produces no unwanted side effects on protected relations. |
| Holdout F1 | F1 on held-out examples meets the threshold (leave-one-out for small sets, k-fold for larger). |
| Ambiguity | No alternative candidate is a near-equal winner. |
| Typed-schema | Relation type metadata is present (or a reviewed waiver is supplied). |

A rule that fails any gate is reported along with the gate it failed. It is never
silently promoted.

## Reliability gates

The search is stochastic, so different random seeds can converge or not. To keep
that honest, dILP is held to seed-level reliability gates that grow stricter as the
subsystem matures. A gate like `5/5` means "five consecutive runs must all succeed."

| Maturity | Gate | Meaning |
|---|---|---|
| Alpha | `5/5` | Five consecutive `train_only` runs must all converge. |
| Beta | `20/20` | 5 seeds across each of 4 stages (reach, grandparent, colleague, plus-2) on the sparse backend — all 20 converge. |
| GA | `50/50` | 50 seeds pass with a Clopper–Pearson lower bound on the success rate. |

<Note>
The GA gate is `50/50` evaluated with a **Clopper–Pearson** lower confidence bound —
a conservative statistical bound on the true success rate, not a loose "50-seed run."
There is no `200/200` gate anywhere in the suite.
</Note>

## A runnable `nn/4` example

This is a self-contained DeepProbLog-style program. A small classifier decides
whether each coin image shows heads or tails, symbolic rules define winning and
losing, and the network is trained purely from the derived symbolic queries. The
synthetic tensors let it run without a dataset download.

```xlog
nn(coin_net, [X], Y, [heads, tails]) :: coin(X, Y).

win(X, Y)  :- coin(X, heads), coin(Y, heads).
lose(X, Y) :- coin(X, tails).
lose(X, Y) :- coin(Y, tails).
```

```python
import torch
import torch.nn as nn
import pyxlog

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 2-class classifier over 3x64x64 coin images -> {heads, tails}
class CoinNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 64 * 64, 64), nn.ReLU(), nn.Linear(64, 2)
        )
    def forward(self, x):
        return torch.log_softmax(self.net(x), dim=-1)  # log-probs over [heads, tails]

# Compile the program and bind the network to the nn(coin_net, ...) declaration.
program = pyxlog.Program.compile(open("coins.xlog").read())
net = CoinNet().to(DEVICE)
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
program.register_network("coin_net", net, opt)

# Synthetic labelled coin images (even idx -> heads, odd idx -> tails).
labels = [i % 2 for i in range(16)]                    # 0=heads, 1=tails
images = torch.zeros(16, 3, 64, 64, device=DEVICE)
for i, y in enumerate(labels):
    images[i, y] = 1.0                                 # class-separable signal
program.add_tensor_source("train", images)

# Supervised atoms: coin(i, heads|tails). The label distribution is trained
# by back-propagating NLL through the circuit.
classes = ["heads", "tails"]
queries = [f"coin({i}, {classes[y]})" for i, y in enumerate(labels)]

for epoch in range(10):
    program.train_epoch(queries, batch_size=8)

# Inference: query a derived symbolic head backed by the trained network.
print(program.query("win(0, 2)"))   # both even -> both heads -> fires high
print(program.query("lose(1, 3)"))  # odd -> tails -> lose fires high
```

**How you know it worked.** After training, `program.query("win(0, 2)")` returns a
high probability (both inputs are even, so both classify as heads, so `win` fires),
and `program.query("lose(1, 3)")` also returns a high probability (odd inputs
classify as tails, so `lose` fires). If both queries stay near their untrained
values, the network did not learn the coin signal.

The optimizer here is Adam. On this multiplicative loss surface plain SGD tends to
plateau, so Adam is the practical default.

## New in the 0.10.0 wheel

These extend the neural-symbolic surface and ship in the `pyxlog` 0.10.0 wheel.

- **Existential-join trainable bodies (Stage B).** A trainable body may join a
  neural predicate to an ordinary relation on a variable that is not in the head. The
  neural predicate is grounded over the real join domain inside the circuit, then
  combined at the head with a logical OR. Per-event features arrive through a
  `domain_inputs=` channel and `register_domain_tensor_source`. The join domain must
  be ground facts, head-binding ids must be `0..N-1`, and only a single join network
  is supported. The planted graph is bounded (roughly six to seven events) by a fixed
  circuit buffer.
- **Joint multi-rule mixtures.** Several trainable clauses that share a head can be
  learned jointly as a noisy-OR mixture — a standard way to combine independent
  pieces of evidence into one probability. `evaluate_joint_mixture` reads the
  held-out result. A candidate may be guard-only, or it may carry a neural conjunct
  (`neural_bodies=`). That conjunct is a small trained head whose output is
  hard-thresholded (using a straight-through estimator, which passes gradients as if
  the threshold were smooth). The thresholded output acts as an on/off gate on the
  candidate's eligibility, so it stays a derivation gate rather than soft truth mass.
- **Graded per-binding candidate masses.** `train_neurosymbolic_program(...,
  candidate_masses={rule_id: tensor})` supplies per-binding confidences in `[0, 1]`.
  Each confidence multiplies into a candidate's eligibility, so the head probability
  becomes the noisy-OR over graded evidence masses. Map head bindings to world steps
  and masses to a fact's per-step confidence, and the guards train against an
  evolving trajectory. Omit the argument and the plain binary behavior is unchanged.
- **GPU-resident zero-host training step.** `forward_backward_grouped(queries,
  expected)` performs one host synchronization per step instead of a per-query
  round-trip, keeping the training loop resident on the device.

<Note>
The higher-level neuro-symbolic driver
(`pyxlog.ilp.neurosymbolic.train_neurosymbolic_program`, with declarative
`trainable_rule(...)` / `train(...)` in-source) is where the Stage-B `domain_inputs=`
channel is exposed.
</Note>

## See also

<Card title="Bounded exact induction" icon="magnifying-glass" href="/neural/exact-induction">
  The non-gradient counterpart — deterministic GPU enumeration of candidate 2-body
  rules across four fixed topologies.
</Card>

<Card title="Python bindings" icon="python" href="/reference/python">
  The full `pyxlog` API surface for registration, training, and querying.
</Card>

<Card title="Arrow and DLPack interop" icon="arrows-left-right" href="/guides/interop">
  How XLOG results and gradient tensors cross into PyTorch without a host round-trip.
</Card>
