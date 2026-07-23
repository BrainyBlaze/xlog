# Probabilistic engines

Attach probabilities to facts and let XLOG compute marginals and conditionals — exactly by GPU knowledge compilation, or approximately by a GPU-resident Monte Carlo megakernel.

Mark some facts as uncertain, tell XLOG what you have observed, and ask how
likely a conclusion is. You write ordinary rules; XLOG carries the probabilities
through them and hands back a probability.

You can ask for two kinds of answer:

- a **marginal** — how likely a fact is, overall;
- a **conditional** — how likely a fact is, *given* what you observed.

XLOG answers with one of two engines: an **exact** engine that computes the true
probability, or an **approximate** Monte Carlo engine that estimates it by
sampling and reports error bars.

## Smallest runnable example

The classic wet-grass model: rain and the sprinkler each make the grass wet, you
observe that the grass is wet, and you ask how likely each cause was.

```xlog
0.7::rain().
0.2::sprinkler().

wet() :- rain().
wet() :- sprinkler().

evidence(wet(), true).

query(rain()).
query(sprinkler()).
```

Run it with the exact engine:

```bash
xlog run wet-grass.xlog
```

**What you should see.** Because the grass is wet, both causes come back more
likely than their starting probabilities (`0.7` and `0.2`), and `rain()` — the
stronger prior — carries most of the posterior mass.

## When to use each engine

Reach for the **exact** engine when you need the true answer and your program is
small — for example when it uses negation, recursion, or small aggregates.

Reach for the **Monte Carlo** engine when the model is larger, or uses a feature
the exact engine does not support, and a sampled estimate with error bars is good
enough.

| Engine | Pragma | What it computes | When to reach for it |
|---|---|---|---|
| Exact | `exact_ddnnf` | Exact marginals and conditionals via knowledge compilation, with gradients | Negation, recursion, and small aggregates where you need an exact answer |
| Monte Carlo | `mc` | Sampled estimates with a confidence interval | Larger models, or anything outside the exact engine's supported fragment |

## Writing a probabilistic program

A **probabilistic fact** is a fact prefixed with a probability and `::`. It holds
with that probability and fails otherwise:

```xlog
0.7::rain().
0.2::sprinkler().
```

An **annotated disjunction** puts several mutually exclusive outcomes on one line,
separated by `;`, each with its own probability. Exactly one outcome is chosen. If
the probabilities sum to less than `1`, the remaining mass is an implicit "none"
outcome:

```xlog
0.6::coin(heads); 0.4::coin(tails).
```

You state what you observed with `evidence(...)` and ask for a probability with
`query(...)`:

```xlog
evidence(wet(), true).
query(rain()).
```

Rules are written exactly as in a deterministic program. The probability lives on
the facts, and the engine propagates it through the derivations.

<Note>
Select the engine with `#pragma prob_engine = exact_ddnnf` or
`#pragma prob_engine = mc`. From Python, the `prob_engine` argument to
`Program.compile(...)` takes precedence over the pragma.
</Note>

## Exact inference

The exact engine (`exact_ddnnf`) computes the true probability by turning your
program into a circuit and then adding up the probability of every world in which
the query holds. Turning a program into a circuit like this is called **knowledge
compilation**; adding up those world probabilities is **weighted model counting**.
The whole path runs on the GPU:

<Frame caption="The exact path compiles provenance to CNF, then to a verified GPU Decision-DNNF, then to the XGCF circuit; the circuit is compiled once and evaluated many times, with gradients flowing back through it.">
  <img className="block dark:hidden" src="/assets/diagrams/knowledge-compilation-light.svg" alt="Exact inference pipeline: PIR provenance to CNF (Tseitin), to GPU Decision-DNNF (compile plus CDCL verify), to the XGCF circuit format, to weighted model counting; the compile stages run once and gradients flow backward through XGCF." />
  <img className="hidden dark:block" src="/assets/diagrams/knowledge-compilation-dark.svg" alt="Exact inference pipeline: PIR provenance to CNF (Tseitin), to GPU Decision-DNNF (compile plus CDCL verify), to the XGCF circuit format, to weighted model counting; the compile stages run once and gradients flow backward through XGCF." />
</Frame>

<Steps>
  <Step title="Provenance">
    Rule evaluation records, for every derived tuple, which probabilistic choices
    support it. This record — a graph over the probabilistic facts and annotated
    disjunctions — is called the tuple's *provenance*.
  </Step>
  <Step title="CNF">
    The provenance graph is encoded into a Boolean formula in the standard
    AND-of-ORs form (`CNF`, conjunctive normal form), with a variable map that
    lives on the device.
  </Step>
  <Step title="Decision-DNNF">
    A GPU compiler turns the `CNF` into a Decision-DNNF circuit — a circuit form
    whose worlds can be counted exactly in a single pass. A GPU SAT solver (using
    `CDCL`, the standard conflict-driven clause-learning algorithm) then checks
    that the circuit is logically equivalent to the formula before it is trusted.
  </Step>
  <Step title="Weighted model counting">
    The verified circuit — XLOG's GPU circuit format, `XGCF` — is evaluated in
    log-space to produce `log P(Q and E)` and `log P(E)`. Their difference is the
    conditional probability `log P(Q | E)`.
  </Step>
</Steps>

Gradients flow back through the same circuit, so a probabilistic program stays
differentiable end to end and can sit inside a training loop. This works through
negation too: a negated literal contributes with the correct sign flip.

<Frame caption="The XGCF circuit is a levelized DAG: literal leaves feed AND and OR nodes evaluated one level per kernel launch in log space, and adjoints propagate back down to produce gradients at the leaves.">
  <img className="block dark:hidden" src="/assets/diagrams/xgcf-circuit-light.svg" alt="XGCF circuit structure: LIT leaf nodes at level 0 feed AND nodes at level 1 and an OR root using logsumexp at level 2, producing the log probability of the query; a dashed backward path carries adjoints from the root back to gradients at the leaves." />
  <img className="hidden dark:block" src="/assets/diagrams/xgcf-circuit-dark.svg" alt="XGCF circuit structure: LIT leaf nodes at level 0 feed AND nodes at level 1 and an OR root using logsumexp at level 2, producing the log probability of the query; a dashed backward path carries adjoints from the root back to gradients at the leaves." />
</Frame>

### Aggregates in exact inference

The exact engine can push aggregates through uncertainty, up to fixed bounds:

- For `sum`, `min`, `max`, and `logsumexp`, it enumerates the outcomes of up to
  `16` uncertain rows per group exactly.
- For `count`, a dedicated count-lifting path handles up to `64` uncertain rows
  per group.

Beyond those caps the engine stops with a typed rejection rather than silently
approximating. The error message tells you to switch to `#pragma prob_engine = mc`
for that program.

## Monte Carlo inference

The Monte Carlo engine (`mc`) estimates probabilities by sampling many possible
worlds and counting how often the query holds.

Its production path is a **megakernel** — a single GPU kernel that evaluates every
sampled world in one launch, followed by one synchronization. Observations are
handled by **rejection sampling**: only sampled worlds that satisfy `evidence(...)`
are counted; the rest are thrown away.

Each query reports an estimate `prob` (with `log_prob`), a standard error
`stderr`, a two-sided confidence interval (`ci_low`, `ci_high`), the sample counts,
and the `seed`. The confidence interval is how you tell whether you drew enough
samples: if it is too wide for your needs, draw more.

<Frame caption="One kernel launch evaluates every sampled world in parallel on the device; counts stay device-resident, and the host reads back only the final estimate.">
  <img className="block dark:hidden" src="/assets/diagrams/mc-resident-megakernel-light.svg" alt="Monte Carlo resident megakernel: a fragment gate rejects unsupported programs with a typed error; supported programs enter a single GPU-resident kernel launch where each world samples facts, derives, and checks the query in parallel; per-world results accumulate into device-resident counts that produce the estimate with confidence interval and seed." />
  <img className="hidden dark:block" src="/assets/diagrams/mc-resident-megakernel-dark.svg" alt="Monte Carlo resident megakernel: a fragment gate rejects unsupported programs with a typed error; supported programs enter a single GPU-resident kernel launch where each world samples facts, derives, and checks the query in parallel; per-world results accumulate into device-resident counts that produce the estimate with confidence interval and seed." />
</Frame>

### Limits of the Monte Carlo engine

The megakernel sizes its GPU memory once, up front, before it allocates anything.
For that to be possible, a program is accepted only when it stays within these
bounds:

| Bound | Limit |
|---|---|
| Predicate arity | `<= 3` |
| Body literals per rule | `<= 3` |
| Distinct variables per rule | `<= 8` |
| Universe (bounded domain) | `<= 65536` |
| Domain size | `<= 256` |

A program that would exceed this memory budget stops with a `ResourceExhausted`
error before any allocation. There is no fallback to a larger, host-sized path.

The engine also rejects — with a typed error — any program that uses negation,
aggregates, or annotated disjunctions. In every one of these cases it refuses
cleanly rather than guessing (it "fails closed").

<Warning>
The `--allow-cpu-oracle` flag (Python: `allow_cpu_oracle=True`) is an explicit,
labeled opt-in that runs a CPU oracle instead of the GPU engine. Its results are
marked `mc_engine: "cpu-oracle"` and are **never** valid GPU-native or zero-host
evidence — treat them as a reference check, not a production result.
</Warning>

## Diagnostics and guarantees

This section is for readers who need to reason about where computation runs and
what the engines promise. You do not need it to write a probabilistic program.

**Monte Carlo — zero host traffic.** For the sampled region, the megakernel keeps
all work on the GPU. It performs `0` tracked host-to-device transfers, `0` tracked
device-to-host transfers, and `0` untracked metadata reads. This holds constant
whether you draw `128` samples or `1024`; the host reads back only the final
estimate.

**Exact engine — GPU-accelerated, host-orchestrated.** The exact engine runs on
the GPU but is driven from a host loop: the forward evaluation launches one kernel
per circuit level. No per-level data is read back — the circuit and its
intermediate values stay on the device, and only `O(1)` scalars (the log-partition
value and per-query gradients) return to the host after evaluation. It therefore
does **not** offer the single-launch, fully device-resident guarantee that the
Monte Carlo engine does.

**No CPU Decision-DNNF compiler.** Production exact inference is always the GPU
compiler and verifier described above. XLOG never bundles a CPU d-DNNF compiler and
never shells out to an external `d4` binary. Decision-DNNF parsing exists only for
tests and fixtures.

<Card title="Rule learning" icon="brain" href="/neural/rule-learning">
  The exact engine's differentiable circuits are the same infrastructure that
  trains neural predicates and learned rules end to end with PyTorch.
</Card>
