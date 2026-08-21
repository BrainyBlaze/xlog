# How XLOG works

One compiler and one CUDA runtime span four reasoning paradigms — here is the path a program takes from source to GPU.

XLOG lets you write deterministic Datalog programs, probabilistic models, epistemic
programs, and differentiable neural-symbolic training programs, then run their
symbolic computation directly on the GPU. SAT/MaxSAT verification is available
through a shared solver service. You write in one typed language, and a shared
compiler plus one CUDA runtime support all four user-facing paradigms.

Because the results stay on the GPU, you can feed them straight into a PyTorch
training loop instead of copying data back and forth. This page follows the path a
program takes from source text to running GPU code.

## When this matters

Read this page when you want a correct mental model of what happens after you write
a program: how it is compiled, why all four user-facing paradigms share one frontend, and
why the output is cheap to hand to a tensor framework.

You do not need it to write your first program. You do need it to reason about
performance, or to understand why XLOG fits inside a training loop rather than beside
it.

This is a conceptual explainer, not a task walkthrough: it builds a mental model of
the compilation path rather than stepping you through a runnable program. For a
hands-on start with real output, see the [quickstart](/get-started/quickstart).

## The pipeline

Every program shares frontend parsing, normalization, type checking, and dependency
analysis. The host machine then compiles and orchestrates the appropriate backend path.
The GPU holds the data and runs the actual computation as CUDA kernels.

The diagram below follows the deterministic relational path. Other reasoning styles
branch from the normalized frontend into their own intermediate forms, as shown later.

<Frame caption="A deterministic program is parsed, stratified, lowered to relational IR, optimized, and executed by dispatching CUDA kernels over device-resident relations.">
  <img className="block dark:hidden" src="/assets/diagrams/compilation-pipeline-light.svg" alt="Deterministic XLOG compilation pipeline: Source, Parser, Stratifier, Lowerer, Optimizer, and Executor run on the host; CUDA kernels and the relation store are GPU-resident." />
  <img className="hidden dark:block" src="/assets/diagrams/compilation-pipeline-dark.svg" alt="Deterministic XLOG compilation pipeline: Source, Parser, Stratifier, Lowerer, Optimizer, and Executor run on the host; CUDA kernels and the relation store are GPU-resident." />
</Frame>

<Steps>
  <Step title="Parse">
    A PEG grammar (a formal grammar that tries its alternatives in a fixed order, so
    every input has one unambiguous parse) turns your program into an abstract syntax
    tree (a structured tree form of your program that the compiler works on).

    The grammar is the definitive surface of the language: if a construct does not
    appear in it, it does not parse.
  </Step>
  <Step title="Stratify">
    For the deterministic relational path, XLOG splits the program into ordered layers,
    called *strata*, so that negation and aggregation are computed in a safe order.

    It does this by finding cycles in the predicate dependency graph
    (strongly-connected-component analysis). If a use of negation or aggregation cannot
    be placed in such an order, XLOG rejects the program.
  </Step>
  <Step title="Lower">
    The stratified deterministic program is rewritten into a relational intermediate
    representation (`RIR`) — your program expressed as database-style operations.

    Those operations are joins, filters, projections, aggregations, and recursive loops
    (fixpoints).
  </Step>
  <Step title="Optimize">
    A cost-aware pass plans the join order and pushes filters closer to the data so
    less work reaches later stages.

    It also promotes eligible joins to *worst-case-optimal multiway joins* — a join
    method that computes multi-way patterns, such as triangles in a graph, directly
    instead of building a large intermediate table.
  </Step>
  <Step title="Execute">
    The executor runs the plan on the host and dispatches CUDA kernels over relations
    that live in GPU memory.

    Deterministic recursion runs as a *semi-naive fixpoint*: a loop that repeats a rule
    until nothing new is derived, processing only the newly-found facts on each round.
  </Step>
</Steps>

## Four paradigms, shared backend routes

XLOG presents four user-facing paradigms — deterministic, probabilistic, epistemic,
and neural-symbolic — and all four share parsing, normalization, type checking, and
dependency analysis. They do not map one-to-one to intermediate representations.
The normalized program fans into three reasoning IR routes: deterministic `RIR`,
probabilistic `PIR`, and epistemic `EIR`. Neural-symbolic training composes neural
predicates with the relevant reasoning route instead of introducing a separate IR.
SAT/MaxSAT features and verification use the shared GPU-resident `GpuCnf` and CDCL
solver service rather than another IR.

Each reasoning branch is built from the normalized program — the parsed and analyzed
rules put into a common standard form — not from the deterministic `RIR`. The
probabilistic and epistemic routes are therefore siblings of the deterministic route,
not layers on top of it. The solver service consumes GPU-resident CNF supplied by
features that need verification or SAT/MaxSAT search.

<Frame caption="One typed frontend fans into three reasoning IR routes plus a shared GPU CNF/CDCL solver service.">
  <img className="block dark:hidden" src="/assets/diagrams/architecture-overview-light.svg" alt="XLOG architecture: Source feeds a shared frontend that fans into deterministic RIR, probabilistic PIR to XGCF, and epistemic EIR reasoning routes, alongside a shared GPU-resident GpuCnf and CDCL solver service." />
  <img className="hidden dark:block" src="/assets/diagrams/architecture-overview-dark.svg" alt="XLOG architecture: Source feeds a shared frontend that fans into deterministic RIR, probabilistic PIR to XGCF, and epistemic EIR reasoning routes, alongside a shared GPU-resident GpuCnf and CDCL solver service." />
</Frame>

Here is what each style computes and where to learn to use it. The glosses below the
table explain the terms in plain language.

| Paradigm | What it computes | Learn more |
|---|---|---|
| **Deterministic Datalog** | Least-model semantics with stratified negation and aggregation, evaluated as a semi-naive fixpoint | [Language reference](/reference/language) |
| **Probabilistic** | Marginal and conditional probabilities via exact knowledge compilation or Monte Carlo sampling | [Probabilistic programming](/probabilistic/engines) |
| **Epistemic** | World-view semantics over modal `know` and `possible` operators | [Epistemic reasoning](/epistemic/overview) |
| **Neural-symbolic** | Learned rules and neural predicates trained end-to-end with PyTorch | [Rule learning](/neural/rule-learning) |

- **Least-model semantics** — the unique smallest set of facts that the rules force to
  be true.
- **Exact knowledge compilation** — turning the program into a circuit that computes
  exact probabilities, rather than estimating them by sampling.
- **World-view semantics** — reasoning about what an agent knows (`know`) or considers
  possible (`possible`); a *world view* is the set of models the program treats as
  possible at once.

## How the paradigms map to intermediate forms

The pieces above become concrete once you see how each path is represented internally.
The normalized AST is the shared source for each reasoning IR; the probabilistic and
epistemic paths do not branch from deterministic RIR.

RIR represents deterministic relational execution. PIR represents probabilistic
execution and compiles into the GPU-evaluable circuit format called `XGCF`. EIR
preserves modal meaning while the high-level dispatcher classifies dependencies.
Neural-symbolic training supplies neural predicate values to, and consumes
differentiable results from, these symbolic routes rather than using a fourth
reasoning IR. SAT/MaxSAT and verification features submit GPU-resident `GpuCnf`
inputs to the shared CDCL solver service.

Inside the epistemic route, one more choice is made for you. XLOG looks at the shape
of the program's dependencies — whether the modal rules form a cycle, and what kind —
and picks how to evaluate it:

| Program shape | How XLOG evaluates it |
|---|---|
| No cycle among the modal rules | *Generate-Propagate-Test*: enumerate the candidate world views, prune the inconsistent ones, then test what is left. |
| A positive cycle, under the default `faeel` semantics | An ordinary least-fixpoint loop over the rewritten rules. It derives only facts the program actually founds, so a rule that merely supports itself concludes nothing. |
| A positive `possible` cycle under `g91` semantics, where each concrete tuple must be checked | Start from an upper bound of candidate tuples and repeatedly filter it against the previous round's snapshot until it stops shrinking. |
| A cycle that runs through negation | The GPU-backed *well-founded* plan: the standard three-valued reading of recursive negation, which leaves a genuinely circular fact undefined instead of guessing it true or false. |

`faeel` and `g91` are the two epistemic semantics you choose between with
`#pragma epistemic_mode`; the [epistemic overview](/epistemic/overview) explains what
each one accepts and when to pick it.

<Frame caption="The normalized AST fans into RIR, PIR, and EIR. SAT/MaxSAT and verification use a separate shared GpuCnf/CDCL service.">
  <img className="block dark:hidden" src="/assets/diagrams/ir-stack-light.svg" alt="XLOG intermediate forms: the normalized AST fans into deterministic RIR and a relational plan, probabilistic PIR and the XGCF circuit format, and epistemic EIR with route dispatch. A separate shared GpuCnf and CDCL service supports SAT, MaxSAT, and verification." />
  <img className="hidden dark:block" src="/assets/diagrams/ir-stack-dark.svg" alt="XLOG intermediate forms: the normalized AST fans into deterministic RIR and a relational plan, probabilistic PIR and the XGCF circuit format, and epistemic EIR with route dispatch. A separate shared GpuCnf and CDCL service supports SAT, MaxSAT, and verification." />
</Frame>

<Note>
  **Internal names, in one place.** These abbreviations name the intermediate forms a
  program passes through. You do not write them, but you will see them in diagnostics
  and diagrams.

  - `RIR` — relational IR: your program as joins, filters, projections, aggregations,
    and fixpoints.
  - `PIR` — probabilistic IR; tracks which facts and probabilities contributed to each
    result (its provenance), and compiles to `XGCF`.
  - `EIR` — epistemic IR; preserves modal semantics and drives execution-route
    classification.
  - `GpuCnf` — the GPU-resident CNF representation consumed by the shared CDCL solver
    service.
  - `XGCF` — the GPU-evaluable circuit format produced by probabilistic knowledge
    compilation.
</Note>

## Compile once, evaluate many

The compiled plan is *structure*, and that structure is stable. Across training
iterations or repeated queries, only the data and the weights change; the plan itself
does not.

So XLOG compiles a program once and reuses the plan on every evaluation. For
probabilistic inference it also reuses the compiled circuit. This is what makes XLOG
usable inside a training loop rather than alongside it.

## Results stay on the GPU

Query outputs and gradient tensors never leave the GPU on their way to your tensor
code. XLOG exposes them through DLPack capsules and Arrow — two standard formats for
sharing tensors and tables between GPU libraries without copying.

Because of that, a downstream tensor computation in PyTorch, JAX, or cuDF reads XLOG's
output directly, with no round-trip back through the host.

<Card title="Why GPU residency matters" icon="microchip" href="/core-concepts/gpu-residency">
  The defining constraint behind XLOG's design — and what separates it from CPU logic
  engines bolted onto a GPU tensor framework.
</Card>
