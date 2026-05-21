# GQM Plan: BFO Universal Case Reasoner

## Measurement Goal

Analyze the cross-domain case reasoner for the purpose of evaluating whether a
stable BFO neuro-symbolic kernel can transfer root-cause and intervention
reasoning across multiple domains with production-grade correctness,
explanation, adaptation, and performance evidence.

Perspective: project owner and production reviewer.

Context: XLOG v0.8.5/v0.8.6-era language, neural, probabilistic, incremental,
and explanation features under the constraints documented in `ROADMAP.md` and
`docs/language-reference.md`.

## Operational Questions And Metrics

| ID | Question | Metric | Required Value |
| --- | --- | --- | --- |
| Q1 | Is the BFO core unchanged across domains? | Core rule edits per domain | 0 |
| Q2 | Are enough domains represented? | Domain adapters | >= 5 |
| Q3 | Are adapters thin? | Adapter/core rule ratio | <= 0.25 per domain |
| Q4 | Is neural evidence real and causally used? | CUDA `nn/4` model affects rankings | Yes |
| Q5 | Does root-cause inference transfer? | Held-out domain root-cause F1 | >= 0.90 |
| Q6 | Are interventions useful? | Accepted intervention precision | >= 0.95 |
| Q7 | Are explanations complete? | Top-level claims with BFO explanation | 100% |
| Q8 | Does neuro-symbolic beat baselines? | Relative uplift over best baseline | >= 15% |
| Q9 | Is online adaptation exact? | Delta output equals full recompute | 100% |
| Q10 | Is the hot path device-resident? | D2H/H2D transfers after initial load | 0 |
| Q11 | Is the result deterministic? | Fixed-seed byte-identical runs | 5/5 |
| Q12 | Is performance production-grade? | p95 core query latency | <= 50 ms |

## Hypotheses

- H1: BFO upper-ontology structure enables transfer across domains with smaller
  adapters than domain-specific rules.
- H2: A held-out domain can be solved zero-shot or few-shot when its cases share
  BFO-level failure structures with training domains.
- H3: Neuro-symbolic reasoning produces higher transfer consistency and better
  explanations than neural-only or symbolic-only baselines.
- H4: Core mutation audits are necessary to distinguish real transfer from
  hidden domain-specific rewriting.

## Data Collection Plan

Validation must record:

- Git SHA, branch, command line, seeds.
- CUDA availability, GPU model, CUDA version, PyTorch version, XLOG runtime mode.
- Hugging Face dataset IDs, splits, row hashes, and per-domain source
  provenance for production transfer cases.
- Domain inventory, adapter size, core rule checksum, holdout protocol.
- Baseline results: neural-only, domain-symbolic, shared-symbolic,
  neuro-symbolic.
- Root-cause, failure-chain, intervention, and explanation records per domain,
  plus held-out confusion counts recomputed from those records.
- v0.8.0/v0.8.5/v0.8.6 bundle reuse probes and their runtime counters.
- Transfer counters before and after hot loops.
- Wall time, p50/p95 latency, GPU memory peak, memory drift.
- Explanation JSON for every top-level claim.
- Raw outputs and generated evidence paths.

## Analysis Plan

The branch passes only if every P0 metric in `REQUIREMENTS.md` passes under
`./validate.sh --strict --gpu-required`. Missing runtime support is a production
failure with evidence, not a reason to weaken the requirement.
