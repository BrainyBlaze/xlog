# MC GPU-Resident Engine Agent Dispatch

## Objective

Deliver the full production-grade GPU-resident Datalog/MC execution engine in
`/home/dev/projects/xlog/.worktrees/mc-gpu-resident-engine` on branch
`feat/mc-gpu-resident-engine`.

Continue from the current worktree. Preserve the existing dense/resident checkpoint
only as a precursor. The final deliverable is the real sparse WCOJ/tensorized
world-batched MC engine.

## Authoritative Inputs

- Design checkpoint: `docs/plans/2026-05-31-wcoj-world-batched-mc-engine.md`
- Current resident engine: `crates/xlog-prob/src/mc/resident.rs`
- Resident tests: `crates/xlog-prob/tests/mc_resident.rs`
- Architecture doc: `docs/architecture/xlog-prob.md`
- Release surfaces: `ROADMAP.md`, `CHANGELOG.md`

## Main Requirement

No host interaction whatsoever inside the measured MC execution loop.

This means:

- `host_loop_iterations = 0`
- `host_fixpoint_iterations = 0`
- `per_sample_host_launches = 0`
- `per_operator_host_allocations = 0` inside the measured region
- `tracked_htod = 0`
- `tracked_dtoh = 0`
- `untracked_metadata_reads = 0`

Do not satisfy this by hiding work behind untracked metadata reads or by narrowing the
claim to zero tracked transfer.

## Production Requirements

- Treat world/sample id as a first-class device dimension.
- Use sparse world-segmented columnar relations for the production engine.
- Use WCOJ/tensorized joins over world-batched relations.
- Replace host count -> host read -> allocate -> materialize operator chaining with
  device-resident sizing and allocation.
- Keep row counts, offsets, query/evidence counts, fixpoint state, and convergence
  flags device-resident.
- Implement device-side recursive/fixpoint orchestration for recursive MC programs.
- Engineer memory budgeting with deterministic preallocation, bounded arenas,
  device-side counters/offsets, and robust diagnostics.
- Do not use CPU fallback, host sizing fallback, or toy-only shortcuts.

## Definition Of Done

- Sparse WCOJ world-batched single join works without host interaction.
- Sparse WCOJ multiway join works without host interaction.
- Recursive transitive closure runs through device-side fixpoint and produces
  non-base derived tuples.
- At least one recursive program requiring more than one fixpoint iteration is proven
  by exact output and device iteration trace.
- `evaluate_gpu_device*` is rewired to the production sparse resident engine where
  applicable.
- Dense engine is documented and classified as a bounded precursor, not the final
  general engine.
- Docs, `ROADMAP.md`, `CHANGELOG.md`, architecture docs, and tests all agree.

## Anti-Gaming Rules

- Do not count dense-only pilots as proof of the general engine.
- Do not count zero tracked-transfer as proof of no-host execution.
- Do not count CPU oracle, parser-only, docs-only, or artifact-only tests as semantic
  evidence.
- Do not special-case `reach`, `edge`, or example names.
- Do not weaken exact tuple/count assertions to non-empty checks.
- Do not hide host work behind untracked metadata reads.
- Do not run broad CPU-heavy suites as release evidence.

## KPIs

- Main no-host counters are all zero inside the measured region.
- At least six exact-value sparse resident pilots pass.
- Recursive fixpoint trace records iterations and stable convergence.
- Targeted epistemic prob gates still pass.
- Release docs contain no stale dense-only or zero-tracked-only closure claim.

## Required Gates

```bash
cargo test -p xlog-prob --release --features host-io --test mc_resident -- --test-threads=1
cargo test -p xlog-prob --release --features host-io --test mc_gpu_native -- --test-threads=1
cargo test -p xlog-prob --release --features host-io --test gpu_mc_device_counts -- --test-threads=1
cargo test -p xlog-prob --release --features host-io --test epistemic_prob_gpu_accepted_evidence -- --test-threads=1
cargo test -p xlog-prob --release --features host-io --test epistemic_prob_production_reuse -- --test-threads=1
cargo fmt --check
git diff --check
rg -n '^(<<<<<<<|>>>>>>>)' --glob '!target/**'
```

## Final Report

Return:

- exact files changed
- implementation architecture
- supported fragments
- exact no-host instrumentation output
- recursive fixpoint evidence
- all gate outputs
- remaining engineering risks, if any
- verdict: `MERGE_CANDIDATE` or `HOLD_FOR_FIXES`
