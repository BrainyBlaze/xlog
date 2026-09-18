# XLOG Engineering Standards

These standards apply to every production change, including code, tests, examples,
benchmarks, build scripts, and user-facing documentation. They are merge criteria,
not optional guidance. `CONTRIBUTING.md` describes the contribution workflow;
`AGENTS.md` adds execution rules for automated contributors.

## Current Human Execution Direction

Implementation must advance the complete governing design through actual production
code. Following the requested registration of lessons in this file and `AGENTS.md`,
do not add tests or meta-work, run local component experiments, or loop on isolated
component improvements. Generic testing and documentation prescriptions below do
not authorize those activities against this direction. Do not delete existing tests
or weaken repository checks to conceal this boundary.

Runtime runs are permitted only on explicitly authorized RunPod GPUs through the
real production path. Reading source, compiling, and fixing actual build failures
are not runtime evidence or permission for local model execution. If a required
integration check cannot run within this direction, report its exact unmet condition
to the lead rather than inventing a substitute or bypassing branch protection.

Every coherent change must receive an atomic commit and be pushed promptly. The lead
owns integration into remote `main`. Source integration and complete runtime
qualification are distinct: deliver code for integration without waiting for the
entire project to become qualified, and never claim qualification from a merge.
Standing human authorization for commit and push must not be replaced by a peer's
older or narrower interpretation. Resource, installation, and execution limits remain
separate. No unreviewed bulk staging, unrelated edits, or AI attribution trailers.

## Lessons Applied to Production Work

- The design's complete observable behavior is the target. An opcode declaration,
  schema, validator, typed refusal, passing component, or audit does not implement a
  missing transition, learning step, structural operation, or restored continuation.
  Choose the next change by the missing connected production behavior, not by how
  easily a local activity can be completed.
- Follow each change from its authentic producer through native execution, public
  bindings, output delivery, refusal, publication, and restoration. Keep one schema
  and one numerical implementation. Do not invent producer-owned fields, fabricate
  native objects, detach embeddings, substitute recorded outputs, or add a second
  loader or model lowering to make an interface appear connected.
- An invalid branch must not execute forbidden model or state-changing work. Masking
  its output afterward is not conditional execution. Maintain the original graph,
  backward path, ownership, and device-side control rather than reading validity on
  the host or silently changing the numerical model.
- Preserve real invocation, envelope, allocation, and random-stream generations.
  Owner-bearing storage must survive asynchronous launches and captured graphs;
  publish only after validating the exact candidate and generation. Refusal and
  failure must not expose partial updates or advance protected state.
- Gradient delivery must reach the original physical model parameters. A receipt,
  retained buffer, replayable carrier, or parameter-shaped array alone is not an
  optimizer update. Restoration must preserve the state needed for the next real
  transition, including parameters, optimizer state, random state, and publication.
- Bounded work must account for actual execution through final output and publication,
  not merely a prefix, an estimate, or a renamed counter. Device residency requires
  measured ZERO TRANSFER in the declared hot loop; source inspection, compiled
  kernels, and historical narrower GPU runs do not prove the current full path.
- Delegation does not transfer integration responsibility. Actual code and callers
  must reach the canonical WSL tree, an atomic commit, the remote branch, and the
  lead's main integration. An agent's report or an isolated worktree is not delivery.
  Do not relocate work to Windows-mounted storage, overwrite dirty work, or remove
  existing copies without explicit authority.
- Use direct foreground commands and obey the current delegation limits. Coordinate
  producer changes with their existing owners, keep dependent interfaces authentic,
  and communicate actionable outcomes instead of status loops. Investigate a real
  blocker and continue independent production work; do not manufacture new approval
  gates, substitute tasks, or completion claims.
- State evidence precisely: local commit, remote push, lead integration, compilation,
  hardware execution, and complete-design acceptance are separate facts. Preserve
  failed attempts and valid historical results without promoting them to a stronger
  or newer claim. A protected branch rejecting a push is not permission to bypass it.

## Goal-Directed Execution

Optimize time to the complete, verified user goal, not the number of tasks, tests,
documents, messages, or approvals produced.

- Choose work that implements required behavior, closes a production integration,
  removes a demonstrated bottleneck, or supplies evidence needed for the next real
  decision. Do not repeatedly polish an isolated component while its required
  production consumer remains disconnected.
- Establish the actual producer contract before dependent implementation: data
  meaning, layout, validity, ownership, lifetime, numerical behavior, and restoration.
  Carry the smallest coherent production path through its consumers and failure
  boundaries; independent component tests do not establish that the path works.
- Necessary source, caller, dependency, history, and documentation research is part of
  an assigned implementation within existing access rights. Do not invent per-file,
  per-function, or tool-read-count approval gates. Navigation confidence and context
  budgets guide efficient exploration; they do not prove absence or define authority.
- Keep independent authorized work parallel. A data run, hardware gate, or another
  owner's unfinished interface blocks only work that actually depends on it. Identify
  the real bottleneck before an expensive run or an infrastructure change.
- Verify changed behavior and affected boundaries, including required broader checks.
  Reuse valid evidence for unchanged code; repeat a check when the change, environment,
  or acceptance requirement justifies it, not to manufacture progress.
- Coordinate around concrete results, interface changes, decisions, and actionable
  blockers. Do not reopen settled choices or repeat unchanged status and acknowledgment
  exchanges in place of implementation.
- Preserve explicit access, write-set, resource, safety, and publication limits.
  Urgency and documentation changes do not themselves authorize restricted reads,
  GPU runs, spending, installation, commit, push, merge, or release. If new authority
  is genuinely required, request one sufficient connected scope from the responsible
  decision-maker and continue independent permitted work.

## Research Before Editing

Every non-trivial change starts with an exact understanding of the current codebase.

- Confirm the repository root, branch, current commit, and clean/dirty state before
  editing.
- Read the governing public contract, architecture documentation, relevant examples,
  tests, and release notes. Inspect git history when it explains why an invariant or
  interface exists.
- Trace the complete production path: definitions, callers, configuration sources,
  error mappings, bindings, serialization boundaries, and documentation. Do not infer
  behavior from a single file or a test name.
- Search for existing types, helpers, kernels, adapters, and test fixtures before
  creating anything. Record the behavioral gap and the invariant the change must
  preserve.
- Reproduce a reported bug through the authoritative path before changing code. For a
  feature, identify its real entry point and downstream consumers before designing it.

Research is complete when the implementation location, affected contracts, regression
risk, and behavioral verification are known. Writing a plan or inventory is not a
substitute for the requested implementation.

## One Coherent Implementation

- Maintain one authoritative implementation for each behavior. Extend the canonical
  path instead of adding a parallel helper, parser, adapter, execution path, or source
  of configuration.
- Reuse existing code when its abstraction matches the invariant. If it does not,
  improve the abstraction at its natural owner rather than copying its logic.
- Do not introduce copy-and-paste implementations, near-duplicate algorithms, shadow
  data models, or competing public APIs. Shared invariants must be expressed once and
  tested at that boundary.
- Add a new abstraction only when it removes real duplication or establishes a durable
  boundary. Do not add speculative layers, single-use wrappers, or configuration for
  hypothetical future work.
- A new public API must have a real supported consumer, production-path behavior tests,
  clear ownership semantics, and user documentation. Public surface without a wired
  consumer is dead code.

## Root-Cause Fixes Only

- Fix the cause, not the symptom. Disabling checks, weakening assertions, suppressing
  diagnostics, excluding files, swallowing errors, or relabeling a failure is not a
  fix.
- Do not use workarounds, kludges, hidden feature flags, environment-variable escape
  hatches, hardcoded semantic shortcuts, or test-only branches to make a requirement
  appear satisfied.
- Production paths must not contain placeholders, stubs, fake records, toy algorithms,
  or mocked success. Small examples and fixtures are acceptable only when they exercise
  the same implementation and semantics as production.
- Do not introduce a fallback execution path as a substitute for supported production
  behavior. A diagnostic or reference oracle must be explicit, opt-in, clearly labeled,
  and excluded from production-performance or hardware-native evidence.
- Failures must be explicit, typed, and actionable. Never silently fall back to a less
  capable backend, stale result, default configuration, host path, or approximate
  behavior.
- Do not add legacy branches or compatibility shims. When replacing an internal path,
  migrate its callers and remove the obsolete path in the same change. A public API
  migration that genuinely requires a transition is separate, explicitly approved
  work with a documented removal release; it must never be introduced incidentally.

## No Dead Code or Deferred Debt

- Remove imports, flags, helpers, branches, tests, docs, and assets made obsolete by a
  change. Do not keep the old path "just in case."
- Every new branch must be reachable from a supported entry point and covered by a
  behavior-level test. Every new error variant must have a real producer and a boundary
  mapping where applicable.
- Do not merge commented-out code, required behavior left as `TODO` or `FIXME`, knowingly
  unreachable paths, unused public functions, or incomplete migrations.
- Do not defer correctness, safety, cleanup, documentation, or required validation as
  technical debt. If the proper solution cannot fit the approved scope, stop and obtain
  a scope decision instead of landing a temporary substitute.

## Design and Implementation Quality

- Keep changes surgical and cohesive. Each changed line must trace to the stated
  behavior, its tests, documentation, or removal of code made obsolete by that behavior.
- Preserve established architecture, naming, formatting, ownership, error, and
  configuration conventions unless the task explicitly changes the convention.
- Make units, limits, ownership, lifetimes, and synchronization boundaries explicit in
  types and names. Avoid magic numbers and implicit global state.
- Validate inputs at the owning boundary and preserve one precedence rule for layered
  configuration. Explicit caller input must not be confused with an omitted default.
- Treat concurrency, device synchronization, overflow, partial results, and resource
  exhaustion as correctness concerns. Never publish a result whose validity status has
  not been checked.
- Every `unsafe` block and host/device boundary must state and enforce its buffer-shape,
  ownership, lifetime, indexing, and synchronization invariants at the nearest testable
  boundary.
- Prefer simple code whose invariants can be explained locally. If an implementation is
  substantially larger or more indirect than the behavioral change, simplify it before
  review.

## Tests and Evidence

- Add a failing regression or behavior test first when practical, observe the expected
  failure, then implement the fix through the real path.
- Test externally observable semantics, not file existence, helper call counts, or an
  implementation detail that can pass while production remains broken.
- Cover success, boundary, invalid-input, and failure propagation cases appropriate to
  the risk. Cross-language and GPU-facing behavior requires verification at those real
  boundaries.
- Benchmarks must be reproducible: commit the runner, pin workload generation and
  parameters, record exact commands and versions, validate result equivalence, report
  failures, and separate measured regions from setup and teardown.
- Evidence must come from the exact commit under review. State what ran, where it ran,
  what passed or failed, and which stronger gate remains untested.

## Contribution and Review Discipline

- Keep each pull request focused on one coherent outcome. Do not mix opportunistic
  refactors or formatting churn with behavioral work.
- Explain the root cause, design choice, public behavior, risk, migration or removal,
  and exact verification commands. Include failed attempts when they affect confidence.
- Update user documentation, examples, bindings, changelog entries, and error references
  in the same pull request when their contract changes.
- Resolve review comments technically: verify the concern, repair the governing path,
  and add regression coverage. Never satisfy review by bypassing the check that exposed
  the defect.
- Never merge with failing required checks, unresolved review threads, known required
  follow-up work, generated files that do not match their source, or unverified claims.
- Do not add AI attribution or generated-by trailers to commits.

## Completion Checklist

Before requesting merge, verify all of the following:

1. The change uses the existing canonical architecture and adds no duplicate path.
2. The root cause or requested behavior is implemented in production code.
3. No obsolete, unreachable, placeholder, fallback, compatibility, or temporary code was
   introduced or left behind.
4. Behavior-level regressions pass through every affected public boundary.
5. Relevant formatting, lint, unit, integration, packaging, documentation, and hardware
   checks pass on the exact commit.
6. Documentation and examples describe the implemented behavior without overclaiming.
7. The diff is focused, understandable, and free of unrelated or local-only artifacts.
