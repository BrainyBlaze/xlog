# Roadmap

A concise capability roadmap for XLOG's implemented architecture and future engineering priorities.

<Note>
For contributors — how xlog's release state and architecture priorities are
organized internally. This is a roadmap, not a user guide.
</Note>

This roadmap describes implemented capabilities and forward-looking architecture
priorities. It does not duplicate a mutable package version. Use
`CHANGELOG.md` and the selected artifact's release notes to determine packaged
availability.

## Released capabilities

Tagged artifacts include the shared Rust workspace, CUDA-backed deterministic
execution, probabilistic and epistemic surfaces, solver services, Python
packaging, and the `xlog` CLI. Aggregate-fused worst-case-optimal joins (WCOJ),
GPU Free Join, factorized recursive deltas, joint neural-symbolic mixtures, and
grouped training are available in tagged artifacts beginning with 0.10.0.

Two versions have followed. 0.11.0 added a device-resident joint-constraint
solver — a GPU handle that picks one label per entity under pairwise constraints
and hands its result buffers back without a host copy — plus a memoized
dynamic-programming solve stage for chain components too large to enumerate.
0.12.0 added exact log-evidence and the CNF-variable-to-fact map on probabilistic
results (`EvalResult.log_z_e`, `CompiledProgram.prob_var_map`), native n-ary
relation provenance, batched same-head rule unions, and imported-module pragma
warnings.

The public Rust release boundary is the set of publishable crates listed in
[Release Process](/architecture/release-process). `pyxlog` is shipped through
Python packaging, not through crates.io.

## Stable Architecture Commitments

XLOG's architecture is organized around a fixed set of commitments. These
commitments — not old milestone labels or implementation-board names — define
the architecture:

- **One frontend** for deterministic, probabilistic, epistemic, and
  neural-symbolic programs.
- **An explicit boundary** between host-side control logic and the GPU
  data plane.
- **Inspectable reasoning contracts** carried by three internal intermediate
  representations: RIR for deterministic relational execution, PIR for
  probabilistic provenance, and EIR for epistemic semantics. SAT/MaxSAT and
  verification use the separate shared `GpuCnf` representation and CDCL solver
  service.
- **CUDA kernels** that operate over device-resident buffers (data that stays on
  the GPU).
- **Fail-closed behavior**: when a proof, route, or capacity contract is not
  met, execution stops rather than returning an unchecked result.
- **Route counters and telemetry**: dispatch counts that record which optimized
  execution path actually fired.

## Available since 0.12.0

These public contracts are part of 0.12.0:

- **Native relation evidence** (breaking change) — ordered semantic roles and
  provenance records bind to complete facts of any positive arity and follow every
  relation mutation atomically. This *replaced* the Python sidecar API rather than
  extending it: `evidence()` no longer returns `source_path`, `source_hash`,
  `row_hashes`, `accepted_count`, `rejected_count`, `output_hash`, or
  `decision_counts`; `evidence()` and `relation()` raise `KeyError` for an unknown
  relation instead of returning an empty dict; and `RelationEvidence` is an
  immutable native class with no `RelationEvidence(session, name)` constructor.
- **Exact relation manifests** — versioned metadata and DLPack columns reconstruct
  persistent rows, roles, and whole-fact evidence in another compatible session.
- **Owned persistent relation snapshots** — persistent DLPack imports are copied
  device-to-device into session-owned storage, while transient evaluation imports
  remain zero-copy.
- **Deterministic delta behavior** — canceled updates do not advance relation
  versions or emit callbacks, and invalid batch keys are rejected before mutation.
- **Batched multiway unions** — same-head rule outputs merge through one chunked
  multiway union per evaluation pass, with the byte budget controlled by
  `XLOG_UNION_CHUNK_BYTES`.
- **Ignored-pragma warnings** — an imported-module pragma produces
  `warning[W0510]` on stderr because pragmas apply only in the entry file.

## Architecture Priorities

The next architecture work stays focused on runtime facts that users can
observe:

- clearer route telemetry for worst-case-optimal join (WCOJ), Free Join,
  aggregate fusion, and recursive delta routing;
- stronger release evidence for CUDA-required behavior;
- explicit no-host-transfer counters wherever a path claims device-resident data
  flow (data that never leaves the GPU);
- sharper failure diagnostics for unsupported probabilistic, epistemic, and
  solver shapes;
- documentation examples that show how to verify route dispatch directly,
  instead of inferring it from the final answers.

## Out-Of-Scope For This Roadmap

This page is a release-state view, not the historical task board. Internal phase
labels, board codes, and dated task bundles are not part of this roadmap. Where a
historical item still matters, it is documented as shipped behavior, the command
that verifies it, and the release boundary — not as a task entry.
