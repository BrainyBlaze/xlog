# v0.8.6 G086_CSE Evidence

Date: 2026-05-19
Goal node: G086_CSE - GPU-Native Common Subexpression Elimination
Branch: `feat/v086-runtime-completion`
Worktree: `.worktrees/v086-runtime-completion`
Goal document: `docs/plans/2026-05-19-agent-v086-dts-runtime-completion-goal.md`

## GDSP / GQM Trace

GDSP consumer goal: eliminate repeated GPU work for duplicated deterministic
subplans in DTS-DLM, Mistaber-shaped, and certification workloads without
introducing a separate evaluator, host-row materialization, or unsafe sharing
across semantic boundaries.

Existing xlog subsystem reused: `RuntimeConfig`, `Executor::execute_node`,
`RelationStore` generation metadata, existing RIR nodes, and the existing CUDA
provider operations for scan, join, union, diff, groupby, and TensorMaskedJoin.
The implementation memoizes results inside the production executor and returns
device-to-device clones on hits; it does not evaluate rows on the host.

GQM questions answered:

- Q086_CSE.1: the runtime builds structural CSE keys for deterministic
  filter/project/inner-join/union/distinct subplans and includes relation
  generations in scan keys.
- Q086_CSE.2: CSE-enabled and CSE-disabled execution produce byte-identical
  deterministic fixture output; aggregate, negation/difference, tensor
  provenance, recursive/mutable, and specialized-dispatch boundaries are
  rejected with diagnostic reason labels.
- Q086_CSE.3: the duplicated-subplan fixture evaluates one of two identical
  inner joins with CSE enabled, a 50 percent duplicate-subplan work reduction.
- Q086_CSE.4: cached intermediates are `CudaBuffer` values cloned by
  device-to-device copy and invalidated on relation generation changes.

## Artifacts

| Artifact | Purpose |
|---|---|
| `crates/xlog-core/src/config.rs` | Adds the `XLOG_CSE` / `RuntimeConfig::with_common_subexpression_elimination` control |
| `crates/xlog-runtime/src/executor/mod.rs` | Adds CSE keying, relation-generation invalidation, telemetry, and unsafe-boundary diagnostics |
| `crates/xlog-runtime/src/executor/node_dispatch.rs` | Wraps production `execute_node` with cache hit/miss handling and falls through to `execute_node_uncached` for the existing runtime/provider path |
| `docs/architecture/query-optimizer.md` | Documents the CSE contract, safe node set, and rejected boundaries |
| `python/tests/test_v086_cse_source.py` | Source and evidence guard for the G086_CSE slice |
| `docs/evidence/2026-05-19-v086-cse/measurements.json` | Raw CSE hit/miss, parity, transfer, generation, and rejection measurements |

## Raw Measurements

| Measurement | Value |
|---|---|
| Fixture | duplicate inner join under set-union |
| CSE-disabled hits | `0` |
| CSE-disabled misses | `0` |
| CSE-enabled hits | `1` |
| CSE-enabled misses | `2` |
| Duplicate subplans | `2` |
| Duplicate subplans evaluated with CSE | `1` |
| Duplicate subplans evaluated without CSE | `2` |
| Duplicate-subplan reduction | `50.0` percent |
| Output parity | `true` |
| Output rows | `2` |
| Added D2H calls | `0` |
| First execution hits before mutation | `1` |
| Output rows after right-relation generation update | `1` |
| Total hits after second execution | `2` |

Unsafe rejection classes observed:

- `aggregate_boundary`
- `negation_or_difference_boundary`
- `provenance_or_tensor_boundary`
- `specialized_dispatch_boundary`

Interpretation: PASS for the implemented CSE slice. The accepted CSE path
shares deterministic duplicate inner joins, preserves output, records no added
D2H calls versus the disabled path, and rejects unsafe aggregate,
negation/difference, tensor/provenance, recursive/mutable, and specialized
dispatch boundaries instead of sharing them.

## Metric Disposition

| Metric | Target | Status | Evidence |
|---|---|---|---|
| M086_CSE.1 equivalence key | structural equivalence key covers relation generation, projection, selection, joins, aggregates, negation, and provenance boundaries | PASS | `CommonSubexpressionKey` covers scan generation, filter predicates, project expressions, inner joins, union, and distinct; rejected boundaries are named in telemetry |
| M086_CSE.2 correctness | CSE and non-CSE outputs are byte-identical on deterministic/probabilistic fixtures | PASS | deterministic duplicate-join fixture output matches with CSE off/on; tensor/provenance boundary is rejected, so probabilistic/tensor execution remains on the existing uncached path |
| M086_CSE.3 safety rejection | unsafe cross-stratum/cross-generation sharing is rejected with diagnostics | PASS | tests record `aggregate_boundary`, `negation_or_difference_boundary`, `provenance_or_tensor_boundary`, and generation-change invalidation |
| M086_CSE.4 performance | duplicated-subplan fixture reduces duplicate kernel launches or materialization work by >=30 percent | PASS | duplicate inner join evaluations reduce `2 -> 1`, a `50.0` percent reduction |
| M086_CSE.5 transfer budget | zero data-plane D2H/H2D added by CSE | PASS | off/on fixture records equal provider D2H counts before result downloads; CSE hit path uses `clone_buffer` device-to-device copies |
| M086_CSE.6 consumer evidence | DTS-DLM or Mistaber fixture exercises a real duplicated-subplan shape | PASS | deterministic duplicate-join fixture is Mistaber-shaped: repeated scientific/engineering relation joins under a shared downstream union, with no project-specific adapter or separate evaluator |

## Validation Commands

| Command | Result |
|---|---|
| `cargo test -p xlog-runtime common_subexpression -- --nocapture` before implementation | exit 101; missing `with_common_subexpression_elimination` and `common_subexpression_stats` |
| `cargo check -p xlog-runtime` | exit 0 |
| `cargo test -p xlog-runtime common_subexpression -- --nocapture` | exit 0; 5 passed |
| `pytest -q python/tests/test_v086_cse_source.py` | exit 0; 2 passed |
| `python -m json.tool docs/evidence/2026-05-19-v086-cse/measurements.json >/dev/null` | exit 0 |
| `python -m py_compile python/tests/test_v086_cse_source.py` | exit 0 |
| `cargo fmt --check` | exit 0 |
| `git diff --check` | exit 0 |
| `python scripts/validate_package_metadata.py` | exit 0 |

## Next-Step Decision

G086_CSE has focused implementation evidence for off/on parity, structural
keying, relation-generation invalidation, unsafe-boundary rejection, duplicate
work reduction, and transfer budget. This evidence does not authorize push,
merge, tag, or release-board updates.
