# CUDA Certification

How XLOG separates ordinary CI, docs validation, CUDA-required release gates, and staged reliability evidence.

<Note>
For contributors — how XLOG's validation gates work internally. This page
explains what each gate proves and why they are not interchangeable.
</Note>

XLOG runs several validation layers, and each one answers a different question.
Knowing which gate proves what keeps you from treating a cheap check as if it
were a release certification. Do not collapse them into a single test count.

A fixed pass-count snapshot (for example "9/9 passed") is not a certification on
its own. What actually certifies a result is three things: the exact command
that ran, the hardware it ran on, and the evidence it produced. Prefer those
over a bare number.

## Validation Layers

The layers below run in different places and prove different things.

| Layer | Where it runs | What it proves |
| --- | --- | --- |
| GitHub CI | GitHub-hosted runners | Formatting, workflow hygiene, package metadata, no-GPU CUDA build, and non-GPU checks. |
| Targeted CUDA CI | Self-hosted CUDA runner after pushes to `main`, a maintainer dispatch from `main` for a reviewed full commit SHA, or a pull request whose head branch belongs to this repository | Installed-wheel contracts and exact-count resident-graph runtime, preparation, production, scaling, and semantic acceptance on a real GPU. Pull requests from forks are skipped before a CUDA runner is assigned. |
| Docs site CI | GitHub Actions for documentation and documentation-build inputs | Mintlify validation, broken-link checks, and static export; pushes to `main` also publish `docs-dist`. |
| GPU release validation | Maintainer-run CUDA host | Actual CUDA behavior through `scripts/validate_release_gpu.sh`. |
| Subsystem reliability gates | Subsystem-specific suites | Statistical or staged reliability for neural-symbolic and other higher-level engines. |

Green GitHub-hosted CI does not certify GPU correctness. The targeted
self-hosted job exercises focused Python and Rust surfaces after merge, an
explicit reviewed-SHA dispatch, or a same-repository pull request. Its job
condition compares the pull request's head repository with the current
repository, so an external fork cannot claim the persistent CUDA runner. The
workflow uses the ordinary `pull_request` event and never
`pull_request_target`. Release certification still requires the complete manual
gate on a CUDA machine.

## GPU Release Gate

This is the canonical manual gate for GPU behavior. It needs a clean checkout of
the pinned acceptance corpus at the exact reviewed commit. Create one without
reusing a developer checkout:

```bash
PINNED_CORPUS_SHA=74f2895486737b4caa42229389d309994e7ad3ea
PINNED_CORPUS_URL=https://github.com/BrainyBlaze/mistaber-xlog.git
corpus_parent="$(mktemp -d)"
export XLOG_PINNED_CORPUS_ROOT="$corpus_parent/pinned-corpus"

git init --quiet "$XLOG_PINNED_CORPUS_ROOT"
git -C "$XLOG_PINNED_CORPUS_ROOT" remote add origin "$PINNED_CORPUS_URL"
git -C "$XLOG_PINNED_CORPUS_ROOT" fetch --quiet --depth=1 --no-tags origin "$PINNED_CORPUS_SHA"
git -C "$XLOG_PINNED_CORPUS_ROOT" -c advice.detachedHead=false checkout --quiet --detach FETCH_HEAD
git -C "$XLOG_PINNED_CORPUS_ROOT" submodule sync --quiet --recursive
git -C "$XLOG_PINNED_CORPUS_ROOT" submodule update --init --recursive --depth=1

test "$(git -C "$XLOG_PINNED_CORPUS_ROOT" rev-parse HEAD)" = "$PINNED_CORPUS_SHA"
test -z "$(git -C "$XLOG_PINNED_CORPUS_ROOT" status --porcelain=v1 --untracked-files=all)"
git -C "$XLOG_PINNED_CORPUS_ROOT" diff --quiet --no-ext-diff --submodule=diff
git -C "$XLOG_PINNED_CORPUS_ROOT" diff --cached --quiet --no-ext-diff --submodule=diff
submodule_status="$(git -C "$XLOG_PINNED_CORPUS_ROOT" submodule status --recursive)"
while IFS= read -r line; do
  test -z "$line" || test "${line:0:1}" = " "
done <<< "$submodule_status"
git -C "$XLOG_PINNED_CORPUS_ROOT" submodule foreach --quiet --recursive \
  'test -z "$(git status --porcelain=v1 --untracked-files=all)"'

scripts/validate_release_gpu.sh --mode release
```

The script does the following:

- sets the `XLOG_REQUIRE_CUDA=1` environment variable, which forces a hard
  failure if CUDA cannot initialize (so a broken GPU cannot be silently
  skipped);
- requires a visible NVIDIA GPU through `nvidia-smi`;
- runs release doctor checks;
- builds the workspace and `xlog-cli` release binary;
- stages Python and CLI kernel artifacts;
- builds the `pyxlog` wheel and CLI archive;
- creates a fresh virtual environment with the host's site packages visible,
  force-installs that exact wheel without resolving dependencies, verifies the
  imported native module, and runs the native relation-provenance, contract,
  public-API, and callback suites with CUDA required;
- runs the host-readable probabilistic CLI suite and the accepted epistemic GPU
  evidence adapter suite in release mode;
- enables the resident-graph test feature and requires exactly 20 passing
  runtime-module tests with no failures, ignored tests, measured tests, or
  CUDA-skip messages;
- runs the exact prepare-only, production-launch, disconnected-rule scaling,
  and semantic-matrix acceptance tests, requiring exactly one passing test from
  each filter;
- runs `xlog-cuda-tests` certification in release mode;
- runs a basic `xlog run` smoke command;
- verifies that packaged artifacts include the expected kernel files.

Use `--mode smoke` for a shorter CUDA smoke gate. Use `--dry-run` only to
inspect the command sequence; it does not certify GPU behavior.

### Run only the resident acceptance gates

Use these commands when reviewing the resident recursive path without building
release archives. Keep `XLOG_REQUIRE_CUDA=1` set so missing CUDA fails instead of
turning a skipped test into a pass. The prepare, production, and scaling tests
also require the clean `XLOG_PINNED_CORPUS_ROOT` checkout prepared above.

```bash
export XLOG_REQUIRE_CUDA=1

cargo test --locked --release -p xlog-runtime --lib \
  --features resident-graph-tests \
  executor::resident_graph_tests -- \
  --nocapture --test-threads=1

cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::pinned_corpus_prepares_resident_graph_without_launching_it -- \
  --ignored --exact --nocapture --test-threads=1

cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::pinned_corpus_certifies_and_runs_through_the_resident_production_path -- \
  --ignored --exact --nocapture --test-threads=1

cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::resident_disconnected_four_thousand_rule_scaling_acceptance -- \
  --ignored --exact --nocapture --test-threads=1

cargo test --locked --release -p xlog-gpu --lib \
  logic::tests::resident_semantic_acceptance_matrix -- \
  --ignored --exact --nocapture --test-threads=1
```

The workflow and release script add a fail-closed summary check around these
commands. It requires 20 passes from the runtime module and one pass from each
exact filter, with zero failures, ignored tests, measured tests, or CUDA-skip
messages. A raw Cargo exit code without those counts is not acceptance evidence.

The production test performs five serialized required-resident runs and requires
an end-to-end median at or below 1.25 seconds and every run at or below 1.75
seconds. Ordinary compilation eagerly seeds its resident-plan certification;
unsupported routes or certification errors remain cached selection evidence and
do not make compilation fail. The five evaluation timers therefore exclude
compilation and certification. The test reports compile-and-certification time
and compile-plus-first-resident time separately from those five samples. The
test also executes the full ordinary plan and the certified dependency-closed
ordinary plan through the disabled-resident evaluator. Their query results must
match. The resident semantic scan and filter counters are compared with the
actual operation profile plus successful-chain equivalents from that
dependency-closed execution, while the full-plan execution remains the query
and full-store reference. Resident physical operation statistics continue to
describe work that actually executed. The scaling test compares five
required-resident runs on the pinned base corpus with
five required-resident runs after adding 4,000 unqueried disconnected rules.
The augmented median may be no more than 10% or 100 milliseconds slower,
whichever allowance is larger. The semantic matrix checks recursion, negation,
constraints, multiple queries, nullary relations, same-named relations with
different arities, caller-provided inputs, and documented non-resident cases.

## Docs Gate

This gate validates the documentation site. It triggers when documentation or a
documentation-build input changes, including the docs tree, build scripts and
deployment configuration, workspace manifests and crates, or the workflow
itself. It uses Node 22, installs `mint@4.2.666`, and runs:

```bash
mint validate
mint broken-links
mint export
```

On `main`, the exported static bundle is pushed to the `docs-dist` branch. The
DigitalOcean App Platform site serves that branch at `xlog.md`.

## Reliability Gates

Reliability gates measure how consistently a higher-level engine produces
correct results across repeated runs. They are not the same as CUDA kernel
certification. The staged reliability labels used in the repository are:

- alpha: 5/5;
- beta: 20/20, defined as 5 seeds across 4 stages;
- GA: 50/50 with Clopper-Pearson confidence accounting. Clopper-Pearson is a
  conservative way to put a confidence interval on a pass rate from a limited
  number of trials.

These labels belong only to the subsystem that defines and runs that gate. They
are not global CUDA test counts, so do not reuse them that way.

## Single-pass epistemic candidate bounds

The Generate-Propagate-Test route enumerates candidate answers over a program's
epistemic literals (the atoms whose truth the reasoning is uncertain about).
This single-pass route has no public fixed-literal limit and no large hardcoded
candidate-count bound. Instead, it computes two concrete bounds:

- `max_candidates = 2^(number of epistemic literals)`;
- a per-reduction limit held in the configurable `max_models_per_reduction`
  field, which defaults to `DEFAULT_EPISTEMIC_MAX_MODELS_PER_REDUCTION = 1024`
  models.

These limits apply to Generate-Propagate-Test reductions only. Three other
epistemic routes never enumerate the candidate lattice, so these bounds do not
constrain them:

- the default `faeel` reduction, which is *founded* — a fact is known only if
  some rule derives it — and computes an ordinary least fixpoint;
- the `g91` route for supported exact-tuple `possible` cycles, which runs a GPU
  upper-bound pass and then descends through frozen-snapshot refinements to the
  greatest set of mutually compatible tuples;
- the well-founded-semantics route for cycles through negation, the standard
  three-valued treatment where a fact can be true, false, or undefined.

The `g91` and well-founded routes keep relation evaluation and set comparison on
the GPU while the host orchestrates fixpoint iterations. For what those routes do
and where they stop, see
[Epistemic Execution Internals](/architecture/epistemic-internals).

## Evidence Requirements

A certification claim should include:

- exact command;
- commit or release tag;
- CUDA toolkit and GPU class;
- whether `XLOG_REQUIRE_CUDA=1` was active;
- route counters or transfer telemetry when the claim depends on a specific
  optimized path;
- artifact or log location when the evidence is durable.

If those details are missing, phrase the result as a local check, not a release
certification.
