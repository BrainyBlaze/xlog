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

The four layers below run in different places and prove different things.

| Layer | Where it runs | What it proves |
| --- | --- | --- |
| GitHub CI | GitHub-hosted runners | Formatting, workflow hygiene, package metadata, no-GPU CUDA build, and non-GPU checks. |
| Docs site CI | GitHub Actions for `docs/**` | Mintlify validation, broken-link checks, and static export publication. |
| GPU release validation | Maintainer-run CUDA host | Actual CUDA behavior through `scripts/validate_release_gpu.sh`. |
| Subsystem reliability gates | Subsystem-specific suites | Statistical or staged reliability for neural-symbolic and other higher-level engines. |

Green GitHub CI does not certify GPU correctness. GPU certification requires a
CUDA machine.

## GPU Release Gate

This is the canonical manual gate for GPU behavior. A maintainer runs it on a
real CUDA host before a release:

```bash
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
- runs `xlog-cuda-tests` certification in release mode;
- runs a basic `xlog run` smoke command;
- verifies that packaged artifacts include the expected kernel files.

Use `--mode smoke` for a shorter CUDA smoke gate. Use `--dry-run` only to
inspect the command sequence; it does not certify GPU behavior.

## Docs Gate

This gate validates the documentation site. It triggers on edits under
`docs/**` or on changes to the workflow itself. It uses Node 22, installs
`mint@4.2.666`, and runs:

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

## Epistemic Candidate Bounds

"Epistemic execution" is XLOG's mode for reasoning about what a program can and
cannot conclude — it enumerates candidate answers over the program's epistemic
literals (the atoms whose truth the reasoning is uncertain about). This mode has
no public fixed-literal limit and no large hardcoded candidate-count bound.
Instead, the source computes two concrete bounds:

- `max_candidates = 2^(number of epistemic literals)`;
- a per-reduction cap of `MAX_MODELS_PER_REDUCTION = 1024` models.

Those are the actual limits that apply to epistemic planning.

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
