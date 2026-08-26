# Contributing to XLOG

Thanks for contributing. Keep changes small, testable, and aligned with the current public support contract.

## Engineering Standard

Read [XLOG Engineering Standards](ENGINEERING.md) before proposing a change. It is the
merge contract for research, reuse, root-cause fixes, production-path testing, removal
of obsolete code, reproducible benchmarks, and honest evidence. A change is not ready
for review if it adds a duplicate implementation, silent fallback, compatibility shim,
placeholder, unused public surface, or required follow-up debt.

Before editing, trace the complete affected production path and search the codebase,
tests, examples, documentation, and relevant history for the existing implementation.
Bug fixes must include a reproduction; features must identify a real supported entry
point and consumer.

## Supported Platform

The first public support contract is:

- Linux `x86_64`
- NVIDIA GPU
- CUDA Toolkit 13.x

General GitHub-hosted pull-request CI is non-GPU. The path-filtered CUDA workflow's
Python wheel job also runs for qualifying same-repository pull requests and pushes to
`main`; fork pull requests are excluded from the persistent runner. Maintainers can
dispatch the remaining CUDA suites from `main` for an explicitly reviewed full commit
SHA. Fork code must be validated on an isolated CUDA host. If you change CUDA kernels,
GPU execution paths, packaging, or installation flows, validate on supported hardware
before asking for review.

## Local Setup

Run the setup doctor before building:

```bash
make doctor
```

Common local build commands:

```bash
make build
make build-host-io
```

If you want local parity with the workflow/shell lint jobs, install:

- `shellcheck` via your package manager, for example `sudo apt-get install -y shellcheck`
- `actionlint` from the release binary that CI uses in `.github/workflows/ci.yml`

## Local Checks

Run the checks that match your change before opening a pull request.

Minimum checks for most changes:

```bash
cargo fmt --all --check
make check
```

Recommended checks when your change affects examples, docs that reference commands, or Python packaging:

```bash
python scripts/validate_examples.py --mode ci
```

Required on a supported CUDA machine when your change affects CUDA kernels or
low-level GPU execution:

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

When your change affects `pyxlog`, persistent relations, DLPack ownership or
stream ordering, Python packaging, or the release path, run the complete release
validator instead:

```bash
bash scripts/validate_release_gpu.sh --mode release
```

That validator builds the distributable artifacts, installs the exact wheel it
produced, runs the native relation and callback suites with CUDA required, and
then runs the CUDA certification suite. The focused certification command is not
a substitute for this package-level gate.

If you build `pyxlog` wheels or run ad-hoc Python probes against saved artifacts, keep the kernel
path explicit. The packaged wheel should ship `pyxlog/kernels/`, but source-tree and probe
workflows should still export `XLOG_CUBIN_DIR` before importing `pyxlog`:

```bash
export XLOG_CUBIN_DIR=$PWD/crates/pyxlog/python/pyxlog/kernels
```

## Pull Request Expectations

`main` is protected by an active repository ruleset with no bypass actors. Every
change must arrive through a pull request, resolve its review threads, and pass the
up-to-date universal jobs in `.github/workflows/ci.yml`; force pushes to `main` and
deletion of `main` are prohibited. Path-filtered hardware workflows remain additional gates
for changes in their scope rather than substitutes for the universal checks.

Each pull request should:

- explain the root cause or production-path design and why the existing architecture is the correct owner
- describe the user-visible change and the risk area
- state whether the work was validated on the supported Linux `x86_64` + NVIDIA CUDA platform
- list the exact commands you ran locally
- include reproduction steps for bug fixes
- include behavior-level tests for new or changed semantics
- update docs when behavior, setup, or support expectations changed
- stay focused; separate unrelated refactors into different pull requests
- remove code made obsolete by the change and introduce no placeholders, silent fallbacks, or unresolved required debt

If a change was only validated in GitHub-hosted CI, say so explicitly. That is useful signal, but it does not replace real GPU validation for CUDA-facing changes.

## Review Notes

- Prefer targeted changes over broad cleanup.
- Verify that a proposed helper or API reuses the canonical implementation instead of duplicating it.
- Reject check suppression, weakened assertions, test-only success paths, and compatibility code that was not explicitly approved.
- Do not merge changes that weaken the supported-platform story without updating the docs and templates in the same pull request.
- If you are unsure whether something needs GPU validation, assume it does and call that out in the PR.
