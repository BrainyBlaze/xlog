# Release Process

XLOG uses a split release model:

- GitHub-hosted Actions validate non-GPU concerns only.
- Real CUDA validation happens manually on a supported Linux x86_64 machine before publication.

This keeps public CI honest. A green PR does not prove GPU correctness.

## Supported Release Surface

Public releases currently target:

- Linux `x86_64`
- NVIDIA CUDA Toolkit 12.x
- GitHub release CLI archives
- `pyxlog` wheels on PyPI
- Rust crates on crates.io for the stable public Rust surface

Everything else remains unsupported for the first public release.

## Crate Publish Policy

Published on crates.io:

- `xlog-core`
- `xlog-ir`
- `xlog-cuda`
- `xlog-stats`
- `xlog-runtime`
- `xlog-logic`
- `xlog-solve`
- `xlog-gpu`
- `xlog-prob`
- `xlog-cli`

Internal only (`publish = false`):

- `pyxlog`
- `xlog-neural`
- `xlog-induce`
- `xlog-cuda-tests`

`release-plz` manages only the public crates above. GitHub tags and GitHub releases are emitted for
`xlog-cli` only, using plain repository tags such as `v0.5.1`. The library crates are published to
crates.io as part of the same release wave, but they do not each create their own GitHub release.

## Required Repository Configuration

Before the first public release, configure:

- `CARGO_REGISTRY_TOKEN` GitHub Actions secret for crates.io publishing
- `RELEASE_PLZ_GITHUB_TOKEN` GitHub Actions secret if you want CI to run automatically on
  release-plz-created PRs. Without it, release-plz falls back to `GITHUB_TOKEN`, which can create
  the PR but will not trigger follow-on PR workflows.
- PyPI Trusted Publishing for `.github/workflows/python-publish.yml`
- GitHub Actions environment `pypi` for the PyPI publish job
- Repository setting `Allow GitHub Actions to create and approve pull requests`

## Automation Layout

`.github/workflows/release-plz.yml`:

- on pushes to `main`, runs `release-plz` in `release-pr` mode to open or update the release PR
- on manual `workflow_dispatch`, runs `release-plz` in `release` mode after the maintainer confirms
  that real-GPU validation has already passed

`.github/workflows/python-publish.yml`:

- builds Linux `x86_64` `pyxlog` wheels in the CUDA container for CPython 3.8 through 3.12
- builds a source distribution
- uploads the distributions to the matching GitHub release
- publishes the distributions to PyPI using trusted publishing

`.github/workflows/github-release.yml`:

- builds the host-io-enabled CLI release archive with bundled kernels
- writes `SHA256SUMS`
- uploads the archive and checksums to the matching GitHub release

## Human Release Gate

Do not publish from GitHub Actions before running the manual GPU gate on the release candidate.

1. Wait for the release-plz PR to open or update.
2. Check out the release-plz PR commit on a real supported machine.
3. Run:

```bash
bash scripts/validate_release_gpu.sh --mode release
```

4. Record the machine, commit SHA, and result.
5. Merge the release-plz PR only after that validation passes.
6. Manually run the `release-plz` workflow with:

```text
confirm_gpu_validation = true
gpu_validation_notes = <host / commit / evidence>
```

That manual dispatch publishes crates, uploads the CLI archive to the GitHub release, and publishes
the Python distributions to PyPI.

## Local Commands

Non-GPU release sanity:

```bash
make lint-workflows
make lint-shell
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --no-deps -- \
  -A clippy::approx_constant \
  -D clippy::dbg_macro \
  -D clippy::todo \
  -D clippy::unimplemented
cargo build --workspace --locked --release --exclude pyxlog
cargo build --locked --release -p xlog-cli --features host-io
```

Manual GPU gate:

```bash
bash scripts/validate_release_gpu.sh --mode release
```

Bundle a local CLI release archive:

```bash
make package PACKAGE_OUTPUT=/tmp/xlog-dist
```
