# Public Release CI/CD Design

**Date:** 2026-04-19
**Status:** Approved design direction, pending implementation plan
**Scope:** First public open-source release of XLOG

## Goal

Prepare XLOG for public open-source release with:

- reproducible source setup for supported users
- public GitHub release binaries
- PyPI wheel publishing for `pyxlog`
- crates.io publishing for the crates that should be public
- non-GPU validation in GitHub Actions
- GPU validation outside GitHub Actions on a real CUDA machine

## Explicit Support Contract

The first public release supports exactly:

- Linux
- x86_64
- NVIDIA CUDA

Everything else is unsupported for the initial public release:

- no CPU-only support target
- no macOS support target
- no Windows support target
- no GitHub-hosted GPU CI

The repository should fail fast and explain this clearly in docs and setup tooling.

## Constraints

1. GitHub Actions must not be responsible for running GPU validation because the required GPU instance is not available there.
2. Public CI still needs value, so non-GPU checks should run automatically on pull requests.
3. The source checkout experience must work on the current supported environment without requiring hidden tribal knowledge.
4. Runtime kernel artifacts must be generated and packaged, not stored as tracked build output in git.

## Current Problems

### 1. PTX artifacts are tracked in git

`kernels/*.ptx` are currently tracked even though the runtime already has a build-time artifact model:

- `crates/xlog-cuda/build.rs` compiles kernels into `OUT_DIR`
- runtime resolution already prefers generated cubins/PTX from `XLOG_CUBIN_DIR`, packaged `kernels/`, or `OUT_DIR`

This means the tracked PTX files are legacy baggage and create drift risk.

### 2. Quickstart does not match real install behavior

The README currently tells users to:

```bash
cargo build --release
xlog run reachability.xlog
```

That is incorrect for a fresh shell because `cargo build` does not install `xlog` onto `PATH`.
The working command is currently:

```bash
./target/release/xlog run reachability.xlog
```

### 3. Probabilistic CLI setup is easy to misuse

Host-readable `xlog prob` output depends on building `xlog-cli` with `--features host-io`.
This must be reflected consistently in setup docs and release artifacts.

### 4. Python wheel packaging is not release-safe

The current `pyxlog` install places `_native.so` in the package, but not packaged kernel artifacts.
That implies a wheel may accidentally depend on build-tree state such as `OUT_DIR`, which is not acceptable for public distribution.

### 5. Existing GitHub Actions are not valid public CI

Current workflows are manual-only and partially dead:

- they only trigger on `workflow_dispatch`
- some contain `pull_request` / `push` branches that can never execute
- they are built around self-hosted CUDA runners

### 6. Existing release validation command is internally inconsistent

The current CUDA workflow uses:

```bash
cargo test --workspace --all-targets --release
```

That does not currently represent a valid universal release gate because PyO3-linked test surfaces do not fit that command model.

### 7. Public repository governance is missing

The repo currently lacks standard public OSS scaffolding:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- issue forms
- PR template
- Dependabot configuration
- explicit crate publishing policy

## Design Principles

### 1. Split validation planes

There are two different validation surfaces and they must remain separate:

- GitHub-hosted CI validates everything that does not require a GPU
- release/gating validation on a real CUDA machine validates GPU functionality

The design should not pretend those are the same thing.

### 2. Generated artifacts stay out of git

Kernel outputs are build artifacts. Only source inputs should be tracked.

Tracked:

- `kernels/*.cu`
- kernel manifest definitions
- packaging scripts

Not tracked:

- `kernels/*.ptx`
- generated cubins
- generated staged release bundles

### 3. Public install flows must be explicit

Users need a clear path for each supported installation mode:

- source checkout
- GitHub release binary
- PyPI install
- crates.io install where applicable

Each path needs a matching artifact layout and a matching doc path.

### 4. Keep tooling boring

This release should prefer small, composable, well-known tools over a larger release framework that hides too much:

- `release-plz` for Rust release automation
- `maturin` / `maturin-action` for Python wheels
- thin repository scripts for setup and validation
- GitHub Actions only for non-GPU automation

## Recommended Architecture

### A. Artifact Model

### Runtime kernel lookup

Kernel lookup should support three valid runtime locations:

1. explicit override via `XLOG_CUBIN_DIR`
2. package-adjacent or binary-adjacent `kernels/`
3. `OUT_DIR` for in-tree development builds

This is already close to the current design for the CLI/binary path. The implementation must be extended and normalized so Python package installs also resolve packaged kernels reliably.

### Release artifact layout

#### GitHub binary release

Each release archive should contain:

- `bin/xlog` or `xlog`
- `kernels/`
- `LICENSE*`
- `README` / install note

#### PyPI wheel

The wheel should contain:

- `pyxlog/_native.*.so`
- package-visible kernel data directory
- Python wrapper code

Wheel runtime must not require access to the build tree.

### PTX generation policy

- remove tracked `kernels/*.ptx` from git
- add `kernels/*.ptx` and generated cubins to `.gitignore`
- update tests that currently read PTX from the repo root to read generated or staged artifacts instead
- keep one canonical manifest for kernel source names so build scripts, packaging, and tests stay aligned

### B. Setup and Developer UX

### Add a first-class environment doctor

Introduce a small setup/preflight tool, e.g. `scripts/xlog_doctor.py`, with a stable command wrapper:

```bash
python scripts/xlog_doctor.py
```

or

```bash
make doctor
```

The doctor should verify:

- OS is Linux
- architecture is x86_64
- NVIDIA driver is visible
- `nvcc` is available
- Rust toolchain is installed
- Python is installed
- CUDA loader compatibility is satisfied, including WSL shim conditions if needed
- whether the requested workflow needs `host-io`

The output should be explicit and actionable:

- PASS
- FAIL with concrete next step
- UNSUPPORTED with explanation

### Quickstart restructuring

The README should separate:

1. source quickstart
2. binary install
3. Python install

The source quickstart should use the real binary path unless an install step is performed:

```bash
./target/release/xlog run reachability.xlog
```

### Thin task entrypoints

Add a small top-level `Makefile` to standardize common commands:

- `make doctor`
- `make build`
- `make build-host-io`
- `make check`
- `make package`
- `make validate-release-local`

The Makefile should delegate to scripts and cargo rather than reimplement logic.

### C. CI Design

### PR CI on GitHub-hosted runners

These jobs should run automatically on pull requests and pushes to main:

1. workflow lint
   - `actionlint`

2. shell/script lint
   - `shellcheck` on `scripts/*.sh`

3. formatting
   - `cargo fmt --check`

4. static Rust lint
   - `cargo clippy --workspace --all-targets -- -D warnings`

5. packaging metadata validation
   - `cargo metadata`
   - `maturin pep517 check` or equivalent wheel metadata validation
   - release config validation

6. docs/setup validation
   - verify documented quickstart commands stay accurate
   - verify doctor command runs

7. non-GPU build validation in a CUDA toolkit container
   - build workspace in a container image that provides `nvcc`
   - do not execute GPU tests

This CI is not a correctness proof for GPU behavior. It is a guardrail against broken packaging, broken setup, broken scripts, and broken docs.

### GPU validation lane outside Actions

The release gate should be a canonical script that runs on a real CUDA machine.

Example responsibilities:

- build release artifacts
- run selected GPU Rust tests
- run example validation
- run Python smoke tests
- verify packaged CLI and wheel against staged kernel assets

This should be one script with machine-readable exit status, not a collection of tribal commands.

### D. Release Automation

### Rust releases

Use `release-plz` for:

- release PR creation
- version bumping
- changelog updates
- tagging
- publishing selected crates to crates.io

Not every workspace crate should necessarily be published. Internal crates should be marked `publish = false`.

### Python releases

Use `maturin` / `maturin-action` for:

- wheel build
- wheel verification
- PyPI publish

The wheel publish job must consume the same staged kernel layout that local release validation checks.

### GitHub releases

GitHub releases should attach:

- CLI archives
- checksums
- wheel artifacts if desired for traceability
- release notes generated from the release process

### E. Public Repository Hygiene

Add:

- `CONTRIBUTING.md`
- `CODE_OF_CONDUCT.md`
- `SECURITY.md`
- `.github/ISSUE_TEMPLATE/` forms
- `.github/PULL_REQUEST_TEMPLATE.md`
- `.github/dependabot.yml`

Dependabot should cover:

- Cargo
- GitHub Actions
- Python packaging files where relevant

## Non-Goals

These are intentionally out of scope for the first public release:

- supporting CPU-only execution
- supporting non-Linux platforms
- running GPU tests on GitHub-hosted Actions
- introducing a large meta-toolchain such as `mise` or `cargo-dist`
- solving every historical documentation artifact in the repository

## Implementation Order

1. Fix kernel artifact model
   - stop tracking PTX
   - update runtime lookup and packaging
   - update tests that assume repo-root PTX

2. Fix public setup path
   - add doctor
   - fix README quickstart
   - add Makefile wrappers

3. Add non-GPU GitHub Actions
   - lint
   - formatting
   - non-GPU build/package checks

4. Add release automation
   - `release-plz`
   - wheel publish
   - GitHub release packaging

5. Add public OSS hygiene files

6. Add and document the GPU release-validation script

## Risks

### Packaging risk

The highest-risk part of this work is making kernel artifacts resolve correctly from:

- development builds
- release binaries
- Python wheels

This must be treated as the primary integration seam.

### False-green CI risk

If GitHub Actions are described as “CI passed” without clarifying that GPU validation is separate, contributors will infer stronger guarantees than actually exist.

### Release surface sprawl

Publishing too many internal crates on crates.io creates long-term maintenance burden. Public crate selection should be explicit, conservative, and documented.

## Recommendation

Proceed with a split-plane release model:

- GitHub-hosted Actions for non-GPU checks only
- GPU release validation on a real CUDA machine outside Actions
- generated kernel artifacts packaged into release outputs
- explicit Linux+x86_64+CUDA-only support contract

This is the most robust design that fits the actual XLOG architecture and the available infrastructure.
