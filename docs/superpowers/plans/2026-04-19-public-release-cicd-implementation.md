# Public Release CI/CD Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prepare XLOG for a first public open-source release with Linux+x86_64+CUDA-only support, generated kernel artifacts, correct source/binary/PyPI setup flows, non-GPU GitHub Actions, explicit GPU release validation outside Actions, and standard public OSS repository scaffolding.

**Architecture:** Split the work into six independent layers: kernel artifact model, setup UX, non-GPU CI, release automation, OSS governance, and GPU release validation. Keep runtime kernel lookup centered on generated artifacts (`OUT_DIR`, package-adjacent `kernels/`, or `XLOG_CUBIN_DIR`) and make every public install path consume the same staged kernel layout.

**Tech Stack:** Rust workspace, Cargo, PyO3/maturin, GitHub Actions, release-plz, shell/Python utility scripts, Contributor Covenant templates, Dependabot

**Spec:** `docs/superpowers/specs/2026-04-19-public-release-cicd-design.md`

---

## File Structure

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `.gitignore` | Stop allowing tracked PTX files and ignore generated kernel artifacts |
| Delete | `kernels/*.ptx` | Remove tracked generated PTX files from git |
| Create | `Makefile` | Stable entrypoints for doctor/build/check/package/release validation |
| Create | `scripts/xlog_doctor.py` | Supported-environment preflight checker |
| Create | `scripts/stage_kernels.py` | Copy generated cubins/PTX into a release/package-ready `kernels/` directory |
| Create | `scripts/package_cli_release.sh` | Assemble GitHub binary release archive |
| Create | `scripts/validate_release_gpu.sh` | Canonical real-GPU release validation entrypoint |
| Modify | `README.md` | Correct install paths, quickstart, support contract, release install docs |
| Modify | `crates/xlog-cuda/build.rs` | Emit/stage data expected by runtime + packaging scripts |
| Modify | `crates/xlog-cuda/src/provider/mod.rs` or split helper file | Normalize kernel path resolution logic |
| Create | `crates/xlog-cuda/src/provider/kernel_paths.rs` | Isolate runtime kernel path resolution |
| Modify | `crates/xlog-cuda/tests/ptx_validation.rs` | Validate generated/staged kernel artifacts instead of repo-root PTX |
| Modify | `crates/xlog-cuda-tests/src/categories/c01_toolchain.rs` | Stop assuming repo-tracked PTX files |
| Modify | `crates/pyxlog/pyproject.toml` | Include packaged kernel assets in wheel/sdist |
| Modify | `crates/pyxlog/python/pyxlog/__init__.py` and/or Python helper | Point wheel/runtime at package-adjacent kernels |
| Create | `python/tests/test_xlog_doctor.py` | Smoke tests for doctor CLI behavior |
| Create | `python/tests/test_kernel_packaging_layout.py` | Verify package/binary kernel layout invariants |
| Create | `.github/workflows/ci.yml` | PR/push non-GPU CI |
| Create | `.github/workflows/release-plz.yml` | Rust release automation |
| Create | `.github/workflows/python-publish.yml` | Wheel build/publish automation |
| Create | `.github/workflows/github-release.yml` | Binary release packaging/upload |
| Create | `release-plz.toml` | Workspace release policy |
| Modify | `Cargo.toml` and selected `crates/*/Cargo.toml` | Publish policy, workspace metadata, release automation support |
| Create | `CONTRIBUTING.md` | Public contribution process |
| Create | `CODE_OF_CONDUCT.md` | Community standards |
| Create | `SECURITY.md` | Security reporting path and support policy |
| Create | `.github/PULL_REQUEST_TEMPLATE.md` | PR checklist |
| Create | `.github/ISSUE_TEMPLATE/config.yml` | Issue chooser config |
| Create | `.github/ISSUE_TEMPLATE/bug_report.yml` | Bug intake form |
| Create | `.github/ISSUE_TEMPLATE/feature_request.yml` | Feature intake form |
| Create | `.github/dependabot.yml` | Dependency update policy |
| Create | `docs/release-process.md` | Human-readable release procedure and GPU gate |

---

## Task 1: Fix Kernel Artifact Model

**Files:**
- Modify: `.gitignore`
- Delete: `kernels/*.ptx`
- Create: `scripts/stage_kernels.py`
- Modify: `crates/xlog-cuda/build.rs`
- Modify/Create: `crates/xlog-cuda/src/provider/kernel_paths.rs`
- Modify: `crates/xlog-cuda/src/provider/mod.rs`
- Modify: `crates/xlog-cuda/tests/ptx_validation.rs`
- Modify: `crates/xlog-cuda-tests/src/categories/c01_toolchain.rs`
- Test: `python/tests/test_kernel_packaging_layout.py`

- [ ] **Step 1: Write failing tests for generated/staged kernel lookup**

Create `python/tests/test_kernel_packaging_layout.py` with two focused checks:

```python
from pathlib import Path

def test_repo_does_not_require_tracked_ptx_files():
    kernels = Path("kernels")
    assert not any(p.suffix == ".ptx" for p in kernels.iterdir())

def test_stage_kernels_help():
    ...
```

Add a Rust-side unit test around the kernel path resolver that prefers:

1. `XLOG_CUBIN_DIR`
2. package/binary-adjacent `kernels/`
3. `OUT_DIR`

- [ ] **Step 2: Run tests to verify they fail on the current state**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_kernel_packaging_layout.py
cargo test -q -p xlog-cuda kernel_path
```

Expected:

- Python test fails because `.ptx` files are still tracked
- Rust resolver test fails because the helper does not exist yet

- [ ] **Step 3: Move kernel path resolution into an isolated helper**

Create `crates/xlog-cuda/src/provider/kernel_paths.rs` and move all path precedence logic out of `provider/mod.rs`.

Implement a small API such as:

```rust
pub struct KernelArtifactLocator { ... }

impl KernelArtifactLocator {
    pub fn resolve_module_path(name: &str, cc: u32) -> Option<(PathBuf, bool)> { ... }
}
```

Requirements:

- no direct dependence on repo-root `kernels/*.ptx`
- explicit support for staged package/binary `kernels/`
- explicit unit tests for precedence order

- [ ] **Step 4: Add a staging script and update build/package seams**

Create `scripts/stage_kernels.py` that:

- locates generated cubins/PTX in the build output
- copies them into a specified `kernels/` destination
- prints a deterministic manifest of staged files

Required CLI:

```bash
python scripts/stage_kernels.py --from-out-dir <dir> --to <dir>
```

Update `crates/xlog-cuda/build.rs` comments and emitted metadata so the staging script has one canonical source of truth.

- [ ] **Step 5: Remove tracked PTX and update tests**

Run:

```bash
git rm kernels/*.ptx
```

Update:

- `.gitignore` to ignore generated `.ptx` and cubins
- `crates/xlog-cuda/tests/ptx_validation.rs`
- `crates/xlog-cuda-tests/src/categories/c01_toolchain.rs`

so those tests validate generated or staged kernel assets rather than repo-tracked PTX.

- [ ] **Step 6: Re-run focused checks**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_kernel_packaging_layout.py
cargo test -q -p xlog-cuda kernel_path
cargo test -q -p xlog-cuda --test ptx_validation
```

Expected:

- all pass
- no test assumes `kernels/*.ptx` are tracked source files

- [ ] **Step 7: Commit**

```bash
git add .gitignore kernels scripts/stage_kernels.py crates/xlog-cuda python/tests/test_kernel_packaging_layout.py
git commit -m "build: stage generated CUDA kernels instead of tracking PTX"
```

---

## Task 2: Fix Public Setup UX

**Files:**
- Create: `Makefile`
- Create: `scripts/xlog_doctor.py`
- Modify: `README.md`
- Test: `python/tests/test_xlog_doctor.py`

- [ ] **Step 1: Write failing doctor tests**

Create `python/tests/test_xlog_doctor.py` covering:

- `--help` works
- unsupported platform emits `UNSUPPORTED`
- missing `nvcc` or missing GPU emits actionable `FAIL`
- a no-op smoke path exits `0` on supported env

Minimal shape:

```python
def test_doctor_help():
    ...

def test_doctor_reports_missing_binary():
    ...
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_xlog_doctor.py
```

Expected:

- FAIL because `scripts/xlog_doctor.py` does not exist yet

- [ ] **Step 3: Implement `scripts/xlog_doctor.py`**

The doctor must check:

- `platform.system() == "Linux"`
- `platform.machine() == "x86_64"`
- `nvidia-smi` visibility
- `nvcc --version`
- `rustc` / `cargo`
- Python version
- CUDA loader compatibility including current WSL shim behavior
- whether `--workflow prob-cli` requires `host-io`

CLI shape:

```bash
python scripts/xlog_doctor.py
python scripts/xlog_doctor.py --workflow run-cli
python scripts/xlog_doctor.py --workflow prob-cli
python scripts/xlog_doctor.py --json
```

- [ ] **Step 4: Add a thin Makefile**

Create `Makefile` targets:

- `doctor`
- `build`
- `build-host-io`
- `check`
- `package`
- `validate-release-local`

Each target should delegate to an existing script or cargo command, not hide complex shell logic inside the Makefile.

- [ ] **Step 5: Rewrite README install and quickstart sections**

Update `README.md` so it clearly separates:

1. Supported platform contract
2. Source install
3. GitHub release binary install
4. PyPI install
5. crates.io install if used for the CLI crate

Critical doc corrections:

- source quickstart uses `./target/release/xlog`
- `xlog prob` host-readable examples explicitly require `host-io`
- doctor is the first command after clone

- [ ] **Step 6: Re-run focused checks**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_xlog_doctor.py
python scripts/xlog_doctor.py
make doctor
rg -n './target/release/xlog|Linux x86_64|host-io' README.md
```

Expected:

- doctor tests pass
- doctor prints supported-env summary on this machine
- README contains corrected quickstart/install language

- [ ] **Step 7: Commit**

```bash
git add Makefile scripts/xlog_doctor.py README.md python/tests/test_xlog_doctor.py
git commit -m "docs: add public setup doctor and correct quickstart flow"
```

---

## Task 3: Make Python and Binary Packaging Use the Same Kernel Layout

**Files:**
- Modify: `crates/pyxlog/pyproject.toml`
- Modify: `crates/pyxlog/python/pyxlog/__init__.py`
- Create or modify: `crates/pyxlog/python/pyxlog/_kernel_paths.py`
- Modify: `scripts/stage_kernels.py`
- Create: `scripts/package_cli_release.sh`
- Test: `python/tests/test_kernel_packaging_layout.py`

- [ ] **Step 1: Write a failing package-layout smoke test**

Extend `python/tests/test_kernel_packaging_layout.py` with checks that a staged Python package layout contains:

- `_native.*.so`
- package-adjacent kernel directory
- no runtime dependence on `OUT_DIR`

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_kernel_packaging_layout.py -k package
```

Expected:

- FAIL because wheel/package layout is not yet staged

- [ ] **Step 3: Package kernel assets into the Python distribution**

Update `crates/pyxlog/pyproject.toml` so the wheel/sdist includes staged kernel assets.

Implementation should use a package-local location such as:

```text
crates/pyxlog/python/pyxlog/kernels/
```

or an adjacent data directory with a stable lookup helper.

- [ ] **Step 4: Teach `pyxlog` runtime to find package-local kernels**

Add a Python-side helper or native-side environment bridge so `pyxlog` resolves package kernels before falling back to build-tree behavior.

Preferred behavior:

- wheel install uses package-local kernels
- editable/dev installs can still use `OUT_DIR`

- [ ] **Step 5: Add CLI release packager**

Create `scripts/package_cli_release.sh` that:

- builds the release binary
- stages kernels into a release directory
- writes a tarball layout suitable for GitHub releases

CLI:

```bash
scripts/package_cli_release.sh --output dist/
```

- [ ] **Step 6: Re-run packaging checks**

Run:

```bash
source .venv/bin/activate && pytest -q python/tests/test_kernel_packaging_layout.py
source .venv/bin/activate && maturin build -m crates/pyxlog/Cargo.toml --out /tmp/xlog-wheel-test
bash scripts/package_cli_release.sh --output /tmp/xlog-cli-dist
```

Expected:

- wheel build succeeds
- release bundle contains `xlog` and `kernels/`
- tests confirm staged layout invariants

- [ ] **Step 7: Commit**

```bash
git add crates/pyxlog scripts/package_cli_release.sh scripts/stage_kernels.py python/tests/test_kernel_packaging_layout.py
git commit -m "packaging: stage kernels for pyxlog wheels and CLI releases"
```

---

## Task 4: Add Non-GPU GitHub Actions

**Files:**
- Create: `.github/workflows/ci.yml`
- Possibly create: `.github/actions/` helper scripts only if needed
- Modify: `README.md` badges if current badges become misleading

- [ ] **Step 1: Write a local validation command for workflow syntax**

Use `actionlint` as the local workflow gate and ensure shell-bearing steps are compatible with `shellcheck`.

Document the local commands in the workflow comments and `Makefile`.

- [ ] **Step 2: Create `ci.yml` with PR/push triggers**

Required jobs:

1. `lint-workflows`
2. `lint-shell`
3. `fmt`
4. `clippy`
5. `package-metadata`
6. `cuda-build-no-gpu`

`cuda-build-no-gpu` must run in a CUDA toolkit container so `nvcc` is available but must not execute GPU tests.

- [ ] **Step 3: Make the build job use valid non-GPU commands**

Do **not** use the currently broken release gate:

```bash
cargo test --workspace --all-targets --release
```

Instead split it into safe checks such as:

```bash
cargo build --workspace --release
cargo test --workspace --exclude pyxlog --exclude xlog-neural --release
```

Adjust the exact command set to fit the repo’s PyO3/test model.

- [ ] **Step 4: Add docs/setup validation in CI**

Add a step that runs:

```bash
python scripts/xlog_doctor.py --help
```

and validates the documented quickstart assumptions using lightweight grep or a dedicated script.

- [ ] **Step 5: Validate workflow locally**

Run:

```bash
actionlint
shellcheck scripts/*.sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
```

Expected:

- all pass locally before relying on GitHub

- [ ] **Step 6: Update misleading status badges**

Either:

- replace the current hard-coded CUDA badge with real workflow badges, or
- remove it until a truthful badge structure exists

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/ci.yml README.md
git commit -m "ci: add non-gpu pull request validation workflow"
```

---

## Task 5: Add Release Automation

**Files:**
- Create: `release-plz.toml`
- Create: `.github/workflows/release-plz.yml`
- Create: `.github/workflows/python-publish.yml`
- Create: `.github/workflows/github-release.yml`
- Modify: `Cargo.toml`
- Modify: selected `crates/*/Cargo.toml`
- Create/modify: `docs/release-process.md`

- [ ] **Step 1: Define crate publish policy**

Audit every workspace crate and decide:

- public on crates.io
- internal only (`publish = false`)

Start conservatively. Any crate not intended for external use should be marked `publish = false`.

At minimum, document the decision matrix in `docs/release-process.md`.

- [ ] **Step 2: Encode Rust release policy in Cargo manifests**

Update relevant `Cargo.toml` files with:

- `publish = false` where appropriate
- missing metadata fields if needed for crates.io

- [ ] **Step 3: Add `release-plz.toml`**

Configure:

- workspace release behavior
- changelog generation
- package inclusion/exclusion
- tag naming behavior consistent with the chosen publish surface

- [ ] **Step 4: Create Rust release workflow**

Create `.github/workflows/release-plz.yml` that:

- runs on push to main and manual dispatch
- creates/updates release PRs
- publishes crates when a release PR is merged

Keep permissions minimal and explicit.

- [ ] **Step 5: Create Python publish workflow**

Create `.github/workflows/python-publish.yml` that:

- builds wheels with `maturin`
- validates wheel contents
- publishes to PyPI on tagged release or manual trigger

The build must use the same kernel-staging path validated locally.

- [ ] **Step 6: Create GitHub release packaging workflow**

Create `.github/workflows/github-release.yml` that:

- builds/stages CLI release archives
- uploads checksums
- attaches artifacts to GitHub releases

This workflow must not claim to validate GPU correctness. It packages known-good release inputs only.

- [ ] **Step 7: Validate release config locally**

Run:

```bash
cargo metadata >/dev/null
actionlint
rg -n 'publish = false|release-plz' Cargo.toml crates .github/workflows
```

Expected:

- release config is internally consistent
- workflows lint cleanly

- [ ] **Step 8: Commit**

```bash
git add release-plz.toml .github/workflows Cargo.toml crates docs/release-process.md
git commit -m "release: add Rust, PyPI, and GitHub release automation"
```

---

## Task 6: Add Public OSS Governance Files

**Files:**
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/config.yml`
- Create: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Create: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Create: `.github/dependabot.yml`

- [ ] **Step 1: Create governance documents**

Add concise but real policy files:

- `CONTRIBUTING.md` with supported platform, local checks, and PR expectations
- `CODE_OF_CONDUCT.md` using a standard template
- `SECURITY.md` with vulnerability reporting instructions and support scope

- [ ] **Step 2: Add PR and issue templates**

Templates should collect:

- environment info
- whether the reporter is on supported platform
- repro steps
- expected vs actual behavior
- GPU / driver / CUDA version for relevant bugs

- [ ] **Step 3: Add Dependabot**

Create `.github/dependabot.yml` covering:

- Cargo
- GitHub Actions
- Python ecosystem where applicable

Group updates conservatively to avoid PR spam.

- [ ] **Step 4: Validate docs/templates locally**

Run:

```bash
rg -n 'Linux|CUDA|supported|security|dependabot' CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md .github
```

Expected:

- all governance files exist
- support policy and contribution path are explicit

- [ ] **Step 5: Commit**

```bash
git add CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md .github/PULL_REQUEST_TEMPLATE.md .github/ISSUE_TEMPLATE .github/dependabot.yml
git commit -m "docs: add public repository governance and contribution templates"
```

---

## Task 7: Add Canonical GPU Release Validation Script

**Files:**
- Create: `scripts/validate_release_gpu.sh`
- Modify/Create: `docs/release-process.md`
- Modify: `Makefile`
- Optional Test: `python/tests/test_validate_release_gpu_cli.py` or shell smoke equivalent

- [ ] **Step 1: Write a failing smoke test or dry-run contract**

If using Python test coverage:

```python
def test_validate_release_gpu_help():
    ...
```

If keeping shell-only, require:

```bash
scripts/validate_release_gpu.sh --help
scripts/validate_release_gpu.sh --dry-run
```

to be stable and testable in non-GPU environments.

- [ ] **Step 2: Implement the script**

The script must:

- run doctor
- build release artifacts
- stage kernels
- run selected Rust GPU tests
- run example validation
- run Python smoke validation
- verify release bundle and wheel layouts

Suggested flags:

```bash
scripts/validate_release_gpu.sh --dry-run
scripts/validate_release_gpu.sh --mode release
```

- [ ] **Step 3: Document the human release gate**

In `docs/release-process.md`, document:

1. run non-GPU CI
2. run `scripts/validate_release_gpu.sh` on a real CUDA machine
3. only then publish/create release

Make the split-plane model explicit so contributors do not mistake GitHub CI for full validation.

- [ ] **Step 4: Wire into Makefile**

Add:

```make
validate-release-local:
	./scripts/validate_release_gpu.sh --mode release
```

- [ ] **Step 5: Run local smoke checks**

Run:

```bash
scripts/validate_release_gpu.sh --help
scripts/validate_release_gpu.sh --dry-run
rg -n 'validate_release_gpu|release gate|non-GPU CI' docs/release-process.md Makefile
```

Expected:

- script help/dry-run succeeds without requiring a full release
- docs reflect the real release sequence

- [ ] **Step 6: Commit**

```bash
git add scripts/validate_release_gpu.sh docs/release-process.md Makefile
git commit -m "release: add canonical gpu release validation gate"
```

---

## Final Verification Pass

- [ ] **Step 1: Run repository-level non-GPU checks**

Run:

```bash
actionlint
shellcheck scripts/*.sh
cargo fmt --check
cargo clippy --workspace --all-targets -- -D warnings
source .venv/bin/activate && pytest -q python/tests/test_xlog_doctor.py python/tests/test_kernel_packaging_layout.py
```

Expected:

- all non-GPU checks pass

- [ ] **Step 2: Run build/package verification**

Run:

```bash
cargo build --workspace --release
source .venv/bin/activate && maturin build -m crates/pyxlog/Cargo.toml --out /tmp/xlog-wheel-final
bash scripts/package_cli_release.sh --output /tmp/xlog-cli-final
```

Expected:

- workspace builds in release mode
- wheel builds
- CLI release archive stages kernels correctly

- [ ] **Step 3: Run GPU release validation on the real CUDA machine**

Run:

```bash
./scripts/validate_release_gpu.sh --mode release
```

Expected:

- zero failures
- release bundle and wheel validated on supported hardware

- [ ] **Step 4: Review final diff**

Run:

```bash
git status --short
git diff --stat
```

Expected:

- only intended files changed
- no generated build artifacts accidentally tracked

- [ ] **Step 5: Final release-readiness summary commit**

```bash
git add -A
git commit -m "release: prepare repository for public open-source distribution"
```

