# Release Readiness Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the proven public-release readiness gaps without enabling branch rulesets or changing repository visibility yet.

**Architecture:** Keep the fixes narrowly scoped to the audited problem areas: public-facing docs, release/validation scripts, workflow definitions, maintainer metadata, and a small set of high-signal warning cleanups. Apply repository-side GitHub settings only where they are safe before merge; defer settings that would break current unmerged workflow files.

**Tech Stack:** Markdown docs, GitHub Actions YAML, shell scripts, Python CLI helpers/tests, Rust crates, GitHub REST API via `gh`

---

### Task 1: Align Public Docs With The Actual Release Surface

**Files:**
- Modify: `README.md`
- Modify: `docs/architecture/python-bindings.md`
- Modify: `docs/ROADMAP.md`
- Modify: `docs/release-process.md`
- Test: `python/tests/test_validate_package_metadata.py`
- Test: `python/tests/test_sync_readme_release_version.py`

- [ ] Update install and limitation docs so published GitHub/PyPI/crates.io artifacts are described as available, while `main` is documented as potentially ahead of the latest published tag.
- [ ] Fix the README license badge target and any stale “planned install” wording in current user-facing docs.
- [ ] Update release-process wording that still speaks about “before the first public release” when it now describes the active release flow.
- [ ] Extend or adjust Python tests so the release README helpers still validate the intended contract.

### Task 2: Repair Release Preflight And Metadata Validation UX

**Files:**
- Modify: `scripts/preflight_release_publish.sh`
- Modify: `scripts/validate_package_metadata.py`
- Modify: `python/tests/test_validate_package_metadata.py`
- Create or Modify: `python/tests/test_preflight_release_publish_cli.py`

- [ ] Replace the current `cargo publish --dry-run` preflight with a package-validation path that works for unreleased workspace version waves.
- [ ] Make `validate_package_metadata.py` self-sufficient when invoked locally without a pre-generated `cargo-metadata.json`.
- [ ] Add or update targeted tests for the metadata helper and preflight CLI behavior.

### Task 3: Clean Up Workflow Hygiene And Action Pinning

**Files:**
- Modify: `.github/workflows/bench.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/cuda-ci.yml`
- Modify: `.github/workflows/fuzz.yml`
- Modify: `.github/workflows/github-release.yml`
- Modify: `.github/workflows/python-publish.yml`
- Modify: `.github/workflows/release-plz.yml`

- [ ] Make `bench.yml` internally consistent with its manual-only execution model.
- [ ] Pin third-party GitHub Actions to immutable commit SHAs across workflow files.
- [ ] Keep behavior unchanged except where the audit already proved the workflow logic was dead or misleading.

### Task 4: Improve Maintainer-Facing OSS Surface

**Files:**
- Modify: `Makefile`
- Modify: `CONTRIBUTING.md`
- Create: `SUPPORT.md`
- Create: `.github/CODEOWNERS`
- Create: `.github/FUNDING.yml`
- Modify: public crate `Cargo.toml` files as needed

- [ ] Make local lint targets fail with actionable install guidance instead of raw command-not-found errors.
- [ ] Add missing public maintainer metadata files with real, non-placeholder content.
- [ ] Improve publishable crate metadata where it is safe to do so without inventing an unverified MSRV contract.

### Task 5: Reduce High-Signal Warning Debt

**Files:**
- Modify: `crates/xlog-cuda/src/arrow_device.rs`
- Modify: `crates/xlog-cuda/src/cuda_compat.rs`
- Modify: `crates/xlog-cuda/src/provider/filter.rs`
- Modify: `crates/xlog-logic/tests/optimizer_integration.rs`
- Modify: `crates/pyxlog/src/program.rs`

- [ ] Fix the unsafe/documentation warnings and obvious correctness-adjacent clippy findings identified in the audit.
- [ ] Avoid broad warning sweeps; keep the cleanup to the highest-signal sites already inspected.

### Task 6: Apply Safe Repository-Side GitHub Settings

**Files:**
- No repo files; apply live settings via `gh`

- [ ] Enable vulnerability alerts and automated security fixes.
- [ ] Enable dependabot security updates / secret scanning enhancements that are safe on the current private repo.
- [ ] Defer Actions SHA pinning enforcement until the workflow pinning changes are merged on `main`.
