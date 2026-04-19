# CUDA 13 Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move XLOG from a mixed CUDA 12/13 configuration to a single CUDA 13 public contract with exact 13.1.1 build pins.

**Architecture:** Update the CUDA dependency/config layer first, then align CI/release toolchains, then update public guidance and support messaging, and finally verify locally on the active CUDA 13.1 machine. Historical documents remain unchanged so the repo does not rewrite past evidence.

**Tech Stack:** Cargo workspace manifests, `cudarc`, GitHub Actions, Bash packaging scripts, Python doctor script, Markdown docs

---

### Task 1: Audit Current CUDA Touchpoints

**Files:**
- Inspect: `Cargo.toml`
- Inspect: `.github/workflows/ci.yml`
- Inspect: `.github/workflows/release-plz.yml`
- Inspect: `.github/workflows/python-publish.yml`
- Inspect: `.github/workflows/github-release.yml`
- Inspect: `README.md`
- Inspect: `CONTRIBUTING.md`
- Inspect: `SECURITY.md`
- Inspect: `docs/BENCHMARKS.md`
- Inspect: `docs/release-process.md`
- Inspect: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Inspect: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Inspect: `scripts/xlog_doctor.py`

- [ ] **Step 1: Locate current source-of-truth CUDA references**

Run: `rg -n "CUDA Toolkit 12|cuda-12060|12\\.4\\.1-devel|CUDA Toolkit 12.x" README.md CONTRIBUTING.md SECURITY.md docs/BENCHMARKS.md docs/release-process.md .github/ISSUE_TEMPLATE scripts Cargo.toml .github/workflows -S`
Expected: references limited to current source-of-truth files, not historical docs

- [ ] **Step 2: Confirm the active local CUDA environment**

Run: `nvcc --version && nvidia-smi`
Expected: local machine reports CUDA 13.1

### Task 2: Upgrade Workspace CUDA Binding Configuration

**Files:**
- Modify: `Cargo.toml`
- Verify: `Cargo.lock`

- [ ] **Step 1: Update the workspace `cudarc` dependency to a CUDA 13-capable release and feature set**

Change `Cargo.toml` so the workspace no longer uses `cuda-12060`.

- [ ] **Step 2: Refresh the lockfile**

Run: `cargo update -p cudarc`
Expected: `Cargo.lock` records the new `cudarc` resolution

- [ ] **Step 3: Compile-check the narrowest CUDA crate first**

Run: `XLOG_NO_CUBIN=1 cargo check -p xlog-cuda --locked`
Expected: PASS or actionable API errors to fix

### Task 3: Re-pin CI and Release Toolchains

**Files:**
- Modify: `.github/workflows/ci.yml`
- Modify: `.github/workflows/release-plz.yml`
- Modify: `.github/workflows/python-publish.yml`
- Modify: `.github/workflows/github-release.yml`

- [ ] **Step 1: Replace CUDA 12.4.1 container images with exact CUDA 13.1.1 images**

Use `nvidia/cuda:13.1.1-devel-ubuntu22.04` consistently.

- [ ] **Step 2: Keep runtime behavior otherwise unchanged**

Do not expand scope into new matrix variants or new GPU job types.

- [ ] **Step 3: Lint workflow YAML**

Run: `make lint-workflows`
Expected: PASS

### Task 4: Update Public Support Guidance

**Files:**
- Modify: `README.md`
- Modify: `CONTRIBUTING.md`
- Modify: `SECURITY.md`
- Modify: `docs/BENCHMARKS.md`
- Modify: `docs/release-process.md`
- Modify: `.github/ISSUE_TEMPLATE/bug_report.yml`
- Modify: `.github/ISSUE_TEMPLATE/feature_request.yml`
- Modify: `scripts/xlog_doctor.py`
- Modify: `python/tests/test_xlog_doctor.py`

- [ ] **Step 1: Replace current support wording from CUDA 12.x to CUDA 13.x**

Update only current support/setup language, not historical evidence documents.

- [ ] **Step 2: Keep exact patch pin out of public docs where a major/minor family is sufficient**

Public docs should say `CUDA 13.x`, not `13.1.1`, unless the file is specifically about CI/release implementation.

- [ ] **Step 3: Update tests that assert doctor text**

Run: `python -m pytest python/tests/test_xlog_doctor.py -q`
Expected: PASS

### Task 5: Full Local Verification

**Files:**
- Verify: workspace and changed files

- [ ] **Step 1: Run formatting/sanity checks**

Run: `git diff --check`
Expected: PASS

- [ ] **Step 2: Run workflow and tracked-ignore checks**

Run: `make lint-workflows && make check-tracked-ignored`
Expected: PASS

- [ ] **Step 3: Run the non-GPU Rust/package verification that is practical locally**

Run: `XLOG_NO_CUBIN=1 cargo check -p xlog-cuda --locked`
Expected: PASS

- [ ] **Step 4: Record residual risk explicitly**

If full GPU execution or GitHub-hosted release publish cannot be validated here, document that clearly in the completion summary.
