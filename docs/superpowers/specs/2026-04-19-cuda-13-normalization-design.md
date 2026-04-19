# CUDA 13 Normalization Design

**Goal:** Normalize XLOG to a single CUDA 13 story for public release readiness, with public docs describing CUDA 13.x support and build infrastructure pinned to CUDA 13.1.1.

**Decision:** Use a split policy:
- Public contract: `CUDA 13.x`
- Exact build pins: `CUDA 13.1.1`
- Rust binding target: CUDA 13.1-compatible `cudarc` configuration

**In Scope:**
- Workspace CUDA dependency configuration
- GitHub Actions CUDA container images
- Public support/setup docs
- Issue templates
- Doctor messaging and any tests that assert current support guidance

**Out of Scope:**
- Historical reports, experiment logs, and archival design docs that describe past CUDA 12.x or 12.8 runs
- Multi-variant CUDA release artifacts
- GPU runtime validation in GitHub Actions beyond current non-GPU compilation checks

**Approach:**
1. Upgrade the workspace CUDA binding configuration so the repository configuration itself is CUDA 13-aware rather than merely compiling against a newer local toolkit by accident.
2. Pin CI and release container images to `nvidia/cuda:13.1.1-devel-ubuntu22.04` for reproducibility.
3. Update public-facing docs and support templates to describe the supported contract as `CUDA 13.x`.
4. Verify locally on the current machine, which is running CUDA 13.1.

**Risks:**
- `cudarc` API drift between `0.12.x` and current CUDA 13-capable versions may require code changes outside simple manifest edits.
- CUDA 13.x may surface stricter compile-time or link-time behavior than the 12.x-based CI image.
- Some user-facing guidance may implicitly depend on a broader `12.x` statement and need synchronized updates.

**Acceptance Criteria:**
- The repository no longer states public CUDA 12.x support in current source-of-truth files.
- Rust manifests no longer pin CUDA 12-only bindings.
- CI/release workflows no longer use CUDA 12.4.1 images.
- Local verification succeeds on the current CUDA 13.1 machine for the non-GPU checks we can run here.
