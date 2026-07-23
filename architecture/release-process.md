# Release Process

XLOG's release boundaries for crates, Python wheels, CLI archives, CUDA validation, and the xlog.md docs deployment.

<Note>
For contributors — how XLOG is released internally. If you just want to install
and use XLOG, you do not need this page.
</Note>

This page explains how a release is built, validated, and published. It covers
release mechanics only, not promises about which features are available. Anything
listed in `CHANGELOG.md` under `[Unreleased]` lives on `main` and is not in any
shipped release artifact yet.

XLOG uses a split release model. GitHub Actions check source hygiene and package
metadata automatically. The heavier CUDA validation is run by hand on a real GPU
host before anything is published.

## Published Artifacts

Public releases target:

- Linux `x86_64`;
- NVIDIA CUDA Toolkit 13.x;
- crates.io packages for the public Rust crates;
- `pyxlog` Python wheels;
- GitHub release CLI archives for `xlog`.

The workspace version is currently `0.9.2`.

## Rust Package Boundary

Not every crate in the workspace is published. The two lists below split the
workspace into what goes to crates.io and what stays internal.

Publishable crates:

- `xlog-cli`
- `xlog-core`
- `xlog-cuda`
- `xlog-gpu`
- `xlog-ir`
- `xlog-logic`
- `xlog-prob`
- `xlog-runtime`
- `xlog-solve`
- `xlog-stats`

Workspace packages that are not published to crates.io:

- `pyxlog`
- `xlog-neural`
- `xlog-induce`
- `xlog-cuda-tests`
- `xlog-integration`

`pyxlog` ships as a Python package on PyPI, not as a crates.io Rust crate.

## CUDA Artifact Model

XLOG runs GPU kernels, and those compiled kernels are not checked into the tree.
The CUDA source lives under `crates/xlog-cuda/kernels`. The Rust build and
packaging scripts compile it into the binary artifacts.

At runtime, XLOG looks for a compiled kernel in four places, in this order:

1. `XLOG_CUBIN_DIR` (an environment variable pointing at a kernel directory);
2. a `kernels/` folder next to the installed package or binary;
3. Cargo's `OUT_DIR`, used for source-tree builds;
4. embedded portable PTX — GPU assembly compiled into the Rust binary as a
   last-resort fallback.

Release archives and Python wheels include pre-staged kernel artifacts. This lets
consumers run GPU code without recompiling kernels the first time.

## GPU Release Validation

Automated CI cannot prove the GPU path works, because GitHub-hosted runners have
no CUDA hardware. So before publication a maintainer runs the validation script
on a real GPU host:

```bash
scripts/validate_release_gpu.sh --mode release
```

This script requires CUDA hardware. It sets `XLOG_REQUIRE_CUDA=1`, builds the
release artifacts, runs the CUDA certification suite, and checks that kernels are
laid out correctly in the package. A green GitHub-hosted CI run does not satisfy
this gate — only a passing GPU run does.

## Docs Deployment

The public docs live at `https://xlog.md`. Here is how a change to them reaches
that site.

Source pages are MDX files under `docs/`. The GitHub workflow
`.github/workflows/docs.yml` runs whenever `docs/**` changes on a pull
request or on `main`. It pins Node 22 and `mint@4.2.666`. It validates the
Mintlify site, exports a static bundle, and force-pushes the generated site to
the `docs-dist` branch on `main`.

DigitalOcean App Platform serves the `docs-dist` branch, configured by
`.do/docs-app.yaml`. That app owns both `xlog.md` and `www.xlog.md`.

The generated static HTML is never hand-edited in the source tree. To change the
docs, edit the MDX in `docs/`, then let validation, export, and the workflow
publish `docs-dist`.

## Main-Only Feature Boundary

The features below are built and living on `main`, but they are unreleased beyond
`0.9.2` — no shipped release artifact contains them yet. A short plain-language
gloss follows each internal name:

- aggregate-fused WCOJ — worst-case-optimal join (a join strategy that avoids
  large intermediate tables) fused with aggregation;
- GPU Free Join and factorized count-by-root — Free Join is a generalized
  multiway join executed on the GPU; count-by-root counts results grouped by a
  root variable using a compact factorized representation;
- factorized recursive deltas — incremental fixpoint updates that keep results in
  that same compact factorized form;
- factorized non-count aggregate folding — folding aggregates other than counting
  into the factorized representation;
- joint multi-rule mixtures — evaluating several rules together as one mixture;
- Stage-B existential joins — joins that check whether a matching row exists
  ("Stage B" is an internal phase label);
- grouped forward/backward neural-symbolic training — training that mixes neural
  and symbolic components, processed in grouped forward and backward passes.

Because these are main-only, docs that mention them mark them with the main-only
label until a release artifact ships that includes them.

## Required Secrets

Release workflows read repository secrets to publish to crates.io and PyPI, to
automate GitHub releases, and to hold DigitalOcean deployment metadata. Store
those secrets in GitHub Actions settings. Do not commit tokens or generated
credentials to the repository.

## Release Checklist

For a release candidate:

1. Confirm `CHANGELOG.md` separates `[Unreleased]` from the target release.
2. Run ordinary CI and docs validation.
3. Run CUDA release validation on a supported GPU host.
4. Build and inspect the Python wheel and CLI archive layouts.
5. Publish Rust crates, Python wheels, and GitHub release artifacts.
6. Confirm `xlog.md` serves the exported docs over HTTPS.
