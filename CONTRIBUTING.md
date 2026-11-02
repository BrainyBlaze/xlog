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

GitHub-hosted pull-request CI checks source hygiene and compiles only the five
CUDA-independent production libraries described in
[the CI policy](ENGINEERING.md#source-ci-and-manual-validation). It neither builds
GPU artifacts nor runs tests. GPU builds are manual and LOCAL only; heavy
integration/end-to-end, benchmark, and certification runs are separately selected
manual procedures. A source change may be reviewed and integrated before runtime
qualification, but it must not be described as GPU-validated until the real run
has completed. Fork code requires an isolated, authorized execution host.

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

### Containerized Python wheel delivery

Build the reusable Linux `x86_64` toolchain image once, then retain its image digest
for subsequent manual LOCAL builds. The image pins CUDA 13.3, Rust 1.99.1, CPython
3.10.22, uv 0.13.1, and maturin 1.15.0. Distribution packages are resolved during
image creation; retain the built image and its package records instead of assuming
that a later image rebuild has identical bytes.

```bash
docker build --platform linux/amd64 \
  -f docker/pyxlog-toolchain.Dockerfile \
  -t xlog-pyxlog-toolchain:cp310 docker
mkdir -p target/container-cache target/container-wheels
docker run --name xlog-pyxlog-build --init \
  --memory 32000000000 --memory-swap 32000000000 --cpus 4 \
  --mount type=bind,src="$PWD",dst=/src,readonly \
  --mount type=bind,src="$PWD/target/container-cache",dst=/cache \
  --mount type=bind,src="$PWD/target/container-wheels",dst=/out \
  --env SOURCE_DATE_EPOCH="$(git log -1 --format=%ct)" \
  xlog-pyxlog-toolchain:cp310
```

Use a clean checkout at the intended source revision and an owned output directory;
the canonical builder replaces existing `pyxlog` wheels in that directory. Cargo
dependencies and compiled objects persist in `/cache`. The container uses the
canonical Python builder with `--build-only`, so it neither installs nor imports
the wheel. It limits compilation to two Cargo jobs and two hours. Keep the named
container until its terminal state, logs, process-group resource records, and actual
toolchain/package identities have been collected. Resource authorization remains
separate from these example bounds.

For `semantic-policy`, append `--features semantic-policy`, mount the original policy
header read-only, and pass `XLOG_SEMANTIC_POLICY_HEADER`,
`XLOG_SEMANTIC_POLICY_SHA256`, and `XLOG_SEMANTIC_POLICY_NAMESPACE` for that header.
Retain its source identity and the actual compiled-header digest with the build
records. Select `XLOG_CUBIN_ARCHS` for the intended GPU; portable PTX remains part of
the canonical CUDA build. Do not rewrite kernel targets after compilation.

Package an admitted maintainer release directory, containing the wheel and its
original build records, as a small artifact image without repeating compilation:

```bash
docker build --platform linux/amd64 \
  -f docker/pyxlog-artifact.Dockerfile \
  -t xlog-pyxlog-artifact:admitted "$XLOG_RELEASE_DIR"
docker image save --output "$XLOG_ARTIFACT_ARCHIVE" xlog-pyxlog-artifact:admitted
```

Transfer that archive and use `docker image load`, or publish the image through the
maintainer's authorized registry. A downstream Dockerfile can copy `/xlog-release/`
from the transferred image into its existing artifact input directory with
`COPY --from`. Keep the exact wheel, native-library, kernel, policy, source, and
toolchain admission checks at the consumer. An image tag or a successful image build
does not replace these identities or establish hardware/runtime qualification.
GPU artifacts are never built or published by source CI.

If you want local parity with the workflow/shell lint jobs, install:

- `shellcheck` via your package manager, for example `sudo apt-get install -y shellcheck`
- `actionlint` from the release binary that CI uses in `.github/workflows/ci.yml`

## Local Checks

Run bounded source checks before opening a pull request:

```bash
cargo fmt --all --check
make lint-workflows
make lint-shell
make check-tracked-ignored
make check
```

`make test` is the explicit manual workspace suite, not a source check. Runtime
and resource authority is required independently for it and for certifications.

Example validators require an already installed `pyxlog` wheel in their selected
Python interpreter. Use the canonical `scripts/install_pyxlog_for_python.py`
procedure for an authorized manual LOCAL build/install; validators do not build
and stage competing debug copies or modify driver-library search paths.

Select the following only for an explicitly authorized manual example/runtime
validation. Despite its historical mode name, this command builds CLI artifacts
and executes examples; it is not a fast-CI command:

```bash
python scripts/validate_examples.py --mode ci
```

For an authorized manual validation of CUDA kernels or low-level GPU execution:

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

For package-level runtime/release acceptance of `pyxlog`, persistent relations,
DLPack ownership, stream ordering, or packaging, select the complete manual
release validator instead:

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
up-to-date `production-build` source check in `.github/workflows/ci.yml`; force
pushes to `main` and deletion of `main` are prohibited. Hardware/runtime acceptance
is recorded separately and is not inferred from that check or from source merge.

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
