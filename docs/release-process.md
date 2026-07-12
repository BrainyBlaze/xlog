# Release Process

XLOG uses a split release model:

- GitHub-hosted Actions validate non-GPU concerns only.
- Real CUDA validation happens manually on a supported Linux x86_64 machine before publication.

This keeps public CI honest. A green PR does not prove GPU correctness.

## Supported Release Surface

Public releases currently target:

- Linux `x86_64`
- NVIDIA CUDA Toolkit 13.x
- GitHub release CLI archives
- `pyxlog` wheels on PyPI
- Rust crates on crates.io for the stable public Rust surface

Everything else remains unsupported under the current public release contract.

## Consumer Build And Kernel Artifact Model

Generated CUDA artifacts are not source files. Keep `.ptx` and `.cubin` outputs out of git; they are
produced from `kernels/*.cu` by `crates/xlog-cuda/build.rs` and staged only for packaged release
artifacts.

Runtime kernel lookup is intentionally layered:

1. `XLOG_CUBIN_DIR`
2. package- or binary-adjacent `kernels/`
3. Cargo build `OUT_DIR` for source-tree builds
4. embedded portable PTX compiled into the Rust binary

The supported consumer paths are:

- `cargo install xlog-cli --features host-io`: requires Rust, Cargo, CUDA Toolkit 13.x, and `nvcc`
  at install time. The installed binary embeds portable PTX, so it does not depend on Cargo
  retaining build-output sidecars after installation.
- GitHub release archive: requires no Rust build. The archive contains the `xlog` binary plus a
  staged `kernels/` directory with release-built CUDA artifacts.
- PyPI wheel: `pip install pyxlog` installs the native extension plus packaged
  `pyxlog/kernels/`; importing `pyxlog` sets `XLOG_CUBIN_DIR` to that packaged directory when it is
  present.
- Local Python development: run
  `python scripts/install_pyxlog_for_python.py --python /path/to/project/python` so the local wheel
  is built for and installed into the exact interpreter used by the downstream project.

Native Linux GPU runners normally expose `/dev/nvidia*`. WSL2 CUDA runners normally expose
`/dev/dxg` instead; absence of `/dev/nvidia*` is expected there. `scripts/xlog_doctor.py` accepts
either device model for non-release workflow checks.

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
`xlog-cli` only, using package-qualified tags such as `xlog-cli-v0.5.1`. This avoids collisions
with legacy repository-wide tags that are not tied to a published crates.io release. The library
crates are published to crates.io as part of the same release wave, but they do not each create
their own GitHub release.

## Required Repository Configuration

Keep these repository settings and secrets configured:

- `CARGO_REGISTRY_TOKEN` GitHub Actions secret for crates.io publishing
- `PYPI_API_TOKEN` GitHub Actions secret for uploading `pyxlog` distributions to PyPI
- `RELEASE_PLZ_GITHUB_TOKEN` GitHub Actions secret for release-plz PR creation and release
  tagging. This is required on `BrainyBlaze/xlog`: the organization currently blocks GitHub
  Actions from creating pull requests with `github.token`, so release-plz must use a dedicated
  credential instead.
- Recommended credential for `RELEASE_PLZ_GITHUB_TOKEN`: a fine-grained PAT with:
  - resource owner `BrainyBlaze`
  - repository access `Only select repositories` -> `xlog`
  - repository permission `Contents: Read and write`
  - repository permission `Pull requests: Read and write`
  - repository permission `Administration: Read and write` only if protected tags are enabled
- If the organization ever allows it, the repository setting `Allow GitHub Actions to create and
  approve pull requests` would let `github.token` create PRs. `BrainyBlaze/xlog` currently cannot
  rely on that path.

Current GitHub-side status for `BrainyBlaze/xlog`:

- repository workflow default permissions are `read`
- workflow files request the write scopes they need explicitly
- enabling “GitHub Actions can create or approve pull requests” at the repo level currently returns
  a `409 Conflict` because the organization disallows it
- therefore `RELEASE_PLZ_GITHUB_TOKEN` is the supported release-plz credential for this repository

GitHub fine-grained PAT creation is currently a browser flow, not a CLI or REST automation flow.
Use this prefilled GitHub URL to create the replacement token, then update the repository secret:

```text
https://github.com/settings/personal-access-tokens/new?name=xlog%20release-plz&description=Release-plz%20automation%20for%20BrainyBlaze%2Fxlog&target_name=BrainyBlaze&expires_in=90&contents=write&pull_requests=write
```

After opening that page, set `Repository access` to `Only select repositories`, choose `xlog`, and
generate the token. Then replace the existing repository secret:

```bash
gh secret set RELEASE_PLZ_GITHUB_TOKEN -R BrainyBlaze/xlog
```

## Automation Layout

`.github/workflows/release-plz.yml`:

- on pushes to `main`, runs `release-plz` in `release-pr` mode to open or update the release PR
- on manual `workflow_dispatch`, runs `release-plz` in `release` mode after the maintainer confirms
  that real-GPU validation has already passed
- a workspace-level worthiness check inside the workflow (before `release-plz release-pr`)
  restricts release PR creation to pushes that carry at least one commit since the last
  `xlog-cli-v*` tag whose subject starts with `feat:`, `fix:`, `perf:`, `refactor:`, `docs:`,
  `build:`, `ci:`, `test:`, or `revert:` — including scoped forms such as `build(deps):` and
  breaking-change forms such as `feat!:`
- plain `chore:` commits are treated as non-release maintenance, so merging a release PR with a
  title such as `chore: release v0.5.2` does not immediately queue the next release PR
- before the manual publish step, runs `scripts/preflight_release_publish.sh`, which currently
  validates the publishable crate package layouts without requiring crates.io to already know about
  the new interdependent workspace versions
- on normal `push` runs, it is expected for the publish-only gates inside the job to be skipped:
  `Require explicit GPU gate acknowledgement`, `Require publish secrets`, and
  `Preflight crate publish verification` are guarded by `github.event_name == 'workflow_dispatch'`
  and therefore do not run during `release-pr` mode

`.github/workflows/python-publish.yml`:

- builds Linux `x86_64` `pyxlog` wheels in the CUDA container for CPython 3.8 through 3.12
- builds a source distribution
- uploads the distributions to the matching GitHub release
- publishes the distributions to PyPI using `PYPI_API_TOKEN`
- can also be rerun manually with `workflow_dispatch` by supplying an existing `xlog-cli` tag
- stages `crates/pyxlog/python/pyxlog/kernels/` with `bash scripts/stage_pyxlog_kernels.sh`
  before invoking `maturin build`; do not remove that step unless wheel inspection still proves
  `pyxlog/kernels/` is present in the built artifact

`.github/workflows/github-release.yml`:

- builds the host-io-enabled CLI release archive with bundled kernels
- writes `SHA256SUMS`
- uploads the archive and checksums to the matching GitHub release
- can also be rerun manually with `workflow_dispatch` by supplying an existing `xlog-cli` tag

## Expected Skipped Jobs

On a normal `push` to `main`, `release-plz.yml` runs in `release-pr` mode. That mode opens or
updates the release PR, but it does not publish crates or create a new `xlog-cli` GitHub release
tag in that same run.

Because of that, these downstream jobs are expected to show as `skipped` on many successful
`main` push runs:

- `python-publish`
- `github-release-assets`

They are gated by:

```yaml
if: needs.release-plz.outputs.cli_release_created == 'true'
```

So a skipped result there means `release-plz` did not create a new `xlog-cli` release in that run.
It does not mean publication failed. Those jobs run only when a publish actually happened and
`cli_release_created=true`, which normally occurs in the manual `workflow_dispatch` publish flow or
when the same workflow invocation actually emits the new CLI release.

## Release Commit Policy

Release-plz PR preparation is intentionally narrower than "every commit that touched packaged
files". A workspace-level worthiness check in the release-plz workflow (run before
`release-plz release-pr`; it replaced the per-package `release_commits` config rule, whose
per-crate filtering produced partial release sets that broke intra-workspace version
requirements under the shared workspace version) treats the following commit types as
release-worthy:

- `feat`
- `fix`
- `perf`
- `refactor`
- `docs`
- `build`
- `ci`
- `test`
- `revert`

Everything else is treated as non-release by default, especially generic `chore:` commits. This is
intentional:

- merged release PR commits use `chore: release ...`
- the README sync bot commit now uses `chore(release): ...`
- those release-maintenance commits must not recursively open the next release PR on `main`

If a maintenance change should ship in the next release, use a more specific Conventional Commit
type such as `build:`, `ci:`, `fix:`, `refactor:`, or `perf:` instead of bare `chore:`.

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

## Recovery When Publish Is Skipped

`release-plz release` publishes only when `HEAD` is associated with a merged PR whose branch name
starts with `release-plz-`. A manual dispatch from an arbitrary `main` commit will not publish and
should now fail loudly instead of succeeding with `releases=[]`.

If you merge a post-release fix directly onto `main` after the release PR, recover with this flow:

1. Create a follow-up branch from current `main` whose name starts with `release-plz-`, for example
   `release-plz-fix-publish`.
2. Put the required CI/package/docs fix on that branch.
3. Open a PR back to `main` and merge it with the standard merge strategy, not squash.
4. Re-run the manual GPU validation on the merged commit if the fix affects release artifacts.
5. Dispatch `release-plz.yml` from that merge commit with the validation attestation fields.

This mirrors the recovery path recommended by release-plz itself for CI fixes that must still lead
to a publishable release commit.

## Recovery When Artifact Upload Fails

If crates have already been published and the follow-on artifact workflows fail, rerun the
artifact workflows directly against the existing CLI release tag instead of re-dispatching
`release-plz.yml`.

1. Confirm the published CLI tag, for example `xlog-cli-v0.5.0`.
2. Re-run Python publishing with `python-publish.yml` and `tag_name=<that tag>`.
3. Re-run GitHub asset upload with `github-release.yml` and `tag_name=<that tag>`.

This avoids a dead end where `release-plz release` has nothing new to publish and therefore cannot
re-trigger artifact jobs that are keyed off `cli_release_created=true`.

## Local Commands

Non-GPU release sanity:

```bash
make lint-workflows
make lint-shell
bash scripts/preflight_release_publish.sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked --no-deps -- \
  -A clippy::approx_constant \
  -D clippy::dbg_macro \
  -D clippy::todo \
  -D clippy::unimplemented
cargo build --workspace --locked --release --exclude pyxlog
cargo build --locked --release -p xlog-cli --features host-io
bash scripts/stage_pyxlog_kernels.sh
```

Manual GPU gate:

```bash
bash scripts/validate_release_gpu.sh --mode release
```

Bundle a local CLI release archive:

```bash
make package PACKAGE_OUTPUT=/tmp/xlog-dist
```
