# Contributing to XLOG

Thanks for contributing. Keep changes small, testable, and aligned with the current public support contract.

## Supported Platform

The first public support contract is:

- Linux `x86_64`
- NVIDIA GPU
- CUDA Toolkit 12.x

GitHub-hosted CI is non-GPU only. GPU validation happens outside GitHub Actions on a real CUDA machine. If you change CUDA kernels, GPU execution paths, packaging, or installation flows, validate on supported hardware before asking for review.

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

Required on a supported CUDA machine when your change affects GPU behavior, CUDA kernels, or release validation:

```bash
cargo test -p xlog-cuda-tests --test certification_suite --release
```

## Pull Request Expectations

Each pull request should:

- describe the user-visible change and the risk area
- state whether the work was validated on the supported Linux `x86_64` + NVIDIA CUDA platform
- list the exact commands you ran locally
- include reproduction steps for bug fixes
- update docs when behavior, setup, or support expectations changed
- stay focused; separate unrelated refactors into different pull requests

If a change was only validated in GitHub-hosted CI, say so explicitly. That is useful signal, but it does not replace real GPU validation for CUDA-facing changes.

## Review Notes

- Prefer targeted changes over broad cleanup.
- Do not merge changes that weaken the supported-platform story without updating the docs and templates in the same pull request.
- If you are unsure whether something needs GPU validation, assume it does and call that out in the PR.
