# Support

## What To Use

- GitHub Issues: reproducible bugs, packaging problems, documentation mismatches, and supported-platform setup failures
- GitHub Discussions: usage questions, integration ideas, and broader design conversation
- Private vulnerability reports: use the process in [SECURITY.md](SECURITY.md)

## Supported Platform Contract

Public support currently covers:

- Linux `x86_64`
- NVIDIA GPU
- CUDA Toolkit 13.x

GitHub-hosted CI is non-GPU only. If your question depends on CUDA execution, include the GPU model,
driver version, CUDA toolkit/runtime version, and whether `nvidia-smi` / `nvcc --version` work on
your machine.

## Before Opening A Ticket

Please include:

- XLOG version, release tag, or commit SHA
- Exact commands you ran
- Minimal reproduction steps
- Full error text or stack trace
- Whether the issue reproduces on the supported platform contract

For local setup questions, run:

```bash
python scripts/xlog_doctor.py
```

and include the output that matters.
