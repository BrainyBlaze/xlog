# Security Policy

## Supported Scope

Public support currently covers:

- Linux `x86_64`
- NVIDIA GPUs
- CUDA Toolkit 12.x

GitHub-hosted CI is non-GPU only. GPU validation and release validation happen outside GitHub Actions on a real CUDA machine. Reports that depend on unsupported platforms may still be reviewed, but fixes and response times are best-effort.

## Reporting a Vulnerability

Please do not open public GitHub issues for suspected security vulnerabilities.

Report vulnerabilities privately through one of these channels:

1. Use GitHub's private vulnerability reporting flow for this repository if the `Report a vulnerability` button is available.
2. If private reporting is not available in the repository UI, contact the maintainers privately through GitHub and include `SECURITY: XLOG vulnerability report` in the subject or first line.

Include, when possible:

- affected version, branch, or commit SHA
- impact summary
- reproduction steps or proof of concept
- whether the issue requires the supported Linux `x86_64` + NVIDIA CUDA environment
- GPU model, NVIDIA driver version, and CUDA version if the issue is CUDA-related

We will acknowledge receipt, triage the report, and work toward a fix before public disclosure. Please give us reasonable time to investigate and prepare a coordinated response.

## Support Expectations

- Setup questions, unsupported platforms, and generic correctness bugs should go through normal GitHub issues.
- CUDA-specific security reports should clearly state whether they were reproduced on the supported platform contract.
- We may ask reporters to retest against the latest `main` branch or a candidate fix.
