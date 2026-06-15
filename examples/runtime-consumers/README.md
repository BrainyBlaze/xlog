# Runtime Consumer Certification

These examples certify the composed runtime and optimizer surface for
external consumers. They are intentionally ordinary `.xlog` programs:
the validator runs them through `xlog-cli` and separately records evidence from
the production runtime/provider gates rather than using private hooks.

The suite covers:

- an external consumer-shaped delta and optimizer fixture;
- two neutral scientific/engineering workloads derived from external-consumer
  flow and signal reasoning, without project-specific terms in the programs;
- a runtime-substrate fixture that documents the exact/index/CSE/adaptive
  primitives required by later epistemic and solver work;
- public compatibility by invoking the existing validators.

Run the runtime-consumer certification validator from the repository's scripts
directory.
