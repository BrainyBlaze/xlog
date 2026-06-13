# v0.8.6 Runtime Consumer Certification

These examples certify the composed v0.8.6 runtime and optimizer surface for
the named release consumers. They are intentionally ordinary `.xlog` programs:
the validator runs them through `xlog-cli` and separately records evidence from
the production runtime/provider gates rather than using private hooks.

The suite covers:

- an external consumer-shaped delta and optimizer fixture;
- two neutral scientific/engineering workloads derived from Mistaber-style
  flow and signal reasoning, without project-specific terms in the programs;
- a v0.9.0 substrate fixture that documents the exact/index/CSE/adaptive
  primitives required by later epistemic and solver work;
- v0.8.0 and v0.8.5 compatibility by invoking the existing validators.

Run:

```bash
python scripts/validate_v086_examples.py
```
