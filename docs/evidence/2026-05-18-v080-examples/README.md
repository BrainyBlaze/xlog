# v0.8.0 DTS Example Suite Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_EXAMPLES post-close hardening amendment.

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `examples/v080-dts/` | Five focused v0.8.0 DTS showcase examples. |
| `scripts/validate_v080_examples.py` | Runs all five examples and writes a single validation summary JSON. |
| `validation_summary.json` | Aggregated example evidence with per-example PASS statuses and required metrics. |
| `python/tests/test_v080_examples_source.py` | Source/file-layout guard for the suite and validator contract. |

## Validation Command

```bash
VIRTUAL_ENV=/tmp/xlog-v080-cert-venv \
PATH=/tmp/xlog-v080-cert-venv/bin:$PATH \
/tmp/xlog-v080-cert-venv/bin/python scripts/validate_v080_examples.py \
  --output docs/evidence/2026-05-18-v080-examples/validation_summary.json
```

Expected summary gates:

- `example_count = 5`
- every `per_example[*].status = PASS`
- CUDA tensor checks reported where applicable
- host-transfer and graph diagnostics reported where exposed
- relation delta equivalence and row counts reported
- deterministic top-k selected labels reported
- exact-induction Python/native parity reported
- async completion and streaming chunk counts reported

Fresh result on this branch: exit 0; `example_count=5`; all five
`per_example` statuses are `PASS`; selected top-k labels are
`["reject", "accept"]`; exact-induction parity reports `summary=true` and
`ordered_candidates=true`.

No push, tag, release-board update, or merge is authorized by this evidence.
