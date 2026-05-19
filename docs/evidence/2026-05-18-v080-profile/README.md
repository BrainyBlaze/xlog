# v0.8.0 Profile-Gated Optimizer Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_PROFILE profile-gated optimizer/index work.

## Decision

No optimizer or index implementation is authorized in this sub-goal.

The available DTS-DLM profile evidence names `session.evaluate()` as hot and
shows only the body-len-2 mixed unary/chain shape. That profile-backed issue
was already handled by the integrated W63 chain dispatcher. The current v0.8.0
DTS evidence does not name duplicated subplans, adaptive re-optimization, or
index rebuild cost as a release blocker.

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `profile_decision.json` | Machine-readable G080_PROFILE decision and raw profile numbers. |
| `docs/evidence/2026-05-14-g39-pre-profiler-trace/report.md` | DTS-DLM 50-doc m37c-prime arm-C profiler trace. |
| `docs/evidence/2026-05-14-g39-w63-chain-prod/report.md` | Profile-backed W63 chain dispatcher production evidence. |
| `docs/evidence/2026-05-18-v080-delta/runtime_probe.json` | Session-delta evidence relevant to the index-rebuild question. |

## Profile Inventory

G_PRE profiler trace:

- Run id: `g39-pre-50doc-20260517-r1`
- Docs: `50`
- Failures: `0`
- Arm-C wall time: `1759.5450429916382s`
- `stage_4_total_ns=1575123109664`
- `session_evaluate_ns=1521849112237`
- `evaluate_pct=0.9661778834300995`

Rule-shape histogram:

```json
{
  "chain_2_mixed_unary": 1556,
  "triangle_3": 0,
  "cycle_4": 0,
  "clique_k": 0,
  "recursive": 0,
  "mixed_deep_join": 0
}
```

Phase breakdown:

```json
{
  "put_relation": 0.00010936660946886062,
  "evaluate": 0.9661778834300995,
  "export_relation": 0.0,
  "enrich_support_sorts": 0.028937228075285436
}
```

W63 profile-backed response:

- Trace subset: `128` chain-shaped G_PRE invocations
- Recorded baseline: `evaluate_ns=81019.497ms`
- ChainJoin replay: `86.819ms`
- Ratio: `933.200479x`
- Dispatches: `128`
- Output rows: `12998`

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_PROFILE.1 profile evidence | required before any optimizer/index implementation | PASS | G_PRE and W63 production evidence are inventoried in `profile_decision.json`. |
| M080_PROFILE.2 improvement gate | implemented change improves profiled bottleneck by at least 1.2x or removes a correctness blocker | N/A | No new optimizer/index implementation is authorized because no current profile names a new bottleneck. The prior profile-backed W63 response exceeded the 1.2x gate with `933.200479x` on the trace subset. |
| M080_PROFILE.3 non-regression | no DTS cert or WCOJ regression | PASS | This sub-goal changes only evidence docs. No runtime, optimizer, or index source changed. |

## Validation Commands

| Command | Result |
|---------|--------|
| `/tmp/xlog-v080-cert-venv/bin/python -m json.tool docs/evidence/2026-05-18-v080-profile/profile_decision.json` | exit 0 |
| `git status --short --branch` before commit | only `docs/evidence/2026-05-18-v080-profile/` untracked for this sub-goal |
| `git diff --check` | exit 0 |

No push, tag, release-board update, merge, or final v0.8.0 closure claim is
authorized by this evidence.
