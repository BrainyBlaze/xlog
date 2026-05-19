# v0.8.0 DTS-DLM Certification Pack Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_CERT certification pack and pyxlog public-surface manifest.

---

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `pyxlog_api_manifest.json` | Machine-readable manifest for DTS-required pyxlog symbols, signature compatibility, hot-path transfer target, deterministic replay target, and CUDA Graph telemetry availability. |
| `runtime_probe.json` | Branch-local pyxlog 0.8.0 runtime probe: 100 fixed-session replays, max tracked host transfers after reset+evaluate, and CUDA Graph counters. |
| `scripts/v080_dts_cert.py` | Manifest generator and verifier used by this evidence. |
| `python/tests/test_v080_dts_cert.py` | Regression tests for manifest coverage and verifier behavior. |
| `crates/pyxlog/python/pyxlog/_native.pyi` | Updated to expose existing `LogicRelationSession` diagnostic methods in the Python type stub. |

---

## Validation Commands

| Command | Result |
|---------|--------|
| `pytest -q python/tests/test_v080_dts_cert.py` | exit 0; 3 passed |
| `python scripts/v080_dts_cert.py manifest --repo-root . --output docs/evidence/2026-05-18-v080-cert/pyxlog_api_manifest.json` | exit 0 |
| `python scripts/v080_dts_cert.py verify docs/evidence/2026-05-18-v080-cert/pyxlog_api_manifest.json` | exit 0; `PASS symbol_coverage=17/17 signature_drift=0` |
| `/tmp/xlog-v080-cert-venv/bin/python` branch-local runtime probe | exit 0; pyxlog `0.8.0`, 100/100 bit-exact replays, max host-transfer stats all zero |
| `cargo test -p xlog-integration --test test_m37a_surface_preservation` | exit 0; 4 passed |

The user-level Python environment still imports `pyxlog 0.6.2`. The runtime
probe used an isolated virtualenv at `/tmp/xlog-v080-cert-venv` with this
branch's locally built `pyxlog-0.8.0-cp310-cp310-linux_x86_64.whl`.

---

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_CERT.1 API manifest | machine-readable manifest under `docs/evidence/...` | PASS | `pyxlog_api_manifest.json` |
| M080_CERT.2 required symbol coverage | 100 percent DTS-required pyxlog symbols present | PASS | verifier reports `17/17` |
| M080_CERT.3 signature drift | 0 unapproved breaking changes | PASS | verifier reports `signature_drift=0` |
| M080_CERT.4 host-transfer delta | `dtoh_bytes=0`, `dtoh_calls=0`, `htod_bytes=0`, `htod_calls=0` on certified hot path | PASS | `runtime_probe.json` records all four max stats as zero after `reset_host_transfer_stats()` + `evaluate()` |
| M080_CERT.5 determinism | 100/100 fixed-fixture replays bit-exact | PASS | `runtime_probe.json` records `bit_exact_replays=100` out of `replays=100` |
| M080_CERT.6 graph telemetry | graph counters available or explicit unavailable reason recorded | PASS | manifest records `status=available` and the four `csm_cuda_graph_*` counters from `crates/pyxlog/src/logic.rs` |

---

## Stub Compatibility Note

`LogicRelationSession` already exposed `host_transfer_stats`,
`reset_host_transfer_stats`, and `cuda_graph_stats` in Rust. The certification
test intentionally failed until `_native.pyi` documented those methods. This is
a stub compatibility fix, not a runtime behavior change.

---

## Next Sub-Goal

Proceed to G080_PYAPI: Python runtime/session API productization. The source
manifest is green, but no push, tag, release-board update, merge, or final
v0.8.0 closure claim is authorized by this evidence.
