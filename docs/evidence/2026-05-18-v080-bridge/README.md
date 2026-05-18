# v0.8.0 Neural-Symbolic Bridge Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_BRIDGE M37-A+B neural-symbolic bridge support.

---

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `runtime_probe.json` | Branch-local pyxlog probe for a LearnedBridge-shaped module, embedding lookup, deterministic top-k, Belnap/semantic losses, and cache telemetry. |
| `python/tests/test_v080_bridge_source.py` | Source-surface regression for bridge helpers and evidence presence. |
| `crates/pyxlog/src/neural.rs` | Bridge helper implementation and circuit-cache hit/miss telemetry. |
| `crates/pyxlog/python/pyxlog/_native.pyi` | Public type-stub surface for bridge helpers. |
| `docs/architecture/python-bindings.md` | Public documentation for v0.8.0 DTS-DLM bridge helpers. |

---

## Validation Commands

| Command | Result |
|---------|--------|
| `pytest -q python/tests/test_v080_bridge_source.py` | exit 0; 3 passed |
| `cargo check -p pyxlog` | exit 0 |
| `/tmp/xlog-v080-cert-venv/bin/python` branch-local bridge probe | exit 0; pyxlog `0.7.0` |

The runtime probe used the isolated virtualenv at `/tmp/xlog-v080-cert-venv`
after reinstalling this branch's locally built `pyxlog-0.7.0` wheel.

---

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_BRIDGE.1 gradient smoke | finite CUDA loss, nonzero gradient, parameter update observed | PASS | `runtime_probe.json`: `initial_loss=0.8466`, `grad_norm=1.5468`, `parameter_update_norm=0.3094` |
| M080_BRIDGE.2 DTS-shaped module | LearnedBridge-shaped fixture works with pyxlog network registration | PASS | `runtime_probe.json`: `learned_bridge_module=LearnedBridge(linear: 4 -> 3, softmax)` |
| M080_BRIDGE.3 Belnap helper tests | pro/contra/quarantine helper outputs match documented formulas | PASS | `runtime_probe.json`: `belnap_loss=0.0`, `belnap_expected=0.0`, formula recorded |
| M080_BRIDGE.4 deterministic top-k | fixed seed and tie inputs produce stable ordered results | PASS | `runtime_probe.json`: `deterministic_topk_indices=[2,0,1]` for tied 0.5 scores |
| M080_BRIDGE.5 neural cache telemetry | hit/miss counters available | PASS | `runtime_probe.json`: `cache_stats.circuit_cache_hits=3`, `circuit_cache_misses=1`, `template_compile_count=1` |
| M080_BRIDGE.6 repeated-query speedup | at least 50x on cache-hit microbench or RCA + amended target | PASS | `runtime_probe.json`: `cache_speedup=1536.0x` comparing first compile path to cached query average |

---

## API Notes

- `belnap_loss(...)` returns tensor terms for `loss`, `pro_reward`, `contra_penalty`, `quarantine_penalty`, and `cfr_regret_proxy`.
- `semantic_loss_tensor(...)`, `mse_loss_tensor(...)`, and `infoloss_tensor(...)` operate on PyTorch tensors and preserve autograd.
- `deterministic_topk(...)` uses stable descending order and lower-index tie breaking.
- `neural_cache_stats()` reports circuit-cache hit/miss counters and registered-network cache/top-k/deterministic configuration.
- Belnap pro/contra/quarantine semantics remain in the Python/ML layer; Stage-4 structural kernels were not changed for those semantics.

No push, tag, release-board update, merge, or final v0.8.0 closure claim is authorized by this evidence.
