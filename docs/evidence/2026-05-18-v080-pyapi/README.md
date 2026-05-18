# v0.8.0 Python Runtime And Session API Evidence

**Date:** 2026-05-18
**Branch:** `feat/v080-dts-ml-python-productization`
**Scope:** G080_PYAPI async evaluation, streaming result chunks, per-call memory limits, progress counters, and diagnostics.

---

## Artifacts

| Artifact | Purpose |
|----------|---------|
| `runtime_probe.json` | Isolated branch-local pyxlog 0.7.0 probe for async, streaming, progress, memory-limit, and diagnostics behavior. |
| `scripts/v080_pyxlog_runtime_probe.py` | Committed reproducer for the PYAPI runtime probe. |
| `python/tests/test_v080_pyapi_source.py` | Source-surface regression test for new runtime/session API names and native memory hooks. |
| `crates/pyxlog/python/pyxlog/__init__.py` | Python async wrapper, progress counters, and chunk streaming helpers. |
| `crates/pyxlog/src/logic.rs`, `crates/pyxlog/src/program.rs`, `crates/pyxlog/src/lib.rs` | Native per-call memory checks and memory diagnostics. |
| `docs/architecture/python-bindings.md` | Public API documentation for v0.8.0 runtime controls. |

---

## Validation Commands

| Command | Result |
|---------|--------|
| `pytest -q python/tests/test_v080_pyapi_source.py` | exit 0; 4 passed |
| `/tmp/xlog-v080-cert-venv/bin/python scripts/v080_pyxlog_runtime_probe.py --probe pyapi --output docs/evidence/2026-05-18-v080-pyapi/runtime_probe.json` | exit 0; regenerated `runtime_probe.json` |
| `cargo fmt --check` | exit 0 |
| `cargo check -p pyxlog` | exit 0 |
| `cargo test -p pyxlog --lib` | exit 0; 7 passed |
| `/tmp/xlog-v080-cert-venv/bin/python` branch-local runtime probe | exit 0; pyxlog `0.7.0` |

The runtime probe uses the isolated virtualenv at `/tmp/xlog-v080-cert-venv`
after reinstalling this branch's locally built `pyxlog-0.7.0` wheel. Some
allocation byte counts are local CUDA-environment observations; the release
gate is the behavior they evidence: async completion, CUDA chunk tensors,
typed over-limit failure, progress counters, and diagnostics keys.

---

## Metric Status

| Metric | Target | Status | Evidence |
|--------|--------|--------|----------|
| M080_PYAPI.1 async API tests | pass for logic session and program evaluation | PASS | `runtime_probe.json`: `async_done_after_result=true`, `program_progress_after_async.evaluations_completed=1` |
| M080_PYAPI.2 streaming tests | chunked output equals non-streaming output as a row set | PASS | `runtime_probe.json`: `chunk_rows=[3,3,2]`, `stream_chunk_rows=[3,3,2]`, `logic_rows=8` |
| M080_PYAPI.3 memory override tests | per-call limit accepted and enforced; over-limit failure is typed | PASS | `runtime_probe.json`: 512 MiB accepted; 1 MiB over-limit raised `MemoryError` with `allocated_bytes=3200004` |
| M080_PYAPI.4 progress counters | deterministic counters exposed for recursive and neural-symbolic long calls | PASS | async wrappers expose `progress_stats()`; probe records started/completed counters for logic session and probabilistic program |
| M080_PYAPI.5 docs | `docs/architecture/python-bindings.md` updated | PASS | new `v0.8.0 Runtime Controls And Diagnostics` section |
| M080_PYAPI.6 compatibility | old `evaluate(...)` and `session.evaluate()` calls continue passing | PASS | pyxlog unit tests passed; wrappers preserve the existing call path and add optional `memory_mb` |

---

## API Notes

- `evaluate_async(...)` returns an awaitable `AsyncEvaluation` backed by a small Python `ThreadPoolExecutor`.
- `LogicQueryResult.iter_chunks(...)` and `LogicEvalResult.iter_query_chunks(...)` yield `LogicQueryChunk` objects with CUDA tensor views. Those tensors are DLPack-compatible and avoid host copies.
- `memory_stats()` reports `allocated_bytes`, `memory_limit_bytes`, `peak_memory_bytes`, and `status`.
- Per-call `memory_mb` is a pre-evaluation guard over the provider's tracked allocation. The compile-time provider budget remains the hard allocator budget.
- CUDA Graph and host-transfer diagnostics are surfaced on both logic sessions and probabilistic programs.

No push, tag, release-board update, merge, or final v0.8.0 closure claim is authorized by this evidence.
