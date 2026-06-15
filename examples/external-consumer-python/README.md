# External Consumer Example Suite

Focused post-close hardening examples for the external consumer ML/Python
productization surface. Each directory contains a small `.xlog` program and a
self-checking `run.py` that emits one JSON result.

Run the full suite with the external-consumer compatibility validator and write
its summary to the JSON path chosen for the current validation run.

The examples require CUDA and an installed or editable in-tree `pyxlog` build.

| Example | Purpose | Surfaces |
|---|---|---|
| `01_async_streaming_reachability` | Recursive external consumer-style graph query | `LogicProgram`, sessions, `evaluate_async`, `evaluate_stream`, chunked result tensors, memory/progress/graph/host-transfer stats |
| `02_relation_deltas` | Relation maintenance analog | `put_relation`, `insert_relation`, `delete_relation`, `apply_relation_delta`, `delta_stats`, full-replacement equivalence |
| `03_neural_bridge_topk_belnap` | Neural bridge fixture | `.xlog` neural predicate, `register_network(k=..., det=True)`, `forward_backward_tensor`, deterministic top-k, cache stats, Belnap helper |
| `04_native_exact_induction` | external consumer-shaped exact ILP topology selection | learnable `.xlog`, `IlpProgramFactory`, `induce_exact(..., backend="native")`, Python/native parity |
| `05_probabilistic_async_diagnostics` | Probabilistic API diagnostics | probabilistic `.xlog`, `Program.evaluate_async`, exact and MC mode, memory/progress/graph/host-transfer stats |

This suite is a post-close hardening addendum. It does not rewrite the original
closure gate and does not authorize push, merge, tag, or release-board changes.
