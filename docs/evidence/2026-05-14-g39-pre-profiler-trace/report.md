# Goal-039 G_PRE Profiler Trace

Date: 2026-05-17.
Branch: `feat/g39-pre-profiler-trace`.
Base: `feat/w67b-step11-close38b` at `9ce14c4e71760027fcc8f5d8e0bd290934fb4f78`.

## Scope

G_PRE measures the `session.evaluate()` share of DTS-DLM Stage 4 wall time on the
m37c-prime arm-C path. The trace uses temporary DTS-DLM instrumentation scoped to
`src/dts_dlm/propagate/xlog_executor.py` and
`src/dts_dlm/propagate/runtime.py`; that instrumentation is reverted after
evidence capture per lock 16.

The xlog pyxlog extension was built from this branch. `maturin develop` could not
run without a virtualenv, and `maturin build` compiled `libpyxlog.so` but panicked
during wheel packaging in maturin policy handling. The run therefore used a
temporary import path containing the built `target/release/libpyxlog.so` plus this
branch's generated CUDA cubins.

## Raw Artifacts

- `g39-pre-trace-50.jsonl`: run-level G_PRE trace, 3,393 JSONL events.
- `arm_C.jsonl`: m37c-prime arm-C result rows, 50 docs.
- `manifest.json`: m37c-prime run manifest.

Run id: `g39-pre-50doc-20260517-r1`.
Docs: 50.
Failures: 0.
Manifest arm-C wall time: 1,759.5450429916382 s.

Trace event counts:

```json
{
  "xlog_executor_init": 281,
  "xlog_evaluate_step": 1556,
  "stage4_propagate_once": 1556
}
```

`xlog_evaluate_step` statuses:

```json
{
  "ok": 956,
  "no_support": 600
}
```

## M_PRE.1 evaluate_pct

Metric definition: `time(session.evaluate()) / time(stage_4_total)`.

```json
{
  "stage_4_total_ns": 1575123109664,
  "session_evaluate_ns": 1521849112237,
  "evaluate_pct": 0.9661778834300995
}
```

Decision-rule outcome: **G_W63 priority HIGH**.

`evaluate_pct = 0.9661778834300995` is greater than the `0.60` HIGH threshold, so
G_W63_CHAIN sequences before G_E2E under goal-039 section 3.1.

Additional hot-loop context:

```json
{
  "xlog_evaluate_step_ns": 1524293708114,
  "xlog_evaluate_step_pct": 0.9677298864843379,
  "stage4_xlog_step_ratio_mean": 0.9808732441179449,
  "stage4_xlog_step_ratio_median": 0.9818509965476615
}
```

## M_PRE.2 rule_shape_histogram

All observed `evaluate()` invocations in this arm-C trace compile to the existing
body-len-2 mixed unary/chain shape. No triangle, cycle, K-clique, recursive, or
deep-join invocation was observed in this 50-doc arm-C run.

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

Support-clause counts observed in trace metadata:

```json
{
  "init_body_1": 281,
  "init_body_2": 2698,
  "step_body_1": 956,
  "step_body_2": 11168
}
```

## M_PRE.3 phase_breakdown

Fractions use summed `stage4_propagate_once.total_ns` as the denominator.

```json
{
  "put_relation": 0.00010936660946886062,
  "evaluate": 0.9661778834300995,
  "export_relation": 0.0,
  "enrich_support_sorts": 0.028937228075285436
}
```

Supplemental fractions:

```json
{
  "result_materialize": 0.0007517659818048085,
  "confidence": 0.0017991951598021628,
  "justification": 0.00111124451813381
}
```

Raw nanoseconds:

```json
{
  "put_relation_ns": 172265874,
  "evaluate_ns": 1521849112237,
  "export_relation_ns": 0,
  "enrich_support_sorts_ns": 45579696671,
  "result_materialize_ns": 1184123971,
  "confidence_ns": 2833953875,
  "justification_ns": 1750346921
}
```

## Validation

- `python3 -m py_compile src/dts_dlm/propagate/xlog_executor.py src/dts_dlm/propagate/runtime.py`: PASS before trace.
- `ruff check src/dts_dlm/propagate/xlog_executor.py src/dts_dlm/propagate/runtime.py`: PASS before trace.
- `python3 -m pytest -q src/tests/propagate/test_m29_evaluate_capture.py src/tests/propagate/test_m32_compile_source_capture.py`: PASS, 8 passed.
- `m37c_prime_pilot --arms C` on 50 docs: PASS, 50 docs, 0 failures.

Temporary DTS-DLM instrumentation was reverted after trace capture. xlog evidence
artifacts above are the only intended committed files for G_PRE.
