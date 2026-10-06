# pyxlog

`pyxlog` provides Python bindings for the XLOG GPU-accelerated probabilistic
logic programming runtime:

- **GPU-resident Datalog** evaluation with CUDA-backed DLPack producers and
  zero-copy transient inputs and exported views (PyTorch, JAX, cuDF); persistent
  relation replacements take owned device-to-device snapshots.
- **Probabilistic inference**: exact weighted model counting via knowledge
  compilation to device-resident arithmetic circuits, plus seeded Monte Carlo
  sampling with confidence intervals.
- **Neural-symbolic training**: neural predicates, circuit caching across
  iterations, and PyTorch autograd integration.
- **Differentiable ILP** rule learning with GPU-resident credit assignment.
- **Native relation evidence** with ordered roles, whole-fact provenance, and
  exact versioned manifest export/import paired with DLPack relation columns.

Requires Linux x86_64 with an NVIDIA GPU and CUDA 13.x.

## Quick start

```python
import torch
from pyxlog import LogicProgram

prog = LogicProgram.compile("""
pred edge(i64, i64).
pred reach(i64, i64).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(X, Y).
""".strip(), device=0, memory_mb=1024)

edge_src = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
edge_dst = torch.tensor([2, 3], device="cuda", dtype=torch.int64)
result = prog.evaluate({"edge": [edge_src, edge_dst]})

q0 = result.queries[0]
out_src = torch.utils.dlpack.from_dlpack(q0.tensors[0])
out_dst = torch.utils.dlpack.from_dlpack(q0.tensors[1])
print(torch.stack([out_src, out_dst], dim=1).cpu().tolist())
# [[1, 2], [1, 3], [2, 3]]
```

The package is built from the `BrainyBlaze/xlog` repository and exposes the
native extension module together with the staged CUDA kernel artifacts needed by
the runtime.

At import time, `pyxlog` prefers packaged kernel artifacts under
`pyxlog/kernels/` and exports that path to `XLOG_CUBIN_DIR` automatically when
the wheel includes them. For source-tree validation, ad-hoc probe scripts, or
artifact runners that execute without the packaged kernel directory, set
`XLOG_CUBIN_DIR` explicitly to a directory containing `.cubin` or
`.portable.ptx` files before importing `pyxlog`.

Project documentation, setup instructions, and release notes live in the
repository root:

- https://github.com/BrainyBlaze/xlog

Use the root project README for installation requirements, CUDA expectations,
and end-to-end examples. For local source-tree installs, use the repository
helper from the root directory so the wheel is built for and installed into the
same Python interpreter your downstream project uses:

```bash
python scripts/install_pyxlog_for_python.py --python /usr/local/bin/python --user
```

## Completed semantic observations

With the `semantic-policy` build feature, the native prepared step's
`execution_measurements` returns `SemanticCompletedExecutionObservation` after
the original segment completes and its receipts and work are validated. Obtain
it before step retirement and read `device_work`, `model_work`, `native_work`,
and `model_calls` as read-only attributes, not dictionary keys.

`SemanticModelEvaluation.finish` similarly returns
`SemanticCompletedModelEvaluation`. Its read-only attributes are `status`,
`loss_bits`, `model_work`, `model_calls`, `operation_count`, `work_bound`, and
`retained_allocation_bytes`. A known numerical refusal still has a completed
observation; an incomplete or unknown invocation raises and retains its original
owner instead of returning measurement values.

Neither completed type has a public constructor or retains the retired model,
selected cohort, CUDA storage, or reader. Retain the original object when passing
native execution custody; a cold dictionary projection is only a serialization,
not a replacement for that object. These observations describe the original
execution, not preparation expenditure or a whole-process physical memory peak.
