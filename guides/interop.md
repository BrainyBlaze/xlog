# Arrow, DLPack, and cuDF interop

Move relation columns between XLOG and the GPU tensor ecosystem — which paths keep data on the device and which pay a host round-trip.

XLOG's query results and input relations are columnar tensors that live in device
memory. To use them from PyTorch, JAX, cuDF, or another GPU framework, you need an
interchange format. DLPack query-result handoff and transient evaluation input are
zero-copy, as is exporting a buffer from the joint-constraint solver's carrier.
Persistent session replacement is deliberately different: it makes one
device-to-device copy into session-owned storage. None of these paths use host row
data. XLOG's CUDA backend (`crates/xlog-cuda`) also offers the Arrow C Device
interface and host-serialized Arrow IPC.

## Zero-copy versus copy at a glance

The distinction is the whole point of this page. A **zero-copy** path hands the
consumer a pointer to the same device buffer XLOG already holds — no bytes move. A
**copy** path moves bytes into a different allocation. Persistent DLPack ownership
and retention copies stay on the GPU; Arrow IPC instead serializes through host
memory and pays a device-to-host-to-device round trip for GPU-to-GPU interchange.

| Path | Direction | Data movement | Status |
|---|---|---|---|
| DLPack query results (per column) | Export | Zero-copy handoff, stays on GPU | Stable |
| DLPack transient evaluation input (per column) | Import | Zero-copy, stays on GPU | Stable |
| DLPack persistent session replacement | Import | One device-to-device ownership snapshot; no host copy | Stable |
| DLPack persistent stored-relation export | Export | One device-to-device retention copy, then a zero-copy capsule handoff; no host copy | Stable |
| DLPack joint-solver carrier buffer | Export | Zero-copy view of the solver's own device allocation; XLOG stays the owner | Stable |
| Arrow C Device interface | Export | Zero-copy, stays on GPU | Stable |
| Arrow C Device interface | Import | Zero-copy, stays on GPU | Experimental, feature-gated |
| Arrow IPC stream | Export and import | Host copy — GPU to host, host to GPU | Stable |

<Note>
Arrow **IPC** and the Arrow **C Device** interface are both "Arrow," but they sit on
opposite sides of the copy boundary. IPC is a host-serialized byte stream; the C Device
interface passes CUDA device pointers. Reach for IPC only when the consumer genuinely
needs host-side Arrow (a `pyarrow.Table`, a file on disk); reach for the C Device
interface or DLPack when the data must stay on the GPU.
</Note>

## DLPack — zero-copy handoff and owned persistence

DLPack is the most direct way to share a column with a tensor framework. XLOG exports
each relation column as a contiguous 1-D device tensor and consumes DLPack tensors the
same way, so a query result flows into PyTorch without leaving the GPU:

```python
import pyxlog
import torch

program = pyxlog.LogicProgram.compile("""
    pred edge(u32, u32).
    pred reach(u32, u32).

    edge(1, 2). edge(2, 3). edge(3, 4).

    reach(X, Y) :- edge(X, Y).
    reach(X, Z) :- reach(X, Y), edge(Y, Z).

    ?- reach(1, N).
""")

result = program.evaluate()

for q in result.queries:
    # Each column is a DLPack tensor; torch.from_dlpack wraps it with no copy.
    cols = [torch.from_dlpack(t) for t in q.tensors]
    print(q.relation_name, q.num_rows, cols)
```

**Confirm it stayed on the GPU.** Check each wrapped tensor's device: `cols[0].device`
reports `cuda` (the same device XLOG computed on). A host round-trip would surface as
`cpu` instead.

Input relations travel the same path in reverse. `evaluate(dlpack_inputs=...)` takes a
mapping from relation name to a sequence of 1-D DLPack columns — one tensor per column,
never a single 2-D tensor:

```python
# Two 1-D columns for edge(u32, u32), already resident on the GPU.
edge_src = torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32)
edge_dst = torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32)

result = program.evaluate(dlpack_inputs={"edge": [edge_src, edge_dst]})
```

The Rust surface mirrors this with `to_dlpack_table` for export and
`from_dlpack_tensors` / `from_dlpack_tensors_with_schema` for import — the latter
validates the incoming columns against a declared schema instead of inferring it.

Persistent `LogicRelationSession` replacements (`put_relation`,
`put_relation_with_provenance`, and `put_relation_from_manifest`) take one
device-to-device snapshot after DLPack import. The session must own stable rows:
a producer tensor remains usable under the DLPack sharing contract, so retaining
its pointer would let later in-place writes bypass relation versions, callbacks,
and evidence validation. Transient `evaluate(dlpack_inputs=...)` inputs remain
zero-copy, as does wrapping a query-result capsule in the consumer framework.
Exporting a stored persistent relation first leaves a device-to-device clone in
the session so it can transfer ownership of the exported buffer; consuming the
returned capsule itself adds no further copy.

For each tensor-like input, XLOG calls `__dlpack_device__()` exactly once. Only
CUDA device memory (`kDLCUDA`) is accepted; another device raises `BufferError`
before any capsule request. XLOG then calls `__dlpack__(stream=1)` exactly once,
using the legacy default stream so pending producer work completes before XLOG
reads it. If that call fails, XLOG propagates the exception without retrying.
A raw capsule bypasses both calls and cannot perform the handshake: create it for
stream `1` or synchronize its producer before passing it to XLOG. Raw capsules
are single-consumer objects, and the native importer still validates the device
header. Capsules exported by XLOG are already ready for this consumer stream.

<Warning>
DLPack columns are contiguous 1-D device buffers. A 2-D tensor, a non-contiguous view,
or a strided slice is not a valid column — materialize a contiguous copy on your side
before handing it to XLOG.
</Warning>

## Provenance-aware DLPack round trips

A persistent logic session can export DLPack columns together with the native roles and
whole-fact evidence needed to reconstruct the relation in another compatible session:

```python
source = program.session()
source.put_relation_with_provenance(
    "edge",
    [edge_src, edge_dst],
    roles=[{"name": "source"}, {"name": "target"}],
    facts=[{
        "tuple": [1, 2],
        "provenance": [
            {"source": "extractor", "document": "record-17"},
            {"source": "review", "kind": "confirmation"},
        ],
    }],
)

exported = source.export_relation_with_provenance("edge")

target = program.session()
snapshot = target.put_relation_from_manifest(
    "edge",
    exported["columns"],
    exported["manifest"],
)
```

`target.relation("edge")` returns a native `RelationEvidence` snapshot after the
import. Invalid role, whole-fact provenance, insert-evidence, or manifest input raises
`pyxlog.RelationMetadataError`, a `ValueError` subclass. `relation(name)` and
`evidence(name)` raise `KeyError` for an unstored relation, while
`export_relation_with_provenance(name)` raises `ValueError`. Without `pyxlog._native`,
both package-level type names remain importable: the fallback metadata error subclasses
`ValueError`, and constructing fallback `RelationEvidence` raises `RuntimeError`.
Native instances and session operations require the extension. Source builds with the
extension expose this API; packaged availability is determined by release notes.

The manifest has a strict, versioned shape:

```python
{
    "format": "xlog.relation-provenance",
    "version": 1,
    "predicate": {
        "name": "edge",
        "arity": 2,
        "schema_sha256": "sha256:...",
    },
    "row_count": 3,
    "metadata_present": True,
    "roles": [
        {"name": "source", "sort": None, "type": "u32"},
        {"name": "target", "sort": None, "type": "u32"},
    ],
    "facts": [{
        "identity": "sha256:...",
        "cells": [
            {"type": "u32", "hex": "01000000"},
            {"type": "u32", "hex": "02000000"},
        ],
        "provenance": [
            {
                "source": "extractor",
                "document": "record-17",
                "span": None,
                "content_hash": None,
                "kind": None,
                "polarity": None,
            },
            {
                "source": "review",
                "document": None,
                "span": None,
                "content_hash": None,
                "kind": "confirmation",
                "polarity": None,
            },
        ],
    }],
}
```

All displayed fields are required, and every dictionary rejects unknown fields.
`metadata_present` must be a Python `bool`; when false, `roles` and `facts` are empty and
the import resets any positional role contract. Each exact cell has only `type` and
`hex`: the type matches the compiled column, the lowercase hex has exactly the scalar's
byte width, and a boolean is `00` or `01`. A non-null span has only non-negative integer
`start` and `end` values (not booleans), both representable as `u64`, with
`start <= end`. Nullary relations reject both provenance-manifest import and export.
Import checks the predicate name, arity,
compiled schema fingerprint, fact identities, DLPack column contract, row count, and
complete-fact membership before replacing the target relation.

The schema fingerprint is a domain-separated SHA-256 digest of the predicate name and
arity plus each compiled argument's ordered name, scalar type, and optional source-domain
alias. Each fact identity separately hashes the predicate name and arity plus every
ordered cell's scalar type, byte length, and exact little-endian bytes. It is independent
of row position, role labels, and provenance records, and import recomputes it rather
than trusting the manifest string.

Fact order, record order, repeated fact entries, and exact duplicate records are
normalized deterministically. Different records attached to the same complete tuple are
preserved. A failed import leaves the target rows and evidence unchanged.

<Warning>
The manifest is JSON-compatible metadata, but the paired DLPack columns are process-local
ownership objects. Static manifest, schema, and column-count validation happens before
capsule consumption. Once column import starts, every supplied capsule is consumed
before dtype, equal-column-length, manifest row-count, and membership validation
completes. Target replacement remains atomic on failure, but spent source capsules
cannot be reused. Successful import consumes each capsule exactly once, so do not first
pass the exported columns to another DLPack consumer and then try to import them. The
capsules retain the device buffers even if the source session is released, but they are
not cross-process or on-disk persistence. Use Arrow IPC when portability matters,
accepting its host copy.
</Warning>

## Joint-solver carrier buffers — writing into XLOG's own allocation

The paths above move *relation columns*. `pyxlog.JointConstraintCarrier` is a different
shape of interop: it is a device-resident workspace for XLOG's joint constraint solver,
and it lets a model write scores straight into the buffers the solver will read, with no
intermediate relation and no copy.

Create the carrier, register a schema, bind the per-label signature masks, then export
the buffer you want to fill:

```python
import pyxlog
import torch
from torch.utils.dlpack import from_dlpack

# device, entities, domain_lanes, candidates, labels, fuel_limit
carrier = pyxlog.JointConstraintCarrier(0, 3, 1, 2, 4, 64)
carrier.register_schema(catalog_sha, pyxlog.SOLVER_ABI_IDENTITY)
carrier.bind_signatures(head_masks, tail_masks)

scores = from_dlpack(carrier.export_buffer("scores"))   # (candidates, labels) float32
scores.copy_(model_logits)                              # lands in the solver's memory
carrier.solve_label_feasibility(3)                      # 3 = the abstain label
```

**Confirm it stayed zero-copy.** Export the same buffer twice and write through the
first view: the second view sees the write, because both alias one allocation. A copying
path would not.

`export_buffer(name)` accepts `domains`, `scores`, `constraints`, `outputs`,
`feasible_sets`, `logical_counts`, `map_results`, and `solve_status`; any other name
raises `ValueError`. XLOG owns every one of these allocations, so a view is valid only
while the carrier is alive — this is the opposite of the relation-export capsules above,
which keep their buffers alive on their own.

<Warning>
Writes you make through an exported view happen on **your** stream, not XLOG's. Either
call `note_producer_stream(...)` with that stream's handle after the writes — the next
solve then waits for it on the device — or synchronize on the host before solving.
Pass a dedicated `torch.cuda.Stream()`: torch's default stream carries raw handle `0`,
which XLOG cannot tell apart from a null handle and refuses. `note_consumer_stream(...)`
is the mirror image, making a successful solve enqueue a wait on your stream rather than
forcing a host barrier; those registrations are one-shot and are cleared if the solve
fails.
</Warning>

Two exceptions belong to this surface, both `RuntimeError` subclasses:

- `pyxlog.CarrierRefused` — a typed refusal whose message names the exact precondition
  you broke and its values. It covers registering a schema twice, a zero-valued capacity
  dimension, binding signatures or solving before `register_schema`, rebinding
  already-bound masks, a mask whose word count does not match the carrier's capacity,
  running the top-two stage before feasibility, a malformed component plan, an
  out-of-range abstain label, and a joint-solve kernel that would not load on this device.
- `pyxlog.SolverResourceExhausted` — the fuel budget you passed to the constructor ran
  out, reported as `solver fuel exhausted: spent {spent} of {limit} node expansions`. The
  meter lives in the carrier session and saturates, so a retry repeats the identical
  refusal instead of slipping more work past the budget. Read the running total from the
  `fuel_spent` property.

## Arrow C Device interface — zero-copy Arrow

When the consumer speaks Arrow but you still want to stay on the GPU, the Arrow C
Device interface exports device-resident columns as an `ArrowDeviceArray` carrying CUDA
device pointers. The device descriptor reports `device_type = ARROW_DEVICE_CUDA` and the
originating `device_id`, so a RAPIDS consumer knows the buffers are already on the
right device.

Export covers `u32`, `u64`, `i32`, `i64`, `f32`, `f64`, `bool` (bit-packed), and
`symbol` (exported as `UInt32`). Symbol columns carry `xlog.symbol=true` and
`xlog.symbol_encoding=u32` in their schema metadata so the interned-string encoding
survives the boundary. Ownership stays explicit: `ArrowDeviceArrayOwned` keeps the GPU
buffers alive until the FFI handle is released.

Import over the same interface is **experimental** and compiled out by default. Build
`pyxlog` with the `arrow-device-import` feature to enable it:

```bash
maturin develop --features arrow-device-import
```

With that feature, import wraps an `ArrowDeviceArrayOwned`'s device pointers as XLOG
columns without copying. It accepts numeric types and `Symbol` (as `UInt32` marked
`xlog.symbol=true`); it currently **rejects nulls** and does **not** yet accept
bit-packed `Bool`.

## Arrow IPC — the host-copy path

Arrow IPC is the interoperability escape hatch: it serializes relations to a standard
Arrow stream that any Arrow-native tool can read, including plain CPU `pyarrow`. It is
**not zero-copy** — export downloads GPU to host, and import uploads host to GPU. Use it
when you need a portable artifact or a host-side table, not on a hot path.

The Rust side writes a stream file with `write_arrow_ipc_stream_file`; Python reads it
back through `pyarrow` and, if desired, up-loads it into cuDF:

```python
import pyarrow.ipc as ipc
import cudf

with open("data.arrow", "rb") as f:
    table = ipc.open_stream(f).read_all()

df = cudf.DataFrame.from_arrow(table)   # host table -> GPU (upload)
print(df)
```

That `from_arrow` step is exactly the host-to-GPU copy the zero-copy paths avoid — a
reminder to prefer DLPack or the Arrow C Device interface whenever the consumer can take
device pointers directly.

<Card title="How results stay on the GPU" icon="microchip" href="/core-concepts/gpu-residency">
  Why zero-copy interop matters — XLOG keeps semantic state device-resident so a
  downstream tensor computation reads its output without a host round-trip.
</Card>
