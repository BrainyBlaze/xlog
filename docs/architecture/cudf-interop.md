# cuDF / Arrow Interop

XLOG’s CUDA backend (`crates/xlog-cuda`) can export/import `CudaBuffer` data using Apache Arrow.
This enables interoperability with the RAPIDS ecosystem (cuDF) and other Arrow-native tools.

## Current State

- Export/import is **compatible** with Arrow and cuDF workflows.
- Arrow IPC export/import is **not zero-copy** today: export downloads GPU → host; import uploads host → GPU.
- Arrow C Data Interface **device export is zero-copy** and keeps buffers on GPU.
- Arrow C Data Interface **device import is available experimentally** (feature-gated) for supported types.
- A **zero-copy** path exists via DLPack export/import (per-column) from device memory (contiguous 1D columns).

## Rust API

- `xlog_cuda::CudaKernelProvider::to_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::from_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::to_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::from_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::write_arrow_ipc_stream_file`
- `xlog_cuda::CudaKernelProvider::read_arrow_ipc_stream_file`
- `xlog_cuda::CudaKernelProvider::to_arrow_device_record_batch`
- `xlog_cuda::CudaKernelProvider::from_arrow_device_record_batch` (**experimental**, requires `--features arrow-device-import`)
- `xlog_cuda::ArrowDeviceArray` / `xlog_cuda::ArrowDeviceArrayOwned`
- `xlog_cuda::CudaKernelProvider::to_dlpack_table` (zero-copy export)
- `xlog_cuda::CudaKernelProvider::from_dlpack_tensors` (zero-copy import, infers schema)
- `xlog_cuda::CudaKernelProvider::from_dlpack_tensors_with_schema` (zero-copy import, checks schema)

## Python cuDF Example (via Arrow IPC)

1. In Rust, write an Arrow IPC stream file using `write_arrow_ipc_stream_file(...)`.
2. In Python:

```python
import pyarrow as pa
import pyarrow.ipc as ipc
import cudf

with open("data.arrow", "rb") as f:
    reader = ipc.open_stream(f)
    table = reader.read_all()

df = cudf.DataFrame.from_arrow(table)
print(df)
```

## Zero-Copy (DLPack)

DLPack provides a GPU-native interchange path that avoids host copies. The current implementation includes:
- ✅ DLPack export (current): produces DLPack `DLManagedTensor` pointers for each column without copies
- ✅ DLPack import (current): consumes DLPack `DLManagedTensor` pointers and wraps them without copies
- ✅ Python capsule/FFI layer: `crates/pyxlog` builds a `pyxlog` module via `maturin` that:
  - accepts DLPack capsules / `__dlpack__` producers for input relations
  - returns DLPack capsules for query result columns
  - provides a `dlpack_roundtrip(...)` helper for low-level DLPack validation

## Zero-Copy (Arrow C Data Interface, device export/import)

The CUDA backend can export and (experimentally) import device-resident Arrow C Data Interface handles without host transfers:

- **Export**: produces an `ArrowDeviceArray` with CUDA device pointers.
- **Import (experimental)**: consumes an `ArrowDeviceArrayOwned` and wraps device pointers as `CudaColumn` without copies.
- **Device descriptor**: `device_type = ARROW_DEVICE_CUDA`, `device_id = <cuda device>`.
- **Supported types (export)**: `U32`, `U64`, `I32`, `I64`, `F32`, `F64`, `Bool` (bit-packed), and `Symbol` (exported as `UInt32`).
- **Supported types (import)**: numeric types + `Symbol` (as `UInt32` with `xlog.symbol=true`). Import currently rejects nulls and does not support bit-packed `Bool` yet.
- **Symbol metadata**: schema fields include `xlog.symbol=true` and `xlog.symbol_encoding=u32`.
- **Ownership**: `ArrowDeviceArrayOwned` keeps GPU buffers alive; releasing the FFI handle frees keepalive state.

### Python (experimental)

When built with `pyxlog` feature `arrow-device-import`, Python exposes:

- `pyxlog.export_arrow_device(...) -> PyCapsule` (name `arrow_device_array`)
- `pyxlog.import_arrow_device(...) -> (dlpack_tensors, names, num_rows)`

## Python cuDF Example (via DLPack)

This uses cuDF as a DLPack producer and round-trips a GPU column through XLOG’s DLPack boundary:

```python
import cupy as cp
import cudf
from pyxlog import dlpack_roundtrip

s = cudf.Series([1, 2, 3], dtype="int32")

# Returns a DLPack capsule for the round-tripped column.
out_capsule = dlpack_roundtrip(s, device=0, memory_mb=32768)

# Convert back to a CuPy array to validate the bytes made the round trip.
out = cp.fromDlpack(out_capsule)
assert out.tolist() == [1, 2, 3]
```
