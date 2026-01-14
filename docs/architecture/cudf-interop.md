# cuDF / Arrow Interop

XLOG’s CUDA backend (`crates/xlog-cuda`) can export/import `CudaBuffer` data using Apache Arrow.
This enables interoperability with the RAPIDS ecosystem (cuDF) and other Arrow-native tools.

## Current State

- Export/import is **compatible** with Arrow and cuDF workflows.
- Arrow export/import is **not zero-copy** today: export downloads GPU → host; import uploads host → GPU.
- A **zero-copy** path exists via DLPack export/import (per-column) from device memory (contiguous 1D columns).

## Rust API

- `xlog_cuda::CudaKernelProvider::to_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::from_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::to_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::from_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::write_arrow_ipc_stream_file`
- `xlog_cuda::CudaKernelProvider::read_arrow_ipc_stream_file`
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
- ✅ Python capsule/FFI layer (Phase 4): `crates/xlog-gpu-py` builds a `xlog_gpu` module via `maturin` that:
  - accepts DLPack capsules / `__dlpack__` producers for input relations
  - returns DLPack capsules for query result columns
  - provides a `dlpack_roundtrip(...)` helper for low-level DLPack validation

## Python cuDF Example (via DLPack)

This uses cuDF as a DLPack producer and round-trips a GPU column through XLOG’s DLPack boundary:

```python
import cupy as cp
import cudf
from xlog_gpu import dlpack_roundtrip

s = cudf.Series([1, 2, 3], dtype="int32")

# Returns a DLPack capsule for the round-tripped column.
out_capsule = dlpack_roundtrip(s, device=0, memory_mb=1024)

# Convert back to a CuPy array to validate the bytes made the round trip.
out = cp.fromDlpack(out_capsule)
assert out.tolist() == [1, 2, 3]
```
