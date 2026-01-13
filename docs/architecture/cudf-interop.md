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

## Next Work (Zero-Copy)

True zero-copy cuDF interop needs a GPU-native interchange path:
- ✅ DLPack export (current): produces DLPack `DLManagedTensor` pointers for each column without copies
- ✅ DLPack import (current): consumes DLPack `DLManagedTensor` pointers and wraps them without copies
- Next: Python capsule/FFI layer + cuDF example (DLPack) or CUDA-aware Arrow memory
