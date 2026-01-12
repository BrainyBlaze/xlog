# cuDF / Arrow Interop

XLOG’s CUDA backend (`crates/xlog-cuda`) can export/import `CudaBuffer` data using Apache Arrow.
This enables interoperability with the RAPIDS ecosystem (cuDF) and other Arrow-native tools.

## Current State

- Export/import is **compatible** with Arrow and cuDF workflows.
- It is **not zero-copy** today: export downloads GPU → host; import uploads host → GPU.

## Rust API

- `xlog_cuda::CudaKernelProvider::to_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::from_arrow_record_batch`
- `xlog_cuda::CudaKernelProvider::to_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::from_arrow_ipc_stream`
- `xlog_cuda::CudaKernelProvider::write_arrow_ipc_stream_file`
- `xlog_cuda::CudaKernelProvider::read_arrow_ipc_stream_file`

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

True zero-copy cuDF interop likely requires a GPU-native interchange path (e.g., DLPack or a CUDA-aware Arrow memory representation).
