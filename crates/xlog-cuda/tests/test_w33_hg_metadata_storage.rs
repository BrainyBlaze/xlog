use std::sync::Arc;

use xlog_core::{MemoryBudget, ScalarType, Schema};
use xlog_cuda::device_runtime::{LogRecord, LoggingSink, SinkError, StreamId};
use xlog_cuda::memory::CudaBuffer;
use xlog_cuda::{CudaKernelProvider, CudaProviderBuilder, GpuMemoryManager};

struct DiscardSink;

impl LoggingSink for DiscardSink {
    fn emit(&self, _record: LogRecord) -> Result<(), SinkError> {
        Ok(())
    }
}

struct Fix {
    memory: Arc<GpuMemoryManager>,
    provider: Arc<CudaKernelProvider>,
    stream: StreamId,
}

fn make_fix() -> Option<Fix> {
    let provider = Arc::new(
        CudaProviderBuilder::new(0, MemoryBudget::with_limit(512 * 1024 * 1024))
            .with_logging_sink(Arc::new(DiscardSink) as Arc<dyn LoggingSink>)
            .build()
            .ok()?,
    );
    let memory = Arc::clone(provider.memory());
    let stream = memory.runtime()?.stream_pool().acquire().ok()?;
    Some(Fix {
        memory,
        provider,
        stream,
    })
}

fn upload_grouped_binary(memory: &Arc<GpuMemoryManager>, rows: u32, keys: u32) -> CudaBuffer {
    let bytes_per_col = rows as usize * std::mem::size_of::<u32>();
    let mut col0 = memory.alloc::<u8>(bytes_per_col).expect("alloc col0");
    let mut col1 = memory.alloc::<u8>(bytes_per_col).expect("alloc col1");
    let mut d_num_rows = memory.alloc::<u32>(1).expect("alloc rows");

    let mut host0 = Vec::with_capacity(bytes_per_col);
    let mut host1 = Vec::with_capacity(bytes_per_col);
    let rows_per_key = rows.div_ceil(keys);
    for i in 0..rows {
        let key = (i / rows_per_key).min(keys - 1);
        host0.extend_from_slice(&key.to_le_bytes());
        host1.extend_from_slice(&i.to_le_bytes());
    }

    let device = memory.device().inner();
    device
        .htod_sync_copy_into(&host0, &mut col0)
        .expect("htod col0");
    device
        .htod_sync_copy_into(&host1, &mut col1)
        .expect("htod col1");
    device
        .htod_sync_copy_into(&[rows], &mut d_num_rows)
        .expect("htod row count");

    let schema = Schema::new(vec![
        ("col0".to_string(), ScalarType::U32),
        ("col1".to_string(), ScalarType::U32),
    ]);
    CudaBuffer::from_columns_with_host_count(
        vec![col0.into(), col1.into()],
        rows as u64,
        d_num_rows,
        schema,
        rows,
    )
}

#[test]
fn metadata_storage_overhead_under_one_percent_at_paper_scale() {
    let Some(fix) = make_fix() else {
        eprintln!("Skipping: CUDA runtime unavailable");
        return;
    };
    let relation = upload_grouped_binary(&fix.memory, 977_000, 1_000);
    let metadata = fix
        .provider
        .wcoj_build_metadata_u32_recorded(&relation, 0, fix.stream)
        .expect("build metadata");
    let relation_bytes = relation.estimated_bytes();
    let metadata_bytes = metadata.metadata_bytes();
    assert!(
        metadata_bytes * 100 < relation_bytes,
        "metadata bytes {metadata_bytes} must be under 1% of relation bytes {relation_bytes}"
    );
    assert_eq!(metadata.key_count, 1_000);
}
