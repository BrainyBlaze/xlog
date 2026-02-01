#[cfg(feature = "arrow-device-import")]
mod tests {
    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType, Schema};
    use xlog_cuda::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    fn setup_provider() -> Option<Arc<CudaKernelProvider>> {
        let device = match CudaDevice::new(0) {
            Ok(d) => Arc::new(d),
            Err(e) => {
                eprintln!("Skipping: CUDA runtime unavailable: {}", e);
                return None;
            }
        };
        let memory = Arc::new(GpuMemoryManager::new(
            device.clone(),
            MemoryBudget::with_limit(512 * 1024 * 1024),
        ));
        CudaKernelProvider::new(device, memory).ok().map(Arc::new)
    }

    #[test]
    fn arrow_device_roundtrip_import_export() {
        let Some(provider) = setup_provider() else {
            eprintln!("Skipping: no CUDA device");
            return;
        };

        let schema = Schema::new(vec![("x".to_string(), ScalarType::U32)]);
        let buffer = provider
            .create_buffer_from_u32_slice(&[1, 2, 3], schema.clone())
            .expect("create buffer");

        let arrow_dev = provider
            .to_arrow_device_record_batch(buffer)
            .expect("export device record batch");

        let imported = provider
            .from_arrow_device_record_batch(arrow_dev)
            .expect("import device record batch");

        assert_eq!(imported.schema(), &schema);
        assert_eq!(imported.num_rows(), 3);

        let batch = provider
            .to_arrow_record_batch(&imported)
            .expect("to arrow record batch");
        let array = batch
            .column(0)
            .as_any()
            .downcast_ref::<arrow::array::UInt32Array>()
            .expect("u32 array");
        assert_eq!(array.len(), 3);
        assert_eq!(array.value(0), 1);
        assert_eq!(array.value(1), 2);
        assert_eq!(array.value(2), 3);
    }
}

#[cfg(not(feature = "arrow-device-import"))]
#[test]
fn arrow_device_import_feature_disabled() {
    eprintln!("arrow-device-import feature disabled; skipping");
}
