use xlog_core::{Result, ScalarType, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::wcoj_metadata::WcojRelationMetadata;
use crate::{CudaBuffer, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;

impl CudaKernelProvider {
    pub fn wcoj_build_metadata_u32_recorded(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u32>> {
        self.validate_metadata_column(input, key_col_idx, MetadataWidth::U32)?;
        let keys = metadata_column_u32(input, key_col_idx)?;
        self.build_metadata_u32_from_column(input, key_col_idx, keys, launch_stream)
    }

    pub fn wcoj_build_metadata_u64_recorded(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u64>> {
        self.validate_metadata_column(input, key_col_idx, MetadataWidth::U64)?;
        let keys = metadata_column_u64(input, key_col_idx)?;
        self.build_metadata_u64_from_column(input, key_col_idx, keys, launch_stream)
    }

    fn validate_metadata_column(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        width: MetadataWidth,
    ) -> Result<()> {
        if key_col_idx >= input.arity() {
            return Err(XlogError::Kernel(format!(
                "wcoj_build_metadata: key column {} out of range for arity {}",
                key_col_idx,
                input.arity()
            )));
        }
        let ty = input.schema().column_type(key_col_idx).ok_or_else(|| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata: column {} type missing",
                key_col_idx
            ))
        })?;
        match (width, ty) {
            (MetadataWidth::U32, ScalarType::U32 | ScalarType::Symbol)
            | (MetadataWidth::U64, ScalarType::U64) => Ok(()),
            (MetadataWidth::U32, other) => Err(XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: column {} must be U32 or Symbol, got {:?}",
                key_col_idx, other
            ))),
            (MetadataWidth::U64, other) => Err(XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: column {} must be U64, got {:?}",
                key_col_idx, other
            ))),
        }
    }

    fn build_metadata_u32_from_column(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u32>,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u32>> {
        let n = self.metadata_logical_rows(input)?;
        if n == 0 {
            return Ok(WcojRelationMetadata {
                unique_keys: self.memory().alloc::<u32>(0)?,
                fan_out: self.memory().alloc::<u32>(0)?,
                prefix_sum: self.memory().alloc::<u32>(0)?,
                total: 0,
                key_count: 0,
                row_count: 0,
            });
        }
        let mut boundary_mask = self.memory().alloc::<u32>(n as usize)?;
        let mut boundary_prefix = self.memory().alloc::<u32>(n as usize)?;
        self.mark_metadata_boundaries_u32(
            input,
            key_col_idx,
            keys,
            n,
            &mut boundary_mask,
            &mut boundary_prefix,
            launch_stream,
        )?;
        let key_count = self.metadata_scanned_count(&boundary_mask, &boundary_prefix, n)?;

        let mut unique_keys = self.memory().alloc::<u32>(key_count as usize)?;
        let mut fan_out = self.memory().alloc::<u32>(key_count as usize)?;
        let mut prefix_sum = self.memory().alloc::<u32>(key_count as usize)?;
        self.scatter_metadata_u32(
            input,
            key_col_idx,
            keys,
            n,
            &boundary_mask,
            &boundary_prefix,
            &mut unique_keys,
            &mut fan_out,
            &mut prefix_sum,
            launch_stream,
        )?;
        let total = self.metadata_scanned_total(&fan_out, &prefix_sum, key_count)?;

        Ok(WcojRelationMetadata {
            unique_keys,
            fan_out,
            prefix_sum,
            total,
            key_count,
            row_count: n,
        })
    }

    fn build_metadata_u64_from_column(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u64>,
        launch_stream: StreamId,
    ) -> Result<WcojRelationMetadata<u64>> {
        let n = self.metadata_logical_rows(input)?;
        if n == 0 {
            return Ok(WcojRelationMetadata {
                unique_keys: self.memory().alloc::<u64>(0)?,
                fan_out: self.memory().alloc::<u32>(0)?,
                prefix_sum: self.memory().alloc::<u32>(0)?,
                total: 0,
                key_count: 0,
                row_count: 0,
            });
        }
        let mut boundary_mask = self.memory().alloc::<u32>(n as usize)?;
        let mut boundary_prefix = self.memory().alloc::<u32>(n as usize)?;
        self.mark_metadata_boundaries_u64(
            input,
            key_col_idx,
            keys,
            n,
            &mut boundary_mask,
            &mut boundary_prefix,
            launch_stream,
        )?;
        let key_count = self.metadata_scanned_count(&boundary_mask, &boundary_prefix, n)?;

        let mut unique_keys = self.memory().alloc::<u64>(key_count as usize)?;
        let mut fan_out = self.memory().alloc::<u32>(key_count as usize)?;
        let mut prefix_sum = self.memory().alloc::<u32>(key_count as usize)?;
        self.scatter_metadata_u64(
            input,
            key_col_idx,
            keys,
            n,
            &boundary_mask,
            &boundary_prefix,
            &mut unique_keys,
            &mut fan_out,
            &mut prefix_sum,
            launch_stream,
        )?;
        let total = self.metadata_scanned_total(&fan_out, &prefix_sum, key_count)?;

        Ok(WcojRelationMetadata {
            unique_keys,
            fan_out,
            prefix_sum,
            total,
            key_count,
            row_count: n,
        })
    }

    fn mark_metadata_boundaries_u32(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u32>,
        n: u32,
        boundary_mask: &mut TrackedCudaSlice<u32>,
        boundary_prefix: &mut TrackedCudaSlice<u32>,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_build_metadata_u32_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_build_metadata_u32_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(key_col_idx).expect("metadata key column"));
        rec.write(boundary_mask);
        rec.write(boundary_prefix);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: mark preflight failed: {e}"
            ))
        })?;

        let kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_BUILD_METADATA_MARK_BOUNDARIES_U32,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_build_metadata_mark_boundaries_u32 kernel not found".to_string(),
                )
            })?;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsafe {
            kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (keys, n, &mut *boundary_mask, &mut *boundary_prefix),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_build_metadata_mark_boundaries_u32 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            boundary_prefix,
            n,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: mark commit failed: {e}"
            ))
        })?;
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: mark stream sync failed: {e}"
            ))
        })?;
        Ok(())
    }

    fn mark_metadata_boundaries_u64(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u64>,
        n: u32,
        boundary_mask: &mut TrackedCudaSlice<u32>,
        boundary_prefix: &mut TrackedCudaSlice<u32>,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_build_metadata_u64_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_build_metadata_u64_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(key_col_idx).expect("metadata key column"));
        rec.write(boundary_mask);
        rec.write(boundary_prefix);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: mark preflight failed: {e}"
            ))
        })?;

        let kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_BUILD_METADATA_MARK_BOUNDARIES_U64,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_build_metadata_mark_boundaries_u64 kernel not found".to_string(),
                )
            })?;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsafe {
            kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (keys, n, &mut *boundary_mask, &mut *boundary_prefix),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_build_metadata_mark_boundaries_u64 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            boundary_prefix,
            n,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: mark commit failed: {e}"
            ))
        })?;
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: mark stream sync failed: {e}"
            ))
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_metadata_u32(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u32>,
        n: u32,
        boundary_mask: &TrackedCudaSlice<u32>,
        boundary_prefix: &TrackedCudaSlice<u32>,
        unique_keys: &mut TrackedCudaSlice<u32>,
        fan_out: &mut TrackedCudaSlice<u32>,
        prefix_sum: &mut TrackedCudaSlice<u32>,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_build_metadata_u32_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_build_metadata_u32_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(key_col_idx).expect("metadata key column"));
        rec.read(boundary_mask);
        rec.read(boundary_prefix);
        rec.write(unique_keys);
        rec.write(fan_out);
        rec.write(prefix_sum);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: scatter preflight failed: {e}"
            ))
        })?;

        let kernel = self
            .device()
            .inner()
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_BUILD_METADATA_SCATTER_U32)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_build_metadata_scatter_u32 kernel not found".to_string())
            })?;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsafe {
            kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        keys,
                        n,
                        boundary_mask,
                        boundary_prefix,
                        &mut *unique_keys,
                        &mut *fan_out,
                        &mut *prefix_sum,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_build_metadata_scatter_u32 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            prefix_sum,
            unique_keys.len() as u32,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: scatter commit failed: {e}"
            ))
        })?;
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u32_recorded: scatter stream sync failed: {e}"
            ))
        })?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn scatter_metadata_u64(
        &self,
        input: &CudaBuffer,
        key_col_idx: usize,
        keys: &TrackedCudaSlice<u64>,
        n: u32,
        boundary_mask: &TrackedCudaSlice<u32>,
        boundary_prefix: &TrackedCudaSlice<u32>,
        unique_keys: &mut TrackedCudaSlice<u64>,
        fan_out: &mut TrackedCudaSlice<u32>,
        prefix_sum: &mut TrackedCudaSlice<u32>,
        launch_stream: StreamId,
    ) -> Result<()> {
        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "wcoj_build_metadata_u64_recorded requires a runtime-backed GpuMemoryManager"
                    .to_string(),
            )
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "wcoj_build_metadata_u64_recorded: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(input.num_rows_device());
        rec.read_column(input.column(key_col_idx).expect("metadata key column"));
        rec.read(boundary_mask);
        rec.read(boundary_prefix);
        rec.write(unique_keys);
        rec.write(fan_out);
        rec.write(prefix_sum);
        rec.preflight(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: scatter preflight failed: {e}"
            ))
        })?;

        let kernel = self
            .device()
            .inner()
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_BUILD_METADATA_SCATTER_U64)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_build_metadata_scatter_u64 kernel not found".to_string())
            })?;
        let grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
        unsafe {
            kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        keys,
                        n,
                        boundary_mask,
                        boundary_prefix,
                        &mut *unique_keys,
                        &mut *fan_out,
                        &mut *prefix_sum,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_build_metadata_scatter_u64 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            prefix_sum,
            unique_keys.len() as u32,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec.commit(runtime).map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: scatter commit failed: {e}"
            ))
        })?;
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!(
                "wcoj_build_metadata_u64_recorded: scatter stream sync failed: {e}"
            ))
        })?;
        Ok(())
    }

    fn metadata_scanned_count(
        &self,
        boundary_mask: &TrackedCudaSlice<u32>,
        boundary_prefix: &TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<u32> {
        let last = n - 1;
        let prefix_last = self.dtoh_scalar_untracked::<u32>(boundary_prefix, last as usize)?;
        let mask_last = self.dtoh_scalar_untracked::<u32>(boundary_mask, last as usize)?;
        Ok(prefix_last + mask_last)
    }

    fn metadata_scanned_total(
        &self,
        fan_out: &TrackedCudaSlice<u32>,
        prefix_sum: &TrackedCudaSlice<u32>,
        key_count: u32,
    ) -> Result<u64> {
        if key_count == 0 {
            return Ok(0);
        }
        let last = key_count - 1;
        let prefix_last = self.dtoh_scalar_untracked::<u32>(prefix_sum, last as usize)?;
        let fan_out_last = self.dtoh_scalar_untracked::<u32>(fan_out, last as usize)?;
        Ok(u64::from(prefix_last) + u64::from(fan_out_last))
    }

    fn metadata_logical_rows(&self, input: &CudaBuffer) -> Result<u32> {
        if let Some(cached) = input.cached_row_count() {
            return Ok(cached);
        }
        self.dtoh_scalar_untracked::<u32>(input.num_rows_device(), 0)
    }
}

#[derive(Clone, Copy)]
enum MetadataWidth {
    U32,
    U64,
}

fn metadata_column_u32<'a>(
    input: &'a CudaBuffer,
    key_col_idx: usize,
) -> Result<&'a TrackedCudaSlice<u32>> {
    let col = input.column(key_col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_build_metadata_u32_recorded: column {} not found",
            key_col_idx
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => unsafe {
            Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u32>))
        },
        _ => Err(XlogError::Kernel(
            "wcoj_build_metadata_u32_recorded: key column must be an owned CudaColumn".to_string(),
        )),
    }
}

fn metadata_column_u64<'a>(
    input: &'a CudaBuffer,
    key_col_idx: usize,
) -> Result<&'a TrackedCudaSlice<u64>> {
    let col = input.column(key_col_idx).ok_or_else(|| {
        XlogError::Kernel(format!(
            "wcoj_build_metadata_u64_recorded: column {} not found",
            key_col_idx
        ))
    })?;
    match col {
        CudaColumn::Owned(slice) => unsafe {
            Ok(&*(slice as *const TrackedCudaSlice<u8> as *const TrackedCudaSlice<u64>))
        },
        _ => Err(XlogError::Kernel(
            "wcoj_build_metadata_u64_recorded: key column must be an owned CudaColumn".to_string(),
        )),
    }
}
