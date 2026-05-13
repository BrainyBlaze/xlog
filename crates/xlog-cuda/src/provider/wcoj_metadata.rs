use std::ffi::c_void;

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::wcoj_metadata::{WcojRelationMetadata, WcojTriangleHgWorkPlanU32};
use crate::{AsKernelParam, CudaBuffer, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;
const HG_COUNT_BLOCK_SIZE: u32 = 1024;

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

    pub fn wcoj_triangle_hg_work_plan_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<WcojTriangleHgWorkPlanU32> {
        let ctx = "wcoj_triangle_hg_work_plan_u32_recorded";
        if block_work_unit == 0 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: block_work_unit must be nonzero"
            )));
        }
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;

        let n_xy = self.metadata_logical_rows(e_xy)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;
        let prefix_len = n_xy
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: prefix length overflow")))?;
        let mut xy_work_prefix = self.memory().alloc::<u32>(prefix_len as usize)?;
        let mut xy_yz_start = self.memory().alloc::<u32>(n_xy as usize)?;
        let mut xy_yz_end = self.memory().alloc::<u32>(n_xy as usize)?;
        let mut xy_xz_start = self.memory().alloc::<u32>(n_xy as usize)?;
        let mut xy_xz_end = self.memory().alloc::<u32>(n_xy as usize)?;

        if n_xy == 0 {
            let scratch_x = self.memory().alloc::<u32>(1)?;
            let scratch_y = self.memory().alloc::<u32>(1)?;
            let scratch_z = self.memory().alloc::<u32>(1)?;
            return Ok(WcojTriangleHgWorkPlanU32 {
                xy_work_prefix,
                xy_yz_start,
                xy_yz_end,
                xy_xz_start,
                xy_xz_end,
                scratch_x,
                scratch_y,
                scratch_z,
                total_work: 0,
                block_work_unit,
                row_count: 0,
            });
        }

        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!("{ctx} requires a runtime-backed GpuMemoryManager"))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{ctx}: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let xy_col0 = metadata_column_u32(e_xy, 0)?;
        let xy_col1 = metadata_column_u32(e_xy, 1)?;
        let yz_col0 = metadata_column_u32(e_yz, 0)?;
        let xz_col0 = metadata_column_u32(e_xz, 0)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_xy.column(0).expect("xy.col0"));
        rec.read_column(e_xy.column(1).expect("xy.col1"));
        rec.read_column(e_yz.column(0).expect("yz.col0"));
        rec.read_column(e_xz.column(0).expect("xz.col0"));
        rec.write(&xy_work_prefix);
        rec.write(&xy_yz_start);
        rec.write(&xy_yz_end);
        rec.write(&xy_xz_start);
        rec.write(&xy_xz_end);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;

        let kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_TRIANGLE_BUILD_HG_WORK_PLAN_U32,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_triangle_build_hg_work_plan_u32 kernel not found".to_string(),
                )
            })?;
        let grid = n_xy.div_ceil(BLOCK_SIZE);
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
                        xy_col0,
                        xy_col1,
                        n_xy,
                        yz_col0,
                        n_yz,
                        xz_col0,
                        n_xz,
                        &mut xy_work_prefix,
                        &mut xy_yz_start,
                        &mut xy_yz_end,
                        &mut xy_xz_start,
                        &mut xy_xz_end,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_triangle_build_hg_work_plan_u32 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut xy_work_prefix,
            prefix_len,
            &cu_stream,
            launch_stream,
            runtime,
        )?;
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: stream sync failed: {e}")))?;
        let total_work = self.dtoh_scalar_untracked::<u32>(&xy_work_prefix, n_xy as usize)?;
        let scratch_slots = if total_work == 0 {
            1usize
        } else {
            let grid = total_work.div_ceil(block_work_unit);
            (grid as usize)
                .checked_mul(block_work_unit as usize)
                .ok_or_else(|| XlogError::Kernel(format!("{ctx}: scratch slot overflow")))?
        };
        let scratch_x = self.memory().alloc::<u32>(scratch_slots)?;
        let scratch_y = self.memory().alloc::<u32>(scratch_slots)?;
        let scratch_z = self.memory().alloc::<u32>(scratch_slots)?;

        Ok(WcojTriangleHgWorkPlanU32 {
            xy_work_prefix,
            xy_yz_start,
            xy_yz_end,
            xy_xz_start,
            xy_xz_end,
            scratch_x,
            scratch_y,
            scratch_z,
            total_work,
            block_work_unit,
            row_count: n_xy,
        })
    }

    pub fn wcoj_triangle_count_hg_u32_recorded(
        &self,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        plan: &WcojTriangleHgWorkPlanU32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_count_hg_u32_recorded";
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;
        let grid = if plan.total_work == 0 {
            1
        } else {
            plan.total_work.div_ceil(plan.block_work_unit)
        };
        let bytes_count = (grid as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: count byte size overflow")))?;
        let mut count_bytes = self.memory().alloc::<u8>(bytes_count)?;
        let d_num_rows = self.memory().alloc::<u32>(1)?;

        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!("{ctx} requires a runtime-backed GpuMemoryManager"))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{ctx}: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;
        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_yz.column(1).expect("yz.col1"));
        rec.read_column(e_xz.column(1).expect("xz.col1"));
        rec.read(&plan.xy_work_prefix);
        rec.read(&plan.xy_yz_start);
        rec.read(&plan.xy_yz_end);
        rec.read(&plan.xy_xz_start);
        rec.read(&plan.xy_xz_end);
        rec.write(&count_bytes);
        rec.write(&d_num_rows);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;

        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *d_num_rows.device_ptr(),
                &grid as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: H2D d_num_rows failed: {res:?}"
                )));
            }
        }

        let kernel = self
            .device()
            .inner()
            .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_U32)
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_triangle_count_hg_u32 kernel not found".to_string())
            })?;
        let count_u32 = unsafe { reinterpret_u8_as_u32(&mut count_bytes) };
        let mut params: Vec<*mut c_void> = vec![
            yz_col1.as_kernel_param(),
            n_yz.as_kernel_param(),
            xz_col1.as_kernel_param(),
            n_xz.as_kernel_param(),
            (&plan.xy_work_prefix).as_kernel_param(),
            (&plan.xy_yz_start).as_kernel_param(),
            (&plan.xy_yz_end).as_kernel_param(),
            (&plan.xy_xz_start).as_kernel_param(),
            (&plan.xy_xz_end).as_kernel_param(),
            plan.row_count.as_kernel_param(),
            plan.total_work.as_kernel_param(),
            plan.block_work_unit.as_kernel_param(),
            count_u32.as_kernel_param(),
        ];
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
                    &mut params,
                )
                .map_err(|e| XlogError::Kernel(format!("{ctx}: launch failed: {e}")))?;
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: stream sync failed: {e}")))?;

        let schema = Schema::new(vec![("count".to_string(), ScalarType::U32)]);
        Ok(CudaBuffer::from_columns_with_host_count(
            vec![count_bytes.into()],
            grid as u64,
            d_num_rows,
            schema,
            grid,
        ))
    }

    pub fn wcoj_triangle_hg_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_hg_u32_recorded";
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        let plan = self.wcoj_triangle_hg_work_plan_u32_recorded(
            e_xy,
            e_yz,
            e_xz,
            block_work_unit,
            launch_stream,
        )?;
        self.wcoj_triangle_hg_u32_with_plan_recorded(e_xy, e_yz, e_xz, &plan, launch_stream)
    }

    pub fn wcoj_triangle_hg_u32_with_plan_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        plan: &WcojTriangleHgWorkPlanU32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_hg_u32_with_plan_recorded";
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        let out_schema = Schema::new(vec![
            (
                "x".to_string(),
                e_xy.schema().column_type(0).expect("xy.col0 type").clone(),
            ),
            (
                "y".to_string(),
                e_xy.schema().column_type(1).expect("xy.col1 type").clone(),
            ),
            (
                "z".to_string(),
                e_yz.schema().column_type(1).expect("yz.col1 type").clone(),
            ),
        ]);
        if plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let bytes_count = (grid as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: count byte size overflow")))?;
        let mut count_bytes = self.memory().alloc::<u8>(bytes_count)?;
        let mut offsets = self.memory().alloc::<u32>(grid as usize)?;
        let total_rows_device = self.memory().alloc::<u32>(1)?;

        let runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(format!("{ctx} requires a runtime-backed GpuMemoryManager"))
        })?;
        let cu_stream = runtime
            .stream_pool()
            .resolve(launch_stream)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "{ctx}: launch_stream StreamId({}) does not resolve",
                    launch_stream.0
                ))
            })?;

        let xy_col0 = metadata_column_u32(e_xy, 0)?;
        let xy_col1 = metadata_column_u32(e_xy, 1)?;
        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;

        let count_u32 = unsafe { reinterpret_u8_as_u32(&mut count_bytes) };

        let mut rec_count = LaunchRecorder::new_strict(launch_stream);
        rec_count.read(e_xy.num_rows_device());
        rec_count.read(e_yz.num_rows_device());
        rec_count.read(e_xz.num_rows_device());
        rec_count.read_column(e_xy.column(0).expect("xy.col0"));
        rec_count.read_column(e_xy.column(1).expect("xy.col1"));
        rec_count.read_column(e_yz.column(1).expect("yz.col1"));
        rec_count.read_column(e_xz.column(1).expect("xz.col1"));
        rec_count.read(&plan.xy_work_prefix);
        rec_count.read(&plan.xy_yz_start);
        rec_count.read(&plan.xy_yz_end);
        rec_count.read(&plan.xy_xz_start);
        rec_count.read(&plan.xy_xz_end);
        rec_count.write(count_u32);
        rec_count.write(&plan.scratch_x);
        rec_count.write(&plan.scratch_y);
        rec_count.write(&plan.scratch_z);
        rec_count
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: cached count preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_CACHED_U32)
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_count_hg_cached_u32 kernel not found".to_string(),
                    )
                })?;
            let mut params: Vec<*mut c_void> = vec![
                xy_col0.as_kernel_param(),
                xy_col1.as_kernel_param(),
                yz_col1.as_kernel_param(),
                n_yz.as_kernel_param(),
                xz_col1.as_kernel_param(),
                n_xz.as_kernel_param(),
                (&plan.xy_work_prefix).as_kernel_param(),
                (&plan.xy_yz_start).as_kernel_param(),
                (&plan.xy_yz_end).as_kernel_param(),
                (&plan.xy_xz_start).as_kernel_param(),
                (&plan.xy_xz_end).as_kernel_param(),
                plan.row_count.as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                count_u32.as_kernel_param(),
                (&plan.scratch_x).as_kernel_param(),
                (&plan.scratch_y).as_kernel_param(),
                (&plan.scratch_z).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (grid, 1, 1),
                            block_dim: (HG_COUNT_BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: cached count launch failed: {e}"))
                    })?;
            }
        }
        rec_count
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: cached count commit failed: {e}")))?;

        let mut rec_scan = LaunchRecorder::new_strict(launch_stream);
        rec_scan.read(count_u32);
        rec_scan.write(&offsets);
        rec_scan.write(&total_rows_device);
        rec_scan
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: scan preflight failed: {e}")))?;
        let total_rows = if grid <= 1024 {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_SCAN_HG_BLOCK_COUNTS_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_scan_hg_block_counts_u32 kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                count_u32.as_kernel_param(),
                grid.as_kernel_param(),
                (&offsets).as_kernel_param(),
                (&total_rows_device).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1024, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: HG block-count scan failed: {e}"))
                    })?;
            }
            rec_scan
                .commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: scan commit failed: {e}")))?;
            cu_stream
                .synchronize()
                .map_err(|e| XlogError::Kernel(format!("{ctx}: scan stream sync failed: {e}")))?;
            u64::from(self.dtoh_scalar_untracked::<u32>(&total_rows_device, 0)?)
        } else {
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *offsets.device_ptr(),
                    *count_u32.device_ptr(),
                    bytes_count,
                    cu_stream.cu_stream(),
                );
                if res != sys::cudaError_enum::CUDA_SUCCESS {
                    return Err(XlogError::Kernel(format!(
                        "{ctx}: DtoD count to offsets failed: {res:?}"
                    )));
                }
            }
            self.multiblock_scan_u32_inplace_on_stream(
                &mut offsets,
                grid,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            rec_scan
                .commit(runtime)
                .map_err(|e| XlogError::Kernel(format!("{ctx}: scan commit failed: {e}")))?;
            cu_stream
                .synchronize()
                .map_err(|e| XlogError::Kernel(format!("{ctx}: scan stream sync failed: {e}")))?;
            let last = grid - 1;
            u64::from(self.dtoh_scalar_untracked::<u32>(&offsets, last as usize)?)
                + u64::from(self.dtoh_scalar_untracked::<u32>(count_u32, last as usize)?)
        };
        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }
        let total_rows_u32 = u32::try_from(total_rows).map_err(|_| {
            XlogError::Kernel(format!("{ctx}: total rows {total_rows} exceed u32::MAX"))
        })?;

        let bytes_per_col = (total_rows_u32 as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: output byte size overflow")))?;
        let mut out_x = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory().alloc::<u8>(bytes_per_col)?;
        let out_d_num_rows = self.memory().alloc::<u32>(1)?;

        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(count_u32);
        rec_mat.read(&offsets);
        rec_mat.read(&plan.scratch_x);
        rec_mat.read(&plan.scratch_y);
        rec_mat.read(&plan.scratch_z);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat.write(&out_d_num_rows);
        rec_mat
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: materialize preflight failed: {e}")))?;
        unsafe {
            let res = sys::cuMemcpyHtoDAsync_v2(
                *out_d_num_rows.device_ptr(),
                &total_rows_u32 as *const u32 as *const c_void,
                std::mem::size_of::<u32>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: H2D output row count failed: {res:?}"
                )));
            }
        }
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_HG_CACHED_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_materialize_hg_cached_u32 kernel not found".to_string(),
                    )
                })?;
            let out_x_u32 = unsafe { reinterpret_u8_as_u32(&mut out_x) };
            let out_y_u32 = unsafe { reinterpret_u8_as_u32(&mut out_y) };
            let out_z_u32 = unsafe { reinterpret_u8_as_u32(&mut out_z) };
            let mut params: Vec<*mut c_void> = vec![
                count_u32.as_kernel_param(),
                (&offsets).as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                total_rows_u32.as_kernel_param(),
                (&plan.scratch_x).as_kernel_param(),
                (&plan.scratch_y).as_kernel_param(),
                (&plan.scratch_z).as_kernel_param(),
                out_x_u32.as_kernel_param(),
                out_y_u32.as_kernel_param(),
                out_z_u32.as_kernel_param(),
            ];
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
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: materialize launch failed: {e}"))
                    })?;
            }
        }
        rec_mat
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: materialize commit failed: {e}")))?;
        cu_stream.synchronize().map_err(|e| {
            XlogError::Kernel(format!("{ctx}: materialize stream sync failed: {e}"))
        })?;

        Ok(CudaBuffer::from_columns_with_host_count(
            vec![out_x.into(), out_y.into(), out_z.into()],
            total_rows_u32 as u64,
            out_d_num_rows,
            out_schema,
            total_rows_u32,
        ))
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

fn validate_binary_u32(ctx: &str, label: &str, input: &CudaBuffer) -> Result<()> {
    if input.arity() != 2 {
        return Err(XlogError::Kernel(format!(
            "{ctx}: {label} must be 2-column, got arity {}",
            input.arity()
        )));
    }
    for col_idx in 0..2 {
        let ty = input.schema().column_type(col_idx).ok_or_else(|| {
            XlogError::Kernel(format!("{ctx}: {label}.col{col_idx} type missing"))
        })?;
        if !matches!(ty, ScalarType::U32 | ScalarType::Symbol) {
            return Err(XlogError::Kernel(format!(
                "{ctx}: {label}.col{col_idx} must be U32 or Symbol, got {:?}",
                ty
            )));
        }
    }
    Ok(())
}

unsafe fn reinterpret_u8_as_u32(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u32> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u32>)
}
