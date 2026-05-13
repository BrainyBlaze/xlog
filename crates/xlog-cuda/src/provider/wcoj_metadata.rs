use std::ffi::c_void;

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{wcoj_kernels, CudaKernelProvider, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::wcoj_metadata::{
    WcojCycle4HgWorkPlanU32, WcojRelationMetadata, WcojTriangleHgWorkPlanU32,
    WcojTriangleHgWorkPlanU64,
};
use crate::{AsKernelParam, CudaBuffer, LaunchAsync, LaunchConfig};

const BLOCK_SIZE: u32 = 256;
const HG_COUNT_BLOCK_SIZE: u32 = 512;

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
            let block_counts = self.memory().alloc::<u32>(1)?;
            let block_offsets = self.memory().alloc::<u32>(1)?;
            let scratch_x = self.memory().alloc::<u32>(1)?;
            let scratch_y = self.memory().alloc::<u32>(1)?;
            let scratch_z = self.memory().alloc::<u32>(1)?;
            return Ok(WcojTriangleHgWorkPlanU32 {
                xy_work_prefix,
                xy_yz_start,
                xy_yz_end,
                xy_xz_start,
                xy_xz_end,
                block_counts,
                block_offsets,
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
        let grid = if total_work == 0 {
            1
        } else {
            total_work.div_ceil(block_work_unit)
        };
        let block_counts = self.memory().alloc::<u32>(grid as usize)?;
        let block_offsets = self.memory().alloc::<u32>(grid as usize)?;

        Ok(WcojTriangleHgWorkPlanU32 {
            xy_work_prefix,
            xy_yz_start,
            xy_yz_end,
            xy_xz_start,
            xy_xz_end,
            block_counts,
            block_offsets,
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
        let mut local_counts = None;
        let mut local_offsets = None;
        if grid > 1024 {
            local_counts = Some(self.memory().alloc::<u32>(grid as usize)?);
            local_offsets = Some(self.memory().alloc::<u32>(grid as usize)?);
        }
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

        let count_u32 = if grid <= 1024 {
            &plan.block_counts
        } else {
            local_counts
                .as_ref()
                .expect("local HG counts allocated when grid exceeds single-block scan")
        };
        let mut rec_hg = LaunchRecorder::new_strict(launch_stream);
        rec_hg.read(e_xy.num_rows_device());
        rec_hg.read(e_yz.num_rows_device());
        rec_hg.read(e_xz.num_rows_device());
        rec_hg.read_column(e_xy.column(0).expect("xy.col0"));
        rec_hg.read_column(e_xy.column(1).expect("xy.col1"));
        rec_hg.read_column(e_yz.column(1).expect("yz.col1"));
        rec_hg.read_column(e_xz.column(1).expect("xz.col1"));
        rec_hg.read(&plan.xy_work_prefix);
        rec_hg.read(&plan.xy_yz_start);
        rec_hg.read(&plan.xy_yz_end);
        rec_hg.read(&plan.xy_xz_start);
        rec_hg.read(&plan.xy_xz_end);
        rec_hg.read_write(count_u32);
        if grid <= 1024 {
            rec_hg.read_write(&plan.block_offsets);
        } else {
            rec_hg.read_write(
                local_offsets
                    .as_ref()
                    .expect("local HG offsets allocated when grid exceeds single-block scan"),
            );
        }
        rec_hg.write(&total_rows_device);
        rec_hg.read_write(&plan.scratch_x);
        rec_hg.read_write(&plan.scratch_y);
        rec_hg.read_write(&plan.scratch_z);
        rec_hg
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: HG preflight failed: {e}")))?;
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
        if grid <= 1024 {
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
                (&plan.block_offsets).as_kernel_param(),
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
        } else {
            let offsets_mut = local_offsets
                .as_mut()
                .expect("local HG offsets allocated when grid exceeds single-block scan");
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *offsets_mut.device_ptr(),
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
                offsets_mut,
                grid,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            let total_kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_COMPUTE_TOTAL)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_compute_total kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                count_u32.as_kernel_param(),
                (&*offsets_mut).as_kernel_param(),
                grid.as_kernel_param(),
                (&total_rows_device).as_kernel_param(),
            ];
            unsafe {
                total_kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: HG total reducer failed: {e}"))
                    })?;
            }
        }
        rec_hg
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: HG count commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count stream sync failed: {e}")))?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&total_rows_device, 0)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: read total rows failed: {e}")))?;
        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let bytes_per_col = (total_rows as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: output byte size overflow")))?;
        let mut out_x = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory().alloc::<u8>(bytes_per_col)?;
        let materialize_offsets = if grid <= 1024 {
            &plan.block_offsets
        } else {
            local_offsets
                .as_ref()
                .expect("local HG offsets allocated when grid exceeds single-block scan")
        };
        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(count_u32);
        rec_mat.read(materialize_offsets);
        rec_mat.read(&plan.scratch_x);
        rec_mat.read(&plan.scratch_y);
        rec_mat.read(&plan.scratch_z);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: materialize preflight failed: {e}")))?;
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
                materialize_offsets.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                total_rows.as_kernel_param(),
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
            total_rows as u64,
            total_rows_device,
            out_schema,
            total_rows,
        ))
    }

    pub fn wcoj_triangle_hg_work_plan_u64_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<WcojTriangleHgWorkPlanU64> {
        let ctx = "wcoj_triangle_hg_work_plan_u64_recorded";
        if block_work_unit == 0 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: block_work_unit must be nonzero"
            )));
        }
        validate_binary_u64(ctx, "e_xy", e_xy)?;
        validate_binary_u64(ctx, "e_yz", e_yz)?;
        validate_binary_u64(ctx, "e_xz", e_xz)?;

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
            let block_counts = self.memory().alloc::<u32>(1)?;
            let block_offsets = self.memory().alloc::<u32>(1)?;
            return Ok(WcojTriangleHgWorkPlanU64 {
                xy_work_prefix,
                xy_yz_start,
                xy_yz_end,
                xy_xz_start,
                xy_xz_end,
                block_counts,
                block_offsets,
                total_work: 0,
                block_work_unit,
                row_count: n_xy,
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

        let xy_col0 = metadata_column_u64(e_xy, 0)?;
        let xy_col1 = metadata_column_u64(e_xy, 1)?;
        let yz_col0 = metadata_column_u64(e_yz, 0)?;
        let xz_col0 = metadata_column_u64(e_xz, 0)?;

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
                wcoj_kernels::WCOJ_TRIANGLE_BUILD_HG_WORK_PLAN_U64,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_triangle_build_hg_work_plan_u64 kernel not found".to_string(),
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
                        "wcoj_triangle_build_hg_work_plan_u64 launch failed: {e}"
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
        let grid = if total_work == 0 {
            1
        } else {
            total_work.div_ceil(block_work_unit)
        };
        let block_counts = self.memory().alloc::<u32>(grid as usize)?;
        let block_offsets = self.memory().alloc::<u32>(grid as usize)?;

        Ok(WcojTriangleHgWorkPlanU64 {
            xy_work_prefix,
            xy_yz_start,
            xy_yz_end,
            xy_xz_start,
            xy_xz_end,
            block_counts,
            block_offsets,
            total_work,
            block_work_unit,
            row_count: n_xy,
        })
    }

    pub fn wcoj_triangle_hg_u64_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_hg_u64_recorded";
        validate_binary_u64(ctx, "e_xy", e_xy)?;
        validate_binary_u64(ctx, "e_yz", e_yz)?;
        validate_binary_u64(ctx, "e_xz", e_xz)?;
        let plan = self.wcoj_triangle_hg_work_plan_u64_recorded(
            e_xy,
            e_yz,
            e_xz,
            block_work_unit,
            launch_stream,
        )?;
        let out_schema = Schema::new(vec![
            ("col0".to_string(), ScalarType::U64),
            ("col1".to_string(), ScalarType::U64),
            ("col2".to_string(), ScalarType::U64),
        ]);
        if plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let bytes_count = (grid as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: count byte size overflow")))?;
        let mut local_counts = None;
        let mut local_offsets = None;
        if grid > 1024 {
            local_counts = Some(self.memory().alloc::<u32>(grid as usize)?);
            local_offsets = Some(self.memory().alloc::<u32>(grid as usize)?);
        }
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

        let xy_col0 = metadata_column_u64(e_xy, 0)?;
        let xy_col1 = metadata_column_u64(e_xy, 1)?;
        let yz_col1 = metadata_column_u64(e_yz, 1)?;
        let xz_col1 = metadata_column_u64(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;

        let count_u32 = if grid <= 1024 {
            &plan.block_counts
        } else {
            local_counts
                .as_ref()
                .expect("local HG counts allocated when grid exceeds single-block scan")
        };
        let mut rec_hg = LaunchRecorder::new_strict(launch_stream);
        rec_hg.read(e_xy.num_rows_device());
        rec_hg.read(e_yz.num_rows_device());
        rec_hg.read(e_xz.num_rows_device());
        rec_hg.read_column(e_yz.column(1).expect("yz.col1"));
        rec_hg.read_column(e_xz.column(1).expect("xz.col1"));
        rec_hg.read(&plan.xy_work_prefix);
        rec_hg.read(&plan.xy_yz_start);
        rec_hg.read(&plan.xy_yz_end);
        rec_hg.read(&plan.xy_xz_start);
        rec_hg.read(&plan.xy_xz_end);
        rec_hg.read_write(count_u32);
        if grid <= 1024 {
            rec_hg.read_write(&plan.block_offsets);
        } else {
            rec_hg.read_write(
                local_offsets
                    .as_ref()
                    .expect("local HG offsets allocated when grid exceeds single-block scan"),
            );
        }
        rec_hg.write(&total_rows_device);
        rec_hg
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: HG preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_U64)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_triangle_count_hg_u64 kernel not found".to_string())
                })?;
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
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: count launch failed: {e}")))?;
            }
        }
        if grid <= 1024 {
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
                (&plan.block_offsets).as_kernel_param(),
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
        } else {
            let offsets_mut = local_offsets
                .as_mut()
                .expect("local HG offsets allocated when grid exceeds single-block scan");
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *offsets_mut.device_ptr(),
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
                offsets_mut,
                grid,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            let total_kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_COMPUTE_TOTAL)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_compute_total kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                count_u32.as_kernel_param(),
                (&*offsets_mut).as_kernel_param(),
                grid.as_kernel_param(),
                (&total_rows_device).as_kernel_param(),
            ];
            unsafe {
                total_kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: HG total reducer failed: {e}"))
                    })?;
            }
        }
        rec_hg
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: HG count commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count stream sync failed: {e}")))?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&total_rows_device, 0)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: read total rows failed: {e}")))?;
        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let bytes_per_col = (total_rows as usize)
            .checked_mul(std::mem::size_of::<u64>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: output byte size overflow")))?;
        let mut out_x = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory().alloc::<u8>(bytes_per_col)?;
        let materialize_offsets = if grid <= 1024 {
            &plan.block_offsets
        } else {
            local_offsets
                .as_ref()
                .expect("local HG offsets allocated when grid exceeds single-block scan")
        };
        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(materialize_offsets);
        rec_mat.read(e_xy.num_rows_device());
        rec_mat.read(e_yz.num_rows_device());
        rec_mat.read(e_xz.num_rows_device());
        rec_mat.read_column(e_xy.column(0).expect("xy.col0"));
        rec_mat.read_column(e_xy.column(1).expect("xy.col1"));
        rec_mat.read_column(e_yz.column(1).expect("yz.col1"));
        rec_mat.read_column(e_xz.column(1).expect("xz.col1"));
        rec_mat.read(&plan.xy_work_prefix);
        rec_mat.read(&plan.xy_yz_start);
        rec_mat.read(&plan.xy_yz_end);
        rec_mat.read(&plan.xy_xz_start);
        rec_mat.read(&plan.xy_xz_end);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: materialize preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_HG_U64)
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_materialize_hg_u64 kernel not found".to_string(),
                    )
                })?;
            let out_x_u64 = unsafe { reinterpret_u8_as_u64(&mut out_x) };
            let out_y_u64 = unsafe { reinterpret_u8_as_u64(&mut out_y) };
            let out_z_u64 = unsafe { reinterpret_u8_as_u64(&mut out_z) };
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
                materialize_offsets.as_kernel_param(),
                total_rows.as_kernel_param(),
                out_x_u64.as_kernel_param(),
                out_y_u64.as_kernel_param(),
                out_z_u64.as_kernel_param(),
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
            total_rows as u64,
            total_rows_device,
            out_schema,
            total_rows,
        ))
    }

    pub fn wcoj_4cycle_hg_work_plan_u32_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<WcojCycle4HgWorkPlanU32> {
        let ctx = "wcoj_4cycle_hg_work_plan_u32_recorded";
        if block_work_unit == 0 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: block_work_unit must be nonzero"
            )));
        }
        validate_binary_u32(ctx, "e1", e1)?;
        validate_binary_u32(ctx, "e2", e2)?;
        validate_binary_u32(ctx, "e3", e3)?;
        validate_binary_u32(ctx, "e4", e4)?;

        let n_e1 = self.metadata_logical_rows(e1)?;
        let n_e2 = self.metadata_logical_rows(e2)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let prefix_len = n_e1
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: prefix length overflow")))?;
        let mut e1_work_prefix = self.memory().alloc::<u32>(prefix_len as usize)?;
        let mut e1_e2_start = self.memory().alloc::<u32>(n_e1 as usize)?;
        let mut e1_e2_end = self.memory().alloc::<u32>(n_e1 as usize)?;

        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || self.metadata_logical_rows(e4)? == 0 {
            let block_counts = self.memory().alloc::<u32>(1)?;
            let block_offsets = self.memory().alloc::<u32>(1)?;
            return Ok(WcojCycle4HgWorkPlanU32 {
                e1_work_prefix,
                e1_e2_start,
                e1_e2_end,
                block_counts,
                block_offsets,
                total_work: 0,
                block_work_unit,
                row_count: n_e1,
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

        let e1_col1 = metadata_column_u32(e1, 1)?;
        let e2_col0 = metadata_column_u32(e2, 0)?;
        let e2_col1 = metadata_column_u32(e2, 1)?;
        let e3_col0 = metadata_column_u32(e3, 0)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read_column(e1.column(1).expect("e1.col1"));
        rec.read_column(e2.column(0).expect("e2.col0"));
        rec.read_column(e2.column(1).expect("e2.col1"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.write(&e1_work_prefix);
        rec.write(&e1_e2_start);
        rec.write(&e1_e2_end);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;

        let kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_4CYCLE_BUILD_HG_WORK_PLAN_U32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_4cycle_build_hg_work_plan_u32 kernel not found".to_string())
            })?;
        let grid = n_e1.div_ceil(BLOCK_SIZE);
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
                        e1_col1,
                        n_e1,
                        e2_col0,
                        e2_col1,
                        n_e2,
                        e3_col0,
                        n_e3,
                        &mut e1_work_prefix,
                        &mut e1_e2_start,
                        &mut e1_e2_end,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_build_hg_work_plan_u32 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut e1_work_prefix,
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
        let total_work = self.dtoh_scalar_untracked::<u32>(&e1_work_prefix, n_e1 as usize)?;
        let grid = if total_work == 0 {
            1
        } else {
            total_work.div_ceil(block_work_unit)
        };
        let block_counts = self.memory().alloc::<u32>(grid as usize)?;
        let block_offsets = self.memory().alloc::<u32>(grid as usize)?;

        Ok(WcojCycle4HgWorkPlanU32 {
            e1_work_prefix,
            e1_e2_start,
            e1_e2_end,
            block_counts,
            block_offsets,
            total_work,
            block_work_unit,
            row_count: n_e1,
        })
    }

    pub fn wcoj_4cycle_hg_u32_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_4cycle_hg_u32_recorded";
        validate_binary_u32(ctx, "e1", e1)?;
        validate_binary_u32(ctx, "e2", e2)?;
        validate_binary_u32(ctx, "e3", e3)?;
        validate_binary_u32(ctx, "e4", e4)?;
        let plan = self.wcoj_4cycle_hg_work_plan_u32_recorded(
            e1,
            e2,
            e3,
            e4,
            block_work_unit,
            launch_stream,
        )?;
        let out_schema = Schema::new(vec![
            (
                "col0".to_string(),
                e1.schema().column_type(0).expect("e1.col0 type").clone(),
            ),
            (
                "col1".to_string(),
                e1.schema().column_type(1).expect("e1.col1 type").clone(),
            ),
            (
                "col2".to_string(),
                e2.schema().column_type(1).expect("e2.col1 type").clone(),
            ),
            (
                "col3".to_string(),
                e3.schema().column_type(1).expect("e3.col1 type").clone(),
            ),
        ]);
        if plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let bytes_count = (grid as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: count byte size overflow")))?;
        let mut local_counts = None;
        let mut local_offsets = None;
        if grid > 1024 {
            local_counts = Some(self.memory().alloc::<u32>(grid as usize)?);
            local_offsets = Some(self.memory().alloc::<u32>(grid as usize)?);
        }
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

        let e1_col0 = metadata_column_u32(e1, 0)?;
        let e1_col1 = metadata_column_u32(e1, 1)?;
        let e2_col1 = metadata_column_u32(e2, 1)?;
        let e3_col0 = metadata_column_u32(e3, 0)?;
        let e3_col1 = metadata_column_u32(e3, 1)?;
        let e4_col0 = metadata_column_u32(e4, 0)?;
        let e4_col1 = metadata_column_u32(e4, 1)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let n_e4 = self.metadata_logical_rows(e4)?;

        let count_u32 = if grid <= 1024 {
            &plan.block_counts
        } else {
            local_counts
                .as_ref()
                .expect("local HG counts allocated when grid exceeds single-block scan")
        };
        let mut rec_hg = LaunchRecorder::new_strict(launch_stream);
        rec_hg.read(e1.num_rows_device());
        rec_hg.read(e2.num_rows_device());
        rec_hg.read(e3.num_rows_device());
        rec_hg.read(e4.num_rows_device());
        rec_hg.read_column(e1.column(0).expect("e1.col0"));
        rec_hg.read_column(e1.column(1).expect("e1.col1"));
        rec_hg.read_column(e2.column(1).expect("e2.col1"));
        rec_hg.read_column(e3.column(0).expect("e3.col0"));
        rec_hg.read_column(e3.column(1).expect("e3.col1"));
        rec_hg.read_column(e4.column(0).expect("e4.col0"));
        rec_hg.read_column(e4.column(1).expect("e4.col1"));
        rec_hg.read(&plan.e1_work_prefix);
        rec_hg.read(&plan.e1_e2_start);
        rec_hg.read(&plan.e1_e2_end);
        rec_hg.read_write(count_u32);
        if grid <= 1024 {
            rec_hg.read_write(&plan.block_offsets);
        } else {
            rec_hg.read_write(
                local_offsets
                    .as_ref()
                    .expect("local HG offsets allocated when grid exceeds single-block scan"),
            );
        }
        rec_hg.write(&total_rows_device);
        rec_hg
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: HG preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_COUNT_HG_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_4cycle_count_hg_u32 kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                plan.row_count.as_kernel_param(),
                e2_col1.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&plan.e1_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
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
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: count launch failed: {e}")))?;
            }
        }
        if grid <= 1024 {
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
                (&plan.block_offsets).as_kernel_param(),
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
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: scan failed: {e}")))?;
            }
        } else {
            let offsets_mut = local_offsets
                .as_mut()
                .expect("local HG offsets allocated when grid exceeds single-block scan");
            unsafe {
                let res = sys::cuMemcpyDtoDAsync_v2(
                    *offsets_mut.device_ptr(),
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
                offsets_mut,
                grid,
                &cu_stream,
                launch_stream,
                runtime,
            )?;
            let total_kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_COMPUTE_TOTAL)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_compute_total kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                count_u32.as_kernel_param(),
                (&*offsets_mut).as_kernel_param(),
                grid.as_kernel_param(),
                (&total_rows_device).as_kernel_param(),
            ];
            unsafe {
                total_kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (1, 1, 1),
                            block_dim: (1, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: total failed: {e}")))?;
            }
        }
        rec_hg
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count stream sync failed: {e}")))?;
        let total_rows = self
            .dtoh_scalar_untracked::<u32>(&total_rows_device, 0)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: read total rows failed: {e}")))?;
        if total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }

        let bytes_per_col = (total_rows as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: output byte size overflow")))?;
        let mut out_w = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_x = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory().alloc::<u8>(bytes_per_col)?;
        let materialize_offsets = if grid <= 1024 {
            &plan.block_offsets
        } else {
            local_offsets
                .as_ref()
                .expect("local HG offsets allocated when grid exceeds single-block scan")
        };
        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
        rec_mat.read(materialize_offsets);
        rec_mat.read(e1.num_rows_device());
        rec_mat.read(e2.num_rows_device());
        rec_mat.read(e3.num_rows_device());
        rec_mat.read(e4.num_rows_device());
        rec_mat.read_column(e1.column(0).expect("e1.col0"));
        rec_mat.read_column(e1.column(1).expect("e1.col1"));
        rec_mat.read_column(e2.column(1).expect("e2.col1"));
        rec_mat.read_column(e3.column(0).expect("e3.col0"));
        rec_mat.read_column(e3.column(1).expect("e3.col1"));
        rec_mat.read_column(e4.column(0).expect("e4.col0"));
        rec_mat.read_column(e4.column(1).expect("e4.col1"));
        rec_mat.read(&plan.e1_work_prefix);
        rec_mat.read(&plan.e1_e2_start);
        rec_mat.read(&plan.e1_e2_end);
        rec_mat.write(&out_w);
        rec_mat.write(&out_x);
        rec_mat.write(&out_y);
        rec_mat.write(&out_z);
        rec_mat
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: materialize preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_MATERIALIZE_HG_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_4cycle_materialize_hg_u32 kernel not found".to_string())
                })?;
            let out_w_u32 = unsafe { reinterpret_u8_as_u32(&mut out_w) };
            let out_x_u32 = unsafe { reinterpret_u8_as_u32(&mut out_x) };
            let out_y_u32 = unsafe { reinterpret_u8_as_u32(&mut out_y) };
            let out_z_u32 = unsafe { reinterpret_u8_as_u32(&mut out_z) };
            let mut params: Vec<*mut c_void> = vec![
                e1_col0.as_kernel_param(),
                e1_col1.as_kernel_param(),
                plan.row_count.as_kernel_param(),
                e2_col1.as_kernel_param(),
                e3_col0.as_kernel_param(),
                e3_col1.as_kernel_param(),
                n_e3.as_kernel_param(),
                e4_col0.as_kernel_param(),
                e4_col1.as_kernel_param(),
                n_e4.as_kernel_param(),
                (&plan.e1_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                materialize_offsets.as_kernel_param(),
                total_rows.as_kernel_param(),
                out_w_u32.as_kernel_param(),
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
            vec![out_w.into(), out_x.into(), out_y.into(), out_z.into()],
            total_rows as u64,
            total_rows_device,
            out_schema,
            total_rows,
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

fn validate_binary_u64(ctx: &str, label: &str, input: &CudaBuffer) -> Result<()> {
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
        if !matches!(ty, ScalarType::U64) {
            return Err(XlogError::Kernel(format!(
                "{ctx}: {label}.col{col_idx} must be U64, got {:?}",
                ty
            )));
        }
    }
    Ok(())
}

unsafe fn reinterpret_u8_as_u32(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u32> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u32>)
}

unsafe fn reinterpret_u8_as_u64(slice: &mut TrackedCudaSlice<u8>) -> &mut TrackedCudaSlice<u64> {
    &mut *(slice as *mut TrackedCudaSlice<u8> as *mut TrackedCudaSlice<u64>)
}
