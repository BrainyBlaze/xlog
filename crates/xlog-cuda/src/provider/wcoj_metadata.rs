use std::collections::BTreeMap;
use std::ffi::c_void;

use cudarc::driver::sys;
use xlog_core::{AggOp, Result, ScalarType, Schema, XlogError};

use super::{arith_kernels, wcoj_kernels, CudaKernelProvider, ARITH_MODULE, WCOJ_MODULE};
use crate::device_runtime::StreamId;
use crate::launch::LaunchRecorder;
use crate::memory::{CudaColumn, TrackedCudaSlice};
use crate::wcoj_metadata::{
    Wcoj4CycleRootAggValue, WcojCycle4HgWorkPlanU32, WcojCycle4HgWorkPlanU64,
    WcojRelationMetadata, WcojRootAggValue, WcojTriangleHgCountPhaseU32,
    WcojTriangleHgWorkPlanU32, WcojTriangleHgWorkPlanU64,
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

        self.htod_launch_metadata_async_copy_one(
            &grid,
            &d_num_rows,
            &cu_stream,
            &format!("{ctx}: d_num_rows"),
        )?;

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

    /// D1 aggregate-fused triangle: evaluate
    /// `q(X, count) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` grouped by the
    /// variable-order root X, WITHOUT materializing the triangle rows.
    ///
    /// Pipeline (all recorded; the triangle result never exists as rows):
    /// 1. the standard histogram-guided work plan;
    /// 2. `wcoj_triangle_groupby_root_count_hg_u32` accumulates per-e_xy-row
    ///    match counts (integer atomicAdd — order-insensitive, deterministic
    ///    values) into a zero-initialized `n_xy`-long array;
    /// 3. a 2-column (X, count) staging buffer over the *input* rows is
    ///    compacted to count>0 rows (group-by over the join result must not
    ///    emit roots with no completion) and reduced per X via the recorded
    ///    groupby Sum (rows are already X-sorted because e_xy is lex-sorted).
    ///
    /// All reduction work is O(n_xy) — input-sized, never join-output-sized.
    ///
    /// Output schema matches the unfused materialize+groupby-count baseline:
    /// `col0` = X (e_xy.col0 type, U32/Symbol), `col1` = count (U64).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U32/Symbol, or
    ///   any kernel launch fails.
    pub fn wcoj_triangle_groupby_root_count_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_groupby_root_count_u32_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e_xy = &self.wcoj_layout_u32_recorded(e_xy, launch_stream)?;
        let e_yz = &self.wcoj_layout_u32_recorded(e_yz, launch_stream)?;
        let e_xz = &self.wcoj_layout_u32_recorded(e_xz, launch_stream)?;
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
        let n_xy = plan.row_count;
        let x_type = e_xy.schema().column_type(0).expect("xy.col0 type");
        let out_schema = Schema::new(vec![
            ("x".to_string(), x_type),
            ("count".to_string(), ScalarType::U64),
        ]);
        if n_xy == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;

        // Per-e_xy-row match counters, zero-initialized. Allocated as the
        // u8-backed column layout so the array doubles as the staging
        // buffer's count column after the kernel fills it.
        let mut row_counts = self
            .memory()
            .alloc::<u8>(n_xy as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_yz.column(1).expect("yz.col1"));
        rec.read_column(e_xz.column(1).expect("xz.col1"));
        rec.read(&plan.xy_work_prefix);
        rec.read(&plan.xy_yz_start);
        rec.read(&plan.xy_yz_end);
        rec.read(&plan.xy_xz_start);
        rec.read(&plan.xy_xz_end);
        rec.write(&row_counts);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_COUNT_HG_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_groupby_root_count_hg_u32 kernel not found".to_string(),
                    )
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
                (&row_counts).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-count launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Staging buffer (X, count) over the n_xy input rows: X is a
        // device-to-device copy of e_xy.col0; the count column is the
        // kernel-filled array. Rows stay lex-sorted by X.
        let x_src = match e_xy.column(0).expect("xy.col0") {
            CudaColumn::Owned(slice) => slice,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: e_xy.col0 must be an owned CudaColumn"
                )))
            }
        };
        let x_copy = self
            .memory()
            .alloc::<u8>(n_xy as usize * std::mem::size_of::<u32>())?;
        // Explicit-length copy: layout-normalized columns are allocated at
        // capacity, which can exceed the logical n_xy * 4 bytes a full-slice
        // typed copy would assert on.
        unsafe {
            let res = sys::cuMemcpyDtoD_v2(
                *x_copy.device_ptr(),
                *x_src.device_ptr(),
                n_xy as usize * std::mem::size_of::<u32>(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: copy X column failed: {res:?}"
                )));
            }
        }
        let mut d_num_rows = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .dtod_copy(e_xy.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy row count failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("x".to_string(), x_type),
            ("count".to_string(), ScalarType::U32),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![x_copy.into(), row_counts.into()],
            n_xy as u64,
            d_num_rows,
            staging_schema,
            n_xy,
        );

        // Keep only roots with at least one completed triangle, then reduce
        // per X. Both steps run over input-sized data.
        let mask = self.compare_const_mask_recorded::<u32>(
            &staging,
            1,
            0u32,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        let compacted =
            self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)?;
        self.groupby_multi_agg_recorded(
            &compacted,
            &[0],
            &[(1, xlog_core::AggOp::Sum)],
            launch_stream,
        )
    }

    /// D1 widening — aggregate-fused triangle sum/min/max: evaluate
    /// `q(X, agg(V)) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` with
    /// `agg ∈ {Sum, Min, Max}` and `V ∈ {Y, Z}` grouped by the
    /// variable-order root X, WITHOUT materializing the triangle rows.
    ///
    /// Pipeline (all recorded; the triangle result never exists as rows):
    /// 1. the standard histogram-guided work plan;
    /// 2. the per-op fused kernel accumulates, per e_xy row, a match count
    ///    (compaction mask) and the per-row partial aggregate (integer
    ///    atomics — order-insensitive, deterministic values). Sum partials
    ///    are u64 (a per-row partial can exceed `u32::MAX`); min partials
    ///    start at `u32::MAX`, max partials at 0;
    /// 3. a 3-column (X, count, agg) staging buffer over the *input* rows
    ///    is compacted to count>0 rows (groups with no completion must be
    ///    absent) and reduced per X via the recorded groupby with the same
    ///    `AggOp` (Sum over the u64 partials; Min/Max over u32).
    ///
    /// All reduction work is O(n_xy) — input-sized, never join-output-sized.
    ///
    /// Output schema matches the unfused materialize+groupby baseline:
    /// `col0` = X (e_xy.col0 type, U32/Symbol), `col1` = U64 for Sum,
    /// U32 for Min/Max.
    ///
    /// Bag semantics: every (Y, Z) completion contributes its value,
    /// exactly like aggregating the materialized projection.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if `agg_op` is not Sum/Min/Max, the value
    ///   columns are not plain U32, the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U32/Symbol, or
    ///   any kernel launch fails.
    pub fn wcoj_triangle_groupby_root_agg_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        agg_op: AggOp,
        value: WcojRootAggValue,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_groupby_root_agg_u32_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e_xy = &self.wcoj_layout_u32_recorded(e_xy, launch_stream)?;
        let e_yz = &self.wcoj_layout_u32_recorded(e_yz, launch_stream)?;
        let e_xz = &self.wcoj_layout_u32_recorded(e_xz, launch_stream)?;
        let (kernel_name, agg_elem_size, agg_scalar, agg_name) = match agg_op {
            AggOp::Sum => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_SUM_HG_U32,
                std::mem::size_of::<u64>(),
                ScalarType::U64,
                "sum_0",
            ),
            AggOp::Min => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_MIN_HG_U32,
                std::mem::size_of::<u32>(),
                ScalarType::U32,
                "min_0",
            ),
            AggOp::Max => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_MAX_HG_U32,
                std::mem::size_of::<u32>(),
                ScalarType::U32,
                "max_0",
            ),
            other => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: unsupported AggOp {other:?} (Sum/Min/Max only; use \
                     wcoj_triangle_groupby_root_count_u32_recorded for Count)"
                )))
            }
        };
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        // The aggregate value is arithmetic: require plain U32 value
        // columns (Symbol ids are not summable/orderable data).
        let value_cols: &[(&CudaBuffer, &str)] = match value {
            WcojRootAggValue::Y => &[(e_xy, "e_xy")],
            WcojRootAggValue::Z => &[(e_yz, "e_yz"), (e_xz, "e_xz")],
        };
        for (buf, label) in value_cols {
            let ty = buf.schema().column_type(1).expect("validated 2-col");
            if ty != ScalarType::U32 {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: {label}.col1 supplies the aggregate value and must be U32, got {ty:?}"
                )));
            }
        }

        let plan = self.wcoj_triangle_hg_work_plan_u32_recorded(
            e_xy,
            e_yz,
            e_xz,
            block_work_unit,
            launch_stream,
        )?;
        let n_xy = plan.row_count;
        let x_type = e_xy.schema().column_type(0).expect("xy.col0 type");
        let out_schema = Schema::new(vec![
            ("x".to_string(), x_type),
            (agg_name.to_string(), agg_scalar),
        ]);
        if n_xy == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;
        let xy_col1 = metadata_column_u32(e_xy, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;
        let value_from_z: u32 = match value {
            WcojRootAggValue::Y => 0,
            WcojRootAggValue::Z => 1,
        };

        // Per-e_xy-row match counters + aggregate partials, allocated as
        // the u8-backed column layout so the arrays double as the staging
        // buffer's columns after the kernel fills them.
        let mut row_counts = self
            .memory()
            .alloc::<u8>(n_xy as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;
        let mut row_agg = self.memory().alloc::<u8>(n_xy as usize * agg_elem_size)?;
        self.device()
            .inner()
            .memset_zeros(&mut row_agg)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row aggregates failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_yz.column(1).expect("yz.col1"));
        rec.read_column(e_xz.column(1).expect("xz.col1"));
        rec.read_column(e_xy.column(1).expect("xy.col1"));
        rec.read(&plan.xy_work_prefix);
        rec.read(&plan.xy_yz_start);
        rec.read(&plan.xy_yz_end);
        rec.read(&plan.xy_xz_start);
        rec.read(&plan.xy_xz_end);
        rec.write(&row_counts);
        rec.write(&row_agg);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        if matches!(agg_op, AggOp::Min) {
            // Min identity: u32::MAX (compaction drops untouched rows).
            let fill = self
                .device()
                .inner()
                .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("arith_fill_const_u32 kernel not found".to_string())
                })?;
            let row_agg_u32 = unsafe { reinterpret_u8_as_u32(&mut row_agg) };
            // SAFETY: arith_fill_const_u32(value, n, output)
            unsafe {
                fill.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig::for_num_elems(n_xy),
                        (u32::MAX, n_xy, &mut *row_agg_u32),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: min identity fill failed: {e}"))
                    })?;
            }
        }
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, kernel_name)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("{kernel_name} kernel not found"))
                })?;
            let mut params: Vec<*mut c_void> = vec![
                yz_col1.as_kernel_param(),
                n_yz.as_kernel_param(),
                xz_col1.as_kernel_param(),
                n_xz.as_kernel_param(),
                xy_col1.as_kernel_param(),
                value_from_z.as_kernel_param(),
                (&plan.xy_work_prefix).as_kernel_param(),
                (&plan.xy_yz_start).as_kernel_param(),
                (&plan.xy_yz_end).as_kernel_param(),
                (&plan.xy_xz_start).as_kernel_param(),
                (&plan.xy_xz_end).as_kernel_param(),
                plan.row_count.as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
                (&row_agg).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-agg launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Staging buffer (X, count, agg) over the n_xy input rows: X is a
        // device-to-device copy of e_xy.col0; count and agg are the
        // kernel-filled arrays. Rows stay lex-sorted by X.
        let x_src = match e_xy.column(0).expect("xy.col0") {
            CudaColumn::Owned(slice) => slice,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: e_xy.col0 must be an owned CudaColumn"
                )))
            }
        };
        let x_copy = self
            .memory()
            .alloc::<u8>(n_xy as usize * std::mem::size_of::<u32>())?;
        // Explicit-length copy: layout-normalized columns are allocated at
        // capacity, which can exceed the logical n_xy * 4 bytes a full-slice
        // typed copy would assert on.
        unsafe {
            let res = sys::cuMemcpyDtoD_v2(
                *x_copy.device_ptr(),
                *x_src.device_ptr(),
                n_xy as usize * std::mem::size_of::<u32>(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: copy X column failed: {res:?}"
                )));
            }
        }
        let mut d_num_rows = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .dtod_copy(e_xy.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy row count failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("x".to_string(), x_type),
            ("count".to_string(), ScalarType::U32),
            ("agg".to_string(), agg_scalar),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![x_copy.into(), row_counts.into(), row_agg.into()],
            n_xy as u64,
            d_num_rows,
            staging_schema,
            n_xy,
        );

        // Keep only roots with at least one completed triangle, then reduce
        // per X with the same AggOp. Both steps run over input-sized data.
        let mask = self.compare_const_mask_recorded::<u32>(
            &staging,
            1,
            0u32,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        let compacted =
            self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)?;
        self.groupby_multi_agg_recorded(&compacted, &[0], &[(2, agg_op)], launch_stream)
    }

    /// D1 widening — u64-key sibling of
    /// [`Self::wcoj_triangle_groupby_root_count_u32_recorded`]: evaluate
    /// `q(X, count)` over the triangle shape grouped by the root X for U64
    /// relations, WITHOUT materializing the triangle rows.
    ///
    /// The recorded groupby is U32/Symbol-key only, so the per-X reduction
    /// reuses the WCOJ relation metadata instead: e_xy is lex-sorted, so
    /// `wcoj_build_metadata_u64_recorded` yields one (unique X, group start)
    /// pair per root, and `wcoj_groupby_root_segment_sum_counts_u32`
    /// accumulates the per-row match counts into per-unique-root u64
    /// totals (integer atomicAdd — deterministic). Roots with zero
    /// completions are compacted away. All reduction work is O(n_xy).
    ///
    /// Output schema matches the unfused materialize+groupby baseline:
    /// `col0` = X (U64), `col1` = count (U64).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U64, or any
    ///   kernel launch fails.
    pub fn wcoj_triangle_groupby_root_count_u64_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_groupby_root_count_u64_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e_xy = &self.wcoj_layout_u64_recorded(e_xy, launch_stream)?;
        let e_yz = &self.wcoj_layout_u64_recorded(e_yz, launch_stream)?;
        let e_xz = &self.wcoj_layout_u64_recorded(e_xz, launch_stream)?;
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
        let n_xy = plan.row_count;
        let out_schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U64),
            ("count".to_string(), ScalarType::U64),
        ]);
        if n_xy == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let yz_col1 = metadata_column_u64(e_yz, 1)?;
        let xz_col1 = metadata_column_u64(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;

        // Per-e_xy-row match counters, zero-initialized.
        let mut row_counts = self.memory().alloc::<u32>(n_xy as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_yz.column(1).expect("yz.col1"));
        rec.read_column(e_xz.column(1).expect("xz.col1"));
        rec.read(&plan.xy_work_prefix);
        rec.read(&plan.xy_yz_start);
        rec.read(&plan.xy_yz_end);
        rec.read(&plan.xy_xz_start);
        rec.read(&plan.xy_xz_end);
        rec.write(&row_counts);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_COUNT_HG_U64,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_groupby_root_count_hg_u64 kernel not found".to_string(),
                    )
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
                (&row_counts).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-count launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Per-X reduction via the relation metadata: one (unique X, group
        // start) pair per root; e_xy is lex-sorted by X so group rows are
        // contiguous.
        let meta = self.wcoj_build_metadata_u64_recorded(e_xy, 0, launch_stream)?;
        let key_count = meta.key_count;
        if key_count == 0 {
            return self.create_empty_buffer(out_schema);
        }
        let mut sums = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        self.device()
            .inner()
            .memset_zeros(&mut sums)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero group sums failed: {e}")))?;

        let mut rec_sum = LaunchRecorder::new_strict(launch_stream);
        rec_sum.read(&row_counts);
        rec_sum.read(&meta.prefix_sum);
        rec_sum.write(&sums);
        rec_sum
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_SUM_COUNTS_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_groupby_root_segment_sum_counts_u32 kernel not found".to_string(),
                    )
                })?;
            let reduce_grid = n_xy.div_ceil(BLOCK_SIZE);
            let mut params: Vec<*mut c_void> = vec![
                (&row_counts).as_kernel_param(),
                n_xy.as_kernel_param(),
                (&meta.prefix_sum).as_kernel_param(),
                key_count.as_kernel_param(),
                (&sums).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (reduce_grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce launch failed: {e}")))?;
            }
        }
        rec_sum
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce commit failed: {e}")))?;

        // (unique X, total) buffer over the key_count roots, then drop the
        // roots with no completion. The copies run on launch_stream and the
        // fresh destination blocks are registered through the strict
        // recorder BEFORE the enqueue — a raw async copy into a freshly
        // pool-allocated block without recording is a visibility race.
        let x_copy = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        let d_num_rows = self.memory().alloc::<u32>(1)?;
        let mut rec_copy = LaunchRecorder::new_strict(launch_stream);
        rec_copy.read(&meta.unique_keys);
        rec_copy.write(&x_copy);
        rec_copy.write(&d_num_rows);
        rec_copy
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy preflight failed: {e}")))?;
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *x_copy.device_ptr(),
                *meta.unique_keys.device_ptr(),
                key_count as usize * std::mem::size_of::<u64>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: DtoD unique keys copy failed: {res:?}"
                )));
            }
        }
        self.htod_launch_metadata_async_copy_one(
            &key_count,
            &d_num_rows,
            &cu_stream,
            &format!("{ctx}: d_num_rows"),
        )?;
        rec_copy
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: stream sync failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U64),
            ("count".to_string(), ScalarType::U64),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![x_copy.into(), sums.into()],
            u64::from(key_count),
            d_num_rows,
            staging_schema,
            key_count,
        );
        let mask = self.compare_const_mask_recorded::<u64>(
            &staging,
            1,
            0u64,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)
    }

    /// S1c widening — u64-key sibling of
    /// [`Self::wcoj_triangle_groupby_root_agg_u32_recorded`]: evaluate
    /// `q(X, agg(V)) :- e_xy(X,Y), e_yz(Y,Z), e_xz(X,Z)` with
    /// `agg ∈ {Sum, Min, Max}` and `V ∈ {Y, Z}` over U64 relations,
    /// grouped by the variable-order root X, WITHOUT materializing the
    /// triangle rows.
    ///
    /// The recorded groupby is U32/Symbol-key only, so the per-X reduction
    /// reuses the WCOJ relation metadata (one unique root per group, e_xy
    /// lex-sorted) like the u64 count path:
    /// 1. the per-op fused kernel accumulates, per e_xy row, a match count
    ///    and a u64 aggregate partial (integer atomics — deterministic;
    ///    sum wraps on overflow exactly like `groupby_sum_u64`; min
    ///    partials start at `u64::MAX`, max partials at 0);
    /// 2. `wcoj_groupby_root_segment_sum_counts_u32` reduces per-row match
    ///    counts to per-unique-root totals (the presence mask), and the
    ///    per-op `wcoj_groupby_root_segment_{sum,min,max}_values_u64`
    ///    kernel folds the per-row partials into per-unique-root u64
    ///    aggregates, skipping zero-match rows;
    /// 3. a (X, agg) staging buffer over the unique roots is compacted to
    ///    count>0 groups.
    ///
    /// All reduction work is O(n_xy) — input-sized, never join-output-sized.
    ///
    /// Output schema matches the unfused materialize+groupby baseline
    /// (legacy groupby widened to u64 values): `col0` = X (U64),
    /// `col1` = U64 for sum, min and max alike.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if `agg_op` is not Sum/Min/Max, the manager
    ///   has no runtime, the launch stream does not resolve, an input is
    ///   not 2-column U64, or any kernel launch fails.
    pub fn wcoj_triangle_groupby_root_agg_u64_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        agg_op: AggOp,
        value: WcojRootAggValue,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_groupby_root_agg_u64_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e_xy = &self.wcoj_layout_u64_recorded(e_xy, launch_stream)?;
        let e_yz = &self.wcoj_layout_u64_recorded(e_yz, launch_stream)?;
        let e_xz = &self.wcoj_layout_u64_recorded(e_xz, launch_stream)?;
        let (kernel_name, segment_kernel_name, agg_name) = match agg_op {
            AggOp::Sum => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_SUM_HG_U64,
                wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_SUM_VALUES_U64,
                "sum_0",
            ),
            AggOp::Min => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_MIN_HG_U64,
                wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_MIN_VALUES_U64,
                "min_0",
            ),
            AggOp::Max => (
                wcoj_kernels::WCOJ_TRIANGLE_GROUPBY_ROOT_MAX_HG_U64,
                wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_MAX_VALUES_U64,
                "max_0",
            ),
            other => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: unsupported AggOp {other:?} (Sum/Min/Max only; use \
                     wcoj_triangle_groupby_root_count_u64_recorded for Count)"
                )))
            }
        };
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
        let n_xy = plan.row_count;
        let out_schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U64),
            (agg_name.to_string(), ScalarType::U64),
        ]);
        if n_xy == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let yz_col1 = metadata_column_u64(e_yz, 1)?;
        let xz_col1 = metadata_column_u64(e_xz, 1)?;
        let xy_col1 = metadata_column_u64(e_xy, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;
        let value_from_z: u32 = match value {
            WcojRootAggValue::Y => 0,
            WcojRootAggValue::Z => 1,
        };

        // Per-e_xy-row match counters + u64 aggregate partials.
        let mut row_counts = self.memory().alloc::<u32>(n_xy as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;
        let mut row_agg = self
            .memory()
            .alloc::<u8>(n_xy as usize * std::mem::size_of::<u64>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_agg)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row aggregates failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e_xy.num_rows_device());
        rec.read(e_yz.num_rows_device());
        rec.read(e_xz.num_rows_device());
        rec.read_column(e_yz.column(1).expect("yz.col1"));
        rec.read_column(e_xz.column(1).expect("xz.col1"));
        rec.read_column(e_xy.column(1).expect("xy.col1"));
        rec.read(&plan.xy_work_prefix);
        rec.read(&plan.xy_yz_start);
        rec.read(&plan.xy_yz_end);
        rec.read(&plan.xy_xz_start);
        rec.read(&plan.xy_xz_end);
        rec.write(&row_counts);
        rec.write(&row_agg);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        if matches!(agg_op, AggOp::Min) {
            // Min identity: u64::MAX (compaction drops untouched groups).
            let fill = self
                .device()
                .inner()
                .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U64)
                .ok_or_else(|| {
                    XlogError::Kernel("arith_fill_const_u64 kernel not found".to_string())
                })?;
            let row_agg_u64 = unsafe { reinterpret_u8_as_u64(&mut row_agg) };
            // SAFETY: arith_fill_const_u64(value, n, output)
            unsafe {
                fill.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig::for_num_elems(n_xy),
                        (u64::MAX, n_xy, &mut *row_agg_u64),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: min identity fill failed: {e}"))
                    })?;
            }
        }
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, kernel_name)
                .ok_or_else(|| XlogError::Kernel(format!("{kernel_name} kernel not found")))?;
            let mut params: Vec<*mut c_void> = vec![
                yz_col1.as_kernel_param(),
                n_yz.as_kernel_param(),
                xz_col1.as_kernel_param(),
                n_xz.as_kernel_param(),
                xy_col1.as_kernel_param(),
                value_from_z.as_kernel_param(),
                (&plan.xy_work_prefix).as_kernel_param(),
                (&plan.xy_yz_start).as_kernel_param(),
                (&plan.xy_yz_end).as_kernel_param(),
                (&plan.xy_xz_start).as_kernel_param(),
                (&plan.xy_xz_end).as_kernel_param(),
                plan.row_count.as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
                (&row_agg).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-agg launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Per-X reduction via the relation metadata: one (unique X, group
        // start) pair per root; e_xy is lex-sorted by X so group rows are
        // contiguous.
        let meta = self.wcoj_build_metadata_u64_recorded(e_xy, 0, launch_stream)?;
        let key_count = meta.key_count;
        if key_count == 0 {
            return self.create_empty_buffer(out_schema);
        }
        let mut count_sums = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        self.device()
            .inner()
            .memset_zeros(&mut count_sums)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero group counts failed: {e}")))?;
        let mut group_agg = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        self.device()
            .inner()
            .memset_zeros(&mut group_agg)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero group aggregates failed: {e}")))?;

        let mut rec_reduce = LaunchRecorder::new_strict(launch_stream);
        rec_reduce.read(&row_counts);
        rec_reduce.read(&row_agg);
        rec_reduce.read(&meta.prefix_sum);
        rec_reduce.write(&count_sums);
        rec_reduce.write(&group_agg);
        rec_reduce
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce preflight failed: {e}")))?;
        if matches!(agg_op, AggOp::Min) {
            let fill = self
                .device()
                .inner()
                .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U64)
                .ok_or_else(|| {
                    XlogError::Kernel("arith_fill_const_u64 kernel not found".to_string())
                })?;
            let group_agg_u64 = unsafe { reinterpret_u8_as_u64(&mut group_agg) };
            // SAFETY: arith_fill_const_u64(value, n, output)
            unsafe {
                fill.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig::for_num_elems(key_count),
                        (u64::MAX, key_count, &mut *group_agg_u64),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: group min identity fill failed: {e}"))
                    })?;
            }
        }
        let reduce_grid = n_xy.div_ceil(BLOCK_SIZE);
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_SUM_COUNTS_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_groupby_root_segment_sum_counts_u32 kernel not found".to_string(),
                    )
                })?;
            let mut params: Vec<*mut c_void> = vec![
                (&row_counts).as_kernel_param(),
                n_xy.as_kernel_param(),
                (&meta.prefix_sum).as_kernel_param(),
                key_count.as_kernel_param(),
                (&count_sums).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (reduce_grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: count reduce launch failed: {e}"))
                    })?;
            }
        }
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, segment_kernel_name)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("{segment_kernel_name} kernel not found"))
                })?;
            let mut params: Vec<*mut c_void> = vec![
                (&row_counts).as_kernel_param(),
                (&row_agg).as_kernel_param(),
                n_xy.as_kernel_param(),
                (&meta.prefix_sum).as_kernel_param(),
                key_count.as_kernel_param(),
                (&group_agg).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (reduce_grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: agg reduce launch failed: {e}"))
                    })?;
            }
        }
        rec_reduce
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce commit failed: {e}")))?;

        // (unique X, agg) staging plus a counts-only buffer whose mask
        // drops groups with no completion. Fresh destination blocks are
        // registered through the strict recorder BEFORE the enqueue (a raw
        // async copy into a freshly pool-allocated block without recording
        // is a visibility race).
        let x_copy = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        let d_num_rows_agg = self.memory().alloc::<u32>(1)?;
        let d_num_rows_counts = self.memory().alloc::<u32>(1)?;
        let mut rec_copy = LaunchRecorder::new_strict(launch_stream);
        rec_copy.read(&meta.unique_keys);
        rec_copy.write(&x_copy);
        rec_copy.write(&d_num_rows_agg);
        rec_copy.write(&d_num_rows_counts);
        rec_copy
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy preflight failed: {e}")))?;
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *x_copy.device_ptr(),
                *meta.unique_keys.device_ptr(),
                key_count as usize * std::mem::size_of::<u64>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: DtoD unique keys copy failed: {res:?}"
                )));
            }
        }
        self.htod_launch_metadata_async_copy_one(
            &key_count,
            &d_num_rows_agg,
            &cu_stream,
            &format!("{ctx}: d_num_rows_agg"),
        )?;
        self.htod_launch_metadata_async_copy_one(
            &key_count,
            &d_num_rows_counts,
            &cu_stream,
            &format!("{ctx}: d_num_rows_counts"),
        )?;
        rec_copy
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: stream sync failed: {e}")))?;

        let counts_schema = Schema::new(vec![("count".to_string(), ScalarType::U64)]);
        let counts_buf = CudaBuffer::from_columns_with_host_count(
            vec![count_sums.into()],
            u64::from(key_count),
            d_num_rows_counts,
            counts_schema,
            key_count,
        );
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![x_copy.into(), group_agg.into()],
            u64::from(key_count),
            d_num_rows_agg,
            out_schema,
            key_count,
        );
        let mask = self.compare_const_mask_recorded::<u64>(
            &counts_buf,
            0,
            0u64,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)
    }

    pub fn wcoj_triangle_hg_count_phase_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        plan: &WcojTriangleHgWorkPlanU32,
        launch_stream: StreamId,
    ) -> Result<WcojTriangleHgCountPhaseU32> {
        let ctx = "wcoj_triangle_hg_count_phase_u32_recorded";
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        if grid > 1024 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: spike phase path requires grid <= 1024, got {grid}"
            )));
        }
        let total_rows_device = self.memory().alloc::<u32>(1)?;
        if plan.total_work == 0 {
            return Ok(WcojTriangleHgCountPhaseU32 {
                total_rows_device,
                total_rows: 0,
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

        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;

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
        rec_hg.read_write(&plan.block_counts);
        rec_hg.read_write(&plan.block_offsets);
        rec_hg.write(&total_rows_device);
        rec_hg
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: count preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_COUNT_HG_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_triangle_count_hg_u32 kernel not found".to_string())
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
                (&plan.block_counts).as_kernel_param(),
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
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_SCAN_HG_BLOCK_COUNTS_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_scan_hg_block_counts_u32 kernel not found".to_string())
                })?;
            let mut params: Vec<*mut c_void> = vec![
                (&plan.block_counts).as_kernel_param(),
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
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: scan launch failed: {e}")))?;
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
        Ok(WcojTriangleHgCountPhaseU32 {
            total_rows_device,
            total_rows,
        })
    }

    pub fn wcoj_triangle_hg_materialize_phase_u32_recorded(
        &self,
        e_xy: &CudaBuffer,
        e_yz: &CudaBuffer,
        e_xz: &CudaBuffer,
        plan: &WcojTriangleHgWorkPlanU32,
        count: WcojTriangleHgCountPhaseU32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_triangle_hg_materialize_phase_u32_recorded";
        validate_binary_u32(ctx, "e_xy", e_xy)?;
        validate_binary_u32(ctx, "e_yz", e_yz)?;
        validate_binary_u32(ctx, "e_xz", e_xz)?;
        let out_schema = Schema::new(vec![
            (
                "x".to_string(),
                e_xy.schema().column_type(0).expect("xy.col0 type"),
            ),
            (
                "y".to_string(),
                e_xy.schema().column_type(1).expect("xy.col1 type"),
            ),
            (
                "z".to_string(),
                e_yz.schema().column_type(1).expect("yz.col1 type"),
            ),
        ]);
        if count.total_rows == 0 {
            return self.create_empty_buffer(out_schema);
        }
        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        if grid > 1024 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: spike phase path requires grid <= 1024, got {grid}"
            )));
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
        let yz_col1 = metadata_column_u32(e_yz, 1)?;
        let xz_col1 = metadata_column_u32(e_xz, 1)?;
        let n_yz = self.metadata_logical_rows(e_yz)?;
        let n_xz = self.metadata_logical_rows(e_xz)?;
        let bytes_per_col = (count.total_rows as usize)
            .checked_mul(std::mem::size_of::<u32>())
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: output byte size overflow")))?;
        let mut out_x = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_y = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut out_z = self.memory().alloc::<u8>(bytes_per_col)?;
        let mut rec_mat = LaunchRecorder::new_strict(launch_stream);
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
        rec_mat.read(&plan.block_offsets);
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
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_TRIANGLE_MATERIALIZE_HG_U32)
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_triangle_materialize_hg_u32 kernel not found".to_string(),
                    )
                })?;
            let out_x_u32 = unsafe { reinterpret_u8_as_u32(&mut out_x) };
            let out_y_u32 = unsafe { reinterpret_u8_as_u32(&mut out_y) };
            let out_z_u32 = unsafe { reinterpret_u8_as_u32(&mut out_z) };
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
                (&plan.block_offsets).as_kernel_param(),
                count.total_rows.as_kernel_param(),
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
            count.total_rows as u64,
            count.total_rows_device,
            out_schema,
            count.total_rows,
        ))
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
                e_xy.schema().column_type(0).expect("xy.col0 type"),
            ),
            (
                "y".to_string(),
                e_xy.schema().column_type(1).expect("xy.col1 type"),
            ),
            (
                "z".to_string(),
                e_yz.schema().column_type(1).expect("yz.col1 type"),
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
        let e2_prefix_len = n_e2
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: e2 prefix length overflow")))?;
        let mut e1_work_prefix = self.memory().alloc::<u32>(prefix_len as usize)?;
        let mut e2_work_prefix = self.memory().alloc::<u32>(e2_prefix_len as usize)?;
        let mut e1_e2_start = self.memory().alloc::<u32>(n_e1 as usize)?;
        let mut e1_e2_end = self.memory().alloc::<u32>(n_e1 as usize)?;

        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || self.metadata_logical_rows(e4)? == 0 {
            let block_counts = self.memory().alloc::<u32>(1)?;
            let block_offsets = self.memory().alloc::<u32>(1)?;
            return Ok(WcojCycle4HgWorkPlanU32 {
                e1_work_prefix,
                e2_work_prefix,
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
        rec.read_write(&e2_work_prefix);
        rec.write(&e1_work_prefix);
        rec.write(&e1_e2_start);
        rec.write(&e1_e2_end);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;

        let e2_kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_4CYCLE_BUILD_E2_WORK_PREFIX_U32,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_4cycle_build_e2_work_prefix_u32 kernel not found".to_string(),
                )
            })?;
        let e2_grid = n_e2.div_ceil(BLOCK_SIZE);
        unsafe {
            e2_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (e2_grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (e2_col1, n_e2, e3_col0, n_e3, &mut e2_work_prefix),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_build_e2_work_prefix_u32 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut e2_work_prefix,
            e2_prefix_len,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

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
                        n_e2,
                        &e2_work_prefix,
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
            e2_work_prefix,
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
                e1.schema().column_type(0).expect("e1.col0 type"),
            ),
            (
                "col1".to_string(),
                e1.schema().column_type(1).expect("e1.col1 type"),
            ),
            (
                "col2".to_string(),
                e2.schema().column_type(1).expect("e2.col1 type"),
            ),
            (
                "col3".to_string(),
                e3.schema().column_type(1).expect("e3.col1 type"),
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
        rec_hg.read(&plan.e2_work_prefix);
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
                (&plan.e2_work_prefix).as_kernel_param(),
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
        rec_mat.read(&plan.e2_work_prefix);
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
                (&plan.e2_work_prefix).as_kernel_param(),
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

    /// S1c — aggregate-fused 4-cycle count: evaluate
    /// `q(W, count) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)` grouped by the
    /// variable-order root W, WITHOUT materializing the 4-cycle rows.
    ///
    /// Pipeline (all recorded; the 4-cycle result never exists as rows):
    /// 1. the standard 4-cycle histogram-guided work plan;
    /// 2. `wcoj_4cycle_groupby_root_count_hg_u32` accumulates, per e1 row,
    ///    a match count (integer atomicAdd — order-insensitive,
    ///    deterministic values);
    /// 3. a (W, count) staging buffer over the *input* rows is compacted
    ///    to count>0 rows (roots with no completion must be absent) and
    ///    reduced per W via the recorded groupby Sum.
    ///
    /// All reduction work is O(n_e1) — input-sized, never join-output-sized.
    ///
    /// Output schema matches the unfused materialize+groupby baseline:
    /// `col0` = W (e1.col0 type, U32/Symbol), `col1` = count (U64).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U32/Symbol, or
    ///   any kernel launch fails.
    pub fn wcoj_4cycle_groupby_root_count_u32_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_4cycle_groupby_root_count_u32_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e1 = &self.wcoj_layout_u32_recorded(e1, launch_stream)?;
        let e2 = &self.wcoj_layout_u32_recorded(e2, launch_stream)?;
        let e3 = &self.wcoj_layout_u32_recorded(e3, launch_stream)?;
        let e4 = &self.wcoj_layout_u32_recorded(e4, launch_stream)?;
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
        let n_e1 = plan.row_count;
        let w_type = e1.schema().column_type(0).expect("e1.col0 type");
        let out_schema = Schema::new(vec![
            ("w".to_string(), w_type),
            ("count".to_string(), ScalarType::U64),
        ]);
        if n_e1 == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let e1_col0 = metadata_column_u32(e1, 0)?;
        let e1_col1 = metadata_column_u32(e1, 1)?;
        let e2_col1 = metadata_column_u32(e2, 1)?;
        let e3_col0 = metadata_column_u32(e3, 0)?;
        let e3_col1 = metadata_column_u32(e3, 1)?;
        let e4_col0 = metadata_column_u32(e4, 0)?;
        let e4_col1 = metadata_column_u32(e4, 1)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let n_e4 = self.metadata_logical_rows(e4)?;

        // Per-e1-row match counters, zero-initialized. Allocated as the
        // u8-backed column layout so the array doubles as the staging
        // buffer's count column after the kernel fills it.
        let mut row_counts = self
            .memory()
            .alloc::<u8>(n_e1 as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read(e4.num_rows_device());
        rec.read_column(e1.column(0).expect("e1.col0"));
        rec.read_column(e1.column(1).expect("e1.col1"));
        rec.read_column(e2.column(1).expect("e2.col1"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_column(e3.column(1).expect("e3.col1"));
        rec.read_column(e4.column(0).expect("e4.col0"));
        rec.read_column(e4.column(1).expect("e4.col1"));
        rec.read(&plan.e1_work_prefix);
        rec.read(&plan.e2_work_prefix);
        rec.read(&plan.e1_e2_start);
        rec.read(&plan.e1_e2_end);
        rec.write(&row_counts);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_4CYCLE_GROUPBY_ROOT_COUNT_HG_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_4cycle_groupby_root_count_hg_u32 kernel not found".to_string(),
                    )
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
                (&plan.e2_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-count launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Staging buffer (W, count) over the n_e1 input rows: W is a
        // device-to-device copy of e1.col0; the count column is the
        // kernel-filled array. Rows stay lex-sorted by W.
        let w_src = match e1.column(0).expect("e1.col0") {
            CudaColumn::Owned(slice) => slice,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: e1.col0 must be an owned CudaColumn"
                )))
            }
        };
        let mut w_copy = self
            .memory()
            .alloc::<u8>(n_e1 as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .dtod_copy(w_src, &mut w_copy)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy W column failed: {e}")))?;
        let mut d_num_rows = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .dtod_copy(e1.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy row count failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("w".to_string(), w_type),
            ("count".to_string(), ScalarType::U32),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![w_copy.into(), row_counts.into()],
            n_e1 as u64,
            d_num_rows,
            staging_schema,
            n_e1,
        );

        // Keep only roots with at least one completed 4-cycle, then reduce
        // per W. Both steps run over input-sized data.
        let mask = self.compare_const_mask_recorded::<u32>(
            &staging,
            1,
            0u32,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        let compacted =
            self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)?;
        self.groupby_multi_agg_recorded(
            &compacted,
            &[0],
            &[(1, xlog_core::AggOp::Sum)],
            launch_stream,
        )
    }

    /// S1d — aggregate-fused 4-cycle sum/min/max: evaluate
    /// `q(W, agg(V)) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)` with
    /// `agg ∈ {Sum, Min, Max}` and `V ∈ {X, Y, Z}` grouped by the
    /// variable-order root W, WITHOUT materializing the 4-cycle rows.
    ///
    /// Pipeline (all recorded; the 4-cycle result never exists as rows):
    /// 1. the standard 4-cycle histogram-guided work plan;
    /// 2. the per-op fused kernel accumulates, per e1 row, a match count
    ///    (compaction mask) and the per-row partial aggregate (integer
    ///    atomics — order-insensitive, deterministic values). Sum partials
    ///    are u64 (a per-row partial can exceed `u32::MAX`); min partials
    ///    start at `u32::MAX`, max partials at 0;
    /// 3. a 3-column (W, count, agg) staging buffer over the *input* rows
    ///    is compacted to count>0 rows (roots with no completion must be
    ///    absent) and reduced per W via the recorded groupby with the same
    ///    `AggOp` (Sum over the u64 partials; Min/Max over u32).
    ///
    /// All reduction work is O(n_e1) — input-sized, never join-output-sized.
    ///
    /// Output schema matches the unfused materialize+groupby baseline:
    /// `col0` = W (e1.col0 type, U32/Symbol), `col1` = U64 for Sum,
    /// U32 for Min/Max.
    ///
    /// Bag semantics: every (X, Y, Z) completion contributes its value,
    /// exactly like aggregating the materialized projection.
    ///
    /// # Errors
    /// * `XlogError::Kernel` if `agg_op` is not Sum/Min/Max, the value
    ///   column is not plain U32, the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U32/Symbol, or
    ///   any kernel launch fails.
    #[allow(clippy::too_many_arguments)]
    pub fn wcoj_4cycle_groupby_root_agg_u32_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        agg_op: AggOp,
        value: Wcoj4CycleRootAggValue,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_4cycle_groupby_root_agg_u32_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e1 = &self.wcoj_layout_u32_recorded(e1, launch_stream)?;
        let e2 = &self.wcoj_layout_u32_recorded(e2, launch_stream)?;
        let e3 = &self.wcoj_layout_u32_recorded(e3, launch_stream)?;
        let e4 = &self.wcoj_layout_u32_recorded(e4, launch_stream)?;
        let (kernel_name, agg_elem_size, agg_scalar, agg_name) = match agg_op {
            AggOp::Sum => (
                wcoj_kernels::WCOJ_4CYCLE_GROUPBY_ROOT_SUM_HG_U32,
                std::mem::size_of::<u64>(),
                ScalarType::U64,
                "sum_0",
            ),
            AggOp::Min => (
                wcoj_kernels::WCOJ_4CYCLE_GROUPBY_ROOT_MIN_HG_U32,
                std::mem::size_of::<u32>(),
                ScalarType::U32,
                "min_0",
            ),
            AggOp::Max => (
                wcoj_kernels::WCOJ_4CYCLE_GROUPBY_ROOT_MAX_HG_U32,
                std::mem::size_of::<u32>(),
                ScalarType::U32,
                "max_0",
            ),
            other => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: unsupported AggOp {other:?} (Sum/Min/Max only; use \
                     wcoj_4cycle_groupby_root_count_u32_recorded for Count)"
                )))
            }
        };
        validate_binary_u32(ctx, "e1", e1)?;
        validate_binary_u32(ctx, "e2", e2)?;
        validate_binary_u32(ctx, "e3", e3)?;
        validate_binary_u32(ctx, "e4", e4)?;
        // The aggregate value is arithmetic: require a plain U32 value
        // column (Symbol ids are not summable/orderable data). The column
        // checked is exactly the one the kernel reads the value from —
        // and the one whose type the materialized (W, X, Y, Z) baseline
        // schema carries (`build_4cycle_head_schema`).
        let (value_buf, value_label) = match value {
            Wcoj4CycleRootAggValue::X => (e1, "e1"),
            Wcoj4CycleRootAggValue::Y => (e2, "e2"),
            Wcoj4CycleRootAggValue::Z => (e3, "e3"),
        };
        {
            let ty = value_buf.schema().column_type(1).expect("validated 2-col");
            if ty != ScalarType::U32 {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: {value_label}.col1 supplies the aggregate value and must be U32, \
                     got {ty:?}"
                )));
            }
        }

        let plan = self.wcoj_4cycle_hg_work_plan_u32_recorded(
            e1,
            e2,
            e3,
            e4,
            block_work_unit,
            launch_stream,
        )?;
        let n_e1 = plan.row_count;
        let w_type = e1.schema().column_type(0).expect("e1.col0 type");
        let out_schema = Schema::new(vec![
            ("w".to_string(), w_type),
            (agg_name.to_string(), agg_scalar),
        ]);
        if n_e1 == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let e1_col0 = metadata_column_u32(e1, 0)?;
        let e1_col1 = metadata_column_u32(e1, 1)?;
        let e2_col1 = metadata_column_u32(e2, 1)?;
        let e3_col0 = metadata_column_u32(e3, 0)?;
        let e3_col1 = metadata_column_u32(e3, 1)?;
        let e4_col0 = metadata_column_u32(e4, 0)?;
        let e4_col1 = metadata_column_u32(e4, 1)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let n_e4 = self.metadata_logical_rows(e4)?;
        let value_sel: u32 = match value {
            Wcoj4CycleRootAggValue::X => 0,
            Wcoj4CycleRootAggValue::Y => 1,
            Wcoj4CycleRootAggValue::Z => 2,
        };

        // Per-e1-row match counters + aggregate partials, allocated as
        // the u8-backed column layout so the arrays double as the staging
        // buffer's columns after the kernel fills them.
        let mut row_counts = self
            .memory()
            .alloc::<u8>(n_e1 as usize * std::mem::size_of::<u32>())?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;
        let mut row_agg = self.memory().alloc::<u8>(n_e1 as usize * agg_elem_size)?;
        self.device()
            .inner()
            .memset_zeros(&mut row_agg)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row aggregates failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read(e4.num_rows_device());
        rec.read_column(e1.column(0).expect("e1.col0"));
        rec.read_column(e1.column(1).expect("e1.col1"));
        rec.read_column(e2.column(1).expect("e2.col1"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_column(e3.column(1).expect("e3.col1"));
        rec.read_column(e4.column(0).expect("e4.col0"));
        rec.read_column(e4.column(1).expect("e4.col1"));
        rec.read(&plan.e1_work_prefix);
        rec.read(&plan.e2_work_prefix);
        rec.read(&plan.e1_e2_start);
        rec.read(&plan.e1_e2_end);
        rec.write(&row_counts);
        rec.write(&row_agg);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        if matches!(agg_op, AggOp::Min) {
            // Min identity: u32::MAX (compaction drops untouched rows).
            let fill = self
                .device()
                .inner()
                .get_func(ARITH_MODULE, arith_kernels::ARITH_FILL_CONST_U32)
                .ok_or_else(|| {
                    XlogError::Kernel("arith_fill_const_u32 kernel not found".to_string())
                })?;
            let row_agg_u32 = unsafe { reinterpret_u8_as_u32(&mut row_agg) };
            // SAFETY: arith_fill_const_u32(value, n, output)
            unsafe {
                fill.clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig::for_num_elems(n_e1),
                        (u32::MAX, n_e1, &mut *row_agg_u32),
                    )
                    .map_err(|e| {
                        XlogError::Kernel(format!("{ctx}: min identity fill failed: {e}"))
                    })?;
            }
        }
        {
            let kernel = self
                .device()
                .inner()
                .get_func(WCOJ_MODULE, kernel_name)
                .ok_or_else(|| {
                    XlogError::Kernel(format!("{kernel_name} kernel not found"))
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
                value_sel.as_kernel_param(),
                (&plan.e1_work_prefix).as_kernel_param(),
                (&plan.e2_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
                (&row_agg).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-agg launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Staging buffer (W, count, agg) over the n_e1 input rows: W is a
        // device-to-device copy of e1.col0; count and agg are the
        // kernel-filled arrays. Rows stay lex-sorted by W.
        let w_src = match e1.column(0).expect("e1.col0") {
            CudaColumn::Owned(slice) => slice,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: e1.col0 must be an owned CudaColumn"
                )))
            }
        };
        let w_copy = self
            .memory()
            .alloc::<u8>(n_e1 as usize * std::mem::size_of::<u32>())?;
        // Explicit-length copy: layout-normalized columns are allocated at
        // capacity, which can exceed the logical n_e1 * 4 bytes a full-slice
        // typed copy would assert on.
        unsafe {
            let res = sys::cuMemcpyDtoD_v2(
                *w_copy.device_ptr(),
                *w_src.device_ptr(),
                n_e1 as usize * std::mem::size_of::<u32>(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: copy W column failed: {res:?}"
                )));
            }
        }
        let mut d_num_rows = self.memory().alloc::<u32>(1)?;
        self.device()
            .inner()
            .dtod_copy(e1.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy row count failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("w".to_string(), w_type),
            ("count".to_string(), ScalarType::U32),
            ("agg".to_string(), agg_scalar),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![w_copy.into(), row_counts.into(), row_agg.into()],
            n_e1 as u64,
            d_num_rows,
            staging_schema,
            n_e1,
        );

        // Keep only roots with at least one completed 4-cycle, then reduce
        // per W with the same AggOp. Both steps run over input-sized data.
        let mask = self.compare_const_mask_recorded::<u32>(
            &staging,
            1,
            0u32,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        let compacted =
            self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)?;
        self.groupby_multi_agg_recorded(&compacted, &[0], &[(2, agg_op)], launch_stream)
    }

    /// S1d slice 2 — u64-key sibling of
    /// [`Self::wcoj_4cycle_groupby_root_count_u32_recorded`]: evaluate
    /// `q(W, count) :- e1(W,X), e2(X,Y), e3(Y,Z), e4(Z,W)` grouped by the
    /// variable-order root W for U64 relations, WITHOUT materializing the
    /// 4-cycle rows.
    ///
    /// The recorded groupby is U32/Symbol-key only, so the per-W reduction
    /// reuses the WCOJ relation metadata instead (mirroring
    /// [`Self::wcoj_triangle_groupby_root_count_u64_recorded`]): e1 is
    /// lex-sorted, so `wcoj_build_metadata_u64_recorded` yields one
    /// (unique W, group start) pair per root, and
    /// `wcoj_groupby_root_segment_sum_counts_u32` accumulates the per-row
    /// match counts into per-unique-root u64 totals (integer atomicAdd —
    /// deterministic). Roots with zero completions are compacted away.
    /// All reduction work is O(n_e1).
    ///
    /// Output schema matches the unfused materialize+groupby baseline:
    /// `col0` = W (U64), `col1` = count (U64).
    ///
    /// # Errors
    /// * `XlogError::Kernel` if the manager has no runtime, the launch
    ///   stream does not resolve, an input is not 2-column U64, or any
    ///   kernel launch fails.
    pub fn wcoj_4cycle_groupby_root_count_u64_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_4cycle_groupby_root_count_u64_recorded";
        // Layout-normalize per dispatch (sorted-fast-path clone when the
        // input is already lex-sorted + unique): the fused path must give
        // the same guarantee as the unfused pipeline instead of trusting
        // store-buffer sortedness — unsorted/duplicated inputs previously
        // produced silently wrong (empty) fused results.
        let e1 = &self.wcoj_layout_u64_recorded(e1, launch_stream)?;
        let e2 = &self.wcoj_layout_u64_recorded(e2, launch_stream)?;
        let e3 = &self.wcoj_layout_u64_recorded(e3, launch_stream)?;
        let e4 = &self.wcoj_layout_u64_recorded(e4, launch_stream)?;
        validate_binary_u64(ctx, "e1", e1)?;
        validate_binary_u64(ctx, "e2", e2)?;
        validate_binary_u64(ctx, "e3", e3)?;
        validate_binary_u64(ctx, "e4", e4)?;
        let plan = self.wcoj_4cycle_hg_work_plan_u64_recorded(
            e1,
            e2,
            e3,
            e4,
            block_work_unit,
            launch_stream,
        )?;
        let n_e1 = plan.row_count;
        let out_schema = Schema::new(vec![
            ("w".to_string(), ScalarType::U64),
            ("count".to_string(), ScalarType::U64),
        ]);
        if n_e1 == 0 || plan.total_work == 0 {
            return self.create_empty_buffer(out_schema);
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

        let e1_col0 = metadata_column_u64(e1, 0)?;
        let e1_col1 = metadata_column_u64(e1, 1)?;
        let e2_col1 = metadata_column_u64(e2, 1)?;
        let e3_col0 = metadata_column_u64(e3, 0)?;
        let e3_col1 = metadata_column_u64(e3, 1)?;
        let e4_col0 = metadata_column_u64(e4, 0)?;
        let e4_col1 = metadata_column_u64(e4, 1)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let n_e4 = self.metadata_logical_rows(e4)?;

        // Per-e1-row match counters, zero-initialized.
        let mut row_counts = self.memory().alloc::<u32>(n_e1 as usize)?;
        self.device()
            .inner()
            .memset_zeros(&mut row_counts)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero row counts failed: {e}")))?;

        let grid = plan.total_work.div_ceil(plan.block_work_unit);
        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read(e4.num_rows_device());
        rec.read_column(e1.column(0).expect("e1.col0"));
        rec.read_column(e1.column(1).expect("e1.col1"));
        rec.read_column(e2.column(1).expect("e2.col1"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_column(e3.column(1).expect("e3.col1"));
        rec.read_column(e4.column(0).expect("e4.col0"));
        rec.read_column(e4.column(1).expect("e4.col1"));
        rec.read(&plan.e1_work_prefix);
        rec.read(&plan.e2_work_prefix);
        rec.read(&plan.e1_e2_start);
        rec.read(&plan.e1_e2_end);
        rec.write(&row_counts);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_4CYCLE_GROUPBY_ROOT_COUNT_HG_U64,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_4cycle_groupby_root_count_hg_u64 kernel not found".to_string(),
                    )
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
                (&plan.e2_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                (&row_counts).as_kernel_param(),
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
                        XlogError::Kernel(format!("{ctx}: groupby-count launch failed: {e}"))
                    })?;
            }
        }
        rec.commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: commit failed: {e}")))?;

        // Per-W reduction via the relation metadata: one (unique W, group
        // start) pair per root; e1 is lex-sorted by W so group rows are
        // contiguous.
        let meta = self.wcoj_build_metadata_u64_recorded(e1, 0, launch_stream)?;
        let key_count = meta.key_count;
        if key_count == 0 {
            return self.create_empty_buffer(out_schema);
        }
        let mut sums = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        self.device()
            .inner()
            .memset_zeros(&mut sums)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: zero group sums failed: {e}")))?;

        let mut rec_sum = LaunchRecorder::new_strict(launch_stream);
        rec_sum.read(&row_counts);
        rec_sum.read(&meta.prefix_sum);
        rec_sum.write(&sums);
        rec_sum
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce preflight failed: {e}")))?;
        {
            let kernel = self
                .device()
                .inner()
                .get_func(
                    WCOJ_MODULE,
                    wcoj_kernels::WCOJ_GROUPBY_ROOT_SEGMENT_SUM_COUNTS_U32,
                )
                .ok_or_else(|| {
                    XlogError::Kernel(
                        "wcoj_groupby_root_segment_sum_counts_u32 kernel not found".to_string(),
                    )
                })?;
            let reduce_grid = n_e1.div_ceil(BLOCK_SIZE);
            let mut params: Vec<*mut c_void> = vec![
                (&row_counts).as_kernel_param(),
                n_e1.as_kernel_param(),
                (&meta.prefix_sum).as_kernel_param(),
                key_count.as_kernel_param(),
                (&sums).as_kernel_param(),
            ];
            unsafe {
                kernel
                    .clone()
                    .launch_on_stream(
                        &cu_stream,
                        LaunchConfig {
                            grid_dim: (reduce_grid, 1, 1),
                            block_dim: (BLOCK_SIZE, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        &mut params,
                    )
                    .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce launch failed: {e}")))?;
            }
        }
        rec_sum
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: reduce commit failed: {e}")))?;

        // (unique W, total) buffer over the key_count roots, then drop the
        // roots with no completion. The copies run on launch_stream and the
        // fresh destination blocks are registered through the strict
        // recorder BEFORE the enqueue — a raw async copy into a freshly
        // pool-allocated block without recording is a visibility race.
        let w_copy = self
            .memory()
            .alloc::<u8>(key_count as usize * std::mem::size_of::<u64>())?;
        let d_num_rows = self.memory().alloc::<u32>(1)?;
        let mut rec_copy = LaunchRecorder::new_strict(launch_stream);
        rec_copy.read(&meta.unique_keys);
        rec_copy.write(&w_copy);
        rec_copy.write(&d_num_rows);
        rec_copy
            .preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy preflight failed: {e}")))?;
        unsafe {
            let res = sys::cuMemcpyDtoDAsync_v2(
                *w_copy.device_ptr(),
                *meta.unique_keys.device_ptr(),
                key_count as usize * std::mem::size_of::<u64>(),
                cu_stream.cu_stream(),
            );
            if res != sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{ctx}: DtoD unique keys copy failed: {res:?}"
                )));
            }
        }
        self.htod_launch_metadata_async_copy_one(
            &key_count,
            &d_num_rows,
            &cu_stream,
            &format!("{ctx}: d_num_rows"),
        )?;
        rec_copy
            .commit(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: copy commit failed: {e}")))?;
        cu_stream
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("{ctx}: stream sync failed: {e}")))?;
        let staging_schema = Schema::new(vec![
            ("w".to_string(), ScalarType::U64),
            ("count".to_string(), ScalarType::U64),
        ]);
        let staging = CudaBuffer::from_columns_with_host_count(
            vec![w_copy.into(), sums.into()],
            u64::from(key_count),
            d_num_rows,
            staging_schema,
            key_count,
        );
        let mask = self.compare_const_mask_recorded::<u64>(
            &staging,
            1,
            0u64,
            crate::CompareOp::Gt,
            launch_stream,
        )?;
        self.compact_buffer_by_device_mask_counted_recorded(&staging, &mask, launch_stream)
    }

    pub fn wcoj_4cycle_hg_work_plan_u64_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<WcojCycle4HgWorkPlanU64> {
        let ctx = "wcoj_4cycle_hg_work_plan_u64_recorded";
        if block_work_unit == 0 {
            return Err(XlogError::Kernel(format!(
                "{ctx}: block_work_unit must be nonzero"
            )));
        }
        validate_binary_u64(ctx, "e1", e1)?;
        validate_binary_u64(ctx, "e2", e2)?;
        validate_binary_u64(ctx, "e3", e3)?;
        validate_binary_u64(ctx, "e4", e4)?;

        let n_e1 = self.metadata_logical_rows(e1)?;
        let n_e2 = self.metadata_logical_rows(e2)?;
        let n_e3 = self.metadata_logical_rows(e3)?;
        let prefix_len = n_e1
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: prefix length overflow")))?;
        let e2_prefix_len = n_e2
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel(format!("{ctx}: e2 prefix length overflow")))?;
        let mut e1_work_prefix = self.memory().alloc::<u32>(prefix_len as usize)?;
        let mut e2_work_prefix = self.memory().alloc::<u32>(e2_prefix_len as usize)?;
        let mut e1_e2_start = self.memory().alloc::<u32>(n_e1 as usize)?;
        let mut e1_e2_end = self.memory().alloc::<u32>(n_e1 as usize)?;

        if n_e1 == 0 || n_e2 == 0 || n_e3 == 0 || self.metadata_logical_rows(e4)? == 0 {
            let block_counts = self.memory().alloc::<u32>(1)?;
            let block_offsets = self.memory().alloc::<u32>(1)?;
            return Ok(WcojCycle4HgWorkPlanU64 {
                e1_work_prefix,
                e2_work_prefix,
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

        let e1_col1 = metadata_column_u64(e1, 1)?;
        let e2_col0 = metadata_column_u64(e2, 0)?;
        let e2_col1 = metadata_column_u64(e2, 1)?;
        let e3_col0 = metadata_column_u64(e3, 0)?;

        let mut rec = LaunchRecorder::new_strict(launch_stream);
        rec.read(e1.num_rows_device());
        rec.read(e2.num_rows_device());
        rec.read(e3.num_rows_device());
        rec.read_column(e1.column(1).expect("e1.col1"));
        rec.read_column(e2.column(0).expect("e2.col0"));
        rec.read_column(e2.column(1).expect("e2.col1"));
        rec.read_column(e3.column(0).expect("e3.col0"));
        rec.read_write(&e2_work_prefix);
        rec.write(&e1_work_prefix);
        rec.write(&e1_e2_start);
        rec.write(&e1_e2_end);
        rec.preflight(runtime)
            .map_err(|e| XlogError::Kernel(format!("{ctx}: preflight failed: {e}")))?;

        let e2_kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_4CYCLE_BUILD_E2_WORK_PREFIX_U64,
            )
            .ok_or_else(|| {
                XlogError::Kernel(
                    "wcoj_4cycle_build_e2_work_prefix_u64 kernel not found".to_string(),
                )
            })?;
        let e2_grid = n_e2.div_ceil(BLOCK_SIZE);
        unsafe {
            e2_kernel
                .clone()
                .launch_on_stream(
                    &cu_stream,
                    LaunchConfig {
                        grid_dim: (e2_grid, 1, 1),
                        block_dim: (BLOCK_SIZE, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (e2_col1, n_e2, e3_col0, n_e3, &mut e2_work_prefix),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_build_e2_work_prefix_u64 launch failed: {e}"
                    ))
                })?;
        }
        self.multiblock_scan_u32_inplace_on_stream(
            &mut e2_work_prefix,
            e2_prefix_len,
            &cu_stream,
            launch_stream,
            runtime,
        )?;

        let kernel = self
            .device()
            .inner()
            .get_func(
                WCOJ_MODULE,
                wcoj_kernels::WCOJ_4CYCLE_BUILD_HG_WORK_PLAN_U64,
            )
            .ok_or_else(|| {
                XlogError::Kernel("wcoj_4cycle_build_hg_work_plan_u64 kernel not found".to_string())
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
                        n_e2,
                        &e2_work_prefix,
                        &mut e1_work_prefix,
                        &mut e1_e2_start,
                        &mut e1_e2_end,
                    ),
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "wcoj_4cycle_build_hg_work_plan_u64 launch failed: {e}"
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

        Ok(WcojCycle4HgWorkPlanU64 {
            e1_work_prefix,
            e2_work_prefix,
            e1_e2_start,
            e1_e2_end,
            block_counts,
            block_offsets,
            total_work,
            block_work_unit,
            row_count: n_e1,
        })
    }

    pub fn wcoj_4cycle_hg_u64_recorded(
        &self,
        e1: &CudaBuffer,
        e2: &CudaBuffer,
        e3: &CudaBuffer,
        e4: &CudaBuffer,
        block_work_unit: u32,
        launch_stream: StreamId,
    ) -> Result<CudaBuffer> {
        let ctx = "wcoj_4cycle_hg_u64_recorded";
        validate_binary_u64(ctx, "e1", e1)?;
        validate_binary_u64(ctx, "e2", e2)?;
        validate_binary_u64(ctx, "e3", e3)?;
        validate_binary_u64(ctx, "e4", e4)?;
        let plan = self.wcoj_4cycle_hg_work_plan_u64_recorded(
            e1,
            e2,
            e3,
            e4,
            block_work_unit,
            launch_stream,
        )?;
        let out_schema = Schema::new(vec![
            ("col0".to_string(), ScalarType::U64),
            ("col1".to_string(), ScalarType::U64),
            ("col2".to_string(), ScalarType::U64),
            ("col3".to_string(), ScalarType::U64),
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

        let e1_col0 = metadata_column_u64(e1, 0)?;
        let e1_col1 = metadata_column_u64(e1, 1)?;
        let e2_col1 = metadata_column_u64(e2, 1)?;
        let e3_col0 = metadata_column_u64(e3, 0)?;
        let e3_col1 = metadata_column_u64(e3, 1)?;
        let e4_col0 = metadata_column_u64(e4, 0)?;
        let e4_col1 = metadata_column_u64(e4, 1)?;
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
        rec_hg.read(&plan.e2_work_prefix);
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
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_COUNT_HG_U64)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_4cycle_count_hg_u64 kernel not found".to_string())
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
                (&plan.e2_work_prefix).as_kernel_param(),
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
            .checked_mul(std::mem::size_of::<u64>())
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
        rec_mat.read(&plan.e2_work_prefix);
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
                .get_func(WCOJ_MODULE, wcoj_kernels::WCOJ_4CYCLE_MATERIALIZE_HG_U64)
                .ok_or_else(|| {
                    XlogError::Kernel("wcoj_4cycle_materialize_hg_u64 kernel not found".to_string())
                })?;
            let out_w_u64 = unsafe { reinterpret_u8_as_u64(&mut out_w) };
            let out_x_u64 = unsafe { reinterpret_u8_as_u64(&mut out_x) };
            let out_y_u64 = unsafe { reinterpret_u8_as_u64(&mut out_y) };
            let out_z_u64 = unsafe { reinterpret_u8_as_u64(&mut out_z) };
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
                (&plan.e2_work_prefix).as_kernel_param(),
                (&plan.e1_e2_start).as_kernel_param(),
                (&plan.e1_e2_end).as_kernel_param(),
                plan.total_work.as_kernel_param(),
                plan.block_work_unit.as_kernel_param(),
                materialize_offsets.as_kernel_param(),
                total_rows.as_kernel_param(),
                out_w_u64.as_kernel_param(),
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
                per_candidate_root: BTreeMap::new(),
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

        Ok(WcojRelationMetadata {
            unique_keys,
            fan_out,
            prefix_sum,
            per_candidate_root: BTreeMap::new(),
            total: u64::from(n),
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
                per_candidate_root: BTreeMap::new(),
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

        Ok(WcojRelationMetadata {
            unique_keys,
            fan_out,
            prefix_sum,
            per_candidate_root: BTreeMap::new(),
            total: u64::from(n),
            key_count,
            row_count: n,
        })
    }

    #[allow(clippy::too_many_arguments)]
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
        let grid = n.div_ceil(BLOCK_SIZE);
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

    #[allow(clippy::too_many_arguments)]
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
        let grid = n.div_ceil(BLOCK_SIZE);
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
        let grid = n.div_ceil(BLOCK_SIZE);
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
        let grid = n.div_ceil(BLOCK_SIZE);
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

fn metadata_column_u32(input: &CudaBuffer, key_col_idx: usize) -> Result<&TrackedCudaSlice<u32>> {
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

fn metadata_column_u64(input: &CudaBuffer, key_col_idx: usize) -> Result<&TrackedCudaSlice<u64>> {
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
