// ---------------------------------------------------------------------------
// ILP GPU loss/grad computation helpers
// ---------------------------------------------------------------------------
//
// Phases C (COO build), D (sort + CSR), and E (forward/backward/reduce)
// extracted from `compute_ilp_loss_grad_gpu` in ilp.rs.
// ---------------------------------------------------------------------------

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use xlog_core::{ScalarType, Schema};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::type_seam::GpuScalar;
use xlog_cuda::CudaKernelProvider;

use super::{dlpack_capsule_from_tensor, types};

// ---------------------------------------------------------------------------
// CooTask — shared between ilp.rs (Phase B) and this module (Phase C)
// ---------------------------------------------------------------------------

/// Task descriptor for COO fill (one per mask-query x candidate).
pub(crate) struct CooTask {
    pub(crate) d_mask: TrackedCudaSlice<u8>,
    pub(crate) fact_indices_idx: usize,
    pub(crate) cidx: u32,
    pub(crate) num_query: u32,
}

// ---------------------------------------------------------------------------
// Phase C: COO construction
// ---------------------------------------------------------------------------

/// Phase C (non-chunked): single-pass COO fill on device.
///
/// Over-allocates COO arrays at `upper_bound` and fills sentinels for unused
/// slots. Returns `(d_coo_facts, d_coo_cands, actual_nnz)` where actual_nnz
/// equals upper_bound (sentinels sort to end and are ignored later).
pub(crate) fn build_coo_single(
    provider: &CudaKernelProvider,
    tasks: &[CooTask],
    fact_indices_buffers: &[TrackedCudaSlice<u32>],
    num_facts: u32,
    num_cands: u32,
    upper_bound: u32,
) -> PyResult<(TrackedCudaSlice<u32>, TrackedCudaSlice<u32>, u32)> {
    let num_tasks = tasks.len();

    let mut offsets_host = vec![0u32; num_tasks];
    {
        let mut running = 0u32;
        for i in 0..num_tasks {
            offsets_host[i] = running;
            running += tasks[i].num_query;
        }
        debug_assert_eq!(running, upper_bound);
    }

    let mut d_offsets = provider
        .memory()
        .alloc::<u32>(num_tasks)
        .map_err(|e| types::gpu_err("alloc d_offsets", e))?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&offsets_host, &mut d_offsets)
        .map_err(|e| types::gpu_err("htod d_offsets", e))?;

    let sentinel_fact = num_facts;
    let sentinel_cand = num_cands;
    let mut d_coo_facts = provider
        .memory()
        .alloc::<u32>(upper_bound as usize)
        .map_err(|e| types::gpu_err("alloc coo_facts", e))?;
    let mut d_coo_cands = provider
        .memory()
        .alloc::<u32>(upper_bound as usize)
        .map_err(|e| types::gpu_err("alloc coo_cands", e))?;
    {
        let sentinel_facts_vec = vec![sentinel_fact; upper_bound as usize];
        let sentinel_cands_vec = vec![sentinel_cand; upper_bound as usize];
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&sentinel_facts_vec, &mut d_coo_facts)
            .map_err(|e| types::gpu_err("sentinel facts", e))?;
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&sentinel_cands_vec, &mut d_coo_cands)
            .map_err(|e| types::gpu_err("sentinel cands", e))?;
    }

    for (task_idx, task) in tasks.iter().enumerate() {
        let d_prefix = provider
            .scan_u8_mask_device(&task.d_mask, task.num_query)
            .map_err(|e| types::gpu_err("scan mask", e))?;

        provider
            .ilp_coo_fill_from_mask_launch(
                &task.d_mask,
                &d_prefix,
                &fact_indices_buffers[task.fact_indices_idx],
                task_idx as u32,
                task.cidx,
                task.num_query,
                &d_offsets,
                &mut d_coo_facts,
                &mut d_coo_cands,
            )
            .map_err(|e| types::gpu_err("coo fill", e))?;
    }

    Ok((d_coo_facts, d_coo_cands, upper_bound))
}

/// Phase C (chunked): two-pass GPU-only bounded-memory merge.
///
/// Pass 1: Count NNZ per task on device (bounded temp per chunk).
/// Pass 2: Fill COO at pre-computed offsets (bounded temp per chunk).
/// The final COO buffer is exact-NNZ sized and may exceed `coo_chunk_budget`.
///
/// Returns `(d_coo_facts, d_coo_cands, total_nnz)`, or `None` if total_nnz
/// is zero (caller should handle the empty-COO path).
pub(crate) fn build_coo_chunked(
    provider: &CudaKernelProvider,
    tasks: &[CooTask],
    fact_indices_buffers: &[TrackedCudaSlice<u32>],
    _num_facts: u32,
    coo_chunk_budget: u64,
) -> PyResult<Option<(TrackedCudaSlice<u32>, TrackedCudaSlice<u32>, u32)>> {
    let num_tasks = tasks.len();
    let max_queries_per_chunk = (coo_chunk_budget / 8).max(1) as u32;

    // Allocate task_counts array (num_tasks + 1 for exclusive scan).
    let tc_len = num_tasks + 1;
    let mut d_task_counts = provider
        .memory()
        .alloc::<u32>(tc_len)
        .map_err(|e| types::gpu_err("alloc task_counts", e))?;
    // Zero-init the entire array (counts default to 0, last slot = 0 for scan).
    {
        let zeros = vec![0u32; tc_len];
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&zeros, &mut d_task_counts)
            .map_err(|e| types::gpu_err("zero task_counts", e))?;
    }

    // ── Pass 1: Count NNZ per task ──
    {
        let mut chunk_start = 0usize;
        while chunk_start < tasks.len() {
            let mut chunk_end = chunk_start;
            let mut chunk_sum = 0u32;
            while chunk_end < tasks.len() {
                let nq = tasks[chunk_end].num_query;
                if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                    break;
                }
                chunk_sum += nq;
                chunk_end += 1;
            }
            if chunk_end == chunk_start {
                chunk_end = chunk_start + 1;
            }

            for task_idx in chunk_start..chunk_end {
                let task = &tasks[task_idx];
                provider
                    .count_mask_into_slot(
                        &task.d_mask,
                        task.num_query,
                        &mut d_task_counts,
                        task_idx,
                    )
                    .map_err(|e| types::gpu_err("count_mask_into_slot", e))?;
            }

            chunk_start = chunk_end;
        }
    }

    // Synchronize to ensure all count kernels have completed.
    provider
        .device()
        .synchronize()
        .map_err(|e| types::gpu_err("sync pass1", e))?;

    // Compute per-task write offsets via exclusive scan.
    // d_task_counts[0..num_tasks] has counts; after scan,
    // d_task_counts[i] = sum of counts[0..i] = write offset for task i.
    // d_task_counts[num_tasks] = total_nnz.
    provider
        .exclusive_scan_u32_inplace(&mut d_task_counts, tc_len as u32)
        .map_err(|e| types::gpu_err("scan task_counts", e))?;

    // Read total_nnz from the last element (metadata-only, untracked).
    let total_nnz: u32 = provider
        .dtoh_scalar_untracked(&d_task_counts, num_tasks)
        .map_err(|e| types::gpu_err("read total_nnz", e))?;

    if total_nnz == 0 {
        return Ok(None);
    }

    // Allocate exact-NNZ global COO buffers (may exceed coo_chunk_budget).
    let mut d_coo_facts = provider
        .memory()
        .alloc::<u32>(total_nnz as usize)
        .map_err(|e| types::gpu_err("alloc global coo_facts", e))?;
    let mut d_coo_cands = provider
        .memory()
        .alloc::<u32>(total_nnz as usize)
        .map_err(|e| types::gpu_err("alloc global coo_cands", e))?;

    // ── Pass 2: Fill COO at pre-computed offsets ──
    // d_task_counts now contains offsets (after exclusive scan).
    // ilp_coo_fill_from_mask_launch reads d_offsets[offset_idx] for the
    // write base position. We pass d_task_counts as d_offsets with
    // offset_idx = task_idx.
    {
        let mut chunk_start = 0usize;
        while chunk_start < tasks.len() {
            let mut chunk_end = chunk_start;
            let mut chunk_sum = 0u32;
            while chunk_end < tasks.len() {
                let nq = tasks[chunk_end].num_query;
                if chunk_sum + nq > max_queries_per_chunk && chunk_sum > 0 {
                    break;
                }
                chunk_sum += nq;
                chunk_end += 1;
            }
            if chunk_end == chunk_start {
                chunk_end = chunk_start + 1;
            }

            for task_idx in chunk_start..chunk_end {
                let task = &tasks[task_idx];

                let d_prefix = provider
                    .scan_u8_mask_device(&task.d_mask, task.num_query)
                    .map_err(|e| types::gpu_err("scan mask", e))?;

                provider
                    .ilp_coo_fill_from_mask_launch(
                        &task.d_mask,
                        &d_prefix,
                        &fact_indices_buffers[task.fact_indices_idx],
                        task_idx as u32,
                        task.cidx,
                        task.num_query,
                        &d_task_counts, // offsets after scan
                        &mut d_coo_facts,
                        &mut d_coo_cands,
                    )
                    .map_err(|e| types::gpu_err("coo fill", e))?;
                // d_prefix dropped here (bounded temp).
            }

            chunk_start = chunk_end;
        }
    }

    Ok(Some((d_coo_facts, d_coo_cands, total_nnz)))
}

// ---------------------------------------------------------------------------
// Phase D: Sort COO + device-side CSR build
// ---------------------------------------------------------------------------

/// Sort all COO entries by fact index and build CSR row offsets.
///
/// In the non-chunked path, sentinels (fact = num_facts) sort to the end
/// and are ignored by the histogram kernel's `f < num_facts` guard. In the
/// chunked path, the COO is exact-NNZ (no sentinels).
pub(crate) fn sort_and_build_csr(
    provider: &CudaKernelProvider,
    d_coo_facts: &mut TrackedCudaSlice<u32>,
    d_coo_cands: &mut TrackedCudaSlice<u32>,
    actual_nnz: u32,
    num_facts: u32,
) -> PyResult<TrackedCudaSlice<u32>> {
    let mut scratch = xlog_cuda::provider::RadixSortScratch::new(provider, actual_nnz)
        .map_err(|e| types::gpu_err("sort scratch", e))?;
    provider
        .radix_sort_u32_pairs(d_coo_facts, d_coo_cands, actual_nnz, &mut scratch)
        .map_err(|e| types::gpu_err("radix sort", e))?;

    let d_hist = provider
        .ilp_csr_histogram_launch(d_coo_facts, actual_nnz, num_facts)
        .map_err(|e| types::gpu_err("csr histogram", e))?;

    let mut d_row_offsets = provider
        .memory()
        .alloc::<u32>((num_facts + 1) as usize)
        .map_err(|e| types::gpu_err("alloc row_offsets", e))?;
    {
        let zeros = vec![0u32; (num_facts + 1) as usize];
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&zeros, &mut d_row_offsets)
            .map_err(|e| types::gpu_err("zero row_offsets", e))?;
    }
    {
        let mut dst_view = d_row_offsets
            .try_slice_mut(0..num_facts as usize)
            .ok_or_else(|| PyRuntimeError::new_err("row_offsets slice failed"))?;
        provider
            .device()
            .inner()
            .dtod_copy(&d_hist, &mut dst_view)
            .map_err(|e| types::gpu_err("dtod hist->row_offsets", e))?;
    }
    provider
        .exclusive_scan_u32_inplace(&mut d_row_offsets, num_facts + 1)
        .map_err(|e| types::gpu_err("scan row_offsets", e))?;

    Ok(d_row_offsets)
}

// ---------------------------------------------------------------------------
// Phase E: Forward + backward + device-side reduction
// ---------------------------------------------------------------------------

/// Run forward credit-gather, NLL loss reduction, and backward gradient
/// scatter on device. Returns `(loss_capsule, grad_capsule)` DLPack objects.
pub(crate) fn forward_backward_reduce(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    d_row_offsets: &TrackedCudaSlice<u32>,
    d_coo_cands: &TrackedCudaSlice<u32>,
    cand_col: &xlog_cuda::CudaColumn,
    d_is_positive: &TrackedCudaSlice<u8>,
    num_facts: u32,
    num_cands: u32,
    is_f64: bool,
) -> PyResult<(PyObject, PyObject)> {
    if is_f64 {
        let eps_f64 = 1e-8f64;

        let (credit_out, loss_contrib) = provider
            .ilp_credit_forward_f64_launch(
                d_row_offsets,
                d_coo_cands,
                cand_col,
                d_is_positive,
                num_facts,
                eps_f64,
            )
            .map_err(|e| types::gpu_err("forward f64", e))?;

        let d_total_loss = provider
            .ilp_reduce_sum_f64_launch(&loss_contrib, num_facts)
            .map_err(|e| types::gpu_err("reduce f64", e))?;

        let d_grad = provider
            .ilp_credit_backward_f64_launch(
                d_row_offsets,
                d_coo_cands,
                &credit_out,
                d_is_positive,
                num_facts,
                num_cands,
            )
            .map_err(|e| types::gpu_err("backward f64", e))?;

        export_loss_grad_device(
            provider,
            py,
            d_total_loss,
            d_grad,
            num_cands,
            ScalarType::F64,
        )
    } else {
        let eps_f32 = 1e-8f32;

        let (credit_out, loss_contrib) = provider
            .ilp_credit_forward_f32_launch(
                d_row_offsets,
                d_coo_cands,
                cand_col,
                d_is_positive,
                num_facts,
                eps_f32,
            )
            .map_err(|e| types::gpu_err("forward f32", e))?;

        let d_total_loss = provider
            .ilp_reduce_sum_f32_launch(&loss_contrib, num_facts)
            .map_err(|e| types::gpu_err("reduce f32", e))?;

        let d_grad = provider
            .ilp_credit_backward_f32_launch(
                d_row_offsets,
                d_coo_cands,
                &credit_out,
                d_is_positive,
                num_facts,
                num_cands,
            )
            .map_err(|e| types::gpu_err("backward f32", e))?;

        export_loss_grad_device(
            provider,
            py,
            d_total_loss,
            d_grad,
            num_cands,
            ScalarType::F32,
        )
    }
}

// ---------------------------------------------------------------------------
// DLPack export helpers (moved from plain impl CompiledIlpProgram)
// ---------------------------------------------------------------------------

/// Export device-resident loss scalar and grad vector as DLPack capsules.
///
/// Handles both f32 and f64 scalar types. Pass `scalar_type` matching the
/// element type `T` of the incoming slices.
pub(crate) fn export_loss_grad_device<T: GpuScalar>(
    provider: &CudaKernelProvider,
    py: Python<'_>,
    d_loss_val: TrackedCudaSlice<T>,
    d_grad: TrackedCudaSlice<T>,
    num_cands: u32,
    scalar_type: ScalarType,
) -> PyResult<(PyObject, PyObject)> {
    let schema = Schema::new(vec![("col_0".to_string(), scalar_type)]);

    let mut d_loss_nrows = provider
        .memory()
        .alloc::<u32>(1)
        .map_err(|e| types::gpu_err("alloc", e))?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[1u32], &mut d_loss_nrows)
        .map_err(|e| types::gpu_err("htod", e))?;

    let loss_buf = xlog_cuda::CudaBuffer::from_columns(
        vec![d_loss_val.into_bytes().into()],
        1,
        d_loss_nrows,
        schema.clone(),
    );
    let loss_dl = provider
        .to_dlpack_table(loss_buf)
        .column(0)
        .map_err(|e| types::gpu_err("DLPack loss", e))?;
    let loss_capsule = dlpack_capsule_from_tensor(py, loss_dl)?;

    let mut d_grad_nrows = provider
        .memory()
        .alloc::<u32>(1)
        .map_err(|e| types::gpu_err("alloc", e))?;
    provider
        .device()
        .inner()
        .htod_sync_copy_into(&[num_cands], &mut d_grad_nrows)
        .map_err(|e| types::gpu_err("htod", e))?;

    let grad_buf = xlog_cuda::CudaBuffer::from_columns(
        vec![d_grad.into_bytes().into()],
        num_cands as u64,
        d_grad_nrows,
        schema,
    );
    let grad_dl = provider
        .to_dlpack_table(grad_buf)
        .column(0)
        .map_err(|e| types::gpu_err("DLPack grad", e))?;
    let grad_capsule = dlpack_capsule_from_tensor(py, grad_dl)?;

    Ok((loss_capsule, grad_capsule))
}
