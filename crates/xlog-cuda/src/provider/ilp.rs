//! ILP (Inductive Logic Programming) kernel operations: credit/loss, COO fill, CSR histogram, reduce_sum.

use std::marker::PhantomData;

use cudarc::driver::{DevicePtr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, Schema, XlogError};

use super::{ilp_credit_kernels, ilp_kernels, RawCudaView, ILP_CREDIT_MODULE, ILP_MODULE};
use crate::memory::{CudaBuffer, CudaColumn, TrackedCudaSlice};

impl super::CudaKernelProvider {
    fn ilp_i32_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, i32>> {
        let required_bytes = num_elements * std::mem::size_of::<i32>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} i32 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if (ptr as usize) % std::mem::align_of::<i32>() != 0 {
            return Err(XlogError::Kernel(
                "Column device pointer is not i32-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            _marker: PhantomData,
        })
    }

    fn ilp_i64_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, i64>> {
        let required_bytes = num_elements * std::mem::size_of::<i64>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} i64 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if (ptr as usize) % std::mem::align_of::<i64>() != 0 {
            return Err(XlogError::Kernel(
                "Column device pointer is not i64-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            _marker: PhantomData,
        })
    }

    pub fn build_selected_id_mask(
        &self,
        ids_buf: &CudaBuffer,
        candidate_count: usize,
    ) -> Result<CudaBuffer> {
        let selected_len = usize::try_from(ids_buf.num_rows())
            .map_err(|_| XlogError::Kernel("selected id row count overflow".to_string()))?;
        let candidate_count_u32 = u32::try_from(candidate_count).map_err(|_| {
            XlogError::Kernel(format!(
                "candidate count {} exceeds u32::MAX for strict sparse mask",
                candidate_count
            ))
        })?;

        let mut active_flags = self.memory.alloc::<u32>(candidate_count)?;
        if candidate_count > 0 {
            self.device
                .inner()
                .memset_zeros(&mut active_flags)
                .map_err(|e| XlogError::Kernel(format!("zero strict sparse mask: {}", e)))?;
        }

        if selected_len > 0 {
            let selected_len_u32 = u32::try_from(selected_len).map_err(|_| {
                XlogError::Kernel(format!(
                    "selected id count {} exceeds u32::MAX for strict sparse mask",
                    selected_len
                ))
            })?;
            let block_size = 256u32;
            let grid_size = (selected_len_u32 + block_size - 1) / block_size;
            let ids_col = ids_buf
                .column(0)
                .ok_or_else(|| XlogError::Kernel("selected id buffer has no column".to_string()))?;
            match ids_buf.schema().column_type(0).ok_or_else(|| {
                XlogError::Kernel("selected id buffer has no schema type".to_string())
            })? {
                ScalarType::U32 | ScalarType::Symbol => {
                    let ids_view = self.column_as_u32_view(ids_col, selected_len)?;
                    let func = self
                        .device
                        .inner()
                        .get_func(ILP_MODULE, ilp_kernels::ILP_MARK_SELECTED_IDS_U32)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "ilp_mark_selected_ids_u32 kernel not found".to_string(),
                            )
                        })?;
                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        func.clone().launch(
                            LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ids_view,
                                selected_len_u32,
                                candidate_count_u32,
                                &mut active_flags,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "strict sparse selected-id scatter failed: {}",
                            e
                        ))
                    })?;
                }
                ScalarType::I32 => {
                    let ids_view = self.ilp_i32_view(ids_col, selected_len)?;
                    let func = self
                        .device
                        .inner()
                        .get_func(ILP_MODULE, ilp_kernels::ILP_MARK_SELECTED_IDS_I32)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "ilp_mark_selected_ids_i32 kernel not found".to_string(),
                            )
                        })?;
                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        func.clone().launch(
                            LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ids_view,
                                selected_len_u32,
                                candidate_count_u32,
                                &mut active_flags,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "strict sparse selected-id scatter failed: {}",
                            e
                        ))
                    })?;
                }
                ScalarType::I64 => {
                    let ids_view = self.ilp_i64_view(ids_col, selected_len)?;
                    let func = self
                        .device
                        .inner()
                        .get_func(ILP_MODULE, ilp_kernels::ILP_MARK_SELECTED_IDS_I64)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "ilp_mark_selected_ids_i64 kernel not found".to_string(),
                            )
                        })?;
                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        func.clone().launch(
                            LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ids_view,
                                selected_len_u32,
                                candidate_count_u32,
                                &mut active_flags,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "strict sparse selected-id scatter failed: {}",
                            e
                        ))
                    })?;
                }
                ScalarType::U64 => {
                    let ids_view = self.column_as_u64_view(ids_col, selected_len)?;
                    let func = self
                        .device
                        .inner()
                        .get_func(ILP_MODULE, ilp_kernels::ILP_MARK_SELECTED_IDS_U64)
                        .ok_or_else(|| {
                            XlogError::Kernel(
                                "ilp_mark_selected_ids_u64 kernel not found".to_string(),
                            )
                        })?;
                    // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                    unsafe {
                        func.clone().launch(
                            LaunchConfig {
                                grid_dim: (grid_size, 1, 1),
                                block_dim: (block_size, 1, 1),
                                shared_mem_bytes: 0,
                            },
                            (
                                &ids_view,
                                selected_len_u32,
                                candidate_count_u32,
                                &mut active_flags,
                            ),
                        )
                    }
                    .map_err(|e| {
                        XlogError::Kernel(format!(
                            "strict sparse selected-id scatter failed: {}",
                            e
                        ))
                    })?;
                }
                other => {
                    return Err(XlogError::Kernel(format!(
                        "selected candidate ids must be I32/I64/U32/U64, got {:?}",
                        other
                    )));
                }
            }

            self.device
                .synchronize()
                .map_err(|e| XlogError::Kernel(format!("strict sparse scatter sync: {}", e)))?;
        }

        let d_num_rows = self.upload_device_row_count(candidate_count_u32)?;
        Ok(CudaBuffer::from_columns_with_host_count(
            vec![active_flags.into_bytes().into()],
            candidate_count as u64,
            d_num_rows,
            Schema::new(vec![("active".to_string(), ScalarType::U32)]),
            candidate_count_u32,
        ))
    }

    pub fn validate_selected_ids(
        &self,
        ids_buf: &CudaBuffer,
        candidate_count: usize,
    ) -> Result<()> {
        let selected_len = usize::try_from(ids_buf.num_rows())
            .map_err(|_| XlogError::Kernel("selected id row count overflow".to_string()))?;
        let candidate_count_u32 = u32::try_from(candidate_count).map_err(|_| {
            XlogError::Kernel(format!(
                "candidate count {} exceeds u32::MAX for strict sparse mask",
                candidate_count
            ))
        })?;

        if selected_len == 0 {
            return Ok(());
        }

        let selected_len_u32 = u32::try_from(selected_len).map_err(|_| {
            XlogError::Kernel(format!(
                "selected id count {} exceeds u32::MAX for strict sparse mask",
                selected_len
            ))
        })?;
        let block_size = 256u32;
        let grid_size = (selected_len_u32 + block_size - 1) / block_size;
        let ids_col = ids_buf
            .column(0)
            .ok_or_else(|| XlogError::Kernel("selected id buffer has no column".to_string()))?;

        let mut seen_flags = self.memory.alloc::<u32>(candidate_count)?;
        if candidate_count > 0 {
            self.device
                .inner()
                .memset_zeros(&mut seen_flags)
                .map_err(|e| {
                    XlogError::Kernel(format!("zero strict sparse validation flags: {}", e))
                })?;
        }

        let mut error_code = self.memory.alloc::<u32>(1)?;
        let mut error_pos = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .memset_zeros(&mut error_code)
            .map_err(|e| XlogError::Kernel(format!("zero strict sparse error code: {}", e)))?;
        self.device
            .inner()
            .memset_zeros(&mut error_pos)
            .map_err(|e| XlogError::Kernel(format!("zero strict sparse error pos: {}", e)))?;

        match ids_buf.schema().column_type(0).ok_or_else(|| {
            XlogError::Kernel("selected id buffer has no schema type".to_string())
        })? {
            ScalarType::U32 | ScalarType::Symbol => {
                let ids_view = self.column_as_u32_view(ids_col, selected_len)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ILP_MODULE, ilp_kernels::ILP_VALIDATE_SELECTED_IDS_U32)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "ilp_validate_selected_ids_u32 kernel not found".to_string(),
                        )
                    })?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (grid_size, 1, 1),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ids_view,
                            selected_len_u32,
                            candidate_count_u32,
                            &mut seen_flags,
                            &mut error_code,
                            &mut error_pos,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("strict sparse selected-id validation failed: {}", e))
                })?;
            }
            ScalarType::I32 => {
                let ids_view = self.ilp_i32_view(ids_col, selected_len)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ILP_MODULE, ilp_kernels::ILP_VALIDATE_SELECTED_IDS_I32)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "ilp_validate_selected_ids_i32 kernel not found".to_string(),
                        )
                    })?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (grid_size, 1, 1),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ids_view,
                            selected_len_u32,
                            candidate_count_u32,
                            &mut seen_flags,
                            &mut error_code,
                            &mut error_pos,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("strict sparse selected-id validation failed: {}", e))
                })?;
            }
            ScalarType::I64 => {
                let ids_view = self.ilp_i64_view(ids_col, selected_len)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ILP_MODULE, ilp_kernels::ILP_VALIDATE_SELECTED_IDS_I64)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "ilp_validate_selected_ids_i64 kernel not found".to_string(),
                        )
                    })?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (grid_size, 1, 1),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ids_view,
                            selected_len_u32,
                            candidate_count_u32,
                            &mut seen_flags,
                            &mut error_code,
                            &mut error_pos,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("strict sparse selected-id validation failed: {}", e))
                })?;
            }
            ScalarType::U64 => {
                let ids_view = self.column_as_u64_view(ids_col, selected_len)?;
                let func = self
                    .device
                    .inner()
                    .get_func(ILP_MODULE, ilp_kernels::ILP_VALIDATE_SELECTED_IDS_U64)
                    .ok_or_else(|| {
                        XlogError::Kernel(
                            "ilp_validate_selected_ids_u64 kernel not found".to_string(),
                        )
                    })?;
                // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (grid_size, 1, 1),
                            block_dim: (block_size, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (
                            &ids_view,
                            selected_len_u32,
                            candidate_count_u32,
                            &mut seen_flags,
                            &mut error_code,
                            &mut error_pos,
                        ),
                    )
                }
                .map_err(|e| {
                    XlogError::Kernel(format!("strict sparse selected-id validation failed: {}", e))
                })?;
            }
            other => {
                return Err(XlogError::Kernel(format!(
                    "selected candidate ids must be I32/I64/U32/U64, got {:?}",
                    other
                )));
            }
        }

        self.device
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("strict sparse validation sync: {}", e)))?;

        let error_code_host = self.dtoh_scalar_untracked(&error_code, 0)?;
        if error_code_host == 0 {
            return Ok(());
        }
        let error_pos_host = self.dtoh_scalar_untracked(&error_pos, 0)?;
        match error_code_host {
            1 => Err(XlogError::Kernel(format!(
                "selected candidate id out of range at position {}",
                error_pos_host
            ))),
            2 => Err(XlogError::Kernel(format!(
                "duplicate selected candidate id at position {}",
                error_pos_host
            ))),
            code => Err(XlogError::Kernel(format!(
                "strict sparse selected-id validation failed with error code {}",
                code
            ))),
        }
    }

    pub fn filter_buffer_by_candidate_flag(
        &self,
        input: &CudaBuffer,
        candidate_flags: &CudaBuffer,
        candidate_idx: usize,
    ) -> Result<CudaBuffer> {
        if input.is_empty() {
            return self.create_empty_buffer(input.schema().clone());
        }
        if candidate_idx >= candidate_flags.num_rows() as usize {
            return Err(XlogError::Kernel(format!(
                "candidate flag index {} out of range [0, {})",
                candidate_idx,
                candidate_flags.num_rows()
            )));
        }

        let flag_col = candidate_flags
            .column(0)
            .ok_or_else(|| XlogError::Kernel("candidate flag buffer has no column".to_string()))?;
        let flag_view = self.column_as_u32_view(flag_col, candidate_flags.num_rows() as usize)?;
        let row_count = u32::try_from(input.num_rows()).map_err(|_| {
            XlogError::Kernel(format!(
                "strict sparse row count {} exceeds u32::MAX",
                input.num_rows()
            ))
        })?;
        let candidate_idx_u32 = u32::try_from(candidate_idx).map_err(|_| {
            XlogError::Kernel(format!(
                "candidate flag index {} exceeds u32::MAX",
                candidate_idx
            ))
        })?;

        let mut row_mask = self.memory.alloc::<u8>(row_count as usize)?;
        let func = self
            .device
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::ILP_BROADCAST_CANDIDATE_FLAG)
            .ok_or_else(|| {
                XlogError::Kernel("ilp_broadcast_candidate_flag kernel not found".to_string())
            })?;
        let block_size = 256u32;
        let grid_size = (row_count + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&flag_view, candidate_idx_u32, row_count, &mut row_mask),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("strict sparse flag broadcast failed: {}", e)))?;

        self.filter_by_device_mask(input, &row_mask)
    }

    // ─── ILP credit kernel launchers ───────────────────────────────────

    /// Launch `ilp_coo_fill` kernel: writes `(compacted_fact_indices[i], cidx)`
    /// pairs at `coo_fact[offset..]` and `coo_cand[offset..]`.
    pub fn ilp_coo_fill_launch(
        &self,
        compacted_fact_indices: &TrackedCudaSlice<u32>,
        cidx: u32,
        count: u32,
        offset: u32,
        coo_fact: &mut TrackedCudaSlice<u32>,
        coo_cand: &mut TrackedCudaSlice<u32>,
    ) -> Result<()> {
        if count == 0 {
            return Ok(());
        }
        let func = self
            .device
            .inner()
            .get_func(ILP_CREDIT_MODULE, ilp_credit_kernels::ILP_COO_FILL)
            .ok_or_else(|| XlogError::Kernel("ilp_coo_fill kernel not found".to_string()))?;
        let block_size = 256u32;
        let grid_size = (count + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    compacted_fact_indices,
                    cidx,
                    count,
                    offset,
                    coo_fact,
                    coo_cand,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_coo_fill failed: {}", e)))?;
        self.device.synchronize()?;
        Ok(())
    }

    /// Launch `ilp_credit_forward_f32`: CSR credit gather + clamp + NLL loss.
    /// Returns `(credit_out, loss_contrib)` device slices of length `num_facts`.
    pub fn ilp_credit_forward_f32_launch(
        &self,
        row_offsets: &TrackedCudaSlice<u32>,
        col_indices: &TrackedCudaSlice<u32>,
        cand_probs: &CudaColumn, // raw byte column from CudaBuffer
        is_positive: &TrackedCudaSlice<u8>,
        num_facts: u32,
        eps: f32,
    ) -> Result<(TrackedCudaSlice<f32>, TrackedCudaSlice<f32>)> {
        let mut credit_out = self.memory.alloc::<f32>(num_facts as usize)?;
        let mut loss_contrib = self.memory.alloc::<f32>(num_facts as usize)?;
        if num_facts == 0 {
            return Ok((credit_out, loss_contrib));
        }
        let func = self
            .device
            .inner()
            .get_func(
                ILP_CREDIT_MODULE,
                ilp_credit_kernels::ILP_CREDIT_FORWARD_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("ilp_credit_forward_f32 kernel not found".to_string())
            })?;
        let block_size = 256u32;
        let grid_size = (num_facts + block_size - 1) / block_size;
        // reinterpret the u8 byte column as f32 for the kernel
        let cand_view = RawCudaView::<f32> {
            ptr: *cudarc::driver::DevicePtr::device_ptr(cand_probs),
            len: cudarc::driver::DeviceSlice::len(cand_probs) / 4,
            _marker: PhantomData,
        };
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    row_offsets,
                    col_indices,
                    &cand_view,
                    is_positive,
                    num_facts,
                    eps,
                    &mut credit_out,
                    &mut loss_contrib,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_credit_forward_f32 failed: {}", e)))?;
        self.device.synchronize()?;
        Ok((credit_out, loss_contrib))
    }

    /// Launch `ilp_credit_forward_f64`: CSR credit gather + clamp + NLL loss.
    /// Returns `(credit_out, loss_contrib)` device slices of length `num_facts`.
    pub fn ilp_credit_forward_f64_launch(
        &self,
        row_offsets: &TrackedCudaSlice<u32>,
        col_indices: &TrackedCudaSlice<u32>,
        cand_probs: &CudaColumn, // raw byte column from CudaBuffer
        is_positive: &TrackedCudaSlice<u8>,
        num_facts: u32,
        eps: f64,
    ) -> Result<(TrackedCudaSlice<f64>, TrackedCudaSlice<f64>)> {
        let mut credit_out = self.memory.alloc::<f64>(num_facts as usize)?;
        let mut loss_contrib = self.memory.alloc::<f64>(num_facts as usize)?;
        if num_facts == 0 {
            return Ok((credit_out, loss_contrib));
        }
        let func = self
            .device
            .inner()
            .get_func(
                ILP_CREDIT_MODULE,
                ilp_credit_kernels::ILP_CREDIT_FORWARD_F64,
            )
            .ok_or_else(|| {
                XlogError::Kernel("ilp_credit_forward_f64 kernel not found".to_string())
            })?;
        let block_size = 256u32;
        let grid_size = (num_facts + block_size - 1) / block_size;
        let cand_view = RawCudaView::<f64> {
            ptr: *cudarc::driver::DevicePtr::device_ptr(cand_probs),
            len: cudarc::driver::DeviceSlice::len(cand_probs) / 8,
            _marker: PhantomData,
        };
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    row_offsets,
                    col_indices,
                    &cand_view,
                    is_positive,
                    num_facts,
                    eps,
                    &mut credit_out,
                    &mut loss_contrib,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_credit_forward_f64 failed: {}", e)))?;
        self.device.synchronize()?;
        Ok((credit_out, loss_contrib))
    }

    /// Launch `ilp_credit_backward_f32`: gradient scatter via CSR + atomicAdd.
    /// Returns `d_cand_probs` gradient of length `num_cands` (zeroed, then accumulated).
    pub fn ilp_credit_backward_f32_launch(
        &self,
        row_offsets: &TrackedCudaSlice<u32>,
        col_indices: &TrackedCudaSlice<u32>,
        credit_out: &TrackedCudaSlice<f32>,
        is_positive: &TrackedCudaSlice<u8>,
        num_facts: u32,
        num_cands: u32,
    ) -> Result<TrackedCudaSlice<f32>> {
        let mut d_grad = self.memory.alloc::<f32>(num_cands as usize)?;
        self.device
            .inner()
            .memset_zeros(&mut d_grad)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad: {}", e)))?;
        if num_facts == 0 {
            return Ok(d_grad);
        }
        let func = self
            .device
            .inner()
            .get_func(
                ILP_CREDIT_MODULE,
                ilp_credit_kernels::ILP_CREDIT_BACKWARD_F32,
            )
            .ok_or_else(|| {
                XlogError::Kernel("ilp_credit_backward_f32 kernel not found".to_string())
            })?;
        let block_size = 256u32;
        let grid_size = (num_facts + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    row_offsets,
                    col_indices,
                    credit_out,
                    is_positive,
                    num_facts,
                    &mut d_grad,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_credit_backward_f32 failed: {}", e)))?;
        self.device.synchronize()?;
        Ok(d_grad)
    }

    /// Launch `ilp_credit_backward_f64`: gradient scatter via CSR + atomicAdd.
    /// Returns `d_cand_probs` gradient of length `num_cands` (zeroed, then accumulated).
    pub fn ilp_credit_backward_f64_launch(
        &self,
        row_offsets: &TrackedCudaSlice<u32>,
        col_indices: &TrackedCudaSlice<u32>,
        credit_out: &TrackedCudaSlice<f64>,
        is_positive: &TrackedCudaSlice<u8>,
        num_facts: u32,
        num_cands: u32,
    ) -> Result<TrackedCudaSlice<f64>> {
        let mut d_grad = self.memory.alloc::<f64>(num_cands as usize)?;
        self.device
            .inner()
            .memset_zeros(&mut d_grad)
            .map_err(|e| XlogError::Kernel(format!("Failed to zero grad: {}", e)))?;
        if num_facts == 0 {
            return Ok(d_grad);
        }
        let func = self
            .device
            .inner()
            .get_func(
                ILP_CREDIT_MODULE,
                ilp_credit_kernels::ILP_CREDIT_BACKWARD_F64,
            )
            .ok_or_else(|| {
                XlogError::Kernel("ilp_credit_backward_f64 kernel not found".to_string())
            })?;
        let block_size = 256u32;
        let grid_size = (num_facts + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    row_offsets,
                    col_indices,
                    credit_out,
                    is_positive,
                    num_facts,
                    &mut d_grad,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_credit_backward_f64 failed: {}", e)))?;
        self.device.synchronize()?;
        Ok(d_grad)
    }

    /// GPU-side sum reduction (f32).
    ///
    /// Sums `n` elements of `input` on device and returns a single-element
    /// device buffer containing the result.  The caller must zero the output
    /// buffer *before* launching the kernel — this function handles that.
    pub fn ilp_reduce_sum_f32_launch(
        &self,
        input: &TrackedCudaSlice<f32>,
        n: u32,
    ) -> Result<TrackedCudaSlice<f32>> {
        let mut d_result = self.memory.alloc::<f32>(1)?;
        self.device
            .inner()
            .htod_sync_copy_into(&[0.0f32], &mut d_result)
            .map_err(|e| XlogError::Kernel(format!("ilp_reduce_sum_f32 zero result: {}", e)))?;

        if n == 0 {
            return Ok(d_result);
        }

        let func = self
            .device
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::ILP_REDUCE_SUM_F32)
            .ok_or_else(|| XlogError::Kernel("ilp_reduce_sum_f32 not found".to_string()))?;
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (input, n, &mut d_result),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_reduce_sum_f32: {}", e)))?;
        self.device.synchronize()?;
        Ok(d_result)
    }

    /// GPU-side sum reduction (f64).
    ///
    /// Sums `n` elements of `input` on device and returns a single-element
    /// device buffer containing the result.  Requires sm_60+ for double
    /// atomicAdd (this project targets sm_75 baseline).
    pub fn ilp_reduce_sum_f64_launch(
        &self,
        input: &TrackedCudaSlice<f64>,
        n: u32,
    ) -> Result<TrackedCudaSlice<f64>> {
        let mut d_result = self.memory.alloc::<f64>(1)?;
        self.device
            .inner()
            .htod_sync_copy_into(&[0.0f64], &mut d_result)
            .map_err(|e| XlogError::Kernel(format!("ilp_reduce_sum_f64 zero result: {}", e)))?;

        if n == 0 {
            return Ok(d_result);
        }

        let func = self
            .device
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::ILP_REDUCE_SUM_F64)
            .ok_or_else(|| XlogError::Kernel("ilp_reduce_sum_f64 not found".to_string()))?;
        let block_size = 256u32;
        let grid_size = (n + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (input, n, &mut d_result),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_reduce_sum_f64: {}", e)))?;
        self.device.synchronize()?;
        Ok(d_result)
    }

    /// Fill COO arrays from a device-side mask and prefix-sum.
    ///
    /// For each set bit in `mask`, writes the corresponding `fact_indices` entry
    /// into `coo_fact` and `cand_value` into `coo_cand` at the position
    /// determined by `d_offsets[offset_idx] + prefix_sum[tid]`.
    ///
    /// Parameters:
    /// - `offset_idx`: index into `d_offsets` for the write base position
    /// - `cand_value`: actual candidate index to write into `coo_cand`
    ///
    /// This keeps COO assembly fully on device, eliminating the mask D2H transfer.
    pub fn ilp_coo_fill_from_mask_launch(
        &self,
        mask: &TrackedCudaSlice<u8>,
        prefix_sum: &TrackedCudaSlice<u32>,
        fact_indices: &TrackedCudaSlice<u32>,
        offset_idx: u32,
        cand_value: u32,
        num_query: u32,
        d_offsets: &TrackedCudaSlice<u32>,
        coo_fact: &mut TrackedCudaSlice<u32>,
        coo_cand: &mut TrackedCudaSlice<u32>,
    ) -> Result<()> {
        if num_query == 0 {
            return Ok(());
        }
        let func = self
            .device()
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::ILP_COO_FILL_FROM_MASK)
            .ok_or_else(|| XlogError::Kernel("ilp_coo_fill_from_mask not found".to_string()))?;
        let block_size = 256u32;
        let grid_size = (num_query + block_size - 1) / block_size;
        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_size, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    mask,
                    prefix_sum,
                    fact_indices,
                    offset_idx,
                    cand_value,
                    num_query,
                    d_offsets,
                    coo_fact,
                    coo_cand,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("ilp_coo_fill_from_mask: {}", e)))?;
        self.device()
            .inner()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("ilp_coo_fill_from_mask sync: {}", e)))?;
        Ok(())
    }

    /// Build a histogram of fact indices from sorted COO data.
    ///
    /// For each entry in `sorted_facts[0..nnz]`, atomically increments
    /// the corresponding bin in the output histogram. The result is a
    /// device-side count array of length `num_facts`, suitable for
    /// prefix-sum to produce CSR `row_offsets`.
    ///
    /// The caller provides sorted fact indices; the histogram is
    /// zero-initialized internally.
    pub fn ilp_csr_histogram_launch(
        &self,
        sorted_facts: &TrackedCudaSlice<u32>,
        nnz: u32,
        num_facts: u32,
    ) -> Result<TrackedCudaSlice<u32>> {
        let mut d_hist = self.memory().alloc::<u32>(num_facts as usize)?;
        // Zero the histogram
        let zeros = vec![0u32; num_facts as usize];
        self.device()
            .inner()
            .htod_sync_copy_into(&zeros, &mut d_hist)
            .map_err(|e| XlogError::Kernel(format!("ilp_csr_histogram zero hist: {}", e)))?;

        if nnz == 0 {
            return Ok(d_hist);
        }

        let func = self
            .device()
            .inner()
            .get_func(ILP_MODULE, ilp_kernels::ILP_CSR_HISTOGRAM)
            .ok_or_else(|| XlogError::Kernel("ilp_csr_histogram kernel not found".to_string()))?;

        let block_size = 256u32;
        let grid_size = (nnz + block_size - 1) / block_size;

        // SAFETY: kernel arguments match the PTX signature; device buffers were allocated with sufficient size
        unsafe {
            func.clone()
                .launch(
                    cudarc::driver::LaunchConfig {
                        grid_dim: (grid_size, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (sorted_facts, nnz, num_facts, &mut d_hist),
                )
                .map_err(|e| XlogError::Kernel(format!("ilp_csr_histogram launch: {}", e)))?;
        }

        self.device()
            .inner()
            .synchronize()
            .map_err(|e| XlogError::Kernel(format!("ilp_csr_histogram sync: {}", e)))?;

        Ok(d_hist)
    }
}
