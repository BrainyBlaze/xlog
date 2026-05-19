//! Launcher for the M8 Phase 1 bounded exact-induction kernel.
//!
//! Drives `kernels/ilp_exact.cu`'s `ilp_exact_score` kernel: scores all
//! `(topology, L, R)` triples for a single `induce_exact` call in one
//! launch and returns the positive/negative coverage count arrays to host.
//!
//! Design: `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md`.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use crate::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::{ilp_exact_kernels, RawCudaView, ILP_EXACT_MODULE};
use crate::memory::CudaBuffer;

const ILP_EXACT_BLOCK_SIZE: u32 = 256;
const ENV_ILP_EXACT_CHAIN_SMEM: &str = "XLOG_ILP_EXACT_CHAIN_SMEM";
const ENV_ILP_EXACT_CHAIN_SMEM_MIN_ROWS: &str = "XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS";
const DEFAULT_ILP_EXACT_CHAIN_SMEM_MIN_ROWS: u32 = 256;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ExactPairLayout {
    U64,
    U32,
    Symbol,
}

impl ExactPairLayout {
    fn elem_size(self) -> usize {
        match self {
            Self::U64 => std::mem::size_of::<u64>(),
            Self::U32 | Self::Symbol => std::mem::size_of::<u32>(),
        }
    }
}

fn ilp_exact_chain_smem_enabled() -> bool {
    match std::env::var(ENV_ILP_EXACT_CHAIN_SMEM) {
        Ok(value) => !matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "0" | "false" | "off" | "no"
        ),
        Err(_) => true,
    }
}

fn chain_smem_shared_bytes(layout: ExactPairLayout) -> u32 {
    let block = ILP_EXACT_BLOCK_SIZE as usize;
    let bytes = (2usize * block * layout.elem_size()) + (block * std::mem::size_of::<u32>());
    u32::try_from(bytes).expect("chain smem byte count fits in u32")
}

fn ilp_exact_chain_smem_min_rows() -> u32 {
    std::env::var(ENV_ILP_EXACT_CHAIN_SMEM_MIN_ROWS)
        .ok()
        .and_then(|value| value.trim().parse::<u32>().ok())
        .unwrap_or(DEFAULT_ILP_EXACT_CHAIN_SMEM_MIN_ROWS)
}

impl super::CudaKernelProvider {
    /// Score every `(topology, L, R)` triple for one `induce_exact` call.
    ///
    /// Returns `(pos_covered, neg_covered)`, each of length `4 * C * C`
    /// where `C = candidate_buffers.len()`. Slot ordering:
    /// `slot = topology * (C * C) + L * C + R`, with topology indices
    /// `chain=0, star=1, fanout=2, fanin=3`.
    ///
    /// Host-side contract:
    ///   * All buffers must be arity 2 with one matching pair type: `U64`,
    ///     `U32`, or `Symbol`.
    ///   * `cached_row_count()` must be populated on every buffer (DLPack
    ///     ingest and `create_empty_buffer` both guarantee this).
    ///   * `negatives` is always a valid buffer — the caller constructs
    ///     an empty pair buffer matching the positive pair type when there are
    ///     no negatives.
    ///
    /// D2H budget: **2** counter-tracked transfers (one per count array).
    /// Setup H2D / D2D copies are not D2H-counted.
    pub fn ilp_exact_score(
        &self,
        candidate_buffers: &[&CudaBuffer],
        positives: &CudaBuffer,
        negatives: &CudaBuffer,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let c = candidate_buffers.len();
        if c == 0 {
            return Err(XlogError::Kernel(
                "ilp_exact_score: candidate list is empty (filter at the engine)".to_string(),
            ));
        }
        let c_u32 = u32::try_from(c).map_err(|_| {
            XlogError::Kernel(format!(
                "ilp_exact_score: candidate count {} exceeds u32::MAX",
                c
            ))
        })?;

        // ── Validate shapes and gather host-side row counts ────────────────
        let layout = validate_exact_pair_buffer(positives, "positives")?;
        require_exact_pair_layout(negatives, "negatives", layout)?;
        let pos_rows = cached_rows(positives, "positives")?;
        let neg_rows = cached_rows(negatives, "negatives")?;

        let mut cand_rows: Vec<u32> = Vec::with_capacity(c);
        for (i, buf) in candidate_buffers.iter().enumerate() {
            let label = format!("candidate[{}]", i);
            require_exact_pair_layout(buf, &label, layout)?;
            cand_rows.push(cached_rows(buf, &label)?);
        }

        // ── Exclusive prefix sum of row counts (cand_offsets, length C+1) ─
        let mut cand_offsets_host: Vec<u32> = Vec::with_capacity(c + 1);
        let mut running: u32 = 0;
        cand_offsets_host.push(0);
        for &r in &cand_rows {
            running = running.checked_add(r).ok_or_else(|| {
                XlogError::Kernel("ilp_exact_score: candidate row count overflow u32".to_string())
            })?;
            cand_offsets_host.push(running);
        }
        let total_rows = running as usize;
        let elem_size = layout.elem_size();
        let total_bytes = total_rows * elem_size;

        let device = self.device.inner();

        // ── Concatenate candidate columns via D2D copies ──────────────────
        // Setup-phase D→D; neither counted by the D2H gate nor by the
        // transfer tracker as a host-to-device round trip.
        let mut cand_arg0_buf = self.memory.alloc::<u8>(total_bytes)?;
        let mut cand_arg1_buf = self.memory.alloc::<u8>(total_bytes)?;
        if total_bytes > 0 {
            let mut byte_offset: usize = 0;
            for (i, buf) in candidate_buffers.iter().enumerate() {
                let rows = cand_rows[i] as usize;
                if rows == 0 {
                    continue;
                }
                let bytes = rows * elem_size;

                let src0 = buf.column(0).ok_or_else(|| {
                    XlogError::Kernel(format!("candidate[{}] missing column 0", i))
                })?;
                let src1 = buf.column(1).ok_or_else(|| {
                    XlogError::Kernel(format!("candidate[{}] missing column 1", i))
                })?;
                let src_view0 = self.column_bytes_view(src0, bytes)?;
                let src_view1 = self.column_bytes_view(src1, bytes)?;
                let mut dst0 = cand_arg0_buf.slice_mut(byte_offset..byte_offset + bytes);
                let mut dst1 = cand_arg1_buf.slice_mut(byte_offset..byte_offset + bytes);
                device.dtod_copy(&src_view0, &mut dst0).map_err(|e| {
                    XlogError::Kernel(format!(
                        "ilp_exact_score: d2d concat arg0 (candidate {}): {}",
                        i, e
                    ))
                })?;
                device.dtod_copy(&src_view1, &mut dst1).map_err(|e| {
                    XlogError::Kernel(format!(
                        "ilp_exact_score: d2d concat arg1 (candidate {}): {}",
                        i, e
                    ))
                })?;
                byte_offset += bytes;
            }
        }

        // ── Upload cand_offsets (H→D, not D2H-counted) ────────────────────
        let mut cand_offsets_buf = self.memory.alloc::<u32>(c + 1)?;
        device
            .htod_sync_copy_into(&cand_offsets_host, &mut cand_offsets_buf)
            .map_err(|e| XlogError::Kernel(format!("ilp_exact_score: h2d cand_offsets: {}", e)))?;

        // ── Alloc output count arrays ─────────────────────────────────────
        let n_slots = 4usize
            .checked_mul(c)
            .and_then(|v| v.checked_mul(c))
            .ok_or_else(|| {
                XlogError::Kernel("ilp_exact_score: n_slots = 4 * C * C overflow".to_string())
            })?;
        let mut pos_covered_buf = self.memory.alloc::<u32>(n_slots)?;
        let mut neg_covered_buf = self.memory.alloc::<u32>(n_slots)?;
        // Kernel writes every slot exactly once — no zero-init required.

        let pos_col0 = positives
            .column(0)
            .ok_or_else(|| XlogError::Kernel("positives: missing column 0".to_string()))?;
        let pos_col1 = positives
            .column(1)
            .ok_or_else(|| XlogError::Kernel("positives: missing column 1".to_string()))?;
        let neg_col0 = negatives
            .column(0)
            .ok_or_else(|| XlogError::Kernel("negatives: missing column 0".to_string()))?;
        let neg_col1 = negatives
            .column(1)
            .ok_or_else(|| XlogError::Kernel("negatives: missing column 1".to_string()))?;

        // ── Launch ────────────────────────────────────────────────────────
        let max_candidate_rows = cand_rows.iter().copied().max().unwrap_or(0);
        let chain_smem_enabled =
            ilp_exact_chain_smem_enabled() && max_candidate_rows >= ilp_exact_chain_smem_min_rows();
        let shared_mem_bytes = if chain_smem_enabled {
            chain_smem_shared_bytes(layout)
        } else {
            0
        };
        match layout {
            ExactPairLayout::U64 => {
                let cand_arg0_view = RawCudaView::<u64> {
                    ptr: *cand_arg0_buf.device_ptr(),
                    len: total_rows,
                    stream: cand_arg0_buf.stream().clone(),
                    source_block: None,
                    _marker: PhantomData,
                };
                let cand_arg1_view = RawCudaView::<u64> {
                    ptr: *cand_arg1_buf.device_ptr(),
                    len: total_rows,
                    stream: cand_arg1_buf.stream().clone(),
                    source_block: None,
                    _marker: PhantomData,
                };
                let pos_arg0_view = self.column_as_u64_view(pos_col0, pos_rows as usize)?;
                let pos_arg1_view = self.column_as_u64_view(pos_col1, pos_rows as usize)?;
                let neg_arg0_view = self.column_as_u64_view(neg_col0, neg_rows as usize)?;
                let neg_arg1_view = self.column_as_u64_view(neg_col1, neg_rows as usize)?;
                let kernel_name = if chain_smem_enabled {
                    ilp_exact_kernels::ILP_EXACT_SCORE_CHAIN_SMEM
                } else {
                    ilp_exact_kernels::ILP_EXACT_SCORE
                };
                let func = device
                    .get_func(ILP_EXACT_MODULE, kernel_name)
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{} kernel not loaded", kernel_name))
                    })?;
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (c_u32, c_u32, 4),
                            block_dim: (ILP_EXACT_BLOCK_SIZE, 1, 1),
                            shared_mem_bytes,
                        },
                        (
                            &cand_arg0_view,
                            &cand_arg1_view,
                            &cand_offsets_buf,
                            c_u32,
                            &pos_arg0_view,
                            &pos_arg1_view,
                            pos_rows,
                            &neg_arg0_view,
                            &neg_arg1_view,
                            neg_rows,
                            &mut pos_covered_buf,
                            &mut neg_covered_buf,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("ilp_exact_score launch: {}", e)))?;
            }
            ExactPairLayout::U32 | ExactPairLayout::Symbol => {
                let cand_arg0_view = RawCudaView::<u32> {
                    ptr: *cand_arg0_buf.device_ptr(),
                    len: total_rows,
                    stream: cand_arg0_buf.stream().clone(),
                    source_block: None,
                    _marker: PhantomData,
                };
                let cand_arg1_view = RawCudaView::<u32> {
                    ptr: *cand_arg1_buf.device_ptr(),
                    len: total_rows,
                    stream: cand_arg1_buf.stream().clone(),
                    source_block: None,
                    _marker: PhantomData,
                };
                let pos_arg0_view = self.column_as_u32_view(pos_col0, pos_rows as usize)?;
                let pos_arg1_view = self.column_as_u32_view(pos_col1, pos_rows as usize)?;
                let neg_arg0_view = self.column_as_u32_view(neg_col0, neg_rows as usize)?;
                let neg_arg1_view = self.column_as_u32_view(neg_col1, neg_rows as usize)?;
                let kernel_name = if chain_smem_enabled {
                    ilp_exact_kernels::ILP_EXACT_SCORE_CHAIN_SMEM_U32
                } else {
                    ilp_exact_kernels::ILP_EXACT_SCORE_U32
                };
                let func = device
                    .get_func(ILP_EXACT_MODULE, kernel_name)
                    .ok_or_else(|| {
                        XlogError::Kernel(format!("{} kernel not loaded", kernel_name))
                    })?;
                unsafe {
                    func.clone().launch(
                        LaunchConfig {
                            grid_dim: (c_u32, c_u32, 4),
                            block_dim: (ILP_EXACT_BLOCK_SIZE, 1, 1),
                            shared_mem_bytes,
                        },
                        (
                            &cand_arg0_view,
                            &cand_arg1_view,
                            &cand_offsets_buf,
                            c_u32,
                            &pos_arg0_view,
                            &pos_arg1_view,
                            pos_rows,
                            &neg_arg0_view,
                            &neg_arg1_view,
                            neg_rows,
                            &mut pos_covered_buf,
                            &mut neg_covered_buf,
                        ),
                    )
                }
                .map_err(|e| XlogError::Kernel(format!("ilp_exact_score_u32 launch: {}", e)))?;
            }
        }

        self.device.synchronize()?;

        // ── Download outputs ──────────────────────────────────────────────
        // Two D→H transfers, counted in the D2H gate. Each increments by 1;
        // total 2 regardless of candidate count — well within the test's
        // `large ≤ small + 2` slack.
        let mut pos_covered = vec![0u32; n_slots];
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        device
            .dtoh_sync_copy_into(&pos_covered_buf, &mut pos_covered)
            .map_err(|e| XlogError::Kernel(format!("ilp_exact_score: dtoh pos_covered: {}", e)))?;

        let mut neg_covered = vec![0u32; n_slots];
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        device
            .dtoh_sync_copy_into(&neg_covered_buf, &mut neg_covered)
            .map_err(|e| XlogError::Kernel(format!("ilp_exact_score: dtoh neg_covered: {}", e)))?;

        Ok((pos_covered, neg_covered))
    }
}

fn validate_exact_pair_buffer(buf: &CudaBuffer, label: &str) -> Result<ExactPairLayout> {
    if buf.arity() != 2 {
        return Err(XlogError::Kernel(format!(
            "ilp_exact_score: {} buffer arity = {}, expected 2",
            label,
            buf.arity(),
        )));
    }
    let mut layout: Option<ExactPairLayout> = None;
    for col_idx in 0..2 {
        let t = buf.schema().column_type(col_idx).ok_or_else(|| {
            XlogError::Kernel(format!(
                "ilp_exact_score: {} buffer missing column {} type",
                label, col_idx,
            ))
        })?;
        let col_layout = match t {
            ScalarType::U64 => ExactPairLayout::U64,
            ScalarType::U32 => ExactPairLayout::U32,
            ScalarType::Symbol => ExactPairLayout::Symbol,
            _ => {
                return Err(XlogError::Kernel(format!(
                    "ilp_exact_score: {} buffer column {} type = {:?}, expected U64, U32, or Symbol",
                    label, col_idx, t,
                )));
            }
        };
        if let Some(expected) = layout {
            if expected != col_layout {
                return Err(XlogError::Kernel(format!(
                    "ilp_exact_score: {} buffer column {} type mismatch: {:?} vs {:?}",
                    label, col_idx, expected, col_layout,
                )));
            }
        } else {
            layout = Some(col_layout);
        }
    }
    Ok(layout.expect("arity 2 loop sets layout"))
}

fn require_exact_pair_layout(
    buf: &CudaBuffer,
    label: &str,
    expected: ExactPairLayout,
) -> Result<()> {
    let actual = validate_exact_pair_buffer(buf, label)?;
    if actual != expected {
        return Err(XlogError::Kernel(format!(
            "ilp_exact_score: {} buffer type mismatch: expected {:?}, got {:?}",
            label, expected, actual,
        )));
    }
    Ok(())
}

fn cached_rows(buf: &CudaBuffer, label: &str) -> Result<u32> {
    buf.cached_row_count().ok_or_else(|| {
        XlogError::Kernel(format!(
            "ilp_exact_score: {} buffer has no cached row count \
             (DLPack ingest and create_empty_buffer both populate it)",
            label
        ))
    })
}

#[cfg(test)]
mod tests {
    //! CUDA-gated correctness tests for the ilp_exact launcher.
    //!
    //! Pinned to a hand-computed fixture so the kernel's coverage arithmetic
    //! can be verified without relying on the Python backend as oracle. The
    //! fixture uses C=2 candidate relations so the expected flat output
    //! (4 × C × C = 16 slots per count array) is tractable to enumerate.

    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType, Schema};

    use crate::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    fn make_provider() -> Option<CudaKernelProvider> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    /// Build a `(u64, u64)` pair buffer from parallel host-side column arrays.
    /// Uses `create_buffer_from_slice` per column then recombines, relying on
    /// the provider's buffer-from-columns path to set the cached row count.
    fn pair_buffer(provider: &CudaKernelProvider, arg0: &[u64], arg1: &[u64]) -> crate::CudaBuffer {
        assert_eq!(arg0.len(), arg1.len());
        let schema = Schema::new(vec![
            ("arg0".to_string(), ScalarType::U64),
            ("arg1".to_string(), ScalarType::U64),
        ]);
        if arg0.is_empty() {
            return provider
                .create_empty_buffer(schema)
                .expect("empty pair buffer");
        }
        // Pack both columns as a single 2-column buffer by constructing
        // byte-columns manually — mirrors what `from_dlpack_tensors_with_schema`
        // does for the in-process launcher tests.
        let device = provider.device().inner();
        let arg0_bytes: Vec<u8> = arg0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let arg1_bytes: Vec<u8> = arg1.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col0 = provider
            .memory()
            .alloc::<u8>(arg0_bytes.len())
            .expect("alloc");
        let mut col1 = provider
            .memory()
            .alloc::<u8>(arg1_bytes.len())
            .expect("alloc");
        device
            .htod_sync_copy_into(&arg0_bytes, &mut col0)
            .expect("h2d arg0");
        device
            .htod_sync_copy_into(&arg1_bytes, &mut col1)
            .expect("h2d arg1");
        provider
            .buffer_from_columns(vec![col0.into(), col1.into()], arg0.len() as u64, schema)
            .expect("buffer_from_columns")
    }

    fn pair_buffer_u32(
        provider: &CudaKernelProvider,
        arg0: &[u32],
        arg1: &[u32],
        typ: ScalarType,
    ) -> crate::CudaBuffer {
        assert_eq!(arg0.len(), arg1.len());
        assert!(matches!(typ, ScalarType::U32 | ScalarType::Symbol));
        let schema = Schema::new(vec![("arg0".to_string(), typ), ("arg1".to_string(), typ)]);
        if arg0.is_empty() {
            return provider
                .create_empty_buffer(schema)
                .expect("empty pair buffer");
        }
        let device = provider.device().inner();
        let arg0_bytes: Vec<u8> = arg0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let arg1_bytes: Vec<u8> = arg1.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col0 = provider
            .memory()
            .alloc::<u8>(arg0_bytes.len())
            .expect("alloc");
        let mut col1 = provider
            .memory()
            .alloc::<u8>(arg1_bytes.len())
            .expect("alloc");
        device
            .htod_sync_copy_into(&arg0_bytes, &mut col0)
            .expect("h2d arg0");
        device
            .htod_sync_copy_into(&arg1_bytes, &mut col1)
            .expect("h2d arg1");
        provider
            .buffer_from_columns(vec![col0.into(), col1.into()], arg0.len() as u64, schema)
            .expect("buffer_from_columns")
    }

    fn pair_buffer_i32(
        provider: &CudaKernelProvider,
        arg0: &[i32],
        arg1: &[i32],
    ) -> crate::CudaBuffer {
        assert_eq!(arg0.len(), arg1.len());
        let schema = Schema::new(vec![
            ("arg0".to_string(), ScalarType::I32),
            ("arg1".to_string(), ScalarType::I32),
        ]);
        if arg0.is_empty() {
            return provider
                .create_empty_buffer(schema)
                .expect("empty pair buffer");
        }
        let device = provider.device().inner();
        let arg0_bytes: Vec<u8> = arg0.iter().flat_map(|v| v.to_le_bytes()).collect();
        let arg1_bytes: Vec<u8> = arg1.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col0 = provider
            .memory()
            .alloc::<u8>(arg0_bytes.len())
            .expect("alloc");
        let mut col1 = provider
            .memory()
            .alloc::<u8>(arg1_bytes.len())
            .expect("alloc");
        device
            .htod_sync_copy_into(&arg0_bytes, &mut col0)
            .expect("h2d arg0");
        device
            .htod_sync_copy_into(&arg1_bytes, &mut col1)
            .expect("h2d arg1");
        provider
            .buffer_from_columns(vec![col0.into(), col1.into()], arg0.len() as u64, schema)
            .expect("buffer_from_columns")
    }

    /// Hand-computed coverage for C=2 candidates {p_B, p_C} against positives
    /// `{(1,4), (2,5)}` and negatives `{(7,8)}`. The only non-zero coverage
    /// is `chain(p_B, p_C) = 2` (both positives covered via chain joins
    /// z=2 and z=3). Everything else is zero by direct enumeration of the
    /// four topology templates — see
    /// `docs/plans/2026-04-17-m8-ilp-exact-kernel-design.md` for the
    /// templates. Also exercises the negative-scoring path with one negative
    /// that no topology-L-R combination covers.
    #[test]
    fn ilp_exact_score_matches_hand_computed_fixture() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Candidate relations.
        let p_b = pair_buffer(&provider, &[1, 2], &[2, 3]);
        let p_c = pair_buffer(&provider, &[2, 3, 4], &[4, 5, 6]);

        // Positives: {(1,4), (2,5)}. Negatives: {(7,8)}.
        let positives = pair_buffer(&provider, &[1, 2], &[4, 5]);
        let negatives = pair_buffer(&provider, &[7], &[8]);

        let (pos, neg) = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .expect("ilp_exact_score launch");

        // Slot layout: topology * C² + L * C + R, with C=2.
        //   topology: chain=0, star=1, fanout=2, fanin=3.
        //   L/R: p_B=0, p_C=1.
        // Only chain(p_B=0, p_C=1) → slot 0*4 + 0*2 + 1 = 1 is non-zero.
        let mut expected_pos = vec![0u32; 16];
        expected_pos[1] = 2;
        assert_eq!(
            pos, expected_pos,
            "positives coverage mismatch: expected {:?}, got {:?}",
            expected_pos, pos,
        );

        // All negatives coverage slots are zero: no (L, R, topology) covers (7, 8).
        let expected_neg = vec![0u32; 16];
        assert_eq!(
            neg, expected_neg,
            "negatives coverage mismatch: expected {:?}, got {:?}",
            expected_neg, neg,
        );
    }

    /// Determinism: the same inputs produce identical outputs on repeat runs.
    /// The kernel relies on integer counts + each block owning one unique
    /// output slot, so determinism is structural — no associativity or
    /// floating-point ordering concerns. Still worth pinning as a regression
    /// guard in case a future change swaps in atomics or shared state.
    #[test]
    fn ilp_exact_score_is_deterministic_across_runs() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer(&provider, &[1, 2], &[2, 3]);
        let p_c = pair_buffer(&provider, &[2, 3, 4], &[4, 5, 6]);
        let positives = pair_buffer(&provider, &[1, 2], &[4, 5]);
        let negatives = pair_buffer(&provider, &[7], &[8]);

        let run_a = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .unwrap();
        let run_b = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .unwrap();
        assert_eq!(run_a.0, run_b.0, "pos coverage drifted across runs");
        assert_eq!(run_a.1, run_b.1, "neg coverage drifted across runs");
    }

    /// Empty negatives: when the caller supplies a zero-row negatives buffer
    /// (the engine's normal treatment of `None`), the kernel must not
    /// dereference the negative pointers and must leave all `neg_covered`
    /// slots at zero.
    #[test]
    fn ilp_exact_score_handles_empty_negatives() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer(&provider, &[1, 2], &[2, 3]);
        let p_c = pair_buffer(&provider, &[2, 3, 4], &[4, 5, 6]);
        let positives = pair_buffer(&provider, &[1, 2], &[4, 5]);
        let negatives = pair_buffer(&provider, &[], &[]);

        let (pos, neg) = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .unwrap();

        let mut expected_pos = vec![0u32; 16];
        expected_pos[1] = 2;
        assert_eq!(pos, expected_pos);
        assert_eq!(neg, vec![0u32; 16]);
    }

    #[test]
    fn ilp_exact_score_accepts_u32_pair_buffers() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer_u32(&provider, &[1, 2], &[2, 3], ScalarType::U32);
        let p_c = pair_buffer_u32(&provider, &[2, 3, 4], &[4, 5, 6], ScalarType::U32);
        let positives = pair_buffer_u32(&provider, &[1, 2], &[4, 5], ScalarType::U32);
        let negatives = pair_buffer_u32(&provider, &[7], &[8], ScalarType::U32);

        let (pos, neg) = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .expect("U32 ilp_exact_score launch");

        let mut expected_pos = vec![0u32; 16];
        expected_pos[1] = 2;
        assert_eq!(pos, expected_pos);
        assert_eq!(neg, vec![0u32; 16]);
    }

    #[test]
    fn ilp_exact_score_accepts_symbol_pair_buffers() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer_u32(&provider, &[1, 2], &[2, 3], ScalarType::Symbol);
        let p_c = pair_buffer_u32(&provider, &[2, 3, 4], &[4, 5, 6], ScalarType::Symbol);
        let positives = pair_buffer_u32(&provider, &[1, 2], &[4, 5], ScalarType::Symbol);
        let negatives = pair_buffer_u32(&provider, &[7], &[8], ScalarType::Symbol);

        let (pos, neg) = provider
            .ilp_exact_score(&[&p_b, &p_c], &positives, &negatives)
            .expect("Symbol ilp_exact_score launch");

        let mut expected_pos = vec![0u32; 16];
        expected_pos[1] = 2;
        assert_eq!(pos, expected_pos);
        assert_eq!(neg, vec![0u32; 16]);
    }

    #[test]
    fn ilp_exact_score_rejects_mixed_pair_types() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer_u32(&provider, &[1, 2], &[2, 3], ScalarType::U32);
        let positives = pair_buffer(&provider, &[1, 2], &[4, 5]);
        let negatives = pair_buffer(&provider, &[7], &[8]);

        let err = provider
            .ilp_exact_score(&[&p_b], &positives, &negatives)
            .expect_err("mixed U64/U32 buffers must be rejected");
        assert!(
            err.to_string().contains("expected U64") || err.to_string().contains("type mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn ilp_exact_score_rejects_unsupported_pair_types() {
        let provider = match make_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let p_b = pair_buffer_i32(&provider, &[1, 2], &[2, 3]);
        let positives = pair_buffer_i32(&provider, &[1, 2], &[4, 5]);
        let negatives = pair_buffer_i32(&provider, &[7], &[8]);

        let err = provider
            .ilp_exact_score(&[&p_b], &positives, &negatives)
            .expect_err("I32 pair buffers must be rejected");
        assert!(
            err.to_string().contains("expected U64, U32, or Symbol"),
            "unexpected error: {err}"
        );
    }
}
