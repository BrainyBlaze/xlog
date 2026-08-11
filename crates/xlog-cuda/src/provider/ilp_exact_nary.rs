//! Launcher for the n-ary bounded exact-induction scoring kernel.
//!
//! Drives `kernels/ilp_exact_nary.cu`'s `ilp_exact_nary_score`: one block
//! per flattened pattern, threads striding the example tuples, block-reduced
//! positive/negative coverage counts per pattern.
//!
//! Layering: this launcher consumes PRIMITIVE flat host arrays — the
//! encoding produced by `xlog_induce::nary_layout::flatten_patterns` — so
//! xlog-cuda keeps no dependency on xlog-induce. Every structural
//! invariant the kernel relies on (offsets in bounds, per-slot arity
//! consistency, binding indexes inside the fixed device state) is
//! re-validated here fail-closed before any upload; the kernel never sees
//! an out-of-bounds batch.

use crate::{LaunchAsync, LaunchConfig};
use std::sync::atomic::Ordering;
use xlog_core::{Result, XlogError};

use super::{ilp_exact_nary_kernels, ILP_EXACT_NARY_MODULE};

const ILP_EXACT_NARY_BLOCK_SIZE: u32 = 256;

// Device-contract bounds; must equal both the kernel's fixed state and
// xlog_induce::nary_layout's flattener bounds.
const NARY_MAX_BODY_ATOMS: u32 = 8;
const NARY_MAX_JOIN_VARS: u32 = 8;

const NARY_JOIN_FLAG: u32 = 0x8000_0000;
const NARY_INDEX_MASK: u32 = 0xFF;

/// Flat n-ary scoring request: pattern batch + candidate relations +
/// example tuples, all host-side, exactly as the kernel consumes them.
pub struct IlpExactNaryRequest<'a> {
    pub body_offset: &'a [u32],
    pub body_len: &'a [u32],
    pub atom_candidate_slot: &'a [u32],
    pub atom_arity: &'a [u32],
    pub atom_binding_offset: &'a [u32],
    pub binding_codes: &'a [u32],
    /// Concatenated row-major u64 relation rows.
    pub cand_values: &'a [u64],
    /// Element offset of each relation's first row in `cand_values`.
    pub cand_value_offset: &'a [u32],
    pub cand_rows: &'a [u32],
    /// Row-major example tuples, stride = `head_arity`.
    pub pos_values: &'a [u64],
    pub neg_values: &'a [u64],
    pub head_arity: u32,
}

fn nary_err(detail: impl std::fmt::Display) -> XlogError {
    XlogError::Kernel(format!("ilp_exact_nary_score: {detail}"))
}

fn validate_request(request: &IlpExactNaryRequest<'_>) -> Result<(u32, u32, u32)> {
    let patterns = request.body_offset.len();
    if patterns == 0 {
        return Err(nary_err("empty pattern batch"));
    }
    if request.body_len.len() != patterns {
        return Err(nary_err("body_len length != pattern count"));
    }
    let atoms = request.atom_candidate_slot.len();
    if request.atom_arity.len() != atoms || request.atom_binding_offset.len() != atoms {
        return Err(nary_err("atom array lengths disagree"));
    }
    let bindings = request.binding_codes.len();
    let slots = request.cand_value_offset.len();
    if request.cand_rows.len() != slots {
        return Err(nary_err("cand_rows length != cand_value_offset length"));
    }
    if request.head_arity == 0 {
        return Err(nary_err("head_arity must be >= 1"));
    }
    let head_arity = request.head_arity as usize;
    if request.pos_values.len() % head_arity != 0 {
        return Err(nary_err("pos_values length not a multiple of head_arity"));
    }
    if request.neg_values.len() % head_arity != 0 {
        return Err(nary_err("neg_values length not a multiple of head_arity"));
    }

    // Per-slot arity is implied by the atoms that read the slot; it must
    // be consistent and its rows must fit the concatenated value buffer.
    let mut slot_arity: Vec<Option<u32>> = vec![None; slots];
    for (pattern, (&offset, &len)) in request.body_offset.iter().zip(request.body_len).enumerate() {
        if len == 0 {
            return Err(nary_err(format!("pattern {pattern} has an empty body")));
        }
        if len > NARY_MAX_BODY_ATOMS {
            return Err(nary_err(format!(
                "pattern {pattern} has {len} body atoms; device bound is \
                 {NARY_MAX_BODY_ATOMS}"
            )));
        }
        let end = offset
            .checked_add(len)
            .ok_or_else(|| nary_err("body offset overflow"))?;
        if end as usize > atoms {
            return Err(nary_err(format!(
                "pattern {pattern} body [{offset}, {end}) exceeds {atoms} atoms"
            )));
        }
        for atom in offset as usize..end as usize {
            let slot = request.atom_candidate_slot[atom] as usize;
            if slot >= slots {
                return Err(nary_err(format!(
                    "atom {atom} references candidate slot {slot} of {slots}"
                )));
            }
            let arity = request.atom_arity[atom];
            if arity == 0 {
                return Err(nary_err(format!("atom {atom} has arity 0")));
            }
            match slot_arity[slot] {
                None => slot_arity[slot] = Some(arity),
                Some(existing) if existing == arity => {}
                Some(existing) => {
                    return Err(nary_err(format!(
                        "candidate slot {slot} read at arity {arity} and \
                         arity {existing}; relation arity must be consistent"
                    )));
                }
            }
            let rows = request.cand_rows[slot] as u64;
            let value_end = request.cand_value_offset[slot] as u64 + rows * arity as u64;
            if value_end > request.cand_values.len() as u64 {
                return Err(nary_err(format!(
                    "candidate slot {slot} needs values up to {value_end}, \
                     buffer holds {}",
                    request.cand_values.len()
                )));
            }
            let binding_offset = request.atom_binding_offset[atom];
            let binding_end = binding_offset
                .checked_add(arity)
                .ok_or_else(|| nary_err("binding offset overflow"))?;
            if binding_end as usize > bindings {
                return Err(nary_err(format!(
                    "atom {atom} bindings [{binding_offset}, {binding_end}) \
                     exceed {bindings} codes"
                )));
            }
            for position in binding_offset as usize..binding_end as usize {
                let code = request.binding_codes[position];
                let index = code & NARY_INDEX_MASK;
                if code & NARY_JOIN_FLAG != 0 {
                    if index >= NARY_MAX_JOIN_VARS {
                        return Err(nary_err(format!(
                            "binding {position} join index {index} >= device \
                             bound {NARY_MAX_JOIN_VARS}"
                        )));
                    }
                } else if index >= request.head_arity {
                    return Err(nary_err(format!(
                        "binding {position} head index {index} >= head arity \
                         {}",
                        request.head_arity
                    )));
                }
            }
        }
    }

    let patterns_u32 =
        u32::try_from(patterns).map_err(|_| nary_err("pattern count exceeds u32"))?;
    let num_pos = u32::try_from(request.pos_values.len() / head_arity)
        .map_err(|_| nary_err("positive tuple count exceeds u32"))?;
    let num_neg = u32::try_from(request.neg_values.len() / head_arity)
        .map_err(|_| nary_err("negative tuple count exceeds u32"))?;
    Ok((patterns_u32, num_pos, num_neg))
}

impl super::CudaKernelProvider {
    /// Score every flattened pattern against the example tuples on GPU.
    ///
    /// Returns `(pos_covered, neg_covered)`, one slot per pattern in
    /// batch order. D2H budget: **2** counter-tracked transfers (one per
    /// count array); all uploads are setup-phase H2D.
    pub fn ilp_exact_nary_score(
        &self,
        request: &IlpExactNaryRequest<'_>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let (num_patterns, num_pos, num_neg) = validate_request(request)?;
        let device = self.device.inner();

        // Pack the six pattern arrays into ONE u32 buffer in the exact
        // section order the kernel unpacks (see ilp_exact_nary.cu): the
        // launch ABI caps the argument tuple, so the batch rides packed.
        let atoms = request.atom_candidate_slot.len();
        let mut batch_host: Vec<u32> =
            Vec::with_capacity(2 * num_patterns as usize + 3 * atoms + request.binding_codes.len());
        batch_host.extend_from_slice(request.body_offset);
        batch_host.extend_from_slice(request.body_len);
        batch_host.extend_from_slice(request.atom_candidate_slot);
        batch_host.extend_from_slice(request.atom_arity);
        batch_host.extend_from_slice(request.atom_binding_offset);
        batch_host.extend_from_slice(request.binding_codes);
        let params_host: Vec<u32> = vec![
            num_patterns,
            u32::try_from(atoms).map_err(|_| nary_err("atom count exceeds u32"))?,
            num_pos,
            num_neg,
            request.head_arity,
        ];

        // Zero-length inputs still get a 1-element allocation: the kernel
        // reads them only within the true counts.
        macro_rules! upload {
            ($name:ident, $ty:ty, $host:expr) => {{
                let host: &[$ty] = $host;
                let mut buf = self.memory.alloc::<$ty>(host.len().max(1))?;
                if !host.is_empty() {
                    self.htod_sync_copy_into_tracked(host, &mut buf)
                        .map_err(|e| {
                            nary_err(format!(concat!("h2d ", stringify!($name), ": {}"), e))
                        })?;
                }
                buf
            }};
        }

        let batch_buf = upload!(batch, u32, &batch_host);
        let params_buf = upload!(params, u32, &params_host);
        let cand_values_buf = upload!(cand_values, u64, request.cand_values);
        let cand_value_offset_buf = upload!(cand_value_offset, u32, request.cand_value_offset);
        let cand_rows_buf = upload!(cand_rows, u32, request.cand_rows);
        let pos_values_buf = upload!(pos_values, u64, request.pos_values);
        let neg_values_buf = upload!(neg_values, u64, request.neg_values);

        let mut pos_covered_buf = self.memory.alloc::<u32>(num_patterns as usize)?;
        let mut neg_covered_buf = self.memory.alloc::<u32>(num_patterns as usize)?;
        // The kernel writes every pattern slot exactly once — no zero-init.

        let func = device
            .get_func(
                ILP_EXACT_NARY_MODULE,
                ilp_exact_nary_kernels::ILP_EXACT_NARY_SCORE,
            )
            .ok_or_else(|| nary_err("kernel not loaded"))?;
        unsafe {
            func.launch(
                LaunchConfig {
                    grid_dim: (num_patterns, 1, 1),
                    block_dim: (ILP_EXACT_NARY_BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &batch_buf,
                    &params_buf,
                    &cand_values_buf,
                    &cand_value_offset_buf,
                    &cand_rows_buf,
                    &pos_values_buf,
                    &neg_values_buf,
                    &mut pos_covered_buf,
                    &mut neg_covered_buf,
                ),
            )
            .map_err(|e| nary_err(format!("launch: {e}")))?;
        }
        self.device.synchronize()?;

        let mut pos_covered = vec![0u32; num_patterns as usize];
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        device
            .dtoh_sync_copy_into(&pos_covered_buf, &mut pos_covered)
            .map_err(|e| nary_err(format!("dtoh pos_covered: {e}")))?;
        let mut neg_covered = vec![0u32; num_patterns as usize];
        self.d2h_transfer_count.fetch_add(1, Ordering::Relaxed);
        device
            .dtoh_sync_copy_into(&neg_covered_buf, &mut neg_covered)
            .map_err(|e| nary_err(format!("dtoh neg_covered: {e}")))?;
        Ok((pos_covered, neg_covered))
    }
}

#[cfg(test)]
mod tests {
    //! CUDA-gated correctness tests for the n-ary launcher, pinned to the
    //! same hand-computed fixtures the host reference scorer freezes
    //! (`xlog_induce::nary_reference`). Skipped silently without a GPU;
    //! the pod leg runs them for real.

    use std::sync::Arc;

    use xlog_core::MemoryBudget;

    use super::IlpExactNaryRequest;
    use crate::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    fn make_provider() -> Option<CudaKernelProvider> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    const JOIN: u32 = 0x8000_0000;

    /// chain(L=0, R=1) over the shipped binary-kernel fixture:
    /// p_B={(1,2),(2,3)}, p_C={(2,4),(3,5),(4,6)}, positives {(1,4),(2,5)},
    /// negatives {(7,8)} — covers both positives, no negative.
    #[test]
    fn nary_kernel_matches_binary_chain_fixture() {
        let Some(provider) = make_provider() else {
            return;
        };
        let request = IlpExactNaryRequest {
            body_offset: &[0],
            body_len: &[2],
            atom_candidate_slot: &[0, 1],
            atom_arity: &[2, 2],
            atom_binding_offset: &[0, 2],
            // L(Head0, Join0), R(Join0, Head1)
            binding_codes: &[0, JOIN, JOIN, 1],
            cand_values: &[1, 2, 2, 3, 2, 4, 3, 5, 4, 6],
            cand_value_offset: &[0, 4],
            cand_rows: &[2, 3],
            pos_values: &[1, 4, 2, 5],
            neg_values: &[7, 8],
            head_arity: 2,
        };
        let (pos, neg) = provider.ilp_exact_nary_score(&request).unwrap();
        assert_eq!(pos, vec![2]);
        assert_eq!(neg, vec![0]);
    }

    /// Ternary fixture from the reference suite: H(x0,x1,x2) :-
    /// T(x0,x1,z0), P(z0,x2) with T={(1,2,9),(4,5,8)}, P={(9,3),(8,7)}.
    /// Of positives {(1,2,3),(4,5,6),(1,2,7)} exactly one is covered.
    #[test]
    fn nary_kernel_matches_ternary_reference_fixture() {
        let Some(provider) = make_provider() else {
            return;
        };
        let request = IlpExactNaryRequest {
            body_offset: &[0],
            body_len: &[2],
            atom_candidate_slot: &[0, 1],
            atom_arity: &[3, 2],
            atom_binding_offset: &[0, 3],
            binding_codes: &[0, 1, JOIN, JOIN, 2],
            cand_values: &[1, 2, 9, 4, 5, 8, 9, 3, 8, 7],
            cand_value_offset: &[0, 6],
            cand_rows: &[2, 2],
            pos_values: &[1, 2, 3, 4, 5, 6, 1, 2, 7],
            neg_values: &[],
            head_arity: 3,
        };
        let (pos, neg) = provider.ilp_exact_nary_score(&request).unwrap();
        assert_eq!(pos, vec![1]);
        assert_eq!(neg, vec![0]);
    }

    /// Backtracking fixture: T={(1,8),(1,9)}, P={(9,2)} — the first T row
    /// dead-ends and the kernel must revisit T at row 2 to find the cover.
    #[test]
    fn nary_kernel_backtracks_across_atoms() {
        let Some(provider) = make_provider() else {
            return;
        };
        let request = IlpExactNaryRequest {
            body_offset: &[0],
            body_len: &[2],
            atom_candidate_slot: &[0, 1],
            atom_arity: &[2, 2],
            atom_binding_offset: &[0, 2],
            binding_codes: &[0, JOIN, JOIN, 1],
            cand_values: &[1, 8, 1, 9, 9, 2],
            cand_value_offset: &[0, 4],
            cand_rows: &[2, 1],
            pos_values: &[1, 2, 1, 3],
            neg_values: &[],
            head_arity: 2,
        };
        let (pos, neg) = provider.ilp_exact_nary_score(&request).unwrap();
        assert_eq!(pos, vec![1]);
        assert_eq!(neg, vec![0]);
    }

    #[test]
    fn validation_refuses_malformed_batches_without_a_device() {
        // Validation is host-side and must refuse BEFORE any CUDA work,
        // so these run everywhere.
        use super::validate_request;

        let base = IlpExactNaryRequest {
            body_offset: &[0],
            body_len: &[1],
            atom_candidate_slot: &[0],
            atom_arity: &[2],
            atom_binding_offset: &[0],
            binding_codes: &[0, 1],
            cand_values: &[1, 2],
            cand_value_offset: &[0],
            cand_rows: &[1],
            pos_values: &[1, 2],
            neg_values: &[],
            head_arity: 2,
        };
        assert!(validate_request(&base).is_ok());

        let empty = IlpExactNaryRequest {
            body_offset: &[],
            body_len: &[],
            ..base
        };
        assert!(validate_request(&empty).is_err());

        let bad_slot = IlpExactNaryRequest {
            atom_candidate_slot: &[5],
            ..base
        };
        assert!(validate_request(&bad_slot).is_err());

        let short_values = IlpExactNaryRequest {
            cand_rows: &[9],
            ..base
        };
        assert!(validate_request(&short_values).is_err());

        let head_out_of_range = IlpExactNaryRequest {
            binding_codes: &[0, 7],
            ..base
        };
        assert!(validate_request(&head_out_of_range).is_err());

        let ragged_tuples = IlpExactNaryRequest {
            pos_values: &[1, 2, 3],
            ..base
        };
        assert!(validate_request(&ragged_tuples).is_err());
    }
}
