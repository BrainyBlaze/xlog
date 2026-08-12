//! Launcher for the n-ary bounded exact-induction scoring kernel.
//!
//! Drives `kernels/ilp_exact_nary.cu`'s `ilp_exact_nary_score`: one block
//! per flattened pattern, threads striding the example tuples, block-reduced
//! positive/negative coverage counts per pattern.
//!
//! Two entry points share one launch core:
//!
//! * [`CudaKernelProvider::ilp_exact_nary_score`] — host-slice inputs.
//!   The CPU-side leg of the parity chain (reference == flat == device)
//!   and the harness the pod fixtures drive.
//! * [`CudaKernelProvider::ilp_exact_nary_score_device`] — the PRODUCTION
//!   ingest: candidate relations and example tuples arrive as
//!   device-resident columnar [`CudaBuffer`]s and are concatenated with
//!   device-to-device column copies. No relation or example value ever
//!   visits the host; the only D2H is the two per-pattern count arrays.
//!
//! Everything is COLUMN-MAJOR in element units: relation cell
//! (row, position) sits at `cand_value_offset[slot] + position *
//! cand_rows[slot] + row`, and example position `i` of tuple `q` sits at
//! `i * count + q` — the same formulas the flat host interpreter and the
//! kernel share.
//!
//! Example tuples are a BAG, not a set: a duplicated positive counts once
//! per occurrence, so a caller that passes duplicates inflates coverage.
//! This is consistent across the reference, flat and device layers and is
//! the caller's contract to uphold.
//!
//! Per-launch runtime is NOT bounded by the pattern cap: one block walks a
//! backtracking search whose worst case is `rows^body_len`. On a display
//! GPU a wide relation set can therefore trip the driver watchdog; bound
//! the inputs, not just the pattern count.
//!
//! Layering: pattern arrays are PRIMITIVE flat host slices (the encoding
//! produced by `xlog_induce::nary_layout::flatten_patterns`), so
//! xlog-cuda keeps no dependency on xlog-induce. Every structural
//! invariant the kernel relies on (offsets in bounds, per-slot arity
//! consistency, binding indexes inside the fixed device state) is
//! re-validated here fail-closed before any upload; the kernel never sees
//! an out-of-bounds batch.

use std::marker::PhantomData;
use std::sync::atomic::Ordering;

use crate::memory::{CudaBuffer, TrackedCudaSlice};
use crate::{LaunchAsync, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::{ilp_exact_nary_kernels, RawCudaView, ILP_EXACT_NARY_MODULE};

/// MUST equal `ILP_EXACT_NARY_BLOCK_SIZE` in `kernels/ilp_exact_nary.cu`:
/// the kernel sizes its static `__shared__` scratch from that macro, so a
/// larger launch dimension would let threads write past it. The block
/// reduction additionally assumes a power of two.
const ILP_EXACT_NARY_BLOCK_SIZE: u32 = 256;
const _: () = assert!(
    ILP_EXACT_NARY_BLOCK_SIZE.is_power_of_two() && ILP_EXACT_NARY_BLOCK_SIZE <= 256,
    "block size must stay a power of two within the kernel's shared scratch",
);

// Device-contract bounds; must equal both the kernel's fixed state and
// xlog_induce::nary_layout's flattener bounds. Head arity is bounded by
// the kernel's per-thread example gather array.
const NARY_MAX_BODY_ATOMS: u32 = 8;
const NARY_MAX_JOIN_VARS: u32 = 8;
const NARY_MAX_HEAD_ARITY: u32 = 8;
// The kernel is safe with a wider atom (its position loop is bounded by
// the binding-code and relation extents), but the contract publishes
// arity <= 8 and this launcher is exported raw — without this the bound
// would hold only for callers who came through the flattener.
const NARY_MAX_ATOM_ARITY: u32 = 8;

const NARY_JOIN_FLAG: u32 = 0x8000_0000;
const NARY_INDEX_MASK: u32 = 0xFF;

const U64_SIZE: usize = std::mem::size_of::<u64>();

/// Flattened pattern batch (host-born: patterns are enumerated on host).
pub struct IlpExactNaryPatterns<'a> {
    pub body_offset: &'a [u32],
    pub body_len: &'a [u32],
    pub atom_candidate_slot: &'a [u32],
    pub atom_arity: &'a [u32],
    pub atom_binding_offset: &'a [u32],
    pub binding_codes: &'a [u32],
    pub head_arity: u32,
}

/// Host-slice scoring request: the pattern batch plus candidate relations
/// and example tuples as flat host arrays, exactly as the kernel consumes
/// them (column-major, element units).
pub struct IlpExactNaryRequest<'a> {
    pub body_offset: &'a [u32],
    pub body_len: &'a [u32],
    pub atom_candidate_slot: &'a [u32],
    pub atom_arity: &'a [u32],
    pub atom_binding_offset: &'a [u32],
    pub binding_codes: &'a [u32],
    /// Concatenated COLUMN-MAJOR u64 relation values: within a
    /// relation, cell (row, position) sits at `cand_value_offset[slot]
    /// + position * cand_rows[slot] + row`. All offsets are u32
    /// ELEMENT indexes, never bytes.
    pub cand_values: &'a [u64],
    /// Element offset of each relation's first value in `cand_values`.
    pub cand_value_offset: &'a [u32],
    pub cand_rows: &'a [u32],
    /// COLUMNAR example tuples: position `i` of example `q` sits at
    /// `i * example_count + q`; length is a multiple of `head_arity`.
    pub pos_values: &'a [u64],
    pub neg_values: &'a [u64],
    pub head_arity: u32,
}

impl<'a> IlpExactNaryRequest<'a> {
    fn patterns(&self) -> IlpExactNaryPatterns<'a> {
        IlpExactNaryPatterns {
            body_offset: self.body_offset,
            body_len: self.body_len,
            atom_candidate_slot: self.atom_candidate_slot,
            atom_arity: self.atom_arity,
            atom_binding_offset: self.atom_binding_offset,
            binding_codes: self.binding_codes,
            head_arity: self.head_arity,
        }
    }
}

fn nary_err(detail: impl std::fmt::Display) -> XlogError {
    XlogError::Kernel(format!("ilp_exact_nary_score: {detail}"))
}

/// Validate the pattern batch against the candidate slot table. Shared by
/// the host-slice and device-buffer paths; returns the pattern count.
fn validate_batch(
    patterns: &IlpExactNaryPatterns<'_>,
    cand_value_offset: &[u32],
    cand_rows: &[u32],
    values_len: u64,
) -> Result<u32> {
    let pattern_count = patterns.body_offset.len();
    if pattern_count == 0 {
        return Err(nary_err("empty pattern batch"));
    }
    if patterns.body_len.len() != pattern_count {
        return Err(nary_err("body_len length != pattern count"));
    }
    let atoms = patterns.atom_candidate_slot.len();
    if patterns.atom_arity.len() != atoms || patterns.atom_binding_offset.len() != atoms {
        return Err(nary_err("atom array lengths disagree"));
    }
    let bindings = patterns.binding_codes.len();
    let slots = cand_value_offset.len();
    if cand_rows.len() != slots {
        return Err(nary_err("cand_rows length != cand_value_offset length"));
    }
    if patterns.head_arity == 0 {
        return Err(nary_err("head_arity must be >= 1"));
    }
    if patterns.head_arity > NARY_MAX_HEAD_ARITY {
        return Err(nary_err(format!(
            "head_arity {} exceeds device bound {NARY_MAX_HEAD_ARITY}",
            patterns.head_arity
        )));
    }

    // Per-slot arity is implied by the atoms that read the slot; it must
    // be consistent and its rows must fit the concatenated value buffer.
    let mut slot_arity: Vec<Option<u32>> = vec![None; slots];
    for (pattern, (&offset, &len)) in patterns
        .body_offset
        .iter()
        .zip(patterns.body_len)
        .enumerate()
    {
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
            let slot = patterns.atom_candidate_slot[atom] as usize;
            if slot >= slots {
                return Err(nary_err(format!(
                    "atom {atom} references candidate slot {slot} of {slots}"
                )));
            }
            let arity = patterns.atom_arity[atom];
            if arity == 0 {
                return Err(nary_err(format!("atom {atom} has arity 0")));
            }
            if arity > NARY_MAX_ATOM_ARITY {
                return Err(nary_err(format!(
                    "atom {atom} has arity {arity} > the device contract's \
                     {NARY_MAX_ATOM_ARITY}"
                )));
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
            let rows = cand_rows[slot] as u64;
            let value_end = cand_value_offset[slot] as u64 + rows * arity as u64;
            if value_end > values_len {
                return Err(nary_err(format!(
                    "candidate slot {slot} needs values up to {value_end}, \
                     buffer holds {values_len}"
                )));
            }
            let binding_offset = patterns.atom_binding_offset[atom];
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
                let code = patterns.binding_codes[position];
                let index = code & NARY_INDEX_MASK;
                if code & NARY_JOIN_FLAG != 0 {
                    if index >= NARY_MAX_JOIN_VARS {
                        return Err(nary_err(format!(
                            "binding {position} join index {index} >= device \
                             bound {NARY_MAX_JOIN_VARS}"
                        )));
                    }
                } else if index >= patterns.head_arity {
                    return Err(nary_err(format!(
                        "binding {position} head index {index} >= head arity \
                         {}",
                        patterns.head_arity
                    )));
                }
            }
        }
    }

    u32::try_from(pattern_count).map_err(|_| nary_err("pattern count exceeds u32"))
}

fn validate_request(request: &IlpExactNaryRequest<'_>) -> Result<(u32, u32, u32)> {
    let head_arity = request.head_arity.max(1) as usize;
    if request.pos_values.len() % head_arity != 0 {
        return Err(nary_err("pos_values length not a multiple of head_arity"));
    }
    if request.neg_values.len() % head_arity != 0 {
        return Err(nary_err("neg_values length not a multiple of head_arity"));
    }
    let num_patterns = validate_batch(
        &request.patterns(),
        request.cand_value_offset,
        request.cand_rows,
        request.cand_values.len() as u64,
    )?;
    let num_pos = u32::try_from(request.pos_values.len() / head_arity)
        .map_err(|_| nary_err("positive tuple count exceeds u32"))?;
    let num_neg = u32::try_from(request.neg_values.len() / head_arity)
        .map_err(|_| nary_err("negative tuple count exceeds u32"))?;
    Ok((num_patterns, num_pos, num_neg))
}

fn u64_view<'a>(slice: &'a TrackedCudaSlice<u8>, elements: usize) -> RawCudaView<'a, u64> {
    RawCudaView {
        ptr: *slice.device_ptr(),
        len: elements,
        stream: slice.stream().clone(),
        _marker: PhantomData,
    }
}

/// One columnar u64 buffer requirement, validated fail-closed.
fn require_u64_columns(buf: &CudaBuffer, label: &str) -> Result<(u32, u32)> {
    let arity = u32::try_from(buf.arity()).map_err(|_| nary_err(format!("{label}: arity")))?;
    if arity == 0 {
        return Err(nary_err(format!("{label}: buffer has arity 0")));
    }
    for column in 0..buf.arity() {
        match buf.schema().column_type(column) {
            Some(ScalarType::U64) => {}
            other => {
                return Err(nary_err(format!(
                    "{label}: column {column} is {other:?}, expected U64"
                )));
            }
        }
    }
    let rows = buf
        .cached_row_count()
        .ok_or_else(|| nary_err(format!("{label}: cached_row_count absent")))?;
    let rows = u32::try_from(rows).map_err(|_| nary_err(format!("{label}: row count")))?;
    Ok((arity, rows))
}

impl super::CudaKernelProvider {
    /// Score every flattened pattern against the example tuples on GPU,
    /// from HOST slices.
    ///
    /// Returns `(pos_covered, neg_covered)`, one slot per pattern in
    /// batch order. D2H budget: **2** counter-tracked transfers (one per
    /// count array); all uploads are setup-phase H2D.
    pub fn ilp_exact_nary_score(
        &self,
        request: &IlpExactNaryRequest<'_>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let (num_patterns, num_pos, num_neg) = validate_request(request)?;

        macro_rules! upload_u64_bytes {
            ($host:expr) => {{
                let host: &[u64] = $host;
                let bytes: Vec<u8> = host.iter().flat_map(|v| v.to_le_bytes()).collect();
                let mut buf = self.memory.alloc::<u8>(bytes.len().max(1))?;
                if !bytes.is_empty() {
                    self.htod_sync_copy_into_tracked(&bytes, &mut buf)
                        .map_err(|e| nary_err(format!("h2d u64 values: {e}")))?;
                }
                buf
            }};
        }

        let cand_values_buf = upload_u64_bytes!(request.cand_values);
        let pos_values_buf = upload_u64_bytes!(request.pos_values);
        let neg_values_buf = upload_u64_bytes!(request.neg_values);

        self.launch_nary(
            &request.patterns(),
            num_patterns,
            num_pos,
            num_neg,
            request.cand_value_offset,
            request.cand_rows,
            u64_view(&cand_values_buf, request.cand_values.len()),
            u64_view(&pos_values_buf, request.pos_values.len()),
            u64_view(&neg_values_buf, request.neg_values.len()),
        )
    }

    /// Score every flattened pattern with DEVICE-RESIDENT relations and
    /// examples — the production ingest.
    ///
    /// `candidates[slot]` and the example buffers are columnar
    /// [`CudaBuffer`]s with all-U64 columns; ingestion is one
    /// device-to-device copy per column into the concatenated columnar
    /// value buffers. No relation or example value crosses the host
    /// boundary; the only D2H is the two count arrays (counter-tracked).
    /// Example buffers must have arity == `patterns.head_arity`.
    pub fn ilp_exact_nary_score_device(
        &self,
        patterns: &IlpExactNaryPatterns<'_>,
        candidates: &[&CudaBuffer],
        positives: &CudaBuffer,
        negatives: &CudaBuffer,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        // Slot table from the buffers themselves.
        let mut cand_value_offset: Vec<u32> = Vec::with_capacity(candidates.len());
        let mut cand_rows: Vec<u32> = Vec::with_capacity(candidates.len());
        let mut arities: Vec<u32> = Vec::with_capacity(candidates.len());
        let mut total_elems: u32 = 0;
        for (slot, buf) in candidates.iter().enumerate() {
            let (arity, rows) = require_u64_columns(buf, &format!("candidate[{slot}]"))?;
            cand_value_offset.push(total_elems);
            cand_rows.push(rows);
            arities.push(arity);
            let elems = arity
                .checked_mul(rows)
                .and_then(|e| total_elems.checked_add(e))
                .ok_or_else(|| nary_err("candidate value count exceeds u32"))?;
            total_elems = elems;
        }
        let num_patterns =
            validate_batch(patterns, &cand_value_offset, &cand_rows, total_elems as u64)?;

        // The batch declares each atom's arity; the buffers know their own
        // column count. A claimed arity wider than the relation reads
        // straight past the slot into the NEXT relation's column and scores
        // real-looking garbage, so the two must agree exactly.
        for (atom, (&slot, &claimed)) in patterns
            .atom_candidate_slot
            .iter()
            .zip(patterns.atom_arity.iter())
            .enumerate()
        {
            let actual = arities.get(slot as usize).copied().ok_or_else(|| {
                nary_err(format!("atom {atom}: candidate slot {slot} out of range"))
            })?;
            if claimed != actual {
                return Err(nary_err(format!(
                    "atom {atom} claims arity {claimed} for candidate slot \
                     {slot}, but that relation has {actual} columns; the \
                     kernel would read past the slot into the next relation",
                )));
            }
        }

        let (pos_arity, num_pos) = require_u64_columns(positives, "positives")?;
        let (neg_arity, num_neg) = require_u64_columns(negatives, "negatives")?;
        if pos_arity != patterns.head_arity {
            return Err(nary_err(format!(
                "positives arity {pos_arity} != head arity {}",
                patterns.head_arity
            )));
        }
        // Empty negatives are exempt from the arity match on purpose: a
        // zero-row buffer is never dereferenced by the kernel, and callers
        // legitimately pass a schema-arbitrary empty placeholder. Positives
        // are checked unconditionally because an empty positive set is a
        // real request shape whose head arity still defines the layout.
        if neg_arity != patterns.head_arity && num_neg != 0 {
            return Err(nary_err(format!(
                "negatives arity {neg_arity} != head arity {}",
                patterns.head_arity
            )));
        }

        // ── D2D columnar concatenation (setup-phase, never host) ──────
        let concat =
            |bufs: &[(&CudaBuffer, u32, u32)], total: usize| -> Result<TrackedCudaSlice<u8>> {
                let mut out = self.memory.alloc::<u8>((total * U64_SIZE).max(1))?;
                let device = self.device.inner();
                let mut element_offset: usize = 0;
                for (buf, arity, rows) in bufs {
                    let rows = *rows as usize;
                    for column in 0..*arity as usize {
                        if rows == 0 {
                            continue;
                        }
                        let bytes = rows * U64_SIZE;
                        let col = buf
                            .column(column)
                            .ok_or_else(|| nary_err(format!("missing column {column}")))?;
                        let src = self.column_bytes_view(col, bytes)?;
                        let byte_offset = element_offset * U64_SIZE;
                        let mut dst = out.slice_mut(byte_offset..byte_offset + bytes);
                        device
                            .dtod_copy(&src, &mut dst)
                            .map_err(|e| nary_err(format!("d2d column concat: {e}")))?;
                        element_offset += rows;
                    }
                }
                Ok(out)
            };

        let cand_triples: Vec<(&CudaBuffer, u32, u32)> = candidates
            .iter()
            .zip(arities.iter().zip(cand_rows.iter()))
            .map(|(buf, (&a, &r))| (*buf, a, r))
            .collect();
        let cand_values_buf = concat(&cand_triples, total_elems as usize)?;
        let pos_elems = (pos_arity as usize) * (num_pos as usize);
        let pos_values_buf = concat(&[(positives, pos_arity, num_pos)], pos_elems)?;
        let neg_elems = (neg_arity as usize) * (num_neg as usize);
        let neg_values_buf = concat(&[(negatives, neg_arity, num_neg)], neg_elems)?;

        self.launch_nary(
            patterns,
            num_patterns,
            num_pos,
            num_neg,
            &cand_value_offset,
            &cand_rows,
            u64_view(&cand_values_buf, total_elems as usize),
            u64_view(&pos_values_buf, pos_elems),
            u64_view(&neg_values_buf, neg_elems),
        )
    }

    /// Shared launch core: pack + upload the pattern batch and slot
    /// table, launch, and read back the two count arrays.
    #[allow(clippy::too_many_arguments)]
    fn launch_nary(
        &self,
        patterns: &IlpExactNaryPatterns<'_>,
        num_patterns: u32,
        num_pos: u32,
        num_neg: u32,
        cand_value_offset: &[u32],
        cand_rows: &[u32],
        cand_values: RawCudaView<'_, u64>,
        pos_values: RawCudaView<'_, u64>,
        neg_values: RawCudaView<'_, u64>,
    ) -> Result<(Vec<u32>, Vec<u32>)> {
        let device = self.device.inner();

        // Pack the six pattern arrays into ONE u32 buffer in the exact
        // section order the kernel unpacks (see ilp_exact_nary.cu): the
        // launch ABI caps the argument tuple, so the batch rides packed.
        let atoms = patterns.atom_candidate_slot.len();
        let mut batch_host: Vec<u32> = Vec::with_capacity(
            2 * num_patterns as usize + 3 * atoms + patterns.binding_codes.len(),
        );
        batch_host.extend_from_slice(patterns.body_offset);
        batch_host.extend_from_slice(patterns.body_len);
        batch_host.extend_from_slice(patterns.atom_candidate_slot);
        batch_host.extend_from_slice(patterns.atom_arity);
        batch_host.extend_from_slice(patterns.atom_binding_offset);
        batch_host.extend_from_slice(patterns.binding_codes);
        let params_host: Vec<u32> = vec![
            num_patterns,
            u32::try_from(atoms).map_err(|_| nary_err("atom count exceeds u32"))?,
            num_pos,
            num_neg,
            patterns.head_arity,
        ];

        macro_rules! upload_u32 {
            ($name:ident, $host:expr) => {{
                let host: &[u32] = $host;
                let mut buf = self.memory.alloc::<u32>(host.len().max(1))?;
                if !host.is_empty() {
                    self.htod_sync_copy_into_tracked(host, &mut buf)
                        .map_err(|e| {
                            nary_err(format!(concat!("h2d ", stringify!($name), ": {}"), e))
                        })?;
                }
                buf
            }};
        }

        let batch_buf = upload_u32!(batch, &batch_host);
        let params_buf = upload_u32!(params, &params_host);
        let cand_value_offset_buf = upload_u32!(cand_value_offset, cand_value_offset);
        let cand_rows_buf = upload_u32!(cand_rows, cand_rows);

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
                    &cand_values,
                    &cand_value_offset_buf,
                    &cand_rows_buf,
                    &pos_values,
                    &neg_values,
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
    //! the pod leg runs them for real. All values are COLUMNAR.

    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType, Schema};

    use super::{IlpExactNaryPatterns, IlpExactNaryRequest};
    use crate::{CudaDevice, CudaKernelProvider, GpuMemoryManager};

    fn make_provider() -> Option<CudaKernelProvider> {
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    const JOIN: u32 = 0x8000_0000;

    /// Build a columnar u64 buffer from per-column host arrays (mirrors
    /// the binary launcher's test helper, generalized to N columns).
    fn columnar_buffer(provider: &CudaKernelProvider, columns: &[&[u64]]) -> crate::CudaBuffer {
        let rows = columns[0].len();
        let schema = Schema::new(
            (0..columns.len())
                .map(|i| (format!("arg{i}"), ScalarType::U64))
                .collect(),
        );
        if rows == 0 {
            return provider.create_empty_buffer(schema).expect("empty buffer");
        }
        let device = provider.device().inner();
        let mut device_columns = Vec::with_capacity(columns.len());
        for column in columns {
            assert_eq!(column.len(), rows, "ragged columns");
            let bytes: Vec<u8> = column.iter().flat_map(|v| v.to_le_bytes()).collect();
            let mut buf = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
            device
                .htod_sync_copy_into(&bytes, &mut buf)
                .expect("h2d column");
            device_columns.push(buf.into());
        }
        provider
            .buffer_from_columns(device_columns, rows as u64, schema)
            .expect("buffer_from_columns")
    }

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
            // Columnar: p_B cols [1,2],[2,3]; p_C cols [2,3,4],[4,5,6].
            cand_values: &[1, 2, 2, 3, 2, 3, 4, 4, 5, 6],
            cand_value_offset: &[0, 4],
            cand_rows: &[2, 3],
            // Columnar examples: positives cols [1,2],[4,5].
            pos_values: &[1, 2, 4, 5],
            neg_values: &[7, 8],
            head_arity: 2,
        };
        let (pos, neg) = provider.ilp_exact_nary_score(&request).unwrap();
        assert_eq!(pos, vec![2]);
        assert_eq!(neg, vec![0]);
    }

    /// The same chain fixture through the PRODUCTION device-buffer path:
    /// candidates and examples live as columnar CudaBuffers and are
    /// ingested with D2D column copies only.
    #[test]
    fn nary_device_ingest_matches_host_path() {
        let Some(provider) = make_provider() else {
            return;
        };
        let p_b = columnar_buffer(&provider, &[&[1, 2], &[2, 3]]);
        let p_c = columnar_buffer(&provider, &[&[2, 3, 4], &[4, 5, 6]]);
        let positives = columnar_buffer(&provider, &[&[1, 2], &[4, 5]]);
        let negatives = columnar_buffer(&provider, &[&[7], &[8]]);
        let patterns = IlpExactNaryPatterns {
            body_offset: &[0],
            body_len: &[2],
            atom_candidate_slot: &[0, 1],
            atom_arity: &[2, 2],
            atom_binding_offset: &[0, 2],
            binding_codes: &[0, JOIN, JOIN, 1],
            head_arity: 2,
        };
        let (pos, neg) = provider
            .ilp_exact_nary_score_device(&patterns, &[&p_b, &p_c], &positives, &negatives)
            .unwrap();
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
            // Columnar: T cols [1,4],[2,5],[9,8]; P cols [9,8],[3,7].
            cand_values: &[1, 4, 2, 5, 9, 8, 9, 8, 3, 7],
            cand_value_offset: &[0, 6],
            cand_rows: &[2, 2],
            // Columnar examples: cols [1,4,1],[2,5,2],[3,6,7].
            pos_values: &[1, 4, 1, 2, 5, 2, 3, 6, 7],
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
            // Columnar: T cols [1,1],[8,9]; P single row [9,2].
            cand_values: &[1, 1, 8, 9, 9, 2],
            cand_value_offset: &[0, 4],
            cand_rows: &[2, 1],
            // Columnar examples: positives {(1,2),(1,3)} -> cols [1,1],[2,3].
            pos_values: &[1, 1, 2, 3],
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

        let wide_head = IlpExactNaryRequest {
            head_arity: 9,
            pos_values: &[1, 2, 3, 4, 5, 6, 7, 8, 9],
            binding_codes: &[0, 1],
            ..base
        };
        assert!(validate_request(&wide_head).is_err());
    }
}
