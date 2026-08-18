//! Fixed-address relational primitives for CUDA conditional graphs.
//!
//! Every allocation happens in the `prepare_*` methods. The `record_*` methods
//! only enqueue fixed-address memset and kernel nodes on the supplied stream,
//! so they are safe to call while a conditional graph body is being captured.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::sys;
use xlog_core::{Result, ScalarType, Schema, XlogError};

use crate::cuda_compat::{AsKernelParam, DeviceSlice, LaunchAsync, LaunchConfig};
use crate::launch::LaunchRecorder;
use crate::memory::{GpuMemoryReservation, TrackedCudaSlice};
use crate::{CudaBuffer, CudaColumn, CudaStream};

use super::CudaKernelProvider;

const MODULE: &str = "xlog_resident_relational";
const BLOCK_SIZE: u32 = 256;

/// Maximum relation width accepted by the resident relational wire ABI.
pub const RESIDENT_RELATIONAL_MAX_ARITY: usize = 17;

/// Stable terminal status codes shared by every resident CUDA primitive.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentTerminalCode {
    Running = 0,
    Success = 1,
    IterationLimit = 2,
    CapacityOverflow = 3,
    ResourceExhausted = 4,
}

/// Stable resource identifiers for [`ResidentTerminalStatus::resource_code`].
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentResourceCode {
    None = 0,
    SetHashSlots = 1,
    JoinBuckets = 2,
    JoinChains = 3,
    InputRows = 4,
    OutputRows = 5,
}

/// Canonical host/device terminal receipt for resident graph execution.
///
/// CUDA reserves `0xffff_fffe` internally while the winning thread publishes
/// a payload. It writes the public terminal code last after a system fence, so
/// concurrent failures cannot produce a mixed receipt.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentTerminalStatus {
    pub code: u32,
    pub op_id: u32,
    pub resource_code: u32,
    pub iterations: u32,
    pub limit: u32,
    pub reserved: u32,
    pub required: u64,
    pub capacity: u64,
}

// SAFETY: repr(C), no references or padding with validity requirements, and
// every bit pattern is valid for all fields.
unsafe impl cudarc::driver::DeviceRepr for ResidentTerminalStatus {}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ResidentRelationView {
    columns: [u64; RESIDENT_RELATIONAL_MAX_ARITY],
    widths: [u32; RESIDENT_RELATIONAL_MAX_ARITY],
    arity: u32,
    capacity: u32,
    reserved: u32,
    num_rows: u64,
}

impl AsKernelParam for ResidentRelationView {
    fn as_kernel_param(&self) -> *mut c_void {
        (self as *const Self).cast_mut().cast()
    }
}

/// A fixed-capacity relation whose storage remains stable for a graph lifetime.
pub struct ResidentRelation {
    buffer: CudaBuffer,
}

impl ResidentRelation {
    pub fn buffer(&self) -> &CudaBuffer {
        &self.buffer
    }

    /// Consume the stable owner after terminal receipt validation.
    ///
    /// This performs no transfer, synchronization, or host-count observation.
    pub fn into_buffer(self) -> CudaBuffer {
        self.buffer
    }

    /// Consume a synchronized final owner with its receipt-selected schema.
    ///
    /// This changes metadata only after all captured work is complete and
    /// validates that the selected schema has the same physical layout.
    pub fn into_buffer_with_observed_schema(mut self, schema: Schema) -> Result<CudaBuffer> {
        if schema.arity() != self.buffer.arity() {
            return Err(XlogError::Kernel(format!(
                "resident observed schema arity {} does not match allocation arity {}",
                schema.arity(),
                self.buffer.arity()
            )));
        }
        for column in 0..schema.arity() {
            let old_width = width(
                self.buffer
                    .schema()
                    .column_type(column)
                    .expect("matching arity"),
            )?;
            let new_width = width(schema.column_type(column).expect("matching arity"))?;
            if old_width != new_width {
                return Err(XlogError::Kernel(format!(
                    "resident observed schema column {column} changes physical width from \
                     {old_width} to {new_width} bytes"
                )));
            }
        }
        self.buffer.set_schema(schema);
        Ok(self.buffer)
    }

    /// Retag a cold scratch allocation without changing any device address.
    ///
    /// This is only valid before the graph that will use this relation is
    /// launched. Captured kernel nodes own copies of their pointer/width
    /// arguments; callers must not retag a relation while a launch is in flight.
    /// Final head relations must not use this scratch-pool operation.
    pub fn retag_schema_for_capture(&mut self, schema: Schema) -> Result<()> {
        if schema.arity() != self.buffer.arity() {
            return Err(XlogError::Kernel(format!(
                "resident scratch retag arity {} does not match allocation arity {}",
                schema.arity(),
                self.buffer.arity()
            )));
        }
        for column in 0..schema.arity() {
            let old_width = width(
                self.buffer
                    .schema()
                    .column_type(column)
                    .expect("matching arity"),
            )?;
            let new_width = width(schema.column_type(column).expect("matching arity"))?;
            if old_width != new_width {
                return Err(XlogError::Kernel(format!(
                    "resident scratch retag column {column} changes physical width from \
                     {old_width} to {new_width} bytes"
                )));
            }
        }
        self.buffer.set_schema(schema);
        Ok(())
    }

    pub fn capacity(&self) -> u32 {
        self.buffer
            .num_rows()
            .try_into()
            .expect("resident capacity")
    }

    pub fn num_rows_device(&self) -> &TrackedCudaSlice<u32> {
        self.buffer.num_rows_device()
    }

    /// Add every owned allocation to the transaction's strict recorder.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        for column in self.buffer.columns() {
            recorder.write_column(column);
        }
        recorder.read_write(self.buffer.num_rows_device());
    }
}

/// Preallocated open-addressed full-row set workspace.
pub struct ResidentSetWorkspace {
    slots: TrackedCudaSlice<u64>,
    required: TrackedCudaSlice<u64>,
    candidate_capacity: u32,
    slot_mask: u32,
}

impl ResidentSetWorkspace {
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.slots);
        recorder.read_write(&self.required);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u32, u32) {
        (
            self.slots.device_ptr_value(),
            self.required.device_ptr_value(),
            self.slot_mask,
            self.candidate_capacity,
        )
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 2]> {
        Ok([
            self.slots.runtime_allocation_identity()?,
            self.required.runtime_allocation_identity()?,
        ])
    }
}

/// Preallocated one-key hash-join workspace.
pub struct ResidentJoinWorkspace {
    bucket_heads: TrackedCudaSlice<u32>,
    next: TrackedCudaSlice<u32>,
    required: TrackedCudaSlice<u64>,
    right_capacity: u32,
    bucket_mask: u32,
}

impl ResidentJoinWorkspace {
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.bucket_heads);
        recorder.read_write(&self.next);
        recorder.read_write(&self.required);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u64, u32, u32) {
        (
            self.bucket_heads.device_ptr_value(),
            self.next.device_ptr_value(),
            self.required.device_ptr_value(),
            self.bucket_mask,
            self.right_capacity,
        )
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 3]> {
        Ok([
            self.bucket_heads.runtime_allocation_identity()?,
            self.next.runtime_allocation_identity()?,
            self.required.runtime_allocation_identity()?,
        ])
    }
}

/// Owns the single canonical terminal receipt used by a resident graph.
pub struct ResidentConvergenceControl {
    status: TrackedCudaSlice<ResidentTerminalStatus>,
    changed: TrackedCudaSlice<u32>,
    loop_iterations: TrackedCudaSlice<u32>,
}

/// Device-owned physical and legacy-semantic resident scan/filter counters.
pub struct ResidentDeviceTrace {
    scan_invocations: TrackedCudaSlice<u32>,
    filter_invocations: TrackedCudaSlice<u32>,
    semantic_scan_invocations: TrackedCudaSlice<u32>,
    semantic_filter_invocations: TrackedCudaSlice<u32>,
}

/// Device-owned first-nonempty schema selection for staged relation heads.
pub struct ResidentSchemaWinners {
    seen_nonempty: TrackedCudaSlice<u32>,
    winner_schema_ids: TrackedCudaSlice<u32>,
    default_schema_ids: Vec<u32>,
    len: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidentReceiptPointeeRole {
    RelationCount(u32),
    ScanTrace,
    FilterTrace,
    SemanticScanTrace,
    SemanticFilterTrace,
    SchemaWinner(u32),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct ResidentReceiptPointee {
    pub(crate) role: ResidentReceiptPointeeRole,
    pub(crate) ptr: u64,
    pub(crate) range_end: u64,
    pub(crate) manager_id: usize,
    pub(crate) block: Option<crate::device_runtime::BlockId>,
}

fn checked_receipt_pointee(
    role: ResidentReceiptPointeeRole,
    ptr: u64,
    manager_id: usize,
    runtime_block: Option<(crate::device_runtime::BlockId, usize)>,
) -> Result<ResidentReceiptPointee> {
    let range_end = ptr.checked_add(4).ok_or_else(|| {
        XlogError::Kernel(
            "resident receipt pointee range overflows the device address space".into(),
        )
    })?;
    let block = match runtime_block {
        Some((block, bytes)) => {
            let block_bytes = u64::try_from(bytes).map_err(|_| {
                XlogError::Kernel("resident receipt runtime block size is not representable".into())
            })?;
            let block_end = block.ptr.checked_add(block_bytes).ok_or_else(|| {
                XlogError::Kernel("resident receipt runtime block range overflows".into())
            })?;
            if ptr < block.ptr || range_end > block_end {
                return Err(XlogError::Kernel(
                    "resident receipt pointee is outside its runtime block".into(),
                ));
            }
            Some(block)
        }
        None => None,
    };
    Ok(ResidentReceiptPointee {
        role,
        ptr,
        range_end,
        manager_id,
        block,
    })
}

fn validate_receipt_pointee_ranges(pointees: &[ResidentReceiptPointee]) -> Result<()> {
    for (index, pointee) in pointees.iter().enumerate() {
        for previous in &pointees[..index] {
            if pointee.ptr < previous.range_end && previous.ptr < pointee.range_end {
                return Err(XlogError::Kernel(
                    "resident receipt pointee ranges overlap".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_receipt_pointee_owners(
    pointees: &[ResidentReceiptPointee],
    manager_id: usize,
    device_ordinal: u32,
) -> Result<()> {
    for pointee in pointees {
        if pointee.manager_id != manager_id {
            return Err(XlogError::Kernel(
                "resident receipt pointee belongs to a foreign memory manager".into(),
            ));
        }
        let block = pointee.block.ok_or_else(|| {
            XlogError::Kernel("resident receipt pointee has no runtime block identity".into())
        })?;
        if block.device_ordinal != device_ordinal {
            return Err(XlogError::Kernel(
                "resident receipt pointee belongs to a foreign CUDA device".into(),
            ));
        }
    }
    Ok(())
}

fn validate_receipt_schedule_block_mapping(
    pointees: &[ResidentReceiptPointee],
    expected_blocks: &[crate::device_runtime::BlockId],
) -> Result<()> {
    if pointees.len() != expected_blocks.len()
        || pointees
            .iter()
            .zip(expected_blocks)
            .any(|(pointee, expected)| pointee.block != Some(*expected))
    {
        return Err(XlogError::Kernel(
            "resident receipt runtime-block mapping differs from the schedule".into(),
        ));
    }
    Ok(())
}

fn record_receipt_pointee_uses(pointees: &[ResidentReceiptPointee], recorder: &mut LaunchRecorder) {
    for pointee in pointees {
        recorder.read_optional_block_identity(pointee.block);
    }
}

fn validate_receipt_schedule_mapping(
    pointees: &[ResidentReceiptPointee],
    relation_counts: &[u64],
    trace_counts: [u64; 4],
    schema_winners: &[u64],
) -> Result<()> {
    if relation_counts.len() != schema_winners.len() {
        return Err(XlogError::Kernel(
            "resident receipt relation and schema-winner mappings differ in length".into(),
        ));
    }
    let expected_len = relation_counts
        .len()
        .checked_mul(2)
        .and_then(|count| count.checked_add(4))
        .ok_or_else(|| XlogError::Kernel("resident receipt mapping length overflow".into()))?;
    if pointees.len() != expected_len {
        return Err(XlogError::Kernel(
            "resident receipt pointee manifest has the wrong length".into(),
        ));
    }
    for (index, ptr) in relation_counts.iter().copied().enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| XlogError::Kernel("resident receipt head index overflow".into()))?;
        if pointees[index as usize].role != ResidentReceiptPointeeRole::RelationCount(index)
            || pointees[index as usize].ptr != ptr
        {
            return Err(XlogError::Kernel(
                "resident receipt relation-count mapping differs from the schedule".into(),
            ));
        }
    }
    let trace_offset = relation_counts.len();
    if pointees[trace_offset].role != ResidentReceiptPointeeRole::ScanTrace
        || pointees[trace_offset].ptr != trace_counts[0]
        || pointees[trace_offset + 1].role != ResidentReceiptPointeeRole::FilterTrace
        || pointees[trace_offset + 1].ptr != trace_counts[1]
        || pointees[trace_offset + 2].role != ResidentReceiptPointeeRole::SemanticScanTrace
        || pointees[trace_offset + 2].ptr != trace_counts[2]
        || pointees[trace_offset + 3].role != ResidentReceiptPointeeRole::SemanticFilterTrace
        || pointees[trace_offset + 3].ptr != trace_counts[3]
    {
        return Err(XlogError::Kernel(
            "resident receipt trace mapping differs from the schedule".into(),
        ));
    }
    let winner_offset = trace_offset + 4;
    for (index, ptr) in schema_winners.iter().copied().enumerate() {
        let index_u32 = u32::try_from(index)
            .map_err(|_| XlogError::Kernel("resident receipt head index overflow".into()))?;
        let pointee = &pointees[winner_offset + index];
        if pointee.role != ResidentReceiptPointeeRole::SchemaWinner(index_u32) || pointee.ptr != ptr
        {
            return Err(XlogError::Kernel(
                "resident receipt schema-winner mapping differs from the schedule".into(),
            ));
        }
    }
    Ok(())
}

impl ResidentSchemaWinners {
    /// Add both fixed-address arrays to the transaction's strict recorder.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.seen_nonempty);
        recorder.read_write(&self.winner_schema_ids);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u32) {
        (
            self.seen_nonempty.device_ptr_value(),
            self.winner_schema_ids.device_ptr_value(),
            self.len,
        )
    }

    pub(crate) fn default_schema_ids(&self) -> &[u32] {
        &self.default_schema_ids
    }

    pub fn len(&self) -> u32 {
        self.len
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 2]> {
        Ok([
            self.seen_nonempty.runtime_allocation_identity()?,
            self.winner_schema_ids.runtime_allocation_identity()?,
        ])
    }
}

impl ResidentDeviceTrace {
    /// Add all fixed-address counters to the transaction's strict recorder.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.scan_invocations);
        recorder.read_write(&self.filter_invocations);
        recorder.read_write(&self.semantic_scan_invocations);
        recorder.read_write(&self.semantic_filter_invocations);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u64, u64) {
        (
            self.scan_invocations.device_ptr_value(),
            self.filter_invocations.device_ptr_value(),
            self.semantic_scan_invocations.device_ptr_value(),
            self.semantic_filter_invocations.device_ptr_value(),
        )
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 4]> {
        Ok([
            self.scan_invocations.runtime_allocation_identity()?,
            self.filter_invocations.runtime_allocation_identity()?,
            self.semantic_scan_invocations
                .runtime_allocation_identity()?,
            self.semantic_filter_invocations
                .runtime_allocation_identity()?,
        ])
    }
}

/// One fixed-size device receipt: terminal wire record, changed flag, then u32 counts.
pub struct ResidentPackedReceipt {
    count_ptrs: TrackedCudaSlice<u64>,
    bytes: TrackedCudaSlice<u8>,
    pointees: Vec<ResidentReceiptPointee>,
    count_len: u32,
    relation_count_len: u32,
    device_trace_field_count: u32,
    schema_winner_count: u32,
}

/// Exact-size page-locked destination for the single final resident receipt.
#[derive(Debug)]
pub struct ResidentPinnedReceipt {
    ptr: std::ptr::NonNull<u8>,
    len: usize,
}

// SAFETY: the allocation has unique ownership, exposes no host pointer, and
// CUDA permits page-locked host allocations to be freed from another thread.
unsafe impl Send for ResidentPinnedReceipt {}

impl ResidentPinnedReceipt {
    pub fn len_bytes(&self) -> usize {
        self.len
    }
}

impl Drop for ResidentPinnedReceipt {
    fn drop(&mut self) {
        // SAFETY: `ptr` was returned by `cuMemHostAlloc` and is freed once here.
        let _ = unsafe { sys::cuMemFreeHost(self.ptr.as_ptr().cast()) };
    }
}

impl ResidentPackedReceipt {
    pub(crate) fn pointee_manifest(&self) -> &[ResidentReceiptPointee] {
        &self.pointees
    }

    pub fn device_bytes(&self) -> &TrackedCudaSlice<u8> {
        &self.bytes
    }

    pub fn len_bytes(&self) -> usize {
        self.bytes.len()
    }

    pub fn relation_count_len(&self) -> u32 {
        self.relation_count_len
    }

    pub fn device_trace_field_count(&self) -> u32 {
        self.device_trace_field_count
    }

    pub fn schema_winner_count(&self) -> u32 {
        self.schema_winner_count
    }

    pub fn total_count_field_len(&self) -> u32 {
        self.count_len
    }

    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.count_ptrs);
        recorder.write(&self.bytes);
        record_receipt_pointee_uses(&self.pointees, recorder);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u32, u32) {
        (
            self.count_ptrs.device_ptr_value(),
            self.bytes.device_ptr_value(),
            self.count_len,
            u32::try_from(self.bytes.len()).unwrap_or(u32::MAX),
        )
    }

    pub(crate) fn validate_schedule_pointees(
        &self,
        manager_id: usize,
        device_ordinal: u32,
        relation_counts: &[u64],
        trace_counts: [u64; 4],
        schema_winners: &[u64],
        expected_blocks: &[crate::device_runtime::BlockId],
    ) -> Result<()> {
        validate_receipt_pointee_owners(&self.pointees, manager_id, device_ordinal)?;
        validate_receipt_schedule_mapping(
            &self.pointees,
            relation_counts,
            trace_counts,
            schema_winners,
        )?;
        validate_receipt_schedule_block_mapping(&self.pointees, expected_blocks)
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 2]> {
        Ok([
            self.count_ptrs.runtime_allocation_identity()?,
            self.bytes.runtime_allocation_identity()?,
        ])
    }
}

impl ResidentConvergenceControl {
    pub fn status_device(&self) -> &TrackedCudaSlice<ResidentTerminalStatus> {
        &self.status
    }

    pub fn status_device_ptr(&self) -> u64 {
        self.status.device_ptr_value()
    }

    pub fn iterations_device_ptr(&self) -> u64 {
        self.status.device_ptr_value() + 12
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 3]> {
        Ok([
            self.status.runtime_allocation_identity()?,
            self.changed.runtime_allocation_identity()?,
            self.loop_iterations.runtime_allocation_identity()?,
        ])
    }

    pub fn changed_device(&self) -> &TrackedCudaSlice<u32> {
        &self.changed
    }

    pub fn changed_device_ptr(&self) -> u64 {
        self.changed.device_ptr_value()
    }

    pub fn loop_iterations_device(&self) -> &TrackedCudaSlice<u32> {
        &self.loop_iterations
    }

    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.status);
        recorder.read_write(&self.changed);
        recorder.read_write(&self.loop_iterations);
    }
}

/// One-key join kinds required by the resident corpus.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentJoinKind {
    Inner,
    Semi,
}

pub(crate) fn checked_hash_slot_capacity(candidate_capacity: u64) -> Result<u64> {
    let doubled = candidate_capacity
        .checked_mul(2)
        .ok_or_else(|| XlogError::Kernel("resident hash capacity overflow".to_string()))?;
    let slots = doubled
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| XlogError::Kernel("resident hash capacity overflow".to_string()))?;
    if slots > u64::from(u32::MAX) + 1 {
        return Err(XlogError::Kernel(format!(
            "resident hash requires {slots} slots, exceeding the u32 index space"
        )));
    }
    Ok(slots)
}

fn checked_capacity(capacity: u64, label: &str) -> Result<u32> {
    u32::try_from(capacity).map_err(|_| {
        XlogError::Kernel(format!(
            "resident {label} capacity {capacity} exceeds u32::MAX"
        ))
    })
}

/// Exact manager-tracked bytes for one fixed-capacity resident relation.
pub fn resident_relation_device_bytes(schema: &Schema, capacity: u64) -> Result<u64> {
    checked_capacity(capacity, "relation")?;
    capacity
        .checked_mul(schema.row_size_bytes() as u64)
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| XlogError::Kernel("resident relation byte overflow".into()))
}

/// Exact manager-tracked bytes for the shared full-row set workspace.
pub fn resident_set_workspace_device_bytes(candidate_capacity: u64) -> Result<u64> {
    let slots = checked_hash_slot_capacity(candidate_capacity)?;
    slots
        .checked_mul(std::mem::size_of::<u64>() as u64)
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>() as u64))
        .ok_or_else(|| XlogError::Kernel("resident set workspace byte overflow".into()))
}

/// Exact manager-tracked bytes for the shared one-key join workspace.
pub fn resident_join_workspace_device_bytes(right_capacity: u64) -> Result<u64> {
    let buckets = checked_hash_slot_capacity(right_capacity)?;
    buckets
        .checked_mul(std::mem::size_of::<u32>() as u64)
        .and_then(|bytes| {
            right_capacity
                .checked_mul(std::mem::size_of::<u32>() as u64)
                .and_then(|next| bytes.checked_add(next))
        })
        .and_then(|bytes| bytes.checked_add(std::mem::size_of::<u64>() as u64))
        .ok_or_else(|| XlogError::Kernel("resident join workspace byte overflow".into()))
}

/// Exact manager-tracked bytes for convergence status and counters.
pub const fn resident_control_device_bytes() -> u64 {
    (std::mem::size_of::<ResidentTerminalStatus>() + 2 * std::mem::size_of::<u32>()) as u64
}

/// Exact manager-tracked bytes for physical and legacy-semantic invocation counters.
pub const fn resident_device_trace_bytes() -> u64 {
    (4 * std::mem::size_of::<u32>()) as u64
}

/// Exact manager-tracked bytes for device first-nonempty schema selection.
pub fn resident_schema_winners_device_bytes(head_count: usize) -> Result<u64> {
    u64::try_from(head_count.max(1))
        .ok()
        .and_then(|count| count.checked_mul(2 * std::mem::size_of::<u32>() as u64))
        .ok_or_else(|| XlogError::Kernel("resident schema winner byte overflow".into()))
}

fn resident_packed_receipt_device_bytes_for_fields(
    head_count: usize,
    schema_winner_count: usize,
) -> Result<u64> {
    let count_fields = head_count
        .checked_add(4)
        .and_then(|fields| fields.checked_add(schema_winner_count))
        .ok_or_else(|| XlogError::Kernel("resident receipt count overflow".into()))?;
    let pointer_bytes = u64::try_from(count_fields.max(1))
        .ok()
        .and_then(|count| count.checked_mul(std::mem::size_of::<u64>() as u64))
        .ok_or_else(|| XlogError::Kernel("resident receipt pointer byte overflow".into()))?;
    let packed_bytes = std::mem::size_of::<ResidentTerminalStatus>()
        .checked_add(
            count_fields
                .checked_add(1)
                .and_then(|fields| fields.checked_mul(std::mem::size_of::<u32>()))
                .ok_or_else(|| XlogError::Kernel("resident receipt byte overflow".into()))?,
        )
        .ok_or_else(|| XlogError::Kernel("resident receipt byte overflow".into()))?;
    pointer_bytes
        .checked_add(packed_bytes as u64)
        .ok_or_else(|| XlogError::Kernel("resident receipt total byte overflow".into()))
}

/// Exact manager-tracked bytes for the packed device receipt and pointer table.
pub fn resident_packed_receipt_device_bytes(head_count: usize) -> Result<u64> {
    resident_packed_receipt_device_bytes_for_fields(head_count, 0)
}

/// Exact device bytes when the receipt also carries one schema winner per head.
pub fn resident_packed_receipt_with_schema_winners_device_bytes(head_count: usize) -> Result<u64> {
    resident_packed_receipt_device_bytes_for_fields(head_count, head_count)
}

fn width(ty: ScalarType) -> Result<u32> {
    match ty {
        ScalarType::Symbol | ScalarType::U32 => Ok(4),
        ScalarType::U64 => Ok(8),
        other => Err(XlogError::Kernel(format!(
            "resident relational type {other:?} is unsupported; expected Symbol, U32, or U64"
        ))),
    }
}

fn validate_schema(schema: &Schema) -> Result<()> {
    if schema.arity() > RESIDENT_RELATIONAL_MAX_ARITY {
        return Err(XlogError::Kernel(format!(
            "resident relational arity {} exceeds {RESIDENT_RELATIONAL_MAX_ARITY}",
            schema.arity()
        )));
    }
    for column in 0..schema.arity() {
        width(schema.column_type(column).expect("schema arity checked"))?;
    }
    Ok(())
}

fn relation_view(buffer: &CudaBuffer) -> Result<ResidentRelationView> {
    validate_schema(buffer.schema())?;
    let capacity = checked_capacity(buffer.num_rows(), "relation")?;
    let mut columns = [0; RESIDENT_RELATIONAL_MAX_ARITY];
    let mut widths = [0; RESIDENT_RELATIONAL_MAX_ARITY];
    for column in 0..buffer.arity() {
        columns[column] = *buffer.column(column).expect("arity checked").device_ptr();
        widths[column] = width(
            buffer
                .schema()
                .column_type(column)
                .expect("schema arity checked"),
        )?;
    }
    Ok(ResidentRelationView {
        columns,
        widths,
        arity: buffer.arity() as u32,
        capacity,
        reserved: 0,
        num_rows: buffer.num_rows_device().device_ptr_value(),
    })
}

fn ensure_same_schema(a: &CudaBuffer, b: &CudaBuffer) -> Result<()> {
    if !same_physical_layout(a.schema(), b.schema()) {
        return Err(XlogError::Kernel(
            "resident set operands must have identical physical layouts".to_string(),
        ));
    }
    Ok(())
}

fn same_physical_layout(left: &Schema, right: &Schema) -> bool {
    left.arity() == right.arity()
        && (0..left.arity()).all(|column| left.column_type(column) == right.column_type(column))
}

fn launch_config(capacity: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (capacity.max(1).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn memset_u8_async(
    ptr: sys::CUdeviceptr,
    value: u8,
    bytes: usize,
    stream: &CudaStream,
    label: &str,
) -> Result<()> {
    if bytes == 0 {
        return Ok(());
    }
    // SAFETY: all callers pass live owned allocations and their exact byte size.
    let code = unsafe { sys::cuMemsetD8Async(ptr, value, bytes, stream.cu_stream()) };
    if code != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(XlogError::Kernel(format!(
            "resident {label} memset failed: {code:?}"
        )));
    }
    Ok(())
}

impl CudaKernelProvider {
    /// Allocate a fixed-capacity resident output relation on the cold path.
    pub fn prepare_resident_relation(
        &self,
        schema: Schema,
        capacity: u64,
    ) -> Result<ResidentRelation> {
        self.prepare_resident_relation_with_reservation(schema, capacity, None)
    }

    /// Allocate a resident relation exclusively from an admitted transaction.
    pub fn prepare_resident_relation_in_reservation(
        &self,
        schema: Schema,
        capacity: u64,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentRelation> {
        self.prepare_resident_relation_with_reservation(schema, capacity, Some(reservation))
    }

    fn prepare_resident_relation_with_reservation(
        &self,
        schema: Schema,
        capacity: u64,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentRelation> {
        if schema.arity() > RESIDENT_RELATIONAL_MAX_ARITY {
            return Err(XlogError::Kernel(format!(
                "resident relation arity {} exceeds {RESIDENT_RELATIONAL_MAX_ARITY}",
                schema.arity()
            )));
        }
        for column in 0..schema.arity() {
            width(schema.column_type(column).expect("schema arity checked"))?;
        }
        let capacity_u32 = checked_capacity(capacity, "output")?;
        let mut columns = Vec::with_capacity(schema.arity());
        for column in 0..schema.arity() {
            let bytes = (capacity as usize)
                .checked_mul(width(schema.column_type(column).expect("schema"))? as usize)
                .ok_or_else(|| XlogError::Kernel("resident column byte overflow".to_string()))?;
            let column = match reservation.as_deref_mut() {
                Some(reservation) => reservation.alloc::<u8>(bytes)?,
                None => self.memory().alloc::<u8>(bytes)?,
            };
            columns.push(CudaColumn::Owned(column));
        }
        let d_num_rows = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        let buffer = CudaBuffer::from_columns(columns, capacity, d_num_rows, schema);
        debug_assert_eq!(capacity_u32 as u64, capacity);
        Ok(ResidentRelation { buffer })
    }

    /// Initialize a private resident relation's logical set cardinality.
    ///
    /// This cold-path write addresses the device scalar directly so it does
    /// not populate or invalidate the buffer's host row-count cache.
    pub fn initialize_resident_relation_count(
        &self,
        relation: &mut ResidentRelation,
        initial_count: u32,
    ) -> Result<()> {
        if initial_count > 1 {
            return Err(XlogError::Kernel(format!(
                "resident relation initial count {initial_count} is invalid; expected 0 or 1"
            )));
        }
        let code = unsafe {
            sys::cuMemsetD32_v2(
                relation.num_rows_device().device_ptr_value(),
                initial_count,
                1,
            )
        };
        if code != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(XlogError::Kernel(format!(
                "resident relation count initialization failed: {code:?}"
            )));
        }
        Ok(())
    }

    /// Enqueue a graph-capturable logical-count clear on the supplied stream.
    pub fn record_resident_relation_clear_on_stream(
        &self,
        relation: &ResidentRelation,
        stream: &CudaStream,
    ) -> Result<()> {
        memset_u8_async(
            relation.num_rows_device().device_ptr_value(),
            0,
            std::mem::size_of::<u32>(),
            stream,
            "relation count clear",
        )
    }

    pub fn prepare_resident_set_workspace(
        &self,
        candidate_capacity: u64,
    ) -> Result<ResidentSetWorkspace> {
        self.prepare_resident_set_workspace_with_reservation(candidate_capacity, None)
    }

    /// Allocate set scratch exclusively from an admitted transaction.
    pub fn prepare_resident_set_workspace_in_reservation(
        &self,
        candidate_capacity: u64,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentSetWorkspace> {
        self.prepare_resident_set_workspace_with_reservation(candidate_capacity, Some(reservation))
    }

    fn prepare_resident_set_workspace_with_reservation(
        &self,
        candidate_capacity: u64,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentSetWorkspace> {
        let candidate_capacity_u32 = checked_capacity(candidate_capacity, "set candidate")?;
        let slots = checked_hash_slot_capacity(candidate_capacity)?;
        let slot_storage = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u64>(slots as usize)?,
            None => self.memory().alloc::<u64>(slots as usize)?,
        };
        let required = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u64>(1)?,
            None => self.memory().alloc::<u64>(1)?,
        };
        Ok(ResidentSetWorkspace {
            slots: slot_storage,
            required,
            candidate_capacity: candidate_capacity_u32,
            slot_mask: (slots - 1) as u32,
        })
    }

    pub fn prepare_resident_join_workspace(
        &self,
        right_capacity: u64,
    ) -> Result<ResidentJoinWorkspace> {
        self.prepare_resident_join_workspace_with_reservation(right_capacity, None)
    }

    /// Allocate join scratch exclusively from an admitted transaction.
    pub fn prepare_resident_join_workspace_in_reservation(
        &self,
        right_capacity: u64,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentJoinWorkspace> {
        self.prepare_resident_join_workspace_with_reservation(right_capacity, Some(reservation))
    }

    fn prepare_resident_join_workspace_with_reservation(
        &self,
        right_capacity: u64,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentJoinWorkspace> {
        let right_capacity_u32 = checked_capacity(right_capacity, "join right")?;
        let buckets = checked_hash_slot_capacity(right_capacity)?;
        let bucket_heads = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(buckets as usize)?,
            None => self.memory().alloc::<u32>(buckets as usize)?,
        };
        let next = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(right_capacity as usize)?,
            None => self.memory().alloc::<u32>(right_capacity as usize)?,
        };
        let required = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u64>(1)?,
            None => self.memory().alloc::<u64>(1)?,
        };
        Ok(ResidentJoinWorkspace {
            bucket_heads,
            next,
            required,
            right_capacity: right_capacity_u32,
            bucket_mask: (buckets - 1) as u32,
        })
    }

    pub fn prepare_resident_convergence_control(&self) -> Result<ResidentConvergenceControl> {
        self.prepare_resident_convergence_control_with_reservation(None)
    }

    /// Allocate convergence state exclusively from an admitted transaction.
    pub fn prepare_resident_convergence_control_in_reservation(
        &self,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentConvergenceControl> {
        self.prepare_resident_convergence_control_with_reservation(Some(reservation))
    }

    fn prepare_resident_convergence_control_with_reservation(
        &self,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentConvergenceControl> {
        let status = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<ResidentTerminalStatus>(1)?,
            None => self.memory().alloc::<ResidentTerminalStatus>(1)?,
        };
        let changed = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        let loop_iterations = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        Ok(ResidentConvergenceControl {
            status,
            changed,
            loop_iterations,
        })
    }

    /// Allocate fixed-address device counters on the cold path.
    pub fn prepare_resident_device_trace(&self) -> Result<ResidentDeviceTrace> {
        self.prepare_resident_device_trace_with_reservation(None)
    }

    /// Allocate device trace counters exclusively from an admitted transaction.
    pub fn prepare_resident_device_trace_in_reservation(
        &self,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentDeviceTrace> {
        self.prepare_resident_device_trace_with_reservation(Some(reservation))
    }

    fn prepare_resident_device_trace_with_reservation(
        &self,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentDeviceTrace> {
        let scan_invocations = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        let filter_invocations = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        let semantic_scan_invocations = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        let semantic_filter_invocations = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(1)?,
            None => self.memory().alloc::<u32>(1)?,
        };
        Ok(ResidentDeviceTrace {
            scan_invocations,
            filter_invocations,
            semantic_scan_invocations,
            semantic_filter_invocations,
        })
    }

    /// Allocate device first-nonempty selectors and upload their cold defaults.
    pub fn prepare_resident_schema_winners(
        &self,
        default_schema_ids: &[u32],
    ) -> Result<ResidentSchemaWinners> {
        self.prepare_resident_schema_winners_with_reservation(default_schema_ids, None)
    }

    /// Allocate schema selectors exclusively from an admitted transaction.
    pub fn prepare_resident_schema_winners_in_reservation(
        &self,
        default_schema_ids: &[u32],
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentSchemaWinners> {
        self.prepare_resident_schema_winners_with_reservation(default_schema_ids, Some(reservation))
    }

    fn prepare_resident_schema_winners_with_reservation(
        &self,
        default_schema_ids: &[u32],
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentSchemaWinners> {
        let len = u32::try_from(default_schema_ids.len())
            .map_err(|_| XlogError::Kernel("too many resident schema winners".into()))?;
        let allocation_len = default_schema_ids.len().max(1);
        let seen_nonempty = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(allocation_len)?,
            None => self.memory().alloc::<u32>(allocation_len)?,
        };
        let mut winner_schema_ids = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(allocation_len)?,
            None => self.memory().alloc::<u32>(allocation_len)?,
        };
        if !default_schema_ids.is_empty() {
            self.device()
                .inner()
                .htod_sync_copy_into(default_schema_ids, &mut winner_schema_ids)
                .map_err(|error| {
                    XlogError::Kernel(format!("resident schema winner upload failed: {error}"))
                })?;
        }
        Ok(ResidentSchemaWinners {
            seen_nonempty,
            winner_schema_ids,
            default_schema_ids: default_schema_ids.to_vec(),
            len,
        })
    }

    fn resident_receipt_pointee<T: cudarc::driver::DeviceRepr>(
        &self,
        role: ResidentReceiptPointeeRole,
        slice: &TrackedCudaSlice<T>,
        byte_offset: usize,
    ) -> Result<ResidentReceiptPointee> {
        let manager_id = slice.memory_manager_ptr_value();
        if manager_id != Arc::as_ptr(self.memory()) as usize {
            return Err(XlogError::Kernel(
                "resident receipt pointee belongs to a foreign memory manager".into(),
            ));
        }
        let provider_context = self.device().inner().stream().context();
        let slice_context = DeviceSlice::stream(slice).context();
        if !Arc::ptr_eq(slice_context, provider_context)
            || slice_context.cu_ctx() != provider_context.cu_ctx()
        {
            return Err(XlogError::Kernel(
                "resident receipt pointee belongs to a foreign CUDA context".into(),
            ));
        }
        let slice_bytes = slice
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| XlogError::Kernel("resident receipt slice byte size overflow".into()))?;
        let field_end = byte_offset
            .checked_add(4)
            .ok_or_else(|| XlogError::Kernel("resident receipt pointee offset overflow".into()))?;
        if field_end > slice_bytes {
            return Err(XlogError::Kernel(
                "resident receipt pointee is outside its source slice".into(),
            ));
        }
        let byte_offset = u64::try_from(byte_offset)
            .map_err(|_| XlogError::Kernel("resident receipt pointee offset overflow".into()))?;
        let ptr = slice
            .device_ptr_value()
            .checked_add(byte_offset)
            .ok_or_else(|| XlogError::Kernel("resident receipt pointee address overflow".into()))?;
        let runtime_block = match slice.runtime_block() {
            Some(block) => {
                if block.state != crate::device_runtime::BlockState::Live {
                    return Err(XlogError::Kernel(
                        "resident receipt pointee runtime block is not live".into(),
                    ));
                }
                Some((
                    crate::device_runtime::BlockId::from_block(block),
                    block.bytes,
                ))
            }
            None => None,
        };
        checked_receipt_pointee(role, ptr, manager_id, runtime_block)
    }

    /// Allocate and upload the immutable relation-count pointer table on the cold path.
    pub fn prepare_resident_packed_receipt(
        &self,
        relations: &[&ResidentRelation],
    ) -> Result<ResidentPackedReceipt> {
        let relation_count_len = u32::try_from(relations.len())
            .map_err(|_| XlogError::Kernel("resident receipt has too many count fields".into()))?;
        let pointees = relations
            .iter()
            .enumerate()
            .map(|(index, relation)| {
                let index = u32::try_from(index).map_err(|_| {
                    XlogError::Kernel("resident receipt head index overflow".into())
                })?;
                self.resident_receipt_pointee(
                    ResidentReceiptPointeeRole::RelationCount(index),
                    relation.num_rows_device(),
                    0,
                )
            })
            .collect::<Result<Vec<_>>>()?;
        self.prepare_resident_packed_receipt_from_pointees(pointees, relation_count_len, 0, 0, None)
    }

    /// Extend the final receipt with physical then legacy-semantic scan/filter counters.
    pub fn prepare_resident_packed_receipt_with_trace(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
    ) -> Result<ResidentPackedReceipt> {
        self.prepare_resident_packed_receipt_with_trace_and_reservation(relations, trace, None)
    }

    /// Allocate the device receipt exclusively from an admitted transaction.
    pub fn prepare_resident_packed_receipt_with_trace_in_reservation(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentPackedReceipt> {
        self.prepare_resident_packed_receipt_with_trace_and_reservation(
            relations,
            trace,
            Some(reservation),
        )
    }

    fn prepare_resident_packed_receipt_with_trace_and_reservation(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
        reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentPackedReceipt> {
        let relation_count_len = u32::try_from(relations.len())
            .map_err(|_| XlogError::Kernel("resident receipt has too many count fields".into()))?;
        let mut pointees = Vec::with_capacity(relations.len().saturating_add(4));
        for (index, relation) in relations.iter().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| XlogError::Kernel("resident receipt head index overflow".into()))?;
            pointees.push(self.resident_receipt_pointee(
                ResidentReceiptPointeeRole::RelationCount(index),
                relation.num_rows_device(),
                0,
            )?);
        }
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::ScanTrace,
            &trace.scan_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::FilterTrace,
            &trace.filter_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::SemanticScanTrace,
            &trace.semantic_scan_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::SemanticFilterTrace,
            &trace.semantic_filter_invocations,
            0,
        )?);
        self.prepare_resident_packed_receipt_from_pointees(
            pointees,
            relation_count_len,
            4,
            0,
            reservation,
        )
    }

    /// Extend the final receipt with trace counters and staged-head schema winners.
    pub fn prepare_resident_packed_receipt_with_trace_and_schema_winners(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
        winners: &ResidentSchemaWinners,
    ) -> Result<ResidentPackedReceipt> {
        self.prepare_resident_packed_receipt_with_trace_and_schema_winners_and_reservation(
            relations, trace, winners, None,
        )
    }

    /// Allocate the schema-winning receipt exclusively from an admitted transaction.
    pub fn prepare_resident_packed_receipt_with_trace_and_schema_winners_in_reservation(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
        winners: &ResidentSchemaWinners,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentPackedReceipt> {
        self.prepare_resident_packed_receipt_with_trace_and_schema_winners_and_reservation(
            relations,
            trace,
            winners,
            Some(reservation),
        )
    }

    fn prepare_resident_packed_receipt_with_trace_and_schema_winners_and_reservation(
        &self,
        relations: &[&ResidentRelation],
        trace: &ResidentDeviceTrace,
        winners: &ResidentSchemaWinners,
        reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentPackedReceipt> {
        if winners.len() as usize != relations.len() {
            return Err(XlogError::Kernel(format!(
                "resident schema winner count {} does not match relation count {}",
                winners.len(),
                relations.len()
            )));
        }
        let relation_count_len = u32::try_from(relations.len())
            .map_err(|_| XlogError::Kernel("resident receipt has too many count fields".into()))?;
        let mut pointees = Vec::with_capacity(relations.len().saturating_mul(2).saturating_add(4));
        for (index, relation) in relations.iter().enumerate() {
            let index = u32::try_from(index)
                .map_err(|_| XlogError::Kernel("resident receipt head index overflow".into()))?;
            pointees.push(self.resident_receipt_pointee(
                ResidentReceiptPointeeRole::RelationCount(index),
                relation.num_rows_device(),
                0,
            )?);
        }
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::ScanTrace,
            &trace.scan_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::FilterTrace,
            &trace.filter_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::SemanticScanTrace,
            &trace.semantic_scan_invocations,
            0,
        )?);
        pointees.push(self.resident_receipt_pointee(
            ResidentReceiptPointeeRole::SemanticFilterTrace,
            &trace.semantic_filter_invocations,
            0,
        )?);
        for index in 0..relations.len() {
            let index_u32 = u32::try_from(index)
                .map_err(|_| XlogError::Kernel("resident receipt head index overflow".into()))?;
            let byte_offset = index
                .checked_mul(std::mem::size_of::<u32>())
                .ok_or_else(|| {
                    XlogError::Kernel("resident receipt winner offset overflow".into())
                })?;
            pointees.push(self.resident_receipt_pointee(
                ResidentReceiptPointeeRole::SchemaWinner(index_u32),
                &winners.winner_schema_ids,
                byte_offset,
            )?);
        }
        self.prepare_resident_packed_receipt_from_pointees(
            pointees,
            relation_count_len,
            4,
            winners.len(),
            reservation,
        )
    }

    fn prepare_resident_packed_receipt_from_pointees(
        &self,
        pointees: Vec<ResidentReceiptPointee>,
        relation_count_len: u32,
        device_trace_field_count: u32,
        schema_winner_count: u32,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentPackedReceipt> {
        validate_receipt_pointee_ranges(&pointees)?;
        let expected_count_len = relation_count_len
            .checked_add(device_trace_field_count)
            .and_then(|count| count.checked_add(schema_winner_count))
            .ok_or_else(|| XlogError::Kernel("resident receipt field count overflow".into()))?;
        if usize::try_from(expected_count_len).ok() != Some(pointees.len()) {
            return Err(XlogError::Kernel(
                "resident receipt pointee manifest count does not match its shape".into(),
            ));
        }
        let count_ptrs: Vec<u64> = pointees.iter().map(|pointee| pointee.ptr).collect();
        let count_len = u32::try_from(count_ptrs.len())
            .map_err(|_| XlogError::Kernel("resident receipt has too many count fields".into()))?;
        let mut d_count_ptrs = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u64>(count_ptrs.len().max(1))?,
            None => self.memory().alloc::<u64>(count_ptrs.len().max(1))?,
        };
        if !count_ptrs.is_empty() {
            self.device()
                .inner()
                .htod_sync_copy_into(&count_ptrs, &mut d_count_ptrs)
                .map_err(|error| {
                    XlogError::Kernel(format!("resident receipt pointer upload failed: {error}"))
                })?;
        }
        let u32_fields = count_ptrs
            .len()
            .checked_add(1)
            .ok_or_else(|| XlogError::Kernel("resident receipt field count overflow".into()))?;
        let bytes =
            std::mem::size_of::<ResidentTerminalStatus>()
                .checked_add(u32_fields.checked_mul(4).ok_or_else(|| {
                    XlogError::Kernel("resident receipt byte size overflow".into())
                })?)
                .ok_or_else(|| XlogError::Kernel("resident receipt byte size overflow".into()))?;
        let bytes = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u8>(bytes)?,
            None => self.memory().alloc::<u8>(bytes)?,
        };
        Ok(ResidentPackedReceipt {
            count_ptrs: d_count_ptrs,
            bytes,
            pointees,
            count_len,
            relation_count_len,
            device_trace_field_count,
            schema_winner_count,
        })
    }

    /// Allocate the final receipt's exact byte length in page-locked memory.
    pub fn prepare_resident_pinned_receipt(
        &self,
        receipt: &ResidentPackedReceipt,
    ) -> Result<ResidentPinnedReceipt> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: CUDA initializes `ptr` on success; the owner frees it once.
        let code = unsafe { sys::cuMemHostAlloc(&mut ptr, receipt.len_bytes(), 0) };
        if code != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(XlogError::Kernel(format!(
                "resident pinned receipt allocation failed: {code:?}"
            )));
        }
        let ptr = std::ptr::NonNull::new(ptr.cast()).ok_or_else(|| {
            XlogError::Kernel("resident pinned receipt allocation returned null".into())
        })?;
        Ok(ResidentPinnedReceipt {
            ptr,
            len: receipt.len_bytes(),
        })
    }

    /// Copy the packed receipt once after core execution has synchronized.
    ///
    /// The destination was allocated on the cold path. This copy is accounted
    /// only in the final-observation counters after its stream wait succeeds.
    pub fn observe_resident_packed_receipt(
        &self,
        receipt: &ResidentPackedReceipt,
        pinned: &mut ResidentPinnedReceipt,
        stream: &CudaStream,
    ) -> Result<Vec<u8>> {
        if pinned.len != receipt.len_bytes() {
            return Err(XlogError::Kernel(format!(
                "resident pinned receipt size {} does not match device receipt size {}",
                pinned.len,
                receipt.len_bytes()
            )));
        }
        // SAFETY: both allocations are live for `pinned.len` bytes and the
        // mutable owner prevents host access until this stream is synchronized.
        let code = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                pinned.ptr.as_ptr().cast(),
                receipt.device_bytes().device_ptr_value(),
                pinned.len,
                stream.cu_stream(),
            )
        };
        if code != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(XlogError::Kernel(format!(
                "resident final receipt copy failed: {code:?}"
            )));
        }
        stream.synchronize().map_err(|error| {
            XlogError::Kernel(format!("resident final receipt wait failed: {error}"))
        })?;
        // SAFETY: the copy and stream wait succeeded for the complete owner.
        let bytes = unsafe { std::slice::from_raw_parts(pinned.ptr.as_ptr(), pinned.len) }.to_vec();
        self.record_final_observation_transfer(pinned.len as u64);
        Ok(bytes)
    }

    /// Enqueue one device-only terminal/count pack before the single final D2H.
    pub fn record_resident_receipt_pack_on_stream(
        &self,
        control: &ResidentConvergenceControl,
        receipt: &ResidentPackedReceipt,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_receipt_pack")
            .ok_or_else(|| XlogError::Kernel("resident_receipt_pack kernel missing".into()))?;
        let status = control.status.device_ptr_value();
        let changed = control.changed.device_ptr_value();
        let count_ptrs = receipt.count_ptrs.device_ptr_value();
        let output = receipt.bytes.device_ptr_value();
        let mut params = vec![
            status.as_kernel_param(),
            changed.as_kernel_param(),
            count_ptrs.as_kernel_param(),
            receipt.count_len.as_kernel_param(),
            output.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_receipt_pack.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident receipt pack launch: {error}")))
    }

    /// Initialize staged-head seen flags from their post-seed device counts.
    pub fn record_resident_schema_winners_initialize_on_stream(
        &self,
        winners: &ResidentSchemaWinners,
        receipt: &ResidentPackedReceipt,
        stream: &CudaStream,
    ) -> Result<()> {
        if receipt.relation_count_len != winners.len {
            return Err(XlogError::Kernel(format!(
                "resident schema winner count {} does not match receipt relation count {}",
                winners.len, receipt.relation_count_len
            )));
        }
        if winners.len == 0 {
            return Ok(());
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_schema_winners_initialize")
            .ok_or_else(|| {
                XlogError::Kernel("resident_schema_winners_initialize kernel missing".into())
            })?;
        let count_ptrs = receipt.count_ptrs.device_ptr_value();
        let seen_nonempty = winners.seen_nonempty.device_ptr_value();
        let mut params = vec![
            count_ptrs.as_kernel_param(),
            winners.len.as_kernel_param(),
            seen_nonempty.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_schema_winners_initialize.
        unsafe {
            function.clone().launch_on_stream(
                stream,
                LaunchConfig::for_num_elems(winners.len),
                &mut params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident schema winner init launch: {error}")))
    }

    /// Select the first nonempty contribution's schema without host observation.
    pub fn record_resident_schema_winner_mark_on_stream(
        &self,
        contribution_count: &TrackedCudaSlice<u32>,
        winners: &ResidentSchemaWinners,
        head_index: u32,
        schema_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        if head_index >= winners.len {
            return Err(XlogError::Kernel(format!(
                "resident schema winner index {head_index} exceeds {} heads",
                winners.len
            )));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_schema_winner_mark")
            .ok_or_else(|| {
                XlogError::Kernel("resident_schema_winner_mark kernel missing".into())
            })?;
        let contribution_count = contribution_count.device_ptr_value();
        let seen_nonempty = winners.seen_nonempty.device_ptr_value();
        let winner_schema_ids = winners.winner_schema_ids.device_ptr_value();
        let mut params = vec![
            contribution_count.as_kernel_param(),
            seen_nonempty.as_kernel_param(),
            winner_schema_ids.as_kernel_param(),
            head_index.as_kernel_param(),
            schema_id.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_schema_winner_mark.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident schema winner mark launch: {error}")))
    }

    /// Reset per-run trace counters inside the captured graph.
    pub fn record_resident_device_trace_initialize_on_stream(
        &self,
        trace: &ResidentDeviceTrace,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_trace_initialize")
            .ok_or_else(|| XlogError::Kernel("resident_trace_initialize kernel missing".into()))?;
        let scan_invocations = trace.scan_invocations.device_ptr_value();
        let filter_invocations = trace.filter_invocations.device_ptr_value();
        let semantic_scan_invocations = trace.semantic_scan_invocations.device_ptr_value();
        let semantic_filter_invocations = trace.semantic_filter_invocations.device_ptr_value();
        let mut params = vec![
            scan_invocations.as_kernel_param(),
            filter_invocations.as_kernel_param(),
            semantic_scan_invocations.as_kernel_param(),
            semantic_filter_invocations.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_trace_initialize.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident trace init launch: {error}")))
    }

    /// Count one actual resident scan invocation on the device.
    pub fn record_resident_scan_trace_on_stream(
        &self,
        trace: &ResidentDeviceTrace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.record_resident_trace_increment_on_stream(
            trace.scan_invocations.device_ptr_value(),
            trace.semantic_scan_invocations.device_ptr_value(),
            "scan",
            stream,
        )
    }

    /// Count one actual resident filter invocation on the device.
    pub fn record_resident_filter_trace_on_stream(
        &self,
        trace: &ResidentDeviceTrace,
        stream: &CudaStream,
    ) -> Result<()> {
        self.record_resident_trace_increment_on_stream(
            trace.filter_invocations.device_ptr_value(),
            trace.semantic_filter_invocations.device_ptr_value(),
            "filter",
            stream,
        )
    }

    fn record_resident_trace_increment_on_stream(
        &self,
        counter: u64,
        semantic_counter: u64,
        label: &str,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_trace_increment")
            .ok_or_else(|| XlogError::Kernel("resident_trace_increment kernel missing".into()))?;
        let mut params = vec![
            counter.as_kernel_param(),
            semantic_counter.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_trace_increment.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| {
            XlogError::Kernel(format!("resident {label} trace increment launch: {error}"))
        })
    }

    /// Enqueue initialization once before the conditional WHILE node executes.
    pub fn record_resident_control_initialize_on_stream(
        &self,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_control_initialize")
            .ok_or_else(|| {
                XlogError::Kernel("resident_control_initialize kernel missing".into())
            })?;
        let status = control.status.device_ptr_value();
        let changed = control.changed.device_ptr_value();
        let loop_iterations = control.loop_iterations.device_ptr_value();
        let mut params = vec![
            status.as_kernel_param(),
            changed.as_kernel_param(),
            loop_iterations.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches the manifest kernel signature.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident control init launch: {error}")))
    }

    /// Reset per-SCC state without clearing a prior transaction error.
    /// A zero limit publishes IterationLimit with zero completed body replays.
    pub fn record_resident_scc_begin_on_stream(
        &self,
        iteration_limit: u32,
        op_id: u32,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_scc_begin")
            .ok_or_else(|| XlogError::Kernel("resident_scc_begin kernel missing".into()))?;
        let status = control.status.device_ptr_value();
        let changed = control.changed.device_ptr_value();
        let loop_iterations = control.loop_iterations.device_ptr_value();
        let mut params = vec![
            iteration_limit.as_kernel_param(),
            op_id.as_kernel_param(),
            status.as_kernel_param(),
            changed.as_kernel_param(),
            loop_iterations.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_scc_begin.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident SCC begin launch: {error}")))
    }

    pub fn record_resident_dedup_on_stream(
        &self,
        input: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentSetWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_schema(input, output.buffer())?;
        self.record_resident_set_on_stream(
            input, input, output, workspace, control, op_id, stream, 0,
        )
    }

    pub fn record_resident_union_on_stream(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentSetWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_schema(left, right)?;
        ensure_same_schema(left, output.buffer())?;
        self.record_resident_set_on_stream(
            left, right, output, workspace, control, op_id, stream, 1,
        )
    }

    pub fn record_resident_diff_on_stream(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentSetWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        ensure_same_schema(left, right)?;
        ensure_same_schema(left, output.buffer())?;
        self.record_resident_set_on_stream(
            left, right, output, workspace, control, op_id, stream, 2,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn record_resident_set_on_stream(
        &self,
        left: &CudaBuffer,
        right: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentSetWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
        mode: u32,
    ) -> Result<()> {
        let total = left
            .num_rows()
            .checked_add(if mode == 0 { 0 } else { right.num_rows() })
            .ok_or_else(|| XlogError::Kernel("resident set capacity overflow".into()))?;
        if total > u64::from(workspace.candidate_capacity) {
            return Err(XlogError::Kernel(format!(
                "resident set workspace capacity {} is below required candidate capacity {total}",
                workspace.candidate_capacity
            )));
        }
        memset_u8_async(
            workspace.slots.device_ptr_value(),
            0,
            workspace.slots.len() * 8,
            stream,
            "set slots",
        )?;
        memset_u8_async(
            workspace.required.device_ptr_value(),
            0,
            8,
            stream,
            "set required",
        )?;
        memset_u8_async(
            output.num_rows_device().device_ptr_value(),
            0,
            4,
            stream,
            "set output count",
        )?;
        let left_view = relation_view(left)?;
        let right_view = relation_view(right)?;
        let output_view = relation_view(output.buffer())?;
        let insert = self
            .device()
            .inner()
            .get_func(MODULE, "resident_set_insert")
            .ok_or_else(|| XlogError::Kernel("resident_set_insert kernel missing".into()))?;
        let slots = workspace.slots.device_ptr_value();
        let required = workspace.required.device_ptr_value();
        let status = control.status.device_ptr_value();
        let launch_insert = |candidate: ResidentRelationView,
                             source_tag: u32,
                             emit_rows: u32,
                             materialize: u32|
         -> Result<()> {
            let mut params = vec![
                candidate.as_kernel_param(),
                left_view.as_kernel_param(),
                right_view.as_kernel_param(),
                source_tag.as_kernel_param(),
                emit_rows.as_kernel_param(),
                materialize.as_kernel_param(),
                slots.as_kernel_param(),
                workspace.slot_mask.as_kernel_param(),
                output_view.as_kernel_param(),
                required.as_kernel_param(),
                status.as_kernel_param(),
                op_id.as_kernel_param(),
            ];
            // SAFETY: parameter list exactly matches resident_set_insert.
            unsafe {
                insert.clone().launch_on_stream(
                    stream,
                    launch_config(candidate.capacity),
                    &mut params,
                )
            }
            .map_err(|error| XlogError::Kernel(format!("resident set insert launch: {error}")))
        };
        let launch_pass = |materialize: u32| -> Result<()> {
            match mode {
                0 => launch_insert(left_view, 0, 1, materialize)?,
                1 => {
                    launch_insert(left_view, 0, 1, materialize)?;
                    launch_insert(right_view, 1, 1, materialize)?;
                }
                2 => {
                    launch_insert(right_view, 1, 0, materialize)?;
                    launch_insert(left_view, 0, 1, materialize)?;
                }
                _ => unreachable!(),
            }
            Ok(())
        };
        launch_pass(0)?;
        self.record_resident_finalize_on_stream(
            "resident_set_finalize",
            required,
            output_view,
            status,
            op_id,
            stream,
        )?;
        memset_u8_async(
            workspace.slots.device_ptr_value(),
            0,
            workspace.slots.len() * 8,
            stream,
            "set materialization slots",
        )?;
        memset_u8_async(
            workspace.required.device_ptr_value(),
            0,
            8,
            stream,
            "set materialization required",
        )?;
        launch_pass(1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn record_resident_join_on_stream(
        &self,
        kind: ResidentJoinKind,
        left: &CudaBuffer,
        left_key: usize,
        right: &CudaBuffer,
        right_key: usize,
        output: &ResidentRelation,
        workspace: &ResidentJoinWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        let left_view = relation_view(left)?;
        let right_view = relation_view(right)?;
        if right_view.capacity > workspace.right_capacity {
            return Err(XlogError::Kernel(format!(
                "resident join workspace right capacity {} is below relation capacity {}",
                workspace.right_capacity, right_view.capacity
            )));
        }
        if left_key >= left.arity() || right_key >= right.arity() {
            return Err(XlogError::Kernel(
                "resident join key index out of bounds".into(),
            ));
        }
        if width(left.schema().column_type(left_key).expect("key checked"))?
            != width(right.schema().column_type(right_key).expect("key checked"))?
        {
            return Err(XlogError::Kernel(
                "resident join key widths must match".into(),
            ));
        }
        let expected_schema = match kind {
            ResidentJoinKind::Semi => left.schema().clone(),
            ResidentJoinKind::Inner => {
                let mut columns = left.schema().columns.clone();
                columns.extend(right.schema().columns.iter().cloned());
                Schema::new(columns)
            }
        };
        if !same_physical_layout(output.buffer().schema(), &expected_schema) {
            return Err(XlogError::Kernel(
                "resident join output schema does not match join kind".into(),
            ));
        }
        let output_view = relation_view(output.buffer())?;
        memset_u8_async(
            workspace.bucket_heads.device_ptr_value(),
            0xff,
            workspace.bucket_heads.len() * 4,
            stream,
            "join buckets",
        )?;
        memset_u8_async(
            workspace.required.device_ptr_value(),
            0,
            8,
            stream,
            "join required",
        )?;
        memset_u8_async(
            output.num_rows_device().device_ptr_value(),
            0,
            4,
            stream,
            "join output count",
        )?;
        let buckets = workspace.bucket_heads.device_ptr_value();
        let next = workspace.next.device_ptr_value();
        let required = workspace.required.device_ptr_value();
        let status = control.status.device_ptr_value();
        let left_key = left_key as u32;
        let right_key = right_key as u32;
        let build = self
            .device()
            .inner()
            .get_func(MODULE, "resident_join_build")
            .ok_or_else(|| XlogError::Kernel("resident_join_build kernel missing".into()))?;
        let mut build_params = vec![
            right_view.as_kernel_param(),
            right_key.as_kernel_param(),
            buckets.as_kernel_param(),
            workspace.bucket_mask.as_kernel_param(),
            next.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_join_build.
        unsafe {
            build.clone().launch_on_stream(
                stream,
                launch_config(right_view.capacity),
                &mut build_params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident join build launch: {error}")))?;
        let probe_name = match kind {
            ResidentJoinKind::Inner => "resident_join_probe_inner",
            ResidentJoinKind::Semi => "resident_join_probe_semi",
        };
        let probe = self
            .device()
            .inner()
            .get_func(MODULE, probe_name)
            .ok_or_else(|| XlogError::Kernel(format!("{probe_name} kernel missing")))?;
        let launch_probe = |materialize: u32| -> Result<()> {
            let mut probe_params = vec![
                left_view.as_kernel_param(),
                left_key.as_kernel_param(),
                right_view.as_kernel_param(),
                right_key.as_kernel_param(),
                buckets.as_kernel_param(),
                workspace.bucket_mask.as_kernel_param(),
                next.as_kernel_param(),
                output_view.as_kernel_param(),
                required.as_kernel_param(),
                materialize.as_kernel_param(),
                status.as_kernel_param(),
                op_id.as_kernel_param(),
            ];
            // SAFETY: both probe variants have the same parameter ABI.
            unsafe {
                probe.clone().launch_on_stream(
                    stream,
                    launch_config(left_view.capacity),
                    &mut probe_params,
                )
            }
            .map_err(|error| XlogError::Kernel(format!("resident join probe launch: {error}")))
        };
        launch_probe(0)?;
        self.record_resident_finalize_on_stream(
            "resident_join_finalize",
            required,
            output_view,
            status,
            op_id,
            stream,
        )?;
        memset_u8_async(
            workspace.required.device_ptr_value(),
            0,
            8,
            stream,
            "join materialization required",
        )?;
        launch_probe(1)
    }

    #[allow(clippy::too_many_arguments)]
    fn record_resident_finalize_on_stream(
        &self,
        kernel: &str,
        required: u64,
        output: ResidentRelationView,
        status: u64,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, kernel)
            .ok_or_else(|| XlogError::Kernel(format!("{kernel} kernel missing")))?;
        let mut params = vec![
            required.as_kernel_param(),
            output.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: set and join finalize share this exact parameter ABI.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident finalize launch: {error}")))
    }

    /// Enqueue the per-iteration whole-SCC changed aggregate reset.
    pub fn record_resident_changed_reset_on_stream(
        &self,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_changed_reset")
            .ok_or_else(|| XlogError::Kernel("resident_changed_reset kernel missing".into()))?;
        let changed = control.changed.device_ptr_value();
        let mut params = vec![changed.as_kernel_param()];
        // SAFETY: parameter list exactly matches resident_changed_reset.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident changed reset launch: {error}")))
    }

    /// Atomically fold one recursive head's novel count into the SCC aggregate.
    pub fn record_resident_changed_mark_on_stream(
        &self,
        novel_count: &TrackedCudaSlice<u32>,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        if novel_count.is_empty() {
            return Err(XlogError::Kernel(
                "resident changed mark requires one device count".into(),
            ));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_changed_mark")
            .ok_or_else(|| XlogError::Kernel("resident_changed_mark kernel missing".into()))?;
        let novel = novel_count.device_ptr_value();
        let changed = control.changed.device_ptr_value();
        let mut params = vec![novel.as_kernel_param(), changed.as_kernel_param()];
        // SAFETY: parameter list exactly matches resident_changed_mark.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident changed mark launch: {error}")))
    }

    /// Enqueue the single WHILE-tail device convergence decision.
    pub fn record_resident_convergence_on_stream(
        &self,
        conditional_handle: u64,
        iteration_limit: u32,
        op_id: u32,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        if iteration_limit == 0 {
            return Err(XlogError::Kernel(
                "resident convergence requires a positive iteration limit".into(),
            ));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_convergence")
            .ok_or_else(|| XlogError::Kernel("resident_convergence kernel missing".into()))?;
        let status = control.status.device_ptr_value();
        let changed = control.changed.device_ptr_value();
        let loop_iterations = control.loop_iterations.device_ptr_value();
        let mut params = vec![
            conditional_handle.as_kernel_param(),
            iteration_limit.as_kernel_param(),
            op_id.as_kernel_param(),
            status.as_kernel_param(),
            changed.as_kernel_param(),
            loop_iterations.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_convergence.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident convergence launch: {error}")))
    }

    /// Publish transaction Success after every parent segment and SCC completes.
    pub fn record_resident_terminal_success_on_stream(
        &self,
        op_id: u32,
        control: &ResidentConvergenceControl,
        stream: &CudaStream,
    ) -> Result<()> {
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_terminal_success")
            .ok_or_else(|| XlogError::Kernel("resident_terminal_success kernel missing".into()))?;
        let status = control.status.device_ptr_value();
        let mut params = vec![op_id.as_kernel_param(), status.as_kernel_param()];
        // SAFETY: parameter list exactly matches resident_terminal_success.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident terminal success launch: {error}")))
    }

    /// Enqueue a device-written terminal status for execution-path tests.
    ///
    /// The kernel uses the same first-terminal-status publication protocol as
    /// production failures; this API never copies a status record to device.
    #[doc(hidden)]
    pub fn record_resident_test_status_on_stream(
        &self,
        control: &ResidentConvergenceControl,
        injected: ResidentTerminalStatus,
        stream: &CudaStream,
    ) -> Result<()> {
        if !matches!(
            injected.code,
            code if code == ResidentTerminalCode::Success as u32
                || code == ResidentTerminalCode::IterationLimit as u32
                || code == ResidentTerminalCode::CapacityOverflow as u32
                || code == ResidentTerminalCode::ResourceExhausted as u32
        ) {
            return Err(XlogError::Kernel(format!(
                "resident test status code {} is not terminal",
                injected.code
            )));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, "resident_test_status")
            .ok_or_else(|| XlogError::Kernel("resident_test_status kernel missing".into()))?;
        let code = injected.code;
        let op_id = injected.op_id;
        let resource_code = injected.resource_code;
        let iterations = injected.iterations;
        let limit = injected.limit;
        let required = injected.required;
        let capacity = injected.capacity;
        let status = control.status.device_ptr_value();
        let mut params = vec![
            code.as_kernel_param(),
            op_id.as_kernel_param(),
            resource_code.as_kernel_param(),
            iterations.as_kernel_param(),
            limit.as_kernel_param(),
            required.as_kernel_param(),
            capacity.as_kernel_param(),
            status.as_kernel_param(),
        ];
        // SAFETY: parameter list exactly matches resident_test_status.
        unsafe {
            function
                .clone()
                .launch_on_stream(stream, LaunchConfig::for_num_elems(1), &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident test status launch: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use xlog_core::{MemoryBudget, ScalarType, Schema};

    use crate::{cuda_graph::CapturedCudaGraph, CudaDevice, GpuMemoryManager};

    use super::{
        checked_hash_slot_capacity, checked_receipt_pointee, record_receipt_pointee_uses,
        resident_device_trace_bytes, resident_packed_receipt_with_schema_winners_device_bytes,
        resident_schema_winners_device_bytes, validate_receipt_pointee_owners,
        validate_receipt_pointee_ranges, validate_receipt_schedule_block_mapping,
        validate_receipt_schedule_mapping, ResidentJoinKind, ResidentPackedReceipt,
        ResidentReceiptPointee, ResidentReceiptPointeeRole, ResidentResourceCode,
        ResidentTerminalCode, ResidentTerminalStatus,
    };

    #[test]
    fn dual_device_trace_and_packed_receipt_have_exact_bytes() {
        assert_eq!(resident_device_trace_bytes(), 16);
        assert_eq!(
            resident_packed_receipt_with_schema_winners_device_bytes(3).unwrap(),
            164
        );
    }

    #[test]
    fn schema_winners_retain_immutable_host_defaults_for_replay() {
        let _: fn(&super::ResidentSchemaWinners) -> &[u32] =
            super::ResidentSchemaWinners::default_schema_ids;
    }

    #[test]
    fn receipt_pointee_manifest_rejects_overlap_and_checked_range_overflow() {
        let block = crate::device_runtime::BlockId {
            ptr: 0x1000,
            generation: crate::device_runtime::Generation(3),
            alloc_stream: crate::device_runtime::StreamId(4),
            device_ordinal: 0,
        };
        let relation = checked_receipt_pointee(
            ResidentReceiptPointeeRole::RelationCount(0),
            0x1000,
            7,
            Some((block, 16)),
        )
        .expect("valid relation count");
        let overlapping = checked_receipt_pointee(
            ResidentReceiptPointeeRole::ScanTrace,
            0x1002,
            7,
            Some((block, 16)),
        )
        .expect("individually valid overlapping counter");
        assert!(validate_receipt_pointee_ranges(&[relation, overlapping]).is_err());

        assert!(checked_receipt_pointee(
            ResidentReceiptPointeeRole::FilterTrace,
            u64::MAX - 1,
            7,
            None,
        )
        .is_err());
        assert!(checked_receipt_pointee(
            ResidentReceiptPointeeRole::FilterTrace,
            0x100e,
            7,
            Some((block, 16)),
        )
        .is_err());
    }

    #[test]
    fn receipt_manifest_mapping_requires_exact_role_and_pointer_order() {
        let zero_heads = [
            checked_receipt_pointee(ResidentReceiptPointeeRole::ScanTrace, 0x800, 7, None).unwrap(),
            checked_receipt_pointee(ResidentReceiptPointeeRole::FilterTrace, 0x810, 7, None)
                .unwrap(),
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::SemanticScanTrace,
                0x820,
                7,
                None,
            )
            .unwrap(),
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::SemanticFilterTrace,
                0x830,
                7,
                None,
            )
            .unwrap(),
        ];
        validate_receipt_schedule_mapping(&zero_heads, &[], [0x800, 0x810, 0x820, 0x830], &[])
            .expect("zero-head receipt mapping");

        let one_head = [
            checked_receipt_pointee(ResidentReceiptPointeeRole::RelationCount(0), 0x900, 7, None)
                .unwrap(),
            checked_receipt_pointee(ResidentReceiptPointeeRole::ScanTrace, 0x910, 7, None).unwrap(),
            checked_receipt_pointee(ResidentReceiptPointeeRole::FilterTrace, 0x920, 7, None)
                .unwrap(),
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::SemanticScanTrace,
                0x930,
                7,
                None,
            )
            .unwrap(),
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::SemanticFilterTrace,
                0x940,
                7,
                None,
            )
            .unwrap(),
            checked_receipt_pointee(ResidentReceiptPointeeRole::SchemaWinner(0), 0x950, 7, None)
                .unwrap(),
        ];
        validate_receipt_schedule_mapping(
            &one_head,
            &[0x900],
            [0x910, 0x920, 0x930, 0x940],
            &[0x950],
        )
        .expect("one-head receipt mapping");

        let entries = [
            (ResidentReceiptPointeeRole::RelationCount(0), 0x1000),
            (ResidentReceiptPointeeRole::RelationCount(1), 0x1010),
            (ResidentReceiptPointeeRole::ScanTrace, 0x1020),
            (ResidentReceiptPointeeRole::FilterTrace, 0x1030),
            (ResidentReceiptPointeeRole::SemanticScanTrace, 0x1040),
            (ResidentReceiptPointeeRole::SemanticFilterTrace, 0x1050),
            (ResidentReceiptPointeeRole::SchemaWinner(0), 0x1060),
            (ResidentReceiptPointeeRole::SchemaWinner(1), 0x1070),
        ]
        .map(|(role, ptr)| checked_receipt_pointee(role, ptr, 7, None).unwrap());

        validate_receipt_schedule_mapping(
            &entries,
            &[0x1000, 0x1010],
            [0x1020, 0x1030, 0x1040, 0x1050],
            &[0x1060, 0x1070],
        )
        .expect("exact receipt mapping");
        assert!(validate_receipt_schedule_mapping(
            &entries,
            &[0x1010, 0x1000],
            [0x1020, 0x1030, 0x1040, 0x1050],
            &[0x1060, 0x1070],
        )
        .is_err());
        assert!(validate_receipt_schedule_mapping(
            &entries,
            &[0x1000, 0x1010],
            [0x1020, 0x1030, 0x1040, 0x1050],
            &[0x1060, 0x9999],
        )
        .is_err());
    }

    #[test]
    fn additive_receipt_manifest_requires_exact_manager_and_runtime_block_owner() {
        let block = crate::device_runtime::BlockId {
            ptr: 0x2000,
            generation: crate::device_runtime::Generation(1),
            alloc_stream: crate::device_runtime::StreamId(2),
            device_ordinal: 3,
        };
        let owned = checked_receipt_pointee(
            ResidentReceiptPointeeRole::RelationCount(0),
            0x2000,
            17,
            Some((block, 8)),
        )
        .unwrap();
        validate_receipt_pointee_owners(&[owned], 17, 3).expect("exact owner");

        let foreign_manager = checked_receipt_pointee(
            ResidentReceiptPointeeRole::RelationCount(0),
            0x2000,
            99,
            Some((block, 8)),
        )
        .unwrap();
        assert!(validate_receipt_pointee_owners(&[foreign_manager], 17, 3).is_err());
        let untracked = checked_receipt_pointee(
            ResidentReceiptPointeeRole::RelationCount(0),
            0x2000,
            17,
            None,
        )
        .unwrap();
        assert!(validate_receipt_pointee_owners(&[untracked], 17, 3).is_err());
        assert!(validate_receipt_pointee_owners(&[owned], 17, 4).is_err());
    }

    #[test]
    fn receipt_schedule_mapping_requires_full_block_identity() {
        let owned = crate::device_runtime::BlockId {
            ptr: 0x2000,
            generation: crate::device_runtime::Generation(1),
            alloc_stream: crate::device_runtime::StreamId(2),
            device_ordinal: 3,
        };
        let different_stream = crate::device_runtime::BlockId {
            alloc_stream: crate::device_runtime::StreamId(9),
            ..owned
        };
        let pointee = checked_receipt_pointee(
            ResidentReceiptPointeeRole::RelationCount(0),
            0x2000,
            17,
            Some((owned, 8)),
        )
        .unwrap();

        validate_receipt_schedule_block_mapping(&[pointee], &[owned])
            .expect("exact allocation generation and stream");
        assert!(validate_receipt_schedule_block_mapping(&[pointee], &[different_stream]).is_err());
    }

    #[test]
    fn packed_receipt_retains_immutable_pointee_manifest() {
        let _: fn(&ResidentPackedReceipt) -> &[ResidentReceiptPointee] =
            ResidentPackedReceipt::pointee_manifest;
    }

    #[test]
    fn receipt_manifest_records_every_runtime_pointee_as_read() {
        let first = crate::device_runtime::BlockId {
            ptr: 0x3000,
            generation: crate::device_runtime::Generation(1),
            alloc_stream: crate::device_runtime::StreamId(2),
            device_ordinal: 0,
        };
        let second = crate::device_runtime::BlockId {
            ptr: 0x4000,
            ..first
        };
        let pointees = [
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::ScanTrace,
                first.ptr,
                17,
                Some((first, 4)),
            )
            .unwrap(),
            checked_receipt_pointee(
                ResidentReceiptPointeeRole::FilterTrace,
                second.ptr,
                17,
                Some((second, 4)),
            )
            .unwrap(),
        ];
        let mut recorder =
            crate::launch::LaunchRecorder::new_strict(crate::device_runtime::StreamId(8));

        record_receipt_pointee_uses(&pointees, &mut recorder);

        assert_eq!(recorder.recorded_count(), 2);
    }

    #[test]
    fn relational_external_owners_expose_complete_allocation_snapshots() {
        type Owner = Option<crate::memory::RuntimeAllocationIdentity>;
        let _: fn(&super::ResidentSetWorkspace) -> xlog_core::Result<[Owner; 2]> =
            super::ResidentSetWorkspace::schedule_owner_snapshots;
        let _: fn(&super::ResidentJoinWorkspace) -> xlog_core::Result<[Owner; 3]> =
            super::ResidentJoinWorkspace::schedule_owner_snapshots;
        let _: fn(&super::ResidentConvergenceControl) -> xlog_core::Result<[Owner; 3]> =
            super::ResidentConvergenceControl::schedule_owner_snapshots;
        let _: fn(&super::ResidentDeviceTrace) -> xlog_core::Result<[Owner; 4]> =
            super::ResidentDeviceTrace::schedule_owner_snapshots;
        let _: fn(&super::ResidentSchemaWinners) -> xlog_core::Result<[Owner; 2]> =
            super::ResidentSchemaWinners::schedule_owner_snapshots;
        let _: fn(&ResidentPackedReceipt) -> xlog_core::Result<[Owner; 2]> =
            ResidentPackedReceipt::schedule_owner_snapshots;
    }

    fn provider() -> Option<super::CudaKernelProvider> {
        let device = match CudaDevice::new(0) {
            Ok(device) => Arc::new(device),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA device initialization failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident CUDA test: {error}");
                return None;
            }
        };
        let memory = Arc::new(GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(512 * 1024 * 1024),
        ));
        match super::CudaKernelProvider::new(device, memory) {
            Ok(provider) => Some(provider),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but resident provider setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident CUDA test: {error}");
                None
            }
        }
    }

    fn pair_schema() -> Schema {
        Schema::new(vec![
            ("key".to_string(), ScalarType::Symbol),
            ("value".to_string(), ScalarType::U64),
        ])
    }

    fn pair_buffer(
        provider: &super::CudaKernelProvider,
        keys: &[u32],
        values: &[u64],
    ) -> crate::CudaBuffer {
        let key_bytes: Vec<u8> = keys.iter().flat_map(|value| value.to_le_bytes()).collect();
        let value_bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        provider
            .create_buffer_from_slices(&[&key_bytes, &value_bytes], pair_schema())
            .expect("pair buffer")
    }

    #[test]
    fn terminal_status_wire_layout_is_stable() {
        assert_eq!(std::mem::size_of::<ResidentTerminalStatus>(), 40);
        assert_eq!(std::mem::align_of::<ResidentTerminalStatus>(), 8);
        assert_eq!(ResidentTerminalCode::Running as u32, 0);
        assert_eq!(ResidentTerminalCode::Success as u32, 1);
        assert_eq!(ResidentTerminalCode::IterationLimit as u32, 2);
        assert_eq!(ResidentTerminalCode::CapacityOverflow as u32, 3);
        assert_eq!(ResidentTerminalCode::ResourceExhausted as u32, 4);
        assert_eq!(ResidentResourceCode::SetHashSlots as u32, 1);
        assert_eq!(ResidentResourceCode::JoinBuckets as u32, 2);
        assert_eq!(ResidentResourceCode::JoinChains as u32, 3);
        assert_eq!(ResidentResourceCode::InputRows as u32, 4);
        assert_eq!(ResidentResourceCode::OutputRows as u32, 5);
    }

    #[test]
    fn hash_capacity_is_checked_and_keeps_load_at_most_one_half() {
        assert_eq!(checked_hash_slot_capacity(0).unwrap(), 1);
        assert_eq!(checked_hash_slot_capacity(1).unwrap(), 2);
        assert_eq!(checked_hash_slot_capacity(4_994).unwrap(), 16_384);
        assert_eq!(checked_hash_slot_capacity(9_988).unwrap(), 32_768);
        assert!(checked_hash_slot_capacity(u64::from(u32::MAX)).is_err());
    }

    #[test]
    fn schema_winner_workspace_bytes_are_exact_and_checked() {
        assert_eq!(resident_schema_winners_device_bytes(0).unwrap(), 8);
        assert_eq!(resident_schema_winners_device_bytes(1).unwrap(), 8);
        assert_eq!(resident_schema_winners_device_bytes(3).unwrap(), 24);
        assert!(resident_schema_winners_device_bytes(usize::MAX).is_err());
    }

    #[test]
    fn real_cuda_schema_winner_uses_first_nonempty_and_packs_one_receipt() {
        let Some(provider) = provider() else { return };
        let mut empty = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("empty relation");
        provider
            .initialize_resident_relation_count(&mut empty, 0)
            .expect("empty count");
        let mut nonempty = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("nonempty relation");
        provider
            .initialize_resident_relation_count(&mut nonempty, 1)
            .expect("nonempty count");
        let mut all_empty = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("all-empty relation");
        provider
            .initialize_resident_relation_count(&mut all_empty, 0)
            .expect("all-empty count");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let trace = provider.prepare_resident_device_trace().expect("trace");
        let winners = provider
            .prepare_resident_schema_winners(&[11, 22, 33])
            .expect("schema winners");
        let receipt = provider
            .prepare_resident_packed_receipt_with_trace_and_schema_winners(
                &[&empty, &nonempty, &all_empty],
                &trace,
                &winners,
            )
            .expect("receipt");
        let mut pinned = provider
            .prepare_resident_pinned_receipt(&receipt)
            .expect("pinned receipt");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_device_trace_initialize_on_stream(&trace, &stream)?;
            provider
                .record_resident_schema_winners_initialize_on_stream(&winners, &receipt, &stream)?;
            provider.record_resident_schema_winner_mark_on_stream(
                empty.num_rows_device(),
                &winners,
                0,
                33,
                &stream,
            )?;
            provider.record_resident_schema_winner_mark_on_stream(
                nonempty.num_rows_device(),
                &winners,
                0,
                44,
                &stream,
            )?;
            provider.record_resident_schema_winner_mark_on_stream(
                all_empty.num_rows_device(),
                &winners,
                2,
                88,
                &stream,
            )?;
            provider.record_resident_schema_winner_mark_on_stream(
                nonempty.num_rows_device(),
                &winners,
                0,
                55,
                &stream,
            )?;
            provider.record_resident_schema_winner_mark_on_stream(
                empty.num_rows_device(),
                &winners,
                1,
                66,
                &stream,
            )?;
            provider.record_resident_schema_winner_mark_on_stream(
                nonempty.num_rows_device(),
                &winners,
                1,
                77,
                &stream,
            )?;
            provider.record_resident_terminal_success_on_stream(91, &control, &stream)?;
            provider.record_resident_receipt_pack_on_stream(&control, &receipt, &stream)
        })
        .expect("capture schema winners");

        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        provider.reset_final_observation_transfer_stats();
        graph.launch(&stream).expect("launch schema winners");
        stream.synchronize().expect("schema winner core sync");
        let ordinary = provider.host_transfer_stats();
        let launch_metadata = provider.host_launch_metadata_transfer_stats();
        assert_eq!(ordinary.htod_calls, 0);
        assert_eq!(ordinary.htod_bytes, 0);
        assert_eq!(ordinary.dtoh_calls, 0);
        assert_eq!(ordinary.dtoh_bytes, 0);
        assert_eq!(launch_metadata.htod_calls, 0);
        assert_eq!(launch_metadata.htod_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);

        let bytes = provider
            .observe_resident_packed_receipt(&receipt, &mut pinned, &stream)
            .expect("observe schema winners");
        assert_eq!(receipt.relation_count_len(), 3);
        assert_eq!(receipt.device_trace_field_count(), 4);
        assert_eq!(receipt.schema_winner_count(), 3);
        assert_eq!(receipt.total_count_field_len(), 10);
        assert_eq!(receipt.len_bytes(), 84);
        assert_eq!(u32::from_ne_bytes(bytes[72..76].try_into().unwrap()), 44);
        assert_eq!(u32::from_ne_bytes(bytes[76..80].try_into().unwrap()), 22);
        assert_eq!(u32::from_ne_bytes(bytes[80..84].try_into().unwrap()), 33);
        let final_observation = provider.final_observation_transfer_stats();
        assert_eq!(final_observation.dtoh_calls, 1);
        assert_eq!(final_observation.dtoh_bytes, 84);
        assert_eq!(final_observation.pinned_receipts, 1);
    }

    #[test]
    fn real_cuda_relation_count_initialization_is_checked_and_cache_free() {
        let Some(provider) = provider() else { return };
        let mut relation = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("relation");

        provider
            .initialize_resident_relation_count(&mut relation, 1)
            .expect("singleton initialization");
        assert_eq!(relation.buffer().cached_row_count(), None);
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(relation.num_rows_device())
                .expect("singleton count"),
            vec![1]
        );

        provider
            .initialize_resident_relation_count(&mut relation, 0)
            .expect("empty initialization");
        assert_eq!(relation.buffer().cached_row_count(), None);
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(relation.num_rows_device())
                .expect("empty count"),
            vec![0]
        );

        let error = provider
            .initialize_resident_relation_count(&mut relation, 2)
            .expect_err("non-set count must be rejected");
        assert!(error.to_string().contains("expected 0 or 1"));
        assert_eq!(relation.buffer().cached_row_count(), None);
    }

    #[test]
    fn real_cuda_relation_count_clear_is_graph_capturable_on_supplied_stream() {
        let Some(provider) = provider() else { return };
        let mut relation = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("relation");
        provider
            .initialize_resident_relation_count(&mut relation, 1)
            .expect("singleton initialization");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");

        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_relation_clear_on_stream(&relation, &stream)
        })
        .expect("capture relation clear");
        graph.launch(&stream).expect("launch relation clear");
        stream.synchronize().expect("relation clear sync");

        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(relation.num_rows_device())
                .expect("cleared count"),
            vec![0]
        );
        assert_eq!(relation.buffer().cached_row_count(), None);
    }

    #[test]
    fn real_cuda_final_receipt_uses_one_pinned_dtoh_without_ordinary_counters() {
        let Some(provider) = provider() else { return };
        let mut relation = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("relation");
        provider
            .initialize_resident_relation_count(&mut relation, 1)
            .expect("singleton initialization");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let receipt = provider
            .prepare_resident_packed_receipt(&[&relation])
            .expect("device receipt");
        let mut pinned = provider
            .prepare_resident_pinned_receipt(&receipt)
            .expect("pinned receipt");
        assert_eq!(pinned.len_bytes(), receipt.len_bytes());
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_terminal_success_on_stream(77, &control, &stream)?;
            provider.record_resident_receipt_pack_on_stream(&control, &receipt, &stream)
        })
        .expect("capture receipt production");

        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        provider.reset_final_observation_transfer_stats();
        graph.launch(&stream).expect("launch receipt production");
        stream.synchronize().expect("core graph sync");
        let bytes = provider
            .observe_resident_packed_receipt(&receipt, &mut pinned, &stream)
            .expect("final receipt observation");

        assert_eq!(bytes.len(), receipt.len_bytes());
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 1);
        assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 77);
        assert_eq!(u32::from_ne_bytes(bytes[40..44].try_into().unwrap()), 0);
        assert_eq!(u32::from_ne_bytes(bytes[44..48].try_into().unwrap()), 1);
        let ordinary = provider.host_transfer_stats();
        let launch_metadata = provider.host_launch_metadata_transfer_stats();
        assert_eq!(ordinary.htod_calls, 0);
        assert_eq!(ordinary.htod_bytes, 0);
        assert_eq!(ordinary.dtoh_calls, 0);
        assert_eq!(ordinary.dtoh_bytes, 0);
        assert_eq!(launch_metadata.htod_calls, 0);
        assert_eq!(launch_metadata.htod_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        let final_observation = provider.final_observation_transfer_stats();
        assert_eq!(final_observation.dtoh_calls, 1);
        assert_eq!(final_observation.dtoh_bytes, receipt.len_bytes() as u64);
        assert_eq!(final_observation.pinned_receipts, 1);
    }

    #[test]
    fn real_cuda_device_trace_packs_exact_invocation_counts_in_final_receipt() {
        let Some(provider) = provider() else { return };
        let mut relation = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("relation");
        provider
            .initialize_resident_relation_count(&mut relation, 1)
            .expect("singleton initialization");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let trace = provider
            .prepare_resident_device_trace()
            .expect("device trace");
        let receipt = provider
            .prepare_resident_packed_receipt_with_trace(&[&relation], &trace)
            .expect("traced device receipt");
        let mut pinned = provider
            .prepare_resident_pinned_receipt(&receipt)
            .expect("pinned receipt");
        assert_eq!(receipt.relation_count_len(), 1);
        assert_eq!(receipt.device_trace_field_count(), 4);
        assert_eq!(receipt.total_count_field_len(), 5);
        assert_eq!(receipt.len_bytes(), 64);
        assert_eq!(pinned.len_bytes(), receipt.len_bytes());
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_device_trace_initialize_on_stream(&trace, &stream)?;
            provider.record_resident_scan_trace_on_stream(&trace, &stream)?;
            provider.record_resident_scan_trace_on_stream(&trace, &stream)?;
            provider.record_resident_filter_trace_on_stream(&trace, &stream)?;
            provider.record_resident_scan_trace_on_stream(&trace, &stream)?;
            provider.record_resident_filter_trace_on_stream(&trace, &stream)?;
            provider.record_resident_terminal_success_on_stream(88, &control, &stream)?;
            provider.record_resident_receipt_pack_on_stream(&control, &receipt, &stream)
        })
        .expect("capture traced receipt production");

        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        provider.reset_final_observation_transfer_stats();
        graph
            .launch(&stream)
            .expect("launch traced receipt production");
        stream.synchronize().expect("core graph sync");
        let ordinary_before_observation = provider.host_transfer_stats();
        let launch_metadata_before_observation = provider.host_launch_metadata_transfer_stats();
        assert_eq!(ordinary_before_observation.htod_calls, 0);
        assert_eq!(ordinary_before_observation.htod_bytes, 0);
        assert_eq!(ordinary_before_observation.dtoh_calls, 0);
        assert_eq!(ordinary_before_observation.dtoh_bytes, 0);
        assert_eq!(launch_metadata_before_observation.htod_calls, 0);
        assert_eq!(launch_metadata_before_observation.htod_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);

        let bytes = provider
            .observe_resident_packed_receipt(&receipt, &mut pinned, &stream)
            .expect("final traced receipt observation");
        assert_eq!(bytes.len(), receipt.len_bytes());
        assert_eq!(u32::from_ne_bytes(bytes[0..4].try_into().unwrap()), 1);
        assert_eq!(u32::from_ne_bytes(bytes[4..8].try_into().unwrap()), 88);
        assert_eq!(u32::from_ne_bytes(bytes[40..44].try_into().unwrap()), 0);
        assert_eq!(u32::from_ne_bytes(bytes[44..48].try_into().unwrap()), 1);
        assert_eq!(u32::from_ne_bytes(bytes[48..52].try_into().unwrap()), 3);
        assert_eq!(u32::from_ne_bytes(bytes[52..56].try_into().unwrap()), 2);
        assert_eq!(u32::from_ne_bytes(bytes[56..60].try_into().unwrap()), 3);
        assert_eq!(u32::from_ne_bytes(bytes[60..64].try_into().unwrap()), 2);
        let ordinary_after_observation = provider.host_transfer_stats();
        assert_eq!(ordinary_after_observation.htod_calls, 0);
        assert_eq!(ordinary_after_observation.dtoh_calls, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        let final_observation = provider.final_observation_transfer_stats();
        assert_eq!(final_observation.dtoh_calls, 1);
        assert_eq!(final_observation.dtoh_bytes, receipt.len_bytes() as u64);
        assert_eq!(final_observation.pinned_receipts, 1);
    }

    #[test]
    fn real_cuda_test_status_kernel_publishes_first_error_after_indexed_op() {
        let Some(provider) = provider() else { return };
        let input = pair_buffer(&provider, &[1], &[10]);
        let output = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("output");
        let workspace = provider
            .prepare_resident_set_workspace(1)
            .expect("workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let first = ResidentTerminalStatus {
            code: ResidentTerminalCode::ResourceExhausted as u32,
            op_id: 61,
            resource_code: ResidentResourceCode::SetHashSlots as u32,
            iterations: 2,
            limit: 9,
            reserved: u32::MAX,
            required: 33,
            capacity: 16,
        };
        let second = ResidentTerminalStatus {
            code: ResidentTerminalCode::CapacityOverflow as u32,
            op_id: 62,
            resource_code: ResidentResourceCode::OutputRows as u32,
            iterations: 3,
            limit: 10,
            reserved: 0,
            required: 65,
            capacity: 32,
        };
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_dedup_on_stream(
                &input, &output, &workspace, &control, 61, &stream,
            )?;
            provider.record_resident_test_status_on_stream(&control, first, &stream)?;
            provider.record_resident_test_status_on_stream(&control, second, &stream)?;
            provider.record_resident_terminal_success_on_stream(63, &control, &stream)
        })
        .expect("capture injected device status");

        provider.reset_host_transfer_stats();
        graph
            .launch(&stream)
            .expect("launch injected device status");
        stream.synchronize().expect("injected status sync");
        let observed = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("status observation")[0];

        assert_eq!(
            observed,
            ResidentTerminalStatus {
                reserved: 0,
                ..first
            }
        );
        let transfers = provider.host_transfer_stats();
        let launch_metadata = provider.host_launch_metadata_transfer_stats();
        assert_eq!(transfers.htod_calls, 0);
        assert_eq!(transfers.htod_bytes, 0);
        assert_eq!(launch_metadata.htod_calls, 0);
        assert_eq!(launch_metadata.htod_bytes, 0);
    }

    #[test]
    fn real_cuda_graph_union_deduplicates_full_mixed_width_rows() {
        let Some(provider) = provider() else { return };
        let left = pair_buffer(&provider, &[1, 1, 2], &[10, 10, 20]);
        let right = pair_buffer(&provider, &[2, 3], &[20, 30]);
        let output = provider
            .prepare_resident_relation(pair_schema(), 3)
            .expect("output");
        let workspace = provider
            .prepare_resident_set_workspace(5)
            .expect("workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_union_on_stream(
                &left, &right, &output, &workspace, &control, 7, &stream,
            )
        })
        .expect("capture resident union");
        graph.launch(&stream).expect("launch resident union");
        stream.synchronize().expect("resident union sync");
        let keys = provider
            .download_column::<u32>(output.buffer(), 0)
            .expect("keys");
        let values = provider
            .download_column::<u64>(output.buffer(), 1)
            .expect("values");
        let mut rows: Vec<_> = keys.into_iter().zip(values).collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![(1, 10), (2, 20), (3, 30)]);
    }

    #[test]
    fn real_cuda_graph_diff_compares_and_deduplicates_full_rows() {
        let Some(provider) = provider() else { return };
        let left = pair_buffer(&provider, &[1, 1, 1, 2], &[10, 10, 11, 20]);
        let right = pair_buffer(&provider, &[1, 9], &[10, 90]);
        let output = provider
            .prepare_resident_relation(pair_schema(), 2)
            .expect("output");
        let workspace = provider
            .prepare_resident_set_workspace(6)
            .expect("workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_diff_on_stream(
                &left, &right, &output, &workspace, &control, 10, &stream,
            )
        })
        .expect("capture resident diff");
        graph.launch(&stream).expect("launch resident diff");
        stream.synchronize().expect("resident diff sync");
        let keys = provider
            .download_column::<u32>(output.buffer(), 0)
            .expect("keys");
        let values = provider
            .download_column::<u64>(output.buffer(), 1)
            .expect("values");
        let mut rows: Vec<_> = keys.into_iter().zip(values).collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![(1, 11), (2, 20)]);
    }

    #[test]
    fn real_cuda_nullary_union_implements_unit_set_semantics() {
        let Some(provider) = provider() else { return };
        let unit_schema = Schema::new(Vec::new());
        let left = provider
            .create_zero_arity_buffer(unit_schema.clone(), 1)
            .expect("left unit");
        let right = provider
            .create_zero_arity_buffer(unit_schema.clone(), 1)
            .expect("right unit");
        let output = provider
            .prepare_resident_relation(unit_schema, 1)
            .expect("unit output");
        let workspace = provider
            .prepare_resident_set_workspace(2)
            .expect("workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_union_on_stream(
                &left, &right, &output, &workspace, &control, 11, &stream,
            )
        })
        .expect("capture nullary union");
        graph.launch(&stream).expect("launch nullary union");
        stream.synchronize().expect("nullary union sync");
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(output.num_rows_device())
                .expect("unit count"),
            vec![1]
        );
    }

    #[test]
    fn real_cuda_inner_and_semi_join_preserve_required_multiplicity() {
        let Some(provider) = provider() else { return };
        let left = pair_buffer(&provider, &[1, 1, 2], &[10, 11, 12]);
        let right = pair_buffer(&provider, &[1, 1, 3], &[20, 21, 22]);
        let mut inner_columns = pair_schema().columns;
        inner_columns.extend(pair_schema().columns);
        let inner = provider
            .prepare_resident_relation(Schema::new(inner_columns), 4)
            .expect("inner output");
        let semi = provider
            .prepare_resident_relation(pair_schema(), 2)
            .expect("semi output");
        let inner_workspace = provider
            .prepare_resident_join_workspace(3)
            .expect("inner workspace");
        let semi_workspace = provider
            .prepare_resident_join_workspace(3)
            .expect("semi workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_join_on_stream(
                ResidentJoinKind::Inner,
                &left,
                0,
                &right,
                0,
                &inner,
                &inner_workspace,
                &control,
                8,
                &stream,
            )?;
            provider.record_resident_join_on_stream(
                ResidentJoinKind::Semi,
                &left,
                0,
                &right,
                0,
                &semi,
                &semi_workspace,
                &control,
                9,
                &stream,
            )
        })
        .expect("capture resident joins");
        graph.launch(&stream).expect("launch resident joins");
        stream.synchronize().expect("resident joins sync");
        let inner_left_values = provider
            .download_column::<u64>(inner.buffer(), 1)
            .expect("inner left values");
        let inner_right_values = provider
            .download_column::<u64>(inner.buffer(), 3)
            .expect("inner right values");
        let mut pairs: Vec<_> = inner_left_values
            .into_iter()
            .zip(inner_right_values)
            .collect();
        pairs.sort_unstable();
        assert_eq!(pairs, vec![(10, 20), (10, 21), (11, 20), (11, 21)]);
        let mut semi_values = provider
            .download_column::<u64>(semi.buffer(), 1)
            .expect("semi values");
        semi_values.sort_unstable();
        assert_eq!(semi_values, vec![10, 11]);
    }

    #[test]
    fn real_cuda_capacity_failure_reports_exact_required_and_clamps_output() {
        let Some(provider) = provider() else { return };
        let input = pair_buffer(&provider, &[1, 2, 3], &[10, 20, 30]);
        let output = provider
            .prepare_resident_relation(pair_schema(), 1)
            .expect("output");
        let workspace = provider
            .prepare_resident_set_workspace(3)
            .expect("workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider.device().inner().stream();
        provider
            .record_resident_control_initialize_on_stream(&control, stream)
            .expect("init");
        provider
            .record_resident_dedup_on_stream(&input, &output, &workspace, &control, 23, stream)
            .expect("dedup");
        stream.synchronize().expect("capacity sync");
        let receipt = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("receipt")[0];
        assert_eq!(receipt.code, ResidentTerminalCode::CapacityOverflow as u32);
        assert_eq!(receipt.op_id, 23);
        assert_eq!(
            receipt.resource_code,
            ResidentResourceCode::OutputRows as u32
        );
        assert_eq!(receipt.required, 3);
        assert_eq!(receipt.capacity, 1);
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(output.num_rows_device())
                .unwrap(),
            vec![1]
        );
    }

    #[test]
    fn real_cuda_set_overflow_reports_exact_required_without_writing_output_storage() {
        let Some(provider) = provider() else { return };
        let left = pair_buffer(&provider, &[1, 2], &[10, 20]);
        let right = pair_buffer(&provider, &[3, 4], &[30, 40]);
        let sentinel_keys = [0xdead_beef_u32, 0xcafe_babe];
        let sentinel_values = [0x1111_2222_3333_4444_u64, 0x5555_6666_7777_8888];
        let output = super::ResidentRelation {
            buffer: pair_buffer(&provider, &sentinel_keys, &sentinel_values),
        };
        let workspace = provider
            .prepare_resident_set_workspace(4)
            .expect("set workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("set control");
        let stream = provider.device().inner().stream();
        provider
            .record_resident_control_initialize_on_stream(&control, stream)
            .expect("set control initialize");
        provider
            .record_resident_union_on_stream(
                &left, &right, &output, &workspace, &control, 31, stream,
            )
            .expect("overflowing union");
        stream.synchronize().expect("overflowing union sync");
        let status = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("set overflow status")[0];
        assert_eq!(status.code, ResidentTerminalCode::CapacityOverflow as u32);
        assert_eq!(status.op_id, 31);
        assert_eq!(status.required, 4);
        assert_eq!(status.capacity, 2);
        let key_bytes: Vec<u8> = sentinel_keys
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        let value_bytes: Vec<u8> = sentinel_values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(output.buffer().column(0).expect("set key column"))
                .expect("set key storage"),
            key_bytes
        );
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(output.buffer().column(1).expect("set value column"))
                .expect("set value storage"),
            value_bytes
        );
    }

    #[test]
    fn real_cuda_join_overflow_reports_exact_required_without_writing_output_storage() {
        let Some(provider) = provider() else { return };
        let left = pair_buffer(&provider, &[1, 1], &[10, 11]);
        let right = pair_buffer(&provider, &[1, 1], &[20, 21]);
        let mut output_columns = pair_schema().columns;
        output_columns.extend(pair_schema().columns);
        let output_schema = Schema::new(output_columns);
        let sentinel_columns = [
            vec![0xaaaa_aaaa_u32.to_le_bytes(), 0xbbbb_bbbb_u32.to_le_bytes()].concat(),
            vec![
                0x1111_2222_3333_4444_u64.to_le_bytes(),
                0x5555_6666_7777_8888_u64.to_le_bytes(),
            ]
            .concat(),
            vec![0xcccc_cccc_u32.to_le_bytes(), 0xdddd_dddd_u32.to_le_bytes()].concat(),
            vec![
                0x9999_aaaa_bbbb_cccc_u64.to_le_bytes(),
                0xdddd_eeee_ffff_0000_u64.to_le_bytes(),
            ]
            .concat(),
        ];
        let slices: Vec<&[u8]> = sentinel_columns.iter().map(Vec::as_slice).collect();
        let output = super::ResidentRelation {
            buffer: provider
                .create_buffer_from_slices(&slices, output_schema)
                .expect("sentinel join output"),
        };
        let workspace = provider
            .prepare_resident_join_workspace(2)
            .expect("join workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("join control");
        let stream = provider.device().inner().stream();
        provider
            .record_resident_control_initialize_on_stream(&control, stream)
            .expect("join control initialize");
        provider
            .record_resident_join_on_stream(
                ResidentJoinKind::Inner,
                &left,
                0,
                &right,
                0,
                &output,
                &workspace,
                &control,
                32,
                stream,
            )
            .expect("overflowing join");
        stream.synchronize().expect("overflowing join sync");
        let status = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("join overflow status")[0];
        assert_eq!(status.code, ResidentTerminalCode::CapacityOverflow as u32);
        assert_eq!(status.op_id, 32);
        assert_eq!(status.required, 4);
        assert_eq!(status.capacity, 2);
        for (column, expected) in sentinel_columns.iter().enumerate() {
            let actual = provider
                .device()
                .inner()
                .dtoh_sync_copy(output.buffer().column(column).expect("join output column"))
                .expect("join output storage");
            assert_eq!(actual.as_slice(), expected.as_slice());
        }
    }

    #[test]
    fn real_cuda_conditional_while_stops_from_device_convergence() {
        let Some(provider) = provider() else { return };
        let empty = provider
            .create_empty_buffer(Schema::new(vec![("novel".into(), ScalarType::U32)]))
            .expect("empty count owner");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        provider
            .record_resident_control_initialize_on_stream(&control, &stream)
            .expect("control init");
        provider
            .record_resident_scc_begin_on_stream(8, 31, &control, &stream)
            .expect("SCC begin");
        stream.synchronize().expect("cold control init sync");
        let graph = CapturedCudaGraph::conditional_while_on_stream(&stream, 1, true, |body| {
            body.capture_on_stream(&stream, || {
                provider.record_resident_changed_reset_on_stream(&control, &stream)?;
                provider.record_resident_changed_mark_on_stream(
                    empty.num_rows_device(),
                    &control,
                    &stream,
                )?;
                provider.record_resident_convergence_on_stream(
                    body.handle(),
                    8,
                    31,
                    &control,
                    &stream,
                )
            })
        })
        .expect("conditional convergence graph");
        graph.launch(&stream).expect("launch conditional graph");
        provider
            .record_resident_terminal_success_on_stream(99, &control, &stream)
            .expect("terminal success");
        stream.synchronize().expect("conditional convergence sync");
        let receipt = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("convergence receipt")[0];
        assert_eq!(receipt.code, ResidentTerminalCode::Success as u32);
        assert_eq!(receipt.op_id, 99);
        assert_eq!(receipt.iterations, 1);
        assert_eq!(
            provider
                .device()
                .inner()
                .dtoh_sync_copy(control.changed_device())
                .expect("changed flag"),
            vec![0]
        );
    }

    #[test]
    fn real_cuda_multi_head_convergence_ors_every_recursive_head() {
        let Some(provider) = provider() else { return };
        let empty = provider
            .create_empty_buffer(Schema::new(vec![("empty".into(), ScalarType::U32)]))
            .expect("empty head");
        let changed = provider
            .create_buffer_from_slice(
                &[7_u32],
                Schema::new(vec![("changed".into(), ScalarType::U32)]),
            )
            .expect("changed head");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        provider
            .record_resident_control_initialize_on_stream(&control, &stream)
            .expect("control init");
        provider
            .record_resident_scc_begin_on_stream(2, 41, &control, &stream)
            .expect("SCC begin");
        stream.synchronize().expect("control init sync");
        let graph = CapturedCudaGraph::conditional_while_on_stream(&stream, 1, true, |body| {
            body.capture_on_stream(&stream, || {
                provider.record_resident_changed_reset_on_stream(&control, &stream)?;
                provider.record_resident_changed_mark_on_stream(
                    empty.num_rows_device(),
                    &control,
                    &stream,
                )?;
                provider.record_resident_changed_mark_on_stream(
                    changed.num_rows_device(),
                    &control,
                    &stream,
                )?;
                provider.record_resident_convergence_on_stream(
                    body.handle(),
                    2,
                    41,
                    &control,
                    &stream,
                )
            })
        })
        .expect("multi-head conditional graph");
        graph.launch(&stream).expect("multi-head launch");
        stream.synchronize().expect("multi-head sync");
        let receipt = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("multi-head receipt")[0];
        assert_eq!(receipt.code, ResidentTerminalCode::IterationLimit as u32);
        assert_eq!(receipt.op_id, 41);
        assert_eq!(receipt.iterations, 2);
        assert_eq!(receipt.limit, 2);
    }

    #[test]
    fn real_cuda_zero_iteration_limit_fails_before_body_replay() {
        let Some(provider) = provider() else { return };
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = provider.device().inner().stream();
        provider
            .record_resident_control_initialize_on_stream(&control, stream)
            .expect("control init");
        provider
            .record_resident_scc_begin_on_stream(0, 51, &control, stream)
            .expect("zero-limit SCC begin");
        stream.synchronize().expect("zero-limit sync");
        let receipt = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("zero-limit receipt")[0];
        assert_eq!(receipt.code, ResidentTerminalCode::IterationLimit as u32);
        assert_eq!(receipt.op_id, 51);
        assert_eq!(receipt.iterations, 0);
        assert_eq!(receipt.limit, 0);
    }
}
