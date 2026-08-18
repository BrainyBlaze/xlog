use std::ptr::NonNull;
use std::sync::Arc;

use cudarc::driver::{sys, CudaStream, DeviceRepr, LaunchConfig};
use xlog_core::{Result, ScalarType, XlogError};

use super::resident_filter_project::ResidentFilterScratch;
use super::resident_relational::{
    ResidentConvergenceControl, ResidentDeviceTrace, ResidentJoinWorkspace, ResidentPackedReceipt,
    ResidentSchemaWinners, ResidentSetWorkspace, ResidentTerminalStatus,
};
use super::CudaKernelProvider;
use crate::cuda_compat::{AsKernelParam, DeviceSlice, LaunchAsync};
use crate::cuda_graph::{
    CapturedCudaGraph, ConditionalCudaGraphBody, ConditionalCudaGraphSequenceBuilder, CudaGraphNode,
};
use crate::device_runtime::{StreamId, XlogDeviceRuntime};
use crate::launch::LaunchRecorder;
use crate::memory::{
    CudaBuffer, GpuMemoryReservation, RuntimeAllocationIdentity, TrackedCudaSlice,
};

const RESIDENT_SCHEDULE_MAX_ARITY: usize = 17;
const RESIDENT_SCHEDULE_BLOCK_SIZE: u32 = 256;
pub const RESIDENT_SCHEDULE_ABI_VERSION: u32 = 3;
const RESIDENT_SCHEDULE_MAX_ROWS: u64 = 65_536;
const MODULE: &str = "xlog_resident_schedule";
const KERNEL: &str = "resident_schedule_execute";
pub const RESIDENT_SCHEDULE_SLOT_SOURCE: u32 = 1;
pub const RESIDENT_SCHEDULE_SLOT_PERMANENT: u32 = 2;
pub const RESIDENT_SCHEDULE_SLOT_DEFINED: u32 = 4;
const SOURCE_SLOT: u32 = RESIDENT_SCHEDULE_SLOT_SOURCE;

fn validate_runtime_allocation_fields(
    manager_id: usize,
    allocation_ptr: u64,
    allocation_bytes: usize,
    block_id: crate::device_runtime::BlockId,
    block_bytes: usize,
    block_state: crate::device_runtime::BlockState,
    expected_manager_id: usize,
    expected_device_ordinal: u32,
) -> Result<(u64, u64)> {
    if manager_id != expected_manager_id {
        return Err(XlogError::Kernel(
            "resident schedule allocation belongs to a foreign memory manager".into(),
        ));
    }
    if block_state != crate::device_runtime::BlockState::Live {
        return Err(XlogError::Kernel(
            "resident schedule allocation runtime block is not live".into(),
        ));
    }
    if block_id.device_ordinal != expected_device_ordinal {
        return Err(XlogError::Kernel(
            "resident schedule allocation belongs to a foreign CUDA device".into(),
        ));
    }
    let allocation_bytes = u64::try_from(allocation_bytes)
        .map_err(|_| XlogError::Kernel("resident schedule allocation size overflow".into()))?;
    let block_bytes = u64::try_from(block_bytes)
        .map_err(|_| XlogError::Kernel("resident schedule runtime block size overflow".into()))?;
    let allocation_end = allocation_ptr
        .checked_add(allocation_bytes)
        .ok_or_else(|| XlogError::Kernel("resident schedule allocation range overflow".into()))?;
    let block_end = block_id.ptr.checked_add(block_bytes).ok_or_else(|| {
        XlogError::Kernel("resident schedule runtime block range overflow".into())
    })?;
    if allocation_ptr < block_id.ptr || allocation_end > block_end {
        return Err(XlogError::Kernel(
            "resident schedule allocation range is outside its runtime block".into(),
        ));
    }
    Ok((allocation_ptr, allocation_end))
}

fn validate_runtime_allocation_identity(
    identity: &RuntimeAllocationIdentity,
    domain: &ResidentExecutionDomain,
) -> Result<(u64, u64)> {
    let range = validate_runtime_allocation_fields(
        identity.manager_id,
        identity.allocation_ptr,
        identity.allocation_bytes,
        identity.block_id,
        identity.block_bytes,
        identity.block_state,
        domain.memory_manager_identity,
        domain.runtime.device_ordinal(),
    )?;
    if !Arc::ptr_eq(&identity.context, &domain.context)
        || identity.context.cu_ctx() != domain.context.cu_ctx()
    {
        return Err(XlogError::Kernel(
            "resident schedule allocation belongs to a foreign CUDA context".into(),
        ));
    }
    Ok(range)
}

fn validate_schedule_allocation(
    identity: Option<RuntimeAllocationIdentity>,
    domain: &ResidentExecutionDomain,
    ranges: &mut Vec<(u64, u64)>,
) -> Result<RuntimeAllocationIdentity> {
    let identity = identity.ok_or_else(|| {
        XlogError::Kernel(
            "resident schedule requires every allocation to be runtime tracked".into(),
        )
    })?;
    let range = validate_runtime_allocation_identity(&identity, domain)?;
    insert_nonoverlapping_allocation_range(ranges, range)?;
    Ok(identity)
}

fn insert_nonoverlapping_allocation_range(
    ranges: &mut Vec<(u64, u64)>,
    range: (u64, u64),
) -> Result<()> {
    if range.0 >= range.1 {
        return Err(XlogError::Kernel(
            "resident schedule allocation range is empty or reversed".into(),
        ));
    }
    if ranges
        .iter()
        .any(|previous| range.0 < previous.1 && previous.0 < range.1)
    {
        return Err(XlogError::Kernel(
            "resident schedule allocations have overlapping byte ranges".into(),
        ));
    }
    ranges.push(range);
    Ok(())
}

fn validate_receipt_slot_mapping(
    receipt_slots: &[u32],
    slot_flags: &[u32],
    head_count: u32,
) -> Result<Vec<usize>> {
    let head_count = usize::try_from(head_count)
        .map_err(|_| XlogError::Kernel("resident schedule head count overflow".into()))?;
    if receipt_slots.len() != head_count {
        return Err(XlogError::Kernel(
            "resident schedule receipt slot count does not match its head count".into(),
        ));
    }
    let mut validated = Vec::with_capacity(receipt_slots.len());
    for &slot in receipt_slots {
        let slot = usize::try_from(slot)
            .map_err(|_| XlogError::Kernel("resident schedule receipt slot overflow".into()))?;
        let flags = *slot_flags.get(slot).ok_or_else(|| {
            XlogError::Kernel("resident schedule receipt slot is out of range".into())
        })?;
        if flags & RESIDENT_SCHEDULE_SLOT_SOURCE != 0
            || flags & RESIDENT_SCHEDULE_SLOT_PERMANENT == 0
        {
            return Err(XlogError::Kernel(
                "resident schedule receipt slot is not a permanent output".into(),
            ));
        }
        if validated.contains(&slot) {
            return Err(XlogError::Kernel(
                "resident schedule receipt slots contain a duplicate".into(),
            ));
        }
        validated.push(slot);
    }
    Ok(validated)
}

fn validate_execution_domain(
    provider: &CudaKernelProvider,
    domain: &ResidentExecutionDomain,
) -> Result<()> {
    let manager_id = Arc::as_ptr(provider.memory()) as usize;
    let manager_runtime = provider.memory().runtime().ok_or_else(|| {
        XlogError::Kernel(
            "resident execution domain requires a runtime-backed memory manager".into(),
        )
    })?;
    if domain.provider_identity != provider.provider_identity()
        || domain.memory_manager_identity != manager_id
        || !Arc::ptr_eq(manager_runtime, &domain.runtime)
        || !Arc::ptr_eq(provider.device(), provider.memory().device())
        || !Arc::ptr_eq(provider.device(), domain.runtime.device())
    {
        return Err(XlogError::Kernel(
            "resident execution domain provider, manager, and runtime identities differ".into(),
        ));
    }
    let device_ordinal = u32::try_from(provider.device().ordinal()).map_err(|_| {
        XlogError::Kernel("resident execution domain device ordinal overflow".into())
    })?;
    if domain.runtime.device_ordinal() != device_ordinal
        || !domain.runtime.supports_block_use_tracking()
    {
        return Err(XlogError::Kernel(
            "resident execution domain runtime is incompatible with the provider".into(),
        ));
    }
    let resolved_stream = domain
        .runtime
        .stream_pool()
        .resolve(domain.stream_id)
        .ok_or_else(|| {
            XlogError::Kernel(
                "resident execution domain stream id is not owned by the runtime".into(),
            )
        })?;
    if !Arc::ptr_eq(&resolved_stream, &domain.stream) {
        return Err(XlogError::Kernel(
            "resident execution domain stream does not match its runtime stream id".into(),
        ));
    }
    let provider_context = provider.device().inner().stream().context();
    if !Arc::ptr_eq(&domain.context, provider_context)
        || !Arc::ptr_eq(domain.stream.context(), &domain.context)
        || domain.context.cu_ctx() != provider_context.cu_ctx()
        || domain.stream.context().cu_ctx() != domain.context.cu_ctx()
    {
        return Err(XlogError::Kernel(
            "resident execution domain belongs to a foreign CUDA context".into(),
        ));
    }
    Ok(())
}

/// Typed operation tag with the same four-byte representation consumed by CUDA.
#[repr(u32)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum ResidentScheduleOpKind {
    #[default]
    Unit = 0,
    Scan = 1,
    Filter = 2,
    Project = 3,
    JoinInner = 4,
    JoinSemi = 5,
    Union = 6,
    Diff = 7,
    TestStatus = 8,
    TraceDelta = 9,
}

const OP_UNIT: ResidentScheduleOpKind = ResidentScheduleOpKind::Unit;
const OP_SCAN: ResidentScheduleOpKind = ResidentScheduleOpKind::Scan;
const OP_FILTER: ResidentScheduleOpKind = ResidentScheduleOpKind::Filter;
const OP_PROJECT: ResidentScheduleOpKind = ResidentScheduleOpKind::Project;
const OP_JOIN_INNER: ResidentScheduleOpKind = ResidentScheduleOpKind::JoinInner;
const OP_JOIN_SEMI: ResidentScheduleOpKind = ResidentScheduleOpKind::JoinSemi;
const OP_UNION: ResidentScheduleOpKind = ResidentScheduleOpKind::Union;
const OP_DIFF: ResidentScheduleOpKind = ResidentScheduleOpKind::Diff;
const OP_TEST_STATUS: ResidentScheduleOpKind = ResidentScheduleOpKind::TestStatus;
const OP_TRACE_DELTA: ResidentScheduleOpKind = ResidentScheduleOpKind::TraceDelta;

pub const RESIDENT_SCHEDULE_OP_MARK_NOVELTY: u32 = 1;
pub const RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER: u32 = 2;
pub const RESIDENT_SCHEDULE_TRACE_SEMANTIC_GUARD: u32 = 1;
pub const RESIDENT_SCHEDULE_REGION_INITIALIZE: u32 = 1;
pub const RESIDENT_SCHEDULE_REGION_SCC_BEGIN: u32 = 2;
pub const RESIDENT_SCHEDULE_REGION_RECURSIVE: u32 = 4;
pub const RESIDENT_SCHEDULE_REGION_FINALIZE: u32 = 8;

/// Device relation view shared by every compact resident operation.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentRelationView {
    pub columns: [u64; RESIDENT_SCHEDULE_MAX_ARITY],
    pub widths: [u32; RESIDENT_SCHEDULE_MAX_ARITY],
    pub arity: u32,
    pub capacity: u32,
    pub reserved: u32,
    pub num_rows: u64,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentRelationView {}

/// Fixed-address relation table entry with replay-generation metadata.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentRelationSlot {
    pub relation: ResidentRelationView,
    pub generation: u32,
    pub flags: u32,
    pub initial_count: u32,
    pub schema_tag: u32,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentRelationSlot {}

/// One operation in stable schedule order.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentOpDescriptor {
    pub kind: ResidentScheduleOpKind,
    pub flags: u32,
    pub op_id: u32,
    pub out: u32,
    pub in0: u32,
    pub in1: u32,
    pub in0_generation: u32,
    pub in1_generation: u32,
    pub out_generation: u32,
    pub aux_offset: u32,
    pub aux_count: u32,
    pub left_key: u32,
    pub right_key: u32,
    pub scan_delta: u32,
    pub filter_delta: u32,
    pub schema_winner_head: u32,
    pub schema_winner_id: u32,
    pub reserved: u32,
}

// SAFETY: stable C layout, no references, and `kind` is created through its
// public typed variants before this host-owned descriptor is uploaded.
unsafe impl DeviceRepr for ResidentOpDescriptor {}

impl ResidentOpDescriptor {
    /// Construct a nullary Unit leaf that writes one logical empty tuple.
    pub fn unit(op_id: u32, out: u32, out_generation: u32) -> Self {
        Self {
            kind: ResidentScheduleOpKind::Unit,
            op_id,
            out,
            out_generation,
            ..Self::default()
        }
    }

    /// Construct a Scan leaf that binds an existing immutable source slot.
    pub fn scan(op_id: u32, source: u32, source_generation: u32) -> Self {
        Self {
            kind: ResidentScheduleOpKind::Scan,
            op_id,
            out: source,
            in0: source,
            in0_generation: source_generation,
            out_generation: source_generation,
            ..Self::default()
        }
    }

    pub fn with_schema_winner(mut self, head: u32, schema_id: u32) -> Self {
        self.flags |= RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER;
        self.schema_winner_head = head;
        self.schema_winner_id = schema_id;
        self
    }

    pub fn test_status(status: ResidentTerminalStatus) -> Result<Self> {
        if status.reserved != 0 {
            return Err(XlogError::Kernel(
                "resident schedule test status reserved field is nonzero".into(),
            ));
        }
        Ok(Self {
            kind: OP_TEST_STATUS,
            op_id: status.op_id,
            out: status.code,
            in0: status.resource_code,
            in1: status.iterations,
            in0_generation: status.limit,
            in1_generation: status.reserved,
            out_generation: status.required as u32,
            aux_offset: (status.required >> 32) as u32,
            aux_count: status.capacity as u32,
            left_key: (status.capacity >> 32) as u32,
            ..Default::default()
        })
    }

    pub fn trace_delta(
        scan_delta: u32,
        filter_delta: u32,
        semantic_guard: Option<(u32, u32)>,
    ) -> Self {
        let (flags, in0, in0_generation) = match semantic_guard {
            Some((slot, generation)) => (RESIDENT_SCHEDULE_TRACE_SEMANTIC_GUARD, slot, generation),
            None => (0, 0, 0),
        };
        Self {
            kind: OP_TRACE_DELTA,
            flags,
            in0,
            in0_generation,
            scan_delta,
            filter_delta,
            ..Default::default()
        }
    }
}

fn decode_test_status(op: &ResidentOpDescriptor) -> Result<ResidentTerminalStatus> {
    if op.kind != OP_TEST_STATUS
        || op.flags != 0
        || op.in1_generation != 0
        || op.right_key != 0
        || op.scan_delta != 0
        || op.filter_delta != 0
        || op.schema_winner_head != 0
        || op.schema_winner_id != 0
        || op.reserved != 0
    {
        return Err(XlogError::Kernel(
            "resident schedule test status descriptor is invalid".into(),
        ));
    }
    Ok(ResidentTerminalStatus {
        code: op.out,
        op_id: op.op_id,
        resource_code: op.in0,
        iterations: op.in1,
        limit: op.in0_generation,
        reserved: op.in1_generation,
        required: u64::from(op.out_generation) | (u64::from(op.aux_offset) << 32),
        capacity: u64::from(op.aux_count) | (u64::from(op.left_key) << 32),
    })
}

fn decode_trace_delta(op: &ResidentOpDescriptor) -> Result<(u32, u32, Option<(u32, u32)>)> {
    let has_semantic_guard = op.flags & RESIDENT_SCHEDULE_TRACE_SEMANTIC_GUARD != 0;
    if op.kind != OP_TRACE_DELTA
        || op.flags & !RESIDENT_SCHEDULE_TRACE_SEMANTIC_GUARD != 0
        || op.op_id != 0
        || op.out != 0
        || op.in1 != 0
        || op.in1_generation != 0
        || op.out_generation != 0
        || op.aux_offset != 0
        || op.aux_count != 0
        || op.left_key != 0
        || op.right_key != 0
        || op.schema_winner_head != 0
        || op.schema_winner_id != 0
        || op.reserved != 0
        || (!has_semantic_guard && (op.in0 != 0 || op.in0_generation != 0))
    {
        return Err(XlogError::Kernel(
            "resident schedule trace delta descriptor is invalid".into(),
        ));
    }
    Ok((
        op.scan_delta,
        op.filter_delta,
        has_semantic_guard.then_some((op.in0, op.in0_generation)),
    ))
}

/// Contiguous operation range with an explicit barrier boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentWaveDescriptor {
    pub first_op: u32,
    pub op_count: u32,
    pub flags: u32,
    pub reserved: u32,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentWaveDescriptor {}

/// Contiguous wave range executed by one cooperative launch.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentRegionDescriptor {
    pub first_wave: u32,
    pub wave_count: u32,
    pub iteration_limit: u32,
    pub op_id: u32,
    pub flags: u32,
    pub first_slot: u32,
    pub slot_count: u32,
    pub generation_offset: u32,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentRegionDescriptor {}

/// Device pointers and exact bounds for one compact resident schedule.
#[repr(C, align(16))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentScheduleHeader {
    pub slots: u64,
    pub ops: u64,
    pub waves: u64,
    pub regions: u64,
    pub generation_metadata: u64,
    pub filter_comparisons: u64,
    pub project_expressions: u64,
    pub filter_mask: u64,
    pub filter_prefix: u64,
    pub filter_block_sums: u64,
    pub filter_block_offsets: u64,
    pub set_slots: u64,
    pub set_required: u64,
    pub join_buckets: u64,
    pub join_next: u64,
    pub join_required: u64,
    pub status: u64,
    pub changed: u64,
    pub iterations: u64,
    pub scan_trace: u64,
    pub filter_trace: u64,
    pub semantic_scan_trace: u64,
    pub semantic_filter_trace: u64,
    pub schema_seen_nonempty: u64,
    pub schema_winner_ids: u64,
    pub receipt_table: u64,
    pub receipt_bytes: u64,
    pub slot_count: u32,
    pub op_count: u32,
    pub wave_count: u32,
    pub region_count: u32,
    pub filter_comparison_count: u32,
    pub project_expression_count: u32,
    pub filter_capacity: u32,
    pub filter_block_count: u32,
    pub set_slot_mask: u32,
    pub set_candidate_capacity: u32,
    pub join_bucket_mask: u32,
    pub join_right_capacity: u32,
    pub schema_winner_count: u32,
    pub receipt_count: u32,
    pub receipt_byte_count: u32,
    pub generation_metadata_count: u32,
    pub abi_version: u32,
    pub reserved: u32,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentScheduleHeader {}

/// One flattened filter comparison consumed by the compact schedule.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentFilterComparisonDescriptor {
    pub left_kind: u32,
    pub left_column: u32,
    pub right_kind: u32,
    pub right_column: u32,
    pub op: u32,
    pub width: u32,
    pub reserved_zero: u32,
    pub reserved_one: u32,
    pub left_constant: u64,
    pub right_constant: u64,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentFilterComparisonDescriptor {}

impl ResidentFilterComparisonDescriptor {
    pub fn column_constant(column: u32, op: u32, width: u32, constant: u64) -> Self {
        Self {
            left_kind: 0,
            left_column: column,
            right_kind: 1,
            right_column: 0,
            op,
            width,
            reserved_zero: 0,
            reserved_one: 0,
            left_constant: 0,
            right_constant: constant,
        }
    }
}

/// One fixed-width projection expression consumed by the compact schedule.
#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ResidentProjectExpressionDescriptor {
    pub kind: u32,
    pub column: u32,
    pub width: u32,
    pub reserved: u32,
    pub constant: u64,
}

// SAFETY: stable C layout, no references, and every field accepts all bit patterns.
unsafe impl DeviceRepr for ResidentProjectExpressionDescriptor {}

impl ResidentProjectExpressionDescriptor {
    pub fn column(column: u32, width: u32) -> Self {
        Self {
            kind: 0,
            column,
            width,
            reserved: 0,
            constant: 0,
        }
    }

    pub fn constant(width: u32, constant: u64) -> Self {
        Self {
            kind: 1,
            column: 0,
            width,
            reserved: 0,
            constant,
        }
    }
}

/// Exclusive graph-lifetime binding for one stable relation slot.
pub enum ResidentScheduleRelation<'a> {
    Source {
        buffer: &'a CudaBuffer,
        generation: u32,
        initial_count: u32,
    },
    Output {
        buffer: &'a mut CudaBuffer,
        generation: u32,
    },
}

/// Transient slot ownership used while materializing graph-free schedule metadata.
pub enum ResidentScheduleSlotBinding<'a> {
    Source {
        buffer: &'a CudaBuffer,
        generation: u32,
        initial_count: u32,
    },
    Resident {
        buffer: &'a CudaBuffer,
        generation: u32,
        permanent: bool,
    },
}

impl<'a> ResidentScheduleSlotBinding<'a> {
    pub fn source(buffer: &'a CudaBuffer, generation: u32) -> Result<Self> {
        let initial_count = buffer.cached_row_count().ok_or_else(|| {
            XlogError::Kernel(
                "resident schedule source requires a cold-path cached logical row count".into(),
            )
        })?;
        if u64::from(initial_count) > buffer.num_rows() {
            return Err(XlogError::Kernel(
                "resident schedule source count exceeds capacity".into(),
            ));
        }
        Ok(Self::Source {
            buffer,
            generation,
            initial_count,
        })
    }

    pub fn scratch(buffer: &'a CudaBuffer, generation: u32) -> Self {
        Self::Resident {
            buffer,
            generation,
            permanent: false,
        }
    }

    pub fn permanent(buffer: &'a CudaBuffer, generation: u32) -> Self {
        Self::Resident {
            buffer,
            generation,
            permanent: true,
        }
    }

    fn buffer(&self) -> &CudaBuffer {
        match self {
            Self::Source { buffer, .. } | Self::Resident { buffer, .. } => buffer,
        }
    }

    fn generation(&self) -> u32 {
        match self {
            Self::Source { generation, .. } | Self::Resident { generation, .. } => *generation,
        }
    }

    fn flags(&self) -> u32 {
        match self {
            Self::Source { .. } => RESIDENT_SCHEDULE_SLOT_SOURCE | RESIDENT_SCHEDULE_SLOT_DEFINED,
            Self::Resident {
                permanent: true, ..
            } => RESIDENT_SCHEDULE_SLOT_PERMANENT | RESIDENT_SCHEDULE_SLOT_DEFINED,
            Self::Resident {
                permanent: false, ..
            } => 0,
        }
    }

    fn initial_count(&self) -> u32 {
        match self {
            Self::Source { initial_count, .. } => *initial_count,
            Self::Resident { .. } => 0,
        }
    }

    /// Record the externally owned relation with its exact scheduler access.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        match self {
            Self::Source { buffer, .. } => {
                for column in buffer.columns() {
                    recorder.read_column(column);
                }
                recorder.read(buffer.num_rows_device());
            }
            Self::Resident { buffer, .. } => {
                for column in buffer.columns() {
                    recorder.read_column(column);
                    recorder.write_column(column);
                }
                recorder.read_write(buffer.num_rows_device());
            }
        }
    }
}

/// Runtime-owned control/workspace bindings copied into a graph-free header.
pub struct ResidentScheduleExternalBindings<'a> {
    filter_scratch: Option<&'a ResidentFilterScratch>,
    set_workspace: &'a ResidentSetWorkspace,
    join_workspace: &'a ResidentJoinWorkspace,
    control: &'a ResidentConvergenceControl,
    trace: &'a ResidentDeviceTrace,
    schema_winners: &'a ResidentSchemaWinners,
    receipt: &'a ResidentPackedReceipt,
}

impl<'a> ResidentScheduleExternalBindings<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        filter_scratch: Option<&'a ResidentFilterScratch>,
        set_workspace: &'a ResidentSetWorkspace,
        join_workspace: &'a ResidentJoinWorkspace,
        control: &'a ResidentConvergenceControl,
        trace: &'a ResidentDeviceTrace,
        schema_winners: &'a ResidentSchemaWinners,
        receipt: &'a ResidentPackedReceipt,
    ) -> Self {
        Self {
            filter_scratch,
            set_workspace,
            join_workspace,
            control,
            trace,
            schema_winners,
            receipt,
        }
    }

    /// Record every runtime-owned mutable workspace and final receipt binding.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        if let Some(filter_scratch) = self.filter_scratch {
            filter_scratch.record_uses(recorder);
        }
        self.set_workspace.record_uses(recorder);
        self.join_workspace.record_uses(recorder);
        self.control.record_uses(recorder);
        self.trace.record_uses(recorder);
        self.schema_winners.record_uses(recorder);
        self.receipt.record_uses(recorder);
    }
}

impl<'a> ResidentScheduleRelation<'a> {
    pub fn source(buffer: &'a CudaBuffer, generation: u32) -> Result<Self> {
        let initial_count = buffer.cached_row_count().ok_or_else(|| {
            XlogError::Kernel(
                "resident schedule source requires a cold-path cached logical row count".into(),
            )
        })?;
        if u64::from(initial_count) > buffer.num_rows() {
            return Err(XlogError::Kernel(format!(
                "resident schedule source count {initial_count} exceeds capacity {}",
                buffer.num_rows()
            )));
        }
        Ok(Self::Source {
            buffer,
            generation,
            initial_count,
        })
    }

    pub fn output(buffer: &'a mut CudaBuffer, generation: u32) -> Self {
        Self::Output { buffer, generation }
    }

    fn buffer(&self) -> &CudaBuffer {
        match self {
            Self::Source { buffer, .. } => buffer,
            Self::Output { buffer, .. } => buffer,
        }
    }

    fn generation(&self) -> u32 {
        match self {
            Self::Source { generation, .. } | Self::Output { generation, .. } => *generation,
        }
    }

    fn flags(&self) -> u32 {
        match self {
            Self::Source { .. } => SOURCE_SLOT,
            Self::Output { .. } => 0,
        }
    }

    fn initial_count(&self) -> u32 {
        match self {
            Self::Source { initial_count, .. } => *initial_count,
            Self::Output { .. } => 0,
        }
    }

    fn invalidate_output_metadata(&mut self) {
        if let Self::Output { buffer, .. } = self {
            let _ = buffer.num_rows_device_mut();
        }
    }

    fn is_output(&self) -> bool {
        matches!(self, Self::Output { .. })
    }
}

/// One post-synchronization observation of terminal status and row counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentScheduleReceipt {
    pub status: ResidentTerminalStatus,
    pub changed: u32,
    pub counts: Vec<u32>,
}

struct ResidentSchedulePinnedReceipt {
    ptr: NonNull<u8>,
    len: usize,
}

impl ResidentSchedulePinnedReceipt {
    fn allocate(len: usize) -> Result<Self> {
        let mut ptr = std::ptr::null_mut();
        // SAFETY: CUDA initializes `ptr` on success and this owner frees it once.
        let code = unsafe { sys::cuMemHostAlloc(&mut ptr, len, 0) };
        if code != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(XlogError::Kernel(format!(
                "resident schedule pinned receipt allocation failed: {code:?}"
            )));
        }
        let ptr = NonNull::new(ptr.cast()).ok_or_else(|| {
            XlogError::Kernel("resident schedule pinned receipt allocation returned null".into())
        })?;
        Ok(Self { ptr, len })
    }

    fn copy_from_device(&mut self, device_ptr: u64, stream: &CudaStream) -> Result<Vec<u8>> {
        // SAFETY: both owners remain live for `self.len` bytes until the stream
        // wait completes, and `&mut self` excludes concurrent host access.
        let code = unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                self.ptr.as_ptr().cast(),
                device_ptr,
                self.len,
                stream.cu_stream(),
            )
        };
        if code != sys::cudaError_enum::CUDA_SUCCESS {
            return Err(XlogError::Kernel(format!(
                "resident schedule final receipt copy failed: {code:?}"
            )));
        }
        stream.synchronize().map_err(|error| {
            XlogError::Kernel(format!(
                "resident schedule final receipt wait failed: {error}"
            ))
        })?;
        // SAFETY: the complete asynchronous copy and stream wait succeeded.
        Ok(unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }.to_vec())
    }
}

impl Drop for ResidentSchedulePinnedReceipt {
    fn drop(&mut self) {
        // SAFETY: this pointer was returned by `cuMemHostAlloc` and is freed once.
        let _ = unsafe { sys::cuMemFreeHost(self.ptr.as_ptr().cast()) };
    }
}

/// All allocations whose addresses are captured by one compact schedule graph.
pub struct ResidentSchedule<'a> {
    origin_provider_identity: u64,
    origin_memory_manager: usize,
    header: TrackedCudaSlice<ResidentScheduleHeader>,
    _slots: TrackedCudaSlice<ResidentRelationSlot>,
    _ops: TrackedCudaSlice<ResidentOpDescriptor>,
    _waves: TrackedCudaSlice<ResidentWaveDescriptor>,
    _regions: TrackedCudaSlice<ResidentRegionDescriptor>,
    _generation_metadata: TrackedCudaSlice<u32>,
    _filter_comparisons: TrackedCudaSlice<ResidentFilterComparisonDescriptor>,
    _project_expressions: TrackedCudaSlice<ResidentProjectExpressionDescriptor>,
    _filter_mask: TrackedCudaSlice<u32>,
    _filter_prefix: TrackedCudaSlice<u32>,
    _filter_block_sums: TrackedCudaSlice<u32>,
    _filter_block_offsets: TrackedCudaSlice<u32>,
    _set_slots: TrackedCudaSlice<u64>,
    _set_required: TrackedCudaSlice<u64>,
    _join_buckets: TrackedCudaSlice<u32>,
    _join_next: TrackedCudaSlice<u32>,
    _join_required: TrackedCudaSlice<u64>,
    _status: TrackedCudaSlice<ResidentTerminalStatus>,
    _changed: TrackedCudaSlice<u32>,
    _iterations: TrackedCudaSlice<u32>,
    _scan_trace: TrackedCudaSlice<u32>,
    _filter_trace: TrackedCudaSlice<u32>,
    _semantic_scan_trace: TrackedCudaSlice<u32>,
    _semantic_filter_trace: TrackedCudaSlice<u32>,
    _receipt_table: TrackedCudaSlice<u64>,
    receipt_bytes: TrackedCudaSlice<u8>,
    pinned_receipt: ResidentSchedulePinnedReceipt,
    launch_config: LaunchConfig,
    region_count: u32,
    region_descriptors: Vec<ResidentRegionDescriptor>,
    requested_receipt_count: usize,
    receipt_slots: Vec<u32>,
    relations: Vec<ResidentScheduleRelation<'a>>,
}

/// Graph-free compact scheduler metadata owned by the enclosing runtime capsule.
#[derive(Clone)]
pub struct ResidentExecutionDomain {
    provider_identity: u64,
    memory_manager_identity: usize,
    runtime: Arc<XlogDeviceRuntime>,
    stream_id: StreamId,
    stream: Arc<CudaStream>,
    context: Arc<cudarc::driver::CudaContext>,
    marker: Arc<()>,
}

impl ResidentExecutionDomain {
    pub fn new_strict_recorder(&self) -> LaunchRecorder {
        LaunchRecorder::new_strict_bound(
            self.stream_id,
            Arc::clone(&self.runtime),
            Arc::clone(&self.marker),
        )
    }

    pub fn preflight(&self, recorder: &mut LaunchRecorder) -> Result<()> {
        recorder.require_bound_domain(&self.runtime, &self.marker, self.stream_id);
        recorder
            .preflight_bound(&self.runtime)
            .map_err(|error| XlogError::Kernel(format!("resident launch preflight: {error}")))
    }

    pub fn commit(&self, recorder: LaunchRecorder) -> Result<()> {
        recorder
            .commit_bound(&self.runtime, &self.marker)
            .map_err(|error| XlogError::Kernel(format!("resident launch commit: {error}")))
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    pub fn stream_id(&self) -> StreamId {
        self.stream_id
    }
}

/// Graph-free compact scheduler metadata owned by the enclosing runtime capsule.
pub struct ResidentScheduleDeviceProgram {
    origin_provider_identity: u64,
    domain: ResidentExecutionDomain,
    header: TrackedCudaSlice<ResidentScheduleHeader>,
    _slots: TrackedCudaSlice<ResidentRelationSlot>,
    _ops: TrackedCudaSlice<ResidentOpDescriptor>,
    _waves: TrackedCudaSlice<ResidentWaveDescriptor>,
    _regions: TrackedCudaSlice<ResidentRegionDescriptor>,
    _generation_metadata: TrackedCudaSlice<u32>,
    _filter_comparisons: TrackedCudaSlice<ResidentFilterComparisonDescriptor>,
    _project_expressions: TrackedCudaSlice<ResidentProjectExpressionDescriptor>,
    launch_config: LaunchConfig,
    region_descriptors: Vec<ResidentRegionDescriptor>,
}

impl ResidentScheduleDeviceProgram {
    /// Record immutable program tables and the replay-mutated slot table.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.require_bound_domain(
            &self.domain.runtime,
            &self.domain.marker,
            self.domain.stream_id,
        );
        recorder.read(&self.header);
        recorder.read_write(&self._slots);
        recorder.read(&self._ops);
        recorder.read(&self._waves);
        recorder.read(&self._regions);
        recorder.read(&self._generation_metadata);
        recorder.read(&self._filter_comparisons);
        recorder.read(&self._project_expressions);
    }
}

/// One captured schedule plus every allocation and exclusive relation lease.
pub struct ResidentScheduleGraph<'a> {
    graph: CapturedCudaGraph,
    schedule: ResidentSchedule<'a>,
    provider: &'a CudaKernelProvider,
    stream: Arc<CudaStream>,
    in_flight: bool,
}

impl ResidentScheduleGraph<'_> {
    pub fn node_count(&self) -> Result<usize> {
        self.graph.node_count()
    }

    pub fn nodes(&self) -> Result<Vec<CudaGraphNode>> {
        self.graph.nodes()
    }

    pub fn launch(&mut self) -> Result<()> {
        if self.in_flight {
            return Err(XlogError::Kernel(
                "resident schedule launch is already in flight".into(),
            ));
        }
        self.graph.launch(&self.stream)?;
        self.in_flight = true;
        Ok(())
    }

    pub fn synchronize_and_observe(&mut self) -> Result<ResidentScheduleReceipt> {
        if !self.in_flight {
            return Err(XlogError::Kernel(
                "resident schedule has no in-flight launch to observe".into(),
            ));
        }
        self.stream
            .synchronize()
            .map_err(|error| XlogError::Kernel(format!("resident schedule sync: {error}")))?;
        let receipt = self
            .provider
            .observe_resident_schedule(&mut self.schedule, &self.stream)?;
        self.in_flight = false;
        Ok(receipt)
    }

    pub fn relation(&self, slot: usize) -> Result<&CudaBuffer> {
        if self.in_flight {
            return Err(XlogError::Kernel(
                "resident schedule relation is unavailable while launch is in flight".into(),
            ));
        }
        self.schedule
            .relations
            .get(slot)
            .map(ResidentScheduleRelation::buffer)
            .ok_or_else(|| XlogError::Kernel(format!("resident schedule slot {slot} is invalid")))
    }
}

impl Drop for ResidentScheduleGraph<'_> {
    fn drop(&mut self) {
        if self.in_flight {
            let _ = self.stream.synchronize();
        }
    }
}

const _: () = {
    use std::mem::{align_of, offset_of, size_of};

    assert!(size_of::<ResidentRelationView>() == 224);
    assert!(align_of::<ResidentRelationView>() == 8);
    assert!(offset_of!(ResidentRelationView, columns) == 0);
    assert!(offset_of!(ResidentRelationView, widths) == 136);
    assert!(offset_of!(ResidentRelationView, arity) == 204);
    assert!(offset_of!(ResidentRelationView, capacity) == 208);
    assert!(offset_of!(ResidentRelationView, reserved) == 212);
    assert!(offset_of!(ResidentRelationView, num_rows) == 216);

    assert!(size_of::<ResidentRelationSlot>() == 240);
    assert!(align_of::<ResidentRelationSlot>() == 16);
    assert!(offset_of!(ResidentRelationSlot, relation) == 0);
    assert!(offset_of!(ResidentRelationSlot, generation) == 224);
    assert!(offset_of!(ResidentRelationSlot, flags) == 228);
    assert!(offset_of!(ResidentRelationSlot, initial_count) == 232);
    assert!(offset_of!(ResidentRelationSlot, schema_tag) == 236);

    assert!(size_of::<ResidentOpDescriptor>() == 72);
    assert!(align_of::<ResidentOpDescriptor>() == 4);
    assert!(offset_of!(ResidentOpDescriptor, kind) == 0);
    assert!(offset_of!(ResidentOpDescriptor, flags) == 4);
    assert!(offset_of!(ResidentOpDescriptor, op_id) == 8);
    assert!(offset_of!(ResidentOpDescriptor, out) == 12);
    assert!(offset_of!(ResidentOpDescriptor, in0) == 16);
    assert!(offset_of!(ResidentOpDescriptor, in1) == 20);
    assert!(offset_of!(ResidentOpDescriptor, in0_generation) == 24);
    assert!(offset_of!(ResidentOpDescriptor, in1_generation) == 28);
    assert!(offset_of!(ResidentOpDescriptor, out_generation) == 32);
    assert!(offset_of!(ResidentOpDescriptor, aux_offset) == 36);
    assert!(offset_of!(ResidentOpDescriptor, aux_count) == 40);
    assert!(offset_of!(ResidentOpDescriptor, left_key) == 44);
    assert!(offset_of!(ResidentOpDescriptor, right_key) == 48);
    assert!(offset_of!(ResidentOpDescriptor, scan_delta) == 52);
    assert!(offset_of!(ResidentOpDescriptor, filter_delta) == 56);
    assert!(offset_of!(ResidentOpDescriptor, schema_winner_head) == 60);
    assert!(offset_of!(ResidentOpDescriptor, schema_winner_id) == 64);
    assert!(offset_of!(ResidentOpDescriptor, reserved) == 68);

    assert!(size_of::<ResidentWaveDescriptor>() == 16);
    assert!(align_of::<ResidentWaveDescriptor>() == 4);
    assert!(offset_of!(ResidentWaveDescriptor, first_op) == 0);
    assert!(offset_of!(ResidentWaveDescriptor, op_count) == 4);
    assert!(offset_of!(ResidentWaveDescriptor, flags) == 8);
    assert!(offset_of!(ResidentWaveDescriptor, reserved) == 12);

    assert!(size_of::<ResidentRegionDescriptor>() == 32);
    assert!(align_of::<ResidentRegionDescriptor>() == 4);
    assert!(offset_of!(ResidentRegionDescriptor, first_wave) == 0);
    assert!(offset_of!(ResidentRegionDescriptor, wave_count) == 4);
    assert!(offset_of!(ResidentRegionDescriptor, iteration_limit) == 8);
    assert!(offset_of!(ResidentRegionDescriptor, op_id) == 12);
    assert!(offset_of!(ResidentRegionDescriptor, flags) == 16);
    assert!(offset_of!(ResidentRegionDescriptor, first_slot) == 20);
    assert!(offset_of!(ResidentRegionDescriptor, slot_count) == 24);
    assert!(offset_of!(ResidentRegionDescriptor, generation_offset) == 28);

    assert!(size_of::<ResidentScheduleHeader>() == 288);
    assert!(align_of::<ResidentScheduleHeader>() == 16);
    assert!(offset_of!(ResidentScheduleHeader, slots) == 0);
    assert!(offset_of!(ResidentScheduleHeader, ops) == 8);
    assert!(offset_of!(ResidentScheduleHeader, waves) == 16);
    assert!(offset_of!(ResidentScheduleHeader, regions) == 24);
    assert!(offset_of!(ResidentScheduleHeader, generation_metadata) == 32);
    assert!(offset_of!(ResidentScheduleHeader, filter_comparisons) == 40);
    assert!(offset_of!(ResidentScheduleHeader, project_expressions) == 48);
    assert!(offset_of!(ResidentScheduleHeader, filter_mask) == 56);
    assert!(offset_of!(ResidentScheduleHeader, filter_prefix) == 64);
    assert!(offset_of!(ResidentScheduleHeader, filter_block_sums) == 72);
    assert!(offset_of!(ResidentScheduleHeader, filter_block_offsets) == 80);
    assert!(offset_of!(ResidentScheduleHeader, set_slots) == 88);
    assert!(offset_of!(ResidentScheduleHeader, set_required) == 96);
    assert!(offset_of!(ResidentScheduleHeader, join_buckets) == 104);
    assert!(offset_of!(ResidentScheduleHeader, join_next) == 112);
    assert!(offset_of!(ResidentScheduleHeader, join_required) == 120);
    assert!(offset_of!(ResidentScheduleHeader, status) == 128);
    assert!(offset_of!(ResidentScheduleHeader, changed) == 136);
    assert!(offset_of!(ResidentScheduleHeader, iterations) == 144);
    assert!(offset_of!(ResidentScheduleHeader, scan_trace) == 152);
    assert!(offset_of!(ResidentScheduleHeader, filter_trace) == 160);
    assert!(offset_of!(ResidentScheduleHeader, semantic_scan_trace) == 168);
    assert!(offset_of!(ResidentScheduleHeader, semantic_filter_trace) == 176);
    assert!(offset_of!(ResidentScheduleHeader, schema_seen_nonempty) == 184);
    assert!(offset_of!(ResidentScheduleHeader, schema_winner_ids) == 192);
    assert!(offset_of!(ResidentScheduleHeader, receipt_table) == 200);
    assert!(offset_of!(ResidentScheduleHeader, receipt_bytes) == 208);
    assert!(offset_of!(ResidentScheduleHeader, slot_count) == 216);
    assert!(offset_of!(ResidentScheduleHeader, schema_winner_count) == 264);
    assert!(offset_of!(ResidentScheduleHeader, generation_metadata_count) == 276);
    assert!(offset_of!(ResidentScheduleHeader, abi_version) == 280);
    assert!(offset_of!(ResidentScheduleHeader, reserved) == 284);

    assert!(size_of::<ResidentFilterComparisonDescriptor>() == 48);
    assert!(align_of::<ResidentFilterComparisonDescriptor>() == 8);
    assert!(offset_of!(ResidentFilterComparisonDescriptor, left_constant) == 32);
    assert!(offset_of!(ResidentFilterComparisonDescriptor, right_constant) == 40);

    assert!(size_of::<ResidentProjectExpressionDescriptor>() == 24);
    assert!(align_of::<ResidentProjectExpressionDescriptor>() == 8);
    assert!(offset_of!(ResidentProjectExpressionDescriptor, constant) == 16);
};

pub fn resident_schedule_metadata_device_bytes(
    slot_count: usize,
    op_count: usize,
    wave_count: usize,
    region_count: usize,
    generation_metadata_count: usize,
    filter_comparison_count: usize,
    project_expression_count: usize,
) -> Result<u64> {
    let tables = [
        (slot_count.max(1), size_of::<ResidentRelationSlot>()),
        (op_count.max(1), size_of::<ResidentOpDescriptor>()),
        (wave_count.max(1), size_of::<ResidentWaveDescriptor>()),
        (region_count.max(1), size_of::<ResidentRegionDescriptor>()),
        (generation_metadata_count.max(1), size_of::<u32>()),
        (
            filter_comparison_count.max(1),
            size_of::<ResidentFilterComparisonDescriptor>(),
        ),
        (
            project_expression_count.max(1),
            size_of::<ResidentProjectExpressionDescriptor>(),
        ),
    ];
    let total = tables.iter().try_fold(
        size_of::<ResidentScheduleHeader>() as u128,
        |sum, &(count, width)| sum.checked_add((count as u128) * (width as u128)),
    );
    total
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| XlogError::Kernel("resident schedule metadata bytes overflow".into()))
}

fn checked_u32(value: usize, context: &str) -> Result<u32> {
    u32::try_from(value)
        .map_err(|_| XlogError::Kernel(format!("resident schedule {context} exceeds u32")))
}

fn checked_capacity(value: u64, context: &str) -> Result<u32> {
    if value > RESIDENT_SCHEDULE_MAX_ROWS {
        return Err(XlogError::Kernel(format!(
            "resident schedule {context} capacity {value} exceeds {RESIDENT_SCHEDULE_MAX_ROWS}"
        )));
    }
    u32::try_from(value)
        .map_err(|_| XlogError::Kernel(format!("resident schedule {context} exceeds u32")))
}

fn checked_workspace_slots(candidate_capacity: u64, context: &str) -> Result<u32> {
    let doubled = candidate_capacity
        .max(1)
        .checked_mul(2)
        .ok_or_else(|| XlogError::Kernel(format!("resident schedule {context} overflow")))?;
    let slots = doubled
        .checked_next_power_of_two()
        .ok_or_else(|| XlogError::Kernel(format!("resident schedule {context} overflow")))?;
    u32::try_from(slots)
        .map_err(|_| XlogError::Kernel(format!("resident schedule {context} exceeds u32")))
}

fn reset_slot_flags(flags: u32) -> u32 {
    if flags & (RESIDENT_SCHEDULE_SLOT_SOURCE | RESIDENT_SCHEDULE_SLOT_PERMANENT) != 0 {
        flags | RESIDENT_SCHEDULE_SLOT_DEFINED
    } else {
        flags & !RESIDENT_SCHEDULE_SLOT_DEFINED
    }
}

fn reset_slot_state_for_region(flags: u32, generation: u32, count: u32) -> (u32, u32, u32) {
    let fixed = flags & (RESIDENT_SCHEDULE_SLOT_SOURCE | RESIDENT_SCHEDULE_SLOT_PERMANENT) != 0;
    (
        reset_slot_flags(flags),
        generation,
        if fixed { count } else { 0 },
    )
}

fn slot_input_is_ready(flags: u32, generation: u32, expected_generation: u32) -> bool {
    flags & RESIDENT_SCHEDULE_SLOT_DEFINED != 0 && generation == expected_generation
}

fn slot_output_generation_is_valid(flags: u32, generation: u32, output_generation: u32) -> bool {
    flags & RESIDENT_SCHEDULE_SLOT_SOURCE == 0
        && (output_generation == generation || generation.checked_add(1) == Some(output_generation))
}

fn finish_slot_write(flags: u32, success: bool) -> u32 {
    if success {
        flags | RESIDENT_SCHEDULE_SLOT_DEFINED
    } else {
        flags
    }
}

fn checked_schedule_head_count(receipt_count: u32, receipt_byte_count: u32) -> Result<u32> {
    let remainder = receipt_count.checked_sub(4).filter(|value| value % 2 == 0);
    let expected_bytes = receipt_count
        .checked_add(1)
        .and_then(|count| count.checked_mul(size_of::<u32>() as u32))
        .and_then(|count_bytes| {
            (size_of::<ResidentTerminalStatus>() as u32).checked_add(count_bytes)
        });
    if remainder.is_none() || expected_bytes != Some(receipt_byte_count) {
        return Err(XlogError::Kernel(
            "resident schedule packed receipt shape is invalid".into(),
        ));
    }
    Ok(remainder.expect("validated receipt remainder") / 2)
}

fn checked_schedule_winner_count(
    receipt_count: u32,
    receipt_byte_count: u32,
    schema_winner_count: u32,
) -> Result<u32> {
    let head_count = checked_schedule_head_count(receipt_count, receipt_byte_count)?;
    if schema_winner_count != head_count {
        return Err(XlogError::Kernel(
            "resident schedule schema winner count does not match its receipt".into(),
        ));
    }
    Ok(head_count)
}

fn validate_schema_winner_encoding(op: &ResidentOpDescriptor, head_count: u32) -> Result<()> {
    let marks_winner = op.flags & RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER != 0;
    if (marks_winner && op.schema_winner_head >= head_count)
        || (!marks_winner && (op.schema_winner_head != 0 || op.schema_winner_id != 0))
    {
        return Err(XlogError::Kernel(format!(
            "resident schedule operation {} has an invalid schema winner encoding",
            op.op_id
        )));
    }
    Ok(())
}

fn same_relation_layout(left: &ResidentRelationView, right: &ResidentRelationView) -> bool {
    left.arity == right.arity
        && left.widths[..left.arity as usize] == right.widths[..right.arity as usize]
}

fn column_width_matches(relation: &ResidentRelationView, column: u32, width: u32) -> bool {
    let column = column as usize;
    column < relation.arity as usize && relation.widths[column] == width
}

fn validate_flattened_filter_project_descriptors(
    slots: &[ResidentRelationSlot],
    slot_types: &[Vec<ScalarType>],
    ops: &[ResidentOpDescriptor],
    filter_comparisons: &[ResidentFilterComparisonDescriptor],
    project_expressions: &[ResidentProjectExpressionDescriptor],
) -> Result<()> {
    if slots.len() != slot_types.len() {
        return Err(XlogError::Kernel(
            "resident schedule slot type table length is invalid".into(),
        ));
    }
    for (slot, types) in slots.iter().zip(slot_types) {
        if types.len() != slot.relation.arity as usize {
            return Err(XlogError::Kernel(
                "resident schedule slot type table arity is invalid".into(),
            ));
        }
        for (column, scalar) in types.iter().copied().enumerate() {
            if resident_schedule_scalar_width(scalar)? != slot.relation.widths[column] {
                return Err(XlogError::Kernel(
                    "resident schedule slot type width is invalid".into(),
                ));
            }
        }
    }

    let operand_type = |relation: &ResidentRelationView,
                        types: &[ScalarType],
                        kind: u32,
                        column: u32,
                        width: u32,
                        constant: u64|
     -> Result<Option<ScalarType>> {
        if !matches!(width, 4 | 8) {
            return Err(XlogError::Kernel(
                "resident schedule descriptor scalar width is invalid".into(),
            ));
        }
        match kind {
            0 => {
                if constant != 0 || !column_width_matches(relation, column, width) {
                    return Err(XlogError::Kernel(
                        "resident schedule descriptor column is invalid".into(),
                    ));
                }
                Ok(types.get(column as usize).copied())
            }
            1 if column == 0 => Ok(None),
            _ => Err(XlogError::Kernel(
                "resident schedule descriptor operand kind is invalid".into(),
            )),
        }
    };

    let mut filter_cursor = 0_u32;
    let mut project_cursor = 0_u32;
    let filter_total = checked_u32(filter_comparisons.len(), "filter comparison count")?;
    let project_total = checked_u32(project_expressions.len(), "project expression count")?;
    for op in ops {
        if op.kind == OP_FILTER {
            if op.in0 as usize >= slots.len()
                || op.out as usize >= slots.len()
                || op.aux_offset != filter_cursor
                || op.aux_offset > filter_total
                || op.aux_count > filter_total - op.aux_offset
            {
                return Err(XlogError::Kernel(format!(
                    "resident schedule filter {} descriptor range is invalid",
                    op.op_id
                )));
            }
            let input = &slots[op.in0 as usize].relation;
            let output = &slots[op.out as usize].relation;
            if slots[op.in0 as usize].schema_tag != slots[op.out as usize].schema_tag
                || !same_relation_layout(input, output)
            {
                return Err(XlogError::Kernel(format!(
                    "resident schedule filter {} input and output schemas differ",
                    op.op_id
                )));
            }
            let types = &slot_types[op.in0 as usize];
            for comparison in
                &filter_comparisons[op.aux_offset as usize..(op.aux_offset + op.aux_count) as usize]
            {
                if comparison.op > 5
                    || comparison.reserved_zero != 0
                    || comparison.reserved_one != 0
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule filter {} descriptor payload is invalid",
                        op.op_id
                    )));
                }
                let left = operand_type(
                    input,
                    types,
                    comparison.left_kind,
                    comparison.left_column,
                    comparison.width,
                    comparison.left_constant,
                )?;
                let right = operand_type(
                    input,
                    types,
                    comparison.right_kind,
                    comparison.right_column,
                    comparison.width,
                    comparison.right_constant,
                )?;
                if left.is_some() && right.is_some() && left != right {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule filter {} operand types differ",
                        op.op_id
                    )));
                }
            }
            filter_cursor = op.aux_offset + op.aux_count;
        } else if op.kind == OP_PROJECT {
            if op.in0 as usize >= slots.len()
                || op.out as usize >= slots.len()
                || op.aux_offset != project_cursor
                || op.aux_offset > project_total
                || op.aux_count > project_total - op.aux_offset
            {
                return Err(XlogError::Kernel(format!(
                    "resident schedule project {} descriptor range is invalid",
                    op.op_id
                )));
            }
            let input = &slots[op.in0 as usize].relation;
            let output = &slots[op.out as usize].relation;
            if op.aux_count != output.arity {
                return Err(XlogError::Kernel(format!(
                    "resident schedule project {} expression count is invalid",
                    op.op_id
                )));
            }
            let input_types = &slot_types[op.in0 as usize];
            let output_types = &slot_types[op.out as usize];
            for (column, expression) in project_expressions
                [op.aux_offset as usize..(op.aux_offset + op.aux_count) as usize]
                .iter()
                .enumerate()
            {
                if expression.reserved != 0
                    || expression.width != output.widths[column]
                    || !matches!(expression.width, 4 | 8)
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule project {} descriptor payload is invalid",
                        op.op_id
                    )));
                }
                match expression.kind {
                    0 => {
                        if expression.constant != 0
                            || !column_width_matches(input, expression.column, expression.width)
                            || input_types.get(expression.column as usize)
                                != output_types.get(column)
                        {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule project {} column type is invalid",
                                op.op_id
                            )));
                        }
                    }
                    1 if expression.column == 0 => {}
                    _ => {
                        return Err(XlogError::Kernel(format!(
                            "resident schedule project {} expression kind is invalid",
                            op.op_id
                        )));
                    }
                }
            }
            project_cursor = op.aux_offset + op.aux_count;
        }
    }
    if filter_cursor != filter_total || project_cursor != project_total {
        return Err(XlogError::Kernel(
            "resident schedule flattened descriptor tables are not exactly covered".into(),
        ));
    }
    Ok(())
}

fn resident_schedule_scalar_width(scalar: ScalarType) -> Result<u32> {
    match scalar {
        ScalarType::Symbol | ScalarType::U32 => Ok(4),
        ScalarType::U64 => Ok(8),
        unsupported => Err(XlogError::Kernel(format!(
            "resident schedule scalar type {unsupported:?} is unsupported"
        ))),
    }
}

fn resident_schedule_scalar_tag(scalar: ScalarType) -> u32 {
    match scalar {
        ScalarType::Symbol => 1,
        ScalarType::U32 => 2,
        ScalarType::U64 => 3,
        _ => 0,
    }
}

fn validate_initialization_scope(
    initial_region: &ResidentRegionDescriptor,
    slot_count: u32,
) -> Result<()> {
    if initial_region.first_slot != 0 || initial_region.slot_count != slot_count {
        return Err(XlogError::Kernel(
            "resident schedule initialization must cover every relation slot".into(),
        ));
    }
    Ok(())
}

fn validate_wave_partition(waves: &[ResidentWaveDescriptor], op_count: u32) -> Result<()> {
    let mut next_op = 0_u32;
    for wave in waves {
        if wave.first_op != next_op
            || wave.first_op > op_count
            || wave.op_count > op_count - wave.first_op
            || wave.flags != 0
            || wave.reserved != 0
        {
            return Err(XlogError::Kernel(
                "resident schedule waves must exactly partition operations".into(),
            ));
        }
        next_op = wave.first_op + wave.op_count;
    }
    if next_op != op_count {
        return Err(XlogError::Kernel(
            "resident schedule waves must exactly partition operations".into(),
        ));
    }
    Ok(())
}

fn validate_region_control_and_ranges(
    regions: &[ResidentRegionDescriptor],
    wave_count: u32,
    slot_count: u32,
) -> Result<()> {
    if regions.is_empty() {
        return Err(XlogError::Kernel(
            "resident schedule requires at least one region".into(),
        ));
    }
    validate_initialization_scope(&regions[0], slot_count)?;
    let allowed_flags = RESIDENT_SCHEDULE_REGION_INITIALIZE
        | RESIDENT_SCHEDULE_REGION_SCC_BEGIN
        | RESIDENT_SCHEDULE_REGION_RECURSIVE
        | RESIDENT_SCHEDULE_REGION_FINALIZE;
    let mut next_wave = 0_u32;
    for (index, region) in regions.iter().enumerate() {
        if region.first_wave > wave_count
            || region.wave_count > wave_count - region.first_wave
            || region.first_slot > slot_count
            || region.slot_count > slot_count - region.first_slot
            || region.first_wave != next_wave
            || region.flags & !allowed_flags != 0
        {
            return Err(XlogError::Kernel(
                "resident schedule region range or reserved field is invalid".into(),
            ));
        }
        let initializes = region.flags & RESIDENT_SCHEDULE_REGION_INITIALIZE != 0;
        let begins_scc = region.flags & RESIDENT_SCHEDULE_REGION_SCC_BEGIN != 0;
        let recursive = region.flags & RESIDENT_SCHEDULE_REGION_RECURSIVE != 0;
        let finalizes = region.flags & RESIDENT_SCHEDULE_REGION_FINALIZE != 0;
        if initializes != (index == 0)
            || finalizes != (index + 1 == regions.len())
            || (recursive && region.flags != RESIDENT_SCHEDULE_REGION_RECURSIVE)
            || (begins_scc && (recursive || finalizes))
            || (!begins_scc && !recursive && region.iteration_limit != 1)
        {
            return Err(XlogError::Kernel(
                "resident schedule region control flags are invalid".into(),
            ));
        }
        if begins_scc {
            let body = regions.get(index + 1).ok_or_else(|| {
                XlogError::Kernel("resident schedule SCC begin has no recursive body".into())
            })?;
            if body.flags != RESIDENT_SCHEDULE_REGION_RECURSIVE
                || body.iteration_limit != region.iteration_limit
                || body.op_id != region.op_id
            {
                return Err(XlogError::Kernel(
                    "resident schedule SCC begin does not match its recursive body".into(),
                ));
            }
        } else if recursive {
            let seed = index
                .checked_sub(1)
                .and_then(|seed_index| regions.get(seed_index))
                .ok_or_else(|| {
                    XlogError::Kernel("resident schedule recursive body has no SCC begin".into())
                })?;
            if seed.flags & RESIDENT_SCHEDULE_REGION_SCC_BEGIN == 0 {
                return Err(XlogError::Kernel(
                    "resident schedule recursive body has no SCC begin".into(),
                ));
            }
        }
        next_wave = region.first_wave + region.wave_count;
    }
    if next_wave != wave_count {
        return Err(XlogError::Kernel(
            "resident schedule regions do not cover every wave".into(),
        ));
    }
    Ok(())
}

fn validate_generation_baseline_ranges(
    regions: &[ResidentRegionDescriptor],
    generation_base_count: u32,
) -> Result<()> {
    let mut cursor = 0_u32;
    for region in regions {
        if region.generation_offset != cursor {
            return Err(XlogError::Kernel(
                "resident schedule generation baselines are not contiguous".into(),
            ));
        }
        cursor = cursor.checked_add(region.slot_count).ok_or_else(|| {
            XlogError::Kernel("resident schedule generation baseline range overflow".into())
        })?;
        if cursor > generation_base_count {
            return Err(XlogError::Kernel(
                "resident schedule generation baseline range is invalid".into(),
            ));
        }
    }
    if cursor != generation_base_count {
        return Err(XlogError::Kernel(
            "resident schedule generation baseline table has trailing entries".into(),
        ));
    }
    Ok(())
}

fn build_generation_baselines(
    regions: &mut [ResidentRegionDescriptor],
    slot_generations: &[u32],
) -> Result<Vec<u32>> {
    let mut baselines = Vec::new();
    for region in regions {
        if region.generation_offset != 0 {
            return Err(XlogError::Kernel(
                "resident schedule generation baselines are not contiguous".into(),
            ));
        }
        region.generation_offset = checked_u32(baselines.len(), "generation baseline count")?;
        let first = usize::try_from(region.first_slot).unwrap_or(usize::MAX);
        let count = usize::try_from(region.slot_count).unwrap_or(usize::MAX);
        let end = first
            .checked_add(count)
            .filter(|&end| end <= slot_generations.len())
            .ok_or_else(|| {
                XlogError::Kernel(
                    "resident schedule generation baseline slot scope is invalid".into(),
                )
            })?;
        baselines.extend_from_slice(&slot_generations[first..end]);
    }
    checked_u32(baselines.len(), "generation baseline count")?;
    Ok(baselines)
}

fn build_generation_metadata(
    generation_bases: &[u32],
    schema_defaults: &[u32],
) -> Result<Vec<u32>> {
    let metadata_count = generation_bases
        .len()
        .checked_add(schema_defaults.len())
        .ok_or_else(|| {
            XlogError::Kernel("resident schedule generation metadata overflow".into())
        })?;
    let mut metadata = Vec::with_capacity(metadata_count);
    metadata.extend_from_slice(generation_bases);
    metadata.extend_from_slice(schema_defaults);
    Ok(metadata)
}

fn generation_baseline_count_from_metadata(
    generation_metadata_count: u32,
    schema_winner_count: u32,
) -> Result<u32> {
    generation_metadata_count
        .checked_sub(schema_winner_count)
        .ok_or_else(|| {
            XlogError::Kernel(
                "resident schedule generation metadata is shorter than its schema-default tail"
                    .into(),
            )
        })
}

#[cfg(test)]
fn reset_schema_winner_state(
    defaults: &[u32],
    seen_nonempty: &mut [u32],
    winner_ids: &mut [u32],
) -> Result<()> {
    if defaults.len() != seen_nonempty.len() || defaults.len() != winner_ids.len() {
        return Err(XlogError::Kernel(
            "resident schedule schema-winner replay state has the wrong shape".into(),
        ));
    }
    seen_nonempty.fill(0);
    winner_ids.copy_from_slice(defaults);
    Ok(())
}

#[cfg(test)]
fn mark_schema_winner_model(
    contribution_count: u32,
    candidate_id: u32,
    seen_nonempty: &mut u32,
    winner_id: &mut u32,
) {
    if contribution_count != 0 && *seen_nonempty == 0 {
        *seen_nonempty = 1;
        *winner_id = candidate_id;
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResidentScheduleRequirements {
    filter_capacity: u32,
    set_candidate_capacity: u64,
    join_right_capacity: u32,
}

#[allow(clippy::too_many_arguments)]
fn validate_schedule_program(
    slots: &[ResidentRelationSlot],
    slot_types: &[Vec<ScalarType>],
    ops: &[ResidentOpDescriptor],
    waves: &[ResidentWaveDescriptor],
    regions: &[ResidentRegionDescriptor],
    generation_bases: &[u32],
    filter_comparisons: &[ResidentFilterComparisonDescriptor],
    project_expressions: &[ResidentProjectExpressionDescriptor],
    schema_defaults: &[u32],
) -> Result<ResidentScheduleRequirements> {
    validate_flattened_filter_project_descriptors(
        slots,
        slot_types,
        ops,
        filter_comparisons,
        project_expressions,
    )?;
    let op_count = checked_u32(ops.len(), "operation count")?;
    let wave_count = checked_u32(waves.len(), "wave count")?;
    let slot_count = checked_u32(slots.len(), "slot count")?;
    let generation_base_count = checked_u32(generation_bases.len(), "generation baseline count")?;
    validate_wave_partition(waves, op_count)?;
    validate_region_control_and_ranges(regions, wave_count, slot_count)?;
    validate_generation_baseline_ranges(regions, generation_base_count)?;
    let head_count = checked_u32(schema_defaults.len(), "schema default count")?;
    let mut first_schema_candidates = vec![None; schema_defaults.len()];
    let mut requirements = ResidentScheduleRequirements::default();
    let allowed_slot_flags = RESIDENT_SCHEDULE_SLOT_SOURCE
        | RESIDENT_SCHEDULE_SLOT_PERMANENT
        | RESIDENT_SCHEDULE_SLOT_DEFINED;
    for slot in slots {
        if slot.flags & !allowed_slot_flags != 0
            || slot.flags & RESIDENT_SCHEDULE_SLOT_SOURCE != 0
                && slot.flags & RESIDENT_SCHEDULE_SLOT_PERMANENT != 0
        {
            return Err(XlogError::Kernel(
                "resident schedule relation slot flags are invalid".into(),
            ));
        }
    }

    for region in regions {
        let recursive = region.flags == RESIDENT_SCHEDULE_REGION_RECURSIVE;
        let mut novelty_count = 0_usize;
        let mut state = slots
            .iter()
            .map(|slot| (reset_slot_flags(slot.flags), slot.generation))
            .collect::<Vec<_>>();
        for offset in 0..region.slot_count {
            let slot = usize::try_from(region.first_slot + offset).unwrap_or(usize::MAX);
            let baseline = usize::try_from(region.generation_offset + offset).unwrap_or(usize::MAX);
            state[slot] = (
                reset_slot_flags(slots[slot].flags),
                generation_bases[baseline],
            );
        }
        let wave_end = region.first_wave + region.wave_count;
        for wave in &waves[region.first_wave as usize..wave_end as usize] {
            let op_end = wave.first_op + wave.op_count;
            for op in &ops[wave.first_op as usize..op_end as usize] {
                if op.kind == OP_TEST_STATUS {
                    decode_test_status(op)?;
                    continue;
                }
                if op.kind == OP_TRACE_DELTA {
                    let (_, _, semantic_guard) = decode_trace_delta(op)?;
                    if let Some((slot, generation)) = semantic_guard {
                        let slot_index = usize::try_from(slot).unwrap_or(usize::MAX);
                        let slot_end = region.first_slot + region.slot_count;
                        if slot_index >= slots.len() || slot < region.first_slot || slot >= slot_end
                        {
                            return Err(XlogError::Kernel(
                                "resident schedule trace guard is outside its region scope".into(),
                            ));
                        }
                        if !slot_input_is_ready(
                            state[slot_index].0,
                            state[slot_index].1,
                            generation,
                        ) {
                            return Err(XlogError::Kernel(
                                "resident schedule trace guard is undefined or stale".into(),
                            ));
                        }
                    }
                    continue;
                }
                let marks_novelty = op.flags & RESIDENT_SCHEDULE_OP_MARK_NOVELTY != 0;
                let marks_schema_winner = op.flags & RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER != 0;
                if op.kind > OP_DIFF
                    || op.flags
                        & !(RESIDENT_SCHEDULE_OP_MARK_NOVELTY
                            | RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER)
                        != 0
                    || (marks_novelty && (!matches!(op.kind, OP_DIFF | OP_PROJECT) || !recursive))
                    || op.scan_delta != 0
                    || op.filter_delta != 0
                    || op.reserved != 0
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} has an unsupported kind, flag, or payload",
                        op.op_id
                    )));
                }
                validate_schema_winner_encoding(op, head_count)?;
                if marks_novelty {
                    novelty_count += 1;
                }
                if marks_schema_winner {
                    let candidate = &mut first_schema_candidates[op.schema_winner_head as usize];
                    if candidate.is_none() {
                        *candidate = Some(op.schema_winner_id);
                    }
                }
                let uses_in0 = op.kind != OP_UNIT;
                let uses_in1 = matches!(op.kind, OP_JOIN_INNER | OP_JOIN_SEMI | OP_UNION | OP_DIFF);
                let out = usize::try_from(op.out).unwrap_or(usize::MAX);
                let in0 = usize::try_from(op.in0).unwrap_or(usize::MAX);
                let in1 = usize::try_from(op.in1).unwrap_or(usize::MAX);
                if out >= slots.len()
                    || (uses_in0 && in0 >= slots.len())
                    || (uses_in1 && in1 >= slots.len())
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} references a slot out of range",
                        op.op_id
                    )));
                }
                let slot_end = region.first_slot + region.slot_count;
                let in_scope = |slot: u32| slot >= region.first_slot && slot < slot_end;
                if !in_scope(op.out)
                    || (uses_in0 && !in_scope(op.in0))
                    || (uses_in1 && !in_scope(op.in1))
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} references a slot outside its region scope",
                        op.op_id
                    )));
                }
                if op.kind != OP_SCAN
                    && ((uses_in0 && op.out == op.in0) || (uses_in1 && op.out == op.in1))
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} aliases an input and output slot",
                        op.op_id
                    )));
                }
                if op.kind == OP_UNIT
                    && (op.in0 != 0
                        || op.in1 != 0
                        || op.in0_generation != 0
                        || op.in1_generation != 0
                        || op.aux_offset != 0
                        || op.aux_count != 0
                        || op.left_key != 0
                        || op.right_key != 0
                        || slots[out].relation.arity != 0)
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule unit {} has nonzero operands or invalid output",
                        op.op_id
                    )));
                }
                if op.kind == OP_SCAN
                    && (op.in1 != 0
                        || op.in1_generation != 0
                        || op.aux_offset != 0
                        || op.aux_count != 0
                        || op.left_key != 0
                        || op.right_key != 0)
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule scan {} has nonzero operands",
                        op.op_id
                    )));
                }
                match op.kind {
                    OP_UNIT | OP_SCAN => {}
                    OP_FILTER => {
                        if op.in1 != 0
                            || op.in1_generation != 0
                            || op.left_key != 0
                            || op.right_key != 0
                            || slots[in0].schema_tag != slots[out].schema_tag
                            || !same_relation_layout(&slots[in0].relation, &slots[out].relation)
                        {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule filter {} is invalid",
                                op.op_id
                            )));
                        }
                        requirements.filter_capacity = requirements
                            .filter_capacity
                            .max(slots[in0].relation.capacity);
                    }
                    OP_PROJECT => {
                        if op.in1 != 0
                            || op.in1_generation != 0
                            || op.left_key != 0
                            || op.right_key != 0
                            || op.aux_count != slots[out].relation.arity
                        {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule project {} is invalid",
                                op.op_id
                            )));
                        }
                    }
                    OP_UNION | OP_DIFF => {
                        if op.aux_offset != 0
                            || op.aux_count != 0
                            || op.left_key != 0
                            || op.right_key != 0
                            || slots[in0].schema_tag != slots[in1].schema_tag
                            || slots[in0].schema_tag != slots[out].schema_tag
                            || !same_relation_layout(&slots[in0].relation, &slots[in1].relation)
                            || !same_relation_layout(&slots[in0].relation, &slots[out].relation)
                        {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule set operation {} is invalid",
                                op.op_id
                            )));
                        }
                        let candidate_capacity = u64::from(slots[in0].relation.capacity)
                            .checked_add(u64::from(slots[in1].relation.capacity))
                            .ok_or_else(|| {
                                XlogError::Kernel("resident schedule set capacity overflow".into())
                            })?;
                        requirements.set_candidate_capacity =
                            requirements.set_candidate_capacity.max(candidate_capacity);
                    }
                    OP_JOIN_INNER | OP_JOIN_SEMI => {
                        if op.aux_offset != 0 || op.aux_count != 0 {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule join {} has invalid auxiliary operands",
                                op.op_id
                            )));
                        }
                        let left = &slots[in0].relation;
                        let right = &slots[in1].relation;
                        let output = &slots[out].relation;
                        let expected_arity = if op.kind == OP_JOIN_SEMI {
                            left.arity
                        } else {
                            left.arity.checked_add(right.arity).ok_or_else(|| {
                                XlogError::Kernel("resident schedule join arity overflow".into())
                            })?
                        };
                        let left_key = usize::try_from(op.left_key).unwrap_or(usize::MAX);
                        let right_key = usize::try_from(op.right_key).unwrap_or(usize::MAX);
                        let keys_match = left_key < left.arity as usize
                            && right_key < right.arity as usize
                            && left.widths[left_key] == right.widths[right_key]
                            && slot_types[in0].get(left_key) == slot_types[in1].get(right_key);
                        let output_widths_match = output.arity == expected_arity
                            && output.widths[..left.arity as usize]
                                == left.widths[..left.arity as usize]
                            && (op.kind == OP_JOIN_SEMI
                                || output.widths[left.arity as usize..expected_arity as usize]
                                    == right.widths[..right.arity as usize]);
                        let output_types_match = slot_types[out].get(..left.arity as usize)
                            == slot_types[in0].get(..left.arity as usize)
                            && (op.kind == OP_JOIN_SEMI
                                || slot_types[out]
                                    .get(left.arity as usize..expected_arity as usize)
                                    == slot_types[in1].get(..right.arity as usize));
                        if expected_arity as usize > RESIDENT_SCHEDULE_MAX_ARITY
                            || !keys_match
                            || !output_widths_match
                            || !output_types_match
                        {
                            return Err(XlogError::Kernel(format!(
                                "resident schedule join {} key or output schema is invalid",
                                op.op_id
                            )));
                        }
                        requirements.join_right_capacity =
                            requirements.join_right_capacity.max(right.capacity);
                    }
                    _ => unreachable!("operation kind validated above"),
                }
                if uses_in0 && !slot_input_is_ready(state[in0].0, state[in0].1, op.in0_generation) {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} reads an undefined or stale input",
                        op.op_id
                    )));
                }
                if uses_in1 && !slot_input_is_ready(state[in1].0, state[in1].1, op.in1_generation) {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule operation {} reads an undefined or stale input",
                        op.op_id
                    )));
                }
                if op.kind == OP_SCAN {
                    if op.out != op.in0 || op.out_generation != op.in0_generation {
                        return Err(XlogError::Kernel(format!(
                            "resident schedule scan {} is not an identity leaf",
                            op.op_id
                        )));
                    }
                } else {
                    if !slot_output_generation_is_valid(
                        state[out].0,
                        state[out].1,
                        op.out_generation,
                    ) {
                        return Err(XlogError::Kernel(format!(
                            "resident schedule operation {} has an invalid output generation",
                            op.op_id
                        )));
                    }
                    state[out].0 = finish_slot_write(state[out].0, true);
                    state[out].1 = op.out_generation;
                }
            }
        }
        if recursive && novelty_count == 0 {
            return Err(XlogError::Kernel(
                "resident schedule recursive body has no marked novelty output".into(),
            ));
        }
    }
    for (head, (&default, candidate)) in schema_defaults
        .iter()
        .zip(first_schema_candidates)
        .enumerate()
    {
        if candidate != Some(default) {
            return Err(XlogError::Kernel(format!(
                "resident schedule schema default for head {head} does not match its first candidate"
            )));
        }
    }
    Ok(requirements)
}

fn buffers_share_storage(left: &CudaBuffer, right: &CudaBuffer) -> bool {
    fn allocation_aliases(
        left_ptr: u64,
        left_len: u64,
        left_block: Option<&crate::device_runtime::DeviceBlock>,
        right_ptr: u64,
        right_len: u64,
        right_block: Option<&crate::device_runtime::DeviceBlock>,
    ) -> bool {
        let same_runtime_allocation = match (left_block, right_block) {
            (Some(left), Some(right)) => {
                left.ptr == right.ptr && left.generation == right.generation
            }
            _ => false,
        };
        same_runtime_allocation || device_ranges_overlap(left_ptr, left_len, right_ptr, right_len)
    }

    let left_count_aliases_right_count = allocation_aliases(
        left.num_rows_device().device_ptr_value(),
        std::mem::size_of::<u32>() as u64,
        left.num_rows_device().runtime_block(),
        right.num_rows_device().device_ptr_value(),
        std::mem::size_of::<u32>() as u64,
        right.num_rows_device().runtime_block(),
    );
    let left_count_aliases_right_column = right.columns().iter().any(|right_column| {
        allocation_aliases(
            left.num_rows_device().device_ptr_value(),
            std::mem::size_of::<u32>() as u64,
            left.num_rows_device().runtime_block(),
            *right_column.device_ptr(),
            u64::try_from(right_column.len()).unwrap_or(u64::MAX),
            right_column.runtime_block(),
        )
    });
    let left_column_aliases_right_count = left.columns().iter().any(|left_column| {
        allocation_aliases(
            *left_column.device_ptr(),
            u64::try_from(left_column.len()).unwrap_or(u64::MAX),
            left_column.runtime_block(),
            right.num_rows_device().device_ptr_value(),
            std::mem::size_of::<u32>() as u64,
            right.num_rows_device().runtime_block(),
        )
    });
    let columns_alias = left.columns().iter().any(|left_column| {
        right.columns().iter().any(|right_column| {
            allocation_aliases(
                *left_column.device_ptr(),
                u64::try_from(left_column.len()).unwrap_or(u64::MAX),
                left_column.runtime_block(),
                *right_column.device_ptr(),
                u64::try_from(right_column.len()).unwrap_or(u64::MAX),
                right_column.runtime_block(),
            )
        })
    });
    left_count_aliases_right_count
        || left_count_aliases_right_column
        || left_column_aliases_right_count
        || columns_alias
}

fn device_ranges_overlap(left_ptr: u64, left_len: u64, right_ptr: u64, right_len: u64) -> bool {
    if left_len == 0 || right_len == 0 {
        return false;
    }
    let left_end = left_ptr.saturating_add(left_len);
    let right_end = right_ptr.saturating_add(right_len);
    left_ptr < right_end && right_ptr < left_end
}

fn relation_view(buffer: &CudaBuffer) -> Result<(ResidentRelationView, u32)> {
    if buffer.arity() > RESIDENT_SCHEDULE_MAX_ARITY {
        return Err(XlogError::Kernel(format!(
            "resident schedule relation arity {} exceeds {RESIDENT_SCHEDULE_MAX_ARITY}",
            buffer.arity()
        )));
    }
    let capacity = checked_capacity(buffer.num_rows(), "relation")?;
    let mut columns = [0; RESIDENT_SCHEDULE_MAX_ARITY];
    let mut widths = [0; RESIDENT_SCHEDULE_MAX_ARITY];
    let mut schema_tag = 2_166_136_261_u32;
    schema_tag ^= checked_u32(buffer.arity(), "arity")?;
    schema_tag = schema_tag.wrapping_mul(16_777_619);
    for column in 0..buffer.arity() {
        columns[column] = *buffer.column(column).expect("arity checked").device_ptr();
        let scalar = buffer
            .schema()
            .column_type(column)
            .expect("schema arity checked");
        let width = resident_schedule_scalar_width(scalar)?;
        widths[column] = width;
        schema_tag ^= resident_schedule_scalar_tag(scalar);
        schema_tag = schema_tag.wrapping_mul(16_777_619);
    }
    Ok((
        ResidentRelationView {
            columns,
            widths,
            arity: checked_u32(buffer.arity(), "arity")?,
            capacity,
            reserved: 0,
            num_rows: buffer.num_rows_device().device_ptr_value(),
        },
        schema_tag,
    ))
}

fn relation_scalar_types(buffer: &CudaBuffer) -> Vec<ScalarType> {
    (0..buffer.arity())
        .map(|column| {
            buffer
                .schema()
                .column_type(column)
                .expect("buffer schema arity")
        })
        .collect()
}

fn finalize_schedule_output_counts(
    provider: &CudaKernelProvider,
    relations: &[ResidentScheduleRelation<'_>],
    receipt_slots: &[u32],
    counts: &[u32],
) -> Result<()> {
    if counts.len() != receipt_slots.len() {
        return Err(XlogError::Kernel(
            "resident schedule receipt count table is truncated".into(),
        ));
    }
    let mut entries = Vec::new();
    for (slot, relation) in relations.iter().enumerate() {
        if !relation.is_output() {
            continue;
        }
        let receipt_index = receipt_slots
            .iter()
            .position(|candidate| *candidate == slot as u32)
            .ok_or_else(|| {
                XlogError::Kernel(format!(
                    "resident schedule output slot {slot} has no receipt field"
                ))
            })?;
        entries.push((relation.buffer(), counts[receipt_index]));
    }
    provider.finalize_resident_logical_counts(&entries)
}

impl CudaKernelProvider {
    pub fn bind_resident_execution_domain(
        &self,
        runtime: Arc<XlogDeviceRuntime>,
        stream_id: StreamId,
        stream: Arc<CudaStream>,
    ) -> Result<ResidentExecutionDomain> {
        let manager_runtime = self.memory().runtime().ok_or_else(|| {
            XlogError::Kernel(
                "resident execution domain requires a runtime-backed memory manager".into(),
            )
        })?;
        if !Arc::ptr_eq(manager_runtime, &runtime)
            || !Arc::ptr_eq(self.device(), self.memory().device())
            || !Arc::ptr_eq(self.device(), runtime.device())
        {
            return Err(XlogError::Kernel(
                "resident execution domain provider, manager, and runtime identities differ".into(),
            ));
        }
        let device_ordinal = u32::try_from(self.device().ordinal()).map_err(|_| {
            XlogError::Kernel("resident execution domain device ordinal overflow".into())
        })?;
        if runtime.device_ordinal() != device_ordinal || !runtime.supports_block_use_tracking() {
            return Err(XlogError::Kernel(
                "resident execution domain runtime is incompatible with the provider".into(),
            ));
        }
        let resolved_stream = runtime.stream_pool().resolve(stream_id).ok_or_else(|| {
            XlogError::Kernel(
                "resident execution domain stream id is not owned by the runtime".into(),
            )
        })?;
        if !Arc::ptr_eq(&resolved_stream, &stream) {
            return Err(XlogError::Kernel(
                "resident execution domain stream does not match its runtime stream id".into(),
            ));
        }
        let provider_context = self.device().inner().stream().context();
        if !Arc::ptr_eq(stream.context(), provider_context)
            || stream.context().cu_ctx() != provider_context.cu_ctx()
        {
            return Err(XlogError::Kernel(
                "resident execution domain stream belongs to a foreign CUDA context".into(),
            ));
        }
        Ok(ResidentExecutionDomain {
            provider_identity: self.provider_identity(),
            memory_manager_identity: Arc::as_ptr(self.memory()) as usize,
            runtime,
            stream_id,
            stream,
            context: Arc::clone(provider_context),
            marker: Arc::new(()),
        })
    }

    fn upload_resident_schedule_metadata<T>(&self, values: &[T]) -> Result<TrackedCudaSlice<T>>
    where
        T: DeviceRepr + Default + Copy,
    {
        let mut allocation = self.memory.alloc::<T>(values.len().max(1))?;
        if values.is_empty() {
            self.htod_launch_metadata_sync_copy_into(&[T::default()], &mut allocation)?;
        } else {
            self.htod_launch_metadata_sync_copy_into(values, &mut allocation)?;
        }
        Ok(allocation)
    }

    fn upload_resident_schedule_metadata_in_reservation<T>(
        &self,
        values: &[T],
        reservation: &mut GpuMemoryReservation,
    ) -> Result<TrackedCudaSlice<T>>
    where
        T: DeviceRepr + Default + Copy,
    {
        let mut allocation = reservation.alloc::<T>(values.len().max(1))?;
        if values.is_empty() {
            self.htod_launch_metadata_sync_copy_into(&[T::default()], &mut allocation)?;
        } else {
            self.htod_launch_metadata_sync_copy_into(values, &mut allocation)?;
        }
        Ok(allocation)
    }

    /// Validate, allocate, and upload every descriptor before graph capture.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_resident_schedule<'a>(
        &self,
        mut relations: Vec<ResidentScheduleRelation<'a>>,
        ops: &[ResidentOpDescriptor],
        waves: &[ResidentWaveDescriptor],
        regions: &[ResidentRegionDescriptor],
        filter_comparisons: &[ResidentFilterComparisonDescriptor],
        project_expressions: &[ResidentProjectExpressionDescriptor],
        receipt_slots: &[u32],
    ) -> Result<ResidentSchedule<'a>> {
        let slot_count = checked_u32(relations.len(), "slot count")?;
        let op_count = checked_u32(ops.len(), "operation count")?;
        let wave_count = checked_u32(waves.len(), "wave count")?;
        let region_count = checked_u32(regions.len(), "region count")?;
        let filter_comparison_count =
            checked_u32(filter_comparisons.len(), "filter comparison count")?;
        let project_expression_count =
            checked_u32(project_expressions.len(), "project expression count")?;

        let mut slot_descriptors = Vec::with_capacity(relations.len());
        let mut slot_types = Vec::with_capacity(relations.len());
        let mut max_capacity = 0_u32;
        let provider_manager = Arc::as_ptr(&self.memory) as usize;
        let provider_context = self.device.inner().stream().context().cu_ctx();
        let provider_ordinal = self.device.ordinal() as u32;
        for relation in &mut relations {
            let buffer = relation.buffer();
            if buffer.num_rows_device().memory_manager_ptr_value() != provider_manager {
                return Err(XlogError::Kernel(
                    "resident schedule relation belongs to a foreign provider".into(),
                ));
            }
            for column in buffer.columns() {
                if column.stream().context().cu_ctx() != provider_context
                    || column
                        .runtime_block()
                        .is_some_and(|block| block.device_ordinal != provider_ordinal)
                {
                    return Err(XlogError::Kernel(
                        "resident schedule relation belongs to a foreign CUDA context".into(),
                    ));
                }
            }
            let (view, schema_tag) = relation_view(buffer)?;
            if u64::from(relation.initial_count()) > buffer.num_rows() {
                return Err(XlogError::Kernel(format!(
                    "resident schedule initial count {} exceeds capacity {}",
                    relation.initial_count(),
                    buffer.num_rows()
                )));
            }
            max_capacity = max_capacity.max(view.capacity);
            slot_descriptors.push(ResidentRelationSlot {
                relation: view,
                generation: relation.generation(),
                flags: relation.flags(),
                initial_count: relation.initial_count(),
                schema_tag,
            });
            slot_types.push(relation_scalar_types(buffer));
        }
        for (output_slot, output) in relations.iter().enumerate() {
            if !output.is_output() {
                continue;
            }
            for (other_slot, other) in relations.iter().enumerate() {
                if output_slot != other_slot
                    && buffers_share_storage(output.buffer(), other.buffer())
                {
                    return Err(XlogError::Kernel(format!(
                        "resident schedule output slot {output_slot} aliases storage in slot {other_slot}"
                    )));
                }
            }
        }

        let mut region_descriptors = regions.to_vec();
        let slot_generations = slot_descriptors
            .iter()
            .map(|slot| slot.generation)
            .collect::<Vec<_>>();
        let generation_base_values =
            build_generation_baselines(&mut region_descriptors, &slot_generations)?;
        let generation_base_count =
            checked_u32(generation_base_values.len(), "generation baseline count")?;
        validate_generation_baseline_ranges(&region_descriptors, generation_base_count)?;

        let requirements = validate_schedule_program(
            &slot_descriptors,
            &slot_types,
            ops,
            waves,
            &region_descriptors,
            &generation_base_values,
            filter_comparisons,
            project_expressions,
            &[],
        )?;
        let filter_capacity = requirements.filter_capacity;
        let set_candidate_capacity = requirements.set_candidate_capacity;
        let join_right_capacity = requirements.join_right_capacity;
        let set_slot_count = checked_workspace_slots(set_candidate_capacity, "set workspace")?;
        let join_bucket_count =
            checked_workspace_slots(u64::from(join_right_capacity), "join workspace")?;
        let filter_block_count = filter_capacity
            .div_ceil(RESIDENT_SCHEDULE_BLOCK_SIZE)
            .max(1);
        let requested_receipt_count = receipt_slots.len();
        let mut all_receipt_slots = receipt_slots.to_vec();
        for (slot, relation) in relations.iter().enumerate() {
            let slot = checked_u32(slot, "receipt slot")?;
            if relation.is_output() && !all_receipt_slots.contains(&slot) {
                all_receipt_slots.push(slot);
            }
        }
        let receipt_count = checked_u32(all_receipt_slots.len(), "receipt count")?;
        let receipt_byte_count = std::mem::size_of::<ResidentTerminalStatus>()
            .checked_add(
                std::mem::size_of::<u32>()
                    .checked_mul(all_receipt_slots.len() + 1)
                    .ok_or_else(|| {
                        XlogError::Kernel("resident schedule receipt overflow".into())
                    })?,
            )
            .ok_or_else(|| XlogError::Kernel("resident schedule receipt overflow".into()))?;
        let receipt_byte_count_u32 = checked_u32(receipt_byte_count, "receipt byte count")?;
        let mut receipt_count_ptrs = Vec::with_capacity(all_receipt_slots.len());
        for &slot in &all_receipt_slots {
            let index = usize::try_from(slot).unwrap_or(usize::MAX);
            let relation = relations.get(index).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "resident schedule receipt slot {slot} is out of range"
                ))
            })?;
            receipt_count_ptrs.push(relation.buffer().num_rows_device().device_ptr_value());
        }

        let function = self
            .device()
            .inner()
            .get_func(MODULE, KERNEL)
            .ok_or_else(|| XlogError::Kernel("resident_schedule_execute kernel missing".into()))?;
        let cooperative = self
            .device()
            .inner()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COOPERATIVE_LAUNCH)
            .map_err(|error| XlogError::Kernel(format!("query cooperative launch: {error}")))?;
        if cooperative == 0 {
            return Err(XlogError::Kernel(
                "CUDA device does not support cooperative kernel launch".into(),
            ));
        }
        let active_per_sm = function
            .occupancy_max_active_blocks_per_multiprocessor(RESIDENT_SCHEDULE_BLOCK_SIZE, 0, None)
            .map_err(|error| XlogError::Kernel(format!("resident schedule occupancy: {error}")))?;
        let multiprocessors = self
            .device()
            .inner()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| XlogError::Kernel(format!("query multiprocessor count: {error}")))?;
        let cooperative_limit =
            active_per_sm
                .checked_mul(u32::try_from(multiprocessors).map_err(|_| {
                    XlogError::Kernel("CUDA multiprocessor count is invalid".into())
                })?)
                .ok_or_else(|| XlogError::Kernel("resident schedule grid overflow".into()))?;
        if cooperative_limit == 0 {
            return Err(XlogError::Kernel(
                "resident schedule has zero cooperative occupancy".into(),
            ));
        }
        let required_blocks = max_capacity
            .max(filter_capacity)
            .div_ceil(RESIDENT_SCHEDULE_BLOCK_SIZE)
            .max(1);
        let launch_config = LaunchConfig {
            grid_dim: (required_blocks.min(cooperative_limit), 1, 1),
            block_dim: (RESIDENT_SCHEDULE_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let slots = self.upload_resident_schedule_metadata(&slot_descriptors)?;
        let ops = self.upload_resident_schedule_metadata(ops)?;
        let waves = self.upload_resident_schedule_metadata(waves)?;
        let regions = self.upload_resident_schedule_metadata(&region_descriptors)?;
        let generation_bases = self.upload_resident_schedule_metadata(&generation_base_values)?;
        let filter_comparisons = self.upload_resident_schedule_metadata(filter_comparisons)?;
        let project_expressions = self.upload_resident_schedule_metadata(project_expressions)?;
        let filter_mask = self
            .memory
            .alloc::<u32>(usize::try_from(filter_capacity).unwrap_or(0).max(1))?;
        let filter_prefix = self
            .memory
            .alloc::<u32>(usize::try_from(filter_capacity).unwrap_or(0).max(1))?;
        let filter_block_sums = self
            .memory
            .alloc::<u32>(usize::try_from(filter_block_count).unwrap_or(1))?;
        let filter_block_offsets = self
            .memory
            .alloc::<u32>(usize::try_from(filter_block_count).unwrap_or(1))?;
        let set_slots = self
            .memory
            .alloc::<u64>(usize::try_from(set_slot_count).unwrap_or(1))?;
        let set_required = self.memory.alloc::<u64>(1)?;
        let join_buckets = self
            .memory
            .alloc::<u32>(usize::try_from(join_bucket_count).unwrap_or(1))?;
        let join_next = self
            .memory
            .alloc::<u32>(usize::try_from(join_right_capacity).unwrap_or(0).max(1))?;
        let join_required = self.memory.alloc::<u64>(1)?;
        let status =
            self.upload_resident_schedule_metadata(&[ResidentTerminalStatus::default()])?;
        let changed = self.upload_resident_schedule_metadata(&[0_u32])?;
        let iterations = self.upload_resident_schedule_metadata(&[0_u32])?;
        let scan_trace = self.upload_resident_schedule_metadata(&[0_u32])?;
        let filter_trace = self.upload_resident_schedule_metadata(&[0_u32])?;
        let semantic_scan_trace = self.upload_resident_schedule_metadata(&[0_u32])?;
        let semantic_filter_trace = self.upload_resident_schedule_metadata(&[0_u32])?;
        let receipt_table = self.upload_resident_schedule_metadata(&receipt_count_ptrs)?;
        let receipt_bytes = self.memory.alloc::<u8>(receipt_byte_count.max(1))?;
        let pinned_receipt = ResidentSchedulePinnedReceipt::allocate(receipt_byte_count)?;

        let header_value = ResidentScheduleHeader {
            slots: slots.device_ptr_value(),
            ops: ops.device_ptr_value(),
            waves: waves.device_ptr_value(),
            regions: regions.device_ptr_value(),
            generation_metadata: generation_bases.device_ptr_value(),
            filter_comparisons: filter_comparisons.device_ptr_value(),
            project_expressions: project_expressions.device_ptr_value(),
            filter_mask: filter_mask.device_ptr_value(),
            filter_prefix: filter_prefix.device_ptr_value(),
            filter_block_sums: filter_block_sums.device_ptr_value(),
            filter_block_offsets: filter_block_offsets.device_ptr_value(),
            set_slots: set_slots.device_ptr_value(),
            set_required: set_required.device_ptr_value(),
            join_buckets: join_buckets.device_ptr_value(),
            join_next: join_next.device_ptr_value(),
            join_required: join_required.device_ptr_value(),
            status: status.device_ptr_value(),
            changed: changed.device_ptr_value(),
            iterations: iterations.device_ptr_value(),
            scan_trace: scan_trace.device_ptr_value(),
            filter_trace: filter_trace.device_ptr_value(),
            semantic_scan_trace: semantic_scan_trace.device_ptr_value(),
            semantic_filter_trace: semantic_filter_trace.device_ptr_value(),
            schema_seen_nonempty: 0,
            schema_winner_ids: 0,
            receipt_table: receipt_table.device_ptr_value(),
            receipt_bytes: receipt_bytes.device_ptr_value(),
            slot_count,
            op_count,
            wave_count,
            region_count,
            filter_comparison_count,
            project_expression_count,
            filter_capacity,
            filter_block_count,
            set_slot_mask: set_slot_count - 1,
            set_candidate_capacity: u32::try_from(set_candidate_capacity).map_err(|_| {
                XlogError::Kernel("resident schedule set capacity exceeds u32".into())
            })?,
            join_bucket_mask: join_bucket_count - 1,
            join_right_capacity,
            schema_winner_count: 0,
            receipt_count,
            receipt_byte_count: receipt_byte_count_u32,
            generation_metadata_count: generation_base_count,
            abi_version: RESIDENT_SCHEDULE_ABI_VERSION,
            reserved: 0,
        };
        let header = self.upload_resident_schedule_metadata(&[header_value])?;
        for relation in &mut relations {
            relation.invalidate_output_metadata();
        }

        Ok(ResidentSchedule {
            origin_provider_identity: self.provider_identity(),
            origin_memory_manager: Arc::as_ptr(&self.memory) as usize,
            header,
            _slots: slots,
            _ops: ops,
            _waves: waves,
            _regions: regions,
            _generation_metadata: generation_bases,
            _filter_comparisons: filter_comparisons,
            _project_expressions: project_expressions,
            _filter_mask: filter_mask,
            _filter_prefix: filter_prefix,
            _filter_block_sums: filter_block_sums,
            _filter_block_offsets: filter_block_offsets,
            _set_slots: set_slots,
            _set_required: set_required,
            _join_buckets: join_buckets,
            _join_next: join_next,
            _join_required: join_required,
            _status: status,
            _changed: changed,
            _iterations: iterations,
            _scan_trace: scan_trace,
            _filter_trace: filter_trace,
            _semantic_scan_trace: semantic_scan_trace,
            _semantic_filter_trace: semantic_filter_trace,
            _receipt_table: receipt_table,
            receipt_bytes,
            pinned_receipt,
            launch_config,
            region_count,
            region_descriptors,
            requested_receipt_count,
            receipt_slots: all_receipt_slots,
            relations,
        })
    }

    /// Materialize only compact scheduler metadata from the enclosing runtime's
    /// already-admitted reservation. All relation, workspace, control, trace,
    /// schema-winner, receipt, graph, and lifecycle owners remain external.
    #[allow(clippy::too_many_arguments)]
    pub fn prepare_resident_schedule_program_in_reservation<'a>(
        &self,
        domain: &ResidentExecutionDomain,
        bindings: &[ResidentScheduleSlotBinding<'a>],
        ops: &[ResidentOpDescriptor],
        waves: &[ResidentWaveDescriptor],
        regions: &[ResidentRegionDescriptor],
        generation_bases: &[u32],
        filter_comparisons: &[ResidentFilterComparisonDescriptor],
        project_expressions: &[ResidentProjectExpressionDescriptor],
        receipt_slots: &[u32],
        external: ResidentScheduleExternalBindings<'a>,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentScheduleDeviceProgram> {
        validate_execution_domain(self, domain)?;
        if reservation.memory_manager_ptr_value() != domain.memory_manager_identity {
            return Err(XlogError::Kernel(
                "resident schedule reservation belongs to a foreign memory manager".into(),
            ));
        }
        let slot_count = checked_u32(bindings.len(), "slot count")?;
        let op_count = checked_u32(ops.len(), "op count")?;
        let wave_count = checked_u32(waves.len(), "wave count")?;
        let region_count = checked_u32(regions.len(), "region count")?;
        let generation_base_count =
            checked_u32(generation_bases.len(), "generation baseline count")?;
        let filter_comparison_count =
            checked_u32(filter_comparisons.len(), "filter comparison count")?;
        let project_expression_count =
            checked_u32(project_expressions.len(), "project expression count")?;

        if bindings.is_empty() {
            return Err(XlogError::Kernel(
                "resident schedule program requires slots".into(),
            ));
        }
        validate_region_control_and_ranges(regions, wave_count, slot_count)?;
        validate_generation_baseline_ranges(regions, generation_base_count)?;
        validate_wave_partition(waves, op_count)?;

        let mut slot_descriptors = Vec::with_capacity(bindings.len());
        let mut slot_types = Vec::with_capacity(bindings.len());
        let mut slot_count_identities = Vec::with_capacity(bindings.len());
        let mut allocation_ranges = Vec::new();
        let mut max_capacity = 0_u32;
        for binding in bindings {
            for column in binding.buffer().columns() {
                validate_schedule_allocation(
                    column.runtime_allocation_identity()?,
                    domain,
                    &mut allocation_ranges,
                )?;
            }
            slot_count_identities.push(validate_schedule_allocation(
                binding
                    .buffer()
                    .num_rows_device()
                    .runtime_allocation_identity()?,
                domain,
                &mut allocation_ranges,
            )?);
            let (relation, schema_tag) = relation_view(binding.buffer())?;
            max_capacity = max_capacity.max(relation.capacity);
            slot_descriptors.push(ResidentRelationSlot {
                relation,
                generation: binding.generation(),
                flags: binding.flags(),
                initial_count: binding.initial_count(),
                schema_tag,
            });
            slot_types.push(relation_scalar_types(binding.buffer()));
        }

        let (receipt_table, receipt_bytes, receipt_count, receipt_byte_count) =
            external.receipt.schedule_parts();
        let (schema_seen_nonempty, schema_winner_ids, schema_winner_count) =
            external.schema_winners.schedule_parts();
        let head_count =
            checked_schedule_winner_count(receipt_count, receipt_byte_count, schema_winner_count)?;
        let schema_defaults = external.schema_winners.default_schema_ids();
        if schema_defaults.len() != head_count as usize {
            return Err(XlogError::Kernel(
                "resident schedule schema-default count differs from the receipt".into(),
            ));
        }
        let generation_metadata = build_generation_metadata(generation_bases, schema_defaults)?;
        let generation_metadata_count =
            checked_u32(generation_metadata.len(), "generation metadata count")?;
        let requirements = validate_schedule_program(
            &slot_descriptors,
            &slot_types,
            ops,
            waves,
            regions,
            generation_bases,
            filter_comparisons,
            project_expressions,
            schema_defaults,
        )?;

        if let Some(filter_scratch) = external.filter_scratch {
            for snapshot in filter_scratch.schedule_owner_snapshots()? {
                validate_schedule_allocation(snapshot, domain, &mut allocation_ranges)?;
            }
        }
        for snapshot in external.set_workspace.schedule_owner_snapshots()? {
            validate_schedule_allocation(snapshot, domain, &mut allocation_ranges)?;
        }
        for snapshot in external.join_workspace.schedule_owner_snapshots()? {
            validate_schedule_allocation(snapshot, domain, &mut allocation_ranges)?;
        }
        for snapshot in external.control.schedule_owner_snapshots()? {
            validate_schedule_allocation(snapshot, domain, &mut allocation_ranges)?;
        }
        let [scan_trace_snapshot, filter_trace_snapshot, semantic_scan_trace_snapshot, semantic_filter_trace_snapshot] =
            external.trace.schedule_owner_snapshots()?;
        let scan_trace_identity =
            validate_schedule_allocation(scan_trace_snapshot, domain, &mut allocation_ranges)?;
        let filter_trace_identity =
            validate_schedule_allocation(filter_trace_snapshot, domain, &mut allocation_ranges)?;
        let semantic_scan_trace_identity = validate_schedule_allocation(
            semantic_scan_trace_snapshot,
            domain,
            &mut allocation_ranges,
        )?;
        let semantic_filter_trace_identity = validate_schedule_allocation(
            semantic_filter_trace_snapshot,
            domain,
            &mut allocation_ranges,
        )?;
        let [schema_seen_snapshot, schema_winner_snapshot] =
            external.schema_winners.schedule_owner_snapshots()?;
        validate_schedule_allocation(schema_seen_snapshot, domain, &mut allocation_ranges)?;
        let schema_winner_identity =
            validate_schedule_allocation(schema_winner_snapshot, domain, &mut allocation_ranges)?;
        for snapshot in external.receipt.schedule_owner_snapshots()? {
            validate_schedule_allocation(snapshot, domain, &mut allocation_ranges)?;
        }

        if external.receipt.relation_count_len() != head_count
            || external.receipt.device_trace_field_count() != 4
            || external.receipt.schema_winner_count() != head_count
            || external.receipt.total_count_field_len() != receipt_count
        {
            return Err(XlogError::Kernel(
                "resident schedule receipt owner shape differs from the header".into(),
            ));
        }
        let slot_flags: Vec<u32> = slot_descriptors.iter().map(|slot| slot.flags).collect();
        let receipt_slot_indices =
            validate_receipt_slot_mapping(receipt_slots, &slot_flags, head_count)?;
        let mut relation_count_ptrs = Vec::with_capacity(receipt_slot_indices.len());
        let expected_block_count = receipt_slot_indices
            .len()
            .checked_mul(2)
            .and_then(|count| count.checked_add(4))
            .ok_or_else(|| XlogError::Kernel("resident receipt block count overflow".into()))?;
        let mut expected_receipt_blocks = Vec::with_capacity(expected_block_count);
        for slot in receipt_slot_indices {
            relation_count_ptrs.push(bindings[slot].buffer().num_rows_device().device_ptr_value());
            expected_receipt_blocks.push(slot_count_identities[slot].block_id);
        }
        let (scan_trace, filter_trace, semantic_scan_trace, semantic_filter_trace) =
            external.trace.schedule_parts();
        expected_receipt_blocks.push(scan_trace_identity.block_id);
        expected_receipt_blocks.push(filter_trace_identity.block_id);
        expected_receipt_blocks.push(semantic_scan_trace_identity.block_id);
        expected_receipt_blocks.push(semantic_filter_trace_identity.block_id);
        let mut schema_winner_ptrs = Vec::with_capacity(receipt_slots.len());
        for index in 0..head_count {
            let offset = u64::from(index)
                .checked_mul(u64::try_from(std::mem::size_of::<u32>()).map_err(|_| {
                    XlogError::Kernel("resident schema-winner element size overflow".into())
                })?)
                .ok_or_else(|| {
                    XlogError::Kernel("resident schema-winner offset overflow".into())
                })?;
            schema_winner_ptrs.push(schema_winner_ids.checked_add(offset).ok_or_else(|| {
                XlogError::Kernel("resident schema-winner pointer overflow".into())
            })?);
            expected_receipt_blocks.push(schema_winner_identity.block_id);
        }
        external.receipt.validate_schedule_pointees(
            domain.memory_manager_identity,
            domain.runtime.device_ordinal(),
            &relation_count_ptrs,
            [
                scan_trace,
                filter_trace,
                semantic_scan_trace,
                semantic_filter_trace,
            ],
            &schema_winner_ptrs,
            &expected_receipt_blocks,
        )?;

        let filter_capacity = requirements.filter_capacity;
        let set_candidate_capacity = requirements.set_candidate_capacity;
        let join_right_capacity = requirements.join_right_capacity;

        let (
            filter_mask,
            filter_prefix,
            filter_block_sums,
            filter_block_offsets,
            supplied_filter_capacity,
            filter_block_count,
        ) = match external.filter_scratch {
            Some(scratch) => scratch.schedule_parts(),
            None if filter_capacity == 0 => (0, 0, 0, 0, 0, 0),
            None => {
                return Err(XlogError::Kernel(
                    "resident schedule filter scratch is missing".into(),
                ));
            }
        };
        if supplied_filter_capacity < filter_capacity {
            return Err(XlogError::Kernel(
                "resident schedule filter scratch is undersized".into(),
            ));
        }
        let (set_slots, set_required, set_slot_mask, supplied_set_capacity) =
            external.set_workspace.schedule_parts();
        if u64::from(supplied_set_capacity) < set_candidate_capacity {
            return Err(XlogError::Kernel(
                "resident schedule set workspace is undersized".into(),
            ));
        }
        let (join_buckets, join_next, join_required, join_bucket_mask, supplied_join_capacity) =
            external.join_workspace.schedule_parts();
        if supplied_join_capacity < join_right_capacity {
            return Err(XlogError::Kernel(
                "resident schedule join workspace is undersized".into(),
            ));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, KERNEL)
            .ok_or_else(|| XlogError::Kernel("resident_schedule_execute kernel missing".into()))?;
        let active_per_sm = function
            .occupancy_max_active_blocks_per_multiprocessor(RESIDENT_SCHEDULE_BLOCK_SIZE, 0, None)
            .map_err(|error| XlogError::Kernel(format!("resident schedule occupancy: {error}")))?;
        let multiprocessors = self
            .device()
            .inner()
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .map_err(|error| XlogError::Kernel(format!("query multiprocessor count: {error}")))?;
        let cooperative_limit =
            active_per_sm
                .checked_mul(u32::try_from(multiprocessors).map_err(|_| {
                    XlogError::Kernel("CUDA multiprocessor count is invalid".into())
                })?)
                .ok_or_else(|| XlogError::Kernel("resident schedule grid overflow".into()))?;
        if cooperative_limit == 0 {
            return Err(XlogError::Kernel(
                "resident schedule has zero cooperative occupancy".into(),
            ));
        }
        let launch_config = LaunchConfig {
            grid_dim: (
                max_capacity
                    .div_ceil(RESIDENT_SCHEDULE_BLOCK_SIZE)
                    .max(1)
                    .min(cooperative_limit),
                1,
                1,
            ),
            block_dim: (RESIDENT_SCHEDULE_BLOCK_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };

        let required_metadata_bytes = resident_schedule_metadata_device_bytes(
            bindings.len(),
            ops.len(),
            waves.len(),
            regions.len(),
            generation_metadata.len(),
            filter_comparisons.len(),
            project_expressions.len(),
        )?;
        if reservation.remaining_bytes() < required_metadata_bytes {
            return Err(XlogError::Kernel(format!(
                "resident schedule metadata requires {required_metadata_bytes} reserved bytes"
            )));
        }
        let slots =
            self.upload_resident_schedule_metadata_in_reservation(&slot_descriptors, reservation)?;
        let op_table = self.upload_resident_schedule_metadata_in_reservation(ops, reservation)?;
        let wave_table =
            self.upload_resident_schedule_metadata_in_reservation(waves, reservation)?;
        let region_table =
            self.upload_resident_schedule_metadata_in_reservation(regions, reservation)?;
        let generation_table = self
            .upload_resident_schedule_metadata_in_reservation(&generation_metadata, reservation)?;
        let filter_table =
            self.upload_resident_schedule_metadata_in_reservation(filter_comparisons, reservation)?;
        let project_table = self
            .upload_resident_schedule_metadata_in_reservation(project_expressions, reservation)?;

        let header_value = ResidentScheduleHeader {
            slots: slots.device_ptr_value(),
            ops: op_table.device_ptr_value(),
            waves: wave_table.device_ptr_value(),
            regions: region_table.device_ptr_value(),
            generation_metadata: generation_table.device_ptr_value(),
            filter_comparisons: filter_table.device_ptr_value(),
            project_expressions: project_table.device_ptr_value(),
            filter_mask,
            filter_prefix,
            filter_block_sums,
            filter_block_offsets,
            set_slots,
            set_required,
            join_buckets,
            join_next,
            join_required,
            status: external.control.status_device_ptr(),
            changed: external.control.changed_device_ptr(),
            iterations: external.control.loop_iterations_device().device_ptr_value(),
            scan_trace,
            filter_trace,
            semantic_scan_trace,
            semantic_filter_trace,
            schema_seen_nonempty,
            schema_winner_ids,
            receipt_table,
            receipt_bytes,
            slot_count,
            op_count,
            wave_count,
            region_count,
            filter_comparison_count,
            project_expression_count,
            filter_capacity: supplied_filter_capacity,
            filter_block_count,
            set_slot_mask,
            set_candidate_capacity: supplied_set_capacity,
            join_bucket_mask,
            join_right_capacity: supplied_join_capacity,
            schema_winner_count,
            receipt_count,
            receipt_byte_count,
            generation_metadata_count,
            abi_version: RESIDENT_SCHEDULE_ABI_VERSION,
            reserved: 0,
        };
        let header =
            self.upload_resident_schedule_metadata_in_reservation(&[header_value], reservation)?;

        Ok(ResidentScheduleDeviceProgram {
            origin_provider_identity: self.provider_identity(),
            domain: domain.clone(),
            header,
            _slots: slots,
            _ops: op_table,
            _waves: wave_table,
            _regions: region_table,
            _generation_metadata: generation_table,
            _filter_comparisons: filter_table,
            _project_expressions: project_table,
            launch_config,
            region_descriptors: regions.to_vec(),
        })
    }

    fn record_resident_schedule_on_stream(
        &self,
        schedule: &ResidentSchedule<'_>,
        region_index: u32,
        conditional_handle: u64,
        stream: &CudaStream,
    ) -> Result<()> {
        if region_index >= schedule.region_count {
            return Err(XlogError::Kernel(format!(
                "resident schedule region {region_index} is out of range"
            )));
        }
        let recursive = schedule.region_descriptors[region_index as usize].flags
            == RESIDENT_SCHEDULE_REGION_RECURSIVE;
        if recursive != (conditional_handle != 0) {
            return Err(XlogError::Kernel(
                "resident schedule conditional handle does not match the region kind".into(),
            ));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, KERNEL)
            .ok_or_else(|| XlogError::Kernel("resident_schedule_execute kernel missing".into()))?;
        let header = schedule.header.device_ptr_value();
        let mut params = vec![
            header.as_kernel_param(),
            region_index.as_kernel_param(),
            conditional_handle.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_schedule_execute, all captured
        // allocations are retained by `schedule`, and its grid is occupancy-capped.
        unsafe {
            function.launch_cooperative_on_stream(stream, schedule.launch_config, &mut params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident schedule launch: {error}")))
    }

    /// Record one compact scheduler region into a graph owned by the caller.
    ///
    /// # Safety
    /// The caller must register the program, every slot and external owner, and every indirect
    /// receipt pointee with the one enclosing strict recorder before domain-bound preflight.
    /// It must use that same execution domain for domain-bound preflight and domain-bound commit.
    /// The program and all registered owners must remain alive through graph destruction and
    /// completion of all in-flight work. The stream must be the exact stream retained by the
    /// program's execution domain. For a recursive region, the conditional body passed here must
    /// be the one minted for the enclosing graph, and this call must occur inside that body's
    /// active `capture_on_stream` callback.
    pub unsafe fn record_resident_schedule_region_on_stream(
        &self,
        program: &ResidentScheduleDeviceProgram,
        region_index: u32,
        conditional_body: Option<&ConditionalCudaGraphBody>,
        stream: &CudaStream,
    ) -> Result<()> {
        validate_execution_domain(self, &program.domain)?;
        if program.origin_provider_identity != self.provider_identity() {
            return Err(XlogError::Kernel(
                "resident schedule program belongs to a foreign provider".into(),
            ));
        }
        if !std::ptr::eq(program.domain.stream.as_ref(), stream)
            || program.domain.stream.cu_stream() != stream.cu_stream()
            || !Arc::ptr_eq(stream.context(), &program.domain.context)
            || stream.context().cu_ctx() != program.domain.context.cu_ctx()
        {
            return Err(XlogError::Kernel(
                "resident schedule record stream differs from its execution domain".into(),
            ));
        }
        if region_index as usize >= program.region_descriptors.len() {
            return Err(XlogError::Kernel(format!(
                "resident schedule region {region_index} is out of range"
            )));
        }
        let recursive = program.region_descriptors[region_index as usize].flags
            == RESIDENT_SCHEDULE_REGION_RECURSIVE;
        if recursive != conditional_body.is_some() {
            return Err(XlogError::Kernel(
                "resident schedule conditional handle does not match the region kind".into(),
            ));
        }
        let conditional_handle = conditional_body.map_or(0, ConditionalCudaGraphBody::handle);
        if conditional_body.is_some_and(|body| body.context() != program.domain.context.cu_ctx()) {
            return Err(XlogError::Kernel(
                "resident schedule conditional body belongs to a foreign CUDA context".into(),
            ));
        }
        let function = self
            .device()
            .inner()
            .get_func(MODULE, KERNEL)
            .ok_or_else(|| XlogError::Kernel("resident_schedule_execute kernel missing".into()))?;
        let header = program.header.device_ptr_value();
        let mut params = vec![
            header.as_kernel_param(),
            region_index.as_kernel_param(),
            conditional_handle.as_kernel_param(),
        ];
        function
            .launch_cooperative_on_stream(stream, program.launch_config, &mut params)
            .map_err(|error| XlogError::Kernel(format!("resident schedule launch: {error}")))
    }

    /// Capture every ordered region while retaining every raw-pointer owner.
    pub fn capture_resident_schedule<'a>(
        &'a self,
        schedule: ResidentSchedule<'a>,
        region_index: u32,
        stream: Arc<CudaStream>,
    ) -> Result<ResidentScheduleGraph<'a>> {
        if schedule.origin_provider_identity != self.provider_identity()
            || schedule.origin_memory_manager != Arc::as_ptr(&self.memory) as usize
        {
            return Err(XlogError::Kernel(
                "resident schedule belongs to a different CUDA kernel provider".into(),
            ));
        }
        let provider_context = self.device.inner().stream().context();
        if !Arc::ptr_eq(stream.context(), provider_context)
            || stream.context().cu_ctx() != provider_context.cu_ctx()
        {
            return Err(XlogError::Kernel(
                "resident schedule stream belongs to a foreign CUDA context".into(),
            ));
        }
        if region_index != 0 {
            return Err(XlogError::Kernel(
                "resident schedule capture must begin with its first region".into(),
            ));
        }
        let graph_error =
            |error| XlogError::Kernel(format!("resident schedule conditional graph: {error}"));
        let mut builder = ConditionalCudaGraphSequenceBuilder::new(&stream).map_err(graph_error)?;
        for (index, region) in schedule.region_descriptors.iter().enumerate() {
            let region_index = checked_u32(index, "region index")?;
            if region.flags == RESIDENT_SCHEDULE_REGION_RECURSIVE {
                let initial_value = u32::from(region.iteration_limit != 0);
                builder
                    .add_conditional_while(initial_value, true, |body| {
                        let handle = body.handle();
                        body.capture_on_stream(&stream, || {
                            self.record_resident_schedule_on_stream(
                                &schedule,
                                region_index,
                                handle,
                                &stream,
                            )
                        })
                    })
                    .map_err(graph_error)?;
            } else {
                builder
                    .capture_segment_on_stream(&stream, || {
                        self.record_resident_schedule_on_stream(&schedule, region_index, 0, &stream)
                    })
                    .map_err(graph_error)?;
            }
        }
        let graph = builder.instantiate().map_err(graph_error)?;
        Ok(ResidentScheduleGraph {
            graph,
            schedule,
            provider: self,
            stream,
            in_flight: false,
        })
    }

    fn observe_resident_schedule(
        &self,
        schedule: &mut ResidentSchedule<'_>,
        stream: &CudaStream,
    ) -> Result<ResidentScheduleReceipt> {
        if schedule.origin_provider_identity != self.provider_identity()
            || schedule.origin_memory_manager != Arc::as_ptr(&self.memory) as usize
        {
            return Err(XlogError::Kernel(
                "resident schedule belongs to a different CUDA kernel provider".into(),
            ));
        }
        let bytes = schedule
            .pinned_receipt
            .copy_from_device(schedule.receipt_bytes.device_ptr_value(), stream)?;
        self.record_final_observation_transfer(bytes.len() as u64);
        let status_bytes = std::mem::size_of::<ResidentTerminalStatus>();
        let expected_bytes = status_bytes
            .checked_add(
                std::mem::size_of::<u32>()
                    .checked_mul(schedule.receipt_slots.len() + 1)
                    .ok_or_else(|| {
                        XlogError::Kernel("resident schedule receipt size overflow".into())
                    })?,
            )
            .ok_or_else(|| XlogError::Kernel("resident schedule receipt size overflow".into()))?;
        if bytes.len() != expected_bytes {
            return Err(XlogError::Kernel(
                "resident schedule receipt has an invalid byte length".into(),
            ));
        }
        // SAFETY: length is checked and the wire type accepts every bit pattern.
        let status =
            unsafe { std::ptr::read_unaligned(bytes.as_ptr().cast::<ResidentTerminalStatus>()) };
        let changed = u32::from_ne_bytes(
            bytes[status_bytes..status_bytes + 4]
                .try_into()
                .expect("four bytes checked"),
        );
        let mut counts = Vec::new();
        for chunk in bytes[status_bytes + 4..].chunks_exact(4) {
            counts.push(u32::from_ne_bytes(
                chunk.try_into().expect("four-byte chunk"),
            ));
        }
        finalize_schedule_output_counts(
            self,
            &schedule.relations,
            &schedule.receipt_slots,
            &counts,
        )?;
        counts.truncate(schedule.requested_receipt_count);
        Ok(ResidentScheduleReceipt {
            status,
            changed,
            counts,
        })
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{align_of, offset_of, size_of};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    use cudarc::driver::{CudaStream, LaunchConfig};
    use xlog_core::MemoryBudget;

    use crate::cuda_compat::LaunchAsync;
    use crate::cuda_graph::{CapturedCudaGraph, CudaGraphNodeKind};
    use crate::device::CudaFunction;
    use crate::device_runtime::{
        AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LoggingResource, NullSink,
        StreamPool, XlogDeviceRuntime,
    };
    use crate::memory::GpuMemoryManager;
    use crate::provider::resident_filter_project::{
        ResidentFilterComparison, ResidentFilterOperand, ResidentProjectExpr, ResidentScalar,
    };
    use crate::provider::resident_relational::{
        ResidentJoinKind, ResidentResourceCode, ResidentTerminalCode,
    };
    use crate::provider::CompareOp;
    use crate::{CudaBuffer, CudaColumn, CudaDevice, CudaKernelProvider, DlpackManagedTensor};
    use xlog_core::{ScalarType, Schema, XlogError};

    fn cuda_test_device() -> Option<Arc<CudaDevice>> {
        match CudaDevice::new(0) {
            Ok(device) => Some(Arc::new(device)),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA device initialization failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident schedule CUDA test: {error}");
                None
            }
        }
    }

    fn provider() -> Option<CudaKernelProvider> {
        let device = cuda_test_device()?;
        let memory = Arc::new(GpuMemoryManager::new(
            Arc::clone(&device),
            MemoryBudget::with_limit(512 * 1024 * 1024),
        ));
        match CudaKernelProvider::new(device, memory) {
            Ok(provider) => Some(provider),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but resident schedule setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident schedule CUDA test: {error}");
                None
            }
        }
    }

    fn runtime_provider() -> Option<CudaKernelProvider> {
        let device = cuda_test_device()?;
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let sink = Arc::new(NullSink::new());
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let logging: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(LoggingResource::new(async_resource, sink));
        let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(logging, 512 * 1024 * 1024));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            pool,
            budget,
        ));
        let memory = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(512 * 1024 * 1024),
            runtime,
        ));
        match CudaKernelProvider::with_runtime(device, memory) {
            Ok(provider) => Some(provider),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but resident schedule runtime setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident schedule runtime CUDA test: {error}");
                None
            }
        }
    }

    #[track_caller]
    fn schedule_kernel_error(result: xlog_core::Result<super::ResidentSchedule<'_>>) -> String {
        match result {
            Err(XlogError::Kernel(message)) => message,
            Err(error) => panic!("unexpected resident schedule error: {error}"),
            Ok(_) => panic!("malformed resident schedule unexpectedly prepared"),
        }
    }

    fn schema(prefix: &str, types: &[ScalarType]) -> Schema {
        Schema::new(
            types
                .iter()
                .copied()
                .enumerate()
                .map(|(index, scalar)| (format!("{prefix}_{index}"), scalar))
                .collect(),
        )
    }

    fn buffer(provider: &CudaKernelProvider, schema: Schema, columns: &[Vec<u64>]) -> CudaBuffer {
        assert_eq!(columns.len(), schema.arity());
        let encoded: Vec<Vec<u8>> = columns
            .iter()
            .enumerate()
            .map(|(column, values)| {
                if schema
                    .column_type(column)
                    .expect("column type")
                    .size_bytes()
                    == 4
                {
                    values
                        .iter()
                        .flat_map(|value| (*value as u32).to_le_bytes())
                        .collect()
                } else {
                    values
                        .iter()
                        .flat_map(|value| value.to_le_bytes())
                        .collect()
                }
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
        provider
            .create_buffer_from_slices(&slices, schema)
            .expect("resident schedule test input")
    }

    fn columns_from_rows(rows: &[Vec<u64>]) -> Vec<Vec<u64>> {
        let arity = rows.first().map_or(0, Vec::len);
        (0..arity)
            .map(|column| rows.iter().map(|row| row[column]).collect())
            .collect()
    }

    fn rows_in_device_order(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<Vec<u64>> {
        let count = provider
            .device_row_count(buffer)
            .expect("logical row count");
        let columns: Vec<Vec<u64>> = (0..buffer.arity())
            .map(|column| {
                if buffer
                    .schema()
                    .column_type(column)
                    .expect("column type")
                    .size_bytes()
                    == 4
                {
                    provider
                        .download_column::<u32>(buffer, column)
                        .expect("u32 column")
                        .into_iter()
                        .map(u64::from)
                        .collect()
                } else {
                    provider
                        .download_column::<u64>(buffer, column)
                        .expect("u64 column")
                }
            })
            .collect();
        (0..count)
            .map(|row| columns.iter().map(|column| column[row]).collect())
            .collect()
    }

    fn normalized_rows(provider: &CudaKernelProvider, buffer: &CudaBuffer) -> Vec<Vec<u64>> {
        let mut rows = rows_in_device_order(provider, buffer);
        rows.sort_unstable();
        rows
    }

    fn compact_set_rows(
        provider: &CudaKernelProvider,
        relation_schema: Schema,
        left_columns: &[Vec<u64>],
        right_columns: &[Vec<u64>],
        operation_kind: super::ResidentScheduleOpKind,
        output_capacity: u64,
    ) -> (super::ResidentScheduleReceipt, Vec<Vec<u64>>) {
        let left = buffer(provider, relation_schema.clone(), left_columns);
        let right = buffer(provider, relation_schema.clone(), right_columns);
        let mut output = provider
            .prepare_resident_relation(relation_schema, output_capacity)
            .expect("compact set output")
            .into_buffer();
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("compact set left"),
            super::ResidentScheduleRelation::source(&right, 2).expect("compact set right"),
            super::ResidentScheduleRelation::output(&mut output, 3),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: operation_kind,
            op_id: 980,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 981,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 3,
            ..Default::default()
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[2])
            .expect("prepare compact set schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("compact set stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture compact set schedule");
        graph.launch().expect("launch compact set schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("observe compact set schedule");
        let rows = rows_in_device_order(
            provider,
            graph.relation(2).expect("compact set output relation"),
        );
        (receipt, rows)
    }

    fn compact_nullary_set_count(
        provider: &CudaKernelProvider,
        left_present: bool,
        right_present: bool,
        operation_kind: super::ResidentScheduleOpKind,
    ) -> u32 {
        let relation_schema = Schema::new(Vec::<(String, ScalarType)>::new());
        let mut left_relation = provider
            .prepare_resident_relation(relation_schema.clone(), 1)
            .expect("nullary left");
        provider
            .initialize_resident_relation_count(&mut left_relation, 0)
            .expect("initialize nullary left");
        let mut left = left_relation.into_buffer();
        let mut right_relation = provider
            .prepare_resident_relation(relation_schema.clone(), 1)
            .expect("nullary right");
        provider
            .initialize_resident_relation_count(&mut right_relation, 0)
            .expect("initialize nullary right");
        let mut right = right_relation.into_buffer();
        let mut output = provider
            .prepare_resident_relation(relation_schema, 1)
            .expect("nullary set output")
            .into_buffer();
        left.set_cached_row_count_if_unset(0);
        right.set_cached_row_count_if_unset(0);
        let mut operations = Vec::new();
        if left_present {
            operations.push(super::ResidentOpDescriptor {
                kind: super::OP_UNIT,
                op_id: 984,
                out: 0,
                out_generation: 1,
                ..Default::default()
            });
        }
        if right_present {
            operations.push(super::ResidentOpDescriptor {
                kind: super::OP_UNIT,
                op_id: 985,
                out: 1,
                out_generation: 2,
                ..Default::default()
            });
        }
        operations.push(super::ResidentOpDescriptor {
            kind: operation_kind,
            op_id: 986,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        });
        let relations = vec![
            if left_present {
                super::ResidentScheduleRelation::output(&mut left, 1)
            } else {
                super::ResidentScheduleRelation::source(&left, 1).expect("empty nullary left")
            },
            if right_present {
                super::ResidentScheduleRelation::output(&mut right, 2)
            } else {
                super::ResidentScheduleRelation::source(&right, 2).expect("empty nullary right")
            },
            super::ResidentScheduleRelation::output(&mut output, 3),
        ];
        let wave = super::ResidentWaveDescriptor {
            op_count: operations.len() as u32,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 987,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 3,
            ..Default::default()
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &operations, &[wave], &[region], &[], &[], &[2])
            .expect("prepare nullary compact set schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("nullary compact set stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture nullary compact set schedule");
        graph.launch().expect("launch nullary compact set schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("observe nullary compact set schedule");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::Success as u32,
            "nullary set terminal status: {:?}",
            receipt.status
        );
        receipt.counts[0]
    }

    fn passthrough_schedule<'a>(
        provider: &CudaKernelProvider,
        input: &'a CudaBuffer,
        output: &'a mut CudaBuffer,
        op_id: u32,
    ) -> super::ResidentSchedule<'a> {
        let relations = vec![
            super::ResidentScheduleRelation::source(input, 1).expect("passthrough source"),
            super::ResidentScheduleRelation::output(output, 2),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            op_id,
            out: 1,
            in0: 0,
            in0_generation: 1,
            out_generation: 2,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: op_id + 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 2,
            ..Default::default()
        };
        provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[1])
            .expect("prepare passthrough schedule")
    }

    fn run_single_recursive_diff(
        provider: &CudaKernelProvider,
        left_values: &[u64],
        right_values: &[u64],
        iteration_limit: u32,
    ) -> (super::ResidentScheduleReceipt, Vec<CudaGraphNodeKind>) {
        let relation_schema = schema("single_recursive", &[ScalarType::U32]);
        let left = buffer(provider, relation_schema.clone(), &[left_values.to_vec()]);
        let right = buffer(provider, relation_schema.clone(), &[right_values.to_vec()]);
        let mut novelty = provider
            .prepare_resident_relation(relation_schema, left_values.len().max(1) as u64)
            .expect("single recursive novelty output")
            .into_buffer();
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("left source"),
            super::ResidentScheduleRelation::source(&right, 2).expect("right source"),
            super::ResidentScheduleRelation::output(&mut novelty, 3),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_DIFF,
            flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
            op_id: 601,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        };
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit,
                op_id: 600,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit,
                op_id: 600,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 0,
                iteration_limit: 1,
                op_id: 602,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 0,
            },
        ];
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &regions, &[], &[], &[2])
            .expect("prepare single recursive schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("single recursive stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture single recursive schedule");
        let kinds = graph
            .nodes()
            .expect("single recursive inventory")
            .into_iter()
            .map(|node| node.kind)
            .collect();
        graph.launch().expect("launch single recursive schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("single recursive receipt");
        (receipt, kinds)
    }

    fn run_two_scc_diff(
        provider: &CudaKernelProvider,
        first_limit: u32,
        second_limit: u32,
        first_changes: bool,
        second_changes: bool,
    ) -> (
        super::ResidentScheduleReceipt,
        Vec<CudaGraphNodeKind>,
        [u32; 2],
    ) {
        let relation_schema = schema("serial_recursive", &[ScalarType::U32]);
        let first_left = buffer(provider, relation_schema.clone(), &[vec![11]]);
        let first_right = buffer(
            provider,
            relation_schema.clone(),
            &[if first_changes { Vec::new() } else { vec![11] }],
        );
        let second_left = buffer(provider, relation_schema.clone(), &[vec![22]]);
        let second_right = buffer(
            provider,
            relation_schema.clone(),
            &[if second_changes { Vec::new() } else { vec![22] }],
        );
        let mut first_novelty = buffer(provider, relation_schema.clone(), &[vec![0x1111_1111]]);
        let mut second_novelty = buffer(provider, relation_schema, &[vec![0x2222_2222]]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&first_left, 1).expect("first left"),
            super::ResidentScheduleRelation::source(&first_right, 2).expect("first right"),
            super::ResidentScheduleRelation::source(&second_left, 3).expect("second left"),
            super::ResidentScheduleRelation::source(&second_right, 4).expect("second right"),
            super::ResidentScheduleRelation::output(&mut first_novelty, 5),
            super::ResidentScheduleRelation::output(&mut second_novelty, 6),
        ];
        let operations = [
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
                op_id: 711,
                out: 4,
                in0: 0,
                in1: 1,
                in0_generation: 1,
                in1_generation: 2,
                out_generation: 5,
                ..Default::default()
            },
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
                op_id: 712,
                out: 5,
                in0: 2,
                in1: 3,
                in0_generation: 3,
                in1_generation: 4,
                out_generation: 6,
                ..Default::default()
            },
        ];
        let waves = [
            super::ResidentWaveDescriptor {
                first_op: 0,
                op_count: 1,
                ..Default::default()
            },
            super::ResidentWaveDescriptor {
                first_op: 1,
                op_count: 1,
                ..Default::default()
            },
        ];
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit: first_limit,
                op_id: 701,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit: first_limit,
                op_id: 701,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 0,
                iteration_limit: second_limit,
                op_id: 702,
                flags: super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 1,
                iteration_limit: second_limit,
                op_id: 702,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 2,
                wave_count: 0,
                iteration_limit: 1,
                op_id: 703,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
        ];
        let schedule = provider
            .prepare_resident_schedule(relations, &operations, &waves, &regions, &[], &[], &[4, 5])
            .expect("prepare serial recursive schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("serial recursive stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture serial recursive schedule");
        let kinds = graph
            .nodes()
            .expect("serial recursive inventory")
            .into_iter()
            .map(|node| node.kind)
            .collect();
        graph.launch().expect("launch serial recursive schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("serial recursive receipt");
        let stored = [4_usize, 5].map(|slot| {
            let bytes: Vec<u8> = provider
                .device()
                .inner()
                .dtoh_sync_copy(
                    graph
                        .relation(slot)
                        .expect("novelty relation")
                        .column(0)
                        .expect("novelty column"),
                )
                .expect("novelty storage");
            u32::from_le_bytes(bytes[..4].try_into().expect("u32 storage"))
        });
        (receipt, kinds, stored)
    }

    #[allow(dead_code)]
    unsafe fn selected_stream_cooperative_launch_is_available(
        function: CudaFunction,
        stream: &CudaStream,
        config: LaunchConfig,
        params: &mut Vec<*mut std::ffi::c_void>,
    ) {
        function
            .launch_cooperative_on_stream(stream, config, params)
            .expect("selected-stream cooperative launch");
    }

    #[test]
    fn schedule_wire_abi_has_exact_sizes_alignments_and_offsets() {
        assert_eq!(size_of::<super::ResidentScheduleOpKind>(), 4);
        assert_eq!(align_of::<super::ResidentScheduleOpKind>(), 4);
        assert_eq!(size_of::<super::ResidentRelationView>(), 224);
        assert_eq!(align_of::<super::ResidentRelationView>(), 8);
        assert_eq!(offset_of!(super::ResidentRelationView, widths), 136);
        assert_eq!(offset_of!(super::ResidentRelationView, num_rows), 216);

        assert_eq!(size_of::<super::ResidentRelationSlot>(), 240);
        assert_eq!(align_of::<super::ResidentRelationSlot>(), 16);
        assert_eq!(offset_of!(super::ResidentRelationSlot, generation), 224);
        assert_eq!(offset_of!(super::ResidentRelationSlot, schema_tag), 236);

        assert_eq!(size_of::<super::ResidentOpDescriptor>(), 72);
        assert_eq!(align_of::<super::ResidentOpDescriptor>(), 4);
        assert_eq!(offset_of!(super::ResidentOpDescriptor, aux_offset), 36);
        assert_eq!(
            offset_of!(super::ResidentOpDescriptor, schema_winner_head),
            60
        );
        assert_eq!(
            offset_of!(super::ResidentOpDescriptor, schema_winner_id),
            64
        );
        assert_eq!(offset_of!(super::ResidentOpDescriptor, reserved), 68);

        assert_eq!(size_of::<super::ResidentWaveDescriptor>(), 16);
        assert_eq!(align_of::<super::ResidentWaveDescriptor>(), 4);
        assert_eq!(offset_of!(super::ResidentWaveDescriptor, op_count), 4);

        assert_eq!(size_of::<super::ResidentRegionDescriptor>(), 32);
        assert_eq!(align_of::<super::ResidentRegionDescriptor>(), 4);
        assert_eq!(offset_of!(super::ResidentRegionDescriptor, slot_count), 24);
        assert_eq!(
            offset_of!(super::ResidentRegionDescriptor, generation_offset),
            28
        );

        assert_eq!(size_of::<super::ResidentScheduleHeader>(), 288);
        assert_eq!(align_of::<super::ResidentScheduleHeader>(), 16);
        assert_eq!(offset_of!(super::ResidentScheduleHeader, slots), 0);
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, receipt_bytes),
            208
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, generation_metadata),
            32
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, schema_seen_nonempty),
            184
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, schema_winner_ids),
            192
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, semantic_scan_trace),
            168
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, semantic_filter_trace),
            176
        );
        assert_eq!(offset_of!(super::ResidentScheduleHeader, slot_count), 216);
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, generation_metadata_count),
            276
        );
        assert_eq!(
            offset_of!(super::ResidentScheduleHeader, schema_winner_count),
            264
        );
        assert_eq!(offset_of!(super::ResidentScheduleHeader, abi_version), 280);
        assert_eq!(offset_of!(super::ResidentScheduleHeader, reserved), 284);
        assert_eq!(super::RESIDENT_SCHEDULE_ABI_VERSION, 3);
    }

    #[test]
    fn cuda_schedule_wire_matches_active_host_abi() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "constexpr uint32_t kAbiVersion = 3;",
            "uint32_t schema_winner_head;",
            "uint32_t schema_winner_id;",
            "uint32_t generation_offset;",
            "uint64_t generation_metadata;",
            "uint64_t schema_seen_nonempty;",
            "uint64_t schema_winner_ids;",
            "uint64_t semantic_scan_trace;",
            "uint64_t semantic_filter_trace;",
            "uint32_t generation_metadata_count;",
            "uint32_t abi_version;",
            "static_assert(sizeof(ResidentOpDescriptor) == 72",
            "static_assert(sizeof(ResidentScheduleHeader) == 288",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA ABI fragment: {required}"
            );
        }
    }

    #[test]
    fn flattened_descriptor_tables_have_exact_host_and_cuda_layouts() {
        use std::mem::{align_of, offset_of, size_of};

        assert_eq!(size_of::<super::ResidentFilterComparisonDescriptor>(), 48);
        assert_eq!(align_of::<super::ResidentFilterComparisonDescriptor>(), 8);
        assert_eq!(
            offset_of!(super::ResidentFilterComparisonDescriptor, left_constant),
            32
        );
        assert_eq!(
            offset_of!(super::ResidentFilterComparisonDescriptor, right_constant),
            40
        );
        assert_eq!(size_of::<super::ResidentProjectExpressionDescriptor>(), 24);
        assert_eq!(align_of::<super::ResidentProjectExpressionDescriptor>(), 8);
        assert_eq!(
            offset_of!(super::ResidentProjectExpressionDescriptor, constant),
            16
        );

        let cuda = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "static_assert(sizeof(ResidentFilterComparisonDescriptor) == 48",
            "static_assert(offsetof(ResidentFilterComparisonDescriptor, left_constant) == 32",
            "static_assert(offsetof(ResidentFilterComparisonDescriptor, right_constant) == 40",
            "static_assert(sizeof(ResidentProjectExpressionDescriptor) == 24",
            "static_assert(offsetof(ResidentProjectExpressionDescriptor, constant) == 16",
        ] {
            assert!(
                cuda.contains(required),
                "missing CUDA layout assertion: {required}"
            );
        }
    }

    #[test]
    fn flattened_filter_and_project_descriptors_are_fully_validated() {
        let slot = |widths: &[u32], schema_tag| {
            let mut relation = super::ResidentRelationView::default();
            relation.arity = widths.len() as u32;
            relation.capacity = 4;
            relation.widths[..widths.len()].copy_from_slice(widths);
            super::ResidentRelationSlot {
                relation,
                schema_tag,
                ..Default::default()
            }
        };
        let slots = [slot(&[4, 4], 1), slot(&[4, 4], 1), slot(&[4, 8], 2)];
        let slot_types = [
            vec![ScalarType::Symbol, ScalarType::Symbol],
            vec![ScalarType::Symbol, ScalarType::Symbol],
            vec![ScalarType::Symbol, ScalarType::U64],
        ];
        let filter = super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Filter,
            out: 1,
            in0: 0,
            aux_count: 1,
            ..Default::default()
        };
        let project = super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Project,
            out: 2,
            in0: 0,
            aux_count: 2,
            ..Default::default()
        };
        let comparison = super::ResidentFilterComparisonDescriptor {
            left_kind: 0,
            left_column: 0,
            right_kind: 1,
            op: 0,
            width: 4,
            right_constant: 7,
            ..Default::default()
        };
        let expressions = [
            super::ResidentProjectExpressionDescriptor::column(0, 4),
            super::ResidentProjectExpressionDescriptor::constant(8, 9),
        ];
        super::validate_flattened_filter_project_descriptors(
            &slots,
            &slot_types,
            &[filter, project],
            &[comparison],
            &expressions,
        )
        .expect("valid flattened descriptors");

        let invalid_comparisons = [
            super::ResidentFilterComparisonDescriptor {
                left_kind: 2,
                ..comparison
            },
            super::ResidentFilterComparisonDescriptor {
                left_column: 2,
                ..comparison
            },
            super::ResidentFilterComparisonDescriptor {
                width: 8,
                ..comparison
            },
            super::ResidentFilterComparisonDescriptor {
                op: 6,
                ..comparison
            },
            super::ResidentFilterComparisonDescriptor {
                reserved_zero: 1,
                ..comparison
            },
        ];
        for invalid in invalid_comparisons {
            assert!(super::validate_flattened_filter_project_descriptors(
                &slots,
                &slot_types,
                &[filter],
                &[invalid],
                &[],
            )
            .is_err());
        }

        let mismatched_types = [
            vec![ScalarType::Symbol, ScalarType::U32],
            slot_types[1].clone(),
            slot_types[2].clone(),
        ];
        let two_columns = super::ResidentFilterComparisonDescriptor {
            right_kind: 0,
            right_column: 1,
            ..comparison
        };
        assert!(super::validate_flattened_filter_project_descriptors(
            &slots,
            &mismatched_types,
            &[filter],
            &[two_columns],
            &[],
        )
        .is_err());

        for invalid in [
            super::ResidentProjectExpressionDescriptor {
                kind: 2,
                ..expressions[0]
            },
            super::ResidentProjectExpressionDescriptor {
                column: 2,
                ..expressions[0]
            },
            super::ResidentProjectExpressionDescriptor {
                width: 8,
                ..expressions[0]
            },
            super::ResidentProjectExpressionDescriptor {
                reserved: 1,
                ..expressions[0]
            },
        ] {
            assert!(super::validate_flattened_filter_project_descriptors(
                &slots,
                &slot_types,
                &[project],
                &[],
                &[invalid, expressions[1]],
            )
            .is_err());
        }
        assert!(super::validate_flattened_filter_project_descriptors(
            &slots,
            &slot_types,
            &[super::ResidentOpDescriptor {
                aux_offset: u32::MAX,
                aux_count: 1,
                ..filter
            }],
            &[comparison],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_rejects_writing_alias_and_accepts_scan_identity() {
        let mut relation = super::ResidentRelationView::default();
        relation.arity = 1;
        relation.capacity = 4;
        relation.widths[0] = 4;
        let slots = [super::ResidentRelationSlot {
            relation,
            generation: 7,
            flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
            schema_tag: 1,
            ..Default::default()
        }];
        let types = [vec![ScalarType::U32]];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        }];
        let aliased_filter = [super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Filter,
            op_id: 41,
            out: 0,
            in0: 0,
            in0_generation: 7,
            out_generation: 7,
            ..Default::default()
        }];
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &aliased_filter,
            &waves,
            &regions,
            &[7],
            &[],
            &[],
            &[],
        )
        .is_err());

        let scan = [super::ResidentOpDescriptor::scan(42, 0, 7)];
        super::validate_schedule_program(
            &slots,
            &types,
            &scan,
            &waves,
            &regions,
            &[7],
            &[],
            &[],
            &[],
        )
        .expect("scan is the read-only same-slot exception");
    }

    #[test]
    fn shared_validator_rejects_writes_to_immutable_source_slots() {
        let mut relation = super::ResidentRelationView::default();
        relation.capacity = 1;
        let slots = [super::ResidentRelationSlot {
            relation,
            generation: 3,
            flags: super::RESIDENT_SCHEDULE_SLOT_SOURCE | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
            ..Default::default()
        }];
        let op = [super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Unit,
            out: 0,
            out_generation: 3,
            ..Default::default()
        }];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        }];

        assert!(super::validate_schedule_program(
            &slots,
            &[Vec::new()],
            &op,
            &waves,
            &regions,
            &[3],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_rejects_unknown_relation_slot_flags() {
        let mut relation = super::ResidentRelationView::default();
        relation.arity = 1;
        relation.capacity = 1;
        relation.widths[0] = 4;
        let slots = [super::ResidentRelationSlot {
            relation,
            generation: 3,
            flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT
                | super::RESIDENT_SCHEDULE_SLOT_DEFINED
                | 8,
            ..Default::default()
        }];
        let ops = [super::ResidentOpDescriptor::scan(43, 0, 3)];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        }];

        assert!(super::validate_schedule_program(
            &slots,
            &[vec![ScalarType::U32]],
            &ops,
            &waves,
            &regions,
            &[3],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_simulates_scratch_definedness_and_generation_transitions() {
        let mut relation = super::ResidentRelationView::default();
        relation.capacity = 1;
        let slots = [super::ResidentRelationSlot {
            relation,
            generation: 4,
            ..Default::default()
        }];
        let types = [Vec::new()];
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        };

        let scan_before_definition = [super::ResidentOpDescriptor::scan(51, 0, 4)];
        let one_op_wave = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &scan_before_definition,
            &one_op_wave,
            &[region],
            &[4],
            &[],
            &[],
            &[],
        )
        .is_err());

        let producer_then_scan = [
            super::ResidentOpDescriptor::unit(52, 0, 5),
            super::ResidentOpDescriptor::scan(53, 0, 5),
        ];
        let two_op_wave = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 2,
            ..Default::default()
        }];
        super::validate_schedule_program(
            &slots,
            &types,
            &producer_then_scan,
            &two_op_wave,
            &[region],
            &[4],
            &[],
            &[],
            &[],
        )
        .expect("successful producer defines its next-generation scratch output");

        let skipped_generation = [super::ResidentOpDescriptor::unit(54, 0, 6)];
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &skipped_generation,
            &one_op_wave,
            &[region],
            &[4],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_rejects_physical_slots_outside_their_region_scope() {
        let mut relation = super::ResidentRelationView::default();
        relation.arity = 1;
        relation.capacity = 1;
        relation.widths[0] = 4;
        let slots = [
            super::ResidentRelationSlot {
                relation,
                generation: 1,
                flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT
                    | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                schema_tag: 1,
                ..Default::default()
            },
            super::ResidentRelationSlot {
                relation,
                generation: 2,
                flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT
                    | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                schema_tag: 1,
                ..Default::default()
            },
        ];
        let types = [vec![ScalarType::U32], vec![ScalarType::U32]];
        let ops = [super::ResidentOpDescriptor::scan(61, 1, 2)];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit: 1,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE,
                first_slot: 0,
                slot_count: 2,
                generation_offset: 0,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit: 1,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 1,
                generation_offset: 2,
                ..Default::default()
            },
        ];

        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &ops,
            &waves,
            &regions,
            &[1, 2, 1],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_requires_recursive_novelty_and_accepts_final_delta_copy_marker() {
        let mut relation = super::ResidentRelationView::default();
        relation.arity = 1;
        relation.capacity = 1;
        relation.widths[0] = 4;
        let slots = [
            super::ResidentRelationSlot {
                relation,
                generation: 1,
                flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT
                    | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                schema_tag: 1,
                ..Default::default()
            },
            super::ResidentRelationSlot {
                relation,
                generation: 2,
                flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT
                    | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                schema_tag: 1,
                ..Default::default()
            },
            super::ResidentRelationSlot {
                relation,
                generation: 3,
                schema_tag: 1,
                ..Default::default()
            },
        ];
        let types = [
            vec![ScalarType::U32],
            vec![ScalarType::U32],
            vec![ScalarType::U32],
        ];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit: 5,
                op_id: 70,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit: 5,
                op_id: 70,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 3,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 0,
                iteration_limit: 1,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 3,
                generation_offset: 6,
                ..Default::default()
            },
        ];
        let baselines = [1, 2, 3, 1, 2, 3, 1, 2, 3];
        let diff = super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Diff,
            op_id: 71,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        };
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &[diff],
            &waves,
            &regions,
            &baselines,
            &[],
            &[],
            &[],
        )
        .is_err());

        let marked = super::ResidentOpDescriptor {
            flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY
                | super::RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER,
            schema_winner_head: 0,
            schema_winner_id: 99,
            ..diff
        };
        super::validate_schedule_program(
            &slots,
            &types,
            &[marked],
            &waves,
            &regions,
            &baselines,
            &[],
            &[],
            &[99],
        )
        .expect("recursive Diff may mark novelty and a schema candidate together");

        let delta_copy = super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Project,
            flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
            op_id: 72,
            out: 2,
            in0: 0,
            in0_generation: 1,
            out_generation: 3,
            aux_count: 1,
            ..Default::default()
        };
        super::validate_schedule_program(
            &slots,
            &types,
            &[delta_copy],
            &waves,
            &regions,
            &baselines,
            &[],
            &[super::ResidentProjectExpressionDescriptor::column(0, 4)],
            &[],
        )
        .expect("recursive final delta copy may drive convergence");
    }

    #[test]
    fn shared_validator_exempts_exact_pseudo_ops_from_slot_scope() {
        let status = super::ResidentOpDescriptor::test_status(super::ResidentTerminalStatus {
            code: 5,
            op_id: 81,
            resource_code: 7,
            iterations: 9,
            limit: 11,
            required: 13,
            capacity: 17,
            ..Default::default()
        })
        .unwrap();
        let trace = super::ResidentOpDescriptor::trace_delta(2, 3, None);
        let ops = [status, trace];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 2,
            ..Default::default()
        }];
        let regions = [super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            ..Default::default()
        }];
        super::validate_schedule_program(&[], &[], &ops, &waves, &regions, &[], &[], &[], &[])
            .expect("exact pseudo ops do not reference relation slots");

        let invalid_trace = [super::ResidentOpDescriptor {
            reserved: 1,
            ..trace
        }];
        let one_op_wave = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        assert!(super::validate_schedule_program(
            &[],
            &[],
            &invalid_trace,
            &one_op_wave,
            &regions,
            &[],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_requires_zero_only_unit_and_scan_operands() {
        let slots = [super::ResidentRelationSlot {
            relation: super::ResidentRelationView {
                capacity: 1,
                ..Default::default()
            },
            generation: 3,
            flags: super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
            ..Default::default()
        }];
        let types = [Vec::new()];
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let regions = [super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        }];
        let invalid_unit = [super::ResidentOpDescriptor {
            in0: 9,
            ..super::ResidentOpDescriptor::unit(91, 0, 3)
        }];
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &invalid_unit,
            &waves,
            &regions,
            &[3],
            &[],
            &[],
            &[],
        )
        .is_err());

        let invalid_scan = [super::ResidentOpDescriptor {
            in1: 9,
            ..super::ResidentOpDescriptor::scan(92, 0, 3)
        }];
        assert!(super::validate_schedule_program(
            &slots,
            &types,
            &invalid_scan,
            &waves,
            &regions,
            &[3],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn shared_validator_checks_set_and_join_layouts_and_workspace_envelopes() {
        let slot = |arity: u32, widths: &[u32], capacity: u32, generation: u32, permanent| {
            let mut relation = super::ResidentRelationView {
                arity,
                capacity,
                ..Default::default()
            };
            relation.widths[..widths.len()].copy_from_slice(widths);
            super::ResidentRelationSlot {
                relation,
                generation,
                flags: if permanent {
                    super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED
                } else {
                    0
                },
                schema_tag: if widths == [4] { 1 } else { 2 },
                ..Default::default()
            }
        };
        let waves = [super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        }];
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 3,
            generation_offset: 0,
            ..Default::default()
        };

        let set_slots = [
            slot(1, &[4], 3, 1, true),
            slot(1, &[4], 4, 2, true),
            slot(1, &[4], 7, 3, false),
        ];
        let set_types = [
            vec![ScalarType::U32],
            vec![ScalarType::U32],
            vec![ScalarType::U32],
        ];
        let union = [super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::Union,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        }];
        let requirements = super::validate_schedule_program(
            &set_slots,
            &set_types,
            &union,
            &waves,
            &[region],
            &[1, 2, 3],
            &[],
            &[],
            &[],
        )
        .unwrap();
        assert_eq!(requirements.set_candidate_capacity, 7);

        let join_slots = [
            slot(1, &[4], 5, 1, true),
            slot(1, &[8], 6, 2, true),
            slot(2, &[4, 8], 8, 3, false),
        ];
        let join_types = [
            vec![ScalarType::U32],
            vec![ScalarType::U64],
            vec![ScalarType::U32, ScalarType::U64],
        ];
        let invalid_join = [super::ResidentOpDescriptor {
            kind: super::ResidentScheduleOpKind::JoinInner,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            left_key: 0,
            right_key: 0,
            ..Default::default()
        }];
        assert!(super::validate_schedule_program(
            &join_slots,
            &join_types,
            &invalid_join,
            &waves,
            &[region],
            &[1, 2, 3],
            &[],
            &[],
            &[],
        )
        .is_err());
    }

    #[test]
    fn cuda_mirrors_flattened_descriptor_validation_with_subtract_bounds() {
        let cuda = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "op.aux_offset <= header->filter_comparison_count",
            "op.aux_count <= header->filter_comparison_count - op.aux_offset",
            "comparison.left_kind > 1",
            "comparison.right_kind > 1",
            "comparison.op > 5",
            "comparison.reserved_zero != 0",
            "comparison.reserved_one != 0",
            "comparison.left_column >= in0.arity",
            "comparison.width != in0.widths[comparison.left_column]",
            "op.aux_offset <= header->project_expression_count",
            "op.aux_count <= header->project_expression_count - op.aux_offset",
            "expression.kind > 1",
            "expression.reserved != 0",
            "expression.column >= in0.arity",
            "expression.width != out.widths[column]",
        ] {
            assert!(
                cuda.contains(required),
                "missing device validation: {required}"
            );
        }
    }

    #[test]
    fn generation_baseline_ranges_reject_overflow_and_out_of_bounds() {
        let valid = [
            super::ResidentRegionDescriptor {
                slot_count: 2,
                generation_offset: 0,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                slot_count: 3,
                generation_offset: 2,
                ..Default::default()
            },
        ];
        super::validate_generation_baseline_ranges(&valid, 5)
            .expect("concatenated generation baselines");

        let out_of_bounds = [super::ResidentRegionDescriptor {
            slot_count: 6,
            generation_offset: 0,
            ..Default::default()
        }];
        assert_eq!(
            super::validate_generation_baseline_ranges(&out_of_bounds, 5)
                .expect_err("range must exceed generation table")
                .to_string(),
            "Kernel error: resident schedule generation baseline range is invalid"
        );

        let overflow = [
            super::ResidentRegionDescriptor {
                slot_count: u32::MAX,
                generation_offset: 0,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                slot_count: 1,
                generation_offset: u32::MAX,
                ..Default::default()
            },
        ];
        assert_eq!(
            super::validate_generation_baseline_ranges(&overflow, u32::MAX)
                .expect_err("range arithmetic must be checked")
                .to_string(),
            "Kernel error: resident schedule generation baseline range overflow"
        );
    }

    #[test]
    fn generation_baselines_exactly_concatenate_region_scopes() {
        let first = super::ResidentRegionDescriptor {
            slot_count: 1,
            generation_offset: 0,
            ..Default::default()
        };
        let second = super::ResidentRegionDescriptor {
            slot_count: 1,
            generation_offset: 1,
            ..Default::default()
        };
        super::validate_generation_baseline_ranges(&[first, second], 2)
            .expect("exact concatenation");

        assert!(super::validate_generation_baseline_ranges(
            &[
                first,
                super::ResidentRegionDescriptor {
                    generation_offset: 2,
                    ..second
                },
            ],
            3,
        )
        .is_err());
        assert!(super::validate_generation_baseline_ranges(
            &[
                first,
                super::ResidentRegionDescriptor {
                    generation_offset: 0,
                    ..second
                },
            ],
            2,
        )
        .is_err());
        assert!(super::validate_generation_baseline_ranges(&[first, second], 3).is_err());
    }

    #[test]
    fn generation_metadata_appends_schema_defaults_and_derives_baseline_count() {
        let metadata =
            super::build_generation_metadata(&[4, 7, 9], &[101, 202]).expect("generation metadata");
        assert_eq!(metadata, vec![4, 7, 9, 101, 202]);
        assert_eq!(
            super::generation_baseline_count_from_metadata(5, 2)
                .expect("three generation baselines"),
            3
        );
        assert_eq!(
            super::generation_baseline_count_from_metadata(3, 0).expect("no schema-default tail"),
            3
        );
        assert!(super::generation_baseline_count_from_metadata(1, 2).is_err());
    }

    #[test]
    fn schema_winner_replay_reset_restores_defaults_before_ordered_marks() {
        let defaults = [10, 20];
        let mut seen = [1, 1];
        let mut winners = [99, 98];
        super::reset_schema_winner_state(&defaults, &mut seen, &mut winners)
            .expect("reset replay state");
        assert_eq!(seen, [0, 0]);
        assert_eq!(winners, defaults);

        super::mark_schema_winner_model(3, 10, &mut seen[0], &mut winners[0]);
        super::mark_schema_winner_model(2, 30, &mut seen[0], &mut winners[0]);
        assert_eq!(winners[0], 10, "existing nonempty head retains its default");

        super::reset_schema_winner_state(&defaults, &mut seen, &mut winners)
            .expect("reset second replay");
        super::mark_schema_winner_model(0, 20, &mut seen[1], &mut winners[1]);
        super::mark_schema_winner_model(2, 40, &mut seen[1], &mut winners[1]);
        assert_eq!(
            winners[1], 40,
            "empty head accepts its first later contribution"
        );
    }

    #[test]
    fn generation_baselines_are_concatenated_in_region_scope_order() {
        let mut regions = [
            super::ResidentRegionDescriptor {
                first_slot: 0,
                slot_count: 2,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                first_slot: 1,
                slot_count: 2,
                ..Default::default()
            },
        ];
        let baselines = super::build_generation_baselines(&mut regions, &[11, 22, 33])
            .expect("generation baseline table");
        assert_eq!(baselines, vec![11, 22, 22, 33]);
        assert_eq!(regions[0].generation_offset, 0);
        assert_eq!(regions[1].generation_offset, 2);

        let mut invalid = [super::ResidentRegionDescriptor {
            first_slot: 2,
            slot_count: 2,
            ..Default::default()
        }];
        assert_eq!(
            super::build_generation_baselines(&mut invalid, &[11, 22, 33])
                .expect_err("slot scope must be bounded")
                .to_string(),
            "Kernel error: resident schedule generation baseline slot scope is invalid"
        );
    }

    #[test]
    fn cuda_checks_generation_baseline_range_before_resetting_slots() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        let guard = source
            .find("const bool generation_range_valid")
            .expect("device generation range guard");
        let read = source
            .find("generation_metadata[region.generation_offset + index]")
            .expect("device generation baseline read");
        assert!(
            guard < read,
            "generation baseline read must follow its bounds guard"
        );
        assert!(source
            .contains("region.slot_count <= generation_base_count - region.generation_offset"));
    }

    #[test]
    fn schema_winner_encoding_uses_exact_receipt_shape_and_head_bounds() {
        assert_eq!(
            super::checked_schedule_head_count(8, 76).expect("two heads"),
            2
        );
        assert_eq!(
            super::checked_schedule_winner_count(8, 76, 2).expect("matching winner state"),
            2
        );
        assert!(super::checked_schedule_winner_count(8, 76, 1).is_err());
        for (count, bytes) in [(1, 48), (5, 64), (6, 64)] {
            assert!(super::checked_schedule_head_count(count, bytes).is_err());
        }

        let marked = super::ResidentOpDescriptor::default().with_schema_winner(1, 77);
        assert_eq!(marked.flags, super::RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER);
        super::validate_schema_winner_encoding(&marked, 2).expect("bounded winner mark");

        let out_of_bounds = marked.with_schema_winner(2, 88);
        assert!(super::validate_schema_winner_encoding(&out_of_bounds, 2).is_err());

        let unmarked_payload = super::ResidentOpDescriptor {
            schema_winner_id: 77,
            ..Default::default()
        };
        assert!(super::validate_schema_winner_encoding(&unmarked_payload, 2).is_err());
    }

    #[test]
    fn cuda_schema_winner_mark_is_count_gated_and_sticky() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "const bool receipt_shape_valid",
            "*head_count = (header->receipt_count - 4) / 2",
            "header->schema_winner_count != *head_count",
            "op.schema_winner_head >= head_count",
            "*device_ptr<const uint32_t>(output.num_rows) != 0",
            "atomicCAS(&schema_seen_nonempty[op.schema_winner_head], 0U, 1U)",
            "schema_winner_ids[op.schema_winner_head] = op.schema_winner_id",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA winner fragment: {required}"
            );
        }
        let count_gate = source
            .find("*device_ptr<const uint32_t>(output.num_rows) != 0")
            .expect("winner count gate");
        let compare_exchange = source
            .find("atomicCAS(&schema_seen_nonempty[op.schema_winner_head], 0U, 1U)")
            .expect("sticky winner compare-exchange");
        assert!(count_gate < compare_exchange);
        let receipt_guard = source
            .find("if (!receipt_shape_valid) return false;")
            .unwrap();
        let head_derivation = source
            .find("*head_count = (header->receipt_count - 4) / 2")
            .unwrap();
        assert!(receipt_guard < head_derivation);
    }

    #[test]
    fn cuda_initialize_resets_schema_winners_from_metadata_tail_before_waves() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "header->generation_metadata_count < *head_count",
            "const uint32_t generation_base_count = generation_metadata_shape_valid",
            "? header->generation_metadata_count - head_count : 0;",
            "schema_seen_nonempty[head] = 0U;",
            "schema_winner_ids[head] =",
            "generation_metadata[generation_base_count + head];",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA replay-reset fragment: {required}"
            );
        }
        let reset = source
            .find("schema_seen_nonempty[head] = 0U;")
            .expect("schema seen reset");
        let waves = source
            .find("for (uint32_t wave_offset = 0; wave_offset < safe_wave_count; ++wave_offset)")
            .expect("wave loop");
        assert!(
            reset < waves,
            "schema winners reset before the first operation wave"
        );
    }

    #[test]
    fn cuda_validator_enforces_region_scope_and_scan_only_same_slot_aliasing() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "const ResidentRegionDescriptor &region,",
            "const uint32_t region_slot_end = region.first_slot + region.slot_count;",
            "op.kind != kOpScan && (op.out == op.in0 ||",
            "op.out < region.first_slot || op.out >= region_slot_end",
            "op.in0 != 0 || op.in1 != 0",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA scope or alias fragment: {required}"
            );
        }
    }

    #[test]
    fn cuda_validator_mirrors_physical_payload_and_workspace_checks() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "op.reserved != 0 ||",
            "output.relation.capacity > 65536 ||",
            "op.in1 != 0 || op.in1_generation != 0 ||",
            "op.left_key != 0 || op.right_key != 0",
            "static_cast<uint64_t>(in0.capacity) + in1.capacity >",
            "header->set_candidate_capacity",
            "expected_arity > kMaxArity",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA physical-envelope fragment: {required}"
            );
        }
    }

    #[test]
    fn cuda_region_reset_rejects_unknown_or_conflicting_slot_flags() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "slot.flags & ~(kSourceSlot | kPermanentSlot | kDefinedSlot)",
            "(slot.flags & kSourceSlot) != 0 &&",
            "(slot.flags & kPermanentSlot) != 0",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA slot-flag validation: {required}"
            );
        }
    }

    #[test]
    fn test_status_descriptor_round_trips_the_full_terminal_payload() {
        let status = super::ResidentTerminalStatus {
            code: 3,
            op_id: 0x1020_3040,
            resource_code: 0x5060_7080,
            iterations: 0x90a0_b0c0,
            limit: 0xd0e0_f001,
            reserved: 0,
            required: 0x1122_3344_5566_7788,
            capacity: 0x99aa_bbcc_ddee_ff00,
        };
        let descriptor =
            super::ResidentOpDescriptor::test_status(status).expect("test status descriptor");
        assert_eq!(descriptor.kind, super::ResidentScheduleOpKind::TestStatus);
        assert_eq!(super::decode_test_status(&descriptor).unwrap(), status);

        let invalid_unused = super::ResidentOpDescriptor {
            right_key: 1,
            ..descriptor
        };
        assert!(super::decode_test_status(&invalid_unused).is_err());

        let invalid_reserved = super::ResidentTerminalStatus {
            reserved: 1,
            ..status
        };
        assert!(super::ResidentOpDescriptor::test_status(invalid_reserved).is_err());
    }

    #[test]
    fn cuda_test_status_descriptor_publishes_both_u64_fields_without_auxiliary_storage() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "constexpr uint32_t kOpTestStatus = 8",
            "op.kind == kOpTestStatus",
            "static_cast<uint64_t>(op.out_generation) |",
            "(static_cast<uint64_t>(op.aux_offset) << 32)",
            "static_cast<uint64_t>(op.aux_count) |",
            "(static_cast<uint64_t>(op.left_key) << 32)",
            "status->iterations = op.in1",
            "status->limit = op.in0_generation",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA TestStatus fragment: {required}"
            );
        }
    }

    #[test]
    fn trace_delta_descriptor_carries_an_optional_semantic_guard() {
        let descriptor = super::ResidentOpDescriptor::trace_delta(2, 3, None);
        assert_eq!(descriptor.kind, super::ResidentScheduleOpKind::TraceDelta);
        assert_eq!(
            super::decode_trace_delta(&descriptor).unwrap(),
            (2, 3, None)
        );

        let guarded = super::ResidentOpDescriptor::trace_delta(5, 7, Some((11, 13)));
        assert_eq!(guarded.flags, super::RESIDENT_SCHEDULE_TRACE_SEMANTIC_GUARD);
        assert_eq!(guarded.in0, 11);
        assert_eq!(guarded.in0_generation, 13);
        assert_eq!(
            super::decode_trace_delta(&guarded).unwrap(),
            (5, 7, Some((11, 13)))
        );

        let invalid = super::ResidentOpDescriptor {
            op_id: 1,
            ..guarded
        };
        assert!(super::decode_trace_delta(&invalid).is_err());
    }

    #[test]
    fn cuda_trace_delta_executes_after_an_earlier_terminal_status() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "constexpr uint32_t kOpTraceDelta = 9",
            "constexpr uint32_t kOpTraceSemanticGuard = 1",
            "if (op.kind == kOpTraceDelta) {",
            "atomicAdd(device_ptr<uint32_t>(header->scan_trace), op.scan_delta)",
            "atomicAdd(device_ptr<uint32_t>(header->filter_trace), op.filter_delta)",
            "const bool semantic_active =",
            "*device_ptr<const uint32_t>(input_zero.num_rows) != 0",
            "header->semantic_scan_trace",
            "header->semantic_filter_trace",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA TraceDelta fragment: {required}"
            );
        }
        assert!(!source.contains("status->code == kRunning && op.scan_delta"));
        assert!(!source.contains("status->code == kRunning && op.filter_delta"));
    }

    #[test]
    fn cuda_set_ordering_compacts_in_parallel_and_uses_bounded_merge_passes() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "compact_set_winners_by_tile",
            "sort_set_reference_tiles",
            "merge_set_reference_runs",
            "kSetReferenceTileSize = 1024",
            "set_workspace_shape_valid",
            "set_slot_count >= 2ULL * header->set_candidate_capacity",
        ] {
            assert!(
                source.contains(required),
                "missing bounded set-ordering fragment: {required}"
            );
        }
        assert!(
            !source.contains("for (uint32_t slot = 0; slot <= header->set_slot_mask; ++slot)"),
            "set winners must not be compacted by one thread"
        );
        assert!(
            !source.contains("for (uint32_t width = 2; width <= sort_size; width <<= 1)"),
            "set ordering must not use a grid-wide bitonic network"
        );
    }

    #[test]
    fn cuda_packs_the_single_receipt_only_in_the_final_region() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        let guard = source
            .find("if (finalizes && global_rank() == 0)")
            .expect("final-region receipt guard");
        let pack = source
            .find("*reinterpret_cast<ResidentTerminalStatus *>(receipt) = *status")
            .expect("terminal receipt pack");
        assert!(guard < pack);
        assert_eq!(
            source
                .matches("*reinterpret_cast<ResidentTerminalStatus *>(receipt) = *status")
                .count(),
            1,
            "there must be one receipt pack path"
        );
    }

    #[test]
    fn slot_definedness_distinguishes_scratch_permanent_and_failed_writes() {
        let scratch = super::reset_slot_flags(super::RESIDENT_SCHEDULE_SLOT_DEFINED);
        assert_eq!(scratch, 0);
        assert!(!super::slot_input_is_ready(scratch, 7, 7));

        let permanent = super::reset_slot_flags(super::RESIDENT_SCHEDULE_SLOT_PERMANENT);
        assert_eq!(
            permanent,
            super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED
        );
        assert!(super::slot_input_is_ready(permanent, 7, 7));

        let source = super::reset_slot_flags(super::RESIDENT_SCHEDULE_SLOT_SOURCE);
        assert!(super::slot_input_is_ready(source, 7, 7));
        assert!(!super::slot_output_generation_is_valid(source, 7, 7));

        assert!(super::slot_output_generation_is_valid(scratch, 7, 7));
        assert!(super::slot_output_generation_is_valid(scratch, 7, 8));
        assert!(!super::slot_output_generation_is_valid(scratch, 7, 9));
        assert!(!super::slot_output_generation_is_valid(
            scratch,
            u32::MAX,
            0
        ));

        assert_eq!(super::finish_slot_write(scratch, false), scratch);
        assert_eq!(
            super::finish_slot_write(scratch, true),
            super::RESIDENT_SCHEDULE_SLOT_DEFINED
        );
    }

    #[test]
    fn cuda_slot_definedness_is_reset_checked_and_set_after_success() {
        let source = include_str!("../../kernels/resident_schedule.cu");
        for required in [
            "constexpr uint32_t kPermanentSlot = 2",
            "constexpr uint32_t kDefinedSlot = 4",
            "slot.flags |= kDefinedSlot",
            "slot.flags &= ~kDefinedSlot",
            "(input.flags & kDefinedSlot) != 0",
            "slots[op.out].flags |= kDefinedSlot",
            "slots[op.out].generation = op.out_generation",
        ] {
            assert!(
                source.contains(required),
                "missing CUDA definedness fragment: {required}"
            );
        }
        let execution = source.find("execute_filter(grid").unwrap();
        let define = source.find("slots[op.out].flags |= kDefinedSlot").unwrap();
        assert!(
            execution < define,
            "output must become defined only after execution"
        );
    }

    #[test]
    fn schedule_metadata_manifest_uses_exact_flattened_table_bytes() {
        assert_eq!(
            super::resident_schedule_metadata_device_bytes(0, 0, 0, 0, 0, 0, 0).unwrap(),
            724
        );
        assert_eq!(
            super::resident_schedule_metadata_device_bytes(2, 3, 4, 5, 6, 0, 2).unwrap(),
            1_328
        );
        let generation_count = 6;
        let schema_default_count = 2;
        assert_eq!(
            super::resident_schedule_metadata_device_bytes(
                2,
                3,
                4,
                5,
                generation_count + schema_default_count,
                0,
                2,
            )
            .unwrap(),
            1_336
        );
        assert!(
            super::resident_schedule_metadata_device_bytes(usize::MAX, 1, 1, 1, 1, 1, 1).is_err()
        );
    }

    #[test]
    fn additive_schedule_api_records_one_region_into_the_existing_stream() {
        let _record: unsafe fn(
            &super::CudaKernelProvider,
            &super::ResidentScheduleDeviceProgram,
            u32,
            Option<&crate::cuda_graph::ConditionalCudaGraphBody>,
            &CudaStream,
        ) -> xlog_core::Result<()> =
            super::CudaKernelProvider::record_resident_schedule_region_on_stream;
    }

    #[test]
    fn additive_schedule_unsafe_contract_covers_owners_recorder_and_graph_identity() {
        let source = include_str!("resident_schedule.rs");
        let start = source
            .find("/// Record one compact scheduler region into a graph owned by the caller.")
            .expect("additive record docs");
        let end = source[start..]
            .find("pub unsafe fn record_resident_schedule_region_on_stream")
            .map(|offset| start + offset)
            .expect("additive record signature");
        let safety = source[start..end]
            .lines()
            .map(|line| line.trim_start().trim_start_matches("///").trim())
            .collect::<Vec<_>>()
            .join(" ");
        for required in [
            "register the program, every slot and external owner, and every indirect receipt pointee",
            "before domain-bound preflight",
            "through graph destruction and completion of all in-flight work",
            "domain-bound preflight and domain-bound commit",
            "conditional body passed here must be the one minted for the enclosing graph",
        ] {
            assert!(
                safety.contains(required),
                "missing additive safety obligation: {required}"
            );
        }
    }

    #[test]
    fn additive_schedule_uses_one_sealed_execution_domain_and_bound_recorder() {
        let _bind: fn(
            &super::CudaKernelProvider,
            Arc<XlogDeviceRuntime>,
            crate::device_runtime::StreamId,
            Arc<CudaStream>,
        ) -> xlog_core::Result<super::ResidentExecutionDomain> =
            super::CudaKernelProvider::bind_resident_execution_domain;
        let _recorder: fn(&super::ResidentExecutionDomain) -> crate::launch::LaunchRecorder =
            super::ResidentExecutionDomain::new_strict_recorder;
        let _preflight: fn(
            &super::ResidentExecutionDomain,
            &mut crate::launch::LaunchRecorder,
        ) -> xlog_core::Result<()> = super::ResidentExecutionDomain::preflight;
        let _commit: fn(
            &super::ResidentExecutionDomain,
            crate::launch::LaunchRecorder,
        ) -> xlog_core::Result<()> = super::ResidentExecutionDomain::commit;
    }

    #[test]
    fn allocation_owner_validation_uses_checked_live_block_ranges() {
        let block = crate::device_runtime::BlockId {
            ptr: 0x1000,
            generation: crate::device_runtime::Generation(1),
            alloc_stream: crate::device_runtime::StreamId(2),
            device_ordinal: 3,
        };
        assert_eq!(
            super::validate_runtime_allocation_fields(
                7,
                0x1004,
                8,
                block,
                32,
                crate::device_runtime::BlockState::Live,
                7,
                3,
            )
            .unwrap(),
            (0x1004, 0x100c)
        );
        assert!(super::validate_runtime_allocation_fields(
            8,
            0x1004,
            8,
            block,
            32,
            crate::device_runtime::BlockState::Live,
            7,
            3,
        )
        .is_err());
        assert!(super::validate_runtime_allocation_fields(
            7,
            u64::MAX - 1,
            4,
            block,
            32,
            crate::device_runtime::BlockState::Live,
            7,
            3,
        )
        .is_err());
        assert!(super::validate_runtime_allocation_fields(
            7,
            0x1004,
            8,
            block,
            32,
            crate::device_runtime::BlockState::Retired,
            7,
            3,
        )
        .is_err());
    }

    #[test]
    fn allocation_inventory_rejects_partial_aliases_but_allows_adjacency() {
        let mut ranges = Vec::new();
        super::insert_nonoverlapping_allocation_range(&mut ranges, (0x1000, 0x1010))
            .expect("first allocation");
        super::insert_nonoverlapping_allocation_range(&mut ranges, (0x1010, 0x1020))
            .expect("adjacent allocation");
        assert!(
            super::insert_nonoverlapping_allocation_range(&mut ranges, (0x1008, 0x1018)).is_err()
        );
    }

    #[test]
    fn receipt_slot_mapping_requires_exact_unique_permanent_targets() {
        let permanent =
            super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED;
        let source = super::RESIDENT_SCHEDULE_SLOT_SOURCE | super::RESIDENT_SCHEDULE_SLOT_DEFINED;
        let flags = [source, permanent, permanent, 0];

        assert_eq!(
            super::validate_receipt_slot_mapping(&[], &flags, 0).unwrap(),
            Vec::<usize>::new()
        );
        assert_eq!(
            super::validate_receipt_slot_mapping(&[2], &flags, 1).unwrap(),
            vec![2]
        );
        assert_eq!(
            super::validate_receipt_slot_mapping(&[2, 1], &flags, 2).unwrap(),
            vec![2, 1]
        );
        assert!(super::validate_receipt_slot_mapping(&[1], &flags, 2).is_err());
        assert!(super::validate_receipt_slot_mapping(&[1, 1], &flags, 2).is_err());
        assert!(super::validate_receipt_slot_mapping(&[0], &flags, 1).is_err());
        assert!(super::validate_receipt_slot_mapping(&[3], &flags, 1).is_err());
        assert!(super::validate_receipt_slot_mapping(&[4], &flags, 1).is_err());
    }

    #[test]
    fn device_program_construction_consumes_the_runtime_reservation_and_external_bindings() {
        let _prepare: for<'a> fn(
            &super::CudaKernelProvider,
            &super::ResidentExecutionDomain,
            &[super::ResidentScheduleSlotBinding<'a>],
            &[super::ResidentOpDescriptor],
            &[super::ResidentWaveDescriptor],
            &[super::ResidentRegionDescriptor],
            &[u32],
            &[super::ResidentFilterComparisonDescriptor],
            &[super::ResidentProjectExpressionDescriptor],
            &[u32],
            super::ResidentScheduleExternalBindings<'a>,
            &mut crate::memory::GpuMemoryReservation,
        )
            -> xlog_core::Result<super::ResidentScheduleDeviceProgram> =
            super::CudaKernelProvider::prepare_resident_schedule_program_in_reservation;
    }

    #[test]
    fn additive_schedule_metadata_and_external_owners_have_strict_recorder_apis() {
        fn record_slot(
            slot: &super::ResidentScheduleSlotBinding<'_>,
            recorder: &mut crate::launch::LaunchRecorder,
        ) {
            slot.record_uses(recorder)
        }
        fn record_external(
            external: &super::ResidentScheduleExternalBindings<'_>,
            recorder: &mut crate::launch::LaunchRecorder,
        ) {
            external.record_uses(recorder)
        }
        let _program: fn(
            &super::ResidentScheduleDeviceProgram,
            &mut crate::launch::LaunchRecorder,
        ) = super::ResidentScheduleDeviceProgram::record_uses;
        let _slot = record_slot;
        let _external = record_external;
    }

    #[test]
    fn slot_reset_preserves_source_and_permanent_counts_and_clears_scratch() {
        let source = super::reset_slot_state_for_region(super::RESIDENT_SCHEDULE_SLOT_SOURCE, 9, 4);
        assert_eq!(
            source,
            (
                super::RESIDENT_SCHEDULE_SLOT_SOURCE | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                9,
                4,
            )
        );
        let permanent =
            super::reset_slot_state_for_region(super::RESIDENT_SCHEDULE_SLOT_PERMANENT, 7, 3);
        assert_eq!(
            permanent,
            (
                super::RESIDENT_SCHEDULE_SLOT_PERMANENT | super::RESIDENT_SCHEDULE_SLOT_DEFINED,
                7,
                3,
            )
        );
        assert_eq!(super::reset_slot_state_for_region(0, 11, 8), (0, 11, 0));
    }

    #[test]
    fn cuda_reset_and_recorder_keep_source_count_read_only() {
        let cuda = include_str!("../../kernels/resident_schedule.cu");
        let reset_start = cuda
            .find("for (uint32_t index = 0; index < region.slot_count; ++index)")
            .expect("slot reset loop");
        let reset_end = cuda[reset_start..]
            .find("if (status->code == kRunning && recursive)")
            .map(|offset| reset_start + offset)
            .expect("slot reset end");
        let reset = &cuda[reset_start..reset_end];
        assert!(!reset.contains("slot.initial_count"));
        assert_eq!(
            reset
                .matches("*device_ptr<uint32_t>(slot.relation.num_rows) = 0")
                .count(),
            1,
            "only the scratch branch may reset a count word"
        );

        let rust = include_str!("resident_schedule.rs");
        let source_arm = rust
            .find("Self::Source { buffer, .. } => {")
            .expect("source recorder arm");
        let resident_arm = rust[source_arm..]
            .find("Self::Resident { buffer, .. } => {")
            .map(|offset| source_arm + offset)
            .expect("resident recorder arm");
        let source_recorder = &rust[source_arm..resident_arm];
        assert!(source_recorder.contains("recorder.read(buffer.num_rows_device())"));
        assert!(!source_recorder.contains("read_write(buffer.num_rows_device())"));
    }

    #[test]
    fn operation_kind_api_expresses_typed_unit_and_scan_leaves() {
        let unit = super::ResidentOpDescriptor::unit(101, 3, 7);
        assert_eq!(unit.kind, super::ResidentScheduleOpKind::Unit);
        assert_eq!(unit.op_id, 101);
        assert_eq!(unit.out, 3);
        assert_eq!(unit.out_generation, 7);

        let scan = super::ResidentOpDescriptor::scan(102, 4, 9);
        assert_eq!(scan.kind, super::ResidentScheduleOpKind::Scan);
        assert_eq!(scan.op_id, 102);
        assert_eq!(scan.in0, 4);
        assert_eq!(scan.in0_generation, 9);
        assert_eq!(scan.out, 4);
        assert_eq!(scan.out_generation, 9);
    }

    #[test]
    fn initialization_scope_must_cover_every_relation_slot() {
        let full = super::ResidentRegionDescriptor {
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE,
            first_slot: 0,
            slot_count: 3,
            ..Default::default()
        };
        super::validate_initialization_scope(&full, 3).expect("full initialization scope");

        let partial = super::ResidentRegionDescriptor {
            slot_count: 2,
            ..full
        };
        assert_eq!(
            super::validate_initialization_scope(&partial, 3)
                .expect_err("partial initialization scope")
                .to_string(),
            "Kernel error: resident schedule initialization must cover every relation slot"
        );
    }

    #[test]
    fn region_control_placement_is_exact_before_materialization() {
        let region = |flags, first_wave, iteration_limit| super::ResidentRegionDescriptor {
            first_wave,
            wave_count: 1,
            iteration_limit,
            op_id: first_wave,
            flags,
            first_slot: 0,
            slot_count: 2,
            generation_offset: first_wave * 2,
        };

        assert!(super::validate_region_control_and_ranges(
            &[region(
                super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                0,
                1,
            )],
            1,
            2,
        )
        .is_ok());

        let mut valid = [
            region(
                super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                0,
                5,
            ),
            region(super::RESIDENT_SCHEDULE_REGION_RECURSIVE, 1, 5),
            region(super::RESIDENT_SCHEDULE_REGION_FINALIZE, 2, 1),
        ];
        valid[1].op_id = valid[0].op_id;
        assert!(super::validate_region_control_and_ranges(&valid, 3, 2).is_ok());

        for invalid in [
            vec![region(super::RESIDENT_SCHEDULE_REGION_FINALIZE, 0, 1)],
            vec![region(super::RESIDENT_SCHEDULE_REGION_INITIALIZE, 0, 1)],
            vec![
                region(super::RESIDENT_SCHEDULE_REGION_INITIALIZE, 0, 1),
                region(
                    super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                        | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                    1,
                    1,
                ),
            ],
            vec![region(
                super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_RECURSIVE
                    | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                0,
                1,
            )],
            vec![
                region(
                    super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                        | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                    0,
                    5,
                ),
                region(
                    super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN
                        | super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                    1,
                    5,
                ),
                region(super::RESIDENT_SCHEDULE_REGION_FINALIZE, 2, 1),
            ],
            vec![region(
                super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                0,
                2,
            )],
        ] {
            assert!(
                super::validate_region_control_and_ranges(&invalid, invalid.len() as u32, 2)
                    .is_err()
            );
        }
    }

    #[test]
    fn waves_exactly_partition_operations_before_materialization() {
        let wave = |first_op, op_count| super::ResidentWaveDescriptor {
            first_op,
            op_count,
            flags: 0,
            reserved: 0,
        };
        assert!(super::validate_wave_partition(&[], 0).is_ok());
        assert!(super::validate_wave_partition(&[wave(0, 2), wave(2, 1)], 3).is_ok());
        for invalid in [
            vec![wave(1, 2)],
            vec![wave(0, 1), wave(2, 1)],
            vec![wave(0, 2), wave(1, 1)],
            vec![wave(0, 1)],
        ] {
            assert!(super::validate_wave_partition(&invalid, 3).is_err());
        }
        assert!(super::validate_wave_partition(
            &[super::ResidentWaveDescriptor {
                flags: 1,
                ..wave(0, 1)
            }],
            1,
        )
        .is_err());
        assert!(super::validate_wave_partition(
            &[super::ResidentWaveDescriptor {
                reserved: 1,
                ..wave(0, 1)
            }],
            1,
        )
        .is_err());
    }

    #[test]
    fn scalar_envelope_is_symbol_u32_and_u64_only() {
        assert_eq!(
            super::resident_schedule_scalar_width(ScalarType::Symbol).expect("symbol"),
            4
        );
        assert_eq!(
            super::resident_schedule_scalar_width(ScalarType::U32).expect("u32"),
            4
        );
        assert_eq!(
            super::resident_schedule_scalar_width(ScalarType::U64).expect("u64"),
            8
        );
        for unsupported in [
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::Bool,
        ] {
            assert!(
                super::resident_schedule_scalar_width(unsupported).is_err(),
                "{unsupported:?} reached the unsigned scheduler kernel"
            );
        }
    }

    #[test]
    fn device_storage_ranges_reject_partial_overlap() {
        assert!(super::device_ranges_overlap(0x1000, 16, 0x1008, 16));
        assert!(super::device_ranges_overlap(0x1008, 16, 0x1000, 16));
        assert!(!super::device_ranges_overlap(0x1000, 8, 0x1008, 8));
        assert!(!super::device_ranges_overlap(0x1000, 0, 0x1000, 8));
    }

    #[test]
    fn preparation_rejects_filter_column_outside_input_schema() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("value", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![7]]);
        let mut output = buffer(&provider, relation_schema, &[vec![0]]);
        assert_eq!(output.cached_row_count(), Some(1));
        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("source relation"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let operations = [super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            op_id: 401,
            out: 1,
            in0: 0,
            in1: 0,
            in0_generation: 1,
            in1_generation: 0,
            out_generation: 2,
            aux_count: 1,
            ..Default::default()
        }];
        let comparisons = [super::ResidentFilterComparisonDescriptor::column_constant(
            1, 0, 4, 7,
        )];
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 400,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 2,
            ..Default::default()
        };

        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &operations,
                &[wave],
                &[region],
                &comparisons,
                &[],
                &[],
            )),
            "resident schedule descriptor column is invalid"
        );
        assert_eq!(output.cached_row_count(), Some(1));
    }

    #[test]
    fn preparation_rejects_non_nullary_unit_without_invalidating_output() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("not_unit", &[ScalarType::U32]);
        let mut output = buffer(&provider, relation_schema, &[vec![17]]);
        let relations = vec![super::ResidentScheduleRelation::output(&mut output, 2)];
        let operation = super::ResidentOpDescriptor::unit(451, 0, 2);
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 450,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 1,
            ..Default::default()
        };

        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &[operation],
                &[wave],
                &[region],
                &[],
                &[],
                &[0],
            )),
            "resident schedule unit 451 has nonzero operands or invalid output"
        );
        assert_eq!(output.cached_row_count(), Some(1));
    }

    #[test]
    fn real_cuda_unit_and_scan_execute_complete_leaf_semantics() {
        let Some(provider) = provider() else { return };
        let unit_schema = Schema::new(Vec::new());
        let mut unit_output = provider
            .prepare_resident_relation(unit_schema.clone(), 1)
            .expect("unit output")
            .into_buffer();
        let unit_relations = vec![super::ResidentScheduleRelation::output(&mut unit_output, 7)];
        let unit_op = super::ResidentOpDescriptor::unit(461, 0, 7);
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 460,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 1,
            ..Default::default()
        };
        let unit_schedule = provider
            .prepare_resident_schedule(
                unit_relations,
                &[unit_op],
                &[wave],
                &[region],
                &[],
                &[],
                &[0],
            )
            .expect("prepare Unit schedule");
        let unit_stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("Unit stream");
        let mut unit_graph = provider
            .capture_resident_schedule(unit_schedule, 0, Arc::clone(&unit_stream))
            .expect("capture Unit schedule");
        unit_graph.launch().expect("launch Unit schedule");
        let unit_receipt = unit_graph
            .synchronize_and_observe()
            .expect("observe Unit schedule");
        assert_eq!(
            unit_receipt.status.code,
            ResidentTerminalCode::Success as u32
        );
        assert_eq!(unit_receipt.counts, vec![1]);
        drop(unit_graph);
        assert_eq!(unit_output.cached_row_count(), Some(1));

        let scan_schema = schema("scan", &[ScalarType::U32]);
        let scan_source = buffer(&provider, scan_schema, &[vec![3, 5, 8]]);
        let scan_relations =
            vec![super::ResidentScheduleRelation::source(&scan_source, 9).expect("Scan source")];
        let scan_op = super::ResidentOpDescriptor::scan(471, 0, 9);
        let scan_schedule = provider
            .prepare_resident_schedule(
                scan_relations,
                &[scan_op],
                &[wave],
                &[super::ResidentRegionDescriptor {
                    op_id: 470,
                    ..region
                }],
                &[],
                &[],
                &[0],
            )
            .expect("prepare Scan schedule");
        let scan_stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("Scan stream");
        let mut scan_graph = provider
            .capture_resident_schedule(scan_schedule, 0, scan_stream)
            .expect("capture Scan schedule");
        scan_graph.launch().expect("launch Scan schedule");
        let scan_receipt = scan_graph
            .synchronize_and_observe()
            .expect("observe Scan schedule");
        assert_eq!(
            scan_receipt.status.code,
            ResidentTerminalCode::Success as u32
        );
        assert_eq!(scan_receipt.counts, vec![3]);
        assert_eq!(scan_source.cached_row_count(), Some(3));
    }

    #[test]
    fn real_cuda_unit_capacity_zero_reports_exact_overflow_without_storage() {
        let Some(provider) = provider() else { return };
        let mut output = provider
            .prepare_resident_relation(Schema::new(Vec::new()), 0)
            .expect("zero-capacity Unit output")
            .into_buffer();
        let relations = vec![super::ResidentScheduleRelation::output(&mut output, 3)];
        let operation = super::ResidentOpDescriptor::unit(481, 0, 3);
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 480,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 1,
            ..Default::default()
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[0])
            .expect("prepare zero-capacity Unit schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("zero-capacity Unit stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, stream)
            .expect("capture zero-capacity Unit schedule");
        graph.launch().expect("launch zero-capacity Unit schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("observe zero-capacity Unit schedule");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 481);
        assert_eq!(
            receipt.status.resource_code,
            ResidentResourceCode::OutputRows as u32
        );
        assert_eq!(receipt.status.required, 1);
        assert_eq!(receipt.status.capacity, 0);
        assert_eq!(receipt.counts, vec![0]);
    }

    #[test]
    fn preparation_rejects_same_slot_input_output_alias() {
        let Some(provider) = provider() else { return };
        let input = buffer(
            &provider,
            schema("alias", &[ScalarType::U32]),
            &[vec![1, 2]],
        );
        let relations =
            vec![super::ResidentScheduleRelation::source(&input, 7).expect("source relation")];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            op_id: 402,
            out: 0,
            in0: 0,
            in1: 0,
            in0_generation: 7,
            out_generation: 7,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 402,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 1,
            generation_offset: 0,
        };

        let error = match provider.prepare_resident_schedule(
            relations,
            &[operation],
            &[wave],
            &[region],
            &[],
            &[],
            &[],
        ) {
            Err(error) => error,
            Ok(_) => panic!("same-slot input/output alias reached device preparation"),
        };
        assert!(error.to_string().contains("aliases"));
        assert_eq!(input.cached_row_count(), Some(2));
    }

    #[test]
    fn preparation_rejects_cross_slot_shared_runtime_allocation_alias() {
        let Some(provider) = runtime_provider() else {
            return;
        };
        let relation_schema = schema("shared_storage", &[ScalarType::U32]);
        let mut source = buffer(&provider, relation_schema.clone(), &[vec![7]]);
        let mut output = buffer(&provider, relation_schema, &[vec![9]]);
        let shared = Arc::new(
            provider
                .memory
                .alloc::<u8>(4)
                .expect("shared runtime allocation"),
        );
        assert!(
            shared.runtime_block().is_some(),
            "alias witness requires runtime block identity and generation"
        );
        let stream = Arc::clone(provider.device().inner().stream());
        let source_tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        let output_tensor = unsafe { DlpackManagedTensor::from_raw(std::ptr::null_mut()) };
        source.columns[0] =
            CudaColumn::dlpack_xlog_owned(Arc::clone(&shared), Arc::clone(&stream), source_tensor);
        output.columns[0] = CudaColumn::dlpack_xlog_owned(shared, stream, output_tensor);
        let relations = vec![
            super::ResidentScheduleRelation::source(&source, 1).expect("shared source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 0,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 405,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 2,
            generation_offset: 0,
        };
        let error = match provider.prepare_resident_schedule(
            relations,
            &[],
            &[wave],
            &[region],
            &[],
            &[],
            &[],
        ) {
            Err(error) => error,
            Ok(_) => panic!("shared runtime allocation reached scheduler allocation"),
        };
        assert!(error.to_string().contains("aliases storage"));
        assert_eq!(output.cached_row_count(), Some(1));
    }

    #[test]
    fn preparation_rejects_cross_slot_overlapping_raw_views() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("overlap", &[ScalarType::U32]);
        let mut source = buffer(&provider, relation_schema.clone(), &[vec![7]]);
        let mut output = buffer(&provider, relation_schema, &[vec![9]]);
        let backing = provider
            .memory
            .alloc::<u8>(8)
            .expect("overlap backing allocation");
        let base = backing.device_ptr_value();
        let stream = Arc::clone(provider.device().inner().stream());
        source.columns[0] = CudaColumn::dlpack(base, 4, Arc::clone(&stream), unsafe {
            DlpackManagedTensor::from_raw(std::ptr::null_mut())
        });
        output.columns[0] = CudaColumn::dlpack(base + 2, 4, stream, unsafe {
            DlpackManagedTensor::from_raw(std::ptr::null_mut())
        });
        let relations = vec![
            super::ResidentScheduleRelation::source(&source, 1).expect("overlap source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let region = super::ResidentRegionDescriptor {
            iteration_limit: 1,
            op_id: 406,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 2,
            ..Default::default()
        };
        let error =
            match provider.prepare_resident_schedule(relations, &[], &[], &[region], &[], &[], &[])
            {
                Err(error) => error,
                Ok(_) => panic!("overlapping raw views reached scheduler allocation"),
            };
        assert!(error.to_string().contains("aliases storage"));
        assert_eq!(output.cached_row_count(), Some(1));
        drop(backing);
    }

    #[test]
    fn output_count_finalization_is_all_or_nothing() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("receipt", &[ScalarType::U32]);
        let mut first = buffer(&provider, relation_schema.clone(), &[vec![1]]);
        let mut second = buffer(&provider, relation_schema, &[vec![2]]);
        let mut relations = vec![
            super::ResidentScheduleRelation::output(&mut first, 1),
            super::ResidentScheduleRelation::output(&mut second, 2),
        ];
        for relation in &mut relations {
            relation.invalidate_output_metadata();
        }
        let error = super::finalize_schedule_output_counts(&provider, &relations, &[0, 1], &[1, 2])
            .expect_err("second count exceeds capacity");
        assert!(error.to_string().contains("exceeds buffer capacity"));
        drop(relations);
        assert_eq!(first.cached_row_count(), None);
        assert_eq!(second.cached_row_count(), None);
    }

    #[test]
    fn preparation_rejects_relation_from_foreign_provider() {
        let Some(foreign_provider) = provider() else {
            return;
        };
        let Some(provider) = provider() else { return };
        let relation_schema = schema("context", &[ScalarType::U32]);
        let input = buffer(&foreign_provider, relation_schema.clone(), &[vec![9]]);
        let mut output = buffer(&provider, relation_schema, &[vec![0]]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("foreign source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            op_id: 403,
            out: 1,
            in0: 0,
            in0_generation: 1,
            out_generation: 2,
            ..Default::default()
        };

        let error = match provider.prepare_resident_schedule(
            relations,
            &[operation],
            &[],
            &[],
            &[],
            &[],
            &[],
        ) {
            Err(error) => error,
            Ok(_) => panic!("foreign relation reached scheduler allocation"),
        };
        assert!(error.to_string().contains("foreign"));
        assert_eq!(output.cached_row_count(), Some(1));
    }

    #[test]
    fn preparation_rejects_ignored_flags_iteration_limits_and_slot_scopes() {
        let Some(provider) = provider() else { return };
        let base_wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 0,
            ..Default::default()
        };
        let base_region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 410,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            ..Default::default()
        };
        let invalid_wave = super::ResidentWaveDescriptor {
            flags: 1,
            ..base_wave
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[invalid_wave],
                &[base_region],
                &[],
                &[],
                &[],
            )),
            "resident schedule waves must exactly partition operations"
        );
        let invalid_wave_reserved = super::ResidentWaveDescriptor {
            reserved: 1,
            ..base_wave
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[invalid_wave_reserved],
                &[base_region],
                &[],
                &[],
                &[],
            )),
            "resident schedule waves must exactly partition operations"
        );
        let invalid_region = super::ResidentRegionDescriptor {
            flags: base_region.flags | (1 << 31),
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[base_wave],
                &[invalid_region],
                &[],
                &[],
                &[],
            )),
            "resident schedule region range or reserved field is invalid"
        );
        let invalid_region_reserved = super::ResidentRegionDescriptor {
            generation_offset: 1,
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[base_wave],
                &[invalid_region_reserved],
                &[],
                &[],
                &[],
            )),
            "resident schedule generation baselines are not contiguous"
        );
        let invalid_limit = super::ResidentRegionDescriptor {
            iteration_limit: 2,
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[base_wave],
                &[invalid_limit],
                &[],
                &[],
                &[],
            )),
            "resident schedule region control flags are invalid"
        );
        let zero_limit = super::ResidentRegionDescriptor {
            iteration_limit: 0,
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[base_wave],
                &[zero_limit],
                &[],
                &[],
                &[],
            )),
            "resident schedule region control flags are invalid"
        );
        let mismatched_limits = [
            super::ResidentRegionDescriptor {
                iteration_limit: 2,
                op_id: 420,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                iteration_limit: 3,
                op_id: 420,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                ..Default::default()
            },
            super::ResidentRegionDescriptor {
                iteration_limit: 1,
                op_id: 421,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                ..Default::default()
            },
        ];
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                Vec::new(),
                &[],
                &[],
                &mismatched_limits,
                &[],
                &[],
                &[],
            )),
            "resident schedule SCC begin does not match its recursive body"
        );

        let relation_schema = schema("scope", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![1]]);
        let mut output = buffer(&provider, relation_schema, &[vec![0]]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("scope source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            op_id: 411,
            out: 1,
            in0: 0,
            in0_generation: 1,
            out_generation: 2,
            ..Default::default()
        };
        let scoped_wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..base_wave
        };
        let too_narrow = super::ResidentRegionDescriptor {
            first_slot: 0,
            slot_count: 1,
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &[operation],
                &[scoped_wave],
                &[too_narrow],
                &[],
                &[],
                &[],
            )),
            "resident schedule initialization must cover every relation slot"
        );
        assert_eq!(output.cached_row_count(), Some(1));

        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("flag source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let invalid_operation_flags = super::ResidentOpDescriptor {
            flags: 4,
            ..operation
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &[invalid_operation_flags],
                &[scoped_wave],
                &[super::ResidentRegionDescriptor {
                    first_slot: 0,
                    slot_count: 2,
                    ..base_region
                }],
                &[],
                &[],
                &[],
            )),
            "resident schedule operation 411 has an unsupported kind, flag, or payload"
        );
        assert_eq!(output.cached_row_count(), Some(1));

        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("reserved source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let invalid_operation_reserved = super::ResidentOpDescriptor {
            reserved: 1,
            ..operation
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &[invalid_operation_reserved],
                &[scoped_wave],
                &[super::ResidentRegionDescriptor {
                    first_slot: 0,
                    slot_count: 2,
                    ..base_region
                }],
                &[],
                &[],
                &[],
            )),
            "resident schedule operation 411 has an unsupported kind, flag, or payload"
        );
        assert_eq!(output.cached_row_count(), Some(1));

        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("overflowing scope source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let overflowing_scope = super::ResidentRegionDescriptor {
            first_slot: u32::MAX,
            slot_count: 2,
            ..base_region
        };
        assert_eq!(
            schedule_kernel_error(provider.prepare_resident_schedule(
                relations,
                &[operation],
                &[scoped_wave],
                &[overflowing_scope],
                &[],
                &[],
                &[],
            )),
            "resident schedule generation baseline slot scope is invalid"
        );
        assert_eq!(output.cached_row_count(), Some(1));
    }

    #[test]
    fn preparation_accepts_explicit_multihead_recursive_novelty_contract() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("recursive", &[ScalarType::U32]);
        let changed = buffer(&provider, relation_schema.clone(), &[vec![1]]);
        let empty = buffer(&provider, relation_schema.clone(), &[Vec::new()]);
        let stable_left = buffer(&provider, relation_schema.clone(), &[vec![7]]);
        let stable_right = buffer(&provider, relation_schema.clone(), &[vec![7]]);
        let mut first_novel = provider
            .prepare_resident_relation(relation_schema.clone(), 1)
            .expect("first novelty output")
            .into_buffer();
        let mut second_novel = provider
            .prepare_resident_relation(relation_schema, 1)
            .expect("second novelty output")
            .into_buffer();
        let relations = vec![
            super::ResidentScheduleRelation::source(&changed, 1).expect("changed source"),
            super::ResidentScheduleRelation::source(&empty, 2).expect("empty source"),
            super::ResidentScheduleRelation::source(&stable_left, 3).expect("stable source"),
            super::ResidentScheduleRelation::source(&stable_right, 4).expect("stable source"),
            super::ResidentScheduleRelation::output(&mut first_novel, 5),
            super::ResidentScheduleRelation::output(&mut second_novel, 6),
        ];
        let operations = [
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
                op_id: 501,
                out: 4,
                in0: 0,
                in1: 1,
                in0_generation: 1,
                in1_generation: 2,
                out_generation: 5,
                ..Default::default()
            },
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
                op_id: 502,
                out: 5,
                in0: 2,
                in1: 3,
                in0_generation: 3,
                in1_generation: 4,
                out_generation: 6,
                ..Default::default()
            },
        ];
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 2,
            ..Default::default()
        };
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit: 1,
                op_id: 500,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit: 1,
                op_id: 500,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 0,
                iteration_limit: 1,
                op_id: 503,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 6,
                generation_offset: 0,
            },
        ];

        let schedule = provider
            .prepare_resident_schedule(relations, &operations, &[wave], &regions, &[], &[], &[4, 5])
            .expect("explicit recursive schedule contract");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("recursive schedule stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture recursive schedule");
        let nodes = graph.nodes().expect("recursive parent inventory");
        assert_eq!(nodes.len(), 3);
        assert_eq!(nodes[0].kind, CudaGraphNodeKind::Kernel);
        assert_eq!(nodes[1].kind, CudaGraphNodeKind::Conditional);
        assert_eq!(nodes[2].kind, CudaGraphNodeKind::Kernel);
        graph.launch().expect("launch recursive schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("recursive schedule receipt");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::IterationLimit as u32
        );
        assert_eq!(receipt.status.op_id, 500);
        assert_eq!(receipt.status.iterations, 1);
        assert_eq!(receipt.status.limit, 1);
        assert_eq!(receipt.changed, 1);
        assert_eq!(receipt.counts, vec![0, 0]);
    }

    #[test]
    fn real_cuda_recursive_zero_convergence_and_iteration_limit_are_exact() {
        let Some(provider) = provider() else { return };
        let expected_inventory = vec![
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
            CudaGraphNodeKind::Kernel,
        ];

        let (zero_limit, zero_limit_nodes) = run_single_recursive_diff(&provider, &[1], &[], 0);
        assert_eq!(zero_limit_nodes, expected_inventory);
        assert_eq!(
            zero_limit.status.code,
            ResidentTerminalCode::IterationLimit as u32
        );
        assert_eq!(zero_limit.status.op_id, 600);
        assert_eq!(zero_limit.status.iterations, 0);
        assert_eq!(zero_limit.status.limit, 0);
        assert_eq!(zero_limit.changed, 0);
        assert_eq!(zero_limit.counts, vec![0]);

        let (converged, converged_nodes) = run_single_recursive_diff(&provider, &[7], &[7], 3);
        assert_eq!(converged_nodes, expected_inventory);
        assert_eq!(converged.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(converged.status.op_id, 602);
        assert_eq!(converged.status.iterations, 1);
        assert_eq!(converged.status.limit, 0);
        assert_eq!(converged.changed, 0);
        assert_eq!(converged.counts, vec![0]);

        let (limited, limited_nodes) = run_single_recursive_diff(&provider, &[9], &[], 3);
        assert_eq!(limited_nodes, expected_inventory);
        assert_eq!(
            limited.status.code,
            ResidentTerminalCode::IterationLimit as u32
        );
        assert_eq!(limited.status.op_id, 600);
        assert_eq!(limited.status.iterations, 3);
        assert_eq!(limited.status.limit, 3);
        assert_eq!(limited.changed, 1);
        assert_eq!(limited.counts, vec![0]);
    }

    #[test]
    fn real_cuda_two_serial_sccs_preserve_sticky_status_and_aggregate_iterations() {
        let Some(provider) = provider() else { return };
        let expected_inventory = vec![
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
            CudaGraphNodeKind::Kernel,
        ];

        let (success, success_nodes, _) = run_two_scc_diff(&provider, 4, 4, false, false);
        assert_eq!(success_nodes, expected_inventory);
        assert_eq!(success.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(success.status.op_id, 703);
        assert_eq!(success.status.iterations, 2);
        assert_eq!(success.status.limit, 0);
        assert_eq!(success.counts, vec![0, 0]);

        let (first_zero, first_zero_nodes, first_zero_storage) =
            run_two_scc_diff(&provider, 0, 4, true, true);
        assert_eq!(first_zero_nodes, expected_inventory);
        assert_eq!(
            first_zero.status.code,
            ResidentTerminalCode::IterationLimit as u32
        );
        assert_eq!(first_zero.status.op_id, 701);
        assert_eq!(first_zero.status.iterations, 0);
        assert_eq!(first_zero.status.limit, 0);
        assert_eq!(first_zero.counts, vec![0, 0]);
        assert_eq!(first_zero_storage, [0x1111_1111, 0x2222_2222]);

        let (second_zero, second_zero_nodes, second_zero_storage) =
            run_two_scc_diff(&provider, 4, 0, false, true);
        assert_eq!(second_zero_nodes, expected_inventory);
        assert_eq!(
            second_zero.status.code,
            ResidentTerminalCode::IterationLimit as u32
        );
        assert_eq!(second_zero.status.op_id, 702);
        assert_eq!(second_zero.status.iterations, 1);
        assert_eq!(second_zero.status.limit, 0);
        assert_eq!(second_zero.counts, vec![0, 0]);
        assert_eq!(second_zero_storage, [0x1111_1111, 0x2222_2222]);
    }

    #[test]
    fn real_cuda_recursive_overflow_is_sticky_and_stops_downstream_writes() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("recursive_overflow", &[ScalarType::U32]);
        let left = buffer(&provider, relation_schema.clone(), &[vec![1, 2]]);
        let right = buffer(&provider, relation_schema.clone(), &[Vec::new()]);
        let mut novelty = buffer(&provider, relation_schema.clone(), &[vec![0xaaaa_aaaa]]);
        let mut downstream = buffer(&provider, relation_schema, &[vec![0xbbbb_bbbb]]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("left source"),
            super::ResidentScheduleRelation::source(&right, 2).expect("right source"),
            super::ResidentScheduleRelation::output(&mut novelty, 3),
            super::ResidentScheduleRelation::output(&mut downstream, 4),
        ];
        let operations = [
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY,
                op_id: 731,
                out: 2,
                in0: 0,
                in1: 1,
                in0_generation: 1,
                in1_generation: 2,
                out_generation: 3,
                ..Default::default()
            },
            super::ResidentOpDescriptor {
                kind: super::OP_PROJECT,
                op_id: 732,
                out: 3,
                in0: 2,
                in0_generation: 3,
                out_generation: 4,
                aux_count: 1,
                ..Default::default()
            },
        ];
        let expression = super::ResidentProjectExpressionDescriptor::column(0, 4);
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 2,
            ..Default::default()
        };
        let regions = [
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 0,
                iteration_limit: 4,
                op_id: 730,
                flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                    | super::RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
                first_slot: 0,
                slot_count: 4,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 0,
                wave_count: 1,
                iteration_limit: 4,
                op_id: 730,
                flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
                first_slot: 0,
                slot_count: 4,
                generation_offset: 0,
            },
            super::ResidentRegionDescriptor {
                first_wave: 1,
                wave_count: 0,
                iteration_limit: 1,
                op_id: 733,
                flags: super::RESIDENT_SCHEDULE_REGION_FINALIZE,
                first_slot: 0,
                slot_count: 4,
                generation_offset: 0,
            },
        ];
        let schedule = provider
            .prepare_resident_schedule(
                relations,
                &operations,
                &[wave],
                &regions,
                &[],
                &[expression],
                &[2, 3],
            )
            .expect("prepare recursive overflow schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("recursive overflow stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture recursive overflow schedule");
        graph.launch().expect("launch recursive overflow schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("recursive overflow receipt");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 731);
        assert_eq!(receipt.status.iterations, 0);
        assert_eq!(receipt.status.limit, 4);
        assert_eq!(receipt.status.required, 2);
        assert_eq!(receipt.status.capacity, 1);
        assert_eq!(receipt.changed, 0);
        assert_eq!(receipt.counts, vec![0, 0]);
        for (slot, expected) in [(2_usize, 0xaaaa_aaaa_u32), (3, 0xbbbb_bbbb)] {
            let raw: Vec<u8> = provider
                .device()
                .inner()
                .dtoh_sync_copy(
                    graph
                        .relation(slot)
                        .expect("recursive output")
                        .column(0)
                        .expect("recursive output column"),
                )
                .expect("recursive output storage");
            assert_eq!(&raw[..4], &expected.to_le_bytes());
        }
    }

    #[test]
    fn capture_rejects_stream_from_foreign_provider_context() {
        let Some(foreign_provider) = provider() else {
            return;
        };
        let Some(provider) = provider() else { return };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 0,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 420,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            ..Default::default()
        };
        let schedule = provider
            .prepare_resident_schedule(Vec::new(), &[], &[wave], &[region], &[], &[], &[])
            .expect("empty local schedule");
        let foreign_stream = foreign_provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("foreign stream");

        let error = match provider.capture_resident_schedule(schedule, 0, foreign_stream) {
            Err(error) => error,
            Ok(_) => panic!("foreign stream captured a local schedule"),
        };
        assert!(error.to_string().contains("foreign CUDA context"));
    }

    #[test]
    fn real_cuda_selected_stream_cooperative_capture_has_one_kernel_node() {
        let Some(provider) = provider() else { return };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 0,
            flags: 0,
            reserved: 0,
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 91,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 0,
            generation_offset: 0,
        };
        let schedule = provider
            .prepare_resident_schedule(Vec::new(), &[], &[wave], &[region], &[], &[], &[])
            .expect("prepare empty resident schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("non-default stream");
        assert_ne!(
            stream.cu_stream(),
            provider.device().inner().stream().cu_stream(),
            "feasibility gate must capture on a selected non-default stream"
        );

        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture cooperative resident schedule");
        assert_eq!(graph.node_count().expect("node count"), 1);
        let nodes = graph.nodes().expect("node inventory");
        assert_eq!(nodes.len(), 1);
        assert_eq!(nodes[0].kind, CudaGraphNodeKind::Kernel);

        graph.launch().expect("launch resident schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("resident schedule receipt");
        assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(receipt.status.op_id, 91);
    }

    #[test]
    fn real_cuda_graph_lease_rejects_overlap_allows_replay_and_synchronizes_drop() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("graph_lease", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![1, 2]]);
        let mut output = buffer(&provider, relation_schema.clone(), &[vec![0, 0]]);
        let schedule = passthrough_schedule(&provider, &input, &mut output, 901);
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("graph lease stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture graph lease schedule");

        assert_eq!(
            graph
                .relation(1)
                .expect("leased output before launch")
                .cached_row_count(),
            None
        );
        match graph.synchronize_and_observe() {
            Err(XlogError::Kernel(message)) => assert_eq!(
                message,
                "resident schedule has no in-flight launch to observe"
            ),
            Err(error) => panic!("unexpected pre-launch observation error: {error}"),
            Ok(_) => panic!("pre-launch observation unexpectedly succeeded"),
        }

        graph.launch().expect("first graph lease launch");
        match graph.relation(1) {
            Err(XlogError::Kernel(message)) => assert_eq!(
                message,
                "resident schedule relation is unavailable while launch is in flight"
            ),
            Err(error) => panic!("unexpected in-flight relation error: {error}"),
            Ok(_) => panic!("in-flight relation access unexpectedly succeeded"),
        }
        match graph.launch() {
            Err(XlogError::Kernel(message)) => {
                assert_eq!(message, "resident schedule launch is already in flight")
            }
            Err(error) => panic!("unexpected overlapping launch error: {error}"),
            Ok(()) => panic!("overlapping graph launch unexpectedly succeeded"),
        }
        let first = graph
            .synchronize_and_observe()
            .expect("first graph lease receipt");
        assert_eq!(first.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(first.status.op_id, 902);
        assert_eq!(first.counts, vec![2]);
        assert_eq!(
            normalized_rows(&provider, graph.relation(1).expect("first replay output")),
            vec![vec![1], vec![2]]
        );
        match graph.synchronize_and_observe() {
            Err(XlogError::Kernel(message)) => assert_eq!(
                message,
                "resident schedule has no in-flight launch to observe"
            ),
            Err(error) => panic!("unexpected duplicate observation error: {error}"),
            Ok(_) => panic!("duplicate observation unexpectedly succeeded"),
        }

        graph.launch().expect("replayed graph lease launch");
        let replay = graph
            .synchronize_and_observe()
            .expect("replayed graph lease receipt");
        assert_eq!(replay, first);
        drop(graph);
        assert_eq!(output.cached_row_count(), Some(2));
        assert!(!output.canonical_full_row_set_certified());
        assert_eq!(normalized_rows(&provider, &output), vec![vec![1], vec![2]]);

        let mut dropped_output = buffer(&provider, relation_schema, &[vec![0, 0]]);
        let dropped_schedule = passthrough_schedule(&provider, &input, &mut dropped_output, 911);
        let dropped_stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("drop synchronization stream");
        let mut dropped_graph = provider
            .capture_resident_schedule(dropped_schedule, 0, dropped_stream)
            .expect("capture drop synchronization schedule");
        dropped_graph
            .launch()
            .expect("launch drop synchronization schedule");
        drop(dropped_graph);
        assert_eq!(dropped_output.cached_row_count(), None);
        assert!(!dropped_output.canonical_full_row_set_certified());
        assert_eq!(
            normalized_rows(&provider, &dropped_output),
            vec![vec![1], vec![2]]
        );
    }

    #[test]
    fn capture_rejects_sibling_provider_on_the_same_cuda_context() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("provider_identity", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![1]]);
        let mut output = buffer(&provider, relation_schema, &[vec![0]]);
        let schedule = passthrough_schedule(&provider, &input, &mut output, 921);
        let sibling_memory = Arc::new(GpuMemoryManager::new(
            Arc::clone(provider.device()),
            MemoryBudget::with_limit(512 * 1024 * 1024),
        ));
        let sibling = CudaKernelProvider::from_loaded_device(
            Arc::clone(provider.device()),
            sibling_memory,
            None,
        );
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("same-context identity stream");

        match sibling.capture_resident_schedule(schedule, 0, stream) {
            Err(XlogError::Kernel(message)) => assert_eq!(
                message,
                "resident schedule belongs to a different CUDA kernel provider"
            ),
            Err(error) => panic!("unexpected provider identity error: {error}"),
            Ok(_) => panic!("sibling provider captured a foreign resident schedule"),
        }
        assert_eq!(output.cached_row_count(), None);
    }

    #[test]
    fn real_cuda_schedule_receipt_uses_one_pinned_final_dtoh_per_observation() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("receipt_accounting", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![1, 2]]);
        let mut output = buffer(&provider, relation_schema, &[vec![0, 0]]);
        let schedule = passthrough_schedule(&provider, &input, &mut output, 931);
        let expected_bytes =
            size_of::<crate::provider::resident_relational::ResidentTerminalStatus>()
                + 2 * size_of::<u32>();
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("receipt accounting stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, stream)
            .expect("capture receipt accounting schedule");
        provider.reset_host_transfer_stats();
        provider.reset_untracked_metadata_dtoh_count();
        provider.reset_final_observation_transfer_stats();

        graph.launch().expect("launch receipt accounting schedule");
        assert_eq!(provider.host_transfer_stats().dtoh_calls, 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        assert_eq!(provider.final_observation_transfer_stats().dtoh_calls, 0);
        graph
            .synchronize_and_observe()
            .expect("first receipt accounting observation");
        assert_eq!(provider.host_transfer_stats().dtoh_calls, 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        let first = provider.final_observation_transfer_stats();
        assert_eq!(first.dtoh_calls, 1);
        assert_eq!(first.dtoh_bytes, expected_bytes as u64);
        assert_eq!(first.pinned_receipts, 1);

        graph.launch().expect("replay receipt accounting schedule");
        graph
            .synchronize_and_observe()
            .expect("replay receipt accounting observation");
        assert_eq!(provider.host_transfer_stats().dtoh_calls, 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);
        let replay = provider.final_observation_transfer_stats();
        assert_eq!(replay.dtoh_calls, 2);
        assert_eq!(replay.dtoh_bytes, 2 * expected_bytes as u64);
        assert_eq!(replay.pinned_receipts, 2);
    }

    #[test]
    fn real_cuda_compact_union_orders_non_power_of_two_rows_across_merge_tiles() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("ordered_union", &[ScalarType::U32]);
        let mut left: Vec<u64> = (0..1_537).rev().collect();
        left.extend([0, 512, 1_536]);
        let mut right: Vec<u64> = (768..2_305).rev().collect();
        right.extend([768, 1_536, 2_304]);
        let (receipt, rows) = compact_set_rows(
            &provider,
            relation_schema,
            &[left],
            &[right],
            super::OP_UNION,
            2_305,
        );
        assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(receipt.counts, vec![2_305]);
        assert_eq!(
            rows,
            (0..2_305).map(|value| vec![value]).collect::<Vec<_>>()
        );
    }

    #[test]
    fn real_cuda_compact_diff_orders_high_u64_values_across_merge_tiles() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("ordered_diff", &[ScalarType::U64]);
        let base = 1_u64 << 40;
        let mut left: Vec<u64> = (0..4_093).rev().map(|value| base + value).collect();
        left.extend([base, base + 2_047, base + 4_092]);
        let mut right: Vec<u64> = (0..4_093)
            .rev()
            .filter(|value| value % 3 == 0)
            .map(|value| base + value)
            .collect();
        right.extend([base, base + 3, base + 4_092]);
        let expected: Vec<Vec<u64>> = (0..4_093)
            .filter(|value| value % 3 != 0)
            .map(|value| vec![base + value])
            .collect();
        let (receipt, rows) = compact_set_rows(
            &provider,
            relation_schema,
            &[left],
            &[right],
            super::OP_DIFF,
            expected.len() as u64,
        );
        assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(receipt.counts, vec![expected.len() as u32]);
        assert_eq!(rows, expected);
    }

    #[test]
    fn real_cuda_compact_zero_arity_set_truth_table() {
        let Some(provider) = provider() else { return };
        for (left, right, union_count, diff_count) in [
            (false, false, 0, 0),
            (false, true, 1, 0),
            (true, false, 1, 1),
            (true, true, 1, 0),
        ] {
            assert_eq!(
                compact_nullary_set_count(&provider, left, right, super::OP_UNION),
                union_count
            );
            assert_eq!(
                compact_nullary_set_count(&provider, left, right, super::OP_DIFF),
                diff_count
            );
        }
    }

    #[test]
    fn real_cuda_compact_set_max_workspace_is_canonical_and_bounded() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("max_ordered_union", &[ScalarType::U32]);
        let left_values: Vec<u64> = (0..65_536).rev().collect();
        let right_values: Vec<u64> = (0..65_536).map(|index| (index * 40_009) % 65_536).collect();
        let left = buffer(&provider, relation_schema.clone(), &[left_values]);
        let right = buffer(&provider, relation_schema.clone(), &[right_values]);
        let mut output = provider
            .prepare_resident_relation(relation_schema, 65_536)
            .expect("maximum compact set output")
            .into_buffer();
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("maximum left"),
            super::ResidentScheduleRelation::source(&right, 2).expect("maximum right"),
            super::ResidentScheduleRelation::output(&mut output, 3),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_UNION,
            op_id: 982,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            wave_count: 1,
            iteration_limit: 1,
            op_id: 983,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            slot_count: 3,
            ..Default::default()
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[2])
            .expect("prepare maximum compact set schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("maximum compact set stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture maximum compact set schedule");

        for _ in 0..20 {
            graph.launch().expect("maximum compact set warmup launch");
            let receipt = graph
                .synchronize_and_observe()
                .expect("maximum compact set warmup receipt");
            assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
        }
        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        let mut durations = Vec::with_capacity(100);
        for _ in 0..100 {
            let started = Instant::now();
            graph.launch().expect("maximum compact set measured launch");
            let receipt = graph
                .synchronize_and_observe()
                .expect("maximum compact set measured receipt");
            durations.push(started.elapsed());
            assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
            assert_eq!(receipt.counts, vec![65_536]);
        }
        durations.sort_unstable();
        let median: Duration = durations[durations.len() / 2];
        let p95: Duration = durations[durations.len() * 95 / 100];
        eprintln!(
            "maximum compact set launch median_us={} p95_us={}",
            median.as_micros(),
            p95.as_micros()
        );
        let transfers = provider.host_transfer_stats();
        assert_eq!(transfers.htod_calls, 0);
        assert_eq!(transfers.htod_bytes, 0);
        assert_eq!(transfers.dtoh_calls, 0);
        assert_eq!(transfers.dtoh_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);

        let ordered = rows_in_device_order(
            &provider,
            graph.relation(2).expect("maximum compact set relation"),
        );
        assert_eq!(ordered.len(), 65_536);
        assert!(ordered
            .iter()
            .enumerate()
            .all(|(index, row)| row == &[index as u64]));
    }

    #[test]
    fn real_cuda_compact_schedule_matches_primitive_chain_at_arity_seventeen() {
        let Some(provider) = provider() else { return };
        let left_types = [
            ScalarType::Symbol,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::U32,
            ScalarType::U64,
        ];
        let projected_types = &left_types[..8];
        let right_types = left_types;
        let left_columns = vec![
            vec![5, 2, 5, 4],
            vec![9, 7, 11, 12],
            vec![100, 200, 300, 400],
            vec![13, 23, 33, 43],
            vec![1_000, 2_000, 3_000, 4_000],
            vec![15, 25, 35, 45],
            vec![6_000, 7_000, 8_000, 9_000],
            vec![17, 27, 37, 47],
            vec![10_000, 11_000, 12_000, 13_000],
        ];
        let right_columns = vec![
            vec![5, 5, 8],
            vec![50, 60, 80],
            vec![500, 600, 800],
            vec![53, 63, 83],
            vec![5_000, 6_000, 8_000],
            vec![55, 65, 85],
            vec![6_000, 7_000, 9_000],
            vec![57, 67, 87],
            vec![7_000, 8_000, 10_000],
        ];
        let left = buffer(&provider, schema("left", &left_types), &left_columns);
        let right = buffer(&provider, schema("right", &right_types), &right_columns);
        let projected_schema = schema("projected", projected_types);
        let mut joined_columns = projected_schema.columns.clone();
        joined_columns.extend(right.schema().columns.iter().cloned());
        let joined_schema = Schema::new(joined_columns);
        assert_eq!(joined_schema.arity(), 17);

        let left_row_zero: Vec<u64> = left_columns[..8].iter().map(|column| column[0]).collect();
        let left_row_two: Vec<u64> = left_columns[..8].iter().map(|column| column[2]).collect();
        let right_row_zero: Vec<u64> = right_columns.iter().map(|column| column[0]).collect();
        let right_row_one: Vec<u64> = right_columns.iter().map(|column| column[1]).collect();
        let mut duplicate = left_row_zero.clone();
        duplicate.extend(right_row_zero);
        let new_row: Vec<u64> = (0..17).map(|column| 90_000 + column as u64).collect();
        let union_extra = buffer(
            &provider,
            joined_schema.clone(),
            &columns_from_rows(&[duplicate, new_row]),
        );
        let mut removed = left_row_two;
        removed.extend(right_row_one);
        let diff_right = buffer(
            &provider,
            joined_schema.clone(),
            &columns_from_rows(&[removed]),
        );

        let comparisons = [
            ResidentFilterComparison::new(
                ResidentFilterOperand::Column(0),
                CompareOp::Eq,
                ResidentFilterOperand::Constant(ResidentScalar::Symbol(5)),
            ),
            ResidentFilterComparison::new(
                ResidentFilterOperand::Column(1),
                CompareOp::Gt,
                ResidentFilterOperand::Constant(ResidentScalar::U32(8)),
            ),
            ResidentFilterComparison::new(
                ResidentFilterOperand::Column(2),
                CompareOp::Le,
                ResidentFilterOperand::Constant(ResidentScalar::U64(300)),
            ),
        ];
        let project_expressions: Vec<_> = (0..8).map(ResidentProjectExpr::Column).collect();
        let primitive_filter = provider
            .prepare_resident_relation(left.schema().clone(), 4)
            .expect("primitive filter output");
        let primitive_project = provider
            .prepare_resident_relation(projected_schema.clone(), 4)
            .expect("primitive project output");
        let primitive_join = provider
            .prepare_resident_relation(joined_schema.clone(), 4)
            .expect("primitive join output");
        let primitive_union = provider
            .prepare_resident_relation(joined_schema.clone(), 5)
            .expect("primitive union output");
        let primitive_final = provider
            .prepare_resident_relation(joined_schema.clone(), 5)
            .expect("primitive diff output");
        let filter_workspace = provider
            .prepare_resident_filter_workspace(&left, &comparisons)
            .expect("primitive filter workspace");
        let project_workspace = provider
            .prepare_resident_project_workspace(
                primitive_filter.buffer(),
                &projected_schema,
                &project_expressions,
            )
            .expect("primitive project workspace");
        let join_workspace = provider
            .prepare_resident_join_workspace(3)
            .expect("primitive join workspace");
        let union_workspace = provider
            .prepare_resident_set_workspace(6)
            .expect("primitive union workspace");
        let diff_workspace = provider
            .prepare_resident_set_workspace(6)
            .expect("primitive diff workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("primitive control");
        let primitive_stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("primitive stream");
        let primitive_graph = CapturedCudaGraph::capture_on_stream(&primitive_stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &primitive_stream)?;
            provider.record_resident_filter_on_stream(
                &left,
                &primitive_filter,
                &filter_workspace,
                &control,
                101,
                &primitive_stream,
            )?;
            provider.record_resident_project_on_stream(
                primitive_filter.buffer(),
                &primitive_project,
                &project_workspace,
                &control,
                102,
                &primitive_stream,
            )?;
            provider.record_resident_join_on_stream(
                ResidentJoinKind::Inner,
                primitive_project.buffer(),
                0,
                &right,
                0,
                &primitive_join,
                &join_workspace,
                &control,
                103,
                &primitive_stream,
            )?;
            provider.record_resident_union_on_stream(
                primitive_join.buffer(),
                &union_extra,
                &primitive_union,
                &union_workspace,
                &control,
                104,
                &primitive_stream,
            )?;
            provider.record_resident_diff_on_stream(
                primitive_union.buffer(),
                &diff_right,
                &primitive_final,
                &diff_workspace,
                &control,
                105,
                &primitive_stream,
            )
        })
        .expect("primitive chain capture");
        primitive_graph
            .launch(&primitive_stream)
            .expect("primitive chain launch");
        primitive_stream
            .synchronize()
            .expect("primitive chain synchronization");
        let expected = normalized_rows(&provider, primitive_final.buffer());

        let mut scheduled_filter = provider
            .prepare_resident_relation(left.schema().clone(), 4)
            .expect("scheduled filter output")
            .into_buffer();
        let mut scheduled_project = provider
            .prepare_resident_relation(projected_schema.clone(), 4)
            .expect("scheduled project output")
            .into_buffer();
        let mut scheduled_join = provider
            .prepare_resident_relation(joined_schema.clone(), 4)
            .expect("scheduled join output")
            .into_buffer();
        let mut scheduled_union = provider
            .prepare_resident_relation(joined_schema.clone(), 5)
            .expect("scheduled union output")
            .into_buffer();
        let mut scheduled_final = provider
            .prepare_resident_relation(joined_schema, 5)
            .expect("scheduled diff output")
            .into_buffer();
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 10).expect("left source"),
            super::ResidentScheduleRelation::source(&right, 11).expect("right source"),
            super::ResidentScheduleRelation::source(&union_extra, 12).expect("union source"),
            super::ResidentScheduleRelation::source(&diff_right, 13).expect("diff source"),
            super::ResidentScheduleRelation::output(&mut scheduled_filter, 14),
            super::ResidentScheduleRelation::output(&mut scheduled_project, 15),
            super::ResidentScheduleRelation::output(&mut scheduled_join, 16),
            super::ResidentScheduleRelation::output(&mut scheduled_union, 17),
            super::ResidentScheduleRelation::output(&mut scheduled_final, 18),
        ];
        let ops = [
            super::ResidentOpDescriptor {
                kind: super::OP_FILTER,
                flags: 0,
                op_id: 101,
                out: 4,
                in0: 0,
                in1: 0,
                in0_generation: 10,
                in1_generation: 0,
                out_generation: 14,
                aux_offset: 0,
                aux_count: 3,
                left_key: 0,
                right_key: 0,
                scan_delta: 0,
                filter_delta: 0,
                schema_winner_head: 0,
                schema_winner_id: 0,
                reserved: 0,
            },
            super::ResidentOpDescriptor {
                kind: super::OP_PROJECT,
                flags: 0,
                op_id: 102,
                out: 5,
                in0: 4,
                in1: 0,
                in0_generation: 14,
                in1_generation: 0,
                out_generation: 15,
                aux_offset: 0,
                aux_count: 8,
                left_key: 0,
                right_key: 0,
                scan_delta: 0,
                filter_delta: 0,
                schema_winner_head: 0,
                schema_winner_id: 0,
                reserved: 0,
            },
            super::ResidentOpDescriptor {
                kind: super::OP_JOIN_INNER,
                flags: 0,
                op_id: 103,
                out: 6,
                in0: 5,
                in1: 1,
                in0_generation: 15,
                in1_generation: 11,
                out_generation: 16,
                aux_offset: 0,
                aux_count: 0,
                left_key: 0,
                right_key: 0,
                scan_delta: 0,
                filter_delta: 0,
                schema_winner_head: 0,
                schema_winner_id: 0,
                reserved: 0,
            },
            super::ResidentOpDescriptor {
                kind: super::OP_UNION,
                flags: 0,
                op_id: 104,
                out: 7,
                in0: 6,
                in1: 2,
                in0_generation: 16,
                in1_generation: 12,
                out_generation: 17,
                aux_offset: 0,
                aux_count: 0,
                left_key: 0,
                right_key: 0,
                scan_delta: 0,
                filter_delta: 0,
                schema_winner_head: 0,
                schema_winner_id: 0,
                reserved: 0,
            },
            super::ResidentOpDescriptor {
                kind: super::OP_DIFF,
                flags: 0,
                op_id: 105,
                out: 8,
                in0: 7,
                in1: 3,
                in0_generation: 17,
                in1_generation: 13,
                out_generation: 18,
                aux_offset: 0,
                aux_count: 0,
                left_key: 0,
                right_key: 0,
                scan_delta: 0,
                filter_delta: 0,
                schema_winner_head: 0,
                schema_winner_id: 0,
                reserved: 0,
            },
        ];
        let filter_descriptors = [
            super::ResidentFilterComparisonDescriptor::column_constant(0, 0, 4, 5),
            super::ResidentFilterComparisonDescriptor::column_constant(1, 4, 4, 8),
            super::ResidentFilterComparisonDescriptor::column_constant(2, 3, 8, 300),
        ];
        let project_descriptors: Vec<_> = projected_types
            .iter()
            .enumerate()
            .map(|(column, scalar)| {
                super::ResidentProjectExpressionDescriptor::column(
                    column as u32,
                    scalar.size_bytes() as u32,
                )
            })
            .collect();
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 5,
            flags: 0,
            reserved: 0,
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 199,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 9,
            generation_offset: 0,
        };
        let schedule = provider
            .prepare_resident_schedule(
                relations,
                &ops,
                &[wave],
                &[region],
                &filter_descriptors,
                &project_descriptors,
                &[8],
            )
            .expect("prepare compact resident schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("compact schedule stream");
        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("compact schedule capture");
        assert_eq!(graph.node_count().expect("node count"), 1);
        assert_eq!(
            graph.nodes().expect("nodes")[0].kind,
            CudaGraphNodeKind::Kernel
        );
        graph.launch().expect("compact schedule launch");
        let receipt = graph
            .synchronize_and_observe()
            .expect("compact schedule receipt");
        let transfers = provider.host_transfer_stats();
        let launch_metadata = provider.host_launch_metadata_transfer_stats();
        assert_eq!(transfers.htod_calls, 0);
        assert_eq!(transfers.htod_bytes, 0);
        assert_eq!(transfers.dtoh_calls, 0);
        assert_eq!(transfers.dtoh_bytes, 0);
        assert_eq!(launch_metadata.htod_calls, 0);
        assert_eq!(launch_metadata.htod_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);

        assert_eq!(receipt.status.code, ResidentTerminalCode::Success as u32);
        assert_eq!(receipt.status.op_id, 199);
        assert_eq!(receipt.counts, vec![4]);
        let scheduled_final = graph.relation(8).expect("scheduled final relation");
        assert_eq!(scheduled_final.arity(), 17);
        assert_eq!(scheduled_final.cached_row_count(), Some(4));
        assert!(!scheduled_final.canonical_full_row_set_certified());
        assert_eq!(rows_in_device_order(&provider, scheduled_final), expected);
    }

    #[test]
    fn real_cuda_compact_set_overflow_is_exact_and_preserves_output_storage() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("set_overflow", &[ScalarType::U32]);
        let left = buffer(&provider, relation_schema.clone(), &[vec![1, 2]]);
        let right = buffer(&provider, relation_schema.clone(), &[vec![3, 4]]);
        let sentinels = [0xdead_beef_u64, 0xcafe_babe];
        let mut output = buffer(&provider, relation_schema, &[sentinels.to_vec()]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("left source"),
            super::ResidentScheduleRelation::source(&right, 2).expect("right source"),
            super::ResidentScheduleRelation::output(&mut output, 3),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_UNION,
            op_id: 801,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 802,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 3,
            generation_offset: 0,
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[2])
            .expect("prepare set overflow schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("set overflow stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture set overflow schedule");
        graph.launch().expect("launch set overflow schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("set overflow receipt");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 801);
        assert_eq!(receipt.status.required, 4);
        assert_eq!(receipt.status.capacity, 2);
        assert_eq!(receipt.counts, vec![0]);
        let raw: Vec<u8> = provider
            .device()
            .inner()
            .dtoh_sync_copy(
                graph
                    .relation(2)
                    .expect("set overflow output")
                    .column(0)
                    .expect("set overflow column"),
            )
            .expect("set overflow storage");
        let expected: Vec<u8> = sentinels
            .iter()
            .flat_map(|value| (*value as u32).to_le_bytes())
            .collect();
        assert_eq!(raw, expected);
    }

    #[test]
    fn real_cuda_compact_join_overflow_is_exact_and_preserves_output_storage() {
        let Some(provider) = provider() else { return };
        let left_schema = schema("join_left", &[ScalarType::U32]);
        let right_schema = schema("join_right", &[ScalarType::U32]);
        let mut output_columns = left_schema.columns.clone();
        output_columns.extend(right_schema.columns.iter().cloned());
        let output_schema = Schema::new(output_columns);
        let left = buffer(&provider, left_schema, &[vec![1, 1]]);
        let right = buffer(&provider, right_schema, &[vec![1, 1]]);
        let sentinels = [
            vec![0x1111_1111_u64, 0x2222_2222],
            vec![0x3333_3333_u64, 0x4444_4444],
        ];
        let mut output = buffer(&provider, output_schema, &sentinels);
        let relations = vec![
            super::ResidentScheduleRelation::source(&left, 1).expect("left source"),
            super::ResidentScheduleRelation::source(&right, 2).expect("right source"),
            super::ResidentScheduleRelation::output(&mut output, 3),
        ];
        let operation = super::ResidentOpDescriptor {
            kind: super::OP_JOIN_INNER,
            op_id: 811,
            out: 2,
            in0: 0,
            in1: 1,
            in0_generation: 1,
            in1_generation: 2,
            out_generation: 3,
            left_key: 0,
            right_key: 0,
            ..Default::default()
        };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 812,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 3,
            generation_offset: 0,
        };
        let schedule = provider
            .prepare_resident_schedule(relations, &[operation], &[wave], &[region], &[], &[], &[2])
            .expect("prepare join overflow schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("join overflow stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture join overflow schedule");
        graph.launch().expect("launch join overflow schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("join overflow receipt");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 811);
        assert_eq!(receipt.status.required, 4);
        assert_eq!(receipt.status.capacity, 2);
        assert_eq!(receipt.counts, vec![0]);
        for (column, expected) in sentinels.iter().enumerate() {
            let raw: Vec<u8> = provider
                .device()
                .inner()
                .dtoh_sync_copy(
                    graph
                        .relation(2)
                        .expect("join overflow output")
                        .column(column)
                        .expect("join overflow column"),
                )
                .expect("join overflow storage");
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|value| (*value as u32).to_le_bytes())
                .collect();
            assert_eq!(raw, expected, "column {column} was written on overflow");
        }
    }

    #[test]
    fn real_cuda_compact_project_overflow_is_sticky_and_preserves_storage() {
        let Some(provider) = provider() else { return };
        let relation_schema = schema("project_overflow", &[ScalarType::U32]);
        let input = buffer(&provider, relation_schema.clone(), &[vec![1, 2, 3, 4]]);
        let sentinels = [0xaaaa_aaaa_u64, 0xbbbb_bbbb];
        let downstream_sentinels = [0xcccc_cccc_u64, 0xdddd_dddd];
        let mut overflow_output = buffer(&provider, relation_schema.clone(), &[sentinels.to_vec()]);
        let mut downstream = buffer(&provider, relation_schema, &[downstream_sentinels.to_vec()]);
        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("project source"),
            super::ResidentScheduleRelation::output(&mut overflow_output, 2),
            super::ResidentScheduleRelation::output(&mut downstream, 3),
        ];
        let operations = [
            super::ResidentOpDescriptor {
                kind: super::OP_PROJECT,
                op_id: 821,
                out: 1,
                in0: 0,
                in0_generation: 1,
                out_generation: 2,
                aux_offset: 0,
                aux_count: 1,
                ..Default::default()
            },
            super::ResidentOpDescriptor {
                kind: super::OP_PROJECT,
                op_id: 822,
                out: 2,
                in0: 1,
                in0_generation: 2,
                out_generation: 3,
                aux_offset: 1,
                aux_count: 1,
                ..Default::default()
            },
        ];
        let expression = super::ResidentProjectExpressionDescriptor::column(0, 4);
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 2,
            ..Default::default()
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 823,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 3,
            generation_offset: 0,
        };
        let schedule = provider
            .prepare_resident_schedule(
                relations,
                &operations,
                &[wave],
                &[region],
                &[],
                &[expression, expression],
                &[1, 2],
            )
            .expect("prepare project overflow schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("project overflow stream");
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("capture project overflow schedule");
        graph.launch().expect("launch project overflow schedule");
        let receipt = graph
            .synchronize_and_observe()
            .expect("project overflow receipt");
        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 821);
        assert_eq!(receipt.status.required, 4);
        assert_eq!(receipt.status.capacity, 2);
        assert_eq!(receipt.counts, vec![0, 0]);
        for (slot, expected) in [(1_usize, sentinels), (2, downstream_sentinels)] {
            let raw: Vec<u8> = provider
                .device()
                .inner()
                .dtoh_sync_copy(
                    graph
                        .relation(slot)
                        .expect("project output")
                        .column(0)
                        .expect("project column"),
                )
                .expect("project storage");
            let expected: Vec<u8> = expected
                .iter()
                .flat_map(|value| (*value as u32).to_le_bytes())
                .collect();
            assert_eq!(raw, expected, "slot {slot} was written after overflow");
        }
    }

    #[test]
    fn real_cuda_compact_filter_overflow_is_exact_and_preserves_output_storage() {
        let Some(provider) = provider() else { return };
        let types = [ScalarType::Symbol, ScalarType::U32, ScalarType::U64];
        let relation_schema = schema("overflow", &types);
        let input = buffer(
            &provider,
            relation_schema.clone(),
            &[
                vec![1, 2, 3, 4],
                vec![10, 20, 30, 40],
                vec![100, 200, 300, 400],
            ],
        );
        let sentinel_columns = [
            vec![0xdead_beef, 0xcafe_babe],
            vec![0x1234_5678, 0x8765_4321],
            vec![0x0123_4567_89ab_cdef, 0xfedc_ba98_7654_3210],
        ];
        let mut output = buffer(&provider, relation_schema, &sentinel_columns);
        let relations = vec![
            super::ResidentScheduleRelation::source(&input, 1).expect("input source"),
            super::ResidentScheduleRelation::output(&mut output, 2),
        ];
        let op = super::ResidentOpDescriptor {
            kind: super::OP_FILTER,
            flags: 0,
            op_id: 301,
            out: 1,
            in0: 0,
            in1: 0,
            in0_generation: 1,
            in1_generation: 0,
            out_generation: 2,
            aux_offset: 0,
            aux_count: 1,
            left_key: 0,
            right_key: 0,
            scan_delta: 0,
            filter_delta: 0,
            schema_winner_head: 0,
            schema_winner_id: 0,
            reserved: 0,
        };
        let wave = super::ResidentWaveDescriptor {
            first_op: 0,
            op_count: 1,
            flags: 0,
            reserved: 0,
        };
        let region = super::ResidentRegionDescriptor {
            first_wave: 0,
            wave_count: 1,
            iteration_limit: 1,
            op_id: 399,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
            first_slot: 0,
            slot_count: 2,
            generation_offset: 0,
        };
        let comparisons = [super::ResidentFilterComparisonDescriptor::column_constant(
            0, 5, 4, 0,
        )];
        let schedule = provider
            .prepare_resident_schedule(
                relations,
                &[op],
                &[wave],
                &[region],
                &comparisons,
                &[],
                &[1],
            )
            .expect("prepare overflow schedule");
        let stream = provider
            .device()
            .inner()
            .stream()
            .context()
            .new_stream()
            .expect("overflow stream");
        provider.reset_host_transfer_stats();
        provider.reset_d2h_transfer_count();
        provider.reset_untracked_metadata_dtoh_count();
        let mut graph = provider
            .capture_resident_schedule(schedule, 0, Arc::clone(&stream))
            .expect("overflow capture");
        assert_eq!(graph.node_count().expect("overflow node count"), 1);
        graph.launch().expect("overflow launch");
        let receipt = graph.synchronize_and_observe().expect("overflow receipt");
        let transfers = provider.host_transfer_stats();
        let launch_metadata = provider.host_launch_metadata_transfer_stats();
        assert_eq!(transfers.htod_calls + transfers.dtoh_calls, 0);
        assert_eq!(transfers.htod_bytes + transfers.dtoh_bytes, 0);
        assert_eq!(launch_metadata.htod_calls, 0);
        assert_eq!(launch_metadata.htod_bytes, 0);
        assert_eq!(provider.d2h_transfer_count(), 0);
        assert_eq!(provider.untracked_metadata_dtoh_count(), 0);

        assert_eq!(
            receipt.status.code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(receipt.status.op_id, 301);
        assert_eq!(
            receipt.status.resource_code,
            ResidentResourceCode::OutputRows as u32
        );
        assert_eq!(receipt.status.required, 4);
        assert_eq!(receipt.status.capacity, 2);
        assert_eq!(receipt.counts, vec![0]);

        let output = graph.relation(1).expect("overflow output relation");
        for (column, expected) in sentinel_columns.iter().enumerate() {
            let raw: Vec<u8> = provider
                .device()
                .inner()
                .dtoh_sync_copy(output.column(column).expect("output column"))
                .expect("raw output storage");
            let expected: Vec<u8> = if types[column].size_bytes() == 4 {
                expected
                    .iter()
                    .flat_map(|value| (*value as u32).to_le_bytes())
                    .collect()
            } else {
                expected
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            };
            assert_eq!(raw, expected, "column {column} was written on overflow");
        }
    }
}
