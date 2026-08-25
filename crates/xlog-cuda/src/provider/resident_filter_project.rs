//! Fixed-address filter and projection descriptors for resident CUDA graphs.

use xlog_core::{Result, ScalarType, Schema, XlogError};

use crate::cuda_compat::{AsKernelParam, LaunchAsync, LaunchConfig};
use crate::launch::LaunchRecorder;
use crate::memory::{GpuMemoryReservation, TrackedCudaSlice};
use crate::{CudaBuffer, CudaStream};
use cudarc::driver::sys;

use super::resident_relational::{ResidentConvergenceControl, ResidentRelation};
use super::{CompareOp, CudaKernelProvider};

const RESIDENT_FILTER_MAX_ROWS: u64 = 65_536;
const RESIDENT_MAX_ARITY: usize = 17;
const MODULE: &str = "xlog_resident_filter_project";
const BLOCK_SIZE: u32 = 256;

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResidentFilterComparisonDescriptor {
    left_kind: u32,
    left_column: u32,
    right_kind: u32,
    right_column: u32,
    op: u32,
    width: u32,
    reserved_zero: u32,
    reserved_one: u32,
    left_constant: u64,
    right_constant: u64,
}

// SAFETY: the descriptor has a stable C layout, owns no references, and every
// field accepts every bit pattern.
unsafe impl cudarc::driver::DeviceRepr for ResidentFilterComparisonDescriptor {}

#[repr(C, align(8))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct ResidentProjectDescriptor {
    kind: u32,
    column: u32,
    width: u32,
    reserved: u32,
    constant: u64,
}

// SAFETY: the descriptor has a stable C layout, owns no references, and every
// field accepts every bit pattern.
unsafe impl cudarc::driver::DeviceRepr for ResidentProjectDescriptor {}

/// A scalar value supported by the resident filter and projection ABI.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentScalar {
    Symbol(u32),
    U32(u32),
    U64(u64),
}

impl ResidentScalar {
    fn scalar_type(self) -> ScalarType {
        match self {
            Self::Symbol(_) => ScalarType::Symbol,
            Self::U32(_) => ScalarType::U32,
            Self::U64(_) => ScalarType::U64,
        }
    }

    fn bits(self) -> u64 {
        match self {
            Self::Symbol(value) | Self::U32(value) => u64::from(value),
            Self::U64(value) => value,
        }
    }
}

/// One side of a resident filter comparison.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentFilterOperand {
    Column(usize),
    Constant(ResidentScalar),
}

/// One comparison in a flattened conjunction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentFilterComparison {
    left: ResidentFilterOperand,
    op: CompareOp,
    right: ResidentFilterOperand,
}

impl ResidentFilterComparison {
    pub fn new(left: ResidentFilterOperand, op: CompareOp, right: ResidentFilterOperand) -> Self {
        Self { left, op, right }
    }

    pub fn op(&self) -> CompareOp {
        self.op
    }
}

/// One fixed-width output expression for resident projection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResidentProjectExpr {
    Column(usize),
    Constant(ResidentScalar),
}

/// Cold-path allocations reused by every captured execution of one filter.
pub struct ResidentFilterWorkspace {
    comparisons: TrackedCudaSlice<ResidentFilterComparisonDescriptor>,
    mask: TrackedCudaSlice<u32>,
    prefix: TrackedCudaSlice<u32>,
    block_sums: TrackedCudaSlice<u32>,
    block_offsets: TrackedCudaSlice<u32>,
    input_schema: Schema,
    capacity: u32,
    block_count: u32,
    comparison_count: u32,
}

/// Immutable descriptors retained for one captured filter occurrence.
pub struct ResidentFilterDescriptorWorkspace {
    comparisons: TrackedCudaSlice<ResidentFilterComparisonDescriptor>,
    input_schema: Schema,
    capacity: u32,
    comparison_count: u32,
}

impl ResidentFilterDescriptorWorkspace {
    /// Declare the immutable descriptor allocation to the enclosing transaction.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.comparisons);
    }
}

/// Mutable prefix-scan storage shared by sequential captured filters.
pub struct ResidentFilterScratch {
    mask: TrackedCudaSlice<u32>,
    prefix: TrackedCudaSlice<u32>,
    block_sums: TrackedCudaSlice<u32>,
    block_offsets: TrackedCudaSlice<u32>,
    capacity: u32,
    block_count: u32,
}

impl ResidentFilterScratch {
    /// Declare the shared mutable scratch allocation to the enclosing transaction.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read_write(&self.mask);
        recorder.read_write(&self.prefix);
        recorder.read_write(&self.block_sums);
        recorder.read_write(&self.block_offsets);
    }

    pub(crate) fn schedule_parts(&self) -> (u64, u64, u64, u64, u32, u32) {
        (
            self.mask.device_ptr_value(),
            self.prefix.device_ptr_value(),
            self.block_sums.device_ptr_value(),
            self.block_offsets.device_ptr_value(),
            self.capacity,
            self.block_count,
        )
    }

    pub(crate) fn schedule_owner_snapshots(
        &self,
    ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 4]> {
        Ok([
            self.mask.runtime_allocation_identity()?,
            self.prefix.runtime_allocation_identity()?,
            self.block_sums.runtime_allocation_identity()?,
            self.block_offsets.runtime_allocation_identity()?,
        ])
    }
}

impl ResidentFilterWorkspace {
    /// Declare every workspace allocation to the enclosing strict transaction.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.comparisons);
        recorder.read_write(&self.mask);
        recorder.read_write(&self.prefix);
        recorder.read_write(&self.block_sums);
        recorder.read_write(&self.block_offsets);
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

#[repr(C, align(8))]
#[derive(Clone, Copy)]
struct ResidentFilterRelationView {
    columns: [u64; RESIDENT_MAX_ARITY],
    widths: [u32; RESIDENT_MAX_ARITY],
    arity: u32,
    capacity: u32,
    reserved: u32,
    num_rows: u64,
}

impl crate::cuda_compat::AsKernelParam for ResidentFilterRelationView {
    fn as_kernel_param(&self) -> *mut std::ffi::c_void {
        (self as *const Self).cast_mut().cast()
    }
}

fn launch_config(capacity: u32) -> LaunchConfig {
    LaunchConfig {
        grid_dim: (capacity.max(1).div_ceil(BLOCK_SIZE), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn reset_output_count(output: &ResidentRelation, stream: &CudaStream) -> Result<()> {
    // SAFETY: the destination is the live, four-byte device row-count owned by
    // `output`; the asynchronous memset is ordered on the captured stream.
    let code = unsafe {
        sys::cuMemsetD8Async(
            output.num_rows_device().device_ptr_value(),
            0,
            std::mem::size_of::<u32>(),
            stream.cu_stream(),
        )
    };
    if code != sys::cudaError_enum::CUDA_SUCCESS {
        return Err(XlogError::Kernel(format!(
            "resident output count reset failed: {code:?}"
        )));
    }
    Ok(())
}

/// Cold-path immutable projection descriptors retained for graph lifetime.
pub struct ResidentProjectWorkspace {
    descriptors: TrackedCudaSlice<ResidentProjectDescriptor>,
    input_schema: Schema,
    output_schema: Schema,
    input_capacity: u32,
    expression_count: u32,
}

impl ResidentProjectWorkspace {
    /// Declare immutable projection metadata to the enclosing transaction.
    pub fn record_uses(&self, recorder: &mut LaunchRecorder) {
        recorder.read(&self.descriptors);
    }
}

fn supported_width(scalar_type: ScalarType) -> Result<u32> {
    match scalar_type {
        ScalarType::Symbol | ScalarType::U32 => Ok(4),
        ScalarType::U64 => Ok(8),
        unsupported => Err(XlogError::Kernel(format!(
            "resident filter/project type {unsupported:?} is unsupported; expected Symbol, U32, or U64"
        ))),
    }
}

fn checked_relation_view(buffer: &CudaBuffer) -> Result<ResidentFilterRelationView> {
    if buffer.arity() > RESIDENT_MAX_ARITY {
        return Err(XlogError::Kernel(format!(
            "resident filter/project arity {} exceeds {RESIDENT_MAX_ARITY}",
            buffer.arity()
        )));
    }
    let capacity = u32::try_from(buffer.num_rows()).map_err(|_| {
        XlogError::Kernel(format!(
            "resident filter/project capacity {} exceeds u32::MAX",
            buffer.num_rows()
        ))
    })?;
    let mut columns = [0; RESIDENT_MAX_ARITY];
    let mut widths = [0; RESIDENT_MAX_ARITY];
    for column in 0..buffer.arity() {
        columns[column] = *buffer.column(column).expect("arity checked").device_ptr();
        widths[column] = supported_width(
            buffer
                .schema()
                .column_type(column)
                .expect("schema arity checked"),
        )?;
    }
    Ok(ResidentFilterRelationView {
        columns,
        widths,
        arity: buffer.arity() as u32,
        capacity,
        reserved: 0,
        num_rows: buffer.num_rows_device().device_ptr_value(),
    })
}

fn same_physical_layout(left: &Schema, right: &Schema) -> bool {
    left.arity() == right.arity()
        && (0..left.arity()).all(|column| left.column_type(column) == right.column_type(column))
}

fn checked_filter_capacity(capacity: u64) -> Result<u32> {
    if capacity > RESIDENT_FILTER_MAX_ROWS {
        return Err(XlogError::Kernel(format!(
            "resident filter capacity {capacity} exceeds the checked two-level scan limit {RESIDENT_FILTER_MAX_ROWS}"
        )));
    }
    Ok(capacity as u32)
}

/// Exact manager-tracked bytes for one filter's immutable descriptor table.
pub fn resident_filter_descriptor_device_bytes(comparison_count: usize) -> Result<u64> {
    u64::try_from(comparison_count.max(1))
        .ok()
        .and_then(|count| {
            count.checked_mul(std::mem::size_of::<ResidentFilterComparisonDescriptor>() as u64)
        })
        .ok_or_else(|| XlogError::Kernel("resident filter descriptor byte overflow".into()))
}

/// Exact manager-tracked bytes for the shared filter prefix-scan scratch.
pub fn resident_filter_scratch_device_bytes(capacity: u64) -> Result<u64> {
    let capacity = u64::from(checked_filter_capacity(capacity)?).max(1);
    let blocks = capacity.div_ceil(u64::from(BLOCK_SIZE));
    capacity
        .checked_mul(8)
        .and_then(|bytes| {
            blocks
                .checked_mul(8)
                .and_then(|tail| bytes.checked_add(tail))
        })
        .ok_or_else(|| XlogError::Kernel("resident filter scratch byte overflow".into()))
}

/// Exact manager-tracked bytes for one projection descriptor table.
pub fn resident_project_descriptor_device_bytes(expression_count: usize) -> Result<u64> {
    u64::try_from(expression_count.max(1))
        .ok()
        .and_then(|count| {
            count.checked_mul(std::mem::size_of::<ResidentProjectDescriptor>() as u64)
        })
        .ok_or_else(|| XlogError::Kernel("resident project descriptor byte overflow".into()))
}

fn operand_encoding(
    schema: &Schema,
    operand: ResidentFilterOperand,
) -> Result<(u32, u32, ScalarType, u64)> {
    match operand {
        ResidentFilterOperand::Column(column) => {
            let scalar_type = schema.column_type(column).ok_or_else(|| {
                XlogError::Kernel(format!(
                    "resident filter column {column} is outside input arity {}",
                    schema.arity()
                ))
            })?;
            supported_width(scalar_type)?;
            let column = u32::try_from(column)
                .map_err(|_| XlogError::Kernel("resident filter column exceeds u32".into()))?;
            Ok((0, column, scalar_type, 0))
        }
        ResidentFilterOperand::Constant(value) => {
            let scalar_type = value.scalar_type();
            supported_width(scalar_type)?;
            Ok((1, 0, scalar_type, value.bits()))
        }
    }
}

fn encode_filter_comparisons(
    schema: &Schema,
    comparisons: &[ResidentFilterComparison],
) -> Result<Vec<ResidentFilterComparisonDescriptor>> {
    if schema.arity() > RESIDENT_MAX_ARITY {
        return Err(XlogError::Kernel(format!(
            "resident filter input arity {} exceeds {RESIDENT_MAX_ARITY}",
            schema.arity()
        )));
    }
    let comparison_count = u32::try_from(comparisons.len())
        .map_err(|_| XlogError::Kernel("resident filter comparison count exceeds u32".into()))?;
    let mut encoded = Vec::with_capacity(comparison_count as usize);
    for comparison in comparisons {
        let (left_kind, left_column, left_type, left_constant) =
            operand_encoding(schema, comparison.left)?;
        let (right_kind, right_column, right_type, right_constant) =
            operand_encoding(schema, comparison.right)?;
        if left_type != right_type {
            return Err(XlogError::Kernel(format!(
                "resident filter comparison type mismatch: {left_type:?} versus {right_type:?}"
            )));
        }
        encoded.push(ResidentFilterComparisonDescriptor {
            left_kind,
            left_column,
            right_kind,
            right_column,
            op: u32::from(comparison.op as u8),
            width: supported_width(left_type)?,
            reserved_zero: 0,
            reserved_one: 0,
            left_constant,
            right_constant,
        });
    }
    Ok(encoded)
}

fn encode_project_expressions(
    input_schema: &Schema,
    output_schema: &Schema,
    expressions: &[ResidentProjectExpr],
) -> Result<Vec<ResidentProjectDescriptor>> {
    if input_schema.arity() > RESIDENT_MAX_ARITY || output_schema.arity() > RESIDENT_MAX_ARITY {
        return Err(XlogError::Kernel(format!(
            "resident projection arity exceeds {RESIDENT_MAX_ARITY}"
        )));
    }
    if output_schema.arity() != expressions.len() {
        return Err(XlogError::Kernel(format!(
            "resident projection has {} expressions for output arity {}",
            expressions.len(),
            output_schema.arity()
        )));
    }
    let mut encoded = Vec::with_capacity(expressions.len());
    for (output_column, expression) in expressions.iter().copied().enumerate() {
        let output_type = output_schema
            .column_type(output_column)
            .expect("output arity checked");
        let output_width = supported_width(output_type)?;
        let (kind, column, expression_type, constant) = match expression {
            ResidentProjectExpr::Column(column) => {
                let scalar_type = input_schema.column_type(column).ok_or_else(|| {
                    XlogError::Kernel(format!(
                        "resident projection column {column} is outside input arity {}",
                        input_schema.arity()
                    ))
                })?;
                let column = u32::try_from(column).map_err(|_| {
                    XlogError::Kernel("resident projection column exceeds u32".into())
                })?;
                (0, column, scalar_type, 0)
            }
            ResidentProjectExpr::Constant(value) => (1, 0, value.scalar_type(), value.bits()),
        };
        if expression_type != output_type {
            return Err(XlogError::Kernel(format!(
                "resident projection output column {output_column} has type {output_type:?}, but its expression has type {expression_type:?}"
            )));
        }
        encoded.push(ResidentProjectDescriptor {
            kind,
            column,
            width: output_width,
            reserved: 0,
            constant,
        });
    }
    Ok(encoded)
}

impl CudaKernelProvider {
    /// Allocate and populate all filter metadata and scan scratch before capture.
    pub fn prepare_resident_filter_workspace(
        &self,
        input: &CudaBuffer,
        comparisons: &[ResidentFilterComparison],
    ) -> Result<ResidentFilterWorkspace> {
        let capacity = checked_filter_capacity(input.num_rows())?;
        let encoded = encode_filter_comparisons(input.schema(), comparisons)?;
        let comparison_count = u32::try_from(encoded.len()).map_err(|_| {
            XlogError::Kernel("resident filter comparison count exceeds u32".into())
        })?;
        let mut device_comparisons = self
            .memory()
            .alloc::<ResidentFilterComparisonDescriptor>(encoded.len().max(1))?;
        if !encoded.is_empty() {
            self.htod_launch_metadata_sync_copy_into(&encoded, &mut device_comparisons)?;
        }
        let scratch_rows = (capacity as usize).max(1);
        let block_count = capacity.max(1).div_ceil(256);
        Ok(ResidentFilterWorkspace {
            comparisons: device_comparisons,
            mask: self.memory().alloc::<u32>(scratch_rows)?,
            prefix: self.memory().alloc::<u32>(scratch_rows)?,
            block_sums: self.memory().alloc::<u32>(block_count as usize)?,
            block_offsets: self.memory().alloc::<u32>(block_count as usize)?,
            input_schema: input.schema().clone(),
            capacity,
            block_count,
            comparison_count,
        })
    }

    /// Allocate only the immutable descriptors for one filter occurrence.
    pub fn prepare_resident_filter_descriptors(
        &self,
        input: &CudaBuffer,
        comparisons: &[ResidentFilterComparison],
    ) -> Result<ResidentFilterDescriptorWorkspace> {
        self.prepare_resident_filter_descriptors_for_schema(
            input.schema(),
            input.num_rows(),
            comparisons,
        )
    }

    /// Allocate immutable descriptors from a certified schema and fixed capacity.
    pub fn prepare_resident_filter_descriptors_for_schema(
        &self,
        input_schema: &Schema,
        capacity: u64,
        comparisons: &[ResidentFilterComparison],
    ) -> Result<ResidentFilterDescriptorWorkspace> {
        self.prepare_resident_filter_descriptors_with_reservation(
            input_schema,
            capacity,
            comparisons,
            None,
        )
    }

    /// Allocate immutable filter descriptors from an admitted transaction.
    pub fn prepare_resident_filter_descriptors_in_reservation(
        &self,
        input_schema: &Schema,
        capacity: u64,
        comparisons: &[ResidentFilterComparison],
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentFilterDescriptorWorkspace> {
        self.prepare_resident_filter_descriptors_with_reservation(
            input_schema,
            capacity,
            comparisons,
            Some(reservation),
        )
    }

    fn prepare_resident_filter_descriptors_with_reservation(
        &self,
        input_schema: &Schema,
        capacity: u64,
        comparisons: &[ResidentFilterComparison],
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentFilterDescriptorWorkspace> {
        let capacity = checked_filter_capacity(capacity)?;
        let encoded = encode_filter_comparisons(input_schema, comparisons)?;
        let comparison_count = u32::try_from(encoded.len()).map_err(|_| {
            XlogError::Kernel("resident filter comparison count exceeds u32".into())
        })?;
        let mut device_comparisons = match reservation.as_mut() {
            Some(reservation) => {
                reservation.alloc::<ResidentFilterComparisonDescriptor>(encoded.len().max(1))?
            }
            None => self
                .memory()
                .alloc::<ResidentFilterComparisonDescriptor>(encoded.len().max(1))?,
        };
        if !encoded.is_empty() {
            self.htod_launch_metadata_sync_copy_into(&encoded, &mut device_comparisons)?;
        }
        Ok(ResidentFilterDescriptorWorkspace {
            comparisons: device_comparisons,
            input_schema: input_schema.clone(),
            capacity,
            comparison_count,
        })
    }

    /// Allocate one mutable prefix-scan workspace for sequential resident filters.
    pub fn prepare_resident_filter_scratch(&self, capacity: u64) -> Result<ResidentFilterScratch> {
        self.prepare_resident_filter_scratch_with_reservation(capacity, None)
    }

    /// Allocate mutable filter scratch from an admitted transaction.
    pub fn prepare_resident_filter_scratch_in_reservation(
        &self,
        capacity: u64,
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentFilterScratch> {
        self.prepare_resident_filter_scratch_with_reservation(capacity, Some(reservation))
    }

    fn prepare_resident_filter_scratch_with_reservation(
        &self,
        capacity: u64,
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentFilterScratch> {
        let capacity = checked_filter_capacity(capacity)?;
        let scratch_rows = (capacity as usize).max(1);
        let block_count = capacity.max(1).div_ceil(256);
        let mask = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(scratch_rows)?,
            None => self.memory().alloc::<u32>(scratch_rows)?,
        };
        let prefix = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(scratch_rows)?,
            None => self.memory().alloc::<u32>(scratch_rows)?,
        };
        let block_sums = match reservation.as_deref_mut() {
            Some(reservation) => reservation.alloc::<u32>(block_count as usize)?,
            None => self.memory().alloc::<u32>(block_count as usize)?,
        };
        let block_offsets = match reservation.as_mut() {
            Some(reservation) => reservation.alloc::<u32>(block_count as usize)?,
            None => self.memory().alloc::<u32>(block_count as usize)?,
        };
        Ok(ResidentFilterScratch {
            mask,
            prefix,
            block_sums,
            block_offsets,
            capacity,
            block_count,
        })
    }

    /// Allocate immutable projection descriptors before graph capture.
    pub fn prepare_resident_project_workspace(
        &self,
        input: &CudaBuffer,
        output_schema: &Schema,
        expressions: &[ResidentProjectExpr],
    ) -> Result<ResidentProjectWorkspace> {
        self.prepare_resident_project_workspace_for_schemas(
            input.schema(),
            input.num_rows(),
            output_schema,
            expressions,
        )
    }

    /// Allocate projection descriptors from certified schemas and fixed capacity.
    pub fn prepare_resident_project_workspace_for_schemas(
        &self,
        input_schema: &Schema,
        input_capacity: u64,
        output_schema: &Schema,
        expressions: &[ResidentProjectExpr],
    ) -> Result<ResidentProjectWorkspace> {
        self.prepare_resident_project_workspace_with_reservation(
            input_schema,
            input_capacity,
            output_schema,
            expressions,
            None,
        )
    }

    /// Allocate projection descriptors from an admitted transaction.
    pub fn prepare_resident_project_workspace_in_reservation(
        &self,
        input_schema: &Schema,
        input_capacity: u64,
        output_schema: &Schema,
        expressions: &[ResidentProjectExpr],
        reservation: &mut GpuMemoryReservation,
    ) -> Result<ResidentProjectWorkspace> {
        self.prepare_resident_project_workspace_with_reservation(
            input_schema,
            input_capacity,
            output_schema,
            expressions,
            Some(reservation),
        )
    }

    fn prepare_resident_project_workspace_with_reservation(
        &self,
        input_schema: &Schema,
        input_capacity: u64,
        output_schema: &Schema,
        expressions: &[ResidentProjectExpr],
        mut reservation: Option<&mut GpuMemoryReservation>,
    ) -> Result<ResidentProjectWorkspace> {
        let input_capacity = u32::try_from(input_capacity).map_err(|_| {
            XlogError::Kernel(format!(
                "resident projection input capacity {} exceeds u32::MAX",
                input_capacity
            ))
        })?;
        let encoded = encode_project_expressions(input_schema, output_schema, expressions)?;
        let expression_count = u32::try_from(encoded.len()).map_err(|_| {
            XlogError::Kernel("resident projection expression count exceeds u32".into())
        })?;
        let mut device_descriptors = match reservation.as_mut() {
            Some(reservation) => {
                reservation.alloc::<ResidentProjectDescriptor>(encoded.len().max(1))?
            }
            None => self
                .memory()
                .alloc::<ResidentProjectDescriptor>(encoded.len().max(1))?,
        };
        if !encoded.is_empty() {
            self.htod_launch_metadata_sync_copy_into(&encoded, &mut device_descriptors)?;
        }
        Ok(ResidentProjectWorkspace {
            descriptors: device_descriptors,
            input_schema: input_schema.clone(),
            output_schema: output_schema.clone(),
            input_capacity,
            expression_count,
        })
    }

    /// Enqueue a stable, fixed-address filter pipeline on the caller's stream.
    pub fn record_resident_filter_on_stream(
        &self,
        input: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentFilterWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        self.record_resident_filter_parts_on_stream(
            input,
            output,
            &workspace.input_schema,
            &workspace.comparisons,
            workspace.comparison_count,
            workspace.capacity,
            workspace.block_count,
            &workspace.mask,
            &workspace.prefix,
            &workspace.block_sums,
            &workspace.block_offsets,
            control,
            op_id,
            stream,
        )
    }

    /// Record a filter using immutable per-op descriptors and shared mutable scratch.
    #[expect(
        clippy::too_many_arguments,
        reason = "descriptor-backed resident filtering keeps immutable descriptors and mutable scratch owners explicit"
    )]
    pub fn record_resident_filter_with_scratch_on_stream(
        &self,
        input: &CudaBuffer,
        output: &ResidentRelation,
        descriptors: &ResidentFilterDescriptorWorkspace,
        scratch: &ResidentFilterScratch,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        if descriptors.capacity != scratch.capacity {
            return Err(XlogError::Kernel(
                "resident filter descriptors and shared scratch capacities differ".into(),
            ));
        }
        self.record_resident_filter_parts_on_stream(
            input,
            output,
            &descriptors.input_schema,
            &descriptors.comparisons,
            descriptors.comparison_count,
            descriptors.capacity,
            scratch.block_count,
            &scratch.mask,
            &scratch.prefix,
            &scratch.block_sums,
            &scratch.block_offsets,
            control,
            op_id,
            stream,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "the shared resident filter launch core receives every already-validated buffer and stream binding explicitly"
    )]
    fn record_resident_filter_parts_on_stream(
        &self,
        input: &CudaBuffer,
        output: &ResidentRelation,
        input_schema: &Schema,
        comparisons_slice: &TrackedCudaSlice<ResidentFilterComparisonDescriptor>,
        comparison_count: u32,
        capacity: u32,
        block_count: u32,
        mask_slice: &TrackedCudaSlice<u32>,
        prefix_slice: &TrackedCudaSlice<u32>,
        block_sums_slice: &TrackedCudaSlice<u32>,
        block_offsets_slice: &TrackedCudaSlice<u32>,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        if !same_physical_layout(input.schema(), input_schema)
            || !same_physical_layout(input.schema(), output.buffer().schema())
        {
            return Err(XlogError::Kernel(
                "resident filter input, output, and prepared schemas differ".into(),
            ));
        }
        if input.num_rows() > u64::from(capacity) {
            return Err(XlogError::Kernel(
                "resident filter input capacity exceeds prepared workspace".into(),
            ));
        }
        let input_view = checked_relation_view(input)?;
        let output_view = checked_relation_view(output.buffer())?;
        let status = control.status_device_ptr();
        reset_output_count(output, stream)?;

        let mask_scan = self
            .device()
            .inner()
            .get_func(MODULE, "resident_filter_mask_scan")
            .ok_or_else(|| XlogError::Kernel("resident_filter_mask_scan kernel missing".into()))?;
        let comparisons = comparisons_slice.device_ptr_value();
        let mask = mask_slice.device_ptr_value();
        let prefix = prefix_slice.device_ptr_value();
        let block_sums = block_sums_slice.device_ptr_value();
        let mut mask_params = vec![
            input_view.as_kernel_param(),
            comparisons.as_kernel_param(),
            comparison_count.as_kernel_param(),
            mask.as_kernel_param(),
            prefix.as_kernel_param(),
            block_sums.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_filter_mask_scan and all
        // pointers remain live for the graph lifetime.
        unsafe { mask_scan.launch_on_stream(stream, launch_config(capacity), &mut mask_params) }
            .map_err(|error| XlogError::Kernel(format!("resident filter mask launch: {error}")))?;

        let scan_blocks = self
            .device()
            .inner()
            .get_func(MODULE, "resident_filter_scan_blocks")
            .ok_or_else(|| {
                XlogError::Kernel("resident_filter_scan_blocks kernel missing".into())
            })?;
        let block_offsets = block_offsets_slice.device_ptr_value();
        let mut scan_params = vec![
            block_sums.as_kernel_param(),
            block_offsets.as_kernel_param(),
            block_count.as_kernel_param(),
        ];
        // SAFETY: the checked 65,536-row limit guarantees at most 256 blocks.
        unsafe {
            scan_blocks.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (BLOCK_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut scan_params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident filter block scan: {error}")))?;

        let add_offsets = self
            .device()
            .inner()
            .get_func(MODULE, "resident_filter_add_offsets")
            .ok_or_else(|| {
                XlogError::Kernel("resident_filter_add_offsets kernel missing".into())
            })?;
        let mut offset_params = vec![
            prefix.as_kernel_param(),
            block_offsets.as_kernel_param(),
            capacity.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_filter_add_offsets.
        unsafe {
            add_offsets.launch_on_stream(stream, launch_config(capacity), &mut offset_params)
        }
        .map_err(|error| XlogError::Kernel(format!("resident filter add offsets: {error}")))?;

        let finalize = self
            .device()
            .inner()
            .get_func(MODULE, "resident_filter_finalize")
            .ok_or_else(|| XlogError::Kernel("resident_filter_finalize kernel missing".into()))?;
        let mut finalize_params = vec![
            input_view.as_kernel_param(),
            mask.as_kernel_param(),
            prefix.as_kernel_param(),
            output_view.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_filter_finalize.
        unsafe {
            finalize.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut finalize_params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident filter finalize: {error}")))?;

        let compact = self
            .device()
            .inner()
            .get_func(MODULE, "resident_filter_compact")
            .ok_or_else(|| XlogError::Kernel("resident_filter_compact kernel missing".into()))?;
        let mut compact_params = vec![
            input_view.as_kernel_param(),
            mask.as_kernel_param(),
            prefix.as_kernel_param(),
            output_view.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_filter_compact.
        unsafe { compact.launch_on_stream(stream, launch_config(capacity), &mut compact_params) }
            .map_err(|error| XlogError::Kernel(format!("resident filter compact: {error}")))
    }

    /// Enqueue count propagation plus direct column/constant projection.
    pub fn record_resident_project_on_stream(
        &self,
        input: &CudaBuffer,
        output: &ResidentRelation,
        workspace: &ResidentProjectWorkspace,
        control: &ResidentConvergenceControl,
        op_id: u32,
        stream: &CudaStream,
    ) -> Result<()> {
        if !same_physical_layout(input.schema(), &workspace.input_schema)
            || !same_physical_layout(output.buffer().schema(), &workspace.output_schema)
        {
            return Err(XlogError::Kernel(
                "resident projection inputs differ from prepared descriptors".into(),
            ));
        }
        if input.num_rows() > u64::from(workspace.input_capacity) {
            return Err(XlogError::Kernel(
                "resident projection input capacity exceeds prepared workspace".into(),
            ));
        }
        let input_view = checked_relation_view(input)?;
        let output_view = checked_relation_view(output.buffer())?;
        let status = control.status_device_ptr();
        reset_output_count(output, stream)?;
        let finalize = self
            .device()
            .inner()
            .get_func(MODULE, "resident_project_finalize")
            .ok_or_else(|| XlogError::Kernel("resident_project_finalize kernel missing".into()))?;
        let mut finalize_params = vec![
            input_view.as_kernel_param(),
            output_view.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_project_finalize.
        unsafe {
            finalize.launch_on_stream(
                stream,
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut finalize_params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident project finalize: {error}")))?;

        let materialize = self
            .device()
            .inner()
            .get_func(MODULE, "resident_project_materialize")
            .ok_or_else(|| {
                XlogError::Kernel("resident_project_materialize kernel missing".into())
            })?;
        let descriptors = workspace.descriptors.device_ptr_value();
        let mut materialize_params = vec![
            input_view.as_kernel_param(),
            descriptors.as_kernel_param(),
            workspace.expression_count.as_kernel_param(),
            output_view.as_kernel_param(),
            status.as_kernel_param(),
            op_id.as_kernel_param(),
        ];
        // SAFETY: parameters exactly match resident_project_materialize.
        unsafe {
            materialize.launch_on_stream(
                stream,
                launch_config(workspace.input_capacity),
                &mut materialize_params,
            )
        }
        .map_err(|error| XlogError::Kernel(format!("resident project materialize: {error}")))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::{
        checked_filter_capacity, encode_filter_comparisons, encode_project_expressions,
        ResidentFilterComparison, ResidentFilterDescriptorWorkspace, ResidentFilterOperand,
        ResidentFilterScratch, ResidentFilterWorkspace, ResidentProjectExpr,
        ResidentProjectWorkspace, ResidentScalar,
    };
    use crate::provider::resident_relational::{
        ResidentConvergenceControl, ResidentJoinKind, ResidentRelation, ResidentResourceCode,
        ResidentTerminalCode, ResidentTerminalStatus,
    };
    use crate::{
        cuda_graph::CapturedCudaGraph, device_runtime::StreamPool, provider::CompareOp, CudaBuffer,
        CudaKernelProvider, CudaStream,
    };
    use xlog_core::{MemoryBudget, Result, ScalarType, Schema};

    #[test]
    fn filter_scratch_exposes_exact_four_allocation_owner_snapshots() {
        let _: fn(
            &ResidentFilterScratch,
        ) -> Result<[Option<crate::memory::RuntimeAllocationIdentity>; 4]> =
            ResidentFilterScratch::schedule_owner_snapshots;
    }

    fn provider() -> Option<CudaKernelProvider> {
        match crate::CudaProviderBuilder::new(0, MemoryBudget::with_limit(512 * 1024 * 1024))
            .build()
        {
            Ok(provider) => Some(provider),
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but resident filter provider setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping resident filter CUDA test: {error}");
                None
            }
        }
    }

    fn mixed_schema() -> Schema {
        Schema::new(vec![
            ("symbol".to_string(), ScalarType::Symbol),
            ("small".to_string(), ScalarType::U32),
            ("wide".to_string(), ScalarType::U64),
        ])
    }

    fn mixed_buffer(provider: &CudaKernelProvider) -> CudaBuffer {
        let symbol: Vec<u8> = [5_u32, 2, 5, 4]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let small: Vec<u8> = [9_u32, 7, 11, 12]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        let wide: Vec<u8> = [100_u64, 200, 300, 400]
            .into_iter()
            .flat_map(u64::to_le_bytes)
            .collect();
        provider
            .create_buffer_from_slices(&[&symbol, &small, &wide], mixed_schema())
            .expect("mixed input")
    }

    fn capture_stream(provider: &CudaKernelProvider) -> Arc<CudaStream> {
        let pool = StreamPool::new(Arc::clone(provider.device()), 1);
        let stream_id = pool.acquire().expect("non-default capture stream");
        pool.resolve(stream_id).expect("capture stream")
    }

    fn named_u32_schema(prefix: &str, arity: usize) -> Schema {
        Schema::new(
            (0..arity)
                .map(|column| (format!("{prefix}_{column}"), ScalarType::U32))
                .collect(),
        )
    }

    fn u32_buffer(provider: &CudaKernelProvider, prefix: &str, columns: &[Vec<u32>]) -> CudaBuffer {
        let encoded: Vec<Vec<u8>> = columns
            .iter()
            .map(|column| {
                column
                    .iter()
                    .flat_map(|value| value.to_le_bytes())
                    .collect()
            })
            .collect();
        let slices: Vec<&[u8]> = encoded.iter().map(Vec::as_slice).collect();
        provider
            .create_buffer_from_slices(&slices, named_u32_schema(prefix, columns.len()))
            .expect("wide u32 input")
    }

    #[test]
    fn filter_and_project_descriptor_api_is_available() {
        let comparison = ResidentFilterComparison::new(
            ResidentFilterOperand::Column(0),
            CompareOp::Eq,
            ResidentFilterOperand::Constant(ResidentScalar::U32(7)),
        );
        let projection = ResidentProjectExpr::Constant(ResidentScalar::Symbol(11));
        assert_eq!(comparison.op(), CompareOp::Eq);
        assert!(matches!(projection, ResidentProjectExpr::Constant(_)));
    }

    #[test]
    fn filter_workspace_limit_is_checked_before_capture() {
        assert_eq!(checked_filter_capacity(65_536).unwrap(), 65_536);
        assert!(checked_filter_capacity(65_537).is_err());
    }

    #[test]
    fn inputs_larger_than_the_prepared_workspace_are_rejected() {
        let Some(provider) = provider() else {
            return;
        };
        let input = u32_buffer(&provider, "input", &[vec![1, 2, 3]]);
        let input_schema = input.schema().clone();
        let output_schema = named_u32_schema("output", 1);
        let output = provider
            .prepare_resident_relation(output_schema.clone(), 2)
            .expect("bounded output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);

        let comparison = ResidentFilterComparison::new(
            ResidentFilterOperand::Column(0),
            CompareOp::Gt,
            ResidentFilterOperand::Constant(ResidentScalar::U32(0)),
        );
        let descriptors = provider
            .prepare_resident_filter_descriptors_for_schema(&input_schema, 2, &[comparison])
            .expect("filter descriptors");
        let scratch = provider
            .prepare_resident_filter_scratch(2)
            .expect("filter scratch");
        let filter_error = provider
            .record_resident_filter_with_scratch_on_stream(
                &input,
                &output,
                &descriptors,
                &scratch,
                &control,
                0,
                &stream,
            )
            .expect_err("an oversized filter input must be rejected before capture");
        assert!(filter_error
            .to_string()
            .contains("input capacity exceeds prepared workspace"));

        let project = provider
            .prepare_resident_project_workspace_for_schemas(
                &input_schema,
                2,
                &output_schema,
                &[ResidentProjectExpr::Column(0)],
            )
            .expect("project descriptors");
        let project_error = provider
            .record_resident_project_on_stream(&input, &output, &project, &control, 1, &stream)
            .expect_err("an oversized project input must be rejected before capture");
        assert!(project_error
            .to_string()
            .contains("projection input capacity exceeds prepared workspace"));
    }

    #[test]
    fn filter_conjunction_encoding_preserves_types_and_operands() {
        let schema = Schema::new(vec![
            ("symbol".to_string(), ScalarType::Symbol),
            ("small".to_string(), ScalarType::U32),
            ("wide".to_string(), ScalarType::U64),
        ]);
        let comparisons = vec![
            ResidentFilterComparison::new(
                ResidentFilterOperand::Column(0),
                CompareOp::Eq,
                ResidentFilterOperand::Constant(ResidentScalar::Symbol(3)),
            ),
            ResidentFilterComparison::new(
                ResidentFilterOperand::Constant(ResidentScalar::U32(7)),
                CompareOp::Lt,
                ResidentFilterOperand::Column(1),
            ),
            ResidentFilterComparison::new(
                ResidentFilterOperand::Column(2),
                CompareOp::Ge,
                ResidentFilterOperand::Constant(ResidentScalar::U64(11)),
            ),
        ];
        let encoded = encode_filter_comparisons(&schema, &comparisons).unwrap();
        assert_eq!(encoded.len(), 3);
        assert_eq!(encoded[0].width, 4);
        assert_eq!(encoded[1].left_constant, 7);
        assert_eq!(encoded[2].width, 8);
        assert_eq!(encoded[2].right_constant, 11);
    }

    #[test]
    fn filter_encoding_rejects_type_mismatch() {
        let schema = Schema::new(vec![("wide".to_string(), ScalarType::U64)]);
        let comparisons = [ResidentFilterComparison::new(
            ResidentFilterOperand::Column(0),
            CompareOp::Eq,
            ResidentFilterOperand::Constant(ResidentScalar::U32(1)),
        )];
        assert!(encode_filter_comparisons(&schema, &comparisons).is_err());
    }

    #[test]
    fn project_encoding_accepts_nullary_and_checks_output_types() {
        let input = Schema::new(vec![("value".to_string(), ScalarType::U64)]);
        let nullary = Schema::new(vec![]);
        assert!(encode_project_expressions(&input, &nullary, &[])
            .unwrap()
            .is_empty());

        let output = Schema::new(vec![
            ("copied".to_string(), ScalarType::U64),
            ("tag".to_string(), ScalarType::Symbol),
        ]);
        let encoded = encode_project_expressions(
            &input,
            &output,
            &[
                ResidentProjectExpr::Column(0),
                ResidentProjectExpr::Constant(ResidentScalar::Symbol(9)),
            ],
        )
        .unwrap();
        assert_eq!(encoded.len(), 2);
        assert_eq!(encoded[0].column, 0);
        assert_eq!(encoded[1].constant, 9);
    }

    #[test]
    fn cold_path_workspace_api_is_available() {
        let _filter: fn(
            &CudaKernelProvider,
            &CudaBuffer,
            &[ResidentFilterComparison],
        ) -> Result<ResidentFilterWorkspace> =
            CudaKernelProvider::prepare_resident_filter_workspace;
        let _project: fn(
            &CudaKernelProvider,
            &CudaBuffer,
            &Schema,
            &[ResidentProjectExpr],
        ) -> Result<ResidentProjectWorkspace> =
            CudaKernelProvider::prepare_resident_project_workspace;
        let _filter_descriptors: fn(
            &CudaKernelProvider,
            &CudaBuffer,
            &[ResidentFilterComparison],
        ) -> Result<ResidentFilterDescriptorWorkspace> =
            CudaKernelProvider::prepare_resident_filter_descriptors;
        let _filter_scratch: fn(&CudaKernelProvider, u64) -> Result<ResidentFilterScratch> =
            CudaKernelProvider::prepare_resident_filter_scratch;
    }

    #[test]
    fn capture_record_api_is_available() {
        let _filter: fn(
            &CudaKernelProvider,
            &CudaBuffer,
            &ResidentRelation,
            &ResidentFilterWorkspace,
            &ResidentConvergenceControl,
            u32,
            &CudaStream,
        ) -> Result<()> = CudaKernelProvider::record_resident_filter_on_stream;
        let _project: fn(
            &CudaKernelProvider,
            &CudaBuffer,
            &ResidentRelation,
            &ResidentProjectWorkspace,
            &ResidentConvergenceControl,
            u32,
            &CudaStream,
        ) -> Result<()> = CudaKernelProvider::record_resident_project_on_stream;
    }

    #[test]
    fn real_cuda_captured_filter_is_stable_and_supports_conjuncts_and_widths() {
        let Some(provider) = provider() else { return };
        let input = mixed_buffer(&provider);
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
        let workspace = provider
            .prepare_resident_filter_workspace(&input, &comparisons)
            .expect("filter workspace");
        let output = provider
            .prepare_resident_relation(mixed_schema(), 4)
            .expect("filter output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_filter_on_stream(
                &input, &output, &workspace, &control, 41, &stream,
            )
        })
        .expect("filter graph capture");
        graph.launch(&stream).expect("filter graph launch");
        stream.synchronize().expect("filter graph sync");
        assert_eq!(
            provider.download_column::<u32>(output.buffer(), 0).unwrap(),
            vec![5, 5]
        );
        assert_eq!(
            provider.download_column::<u32>(output.buffer(), 1).unwrap(),
            vec![9, 11]
        );
        assert_eq!(
            provider.download_column::<u64>(output.buffer(), 2).unwrap(),
            vec![100, 300]
        );
    }

    #[test]
    fn real_cuda_captured_project_copies_columns_constants_and_nullary_count() {
        let Some(provider) = provider() else { return };
        let input = mixed_buffer(&provider);
        let output_schema = Schema::new(vec![
            ("wide".to_string(), ScalarType::U64),
            ("tag".to_string(), ScalarType::Symbol),
        ]);
        let expressions = [
            ResidentProjectExpr::Column(2),
            ResidentProjectExpr::Constant(ResidentScalar::Symbol(77)),
        ];
        let workspace = provider
            .prepare_resident_project_workspace(&input, &output_schema, &expressions)
            .expect("project workspace");
        let output = provider
            .prepare_resident_relation(output_schema, 4)
            .expect("project output");
        let nullary_schema = Schema::new(vec![]);
        let nullary_workspace = provider
            .prepare_resident_project_workspace(&input, &nullary_schema, &[])
            .expect("nullary workspace");
        let nullary = provider
            .prepare_resident_relation(nullary_schema, 4)
            .expect("nullary output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_project_on_stream(
                &input, &output, &workspace, &control, 42, &stream,
            )?;
            provider.record_resident_project_on_stream(
                &input,
                &nullary,
                &nullary_workspace,
                &control,
                43,
                &stream,
            )
        })
        .expect("project graph capture");
        graph.launch(&stream).expect("project graph launch");
        stream.synchronize().expect("project graph sync");
        assert_eq!(
            provider.download_column::<u64>(output.buffer(), 0).unwrap(),
            vec![100, 200, 300, 400]
        );
        assert_eq!(
            provider.download_column::<u32>(output.buffer(), 1).unwrap(),
            vec![77; 4]
        );
        assert_eq!(provider.device_row_count(nullary.buffer()).unwrap(), 4);
    }

    #[test]
    fn real_cuda_arity_seventeen_inner_join_flows_into_project() {
        let Some(provider) = provider() else { return };
        let mut left_columns = vec![vec![707_u32]; 8];
        left_columns[0] = vec![7];
        let mut right_columns = vec![vec![909_u32, 919]; 9];
        right_columns[0] = vec![7, 7];
        right_columns[8] = vec![908, 918];
        let left = u32_buffer(&provider, "left", &left_columns);
        let right = u32_buffer(&provider, "right", &right_columns);
        let mut join_columns = left.schema().columns.clone();
        join_columns.extend(right.schema().columns.iter().cloned());
        let join_output = provider
            .prepare_resident_relation(Schema::new(join_columns), 2)
            .expect("arity-seventeen join output");
        let join_workspace = provider
            .prepare_resident_join_workspace(2)
            .expect("join workspace");
        let projected_schema = Schema::new(vec![
            ("left_value".to_string(), ScalarType::U32),
            ("right_value".to_string(), ScalarType::U32),
        ]);
        let projected_expressions = [
            ResidentProjectExpr::Column(7),
            ResidentProjectExpr::Column(16),
        ];
        let project_workspace = provider
            .prepare_resident_project_workspace(
                join_output.buffer(),
                &projected_schema,
                &projected_expressions,
            )
            .expect("project workspace");
        let projected = provider
            .prepare_resident_relation(projected_schema, 2)
            .expect("project output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_join_on_stream(
                ResidentJoinKind::Inner,
                &left,
                0,
                &right,
                0,
                &join_output,
                &join_workspace,
                &control,
                51,
                &stream,
            )?;
            provider.record_resident_project_on_stream(
                join_output.buffer(),
                &projected,
                &project_workspace,
                &control,
                52,
                &stream,
            )
        })
        .expect("join and project graph capture");
        graph
            .launch(&stream)
            .expect("join and project graph launch");
        stream.synchronize().expect("join and project graph sync");
        let left_values = provider
            .download_column::<u32>(projected.buffer(), 0)
            .expect("projected left values");
        let right_values = provider
            .download_column::<u32>(projected.buffer(), 1)
            .expect("projected right values");
        let mut rows: Vec<_> = left_values.into_iter().zip(right_values).collect();
        rows.sort_unstable();
        assert_eq!(rows, vec![(707, 908), (707, 918)]);
    }

    #[test]
    fn real_cuda_arity_seventeen_join_overflow_clamps_every_output_column() {
        let Some(provider) = provider() else { return };
        let mut left_columns = vec![vec![707_u32]; 8];
        left_columns[0] = vec![7];
        let mut right_columns = vec![vec![909_u32, 919]; 9];
        right_columns[0] = vec![7, 7];
        let left = u32_buffer(&provider, "left", &left_columns);
        let right = u32_buffer(&provider, "right", &right_columns);
        let mut join_columns = left.schema().columns.clone();
        join_columns.extend(right.schema().columns.iter().cloned());
        let output = provider
            .prepare_resident_relation(Schema::new(join_columns), 1)
            .expect("bounded arity-seventeen output");
        let workspace = provider
            .prepare_resident_join_workspace(2)
            .expect("join workspace");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_join_on_stream(
                ResidentJoinKind::Inner,
                &left,
                0,
                &right,
                0,
                &output,
                &workspace,
                &control,
                53,
                &stream,
            )
        })
        .expect("bounded join graph capture");
        graph.launch(&stream).expect("bounded join graph launch");
        stream.synchronize().expect("bounded join graph sync");
        let status: Vec<ResidentTerminalStatus> = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("status receipt");
        assert_eq!(
            status[0].code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(status[0].op_id, 53);
        assert_eq!(
            status[0].resource_code,
            ResidentResourceCode::OutputRows as u32
        );
        assert_eq!(status[0].required, 2);
        assert_eq!(status[0].capacity, 1);
        assert_eq!(provider.device_row_count(output.buffer()).unwrap(), 1);
        for column in 0..17 {
            assert_eq!(
                provider
                    .download_column::<u32>(output.buffer(), column)
                    .expect("bounded output column")
                    .len(),
                1
            );
        }
    }

    #[test]
    fn resident_relation_rejects_arity_eighteen_on_the_cold_path() {
        let Some(provider) = provider() else { return };
        let Err(error) = provider.prepare_resident_relation(named_u32_schema("too_wide", 18), 1)
        else {
            panic!("arity eighteen must remain outside the resident ABI")
        };
        assert!(error.to_string().contains("arity 18 exceeds 17"), "{error}");
    }

    #[test]
    fn real_cuda_filter_overflow_reports_exact_status_without_oob_write() {
        let Some(provider) = provider() else { return };
        let input = mixed_buffer(&provider);
        let workspace = provider
            .prepare_resident_filter_workspace(&input, &[])
            .expect("filter workspace");
        let output = provider
            .prepare_resident_relation(mixed_schema(), 2)
            .expect("small output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let graph = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_filter_on_stream(
                &input, &output, &workspace, &control, 44, &stream,
            )
        })
        .expect("overflow graph capture");
        graph.launch(&stream).expect("overflow graph launch");
        stream.synchronize().expect("overflow graph sync");
        let status: Vec<ResidentTerminalStatus> = provider
            .device()
            .inner()
            .dtoh_sync_copy(control.status_device())
            .expect("status receipt");
        assert_eq!(
            status[0].code,
            ResidentTerminalCode::CapacityOverflow as u32
        );
        assert_eq!(status[0].op_id, 44);
        assert_eq!(
            status[0].resource_code,
            ResidentResourceCode::OutputRows as u32
        );
        assert_eq!(status[0].required, 4);
        assert_eq!(status[0].capacity, 2);
        assert_eq!(provider.device_row_count(output.buffer()).unwrap(), 0);
    }

    #[test]
    fn real_cuda_overflow_clears_a_reused_output_count() {
        let Some(provider) = provider() else { return };
        let input = mixed_buffer(&provider);
        let small_symbol: Vec<u8> = [1_u32, 2].into_iter().flat_map(u32::to_le_bytes).collect();
        let small_u32: Vec<u8> = [3_u32, 4].into_iter().flat_map(u32::to_le_bytes).collect();
        let small_u64: Vec<u8> = [5_u64, 6].into_iter().flat_map(u64::to_le_bytes).collect();
        let small_input = provider
            .create_buffer_from_slices(&[&small_symbol, &small_u32, &small_u64], mixed_schema())
            .expect("small input");
        let identity = [
            ResidentProjectExpr::Column(0),
            ResidentProjectExpr::Column(1),
            ResidentProjectExpr::Column(2),
        ];
        let project_workspace = provider
            .prepare_resident_project_workspace(&small_input, &mixed_schema(), &identity)
            .expect("identity workspace");
        let filter_workspace = provider
            .prepare_resident_filter_workspace(&input, &[])
            .expect("filter workspace");
        let output = provider
            .prepare_resident_relation(mixed_schema(), 2)
            .expect("reused output");
        let control = provider
            .prepare_resident_convergence_control()
            .expect("control");
        let stream = capture_stream(&provider);
        let seed = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_project_on_stream(
                &small_input,
                &output,
                &project_workspace,
                &control,
                45,
                &stream,
            )
        })
        .expect("seed capture");
        seed.launch(&stream).expect("seed launch");
        stream.synchronize().expect("seed sync");
        assert_eq!(provider.device_row_count(output.buffer()).unwrap(), 2);

        let overflow = CapturedCudaGraph::capture_on_stream(&stream, || {
            provider.record_resident_control_initialize_on_stream(&control, &stream)?;
            provider.record_resident_filter_on_stream(
                &input,
                &output,
                &filter_workspace,
                &control,
                46,
                &stream,
            )
        })
        .expect("overflow capture");
        overflow.launch(&stream).expect("overflow launch");
        stream.synchronize().expect("overflow sync");
        let raw_count: Vec<u32> = provider
            .device()
            .inner()
            .dtoh_sync_copy(output.num_rows_device())
            .expect("raw output count");
        assert_eq!(raw_count, vec![0]);
    }
}
