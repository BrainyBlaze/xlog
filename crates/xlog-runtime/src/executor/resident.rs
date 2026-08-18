use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use std::marker::PhantomData;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{sys::CUevent_flags, CudaEvent};
use xlog_core::{symbol, RelId, Result, ScalarType, Schema, XlogError};
use xlog_cuda::cuda_graph::{
    CapturedCudaGraph, ConditionalCudaGraphSequenceBuilder, CudaConditionalGraphUnavailable,
    CudaGraphNodeKind,
};
use xlog_cuda::device_runtime::{BlockId, ResidentCompletionEvent, XlogDeviceRuntime};
use xlog_cuda::launch::LaunchRecorder;
use xlog_cuda::memory::GpuMemoryReservation;
use xlog_cuda::provider::resident_filter_project::{
    resident_filter_scratch_device_bytes, ResidentFilterScratch,
};
use xlog_cuda::provider::resident_relational::{
    resident_control_device_bytes, resident_device_trace_bytes,
    resident_join_workspace_device_bytes, resident_packed_receipt_with_schema_winners_device_bytes,
    resident_relation_device_bytes, resident_schema_winners_device_bytes,
    resident_set_workspace_device_bytes, ResidentConvergenceControl, ResidentDeviceTrace,
    ResidentJoinKind, ResidentJoinWorkspace, ResidentPackedReceipt, ResidentPinnedReceipt,
    ResidentRelation, ResidentResourceCode, ResidentSchemaWinners, ResidentSetWorkspace,
    ResidentTerminalCode, ResidentTerminalStatus,
};
use xlog_cuda::provider::resident_schedule::{
    resident_schedule_metadata_device_bytes, ResidentExecutionDomain,
    ResidentFilterComparisonDescriptor, ResidentOpDescriptor, ResidentProjectExpressionDescriptor,
    ResidentRegionDescriptor, ResidentScheduleDeviceProgram, ResidentScheduleExternalBindings,
    ResidentScheduleOpKind, ResidentScheduleSlotBinding, ResidentWaveDescriptor,
    RESIDENT_SCHEDULE_OP_MARK_NOVELTY, RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER,
    RESIDENT_SCHEDULE_REGION_FINALIZE, RESIDENT_SCHEDULE_REGION_INITIALIZE,
    RESIDENT_SCHEDULE_REGION_RECURSIVE, RESIDENT_SCHEDULE_REGION_SCC_BEGIN,
};
use xlog_cuda::{
    CompareOp as CudaCompareOp, CudaBuffer, CudaColumn, CudaKernelProvider, CudaStream,
};
use xlog_ir::{
    CompareOp as RirCompareOp, CompiledRule, ConstValue, ExecutionPlan, Expr, JoinType,
    ProjectExpr, RirNode,
};

use super::Executor;
use crate::resident_graph::{
    ResidentGraphCertifiedPlan, ResidentGraphDeclineReason, ResidentGraphDeviceStatus,
    ResidentGraphExecutionError, ResidentGraphPrepareDiagnosticSnapshot,
    ResidentGraphPrepareOptions, ResidentGraphRouteCertificate,
};

const MAX_RESIDENT_CAPACITY: u32 = 65_536;
const RESIDENT_DYNAMIC_SCHEMA_ID: u32 = u32::MAX;

fn resident_schemas_type_compatible(left: &Schema, right: &Schema) -> bool {
    left.arity() == right.arity()
        && (0..left.arity()).all(|column| left.column_type(column) == right.column_type(column))
}

fn resident_intern_schema(candidates: &mut Vec<Schema>, schema: Schema) -> Result<u32> {
    let index = candidates
        .iter()
        .position(|candidate| candidate == &schema)
        .unwrap_or_else(|| {
            candidates.push(schema);
            candidates.len() - 1
        });
    u32::try_from(index)
        .map_err(|_| XlogError::Execution("resident schema candidate count exceeds u32".into()))
}
#[derive(Debug, Clone)]
pub(super) struct ResidentWorkspaceAdmission {
    pub(super) relation_capacity: u32,
    pub(super) head_schemas: BTreeMap<String, Schema>,
    head_schema_choices: BTreeMap<String, Vec<Schema>>,
    rule_schema_ids: Vec<Vec<Option<u32>>>,
    head_schema_selections: BTreeMap<String, ResidentHeadSchemaSelection>,
}

#[derive(Debug, Clone)]
struct ResidentHeadSchemaSelection {
    source_head: String,
    output_schemas_by_source_winner: Vec<Schema>,
}

#[derive(Debug, Clone)]
enum ResidentSchemaVariants {
    Fixed(Schema),
    Dynamic {
        source_head: String,
        schemas: Vec<Schema>,
    },
}

fn resident_schema_variants_from_source(
    source_head: String,
    schemas: Vec<Schema>,
) -> std::result::Result<ResidentSchemaVariants, ResidentGraphDeclineReason> {
    let Some(first) = schemas.first() else {
        return Err(resident_workspace_decline(format!(
            "resident schema source {source_head} has no admitted candidate"
        )));
    };
    if schemas.iter().all(|schema| schema == first) {
        Ok(ResidentSchemaVariants::Fixed(first.clone()))
    } else {
        Ok(ResidentSchemaVariants::Dynamic {
            source_head,
            schemas,
        })
    }
}

fn resident_register_schema_selection(
    target_head: &str,
    selection: ResidentHeadSchemaSelection,
    head_schema_choices: &BTreeMap<String, Vec<Schema>>,
    head_schema_selections: &mut BTreeMap<String, ResidentHeadSchemaSelection>,
) -> std::result::Result<(), ResidentGraphDeclineReason> {
    if selection.source_head == target_head {
        return Err(resident_workspace_decline(format!(
            "resident schema lineage for {target_head} contains a cycle"
        )));
    }
    let source_candidates = head_schema_choices
        .get(&selection.source_head)
        .filter(|candidates| !candidates.is_empty())
        .ok_or_else(|| {
            resident_workspace_decline(format!(
                "resident schema source {} for {target_head} has no admitted candidate",
                selection.source_head
            ))
        })?;
    if source_candidates.len() != selection.output_schemas_by_source_winner.len() {
        return Err(resident_workspace_decline(format!(
            "resident schema source {} for {target_head} has {} candidates but {} mappings",
            selection.source_head,
            source_candidates.len(),
            selection.output_schemas_by_source_winner.len()
        )));
    }

    let mut source = selection.source_head.as_str();
    let mut seen = HashSet::new();
    while let Some(parent) = head_schema_selections.get(source) {
        if source == target_head || !seen.insert(source) {
            return Err(resident_workspace_decline(format!(
                "resident schema lineage for {target_head} contains a cycle"
            )));
        }
        source = parent.source_head.as_str();
    }
    if source == target_head {
        return Err(resident_workspace_decline(format!(
            "resident schema lineage for {target_head} contains a cycle"
        )));
    }

    if let Some(existing) = head_schema_selections.get(target_head) {
        if existing.source_head != selection.source_head
            || existing.output_schemas_by_source_winner != selection.output_schemas_by_source_winner
        {
            return Err(resident_workspace_decline(format!(
                "resident head {target_head} has multiple schema sources or mappings"
            )));
        }
        return Ok(());
    }
    head_schema_selections.insert(target_head.to_string(), selection);
    Ok(())
}

#[derive(Debug, Clone)]
enum ResidentOutputSchemaSelection {
    OwnWinner,
    SourceWinner {
        source_output: usize,
        schemas: Vec<Schema>,
    },
}

#[derive(Debug, Clone)]
struct ResidentOutputSchemaPlan {
    candidates: Vec<Schema>,
    selection: ResidentOutputSchemaSelection,
}

fn resident_resolve_output_schemas(
    plans: &[ResidentOutputSchemaPlan],
    winner_ids: &[u32],
) -> Result<Vec<Schema>> {
    if plans.len() != winner_ids.len() {
        return Err(XlogError::Execution(format!(
            "resident schema plan count {} does not match winner count {}",
            plans.len(),
            winner_ids.len()
        )));
    }
    (0..plans.len())
        .map(|output| {
            resident_resolve_output_schema(plans, winner_ids, output, &mut HashSet::new())
        })
        .collect()
}

fn resident_resolve_output_schema(
    plans: &[ResidentOutputSchemaPlan],
    winner_ids: &[u32],
    output: usize,
    resolving: &mut HashSet<usize>,
) -> Result<Schema> {
    if !resolving.insert(output) {
        return Err(XlogError::Execution(
            "resident schema source lineage contains a cycle".into(),
        ));
    }
    let result = (|| {
        let plan = plans.get(output).ok_or_else(|| {
            XlogError::Execution(format!("resident schema output {output} is missing"))
        })?;
        let winner = *winner_ids.get(output).ok_or_else(|| {
            XlogError::Execution(format!("resident schema winner {output} is missing"))
        })?;
        if winner != RESIDENT_DYNAMIC_SCHEMA_ID {
            return plan
                .candidates
                .get(winner as usize)
                .cloned()
                .ok_or_else(|| {
                    XlogError::Execution(format!(
                        "resident schema winner {winner} exceeds {} candidates",
                        plan.candidates.len()
                    ))
                });
        }
        let ResidentOutputSchemaSelection::SourceWinner {
            source_output,
            schemas,
        } = &plan.selection
        else {
            return Err(XlogError::Execution(
                "resident dynamic schema winner has no source lineage".into(),
            ));
        };
        let source_schema =
            resident_resolve_output_schema(plans, winner_ids, *source_output, resolving)?;
        let source_plan = plans
            .get(*source_output)
            .ok_or_else(|| XlogError::Execution("resident schema source plan is missing".into()))?;
        let source_winner = source_plan
            .candidates
            .iter()
            .position(|candidate| candidate == &source_schema)
            .ok_or_else(|| {
                XlogError::Execution(
                    "resident resolved source schema has no candidate index".into(),
                )
            })?;
        schemas.get(source_winner).cloned().ok_or_else(|| {
            XlogError::Execution(format!(
                "resident schema source winner {source_winner} exceeds {} mappings",
                schemas.len()
            ))
        })
    })();
    resolving.remove(&output);
    result
}

struct ResidentRunOwners {
    provider: Arc<CudaKernelProvider>,
    runtime: Arc<XlogDeviceRuntime>,
    stream: Arc<CudaStream>,
    // Field drop order is intentional: destroy the graph before its compact
    // program/domain and every externally owned device allocation.
    graph: CapturedCudaGraph,
    execution_domain: ResidentExecutionDomain,
    schedule_program: ResidentScheduleDeviceProgram,
    recorder: LaunchRecorder,
    relations: Vec<Option<ResidentRelation>>,
    output_indices: Vec<(String, usize)>,
    filter_scratch: Option<ResidentFilterScratch>,
    set_workspace: ResidentSetWorkspace,
    join_workspace: ResidentJoinWorkspace,
    control: ResidentConvergenceControl,
    device_trace: ResidentDeviceTrace,
    schema_winners: ResidentSchemaWinners,
    receipt: ResidentPackedReceipt,
    pinned_receipt: ResidentPinnedReceipt,
    source_epoch: u64,
    relation_registration: Vec<(RelId, String)>,
    transaction_identity: Arc<()>,
    output_schema_plans: Vec<ResidentOutputSchemaPlan>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct ResidentSourceSetSnapshot {
    name: String,
    version: u64,
    schema: Schema,
    row_capacity: u64,
    column_blocks: Vec<Option<BlockId>>,
    row_count_block: BlockId,
}

#[derive(Clone)]
enum ResidentBufferRef {
    Source(String),
    Private(usize),
}

enum ResidentRecordedOp {
    Unit {
        output: usize,
        op_id: u32,
    },
    Scan {
        relation: ResidentBufferRef,
        op_id: u32,
    },
    TraceDelta {
        scan_delta: u32,
        filter_delta: u32,
        semantic_guard: Option<ResidentBufferRef>,
    },
    Clear {
        output: usize,
    },
    Filter {
        input: ResidentBufferRef,
        output: usize,
        workspace: usize,
        op_id: u32,
    },
    Project {
        input: ResidentBufferRef,
        output: usize,
        workspace: usize,
        op_id: u32,
    },
    Union {
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        output: usize,
        op_id: u32,
    },
    Diff {
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        output: usize,
        op_id: u32,
    },
    Join {
        kind: ResidentJoinKind,
        left: ResidentBufferRef,
        left_key: usize,
        right: ResidentBufferRef,
        right_key: usize,
        output: usize,
        op_id: u32,
    },
    ChangedReset,
    ChangedMark {
        relation: usize,
    },
    TestStatus(ResidentTerminalStatus),
    SchemaWinnerMark {
        contribution: ResidentBufferRef,
        head_index: u32,
        schema_id: u32,
    },
}

fn resident_record_unit_leaf(
    output: usize,
    op_id: u32,
    ops: &mut Vec<ResidentRecordedOp>,
    push_physical: impl FnOnce(&mut Vec<ResidentRecordedOp>, ResidentRecordedOp, u32),
) -> ResidentBufferRef {
    push_physical(ops, ResidentRecordedOp::Unit { output, op_id }, op_id);
    ResidentBufferRef::Private(output)
}

fn resident_new_phase_unit(
    relations: &mut Vec<ResidentLogicalRelation>,
    op_id: u32,
) -> Result<(ResidentBufferRef, ResidentRecordedOp)> {
    let output = relations.len();
    relations.push(ResidentLogicalRelation {
        schema: Schema::new(Vec::new()),
        initial_count: 0,
        permanent: false,
    });
    Ok((
        ResidentBufferRef::Private(output),
        ResidentRecordedOp::Unit { output, op_id },
    ))
}

fn resident_record_scan_leaf(
    relation: ResidentBufferRef,
    op_id: u32,
    semantic_guard: Option<ResidentBufferRef>,
    ops: &mut Vec<ResidentRecordedOp>,
    push_physical: impl FnOnce(&mut Vec<ResidentRecordedOp>, ResidentRecordedOp, u32),
) -> ResidentBufferRef {
    push_physical(
        ops,
        ResidentRecordedOp::Scan {
            relation: relation.clone(),
            op_id,
        },
        op_id,
    );
    ops.push(ResidentRecordedOp::TraceDelta {
        scan_delta: 1,
        filter_delta: 0,
        semantic_guard,
    });
    relation
}

fn resident_semantic_trace_guard(
    override_scan: Option<(RelId, usize, usize)>,
) -> Option<ResidentBufferRef> {
    override_scan.map(|(_, _, delta)| ResidentBufferRef::Private(delta))
}

fn resident_record_schema_winner_mark(
    ops: &mut Vec<ResidentRecordedOp>,
    contribution: ResidentBufferRef,
    head_index: u32,
    schema_id: u32,
) {
    ops.push(ResidentRecordedOp::SchemaWinnerMark {
        contribution,
        head_index,
        schema_id,
    });
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentUnionFoldMode {
    SelfUnion,
    LeftAssociated,
}

fn resident_union_fold_mode(input_count: usize) -> Result<ResidentUnionFoldMode> {
    match input_count {
        0 => Err(XlogError::Execution("resident union has no inputs".into())),
        1 => Ok(ResidentUnionFoldMode::SelfUnion),
        _ => Ok(ResidentUnionFoldMode::LeftAssociated),
    }
}

enum ResidentPhaseMergeStep<T> {
    Deduplicate(T),
    Union(T, T),
}

fn resident_phase_merge<T, E>(
    current: Option<T>,
    contribution: T,
    mut execute: impl FnMut(ResidentPhaseMergeStep<T>) -> std::result::Result<T, E>,
) -> std::result::Result<T, E> {
    let contribution = execute(ResidentPhaseMergeStep::Deduplicate(contribution))?;
    match current {
        Some(current) => execute(ResidentPhaseMergeStep::Union(current, contribution)),
        None => Ok(contribution),
    }
}

enum ResidentCapturePhase {
    Segment {
        ops: Vec<ResidentRecordedOp>,
        scc_begin: Option<(u32, u32)>,
    },
    ConditionalWhile {
        ops: Vec<ResidentRecordedOp>,
        iteration_limit: u32,
        convergence_op_id: u32,
    },
}

struct ResidentCompactLogicalRegion {
    ops: Vec<ResidentRecordedOp>,
    iteration_limit: u32,
    op_id: u32,
    flags: u32,
}

struct ResidentCompactSchedulePlan {
    source_slots: BTreeMap<String, u32>,
    ops: Vec<ResidentOpDescriptor>,
    waves: Vec<ResidentWaveDescriptor>,
    regions: Vec<ResidentRegionDescriptor>,
    generation_bases: Vec<u32>,
    filter_comparisons: Vec<ResidentFilterComparisonDescriptor>,
    project_expressions: Vec<ResidentProjectExpressionDescriptor>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentCaptureParentKind {
    Kernel,
    Conditional,
}

struct ResidentCompactTopology {
    parent_kinds: Vec<ResidentCaptureParentKind>,
    conditional_body_kernel_counts: Vec<usize>,
    hierarchical_node_count: usize,
}

trait ResidentRegionFlags {
    fn region_flags(&self) -> u32;
}

impl ResidentRegionFlags for ResidentCompactLogicalRegion {
    fn region_flags(&self) -> u32 {
        self.flags
    }
}

impl ResidentRegionFlags for ResidentRegionDescriptor {
    fn region_flags(&self) -> u32 {
        self.flags
    }
}

fn resident_compact_topology<R: ResidentRegionFlags>(
    regions: &[R],
) -> Result<ResidentCompactTopology> {
    let parent_kinds = regions
        .iter()
        .map(|region| {
            if region.region_flags() == RESIDENT_SCHEDULE_REGION_RECURSIVE {
                ResidentCaptureParentKind::Conditional
            } else {
                ResidentCaptureParentKind::Kernel
            }
        })
        .collect::<Vec<_>>();
    let conditional_count = parent_kinds
        .iter()
        .filter(|kind| **kind == ResidentCaptureParentKind::Conditional)
        .count();
    let hierarchical_node_count = parent_kinds
        .len()
        .checked_add(conditional_count)
        .ok_or_else(|| XlogError::Execution("resident topology node count overflow".into()))?;
    Ok(ResidentCompactTopology {
        parent_kinds,
        conditional_body_kernel_counts: vec![1; conditional_count],
        hierarchical_node_count,
    })
}

fn resident_validate_parent_graph_kinds(
    actual: &[CudaGraphNodeKind],
    expected: &[ResidentCaptureParentKind],
) -> Result<()> {
    if actual.len() != expected.len()
        || actual.iter().zip(expected).any(|(actual, expected)| {
            !matches!(
                (actual, expected),
                (CudaGraphNodeKind::Kernel, ResidentCaptureParentKind::Kernel)
                    | (
                        CudaGraphNodeKind::Conditional,
                        ResidentCaptureParentKind::Conditional
                    )
            )
        })
    {
        return Err(XlogError::Execution(
            "resident parent graph topology differs from the compact schedule".into(),
        ));
    }
    Ok(())
}

fn resident_validate_conditional_body_node_kinds(
    actual: &[Vec<CudaGraphNodeKind>],
    expected_conditional_count: usize,
) -> Result<Vec<usize>> {
    if actual.len() != expected_conditional_count {
        return Err(XlogError::Execution(format!(
            "resident conditional body inventory has {} bodies, expected {expected_conditional_count}",
            actual.len()
        )));
    }
    let mut kernel_counts = Vec::with_capacity(actual.len());
    for (index, kinds) in actual.iter().enumerate() {
        if kinds.as_slice() != [CudaGraphNodeKind::Kernel] {
            return Err(XlogError::Execution(format!(
                "resident conditional body {index} must contain exactly one kernel node, found {kinds:?}"
            )));
        }
        kernel_counts.push(1);
    }
    Ok(kernel_counts)
}

#[derive(Default)]
struct ResidentCompactDescriptorTables {
    filter_comparisons: Vec<ResidentFilterComparisonDescriptor>,
    filter_ranges: Vec<(u32, u32)>,
    project_expressions: Vec<ResidentProjectExpressionDescriptor>,
    project_ranges: Vec<(u32, u32)>,
}

#[allow(clippy::too_many_arguments)]
fn resident_compact_schedule_metadata_bytes(
    slot_count: usize,
    op_count: usize,
    wave_count: usize,
    region_count: usize,
    generation_base_count: usize,
    head_count: usize,
    filter_comparison_count: usize,
    project_expression_count: usize,
) -> Result<u64> {
    let generation_metadata_count =
        generation_base_count
            .checked_add(head_count)
            .ok_or_else(|| {
                XlogError::Execution("resident generation metadata count overflow".into())
            })?;
    resident_schedule_metadata_device_bytes(
        slot_count,
        op_count,
        wave_count,
        region_count,
        generation_metadata_count,
        filter_comparison_count,
        project_expression_count,
    )
}

fn resident_compact_schema_defaults(
    ops: &[ResidentOpDescriptor],
    head_count: usize,
) -> Result<Vec<u32>> {
    let mut defaults = vec![None; head_count];
    for op in ops {
        if op.flags & RESIDENT_SCHEDULE_OP_MARK_SCHEMA_WINNER == 0 {
            continue;
        }
        let head = usize::try_from(op.schema_winner_head).map_err(|_| {
            XlogError::Execution("resident schema winner head index overflow".into())
        })?;
        let default = defaults.get_mut(head).ok_or_else(|| {
            XlogError::Execution("resident schema winner head is out of range".into())
        })?;
        if default.is_none() {
            *default = Some(op.schema_winner_id);
        }
    }
    defaults
        .into_iter()
        .enumerate()
        .map(|(head, default)| {
            default.ok_or_else(|| {
                XlogError::Execution(format!(
                    "resident head {head} has no schema winner candidate"
                ))
            })
        })
        .collect()
}

fn resident_compact_allocation_bytes(
    relation_bytes: u64,
    filter_scratch_bytes: u64,
    fixed_workspace_bytes: u64,
    private_slot_count: usize,
    head_count: usize,
    plan: &ResidentCompactSchedulePlan,
) -> Result<(u64, u64)> {
    let slot_count = private_slot_count
        .checked_add(plan.source_slots.len())
        .ok_or_else(|| XlogError::Execution("resident compact slot count overflow".into()))?;
    let metadata_bytes = resident_compact_schedule_metadata_bytes(
        slot_count,
        plan.ops.len(),
        plan.waves.len(),
        plan.regions.len(),
        plan.generation_bases.len(),
        head_count,
        plan.filter_comparisons.len(),
        plan.project_expressions.len(),
    )?;
    let required_bytes = relation_bytes
        .checked_add(filter_scratch_bytes)
        .and_then(|bytes| bytes.checked_add(fixed_workspace_bytes))
        .and_then(|bytes| bytes.checked_add(metadata_bytes))
        .ok_or_else(|| XlogError::Execution("resident compact manifest overflow".into()))?;
    Ok((required_bytes, metadata_bytes))
}

fn resident_compact_preflight_device_bytes(
    manifest: &ResidentAllocationManifest,
    plan: &ResidentCompactSchedulePlan,
) -> Result<(u64, u64, u64)> {
    let filter_count = u64::try_from(plan.filter_comparisons.len().max(1)).map_err(|_| {
        XlogError::Execution("resident compact filter descriptor count exceeds u64".into())
    })?;
    let project_count = u64::try_from(plan.project_expressions.len().max(1)).map_err(|_| {
        XlogError::Execution("resident compact project descriptor count exceeds u64".into())
    })?;
    let filter_bytes = u64::try_from(std::mem::size_of::<ResidentFilterComparisonDescriptor>())
        .ok()
        .and_then(|size| size.checked_mul(filter_count))
        .ok_or_else(|| {
            XlogError::Execution("resident compact filter descriptor bytes overflow".into())
        })?;
    let project_bytes = u64::try_from(std::mem::size_of::<ResidentProjectExpressionDescriptor>())
        .ok()
        .and_then(|size| size.checked_mul(project_count))
        .ok_or_else(|| {
            XlogError::Execution("resident compact project descriptor bytes overflow".into())
        })?;
    let descriptor_bytes = filter_bytes
        .checked_add(project_bytes)
        .ok_or_else(|| XlogError::Execution("resident compact descriptor bytes overflow".into()))?;
    let remaining_metadata_bytes = manifest
        .schedule_metadata_bytes
        .checked_sub(descriptor_bytes)
        .ok_or_else(|| {
            XlogError::Execution(
                "resident compact descriptor bytes exceed schedule metadata".into(),
            )
        })?;
    let fixed_bytes = manifest
        .fixed_workspace_bytes
        .checked_add(remaining_metadata_bytes)
        .ok_or_else(|| XlogError::Execution("resident compact fixed bytes overflow".into()))?;
    let reported_total = manifest
        .relation_bytes
        .checked_add(manifest.filter_scratch_bytes)
        .and_then(|bytes| bytes.checked_add(filter_bytes))
        .and_then(|bytes| bytes.checked_add(project_bytes))
        .and_then(|bytes| bytes.checked_add(fixed_bytes))
        .ok_or_else(|| XlogError::Execution("resident compact preflight bytes overflow".into()))?;
    if reported_total != manifest.required_bytes {
        return Err(XlogError::Execution(format!(
            "resident compact preflight component mismatch: reported {reported_total}, required {}",
            manifest.required_bytes
        )));
    }
    Ok((filter_bytes, project_bytes, fixed_bytes))
}

fn resident_compact_tables(
    filter_workspaces: &[ResidentFilterPlan],
    project_workspaces: &[ResidentProjectPlan],
) -> Result<ResidentCompactDescriptorTables> {
    let mut tables = ResidentCompactDescriptorTables::default();
    for workspace in filter_workspaces {
        let offset = u32::try_from(tables.filter_comparisons.len()).map_err(|_| {
            XlogError::Execution("resident filter comparison offset exceeds u32".into())
        })?;
        let count = u32::try_from(workspace.compact_comparisons.len()).map_err(|_| {
            XlogError::Execution("resident filter comparison count exceeds u32".into())
        })?;
        tables
            .filter_comparisons
            .extend_from_slice(&workspace.compact_comparisons);
        tables.filter_ranges.push((offset, count));
    }
    for workspace in project_workspaces {
        let offset = u32::try_from(tables.project_expressions.len()).map_err(|_| {
            XlogError::Execution("resident project expression offset exceeds u32".into())
        })?;
        let count = u32::try_from(workspace.compact_expressions.len()).map_err(|_| {
            XlogError::Execution("resident project expression count exceeds u32".into())
        })?;
        tables
            .project_expressions
            .extend_from_slice(&workspace.compact_expressions);
        tables.project_ranges.push((offset, count));
    }
    Ok(tables)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ResidentCompactSlotRef {
    slot: u32,
    generation: u32,
}

fn resident_compact_slot_ref(
    reference: &ResidentBufferRef,
    assignments: &[ResidentSlotAssignment],
    source_slots: &BTreeMap<String, u32>,
) -> Result<ResidentCompactSlotRef> {
    match reference {
        ResidentBufferRef::Private(logical) => {
            let assignment = assignments.get(*logical).ok_or_else(|| {
                XlogError::Execution(format!(
                    "resident logical relation {logical} has no compact slot assignment"
                ))
            })?;
            Ok(ResidentCompactSlotRef {
                slot: u32::try_from(assignment.slot).map_err(|_| {
                    XlogError::Execution("resident compact slot index exceeds u32".into())
                })?,
                generation: assignment.generation,
            })
        }
        ResidentBufferRef::Source(name) => Ok(ResidentCompactSlotRef {
            slot: *source_slots.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "resident source {name} has no compact slot assignment"
                ))
            })?,
            generation: 0,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn resident_lower_compact_regions<'a>(
    logical_regions: Vec<ResidentCompactLogicalRegion>,
    physical_slots: &[ResidentPhysicalSlotPlan],
    assignments: &[ResidentSlotAssignment],
    source_names: impl IntoIterator<Item = &'a str>,
    tables: ResidentCompactDescriptorTables,
) -> Result<ResidentCompactSchedulePlan> {
    let source_slots = resident_source_slot_map(physical_slots.len(), source_names)?;
    let slot_count = physical_slots
        .len()
        .checked_add(source_slots.len())
        .ok_or_else(|| XlogError::Execution("resident compact slot count overflow".into()))?;
    let slot_count_u32 = u32::try_from(slot_count)
        .map_err(|_| XlogError::Execution("resident compact slot count exceeds u32".into()))?;
    let mut ops = Vec::new();
    let mut waves = Vec::new();
    let mut regions = Vec::with_capacity(logical_regions.len());
    let filter_total = u32::try_from(tables.filter_comparisons.len())
        .map_err(|_| XlogError::Execution("resident filter comparison count exceeds u32".into()))?;
    let project_total = u32::try_from(tables.project_expressions.len()).map_err(|_| {
        XlogError::Execution("resident project expression count exceeds u32".into())
    })?;
    let mut filter_cursor = 0_u32;
    let mut project_cursor = 0_u32;
    let mut generation_bases = Vec::with_capacity(
        logical_regions
            .len()
            .checked_mul(slot_count)
            .ok_or_else(|| {
                XlogError::Execution("resident generation baseline count overflow".into())
            })?,
    );

    for logical_region in logical_regions {
        let region_flags = logical_region.flags;
        let first_wave = u32::try_from(waves.len())
            .map_err(|_| XlogError::Execution("resident wave count exceeds u32".into()))?;
        let generation_offset = u32::try_from(generation_bases.len()).map_err(|_| {
            XlogError::Execution("resident generation baseline offset exceeds u32".into())
        })?;
        let mut first_generations = vec![None; slot_count];
        let mut last_relation_descriptor = None::<(usize, ResidentCompactSlotRef)>;

        for (logical_op_index, logical_op) in logical_region.ops.into_iter().enumerate() {
            let mut emit = |descriptor: ResidentOpDescriptor,
                            output: ResidentCompactSlotRef,
                            references: Vec<ResidentCompactSlotRef>|
             -> Result<()> {
                for reference in &references {
                    let generation = first_generations
                        .get_mut(reference.slot as usize)
                        .ok_or_else(|| {
                            XlogError::Execution(
                                "resident compact operation slot is out of range".into(),
                            )
                        })?;
                    if generation.is_none() {
                        *generation = Some(reference.generation);
                    }
                }
                let op_index = ops.len();
                let first_op = u32::try_from(op_index).map_err(|_| {
                    XlogError::Execution("resident compact op count exceeds u32".into())
                })?;
                ops.push(descriptor);
                waves.push(ResidentWaveDescriptor {
                    first_op,
                    op_count: 1,
                    flags: 0,
                    reserved: 0,
                });
                last_relation_descriptor = Some((op_index, output));
                Ok(())
            };

            match logical_op {
                ResidentRecordedOp::Unit { output, op_id } => {
                    let output = resident_compact_slot_ref(
                        &ResidentBufferRef::Private(output),
                        assignments,
                        &source_slots,
                    )?;
                    emit(
                        ResidentOpDescriptor::unit(op_id, output.slot, output.generation),
                        output,
                        vec![output],
                    )?;
                }
                ResidentRecordedOp::Scan { relation, op_id } => {
                    let source = resident_compact_slot_ref(&relation, assignments, &source_slots)?;
                    emit(
                        ResidentOpDescriptor::scan(op_id, source.slot, source.generation),
                        source,
                        vec![source],
                    )?;
                }
                op @ (ResidentRecordedOp::Filter { .. } | ResidentRecordedOp::Project { .. }) => {
                    let (is_filter, input, output, workspace, op_id) = match op {
                        ResidentRecordedOp::Filter {
                            input,
                            output,
                            workspace,
                            op_id,
                        } => (true, input, output, workspace, op_id),
                        ResidentRecordedOp::Project {
                            input,
                            output,
                            workspace,
                            op_id,
                        } => (false, input, output, workspace, op_id),
                        _ => unreachable!("compact filter/project arm"),
                    };
                    let input = resident_compact_slot_ref(&input, assignments, &source_slots)?;
                    let output = resident_compact_slot_ref(
                        &ResidentBufferRef::Private(output),
                        assignments,
                        &source_slots,
                    )?;
                    let (aux_offset, aux_count) = if is_filter {
                        *tables.filter_ranges.get(workspace).ok_or_else(|| {
                            XlogError::Execution(
                                "resident compact filter workspace is out of range".into(),
                            )
                        })?
                    } else {
                        *tables.project_ranges.get(workspace).ok_or_else(|| {
                            XlogError::Execution(
                                "resident compact project workspace is out of range".into(),
                            )
                        })?
                    };
                    let (cursor, total, table_name) = if is_filter {
                        (&mut filter_cursor, filter_total, "filter")
                    } else {
                        (&mut project_cursor, project_total, "project")
                    };
                    if aux_offset != *cursor || aux_offset > total || aux_count > total - aux_offset
                    {
                        return Err(XlogError::Execution(format!(
                            "resident compact {table_name} descriptor range is not contiguous"
                        )));
                    }
                    *cursor = aux_offset + aux_count;
                    emit(
                        ResidentOpDescriptor {
                            kind: if is_filter {
                                ResidentScheduleOpKind::Filter
                            } else {
                                ResidentScheduleOpKind::Project
                            },
                            op_id,
                            out: output.slot,
                            in0: input.slot,
                            in0_generation: input.generation,
                            out_generation: output.generation,
                            aux_offset,
                            aux_count,
                            ..Default::default()
                        },
                        output,
                        vec![input, output],
                    )?;
                }
                op @ (ResidentRecordedOp::Union { .. } | ResidentRecordedOp::Diff { .. }) => {
                    let (is_union, left, right, output, op_id) = match op {
                        ResidentRecordedOp::Union {
                            left,
                            right,
                            output,
                            op_id,
                        } => (true, left, right, output, op_id),
                        ResidentRecordedOp::Diff {
                            left,
                            right,
                            output,
                            op_id,
                        } => (false, left, right, output, op_id),
                        _ => unreachable!("compact set arm"),
                    };
                    let left = resident_compact_slot_ref(&left, assignments, &source_slots)?;
                    let right = resident_compact_slot_ref(&right, assignments, &source_slots)?;
                    let output = resident_compact_slot_ref(
                        &ResidentBufferRef::Private(output),
                        assignments,
                        &source_slots,
                    )?;
                    emit(
                        ResidentOpDescriptor {
                            kind: if is_union {
                                ResidentScheduleOpKind::Union
                            } else {
                                ResidentScheduleOpKind::Diff
                            },
                            op_id,
                            out: output.slot,
                            in0: left.slot,
                            in1: right.slot,
                            in0_generation: left.generation,
                            in1_generation: right.generation,
                            out_generation: output.generation,
                            ..Default::default()
                        },
                        output,
                        vec![left, right, output],
                    )?;
                }
                ResidentRecordedOp::Join {
                    kind,
                    left,
                    left_key,
                    right,
                    right_key,
                    output,
                    op_id,
                } => {
                    let left = resident_compact_slot_ref(&left, assignments, &source_slots)?;
                    let right = resident_compact_slot_ref(&right, assignments, &source_slots)?;
                    let output = resident_compact_slot_ref(
                        &ResidentBufferRef::Private(output),
                        assignments,
                        &source_slots,
                    )?;
                    emit(
                        ResidentOpDescriptor {
                            kind: match kind {
                                ResidentJoinKind::Inner => ResidentScheduleOpKind::JoinInner,
                                ResidentJoinKind::Semi => ResidentScheduleOpKind::JoinSemi,
                            },
                            op_id,
                            out: output.slot,
                            in0: left.slot,
                            in1: right.slot,
                            in0_generation: left.generation,
                            in1_generation: right.generation,
                            out_generation: output.generation,
                            left_key: u32::try_from(left_key).map_err(|_| {
                                XlogError::Execution(
                                    "resident compact left join key exceeds u32".into(),
                                )
                            })?,
                            right_key: u32::try_from(right_key).map_err(|_| {
                                XlogError::Execution(
                                    "resident compact right join key exceeds u32".into(),
                                )
                            })?,
                            ..Default::default()
                        },
                        output,
                        vec![left, right, output],
                    )?;
                }
                ResidentRecordedOp::TraceDelta {
                    scan_delta,
                    filter_delta,
                    semantic_guard,
                } => {
                    let semantic_guard = semantic_guard
                        .as_ref()
                        .map(|reference| {
                            resident_compact_slot_ref(reference, assignments, &source_slots)
                        })
                        .transpose()?;
                    if let Some(reference) = semantic_guard {
                        let generation = first_generations
                            .get_mut(reference.slot as usize)
                            .ok_or_else(|| {
                                XlogError::Execution(
                                    "resident compact trace guard slot is out of range".into(),
                                )
                            })?;
                        if generation.is_none() {
                            *generation = Some(reference.generation);
                        }
                    }
                    let first_op = u32::try_from(ops.len()).map_err(|_| {
                        XlogError::Execution("resident compact op count exceeds u32".into())
                    })?;
                    ops.push(ResidentOpDescriptor::trace_delta(
                        scan_delta,
                        filter_delta,
                        semantic_guard.map(|reference| (reference.slot, reference.generation)),
                    ));
                    waves.push(ResidentWaveDescriptor {
                        first_op,
                        op_count: 1,
                        flags: 0,
                        reserved: 0,
                    });
                }
                ResidentRecordedOp::TestStatus(status) => {
                    let first_op = u32::try_from(ops.len()).map_err(|_| {
                        XlogError::Execution("resident compact op count exceeds u32".into())
                    })?;
                    ops.push(ResidentOpDescriptor::test_status(status)?);
                    waves.push(ResidentWaveDescriptor {
                        first_op,
                        op_count: 1,
                        flags: 0,
                        reserved: 0,
                    });
                }
                ResidentRecordedOp::SchemaWinnerMark {
                    contribution,
                    head_index,
                    schema_id,
                } => {
                    let contribution =
                        resident_compact_slot_ref(&contribution, assignments, &source_slots)?;
                    let (index, output) = last_relation_descriptor.as_ref().ok_or_else(|| {
                        XlogError::Execution(
                            "resident schema marker has no contribution operation".into(),
                        )
                    })?;
                    if *output != contribution {
                        return Err(XlogError::Execution(
                            "resident schema marker does not match its contribution operation"
                                .into(),
                        ));
                    }
                    ops[*index] = ops[*index].with_schema_winner(head_index, schema_id);
                }
                ResidentRecordedOp::ChangedMark { relation } => {
                    let relation = resident_compact_slot_ref(
                        &ResidentBufferRef::Private(relation),
                        assignments,
                        &source_slots,
                    )?;
                    let (index, output) = last_relation_descriptor.as_ref().ok_or_else(|| {
                        XlogError::Execution(
                            "resident novelty marker has no completed delta copy".into(),
                        )
                    })?;
                    if *output != relation || ops[*index].kind != ResidentScheduleOpKind::Project {
                        return Err(XlogError::Execution(
                            "resident novelty marker does not follow its completed delta copy"
                                .into(),
                        ));
                    }
                    ops[*index].flags |= RESIDENT_SCHEDULE_OP_MARK_NOVELTY;
                }
                ResidentRecordedOp::ChangedReset => {
                    if region_flags != RESIDENT_SCHEDULE_REGION_RECURSIVE || logical_op_index != 0 {
                        return Err(XlogError::Execution(
                            "resident changed reset is not the recursive region entry".into(),
                        ));
                    }
                }
                ResidentRecordedOp::Clear { .. } => {
                    return Err(XlogError::Execution(
                        "resident compact SSA still contains a Clear operation".into(),
                    ));
                }
            }
        }

        for (slot, generation) in first_generations.into_iter().enumerate() {
            let baseline = if slot < physical_slots.len() && physical_slots[slot].permanent {
                0
            } else {
                generation.unwrap_or(0)
            };
            generation_bases.push(baseline);
        }
        let wave_count = u32::try_from(waves.len())
            .map_err(|_| XlogError::Execution("resident wave count exceeds u32".into()))?
            .checked_sub(first_wave)
            .ok_or_else(|| XlogError::Execution("resident wave range underflow".into()))?;
        regions.push(ResidentRegionDescriptor {
            first_wave,
            wave_count,
            iteration_limit: logical_region.iteration_limit,
            op_id: logical_region.op_id,
            flags: region_flags,
            first_slot: 0,
            slot_count: slot_count_u32,
            generation_offset,
        });
    }

    if filter_cursor != filter_total || project_cursor != project_total {
        return Err(XlogError::Execution(
            "resident compact descriptor tables are not exactly covered".into(),
        ));
    }

    Ok(ResidentCompactSchedulePlan {
        source_slots,
        ops,
        waves,
        regions,
        generation_bases,
        filter_comparisons: tables.filter_comparisons,
        project_expressions: tables.project_expressions,
    })
}

impl ResidentCompactLogicalRegion {
    fn initializes(&self) -> bool {
        self.flags & RESIDENT_SCHEDULE_REGION_INITIALIZE != 0
    }

    fn begins_scc(&self) -> bool {
        self.flags & RESIDENT_SCHEDULE_REGION_SCC_BEGIN != 0
    }

    fn recursive(&self) -> bool {
        self.flags == RESIDENT_SCHEDULE_REGION_RECURSIVE
    }

    fn finalizes(&self) -> bool {
        self.flags & RESIDENT_SCHEDULE_REGION_FINALIZE != 0
    }
}

fn resident_compact_regions(
    initial_ops: Vec<ResidentRecordedOp>,
    phases: Vec<ResidentCapturePhase>,
    success_op_id: u32,
) -> Result<Vec<ResidentCompactLogicalRegion>> {
    let mut regions = Vec::new();
    let mut pending = initial_ops;
    let mut pending_seed_region = None;

    for phase in phases {
        match phase {
            ResidentCapturePhase::Segment {
                mut ops,
                scc_begin: None,
            } => {
                if pending_seed_region.is_some() {
                    return Err(XlogError::Execution(
                        "resident SCC seed is not followed by its recursive body".into(),
                    ));
                }
                pending.append(&mut ops);
            }
            ResidentCapturePhase::Segment {
                mut ops,
                scc_begin: Some((iteration_limit, op_id)),
            } => {
                if pending_seed_region.is_some() {
                    return Err(XlogError::Execution(
                        "resident SCC seed is not followed by its recursive body".into(),
                    ));
                }
                pending.append(&mut ops);
                let mut flags = RESIDENT_SCHEDULE_REGION_SCC_BEGIN;
                if regions.is_empty() {
                    flags |= RESIDENT_SCHEDULE_REGION_INITIALIZE;
                }
                regions.push(ResidentCompactLogicalRegion {
                    ops: std::mem::take(&mut pending),
                    iteration_limit,
                    op_id,
                    flags,
                });
                pending_seed_region = Some(regions.len() - 1);
            }
            ResidentCapturePhase::ConditionalWhile {
                ops,
                iteration_limit,
                convergence_op_id,
            } => {
                let seed_region = pending_seed_region.take().ok_or_else(|| {
                    XlogError::Execution("resident recursive body has no preceding SCC seed".into())
                })?;
                if regions[seed_region].iteration_limit != iteration_limit {
                    return Err(XlogError::Execution(
                        "resident SCC seed and recursive body iteration limits differ".into(),
                    ));
                }
                regions[seed_region].op_id = convergence_op_id;
                regions.push(ResidentCompactLogicalRegion {
                    ops,
                    iteration_limit,
                    op_id: convergence_op_id,
                    flags: RESIDENT_SCHEDULE_REGION_RECURSIVE,
                });
            }
        }
    }
    if pending_seed_region.is_some() {
        return Err(XlogError::Execution(
            "resident SCC seed is missing its recursive body".into(),
        ));
    }

    let mut flags = RESIDENT_SCHEDULE_REGION_FINALIZE;
    if regions.is_empty() {
        flags |= RESIDENT_SCHEDULE_REGION_INITIALIZE;
    }
    regions.push(ResidentCompactLogicalRegion {
        ops: pending,
        iteration_limit: 1,
        op_id: success_op_id,
        flags,
    });
    Ok(regions)
}

fn coalesce_resident_capture_phases(
    phases: Vec<ResidentCapturePhase>,
) -> Vec<ResidentCapturePhase> {
    let mut coalesced = Vec::new();
    let mut ordinary_ops = Vec::new();
    let flush_ordinary = |coalesced: &mut Vec<ResidentCapturePhase>,
                          ordinary_ops: &mut Vec<ResidentRecordedOp>| {
        if !ordinary_ops.is_empty() {
            coalesced.push(ResidentCapturePhase::Segment {
                ops: std::mem::take(ordinary_ops),
                scc_begin: None,
            });
        }
    };

    for phase in phases {
        match phase {
            ResidentCapturePhase::Segment {
                mut ops,
                scc_begin: None,
            } => ordinary_ops.append(&mut ops),
            boundary => {
                flush_ordinary(&mut coalesced, &mut ordinary_ops);
                coalesced.push(boundary);
            }
        }
    }
    flush_ordinary(&mut coalesced, &mut ordinary_ops);
    coalesced
}

#[cfg(test)]
mod capture_phase_tests {
    use super::{coalesce_resident_capture_phases, ResidentCapturePhase, ResidentRecordedOp};

    fn segment(output: usize, scc_begin: Option<(u32, u32)>) -> ResidentCapturePhase {
        ResidentCapturePhase::Segment {
            ops: vec![ResidentRecordedOp::Clear { output }],
            scc_begin,
        }
    }

    fn assert_clear_range(phase: &ResidentCapturePhase, expected: std::ops::Range<usize>) {
        let ResidentCapturePhase::Segment {
            ops,
            scc_begin: None,
        } = phase
        else {
            panic!("expected an ordinary capture segment");
        };
        assert_eq!(ops.len(), expected.len());
        for (op, output) in ops.iter().zip(expected) {
            assert!(
                matches!(op, ResidentRecordedOp::Clear { output: actual } if *actual == output)
            );
        }
    }

    #[test]
    fn coalesces_only_maximal_ordinary_phase_runs() {
        let mut phases = (0..1_000)
            .map(|output| segment(output, None))
            .collect::<Vec<_>>();
        phases.push(segment(10_000, Some((64, 41))));
        phases.extend((1_000..1_500).map(|output| segment(output, None)));
        phases.push(ResidentCapturePhase::ConditionalWhile {
            ops: vec![ResidentRecordedOp::Clear { output: 20_000 }],
            iteration_limit: 64,
            convergence_op_id: 42,
        });
        phases.extend((1_500..1_800).map(|output| segment(output, None)));

        let coalesced = coalesce_resident_capture_phases(phases);
        assert_eq!(coalesced.len(), 5);
        assert_clear_range(&coalesced[0], 0..1_000);
        assert!(matches!(
            &coalesced[1],
            ResidentCapturePhase::Segment {
                ops,
                scc_begin: Some((64, 41)),
            } if matches!(ops.as_slice(), [ResidentRecordedOp::Clear { output: 10_000 }])
        ));
        assert_clear_range(&coalesced[2], 1_000..1_500);
        assert!(matches!(
            &coalesced[3],
            ResidentCapturePhase::ConditionalWhile {
                ops,
                iteration_limit: 64,
                convergence_op_id: 42,
            } if matches!(ops.as_slice(), [ResidentRecordedOp::Clear { output: 20_000 }])
        ));
        assert_clear_range(&coalesced[4], 1_500..1_800);
    }
}

struct ResidentBuild<'executor> {
    executor: &'executor Executor,
    certificate: &'executor ResidentGraphRouteCertificate,
    capacity: u32,
    relations: Vec<ResidentLogicalRelation>,
    filter_workspaces: Vec<ResidentFilterPlan>,
    project_workspaces: Vec<ResidentProjectPlan>,
    heads: BTreeMap<String, usize>,
    head_winner_indices: BTreeMap<String, u32>,
    source_names: HashSet<String>,
    source_aliases: HashMap<String, usize>,
    next_op_id: u32,
    injection: Option<crate::resident_graph::ResidentGraphDeviceStatusTestInjection>,
    injection_recorded: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ResidentSourceBindingRoute {
    Direct,
    NormalizeEmpty,
}

fn resident_source_logical_count(cached: Option<u32>) -> Result<u64> {
    cached.map(u64::from).ok_or_else(|| {
        XlogError::Execution("resident source requires a cold-path cached logical row count".into())
    })
}

fn resident_source_binding_route(
    row_count: u64,
    count_tracked: bool,
    columns_tracked: bool,
) -> Result<ResidentSourceBindingRoute> {
    if count_tracked && columns_tracked {
        return Ok(ResidentSourceBindingRoute::Direct);
    }
    if row_count == 0 {
        return Ok(ResidentSourceBindingRoute::NormalizeEmpty);
    }
    Err(XlogError::Execution(
        "nonempty resident source is not fully runtime tracked".into(),
    ))
}

#[derive(Debug, Clone)]
struct ResidentLogicalRelation {
    schema: Schema,
    initial_count: u32,
    permanent: bool,
}

#[derive(Debug, Clone)]
struct ResidentFilterPlan {
    compact_comparisons: Vec<ResidentFilterComparisonDescriptor>,
}

#[derive(Debug, Clone)]
struct ResidentProjectPlan {
    compact_expressions: Vec<ResidentProjectExpressionDescriptor>,
}

#[derive(Debug, Clone, Copy)]
struct ResidentSlotAssignment {
    slot: usize,
    generation: u32,
}

#[derive(Debug, Clone)]
struct ResidentPhysicalSlotPlan {
    schema: Schema,
    initial_count: u32,
    permanent: bool,
}

#[derive(Debug, Clone)]
struct ResidentAllocationManifest {
    slots: Vec<ResidentPhysicalSlotPlan>,
    logical_to_slot: Vec<ResidentSlotAssignment>,
    required_bytes: u64,
    relation_bytes: u64,
    filter_scratch_bytes: u64,
    schedule_metadata_bytes: u64,
    fixed_workspace_bytes: u64,
    logical_relation_values: usize,
    permanent_slots: u32,
    scratch_slots: u32,
    filter_scratch_allocations: u32,
    max_row_bytes: u64,
}

struct ResidentPhysicalBuild {
    relations: Vec<Option<ResidentRelation>>,
    filter_scratch: Option<ResidentFilterScratch>,
    set_workspace: ResidentSetWorkspace,
    join_workspace: ResidentJoinWorkspace,
    control: ResidentConvergenceControl,
}

/// Raw fixed-cost counters retained until outer latency sampling is complete.
#[doc(hidden)]
#[derive(Default)]
pub struct ResidentPrepareDiagnostics {
    sample: u64,
    total_ns: u64,
    required_reservation_bytes: u64,
    logical_relation_values: usize,
    physical_relation_slots: usize,
    relation_device_allocation_calls: u64,
    compact_ops: usize,
    compact_waves: usize,
    compact_regions: usize,
    conditional_regions: usize,
    parent_graph_nodes: usize,
    conditional_body_nodes: usize,
    admission_and_source_snapshot_ns: u64,
    execution_domain_and_build_setup_ns: u64,
    logical_schedule_planning_ns: u64,
    manifest_compact_construction_ns: u64,
    schedule_lowering_ns: u64,
    reservation_ns: u64,
    relation_slot_allocation_ns: u64,
    relation_slot_allocation_ns_max: u64,
    relation_slot_allocation_bytes: u64,
    count_initialization_ns: u64,
    count_initialization_ns_max: u64,
    count_memset_calls: u64,
    workspace_provider_calls: u64,
    filter_scratch_allocation_ns: u64,
    filter_scratch_allocation_bytes: u64,
    set_workspace_allocation_ns: u64,
    set_workspace_allocation_bytes: u64,
    join_workspace_allocation_ns: u64,
    join_workspace_allocation_bytes: u64,
    control_allocation_ns: u64,
    control_allocation_bytes: u64,
    metadata_binding_construction_ns: u64,
    metadata_provider_calls: u64,
    device_trace_preparation_ns: u64,
    device_trace_reserved_bytes: u64,
    device_trace_initial_htod_calls: u64,
    device_trace_initial_htod_bytes: u64,
    schema_winners_preparation_ns: u64,
    schema_winners_reserved_bytes: u64,
    schema_winners_initial_htod_calls: u64,
    schema_winners_initial_htod_bytes: u64,
    receipt_preparation_ns: u64,
    receipt_reserved_bytes: u64,
    receipt_initial_htod_calls: u64,
    receipt_initial_htod_bytes: u64,
    schedule_program_preparation_ns: u64,
    schedule_program_reserved_bytes: u64,
    schedule_program_initial_htod_calls: u64,
    schedule_program_initial_htod_bytes: u64,
    reservation_validation_and_release_ns: u64,
    pinned_receipt_ns: u64,
    graph_body_capture_ns: u64,
    graph_instantiate_ns: u64,
    validation_owner_assembly_ns: u64,
}

fn resident_prepare_diagnostics_for_sample(
    sample: Option<u64>,
) -> Option<ResidentPrepareDiagnostics> {
    sample.map(|sample| ResidentPrepareDiagnostics {
        sample,
        ..ResidentPrepareDiagnostics::default()
    })
}

fn resident_prepare_elapsed_ns(started: std::time::Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn resident_schema_winners_initial_htod(default_count: usize) -> (u64, u64) {
    if default_count == 0 {
        return (0, 0);
    }
    (
        1,
        u64::try_from(default_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<u32>() as u64),
    )
}

fn resident_receipt_initial_htod(output_count: usize) -> (u64, u64) {
    let pointee_count = output_count.saturating_mul(2).saturating_add(4);
    if pointee_count == 0 {
        return (0, 0);
    }
    (
        1,
        u64::try_from(pointee_count)
            .unwrap_or(u64::MAX)
            .saturating_mul(std::mem::size_of::<u64>() as u64),
    )
}

fn resident_schedule_initial_htod(reserved_bytes: u64) -> (u64, u64) {
    (8, reserved_bytes)
}

impl ResidentPrepareDiagnostics {
    pub fn into_snapshot(self) -> ResidentGraphPrepareDiagnosticSnapshot {
        let workspace_allocation_ns = self
            .filter_scratch_allocation_ns
            .saturating_add(self.set_workspace_allocation_ns)
            .saturating_add(self.join_workspace_allocation_ns)
            .saturating_add(self.control_allocation_ns);
        let workspace_allocation_bytes = self
            .filter_scratch_allocation_bytes
            .saturating_add(self.set_workspace_allocation_bytes)
            .saturating_add(self.join_workspace_allocation_bytes)
            .saturating_add(self.control_allocation_bytes);
        let metadata_preparation_ns = self
            .device_trace_preparation_ns
            .saturating_add(self.schema_winners_preparation_ns)
            .saturating_add(self.receipt_preparation_ns)
            .saturating_add(self.schedule_program_preparation_ns);
        let metadata_reserved_bytes = self
            .device_trace_reserved_bytes
            .saturating_add(self.schema_winners_reserved_bytes)
            .saturating_add(self.receipt_reserved_bytes)
            .saturating_add(self.schedule_program_reserved_bytes);
        let additive = [
            self.admission_and_source_snapshot_ns,
            self.execution_domain_and_build_setup_ns,
            self.logical_schedule_planning_ns,
            self.manifest_compact_construction_ns,
            self.schedule_lowering_ns,
            self.reservation_ns,
            self.relation_slot_allocation_ns,
            self.count_initialization_ns,
            workspace_allocation_ns,
            self.metadata_binding_construction_ns,
            metadata_preparation_ns,
            self.reservation_validation_and_release_ns,
            self.pinned_receipt_ns,
            self.graph_body_capture_ns,
            self.graph_instantiate_ns,
            self.validation_owner_assembly_ns,
        ];
        let attributed_ns = additive.iter().copied().fold(0u64, u64::saturating_add);
        let unattributed_ns = self.total_ns.saturating_sub(attributed_ns);
        let metadata_initial_htod_calls = self
            .device_trace_initial_htod_calls
            .saturating_add(self.schema_winners_initial_htod_calls)
            .saturating_add(self.receipt_initial_htod_calls)
            .saturating_add(self.schedule_program_initial_htod_calls);
        let metadata_initial_htod_bytes = self
            .device_trace_initial_htod_bytes
            .saturating_add(self.schema_winners_initial_htod_bytes)
            .saturating_add(self.receipt_initial_htod_bytes)
            .saturating_add(self.schedule_program_initial_htod_bytes);
        let to_u64 = |value: usize| u64::try_from(value).unwrap_or(u64::MAX);
        ResidentGraphPrepareDiagnosticSnapshot {
            sample: self.sample,
            total_ns: self.total_ns,
            admission_and_source_snapshot_ns: self.admission_and_source_snapshot_ns,
            execution_domain_and_build_setup_ns: self.execution_domain_and_build_setup_ns,
            logical_schedule_planning_ns: self.logical_schedule_planning_ns,
            manifest_compact_construction_ns: self.manifest_compact_construction_ns,
            schedule_lowering_ns: self.schedule_lowering_ns,
            reservation_ns: self.reservation_ns,
            relation_preparation_ns: self.relation_slot_allocation_ns,
            count_initialization_ns: self.count_initialization_ns,
            workspace_preparation_ns: workspace_allocation_ns,
            metadata_binding_construction_ns: self.metadata_binding_construction_ns,
            metadata_preparation_ns,
            reservation_validation_and_release_ns: self.reservation_validation_and_release_ns,
            pinned_receipt_ns: self.pinned_receipt_ns,
            graph_body_capture_ns: self.graph_body_capture_ns,
            graph_instantiate_ns: self.graph_instantiate_ns,
            validation_owner_assembly_ns: self.validation_owner_assembly_ns,
            unattributed_ns,
            required_reservation_bytes: self.required_reservation_bytes,
            logical_relation_values: to_u64(self.logical_relation_values),
            physical_relation_slots: to_u64(self.physical_relation_slots),
            relation_device_allocation_calls: self.relation_device_allocation_calls,
            relation_reserved_bytes: self.relation_slot_allocation_bytes,
            relation_slot_preparation_ns_max: self.relation_slot_allocation_ns_max,
            count_memset_calls: self.count_memset_calls,
            count_memset_bytes: self
                .count_memset_calls
                .saturating_mul(std::mem::size_of::<u32>() as u64),
            count_initialization_ns_max: self.count_initialization_ns_max,
            workspace_provider_calls: self.workspace_provider_calls,
            workspace_reserved_bytes: workspace_allocation_bytes,
            filter_scratch_preparation_ns: self.filter_scratch_allocation_ns,
            filter_scratch_reserved_bytes: self.filter_scratch_allocation_bytes,
            set_workspace_preparation_ns: self.set_workspace_allocation_ns,
            set_workspace_reserved_bytes: self.set_workspace_allocation_bytes,
            join_workspace_preparation_ns: self.join_workspace_allocation_ns,
            join_workspace_reserved_bytes: self.join_workspace_allocation_bytes,
            control_preparation_ns: self.control_allocation_ns,
            control_reserved_bytes: self.control_allocation_bytes,
            metadata_provider_calls: self.metadata_provider_calls,
            metadata_reserved_bytes,
            metadata_initial_htod_calls,
            metadata_initial_htod_bytes,
            device_trace_preparation_ns: self.device_trace_preparation_ns,
            device_trace_reserved_bytes: self.device_trace_reserved_bytes,
            device_trace_initial_htod_calls: self.device_trace_initial_htod_calls,
            device_trace_initial_htod_bytes: self.device_trace_initial_htod_bytes,
            schema_winners_preparation_ns: self.schema_winners_preparation_ns,
            schema_winners_reserved_bytes: self.schema_winners_reserved_bytes,
            schema_winners_initial_htod_calls: self.schema_winners_initial_htod_calls,
            schema_winners_initial_htod_bytes: self.schema_winners_initial_htod_bytes,
            receipt_preparation_ns: self.receipt_preparation_ns,
            receipt_reserved_bytes: self.receipt_reserved_bytes,
            receipt_initial_htod_calls: self.receipt_initial_htod_calls,
            receipt_initial_htod_bytes: self.receipt_initial_htod_bytes,
            schedule_program_preparation_ns: self.schedule_program_preparation_ns,
            schedule_program_reserved_bytes: self.schedule_program_reserved_bytes,
            schedule_program_initial_htod_calls: self.schedule_program_initial_htod_calls,
            schedule_program_initial_htod_bytes: self.schedule_program_initial_htod_bytes,
            compact_ops: to_u64(self.compact_ops),
            compact_waves: to_u64(self.compact_waves),
            compact_regions: to_u64(self.compact_regions),
            conditional_regions: to_u64(self.conditional_regions),
            parent_graph_nodes: to_u64(self.parent_graph_nodes),
            conditional_body_nodes: to_u64(self.conditional_body_nodes),
        }
    }
}

#[cfg(test)]
mod prepare_diagnostic_tests {
    use super::{
        resident_prepare_diagnostics_for_sample, resident_receipt_initial_htod,
        resident_schedule_initial_htod, resident_schema_winners_initial_htod,
        ResidentPrepareDiagnostics,
    };

    #[test]
    fn prepare_diagnostic_counts_only_source_proven_initial_transfers() {
        assert_eq!(resident_schema_winners_initial_htod(0), (0, 0));
        assert_eq!(resident_schema_winners_initial_htod(3), (1, 12));
        assert_eq!(resident_receipt_initial_htod(0), (1, 32));
        assert_eq!(resident_receipt_initial_htod(3), (1, 80));
        assert_eq!(resident_schedule_initial_htod(4096), (8, 4096));

        let (schema_calls, schema_bytes) = resident_schema_winners_initial_htod(3);
        let (receipt_calls, receipt_bytes) = resident_receipt_initial_htod(0);
        let (schedule_calls, schedule_bytes) = resident_schedule_initial_htod(4096);
        let diagnostics = ResidentPrepareDiagnostics {
            sample: 9,
            total_ns: 100,
            relation_slot_allocation_ns: 11,
            relation_slot_allocation_ns_max: 7,
            relation_slot_allocation_bytes: 2048,
            workspace_provider_calls: 4,
            count_initialization_ns: 10,
            count_initialization_ns_max: 5,
            count_memset_calls: 3,
            metadata_provider_calls: 4,
            schema_winners_initial_htod_calls: schema_calls,
            schema_winners_initial_htod_bytes: schema_bytes,
            receipt_initial_htod_calls: receipt_calls,
            receipt_initial_htod_bytes: receipt_bytes,
            schedule_program_initial_htod_calls: schedule_calls,
            schedule_program_initial_htod_bytes: schedule_bytes,
            ..ResidentPrepareDiagnostics::default()
        };
        let snapshot = diagnostics.into_snapshot();
        assert_eq!(snapshot.sample, 9);
        assert_eq!(snapshot.total_ns, 100);
        assert_eq!(snapshot.relation_preparation_ns, 11);
        assert_eq!(snapshot.relation_reserved_bytes, 2048);
        assert_eq!(snapshot.relation_slot_preparation_ns_max, 7);
        assert_eq!(snapshot.workspace_provider_calls, 4);
        assert_eq!(snapshot.count_memset_calls, 3);
        assert_eq!(snapshot.count_initialization_ns, 10);
        assert_eq!(snapshot.count_initialization_ns_max, 5);
        assert_eq!(
            snapshot.count_memset_bytes,
            3 * std::mem::size_of::<u32>() as u64
        );
        assert_eq!(snapshot.metadata_provider_calls, 4);
        assert_eq!(snapshot.device_trace_initial_htod_calls, 0);
        assert_eq!(snapshot.device_trace_initial_htod_bytes, 0);
        assert_eq!(snapshot.schema_winners_initial_htod_calls, 1);
        assert_eq!(snapshot.schema_winners_initial_htod_bytes, 12);
        assert_eq!(snapshot.receipt_initial_htod_calls, 1);
        assert_eq!(snapshot.receipt_initial_htod_bytes, 32);
        assert_eq!(snapshot.schedule_program_initial_htod_calls, 8);
        assert_eq!(snapshot.schedule_program_initial_htod_bytes, 4096);
        assert_eq!(snapshot.metadata_initial_htod_calls, 10);
        assert_eq!(snapshot.metadata_initial_htod_bytes, 4140);
    }

    #[test]
    fn prepare_diagnostic_is_absent_when_sampling_is_disabled() {
        assert!(resident_prepare_diagnostics_for_sample(None).is_none());
        assert_eq!(
            resident_prepare_diagnostics_for_sample(Some(23))
                .expect("enabled diagnostics create raw timing state")
                .sample,
            23
        );
    }

    #[test]
    fn prepare_diagnostic_keeps_setup_and_reservation_validation_separate() {
        let snapshot = ResidentPrepareDiagnostics {
            sample: 5,
            total_ns: 100,
            admission_and_source_snapshot_ns: 11,
            execution_domain_and_build_setup_ns: 13,
            metadata_binding_construction_ns: 17,
            reservation_validation_and_release_ns: 19,
            ..ResidentPrepareDiagnostics::default()
        }
        .into_snapshot();

        assert_eq!(snapshot.admission_and_source_snapshot_ns, 11);
        assert_eq!(snapshot.execution_domain_and_build_setup_ns, 13);
        assert_eq!(snapshot.metadata_binding_construction_ns, 17);
        assert_eq!(snapshot.reservation_validation_and_release_ns, 19);
        assert_eq!(snapshot.unattributed_ns, 40);
    }
}

impl ResidentAllocationManifest {
    fn finalize_compact_schedule(
        &mut self,
        schedule: &ResidentCompactSchedulePlan,
        head_count: usize,
    ) -> Result<()> {
        let (required_bytes, schedule_metadata_bytes) = resident_compact_allocation_bytes(
            self.relation_bytes,
            self.filter_scratch_bytes,
            self.fixed_workspace_bytes,
            self.slots.len(),
            head_count,
            schedule,
        )?;
        self.required_bytes = required_bytes;
        self.schedule_metadata_bytes = schedule_metadata_bytes;
        Ok(())
    }
}

fn resident_output_indices(
    heads: &BTreeMap<String, usize>,
    assignments: &[ResidentSlotAssignment],
    slots: &[ResidentPhysicalSlotPlan],
) -> Result<Vec<(String, usize)>> {
    heads
        .iter()
        .map(|(name, logical)| {
            let assignment = assignments.get(*logical).ok_or_else(|| {
                XlogError::Execution("resident head has no physical slot assignment".into())
            })?;
            if !slots
                .get(assignment.slot)
                .is_some_and(|slot| slot.permanent)
            {
                return Err(XlogError::Execution(
                    "resident head is not assigned to a permanent physical slot".into(),
                ));
            }
            Ok((name.clone(), assignment.slot))
        })
        .collect()
}

fn resident_validate_exact_reservation(required: u64, used: u64, remaining: u64) -> Result<()> {
    if used == required && remaining == 0 {
        return Ok(());
    }
    Err(XlogError::Execution(format!(
        "resident allocation manifest accounted for {required} manager-tracked bytes but materialization consumed {used} and left {remaining} reserved bytes"
    )))
}

#[derive(Debug)]
struct ResidentScratchSlotState {
    slot: usize,
    layout: Vec<ScalarType>,
    last_use: usize,
    generation: u32,
}

fn resident_schema_layout(schema: &Schema) -> Vec<ScalarType> {
    schema
        .columns
        .iter()
        .map(|(_, scalar)| scalar.clone())
        .collect()
}

fn resident_private_inputs(op: &ResidentRecordedOp, mut visit: impl FnMut(usize)) {
    let mut visit_ref = |reference: &ResidentBufferRef| {
        if let ResidentBufferRef::Private(index) = reference {
            visit(*index);
        }
    };
    match op {
        ResidentRecordedOp::Scan {
            relation: input, ..
        }
        | ResidentRecordedOp::Filter { input, .. }
        | ResidentRecordedOp::Project { input, .. } => visit_ref(input),
        ResidentRecordedOp::Union { left, right, .. }
        | ResidentRecordedOp::Diff { left, right, .. }
        | ResidentRecordedOp::Join { left, right, .. } => {
            visit_ref(left);
            visit_ref(right);
        }
        ResidentRecordedOp::ChangedMark { relation } => visit(*relation),
        ResidentRecordedOp::TraceDelta { semantic_guard, .. } => {
            if let Some(semantic_guard) = semantic_guard {
                visit_ref(semantic_guard);
            }
        }
        ResidentRecordedOp::SchemaWinnerMark { contribution, .. } => visit_ref(contribution),
        ResidentRecordedOp::Unit { .. }
        | ResidentRecordedOp::Clear { .. }
        | ResidentRecordedOp::ChangedReset
        | ResidentRecordedOp::TestStatus(_) => {}
    }
}

fn resident_private_output(op: &ResidentRecordedOp) -> Option<usize> {
    match op {
        ResidentRecordedOp::Unit { output, .. }
        | ResidentRecordedOp::Clear { output }
        | ResidentRecordedOp::Filter { output, .. }
        | ResidentRecordedOp::Project { output, .. }
        | ResidentRecordedOp::Union { output, .. }
        | ResidentRecordedOp::Diff { output, .. }
        | ResidentRecordedOp::Join { output, .. } => Some(*output),
        ResidentRecordedOp::Scan { .. }
        | ResidentRecordedOp::TraceDelta { .. }
        | ResidentRecordedOp::ChangedReset
        | ResidentRecordedOp::ChangedMark { .. }
        | ResidentRecordedOp::SchemaWinnerMark { .. }
        | ResidentRecordedOp::TestStatus(_) => None,
    }
}

fn resident_source_slot_map<'a>(
    private_slot_count: usize,
    sources: impl IntoIterator<Item = &'a str>,
) -> Result<BTreeMap<String, u32>> {
    let first_source_slot = u32::try_from(private_slot_count)
        .map_err(|_| XlogError::Execution("resident private slot count exceeds u32".into()))?;
    let names = sources
        .into_iter()
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    names
        .into_iter()
        .enumerate()
        .map(|(offset, name)| {
            let offset = u32::try_from(offset).map_err(|_| {
                XlogError::Execution("resident source slot count exceeds u32".into())
            })?;
            let slot = first_source_slot.checked_add(offset).ok_or_else(|| {
                XlogError::Execution("resident source slot index overflow".into())
            })?;
            Ok((name, slot))
        })
        .collect()
}

fn resident_record_lifetimes(
    ops: &[ResidentRecordedOp],
    relations: &[ResidentLogicalRelation],
    definitions: &mut [Option<usize>],
    last_uses: &mut [Option<usize>],
    ordinal: &mut usize,
) -> Result<(usize, usize)> {
    let start = *ordinal;
    for op in ops {
        let current = *ordinal;
        let mut invalid_input = None;
        let mut input_before_definition = None;
        resident_private_inputs(op, |logical| {
            let Some(relation) = relations.get(logical) else {
                invalid_input.get_or_insert(logical);
                return;
            };
            if !relation.permanent {
                if definitions[logical].is_none() {
                    input_before_definition.get_or_insert(logical);
                } else {
                    last_uses[logical] = Some(last_uses[logical].unwrap_or(current).max(current));
                }
            }
        });
        if let Some(logical) = invalid_input {
            return Err(XlogError::Execution(format!(
                "resident logical input relation {logical} is missing"
            )));
        }
        if let Some(logical) = input_before_definition {
            return Err(XlogError::Execution(format!(
                "resident scratch relation {logical} is used before its definition"
            )));
        }
        if let Some(logical) = resident_private_output(op) {
            let relation = relations.get(logical).ok_or_else(|| {
                XlogError::Execution(format!(
                    "resident logical output relation {logical} is missing"
                ))
            })?;
            if !relation.permanent {
                if definitions[logical].replace(current).is_some() {
                    return Err(XlogError::Execution(format!(
                        "resident scratch relation {logical} has multiple definitions"
                    )));
                }
                last_uses[logical] = Some(last_uses[logical].unwrap_or(current).max(current));
            }
        }
        *ordinal = ordinal
            .checked_add(1)
            .ok_or_else(|| XlogError::Execution("resident operation ordinal overflow".into()))?;
    }
    Ok((start, *ordinal))
}

fn resident_validate_slot_assignments(
    relations: &[ResidentLogicalRelation],
    definitions: &[Option<usize>],
    last_uses: &[Option<usize>],
    slots: &[ResidentPhysicalSlotPlan],
    assignments: &[ResidentSlotAssignment],
) -> Result<()> {
    if definitions.len() != relations.len()
        || last_uses.len() != relations.len()
        || assignments.len() != relations.len()
    {
        return Err(XlogError::Execution(
            "resident slot validation vector length mismatch".into(),
        ));
    }

    let mut occupied_permanent_slots = HashSet::new();
    let mut scratch_by_slot = BTreeMap::<usize, Vec<(usize, usize, usize, u32)>>::new();
    let mut assigned_slots = HashSet::new();
    for (logical, relation) in relations.iter().enumerate() {
        let assignment = assignments[logical];
        let slot = slots.get(assignment.slot).ok_or_else(|| {
            XlogError::Execution(format!(
                "resident logical relation {logical} maps to missing physical slot {}",
                assignment.slot
            ))
        })?;
        assigned_slots.insert(assignment.slot);
        if relation.permanent {
            if assignment.generation != 0 {
                return Err(XlogError::Execution(format!(
                    "resident permanent relation {logical} has nonzero generation {}",
                    assignment.generation
                )));
            }
            if !slot.permanent
                || slot.schema != relation.schema
                || slot.initial_count != relation.initial_count
            {
                return Err(XlogError::Execution(format!(
                    "resident permanent relation {logical} has an incompatible physical slot"
                )));
            }
            if !occupied_permanent_slots.insert(assignment.slot) {
                return Err(XlogError::Execution(format!(
                    "resident permanent physical slot {} has multiple owners",
                    assignment.slot
                )));
            }
            continue;
        }

        if slot.permanent
            || resident_schema_layout(&slot.schema) != resident_schema_layout(&relation.schema)
        {
            return Err(XlogError::Execution(format!(
                "resident scratch relation {logical} has an incompatible physical slot"
            )));
        }
        let definition = definitions[logical].ok_or_else(|| {
            XlogError::Execution(format!(
                "resident scratch relation {logical} is never defined"
            ))
        })?;
        let last_use = last_uses[logical].unwrap_or(definition);
        if last_use < definition {
            return Err(XlogError::Execution(format!(
                "resident scratch relation {logical} ends before its definition"
            )));
        }
        scratch_by_slot.entry(assignment.slot).or_default().push((
            definition,
            last_use,
            logical,
            assignment.generation,
        ));
    }

    if assigned_slots.len() != slots.len() {
        return Err(XlogError::Execution(
            "resident allocation manifest contains an unassigned physical slot".into(),
        ));
    }
    for (slot, mut generations) in scratch_by_slot {
        generations.sort_by_key(|(definition, _, logical, _)| (*definition, *logical));
        let mut previous_last_use = None;
        for (expected_generation, (definition, last_use, logical, generation)) in
            generations.into_iter().enumerate()
        {
            let expected_generation = u32::try_from(expected_generation).map_err(|_| {
                XlogError::Execution("resident scratch generation exceeds u32".into())
            })?;
            if generation != expected_generation {
                return Err(XlogError::Execution(format!(
                    "resident scratch relation {logical} in slot {slot} has generation {generation} but expected {expected_generation}"
                )));
            }
            if previous_last_use.is_some_and(|previous| previous >= definition) {
                return Err(XlogError::Execution(format!(
                    "resident scratch generations overlap in physical slot {slot}"
                )));
            }
            previous_last_use = Some(last_use);
        }
    }
    Ok(())
}

fn resident_checked_add(total: &mut u64, bytes: u64, label: &str) -> Result<()> {
    *total = total.checked_add(bytes).ok_or_else(|| {
        XlogError::Execution(format!("resident {label} allocation byte overflow"))
    })?;
    Ok(())
}

/// A fully allocated, instantiated resident transaction that has not launched.
pub struct PreparedResidentGraph<'executor> {
    owners: ResidentRunOwners,
    preflight_report: ResidentGraphPreflightReport,
    has_device_status_writer: bool,
    source_guard: &'executor Executor,
    source_set_snapshots: Vec<ResidentSourceSetSnapshot>,
    prepare_diagnostic: Option<ResidentPrepareDiagnostics>,
}

/// Read-only topology and memory facts captured after preparation and before launch.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ResidentGraphPreflightReport {
    /// Fixed logical row capacity used for every staged relation.
    pub relation_capacity: u32,
    /// Exact manager-tracked bytes reserved before physical materialization.
    pub estimated_required_bytes: u64,
    /// Caller budget remaining when admission ran.
    pub available_bytes_at_admission: u64,
    /// Manager-tracked live allocation growth caused by this prepared transaction.
    pub tracked_device_allocation_bytes: u64,
    /// Exact manager-tracked relation data and logical-count bytes.
    pub relation_device_bytes: u64,
    /// Exact immutable filter descriptor bytes retained by the graph.
    pub filter_descriptor_device_bytes: u64,
    /// Exact mutable filter scratch bytes shared by all sequential filters.
    pub filter_scratch_device_bytes: u64,
    /// Exact immutable project and copy descriptor bytes.
    pub project_descriptor_device_bytes: u64,
    /// Exact set, join, control, trace, packed-receipt, and remaining compact schedule metadata bytes.
    pub fixed_workspace_device_bytes: u64,
    /// Nodes in the instantiated parent graph, excluding conditional body children.
    pub parent_graph_nodes: usize,
    /// Conditional-WHILE nodes in the instantiated parent graph.
    pub conditional_while_nodes: usize,
    /// Actual node kinds in root-to-leaf dependency-chain order.
    pub parent_graph_node_kinds: Vec<CudaGraphNodeKind>,
    /// Actual node kinds in each conditional-WHILE body, in parent order.
    pub conditional_body_node_kinds: Vec<Vec<CudaGraphNodeKind>>,
    /// Actual kernel-node count in each conditional-WHILE body.
    pub conditional_body_kernel_counts: Vec<usize>,
    /// Parent nodes plus every conditional-body child node.
    pub hierarchical_graph_nodes: usize,
    /// Private fixed-capacity relation slots retained by the graph.
    pub private_relation_slots: usize,
    /// Logical relation values before interval coloring.
    pub logical_relation_values: usize,
    /// Permanent unit, staged-head, raw, and delta slots.
    pub permanent_relation_slots: u32,
    /// Final staged relation count included in the terminal receipt.
    pub staged_output_relations: usize,
    /// Layout-compatible scratch slots selected by interval coloring.
    pub scratch_slots: u32,
    /// Number of mutable filter scratch allocations retained by this graph.
    pub filter_scratch_allocations: u32,
    /// Widest allocated permanent or intermediate row in bytes.
    pub max_row_bytes: u64,
}

/// A resident transaction whose one graph launch is in flight.
pub struct ResidentGraphInFlight<'executor> {
    // Field order is intentional: abandoned execution waits before graph and
    // workspace owners are destroyed.
    completion: ResidentCompletionEvent,
    timing_start: CudaEvent,
    timing_end: CudaEvent,
    owners: ResidentRunOwners,
    _executor: PhantomData<&'executor Executor>,
}

/// A resident transaction after its one terminal completion wait.
pub struct ResidentGraphSynchronized<'executor> {
    device_elapsed_ns: u64,
    owners: ResidentRunOwners,
    _executor: PhantomData<&'executor Executor>,
}

struct StagedResidentOutput {
    name: String,
    buffer: CudaBuffer,
}

/// Borrow-free decoded receipt. Store mutation happens only in [`Self::commit`].
pub struct ObservedResidentGraphReceipt {
    encoded_len: usize,
    device_elapsed_ns: u64,
    device_scan_invocations: u64,
    device_filter_invocations: u64,
    semantic_scan_invocations: u64,
    semantic_filter_invocations: u64,
    iterations: u32,
    terminal: std::result::Result<(), ResidentGraphExecutionError>,
    outputs: Vec<StagedResidentOutput>,
    source_epoch: u64,
    relation_registration: Vec<(RelId, String)>,
    transaction_identity: Arc<()>,
    provider: Arc<CudaKernelProvider>,
    phase_timings: Option<ResidentFinalObservationPhaseTimings>,
}

/// Opt-in wall-clock breakdown of final resident receipt observation.
#[doc(hidden)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidentFinalObservationPhaseTimings {
    /// Time spent in the sole pinned device-to-host receipt transfer.
    pub receipt_d2h_ns: u64,
    /// Time spent decoding the receipt, resolving schemas, and staging outputs.
    pub decode_schema_staging_ns: u64,
}

fn resident_source_set_snapshot(
    provider: &CudaKernelProvider,
    name: &str,
    version: u64,
    buffer: &CudaBuffer,
) -> std::result::Result<ResidentSourceSetSnapshot, ResidentGraphDeclineReason> {
    if !buffer.canonical_full_row_set_certified() {
        return Err(ResidentGraphDeclineReason::SourceSetUncertified {
            relation: name.to_owned(),
        });
    }
    let manager_ptr = Arc::as_ptr(provider.memory()) as usize;
    let runtime = provider.memory().runtime().ok_or_else(|| {
        ResidentGraphDeclineReason::SourceSetUncertified {
            relation: format!("{name} memory manager has no device runtime"),
        }
    })?;
    if !Arc::ptr_eq(provider.device(), provider.memory().device())
        || !Arc::ptr_eq(provider.device(), runtime.device())
        || u32::try_from(provider.device().ordinal()).ok() != Some(runtime.device_ordinal())
    {
        return Err(ResidentGraphDeclineReason::SourceSetUncertified {
            relation: format!("{name} memory manager does not match the resident provider"),
        });
    }
    let column_blocks = buffer
        .columns()
        .iter()
        .enumerate()
        .map(|(column_index, column)| {
            let CudaColumn::Owned(column) = column else {
                return Err(ResidentGraphDeclineReason::SourceSetUncertified {
                    relation: format!("{name} column {column_index} is externally owned"),
                });
            };
            if column.memory_manager_ptr_value() != manager_ptr {
                return Err(ResidentGraphDeclineReason::SourceSetUncertified {
                    relation: format!(
                        "{name} column {column_index} belongs to another memory manager"
                    ),
                });
            }
            let block = column.runtime_block().map(BlockId::from_block);
            if block.is_none() && buffer.num_rows() != 0 {
                return Err(ResidentGraphDeclineReason::SourceSetUncertified {
                    relation: format!("{name} column {column_index} is not runtime tracked"),
                });
            }
            Ok(block)
        })
        .collect::<std::result::Result<Vec<_>, _>>()?;
    let row_count = buffer.num_rows_device();
    if row_count.memory_manager_ptr_value() != manager_ptr {
        return Err(ResidentGraphDeclineReason::SourceSetUncertified {
            relation: format!("{name} logical count belongs to another memory manager"),
        });
    }
    let row_count_block = row_count
        .runtime_block()
        .map(BlockId::from_block)
        .ok_or_else(|| ResidentGraphDeclineReason::SourceSetUncertified {
            relation: format!("{name} logical count is not runtime tracked"),
        })?;
    Ok(ResidentSourceSetSnapshot {
        name: name.to_owned(),
        version,
        schema: buffer.schema().clone(),
        row_capacity: buffer.num_rows(),
        column_blocks,
        row_count_block,
    })
}

fn validate_resident_source_set_snapshots(
    executor: &Executor,
    source_epoch: u64,
    snapshots: &[ResidentSourceSetSnapshot],
) -> std::result::Result<(), ResidentGraphExecutionError> {
    if executor.store.mutation_epoch() != source_epoch {
        return Err(resident_decline_error(
            ResidentGraphDeclineReason::SourceSetUncertified {
                relation: "relation store changed after resident preparation".to_owned(),
            },
        ));
    }
    for snapshot in snapshots {
        let Some((buffer, version)) = executor.store.get_with_version(&snapshot.name) else {
            return Err(resident_decline_error(
                ResidentGraphDeclineReason::SourceSetUncertified {
                    relation: snapshot.name.clone(),
                },
            ));
        };
        let current =
            resident_source_set_snapshot(&executor.provider, &snapshot.name, version, buffer)
                .map_err(resident_decline_error)?;
        if &current != snapshot {
            return Err(resident_decline_error(
                ResidentGraphDeclineReason::SourceSetUncertified {
                    relation: snapshot.name.clone(),
                },
            ));
        }
    }
    Ok(())
}

impl<'executor> PreparedResidentGraph<'executor> {
    /// Return immutable prelaunch topology and memory diagnostics.
    pub fn preflight_report(&self) -> &ResidentGraphPreflightReport {
        &self.preflight_report
    }

    /// Move out the opt-in raw prepare diagnostic without deriving or emitting it.
    #[doc(hidden)]
    pub fn take_prepare_diagnostic(&mut self) -> Option<ResidentPrepareDiagnostics> {
        self.prepare_diagnostic.take()
    }

    #[cfg(feature = "resident-graph-tests")]
    pub(crate) fn invalidate_expected_source_epoch(&mut self) {
        self.owners.source_epoch = self.owners.source_epoch.wrapping_add(1);
    }

    /// Launch the already-instantiated graph exactly once.
    pub fn launch(
        mut self,
    ) -> std::result::Result<ResidentGraphInFlight<'executor>, ResidentGraphExecutionError> {
        validate_resident_source_set_snapshots(
            self.source_guard,
            self.owners.source_epoch,
            &self.source_set_snapshots,
        )?;
        self.owners
            .execution_domain
            .preflight(&mut self.owners.recorder)
            .map_err(runtime_error)?;
        let timing_start = self
            .owners
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
            .map_err(|error| {
                runtime_error(format!("resident timing-start event failed: {error}"))
            })?;
        self.owners
            .graph
            .launch(&self.owners.stream)
            .map_err(runtime_error)?;
        let timing_end = match self
            .owners
            .stream
            .record_event(Some(CUevent_flags::CU_EVENT_DEFAULT))
        {
            Ok(event) => event,
            Err(error) => {
                let _ = self.owners.stream.synchronize();
                return Err(runtime_error(format!(
                    "resident timing-end event failed after graph launch: {error}"
                )));
            }
        };
        self.owners
            .runtime
            .record_conditional_graph_launch(self.has_device_status_writer);
        let recorder = std::mem::replace(
            &mut self.owners.recorder,
            self.owners.execution_domain.new_strict_recorder(),
        );
        if let Err(error) = self.owners.execution_domain.commit(recorder) {
            let _ = self.owners.stream.synchronize();
            return Err(runtime_error(error));
        }
        let completion = match self
            .owners
            .runtime
            .record_resident_completion_event(&self.owners.stream)
        {
            Ok(completion) => completion,
            Err(error) => {
                let _ = self.owners.stream.synchronize();
                return Err(runtime_error(error));
            }
        };
        Ok(ResidentGraphInFlight {
            completion,
            timing_start,
            timing_end,
            owners: self.owners,
            _executor: PhantomData,
        })
    }
}

impl<'executor> ResidentGraphInFlight<'executor> {
    /// Wait for the one real completion event and retain every graph owner.
    pub fn synchronize_core(
        mut self,
    ) -> std::result::Result<ResidentGraphSynchronized<'executor>, ResidentGraphExecutionError>
    {
        self.completion.synchronize().map_err(runtime_error)?;
        let elapsed_ms = self
            .timing_start
            .elapsed_ms(&self.timing_end)
            .map_err(|error| {
                runtime_error(format!("resident CUDA-event timing failed: {error}"))
            })?;
        if !elapsed_ms.is_finite() || elapsed_ms < 0.0 {
            return Err(runtime_error("resident CUDA-event timing was not finite"));
        }
        let device_elapsed_ns = (f64::from(elapsed_ms) * 1_000_000.0)
            .round()
            .clamp(0.0, u64::MAX as f64) as u64;
        Ok(ResidentGraphSynchronized {
            device_elapsed_ns,
            owners: self.owners,
            _executor: PhantomData,
        })
    }
}

impl<'executor> ResidentGraphSynchronized<'executor> {
    /// Perform the transaction's only device-to-host observation and decode it.
    pub fn observe_final_receipt(
        mut self,
    ) -> std::result::Result<ObservedResidentGraphReceipt, ResidentGraphExecutionError> {
        let phase_diagnostics =
            std::env::var("XLOG_RESIDENT_LATENCY_DIAGNOSTICS").as_deref() == Ok("1");
        let receipt_d2h_started = phase_diagnostics.then(Instant::now);
        let encoded_len = self.owners.receipt.len_bytes();
        let bytes = self
            .owners
            .provider
            .observe_resident_packed_receipt(
                &self.owners.receipt,
                &mut self.owners.pinned_receipt,
                &self.owners.stream,
            )
            .map_err(runtime_error)?;
        let receipt_d2h_ns = receipt_d2h_started
            .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        let decode_started = phase_diagnostics.then(Instant::now);
        let relation_count_len = self.owners.output_indices.len();
        let schema_winner_count = self.owners.output_schema_plans.len();
        let expected_count_fields = relation_count_len + 4 + schema_winner_count;
        if self.owners.receipt.relation_count_len() as usize != relation_count_len
            || self.owners.receipt.device_trace_field_count() != 4
            || self.owners.receipt.schema_winner_count() as usize != schema_winner_count
            || self.owners.receipt.total_count_field_len() as usize != expected_count_fields
            || bytes.len() != encoded_len
            || encoded_len != 44 + 4 * expected_count_fields
        {
            return Err(runtime_error(format!(
                "malformed resident receipt length: got {}, expected {}",
                bytes.len(),
                44 + 4 * expected_count_fields
            )));
        }
        let field_u32 = |offset| {
            read_receipt_u32(&bytes, offset)
                .ok_or_else(|| runtime_error("truncated resident receipt u32 field"))
        };
        let field_u64 = |offset| {
            read_receipt_u64(&bytes, offset)
                .ok_or_else(|| runtime_error("truncated resident receipt u64 field"))
        };
        let code = field_u32(0)?;
        let op_id = field_u32(4)?;
        let resource_code = field_u32(8)?;
        let iterations = field_u32(12)?;
        let limit = field_u32(16)?;
        let reserved = field_u32(20)?;
        let required = field_u64(24)?;
        let capacity = field_u64(32)?;
        let changed = field_u32(40)?;
        if reserved != 0 || changed > 1 {
            return Err(runtime_error(
                "malformed resident terminal status reserved/changed field",
            ));
        }
        let terminal = match code {
            value if value == ResidentTerminalCode::Success as u32 => {
                if resource_code != ResidentResourceCode::None as u32
                    || limit != 0
                    || required != 0
                    || capacity != 0
                {
                    return Err(runtime_error("malformed resident success payload"));
                }
                Ok(())
            }
            value if value == ResidentTerminalCode::IterationLimit as u32 => {
                if resource_code != ResidentResourceCode::None as u32
                    || required != 0
                    || capacity != 0
                    || iterations > limit
                {
                    return Err(runtime_error("malformed resident iteration-limit payload"));
                }
                Err(ResidentGraphExecutionError::IterationLimit {
                    limit,
                    completed: iterations,
                })
            }
            value if value == ResidentTerminalCode::CapacityOverflow as u32 => {
                if resource_code != ResidentResourceCode::OutputRows as u32 || required <= capacity
                {
                    return Err(runtime_error(
                        "malformed resident capacity-overflow payload",
                    ));
                }
                Err(ResidentGraphExecutionError::CapacityOverflow {
                    op_id,
                    required,
                    capacity,
                })
            }
            value if value == ResidentTerminalCode::ResourceExhausted as u32 => {
                let resource = match resource_code {
                    value
                        if value == ResidentResourceCode::SetHashSlots as u32
                            || value == ResidentResourceCode::JoinBuckets as u32
                            || value == ResidentResourceCode::JoinChains as u32 =>
                    {
                        "workspace_slots"
                    }
                    value if value == ResidentResourceCode::InputRows as u32 => "input_rows",
                    value if value == ResidentResourceCode::OutputRows as u32 => "output_rows",
                    _ => return Err(runtime_error("unknown resident resource code")),
                };
                if required <= capacity {
                    return Err(runtime_error("malformed resident resource payload"));
                }
                Err(ResidentGraphExecutionError::ResourceExhausted {
                    op_id,
                    resource,
                    required,
                    capacity,
                })
            }
            _ => {
                return Err(runtime_error(format!(
                    "unknown resident terminal code {code}"
                )))
            }
        };

        let mut counts = Vec::with_capacity(self.owners.output_indices.len());
        for index in 0..self.owners.output_indices.len() {
            counts.push(field_u32(44 + index * 4)?);
        }
        let device_scan_invocations = u64::from(field_u32(44 + relation_count_len * 4)?);
        let device_filter_invocations = u64::from(field_u32(48 + relation_count_len * 4)?);
        let semantic_scan_invocations = u64::from(field_u32(52 + relation_count_len * 4)?);
        let semantic_filter_invocations = u64::from(field_u32(56 + relation_count_len * 4)?);
        let schema_winner_offset = 60 + relation_count_len * 4;
        let mut schema_winner_ids = Vec::with_capacity(schema_winner_count);
        for index in 0..schema_winner_count {
            schema_winner_ids.push(field_u32(schema_winner_offset + index * 4)?);
        }
        let mut outputs = Vec::new();
        if terminal.is_ok() {
            let selected_schemas = resident_resolve_output_schemas(
                &self.owners.output_schema_plans,
                &schema_winner_ids,
            )
            .map_err(runtime_error)?;
            let mut cache_entries = Vec::with_capacity(counts.len());
            for ((_, relation_index), count) in self
                .owners
                .output_indices
                .iter()
                .zip(counts.iter().copied())
            {
                let relation = private_relation(&self.owners.relations, *relation_index)
                    .map_err(runtime_error)?;
                if count > relation.capacity() {
                    return Err(runtime_error(format!(
                        "resident receipt count {count} exceeds output capacity {}",
                        relation.capacity()
                    )));
                }
                cache_entries.push((relation.buffer(), count));
            }
            self.owners
                .provider
                .finalize_resident_logical_counts(&cache_entries)
                .map_err(runtime_error)?;
            for ((name, relation_index), schema) in
                self.owners.output_indices.iter().zip(selected_schemas)
            {
                let relation = self
                    .owners
                    .relations
                    .get_mut(*relation_index)
                    .and_then(Option::take)
                    .ok_or_else(|| runtime_error("resident output relation owner missing"))?;
                outputs.push(StagedResidentOutput {
                    name: name.clone(),
                    buffer: relation
                        .into_buffer_with_observed_schema(schema)
                        .map_err(runtime_error)?,
                });
            }
        }

        let decode_schema_staging_ns = decode_started
            .map(|started| u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX))
            .unwrap_or(0);
        Ok(ObservedResidentGraphReceipt {
            encoded_len,
            device_elapsed_ns: self.device_elapsed_ns,
            device_scan_invocations,
            device_filter_invocations,
            semantic_scan_invocations,
            semantic_filter_invocations,
            iterations,
            terminal,
            outputs,
            source_epoch: self.owners.source_epoch,
            relation_registration: self.owners.relation_registration.clone(),
            transaction_identity: Arc::clone(&self.owners.transaction_identity),
            provider: Arc::clone(&self.owners.provider),
            phase_timings: phase_diagnostics.then_some(ResidentFinalObservationPhaseTimings {
                receipt_d2h_ns,
                decode_schema_staging_ns,
            }),
        })
    }
}

impl ObservedResidentGraphReceipt {
    /// Exact bytes transferred in the sole final observation.
    pub fn encoded_len(&self) -> usize {
        self.encoded_len
    }

    /// Return opt-in final-observation phase timings when diagnostics were enabled.
    #[doc(hidden)]
    pub fn phase_timings(&self) -> Option<ResidentFinalObservationPhaseTimings> {
        self.phase_timings
    }

    /// Number of relation registrations validated by commit.
    #[doc(hidden)]
    pub fn relation_registration_count(&self) -> usize {
        self.relation_registration.len()
    }

    /// Device execution duration measured between CUDA events around the graph launch.
    pub fn device_elapsed_ns(&self) -> u64 {
        self.device_elapsed_ns
    }

    /// Actual scan invocations counted by device kernels, including WHILE replays.
    pub fn device_scan_invocations(&self) -> u64 {
        self.device_scan_invocations
    }

    /// Actual filter invocations counted by device kernels, including WHILE replays.
    pub fn device_filter_invocations(&self) -> u64 {
        self.device_filter_invocations
    }

    /// Legacy-semantic scan count, excluding recursive variants whose selected delta was empty.
    pub fn semantic_scan_invocations(&self) -> u64 {
        self.semantic_scan_invocations
    }

    /// Legacy-semantic filter count, excluding recursive variants whose selected delta was empty.
    pub fn semantic_filter_invocations(&self) -> u64 {
        self.semantic_filter_invocations
    }

    /// Number of staged relation heads published by a successful commit.
    pub fn staged_output_count(&self) -> u64 {
        self.outputs.len() as u64
    }

    /// Aggregate recursive iterations reported by the device.
    pub fn iterations(&self) -> u32 {
        self.iterations
    }

    /// Atomically publish staged outputs after optimistic validation.
    pub fn commit(
        mut self,
        executor: &mut Executor,
    ) -> std::result::Result<(), ResidentGraphExecutionError> {
        self.terminal?;
        if !Arc::ptr_eq(&self.transaction_identity, &executor.transaction_identity)
            || !Arc::ptr_eq(&self.provider, &executor.provider)
            || self.source_epoch != executor.store.mutation_epoch()
        {
            return Err(ResidentGraphExecutionError::Runtime(
                "resident transaction became stale before commit".into(),
            ));
        }
        let mut current_registration = executor
            .rel_names
            .iter()
            .map(|(rel, name)| (*rel, name.clone()))
            .collect::<Vec<_>>();
        current_registration.sort_by_key(|(rel, name)| (rel.0, name.clone()));
        if current_registration != self.relation_registration {
            return Err(ResidentGraphExecutionError::Runtime(
                "resident relation registration changed before commit".into(),
            ));
        }
        let additional = self
            .outputs
            .iter()
            .filter(|output| !executor.store.contains(&output.name))
            .count();
        executor
            .store
            .try_reserve_relations(additional)
            .map_err(runtime_error)?;
        executor.common_subexpression_cache.clear();
        for output in self.outputs.drain(..) {
            if let Some(&rel) = executor.name_to_rel.get(&output.name) {
                executor.join_index_cache.invalidate_rel(rel);
            }
            executor.store.put_owned(output.name, output.buffer);
        }
        Ok(())
    }
}

fn runtime_error(error: impl std::fmt::Display) -> ResidentGraphExecutionError {
    ResidentGraphExecutionError::Runtime(error.to_string())
}

impl ResidentBuild<'_> {
    fn source_reference(&mut self, name: &str) -> Result<ResidentBufferRef> {
        if let Some(index) = self.source_aliases.get(name) {
            return Ok(ResidentBufferRef::Private(*index));
        }
        let (row_count, count_tracked, columns_tracked, schema) = {
            let source = self.executor.store.get(name).ok_or_else(|| {
                XlogError::Execution(format!(
                    "resident source {name} disappeared during planning"
                ))
            })?;
            (
                resident_source_logical_count(source.cached_row_count())?,
                source.num_rows_device().runtime_block().is_some(),
                source
                    .columns()
                    .iter()
                    .all(|column| column.runtime_block().is_some()),
                source.schema().clone(),
            )
        };
        match resident_source_binding_route(row_count, count_tracked, columns_tracked)? {
            ResidentSourceBindingRoute::Direct => {
                self.source_names.insert(name.to_owned());
                Ok(ResidentBufferRef::Source(name.to_owned()))
            }
            ResidentSourceBindingRoute::NormalizeEmpty => {
                let index = self.allocate_permanent_relation(schema, 0)?;
                self.source_aliases.insert(name.to_owned(), index);
                Ok(ResidentBufferRef::Private(index))
            }
        }
    }

    fn private(&self, index: usize) -> &ResidentLogicalRelation {
        &self.relations[index]
    }

    fn schema(&self, reference: &ResidentBufferRef) -> Result<&Schema> {
        match reference {
            ResidentBufferRef::Source(name) => self
                .executor
                .store
                .get(name)
                .map(CudaBuffer::schema)
                .ok_or_else(|| {
                    XlogError::Execution(format!("missing resident source relation {name}"))
                }),
            ResidentBufferRef::Private(index) => Ok(&self.private(*index).schema),
        }
    }

    fn allocate_relation(
        &mut self,
        schema: Schema,
        initial_count: u32,
        permanent: bool,
    ) -> Result<usize> {
        if initial_count > 1 {
            return Err(XlogError::Execution(
                "resident logical initial count must be zero or one".into(),
            ));
        }
        let index = self.relations.len();
        self.relations.push(ResidentLogicalRelation {
            schema,
            initial_count,
            permanent,
        });
        Ok(index)
    }

    fn allocate_permanent_relation(&mut self, schema: Schema, initial_count: u32) -> Result<usize> {
        self.allocate_relation(schema, initial_count, true)
    }

    fn allocate_scratch_relation(&mut self, schema: Schema) -> Result<usize> {
        self.allocate_relation(schema, 0, false)
    }

    fn allocation_manifest(
        &self,
        initial_ops: &[ResidentRecordedOp],
        phases: &[ResidentCapturePhase],
    ) -> Result<ResidentAllocationManifest> {
        let mut definitions = vec![None; self.relations.len()];
        let mut last_uses = vec![None; self.relations.len()];
        let mut ordinal = 0usize;
        let mut phase_ranges = Vec::with_capacity(phases.len() + 1);
        phase_ranges.push(resident_record_lifetimes(
            initial_ops,
            &self.relations,
            &mut definitions,
            &mut last_uses,
            &mut ordinal,
        )?);
        for phase in phases {
            let ops = match phase {
                ResidentCapturePhase::Segment { ops, .. }
                | ResidentCapturePhase::ConditionalWhile { ops, .. } => ops,
            };
            phase_ranges.push(resident_record_lifetimes(
                ops,
                &self.relations,
                &mut definitions,
                &mut last_uses,
                &mut ordinal,
            )?);
        }

        let mut slots = Vec::<ResidentPhysicalSlotPlan>::new();
        let mut logical_to_slot = vec![
            ResidentSlotAssignment {
                slot: usize::MAX,
                generation: 0,
            };
            self.relations.len()
        ];
        for (logical, relation) in self.relations.iter().enumerate() {
            if relation.permanent {
                let slot = slots.len();
                slots.push(ResidentPhysicalSlotPlan {
                    schema: relation.schema.clone(),
                    initial_count: relation.initial_count,
                    permanent: true,
                });
                logical_to_slot[logical] = ResidentSlotAssignment {
                    slot,
                    generation: 0,
                };
            }
        }

        let mut scratch_relations = self
            .relations
            .iter()
            .enumerate()
            .filter(|(_, relation)| !relation.permanent)
            .map(|(logical, relation)| {
                let definition = definitions[logical].ok_or_else(|| {
                    XlogError::Execution(format!(
                        "resident scratch relation {logical} is never defined"
                    ))
                })?;
                let last_use = last_uses[logical].unwrap_or(definition);
                if !phase_ranges
                    .iter()
                    .any(|(start, end)| *start <= definition && last_use < *end)
                {
                    return Err(XlogError::Execution(format!(
                        "resident scratch relation {logical} crosses a capture phase boundary"
                    )));
                }
                Ok((
                    logical,
                    definition,
                    last_use,
                    resident_schema_layout(&relation.schema),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        scratch_relations.sort_by_key(|(logical, definition, _, _)| (*definition, *logical));

        let mut scratch_slots = Vec::<ResidentScratchSlotState>::new();
        for (logical, definition, last_use, layout) in scratch_relations {
            let reusable = scratch_slots
                .iter_mut()
                .filter(|slot| slot.layout == layout && slot.last_use < definition)
                .min_by_key(|slot| slot.last_use);
            let assignment = if let Some(slot) = reusable {
                slot.generation = slot.generation.checked_add(1).ok_or_else(|| {
                    XlogError::Execution("resident scratch generation overflow".into())
                })?;
                slot.last_use = last_use;
                ResidentSlotAssignment {
                    slot: slot.slot,
                    generation: slot.generation,
                }
            } else {
                let slot_index = slots.len();
                slots.push(ResidentPhysicalSlotPlan {
                    schema: self.relations[logical].schema.clone(),
                    initial_count: 0,
                    permanent: false,
                });
                scratch_slots.push(ResidentScratchSlotState {
                    slot: slot_index,
                    layout,
                    last_use,
                    generation: 0,
                });
                ResidentSlotAssignment {
                    slot: slot_index,
                    generation: 0,
                }
            };
            logical_to_slot[logical] = assignment;
        }
        if logical_to_slot
            .iter()
            .any(|assignment| assignment.slot == usize::MAX)
        {
            return Err(XlogError::Execution(
                "resident logical relation has no physical slot assignment".into(),
            ));
        }
        resident_validate_slot_assignments(
            &self.relations,
            &definitions,
            &last_uses,
            &slots,
            &logical_to_slot,
        )?;

        let capacity = u64::from(self.capacity);
        let mut relation_bytes = 0u64;
        let mut max_row_bytes = 1u64;
        for slot in &slots {
            max_row_bytes = max_row_bytes.max(slot.schema.row_size_bytes() as u64);
            resident_checked_add(
                &mut relation_bytes,
                resident_relation_device_bytes(&slot.schema, capacity)?,
                "relation",
            )?;
        }
        let filter_scratch_bytes = if self.filter_workspaces.is_empty() {
            0
        } else {
            resident_filter_scratch_device_bytes(capacity)?
        };
        let set_candidate_capacity = capacity.checked_mul(2).ok_or_else(|| {
            XlogError::Execution("resident set candidate capacity overflow".into())
        })?;
        let mut fixed_workspace_bytes = 0u64;
        for bytes in [
            resident_set_workspace_device_bytes(set_candidate_capacity)?,
            resident_join_workspace_device_bytes(capacity)?,
            resident_control_device_bytes(),
            resident_device_trace_bytes(),
            resident_schema_winners_device_bytes(self.heads.len())?,
            resident_packed_receipt_with_schema_winners_device_bytes(self.heads.len())?,
        ] {
            resident_checked_add(&mut fixed_workspace_bytes, bytes, "fixed workspace")?;
        }
        let mut required_bytes = 0u64;
        for bytes in [relation_bytes, filter_scratch_bytes, fixed_workspace_bytes] {
            resident_checked_add(&mut required_bytes, bytes, "manifest")?;
        }

        let permanent_slots = u32::try_from(slots.iter().filter(|slot| slot.permanent).count())
            .map_err(|_| {
                XlogError::Execution("resident permanent slot count exceeds u32".into())
            })?;
        Ok(ResidentAllocationManifest {
            slots,
            logical_to_slot,
            required_bytes,
            relation_bytes,
            filter_scratch_bytes,
            schedule_metadata_bytes: 0,
            fixed_workspace_bytes,
            logical_relation_values: self.relations.len(),
            permanent_slots,
            scratch_slots: u32::try_from(scratch_slots.len()).map_err(|_| {
                XlogError::Execution("resident scratch slot count exceeds u32".into())
            })?,
            filter_scratch_allocations: u32::from(!self.filter_workspaces.is_empty()),
            max_row_bytes,
        })
    }

    fn materialize(
        &self,
        manifest: &ResidentAllocationManifest,
        reservation: &mut GpuMemoryReservation,
        mut diagnostics: Option<&mut ResidentPrepareDiagnostics>,
    ) -> Result<ResidentPhysicalBuild> {
        let capacity = u64::from(self.capacity);
        let mut relations = Vec::with_capacity(manifest.slots.len());
        for slot in &manifest.slots {
            let allocation_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
            let allocation_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
            let mut relation = self
                .executor
                .provider
                .prepare_resident_relation_in_reservation(
                    slot.schema.clone(),
                    capacity,
                    reservation,
                )?;
            if let Some(diagnostics) = diagnostics.as_deref_mut() {
                let allocation_ns = resident_prepare_elapsed_ns(
                    allocation_started.expect("diagnostic timer exists when enabled"),
                );
                let allocation_bytes = reservation.used_bytes().saturating_sub(
                    allocation_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
                );
                diagnostics.relation_slot_allocation_ns = diagnostics
                    .relation_slot_allocation_ns
                    .saturating_add(allocation_ns);
                diagnostics.relation_slot_allocation_ns_max = diagnostics
                    .relation_slot_allocation_ns_max
                    .max(allocation_ns);
                diagnostics.relation_slot_allocation_bytes = diagnostics
                    .relation_slot_allocation_bytes
                    .saturating_add(allocation_bytes);
                diagnostics.relation_device_allocation_calls =
                    diagnostics.relation_device_allocation_calls.saturating_add(
                        u64::try_from(slot.schema.arity())
                            .unwrap_or(u64::MAX)
                            .saturating_add(1),
                    );
            }
            let count_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
            self.executor
                .provider
                .initialize_resident_relation_count(&mut relation, slot.initial_count)?;
            if let Some(diagnostics) = diagnostics.as_deref_mut() {
                let count_ns = resident_prepare_elapsed_ns(
                    count_started.expect("diagnostic timer exists when enabled"),
                );
                diagnostics.count_initialization_ns =
                    diagnostics.count_initialization_ns.saturating_add(count_ns);
                diagnostics.count_initialization_ns_max =
                    diagnostics.count_initialization_ns_max.max(count_ns);
                diagnostics.count_memset_calls = diagnostics.count_memset_calls.saturating_add(1);
            }
            relations.push(Some(relation));
        }
        let filter_scratch = if self.filter_workspaces.is_empty() {
            None
        } else {
            let bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
            let started = diagnostics.as_ref().map(|_| std::time::Instant::now());
            let scratch = self
                .executor
                .provider
                .prepare_resident_filter_scratch_in_reservation(capacity, reservation)?;
            if let Some(diagnostics) = diagnostics.as_deref_mut() {
                let elapsed_ns = resident_prepare_elapsed_ns(
                    started.expect("diagnostic timer exists when enabled"),
                );
                let reserved_bytes = reservation.used_bytes().saturating_sub(
                    bytes_before.expect("diagnostic byte snapshot exists when enabled"),
                );
                diagnostics.workspace_provider_calls =
                    diagnostics.workspace_provider_calls.saturating_add(1);
                diagnostics.filter_scratch_allocation_ns = elapsed_ns;
                diagnostics.filter_scratch_allocation_bytes = reserved_bytes;
            }
            Some(scratch)
        };
        let set_candidate_capacity = capacity.checked_mul(2).ok_or_else(|| {
            XlogError::Execution("resident set candidate capacity overflow".into())
        })?;
        let set_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let set_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let set_workspace = self
            .executor
            .provider
            .prepare_resident_set_workspace_in_reservation(set_candidate_capacity, reservation)?;
        if let Some(diagnostics) = diagnostics.as_deref_mut() {
            let set_ns = resident_prepare_elapsed_ns(
                set_started.expect("diagnostic timer exists when enabled"),
            );
            let set_reserved_bytes = reservation.used_bytes().saturating_sub(
                set_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.workspace_provider_calls =
                diagnostics.workspace_provider_calls.saturating_add(1);
            diagnostics.set_workspace_allocation_ns = set_ns;
            diagnostics.set_workspace_allocation_bytes = set_reserved_bytes;
        }
        let join_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let join_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let join_workspace = self
            .executor
            .provider
            .prepare_resident_join_workspace_in_reservation(capacity, reservation)?;
        if let Some(diagnostics) = diagnostics.as_deref_mut() {
            let join_ns = resident_prepare_elapsed_ns(
                join_started.expect("diagnostic timer exists when enabled"),
            );
            let join_reserved_bytes = reservation.used_bytes().saturating_sub(
                join_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.workspace_provider_calls =
                diagnostics.workspace_provider_calls.saturating_add(1);
            diagnostics.join_workspace_allocation_ns = join_ns;
            diagnostics.join_workspace_allocation_bytes = join_reserved_bytes;
        }
        let control_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let control_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let control = self
            .executor
            .provider
            .prepare_resident_convergence_control_in_reservation(reservation)?;
        if let Some(diagnostics) = diagnostics.as_deref_mut() {
            let control_ns = resident_prepare_elapsed_ns(
                control_started.expect("diagnostic timer exists when enabled"),
            );
            let control_reserved_bytes = reservation.used_bytes().saturating_sub(
                control_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.workspace_provider_calls =
                diagnostics.workspace_provider_calls.saturating_add(1);
            diagnostics.control_allocation_ns = control_ns;
            diagnostics.control_allocation_bytes = control_reserved_bytes;
        }
        Ok(ResidentPhysicalBuild {
            relations,
            filter_scratch,
            set_workspace,
            join_workspace,
            control,
        })
    }

    fn next_op_id(&mut self) -> Result<u32> {
        let op_id = self.next_op_id;
        self.next_op_id = self
            .next_op_id
            .checked_add(1)
            .ok_or_else(|| XlogError::Execution("resident physical op id overflow".into()))?;
        Ok(op_id)
    }

    fn push_physical_op(
        &mut self,
        ops: &mut Vec<ResidentRecordedOp>,
        op: ResidentRecordedOp,
        op_id: u32,
    ) {
        ops.push(op);
        if !self.injection_recorded
            && self
                .injection
                .as_ref()
                .is_some_and(|injection| injection.after_op == op_id)
        {
            let status =
                terminal_status_for_injection(&self.injection.as_ref().expect("checked").status);
            ops.push(ResidentRecordedOp::TestStatus(status));
            self.injection_recorded = true;
        }
    }

    fn scan_reference(
        &mut self,
        rel: RelId,
        override_scan: Option<(RelId, usize, usize)>,
        occurrences: &mut HashMap<RelId, usize>,
    ) -> Result<ResidentBufferRef> {
        let occurrence = occurrences.entry(rel).or_insert(0);
        let current = *occurrence;
        *occurrence += 1;
        if let Some((target, target_occurrence, delta)) = override_scan {
            if rel == target && current == target_occurrence {
                return Ok(ResidentBufferRef::Private(delta));
            }
        }
        let name = self
            .executor
            .rel_names
            .get(&rel)
            .ok_or_else(|| XlogError::Execution(format!("unknown resident relation id {rel:?}")))?
            .clone();
        if let Some(index) = self.heads.get(&name) {
            Ok(ResidentBufferRef::Private(*index))
        } else {
            self.source_reference(&name)
        }
    }

    fn plan_node(
        &mut self,
        node: &RirNode,
        override_scan: Option<(RelId, usize, usize)>,
        occurrences: &mut HashMap<RelId, usize>,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        match node {
            RirNode::Unit => {
                let op_id = self.next_op_id()?;
                let (reference, op) = resident_new_phase_unit(&mut self.relations, op_id)?;
                self.push_physical_op(ops, op, op_id);
                Ok(reference)
            }
            RirNode::Scan { rel } => {
                let reference = self.scan_reference(*rel, override_scan, occurrences)?;
                let op_id = self.next_op_id()?;
                Ok(resident_record_scan_leaf(
                    reference,
                    op_id,
                    resident_semantic_trace_guard(override_scan),
                    ops,
                    |ops, op, op_id| self.push_physical_op(ops, op, op_id),
                ))
            }
            RirNode::Filter { input, predicate } => {
                let input_ref = self.plan_node(input, override_scan, occurrences, ops)?;
                let input_schema = self.schema(&input_ref)?.clone();
                let compact_comparisons =
                    resident_compact_filter_descriptors(predicate, &input_schema)?;
                let workspace = ResidentFilterPlan {
                    compact_comparisons,
                };
                let workspace_index = self.filter_workspaces.len();
                self.filter_workspaces.push(workspace);
                let schema = self
                    .certificate
                    .node_schema(node)
                    .ok_or_else(|| XlogError::Execution("resident filter schema missing".into()))?;
                let output = self.allocate_scratch_relation(schema)?;
                let op_id = self.next_op_id()?;
                self.push_physical_op(
                    ops,
                    ResidentRecordedOp::Filter {
                        input: input_ref,
                        output,
                        workspace: workspace_index,
                        op_id,
                    },
                    op_id,
                );
                ops.push(ResidentRecordedOp::TraceDelta {
                    scan_delta: 0,
                    filter_delta: 1,
                    semantic_guard: resident_semantic_trace_guard(override_scan),
                });
                Ok(ResidentBufferRef::Private(output))
            }
            RirNode::Project { input, columns } => {
                let input_ref = self.plan_node(input, override_scan, occurrences, ops)?;
                let input_schema = self.schema(&input_ref)?.clone();
                let certified_schema = self.certificate.node_schema(node).ok_or_else(|| {
                    XlogError::Execution("resident project schema missing".into())
                })?;
                let schema = self.executor.project_schema(&input_schema, columns)?;
                if schema.arity() != certified_schema.arity()
                    || (0..schema.arity()).any(|column| {
                        schema.column_type(column) != certified_schema.column_type(column)
                    })
                {
                    return Err(XlogError::Execution(
                        "resident project physical schema differs from certificate".into(),
                    ));
                }
                let compact_expressions =
                    resident_compact_project_descriptors(columns, &input_schema, &schema)?;
                let workspace = ResidentProjectPlan {
                    compact_expressions,
                };
                let workspace_index = self.project_workspaces.len();
                self.project_workspaces.push(workspace);
                let output = self.allocate_scratch_relation(schema)?;
                let op_id = self.next_op_id()?;
                self.push_physical_op(
                    ops,
                    ResidentRecordedOp::Project {
                        input: input_ref,
                        output,
                        workspace: workspace_index,
                        op_id,
                    },
                    op_id,
                );
                Ok(ResidentBufferRef::Private(output))
            }
            RirNode::Join {
                left,
                right,
                left_keys,
                right_keys,
                join_type,
            } => {
                let left_ref = self.plan_node(left, override_scan, occurrences, ops)?;
                let right_ref = self.plan_node(right, override_scan, occurrences, ops)?;
                let certified_schema = self
                    .certificate
                    .node_schema(node)
                    .ok_or_else(|| XlogError::Execution("resident join schema missing".into()))?;
                let kind = match join_type {
                    JoinType::Inner => ResidentJoinKind::Inner,
                    JoinType::Semi => ResidentJoinKind::Semi,
                    _ => return Err(XlogError::Execution("uncertified resident join".into())),
                };
                // The provider preserves operand column names. The route
                // certificate canonicalizes those names and binds the same
                // physical column types, so construct the provider's exact
                // schema here and normalize names at the next projection.
                let left_schema = self.schema(&left_ref)?.clone();
                let right_schema = self.schema(&right_ref)?.clone();
                let schema = match kind {
                    ResidentJoinKind::Semi => left_schema,
                    ResidentJoinKind::Inner => {
                        let mut columns = left_schema.columns;
                        columns.extend(right_schema.columns);
                        Schema::new(columns)
                    }
                };
                if schema.arity() != certified_schema.arity()
                    || (0..schema.arity()).any(|column| {
                        schema.column_type(column) != certified_schema.column_type(column)
                    })
                {
                    return Err(XlogError::Execution(
                        "resident join physical schema differs from certificate".into(),
                    ));
                }
                let output = self.allocate_scratch_relation(schema).map_err(|error| {
                    XlogError::Execution(format!(
                        "resident join intermediate arity {} at {node:?} could not be allocated: {error}",
                        certified_schema.arity()
                    ))
                })?;
                let op_id = self.next_op_id()?;
                self.push_physical_op(
                    ops,
                    ResidentRecordedOp::Join {
                        kind,
                        left: left_ref,
                        left_key: left_keys[0],
                        right: right_ref,
                        right_key: right_keys[0],
                        output,
                        op_id,
                    },
                    op_id,
                );
                Ok(ResidentBufferRef::Private(output))
            }
            RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
                self.plan_node(fallback, override_scan, occurrences, ops)
            }
            RirNode::Distinct { input, .. } => {
                let input_ref = self.plan_node(input, override_scan, occurrences, ops)?;
                self.plan_dedup(input_ref, node, ops)
            }
            RirNode::Diff { left, right } => {
                let left_ref = self.plan_node(left, override_scan, occurrences, ops)?;
                let right_ref = self.plan_node(right, override_scan, occurrences, ops)?;
                self.plan_diff(left_ref, right_ref, node, ops)
            }
            RirNode::Union { inputs } => {
                let fold_mode = resident_union_fold_mode(inputs.len())?;
                let mut iter = inputs.iter();
                let first = iter
                    .next()
                    .expect("union fold mode rejected the empty case");
                let mut result = self.plan_node(first, override_scan, occurrences, ops)?;
                if fold_mode == ResidentUnionFoldMode::SelfUnion {
                    return self.plan_union(result.clone(), result, node, ops);
                }
                for input in iter {
                    let right = self.plan_node(input, override_scan, occurrences, ops)?;
                    result = self.plan_union(result, right, node, ops)?;
                }
                Ok(result)
            }
            RirNode::Fixpoint { .. }
            | RirNode::GroupBy { .. }
            | RirNode::TensorMaskedJoin { .. } => Err(XlogError::Execution(
                "uncertified node reached resident planner".into(),
            )),
        }
    }

    fn plan_dedup(
        &mut self,
        input: ResidentBufferRef,
        node: &RirNode,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let schema = self
            .certificate
            .node_schema(node)
            .ok_or_else(|| XlogError::Execution("resident dedup schema missing".into()))?;
        let output = self.allocate_scratch_relation(schema)?;
        // Dedup is union against itself in the exact full-row set primitive.
        let op_id = self.next_op_id()?;
        self.push_physical_op(
            ops,
            ResidentRecordedOp::Union {
                left: input.clone(),
                right: input,
                output,
                op_id,
            },
            op_id,
        );
        Ok(ResidentBufferRef::Private(output))
    }

    fn plan_union(
        &mut self,
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        schema_node: &RirNode,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let schema = self
            .certificate
            .node_schema(schema_node)
            .ok_or_else(|| XlogError::Execution("resident union schema missing".into()))?;
        self.plan_union_schema(left, right, schema, ops)
    }

    fn plan_union_schema(
        &mut self,
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        schema: Schema,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let output = self.allocate_scratch_relation(schema)?;
        let op_id = self.next_op_id()?;
        self.push_physical_op(
            ops,
            ResidentRecordedOp::Union {
                left,
                right,
                output,
                op_id,
            },
            op_id,
        );
        Ok(ResidentBufferRef::Private(output))
    }

    fn plan_diff(
        &mut self,
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        schema_node: &RirNode,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let schema = self
            .certificate
            .node_schema(schema_node)
            .ok_or_else(|| XlogError::Execution("resident diff schema missing".into()))?;
        self.plan_diff_schema(left, right, schema, ops)
    }

    fn plan_diff_schema(
        &mut self,
        left: ResidentBufferRef,
        right: ResidentBufferRef,
        schema: Schema,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let output = self.allocate_scratch_relation(schema)?;
        let op_id = self.next_op_id()?;
        self.push_physical_op(
            ops,
            ResidentRecordedOp::Diff {
                left,
                right,
                output,
                op_id,
            },
            op_id,
        );
        Ok(ResidentBufferRef::Private(output))
    }

    fn plan_copy(
        &mut self,
        input: ResidentBufferRef,
        output: usize,
        schema: &Schema,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<()> {
        let compact_expressions = (0..schema.arity())
            .map(|column| {
                let scalar = schema.column_type(column).ok_or_else(|| {
                    XlogError::Execution("resident identity project column is missing".into())
                })?;
                Ok(ResidentProjectExpressionDescriptor::column(
                    u32::try_from(column).map_err(|_| {
                        XlogError::Execution("resident project column exceeds u32".into())
                    })?,
                    u32::try_from(scalar.size_bytes()).map_err(|_| {
                        XlogError::Execution("resident project scalar width exceeds u32".into())
                    })?,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        let workspace = ResidentProjectPlan {
            compact_expressions,
        };
        let workspace_index = self.project_workspaces.len();
        self.project_workspaces.push(workspace);
        let op_id = self.next_op_id()?;
        self.push_physical_op(
            ops,
            ResidentRecordedOp::Project {
                input,
                output,
                workspace: workspace_index,
                op_id,
            },
            op_id,
        );
        Ok(())
    }

    fn normalize_to_schema(
        &mut self,
        input: ResidentBufferRef,
        schema: &Schema,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let input_schema = self.schema(&input)?;
        if input_schema == schema {
            return Ok(input);
        }
        if input_schema.arity() != schema.arity()
            || input_schema
                .columns
                .iter()
                .zip(&schema.columns)
                .any(|(left, right)| left.1 != right.1)
        {
            return Err(XlogError::Execution(format!(
                "resident rule output schema {:?} cannot be normalized to head schema {:?}",
                input_schema, schema
            )));
        }
        let output = self.allocate_scratch_relation(schema.clone())?;
        self.plan_copy(input, output, schema, ops)?;
        Ok(ResidentBufferRef::Private(output))
    }

    fn merge_phase_contribution(
        &mut self,
        current: Option<ResidentBufferRef>,
        contribution: ResidentBufferRef,
        schema: &Schema,
        ops: &mut Vec<ResidentRecordedOp>,
    ) -> Result<ResidentBufferRef> {
        let contribution = self.normalize_to_schema(contribution, schema, ops)?;
        resident_phase_merge(current, contribution, |step| match step {
            ResidentPhaseMergeStep::Deduplicate(input) => {
                self.plan_union_schema(input.clone(), input, schema.clone(), ops)
            }
            ResidentPhaseMergeStep::Union(left, right) => {
                self.plan_union_schema(left, right, schema.clone(), ops)
            }
        })
    }
}

fn private_relation(
    relations: &[Option<ResidentRelation>],
    index: usize,
) -> Result<&ResidentRelation> {
    relations
        .get(index)
        .and_then(Option::as_ref)
        .ok_or_else(|| XlogError::Execution(format!("resident relation slot {index} is missing")))
}

fn collect_recorded_scans(node: &RirNode, output: &mut Vec<RelId>) {
    match node {
        RirNode::Unit | RirNode::TensorMaskedJoin { .. } => {}
        RirNode::Scan { rel } => output.push(*rel),
        RirNode::Filter { input, .. }
        | RirNode::Project { input, .. }
        | RirNode::GroupBy { input, .. }
        | RirNode::Distinct { input, .. } => collect_recorded_scans(input, output),
        RirNode::Join { left, right, .. } | RirNode::Diff { left, right } => {
            collect_recorded_scans(left, output);
            collect_recorded_scans(right, output);
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            collect_recorded_scans(fallback, output);
        }
        RirNode::Union { inputs } => {
            for input in inputs {
                collect_recorded_scans(input, output);
            }
        }
        RirNode::Fixpoint {
            base, recursive, ..
        } => {
            collect_recorded_scans(base, output);
            collect_recorded_scans(recursive, output);
        }
    }
}

fn resident_decline_error(reason: ResidentGraphDeclineReason) -> ResidentGraphExecutionError {
    ResidentGraphExecutionError::Declined(reason)
}

fn conditional_graph_error(error: CudaConditionalGraphUnavailable) -> ResidentGraphExecutionError {
    if error.is_unsupported() {
        resident_decline_error(ResidentGraphDeclineReason::ConditionalGraphUnavailable {
            detail: error.decline_detail(),
        })
    } else {
        runtime_error(error.decline_detail())
    }
}

fn read_receipt_u32(bytes: &[u8], offset: usize) -> Option<u32> {
    bytes
        .get(offset..offset.checked_add(4)?)?
        .try_into()
        .ok()
        .map(u32::from_le_bytes)
}

fn read_receipt_u64(bytes: &[u8], offset: usize) -> Option<u64> {
    bytes
        .get(offset..offset.checked_add(8)?)?
        .try_into()
        .ok()
        .map(u64::from_le_bytes)
}

fn resident_compact_scalar(value: &ConstValue) -> Result<(ScalarType, u64)> {
    match value {
        ConstValue::Symbol(value) => Ok((ScalarType::Symbol, u64::from(symbol::intern(value)))),
        ConstValue::U32(value) => Ok((ScalarType::U32, u64::from(*value))),
        ConstValue::U64(value) => Ok((ScalarType::U64, *value)),
        _ => Err(XlogError::Execution(
            "uncertified resident constant type".into(),
        )),
    }
}

fn resident_compact_filter_descriptors(
    expression: &Expr,
    input_schema: &Schema,
) -> Result<Vec<ResidentFilterComparisonDescriptor>> {
    fn operand(expression: &Expr, input_schema: &Schema) -> Result<(u32, u32, ScalarType, u64)> {
        match expression {
            Expr::Column(column) => {
                let scalar = input_schema.column_type(*column).ok_or_else(|| {
                    XlogError::Execution("resident filter column is out of range".into())
                })?;
                Ok((
                    0,
                    u32::try_from(*column).map_err(|_| {
                        XlogError::Execution("resident filter column exceeds u32".into())
                    })?,
                    scalar,
                    0,
                ))
            }
            Expr::Const(value) => {
                let (scalar, bits) = resident_compact_scalar(value)?;
                Ok((1, 0, scalar, bits))
            }
            _ => Err(XlogError::Execution(
                "uncertified resident filter operand".into(),
            )),
        }
    }

    fn append(
        expression: &Expr,
        input_schema: &Schema,
        output: &mut Vec<ResidentFilterComparisonDescriptor>,
    ) -> Result<()> {
        match expression {
            Expr::And(parts) => {
                for part in parts {
                    append(part, input_schema, output)?;
                }
                Ok(())
            }
            Expr::Compare { left, op, right } => {
                let (left_kind, left_column, left_type, left_constant) =
                    operand(left, input_schema)?;
                let (right_kind, right_column, right_type, right_constant) =
                    operand(right, input_schema)?;
                if left_type != right_type {
                    return Err(XlogError::Execution(
                        "resident filter operand scalar types differ".into(),
                    ));
                }
                let width = u32::try_from(left_type.size_bytes()).map_err(|_| {
                    XlogError::Execution("resident filter scalar width exceeds u32".into())
                })?;
                if !matches!(width, 4 | 8) {
                    return Err(XlogError::Execution(
                        "resident filter scalar width is unsupported".into(),
                    ));
                }
                output.push(ResidentFilterComparisonDescriptor {
                    left_kind,
                    left_column,
                    right_kind,
                    right_column,
                    op: u32::from(resident_compare_op(*op) as u8),
                    width,
                    reserved_zero: 0,
                    reserved_one: 0,
                    left_constant,
                    right_constant,
                });
                Ok(())
            }
            _ => Err(XlogError::Execution(
                "uncertified resident filter expression".into(),
            )),
        }
    }

    let mut output = Vec::new();
    append(expression, input_schema, &mut output)?;
    Ok(output)
}

fn resident_compact_project_descriptors(
    columns: &[ProjectExpr],
    input_schema: &Schema,
    output_schema: &Schema,
) -> Result<Vec<ResidentProjectExpressionDescriptor>> {
    if columns.len() != output_schema.arity() {
        return Err(XlogError::Execution(
            "resident project expression count differs from output arity".into(),
        ));
    }
    columns
        .iter()
        .enumerate()
        .map(|(output_column, expression)| {
            let output_type = output_schema.column_type(output_column).ok_or_else(|| {
                XlogError::Execution("resident project output column is out of range".into())
            })?;
            let width = u32::try_from(output_type.size_bytes()).map_err(|_| {
                XlogError::Execution("resident project scalar width exceeds u32".into())
            })?;
            if !matches!(width, 4 | 8) {
                return Err(XlogError::Execution(
                    "resident project scalar width is unsupported".into(),
                ));
            }
            match expression {
                ProjectExpr::Column(input_column) => {
                    if input_schema.column_type(*input_column) != Some(output_type) {
                        return Err(XlogError::Execution(
                            "resident project column scalar type differs from output".into(),
                        ));
                    }
                    Ok(ResidentProjectExpressionDescriptor::column(
                        u32::try_from(*input_column).map_err(|_| {
                            XlogError::Execution("resident project column exceeds u32".into())
                        })?,
                        width,
                    ))
                }
                ProjectExpr::Computed(Expr::Const(value), declared_type) => {
                    let (constant_type, bits) = resident_compact_scalar(value)?;
                    if constant_type != *declared_type || constant_type != output_type {
                        return Err(XlogError::Execution(
                            "resident project constant scalar type differs from output".into(),
                        ));
                    }
                    Ok(ResidentProjectExpressionDescriptor::constant(width, bits))
                }
                _ => Err(XlogError::Execution(
                    "uncertified resident projection expression".into(),
                )),
            }
        })
        .collect()
}

fn resident_compare_op(op: RirCompareOp) -> CudaCompareOp {
    match op {
        RirCompareOp::Eq => CudaCompareOp::Eq,
        RirCompareOp::Ne => CudaCompareOp::Ne,
        RirCompareOp::Lt => CudaCompareOp::Lt,
        RirCompareOp::Le => CudaCompareOp::Le,
        RirCompareOp::Gt => CudaCompareOp::Gt,
        RirCompareOp::Ge => CudaCompareOp::Ge,
    }
}

fn terminal_status_for_injection(status: &ResidentGraphDeviceStatus) -> ResidentTerminalStatus {
    match status {
        ResidentGraphDeviceStatus::Success { iterations } => ResidentTerminalStatus {
            code: ResidentTerminalCode::Success as u32,
            iterations: *iterations,
            ..ResidentTerminalStatus::default()
        },
        ResidentGraphDeviceStatus::IterationLimit { limit, completed } => ResidentTerminalStatus {
            code: ResidentTerminalCode::IterationLimit as u32,
            iterations: *completed,
            limit: *limit,
            ..ResidentTerminalStatus::default()
        },
        ResidentGraphDeviceStatus::CapacityOverflow {
            op_id,
            required,
            capacity,
        } => ResidentTerminalStatus {
            code: ResidentTerminalCode::CapacityOverflow as u32,
            op_id: *op_id,
            resource_code: ResidentResourceCode::OutputRows as u32,
            required: *required,
            capacity: *capacity,
            ..ResidentTerminalStatus::default()
        },
        ResidentGraphDeviceStatus::ResourceExhausted {
            op_id,
            required,
            capacity,
            ..
        } => ResidentTerminalStatus {
            code: ResidentTerminalCode::ResourceExhausted as u32,
            op_id: *op_id,
            resource_code: ResidentResourceCode::SetHashSlots as u32,
            required: *required,
            capacity: *capacity,
            ..ResidentTerminalStatus::default()
        },
    }
}

fn validate_compact_resident_node_envelope(
    node: &RirNode,
    schema_for: &impl Fn(&RirNode) -> std::result::Result<Schema, ResidentGraphDeclineReason>,
) -> std::result::Result<(), ResidentGraphDeclineReason> {
    match node {
        RirNode::Unit | RirNode::Scan { .. } => Ok(()),
        RirNode::Filter { input, .. } | RirNode::Project { input, .. } => {
            validate_compact_resident_node_envelope(input, schema_for)
        }
        RirNode::Distinct { input, key_cols } => {
            validate_compact_resident_node_envelope(input, schema_for)?;
            let arity = schema_for(input)?.arity();
            if key_cols.len() != arity
                || key_cols
                    .iter()
                    .copied()
                    .enumerate()
                    .any(|(column, key)| column != key)
            {
                return Err(resident_workspace_decline(format!(
                    "compact resident Distinct requires canonical full-row key columns 0..{arity}"
                )));
            }
            Ok(())
        }
        RirNode::Union { inputs } => {
            if inputs.is_empty() {
                return Err(resident_workspace_decline(
                    "compact resident Union requires at least one input",
                ));
            }
            for input in inputs {
                validate_compact_resident_node_envelope(input, schema_for)?;
            }
            Ok(())
        }
        RirNode::Diff { left, right } => {
            validate_compact_resident_node_envelope(left, schema_for)?;
            validate_compact_resident_node_envelope(right, schema_for)
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
        } => {
            if !matches!(join_type, JoinType::Inner | JoinType::Semi) {
                return Err(resident_workspace_decline(format!(
                    "compact resident Join does not support {join_type:?}"
                )));
            }
            if left_keys.len() != 1 || right_keys.len() != 1 {
                return Err(resident_workspace_decline(
                    "compact resident Join requires exactly one key per input",
                ));
            }
            validate_compact_resident_node_envelope(left, schema_for)?;
            validate_compact_resident_node_envelope(right, schema_for)?;
            let left_schema = schema_for(left)?;
            let right_schema = schema_for(right)?;
            let left_key = left_keys[0];
            let right_key = right_keys[0];
            if left_key >= left_schema.arity() || right_key >= right_schema.arity() {
                return Err(resident_workspace_decline(
                    "compact resident Join key is outside its input schema",
                ));
            }
            let left_key_type = left_schema
                .column_type(left_key)
                .expect("left key bounds checked");
            let right_key_type = right_schema
                .column_type(right_key)
                .expect("right key bounds checked");
            if left_key_type != right_key_type
                || !matches!(
                    left_key_type,
                    ScalarType::U32 | ScalarType::U64 | ScalarType::Symbol
                )
            {
                return Err(resident_workspace_decline(
                    "compact resident Join requires matching U32, U64, or Symbol key types",
                ));
            }
            if left_key_type.size_bytes() != right_key_type.size_bytes() {
                return Err(resident_workspace_decline(
                    "compact resident Join key widths differ",
                ));
            }
            let output_arity = match join_type {
                JoinType::Inner => left_schema
                    .arity()
                    .checked_add(right_schema.arity())
                    .ok_or_else(|| {
                        resident_workspace_decline("compact resident Join arity overflow")
                    })?,
                JoinType::Semi => left_schema.arity(),
                _ => unreachable!("join kind checked above"),
            };
            if output_arity > 17 {
                return Err(resident_workspace_decline(
                    "compact resident Join output arity exceeds 17",
                ));
            }
            Ok(())
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            validate_compact_resident_node_envelope(fallback, schema_for)
        }
        RirNode::Fixpoint { .. } | RirNode::GroupBy { .. } | RirNode::TensorMaskedJoin { .. } => {
            Err(resident_workspace_decline(
                "compact resident route contains an unsupported node",
            ))
        }
    }
}

fn resident_generated_query_heads(
    plan: &ExecutionPlan,
) -> std::result::Result<BTreeSet<String>, ResidentGraphDeclineReason> {
    let mut heads = BTreeSet::new();
    let mut rule_positions = HashSet::new();
    for (position, provenance) in plan.generated_query_rules.iter().enumerate() {
        if provenance.query_index != position {
            return Err(resident_workspace_decline(format!(
                "generated query provenance position {position} carries query index {}",
                provenance.query_index
            )));
        }
        if !rule_positions.insert((provenance.scc_index, provenance.rule_index)) {
            return Err(resident_workspace_decline(format!(
                "generated query provenance {} reuses compiled rule scc={} rule={}",
                provenance.query_index, provenance.scc_index, provenance.rule_index
            )));
        }
        let expected_head = format!("__xlog_query_{}", provenance.query_index);
        let rule = plan
            .rules_by_scc
            .get(provenance.scc_index)
            .and_then(|rules| rules.get(provenance.rule_index))
            .ok_or_else(|| {
                resident_workspace_decline(format!(
                    "generated query provenance {} references missing compiled rule scc={} rule={}",
                    provenance.query_index, provenance.scc_index, provenance.rule_index
                ))
            })?;
        if rule.head != expected_head {
            return Err(resident_workspace_decline(format!(
                "generated query provenance {} expects head {expected_head} but references authored head {}",
                provenance.query_index, rule.head
            )));
        }
        let occurrence_count = plan
            .rules_by_scc
            .iter()
            .flatten()
            .filter(|candidate| candidate.head == expected_head)
            .count();
        if occurrence_count != 1 {
            return Err(resident_workspace_decline(format!(
                "generated query head {expected_head} must have exactly one compiled rule, found {occurrence_count}"
            )));
        }
        heads.insert(expected_head);
    }
    Ok(heads)
}

impl Executor {
    /// Prepare one immutable, fixed-capacity conditional graph transaction.
    pub fn prepare_resident_graph<'executor>(
        &'executor self,
        plan: &ExecutionPlan,
        certificate: &ResidentGraphRouteCertificate,
        options: ResidentGraphPrepareOptions,
    ) -> std::result::Result<PreparedResidentGraph<'executor>, ResidentGraphExecutionError> {
        if !certificate.matches_plan(plan).map_err(runtime_error)? {
            return Err(resident_decline_error(
                ResidentGraphDeclineReason::WorkspaceUnbounded {
                    detail: "route certificate does not match the plan being prepared".into(),
                },
            ));
        }
        if !certificate.is_supported() {
            return Err(resident_decline_error(
                certificate.declines().first().cloned().unwrap_or_else(|| {
                    ResidentGraphDeclineReason::WorkspaceUnbounded {
                        detail: "route certificate is not resident-capable".into(),
                    }
                }),
            ));
        }

        self.prepare_resident_graph_after_certification(plan, certificate, options)
    }

    /// Prepare a transaction from a certificate sealed to its exact immutable plan.
    pub fn prepare_certified_resident_graph<'executor>(
        &'executor self,
        certified: &ResidentGraphCertifiedPlan,
        options: ResidentGraphPrepareOptions,
    ) -> std::result::Result<PreparedResidentGraph<'executor>, ResidentGraphExecutionError> {
        let certificate = certified.certificate();
        if !certificate.is_supported() {
            return Err(resident_decline_error(
                certificate.declines().first().cloned().unwrap_or_else(|| {
                    ResidentGraphDeclineReason::WorkspaceUnbounded {
                        detail: "route certificate is not resident-capable".into(),
                    }
                }),
            ));
        }
        self.prepare_resident_graph_after_certification(certified.plan(), certificate, options)
    }

    fn prepare_resident_graph_after_certification<'executor>(
        &'executor self,
        plan: &ExecutionPlan,
        certificate: &ResidentGraphRouteCertificate,
        options: ResidentGraphPrepareOptions,
    ) -> std::result::Result<PreparedResidentGraph<'executor>, ResidentGraphExecutionError> {
        let mut diagnostics =
            resident_prepare_diagnostics_for_sample(options.latency_diagnostic_sample);
        let total_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let admission_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let source_epoch = self.store.mutation_epoch();
        let resident_head_names = plan
            .rules_by_scc
            .iter()
            .flat_map(|rules| rules.iter().map(|rule| rule.head.clone()))
            .collect::<HashSet<_>>();
        let mut source_set_snapshots = Vec::new();
        let mut scans = Vec::new();
        for rules in &plan.rules_by_scc {
            for rule in rules {
                collect_recorded_scans(&rule.body, &mut scans);
            }
        }
        scans.sort_by_key(|rel| rel.0);
        scans.dedup();
        for rel in scans {
            let name = self
                .rel_names
                .get(&rel)
                .ok_or_else(|| runtime_error(format!("unknown resident relation id {rel:?}")))?;
            match self.store.get_with_version(name) {
                Some((buffer, version)) => {
                    let schema = certificate.schema_for(rel).ok_or_else(|| {
                        runtime_error(format!("resident scan {name} has no certified schema"))
                    })?;
                    if buffer.schema() != schema {
                        return Err(runtime_error(format!(
                            "resident scan {name} store schema differs from its certificate"
                        )));
                    }
                    source_set_snapshots.push(
                        resident_source_set_snapshot(&self.provider, name, version, buffer)
                            .map_err(resident_decline_error)?,
                    );
                }
                None if resident_head_names.contains(name) => {}
                None => {
                    return Err(runtime_error(format!(
                        "missing resident source relation {name}"
                    )))
                }
            }
        }
        let admission = self
            .resident_workspace_admission_after_certification(plan, certificate)
            .map_err(resident_decline_error)?;
        for (name, schema) in &admission.head_schemas {
            if let Some(existing) = self.store.get(name) {
                if !resident_schemas_type_compatible(existing.schema(), schema) {
                    return Err(runtime_error(format!(
                        "existing resident head {name} has a physically incompatible schema: existing={:?}, staged={schema:?}",
                        existing.schema()
                    )));
                }
            }
        }
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.admission_and_source_snapshot_ns = resident_prepare_elapsed_ns(
                admission_started.expect("diagnostic timer exists when enabled"),
            );
        }

        let execution_setup_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let runtime =
            Arc::clone(self.provider.memory().runtime().ok_or_else(|| {
                runtime_error("resident graph requires the async device runtime")
            })?);
        let stream_id = if let Some(stream_id) = self.resident_graph_stream.get() {
            *stream_id
        } else {
            let acquired = runtime.stream_pool().acquire().map_err(runtime_error)?;
            match self.resident_graph_stream.set(acquired) {
                Ok(()) => acquired,
                Err(_) => *self
                    .resident_graph_stream
                    .get()
                    .ok_or_else(|| runtime_error("resident stream initialization raced"))?,
            }
        };
        let stream = runtime
            .stream_pool()
            .resolve(stream_id)
            .ok_or_else(|| runtime_error("resident graph stream is no longer live"))?;
        let execution_domain = self
            .provider
            .bind_resident_execution_domain(Arc::clone(&runtime), stream_id, Arc::clone(&stream))
            .map_err(runtime_error)?;

        let mut relation_registration = self
            .rel_names
            .iter()
            .map(|(rel, name)| (*rel, name.clone()))
            .collect::<Vec<_>>();
        relation_registration.sort_by_key(|(rel, name)| (rel.0, name.clone()));

        let mut build = ResidentBuild {
            executor: self,
            certificate,
            capacity: admission.relation_capacity,
            relations: Vec::new(),
            filter_workspaces: Vec::new(),
            project_workspaces: Vec::new(),
            heads: BTreeMap::new(),
            head_winner_indices: BTreeMap::new(),
            source_names: HashSet::new(),
            source_aliases: HashMap::new(),
            next_op_id: 0,
            injection: options.test_device_status,
            injection_recorded: false,
        };
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.execution_domain_and_build_setup_ns = resident_prepare_elapsed_ns(
                execution_setup_started.expect("diagnostic timer exists when enabled"),
            );
        }
        let logical_planning_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        for (winner_index, (name, schema)) in admission.head_schemas.iter().enumerate() {
            let index = build
                .allocate_permanent_relation(schema.clone(), 0)
                .map_err(runtime_error)?;
            build.heads.insert(name.clone(), index);
            build.head_winner_indices.insert(
                name.clone(),
                u32::try_from(winner_index)
                    .map_err(|_| runtime_error("resident head count exceeds u32"))?,
            );
        }

        let mut initial_ops = Vec::new();
        for (name, schema) in &admission.head_schemas {
            if self.store.contains(name) {
                let source = build.source_reference(name).map_err(runtime_error)?;
                build
                    .plan_copy(source.clone(), build.heads[name], schema, &mut initial_ops)
                    .map_err(runtime_error)?;
                resident_record_schema_winner_mark(
                    &mut initial_ops,
                    ResidentBufferRef::Private(build.heads[name]),
                    build.head_winner_indices[name],
                    0,
                );
            }
        }

        let mut scc_by_id = HashMap::new();
        for (index, scc) in plan.sccs.iter().enumerate() {
            if scc_by_id.insert(scc.id, index).is_some() {
                return Err(runtime_error(format!(
                    "duplicate resident SCC id {}",
                    scc.id
                )));
            }
        }
        let ordered_sccs = if plan.strata.is_empty() {
            (0..plan.sccs.len()).collect::<Vec<_>>()
        } else {
            let mut ordered = Vec::new();
            for stratum in &plan.strata {
                for scc_id in &stratum.sccs {
                    ordered.push(*scc_by_id.get(scc_id).ok_or_else(|| {
                        runtime_error(format!(
                            "stratum {} references unknown SCC {scc_id}",
                            stratum.id
                        ))
                    })?);
                }
            }
            ordered
        };
        let mut seen_sccs = HashSet::new();
        for index in &ordered_sccs {
            if !seen_sccs.insert(*index) {
                return Err(runtime_error(format!(
                    "resident execution schedule repeats SCC {}",
                    plan.sccs[*index].id
                )));
            }
        }
        if seen_sccs.len() != plan.sccs.len() {
            return Err(runtime_error(
                "resident execution schedule omits one or more SCCs",
            ));
        }

        let mut phases = Vec::new();
        for scc_index in ordered_sccs {
            let scc = &plan.sccs[scc_index];
            let rules = plan.rules_by_scc.get(scc_index).ok_or_else(|| {
                runtime_error(format!("resident SCC {scc_index} has no rule vector"))
            })?;
            if rules.is_empty() {
                continue;
            }
            if !scc.is_recursive {
                let mut ops = Vec::new();
                let mut phase_heads = BTreeMap::<String, ResidentBufferRef>::new();
                for (rule_index, rule) in rules.iter().enumerate() {
                    let mut occurrences = HashMap::new();
                    let contribution = build
                        .plan_node(&rule.body, None, &mut occurrences, &mut ops)
                        .map_err(runtime_error)?;
                    let schema_id =
                        admission.rule_schema_ids[scc_index][rule_index].ok_or_else(|| {
                            runtime_error(format!(
                                "resident rule {} has no schema winner id",
                                rule.head
                            ))
                        })?;
                    ops.push(ResidentRecordedOp::SchemaWinnerMark {
                        contribution: contribution.clone(),
                        head_index: build.head_winner_indices[&rule.head],
                        schema_id,
                    });
                    let target = *build.heads.get(&rule.head).ok_or_else(|| {
                        runtime_error(format!("resident head {} was not staged", rule.head))
                    })?;
                    let target_schema = build
                        .schema(&ResidentBufferRef::Private(target))
                        .map_err(runtime_error)?
                        .clone();
                    let current = phase_heads
                        .remove(&rule.head)
                        .unwrap_or(ResidentBufferRef::Private(target));
                    let merged = build
                        .merge_phase_contribution(
                            Some(current),
                            contribution,
                            &target_schema,
                            &mut ops,
                        )
                        .map_err(runtime_error)?;
                    phase_heads.insert(rule.head.clone(), merged);
                }
                for (head, value) in phase_heads {
                    let target = build.heads[&head];
                    let target_schema = build
                        .schema(&ResidentBufferRef::Private(target))
                        .map_err(runtime_error)?
                        .clone();
                    build
                        .plan_copy(value, target, &target_schema, &mut ops)
                        .map_err(runtime_error)?;
                }
                phases.push(ResidentCapturePhase::Segment {
                    ops,
                    scc_begin: None,
                });
                continue;
            }

            let mut recursive_heads = BTreeMap::new();
            for rule in rules {
                recursive_heads
                    .entry(rule.head.clone())
                    .or_insert_with(|| rule.meta.schema.clone());
            }
            let mut delta_heads = BTreeMap::new();
            for (name, schema) in &recursive_heads {
                delta_heads.insert(
                    name.clone(),
                    build
                        .allocate_permanent_relation(schema.clone(), 0)
                        .map_err(runtime_error)?,
                );
            }

            let mut seed_ops = Vec::new();
            let mut seed_values = BTreeMap::<String, ResidentBufferRef>::new();
            for (rule_index, rule) in rules.iter().enumerate() {
                let mut occurrences = HashMap::new();
                let contribution = build
                    .plan_node(&rule.body, None, &mut occurrences, &mut seed_ops)
                    .map_err(runtime_error)?;
                let schema_id =
                    admission.rule_schema_ids[scc_index][rule_index].ok_or_else(|| {
                        runtime_error(format!(
                            "resident rule {} has no schema winner id",
                            rule.head
                        ))
                    })?;
                seed_ops.push(ResidentRecordedOp::SchemaWinnerMark {
                    contribution: contribution.clone(),
                    head_index: build.head_winner_indices[&rule.head],
                    schema_id,
                });
                let current = seed_values.remove(&rule.head);
                let merged = build
                    .merge_phase_contribution(
                        current,
                        contribution,
                        &rule.meta.schema,
                        &mut seed_ops,
                    )
                    .map_err(runtime_error)?;
                seed_values.insert(rule.head.clone(), merged);
            }
            for (name, schema) in &recursive_heads {
                let full = build.heads[name];
                let seed = seed_values.remove(name).ok_or_else(|| {
                    runtime_error(format!(
                        "recursive resident seed has no contribution for {name}"
                    ))
                })?;
                let candidate = build
                    .plan_union_schema(
                        ResidentBufferRef::Private(full),
                        seed,
                        schema.clone(),
                        &mut seed_ops,
                    )
                    .map_err(runtime_error)?;
                let novel = build
                    .plan_diff_schema(
                        candidate.clone(),
                        ResidentBufferRef::Private(full),
                        schema.clone(),
                        &mut seed_ops,
                    )
                    .map_err(runtime_error)?;
                build
                    .plan_copy(candidate, full, schema, &mut seed_ops)
                    .map_err(runtime_error)?;
                build
                    .plan_copy(novel, delta_heads[name], schema, &mut seed_ops)
                    .map_err(runtime_error)?;
            }
            let begin_op_id = build.next_op_id().map_err(runtime_error)?;
            phases.push(ResidentCapturePhase::Segment {
                ops: seed_ops,
                scc_begin: Some((self.config.max_iterations, begin_op_id)),
            });

            let mut body_ops = vec![ResidentRecordedOp::ChangedReset];
            let mut body_values = BTreeMap::<String, ResidentBufferRef>::new();
            let recursive_names = recursive_heads.keys().cloned().collect::<HashSet<_>>();
            for (rule_index, rule) in rules.iter().enumerate() {
                let mut rule_scans = Vec::new();
                collect_recorded_scans(&rule.body, &mut rule_scans);
                let mut per_relation_occurrence = HashMap::<RelId, usize>::new();
                for rel in rule_scans {
                    let occurrence = per_relation_occurrence.entry(rel).or_insert(0);
                    let current = *occurrence;
                    *occurrence += 1;
                    let Some(name) = self.rel_names.get(&rel) else {
                        return Err(runtime_error(format!(
                            "recursive resident scan has unknown relation id {rel:?}"
                        )));
                    };
                    if !recursive_names.contains(name) {
                        continue;
                    }
                    let mut planning_occurrences = HashMap::new();
                    let contribution = build
                        .plan_node(
                            &rule.body,
                            Some((rel, current, delta_heads[name])),
                            &mut planning_occurrences,
                            &mut body_ops,
                        )
                        .map_err(runtime_error)?;
                    let schema_id =
                        admission.rule_schema_ids[scc_index][rule_index].ok_or_else(|| {
                            runtime_error(format!(
                                "resident rule {} has no schema winner id",
                                rule.head
                            ))
                        })?;
                    body_ops.push(ResidentRecordedOp::SchemaWinnerMark {
                        contribution: contribution.clone(),
                        head_index: build.head_winner_indices[&rule.head],
                        schema_id,
                    });
                    let current = body_values.remove(&rule.head);
                    let merged = build
                        .merge_phase_contribution(
                            current,
                            contribution,
                            &rule.meta.schema,
                            &mut body_ops,
                        )
                        .map_err(runtime_error)?;
                    body_values.insert(rule.head.clone(), merged);
                }
            }
            for (name, schema) in &recursive_heads {
                let full = build.heads[name];
                let accumulated = match body_values.remove(name) {
                    Some(value) => value,
                    None => build
                        .plan_diff_schema(
                            ResidentBufferRef::Private(full),
                            ResidentBufferRef::Private(full),
                            schema.clone(),
                            &mut body_ops,
                        )
                        .map_err(runtime_error)?,
                };
                let novel = build
                    .plan_diff_schema(
                        accumulated,
                        ResidentBufferRef::Private(full),
                        schema.clone(),
                        &mut body_ops,
                    )
                    .map_err(runtime_error)?;
                let next_full = build
                    .plan_union_schema(
                        ResidentBufferRef::Private(full),
                        novel.clone(),
                        schema.clone(),
                        &mut body_ops,
                    )
                    .map_err(runtime_error)?;
                build
                    .plan_copy(next_full, full, schema, &mut body_ops)
                    .map_err(runtime_error)?;
                build
                    .plan_copy(novel, delta_heads[name], schema, &mut body_ops)
                    .map_err(runtime_error)?;
                // The transient Diff slot can be reused by later body operations. Drive
                // convergence from the completed permanent delta instead.
                body_ops.push(ResidentRecordedOp::ChangedMark {
                    relation: delta_heads[name],
                });
            }
            let convergence_op_id = build.next_op_id().map_err(runtime_error)?;
            phases.push(ResidentCapturePhase::ConditionalWhile {
                ops: body_ops,
                iteration_limit: self.config.max_iterations,
                convergence_op_id,
            });
        }

        let success_op_id = build.next_op_id().map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.logical_schedule_planning_ns = resident_prepare_elapsed_ns(
                logical_planning_started.expect("diagnostic timer exists when enabled"),
            );
        }
        let manifest_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let mut manifest = build
            .allocation_manifest(&initial_ops, &phases)
            .map_err(runtime_error)?;
        let compact_tables =
            resident_compact_tables(&build.filter_workspaces, &build.project_workspaces)
                .map_err(runtime_error)?;
        let compact_regions =
            resident_compact_regions(initial_ops, phases, success_op_id).map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.manifest_compact_construction_ns = resident_prepare_elapsed_ns(
                manifest_started.expect("diagnostic timer exists when enabled"),
            );
            diagnostics.compact_regions = compact_regions.len();
            diagnostics.conditional_regions = compact_regions
                .iter()
                .filter(|region| region.iteration_limit != 0)
                .count();
        }
        let schedule_lowering_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let compact_schedule = resident_lower_compact_regions(
            compact_regions,
            &manifest.slots,
            &manifest.logical_to_slot,
            build.source_names.iter().map(String::as_str),
            compact_tables,
        )
        .map_err(runtime_error)?;
        manifest
            .finalize_compact_schedule(&compact_schedule, build.heads.len())
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.schedule_lowering_ns = resident_prepare_elapsed_ns(
                schedule_lowering_started.expect("diagnostic timer exists when enabled"),
            );
            diagnostics.required_reservation_bytes = manifest.required_bytes;
            diagnostics.logical_relation_values = manifest.logical_relation_values;
            diagnostics.physical_relation_slots = manifest.slots.len();
            diagnostics.compact_ops = compact_schedule.ops.len();
            diagnostics.compact_waves = compact_schedule.waves.len();
        }
        let reservation_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let manifest_available_bytes = self.provider.memory().remaining_bytes();
        let mut reservation = self
            .provider
            .memory()
            .reserve_bytes(manifest.required_bytes)
            .map_err(|error| {
                resident_decline_error(ResidentGraphDeclineReason::WorkspaceUnbounded {
                    detail: format!(
                        "resident allocation manifest reservation of {} bytes failed: {error}",
                        manifest.required_bytes
                    ),
                })
            })?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.reservation_ns = resident_prepare_elapsed_ns(
                reservation_started.expect("diagnostic timer exists when enabled"),
            );
        }
        let physical = build
            .materialize(&manifest, &mut reservation, diagnostics.as_mut())
            .map_err(runtime_error)?;

        if build.injection.is_some() && !build.injection_recorded {
            return Err(runtime_error(
                "test device status requested after a nonexistent physical op",
            ));
        }
        let metadata_binding_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let output_indices =
            resident_output_indices(&build.heads, &manifest.logical_to_slot, &manifest.slots)
                .map_err(runtime_error)?;
        let output_relations = output_indices
            .iter()
            .map(|(_, index)| private_relation(&physical.relations, *index))
            .collect::<Result<Vec<_>>>()
            .map_err(runtime_error)?;
        let output_schema_plans = output_indices
            .iter()
            .map(|(name, _)| {
                let candidates = admission
                    .head_schema_choices
                    .get(name)
                    .cloned()
                    .ok_or_else(|| {
                        runtime_error(format!("resident head {name} has no schema plan"))
                    })?;
                let selection = if let Some(selection) = admission.head_schema_selections.get(name)
                {
                    let source_output = usize::try_from(
                        *build
                            .head_winner_indices
                            .get(&selection.source_head)
                            .ok_or_else(|| {
                                runtime_error(format!(
                                    "resident schema source {} is not staged",
                                    selection.source_head
                                ))
                            })?,
                    )
                    .map_err(|_| runtime_error("resident schema source index overflow"))?;
                    let source_candidates = admission
                        .head_schema_choices
                        .get(&selection.source_head)
                        .ok_or_else(|| runtime_error("resident schema source is missing"))?;
                    if source_candidates.len() != selection.output_schemas_by_source_winner.len() {
                        return Err(runtime_error(
                            "resident schema source mapping has the wrong length",
                        ));
                    }
                    ResidentOutputSchemaSelection::SourceWinner {
                        source_output,
                        schemas: selection.output_schemas_by_source_winner.clone(),
                    }
                } else {
                    ResidentOutputSchemaSelection::OwnWinner
                };
                Ok(ResidentOutputSchemaPlan {
                    candidates,
                    selection,
                })
            })
            .collect::<std::result::Result<Vec<_>, ResidentGraphExecutionError>>()?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let metadata_binding_ns = resident_prepare_elapsed_ns(
                metadata_binding_started.expect("diagnostic timer exists when enabled"),
            );
            diagnostics.metadata_binding_construction_ns = diagnostics
                .metadata_binding_construction_ns
                .saturating_add(metadata_binding_ns);
        }
        let device_trace_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let device_trace_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let device_trace = self
            .provider
            .prepare_resident_device_trace_in_reservation(&mut reservation)
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let device_trace_ns = resident_prepare_elapsed_ns(
                device_trace_started.expect("diagnostic timer exists when enabled"),
            );
            let device_trace_reserved_bytes = reservation.used_bytes().saturating_sub(
                device_trace_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.metadata_provider_calls =
                diagnostics.metadata_provider_calls.saturating_add(1);
            diagnostics.device_trace_preparation_ns = device_trace_ns;
            diagnostics.device_trace_reserved_bytes = device_trace_reserved_bytes;
        }
        let schema_defaults_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let default_schema_ids =
            resident_compact_schema_defaults(&compact_schedule.ops, output_relations.len())
                .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let schema_defaults_ns = resident_prepare_elapsed_ns(
                schema_defaults_started.expect("diagnostic timer exists when enabled"),
            );
            diagnostics.metadata_binding_construction_ns = diagnostics
                .metadata_binding_construction_ns
                .saturating_add(schema_defaults_ns);
        }
        let schema_winners_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let schema_winners_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let schema_winners = self
            .provider
            .prepare_resident_schema_winners_in_reservation(&default_schema_ids, &mut reservation)
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let schema_winners_ns = resident_prepare_elapsed_ns(
                schema_winners_started.expect("diagnostic timer exists when enabled"),
            );
            let schema_winners_reserved_bytes = reservation.used_bytes().saturating_sub(
                schema_winners_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.metadata_provider_calls =
                diagnostics.metadata_provider_calls.saturating_add(1);
            diagnostics.schema_winners_preparation_ns = schema_winners_ns;
            diagnostics.schema_winners_reserved_bytes = schema_winners_reserved_bytes;
            (
                diagnostics.schema_winners_initial_htod_calls,
                diagnostics.schema_winners_initial_htod_bytes,
            ) = resident_schema_winners_initial_htod(default_schema_ids.len());
        }
        let receipt_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let receipt_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let receipt = self
            .provider
            .prepare_resident_packed_receipt_with_trace_and_schema_winners_in_reservation(
                &output_relations,
                &device_trace,
                &schema_winners,
                &mut reservation,
            )
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let receipt_ns = resident_prepare_elapsed_ns(
                receipt_started.expect("diagnostic timer exists when enabled"),
            );
            let receipt_reserved_bytes = reservation.used_bytes().saturating_sub(
                receipt_bytes_before.expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.metadata_provider_calls =
                diagnostics.metadata_provider_calls.saturating_add(1);
            diagnostics.receipt_preparation_ns = receipt_ns;
            diagnostics.receipt_reserved_bytes = receipt_reserved_bytes;
            (
                diagnostics.receipt_initial_htod_calls,
                diagnostics.receipt_initial_htod_bytes,
            ) = resident_receipt_initial_htod(output_relations.len());
        }
        let schedule_bindings_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let mut schedule_bindings = Vec::with_capacity(
            manifest
                .slots
                .len()
                .checked_add(compact_schedule.source_slots.len())
                .ok_or_else(|| runtime_error("resident compact slot binding count overflow"))?,
        );
        for (slot_index, slot) in manifest.slots.iter().enumerate() {
            let relation =
                private_relation(&physical.relations, slot_index).map_err(runtime_error)?;
            schedule_bindings.push(if slot.permanent {
                ResidentScheduleSlotBinding::permanent(relation.buffer(), 0)
            } else {
                ResidentScheduleSlotBinding::scratch(relation.buffer(), 0)
            });
        }
        for (name, slot) in &compact_schedule.source_slots {
            let expected_slot = u32::try_from(schedule_bindings.len())
                .map_err(|_| runtime_error("resident compact slot binding count exceeds u32"))?;
            if *slot != expected_slot {
                return Err(runtime_error(
                    "resident compact source bindings are not contiguous",
                ));
            }
            let source = self.store.get(name).ok_or_else(|| {
                runtime_error(format!("resident source {name} disappeared during prepare"))
            })?;
            schedule_bindings
                .push(ResidentScheduleSlotBinding::source(source, 0).map_err(runtime_error)?);
        }
        let receipt_slots = output_indices
            .iter()
            .map(|(_, slot)| {
                u32::try_from(*slot)
                    .map_err(|_| runtime_error("resident receipt slot index exceeds u32"))
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let schedule_bindings_ns = resident_prepare_elapsed_ns(
                schedule_bindings_started.expect("diagnostic timer exists when enabled"),
            );
            diagnostics.metadata_binding_construction_ns = diagnostics
                .metadata_binding_construction_ns
                .saturating_add(schedule_bindings_ns);
        }
        let schedule_program_bytes_before = diagnostics.as_ref().map(|_| reservation.used_bytes());
        let schedule_program_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let schedule_program = self
            .provider
            .prepare_resident_schedule_program_in_reservation(
                &execution_domain,
                &schedule_bindings,
                &compact_schedule.ops,
                &compact_schedule.waves,
                &compact_schedule.regions,
                &compact_schedule.generation_bases,
                &compact_schedule.filter_comparisons,
                &compact_schedule.project_expressions,
                &receipt_slots,
                ResidentScheduleExternalBindings::new(
                    physical.filter_scratch.as_ref(),
                    &physical.set_workspace,
                    &physical.join_workspace,
                    &physical.control,
                    &device_trace,
                    &schema_winners,
                    &receipt,
                ),
                &mut reservation,
            )
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            let schedule_program_ns = resident_prepare_elapsed_ns(
                schedule_program_started.expect("diagnostic timer exists when enabled"),
            );
            let schedule_program_reserved_bytes = reservation.used_bytes().saturating_sub(
                schedule_program_bytes_before
                    .expect("diagnostic byte snapshot exists when enabled"),
            );
            diagnostics.metadata_provider_calls =
                diagnostics.metadata_provider_calls.saturating_add(1);
            diagnostics.schedule_program_preparation_ns = schedule_program_ns;
            diagnostics.schedule_program_reserved_bytes = schedule_program_reserved_bytes;
            (
                diagnostics.schedule_program_initial_htod_calls,
                diagnostics.schedule_program_initial_htod_bytes,
            ) = resident_schedule_initial_htod(diagnostics.schedule_program_reserved_bytes);
        }
        let reservation_validation_started =
            diagnostics.as_ref().map(|_| std::time::Instant::now());
        let tracked_device_allocation_bytes = reservation.used_bytes();
        resident_validate_exact_reservation(
            manifest.required_bytes,
            tracked_device_allocation_bytes,
            reservation.remaining_bytes(),
        )
        .map_err(runtime_error)?;
        drop(reservation);
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.reservation_validation_and_release_ns = resident_prepare_elapsed_ns(
                reservation_validation_started.expect("diagnostic timer exists when enabled"),
            );
        }
        let pinned_receipt_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let pinned_receipt = self
            .provider
            .prepare_resident_pinned_receipt(&receipt)
            .map_err(runtime_error)?;
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.pinned_receipt_ns = resident_prepare_elapsed_ns(
                pinned_receipt_started.expect("diagnostic timer exists when enabled"),
            );
        }

        let graph_capture_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let mut sequence =
            ConditionalCudaGraphSequenceBuilder::new(&stream).map_err(conditional_graph_error)?;
        let capture_topology =
            resident_compact_topology(&compact_schedule.regions).map_err(runtime_error)?;
        let mut conditional_body_node_kinds =
            Vec::with_capacity(capture_topology.conditional_body_kernel_counts.len());
        for ((region_index, region), parent_kind) in compact_schedule
            .regions
            .iter()
            .enumerate()
            .zip(capture_topology.parent_kinds.iter().copied())
        {
            let region_index = u32::try_from(region_index)
                .map_err(|_| runtime_error("resident compact region index exceeds u32"))?;
            if parent_kind == ResidentCaptureParentKind::Conditional {
                sequence
                    .add_conditional_while(u32::from(region.iteration_limit != 0), true, |body| {
                        body.capture_on_stream(&stream, || unsafe {
                            self.provider.record_resident_schedule_region_on_stream(
                                &schedule_program,
                                region_index,
                                Some(&body),
                                &stream,
                            )
                        })?;
                        conditional_body_node_kinds.push(body.linear_chain_node_kinds()?);
                        Ok(())
                    })
                    .map_err(conditional_graph_error)?;
            } else {
                sequence
                    .capture_segment_on_stream(&stream, || unsafe {
                        self.provider.record_resident_schedule_region_on_stream(
                            &schedule_program,
                            region_index,
                            None,
                            &stream,
                        )
                    })
                    .map_err(conditional_graph_error)?;
            }
        }
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.graph_body_capture_ns = resident_prepare_elapsed_ns(
                graph_capture_started.expect("diagnostic timer exists when enabled"),
            );
        }
        let graph_instantiate_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let graph = sequence
            .instantiate()
            .map_err(conditional_graph_error)?
            .bind_resident_lifecycle(&runtime);
        if let Some(diagnostics) = diagnostics.as_mut() {
            diagnostics.graph_instantiate_ns = resident_prepare_elapsed_ns(
                graph_instantiate_started.expect("diagnostic timer exists when enabled"),
            );
        }

        let validation_started = diagnostics.as_ref().map(|_| std::time::Instant::now());
        let mut recorder = execution_domain.new_strict_recorder();
        for binding in &schedule_bindings {
            binding.record_uses(&mut recorder);
        }
        ResidentScheduleExternalBindings::new(
            physical.filter_scratch.as_ref(),
            &physical.set_workspace,
            &physical.join_workspace,
            &physical.control,
            &device_trace,
            &schema_winners,
            &receipt,
        )
        .record_uses(&mut recorder);
        schedule_program.record_uses(&mut recorder);

        let graph_node_kinds = graph.linear_chain_node_kinds().map_err(runtime_error)?;
        resident_validate_parent_graph_kinds(&graph_node_kinds, &capture_topology.parent_kinds)
            .map_err(runtime_error)?;
        let conditional_body_kernel_counts = resident_validate_conditional_body_node_kinds(
            &conditional_body_node_kinds,
            capture_topology.conditional_body_kernel_counts.len(),
        )
        .map_err(runtime_error)?;
        if conditional_body_kernel_counts != capture_topology.conditional_body_kernel_counts {
            return Err(runtime_error(
                "resident conditional body topology differs from the compact schedule",
            ));
        }
        let body_node_count =
            conditional_body_node_kinds
                .iter()
                .try_fold(0usize, |total, kinds| {
                    total.checked_add(kinds.len()).ok_or_else(|| {
                        runtime_error("resident hierarchical graph node count overflow")
                    })
                })?;
        let hierarchical_graph_nodes = graph_node_kinds
            .len()
            .checked_add(body_node_count)
            .ok_or_else(|| runtime_error("resident hierarchical graph node count overflow"))?;
        if hierarchical_graph_nodes != capture_topology.hierarchical_node_count {
            return Err(runtime_error(
                "resident hierarchical graph topology differs from the compact schedule",
            ));
        }
        let (
            filter_descriptor_device_bytes,
            project_descriptor_device_bytes,
            fixed_workspace_device_bytes,
        ) = resident_compact_preflight_device_bytes(&manifest, &compact_schedule)
            .map_err(runtime_error)?;
        let parent_graph_node_count = graph_node_kinds.len();
        let preflight_report = ResidentGraphPreflightReport {
            relation_capacity: admission.relation_capacity,
            estimated_required_bytes: manifest.required_bytes,
            available_bytes_at_admission: manifest_available_bytes,
            tracked_device_allocation_bytes,
            relation_device_bytes: manifest.relation_bytes,
            filter_descriptor_device_bytes,
            filter_scratch_device_bytes: manifest.filter_scratch_bytes,
            project_descriptor_device_bytes,
            fixed_workspace_device_bytes,
            parent_graph_nodes: graph_node_kinds.len(),
            conditional_while_nodes: graph_node_kinds
                .iter()
                .filter(|kind| **kind == CudaGraphNodeKind::Conditional)
                .count(),
            parent_graph_node_kinds: graph_node_kinds,
            conditional_body_node_kinds,
            conditional_body_kernel_counts,
            hierarchical_graph_nodes,
            private_relation_slots: physical.relations.len(),
            logical_relation_values: manifest.logical_relation_values,
            permanent_relation_slots: manifest.permanent_slots,
            staged_output_relations: output_indices.len(),
            scratch_slots: manifest.scratch_slots,
            filter_scratch_allocations: manifest.filter_scratch_allocations,
            max_row_bytes: manifest.max_row_bytes,
        };
        let has_device_status_writer = build.injection_recorded;
        let owners = ResidentRunOwners {
            provider: Arc::clone(&self.provider),
            runtime,
            stream,
            graph,
            execution_domain,
            schedule_program,
            recorder,
            relations: physical.relations,
            output_indices,
            filter_scratch: physical.filter_scratch,
            set_workspace: physical.set_workspace,
            join_workspace: physical.join_workspace,
            control: physical.control,
            device_trace,
            schema_winners,
            receipt,
            pinned_receipt,
            source_epoch,
            relation_registration,
            transaction_identity: Arc::clone(&self.transaction_identity),
            output_schema_plans,
        };
        let validation_owner_assembly_ns = diagnostics.as_ref().map(|_| {
            resident_prepare_elapsed_ns(
                validation_started.expect("diagnostic timer exists when enabled"),
            )
        });
        let prepare_total_ns = diagnostics.as_ref().map(|_| {
            resident_prepare_elapsed_ns(
                total_started.expect("diagnostic total timer exists when enabled"),
            )
        });
        let prepare_diagnostic = diagnostics.map(|mut diagnostics| {
            diagnostics.total_ns =
                prepare_total_ns.expect("diagnostic total sample exists when enabled");
            diagnostics.validation_owner_assembly_ns = validation_owner_assembly_ns
                .expect("diagnostic validation sample exists when enabled");
            diagnostics.parent_graph_nodes = parent_graph_node_count;
            diagnostics.conditional_body_nodes = body_node_count;
            diagnostics
        });
        Ok(PreparedResidentGraph {
            owners,
            preflight_report,
            has_device_status_writer,
            source_guard: self,
            source_set_snapshots,
            prepare_diagnostic,
        })
    }

    fn resident_schema_variants(
        &self,
        node: &RirNode,
        certificate: &ResidentGraphRouteCertificate,
        head_schema_choices: &BTreeMap<String, Vec<Schema>>,
    ) -> std::result::Result<ResidentSchemaVariants, ResidentGraphDeclineReason> {
        match node {
            RirNode::Unit => certificate
                .node_schema(node)
                .map(ResidentSchemaVariants::Fixed)
                .ok_or_else(|| resident_workspace_decline("resident node schema missing")),
            RirNode::Scan { rel } => {
                let name = self.rel_names.get(rel).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "resident scan relation {rel:?} is not registered"
                    ))
                })?;
                if let Some(schemas) = head_schema_choices.get(name) {
                    return resident_schema_variants_from_source(name.clone(), schemas.clone());
                }
                let schema = self
                    .store
                    .get(name)
                    .map(|buffer| buffer.schema().clone())
                    .or_else(|| certificate.schema_for(*rel).cloned())
                    .ok_or_else(|| {
                        resident_workspace_decline(format!("resident scan {name} has no schema"))
                    })?;
                Ok(ResidentSchemaVariants::Fixed(schema))
            }
            RirNode::Filter { input, .. } | RirNode::Distinct { input, .. } => {
                self.resident_schema_variants(input, certificate, head_schema_choices)
            }
            RirNode::Project { input, columns } => {
                match self.resident_schema_variants(input, certificate, head_schema_choices)? {
                    ResidentSchemaVariants::Fixed(schema) => self
                        .project_schema(&schema, columns)
                        .map(ResidentSchemaVariants::Fixed)
                        .map_err(|error| resident_workspace_decline(error.to_string())),
                    ResidentSchemaVariants::Dynamic {
                        source_head,
                        schemas,
                    } => {
                        let schemas = schemas
                            .iter()
                            .map(|schema| self.project_schema(schema, columns))
                            .collect::<Result<Vec<_>>>()
                            .map_err(|error| resident_workspace_decline(error.to_string()))?;
                        resident_schema_variants_from_source(source_head, schemas)
                    }
                }
            }
            RirNode::Join {
                left,
                right,
                join_type,
                ..
            } => {
                let left = self.resident_schema_variants(left, certificate, head_schema_choices)?;
                match join_type {
                    JoinType::Semi => Ok(left),
                    JoinType::Inner => {
                        let right =
                            self.resident_schema_variants(right, certificate, head_schema_choices)?;
                        match (left, right) {
                            (
                                ResidentSchemaVariants::Fixed(left),
                                ResidentSchemaVariants::Fixed(right),
                            ) => {
                                let mut columns = left.columns;
                                columns.extend(right.columns);
                                Ok(ResidentSchemaVariants::Fixed(Schema::new(columns)))
                            }
                            (
                                ResidentSchemaVariants::Dynamic {
                                    source_head,
                                    schemas,
                                },
                                ResidentSchemaVariants::Fixed(right),
                            ) => {
                                let schemas = schemas
                                    .into_iter()
                                    .map(|left| {
                                        let mut columns = left.columns;
                                        columns.extend(right.columns.iter().cloned());
                                        Schema::new(columns)
                                    })
                                    .collect();
                                resident_schema_variants_from_source(source_head, schemas)
                            }
                            (
                                ResidentSchemaVariants::Fixed(left),
                                ResidentSchemaVariants::Dynamic {
                                    source_head,
                                    schemas,
                                },
                            ) => {
                                let schemas = schemas
                                    .into_iter()
                                    .map(|right| {
                                        let mut columns = left.columns.clone();
                                        columns.extend(right.columns);
                                        Schema::new(columns)
                                    })
                                    .collect();
                                resident_schema_variants_from_source(source_head, schemas)
                            }
                            (
                                ResidentSchemaVariants::Dynamic {
                                    source_head: left_source,
                                    schemas: left_schemas,
                                },
                                ResidentSchemaVariants::Dynamic {
                                    source_head: right_source,
                                    schemas: right_schemas,
                                },
                            ) => {
                                if left_source != right_source
                                    || left_schemas.len() != right_schemas.len()
                                {
                                    return Err(resident_workspace_decline(
                                        "resident join has multiple schema sources",
                                    ));
                                }
                                let schemas = left_schemas
                                    .into_iter()
                                    .zip(right_schemas)
                                    .map(|(left, right)| {
                                        let mut columns = left.columns;
                                        columns.extend(right.columns);
                                        Schema::new(columns)
                                    })
                                    .collect();
                                resident_schema_variants_from_source(left_source, schemas)
                            }
                        }
                    }
                    _ => Err(resident_workspace_decline(
                        "uncertified join reached resident schema planning",
                    )),
                }
            }
            RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
                self.resident_schema_variants(fallback, certificate, head_schema_choices)
            }
            RirNode::Diff { left, .. } => {
                self.resident_schema_variants(left, certificate, head_schema_choices)
            }
            RirNode::Union { inputs } => {
                if inputs.is_empty() {
                    return Err(resident_workspace_decline(
                        "resident schema union has no inputs",
                    ));
                }
                let mut fixed = None;
                for input in inputs {
                    match self.resident_schema_variants(input, certificate, head_schema_choices)? {
                        ResidentSchemaVariants::Fixed(schema) => match &fixed {
                            Some(existing) if existing != &schema => {
                                return Err(resident_workspace_decline(
                                    "resident schema union has ambiguous input schemas",
                                ));
                            }
                            Some(_) => {}
                            None => fixed = Some(schema),
                        },
                        ResidentSchemaVariants::Dynamic { .. } => {
                            return Err(resident_workspace_decline(
                                "resident schema union has ambiguous dynamic lineage",
                            ));
                        }
                    }
                }
                fixed.map(ResidentSchemaVariants::Fixed).ok_or_else(|| {
                    resident_workspace_decline("resident schema union has no inputs")
                })
            }
            RirNode::Fixpoint { .. }
            | RirNode::GroupBy { .. }
            | RirNode::TensorMaskedJoin { .. } => Err(resident_workspace_decline(
                "unsupported node reached resident schema planning",
            )),
        }
    }

    fn resident_workspace_admission_after_certification(
        &self,
        plan: &ExecutionPlan,
        certificate: &ResidentGraphRouteCertificate,
    ) -> std::result::Result<ResidentWorkspaceAdmission, ResidentGraphDeclineReason> {
        for rule in plan.rules_by_scc.iter().flatten() {
            validate_compact_resident_node_envelope(&rule.body, &|node| {
                certificate.node_schema(node).ok_or_else(|| {
                    resident_workspace_decline("compact resident node schema is missing")
                })
            })?;
        }

        let generated_query_heads = resident_generated_query_heads(plan)?;

        let mut generated_query_occurrences = generated_query_heads
            .iter()
            .map(|head| (head.as_str(), 0usize))
            .collect::<BTreeMap<_, _>>();
        for rule in plan.rules_by_scc.iter().flatten() {
            if let Some(count) = generated_query_occurrences.get_mut(rule.head.as_str()) {
                *count = count.checked_add(1).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "generated query head {} rule count overflowed",
                        rule.head
                    ))
                })?;
            }
        }
        for (head, count) in generated_query_occurrences {
            if count != 1 {
                return Err(resident_workspace_decline(format!(
                    "generated query head {head} must have exactly one compiled rule, found {count}"
                )));
            }
        }

        let (required_row_capacity, capacity_witness) =
            self.resident_required_row_capacity(plan, certificate)?;
        let required_row_capacity = required_row_capacity.max(1);
        let required_row_capacity = u32::try_from(required_row_capacity).map_err(|_| {
            ResidentGraphDeclineReason::WorkspaceUnbounded {
                detail: format!(
                    "certified resident row bound {required_row_capacity} exceeds the u32 count ABI; witness: {capacity_witness}"
                ),
            }
        })?;
        let relation_capacity = checked_capacity_class(required_row_capacity).map_err(|error| {
            ResidentGraphDeclineReason::WorkspaceUnbounded {
                detail: format!(
                    "{error}; certified_required_rows={required_row_capacity}; witness: {capacity_witness}"
                ),
            }
        })?;

        let mut head_schemas = BTreeMap::<String, Schema>::new();
        for rules in &plan.rules_by_scc {
            for rule in rules {
                match head_schemas.get(&rule.head) {
                    Some(existing) if existing != &rule.meta.schema => {
                        return Err(ResidentGraphDeclineReason::WorkspaceUnbounded {
                            detail: format!(
                                "relation {} has inconsistent compiled head schemas",
                                rule.head
                            ),
                        });
                    }
                    Some(_) => {}
                    None => {
                        head_schemas.insert(rule.head.clone(), rule.meta.schema.clone());
                    }
                }
            }
        }

        let mut head_schema_choices = head_schemas
            .keys()
            .map(|name| {
                let mut candidates = Vec::new();
                if let Some(existing) = self.store.get(name) {
                    let physical = &head_schemas[name];
                    if !resident_schemas_type_compatible(existing.schema(), physical) {
                        return Err(resident_workspace_decline(format!(
                            "existing resident head {name} is physically incompatible with its compiled schema"
                        )));
                    }
                    if !(generated_query_heads.contains(name)
                        && existing.cached_row_count() == Some(0))
                    {
                        candidates.push(existing.schema().clone());
                    }
                }
                Ok((name.clone(), candidates))
            })
            .collect::<std::result::Result<BTreeMap<_, _>, _>>()?;
        let mut rule_schema_ids = plan
            .rules_by_scc
            .iter()
            .map(|rules| vec![None; rules.len()])
            .collect::<Vec<_>>();
        let mut head_schema_selections = BTreeMap::new();

        for (scc_index, rules) in plan.rules_by_scc.iter().enumerate() {
            let recursive = plan.sccs.get(scc_index).is_some_and(|scc| scc.is_recursive);
            for (rule_index, rule) in rules.iter().enumerate() {
                if generated_query_heads.contains(&rule.head) {
                    continue;
                }
                let variants =
                    self.resident_schema_variants(&rule.body, certificate, &head_schema_choices)?;
                let physical = &head_schemas[&rule.head];
                let validate_schema = |schema: &Schema| {
                    if !resident_schemas_type_compatible(schema, physical) {
                        return Err(resident_workspace_decline(format!(
                            "resident rule contribution for {} is physically incompatible with its compiled head",
                            rule.head
                        )));
                    }
                    if recursive {
                        let rel = self.name_to_rel.get(&rule.head).ok_or_else(|| {
                            resident_workspace_decline(format!(
                                "recursive resident head {} is not registered",
                                rule.head
                            ))
                        })?;
                        let candidate = certificate.schema_for(*rel).ok_or_else(|| {
                            resident_workspace_decline(format!(
                                "recursive resident head {} has no catalog schema candidate",
                                rule.head
                            ))
                        })?;
                        if !resident_schemas_type_compatible(schema, candidate) {
                            return Err(resident_workspace_decline(format!(
                                "recursive resident head {} contribution schema is physically incompatible with its catalog candidate: contribution={schema:?}, catalog={candidate:?}",
                                rule.head
                            )));
                        }
                        if schema.key_columns != candidate.key_columns {
                            return Err(resident_workspace_decline(format!(
                                "recursive resident head {} contribution key columns do not equal its catalog candidate: contribution={:?}, catalog={:?}",
                                rule.head, schema.key_columns, candidate.key_columns
                            )));
                        }
                    }
                    Ok(())
                };
                match &variants {
                    ResidentSchemaVariants::Fixed(schema) => validate_schema(schema)?,
                    ResidentSchemaVariants::Dynamic { schemas, .. } => {
                        for schema in schemas {
                            validate_schema(schema)?;
                        }
                    }
                }
                match variants {
                    ResidentSchemaVariants::Fixed(schema) => {
                        let schema_id = resident_intern_schema(
                            head_schema_choices
                                .get_mut(&rule.head)
                                .expect("compiled head choice exists"),
                            schema,
                        )
                        .map_err(|error| resident_workspace_decline(error.to_string()))?;
                        rule_schema_ids[scc_index][rule_index] = Some(schema_id);
                    }
                    ResidentSchemaVariants::Dynamic {
                        source_head,
                        schemas,
                    } => {
                        resident_register_schema_selection(
                            &rule.head,
                            ResidentHeadSchemaSelection {
                                source_head,
                                output_schemas_by_source_winner: schemas.clone(),
                            },
                            &head_schema_choices,
                            &mut head_schema_selections,
                        )?;
                        for schema in schemas {
                            resident_intern_schema(
                                head_schema_choices
                                    .get_mut(&rule.head)
                                    .expect("compiled head choice exists"),
                                schema,
                            )
                            .map_err(|error| resident_workspace_decline(error.to_string()))?;
                        }
                        rule_schema_ids[scc_index][rule_index] = Some(RESIDENT_DYNAMIC_SCHEMA_ID);
                    }
                }
            }
        }

        let mut seen_query_heads = HashSet::new();
        for (scc_index, rules) in plan.rules_by_scc.iter().enumerate() {
            for (rule_index, rule) in rules.iter().enumerate() {
                if !generated_query_heads.contains(&rule.head) {
                    continue;
                }
                if !seen_query_heads.insert(rule.head.clone()) {
                    return Err(resident_workspace_decline(format!(
                        "synthetic query head {} has more than one compiled rule",
                        rule.head
                    )));
                }
                let variants =
                    self.resident_schema_variants(&rule.body, certificate, &head_schema_choices)?;
                let physical = &head_schemas[&rule.head];
                match variants {
                    ResidentSchemaVariants::Fixed(schema) => {
                        if !resident_schemas_type_compatible(&schema, physical) {
                            return Err(resident_workspace_decline(format!(
                                "synthetic query head {} is physically incompatible with its compiled head",
                                rule.head
                            )));
                        }
                        let schema_id = resident_intern_schema(
                            head_schema_choices
                                .get_mut(&rule.head)
                                .expect("compiled query head choice exists"),
                            schema,
                        )
                        .map_err(|error| resident_workspace_decline(error.to_string()))?;
                        rule_schema_ids[scc_index][rule_index] = Some(schema_id);
                    }
                    ResidentSchemaVariants::Dynamic {
                        source_head,
                        schemas,
                    } => {
                        if schemas
                            .iter()
                            .any(|schema| !resident_schemas_type_compatible(schema, physical))
                        {
                            return Err(resident_workspace_decline(format!(
                                "synthetic query head {} has a physically incompatible dynamic schema",
                                rule.head
                            )));
                        }
                        for schema in &schemas {
                            resident_intern_schema(
                                head_schema_choices
                                    .get_mut(&rule.head)
                                    .expect("compiled query head choice exists"),
                                schema.clone(),
                            )
                            .map_err(|error| resident_workspace_decline(error.to_string()))?;
                        }
                        rule_schema_ids[scc_index][rule_index] = Some(RESIDENT_DYNAMIC_SCHEMA_ID);
                        resident_register_schema_selection(
                            &rule.head,
                            ResidentHeadSchemaSelection {
                                source_head,
                                output_schemas_by_source_winner: schemas,
                            },
                            &head_schema_choices,
                            &mut head_schema_selections,
                        )?;
                    }
                }
            }
        }

        for (name, choices) in &head_schema_choices {
            if choices.is_empty() {
                return Err(resident_workspace_decline(format!(
                    "resident head {name} has no schema candidate"
                )));
            }
        }
        Ok(ResidentWorkspaceAdmission {
            relation_capacity,
            head_schemas,
            head_schema_choices,
            rule_schema_ids,
            head_schema_selections,
        })
    }

    fn resident_required_row_capacity(
        &self,
        plan: &ExecutionPlan,
        certificate: &ResidentGraphRouteCertificate,
    ) -> std::result::Result<(u64, String), ResidentGraphDeclineReason> {
        let mut relation_domains = HashMap::<RelId, ResidentColumnDomains>::new();
        let mut synthetic_domains = HashMap::<String, ResidentColumnDomains>::new();
        let mut source_rows = HashMap::<RelId, u64>::new();
        let plan_relations =
            resident_plan_relation_ids(plan.rules_by_scc.iter().flatten(), &self.name_to_rel);
        for (relation, name) in &self.rel_names {
            if !plan_relations.contains(relation) {
                continue;
            }
            let schema = certificate.schema_for(*relation).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "relation {name} has no schema in the resident certificate"
                ))
            })?;
            let mut columns = vec![BTreeMap::new(); schema.arity()];
            let rows = self.store.get(name).map(CudaBuffer::num_rows).unwrap_or(0);
            if rows != 0 {
                for (column, domain) in columns.iter_mut().enumerate() {
                    domain.insert(format!("source:{}:{column}", relation.0), rows);
                }
            }
            relation_domains.insert(*relation, columns);
            source_rows.insert(*relation, rows);
        }

        loop {
            let mut changed = false;
            for rules in &plan.rules_by_scc {
                for rule in rules {
                    let contribution = resident_node_domains(&rule.body, &relation_domains)?;
                    if contribution.len() != rule.meta.schema.arity() {
                        return Err(resident_workspace_decline(format!(
                            "resident rule head {} has {} certified domains for schema arity {}",
                            rule.head,
                            contribution.len(),
                            rule.meta.schema.arity()
                        )));
                    }
                    let target = if let Some(head) = self.name_to_rel.get(&rule.head) {
                        relation_domains.get_mut(head).ok_or_else(|| {
                            resident_workspace_decline(format!(
                                "resident rule head {} has no domain slot",
                                rule.head
                            ))
                        })?
                    } else {
                        synthetic_domains
                            .entry(rule.head.clone())
                            .or_insert_with(|| vec![BTreeMap::new(); rule.meta.schema.arity()])
                    };
                    changed |= merge_resident_domains(&rule.head, target, contribution)?;
                }
            }
            if !changed {
                break;
            }
        }

        let mut relation_set_bounds = source_rows.clone();
        let mut synthetic_set_bounds = HashMap::<String, u64>::new();
        let mut required = source_rows.values().copied().max().unwrap_or(1).max(1);
        let mut witness = source_rows
            .iter()
            .max_by_key(|(_, rows)| *rows)
            .map(|(relation, rows)| {
                let name = self
                    .rel_names
                    .get(relation)
                    .map(String::as_str)
                    .unwrap_or("<unregistered>");
                format!("source relation {name} ({relation:?}) rows={rows}")
            })
            .unwrap_or_else(|| "unit relation bound=1".to_string());

        let mut scc_by_id = HashMap::new();
        for (index, scc) in plan.sccs.iter().enumerate() {
            if scc_by_id.insert(scc.id, index).is_some() {
                return Err(resident_workspace_decline(format!(
                    "resident row proof found duplicate SCC id {}",
                    scc.id
                )));
            }
        }
        let ordered_sccs = if plan.strata.is_empty() {
            (0..plan.sccs.len()).collect::<Vec<_>>()
        } else {
            let mut ordered = Vec::new();
            for stratum in &plan.strata {
                for scc_id in &stratum.sccs {
                    ordered.push(*scc_by_id.get(scc_id).ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "resident row proof stratum {} references unknown SCC {scc_id}",
                            stratum.id
                        ))
                    })?);
                }
            }
            ordered
        };
        let mut seen_sccs = HashSet::new();
        for scc_index in ordered_sccs {
            if !seen_sccs.insert(scc_index) {
                return Err(resident_workspace_decline(format!(
                    "resident row proof schedule repeats SCC {}",
                    plan.sccs[scc_index].id
                )));
            }
            let scc = &plan.sccs[scc_index];
            let rules = plan.rules_by_scc.get(scc_index).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "resident row proof SCC {scc_index} has no rule vector"
                ))
            })?;

            if scc.is_recursive {
                for name in &scc.predicates {
                    let relation = self.name_to_rel.get(name).ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "recursive relation {name} has no registered relation id"
                        ))
                    })?;
                    let columns = relation_domains.get(relation).ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "recursive relation {relation:?} has no active-domain proof"
                        ))
                    })?;
                    let context = format!("recursive relation {name} ({relation:?})");
                    let finite_cap = resident_domain_product(columns, &context)?;
                    relation_set_bounds.insert(*relation, finite_cap);
                    if finite_cap > required {
                        required = finite_cap;
                        witness = format!(
                            "{context} finite_domain_cap={finite_cap} domains={}",
                            resident_domain_description(columns)
                        );
                    }
                }
            }

            let mut additions = BTreeMap::<String, u64>::new();
            let mut raw_additions = BTreeMap::<String, u64>::new();
            let mut addition_witnesses = BTreeMap::<String, String>::new();
            for (rule_index, rule) in rules.iter().enumerate() {
                let path = format!("scc[{scc_index}].rule[{rule_index}] head={}", rule.head);
                let proof = resident_node_row_bound(
                    &rule.body,
                    &source_rows,
                    &relation_set_bounds,
                    &relation_domains,
                    &path,
                )?;
                let rows = proof.rows;
                let transient = proof.peak;
                let rule_domains = resident_node_domains(&rule.body, &relation_domains)?;
                let set_rows = resident_domain_product_capped(
                    &rule_domains,
                    rows,
                    &format!("{path} projected set"),
                )?;
                if transient > required {
                    required = transient;
                    witness = format!(
                        "{path} transient_bound={transient} proof={} body={:?}",
                        proof.peak_detail, rule.body
                    );
                }
                let total = additions.entry(rule.head.clone()).or_default();
                *total = total.checked_add(set_rows).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "{path} rule-output addition overflow: partial={total} addition={set_rows} raw_rows={rows}"
                    ))
                })?;
                let raw_total = raw_additions.entry(rule.head.clone()).or_default();
                *raw_total = raw_total.checked_add(rows).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "{path} raw rule-output addition overflow: partial={raw_total} addition={rows}"
                    ))
                })?;
                addition_witnesses.insert(
                    rule.head.clone(),
                    format!(
                        "{path} body={:?} raw_rows={rows} per_clause_set_cap={set_rows} domains={}",
                        rule.body,
                        resident_domain_description(&rule_domains)
                    ),
                );
            }

            for (head, addition) in additions {
                let (current, relation) = if let Some(relation) = self.name_to_rel.get(&head) {
                    (
                        relation_set_bounds.get(relation).copied().unwrap_or(0),
                        Some(*relation),
                    )
                } else {
                    (synthetic_set_bounds.get(&head).copied().unwrap_or(0), None)
                };
                let path = addition_witnesses
                    .get(&head)
                    .map(String::as_str)
                    .unwrap_or("unknown rule");
                let raw_addition = raw_additions.get(&head).copied().unwrap_or(0);
                let raw_candidate = current.checked_add(raw_addition).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "{path} staged candidate addition overflow: prior={current} raw_rule_rows={raw_addition}"
                    ))
                })?;
                if raw_candidate > required {
                    required = raw_candidate;
                    witness = format!(
                        "{path} staged_candidate_bound={raw_candidate} prior={current} raw_rule_rows={raw_addition}"
                    );
                }
                if scc.is_recursive {
                    let candidate = current.checked_add(addition).ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "{path} recursive set candidate addition overflow: finite_cap={current} set_rule_rows={addition}"
                        ))
                    })?;
                    if candidate > required {
                        required = candidate;
                        witness = format!(
                            "{path} recursive_candidate_bound={candidate} finite_cap={current} raw_rule_rows={addition}"
                        );
                    }
                } else {
                    let summed_bound = current.checked_add(addition).ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "{path} nonrecursive set addition overflow: prior={current} rule_rows={addition}"
                        ))
                    })?;
                    let columns = if let Some(relation) = relation {
                        relation_domains.get(&relation)
                    } else {
                        synthetic_domains.get(&head)
                    }
                    .ok_or_else(|| {
                        resident_workspace_decline(format!(
                            "{path} nonrecursive head {head} has no active-domain proof"
                        ))
                    })?;
                    let set_bound = resident_domain_product_capped(
                        columns,
                        summed_bound,
                        &format!("{path} merged set {head}"),
                    )?;
                    if let Some(relation) = relation {
                        relation_set_bounds.insert(relation, set_bound);
                    } else {
                        synthetic_set_bounds.insert(head.clone(), set_bound);
                    }
                    if set_bound > required {
                        required = set_bound;
                        witness = format!(
                            "{path} nonrecursive_set_bound={set_bound} prior={current} rule_rows={addition}"
                        );
                    }
                }
            }
        }
        if seen_sccs.len() != plan.sccs.len() {
            return Err(resident_workspace_decline(
                "resident row proof schedule omits one or more SCCs",
            ));
        }
        Ok((required, witness))
    }
}

type ResidentDomain = BTreeMap<String, u64>;
type ResidentColumnDomains = Vec<ResidentDomain>;

fn resident_plan_relation_ids<'a>(
    rules: impl Iterator<Item = &'a CompiledRule>,
    name_to_rel: &HashMap<String, RelId>,
) -> HashSet<RelId> {
    let mut relations = HashSet::new();
    for rule in rules {
        relations.extend(rule.body.referenced_relations());
        if let Some(head) = name_to_rel.get(&rule.head) {
            relations.insert(*head);
        }
    }
    relations
}

fn resident_workspace_decline(detail: impl Into<String>) -> ResidentGraphDeclineReason {
    ResidentGraphDeclineReason::WorkspaceUnbounded {
        detail: detail.into(),
    }
}

fn merge_resident_domains(
    relation: &str,
    target: &mut ResidentColumnDomains,
    contribution: ResidentColumnDomains,
) -> std::result::Result<bool, ResidentGraphDeclineReason> {
    if target.len() != contribution.len() {
        return Err(resident_workspace_decline(format!(
            "resident rule head {relation} domain arity changed"
        )));
    }
    let mut changed = false;
    for (target_domain, contribution_domain) in target.iter_mut().zip(contribution) {
        for (source, bound) in contribution_domain {
            match target_domain.entry(source) {
                std::collections::btree_map::Entry::Vacant(entry) => {
                    entry.insert(bound);
                    changed = true;
                }
                std::collections::btree_map::Entry::Occupied(mut entry) if *entry.get() < bound => {
                    entry.insert(bound);
                    changed = true;
                }
                std::collections::btree_map::Entry::Occupied(_) => {}
            }
        }
    }
    Ok(changed)
}

fn resident_node_domains(
    node: &RirNode,
    relations: &HashMap<RelId, ResidentColumnDomains>,
) -> std::result::Result<ResidentColumnDomains, ResidentGraphDeclineReason> {
    match node {
        RirNode::Unit => Ok(Vec::new()),
        RirNode::Scan { rel } => relations.get(rel).cloned().ok_or_else(|| {
            resident_workspace_decline(format!(
                "resident scan {rel:?} has no source-domain certificate"
            ))
        }),
        RirNode::Filter { input, predicate } => {
            let mut domains = resident_node_domains(input, relations)?;
            resident_refine_filter_domains(&mut domains, predicate)?;
            Ok(domains)
        }
        RirNode::Distinct { input, .. } => resident_node_domains(input, relations),
        RirNode::Project { input, columns } => {
            let input_domains = resident_node_domains(input, relations)?;
            resident_project_domains(&input_domains, columns)
        }
        RirNode::Join {
            left,
            right,
            join_type,
            ..
        } => {
            let mut left_domains = resident_node_domains(left, relations)?;
            match join_type {
                JoinType::Inner => {
                    left_domains.extend(resident_node_domains(right, relations)?);
                    Ok(left_domains)
                }
                JoinType::Semi => Ok(left_domains),
                other => Err(resident_workspace_decline(format!(
                    "resident domain proof does not support {other:?} joins"
                ))),
            }
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            resident_node_domains(fallback, relations)
        }
        RirNode::Union { inputs } => {
            let mut merged: Option<ResidentColumnDomains> = None;
            for input in inputs {
                let contribution = resident_node_domains(input, relations)?;
                if let Some(domains) = merged.as_mut() {
                    if domains.len() != contribution.len() {
                        return Err(resident_workspace_decline(
                            "resident union domain arity mismatch",
                        ));
                    }
                    for (domain, addition) in domains.iter_mut().zip(contribution) {
                        for (source, bound) in addition {
                            domain
                                .entry(source)
                                .and_modify(|current| *current = (*current).max(bound))
                                .or_insert(bound);
                        }
                    }
                } else {
                    merged = Some(contribution);
                }
            }
            merged.ok_or_else(|| resident_workspace_decline("resident union has no inputs"))
        }
        RirNode::Diff { left, .. } => resident_node_domains(left, relations),
        RirNode::GroupBy { .. } | RirNode::Fixpoint { .. } | RirNode::TensorMaskedJoin { .. } => {
            Err(resident_workspace_decline(
                "resident domain proof encountered an uncertified operator",
            ))
        }
    }
}

fn resident_refine_filter_domains(
    domains: &mut ResidentColumnDomains,
    predicate: &Expr,
) -> std::result::Result<(), ResidentGraphDeclineReason> {
    match predicate {
        Expr::And(parts) => {
            for part in parts {
                resident_refine_filter_domains(domains, part)?;
            }
        }
        Expr::Compare {
            left,
            op: RirCompareOp::Eq,
            right,
        } => {
            let (column, value) = match (&**left, &**right) {
                (Expr::Column(column), Expr::Const(value))
                | (Expr::Const(value), Expr::Column(column)) => (*column, value),
                _ => return Ok(()),
            };
            let arity = domains.len();
            let domain = domains.get_mut(column).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "resident equality filter column {column} exceeds input arity {}",
                    arity
                ))
            })?;
            *domain = BTreeMap::from([(format!("constant:{value:?}"), 1)]);
        }
        _ => {}
    }
    Ok(())
}

fn resident_project_domains(
    input: &[ResidentDomain],
    columns: &[ProjectExpr],
) -> std::result::Result<ResidentColumnDomains, ResidentGraphDeclineReason> {
    columns
        .iter()
        .map(|column| match column {
            ProjectExpr::Column(index) => input.get(*index).cloned().ok_or_else(|| {
                resident_workspace_decline(format!(
                    "resident projection column {index} exceeds input arity {}",
                    input.len()
                ))
            }),
            ProjectExpr::Computed(Expr::Const(value), scalar) => Ok(BTreeMap::from([(
                format!("constant:{scalar:?}:{value:?}"),
                1,
            )])),
            _ => Err(resident_workspace_decline(
                "resident projection domain is not a column or constant",
            )),
        })
        .collect()
}

fn resident_domain_product(
    columns: &[ResidentDomain],
    context: &str,
) -> std::result::Result<u64, ResidentGraphDeclineReason> {
    if columns.is_empty() {
        return Ok(1);
    }
    columns
        .iter()
        .enumerate()
        .try_fold(1u64, |product, (column_index, domain)| {
            let values = domain.values().try_fold(0u64, |total, bound| {
                total.checked_add(*bound).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "{context} active-domain sum overflow at column {column_index}: partial={total} addition={bound} lineage={domain:?}"
                    ))
                })
            })?;
            product.checked_mul(values).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "{context} active-domain product overflow at column {column_index}: partial={product} factor={values} lineage={domain:?}"
                ))
            })
        })
}

fn resident_domain_product_capped(
    columns: &[ResidentDomain],
    cap: u64,
    context: &str,
) -> std::result::Result<u64, ResidentGraphDeclineReason> {
    if cap == 0 {
        return Ok(0);
    }
    if columns.is_empty() {
        return Ok(1);
    }
    let mut product = 1u64;
    for (column_index, domain) in columns.iter().enumerate() {
        let mut values = 0u64;
        for bound in domain.values() {
            let addition = (*bound).min(cap.saturating_sub(values));
            values = values.checked_add(addition).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "{context} capped active-domain sum overflow at column {column_index}: partial={values} addition={addition}"
                ))
            })?;
            if values == cap {
                break;
            }
        }
        if values == 0 {
            return Ok(0);
        }
        if product > cap / values {
            return Ok(cap);
        }
        product = product.checked_mul(values).ok_or_else(|| {
            resident_workspace_decline(format!(
                "{context} capped active-domain product overflow at column {column_index}: partial={product} factor={values}"
            ))
        })?;
    }
    Ok(product.min(cap))
}

fn resident_domain_description(columns: &[ResidentDomain]) -> String {
    columns
        .iter()
        .enumerate()
        .map(|(column, domain)| {
            let sum = domain.values().copied().fold(0u64, u64::saturating_add);
            format!("column[{column}] sum={sum} lineage={domain:?}")
        })
        .collect::<Vec<_>>()
        .join("; ")
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum ResidentCapacityBound {
    Finite { rows: u64, proof: String },
    AboveResidentLimit { proof: String },
}

fn resident_capacity_product_bound(proof: &'static str, factors: &[u64]) -> ResidentCapacityBound {
    if factors.contains(&0) {
        return ResidentCapacityBound::Finite {
            rows: 0,
            proof: proof.to_owned(),
        };
    }

    let limit = u64::from(MAX_RESIDENT_CAPACITY);
    let mut rows = 1u64;
    for factor in factors {
        if rows > limit / factor {
            return ResidentCapacityBound::AboveResidentLimit {
                proof: format!("{proof} factors={factors:?}"),
            };
        }
        rows *= factor;
    }
    ResidentCapacityBound::Finite {
        rows,
        proof: proof.to_owned(),
    }
}

fn resident_join_capacity_bound(
    left_rows: u64,
    right_rows: u64,
    left_fanout: u64,
    right_fanout: u64,
    matching_key_values: u64,
) -> ResidentCapacityBound {
    let candidates = [
        resident_capacity_product_bound("left_rows*right_rows", &[left_rows, right_rows]),
        resident_capacity_product_bound("left_rows*right_fanout", &[left_rows, right_fanout]),
        resident_capacity_product_bound("right_rows*left_fanout", &[right_rows, left_fanout]),
        resident_capacity_product_bound(
            "matching_keys*left_fanout*right_fanout",
            &[matching_key_values, left_fanout, right_fanout],
        ),
    ];

    let mut tightest: Option<(u64, String)> = None;
    let mut above = Vec::new();
    for candidate in candidates {
        match candidate {
            ResidentCapacityBound::Finite { rows, proof } => {
                if tightest.as_ref().is_none_or(|(current, _)| rows < *current) {
                    tightest = Some((rows, proof));
                }
            }
            ResidentCapacityBound::AboveResidentLimit { proof } => above.push(proof),
        }
    }
    if let Some((rows, proof)) = tightest {
        ResidentCapacityBound::Finite { rows, proof }
    } else {
        ResidentCapacityBound::AboveResidentLimit {
            proof: above.join("; "),
        }
    }
}

struct ResidentRowProof {
    rows: u64,
    peak: u64,
    domains: ResidentColumnDomains,
    full_row_unique: bool,
    peak_detail: String,
}

fn resident_node_row_bound(
    node: &RirNode,
    source_rows: &HashMap<RelId, u64>,
    relation_set_bounds: &HashMap<RelId, u64>,
    relation_domains: &HashMap<RelId, ResidentColumnDomains>,
    path: &str,
) -> std::result::Result<ResidentRowProof, ResidentGraphDeclineReason> {
    match node {
        RirNode::Unit => Ok(ResidentRowProof {
            rows: 1,
            peak: 1,
            domains: Vec::new(),
            full_row_unique: true,
            peak_detail: format!("{path}.unit rows=1"),
        }),
        RirNode::Scan { rel } => {
            let rows = source_rows
                .get(rel)
                .copied()
                .unwrap_or(0)
                .max(relation_set_bounds.get(rel).copied().unwrap_or(0));
            let domains = relation_domains.get(rel).cloned().ok_or_else(|| {
                resident_workspace_decline(format!(
                    "{path}.scan {rel:?} has no active-domain proof"
                ))
            })?;
            Ok(ResidentRowProof {
                rows,
                peak: rows,
                domains,
                full_row_unique: true,
                peak_detail: format!("{path}.scan rel={rel:?} rows={rows}"),
            })
        }
        RirNode::Filter { input, predicate } => {
            let mut proof = resident_node_row_bound(
                input,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.filter.input"),
            )?;
            resident_refine_filter_domains(&mut proof.domains, predicate)?;
            if proof.full_row_unique {
                proof.rows = resident_domain_product_capped(
                    &proof.domains,
                    proof.rows,
                    &format!("{path}.filter"),
                )?;
            }
            Ok(proof)
        }
        RirNode::Project { input, columns } => {
            let input = resident_node_row_bound(
                input,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.project.input"),
            )?;
            let domains = resident_project_domains(&input.domains, columns)?;
            let full_row_unique = input.full_row_unique
                && resident_projection_is_injective(&input.domains, columns, path)?;
            Ok(ResidentRowProof {
                rows: input.rows,
                peak: input.peak.max(input.rows),
                domains,
                full_row_unique,
                peak_detail: input.peak_detail,
            })
        }
        RirNode::Distinct { input, .. } => {
            let mut proof = resident_node_row_bound(
                input,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.distinct.input"),
            )?;
            proof.rows = resident_domain_product_capped(
                &proof.domains,
                proof.rows,
                &format!("{path}.distinct"),
            )?;
            proof.full_row_unique = true;
            Ok(proof)
        }
        RirNode::Join {
            left,
            right,
            left_keys,
            right_keys,
            join_type,
        } => {
            let left = resident_node_row_bound(
                left,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.join.left"),
            )?;
            let right = resident_node_row_bound(
                right,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.join.right"),
            )?;
            match join_type {
                JoinType::Semi => Ok(ResidentRowProof {
                    rows: left.rows,
                    peak: left.peak.max(right.peak).max(left.rows),
                    domains: left.domains,
                    full_row_unique: left.full_row_unique,
                    peak_detail: if left.peak >= right.peak {
                        left.peak_detail
                    } else {
                        right.peak_detail
                    },
                }),
                JoinType::Inner => {
                    if left_keys.len() != 1 || right_keys.len() != 1 {
                        return Err(resident_workspace_decline(format!(
                            "{path} resident row proof requires exactly one join key per side"
                        )));
                    }
                    let left_key = left_keys[0];
                    let right_key = right_keys[0];
                    let left_fanout = resident_key_fanout_bound(
                        &left.domains,
                        left_key,
                        left.rows,
                        left.full_row_unique,
                        &format!("{path}.join.left"),
                    )?;
                    let right_fanout = resident_key_fanout_bound(
                        &right.domains,
                        right_key,
                        right.rows,
                        right.full_row_unique,
                        &format!("{path}.join.right"),
                    )?;
                    let left_key_values = resident_domain_cardinality_capped(
                        left.domains.get(left_key).ok_or_else(|| {
                            resident_workspace_decline(format!(
                                "{path} left join key {left_key} exceeds arity {}",
                                left.domains.len()
                            ))
                        })?,
                        left.rows,
                        &format!("{path}.join.left_key"),
                    )?;
                    let right_key_values = resident_domain_cardinality_capped(
                        right.domains.get(right_key).ok_or_else(|| {
                            resident_workspace_decline(format!(
                                "{path} right join key {right_key} exceeds arity {}",
                                right.domains.len()
                            ))
                        })?,
                        right.rows,
                        &format!("{path}.join.right_key"),
                    )?;
                    let matching_key_values = left_key_values.min(right_key_values);
                    let (rows, bound_proof) = match resident_join_capacity_bound(
                        left.rows,
                        right.rows,
                        left_fanout,
                        right_fanout,
                        matching_key_values,
                    ) {
                        ResidentCapacityBound::Finite { rows, proof } => (rows, proof),
                        ResidentCapacityBound::AboveResidentLimit { proof } => {
                            return Err(resident_workspace_decline(format!(
                                "{path} inner join exceeds the fixed resident capacity limit: left_rows={} right_rows={} left_fanout={left_fanout} right_fanout={right_fanout} matching_keys={matching_key_values} products={proof}",
                                left.rows, right.rows
                            )));
                        }
                    };
                    let join_detail = format!(
                        "{path}.join left_rows={} right_rows={} left_fanout={left_fanout} right_fanout={right_fanout} matching_keys={matching_key_values} bound={rows} proof={bound_proof}",
                        left.rows, right.rows
                    );
                    let (peak, peak_detail) = if rows >= left.peak && rows >= right.peak {
                        (rows, join_detail)
                    } else if left.peak >= right.peak {
                        (left.peak, left.peak_detail)
                    } else {
                        (right.peak, right.peak_detail)
                    };
                    let mut domains = left.domains;
                    domains.extend(right.domains);
                    Ok(ResidentRowProof {
                        rows,
                        peak,
                        domains,
                        full_row_unique: left.full_row_unique && right.full_row_unique,
                        peak_detail,
                    })
                }
                other => Err(resident_workspace_decline(format!(
                    "resident row proof does not support {other:?} joins"
                ))),
            }
        }
        RirNode::ChainJoin { fallback, .. } | RirNode::MultiWayJoin { fallback, .. } => {
            resident_node_row_bound(
                fallback,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.fallback"),
            )
        }
        RirNode::Union { inputs } => {
            let mut rows = 0u64;
            let mut peak = 0u64;
            let mut peak_detail = format!("{path}.union empty");
            let mut domains: Option<ResidentColumnDomains> = None;
            for (index, input) in inputs.iter().enumerate() {
                let proof = resident_node_row_bound(
                    input,
                    source_rows,
                    relation_set_bounds,
                    relation_domains,
                    &format!("{path}.union[{index}]"),
                )?;
                rows = rows.checked_add(proof.rows).ok_or_else(|| {
                    resident_workspace_decline(format!(
                        "{path} union addition overflow: partial={rows} addition={}",
                        proof.rows
                    ))
                })?;
                if proof.peak > peak {
                    peak = proof.peak;
                    peak_detail = proof.peak_detail;
                }
                if let Some(merged) = domains.as_mut() {
                    merge_domain_columns(merged, proof.domains, path)?;
                } else {
                    domains = Some(proof.domains);
                }
            }
            if rows > peak {
                peak = rows;
                peak_detail = format!("{path}.union raw_rows={rows}");
            }
            Ok(ResidentRowProof {
                rows,
                peak,
                domains: domains
                    .ok_or_else(|| resident_workspace_decline("resident union has no inputs"))?,
                full_row_unique: false,
                peak_detail,
            })
        }
        RirNode::Diff { left, right } => {
            let left = resident_node_row_bound(
                left,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.diff.left"),
            )?;
            let right = resident_node_row_bound(
                right,
                source_rows,
                relation_set_bounds,
                relation_domains,
                &format!("{path}.diff.right"),
            )?;
            let (peak, peak_detail) = if left.peak >= right.peak {
                (left.peak.max(left.rows), left.peak_detail)
            } else {
                (right.peak.max(left.rows), right.peak_detail)
            };
            Ok(ResidentRowProof {
                rows: left.rows,
                peak,
                domains: left.domains,
                full_row_unique: left.full_row_unique,
                peak_detail,
            })
        }
        RirNode::GroupBy { .. } | RirNode::Fixpoint { .. } | RirNode::TensorMaskedJoin { .. } => {
            Err(resident_workspace_decline(
                "resident row proof encountered an uncertified operator",
            ))
        }
    }
}

fn resident_projection_is_injective(
    input_domains: &[ResidentDomain],
    columns: &[ProjectExpr],
    path: &str,
) -> std::result::Result<bool, ResidentGraphDeclineReason> {
    let mut retained = vec![false; input_domains.len()];
    for column in columns {
        if let ProjectExpr::Column(index) = column {
            let slot = retained.get_mut(*index).ok_or_else(|| {
                resident_workspace_decline(format!(
                    "{path} projection column {index} exceeds input arity {}",
                    input_domains.len()
                ))
            })?;
            *slot = true;
        }
    }
    for (index, (retained, domain)) in retained.iter().zip(input_domains).enumerate() {
        if !retained
            && resident_domain_cardinality_capped(
                domain,
                2,
                &format!("{path}.project.omitted[{index}]"),
            )? > 1
        {
            return Ok(false);
        }
    }
    Ok(true)
}

fn resident_domain_cardinality_capped(
    domain: &ResidentDomain,
    cap: u64,
    context: &str,
) -> std::result::Result<u64, ResidentGraphDeclineReason> {
    let mut total = 0u64;
    for bound in domain.values() {
        let addition = (*bound).min(cap.saturating_sub(total));
        total = total.checked_add(addition).ok_or_else(|| {
            resident_workspace_decline(format!(
                "{context} domain-cardinality addition overflow: partial={total} addition={addition}"
            ))
        })?;
        if total == cap {
            break;
        }
    }
    Ok(total)
}

fn resident_key_fanout_bound(
    domains: &[ResidentDomain],
    key: usize,
    rows: u64,
    full_row_unique: bool,
    context: &str,
) -> std::result::Result<u64, ResidentGraphDeclineReason> {
    if key >= domains.len() {
        return Err(resident_workspace_decline(format!(
            "{context} join key {key} exceeds input arity {}",
            domains.len()
        )));
    }
    if !full_row_unique {
        return Ok(rows);
    }
    let non_key = domains
        .iter()
        .enumerate()
        .filter_map(|(index, domain)| (index != key).then_some(domain.clone()))
        .collect::<Vec<_>>();
    resident_domain_product_capped(&non_key, rows, &format!("{context}.fanout"))
}

fn merge_domain_columns(
    target: &mut ResidentColumnDomains,
    addition: ResidentColumnDomains,
    path: &str,
) -> std::result::Result<(), ResidentGraphDeclineReason> {
    if target.len() != addition.len() {
        return Err(resident_workspace_decline(format!(
            "{path} union domain arity mismatch: left={} right={}",
            target.len(),
            addition.len()
        )));
    }
    for (target, addition) in target.iter_mut().zip(addition) {
        for (source, bound) in addition {
            target
                .entry(source)
                .and_modify(|current| *current = (*current).max(bound))
                .or_insert(bound);
        }
    }
    Ok(())
}

fn checked_capacity_class(source_capacity: u32) -> Result<u32> {
    let capacity = source_capacity
        .max(1)
        .checked_next_power_of_two()
        .ok_or_else(|| XlogError::Execution("resident capacity class overflow".to_string()))?;
    if capacity > MAX_RESIDENT_CAPACITY {
        return Err(XlogError::Execution(format!(
            "resident capacity class {capacity} exceeds the fixed scan envelope {MAX_RESIDENT_CAPACITY}"
        )));
    }
    Ok(capacity)
}

#[cfg(test)]
mod tests {
    use super::{
        checked_capacity_class, resident_compact_allocation_bytes,
        resident_compact_filter_descriptors, resident_compact_preflight_device_bytes,
        resident_compact_project_descriptors, resident_compact_regions,
        resident_compact_schedule_metadata_bytes, resident_compact_schema_defaults,
        resident_compact_tables, resident_compact_topology, resident_join_capacity_bound,
        resident_lower_compact_regions, resident_new_phase_unit, resident_output_indices,
        resident_phase_merge, resident_plan_relation_ids, resident_record_lifetimes,
        resident_record_scan_leaf, resident_record_schema_winner_mark, resident_record_unit_leaf,
        resident_register_schema_selection, resident_resolve_output_schema,
        resident_semantic_trace_guard, resident_source_binding_route,
        resident_source_logical_count, resident_source_slot_map, resident_union_fold_mode,
        resident_validate_conditional_body_node_kinds, resident_validate_exact_reservation,
        resident_validate_parent_graph_kinds, resident_validate_slot_assignments,
        validate_compact_resident_node_envelope, CudaGraphNodeKind, ResidentAllocationManifest,
        ResidentBufferRef, ResidentCapacityBound, ResidentCaptureParentKind, ResidentCapturePhase,
        ResidentCompactDescriptorTables, ResidentCompactLogicalRegion, ResidentFilterPlan,
        ResidentGraphDeclineReason, ResidentHeadSchemaSelection, ResidentLogicalRelation,
        ResidentOpDescriptor, ResidentOutputSchemaPlan, ResidentOutputSchemaSelection,
        ResidentPhaseMergeStep, ResidentPhysicalSlotPlan, ResidentProjectPlan, ResidentRecordedOp,
        ResidentSlotAssignment, ResidentUnionFoldMode, ScalarType, Schema,
        RESIDENT_DYNAMIC_SCHEMA_ID,
    };
    use std::collections::{BTreeMap, HashMap, HashSet};
    use xlog_core::RelId;
    use xlog_ir::{
        CompareOp, CompiledRule, ConstValue, Expr, JoinType, ProjectExpr, RirMeta, RirNode,
    };

    #[test]
    fn resident_plan_relations_exclude_registered_but_unreferenced_relations() {
        let source = RelId(1);
        let head = RelId(2);
        let unrelated = RelId(3);
        let rules = [CompiledRule {
            head: "reachable".to_owned(),
            body: RirNode::Scan { rel: source },
            meta: RirMeta::default(),
        }];
        let names = HashMap::from([
            ("source".to_owned(), source),
            ("reachable".to_owned(), head),
            ("unrelated".to_owned(), unrelated),
        ]);

        let relations = resident_plan_relation_ids(rules.iter(), &names);

        assert_eq!(relations, HashSet::from([source, head]));
    }

    #[test]
    fn capacity_class_is_source_bounded_and_checked() {
        assert_eq!(checked_capacity_class(0).unwrap(), 1);
        assert_eq!(checked_capacity_class(4_393).unwrap(), 8_192);
        assert_eq!(checked_capacity_class(65_536).unwrap(), 65_536);
        assert!(checked_capacity_class(65_537).is_err());
    }

    #[test]
    fn schema_lineage_fails_closed_for_cycles_and_missing_sources() {
        let schema = Schema::new(vec![("value".to_owned(), ScalarType::U32)]);
        let selection = |source: &str| ResidentHeadSchemaSelection {
            source_head: source.to_owned(),
            output_schemas_by_source_winner: vec![schema.clone()],
        };
        let choices = BTreeMap::from([
            ("a".to_owned(), vec![schema.clone()]),
            ("b".to_owned(), vec![schema.clone()]),
            ("c".to_owned(), vec![schema.clone()]),
            ("empty".to_owned(), Vec::new()),
        ]);

        let direct_cycle =
            resident_register_schema_selection("a", selection("a"), &choices, &mut BTreeMap::new())
                .expect_err("direct schema lineage cycle must decline");
        assert!(matches!(
            direct_cycle,
            ResidentGraphDeclineReason::WorkspaceUnbounded { detail }
                if detail.contains("contains a cycle")
        ));

        let mut long_cycle = BTreeMap::new();
        resident_register_schema_selection("a", selection("b"), &choices, &mut long_cycle)
            .expect("first acyclic lineage edge");
        resident_register_schema_selection("b", selection("c"), &choices, &mut long_cycle)
            .expect("second acyclic lineage edge");
        let long_cycle =
            resident_register_schema_selection("c", selection("a"), &choices, &mut long_cycle)
                .expect_err("transitive schema lineage cycle must decline");
        assert!(matches!(
            long_cycle,
            ResidentGraphDeclineReason::WorkspaceUnbounded { detail }
                if detail.contains("contains a cycle")
        ));

        for source in ["missing", "empty"] {
            let missing = resident_register_schema_selection(
                "target",
                selection(source),
                &choices,
                &mut BTreeMap::new(),
            )
            .expect_err("missing schema source must decline");
            assert!(matches!(
                missing,
                ResidentGraphDeclineReason::WorkspaceUnbounded { detail }
                    if detail.contains("has no admitted candidate")
            ));
        }

        let dynamic_plan = |source_output| ResidentOutputSchemaPlan {
            candidates: vec![schema.clone()],
            selection: ResidentOutputSchemaSelection::SourceWinner {
                source_output,
                schemas: vec![schema.clone()],
            },
        };
        let resolver_cycle = resident_resolve_output_schema(
            &[dynamic_plan(1), dynamic_plan(0)],
            &[RESIDENT_DYNAMIC_SCHEMA_ID, RESIDENT_DYNAMIC_SCHEMA_ID],
            0,
            &mut HashSet::new(),
        )
        .expect_err("receipt schema resolver cycle must fail");
        assert!(resolver_cycle.to_string().contains("contains a cycle"));

        let missing_plan = resident_resolve_output_schema(
            &[dynamic_plan(1)],
            &[RESIDENT_DYNAMIC_SCHEMA_ID],
            0,
            &mut HashSet::new(),
        )
        .expect_err("missing receipt schema source plan must fail");
        assert!(missing_plan.to_string().contains("output 1 is missing"));
    }

    #[test]
    fn join_capacity_uses_a_finite_tight_bound_when_loose_products_overflow() {
        let bound = resident_join_capacity_bound(u64::MAX, u64::MAX, 1, 1, 1);
        assert_eq!(
            bound,
            ResidentCapacityBound::Finite {
                rows: 1,
                proof: "matching_keys*left_fanout*right_fanout".to_owned(),
            }
        );
    }

    #[test]
    fn true_cartesian_join_remains_above_the_resident_limit() {
        let bound = resident_join_capacity_bound(257, 257, 257, 257, 1);
        assert!(matches!(
            bound,
            ResidentCapacityBound::AboveResidentLimit { .. }
        ));
    }

    #[test]
    fn lifetime_scan_rejects_an_out_of_range_logical_input() {
        let relations = vec![ResidentLogicalRelation {
            schema: Schema::new(vec![("value".to_string(), ScalarType::U32)]),
            initial_count: 0,
            permanent: false,
        }];
        let mut definitions = vec![None];
        let mut last_uses = vec![None];
        let mut ordinal = 0;
        let error = resident_record_lifetimes(
            &[ResidentRecordedOp::Filter {
                input: ResidentBufferRef::Private(1),
                output: 0,
                workspace: 0,
                op_id: 0,
            }],
            &relations,
            &mut definitions,
            &mut last_uses,
            &mut ordinal,
        )
        .expect_err("an invalid logical input must fail before allocation");
        assert!(error
            .to_string()
            .contains("logical input relation 1 is missing"));
    }

    #[test]
    fn stale_scratch_generation_is_rejected_before_materialization() {
        let schema = Schema::new(vec![("value".to_string(), ScalarType::U32)]);
        let relations = vec![
            ResidentLogicalRelation {
                schema: schema.clone(),
                initial_count: 0,
                permanent: false,
            },
            ResidentLogicalRelation {
                schema: schema.clone(),
                initial_count: 0,
                permanent: false,
            },
        ];
        let slots = vec![ResidentPhysicalSlotPlan {
            schema,
            initial_count: 0,
            permanent: false,
        }];
        let assignments = vec![
            ResidentSlotAssignment {
                slot: 0,
                generation: 0,
            },
            ResidentSlotAssignment {
                slot: 0,
                generation: 0,
            },
        ];
        let error = resident_validate_slot_assignments(
            &relations,
            &[Some(0), Some(2)],
            &[Some(0), Some(2)],
            &slots,
            &assignments,
        )
        .expect_err("a stale scratch generation must fail before allocation");
        assert!(error.to_string().contains("generation 0 but expected 1"));
    }

    #[test]
    fn authored_scan_and_unit_are_explicit_lifetime_operations() {
        let relations = vec![ResidentLogicalRelation {
            schema: Schema::new(Vec::<(String, ScalarType)>::new()),
            initial_count: 0,
            permanent: false,
        }];
        let mut definitions = vec![None];
        let mut last_uses = vec![None];
        let mut ordinal = 0;
        let operations = vec![
            ResidentRecordedOp::Unit {
                output: 0,
                op_id: 17,
            },
            ResidentRecordedOp::Scan {
                relation: ResidentBufferRef::Private(0),
                op_id: 18,
            },
            ResidentRecordedOp::TraceDelta {
                scan_delta: 1,
                filter_delta: 0,
                semantic_guard: None,
            },
        ];

        resident_record_lifetimes(
            &operations,
            &relations,
            &mut definitions,
            &mut last_uses,
            &mut ordinal,
        )
        .expect("an emitted Unit defines the relation consumed by the emitted Scan");

        assert_eq!(definitions, vec![Some(0)]);
        assert_eq!(last_uses, vec![Some(1)]);
        assert_eq!(ordinal, operations.len());
    }

    #[test]
    fn authored_leaf_emitters_preserve_scan_identity_and_trace_order() {
        let mut operations = Vec::new();
        let push = |ops: &mut Vec<_>, op, _op_id| ops.push(op);

        let unit = resident_record_unit_leaf(3, 17, &mut operations, push);
        let first = resident_record_scan_leaf(
            ResidentBufferRef::Private(4),
            18,
            None,
            &mut operations,
            push,
        );
        let second = resident_record_scan_leaf(
            ResidentBufferRef::Private(5),
            19,
            None,
            &mut operations,
            push,
        );

        assert!(matches!(unit, ResidentBufferRef::Private(3)));
        assert!(matches!(first, ResidentBufferRef::Private(4)));
        assert!(matches!(second, ResidentBufferRef::Private(5)));
        assert!(matches!(
            operations.as_slice(),
            [
                ResidentRecordedOp::Unit {
                    output: 3,
                    op_id: 17
                },
                ResidentRecordedOp::Scan {
                    relation: ResidentBufferRef::Private(4),
                    op_id: 18
                },
                ResidentRecordedOp::TraceDelta {
                    scan_delta: 1,
                    filter_delta: 0,
                    semantic_guard: None,
                },
                ResidentRecordedOp::Scan {
                    relation: ResidentBufferRef::Private(5),
                    op_id: 19
                },
                ResidentRecordedOp::TraceDelta {
                    scan_delta: 1,
                    filter_delta: 0,
                    semantic_guard: None,
                },
            ]
        ));
    }

    #[test]
    fn recursive_trace_semantics_use_the_selected_delta_and_leave_seed_unguarded() {
        let selected_delta = ResidentBufferRef::Private(9);
        assert!(resident_semantic_trace_guard(None).is_none());
        assert!(matches!(
            resident_semantic_trace_guard(Some((RelId(4), 2, 9))),
            Some(ResidentBufferRef::Private(9))
        ));

        let mut operations = Vec::new();
        let push = |ops: &mut Vec<_>, op, _op_id| ops.push(op);
        resident_record_scan_leaf(
            ResidentBufferRef::Private(4),
            18,
            None,
            &mut operations,
            push,
        );
        resident_record_scan_leaf(
            ResidentBufferRef::Private(5),
            19,
            Some(selected_delta.clone()),
            &mut operations,
            push,
        );

        assert!(matches!(
            operations.as_slice(),
            [
                ResidentRecordedOp::Scan { op_id: 18, .. },
                ResidentRecordedOp::TraceDelta {
                    semantic_guard: None,
                    ..
                },
                ResidentRecordedOp::Scan { op_id: 19, .. },
                ResidentRecordedOp::TraceDelta {
                    semantic_guard: Some(ResidentBufferRef::Private(9)),
                    ..
                },
            ]
        ));
    }

    #[test]
    fn phase_unit_allocator_creates_fresh_scratch_values() {
        let mut relations = Vec::new();

        let (first, first_op) = resident_new_phase_unit(&mut relations, 17).unwrap();
        let (second, second_op) = resident_new_phase_unit(&mut relations, 18).unwrap();

        assert!(matches!(first, ResidentBufferRef::Private(0)));
        assert!(matches!(second, ResidentBufferRef::Private(1)));
        assert!(matches!(
            first_op,
            ResidentRecordedOp::Unit {
                output: 0,
                op_id: 17
            }
        ));
        assert!(matches!(
            second_op,
            ResidentRecordedOp::Unit {
                output: 1,
                op_id: 18
            }
        ));
        assert!(relations
            .iter()
            .all(|relation| !relation.permanent && relation.initial_count == 0));
    }

    #[test]
    fn phase_local_merge_deduplicates_each_contribution_before_ordered_union() {
        let mut steps = Vec::new();
        let first = resident_phase_merge(None, 2_u32, |step| match step {
            ResidentPhaseMergeStep::Deduplicate(value) => {
                steps.push(("dedup", value, 0));
                Ok::<_, &'static str>(value * 10)
            }
            ResidentPhaseMergeStep::Union(left, right) => {
                steps.push(("union", left, right));
                Ok(left + right)
            }
        })
        .unwrap();
        let second = resident_phase_merge(Some(first), 3_u32, |step| match step {
            ResidentPhaseMergeStep::Deduplicate(value) => {
                steps.push(("dedup", value, 0));
                Ok::<_, &'static str>(value * 10)
            }
            ResidentPhaseMergeStep::Union(left, right) => {
                steps.push(("union", left, right));
                Ok(left + right)
            }
        })
        .unwrap();

        assert_eq!(first, 20);
        assert_eq!(second, 50);
        assert_eq!(steps, [("dedup", 2, 0), ("dedup", 3, 0), ("union", 20, 30)]);
    }

    #[test]
    fn initial_copy_schema_marker_is_ordered_after_its_count_producer() {
        let source = ResidentBufferRef::Source("head".to_owned());
        let mut operations = vec![ResidentRecordedOp::Project {
            input: source.clone(),
            output: 0,
            workspace: 0,
            op_id: 7,
        }];

        resident_record_schema_winner_mark(&mut operations, ResidentBufferRef::Private(0), 0, 3);

        assert!(matches!(
            operations.as_slice(),
            [
                ResidentRecordedOp::Project { op_id: 7, .. },
                ResidentRecordedOp::SchemaWinnerMark {
                    contribution: ResidentBufferRef::Private(0),
                    head_index: 0,
                    schema_id: 3,
                }
            ]
        ));
    }

    #[test]
    fn source_slots_are_deduplicated_sorted_and_distinct_from_private_targets() {
        let slots = resident_source_slot_map(3, ["zeta", "head", "zeta", "head"].into_iter())
            .expect("source slots");

        assert_eq!(slots.get("head"), Some(&3));
        assert_eq!(slots.get("zeta"), Some(&4));
        assert_eq!(slots.len(), 2);
        assert_ne!(slots["head"], 0, "stored source and staged target differ");
    }

    #[test]
    fn empty_untracked_sources_are_normalized_before_slot_binding() {
        let logical_count = resident_source_logical_count(Some(0)).unwrap();
        assert_eq!(
            resident_source_binding_route(logical_count, false, false).unwrap(),
            super::ResidentSourceBindingRoute::NormalizeEmpty
        );
        assert_eq!(
            resident_source_binding_route(0, true, true).unwrap(),
            super::ResidentSourceBindingRoute::Direct
        );
        assert!(resident_source_binding_route(1, false, true).is_err());
        assert!(resident_source_binding_route(1, true, false).is_err());
        assert!(resident_source_logical_count(None).is_err());
    }

    #[test]
    fn compact_regions_preserve_phase_order_and_form_five_parent_nodes() {
        let unit = |output, op_id| ResidentRecordedOp::Unit { output, op_id };
        let phases = vec![
            ResidentCapturePhase::Segment {
                ops: vec![unit(10, 10)],
                scc_begin: None,
            },
            ResidentCapturePhase::Segment {
                ops: vec![unit(11, 11)],
                scc_begin: Some((64, 101)),
            },
            ResidentCapturePhase::ConditionalWhile {
                ops: vec![unit(12, 12)],
                iteration_limit: 64,
                convergence_op_id: 102,
            },
            ResidentCapturePhase::Segment {
                ops: vec![unit(20, 20)],
                scc_begin: None,
            },
            ResidentCapturePhase::Segment {
                ops: vec![ResidentRecordedOp::Scan {
                    relation: ResidentBufferRef::Private(20),
                    op_id: 21,
                }],
                scc_begin: Some((32, 201)),
            },
            ResidentCapturePhase::ConditionalWhile {
                ops: vec![unit(22, 22)],
                iteration_limit: 32,
                convergence_op_id: 202,
            },
            ResidentCapturePhase::Segment {
                ops: vec![unit(30, 30)],
                scc_begin: None,
            },
        ];

        let regions = resident_compact_regions(vec![unit(0, 0)], phases, 999).expect("regions");

        assert_eq!(regions.len(), 5);
        assert_eq!(
            regions.iter().filter(|region| region.recursive()).count(),
            2
        );
        assert_eq!(regions.len() + 2, 7, "hierarchical node inventory");
        let topology = resident_compact_topology(&regions).unwrap();
        assert_eq!(
            topology.parent_kinds,
            vec![
                ResidentCaptureParentKind::Kernel,
                ResidentCaptureParentKind::Conditional,
                ResidentCaptureParentKind::Kernel,
                ResidentCaptureParentKind::Conditional,
                ResidentCaptureParentKind::Kernel,
            ]
        );
        assert_eq!(topology.conditional_body_kernel_counts, vec![1, 1]);
        assert_eq!(topology.hierarchical_node_count, 7);
        assert!(regions[0].initializes());
        assert!(regions[0].begins_scc());
        assert_eq!(regions[0].op_id, regions[1].op_id);
        assert_eq!(regions[0].iteration_limit, regions[1].iteration_limit);
        assert_eq!(regions[2].op_id, regions[3].op_id);
        assert_eq!(regions[2].iteration_limit, regions[3].iteration_limit);
        assert!(regions[4].finalizes());
        assert_eq!(regions[4].op_id, 999);
        assert!(matches!(
            regions[2].ops.as_slice(),
            [
                ResidentRecordedOp::Unit { output: 20, .. },
                ResidentRecordedOp::Scan {
                    relation: ResidentBufferRef::Private(20),
                    ..
                }
            ]
        ));
    }

    #[test]
    fn conditional_body_inventory_requires_one_actual_kernel_per_body() {
        let exact = vec![
            vec![CudaGraphNodeKind::Kernel],
            vec![CudaGraphNodeKind::Kernel],
        ];
        assert_eq!(
            resident_validate_conditional_body_node_kinds(&exact, 2).unwrap(),
            vec![1, 1]
        );
        assert!(resident_validate_conditional_body_node_kinds(&exact[..1], 2).is_err());
        assert!(resident_validate_conditional_body_node_kinds(&[Vec::new()], 1).is_err());
        assert!(resident_validate_conditional_body_node_kinds(
            &[vec![CudaGraphNodeKind::Kernel, CudaGraphNodeKind::Kernel]],
            1,
        )
        .is_err());
        assert!(resident_validate_conditional_body_node_kinds(
            &[vec![CudaGraphNodeKind::Memcpy]],
            1,
        )
        .is_err());
    }

    #[test]
    fn compact_parent_graph_kind_validation_is_exact() {
        let expected = vec![
            ResidentCaptureParentKind::Kernel,
            ResidentCaptureParentKind::Conditional,
            ResidentCaptureParentKind::Kernel,
        ];
        assert!(resident_validate_parent_graph_kinds(
            &[
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel
            ],
            &expected,
        )
        .is_ok());
        assert!(resident_validate_parent_graph_kinds(
            &[
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional
            ],
            &expected,
        )
        .is_err());
        assert!(resident_validate_parent_graph_kinds(
            &[CudaGraphNodeKind::Kernel, CudaGraphNodeKind::Conditional],
            &expected,
        )
        .is_err());
    }

    #[test]
    fn compact_descriptors_preserve_physical_generations_and_source_slots() {
        let regions = vec![ResidentCompactLogicalRegion {
            ops: vec![
                ResidentRecordedOp::Unit {
                    output: 0,
                    op_id: 7,
                },
                ResidentRecordedOp::SchemaWinnerMark {
                    contribution: ResidentBufferRef::Private(0),
                    head_index: 0,
                    schema_id: 3,
                },
                ResidentRecordedOp::TraceDelta {
                    scan_delta: 0,
                    filter_delta: 1,
                    semantic_guard: None,
                },
                ResidentRecordedOp::Scan {
                    relation: ResidentBufferRef::Source("source".to_owned()),
                    op_id: 8,
                },
                ResidentRecordedOp::TraceDelta {
                    scan_delta: 1,
                    filter_delta: 0,
                    semantic_guard: None,
                },
            ],
            iteration_limit: 1,
            op_id: 0,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
        }];
        let slots = vec![
            ResidentPhysicalSlotPlan {
                schema: Schema::new(Vec::new()),
                initial_count: 0,
                permanent: true,
            },
            ResidentPhysicalSlotPlan {
                schema: Schema::new(Vec::new()),
                initial_count: 0,
                permanent: false,
            },
        ];
        let assignments = vec![ResidentSlotAssignment {
            slot: 1,
            generation: 4,
        }];

        let plan = resident_lower_compact_regions(
            regions,
            &slots,
            &assignments,
            ["source"].into_iter(),
            Default::default(),
        )
        .expect("compact descriptors");

        assert_eq!(plan.source_slots["source"], 2);
        assert_eq!(plan.ops.len(), 4);
        assert_eq!(plan.ops[0].kind, super::ResidentScheduleOpKind::Unit);
        assert_eq!(plan.ops[0].out, 1);
        assert_eq!(plan.ops[0].out_generation, 4);
        assert_eq!(plan.ops[0].schema_winner_head, 0);
        assert_eq!(plan.ops[0].schema_winner_id, 3);
        assert_eq!(plan.ops[2].kind, super::ResidentScheduleOpKind::Scan);
        assert_eq!(plan.ops[2].out, 2);
        assert_eq!(plan.ops[2].in0_generation, 0);
        assert_eq!(plan.waves.len(), plan.ops.len());
        assert_eq!(plan.regions.len(), 1);
        assert_eq!(plan.generation_bases, vec![0, 4, 0]);
    }

    #[test]
    fn compact_novelty_marker_observes_the_completed_delta_copy() {
        let regions = vec![ResidentCompactLogicalRegion {
            ops: vec![
                ResidentRecordedOp::ChangedReset,
                ResidentRecordedOp::Diff {
                    left: ResidentBufferRef::Private(0),
                    right: ResidentBufferRef::Private(1),
                    output: 2,
                    op_id: 10,
                },
                ResidentRecordedOp::Project {
                    input: ResidentBufferRef::Private(2),
                    output: 3,
                    workspace: 0,
                    op_id: 11,
                },
                ResidentRecordedOp::ChangedMark { relation: 3 },
            ],
            iteration_limit: 3,
            op_id: 12,
            flags: super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
        }];
        let slots = (0..4)
            .map(|slot| ResidentPhysicalSlotPlan {
                schema: Schema::new(Vec::new()),
                initial_count: 0,
                permanent: slot != 2,
            })
            .collect::<Vec<_>>();
        let assignments = (0..4)
            .map(|slot| ResidentSlotAssignment {
                slot,
                generation: 0,
            })
            .collect::<Vec<_>>();
        let tables = ResidentCompactDescriptorTables {
            project_expressions: vec![super::ResidentProjectExpressionDescriptor::column(0, 4)],
            project_ranges: vec![(0, 1)],
            ..Default::default()
        };

        let plan = resident_lower_compact_regions(
            regions,
            &slots,
            &assignments,
            std::iter::empty(),
            tables,
        )
        .expect("the final delta copy must carry the novelty marker");

        assert_eq!(plan.ops.len(), 2);
        assert_eq!(plan.ops[0].kind, super::ResidentScheduleOpKind::Diff);
        assert_eq!(plan.ops[0].flags, 0);
        assert_eq!(plan.ops[1].kind, super::ResidentScheduleOpKind::Project);
        assert_eq!(plan.ops[1].flags, super::RESIDENT_SCHEDULE_OP_MARK_NOVELTY);
        assert_eq!(plan.ops[1].out, 3);
    }

    #[test]
    fn compact_schema_marker_must_name_the_producer_output() {
        let regions = vec![ResidentCompactLogicalRegion {
            ops: vec![
                ResidentRecordedOp::Project {
                    input: ResidentBufferRef::Source("source".to_owned()),
                    output: 0,
                    workspace: 0,
                    op_id: 7,
                },
                ResidentRecordedOp::SchemaWinnerMark {
                    contribution: ResidentBufferRef::Source("source".to_owned()),
                    head_index: 0,
                    schema_id: 0,
                },
            ],
            iteration_limit: 1,
            op_id: 9,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
        }];
        let slots = vec![ResidentPhysicalSlotPlan {
            schema: Schema::new(Vec::new()),
            initial_count: 0,
            permanent: true,
        }];
        let assignments = vec![ResidentSlotAssignment {
            slot: 0,
            generation: 0,
        }];
        let tables = ResidentCompactDescriptorTables {
            project_ranges: vec![(0, 0)],
            ..Default::default()
        };

        assert!(resident_lower_compact_regions(
            regions,
            &slots,
            &assignments,
            ["source"].into_iter(),
            tables,
        )
        .is_err());
    }

    #[test]
    fn compact_schema_defaults_follow_first_marker_per_head() {
        let ops = vec![
            ResidentOpDescriptor::unit(1, 0, 0).with_schema_winner(1, u32::MAX),
            ResidentOpDescriptor::unit(2, 1, 0).with_schema_winner(0, 7),
            ResidentOpDescriptor::unit(3, 2, 0).with_schema_winner(1, 9),
        ];
        assert_eq!(
            resident_compact_schema_defaults(&ops, 2).unwrap(),
            [7, u32::MAX]
        );
        assert!(resident_compact_schema_defaults(&ops[..1], 2).is_err());
    }

    #[test]
    fn compact_changed_reset_is_only_absorbed_at_recursive_region_entry() {
        let slot = ResidentPhysicalSlotPlan {
            schema: Schema::new(Vec::new()),
            initial_count: 0,
            permanent: false,
        };
        let assignment = ResidentSlotAssignment {
            slot: 0,
            generation: 0,
        };
        let region = |ops, flags| ResidentCompactLogicalRegion {
            ops,
            iteration_limit: 4,
            op_id: 9,
            flags,
        };

        assert!(resident_lower_compact_regions(
            vec![region(vec![ResidentRecordedOp::ChangedReset], 0)],
            std::slice::from_ref(&slot),
            std::slice::from_ref(&assignment),
            std::iter::empty(),
            Default::default(),
        )
        .is_err());
        assert!(resident_lower_compact_regions(
            vec![region(
                vec![
                    ResidentRecordedOp::Unit {
                        output: 0,
                        op_id: 1,
                    },
                    ResidentRecordedOp::ChangedReset,
                ],
                super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
            )],
            std::slice::from_ref(&slot),
            std::slice::from_ref(&assignment),
            std::iter::empty(),
            Default::default(),
        )
        .is_err());
        assert!(resident_lower_compact_regions(
            vec![region(
                vec![
                    ResidentRecordedOp::ChangedReset,
                    ResidentRecordedOp::Unit {
                        output: 0,
                        op_id: 1,
                    },
                ],
                super::RESIDENT_SCHEDULE_REGION_RECURSIVE,
            )],
            &[slot],
            &[assignment],
            std::iter::empty(),
            Default::default(),
        )
        .is_ok());
    }

    #[test]
    fn compact_filter_project_tables_preserve_types_widths_and_order() {
        let input = Schema::new(vec![
            ("symbol".to_owned(), ScalarType::Symbol),
            ("number".to_owned(), ScalarType::U64),
        ]);
        let predicate = Expr::And(vec![
            Expr::Compare {
                left: Box::new(Expr::Column(1)),
                op: CompareOp::Ge,
                right: Box::new(Expr::Const(ConstValue::U64(9))),
            },
            Expr::Compare {
                left: Box::new(Expr::Column(0)),
                op: CompareOp::Eq,
                right: Box::new(Expr::Const(ConstValue::Symbol("x".to_owned()))),
            },
        ]);
        let comparisons =
            resident_compact_filter_descriptors(&predicate, &input).expect("filter table");
        assert_eq!(comparisons.len(), 2);
        assert_eq!(comparisons[0].left_column, 1);
        assert_eq!(comparisons[0].width, 8);
        assert_eq!(comparisons[0].right_constant, 9);
        assert_eq!(comparisons[1].left_column, 0);
        assert_eq!(comparisons[1].width, 4);

        let output = Schema::new(vec![
            ("number".to_owned(), ScalarType::U64),
            ("symbol".to_owned(), ScalarType::Symbol),
        ]);
        let expressions = resident_compact_project_descriptors(
            &[
                ProjectExpr::Column(1),
                ProjectExpr::Computed(
                    Expr::Const(ConstValue::Symbol("y".to_owned())),
                    ScalarType::Symbol,
                ),
            ],
            &input,
            &output,
        )
        .expect("project table");
        assert_eq!(expressions.len(), output.arity());
        assert_eq!(expressions[0].column, 1);
        assert_eq!(expressions[0].width, 8);
        assert_eq!(expressions[1].kind, 1);
        assert_eq!(expressions[1].width, 4);

        let mismatch = Expr::Compare {
            left: Box::new(Expr::Column(1)),
            op: CompareOp::Eq,
            right: Box::new(Expr::Const(ConstValue::Symbol("wrong".to_owned()))),
        };
        assert!(resident_compact_filter_descriptors(&mismatch, &input).is_err());
    }

    #[test]
    fn compact_tables_cover_multiple_empty_occurrences_without_duplicate_payloads() {
        let filters = vec![
            ResidentFilterPlan {
                compact_comparisons: Vec::new(),
            },
            ResidentFilterPlan {
                compact_comparisons: Vec::new(),
            },
        ];
        let projects = vec![ResidentProjectPlan {
            compact_expressions: Vec::new(),
        }];

        let tables = resident_compact_tables(&filters, &projects).expect("compact tables");

        assert!(tables.filter_comparisons.is_empty());
        assert_eq!(tables.filter_ranges, [(0, 0), (0, 0)]);
        assert!(tables.project_expressions.is_empty());
        assert_eq!(tables.project_ranges, [(0, 0)]);
    }

    #[test]
    fn compact_descriptor_ranges_follow_operation_order_exactly() {
        let regions = vec![ResidentCompactLogicalRegion {
            ops: vec![
                ResidentRecordedOp::Project {
                    input: ResidentBufferRef::Source("source".to_owned()),
                    output: 0,
                    workspace: 1,
                    op_id: 1,
                },
                ResidentRecordedOp::Project {
                    input: ResidentBufferRef::Source("source".to_owned()),
                    output: 1,
                    workspace: 0,
                    op_id: 2,
                },
            ],
            iteration_limit: 1,
            op_id: 3,
            flags: super::RESIDENT_SCHEDULE_REGION_INITIALIZE
                | super::RESIDENT_SCHEDULE_REGION_FINALIZE,
        }];
        let slots = vec![
            ResidentPhysicalSlotPlan {
                schema: Schema::new(Vec::new()),
                initial_count: 0,
                permanent: false,
            },
            ResidentPhysicalSlotPlan {
                schema: Schema::new(Vec::new()),
                initial_count: 0,
                permanent: false,
            },
        ];
        let assignments = vec![
            ResidentSlotAssignment {
                slot: 0,
                generation: 0,
            },
            ResidentSlotAssignment {
                slot: 1,
                generation: 0,
            },
        ];
        let tables = ResidentCompactDescriptorTables {
            project_expressions: vec![Default::default(), Default::default()],
            project_ranges: vec![(0, 1), (1, 1)],
            ..Default::default()
        };

        assert!(resident_lower_compact_regions(
            regions,
            &slots,
            &assignments,
            ["source"].into_iter(),
            tables,
        )
        .is_err());
    }

    #[test]
    fn compact_metadata_reservation_counts_generation_bases_and_schema_defaults() {
        let actual = resident_compact_schedule_metadata_bytes(3, 5, 5, 2, 6, 4, 0, 0)
            .expect("metadata bytes");
        let expected = super::resident_schedule_metadata_device_bytes(3, 5, 5, 2, 10, 0, 0)
            .expect("expected metadata bytes");
        let missing_defaults = super::resident_schedule_metadata_device_bytes(3, 5, 5, 2, 6, 0, 0)
            .expect("generation-only bytes");

        assert_eq!(actual, expected);
        assert_ne!(actual, missing_defaults);
    }

    #[test]
    fn compact_manifest_replaces_per_occurrence_descriptor_allocations() {
        let plan = super::ResidentCompactSchedulePlan {
            source_slots: [("source".to_owned(), 2)].into_iter().collect(),
            ops: Vec::new(),
            waves: Vec::new(),
            regions: Vec::new(),
            generation_bases: Vec::new(),
            filter_comparisons: Vec::new(),
            project_expressions: Vec::new(),
        };
        let (required, metadata) =
            resident_compact_allocation_bytes(1_000, 2_000, 3_000, 2, 2, &plan).unwrap();
        let expected_metadata =
            resident_compact_schedule_metadata_bytes(3, 0, 0, 0, 0, 2, 0, 0).unwrap();
        assert_eq!(metadata, expected_metadata);
        assert_eq!(required, 1_000 + 2_000 + 3_000 + expected_metadata);
        assert_ne!(
            required,
            1_000 + 48 + 2_000 + 24 + 3_000 + expected_metadata
        );
    }

    #[test]
    fn compact_preflight_bytes_report_flattened_tables_without_double_counting() {
        let assert_components = |filter_count: usize, project_count: usize| {
            let plan = super::ResidentCompactSchedulePlan {
                source_slots: [("source".to_owned(), 2)].into_iter().collect(),
                ops: Vec::new(),
                waves: Vec::new(),
                regions: Vec::new(),
                generation_bases: Vec::new(),
                filter_comparisons: vec![
                    super::ResidentFilterComparisonDescriptor::default();
                    filter_count
                ],
                project_expressions: vec![
                    super::ResidentProjectExpressionDescriptor::default();
                    project_count
                ],
            };
            let mut manifest = ResidentAllocationManifest {
                slots: Vec::new(),
                logical_to_slot: Vec::new(),
                required_bytes: 0,
                relation_bytes: 1_000,
                filter_scratch_bytes: 2_000,
                schedule_metadata_bytes: 0,
                fixed_workspace_bytes: 3_000,
                logical_relation_values: 0,
                permanent_slots: 0,
                scratch_slots: 0,
                filter_scratch_allocations: 1,
                max_row_bytes: 0,
            };
            manifest.finalize_compact_schedule(&plan, 2).unwrap();

            let (filter_bytes, project_bytes, fixed_bytes) =
                resident_compact_preflight_device_bytes(&manifest, &plan).unwrap();
            assert_eq!(
                filter_bytes,
                48 * u64::try_from(filter_count.max(1)).unwrap()
            );
            assert_eq!(
                project_bytes,
                24 * u64::try_from(project_count.max(1)).unwrap()
            );
            assert_eq!(
                manifest.relation_bytes
                    + manifest.filter_scratch_bytes
                    + filter_bytes
                    + project_bytes
                    + fixed_bytes,
                manifest.required_bytes
            );
        };

        assert_components(0, 0);
        assert_components(3, 2);
    }

    #[test]
    fn compact_manifest_is_exact_and_preserves_permanent_head_mapping() {
        let mut manifest = ResidentAllocationManifest {
            slots: vec![
                ResidentPhysicalSlotPlan {
                    schema: Schema::new(Vec::new()),
                    initial_count: 0,
                    permanent: true,
                },
                ResidentPhysicalSlotPlan {
                    schema: Schema::new(Vec::new()),
                    initial_count: 0,
                    permanent: true,
                },
            ],
            logical_to_slot: vec![
                ResidentSlotAssignment {
                    slot: 0,
                    generation: 0,
                },
                ResidentSlotAssignment {
                    slot: 1,
                    generation: 0,
                },
            ],
            required_bytes: 0,
            relation_bytes: 1_000,
            filter_scratch_bytes: 2_000,
            schedule_metadata_bytes: 0,
            fixed_workspace_bytes: 3_000,
            logical_relation_values: 2,
            permanent_slots: 2,
            scratch_slots: 0,
            filter_scratch_allocations: 1,
            max_row_bytes: 1,
        };
        let plan = super::ResidentCompactSchedulePlan {
            source_slots: [("external".to_owned(), 2)].into_iter().collect(),
            ops: Vec::new(),
            waves: Vec::new(),
            regions: Vec::new(),
            generation_bases: Vec::new(),
            filter_comparisons: Vec::new(),
            project_expressions: Vec::new(),
        };
        manifest.finalize_compact_schedule(&plan, 1).unwrap();
        let without_source =
            resident_compact_schedule_metadata_bytes(2, 0, 0, 0, 0, 1, 0, 0).unwrap();
        assert_eq!(manifest.schedule_metadata_bytes - without_source, 240);
        assert_eq!(
            manifest.required_bytes,
            manifest.relation_bytes
                + manifest.filter_scratch_bytes
                + manifest.fixed_workspace_bytes
                + manifest.schedule_metadata_bytes
        );
        resident_validate_exact_reservation(manifest.required_bytes, manifest.required_bytes, 0)
            .unwrap();
        assert!(resident_validate_exact_reservation(manifest.required_bytes, 5_999, 1).is_err());
        let heads = [("head".to_owned(), 0_usize)].into_iter().collect();
        assert_eq!(
            resident_output_indices(&heads, &manifest.logical_to_slot, &manifest.slots).unwrap(),
            vec![("head".to_owned(), 0)]
        );
        let mut scratch_slots = manifest.slots.clone();
        scratch_slots[0].permanent = false;
        assert!(
            resident_output_indices(&heads, &manifest.logical_to_slot, &scratch_slots,).is_err()
        );
    }

    #[test]
    fn compact_distinct_declines_partial_keys_before_planning() {
        let node = RirNode::Distinct {
            input: Box::new(RirNode::Scan { rel: RelId(1) }),
            key_cols: vec![0],
        };
        let schema = Schema::new(vec![
            ("key".to_owned(), ScalarType::U32),
            ("value".to_owned(), ScalarType::U32),
        ]);

        let error = validate_compact_resident_node_envelope(&node, &|_| Ok(schema.clone()))
            .expect_err("a partial-key distinct must decline before allocation");

        let crate::resident_graph::ResidentGraphDeclineReason::WorkspaceUnbounded { detail } =
            error
        else {
            panic!("expected a workspace-envelope decline");
        };
        assert!(detail.contains("canonical full-row key columns"));
    }

    #[test]
    fn compact_distinct_accepts_empty_keys_for_nullary_rows() {
        let node = RirNode::Distinct {
            input: Box::new(RirNode::Unit),
            key_cols: vec![],
        };

        validate_compact_resident_node_envelope(&node, &|_| Ok(Schema::new(Vec::new())))
            .expect("the canonical full-row key for an arity-zero relation is empty");
    }

    #[test]
    fn compact_join_declines_same_width_different_key_types() {
        let node = RirNode::Join {
            left: Box::new(RirNode::Scan { rel: RelId(1) }),
            right: Box::new(RirNode::Scan { rel: RelId(2) }),
            left_keys: vec![0],
            right_keys: vec![0],
            join_type: JoinType::Inner,
        };

        let error = validate_compact_resident_node_envelope(&node, &|node| match node {
            RirNode::Scan { rel: RelId(1) } => {
                Ok(Schema::new(vec![("key".to_owned(), ScalarType::U32)]))
            }
            RirNode::Scan { rel: RelId(2) } => {
                Ok(Schema::new(vec![("key".to_owned(), ScalarType::Symbol)]))
            }
            _ => unreachable!("the envelope asks only for operand schemas"),
        })
        .expect_err("same-width but different key types must decline");

        let crate::resident_graph::ResidentGraphDeclineReason::WorkspaceUnbounded { detail } =
            error
        else {
            panic!("expected a workspace-envelope decline");
        };
        assert!(detail.contains("matching U32, U64, or Symbol key types"));
    }

    #[test]
    fn compact_union_shape_requires_unary_self_union() {
        assert!(resident_union_fold_mode(0).is_err());
        assert_eq!(
            resident_union_fold_mode(1).unwrap(),
            ResidentUnionFoldMode::SelfUnion
        );
        assert_eq!(
            resident_union_fold_mode(4).unwrap(),
            ResidentUnionFoldMode::LeftAssociated
        );
    }
}
