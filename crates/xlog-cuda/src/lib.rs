//! GPU kernel provider for XLOG

pub mod arrow_device;
pub mod cuda_compat;
pub mod cuda_graph;
pub mod device;
pub mod device_pool;
pub mod device_runtime;
pub mod dlpack;
pub mod joint_constraint;
pub mod joint_solver;
pub mod kernel_manifest_data;
pub mod launch;
pub mod memory;
pub mod multi_gpu_memory;
pub mod provider;
mod semantic_hypergraph;
mod semantic_training_view;
mod semantic_transition;
mod semantic_work;
pub mod type_seam;
pub mod wcoj_metadata;
#[cfg(feature = "wcoj-phase-timing")]
pub mod wcoj_phase_timing;

pub(crate) mod embedded_kernel_data {
    include!(concat!(env!("OUT_DIR"), "/embedded_kernel_data.rs"));
}

pub use arrow_device::{ArrowDeviceArray, ArrowDeviceArrayOwned, ARROW_DEVICE_CUDA};
pub use cuda_compat::{
    sys, AsKernelParam, CudaFunction, CudaSlice, CudaStream, CudaView, CudaViewMut, DevicePtr,
    DevicePtrMut, DeviceRepr, DeviceSlice, DriverError, IntoKernelParamStorage, KernelParamStorage,
    KernelScalar, LaunchAsync, LaunchConfig, ValidAsZeroBits,
};
pub use device::CudaDevice;
pub use device_pool::GpuDevicePool;
pub use dlpack::{DLManagedTensor, DlpackManagedTensor, DlpackTable};
pub use joint_constraint::{CarrierBufferId, CarrierError, CarrierExport, JointConstraintCarrier};
pub use joint_solver::{
    Component, ConstraintGraph, Feasibility, FuelMeter, SolveStrategy, SolverError,
    SOLVER_ABI_IDENTITY,
};
pub use memory::{CudaBuffer, CudaColumn, GpuMemoryManager, RuntimeAllocBlock};
pub use multi_gpu_memory::MultiGpuMemoryManager;
pub use provider::{
    circuit_kernels, dedup_kernels, filter_kernels, groupby_kernels, ilp_kernels, join_kernels,
    pack_kernels, pir_kernels, scan_kernels, set_ops_kernels, sort_kernels, CompareOp,
    CudaKernelProvider, CudaProviderBuilder, FjDeltaCols, FjNode, FjPlan, FjSubAtom,
    IlpExactNaryPatterns, IlpExactNaryRequest, JoinIndexV2, JoinType, CIRCUIT_MODULE, DEDUP_MODULE,
    GROUPBY_MODULE, ILP_MODULE, JOIN_MODULE, PACK_MODULE, PIR_MODULE, SCAN_MODULE, SET_OPS_MODULE,
    SORT_MODULE,
};
pub use semantic_hypergraph::{
    SemanticAdmission, SemanticAdmissionLimits, SemanticAdmissionRecords, SemanticArgument,
    SemanticExtents, SemanticForkHandle, SemanticHandleKind, SemanticHypergraph,
    SemanticHypergraphCapacities, SemanticHypergraphError, SemanticHypergraphExecutionStats,
    SemanticInsertOutcome, SemanticInsertedSupport, SemanticPolarity, SemanticPredicateRecord,
    SemanticRecordRole, SemanticRootDigest, SemanticRootHandle, SemanticRootSnapshot,
    SemanticStatementHandle, SemanticStatementIdentity, SemanticStatementKey, SemanticStatementRef,
    SemanticSupportEvent, SemanticSupportHandle, SemanticSupportIdentity, SemanticSupportRecord,
    SemanticSupportRef, SemanticTruth, SemanticTypedRecord, SemanticVersionHandle,
    SemanticVersionIdentity, SemanticVersionRef, SemanticView,
};
pub use semantic_training_view::{
    SemanticSelectedTrainingView, SemanticTrainingCanary, SemanticTrainingCanaryKind,
    SemanticTrainingCanaryRecord, SemanticTrainingCanaryRefusalReason,
    SemanticTrainingCanaryRefusalRecord, SemanticTrainingCanaryResultRecord,
    SemanticTrainingObjective, SemanticTrainingObjectiveGroup, SemanticTrainingObjectiveGroupKind,
    SemanticTrainingObjectiveGroupRecord, SemanticTrainingObjectiveRecord,
    SemanticTrainingRosterRow, SemanticTrainingViewBasis, SemanticTrainingViewOrigin,
    SemanticTrainingViewOriginRecord, SemanticTrainingViewPort, SemanticTrainingViewRow,
    SemanticTrainingViewSelection,
};
pub use semantic_transition::{
    Identity256, SemanticActionCatalogue, SemanticActionDescriptor, SemanticActiveRow,
    SemanticActiveRows, SemanticCatalogueBinding, SemanticCatalogueField,
    SemanticCatalogueSignature, SemanticComponent, SemanticComponentInput,
    SemanticContinuationInput, SemanticFeedback, SemanticFeedbackRecordMaterial,
    SemanticFeedbackSchema, SemanticGradientDeliveryBinding, SemanticModelContext,
    SemanticModelContractLayout, SemanticModelMemory, SemanticModelStorage, SemanticModelView,
    SemanticObservedSource, SemanticParentBinding, SemanticPolicyFieldLayout, SemanticPolicyLayout,
    SemanticPreparedStep, SemanticPreparedStepOutcome, SemanticPublishedIdentity,
    SemanticPublishedLease, SemanticReplayMaterial, SemanticResidentModelMemory,
    SemanticRngBinding, SemanticSourceMapping, SemanticStateRecord, SemanticStateRole,
    SemanticTaskEvaluation, SemanticTaskEvaluationSpec, SemanticTaskFacts, SemanticTaskObservation,
    SemanticTaskObservationRoots, SemanticTaskProgram, SemanticTaskRefusal, SemanticTaskScoring,
    SemanticTensorContentWitness, SemanticTensorInput, SemanticTensorLayout, SemanticTextRow,
    SemanticTextSlot, SemanticTransitionError, SemanticTransitionHostIoStats,
    SemanticTransitionKind, SemanticTransitionLane, SemanticTransitionObservation,
    SemanticTransitionOutcome, SemanticTransitionReceipt, SemanticTransitionRefusal,
    SemanticTransitionSession, SemanticTransitionWork, SEMANTIC_FEEDBACK_SUPPORT_FIELDS,
    SEMANTIC_TRANSITION_COMPONENT_COUNT, SEMANTIC_TRANSITION_GENERATION,
};
#[cfg(feature = "semantic-policy")]
pub use semantic_transition::{SemanticPolicyGradients, SemanticSelectedPolicyGradients};
pub use semantic_work::ModelWorkKind;

pub use wcoj_metadata::{
    HeatDist, LayoutSignature, RootMetadata, VertexId, WcojCycle4HgWorkPlanU32,
    WcojCycle4HgWorkPlanU64, WcojRelationMetadata, WcojTriangleHgWorkPlanU32,
    WcojTriangleHgWorkPlanU64,
};
