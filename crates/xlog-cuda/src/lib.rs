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
pub mod module_backing;
pub mod multi_gpu_memory;
pub mod provider;
mod semantic_hypergraph;
mod semantic_program;
mod semantic_training_view;
mod semantic_transition;
mod semantic_work;
pub use semantic_transition::{
    SemanticDetachedColdTaskContext, SemanticEvaluationForkOrigin,
    SemanticEvaluationForkPreparation, SemanticEvaluationForkTarget,
    SemanticEvaluationForkTargetRecords, SemanticSelectedTaskFacts,
    SemanticSelectedTaskFactsObservation,
};
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
    SemanticRootSupport, SemanticStatementHandle, SemanticStatementIdentity, SemanticStatementKey,
    SemanticStatementRef, SemanticSupportEvent, SemanticSupportHandle, SemanticSupportIdentity,
    SemanticSupportRecord, SemanticSupportRef, SemanticTruth, SemanticTypedRecord,
    SemanticVersionHandle, SemanticVersionIdentity, SemanticVersionRef, SemanticView,
};
pub use semantic_program::{
    SemanticProgramAdmission, SemanticProgramAtom, SemanticProgramFact, SemanticProgramRule,
    SemanticProgramTerm,
};
pub use semantic_training_view::{
    validate_training_view_layout, SemanticSelectedTrainingView, SemanticTrainingCanary,
    SemanticTrainingCanaryKind, SemanticTrainingCanaryRecord, SemanticTrainingCanaryRefusalReason,
    SemanticTrainingCanaryRefusalRecord, SemanticTrainingCanarySource, SemanticTrainingDomain,
    SemanticTrainingManifest, SemanticTrainingObjective, SemanticTrainingObjectiveGroup,
    SemanticTrainingObjectiveGroupKind, SemanticTrainingObjectiveGroupRecord,
    SemanticTrainingObjectiveRecord, SemanticTrainingRosterRow, SemanticTrainingViewBasis,
    SemanticTrainingViewLayout, SemanticTrainingViewOrigin, SemanticTrainingViewOriginRecord,
    SemanticTrainingViewPort, SemanticTrainingViewRow, SemanticTrainingViewSelection,
    SEMANTIC_TRAINING_CANARY_EVALUATOR_ABI,
};
pub use semantic_transition::{
    initial_source_layout, output_intents_from_publication_material,
    pending_output_intents_from_publication_material, semantic_structural_cost_descriptor,
    Identity256, SemanticActionCatalogue, SemanticActionDescriptor, SemanticActiveRow,
    SemanticActiveRows, SemanticCapacityRefusalInput, SemanticCapacityRefusalRecord,
    SemanticCatalogueBinding, SemanticCatalogueField, SemanticCatalogueSignature,
    SemanticCheckpointInitialPrefillObservation, SemanticCheckpointNativeProjection,
    SemanticComponent, SemanticComponentInput, SemanticContinuationInput, SemanticFeedback,
    SemanticFeedbackRecordMaterial, SemanticFeedbackSchema, SemanticFrozenModelContext,
    SemanticGradientDeliveryBinding, SemanticInitialPrefillLease, SemanticInitialSourceLayout,
    SemanticLearningCopyReset, SemanticLearningPhase, SemanticLearningPhaseAdmission,
    SemanticLearningPhaseRecord, SemanticLearningPhaseTransition,
    SemanticLoadedPublicationInputProjection, SemanticModelContext, SemanticModelContractLayout,
    SemanticModelMemory, SemanticModelStorage, SemanticModelView, SemanticModelWorkBufferExport,
    SemanticModelWorkBufferPort, SemanticNonterminalIntentBinding, SemanticObservedSource,
    SemanticOutputIntent, SemanticParentBinding, SemanticPolicyFieldLayout, SemanticPolicyLayout,
    SemanticPreparedResidentModelMemory, SemanticPreparedSegmentCapture,
    SemanticPreparedSegmentContinuation, SemanticPreparedSegmentNonSubmission,
    SemanticPreparedSegmentTransfers, SemanticPreparedStep, SemanticPreparedStepOutcome,
    SemanticPublicationInputProjection, SemanticPublicationReadbackLoan,
    SemanticPublicationTensorProjection, SemanticPublishedAcquisition, SemanticPublishedIdentity,
    SemanticPublishedLease, SemanticPublishedStateObservation,
    SemanticPublishedStateObservationBinding, SemanticReplayAppendBinding,
    SemanticReplayAppendRows, SemanticReplayDecisionBytes, SemanticReplayLoadedMaterial,
    SemanticReplayMaterial, SemanticReplayMaterialBudget, SemanticReplayMaterialOutput,
    SemanticReplayMaterialReservation, SemanticReplayMaterialScratch, SemanticReplayModelBackings,
    SemanticReplayPublicationJoin, SemanticReplayStateMaterialAdmission,
    SemanticReplayStateMaterialProof, SemanticReplayTrainingViewRow, SemanticResidentModelMemory,
    SemanticResolvedReplayMaterial, SemanticRngBinding, SemanticSegmentColdCapacity,
    SemanticSegmentTerminalWait, SemanticSourceMapping, SemanticStateRecord, SemanticStateRole,
    SemanticStructuralCostDescriptor, SemanticTaskAttainmentRequirement,
    SemanticTaskContentIdentity, SemanticTaskEvaluation, SemanticTaskEvaluationSpec,
    SemanticTaskFacts, SemanticTaskGoalWitness, SemanticTaskGround, SemanticTaskMeasurementBinding,
    SemanticTaskObjectiveLaw, SemanticTaskObservation, SemanticTaskObservationRoots,
    SemanticTaskObservationRootsOwner, SemanticTaskPriorityGoal, SemanticTaskPriorityLevel,
    SemanticTaskProgram, SemanticTaskReceiptFamily, SemanticTaskRefusal, SemanticTaskRelease,
    SemanticTaskScoring, SemanticTaskVerificationBinding, SemanticTensorInput,
    SemanticTensorLayout, SemanticTensorOwner, SemanticTextRow, SemanticTextSlot,
    SemanticTokenProvenance, SemanticTransitionError, SemanticTransitionHostIoStats,
    SemanticTransitionKind, SemanticTransitionLane, SemanticTransitionObservation,
    SemanticTransitionOutcome, SemanticTransitionReceipt, SemanticTransitionRefusal,
    SemanticTransitionSession, SemanticTransitionWork, SEMANTIC_FEEDBACK_SUPPORT_FIELDS,
    SEMANTIC_TRANSITION_COMPONENT_COUNT, SEMANTIC_TRANSITION_GENERATION,
};
#[cfg(feature = "semantic-policy")]
pub use semantic_transition::{
    SemanticActorRefreshInitializerRefusal, SemanticCompletedActionComponentMaterial,
    SemanticCompletedActionLaneMaterial, SemanticCompletedActionProjectionMaterial,
    SemanticCompletedEditSolutionMaterial, SemanticCompletedExecutionObservation,
    SemanticCompletedLaneOutcomeMaterial, SemanticCompletedMaterialCapacity,
    SemanticCompletedModelCarrierMaterial, SemanticCompletedReplayMaterials,
    SemanticCompletedStepWitnessMaterial, SemanticCompletedTaskGroundMaterial,
    SemanticCompletedTheoryDeltaMaterial, SemanticModelCanaryStream, SemanticModelCanaryWitness,
    SemanticModelContextCheckpointAttachment, SemanticModelContextContinuation,
    SemanticModelContextPrefillRestoreDeclaration, SemanticModelForwardStream,
    SemanticModelForwardWitness, SemanticPolicyGradients, SemanticPolicyNumericalDomainLayout,
    SemanticPreparedActorRefresh, SemanticPreparedResidentModelRestoration,
    SemanticPrivateModelContextCheckpointOrigin, SemanticPrivateModelContextCheckpointSave,
    SemanticResidentModelRead, SemanticRestoredModelContextPrefill,
    SemanticSegmentInstructionAdmission, SemanticTransitionSessionConstruction,
    SemanticUpdateCanaryMeasurement, SemanticUpdateForwardReceipt,
    SemanticUpdateForwardReceiptMaterial, SemanticUpdateMeasurements,
};
#[cfg(feature = "semantic-policy")]
pub use semantic_transition::{
    SemanticCancelledEvaluationCapture, SemanticCancelledModelEvaluation,
    SemanticCompletedModelEvaluation, SemanticEvaluationCohort, SemanticModelEvaluation,
    SemanticModelEvaluationResult,
};
#[cfg(feature = "semantic-policy")]
pub use semantic_transition::{
    SemanticColdModelWork, SemanticColdModelWorkAdmission, SemanticColdModelWorkDisposition,
    SemanticColdModelWorkOperation, SemanticColdModelWorkPurpose, SemanticColdModelWorkRegion,
    SemanticColdModelWorkReportCursor, SemanticColdNativeWork,
    SemanticColdNativeWorkActivationIdentity, SemanticColdNativeWorkActivationPermit,
    SemanticColdResidentRestorationPrelude,
};
pub use semantic_work::ModelWorkKind;

#[cfg(feature = "semantic-policy")]
pub use semantic_transition::{
    SemanticModelInputGenerationPort, SemanticPreparedSourceCache, SemanticPreparedSourceContent,
    SemanticPreparedSourceContentGeometry, SemanticPreparedSourceContentPorts,
    SemanticSourceAllocationDescriptor, SemanticSourceContentAllocation,
    SemanticSourceContentGeometryKind, SemanticSourceContentGeometrySeed,
    SemanticSourceContentRetirement, SemanticSourceGeometryAllocation,
    SemanticSourceGeometryTensor, SemanticSourceTensorDescriptor,
};

pub use wcoj_metadata::{
    HeatDist, LayoutSignature, RootMetadata, VertexId, WcojCycle4HgWorkPlanU32,
    WcojCycle4HgWorkPlanU64, WcojRelationMetadata, WcojTriangleHgWorkPlanU32,
    WcojTriangleHgWorkPlanU64,
};

pub use semantic_transition::SemanticColdModelWorkResult;
