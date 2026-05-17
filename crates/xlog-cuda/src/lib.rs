//! GPU kernel provider for XLOG

pub mod arrow_device;
pub mod cuda_compat;
pub mod device;
pub mod device_pool;
pub mod device_runtime;
pub mod dlpack;
pub mod kernel_manifest_data;
pub mod launch;
pub mod memory;
pub mod multi_gpu_memory;
pub mod provider;
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
pub use memory::{CudaBuffer, CudaColumn, GpuMemoryManager, RuntimeAllocBlock};
pub use multi_gpu_memory::MultiGpuMemoryManager;
pub use provider::{
    circuit_kernels, dedup_kernels, filter_kernels, groupby_kernels, ilp_kernels, join_kernels,
    pack_kernels, pir_kernels, scan_kernels, set_ops_kernels, sort_kernels, CompareOp,
    CudaKernelProvider, JoinIndexV2, JoinType, CIRCUIT_MODULE, DEDUP_MODULE, FILTER_MODULE,
    GROUPBY_MODULE, ILP_MODULE, JOIN_MODULE, PACK_MODULE, PIR_MODULE, SCAN_MODULE, SET_OPS_MODULE,
    SORT_MODULE,
};
pub use wcoj_metadata::{
    HeatDist, LayoutSignature, RootMetadata, VertexId, WcojCycle4HgWorkPlanU32,
    WcojCycle4HgWorkPlanU64, WcojRelationMetadata, WcojTriangleHgWorkPlanU32,
    WcojTriangleHgWorkPlanU64,
};
