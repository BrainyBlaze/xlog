//! CUDA kernel provider implementation
//!
//! This module provides the `CudaKernelProvider` which manages pre-compiled
//! PTX kernels for GPU execution of relational operations (join, dedup, groupby).

use std::collections::HashMap;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use std::ffi::c_void;
use xlog_core::{Result, Schema, XlogError};

use crate::{
    cuda_compat::{
        AsKernelParam, DeviceParamStorage, DevicePtr, DeviceRepr, DeviceSlice,
        IntoKernelParamStorage, LaunchAsync, LaunchConfig,
    },
    cuda_graph::{CapturedCudaGraph, CsmCudaGraphKey, CudaGraphNode},
    memory::{validate_logical_row_count, CudaColumn, TrackedCudaSlice},
    CudaBuffer, CudaDevice, CudaStream, CudaViewMut, GpuMemoryManager,
};

mod arithmetic;
mod filter;
mod fj;
mod fj_delta;
mod fj_delta_sparse;
mod groupby;
mod ilp;
mod ilp_exact;
mod io;
mod kernel_loading;
pub mod kernel_paths;
mod launch_safe;
mod probabilistic;
mod relational;
mod transfer;
mod wcoj;
mod wcoj_metadata;
mod wcoj_project;

pub use fj::{FjNode, FjPlan, FjSubAtom};
pub use fj_delta::{FjDeltaCols, FJ_DELTA_MAX_DOMAIN};

/// Per-module PTX load timing (populated only when XLOG_WARMUP_PROFILE=1).
#[derive(Debug, Clone, Default)]
pub struct PtxLoadProfile {
    pub total_sec: f64,
    pub per_module_sec: Vec<(String, f64)>,
    pub cubin_loaded: u32,
    pub ptx_fallback: u32,
}

fn warmup_profiling_enabled() -> bool {
    std::env::var("XLOG_WARMUP_PROFILE")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Detect device compute capability as a two-digit number (e.g. 75, 80, 120).
fn detect_compute_capability(device: &Arc<CudaDevice>) -> Result<u32> {
    let major = device
        .inner()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
        )
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM major: {}", e)))?;
    let minor = device
        .inner()
        .attribute(
            cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
        )
        .map_err(|e| XlogError::Kernel(format!("Failed to query SM minor: {}", e)))?;
    Ok((major as u32) * 10 + (minor as u32))
}

#[cfg(test)]
fn resolve_module_path(name: &str, cc: u32) -> Option<(std::path::PathBuf, bool)> {
    kernel_paths::KernelArtifactLocator::from_env().resolve_module_path(name, cc)
}

#[derive(Debug)]
pub(crate) enum KernelModuleSource {
    File { path: PathBuf, is_cubin: bool },
    EmbeddedPortablePtx { ptx: &'static str },
}

pub(crate) fn resolve_module_sources_with_locator(
    name: &str,
    cc: u32,
    locator: &kernel_paths::KernelArtifactLocator,
) -> Vec<KernelModuleSource> {
    let mut sources: Vec<KernelModuleSource> = locator
        .resolve_module_paths(name, cc)
        .into_iter()
        // Skip any staged cubin/PTX whose bytes diverge from what this binary
        // was built against. A stale staged artifact (kernel signature changed
        // but the staged copy was never refreshed) otherwise loads "fine" and
        // then launches a mismatched kernel into an illegal address.
        .filter(|(path, _)| !staged_artifact_is_stale(path))
        .map(|(path, is_cubin)| KernelModuleSource::File { path, is_cubin })
        .collect();

    // ALWAYS append the embedded portable PTX as the final fallback. It is
    // compiled into this binary, so it can never be stale relative to the launch
    // sites — it guarantees a signature-correct kernel even when every staged
    // File artifact was skipped as stale or fails to load. (Previously this was
    // suppressed whenever any portable-PTX *file* existed, which let a stale
    // staged PTX shadow the fresh embedded one.)
    if let Some(ptx) = crate::embedded_kernel_data::portable_ptx(name) {
        sources.push(KernelModuleSource::EmbeddedPortablePtx { ptx });
    }
    sources
}

/// A staged cubin/PTX is "stale" when this binary embeds a canonical integrity
/// hash for that artifact file name and the on-disk bytes do not match it — the
/// staged artifact diverges from what this build produced. Loading such an
/// artifact can launch a mismatched kernel into an illegal address, so it is
/// skipped in favor of a fresh source. Artifacts with no embedded canonical
/// hash (e.g. an arch this build did not produce) are NOT treated as stale — we
/// can only validate what we built — nor are unreadable files (the loader
/// surfaces the IO error).
fn staged_artifact_is_stale(path: &std::path::Path) -> bool {
    let Some(file_name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    let Some(expected) = crate::embedded_kernel_data::canonical_artifact_hash(file_name) else {
        return false;
    };
    match std::fs::read(path) {
        Ok(bytes) => fnv1a_64(&bytes) != expected,
        Err(_) => false,
    }
}

/// FNV-1a 64-bit, matching the build-time hash in `crates/xlog-cuda/build.rs`.
fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for &byte in bytes {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

#[cfg(test)]
mod kernel_source_resolution_tests {
    use super::{
        kernel_paths::KernelArtifactLocator, resolve_module_sources_with_locator,
        KernelModuleSource,
    };
    use std::fs;

    #[test]
    fn keeps_portable_ptx_fallback_when_cubin_exists() {
        let root = std::env::temp_dir().join(format!(
            "xlog-kernel-fallback-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        let kernels = root.join("kernels");
        fs::create_dir_all(&kernels).expect("create kernels dir");
        // Use a name this build does NOT produce, so neither file carries a
        // canonical integrity hash (the staleness skip is exercised separately).
        // This isolates the file-resolution precedence: cubin first, then the
        // portable-PTX file as fallback.
        fs::write(kernels.join("fakekernel.sm_86.cubin"), b"cubin").expect("write cubin");
        fs::write(kernels.join("fakekernel.portable.ptx"), b"ptx").expect("write ptx");
        let expected_cubin = kernels.join("fakekernel.sm_86.cubin");
        let expected_ptx = kernels.join("fakekernel.portable.ptx");

        let locator = KernelArtifactLocator::new(None, Some(kernels.clone()), None);
        let sources = resolve_module_sources_with_locator("fakekernel", 86, &locator);

        assert_eq!(sources.len(), 2);
        assert!(matches!(
            &sources[0],
            KernelModuleSource::File {
                path,
                is_cubin: true
            } if path == &expected_cubin
        ));
        assert!(matches!(
            &sources[1],
            KernelModuleSource::File {
                path,
                is_cubin: false
            } if path == &expected_ptx
        ));

        fs::remove_dir_all(root).expect("remove temp kernels");
    }

    // Locks the FNV-1a contract between build.rs (which embeds canonical
    // artifact hashes) and the runtime (which re-hashes staged artifacts). If
    // these two implementations ever diverge, every staged artifact would read
    // as "stale" — so these canonical FNV-1a 64-bit vectors must hold.
    #[test]
    fn fnv1a_64_matches_known_vectors() {
        assert_eq!(super::fnv1a_64(b""), 0xcbf2_9ce4_8422_2325);
        assert_eq!(super::fnv1a_64(b"a"), 0xaf63_dc4c_8601_ec8c);
        assert_eq!(super::fnv1a_64(b"foobar"), 0x85944171_f73967e8);
    }

    // A file whose name this build did not produce has no canonical hash, so it
    // is conservatively NOT treated as stale (we only validate what we built).
    // A nonexistent path is likewise not "stale" — the loader surfaces IO.
    #[test]
    fn staged_artifact_not_stale_without_canonical_hash() {
        let root = std::env::temp_dir().join(format!(
            "xlog-kernel-stale-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        fs::create_dir_all(&root).expect("create dir");
        let unknown = root.join("definitely_not_a_real_kernel.sm_86.cubin");
        fs::write(&unknown, b"bytes").expect("write");
        assert!(!super::staged_artifact_is_stale(&unknown));
        assert!(!super::staged_artifact_is_stale(
            &root.join("missing.portable.ptx")
        ));
        fs::remove_dir_all(root).expect("remove temp dir");
    }
}

/// Resolve a kernel module from sidecar artifacts or embedded portable PTX.
///
/// Asserts (in debug builds) that `name` is present in the kernel manifest,
/// catching name/order drift between the manifest and provider load blocks.
pub(crate) fn load_module_sources(name: &str, cc: u32) -> Result<Vec<KernelModuleSource>> {
    debug_assert!(
        crate::kernel_manifest_data::KERNEL_CU_NAMES.contains(&name),
        "kernel module '{name}' is not in KERNEL_CU_NAMES manifest — update kernel_manifest_data.rs"
    );
    let locator = kernel_paths::KernelArtifactLocator::from_env();
    let sources = resolve_module_sources_with_locator(name, cc, &locator);
    if sources.is_empty() {
        Err(XlogError::Kernel(format!(
            "{name}: no cubin, sidecar portable PTX, or embedded portable PTX found"
        )))
    } else {
        Ok(sources)
    }
}

#[derive(Clone)]
pub(crate) struct RawCudaView<'a, T> {
    ptr: cudarc::driver::sys::CUdeviceptr,
    len: usize,
    stream: Arc<CudaStream>,
    /// Optional back-reference to the source [`DeviceBlock`]
    /// when this view borrows a region of a runtime-backed
    /// allocation. The launch recorder uses this to attach
    /// cross-stream uses without losing identity through view
    /// construction. `None` for views built from external
    /// memory or legacy paths; strict-mode launch recorders
    /// reject `None` views.
    ///
    /// Read by [`RawCudaView::runtime_block`]; the field
    /// itself is intentionally not directly exposed because
    /// the lifetime of the back-reference is bound to the
    /// view's `'a`.
    #[allow(dead_code)]
    source_block: Option<&'a crate::device_runtime::DeviceBlock>,
    _marker: PhantomData<&'a [T]>,
}

/// Preallocated scratch layout for graph-capturable u32 multi-block scans.
///
/// The legacy stream-aware scan helper allocates recursive `block_sums`
/// buffers inside the helper. CUDA Graph capture records concrete allocation
/// addresses, so bounded CSM CUDA Graph replay needs the scan topology and
/// scratch buffers to be fixed before capture begins.
pub(crate) struct MultiblockScanScratchU32 {
    levels: Vec<TrackedCudaSlice<u32>>,
}

impl MultiblockScanScratchU32 {
    pub(crate) fn levels(&self) -> &[TrackedCudaSlice<u32>] {
        &self.levels
    }
}

pub(crate) struct CsmCudaGraphNodes {
    pub(crate) count: CudaGraphNode,
    pub(crate) total: CudaGraphNode,
    pub(crate) materialize: CudaGraphNode,
    pub(crate) node_count: usize,
}

pub(crate) struct CsmCudaGraphEntry {
    pub(crate) graph: CapturedCudaGraph,
    pub(crate) nodes: CsmCudaGraphNodes,
    pub(crate) per_probe_count: TrackedCudaSlice<u32>,
    pub(crate) per_probe_offsets: TrackedCudaSlice<u32>,
    pub(crate) d_logical_count: TrackedCudaSlice<u32>,
    pub(crate) d_overflow: TrackedCudaSlice<u8>,
    pub(crate) d_output_left: TrackedCudaSlice<u32>,
    pub(crate) d_output_right: TrackedCudaSlice<u32>,
    pub(crate) scan_scratch: MultiblockScanScratchU32,
    pub(crate) probe_capacity: u32,
    pub(crate) output_capacity: u32,
}

impl<'a, T> DeviceSlice<T> for RawCudaView<'a, T> {
    fn len(&self) -> usize {
        self.len
    }

    fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }
}

impl<'a, T> DevicePtr<T> for RawCudaView<'a, T> {
    fn device_ptr<'b>(
        &'b self,
        _stream: &'b CudaStream,
    ) -> (
        cudarc::driver::sys::CUdeviceptr,
        cudarc::driver::SyncOnDrop<'b>,
    ) {
        (self.ptr, cudarc::driver::SyncOnDrop::Sync(None))
    }
}

impl<'a, T> RawCudaView<'a, T> {
    pub fn device_ptr(&self) -> &cudarc::driver::sys::CUdeviceptr {
        &self.ptr
    }

    /// Borrow the back-reference to the source
    /// [`crate::device_runtime::DeviceBlock`], if this view was
    /// constructed from a runtime-backed allocation. Returns
    /// `None` for views built from external memory or legacy
    /// paths.
    ///
    /// Public API reserved for the filter-class migration; no
    /// production caller exists yet.
    #[allow(dead_code)]
    pub fn runtime_block(&self) -> Option<&'a crate::device_runtime::DeviceBlock> {
        self.source_block
    }
}

impl<'a, T: DeviceRepr> AsKernelParam for &RawCudaView<'a, T> {
    fn as_kernel_param(&self) -> *mut c_void {
        ((*self).device_ptr() as *const cudarc::driver::sys::CUdeviceptr)
            .cast_mut()
            .cast()
    }
}

impl<'a, T: DeviceRepr> IntoKernelParamStorage for &'a RawCudaView<'a, T> {
    type Storage = DeviceParamStorage<'a>;

    fn into_kernel_param_storage(self) -> Self::Storage {
        DeviceParamStorage::unsynced(self.ptr)
    }
}

/// Scratch buffers for stable radix sorting of u32 key/value pairs.
pub struct RadixSortScratch {
    keys_b: TrackedCudaSlice<u32>,
    values_b: TrackedCudaSlice<u32>,
    hist: TrackedCudaSlice<u32>,
    prefix: TrackedCudaSlice<u32>,
    ranks: TrackedCudaSlice<u32>,
    len: u32,
}

impl RadixSortScratch {
    pub fn new(provider: &CudaKernelProvider, n: u32) -> Result<Self> {
        let memory = provider.memory();
        let len = n.max(1);
        let keys_b = memory.alloc::<u32>(len as usize)?;
        let values_b = memory.alloc::<u32>(len as usize)?;
        let ranks = memory.alloc::<u32>(len as usize)?;
        let block_size = CudaKernelProvider::SORT_BLOCK_SIZE;
        let grid_size = len.div_ceil(block_size).max(1);
        let hist = memory.alloc::<u32>((grid_size as usize) * 16)?;
        let prefix = memory.alloc::<u32>(16)?;
        Ok(Self {
            keys_b,
            values_b,
            hist,
            prefix,
            ranks,
            len,
        })
    }

    pub fn ensure_capacity(&mut self, provider: &CudaKernelProvider, n: u32) -> Result<()> {
        if n <= self.len {
            return Ok(());
        }
        *self = Self::new(provider, n)?;
        Ok(())
    }
}

/// Module names for loaded PTX modules
pub const JOIN_MODULE: &str = "xlog_join";
pub const DEDUP_MODULE: &str = "xlog_dedup";
pub const GROUPBY_MODULE: &str = "xlog_groupby";
pub const SCAN_MODULE: &str = "xlog_scan";
pub const SORT_MODULE: &str = "xlog_sort";
pub const FILTER_MODULE: &str = "xlog_filter";
pub const SET_OPS_MODULE: &str = "xlog_set_ops";
pub const PACK_MODULE: &str = "xlog_pack";
pub const CIRCUIT_MODULE: &str = "xlog_circuit";
pub const MC_SAMPLE_MODULE: &str = "xlog_mc_sample";
pub const MC_EVAL_MODULE: &str = "xlog_mc_eval";
pub const MC_RESIDENT_MODULE: &str = "xlog_mc_resident";
pub const ARITH_MODULE: &str = "xlog_arith";
pub const SAT_MODULE: &str = "xlog_sat";
pub const D4_MODULE: &str = "xlog_d4";
pub const NEURAL_MODULE: &str = "xlog_neural";
pub const PIR_MODULE: &str = "xlog_pir";
pub const CNF_MODULE: &str = "xlog_cnf";
pub const CACHE_MODULE: &str = "xlog_cache";
pub const WEIGHTS_MODULE: &str = "xlog_weights";
pub const ILP_MODULE: &str = "xlog_ilp";
pub const ILP_CREDIT_MODULE: &str = "xlog_ilp_credit";
pub const ILP_EXACT_MODULE: &str = "xlog_ilp_exact";
pub const EPISTEMIC_MODULE: &str = "xlog_epistemic";
pub const WCOJ_MODULE: &str = "xlog_wcoj";

// Compile-time check: kernel manifest lists exactly 25 modules.
const _: () = assert!(crate::kernel_manifest_data::KERNEL_CU_NAMES.len() == 25);

/// Kernel function names in the GPU WCOJ module.
pub mod wcoj_kernels {
    pub const WCOJ_BUILD_METADATA_MARK_BOUNDARIES_U32: &str =
        "wcoj_build_metadata_mark_boundaries_u32";
    pub const WCOJ_BUILD_METADATA_MARK_BOUNDARIES_U64: &str =
        "wcoj_build_metadata_mark_boundaries_u64";
    pub const WCOJ_BUILD_METADATA_SCATTER_U32: &str = "wcoj_build_metadata_scatter_u32";
    pub const WCOJ_BUILD_METADATA_SCATTER_U64: &str = "wcoj_build_metadata_scatter_u64";
    pub const WCOJ_TRIANGLE_BUILD_HG_WORK_PLAN_U32: &str = "wcoj_triangle_build_hg_work_plan_u32";
    pub const WCOJ_TRIANGLE_COUNT_HG_U32: &str = "wcoj_triangle_count_hg_u32";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_COUNT_HG_U32: &str =
        "wcoj_triangle_groupby_root_count_hg_u32";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_SUM_HG_U32: &str = "wcoj_triangle_groupby_root_sum_hg_u32";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_MIN_HG_U32: &str = "wcoj_triangle_groupby_root_min_hg_u32";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_MAX_HG_U32: &str = "wcoj_triangle_groupby_root_max_hg_u32";
    pub const WCOJ_TRIANGLE_MATERIALIZE_HG_U32: &str = "wcoj_triangle_materialize_hg_u32";
    pub const WCOJ_TRIANGLE_BUILD_HG_WORK_PLAN_U64: &str = "wcoj_triangle_build_hg_work_plan_u64";
    pub const WCOJ_TRIANGLE_COUNT_HG_U64: &str = "wcoj_triangle_count_hg_u64";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_COUNT_HG_U64: &str =
        "wcoj_triangle_groupby_root_count_hg_u64";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_SUM_HG_U64: &str = "wcoj_triangle_groupby_root_sum_hg_u64";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_MIN_HG_U64: &str = "wcoj_triangle_groupby_root_min_hg_u64";
    pub const WCOJ_TRIANGLE_GROUPBY_ROOT_MAX_HG_U64: &str = "wcoj_triangle_groupby_root_max_hg_u64";
    pub const WCOJ_GROUPBY_ROOT_SEGMENT_SUM_COUNTS_U32: &str =
        "wcoj_groupby_root_segment_sum_counts_u32";
    pub const WCOJ_GROUPBY_ROOT_SEGMENT_SUM_VALUES_U64: &str =
        "wcoj_groupby_root_segment_sum_values_u64";
    pub const WCOJ_GROUPBY_ROOT_SEGMENT_MIN_VALUES_U64: &str =
        "wcoj_groupby_root_segment_min_values_u64";
    pub const WCOJ_GROUPBY_ROOT_SEGMENT_MAX_VALUES_U64: &str =
        "wcoj_groupby_root_segment_max_values_u64";
    pub const WCOJ_TRIANGLE_MATERIALIZE_HG_U64: &str = "wcoj_triangle_materialize_hg_u64";
    pub const WCOJ_TRIANGLE_COUNT_HG_CACHED_U32: &str = "wcoj_triangle_count_hg_cached_u32";
    pub const WCOJ_TRIANGLE_MATERIALIZE_HG_CACHED_U32: &str =
        "wcoj_triangle_materialize_hg_cached_u32";
    pub const WCOJ_SCAN_HG_BLOCK_COUNTS_U32: &str = "wcoj_scan_hg_block_counts_u32";
    pub const WCOJ_COMPUTE_TOTAL: &str = "wcoj_compute_total";
    pub const WCOJ_LAYOUT_CHECK_SORTED_UNIQUE_U32: &str = "wcoj_layout_check_sorted_unique_u32";
    pub const WCOJ_LAYOUT_CHECK_SORTED_UNIQUE_U64: &str = "wcoj_layout_check_sorted_unique_u64";
    pub const WCOJ_4CYCLE_BUILD_E2_WORK_PREFIX_U32: &str = "wcoj_4cycle_build_e2_work_prefix_u32";
    pub const WCOJ_4CYCLE_BUILD_HG_WORK_PLAN_U32: &str = "wcoj_4cycle_build_hg_work_plan_u32";
    pub const WCOJ_4CYCLE_COUNT_HG_U32: &str = "wcoj_4cycle_count_hg_u32";
    pub const WCOJ_4CYCLE_GROUPBY_ROOT_COUNT_HG_U32: &str = "wcoj_4cycle_groupby_root_count_hg_u32";
    pub const WCOJ_4CYCLE_GROUPBY_ROOT_SUM_HG_U32: &str = "wcoj_4cycle_groupby_root_sum_hg_u32";
    pub const WCOJ_4CYCLE_GROUPBY_ROOT_MIN_HG_U32: &str = "wcoj_4cycle_groupby_root_min_hg_u32";
    pub const WCOJ_4CYCLE_GROUPBY_ROOT_MAX_HG_U32: &str = "wcoj_4cycle_groupby_root_max_hg_u32";
    pub const WCOJ_4CYCLE_MATERIALIZE_HG_U32: &str = "wcoj_4cycle_materialize_hg_u32";
    pub const WCOJ_4CYCLE_BUILD_E2_WORK_PREFIX_U64: &str = "wcoj_4cycle_build_e2_work_prefix_u64";
    pub const WCOJ_4CYCLE_BUILD_HG_WORK_PLAN_U64: &str = "wcoj_4cycle_build_hg_work_plan_u64";
    pub const WCOJ_4CYCLE_COUNT_HG_U64: &str = "wcoj_4cycle_count_hg_u64";
    pub const WCOJ_4CYCLE_GROUPBY_ROOT_COUNT_HG_U64: &str = "wcoj_4cycle_groupby_root_count_hg_u64";
    pub const WCOJ_4CYCLE_MATERIALIZE_HG_U64: &str = "wcoj_4cycle_materialize_hg_u64";
    // General-arity clique kernels (k=5..8 from a single template).
    pub const WCOJ_CLIQUE5_COUNT_HG_U32: &str = "wcoj_clique5_count_hg_u32";
    pub const WCOJ_CLIQUE5_MATERIALIZE_HG_U32: &str = "wcoj_clique5_materialize_hg_u32";
    pub const WCOJ_CLIQUE5_COUNT_HG_U64: &str = "wcoj_clique5_count_hg_u64";
    pub const WCOJ_CLIQUE5_MATERIALIZE_HG_U64: &str = "wcoj_clique5_materialize_hg_u64";
    pub const WCOJ_CLIQUE6_COUNT_HG_U32: &str = "wcoj_clique6_count_hg_u32";
    pub const WCOJ_CLIQUE6_MATERIALIZE_HG_U32: &str = "wcoj_clique6_materialize_hg_u32";
    pub const WCOJ_CLIQUE6_COUNT_HG_U64: &str = "wcoj_clique6_count_hg_u64";
    pub const WCOJ_CLIQUE6_MATERIALIZE_HG_U64: &str = "wcoj_clique6_materialize_hg_u64";
    pub const WCOJ_CLIQUE7_COUNT_HG_U32: &str = "wcoj_clique7_count_hg_u32";
    pub const WCOJ_CLIQUE7_MATERIALIZE_HG_U32: &str = "wcoj_clique7_materialize_hg_u32";
    pub const WCOJ_CLIQUE7_COUNT_HG_U64: &str = "wcoj_clique7_count_hg_u64";
    pub const WCOJ_CLIQUE7_MATERIALIZE_HG_U64: &str = "wcoj_clique7_materialize_hg_u64";
    pub const WCOJ_CLIQUE8_COUNT_HG_U32: &str = "wcoj_clique8_count_hg_u32";
    pub const WCOJ_CLIQUE8_MATERIALIZE_HG_U32: &str = "wcoj_clique8_materialize_hg_u32";
    pub const WCOJ_CLIQUE8_COUNT_HG_U64: &str = "wcoj_clique8_count_hg_u64";
    pub const WCOJ_CLIQUE8_MATERIALIZE_HG_U64: &str = "wcoj_clique8_materialize_hg_u64";
    pub const WCOJ_CLIQUE5_GROUPBY_ROOT_COUNT_HG_U32: &str =
        "wcoj_clique5_groupby_root_count_hg_u32";
    pub const WCOJ_CLIQUE6_GROUPBY_ROOT_COUNT_HG_U32: &str =
        "wcoj_clique6_groupby_root_count_hg_u32";
    // Free Join frontier engine primitives. The work
    // prefix kernel is width-agnostic (ranges are u32 row indices in
    // every width class); count/emit/probe have u64 data twins.
    pub const FJ_EXPAND_WORK_PREFIX_U32: &str = "fj_expand_work_prefix_u32";
    pub const FJ_EXPAND_COUNT_U32: &str = "fj_expand_count_u32";
    pub const FJ_EXPAND_EMIT_U32: &str = "fj_expand_emit_u32";
    pub const FJ_PROBE_REFINE_U32: &str = "fj_probe_refine_u32";
    pub const FJ_EXPAND_COUNT_U64: &str = "fj_expand_count_u64";
    pub const FJ_EXPAND_EMIT_U64: &str = "fj_expand_emit_u64";
    pub const FJ_PROBE_REFINE_U64: &str = "fj_probe_refine_u64";
    pub const FJ_COUNT_MULTIPLICITY: &str = "fj_count_multiplicity";
    // D3 S3 spike — factorized recursive delta novel-set pipeline.
    pub const FJ_DELTA_RANGE_U32: &str = "fj_delta_range_u32";
    pub const FJ_DELTA_MARK_U32: &str = "fj_delta_mark_u32";
    pub const FJ_DELTA_SUBTRACT_U32: &str = "fj_delta_subtract_u32";
    pub const FJ_DELTA_POPCOUNT: &str = "fj_delta_popcount";
    pub const FJ_DELTA_EMIT_U32: &str = "fj_delta_emit_u32";
    pub const FJ_DELTA_MAX_U32: &str = "fj_delta_max_u32";
    pub const FJ_DELTA_SPARSE_ESTIMATE: &str = "fj_delta_sparse_estimate";
    pub const FJ_DELTA_SPARSE_LOAD_R: &str = "fj_delta_sparse_load_r";
    pub const FJ_DELTA_SPARSE_INSERT_CANDIDATES: &str = "fj_delta_sparse_insert_candidates";
    pub const FJ_DELTA_SPARSE_MARK: &str = "fj_delta_sparse_mark";
    pub const FJ_DELTA_SPARSE_EMIT: &str = "fj_delta_sparse_emit";
}

/// Kernel function names in the Monte Carlo sampling module
pub mod mc_sample_kernels {
    pub const MC_SAMPLE_BERNOULLI: &str = "mc_sample_bernoulli";
}

/// Kernel function names in the Monte Carlo evaluation module
pub mod mc_eval_kernels {
    pub const MC_EVAL_MASK_VAR: &str = "mc_eval_mask_var";
    pub const MC_EVAL_MASK_AD: &str = "mc_eval_mask_ad_choice";
    pub const MC_EVAL_QUERY_EVIDENCE_TRUTH: &str = "mc_eval_query_evidence_truth";
    pub const MC_EVAL_ACCUMULATE_COUNTS: &str = "mc_accumulate_counts";
}

/// Kernel function names in the GPU-resident Datalog/MC engine module.
pub mod mc_resident_kernels {
    /// Single megakernel: evaluates all MC worlds to fixpoint and counts
    /// query/evidence satisfaction with zero host interaction in-region.
    pub const MC_RESIDENT_ENGINE: &str = "mc_resident_engine";
}

/// Kernel function names in the arithmetic module
pub mod arith_kernels {
    pub const ARITH_BINARY_I64: &str = "arith_binary_i64";
    pub const ARITH_BINARY_I32: &str = "arith_binary_i32";
    pub const ARITH_BINARY_U64: &str = "arith_binary_u64";
    pub const ARITH_BINARY_U32: &str = "arith_binary_u32";
    pub const ARITH_BINARY_F64: &str = "arith_binary_f64";
    pub const ARITH_BINARY_F32: &str = "arith_binary_f32";
    pub const ARITH_ABS_I64: &str = "arith_abs_i64";
    pub const ARITH_ABS_I32: &str = "arith_abs_i32";
    pub const ARITH_ABS_F64: &str = "arith_abs_f64";
    pub const ARITH_ABS_F32: &str = "arith_abs_f32";
    pub const ARITH_POW_F64: &str = "arith_pow_f64";
    pub const ARITH_CAST: &str = "arith_cast";
    pub const ARITH_FILL_CONST_U32: &str = "arith_fill_const_u32";
    pub const ARITH_FILL_CONST_U64: &str = "arith_fill_const_u64";
    pub const ARITH_FILL_CONST_I64: &str = "arith_fill_const_i64";
    pub const ARITH_FILL_CONST_I32: &str = "arith_fill_const_i32";
    pub const ARITH_FILL_CONST_F64: &str = "arith_fill_const_f64";
    pub const ARITH_FILL_CONST_F32: &str = "arith_fill_const_f32";
    pub const ARITH_FILL_CONST_U8: &str = "arith_fill_const_u8";
    // Conditional select kernels
    pub const ARITH_SELECT_I64: &str = "arith_select_i64";
    pub const ARITH_SELECT_I32: &str = "arith_select_i32";
    pub const ARITH_SELECT_U64: &str = "arith_select_u64";
    pub const ARITH_SELECT_U32: &str = "arith_select_u32";
    pub const ARITH_SELECT_F64: &str = "arith_select_f64";
    pub const ARITH_SELECT_F32: &str = "arith_select_f32";
}

/// Kernel function names in the epistemic module.
pub mod epistemic_kernels {
    /// Device-side epistemic candidate-assumption generator.
    pub const EPISTEMIC_GENERATE_CANDIDATE_ASSUMPTIONS_U8: &str =
        "epistemic_generate_candidate_assumptions_u8";
    /// Device-side epistemic candidate propagation staging kernel.
    pub const EPISTEMIC_PROPAGATE_CANDIDATES_U8: &str = "epistemic_propagate_candidates_u8";
    /// Device-side epistemic candidate bit validation kernel.
    pub const EPISTEMIC_VALIDATE_CANDIDATE_BITS_U8: &str = "epistemic_validate_candidate_bits_u8";
    /// Device-side model-membership staging kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_U8: &str =
        "epistemic_populate_model_membership_u8";
    /// Device-side tuple-source-backed model-membership kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_U8: &str =
        "epistemic_populate_model_membership_from_tuple_source_u8";
    /// Device-side arity-one tuple-key-backed model-membership kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY1_U8: &str =
        "epistemic_populate_model_membership_from_tuple_source_arity1_u8";
    /// Device-side arity-two tuple-key-backed model-membership kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY2_U8: &str =
        "epistemic_populate_model_membership_from_tuple_source_arity2_u8";
    /// Device-side arity-three tuple-key-backed model-membership kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY3_U8: &str =
        "epistemic_populate_model_membership_from_tuple_source_arity3_u8";
    /// Device-side generic-arity tuple-key-backed model-membership kernel.
    pub const EPISTEMIC_POPULATE_MODEL_MEMBERSHIP_FROM_TUPLE_SOURCE_ARITY_N_U8: &str =
        "epistemic_populate_model_membership_from_tuple_source_arity_n_u8";
    /// Device-side world-view validation kernel.
    pub const EPISTEMIC_VALIDATE_WORLD_VIEWS_U8: &str = "epistemic_validate_world_views_u8";
    /// Device-side world-view integrity-constraint validation kernel.
    pub const EPISTEMIC_VALIDATE_CONSTRAINTS_U8: &str = "epistemic_validate_constraints_u8";
    /// Device-side accepted-candidate materialization staging kernel.
    pub const EPISTEMIC_MATERIALIZE_ACCEPTED_CANDIDATES_U8: &str =
        "epistemic_materialize_accepted_candidates_u8";

    /// Device-side final-result flag materialization staging kernel.
    pub const EPISTEMIC_MATERIALIZE_FINAL_RESULT_FLAGS_U8: &str =
        "epistemic_materialize_final_result_flags_u8";
    /// Device-side final tuple materialization kernel.
    pub const EPISTEMIC_MATERIALIZE_FINAL_TUPLE_COLUMN_U8: &str =
        "epistemic_materialize_final_tuple_column_u8";
    /// Device-side final tuple row-map kernel.
    pub const EPISTEMIC_BUILD_FINAL_TUPLE_ROW_MAP_U8: &str =
        "epistemic_build_final_tuple_row_map_u8";
    /// Device-side final tuple rejection-close kernel.
    pub const EPISTEMIC_CLOSE_FINAL_TUPLE_REJECTIONS_U8: &str =
        "epistemic_close_final_tuple_rejections_u8";
}

/// Kernel function names in the neural fast-path module.
pub mod neural_kernels {
    pub const NEURAL_FILL_AD_CHAIN_F32: &str = "neural_fill_ad_chain_f32";
    pub const NEURAL_SCATTER_AD_CHAIN_GRADS_F32: &str = "neural_scatter_ad_chain_grads_f32";
}

/// Kernel function names in the ILP module.
pub mod ilp_kernels {
    pub const EXTRACT_NONZERO_INDICES: &str = "extract_nonzero_indices";
    pub const ILP_MARK_SELECTED_IDS_U32: &str = "ilp_mark_selected_ids_u32";
    pub const ILP_MARK_SELECTED_IDS_I32: &str = "ilp_mark_selected_ids_i32";
    pub const ILP_MARK_SELECTED_IDS_I64: &str = "ilp_mark_selected_ids_i64";
    pub const ILP_MARK_SELECTED_IDS_U64: &str = "ilp_mark_selected_ids_u64";
    pub const ILP_VALIDATE_SELECTED_IDS_U32: &str = "ilp_validate_selected_ids_u32";
    pub const ILP_VALIDATE_SELECTED_IDS_I32: &str = "ilp_validate_selected_ids_i32";
    pub const ILP_VALIDATE_SELECTED_IDS_I64: &str = "ilp_validate_selected_ids_i64";
    pub const ILP_VALIDATE_SELECTED_IDS_U64: &str = "ilp_validate_selected_ids_u64";
    pub const ILP_BROADCAST_CANDIDATE_FLAG: &str = "ilp_broadcast_candidate_flag";
    pub const ILP_COO_FILL_FROM_MASK: &str = "ilp_coo_fill_from_mask";
    pub const ILP_CSR_HISTOGRAM: &str = "ilp_csr_histogram";
    pub const ILP_REDUCE_SUM_F32: &str = "ilp_reduce_sum_f32";
    pub const ILP_REDUCE_SUM_F64: &str = "ilp_reduce_sum_f64";
}

/// Kernel function names in the ILP credit module.
pub mod ilp_credit_kernels {
    pub const ILP_COO_FILL: &str = "ilp_coo_fill";
    pub const ILP_CREDIT_FORWARD_F32: &str = "ilp_credit_forward_f32";
    pub const ILP_CREDIT_FORWARD_F64: &str = "ilp_credit_forward_f64";
    pub const ILP_CREDIT_BACKWARD_F32: &str = "ilp_credit_backward_f32";
    pub const ILP_CREDIT_BACKWARD_F64: &str = "ilp_credit_backward_f64";
}

/// Kernel function names in the native bounded exact-induction module.
pub mod ilp_exact_kernels {
    pub const ILP_EXACT_SCORE: &str = "ilp_exact_score";
    pub const ILP_EXACT_SCORE_U32: &str = "ilp_exact_score_u32";
    pub const ILP_EXACT_SCORE_CHAIN_SMEM: &str = "ilp_exact_score_chain_smem";
    pub const ILP_EXACT_SCORE_CHAIN_SMEM_U32: &str = "ilp_exact_score_chain_smem_u32";
    pub const ILP_EXACT_SELECT_TOPK: &str = "ilp_exact_select_topk";
}

/// Kernel function names in the PIR interning module.
pub mod pir_kernels {
    pub const PIR_PACK_KEYS: &str = "pir_pack_keys";
    pub const PIR_HASH_KEYS: &str = "pir_hash_keys";
    pub const PIR_MARK_UNIQUE: &str = "pir_mark_unique";
    pub const PIR_FIND_EXISTING: &str = "pir_find_existing";
    pub const PIR_MARK_NEW_GROUPS: &str = "pir_mark_new_groups";
    pub const PIR_BUILD_GROUP_IDS: &str = "pir_build_group_ids";
    pub const PIR_FILL_CHILD_PARENTS: &str = "pir_fill_child_parents";
    pub const PIR_MARK_UNIQUE_PAIRS: &str = "pir_mark_unique_pairs";
    pub const PIR_COMPACT_PAIRS: &str = "pir_compact_pairs";
    pub const PIR_COUNT_CHILDREN: &str = "pir_count_children";
    pub const PIR_WRITE_CHILD_OFFSETS: &str = "pir_write_child_offsets";
    pub const PIR_GATHER_CHILDREN: &str = "pir_gather_children";
    pub const PIR_BUILD_GRAPH_CHILD_COUNTS: &str = "pir_build_graph_child_counts";
    pub const PIR_SUM_COUNTS: &str = "pir_sum_counts";
    pub const PIR_EMIT_NODES_AND_IDS: &str = "pir_emit_nodes_and_ids";
    pub const PIR_UPDATE_COUNTS: &str = "pir_update_counts";
}

/// Kernel function names in the GPU CNF encoder module.
pub mod cnf_kernels {
    pub const CNF_REACHABILITY_INIT: &str = "cnf_reachability_init";
    pub const CNF_REACHABILITY_BFS: &str = "cnf_reachability_bfs";
    pub const CNF_MARK_LEAF_CHOICE: &str = "cnf_mark_leaf_choice";
    pub const CNF_ASSIGN_LEAF_VAR: &str = "cnf_assign_leaf_var";
    pub const CNF_ASSIGN_CHOICE_VAR: &str = "cnf_assign_choice_var";
    pub const CNF_MARK_NODE_VARS: &str = "cnf_mark_node_vars";
    pub const CNF_COUNT_CLAUSES: &str = "cnf_count_clauses";
    pub const CNF_CAPTURE_LAST_COUNTS: &str = "cnf_capture_last_counts";
    pub const CNF_COMPUTE_LEAF_CHOICE_TOTALS: &str = "cnf_compute_leaf_choice_totals";
    pub const CNF_COMPUTE_TOTALS: &str = "cnf_compute_totals";
    pub const CNF_ASSIGN_NODE_VAR: &str = "cnf_assign_node_var";
    pub const CNF_EMIT_CLAUSES: &str = "cnf_emit_clauses";
    pub const CNF_SET_CLAUSE_END: &str = "cnf_set_clause_end";
}

/// Kernel function names in the weights module.
pub mod weights_kernels {
    pub const WEIGHTS_FILL_LEAF: &str = "weights_fill_leaf";
    pub const WEIGHTS_FILL_CHOICE: &str = "weights_fill_choice";
    pub const WEIGHTS_COUNT_LIFT_EXACT: &str = "weights_count_lift_exact";
    pub const WEIGHTS_SET_EVIDENCE_FROM_NODES: &str = "weights_set_evidence_from_nodes";
    pub const WEIGHTS_APPLY_EVIDENCE: &str = "weights_apply_evidence";
    pub const WEIGHTS_MAP_NODES_TO_VARS: &str = "weights_map_nodes_to_vars";
    pub const WEIGHTS_FORCE_VAR_FALSE: &str = "weights_force_var_false";
    pub const WEIGHTS_RESTORE_VAR_FALSE: &str = "weights_restore_var_false";
    pub const WEIGHTS_FORCE_VAR_TRUE: &str = "weights_force_var_true";
    pub const WEIGHTS_RESTORE_VAR_TRUE: &str = "weights_restore_var_true";
    pub const WEIGHTS_COPY_SLOT_TO_BATCH: &str = "weights_copy_slot_to_batch";
    pub const WEIGHTS_APPLY_QUERY_VARS: &str = "weights_apply_query_vars";
    pub const WEIGHTS_RESTORE_QUERY_VARS: &str = "weights_restore_query_vars";
    pub const WEIGHTS_APPLY_QUERY_VARS_FALSE_BATCHED: &str =
        "weights_apply_query_vars_false_batched";
    pub const WEIGHTS_RESTORE_QUERY_VARS_FALSE_BATCHED: &str =
        "weights_restore_query_vars_false_batched";
    pub const WEIGHTS_APPLY_QUERY_VARS_TRUE_BATCHED: &str = "weights_apply_query_vars_true_batched";
    pub const WEIGHTS_RESTORE_QUERY_VARS_TRUE_BATCHED: &str =
        "weights_restore_query_vars_true_batched";
}

/// Kernel function names in the GPU Decision-DNNF compiler module
/// (CNF validation + circuit levelization).
pub mod d4_kernels {
    pub const D4_VALIDATE_CNF: &str = "d4_validate_cnf";
    pub const D4_LEVELIZE_COUNTS: &str = "d4_levelize_counts";
    pub const D4_LEVELIZE_EMIT: &str = "d4_levelize_emit";
    // BFS frontier expansion and unit propagation.
    pub const D4_FRONTIER_PREPARE: &str = "d4_frontier_prepare";
    pub const D4_FRONTIER_EXPAND: &str = "d4_frontier_expand";
    pub const D4_FRONTIER_PREPARE_DENSE: &str = "d4_frontier_prepare_dense";
    pub const D4_FRONTIER_EXPAND_DENSE: &str = "d4_frontier_expand_dense";
    // Per-frontier Decision-DNNF DFS worker (count+emit).
    pub const D4_COMPILE_COUNT: &str = "d4_compile_count";
    pub const D4_COMPILE_EMIT: &str = "d4_compile_emit";
    pub const D4_CAPTURE_EMIT_META: &str = "d4_capture_emit_meta";
    // GPU smoothing with random-variable support and wrapper emission.
    pub const D4_SUPPORT_LEVEL: &str = "d4_support_level";
    pub const D4_SUPPORT_SET_ROOT_BITS: &str = "d4_support_set_root_bits";
    pub const D4_SMOOTH_COUNT: &str = "d4_smooth_count";
    pub const D4_SMOOTH_WRAPPER_COUNTS: &str = "d4_smooth_wrapper_counts";
    pub const D4_SMOOTH_WRAPPER_EDGE_COUNTS_OR: &str = "d4_smooth_wrapper_edge_counts_or";
    pub const D4_SMOOTH_WRAPPER_EDGE_COUNTS_DEC: &str = "d4_smooth_wrapper_edge_counts_dec";
    pub const D4_SMOOTH_INIT_NODES: &str = "d4_smooth_init_nodes";
    pub const D4_SMOOTH_EMIT_LEVEL: &str = "d4_smooth_emit_level";
    pub const D4_SMOOTH_CHECK_EDGE_CAP: &str = "d4_smooth_check_edge_cap";
    // GPU free-variable mask for variables in clauses versus the circuit.
    pub const D4_MARK_VARS_IN_CLAUSES: &str = "d4_mark_vars_in_clauses";
    pub const D4_MARK_VARS_IN_CIRCUIT: &str = "d4_mark_vars_in_circuit";
    pub const D4_BUILD_FREE_VAR_MASK: &str = "d4_build_free_var_mask";
    // GPU-only assertions (tests + invariant enforcement without host reads).
    pub const D4_ASSERT_U32_EQ: &str = "d4_assert_u32_eq";
    pub const D4_ASSERT_BITSET_VAR: &str = "d4_assert_bitset_var";
    pub const D4_ASSERT_DENSE_VAR: &str = "d4_assert_dense_var";
    pub const D4_ASSERT_LEAF_ROOT_AND_DEGREE: &str = "d4_assert_leaf_root_and_degree";
}

/// Kernel function names in the join module
pub mod join_kernels {
    pub const HASH_JOIN_BUILD: &str = "hash_join_build";
    pub const HASH_JOIN_PROBE: &str = "hash_join_probe";
    // V2 kernels for multi-column joins
    pub const COMPUTE_COMPOSITE_HASH: &str = "compute_composite_hash";
    pub const HASH_JOIN_BUCKET_COUNT_V2: &str = "hash_join_bucket_count_v2";
    pub const HASH_JOIN_SCATTER_V2: &str = "hash_join_scatter_v2";
    pub const HASH_JOIN_PROBE_V2: &str = "hash_join_probe_v2";
    pub const HASH_JOIN_PROBE_V2_COUNT_PER_ROW: &str = "hash_join_probe_v2_count_per_row";
    pub const HASH_JOIN_PROBE_V2_MATERIALIZE: &str = "hash_join_probe_v2_materialize";
    pub const HASH_JOIN_TOTAL_FROM_SCAN: &str = "hash_join_total_from_scan";
    pub const HASH_JOIN_CSM_UNMATCHED_MASK: &str = "hash_join_csm_unmatched_mask";
    pub const HASH_JOIN_SEMI: &str = "hash_join_semi";
    pub const HASH_JOIN_ANTI: &str = "hash_join_anti";
    pub const INIT_HASH_TABLE: &str = "init_hash_table";
    /// Nested-loop inner join (emit-pairs design). Reads
    /// the single key column from each side; emits matched
    /// `(left_idx, right_idx)` pairs as two parallel u32 arrays.
    /// Payload columns are materialized after the kernel via
    /// `gather_buffer_by_indices` in the provider fn.
    pub const NESTED_LOOP_JOIN_INNER_U32_1KEY_PAIRS: &str = "nested_loop_join_inner_u32_1key_pairs";
    /// Sort-merge inner join (emit-pairs design,
    /// caller-asserted pre-sorted inputs). Reads the single
    /// key column from each side, performs per-thread binary
    /// search on the right side to find matched-key runs,
    /// emits `(left_idx, right_idx)` pairs as two parallel
    /// u32 arrays. Payload columns materialize after the
    /// kernel via `gather_buffer_by_indices`.
    pub const SORT_MERGE_JOIN_INNER_U32_1KEY_PAIRS: &str = "sort_merge_join_inner_u32_1key_pairs";
}

/// Kernel function names in the dedup module
pub mod dedup_kernels {
    pub const MARK_DUPLICATES: &str = "mark_duplicates";
    pub const MARK_UNIQUE_COLUMNAR: &str = "mark_unique_columnar";
    pub const MARK_UNIQUE_AND_SCAN_COLUMNAR: &str = "mark_unique_and_scan_columnar";
    pub const COMPACT_ROWS: &str = "compact_rows";
    pub const MARK_UNIQUE_FULL_ROW_BYTEWISE: &str = "mark_unique_full_row_bytewise";
    pub const MARK_DIFF_FULL_ROW_TYPED_SORTED: &str = "mark_diff_full_row_typed_sorted";
    pub const SMALL_SORT_FULL_ROW_INDICES_TYPED: &str = "small_sort_full_row_indices_typed";
}

/// Kernel function names in the groupby module
pub mod groupby_kernels {
    pub const DETECT_GROUP_BOUNDARIES: &str = "detect_group_boundaries";
    pub const DETECT_BOUNDARIES: &str = "detect_boundaries";
    pub const EXTRACT_GROUP_KEYS: &str = "extract_group_keys";
    pub const GROUP_IDS_FROM_BOUNDARIES: &str = "group_ids_from_boundaries";
    pub const GROUP_START_INDICES: &str = "group_start_indices";
    pub const CAPTURE_NUM_GROUPS: &str = "capture_num_groups";
    pub const GROUPBY_COUNT: &str = "groupby_count";
    pub const GROUPBY_SUM: &str = "groupby_sum";
    pub const GROUPBY_SUM_U64: &str = "groupby_sum_u64";
    pub const GROUPBY_MIN: &str = "groupby_min";
    pub const GROUPBY_MIN_U64: &str = "groupby_min_u64";
    pub const GROUPBY_MAX: &str = "groupby_max";
    pub const GROUPBY_MAX_U64: &str = "groupby_max_u64";
    pub const GROUPBY_LOGSUMEXP_MAX: &str = "groupby_logsumexp_max";
    pub const GROUPBY_LOGSUMEXP_SUMEXP: &str = "groupby_logsumexp_sumexp";
    pub const GROUPBY_LOGSUMEXP_FINAL: &str = "groupby_logsumexp_final";
}

/// Kernel function names in the scan module
pub mod scan_kernels {
    pub const BLOCK_INCLUSIVE_SCAN: &str = "block_inclusive_scan";
    pub const ADD_BLOCK_OFFSETS: &str = "add_block_offsets";
    pub const EXCLUSIVE_SCAN_MASK: &str = "exclusive_scan_mask";
    pub const COUNT_MASK: &str = "count_mask";
    // Multi-block scan kernels for large prefix sums
    pub const MULTIBLOCK_SCAN_PHASE1: &str = "multiblock_scan_phase1";
    pub const MULTIBLOCK_SCAN_U32_PHASE1: &str = "multiblock_scan_u32_phase1";
    pub const MULTIBLOCK_SCAN_PHASE2: &str = "multiblock_scan_phase2";
    pub const MULTIBLOCK_SCAN_PHASE3: &str = "multiblock_scan_phase3";
}

/// Kernel function names in the sort module
pub mod sort_kernels {
    pub const RADIX_HISTOGRAM: &str = "radix_histogram";
    pub const RADIX_SCATTER: &str = "radix_scatter";
    pub const COMPUTE_RANKS: &str = "compute_ranks";
    pub const RADIX_SCATTER_STABLE: &str = "radix_scatter_stable";
    pub const COMPUTE_DIGIT_PREFIX_SUMS: &str = "compute_digit_prefix_sums";
    pub const INIT_INDICES: &str = "init_indices";
    pub const APPLY_PERMUTATION_U32: &str = "apply_permutation_u32";
    pub const APPLY_PERMUTATION_BYTES: &str = "apply_permutation_bytes";

    pub const GATHER_KEYS_I32_ORDERED_U32: &str = "gather_keys_i32_ordered_u32";
    pub const GATHER_KEYS_F32_ORDERED_U32: &str = "gather_keys_f32_ordered_u32";
    pub const GATHER_KEYS_BOOL_ORDERED_U32: &str = "gather_keys_bool_ordered_u32";

    pub const GATHER_KEYS_U64_LO_U32: &str = "gather_keys_u64_lo_u32";
    pub const GATHER_KEYS_U64_HI_U32: &str = "gather_keys_u64_hi_u32";

    pub const GATHER_KEYS_I64_LO_U32: &str = "gather_keys_i64_lo_u32";
    pub const GATHER_KEYS_I64_HI_U32: &str = "gather_keys_i64_hi_u32";

    pub const GATHER_KEYS_F64_LO_U32: &str = "gather_keys_f64_lo_u32";
    pub const GATHER_KEYS_F64_HI_U32: &str = "gather_keys_f64_hi_u32";
    /// Sort-merge sortedness-detection kernel — single-pass adjacent-
    /// pair check; atomically writes 0 to a u32 flag on
    /// `keys[i] > keys[i+1]`. Caller initializes flag to 1
    /// before launch, reads result post-launch. Used by the
    /// dispatch-site eligibility check at `execute_join` to
    /// validate caller-asserted sortedness before invoking
    /// `sort_merge_join_v2_inner_u32_1key`.
    pub const CHECK_ASCENDING_SORTED_U32: &str = "check_ascending_sorted_u32";
}

/// Kernel function names in the filter module
pub mod filter_kernels {
    pub const FILTER_COMPARE_U32: &str = "filter_compare_u32";
    pub const FILTER_COMPARE_I64: &str = "filter_compare_i64";
    pub const FILTER_COMPARE_F64: &str = "filter_compare_f64";
    pub const FILTER_COMPARE_I32: &str = "filter_compare_i32";
    pub const FILTER_COMPARE_U64: &str = "filter_compare_u64";
    pub const FILTER_COMPARE_F32: &str = "filter_compare_f32";
    pub const FILTER_COMPARE_U8: &str = "filter_compare_u8";
    pub const FILTER_COMPARE_U32_SCAN_PHASE1: &str = "filter_compare_u32_scan_phase1";
    pub const FILTER_COMPARE_F64_SCAN_PHASE1: &str = "filter_compare_f64_scan_phase1";
    pub const FILTER_COMPARE_F32_SCAN_PHASE1: &str = "filter_compare_f32_scan_phase1";
    pub const FILTER_COMPARE_U32_COL: &str = "filter_compare_u32_col";
    pub const FILTER_COMPARE_I32_COL: &str = "filter_compare_i32_col";
    pub const FILTER_COMPARE_I64_COL: &str = "filter_compare_i64_col";
    pub const FILTER_COMPARE_U64_COL: &str = "filter_compare_u64_col";
    pub const FILTER_COMPARE_F32_COL: &str = "filter_compare_f32_col";
    pub const FILTER_COMPARE_F64_COL: &str = "filter_compare_f64_col";
    pub const FILTER_COMPARE_U8_COL: &str = "filter_compare_u8_col";
    pub const FILL_U32_IOTA: &str = "fill_u32_iota";
    pub const FILL_U32_CONST: &str = "fill_u32_const";
    pub const MARK_RANDOM_VARS: &str = "mark_random_vars";
    pub const RANDOM_VAR_TO_BIT_FROM_LIST: &str = "random_var_to_bit_from_list";
    pub const CHECK_RANDOM_VAR_COUNT: &str = "check_random_var_count";
    pub const COMPACT_U32_BY_MASK: &str = "compact_u32_by_mask";
    pub const COMPACT_I64_BY_MASK: &str = "compact_i64_by_mask";
    pub const COMPACT_F64_BY_MASK: &str = "compact_f64_by_mask";
    pub const COMPACT_BYTES_BY_MASK: &str = "compact_bytes_by_mask";
    pub const CAPTURE_COMPACT_COUNT: &str = "capture_compact_count";
    pub const MASK_CLAMP_ROWS: &str = "mask_clamp_rows";
    pub const MASK_AND: &str = "mask_and";
    pub const MASK_OR: &str = "mask_or";
    pub const MASK_NOT: &str = "mask_not";
}

/// Kernel function names in the set_ops module
pub mod set_ops_kernels {
    pub const CONCAT_U32: &str = "concat_u32";
    pub const CONCAT_BYTES: &str = "concat_bytes";
    pub const SORTED_DIFF_MARK: &str = "sorted_diff_mark";
}

/// Kernel function names in the pack module (GPU-side key packing)
pub mod pack_kernels {
    /// Pack multiple columns into row-major byte array
    pub const PACK_KEYS: &str = "pack_keys";
    /// Compute FNV-1a hash from packed keys
    pub const HASH_PACKED_KEYS: &str = "hash_packed_keys";
    /// Fused pack + hash in single pass (optimal for join key preparation)
    pub const PACK_AND_HASH_KEYS: &str = "pack_and_hash_keys";
    /// Fused pack + hash for arbitrary key column counts
    pub const PACK_AND_HASH_KEYS_GENERIC: &str = "pack_and_hash_keys_generic";
    /// Vectorized pack for 8-byte aligned columns
    pub const PACK_KEYS_ALIGNED: &str = "pack_keys_aligned";
    /// Unpack single column from packed row data
    pub const UNPACK_COLUMN: &str = "unpack_column";
    /// Unpack single column with device-resident row count
    pub const UNPACK_COLUMN_COUNTED: &str = "unpack_column_counted";
    /// Gather rows from packed data based on index array
    pub const GATHER_PACKED_ROWS: &str = "gather_packed_rows";
    /// Gather rows with device-resident row count
    pub const GATHER_PACKED_ROWS_COUNTED: &str = "gather_packed_rows_counted";
    /// Scatter write: distribute packed rows to non-contiguous output positions
    pub const SCATTER_PACKED_ROWS: &str = "scatter_packed_rows";
    /// Compare packed keys for equality
    pub const COMPARE_PACKED_KEYS: &str = "compare_packed_keys";
    /// Pack u8 bools into Arrow bitmap bytes
    pub const PACK_BOOLS_TO_BITMAP: &str = "pack_bools_to_bitmap";
}

/// Kernel function names in the circuit module
pub mod circuit_kernels {
    pub const XGCF_FORWARD_LEVEL: &str = "xgcf_forward_level";
    pub const XGCF_BACKWARD_LEVEL_PROPAGATE: &str = "xgcf_backward_level_propagate";
    pub const XGCF_BACKWARD_LEVEL_DECISION_GRAD: &str = "xgcf_backward_level_decision_grad";
    pub const XGCF_BACKWARD_LEVEL_LIT_GRAD: &str = "xgcf_backward_level_lit_grad";
    pub const XGCF_FREE_VAR_APPLY_GRAD: &str = "xgcf_free_var_apply_grad";
    pub const XGCF_FREE_VAR_REDUCE_STAGE: &str = "xgcf_free_var_reduce_stage";
    pub const XGCF_ADD_SCALAR: &str = "xgcf_add_scalar";
    pub const XGCF_FORWARD_LEVEL_CACHED: &str = "xgcf_forward_level_cached";
    pub const XGCF_EVAL_ALL_LEVELS_CACHED: &str = "xgcf_eval_all_levels_cached";
    pub const XGCF_EVAL_ALL_LEVELS_CACHED_BATCHED: &str = "xgcf_eval_all_levels_cached_batched";
    pub const XGCF_BACKWARD_LEVEL_PROPAGATE_CACHED: &str = "xgcf_backward_level_propagate_cached";
    pub const XGCF_BACKWARD_LEVEL_DECISION_GRAD_CACHED: &str =
        "xgcf_backward_level_decision_grad_cached";
    pub const XGCF_BACKWARD_LEVEL_LIT_GRAD_CACHED: &str = "xgcf_backward_level_lit_grad_cached";
    pub const XGCF_BACKWARD_ALL_LEVELS_CACHED: &str = "xgcf_backward_all_levels_cached";
    pub const XGCF_BACKWARD_ALL_LEVELS_CACHED_BATCHED: &str =
        "xgcf_backward_all_levels_cached_batched";
    pub const XGCF_FREE_VAR_APPLY_GRAD_CACHED: &str = "xgcf_free_var_apply_grad_cached";
    pub const XGCF_FREE_VAR_REDUCE_STAGE_CACHED: &str = "xgcf_free_var_reduce_stage_cached";
    pub const XGCF_ADD_SCALAR_CACHED: &str = "xgcf_add_scalar_cached";
    pub const XGCF_SET_ROOT_ADJ_CACHED_BATCHED: &str = "xgcf_set_root_adj_cached_batched";
    pub const XGCF_COPY_ROOT_CACHED: &str = "xgcf_copy_root_cached";
    pub const XGCF_COPY_ROOT_CACHED_META: &str = "xgcf_copy_root_cached_meta";
    pub const XGCF_COPY_ROOT_CACHED_META_BATCHED: &str = "xgcf_copy_root_cached_meta_batched";
}

/// Kernel function names in the cache module
pub mod cache_kernels {
    pub const CACHE_CNF_HASH: &str = "cache_cnf_hash";
    pub const CACHE_LOOKUP_OR_INSERT: &str = "cache_lookup_or_insert";
    pub const CACHE_EVICT_LRU: &str = "cache_evict_lru";
    pub const CACHE_STORE_U8: &str = "cache_store_u8";
    pub const CACHE_STORE_U32: &str = "cache_store_u32";
    pub const CACHE_STORE_I32: &str = "cache_store_i32";
    pub const CACHE_STORE_F64: &str = "cache_store_f64";
    pub const CACHE_STORE_META: &str = "cache_store_meta";
}

/// Kernel function names in the SAT module
pub mod sat_kernels {
    pub const SAT_CDCL_SOLVE: &str = "sat_cdcl_solve";
    pub const SAT_CHECK_MODEL: &str = "sat_check_model";
    pub const SAT_PROOF_MARK_NEEDED: &str = "sat_proof_mark_needed";
    pub const SAT_PROOF_CHECK: &str = "sat_proof_check";
    pub const SAT_ASSERT_STATUS: &str = "sat_assert_status";
    pub const SAT_ASSERT_OK: &str = "sat_assert_ok";
    pub const SAT_XGCF_CNF_COUNTS: &str = "sat_xgcf_cnf_counts";
    pub const SAT_XGCF_CNF_EMIT: &str = "sat_xgcf_cnf_emit";
    pub const SAT_XGCF_CNF_CAPTURE_LAST_COUNTS: &str = "sat_xgcf_cnf_capture_last_counts";
    pub const SAT_XGCF_CNF_COMPUTE_TOTALS: &str = "sat_xgcf_cnf_compute_totals";
    pub const SAT_CNF_WRITE_TERMINATOR: &str = "sat_cnf_write_terminator";
    pub const SAT_CNF_COPY_INTO: &str = "sat_cnf_copy_into";
    pub const SAT_SHIFT_OFFSETS: &str = "sat_shift_offsets";
    pub const SAT_XGCF_WRITE_ROOT_UNIT_CLAUSE: &str = "sat_xgcf_write_root_unit_clause";
    pub const SAT_NOT_PHI_COUNTS: &str = "sat_not_phi_counts";
    pub const SAT_EMIT_NOT_PHI: &str = "sat_emit_not_phi";
}

/// Default maximum output size for join operations.
/// This prevents memory overflow when joining large tables with high cardinality matches.
pub const DEFAULT_JOIN_MAX_OUTPUT: usize = 1_000_000;

/// Nested-loop join eligibility threshold (Cartesian product
/// upper bound). The dispatcher routes to nested-loop iff
/// `num_left * num_right <= NESTED_LOOP_TOTAL_THRESHOLD`; the
/// provider validates the same invariant fail-closed before any
/// allocation.
///
/// This is the **single source of truth** for the threshold.
/// `xlog-runtime`'s dispatch site imports this constant; do NOT
/// redeclare in xlog-runtime (would create either drift risk or
/// a reverse `xlog-cuda → xlog-runtime` dep cycle).
///
/// Value (`4_000_000`) is grounded in the bench-spike at
/// `bench-spike/w42-nested-loop` HEAD `9c0cefc6` (see
/// `docs/evidence/2026-05-07-w42-bench-spike/README.md`):
/// largest symmetric tested cell `L=R=2000` → 4M total wins by
/// 5.41× over hash; the algorithmic crossover is extrapolated to
/// ~10000×10000 = 100M; 4M leaves 6× margin to absorb
/// production-kernel cost asymmetry. The threshold also caps the
/// index-array allocation at 32 MB total (4M × 4 bytes × 2
/// arrays).
pub const NESTED_LOOP_TOTAL_THRESHOLD: u64 = 4_000_000;

/// Comparison operators for filtering
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum CompareOp {
    Eq = 0,
    Ne = 1,
    Lt = 2,
    Le = 3,
    Gt = 4,
    Ge = 5,
}

/// Join types for hash_join_v2
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum JoinType {
    /// Inner join: return rows where keys match on both sides
    Inner,
    /// Semi join: return left rows that have any match in right (no right columns)
    Semi,
    /// Anti join: return left rows that have NO match in right
    Anti,
    /// Left outer join: return all left rows, with nulls for non-matching right
    LeftOuter,
}

/// Result of packing key columns and computing hashes for join operations
struct PackedKeyData {
    /// Computed hash values (one per row)
    hashes: crate::memory::TrackedCudaSlice<u64>,
    /// Packed key data in row-major format
    packed_keys: crate::memory::TrackedCudaSlice<u8>,
    /// Total bytes per row (key stride)
    key_bytes: u32,
}

struct JoinHashTableV2 {
    bucket_counts: crate::memory::TrackedCudaSlice<u32>,
    bucket_offsets: crate::memory::TrackedCudaSlice<u32>,
    bucket_entries: crate::memory::TrackedCudaSlice<u32>,
    bucket_entry_hashes: crate::memory::TrackedCudaSlice<u64>,
    bucket_mask: u32,
}

/// Bucketed hash table for u64 hashes.
pub struct HashTableU64 {
    pub bucket_counts: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_offsets: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_entries: crate::memory::TrackedCudaSlice<u32>,
    pub bucket_entry_hashes: crate::memory::TrackedCudaSlice<u64>,
    pub bucket_mask: u32,
}

/// Cached build-side join index for v2 hash join.
///
/// This captures the packed key bytes and bucketed hash table layout for the build (right) side,
/// enabling reuse across repeated joins on the same relation + key columns.
pub struct JoinIndexV2 {
    right_num_rows: u32,
    right_keys: Vec<usize>,
    key_bytes: u32,
    packed_keys: crate::memory::TrackedCudaSlice<u8>,
    table: JoinHashTableV2,
}

impl JoinIndexV2 {
    /// Key columns (indices) this index was built for.
    pub fn right_keys(&self) -> &[usize] {
        &self.right_keys
    }

    /// Row count of the build-side buffer at index build time.
    pub fn right_num_rows(&self) -> u32 {
        self.right_num_rows
    }

    /// Approximate device memory used by this cached index.
    pub fn estimated_bytes(&self) -> u64 {
        let mut bytes = 0u64;
        bytes = bytes.saturating_add(self.packed_keys.len() as u64);
        bytes = bytes.saturating_add(self.table.bucket_counts.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_offsets.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_entries.len() as u64 * 4);
        bytes = bytes.saturating_add(self.table.bucket_entry_hashes.len() as u64 * 8);
        bytes
    }
}

/// CUDA kernel provider for xlog GPU operations
///
/// Manages pre-compiled PTX modules for relational operations:
/// - **Join**: Hash join with build/probe phases
/// - **Dedup**: Sort-based deduplication with prefix-sum compaction
/// - **GroupBy**: Sorted-input group aggregation (count, sum, min, max)
///
/// PTX modules are loaded at construction time and stored in the CUDA device.
/// Kernel functions can be retrieved using `device.get_func()`.
///
/// # Example
/// ```ignore
/// use std::sync::Arc;
/// use xlog_cuda::{CudaDevice, GpuMemoryManager, CudaKernelProvider};
/// use xlog_core::MemoryBudget;
///
/// let device = Arc::new(CudaDevice::new(0)?);
/// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
/// let provider = CudaKernelProvider::new(device, memory)?;
/// ```
pub struct CudaKernelProvider {
    /// The CUDA device with loaded PTX modules
    device: Arc<CudaDevice>,
    /// GPU memory manager for kernel allocations
    memory: Arc<GpuMemoryManager>,
    /// Tracked host transfers for diagnostics
    transfer_tracker: HostTransferTracker,
    /// PTX load profiling data (populated only when XLOG_WARMUP_PROFILE=1)
    ptx_load_profile: Option<PtxLoadProfile>,
    /// Column-level D2H transfer counter (incremented by each download_column_* call)
    d2h_transfer_count: AtomicU64,
    /// Untracked control-plane metadata D2H read counter. Incremented by every
    /// `dtoh_scalar_untracked` / `dtoh_small_metadata_untracked` call. These are
    /// bounded metadata reads (row counts, scan totals) exempt from the
    /// data-plane transfer contract, but the GPU-resident MC engine's no-host
    /// gate must prove they are *also* zero inside the measured region — hence an
    /// explicit, resettable counter.
    untracked_metadata_dtoh_count: AtomicU64,
    /// Strict deterministic-Datalog D2H gate. When `true`, any data-plane D2H
    /// transfer (column downloads or `dtoh_sync_copy_into_tracked`) increments
    /// the violation counter and returns `XlogError::Execution` from the
    /// originating call. Metadata reads via `dtoh_scalar_untracked` are NOT
    /// gated. See [`CudaKernelProvider::enable_strict_deterministic_d2h`].
    strict_deterministic_d2h: AtomicBool,
    /// Cumulative count of deterministic-D2H gate violations observed since
    /// the last reset. Increments even on the failing path (the originating
    /// call still returns `Err`); kept for telemetry and tests.
    deterministic_d2h_violations: AtomicU64,
    /// Lazy-initialized non-default launch stream used by
    /// env-gated recorded-operator dispatch (filter, sort,
    /// dedup, GroupBy, hash-join). Cached for the provider's
    /// lifetime — the [`crate::device_runtime::StreamPool`]
    /// never returns streams to a free-list, so per-call
    /// acquire would saturate it. One stream per provider is
    /// sufficient because the recorder serializes work on it;
    /// multiple operations chain through commit-order events.
    recorded_op_stream: OnceLock<crate::device_runtime::StreamId>,
    /// Test/diagnostic-only counter for CSM (count-scan-materialize)
    /// invocations selected by the recorded hash-join dispatch.
    /// **Not part of any public stability guarantee** — its existence,
    /// shape, exposure, and increment semantics may change in any
    /// release. Used by the env-dispatch test suite to prove that CSM
    /// was actually selected for eligible Inner / LeftOuter cases (and
    /// not selected for Semi / Anti or when the env gate is off).
    csm_invocations: AtomicU64,
    /// Diagnostic counter for bounded CSM CUDA Graph captures.
    csm_cuda_graph_captures: AtomicU64,
    /// Diagnostic counter for bounded CSM CUDA Graph launches.
    csm_cuda_graph_launches: AtomicU64,
    /// Diagnostic counter for bounded CSM CUDA Graph ineligibility fallbacks.
    csm_cuda_graph_fallbacks: AtomicU64,
    /// Diagnostic counter for bounded CSM CUDA Graph cache replays.
    csm_cuda_graph_cache_hits: AtomicU64,
    /// Diagnostic counter for graph-mode small full-row set-maintenance
    /// sorts. This is test telemetry only; production correctness must not
    /// depend on the value.
    small_full_row_sort_invocations: AtomicU64,
    /// Bounded CSM CUDA Graph replay cache.
    csm_cuda_graph_cache: Mutex<HashMap<CsmCudaGraphKey, CsmCudaGraphEntry>>,
    /// Per-process counter of WCOJ layout fast-path hits. The
    /// fast-path skips `dedup_full_row_recorded` when the input
    /// is already strictly lex-sorted and full-row unique.
    /// Tests + the phase report binary read this counter to
    /// confirm the fast-path actually fired vs. silently fell
    /// through to the existing dedup pipeline.
    wcoj_layout_fast_path_hit_count: AtomicU64,
    /// Diagnostic counter for generic WCOJ layout-sort helper
    /// invocations. Used by K-clique dispatch-plan certifications to
    /// prove K-clique runtime dispatch no longer routes every edge
    /// through the old all-edge `wcoj_layout_sort_*_recorded` path.
    wcoj_layout_sort_invocation_count: AtomicU64,
    /// Diagnostic counter for K-clique leader-edge metadata builds.
    kclique_metadata_build_count: AtomicU64,
    /// Diagnostic counter for cumulative nanoseconds spent building K-clique
    /// leader-edge metadata.
    kclique_metadata_build_nanos: AtomicU64,
    /// Histogram-guided triangle WCOJ routing counter: successful dispatches
    /// accepted through the block-slice provider entry.
    wcoj_triangle_hg_dispatch_count: AtomicU64,
    /// Diagnostic-only: last WCOJ triangle dispatch's per-phase
    /// CUDA-event timings, populated by `wcoj_triangle_*_recorded`
    /// when the `wcoj-phase-timing` Cargo feature is on. Read by
    /// the `wcoj_phase_report` binary in xlog-integration. Field
    /// is absent when the feature is off, so production builds
    /// have zero overhead.
    #[cfg(feature = "wcoj-phase-timing")]
    last_triangle_phase_timing:
        std::sync::Mutex<Option<crate::wcoj_phase_timing::WcojTrianglePhaseTiming>>,
}

#[derive(Default)]
struct HostTransferTracker {
    dtoh_bytes: AtomicU64,
    htod_bytes: AtomicU64,
    dtoh_calls: AtomicU64,
    htod_calls: AtomicU64,
    launch_metadata_htod_bytes: AtomicU64,
    launch_metadata_htod_calls: AtomicU64,
}

#[derive(Debug, Clone, Copy)]
pub struct HostTransferStats {
    pub dtoh_bytes: u64,
    pub htod_bytes: u64,
    pub dtoh_calls: u64,
    pub htod_calls: u64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct HostLaunchMetadataTransferStats {
    pub htod_bytes: u64,
    pub htod_calls: u64,
}

impl HostTransferTracker {
    fn record_dtoh(&self, bytes: u64) {
        self.dtoh_calls.fetch_add(1, Ordering::Relaxed);
        self.dtoh_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_htod(&self, bytes: u64) {
        self.htod_calls.fetch_add(1, Ordering::Relaxed);
        self.htod_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    fn record_htod_launch_metadata(&self, bytes: u64) {
        self.launch_metadata_htod_calls
            .fetch_add(1, Ordering::Relaxed);
        self.launch_metadata_htod_bytes
            .fetch_add(bytes, Ordering::Relaxed);
    }

    fn snapshot(&self) -> HostTransferStats {
        HostTransferStats {
            dtoh_bytes: self.dtoh_bytes.load(Ordering::Relaxed),
            htod_bytes: self.htod_bytes.load(Ordering::Relaxed),
            dtoh_calls: self.dtoh_calls.load(Ordering::Relaxed),
            htod_calls: self.htod_calls.load(Ordering::Relaxed),
        }
    }

    fn launch_metadata_snapshot(&self) -> HostLaunchMetadataTransferStats {
        HostLaunchMetadataTransferStats {
            htod_bytes: self.launch_metadata_htod_bytes.load(Ordering::Relaxed),
            htod_calls: self.launch_metadata_htod_calls.load(Ordering::Relaxed),
        }
    }

    fn reset(&self) {
        self.dtoh_bytes.store(0, Ordering::Relaxed);
        self.htod_bytes.store(0, Ordering::Relaxed);
        self.dtoh_calls.store(0, Ordering::Relaxed);
        self.htod_calls.store(0, Ordering::Relaxed);
        self.launch_metadata_htod_bytes.store(0, Ordering::Relaxed);
        self.launch_metadata_htod_calls.store(0, Ordering::Relaxed);
    }
}

impl CudaKernelProvider {
    /// Create a new CUDA kernel provider
    ///
    /// Loads all kernel modules into the CUDA device.
    /// Prefers cubin for the detected SM arch, falls back to portable PTX (sm_75+).
    ///
    /// # Arguments
    /// * `device` - The CUDA device to load modules into
    /// * `memory` - The GPU memory manager for kernel allocations
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if PTX loading fails
    ///
    /// # Example
    /// ```ignore
    /// let device = Arc::new(CudaDevice::new(0)?);
    /// let memory = Arc::new(GpuMemoryManager::new(device.clone(), MemoryBudget::default()));
    /// let provider = CudaKernelProvider::new(device, memory)?;
    /// ```
    pub fn new(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
        let profiling = warmup_profiling_enabled();
        let ptx_load_profile = Self::load_all_kernel_modules(&device, profiling)?;

        Ok(Self {
            device,
            memory,
            transfer_tracker: HostTransferTracker::default(),
            ptx_load_profile,
            d2h_transfer_count: AtomicU64::new(0),
            untracked_metadata_dtoh_count: AtomicU64::new(0),
            strict_deterministic_d2h: AtomicBool::new(false),
            deterministic_d2h_violations: AtomicU64::new(0),
            recorded_op_stream: OnceLock::new(),
            csm_invocations: AtomicU64::new(0),
            csm_cuda_graph_captures: AtomicU64::new(0),
            csm_cuda_graph_launches: AtomicU64::new(0),
            csm_cuda_graph_fallbacks: AtomicU64::new(0),
            csm_cuda_graph_cache_hits: AtomicU64::new(0),
            small_full_row_sort_invocations: AtomicU64::new(0),
            csm_cuda_graph_cache: Mutex::new(HashMap::new()),
            wcoj_layout_fast_path_hit_count: AtomicU64::new(0),
            wcoj_layout_sort_invocation_count: AtomicU64::new(0),
            kclique_metadata_build_count: AtomicU64::new(0),
            kclique_metadata_build_nanos: AtomicU64::new(0),
            wcoj_triangle_hg_dispatch_count: AtomicU64::new(0),
            #[cfg(feature = "wcoj-phase-timing")]
            last_triangle_phase_timing: std::sync::Mutex::new(None),
        })
    }

    /// Construct a provider whose `GpuMemoryManager` must already
    /// have a v0.6 [`crate::device_runtime::XlogDeviceRuntime`]
    /// attached via [`GpuMemoryManager::with_runtime`].
    ///
    /// Equivalent to [`Self::new`] in every respect — same kernel
    /// loading, same field initialization — but **rejects** managers
    /// that lack a runtime. This guards against the misconfiguration
    /// in which a caller asks for runtime-routed provider semantics
    /// (by calling `with_runtime`) but supplies a legacy manager
    /// built via [`GpuMemoryManager::new`]; without the check, the
    /// resulting provider would silently keep using the cudarc
    /// default allocator and the runtime budget/logging stack would
    /// never observe the allocations the caller expected to be
    /// routed through it.
    ///
    /// Note: a runtime-routed manager passed to [`Self::new`] still
    /// routes correctly — `alloc::<T>` and `alloc_raw` consult
    /// `memory.runtime()` regardless of which provider constructor
    /// was used. `with_runtime` exists for callers that want the
    /// requirement enforced at construction time, not for
    /// correctness of the routing itself.
    ///
    /// This is the **opt-in** runtime entry point for providers.
    /// `Self::new` continues to accept managers without a runtime
    /// (the legacy default) and remains the production constructor
    /// until the runtime stack is certified end-to-end.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if `memory.runtime()` is `None`,
    /// or anything `Self::new` would return.
    ///
    /// # Example
    /// ```ignore
    /// let device = Arc::new(CudaDevice::new(0)?);
    /// let runtime = Arc::new(XlogDeviceRuntime::with_resource(
    ///     Arc::clone(&device),
    ///     0,
    ///     Arc::new(StreamPool::with_defaults(Arc::clone(&device))),
    ///     Box::new(AsyncCudaResource::new(/* ... */)),
    /// ));
    /// let memory = Arc::new(GpuMemoryManager::with_runtime(
    ///     Arc::clone(&device),
    ///     MemoryBudget::default(),
    ///     runtime,
    /// ));
    /// let provider = CudaKernelProvider::with_runtime(device, memory)?;
    /// ```
    pub fn with_runtime(device: Arc<CudaDevice>, memory: Arc<GpuMemoryManager>) -> Result<Self> {
        if memory.runtime().is_none() {
            return Err(XlogError::Kernel(
                "CudaKernelProvider::with_runtime requires a GpuMemoryManager built via \
                 GpuMemoryManager::with_runtime; got a manager with no runtime attached"
                    .to_string(),
            ));
        }
        Self::new(device, memory)
    }

    /// Internal: parse a "boolean" env var. Empty / unset / `"0"`
    /// → false; any other value → true.
    fn env_flag(name: &str) -> bool {
        std::env::var(name)
            .map(|v| !v.is_empty() && v != "0")
            .unwrap_or(false)
    }

    /// Whether the recorded filter dispatch is enabled via env.
    ///
    /// Returns `true` when either `XLOG_USE_RECORDED_FILTERS` or
    /// the umbrella `XLOG_USE_RECORDED_OPS` env var is set.
    /// Combined with a runtime-backed manager, this routes
    /// `filter::<T>` through the recorded launch path.
    ///
    /// Env-gated rather than default-on so the migration is
    /// opt-in for real callers; the existing legacy paths remain
    /// the production default until the runtime stack is
    /// certified end-to-end.
    pub(crate) fn use_recorded_filters_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_FILTERS") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the recorded sort dispatch is enabled via env.
    /// Reads `XLOG_USE_RECORDED_SORT` or the umbrella
    /// `XLOG_USE_RECORDED_OPS`. The recorded-sort path is narrowed
    /// to U32 / Symbol keys only — the public
    /// `sort()` dispatcher checks both this env flag AND key
    /// type compatibility before routing.
    pub(crate) fn use_recorded_sort_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_SORT") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the recorded full-row dedup dispatch is enabled
    /// via env. Reads `XLOG_USE_RECORDED_DEDUP` or the umbrella
    /// `XLOG_USE_RECORDED_OPS`. `dedup_full_row_recorded` is
    /// narrow to all-U32 / Symbol columns.
    pub(crate) fn use_recorded_dedup_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_DEDUP") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the recorded GroupBy dispatch is enabled via
    /// env. Reads `XLOG_USE_RECORDED_GROUPBY` or
    /// `XLOG_USE_RECORDED_OPS`. `groupby_multi_agg_recorded`
    /// supports U32 / Symbol keys + Count / Sum / Min / Max
    /// aggs only.
    pub(crate) fn use_recorded_groupby_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_GROUPBY") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the recorded hash-join dispatch is enabled via
    /// env. Reads `XLOG_USE_RECORDED_HASH_JOIN` or
    /// `XLOG_USE_RECORDED_OPS`. `hash_join_v2_recorded` and
    /// `hash_join_v2_with_index_recorded` cover all four join
    /// types (Inner / Semi / Anti / LeftOuter); the only
    /// hard constraint inherited from `pack_keys` is `≤4`
    /// key columns.
    pub(crate) fn use_recorded_hash_join_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_HASH_JOIN") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the recorded CSM (count-scan-materialize)
    /// dispatch is enabled via env. Reads `XLOG_USE_RECORDED_CSM`
    /// or `XLOG_USE_RECORDED_OPS`. CSM is a sub-strategy of the
    /// recorded hash-join: it is consulted only after the
    /// recorded path has already been selected, and only for
    /// `JoinType::Inner` / `JoinType::LeftOuter` where a CSM
    /// implementation exists. `Semi` / `Anti` are not affected.
    pub(crate) fn use_recorded_csm_env() -> bool {
        Self::env_flag("XLOG_USE_RECORDED_CSM") || Self::env_flag("XLOG_USE_RECORDED_OPS")
    }

    /// Whether the bounded CSM CUDA Graph path is enabled.
    ///
    /// This is narrower than `XLOG_USE_RECORDED_CSM`: callers must first select
    /// the recorded CSM hash-join path, then opt into graph capture/replay with
    /// `XLOG_USE_CSM_CUDA_GRAPH=1` (or the broader `XLOG_USE_CUDA_GRAPHS=1`).
    pub(crate) fn use_csm_cuda_graph_env() -> bool {
        Self::env_flag("XLOG_USE_CSM_CUDA_GRAPH") || Self::env_flag("XLOG_USE_CUDA_GRAPHS")
    }

    /// Test/diagnostic-only telemetry: number of times the recorded
    /// hash-join dispatch routed through a CSM (count-scan-materialize)
    /// method since this provider was created. Increments once per
    /// dispatched call across all four CSM methods (Inner / LeftOuter,
    /// non-indexed / indexed). Used by `test_csm_env_dispatch` to
    /// prove dispatch selection.
    ///
    /// **Not part of any public stability guarantee.** Hidden from
    /// rustdoc with `#[doc(hidden)]` so it does not appear in
    /// generated API docs; the symbol remains callable from
    /// integration tests within this crate but production callers
    /// must not depend on it. May be renamed, gated behind a cargo
    /// feature, or withdrawn in any release without notice.
    #[doc(hidden)]
    pub fn csm_invocations(&self) -> u64 {
        self.csm_invocations.load(Ordering::Relaxed)
    }

    #[doc(hidden)]
    pub fn csm_cuda_graph_captures(&self) -> u64 {
        self.csm_cuda_graph_captures.load(Ordering::Relaxed)
    }

    #[doc(hidden)]
    pub fn csm_cuda_graph_launches(&self) -> u64 {
        self.csm_cuda_graph_launches.load(Ordering::Relaxed)
    }

    #[doc(hidden)]
    pub fn csm_cuda_graph_fallbacks(&self) -> u64 {
        self.csm_cuda_graph_fallbacks.load(Ordering::Relaxed)
    }

    #[doc(hidden)]
    pub fn csm_cuda_graph_cache_hits(&self) -> u64 {
        self.csm_cuda_graph_cache_hits.load(Ordering::Relaxed)
    }

    #[doc(hidden)]
    pub fn small_full_row_sort_invocations(&self) -> u64 {
        self.small_full_row_sort_invocations.load(Ordering::Relaxed)
    }

    /// Lazily acquire one non-default launch stream from the
    /// runtime's [`crate::device_runtime::StreamPool`] for
    /// recorded-operator dispatch, and cache it for this
    /// provider's lifetime. Shared across all env-gated
    /// recorded paths (filter, sort, dedup, GroupBy,
    /// hash-join) — a single stream is sufficient because the
    /// recorder serializes work on it; multiple operations
    /// chain naturally through commit-order events.
    ///
    /// Returns `None` when:
    ///   * the manager has no runtime attached
    ///     (`memory.runtime() == None`), or
    ///   * the stream pool is at capacity and `acquire` fails.
    ///
    /// On a lost race during first init the loser leaks one
    /// stream (the pool keeps it alive); both winners cache
    /// the same `StreamId`. Acceptable cost — practical pool
    /// sizes are large compared to the number of providers
    /// per process.
    pub(crate) fn recorded_op_stream_or_init(&self) -> Option<crate::device_runtime::StreamId> {
        if let Some(s) = self.recorded_op_stream.get() {
            return Some(*s);
        }
        let runtime = self.memory.runtime()?;
        let stream = runtime.stream_pool().acquire().ok()?;
        let _ = self.recorded_op_stream.set(stream);
        self.recorded_op_stream.get().copied()
    }

    /// Take the per-phase WCOJ triangle dispatch timings recorded
    /// by the most recent `wcoj_triangle_*_recorded` call. Reading
    /// clears the slot — designed for one-shot consumption by the
    /// `wcoj_phase_report` binary in xlog-integration. Returns
    /// `None` if no triangle dispatch has fired since the last
    /// read (or since construction).
    ///
    /// Compiled in only with the `wcoj-phase-timing` Cargo
    /// feature; production builds have no such method.
    #[cfg(feature = "wcoj-phase-timing")]
    pub fn take_wcoj_triangle_phase_timing(
        &self,
    ) -> Option<crate::wcoj_phase_timing::WcojTrianglePhaseTiming> {
        self.last_triangle_phase_timing
            .lock()
            .ok()
            .and_then(|mut g| g.take())
    }

    /// Internal: store the phase timings produced by a triangle
    /// dispatch. Overwrites any prior unread slot — the report
    /// binary is expected to read after every `execute_plan`.
    #[cfg(feature = "wcoj-phase-timing")]
    #[allow(dead_code)]
    pub(crate) fn put_wcoj_triangle_phase_timing(
        &self,
        timing: crate::wcoj_phase_timing::WcojTrianglePhaseTiming,
    ) {
        if let Ok(mut g) = self.last_triangle_phase_timing.lock() {
            *g = Some(timing);
        }
    }

    /// Number of times `wcoj_layout_*_recorded` short-circuited
    /// to the fast-path (recorded clone) instead of running
    /// `dedup_full_row_recorded`. Increments by 1 per
    /// fast-path hit (3 hits per dispatch when all inputs are
    /// already sorted+unique). Used by tests + the phase
    /// report to confirm the fast-path fired.
    pub fn wcoj_layout_fast_path_hit_count(&self) -> u64 {
        self.wcoj_layout_fast_path_hit_count.load(Ordering::Relaxed)
    }

    /// Histogram-guided block-slice triangle WCOJ test/diagnostic counter:
    /// successful dispatches that routed through the provider entry.
    pub fn wcoj_triangle_hg_dispatch_count(&self) -> u64 {
        self.wcoj_triangle_hg_dispatch_count.load(Ordering::Relaxed)
    }

    /// Reset the fast-path hit counter to 0. Tests use this to
    /// scope counter assertions to a single dispatch.
    pub fn reset_wcoj_layout_fast_path_hit_count(&self) {
        self.wcoj_layout_fast_path_hit_count
            .store(0, Ordering::Relaxed);
    }

    /// Number of calls to `wcoj_layout_sort_*_recorded` since the
    /// last reset. Diagnostic-only; used by dispatch-plan certification.
    pub fn wcoj_layout_sort_invocation_count(&self) -> u64 {
        self.wcoj_layout_sort_invocation_count
            .load(Ordering::Relaxed)
    }

    /// Reset the WCOJ layout-sort invocation counter to 0.
    pub fn reset_wcoj_layout_sort_invocation_count(&self) {
        self.wcoj_layout_sort_invocation_count
            .store(0, Ordering::Relaxed);
    }

    /// Number of K-clique leader-edge metadata builds since the
    /// last reset.
    pub fn kclique_metadata_build_count(&self) -> u64 {
        self.kclique_metadata_build_count.load(Ordering::Relaxed)
    }

    /// Cumulative nanoseconds spent building K-clique leader-edge
    /// metadata since the last reset.
    pub fn kclique_metadata_build_nanos(&self) -> u64 {
        self.kclique_metadata_build_nanos.load(Ordering::Relaxed)
    }

    /// Reset K-clique metadata build diagnostics.
    pub fn reset_kclique_metadata_build_metrics(&self) {
        self.kclique_metadata_build_count
            .store(0, Ordering::Relaxed);
        self.kclique_metadata_build_nanos
            .store(0, Ordering::Relaxed);
    }

    /// Internal: increment the fast-path counter. Called by
    /// `wcoj_layout_*_recorded` after a successful fast-path
    /// branch. Not part of any public stability guarantee.
    pub(crate) fn record_wcoj_layout_fast_path_hit(&self) {
        self.wcoj_layout_fast_path_hit_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Internal: increment the generic WCOJ layout-sort counter.
    pub(crate) fn record_wcoj_layout_sort_invocation(&self) {
        self.wcoj_layout_sort_invocation_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Internal: record a K-clique leader-edge metadata build.
    pub(crate) fn record_kclique_metadata_build_nanos(&self, nanos: u128) {
        self.kclique_metadata_build_count
            .fetch_add(1, Ordering::Relaxed);
        let nanos = u64::try_from(nanos).unwrap_or(u64::MAX);
        self.kclique_metadata_build_nanos
            .fetch_add(nanos, Ordering::Relaxed);
    }

    /// Runtime hook: record a successful histogram-guided block-slice triangle
    /// dispatch.
    #[doc(hidden)]
    pub fn record_wcoj_triangle_hg_dispatch(&self) {
        self.wcoj_triangle_hg_dispatch_count
            .fetch_add(1, Ordering::Relaxed);
    }

    /// Get the CUDA device
    pub fn device(&self) -> &Arc<CudaDevice> {
        &self.device
    }

    /// Get the GPU memory manager
    pub fn memory(&self) -> &Arc<GpuMemoryManager> {
        &self.memory
    }

    /// Get PTX load profiling data (only populated when XLOG_WARMUP_PROFILE=1).
    pub fn ptx_load_profile(&self) -> Option<&PtxLoadProfile> {
        self.ptx_load_profile.as_ref()
    }

    /// Reset tracked host transfer statistics.
    pub fn reset_host_transfer_stats(&self) {
        self.transfer_tracker.reset();
    }

    /// Snapshot tracked host transfer statistics.
    pub fn host_transfer_stats(&self) -> HostTransferStats {
        self.transfer_tracker.snapshot()
    }

    /// Snapshot launch-parameter H2D uploads tracked separately from
    /// `host_transfer_stats`.
    pub fn host_launch_metadata_transfer_stats(&self) -> HostLaunchMetadataTransferStats {
        self.transfer_tracker.launch_metadata_snapshot()
    }

    /// Read the column-level D2H transfer counter.
    ///
    /// This counter increments once per `download_column_*` call, enabling
    /// callers (e.g. the ILP trainer) to assert that no column downloads
    /// occurred during a performance-critical section.
    pub fn d2h_transfer_count(&self) -> u64 {
        self.d2h_transfer_count.load(Ordering::Relaxed)
    }

    /// Reset the column-level D2H transfer counter to zero.
    pub fn reset_d2h_transfer_count(&self) {
        self.d2h_transfer_count.store(0, Ordering::Relaxed);
    }

    /// Count of untracked control-plane metadata D2H reads
    /// (`dtoh_scalar_untracked` + `dtoh_small_metadata_untracked`).
    pub fn untracked_metadata_dtoh_count(&self) -> u64 {
        self.untracked_metadata_dtoh_count.load(Ordering::Relaxed)
    }

    /// Reset the untracked metadata D2H read counter to zero.
    pub fn reset_untracked_metadata_dtoh_count(&self) {
        self.untracked_metadata_dtoh_count
            .store(0, Ordering::Relaxed);
    }

    /// Enable the strict deterministic-Datalog D2H gate.
    ///
    /// While enabled, any data-plane device-to-host transfer (column downloads
    /// via `download_column` / `download_column_untracked`, and any internal
    /// transfer routed through `dtoh_sync_copy_into_tracked`) increments
    /// [`CudaKernelProvider::deterministic_d2h_violation_count`] and returns
    /// `XlogError::Execution` from the originating call.
    ///
    /// Metadata reads via [`CudaKernelProvider::dtoh_scalar_untracked`] are
    /// allowed and never trip the gate.
    ///
    /// Default is `false`; the runtime opts in via
    /// `RuntimeConfig::strict_deterministic_d2h`. v0.5.5 ships the gate
    /// opt-in only — known-violating relational paths (set difference,
    /// join count/materialize) are scheduled for replacement before the
    /// default flips.
    pub fn enable_strict_deterministic_d2h(&self) {
        self.strict_deterministic_d2h.store(true, Ordering::Relaxed);
    }

    /// Disable the strict deterministic-Datalog D2H gate.
    pub fn disable_strict_deterministic_d2h(&self) {
        self.strict_deterministic_d2h
            .store(false, Ordering::Relaxed);
    }

    /// Returns whether the strict deterministic-Datalog D2H gate is enabled.
    pub fn strict_deterministic_d2h_enabled(&self) -> bool {
        self.strict_deterministic_d2h.load(Ordering::Relaxed)
    }

    /// Cumulative deterministic-D2H gate violations since the last reset.
    pub fn deterministic_d2h_violation_count(&self) -> u64 {
        self.deterministic_d2h_violations.load(Ordering::Relaxed)
    }

    /// Reset the deterministic-D2H violation counter to zero.
    pub fn reset_deterministic_d2h_violations(&self) {
        self.deterministic_d2h_violations
            .store(0, Ordering::Relaxed);
    }

    /// Chokepoint for the deterministic-D2H gate.
    ///
    /// If the gate is enabled, increments the violation counter and returns
    /// `XlogError::Execution` naming the offending operation and byte count.
    /// If the gate is disabled, returns `Ok(())` cheaply.
    pub(crate) fn check_deterministic_d2h(&self, op: &'static str, bytes: u64) -> Result<()> {
        if self.strict_deterministic_d2h.load(Ordering::Relaxed) {
            self.deterministic_d2h_violations
                .fetch_add(1, Ordering::Relaxed);
            return Err(XlogError::Execution(format!(
                "deterministic D2H gate: {} attempted to copy {} bytes from device to host",
                op, bytes
            )));
        }
        Ok(())
    }

    fn dtoh_sync_copy_into_tracked<T: DeviceRepr, Src: DevicePtr<T>>(
        &self,
        src: &Src,
        dst: &mut [T],
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(dst.len())
            .ok_or_else(|| XlogError::Kernel("dtoh size overflow".to_string()))?;
        self.check_deterministic_d2h("dtoh_sync_copy_into_tracked", bytes as u64)?;
        self.transfer_tracker.record_dtoh(bytes as u64);
        self.device
            .inner()
            .dtoh_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy from device: {}", e)))
    }

    /// Hard cap (in bytes) for [`Self::dtoh_small_metadata_untracked`].
    /// Set deliberately small (4 KB) so the helper cannot become a
    /// general-purpose vector D2H escape hatch — it's strictly for
    /// classifier histograms and similar small metadata round-trips.
    pub const DTOH_SMALL_METADATA_MAX_BYTES: usize = 4096;

    /// Read a small metadata vector (≤ [`Self::DTOH_SMALL_METADATA_MAX_BYTES`])
    /// from device to host WITHOUT updating the D2H transfer tracker.
    ///
    /// Sibling of [`Self::dtoh_scalar_untracked`] for callers that need
    /// a few bucket counts (the WCOJ skew classifier reads a 3 × 64 ×
    /// `u32` = 768-byte histogram in one go) instead of `count` separate
    /// scalar reads. Like `dtoh_scalar_untracked`, this method is
    /// whitelisted by the strict deterministic-D2H gate
    /// ([`Self::enable_strict_deterministic_d2h`]) — it does NOT trip
    /// the gate, on purpose, because metadata reads are part of the
    /// determinism contract (just like a scalar `total` after a scan).
    ///
    /// # Hard contract — DO NOT WIDEN THE CAP
    /// The 4 KB cap is the contract. If a caller wants a larger D2H,
    /// it's a data-plane transfer and must go through the tracked
    /// `download_column*` path. Widening this cap turns the helper
    /// into a backdoor for tracked-bypass column reads, which would
    /// silently invalidate the strict deterministic-D2H gate.
    ///
    /// # Errors
    ///   * `XlogError::Kernel` if `count * size_of::<T>()` exceeds
    ///     `DTOH_SMALL_METADATA_MAX_BYTES`.
    ///   * `XlogError::Kernel` if `count` exceeds the device slice's
    ///     length, or if the inner sync copy fails.
    pub fn dtoh_small_metadata_untracked<T: DeviceRepr + Default + Copy>(
        &self,
        src: &crate::memory::TrackedCudaSlice<T>,
        count: usize,
    ) -> Result<Vec<T>> {
        let bytes = count.checked_mul(std::mem::size_of::<T>()).ok_or_else(|| {
            XlogError::Kernel("dtoh_small_metadata_untracked: byte size overflow".to_string())
        })?;
        if bytes > Self::DTOH_SMALL_METADATA_MAX_BYTES {
            return Err(XlogError::Kernel(format!(
                "dtoh_small_metadata_untracked: requested {} bytes exceeds metadata cap of {} bytes \
                 (this is metadata-only; use download_column* for data-plane transfers)",
                bytes,
                Self::DTOH_SMALL_METADATA_MAX_BYTES
            )));
        }
        if count > src.len() {
            return Err(XlogError::Kernel(format!(
                "dtoh_small_metadata_untracked: count={count} > src.len={}",
                src.len()
            )));
        }
        if count == 0 {
            return Ok(Vec::new());
        }
        let slice = src.try_slice(0..count).ok_or_else(|| {
            XlogError::Kernel(format!(
                "dtoh_small_metadata_untracked: try_slice(0..{count}) failed"
            ))
        })?;
        let mut buf: Vec<T> = vec![T::default(); count];
        self.untracked_metadata_dtoh_count
            .fetch_add(1, Ordering::Relaxed);
        self.device
            .inner()
            .dtoh_sync_copy_into(&slice, &mut buf)
            .map_err(|e| {
                XlogError::Kernel(format!("dtoh_small_metadata_untracked: copy failed: {}", e))
            })?;
        Ok(buf)
    }

    /// Read a single scalar from device to host WITHOUT updating the
    /// D2H transfer tracker. Use ONLY for metadata reads (e.g. total_nnz
    /// after an exclusive scan), never for data-plane transfers.
    ///
    /// This makes the "metadata != data-plane" contract explicit and
    /// auditable: callers that bypass tracking must call this method
    /// (which is grep-able) rather than reaching for device().inner().
    pub fn dtoh_scalar_untracked<T: DeviceRepr + Default + Copy>(
        &self,
        src: &crate::memory::TrackedCudaSlice<T>,
        index: usize,
    ) -> Result<T> {
        if index >= src.len() {
            return Err(XlogError::Kernel(format!(
                "dtoh_scalar_untracked: index={} >= len={}",
                index,
                src.len()
            )));
        }
        let slice = src.try_slice(index..index + 1).ok_or_else(|| {
            XlogError::Kernel(format!(
                "dtoh_scalar_untracked: slice failed at index={}",
                index
            ))
        })?;
        let mut buf = [T::default()];
        self.untracked_metadata_dtoh_count
            .fetch_add(1, Ordering::Relaxed);
        self.device
            .inner()
            .dtoh_sync_copy_into(&slice, &mut buf)
            .map_err(|e| XlogError::Kernel(format!("dtoh_scalar_untracked: copy failed: {}", e)))?;
        Ok(buf[0])
    }

    /// Upload host data to device while recording data-plane H2D transfer stats.
    pub fn htod_sync_copy_into_tracked<T: DeviceRepr, Dst: cudarc::driver::DevicePtrMut<T>>(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("htod size overflow".to_string()))?;
        self.transfer_tracker.record_htod(bytes as u64);
        self.device
            .inner()
            .htod_sync_copy_into(src, dst)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy to device: {}", e)))
    }

    /// Allocate a CUDA slice from host data while recording data-plane H2D
    /// transfer stats.
    pub fn htod_sync_copy_tracked<T: DeviceRepr>(
        &self,
        src: &[T],
    ) -> Result<cudarc::driver::CudaSlice<T>> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("htod size overflow".to_string()))?;
        self.transfer_tracker.record_htod(bytes as u64);
        self.device
            .inner()
            .htod_sync_copy(src)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy to device: {}", e)))
    }

    /// Upload bounded launch metadata from host to device while recording it in
    /// the launch-metadata subcounter.
    pub fn htod_launch_metadata_sync_copy_into<
        T: DeviceRepr,
        Dst: cudarc::driver::DevicePtrMut<T>,
    >(
        &self,
        src: &[T],
        dst: &mut Dst,
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>()
            .checked_mul(src.len())
            .ok_or_else(|| XlogError::Kernel("launch metadata htod size overflow".to_string()))?;
        self.transfer_tracker
            .record_htod_launch_metadata(bytes as u64);
        self.device
            .inner()
            .htod_sync_copy_into(src, dst)
            .map_err(|e| {
                XlogError::Kernel(format!("Failed to copy launch metadata to device: {}", e))
            })
    }

    /// Upload one launch-metadata scalar to device on a caller-owned stream
    /// while recording the transfer in the launch-metadata H2D counters.
    pub(crate) fn htod_launch_metadata_async_copy_one<T: DeviceRepr>(
        &self,
        src: &T,
        dst: &TrackedCudaSlice<T>,
        stream: &CudaStream,
        context: &str,
    ) -> Result<()> {
        let bytes = std::mem::size_of::<T>();
        self.transfer_tracker
            .record_htod_launch_metadata(bytes as u64);
        unsafe {
            let res = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                *dst.device_ptr(),
                src as *const T as *const c_void,
                bytes,
                stream.cu_stream(),
            );
            if res != cudarc::driver::sys::cudaError_enum::CUDA_SUCCESS {
                return Err(XlogError::Kernel(format!(
                    "{context}: launch metadata H2D failed: {res:?}"
                )));
            }
        }
        Ok(())
    }

    /// Compute exclusive prefix sum of u8 mask, returns (prefix_sum_vec, total_count)
    ///
    /// This is useful for compaction operations where we need to know:
    /// 1. The output position for each input element (prefix sum)
    /// 2. The total number of elements that pass the mask (count)
    ///
    /// # Arguments
    /// * `mask` - A slice of u8 values (0 or non-zero)
    ///
    /// # Returns
    /// A tuple of:
    /// - `Vec<u32>` containing the exclusive prefix sum
    /// - `u32` containing the total count of non-zero mask elements
    ///
    /// # Example
    /// ```ignore
    /// let mask = vec![1u8, 0, 1, 1, 0, 1];
    /// let (prefix_sum, count) = provider.prefix_sum_mask(&mask)?;
    /// // prefix_sum = [0, 1, 1, 2, 3, 3]
    /// // count = 4
    /// ```
    ///
    /// # Note
    /// For small inputs (<=256 elements), a CPU scan is used for efficiency.
    /// For larger inputs, a three-phase multi-block GPU scan is used.
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if kernel execution fails
    pub fn exclusive_scan_u32_inplace(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n as usize > data.len() {
            return Err(XlogError::Kernel(format!(
                "exclusive_scan_u32_inplace: n={} exceeds slice len={}",
                n,
                data.len()
            )));
        }
        self.multiblock_scan_u32_inplace(data, n)
    }

    fn multiblock_scan_u32_inplace(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;

            // SAFETY: multiblock_scan_phase2(uint32_t* block_sums, uint32_t num_blocks)
            unsafe {
                phase2_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut *data, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;

            self.device.synchronize()?;
            return Ok(());
        }

        let num_blocks = n.div_ceil(block_size);
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_u32_phase1(uint32_t* data, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_u32_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;
        self.device.synchronize()?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut block_sums, num_blocks)?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase3(uint32_t* prefix_sum, const uint32_t* block_offsets, uint32_t n)
        unsafe {
            phase3_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

        self.device.synchronize()?;
        Ok(())
    }

    /// Stream-aware variant of [`Self::multiblock_scan_u32_inplace`].
    ///
    /// Runs every kernel of the recursive scan on `cu_stream`
    /// (no `device.synchronize()`), and records each intermediate
    /// `block_sums` allocation against the runtime so that when
    /// the helper returns and the local drops, the runtime's
    /// deallocate can queue `cuStreamWaitEvent(alloc_stream,
    /// recorded_event)` BEFORE `cuMemFreeAsync` — the same
    /// cross-stream lifetime safety the LaunchRecorder gives
    /// caller-provided buffers.
    ///
    /// `data` is not recorded here: the caller already records
    /// its own write of `data` against the same launch_stream
    /// (typically via `LaunchRecorder::write` BEFORE preflight).
    pub(crate) fn multiblock_scan_u32_inplace_on_stream(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: crate::device_runtime::StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;
            // SAFETY: kernel signature matches; data is mutated in place.
            unsafe {
                phase2_fn.clone().launch_on_stream(
                    cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut *data, n),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!("multiblock_scan_phase2 (on_stream) failed: {}", e))
            })?;
            return Ok(());
        }

        let num_blocks = n.div_ceil(block_size);
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;
        // Fence alloc-ready → launch_stream for block_sums
        // before phase1 kernel writes it. The alloc was queued
        // on the manager's default stream; without this wait,
        // a launch_stream-queued kernel can begin before
        // cuMemAllocAsync completes and read pool-recycled
        // bytes when the streams differ.
        runtime
            .prepare_first_use(
                &block_sums,
                launch_stream,
                crate::device_runtime::Access::Write,
            )
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "multiblock_scan_u32_inplace_on_stream: prepare block_sums failed: {}",
                    e
                ))
            })?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;
        // SAFETY: kernel signature matches.
        unsafe {
            phase1_u32_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "multiblock_scan_u32_phase1 (on_stream) failed: {}",
                e
            ))
        })?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace_on_stream(
                &mut block_sums,
                num_blocks,
                cu_stream,
                launch_stream,
                runtime,
            )?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;
        // SAFETY: kernel signature matches.
        unsafe {
            phase3_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!("multiblock_scan_phase3 (on_stream) failed: {}", e))
        })?;

        // Record `block_sums` use on `launch_stream` BEFORE it
        // drops at end-of-scope. Without this, the runtime's
        // deallocate would queue `cuMemFreeAsync` on alloc_stream
        // without waiting for the launch_stream chain that's
        // still reading/writing block_sums to complete.
        if let Some(b) = block_sums.runtime_block() {
            runtime
                .finish_block_use(
                    crate::device_runtime::BlockId::from_block(b),
                    launch_stream,
                    crate::device_runtime::Access::Write,
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "multiblock_scan_u32_inplace_on_stream: finish_block_use \
                         for intermediate block_sums failed: {}",
                        e
                    ))
                })?;
        } else {
            return Err(XlogError::Kernel(
                "multiblock_scan_u32_inplace_on_stream: intermediate block_sums has no \
                 runtime block — caller must use a runtime-backed manager"
                    .to_string(),
            ));
        }
        Ok(())
    }

    /// Allocate every recursive `block_sums` buffer needed by
    /// [`Self::multiblock_scan_u32_inplace_on_stream_with_scratch`].
    pub(crate) fn multiblock_scan_u32_scratch_for_len(
        &self,
        mut n: u32,
    ) -> Result<MultiblockScanScratchU32> {
        let block_size = 256u32;
        let mut levels = Vec::new();
        while n > block_size {
            let num_blocks = n.div_ceil(block_size);
            levels.push(self.memory.alloc::<u32>(num_blocks as usize)?);
            n = num_blocks;
        }
        Ok(MultiblockScanScratchU32 { levels })
    }

    /// Stream-aware u32 scan with caller-owned scratch.
    ///
    /// This is the CUDA Graph compatible counterpart to
    /// [`Self::multiblock_scan_u32_inplace_on_stream`]: all scratch buffers are
    /// supplied by the caller, so graph capture sees a stable scan topology and
    /// stable intermediate addresses.
    pub(crate) fn multiblock_scan_u32_inplace_on_stream_with_scratch(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
        cu_stream: &cudarc::driver::CudaStream,
        scratch: &mut MultiblockScanScratchU32,
    ) -> Result<()> {
        self.multiblock_scan_u32_inplace_on_stream_with_scratch_levels(
            data,
            n,
            cu_stream,
            &mut scratch.levels,
        )
    }

    fn multiblock_scan_u32_inplace_on_stream_with_scratch_levels(
        &self,
        data: &mut crate::memory::TrackedCudaSlice<u32>,
        n: u32,
        cu_stream: &cudarc::driver::CudaStream,
        scratch_levels: &mut [TrackedCudaSlice<u32>],
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;
            // SAFETY: kernel signature matches; data is mutated in place.
            unsafe {
                phase2_fn.clone().launch_on_stream(
                    cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut *data, n),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "multiblock_scan_phase2 (graph scratch) failed: {}",
                    e
                ))
            })?;
            return Ok(());
        }

        let num_blocks = n.div_ceil(block_size);
        let (block_sums, rest) = scratch_levels.split_first_mut().ok_or_else(|| {
            XlogError::Kernel(format!(
                "multiblock_scan_u32_inplace_on_stream_with_scratch: missing scratch level \
                 for n={n}, num_blocks={num_blocks}"
            ))
        })?;
        if block_sums.len() < num_blocks as usize {
            return Err(XlogError::Kernel(format!(
                "multiblock_scan_u32_inplace_on_stream_with_scratch: scratch level too small \
                 (have {}, need {})",
                block_sums.len(),
                num_blocks
            )));
        }

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;
        // SAFETY: kernel signature matches.
        unsafe {
            phase1_u32_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut *block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "multiblock_scan_u32_phase1 (graph scratch) failed: {}",
                e
            ))
        })?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace_on_stream_with_scratch_levels(
                block_sums, num_blocks, cu_stream, rest,
            )?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;
        // SAFETY: kernel signature matches.
        unsafe {
            phase3_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &*block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "multiblock_scan_phase3 (graph scratch) failed: {}",
                e
            ))
        })?;
        Ok(())
    }

    /// Stream-aware view-inplace variant of
    /// [`Self::multiblock_scan_u32_view_inplace`]. Same shape
    /// as [`Self::multiblock_scan_u32_inplace_on_stream`] but
    /// over a `CudaViewMut` (used by recorded radix sort
    /// digit loops that scan per-digit slices of the histogram
    /// in place). Records intermediate `block_sums` against
    /// the runtime before they drop at end-of-scope.
    pub(crate) fn multiblock_scan_u32_view_inplace_on_stream(
        &self,
        data: &mut CudaViewMut<'_, u32>,
        n: u32,
        cu_stream: &cudarc::driver::CudaStream,
        launch_stream: crate::device_runtime::StreamId,
        runtime: &crate::device_runtime::XlogDeviceRuntime,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }
        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;
            // SAFETY: phase2 kernel signature.
            unsafe {
                phase2_fn.clone().launch_on_stream(
                    cu_stream,
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (data, n),
                )
            }
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "multiblock_scan_phase2 (view on_stream) failed: {}",
                    e
                ))
            })?;
            return Ok(());
        }

        let num_blocks = n.div_ceil(block_size);
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;
        // Fence alloc-ready → launch_stream for block_sums
        // before phase1 kernel writes it. See the inplace
        // variant for the full rationale.
        runtime
            .prepare_first_use(
                &block_sums,
                launch_stream,
                crate::device_runtime::Access::Write,
            )
            .map_err(|e| {
                XlogError::Kernel(format!(
                    "multiblock_scan_u32_view_inplace_on_stream: prepare block_sums failed: {}",
                    e
                ))
            })?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;
        // SAFETY: phase1 kernel signature.
        unsafe {
            phase1_u32_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "multiblock_scan_u32_phase1 (view on_stream) failed: {}",
                e
            ))
        })?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace_on_stream(
                &mut block_sums,
                num_blocks,
                cu_stream,
                launch_stream,
                runtime,
            )?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;
        // SAFETY: phase3 kernel signature.
        unsafe {
            phase3_fn.clone().launch_on_stream(
                cu_stream,
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| {
            XlogError::Kernel(format!(
                "multiblock_scan_phase3 (view on_stream) failed: {}",
                e
            ))
        })?;

        // Record block_sums use before end-of-scope drop.
        if let Some(b) = block_sums.runtime_block() {
            runtime
                .finish_block_use(
                    crate::device_runtime::BlockId::from_block(b),
                    launch_stream,
                    crate::device_runtime::Access::Write,
                )
                .map_err(|e| {
                    XlogError::Kernel(format!(
                        "multiblock_scan_u32_view_inplace_on_stream: finish_block_use \
                     for intermediate block_sums failed: {}",
                        e
                    ))
                })?;
        } else {
            return Err(XlogError::Kernel(
                "multiblock_scan_u32_view_inplace_on_stream: intermediate block_sums has no \
                 runtime block — caller must use a runtime-backed manager"
                    .to_string(),
            ));
        }
        Ok(())
    }

    fn multiblock_scan_u32_view_inplace(
        &self,
        data: &mut CudaViewMut<'_, u32>,
        n: u32,
    ) -> Result<()> {
        if n == 0 {
            return Ok(());
        }

        let device = self.device.inner();
        let block_size = 256u32;

        if n <= block_size {
            let phase2_fn = device
                .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE2)
                .ok_or_else(|| {
                    XlogError::Kernel("Failed to get multiblock_scan_phase2 kernel".to_string())
                })?;

            // SAFETY: multiblock_scan_phase2(uint32_t* block_sums, uint32_t num_blocks)
            unsafe {
                phase2_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (1, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (data, n),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase2 failed: {}", e)))?;

            self.device.synchronize()?;
            return Ok(());
        }

        let num_blocks = n.div_ceil(block_size);
        let mut block_sums = self.memory.alloc::<u32>(num_blocks as usize)?;

        let phase1_u32_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_U32_PHASE1)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_u32_phase1 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_u32_phase1(uint32_t* data, uint32_t* block_sums, uint32_t n)
        unsafe {
            phase1_u32_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &mut block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_u32_phase1 failed: {}", e)))?;
        self.device.synchronize()?;

        if num_blocks > 1 {
            self.multiblock_scan_u32_inplace(&mut block_sums, num_blocks)?;
        }

        let phase3_fn = device
            .get_func(SCAN_MODULE, scan_kernels::MULTIBLOCK_SCAN_PHASE3)
            .ok_or_else(|| {
                XlogError::Kernel("Failed to get multiblock_scan_phase3 kernel".to_string())
            })?;

        // SAFETY: multiblock_scan_phase3(uint32_t* prefix_sum, const uint32_t* block_offsets, uint32_t n)
        unsafe {
            phase3_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (num_blocks, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut *data, &block_sums, n),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("multiblock_scan_phase3 failed: {}", e)))?;

        self.device.synchronize()?;
        Ok(())
    }

    // ============== Internal Helper Methods ==============

    /// Read a buffer's logical row count, using the host cache when available
    /// and falling back to a metadata-only device-to-host read when needed.
    pub fn device_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
        if let Some(n) = buffer.cached_row_count() {
            return Ok(n as usize);
        }
        let mut host_rows = [0u32];
        self.device
            .inner()
            .dtoh_sync_copy_into(buffer.num_rows_device(), &mut host_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to read row count: {}", e)))?;
        buffer.set_cached_row_count_if_unset(host_rows[0]);
        Ok(host_rows[0] as usize)
    }

    /// Read and validate a buffer's logical row count for outward-facing APIs.
    ///
    /// This keeps exported/query-visible lengths tied to the device logical row
    /// count while still rejecting impossible metadata (`logical_rows > row_cap`).
    pub fn validated_logical_row_count(&self, buffer: &CudaBuffer) -> Result<usize> {
        let logical_rows = self.device_row_count(buffer)?;
        validate_logical_row_count(buffer.num_rows(), logical_rows)
    }

    fn clone_device_row_count(&self, buffer: &CudaBuffer) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.device
            .inner()
            .dtod_copy(buffer.num_rows_device(), &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to copy row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn upload_device_row_count(&self, row_count: u32) -> Result<TrackedCudaSlice<u32>> {
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.htod_launch_metadata_sync_copy_into(&[row_count], &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to upload row count: {}", e)))?;
        Ok(d_num_rows)
    }

    fn buffer_from_columns_with_device_count(
        &self,
        columns: Vec<CudaColumn>,
        row_cap: u64,
        schema: Schema,
        src: &CudaBuffer,
    ) -> Result<CudaBuffer> {
        let d_num_rows = self.clone_device_row_count(src)?;
        Ok(CudaBuffer::from_columns(
            columns, row_cap, d_num_rows, schema,
        ))
    }

    fn column_bytes_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_bytes: usize,
    ) -> Result<RawCudaView<'a, u8>> {
        if col.num_bytes() < num_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required",
                col.num_bytes(),
                num_bytes
            )));
        }
        let ptr = *col.device_ptr();
        Ok(RawCudaView {
            ptr,
            len: num_bytes,
            stream: col.stream().clone(),
            source_block: col.runtime_block(),
            _marker: PhantomData,
        })
    }

    fn bytes_as_u32_view<'a>(
        &self,
        bytes: &'a TrackedCudaSlice<u8>,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u32>> {
        let required_bytes = num_elements * std::mem::size_of::<u32>();
        if bytes.len() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Packed keys have {} bytes but {} required for {} u32 elements",
                bytes.len(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *bytes.device_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<u32>()) {
            return Err(XlogError::Kernel(
                "Packed keys device pointer is not u32-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: bytes.stream().clone(),
            source_block: bytes.runtime_block(),
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as a `u32` slice for kernel access.
    fn column_as_u32_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u32>> {
        let required_bytes = num_elements * std::mem::size_of::<u32>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} u32 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<u32>()) {
            return Err(XlogError::Kernel(
                "Column device pointer is not u32-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            source_block: col.runtime_block(),
            _marker: PhantomData,
        })
    }

    fn column_as_u64_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, u64>> {
        let required_bytes = num_elements * std::mem::size_of::<u64>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} u64 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<u64>()) {
            return Err(XlogError::Kernel(
                "Column device pointer is not u64-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            source_block: col.runtime_block(),
            _marker: PhantomData,
        })
    }

    /// Reinterpret a `CudaBuffer` column as an `f64` slice for kernel access.
    fn column_as_f64_view<'a>(
        &self,
        col: &'a CudaColumn,
        num_elements: usize,
    ) -> Result<RawCudaView<'a, f64>> {
        let required_bytes = num_elements * std::mem::size_of::<f64>();
        if col.num_bytes() < required_bytes {
            return Err(XlogError::Kernel(format!(
                "Column has {} bytes but {} required for {} f64 elements",
                col.num_bytes(),
                required_bytes,
                num_elements
            )));
        }
        let ptr = *col.device_ptr();
        if !(ptr as usize).is_multiple_of(std::mem::align_of::<f64>()) {
            return Err(XlogError::Kernel(
                "Column device pointer is not f64-aligned".to_string(),
            ));
        }
        Ok(RawCudaView {
            ptr,
            len: num_elements,
            stream: col.stream().clone(),
            source_block: col.runtime_block(),
            _marker: PhantomData,
        })
    }

    /// Create an empty buffer with the given schema (all columns are empty slices)
    ///
    /// # Arguments
    /// * `schema` - The schema for the empty buffer
    ///
    /// # Returns
    /// A new CudaBuffer with zero rows
    ///
    /// # Errors
    /// Returns `XlogError::Kernel` if allocation fails
    pub fn create_empty_buffer(&self, schema: Schema) -> Result<CudaBuffer> {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            // Allocate zero-length column
            columns.push(self.memory.alloc::<u8>(0)?.into());
        }
        self.buffer_from_columns(columns, 0, schema)
    }

    /// Create a zero-arity (nullary) relation buffer carrying `rows` unit tuples.
    ///
    /// A nullary relation holds exactly when it has at least one row; its single
    /// possible tuple is the empty tuple `()`. `create_buffer_from_slices` with no
    /// column slices routes to `create_empty_buffer` (0 rows), which represents the
    /// relation as *absent* — wrong for an asserted nullary fact. Nullary facts must
    /// use this path so presence is materialized as one row.
    pub fn create_zero_arity_buffer(&self, schema: Schema, rows: u32) -> Result<CudaBuffer> {
        debug_assert_eq!(
            schema.arity(),
            0,
            "create_zero_arity_buffer requires arity 0"
        );
        self.buffer_from_columns(Vec::new(), u64::from(rows), schema)
    }

    pub(crate) fn buffer_from_columns(
        &self,
        columns: Vec<CudaColumn>,
        row_cap: u64,
        schema: Schema,
    ) -> Result<CudaBuffer> {
        let row_u32 = u32::try_from(row_cap)
            .map_err(|_| XlogError::Kernel(format!("Row capacity {} exceeds u32::MAX", row_cap)))?;
        let mut d_num_rows = self.memory.alloc::<u32>(1)?;
        self.htod_launch_metadata_sync_copy_into(&[row_u32], &mut d_num_rows)
            .map_err(|e| XlogError::Kernel(format!("Failed to set row count: {}", e)))?;
        Ok(CudaBuffer::from_columns_with_host_count(
            columns, row_cap, d_num_rows, schema, row_u32,
        ))
    }

    /// Combine schemas from left and right buffers for join result
    fn combine_schemas(&self, left: &Schema, right: &Schema) -> Schema {
        let mut columns = left.columns.clone();
        columns.extend(right.columns.iter().cloned());
        let mut sort_labels = left.sort_labels().to_vec();
        sort_labels.extend(right.sort_labels().iter().cloned());
        Schema::new(columns)
            .with_sort_labels(sort_labels)
            .expect("combined schema sort labels match column arity")
    }

    /// Check if two schemas have compatible types (same arity and column types)
    ///
    /// This ignores column names, which is useful for Datalog operations where
    /// projected relations may have different column names but the same types.
    fn schemas_type_compatible(&self, a: &Schema, b: &Schema) -> bool {
        if a.arity() != b.arity() {
            return false;
        }
        for i in 0..a.arity() {
            if a.column_type(i) != b.column_type(i) {
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device_runtime::{
        AsyncCudaResource, DeviceMemoryResource, GlobalDeviceBudget, LoggingResource, NullSink,
        StreamPool, XlogDeviceRuntime,
    };
    use xlog_core::{AggOp, MemoryBudget, ScalarType};

    fn has_cuda_device() -> bool {
        CudaDevice::new(0).is_ok()
    }

    #[test]
    fn test_kernel_artifact_locator_precedence_order() {
        use super::kernel_paths::KernelArtifactLocator;
        use std::fs;
        use std::path::PathBuf;

        let root = std::env::temp_dir().join(format!(
            "xlog-kernel-paths-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .expect("system clock before UNIX_EPOCH")
                .as_nanos()
        ));
        let cubin_dir = root.join("cubin");
        let package_dir = root.join("bin").join("kernels");
        let out_dir = root.join("out");
        fs::create_dir_all(&cubin_dir).expect("create cubin dir");
        fs::create_dir_all(&package_dir).expect("create package kernels dir");
        fs::create_dir_all(&out_dir).expect("create out dir");

        let name = "xlog_join";
        let cc = 75;
        let cubin_path = cubin_dir.join(format!("{name}.sm_{cc}.cubin"));
        let package_path = package_dir.join(format!("{name}.sm_{cc}.cubin"));
        let out_path = out_dir.join(format!("{name}.sm_{cc}.cubin"));
        fs::write(&cubin_path, b"cubin").expect("write cubin file");
        fs::write(&package_path, b"package").expect("write package file");
        fs::write(&out_path, b"out").expect("write out file");

        let locator = KernelArtifactLocator::new(
            Some(cubin_dir.clone()),
            Some(package_dir.clone()),
            Some(out_dir.clone()),
        );

        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected a kernel artifact");
        assert_eq!(path, cubin_path);
        assert!(is_cubin);

        fs::remove_file(&cubin_path).expect("remove cubin file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected package kernel artifact");
        assert_eq!(path, package_path);
        assert!(is_cubin);

        fs::remove_file(&package_path).expect("remove package file");
        let (path, is_cubin) = locator
            .resolve_module_path(name, cc)
            .expect("expected out dir kernel artifact");
        assert_eq!(path, out_path);
        assert!(is_cubin);

        let _ = fs::remove_dir_all(PathBuf::from(&root));
    }

    #[test]
    fn test_module_resolution_finds_portable_ptx() {
        // Verify resolve_module_path finds portable PTX for all modules.
        // Uses a dummy cc (999) so cubin won't match — only portable PTX.
        for name in crate::kernel_manifest_data::KERNEL_CU_NAMES {
            let result = resolve_module_path(name, 999);
            assert!(
                result.is_some(),
                "resolve_module_path({name}, 999) should find portable PTX"
            );
            let (path, is_cubin) = result.unwrap();
            assert!(
                !is_cubin,
                "{name}: expected portable PTX fallback, got cubin"
            );
            assert!(
                path.to_str().unwrap().ends_with(".portable.ptx"),
                "{name}: path should end with .portable.ptx, got {:?}",
                path
            );
        }
    }

    #[test]
    fn test_module_resolution_falls_back_to_embedded_portable_ptx() {
        use super::kernel_paths::KernelArtifactLocator;

        let locator = KernelArtifactLocator::new(None, None, None);
        for name in crate::kernel_manifest_data::KERNEL_CU_NAMES {
            let sources = resolve_module_sources_with_locator(name, 999, &locator);
            assert_eq!(
                sources.len(),
                1,
                "{name}: expected only embedded portable PTX fallback"
            );

            match &sources[0] {
                KernelModuleSource::EmbeddedPortablePtx { ptx } => {
                    assert!(
                        ptx.contains(".entry"),
                        "{name}: embedded PTX should contain CUDA entry points"
                    );
                }
                KernelModuleSource::File { path, .. } => {
                    panic!(
                        "{name}: expected embedded portable PTX fallback, got file {}",
                        path.display()
                    );
                }
            }
        }
    }

    #[test]
    fn test_embedded_portable_ptx_manifest_matches_kernel_manifest() {
        let embedded_names: std::collections::BTreeSet<_> =
            crate::embedded_kernel_data::EMBEDDED_PORTABLE_PTX
                .iter()
                .map(|artifact| artifact.name)
                .collect();
        let manifest_names: std::collections::BTreeSet<_> =
            crate::kernel_manifest_data::KERNEL_CU_NAMES
                .iter()
                .copied()
                .collect();

        assert_eq!(
            embedded_names, manifest_names,
            "embedded portable PTX table should cover every runtime kernel module"
        );
    }

    #[test]
    fn test_kernel_provider_creation() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024); // 1 GB
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let provider = CudaKernelProvider::new(device.clone(), memory.clone());
        assert!(
            provider.is_ok(),
            "Failed to create kernel provider: {:?}",
            provider.err()
        );

        let provider = provider.unwrap();
        assert!(Arc::ptr_eq(provider.device(), &device));
        assert!(Arc::ptr_eq(provider.memory(), &memory));
    }

    #[test]
    fn test_kernel_functions_accessible() {
        if !has_cuda_device() {
            eprintln!("Skipping test: no CUDA device available");
            return;
        }

        let device = Arc::new(CudaDevice::new(0).expect("Failed to create device"));
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));

        let _provider =
            CudaKernelProvider::new(device.clone(), memory).expect("Failed to create provider");

        // Verify all kernel functions can be retrieved
        let inner = device.inner();

        // Join kernels
        let build_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_BUILD);
        assert!(
            build_fn.is_some(),
            "hash_join_build function should be accessible"
        );

        let probe_fn = inner.get_func(JOIN_MODULE, join_kernels::HASH_JOIN_PROBE);
        assert!(
            probe_fn.is_some(),
            "hash_join_probe function should be accessible"
        );

        // Dedup kernels
        let mark_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::MARK_DUPLICATES);
        assert!(
            mark_fn.is_some(),
            "mark_duplicates function should be accessible"
        );

        let compact_fn = inner.get_func(DEDUP_MODULE, dedup_kernels::COMPACT_ROWS);
        assert!(
            compact_fn.is_some(),
            "compact_rows function should be accessible"
        );

        // GroupBy kernels
        let boundaries_fn =
            inner.get_func(GROUPBY_MODULE, groupby_kernels::DETECT_GROUP_BOUNDARIES);
        assert!(
            boundaries_fn.is_some(),
            "detect_group_boundaries function should be accessible"
        );

        let count_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_COUNT);
        assert!(
            count_fn.is_some(),
            "groupby_count function should be accessible"
        );

        let sum_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_SUM);
        assert!(
            sum_fn.is_some(),
            "groupby_sum function should be accessible"
        );

        let min_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MIN);
        assert!(
            min_fn.is_some(),
            "groupby_min function should be accessible"
        );

        let max_fn = inner.get_func(GROUPBY_MODULE, groupby_kernels::GROUPBY_MAX);
        assert!(
            max_fn.is_some(),
            "groupby_max function should be accessible"
        );

        // Circuit kernels (XGCF forward/backward)
        let xgcf_forward = inner.get_func(CIRCUIT_MODULE, "xgcf_forward_level");
        assert!(
            xgcf_forward.is_some(),
            "xgcf_forward_level function should be accessible"
        );

        let xgcf_backward_propagate =
            inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_propagate");
        assert!(
            xgcf_backward_propagate.is_some(),
            "xgcf_backward_level_propagate function should be accessible"
        );

        let xgcf_backward_decision_grad =
            inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_decision_grad");
        assert!(
            xgcf_backward_decision_grad.is_some(),
            "xgcf_backward_level_decision_grad function should be accessible"
        );

        let xgcf_backward_lit_grad = inner.get_func(CIRCUIT_MODULE, "xgcf_backward_level_lit_grad");
        assert!(
            xgcf_backward_lit_grad.is_some(),
            "xgcf_backward_level_lit_grad function should be accessible"
        );

        // Neural fast-path kernels (AD chain weight fill + gradient scatter)
        let neural_fill = inner.get_func("xlog_neural", "neural_fill_ad_chain_f32");
        assert!(
            neural_fill.is_some(),
            "neural_fill_ad_chain_f32 function should be accessible"
        );
        let neural_scatter = inner.get_func("xlog_neural", "neural_scatter_ad_chain_grads_f32");
        assert!(
            neural_scatter.is_some(),
            "neural_scatter_ad_chain_grads_f32 function should be accessible"
        );
    }

    #[test]
    fn test_module_names_unique() {
        // Ensure module names don't collide
        assert_ne!(JOIN_MODULE, DEDUP_MODULE);
        assert_ne!(JOIN_MODULE, GROUPBY_MODULE);
        assert_ne!(DEDUP_MODULE, GROUPBY_MODULE);
    }

    // Helper function to create test provider
    fn create_test_provider() -> Option<CudaKernelProvider> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    fn create_test_provider_with_runtime() -> Option<(CudaKernelProvider, Arc<XlogDeviceRuntime>)> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let pool = Arc::new(StreamPool::with_defaults(Arc::clone(&device)));
        let sink = Arc::new(NullSink::new());
        let async_resource: Box<dyn DeviceMemoryResource + Send + Sync> = Box::new(
            AsyncCudaResource::new(Arc::clone(&device), 0, Arc::clone(&pool)),
        );
        let logging: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(LoggingResource::new(async_resource, sink));
        let budget: Box<dyn DeviceMemoryResource + Send + Sync> =
            Box::new(GlobalDeviceBudget::new(logging, 1024 * 1024 * 1024));
        let runtime = Arc::new(XlogDeviceRuntime::with_resource(
            Arc::clone(&device),
            0,
            pool,
            budget,
        ));
        let memory = Arc::new(GpuMemoryManager::with_runtime(
            Arc::clone(&device),
            MemoryBudget::with_limit(1024 * 1024 * 1024),
            Arc::clone(&runtime),
        ));
        let provider = CudaKernelProvider::with_runtime(device, memory).ok()?;
        Some((provider, runtime))
    }

    #[test]
    fn test_recorded_join_index_build_runs_on_runtime_stream() {
        let (provider, runtime) = match create_test_provider_with_runtime() {
            Some(fixture) => fixture,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };
        let stream = runtime.stream_pool().acquire().expect("recorded stream");
        let left = create_test_buffer(&provider, &[1, 2, 3, 4], "key");
        let right = create_test_buffer(&provider, &[1, 2, 3, 4], "key");

        let index = provider
            .build_join_index_v2_recorded(&right, &[0], stream)
            .expect("recorded join-index build");
        let joined = provider
            .hash_join_v2_with_index_recorded(
                &left,
                &right,
                &[0],
                &[0],
                JoinType::Inner,
                &index,
                None,
                stream,
            )
            .expect("recorded indexed join consumes recorded build");
        runtime
            .stream_pool()
            .resolve(stream)
            .expect("stream resolves")
            .synchronize()
            .expect("recorded stream synchronized");

        assert_eq!(index.right_num_rows(), 4);
        assert_eq!(index.right_keys(), &[0]);
        assert_eq!(provider.device_row_count(&joined).expect("joined rows"), 4);
    }

    // Helper function to create a CudaBuffer with U32 data
    fn create_test_buffer(
        provider: &CudaKernelProvider,
        data: &[u32],
        col_name: &str,
    ) -> CudaBuffer {
        let schema = Schema::new(vec![(col_name.to_string(), ScalarType::U32)]);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();

        let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");

        provider
            .buffer_from_columns(vec![col.into()], data.len() as u64, schema)
            .expect("buffer")
    }

    // Helper function to create an empty buffer with correct column count
    fn create_empty_test_buffer(provider: &CudaKernelProvider, schema: Schema) -> CudaBuffer {
        let mut columns = Vec::with_capacity(schema.arity());
        for _ in 0..schema.arity() {
            columns.push(provider.memory().alloc::<u8>(0).expect("alloc").into());
        }
        provider
            .buffer_from_columns(columns, 0, schema)
            .expect("buffer")
    }

    // Helper function to read U32 data from CudaBuffer
    fn read_buffer_u32(provider: &CudaKernelProvider, buffer: &CudaBuffer, col: usize) -> Vec<u32> {
        if buffer.is_empty() || buffer.column(col).is_none() {
            return vec![];
        }
        let num_rows = buffer.num_rows() as usize;
        let mut bytes = vec![0u8; num_rows * 4];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(buffer.column(col).unwrap(), &mut bytes)
            .expect("dtoh");
        bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }

    #[test]
    fn test_compact_device_mask_respects_mask_len_smaller_than_row_cap() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let base = create_test_buffer(&provider, &[1, 2, 3, 4, 5, 6, 7, 8], "id");

        let row_cap = 16u64;
        let data: Vec<u32> = (0..row_cap as u32).collect();
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut col = provider.memory().alloc::<u8>(bytes.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes, &mut col)
            .expect("htod");
        let expanded = provider
            .buffer_from_columns_with_device_count(vec![col.into()], row_cap, schema, &base)
            .expect("buffer");

        let mask: Vec<u8> = vec![1, 0, 1, 0, 1, 0, 1, 0];
        let (prefix_sum, count) = provider.prefix_sum_mask(&mask).expect("prefix sum");

        let mut d_mask = provider.memory().alloc::<u8>(mask.len()).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&mask, &mut d_mask)
            .expect("mask htod");

        let mut d_prefix = provider
            .memory()
            .alloc::<u32>(prefix_sum.len())
            .expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&prefix_sum, &mut d_prefix)
            .expect("prefix htod");

        let mut d_out_count = provider.memory().alloc::<u32>(1).expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&[count], &mut d_out_count)
            .expect("count htod");

        let compacted = provider
            .compact_buffer_by_device_mask_device_count(&expanded, &d_mask, &d_prefix, d_out_count)
            .expect("compact");

        assert_eq!(compacted.num_rows(), mask.len() as u64);
        let device_rows = provider.device_row_count(&compacted).expect("row count");
        assert_eq!(device_rows as u32, count);
    }

    #[test]
    fn test_clone_buffer_preserves_device_count() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let ids: Vec<u32> = vec![10, 20, 30];
        let buffer = provider
            .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
            .unwrap();

        let cloned = provider.clone_buffer(&buffer).unwrap();

        let mut host_count = [0u32];
        provider
            .device()
            .inner()
            .dtoh_sync_copy_into(cloned.num_rows_device(), &mut host_count)
            .unwrap();
        assert_eq!(host_count[0], 3);
    }

    /// `clone_buffer` must propagate the host-side `cached_row_count` so
    /// downstream code can read the row count without a D2H round-trip.
    /// Without this propagation, buffers flowed through the relation store
    /// (`CompiledIlpProgram::put_relation` calls `clone_buffer` before
    /// storing) lose their host-visible count, forcing consumers to choose
    /// between an extra D2H (violating the native bounded exact-induction
    /// transfer-budget gates) and a hard error. This test pins the cache-propagation
    /// contract directly.
    #[test]
    fn test_clone_buffer_preserves_cached_row_count() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("id".to_string(), ScalarType::U32)]);
        let ids: Vec<u32> = vec![7, 11, 13, 17];
        let source = provider
            .create_buffer_from_slices(&[bytemuck::cast_slice(&ids)], schema)
            .unwrap();
        // Source's cache is populated by the `create_buffer_from_*` path;
        // verify the precondition so a regression in that path shows up here
        // rather than silently passing the real assertion below.
        assert_eq!(
            source.cached_row_count(),
            Some(4),
            "source buffer should have its cached row count populated by \
             create_buffer_from_slices"
        );

        let cloned = provider.clone_buffer(&source).unwrap();

        assert_eq!(
            cloned.cached_row_count(),
            Some(4),
            "clone_buffer must propagate cached_row_count from source to clone",
        );
    }

    // ============== Hash Join Tests ==============

    #[test]
    fn test_hash_join_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Join empty with empty
        let result = provider.hash_join(&empty, &empty, &[0], &[0]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_hash_join_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = create_test_buffer(&provider, &[1, 2, 3], "left_key");
        let right = create_test_buffer(&provider, &[2, 3, 4], "right_key");

        // Empty key columns
        let result = provider.hash_join(&left, &right, &[], &[0]);
        assert!(result.is_err());

        // Mismatched key lengths
        let result = provider.hash_join(&left, &right, &[0], &[0, 0]);
        assert!(result.is_err());
    }

    // ============== Dedup Tests ==============

    #[test]
    fn test_dedup_empty_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema);

        let result = provider.dedup(&empty, &[0]);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_dedup_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&provider, &[1, 1, 2, 2, 3], "key");

        // Empty key columns
        let result = provider.dedup(&buffer, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_dedup_with_duplicates() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Test dedup with duplicates: [3, 1, 2, 1, 3, 2]
        let buffer = create_test_buffer(&provider, &[3, 1, 2, 1, 3, 2], "key");
        let deduped = provider.dedup(&buffer, &[0]).unwrap();

        let dedup_count = provider
            .device_row_count(&deduped)
            .expect("read dedup row count");
        assert_eq!(dedup_count, 3, "Should have 3 unique values");

        let result = provider.download_column::<u32>(&deduped, 0).unwrap();
        // Result should be sorted and deduped
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_dedup_larger_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create input with duplicates: 0..500 ++ 250..750 = 1000 elements, 750 unique
        let a: Vec<u32> = (0..500).collect();
        let b: Vec<u32> = (250..750).collect();
        let input: Vec<u32> = a.iter().chain(b.iter()).copied().collect();

        let buffer = create_test_buffer(&provider, &input, "key");
        let deduped = provider.dedup(&buffer, &[0]).unwrap();

        let dedup_count = provider
            .device_row_count(&deduped)
            .expect("read dedup row count");
        assert_eq!(dedup_count, 750, "Should have 750 unique values (0..750)");

        // Verify output is sorted
        let result = provider.download_column::<u32>(&deduped, 0).unwrap();
        let is_sorted = result.windows(2).all(|w| w[0] <= w[1]);
        assert!(is_sorted, "Output should be sorted");

        // Verify expected values
        let expected: Vec<u32> = (0..750).collect();
        assert_eq!(result, expected);
    }

    // ============== Union Tests ==============

    #[test]
    fn test_union_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Empty union empty
        let result = provider.union(&empty, &empty);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Non-empty union empty
        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let empty2 = create_empty_test_buffer(&provider, schema);
        let result = provider.union(&a, &empty2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_union_schema_type_mismatch() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[3, 4], "col_b");

        // Different column names but same types should succeed (Datalog union semantics)
        let result = provider.union(&a, &b);
        assert!(result.is_ok());

        // Different arity should fail - create a 2-column buffer
        let two_col_schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U32),
            ("y".to_string(), ScalarType::U32),
        ]);
        let c = provider
            .create_buffer_from_u32_columns(&[&[1, 2], &[3, 4]], two_col_schema)
            .unwrap();
        let result = provider.union(&a, &c);
        assert!(result.is_err());
    }

    // ============== Diff Tests ==============

    #[test]
    fn test_diff_empty_inputs() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema.clone());

        // Empty diff empty
        let result = provider.diff(&empty, &empty);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());

        // Non-empty diff empty should return all of a
        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let empty2 = create_empty_test_buffer(&provider, schema);
        let result = provider.diff(&a, &empty2);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3);
    }

    #[test]
    fn test_diff_basic() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2, 3, 4, 5], "key");
        let b = create_test_buffer(&provider, &[2, 4], "key");

        let result = provider.diff(&a, &b);
        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.num_rows(), 3); // 1, 3, 5

        let values = read_buffer_u32(&provider, &result, 0);
        assert_eq!(values, vec![1, 3, 5]);
    }

    #[test]
    fn test_diff_all_filtered_out() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let a = create_test_buffer(&provider, &[1, 2, 3], "key");
        let b = create_test_buffer(&provider, &[1, 2, 3, 4, 5], "key");

        let result = provider.diff(&a, &b);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_diff_schema_mismatch() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Different column names with same types should work (Datalog semantics)
        let a = create_test_buffer(&provider, &[1, 2], "col_a");
        let b = create_test_buffer(&provider, &[1, 2], "col_b");
        let result = provider.diff(&a, &b);
        assert!(
            result.is_ok(),
            "Same types with different names should succeed"
        );

        // Create buffers with different arities (this should fail)
        let schema_2col = Schema::new(vec![
            ("c0".to_string(), ScalarType::U32),
            ("c1".to_string(), ScalarType::U32),
        ]);

        let bytes_2col: Vec<u8> = [1u32, 2, 3, 4]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let mut col0 = provider
            .memory()
            .alloc::<u8>(bytes_2col.len() / 2)
            .expect("alloc");
        let mut col1 = provider
            .memory()
            .alloc::<u8>(bytes_2col.len() / 2)
            .expect("alloc");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes_2col[..8], &mut col0)
            .expect("htod");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&bytes_2col[8..], &mut col1)
            .expect("htod");
        let buffer_2col = provider
            .buffer_from_columns(vec![col0.into(), col1.into()], 2, schema_2col)
            .expect("buffer");

        let buffer_1col = create_test_buffer(&provider, &[1, 2], "c0");

        let result = provider.diff(&buffer_2col, &buffer_1col);
        assert!(result.is_err(), "Different arities should fail");
    }

    // ============== GroupBy Aggregation Tests ==============

    #[test]
    fn test_groupby_empty_input() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let schema = Schema::new(vec![("key".to_string(), ScalarType::U32)]);
        let empty = create_empty_test_buffer(&provider, schema);

        let result = provider.groupby_agg(&empty, &[0], AggOp::Count, 0);
        assert!(result.is_ok());
        assert!(result.unwrap().is_empty());
    }

    #[test]
    fn test_groupby_validation() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let buffer = create_test_buffer(&provider, &[1, 1, 2, 2, 3], "key");

        // Empty key columns
        let result = provider.groupby_agg(&buffer, &[], AggOp::Count, 0);
        assert!(result.is_err());

        // Value column out of bounds
        let result = provider.groupby_agg(&buffer, &[0], AggOp::Count, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_groupby_logsumexp() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create buffer with U32 keys and F64 values
        // Group 0 (key=1): values 1.0, 2.0 -> logsumexp = log(e^1 + e^2) ≈ 2.31326
        // Group 1 (key=2): values 3.0, 4.0 -> logsumexp = log(e^3 + e^4) ≈ 4.31326
        let keys: Vec<u32> = vec![1, 1, 2, 2];
        let values: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];

        let schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("value".to_string(), ScalarType::F64),
        ]);

        // Create key column
        let key_bytes: Vec<u8> = keys.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut key_col = provider
            .memory()
            .alloc::<u8>(key_bytes.len())
            .expect("alloc key");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&key_bytes, &mut key_col)
            .expect("upload key");

        // Create value column
        let val_bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut val_col = provider
            .memory()
            .alloc::<u8>(val_bytes.len())
            .expect("alloc val");
        provider
            .device()
            .inner()
            .htod_sync_copy_into(&val_bytes, &mut val_col)
            .expect("upload val");

        let buffer = provider
            .buffer_from_columns(vec![key_col.into(), val_col.into()], 4, schema)
            .expect("buffer");

        // Run LogSumExp aggregation grouped by key column (0), aggregating value column (1)
        let result = provider.groupby_agg(&buffer, &[0], AggOp::LogSumExp, 1);
        assert!(
            result.is_ok(),
            "groupby_agg with LogSumExp should succeed: {:?}",
            result.err()
        );

        let result = result.unwrap();
        let group_count = provider
            .device_row_count(&result)
            .expect("read group count");
        assert_eq!(group_count, 2, "Should have 2 groups");

        // Download results
        let result_values = provider
            .download_column::<f64>(&result, 1)
            .expect("download result");

        // Expected values:
        // logsumexp(1.0, 2.0) = 2.0 + log(exp(1.0-2.0) + exp(2.0-2.0)) = 2.0 + log(e^-1 + 1) ≈ 2.31326
        // logsumexp(3.0, 4.0) = 4.0 + log(exp(3.0-4.0) + exp(4.0-4.0)) = 4.0 + log(e^-1 + 1) ≈ 4.31326
        let expected_0 = 2.0_f64 + ((-1.0_f64).exp() + 1.0_f64).ln(); // ≈ 2.31326
        let expected_1 = 4.0_f64 + ((-1.0_f64).exp() + 1.0_f64).ln(); // ≈ 4.31326

        let tolerance = 1e-5;
        assert!(
            (result_values[0] - expected_0).abs() < tolerance,
            "Group 0 logsumexp mismatch: got {}, expected {}",
            result_values[0],
            expected_0
        );
        assert!(
            (result_values[1] - expected_1).abs() < tolerance,
            "Group 1 logsumexp mismatch: got {}, expected {}",
            result_values[1],
            expected_1
        );
    }

    // ============== Schema Helper Tests ==============

    #[test]
    fn test_combine_schemas() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let left = Schema::new(vec![("a".to_string(), ScalarType::U32)]);
        let right = Schema::new(vec![("b".to_string(), ScalarType::U64)]);

        let combined = provider.combine_schemas(&left, &right);
        assert_eq!(combined.arity(), 2);
        assert_eq!(combined.column_type(0), Some(ScalarType::U32));
        assert_eq!(combined.column_type(1), Some(ScalarType::U64));
    }

    #[test]
    fn test_groupby_result_schema() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        let input = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("value".to_string(), ScalarType::U32),
        ]);

        // Count result schema (u64 to match predicate declarations)
        let count_schema =
            provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Count)]);
        assert_eq!(count_schema.arity(), 2);
        assert_eq!(count_schema.column_type(1), Some(ScalarType::U64));

        // Sum result schema
        let sum_schema = provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Sum)]);
        assert_eq!(sum_schema.arity(), 2);
        assert_eq!(sum_schema.column_type(1), Some(ScalarType::U64));

        // Min/Max result schema
        let min_schema = provider.groupby_multi_agg_result_schema(&input, &[0], &[(1, AggOp::Min)]);
        assert_eq!(min_schema.arity(), 2);
        assert_eq!(min_schema.column_type(1), Some(ScalarType::U32));
    }

    #[test]
    fn test_groupby_multi_agg_sum_returns_u64_schema() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device");
                return;
            }
        };

        let schema = Schema::new(vec![
            ("key".to_string(), ScalarType::U32),
            ("val".to_string(), ScalarType::U32),
        ]);

        let result_schema =
            provider.groupby_multi_agg_result_schema(&schema, &[0], &[(1, AggOp::Sum)]);

        // Sum should return U64 to prevent overflow
        assert_eq!(
            result_schema.column_type(1),
            Some(ScalarType::U64),
            "Sum aggregation should return U64 type, not U32"
        );
    }

    #[test]
    fn test_join_custom_max_output() {
        let provider = match create_test_provider() {
            Some(p) => p,
            None => {
                eprintln!("Skipping test: no CUDA device available");
                return;
            }
        };

        // Create buffers that produce more than 10 results when joined
        // Left: [1, 1, 1, 1, 2, 2, 2, 2] - 4 copies of 1, 4 copies of 2
        // Right: [1, 1, 1, 2, 2, 2] - 3 copies of 1, 3 copies of 2
        // Join produces: 4*3 + 4*3 = 24 results
        let left = create_test_buffer(&provider, &[1, 1, 1, 1, 2, 2, 2, 2], "left_key");
        let right = create_test_buffer(&provider, &[1, 1, 1, 2, 2, 2], "right_key");

        // Test with limit of 10 - should get at most 10
        let result_limited = provider
            .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, Some(10))
            .expect("join with limit should succeed");
        assert!(
            result_limited.num_rows() <= 10,
            "With limit 10, got {} rows but expected at most 10",
            result_limited.num_rows()
        );

        // Test with None (default) - should get all 24 results
        let result_unlimited = provider
            .hash_join_v2_with_limit(&left, &right, &[0], &[0], JoinType::Inner, None)
            .expect("join without limit should succeed");
        assert_eq!(
            result_unlimited.num_rows(),
            24,
            "Without limit, expected 24 rows but got {}",
            result_unlimited.num_rows()
        );

        // Test legacy API still works (backward compatibility)
        let result_legacy = provider
            .hash_join_v2(&left, &right, &[0], &[0], JoinType::Inner)
            .expect("legacy hash_join_v2 should succeed");
        assert_eq!(
            result_legacy.num_rows(),
            24,
            "Legacy API without limit, expected 24 rows but got {}",
            result_legacy.num_rows()
        );
    }

    // ============== Arithmetic Operation Tests ==============

    /// Helper to create a test provider for arithmetic tests
    fn create_arith_test_provider() -> Option<CudaKernelProvider> {
        if !has_cuda_device() {
            return None;
        }
        let device = Arc::new(CudaDevice::new(0).ok()?);
        let budget = MemoryBudget::with_limit(1024 * 1024 * 1024);
        let memory = Arc::new(GpuMemoryManager::new(device.clone(), budget));
        CudaKernelProvider::new(device, memory).ok()
    }

    /// Helper to create an i64 buffer for arithmetic tests
    fn create_i64_buffer(provider: &CudaKernelProvider, data: &[i64]) -> CudaBuffer {
        let schema = Schema::new(vec![("col".to_string(), ScalarType::I64)]);
        provider
            .create_buffer_from_slice::<i64>(data, schema)
            .unwrap()
    }

    /// Helper to create an f64 buffer for arithmetic tests
    fn create_f64_buffer(provider: &CudaKernelProvider, data: &[f64]) -> CudaBuffer {
        let schema = Schema::new(vec![("col".to_string(), ScalarType::F64)]);
        provider
            .create_buffer_from_slice::<f64>(data, schema)
            .unwrap()
    }

    #[test]
    fn test_add_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);
        let b = create_i64_buffer(&provider, &[10, 20, 30, 40, 50]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![11, 22, 33, 44, 55]);
    }

    #[test]
    fn test_sub_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20, 30, 40, 50]);
        let b = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);

        let result = provider.sub_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![9, 18, 27, 36, 45]);
    }

    #[test]
    fn test_mul_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[2, 3, 4, 5, 6]);
        let b = create_i64_buffer(&provider, &[3, 4, 5, 6, 7]);

        let result = provider.mul_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![6, 12, 20, 30, 42]);
    }

    #[test]
    fn test_div_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[100, 200, 300, 400]);
        let b = create_i64_buffer(&provider, &[10, 20, 30, 40]);

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![10, 10, 10, 10]);
    }

    #[test]
    fn test_div_columns_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20, 30]);
        let b = create_i64_buffer(&provider, &[2, 0, 3]); // Note: division by zero

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Division by zero returns i64::MAX
        assert_eq!(values, vec![5, i64::MAX, 10]);
    }

    #[test]
    fn test_mod_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[17, 23, 100, 7]);
        let b = create_i64_buffer(&provider, &[5, 7, 30, 3]);

        let result = provider.mod_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![2, 2, 10, 1]);
    }

    #[test]
    fn test_mod_columns_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[10, 20]);
        let b = create_i64_buffer(&provider, &[3, 0]); // Note: mod by zero

        let result = provider.mod_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Mod by zero returns 0
        assert_eq!(values, vec![1, 0]);
    }

    #[test]
    fn test_abs_column_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[-5, 10, -15, 20, 0]);

        let result = provider.abs_column(&a).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![5, 10, 15, 20, 0]);
    }

    #[test]
    fn test_min_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[5, 10, 15, 20]);
        let b = create_i64_buffer(&provider, &[3, 12, 10, 25]);

        let result = provider.min_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![3, 10, 10, 20]);
    }

    #[test]
    fn test_max_columns_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[5, 10, 15, 20]);
        let b = create_i64_buffer(&provider, &[3, 12, 10, 25]);

        let result = provider.max_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, vec![5, 12, 15, 25]);
    }

    #[test]
    fn test_add_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.5, 2.5, 3.5]);
        let b = create_f64_buffer(&provider, &[0.5, 1.5, 2.5]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_mul_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[2.0, 3.0, 4.0]);
        let b = create_f64_buffer(&provider, &[1.5, 2.0, 2.5]);

        let result = provider.mul_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![3.0, 6.0, 10.0]);
    }

    #[test]
    fn test_div_columns_f64_by_zero() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.0, -1.0, 0.0]);
        let b = create_f64_buffer(&provider, &[0.0, 0.0, 0.0]);

        let result = provider.div_columns(&a, &b).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // IEEE 754: 1.0/0.0 = Inf, -1.0/0.0 = -Inf, 0.0/0.0 = NaN
        assert!(values[0].is_infinite() && values[0].is_sign_positive());
        assert!(values[1].is_infinite() && values[1].is_sign_negative());
        assert!(values[2].is_nan());
    }

    #[test]
    fn test_pow_columns() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let base = create_i64_buffer(&provider, &[2, 3, 4, 5]);
        let exp = create_i64_buffer(&provider, &[3, 2, 2, 1]);

        let result = provider.pow_columns(&base, &exp).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // pow always returns f64
        assert_eq!(values, vec![8.0, 9.0, 16.0, 5.0]);
    }

    #[test]
    fn test_pow_columns_fractional_exp() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let base = create_f64_buffer(&provider, &[4.0, 9.0, 27.0]);
        let exp = create_f64_buffer(&provider, &[0.5, 0.5, 1.0 / 3.0]);

        let result = provider.pow_columns(&base, &exp).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        // sqrt(4) = 2, sqrt(9) = 3, cbrt(27) = 3
        assert!((values[0] - 2.0).abs() < 1e-10);
        assert!((values[1] - 3.0).abs() < 1e-10);
        assert!((values[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_cast_i64_to_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 4, 5]);

        let result = provider.cast_column(&a, ScalarType::F64).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_cast_f64_to_i64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.9, 2.1, 3.5, 4.0, 5.7]);

        let result = provider.cast_column(&a, ScalarType::I64).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        // Truncation towards zero
        assert_eq!(values, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_cast_i64_to_i32() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3, 100, 200]);

        let result = provider.cast_column(&a, ScalarType::I32).unwrap();
        let values = provider.download_column::<i32>(&result, 0).unwrap();

        assert_eq!(values, vec![1, 2, 3, 100, 200]);
    }

    #[test]
    fn test_arithmetic_row_count_mismatch() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[1, 2, 3]);
        let b = create_i64_buffer(&provider, &[1, 2]); // Different size

        let result = provider.add_columns(&a, &b);
        assert!(result.is_err());
        let err = result.err().unwrap();
        assert!(err.to_string().contains("Row count mismatch"));
    }

    #[test]
    fn test_arithmetic_empty_buffers() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[]);
        let b = create_i64_buffer(&provider, &[]);

        let result = provider.add_columns(&a, &b).unwrap();
        let values = provider.download_column::<i64>(&result, 0).unwrap();

        assert_eq!(values, Vec::<i64>::new());
    }

    #[test]
    fn test_wrapping_arithmetic_overflow() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_i64_buffer(&provider, &[i64::MAX, i64::MIN]);
        let b = create_i64_buffer(&provider, &[1, -1]);

        // Addition should wrap
        let add_result = provider.add_columns(&a, &b).unwrap();
        let add_values = provider.download_column::<i64>(&add_result, 0).unwrap();
        assert_eq!(add_values[0], i64::MIN); // MAX + 1 wraps to MIN
        assert_eq!(add_values[1], i64::MAX); // MIN - 1 wraps to MAX
    }

    #[test]
    fn test_abs_column_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[-1.5, 2.5, -3.5, 0.0]);

        let result = provider.abs_column(&a).unwrap();
        let values = provider.download_column::<f64>(&result, 0).unwrap();

        assert_eq!(values, vec![1.5, 2.5, 3.5, 0.0]);
    }

    #[test]
    fn test_min_max_columns_f64() {
        let Some(provider) = create_arith_test_provider() else {
            eprintln!("Skipping test: no CUDA device available");
            return;
        };

        let a = create_f64_buffer(&provider, &[1.5, 5.0, 3.0]);
        let b = create_f64_buffer(&provider, &[2.0, 3.0, 4.0]);

        let min_result = provider.min_columns(&a, &b).unwrap();
        let min_values = provider.download_column::<f64>(&min_result, 0).unwrap();
        assert_eq!(min_values, vec![1.5, 3.0, 3.0]);

        let max_result = provider.max_columns(&a, &b).unwrap();
        let max_values = provider.download_column::<f64>(&max_result, 0).unwrap();
        assert_eq!(max_values, vec![2.0, 5.0, 4.0]);
    }
}
