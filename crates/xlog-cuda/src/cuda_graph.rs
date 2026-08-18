//! CUDA Graph RAII helpers for production graph capture/replay.
//!
//! This module intentionally stays close to the CUDA driver API. The bounded
//! CSM CUDA Graph path needs explicit graph lifetime ownership and node
//! inventory before it can safely update graph-exec parameters for runtime
//! pointers and capacity classes.

use std::collections::HashSet;
use std::{
    fmt, mem, ptr,
    sync::{Arc, Mutex, OnceLock},
};

use cudarc::driver::{sys, CudaContext, CudaStream};
use libloading::Library;
use xlog_core::{Result, XlogError};

use crate::device_runtime::XlogDeviceRuntime;

pub const CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION: u32 = 1;
const CONDITIONAL_GRAPH_MINIMUM_DRIVER: i32 = 12_030;

type DriverGetVersionFn = unsafe extern "C" fn(*mut i32) -> sys::CUresult;
type ConditionalHandleCreateFn = unsafe extern "C" fn(
    *mut sys::CUgraphConditionalHandle,
    sys::CUgraph,
    sys::CUcontext,
    u32,
    u32,
) -> sys::CUresult;
type GraphAddNodeFn = unsafe extern "C" fn(
    *mut sys::CUgraphNode,
    sys::CUgraph,
    *const sys::CUgraphNode,
    usize,
    *mut sys::CUgraphNodeParams,
) -> sys::CUresult;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CudaConditionalGraphUnavailable {
    DriverLibraryUnavailable,
    MissingDriverSymbol {
        symbol: &'static str,
    },
    DriverVersionQueryFailed {
        code: sys::CUresult,
    },
    DriverVersionTooOld {
        found: i32,
        required: i32,
    },
    DriverCallFailed {
        operation: &'static str,
        code: sys::CUresult,
    },
    NullDriverHandle {
        operation: &'static str,
    },
    ContextMismatch,
    StreamCaptureBusy,
    BodyPopulationFailed {
        detail: String,
    },
}

impl CudaConditionalGraphUnavailable {
    pub fn is_unsupported(&self) -> bool {
        matches!(
            self,
            Self::DriverLibraryUnavailable
                | Self::MissingDriverSymbol { .. }
                | Self::DriverVersionTooOld { .. }
                | Self::DriverCallFailed {
                    code: sys::CUresult::CUDA_ERROR_NOT_SUPPORTED,
                    ..
                }
        )
    }

    pub fn decline_detail(&self) -> String {
        match self {
            Self::DriverLibraryUnavailable => {
                "CUDA driver library is unavailable for conditional graphs".to_string()
            }
            Self::MissingDriverSymbol { symbol } => {
                format!("CUDA driver is missing required conditional-graph symbol {symbol}")
            }
            Self::DriverVersionQueryFailed { code } => {
                format!("CUDA driver version query failed: {code:?}")
            }
            Self::DriverVersionTooOld { found, required } => {
                format!("CUDA conditional graphs require driver API {required}, found {found}")
            }
            Self::DriverCallFailed { operation, code } => {
                format!("CUDA conditional-graph operation {operation} failed: {code:?}")
            }
            Self::NullDriverHandle { operation } => {
                format!("CUDA conditional-graph operation {operation} returned a null handle")
            }
            Self::ContextMismatch => {
                "CUDA conditional graph and stream belong to different contexts".to_string()
            }
            Self::StreamCaptureBusy => {
                "CUDA stream already has an active graph capture".to_string()
            }
            Self::BodyPopulationFailed { detail } => {
                format!("CUDA conditional graph body population failed: {detail}")
            }
        }
    }

    pub fn body_population(error: impl fmt::Display) -> Self {
        Self::BodyPopulationFailed {
            detail: error.to_string(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct StreamCaptureKey {
    context: usize,
    stream: usize,
}

static ACTIVE_STREAM_CAPTURES: OnceLock<Mutex<HashSet<StreamCaptureKey>>> = OnceLock::new();

#[derive(Debug)]
struct StreamCaptureLease {
    key: StreamCaptureKey,
}

fn try_acquire_stream_capture_key(
    key: StreamCaptureKey,
) -> std::result::Result<StreamCaptureLease, CudaConditionalGraphUnavailable> {
    let registry = ACTIVE_STREAM_CAPTURES.get_or_init(|| Mutex::new(HashSet::new()));
    let mut active = registry
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if !active.insert(key) {
        return Err(CudaConditionalGraphUnavailable::StreamCaptureBusy);
    }
    Ok(StreamCaptureLease { key })
}

fn try_acquire_stream_capture(
    stream: &CudaStream,
) -> std::result::Result<StreamCaptureLease, CudaConditionalGraphUnavailable> {
    let context =
        stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
    try_acquire_stream_capture_key(StreamCaptureKey {
        context: context as usize,
        stream: stream.cu_stream() as usize,
    })
}

impl Drop for StreamCaptureLease {
    fn drop(&mut self) {
        let registry = ACTIVE_STREAM_CAPTURES.get_or_init(|| Mutex::new(HashSet::new()));
        let mut active = registry
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        active.remove(&self.key);
    }
}

impl fmt::Display for CudaConditionalGraphUnavailable {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.decline_detail())
    }
}

impl std::error::Error for CudaConditionalGraphUnavailable {}

impl From<XlogError> for CudaConditionalGraphUnavailable {
    fn from(error: XlogError) -> Self {
        Self::body_population(error)
    }
}

struct ConditionalGraphDriverApi {
    _library: Arc<Library>,
    driver_get_version: DriverGetVersionFn,
    conditional_handle_create: ConditionalHandleCreateFn,
    graph_add_node: GraphAddNodeFn,
}

impl ConditionalGraphDriverApi {
    fn load() -> std::result::Result<Arc<Self>, CudaConditionalGraphUnavailable> {
        #[cfg(target_os = "windows")]
        const CUDA_DRIVER_LIBRARY: &str = "nvcuda.dll";
        #[cfg(not(target_os = "windows"))]
        const CUDA_DRIVER_LIBRARY: &str = "libcuda.so.1";

        let library = Arc::new(
            unsafe { Library::new(CUDA_DRIVER_LIBRARY) }
                .map_err(|_| CudaConditionalGraphUnavailable::DriverLibraryUnavailable)?,
        );
        let driver_get_version = unsafe {
            load_required_symbol(&library, b"cuDriverGetVersion\0", "cuDriverGetVersion")?
        };
        let conditional_handle_create = unsafe {
            load_required_symbol(
                &library,
                b"cuGraphConditionalHandleCreate\0",
                "cuGraphConditionalHandleCreate",
            )?
        };
        let graph_add_node =
            unsafe { load_required_symbol(&library, b"cuGraphAddNode\0", "cuGraphAddNode")? };

        let api = Arc::new(Self {
            _library: library,
            driver_get_version,
            conditional_handle_create,
            graph_add_node,
        });
        api.require_supported_driver()?;
        Ok(api)
    }

    fn require_supported_driver(&self) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
        let mut version = 0;
        let code = unsafe { (self.driver_get_version)(&mut version) };
        if code != sys::CUresult::CUDA_SUCCESS {
            return Err(CudaConditionalGraphUnavailable::DriverVersionQueryFailed { code });
        }
        require_conditional_graph_driver(version)
    }
}

unsafe fn load_required_symbol<F: Copy>(
    library: &Library,
    name: &'static [u8],
    display_name: &'static str,
) -> std::result::Result<F, CudaConditionalGraphUnavailable> {
    library.get::<F>(name).map(|symbol| *symbol).map_err(|_| {
        CudaConditionalGraphUnavailable::MissingDriverSymbol {
            symbol: display_name,
        }
    })
}

fn require_conditional_graph_driver(
    version: i32,
) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
    if version < CONDITIONAL_GRAPH_MINIMUM_DRIVER {
        Err(CudaConditionalGraphUnavailable::DriverVersionTooOld {
            found: version,
            required: CONDITIONAL_GRAPH_MINIMUM_DRIVER,
        })
    } else {
        Ok(())
    }
}

fn conditional_while_node_params(
    handle: sys::CUgraphConditionalHandle,
    ctx: sys::CUcontext,
) -> sys::CUgraphNodeParams {
    let mut params: sys::CUgraphNodeParams = unsafe { mem::zeroed() };
    params.type_ = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL;
    params.__bindgen_anon_1.conditional = sys::CUDA_CONDITIONAL_NODE_PARAMS {
        handle,
        type_: sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE,
        size: 1,
        phGraph_out: ptr::null_mut(),
        ctx,
    };
    params
}

fn conditional_driver_call(
    operation: &'static str,
    code: sys::CUresult,
) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(CudaConditionalGraphUnavailable::DriverCallFailed { operation, code })
    }
}

struct UninstantiatedCudaGraph {
    raw: sys::CUgraph,
    context: Arc<CudaContext>,
}

impl UninstantiatedCudaGraph {
    fn create(
        context: Arc<CudaContext>,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        context
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut raw = ptr::null_mut();
        unsafe {
            conditional_driver_call("cuGraphCreate", sys::cuGraphCreate(&mut raw, 0))?;
        }
        if raw.is_null() {
            Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphCreate",
            })
        } else {
            Ok(Self { raw, context })
        }
    }

    fn raw(&self) -> sys::CUgraph {
        self.raw
    }

    fn into_raw(mut self) -> sys::CUgraph {
        let raw = self.raw;
        self.raw = ptr::null_mut();
        raw
    }
}

impl Drop for UninstantiatedCudaGraph {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            let _ = self.context.bind_to_thread();
            unsafe {
                let _ = sys::cuGraphDestroy(self.raw);
            }
        }
    }
}

/// CUDA-owned body graph produced while adding one conditional-WHILE node.
///
/// The body graph and conditional handle are owned by the parent graph. They
/// must not be destroyed independently. The numeric handle is intended to be
/// passed by value to a device kernel that calls `cudaGraphSetConditional`.
#[derive(Debug)]
pub struct ConditionalCudaGraphBody {
    graph: sys::CUgraph,
    handle: sys::CUgraphConditionalHandle,
    context: sys::CUcontext,
}

impl ConditionalCudaGraphBody {
    pub fn graph(&self) -> sys::CUgraph {
        self.graph
    }

    pub fn handle(&self) -> sys::CUgraphConditionalHandle {
        self.handle
    }

    pub fn context(&self) -> sys::CUcontext {
        self.context
    }

    /// Return this body's actual node kinds in dependency-chain order.
    ///
    /// The body must be a single linear dependency chain. CUDA's node-list
    /// enumeration order is not used as an execution-order signal.
    pub fn linear_chain_node_kinds(
        &self,
    ) -> std::result::Result<Vec<CudaGraphNodeKind>, CudaConditionalGraphUnavailable> {
        let mut check = |operation, code| conditional_driver_call(operation, code);
        let mut shape_error = |error| CudaConditionalGraphUnavailable::BodyPopulationFailed {
            detail: format!("conditional graph body is not a linear dependency chain: {error}"),
        };
        graph_linear_chain_node_kinds_with(self.graph, &mut check, &mut shape_error)
    }

    /// Capture graph-compatible work directly into this conditional body.
    ///
    /// `stream` must be a non-default stream. The callback must not allocate,
    /// synchronize, or record/wait on events.
    /// CUDA conditional bodies allow only kernel, empty, child-graph, device
    /// memcpy/memset, and nested conditional nodes.
    pub fn capture_on_stream<F, E>(
        &self,
        stream: &CudaStream,
        record: F,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable>
    where
        F: FnOnce() -> std::result::Result<(), E>,
        E: fmt::Display,
    {
        let stream_ctx =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if stream_ctx != self.context {
            return Err(CudaConditionalGraphUnavailable::ContextMismatch);
        }
        let _capture_lease = try_acquire_stream_capture(stream)?;

        unsafe {
            conditional_driver_call(
                "cuStreamBeginCaptureToGraph",
                sys::cuStreamBeginCaptureToGraph(
                    stream.cu_stream(),
                    self.graph,
                    ptr::null(),
                    ptr::null(),
                    0,
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                ),
            )?;
        }

        let record_result = record();
        let mut captured = ptr::null_mut();
        let end_result = unsafe {
            conditional_driver_call(
                "cuStreamEndCapture",
                sys::cuStreamEndCapture(stream.cu_stream(), &mut captured),
            )
        };
        if let Err(error) = record_result {
            return Err(CudaConditionalGraphUnavailable::body_population(error));
        }
        end_result?;
        if captured.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuStreamEndCapture",
            });
        }
        if captured != self.graph {
            return Err(CudaConditionalGraphUnavailable::BodyPopulationFailed {
                detail: "capture returned a graph other than the conditional body".to_string(),
            });
        }
        Ok(())
    }
}

/// Builds one dependency-ordered parent graph containing ordinary captured
/// segments and any number of conditional-WHILE nodes.
///
/// This is the topology required by stratified Datalog: work before a
/// recursive strongly connected component executes once, only that component
/// is placed in a device-controlled WHILE, and later strata depend on its
/// completion. The finished value still launches through one `cuGraphLaunch`.
pub struct ConditionalCudaGraphSequenceBuilder {
    graph: UninstantiatedCudaGraph,
    context: Arc<CudaContext>,
    raw_context: sys::CUcontext,
    api: Arc<ConditionalGraphDriverApi>,
    frontier: Vec<sys::CUgraphNode>,
}

impl ConditionalCudaGraphSequenceBuilder {
    /// Create an empty parent graph bound to `stream`'s CUDA context.
    pub fn new(stream: &CudaStream) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        let api = ConditionalGraphDriverApi::load()?;
        let context = stream.context().clone();
        let raw_context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if raw_context != context.cu_ctx() {
            return Err(CudaConditionalGraphUnavailable::ContextMismatch);
        }
        Ok(Self {
            graph: UninstantiatedCudaGraph::create(Arc::clone(&context))?,
            context,
            raw_context,
            api,
            frontier: Vec::new(),
        })
    }

    /// Capture one ordinary graph segment after the current dependency
    /// frontier. The callback may enqueue only capture-compatible operations.
    pub fn capture_segment_on_stream<F, E>(
        &mut self,
        stream: &CudaStream,
        record: F,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable>
    where
        F: FnOnce() -> std::result::Result<(), E>,
        E: fmt::Display,
    {
        self.ensure_stream_context(stream)?;
        let _capture_lease = try_acquire_stream_capture(stream)?;
        unsafe {
            conditional_driver_call(
                "cuStreamBeginCaptureToGraph",
                sys::cuStreamBeginCaptureToGraph(
                    stream.cu_stream(),
                    self.graph.raw(),
                    if self.frontier.is_empty() {
                        ptr::null()
                    } else {
                        self.frontier.as_ptr()
                    },
                    ptr::null(),
                    self.frontier.len(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                ),
            )?;
        }

        let record_result = record();
        let mut captured = ptr::null_mut();
        let end_result = unsafe {
            conditional_driver_call(
                "cuStreamEndCapture",
                sys::cuStreamEndCapture(stream.cu_stream(), &mut captured),
            )
        };
        if let Err(error) = record_result {
            return Err(CudaConditionalGraphUnavailable::body_population(error));
        }
        end_result?;
        if captured != self.graph.raw() {
            return Err(CudaConditionalGraphUnavailable::BodyPopulationFailed {
                detail: "segment capture returned a graph other than its parent".to_string(),
            });
        }
        self.frontier = graph_leaf_nodes(self.graph.raw())?;
        Ok(())
    }

    /// Append one conditional-WHILE node after the current frontier.
    pub fn add_conditional_while<F>(
        &mut self,
        initial_value: u32,
        assign_default_on_launch: bool,
        populate_body: F,
    ) -> std::result::Result<sys::CUgraphConditionalHandle, CudaConditionalGraphUnavailable>
    where
        F: FnOnce(
            ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let mut handle = 0;
        let flags = if assign_default_on_launch {
            sys::CU_GRAPH_COND_ASSIGN_DEFAULT
        } else {
            0
        };
        unsafe {
            conditional_driver_call(
                "cuGraphConditionalHandleCreate",
                (self.api.conditional_handle_create)(
                    &mut handle,
                    self.graph.raw(),
                    self.raw_context,
                    initial_value,
                    flags,
                ),
            )?;
        }

        let mut params = conditional_while_node_params(handle, self.raw_context);
        let mut conditional_node = ptr::null_mut();
        unsafe {
            conditional_driver_call(
                "cuGraphAddNode",
                (self.api.graph_add_node)(
                    &mut conditional_node,
                    self.graph.raw(),
                    if self.frontier.is_empty() {
                        ptr::null()
                    } else {
                        self.frontier.as_ptr()
                    },
                    self.frontier.len(),
                    &mut params,
                ),
            )?;
        }
        if conditional_node.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode",
            });
        }
        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        if conditional.phGraph_out.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode body array",
            });
        }
        let body_graph = unsafe { *conditional.phGraph_out };
        if body_graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode WHILE body",
            });
        }
        populate_body(ConditionalCudaGraphBody {
            graph: body_graph,
            handle,
            context: self.raw_context,
        })?;
        self.frontier.clear();
        self.frontier.push(conditional_node);
        Ok(handle)
    }

    /// Instantiate the complete parent graph exactly once.
    pub fn instantiate(
        self,
    ) -> std::result::Result<CapturedCudaGraph, CudaConditionalGraphUnavailable> {
        let ConditionalCudaGraphSequenceBuilder {
            graph,
            context,
            api,
            ..
        } = self;
        let mut captured =
            unsafe { CapturedCudaGraph::instantiate_owned_graph(graph.into_raw(), context)? };
        captured._conditional_api = Some(api);
        Ok(captured)
    }

    fn ensure_stream_context(
        &self,
        stream: &CudaStream,
    ) -> std::result::Result<(), CudaConditionalGraphUnavailable> {
        let context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if context == self.raw_context {
            Ok(())
        } else {
            Err(CudaConditionalGraphUnavailable::ContextMismatch)
        }
    }
}

fn raw_graph_nodes_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<sys::CUgraphNode>, E> {
    let mut node_count = 0usize;
    unsafe {
        check(
            "cuGraphGetNodes(count)",
            sys::cuGraphGetNodes(graph, ptr::null_mut(), &mut node_count),
        )?;
    }
    let mut nodes = vec![ptr::null_mut(); node_count];
    if node_count != 0 {
        unsafe {
            check(
                "cuGraphGetNodes(nodes)",
                sys::cuGraphGetNodes(graph, nodes.as_mut_ptr(), &mut node_count),
            )?;
        }
        nodes.truncate(node_count);
    }
    Ok(nodes)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum LinearGraphChainError {
    ForeignDependency { node: usize, dependency: usize },
    DuplicateDependency { node: usize, dependency: usize },
    Cycle,
    RootCount { found: usize },
    IncomingDegree { node: usize, dependencies: usize },
    Branch { node: usize, dependents: usize },
    LeafCount { found: usize },
    Disconnected { visited: usize, total: usize },
}

impl fmt::Display for LinearGraphChainError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ForeignDependency { node, dependency } => write!(
                formatter,
                "node {node} depends on foreign enumeration index {dependency}"
            ),
            Self::DuplicateDependency { node, dependency } => write!(
                formatter,
                "node {node} repeats dependency enumeration index {dependency}"
            ),
            Self::Cycle => formatter.write_str("dependency graph contains a cycle"),
            Self::RootCount { found } => {
                write!(
                    formatter,
                    "dependency graph has {found} roots instead of one"
                )
            }
            Self::IncomingDegree { node, dependencies } => write!(
                formatter,
                "non-root node {node} has {dependencies} immediate dependencies instead of one"
            ),
            Self::Branch { node, dependents } => write!(
                formatter,
                "node {node} has {dependents} immediate dependents instead of at most one"
            ),
            Self::LeafCount { found } => {
                write!(
                    formatter,
                    "dependency graph has {found} leaves instead of one"
                )
            }
            Self::Disconnected { visited, total } => write!(
                formatter,
                "dependency chain visits {visited} of {total} enumerated nodes"
            ),
        }
    }
}

fn linear_chain_order(
    immediate_dependencies: &[Vec<usize>],
) -> std::result::Result<Vec<usize>, LinearGraphChainError> {
    let node_count = immediate_dependencies.len();
    let mut outgoing = vec![Vec::new(); node_count];
    for (node, dependencies) in immediate_dependencies.iter().enumerate() {
        let mut unique = HashSet::with_capacity(dependencies.len());
        for &dependency in dependencies {
            if dependency >= node_count {
                return Err(LinearGraphChainError::ForeignDependency { node, dependency });
            }
            if !unique.insert(dependency) {
                return Err(LinearGraphChainError::DuplicateDependency { node, dependency });
            }
            outgoing[dependency].push(node);
        }
    }

    let mut remaining_indegree = immediate_dependencies
        .iter()
        .map(Vec::len)
        .collect::<Vec<_>>();
    let mut ready = remaining_indegree
        .iter()
        .enumerate()
        .filter_map(|(node, &degree)| (degree == 0).then_some(node))
        .collect::<Vec<_>>();
    let mut acyclic_nodes = 0usize;
    while let Some(node) = ready.pop() {
        acyclic_nodes += 1;
        for &dependent in &outgoing[node] {
            remaining_indegree[dependent] -= 1;
            if remaining_indegree[dependent] == 0 {
                ready.push(dependent);
            }
        }
    }
    if acyclic_nodes != node_count {
        return Err(LinearGraphChainError::Cycle);
    }

    let roots = immediate_dependencies
        .iter()
        .enumerate()
        .filter_map(|(node, dependencies)| dependencies.is_empty().then_some(node))
        .collect::<Vec<_>>();
    if roots.len() != 1 {
        return Err(LinearGraphChainError::RootCount { found: roots.len() });
    }
    let root = roots[0];
    for (node, dependencies) in immediate_dependencies.iter().enumerate() {
        if node != root && dependencies.len() != 1 {
            return Err(LinearGraphChainError::IncomingDegree {
                node,
                dependencies: dependencies.len(),
            });
        }
    }
    for (node, dependents) in outgoing.iter().enumerate() {
        if dependents.len() > 1 {
            return Err(LinearGraphChainError::Branch {
                node,
                dependents: dependents.len(),
            });
        }
    }
    let leaf_count = outgoing
        .iter()
        .filter(|dependents| dependents.is_empty())
        .count();
    if leaf_count != 1 {
        return Err(LinearGraphChainError::LeafCount { found: leaf_count });
    }

    let mut order = Vec::with_capacity(node_count);
    let mut visited = vec![false; node_count];
    let mut current = Some(root);
    while let Some(node) = current {
        if visited[node] {
            return Err(LinearGraphChainError::Cycle);
        }
        visited[node] = true;
        order.push(node);
        current = outgoing[node].first().copied();
    }
    if order.len() != node_count {
        return Err(LinearGraphChainError::Disconnected {
            visited: order.len(),
            total: node_count,
        });
    }
    Ok(order)
}

fn raw_node_dependencies_with<E>(
    node: sys::CUgraphNode,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<sys::CUgraphNode>, E> {
    let mut dependency_count = 0usize;
    unsafe {
        check(
            "cuGraphNodeGetDependencies(count)",
            sys::cuGraphNodeGetDependencies(node, ptr::null_mut(), &mut dependency_count),
        )?;
    }
    let mut dependencies = vec![ptr::null_mut(); dependency_count];
    if dependency_count != 0 {
        unsafe {
            check(
                "cuGraphNodeGetDependencies(nodes)",
                sys::cuGraphNodeGetDependencies(
                    node,
                    dependencies.as_mut_ptr(),
                    &mut dependency_count,
                ),
            )?;
        }
        dependencies.truncate(dependency_count);
    }
    Ok(dependencies)
}

fn graph_nodes_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
) -> std::result::Result<Vec<CudaGraphNode>, E> {
    let raw_nodes = raw_graph_nodes_with(graph, check)?;
    let mut nodes = Vec::with_capacity(raw_nodes.len());
    for (index, raw) in raw_nodes.into_iter().enumerate() {
        let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
        unsafe {
            check("cuGraphNodeGetType", sys::cuGraphNodeGetType(raw, &mut ty))?;
        }
        nodes.push(CudaGraphNode {
            index,
            raw,
            kind: CudaGraphNodeKind::from_sys(ty),
        });
    }
    Ok(nodes)
}

fn graph_linear_chain_node_kinds_with<E>(
    graph: sys::CUgraph,
    check: &mut impl FnMut(&'static str, sys::CUresult) -> std::result::Result<(), E>,
    shape_error: &mut impl FnMut(LinearGraphChainError) -> E,
) -> std::result::Result<Vec<CudaGraphNodeKind>, E> {
    let nodes = graph_nodes_with(graph, check)?;
    let mut immediate_dependencies = Vec::with_capacity(nodes.len());
    for (node_index, node) in nodes.iter().enumerate() {
        let raw_dependencies = raw_node_dependencies_with(node.raw, check)?;
        let mut dependency_indices = Vec::with_capacity(raw_dependencies.len());
        for dependency in raw_dependencies {
            let Some(dependency_index) = nodes
                .iter()
                .position(|candidate| candidate.raw == dependency)
            else {
                return Err(shape_error(LinearGraphChainError::ForeignDependency {
                    node: node_index,
                    dependency: nodes.len(),
                }));
            };
            dependency_indices.push(dependency_index);
        }
        immediate_dependencies.push(dependency_indices);
    }
    let order = linear_chain_order(&immediate_dependencies).map_err(shape_error)?;
    Ok(order.into_iter().map(|index| nodes[index].kind).collect())
}

fn graph_leaf_nodes(
    graph: sys::CUgraph,
) -> std::result::Result<Vec<sys::CUgraphNode>, CudaConditionalGraphUnavailable> {
    let mut check = |_, code| conditional_driver_call("cuGraphGetNodes", code);
    let mut nodes = raw_graph_nodes_with(graph, &mut check)?;

    let mut edge_count = 0usize;
    unsafe {
        conditional_driver_call(
            "cuGraphGetEdges",
            sys::cuGraphGetEdges(graph, ptr::null_mut(), ptr::null_mut(), &mut edge_count),
        )?;
    }
    let mut from = vec![ptr::null_mut(); edge_count];
    let mut to = vec![ptr::null_mut(); edge_count];
    if edge_count != 0 {
        unsafe {
            conditional_driver_call(
                "cuGraphGetEdges",
                sys::cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut edge_count),
            )?;
        }
        from.truncate(edge_count);
    }
    nodes.retain(|node| !from.contains(node));
    Ok(nodes)
}

/// Instantiated CUDA Graph with owned graph + exec handles.
pub struct CapturedCudaGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
    context: Arc<CudaContext>,
    _conditional_api: Option<Arc<ConditionalGraphDriverApi>>,
    _resident_lifecycle_lease: Option<Box<dyn Send + Sync>>,
}

// CUDA graph handles are context-owned driver handles. xlog stores them behind
// provider-level synchronization when caching graph executions.
unsafe impl Send for CapturedCudaGraph {}
unsafe impl Sync for CapturedCudaGraph {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CudaGraphNodeKind {
    Kernel,
    Memcpy,
    Memset,
    Host,
    Graph,
    Empty,
    WaitEvent,
    EventRecord,
    ExternalSemaphoresSignal,
    ExternalSemaphoresWait,
    MemAlloc,
    MemFree,
    BatchMemOp,
    Conditional,
}

#[derive(Debug, Clone, Copy)]
pub struct CudaGraphNode {
    pub index: usize,
    pub raw: sys::CUgraphNode,
    pub kind: CudaGraphNodeKind,
}

unsafe impl Send for CudaGraphNode {}
unsafe impl Sync for CudaGraphNode {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CsmCudaGraphJoinKind {
    Inner,
    IndexedInner,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ScanTopology {
    pub input_len: u32,
    pub block_size: u32,
    pub scratch_lengths: Vec<u32>,
    pub kernel_node_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CsmCudaGraphKey {
    pub join_kind: CsmCudaGraphJoinKind,
    pub key_arity: u8,
    pub key_bytes: u32,
    pub probe_capacity_class: u32,
    pub output_capacity_class: u32,
    pub scan_topology: ScanTopology,
    pub node_layout_version: u32,
}

impl CsmCudaGraphKey {
    pub fn inner(
        key_arity: usize,
        key_bytes: u32,
        probe_capacity: u32,
        output_capacity: u32,
    ) -> Result<Self> {
        let key_arity = u8::try_from(key_arity).map_err(|_| {
            XlogError::Kernel(format!(
                "CSM CUDA Graph key arity {} exceeds u8::MAX",
                key_arity
            ))
        })?;
        Ok(Self {
            join_kind: CsmCudaGraphJoinKind::Inner,
            key_arity,
            key_bytes,
            probe_capacity_class: graph_capacity_class_u32(probe_capacity),
            output_capacity_class: graph_capacity_class_u32(output_capacity),
            scan_topology: scan_topology_u32(probe_capacity),
            node_layout_version: CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION,
        })
    }
}

pub fn graph_capacity_class_u32(n: u32) -> u32 {
    if n <= 1 {
        1
    } else {
        n.checked_next_power_of_two().unwrap_or(u32::MAX)
    }
}

pub fn scan_topology_u32(mut n: u32) -> ScanTopology {
    let input_len = n;
    let block_size = 256u32;
    let mut scratch_lengths = Vec::new();
    let mut kernel_node_count = if n == 0 { 0 } else { 1 };
    while n > block_size {
        let num_blocks = n.div_ceil(block_size);
        scratch_lengths.push(num_blocks);
        kernel_node_count += 2;
        n = num_blocks;
    }
    ScanTopology {
        input_len,
        block_size,
        scratch_lengths,
        kernel_node_count,
    }
}

impl CapturedCudaGraph {
    /// Tie resident lifecycle accounting to this real graph/exec owner.
    ///
    /// Binding is idempotent. The lease is created only after graph
    /// instantiation has succeeded and is dropped after this type's `Drop`
    /// implementation destroys the executable and parent graph handles.
    pub fn bind_resident_lifecycle(mut self, runtime: &XlogDeviceRuntime) -> Self {
        if self._resident_lifecycle_lease.is_none() {
            self._resident_lifecycle_lease = Some(Box::new(runtime.resident_graph_handle_lease()));
        }
        self
    }

    /// Create, populate, and instantiate a parent graph containing exactly one
    /// root conditional-WHILE node.
    ///
    /// The handle and body passed to `populate_body` remain valid until this
    /// `CapturedCudaGraph` is dropped. CUDA permits only one live executable
    /// instantiation of a graph containing a conditional node.
    pub fn conditional_while_on_stream<F>(
        stream: &CudaStream,
        initial_value: u32,
        assign_default_on_launch: bool,
        populate_body: F,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable>
    where
        F: FnOnce(
            ConditionalCudaGraphBody,
        ) -> std::result::Result<(), CudaConditionalGraphUnavailable>,
    {
        let api = ConditionalGraphDriverApi::load()?;
        let context = stream.context().clone();
        let raw_context =
            stream_context(stream).map_err(CudaConditionalGraphUnavailable::body_population)?;
        if raw_context != context.cu_ctx() {
            return Err(CudaConditionalGraphUnavailable::ContextMismatch);
        }

        let graph = UninstantiatedCudaGraph::create(Arc::clone(&context))?;
        let mut handle = 0;
        let flags = if assign_default_on_launch {
            sys::CU_GRAPH_COND_ASSIGN_DEFAULT
        } else {
            0
        };
        unsafe {
            conditional_driver_call(
                "cuGraphConditionalHandleCreate",
                (api.conditional_handle_create)(
                    &mut handle,
                    graph.raw(),
                    raw_context,
                    initial_value,
                    flags,
                ),
            )?;
        }

        let mut params = conditional_while_node_params(handle, raw_context);
        let mut conditional_node = ptr::null_mut();
        unsafe {
            conditional_driver_call(
                "cuGraphAddNode",
                (api.graph_add_node)(
                    &mut conditional_node,
                    graph.raw(),
                    ptr::null(),
                    0,
                    &mut params,
                ),
            )?;
        }
        if conditional_node.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode",
            });
        }

        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        if conditional.phGraph_out.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode body array",
            });
        }
        let body_graph = unsafe { *conditional.phGraph_out };
        if body_graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphAddNode WHILE body",
            });
        }
        populate_body(ConditionalCudaGraphBody {
            graph: body_graph,
            handle,
            context: raw_context,
        })?;

        let mut instantiated = unsafe { Self::instantiate_owned_graph(graph.into_raw(), context)? };
        instantiated._conditional_api = Some(api);
        Ok(instantiated)
    }

    /// Instantiate and assume sole ownership of `graph`.
    ///
    /// The graph is destroyed on instantiation failure. On success, this value
    /// destroys the executable first and the source graph second.
    ///
    /// # Safety
    /// `graph` must be a valid, unowned graph in `context`, and no other owner
    /// may destroy or instantiate it while this value exists.
    pub unsafe fn instantiate_owned_graph(
        graph: sys::CUgraph,
        context: Arc<CudaContext>,
    ) -> std::result::Result<Self, CudaConditionalGraphUnavailable> {
        if graph.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "instantiate_owned_graph",
            });
        }
        let owned_graph = UninstantiatedCudaGraph {
            raw: graph,
            context: Arc::clone(&context),
        };
        context
            .bind_to_thread()
            .map_err(CudaConditionalGraphUnavailable::body_population)?;
        let mut exec = ptr::null_mut();
        conditional_driver_call(
            "cuGraphInstantiateWithFlags",
            sys::cuGraphInstantiateWithFlags(&mut exec, owned_graph.raw(), 0),
        )?;
        if exec.is_null() {
            return Err(CudaConditionalGraphUnavailable::NullDriverHandle {
                operation: "cuGraphInstantiateWithFlags",
            });
        }
        Ok(Self {
            graph: owned_graph.into_raw(),
            exec,
            context,
            _conditional_api: None,
            _resident_lifecycle_lease: None,
        })
    }

    /// Capture work submitted by `record` on `stream`, instantiate it, and take
    /// ownership of the resulting graph handles.
    pub fn capture_on_stream<F>(stream: &CudaStream, record: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
        let _capture_lease = try_acquire_stream_capture(stream)
            .map_err(|error| XlogError::Kernel(error.decline_detail()))?;
        unsafe {
            cuda_graph_check(
                "cuStreamBeginCapture_v2",
                sys::cuStreamBeginCapture_v2(
                    stream.cu_stream(),
                    sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
                ),
            )?;
        }

        let record_result = record();
        let mut graph: sys::CUgraph = ptr::null_mut();
        let end_result = unsafe {
            cuda_graph_check(
                "cuStreamEndCapture",
                sys::cuStreamEndCapture(stream.cu_stream(), &mut graph),
            )
        };

        if let Err(record_err) = record_result {
            if end_result.is_ok() && !graph.is_null() {
                unsafe {
                    let _ = sys::cuGraphDestroy(graph);
                }
            }
            return Err(record_err);
        }
        end_result?;
        if graph.is_null() {
            return Err(XlogError::Kernel(
                "cuStreamEndCapture returned a null CUDA graph".to_string(),
            ));
        }

        let mut exec: sys::CUgraphExec = ptr::null_mut();
        unsafe {
            if let Err(err) = cuda_graph_check(
                "cuGraphInstantiateWithFlags",
                sys::cuGraphInstantiateWithFlags(&mut exec, graph, 0),
            ) {
                let _ = sys::cuGraphDestroy(graph);
                return Err(err);
            }
        }
        if exec.is_null() {
            unsafe {
                let _ = sys::cuGraphDestroy(graph);
            }
            return Err(XlogError::Kernel(
                "cuGraphInstantiateWithFlags returned a null CUDA graph exec".to_string(),
            ));
        }

        Ok(Self {
            graph,
            exec,
            context: stream.context().clone(),
            _conditional_api: None,
            _resident_lifecycle_lease: None,
        })
    }

    /// Replay the instantiated graph on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        if stream.context().cu_ctx() != self.context.cu_ctx() {
            return Err(XlogError::Kernel(
                CudaConditionalGraphUnavailable::ContextMismatch.decline_detail(),
            ));
        }
        unsafe {
            cuda_graph_check(
                "cuGraphLaunch",
                sys::cuGraphLaunch(self.exec, stream.cu_stream()),
            )
        }
    }

    /// Number of nodes in the captured graph. Used by bounded CSM CUDA Graph
    /// cache-key and node-inventory certs to prove topology stability.
    pub fn node_count(&self) -> Result<usize> {
        let mut count = 0usize;
        unsafe {
            cuda_graph_check(
                "cuGraphGetNodes(count)",
                sys::cuGraphGetNodes(self.graph, ptr::null_mut(), &mut count),
            )?;
        }
        Ok(count)
    }

    /// Return graph nodes in CUDA's enumeration order with their node type.
    ///
    /// CUDA does not define this list as dependency or execution order. Use
    /// [`Self::linear_chain_node_kinds`] when a linear topology is required.
    pub fn nodes(&self) -> Result<Vec<CudaGraphNode>> {
        let mut check = |operation, code| cuda_graph_check(operation, code);
        graph_nodes_with(self.graph, &mut check)
    }

    /// Return actual node kinds in root-to-leaf dependency order.
    ///
    /// This fails unless the graph is one connected linear chain with exactly
    /// one root, one leaf, one immediate dependency per non-root node, and no
    /// branches, duplicate dependencies, foreign dependencies, or cycles.
    pub fn linear_chain_node_kinds(&self) -> Result<Vec<CudaGraphNodeKind>> {
        let mut check = |operation, code| cuda_graph_check(operation, code);
        let mut shape_error = |error| {
            XlogError::Kernel(format!(
                "CUDA graph is not a linear dependency chain: {error}"
            ))
        };
        graph_linear_chain_node_kinds_with(self.graph, &mut check, &mut shape_error)
    }

    /// Read CUDA's raw kernel-node params for inventory/update code.
    ///
    /// The returned `kernelParams` pointer is CUDA-owned capture metadata. Treat
    /// it as read-only unless constructing a fresh params object for
    /// [`Self::set_kernel_node_params`].
    pub fn kernel_node_params(&self, node: CudaGraphNode) -> Result<sys::CUDA_KERNEL_NODE_PARAMS> {
        if node.kind != CudaGraphNodeKind::Kernel {
            return Err(XlogError::Kernel(format!(
                "kernel_node_params called for non-kernel graph node {:?}",
                node.kind
            )));
        }
        let mut params: sys::CUDA_KERNEL_NODE_PARAMS = unsafe { mem::zeroed() };
        unsafe {
            cuda_graph_check(
                "cuGraphKernelNodeGetParams_v2",
                sys::cuGraphKernelNodeGetParams_v2(node.raw, &mut params),
            )?;
        }
        Ok(params)
    }

    /// Update a kernel node in the instantiated graph.
    ///
    /// # Safety
    /// CUDA requires the replacement params to be topology-compatible with the
    /// captured node. The caller must keep every pointed-to kernel argument
    /// alive until CUDA has consumed the update and launched work that uses it.
    pub unsafe fn set_kernel_node_params(
        &self,
        node: CudaGraphNode,
        params: &sys::CUDA_KERNEL_NODE_PARAMS,
    ) -> Result<()> {
        if node.kind != CudaGraphNodeKind::Kernel {
            return Err(XlogError::Kernel(format!(
                "set_kernel_node_params called for non-kernel graph node {:?}",
                node.kind
            )));
        }
        cuda_graph_check(
            "cuGraphExecKernelNodeSetParams_v2",
            sys::cuGraphExecKernelNodeSetParams_v2(self.exec, node.raw, params),
        )
    }

    /// Read CUDA's raw memset-node params for inventory/update code.
    pub fn memset_node_params(&self, node: CudaGraphNode) -> Result<sys::CUDA_MEMSET_NODE_PARAMS> {
        if node.kind != CudaGraphNodeKind::Memset {
            return Err(XlogError::Kernel(format!(
                "memset_node_params called for non-memset graph node {:?}",
                node.kind
            )));
        }
        let mut params: sys::CUDA_MEMSET_NODE_PARAMS = unsafe { mem::zeroed() };
        unsafe {
            cuda_graph_check(
                "cuGraphMemsetNodeGetParams",
                sys::cuGraphMemsetNodeGetParams(node.raw, &mut params),
            )?;
        }
        Ok(params)
    }

    /// Update a memset node in the instantiated graph.
    pub fn set_memset_node_params(
        &self,
        node: CudaGraphNode,
        params: &sys::CUDA_MEMSET_NODE_PARAMS,
        stream: &CudaStream,
    ) -> Result<()> {
        if node.kind != CudaGraphNodeKind::Memset {
            return Err(XlogError::Kernel(format!(
                "set_memset_node_params called for non-memset graph node {:?}",
                node.kind
            )));
        }
        let ctx = stream_context(stream)?;
        unsafe {
            cuda_graph_check(
                "cuGraphExecMemsetNodeSetParams",
                sys::cuGraphExecMemsetNodeSetParams(self.exec, node.raw, params, ctx),
            )
        }
    }

    /// Raw graph handle for low-level node inventory/update code.
    pub fn graph(&self) -> sys::CUgraph {
        self.graph
    }

    /// Raw instantiated graph handle for low-level graph-exec update code.
    pub fn exec(&self) -> sys::CUgraphExec {
        self.exec
    }
}

impl Drop for CapturedCudaGraph {
    fn drop(&mut self) {
        let _ = self.context.bind_to_thread();
        unsafe {
            if !self.exec.is_null() {
                let _ = sys::cuGraphExecDestroy(self.exec);
            }
            if !self.graph.is_null() {
                let _ = sys::cuGraphDestroy(self.graph);
            }
        }
    }
}

impl CudaGraphNodeKind {
    fn from_sys(kind: sys::CUgraphNodeType) -> Self {
        match kind {
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_KERNEL => Self::Kernel,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMCPY => Self::Memcpy,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEMSET => Self::Memset,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_HOST => Self::Host,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_GRAPH => Self::Graph,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY => Self::Empty,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_WAIT_EVENT => Self::WaitEvent,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EVENT_RECORD => Self::EventRecord,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => {
                Self::ExternalSemaphoresSignal
            }
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT => Self::ExternalSemaphoresWait,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_ALLOC => Self::MemAlloc,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_MEM_FREE => Self::MemFree,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_BATCH_MEM_OP => Self::BatchMemOp,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL => Self::Conditional,
        }
    }
}

fn cuda_graph_check(label: &str, code: sys::CUresult) -> Result<()> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(XlogError::Kernel(format!("{label} failed: {code:?}")))
    }
}

fn stream_context(stream: &CudaStream) -> Result<sys::CUcontext> {
    let mut ctx = ptr::null_mut();
    unsafe {
        cuda_graph_check(
            "cuStreamGetCtx",
            sys::cuStreamGetCtx(stream.cu_stream(), &mut ctx),
        )?;
    }
    if ctx.is_null() {
        Err(XlogError::Kernel(
            "cuStreamGetCtx returned a null CUDA context".to_string(),
        ))
    } else {
        Ok(ctx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::{
        driver::{DevicePtr, LaunchConfig, PushKernelArg},
        nvrtc::compile_ptx,
    };
    use std::sync::Barrier;
    use std::thread;

    #[test]
    fn conditional_graphs_expose_dependency_ordered_linear_chain_inventory() {
        let _: fn(
            &ConditionalCudaGraphBody,
        )
            -> std::result::Result<Vec<CudaGraphNodeKind>, CudaConditionalGraphUnavailable> =
            ConditionalCudaGraphBody::linear_chain_node_kinds;
        let _: fn(&CapturedCudaGraph) -> Result<Vec<CudaGraphNodeKind>> =
            CapturedCudaGraph::linear_chain_node_kinds;
    }

    #[test]
    fn linear_chain_inventory_recovers_dependency_order_from_shuffled_nodes() {
        let enumerated_kinds = [
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
            CudaGraphNodeKind::Kernel,
            CudaGraphNodeKind::Conditional,
        ];
        let immediate_dependencies = vec![vec![2], vec![4], vec![3], vec![], vec![0]];
        let dependency_order = linear_chain_order(&immediate_dependencies).unwrap();
        assert_eq!(dependency_order, vec![3, 2, 0, 4, 1]);
        assert_eq!(
            dependency_order
                .into_iter()
                .map(|index| enumerated_kinds[index])
                .collect::<Vec<_>>(),
            vec![
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
                CudaGraphNodeKind::Conditional,
                CudaGraphNodeKind::Kernel,
            ]
        );
        assert_eq!(linear_chain_order(&[vec![]]).unwrap(), vec![0]);
    }

    #[test]
    fn linear_chain_inventory_rejects_non_linear_dependency_shapes() {
        let cases = [
            (
                "empty graph",
                Vec::new(),
                LinearGraphChainError::RootCount { found: 0 },
            ),
            (
                "branch",
                vec![vec![], vec![0], vec![0]],
                LinearGraphChainError::Branch {
                    node: 0,
                    dependents: 2,
                },
            ),
            (
                "disconnected",
                vec![vec![], vec![]],
                LinearGraphChainError::RootCount { found: 2 },
            ),
            (
                "cycle",
                vec![vec![1], vec![0]],
                LinearGraphChainError::Cycle,
            ),
            (
                "foreign dependency",
                vec![vec![], vec![2]],
                LinearGraphChainError::ForeignDependency {
                    node: 1,
                    dependency: 2,
                },
            ),
            (
                "duplicate edge",
                vec![vec![], vec![0, 0]],
                LinearGraphChainError::DuplicateDependency {
                    node: 1,
                    dependency: 0,
                },
            ),
        ];
        for (case, dependencies, expected) in cases {
            assert_eq!(linear_chain_order(&dependencies), Err(expected), "{case}");
        }
    }

    #[test]
    fn same_stream_capture_registry_is_deterministically_busy_until_release() {
        let key = StreamCaptureKey {
            context: 0x51,
            stream: 0x73,
        };
        let entered = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let first_entered = Arc::clone(&entered);
        let first_release = Arc::clone(&release);
        let first = thread::spawn(move || {
            let _lease = try_acquire_stream_capture_key(key).expect("first capture lease");
            first_entered.wait();
            first_release.wait();
        });
        entered.wait();
        assert_eq!(
            try_acquire_stream_capture_key(key).expect_err("same stream must be busy"),
            CudaConditionalGraphUnavailable::StreamCaptureBusy
        );
        release.wait();
        first.join().expect("first capture thread");
        drop(try_acquire_stream_capture_key(key).expect("capture after release"));
    }

    #[test]
    fn different_stream_capture_registry_entries_can_coexist() {
        let first = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0x91,
            stream: 0x92,
        })
        .expect("first stream capture");
        let second = try_acquire_stream_capture_key(StreamCaptureKey {
            context: 0x91,
            stream: 0x93,
        })
        .expect("different stream capture");
        drop((first, second));
    }

    #[test]
    fn real_capture_helper_returns_typed_busy_before_beginning_driver_capture() {
        let context = match CudaContext::new(0) {
            Ok(context) => context,
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA unavailable: {error}");
                return;
            }
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let _lease = try_acquire_stream_capture(&stream).expect("held stream capture lease");
        let mut builder = match ConditionalCudaGraphSequenceBuilder::new(&stream) {
            Ok(builder) => builder,
            Err(error) if error.is_unsupported() => return,
            Err(error) => panic!("sequence builder failed: {error}"),
        };
        let error = builder
            .capture_segment_on_stream(&stream, || Ok::<(), XlogError>(()))
            .expect_err("same stream capture must decline before driver capture");
        assert_eq!(error, CudaConditionalGraphUnavailable::StreamCaptureBusy);
    }

    #[cfg(unix)]
    #[test]
    fn missing_driver_symbol_is_a_typed_error_instead_of_a_panic() {
        let library = libloading::Library::from(libloading::os::unix::Library::this());
        let error = unsafe {
            load_required_symbol::<ConditionalHandleCreateFn>(
                &library,
                b"xlog_missing_cuda_conditional_symbol_for_test\0",
                "xlog_missing_cuda_conditional_symbol_for_test",
            )
        }
        .expect_err("the deliberately absent symbol must fail closed");

        assert_eq!(
            error,
            CudaConditionalGraphUnavailable::MissingDriverSymbol {
                symbol: "xlog_missing_cuda_conditional_symbol_for_test",
            }
        );
        assert!(error.is_unsupported());
        assert_eq!(
            error.decline_detail(),
            "CUDA driver is missing required conditional-graph symbol \
             xlog_missing_cuda_conditional_symbol_for_test"
        );
    }

    #[test]
    fn driver_versions_before_cuda_twelve_three_decline_conditionals() {
        let error = require_conditional_graph_driver(12_020).expect_err("CUDA 12.2 is too old");
        assert_eq!(
            error,
            CudaConditionalGraphUnavailable::DriverVersionTooOld {
                found: 12_020,
                required: 12_030,
            }
        );
        assert!(error.is_unsupported());
        assert_eq!(
            error.decline_detail(),
            "CUDA conditional graphs require driver API 12030, found 12020"
        );
        require_conditional_graph_driver(12_030).expect("CUDA 12.3 is supported");
    }

    #[test]
    fn while_node_params_use_the_driver_abi_and_return_one_body() {
        let handle = 0x1234_u64;
        let ctx = 0x5678_usize as sys::CUcontext;
        let params = conditional_while_node_params(handle, ctx);

        assert_eq!(
            params.type_,
            sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_CONDITIONAL
        );
        let conditional = unsafe { params.__bindgen_anon_1.conditional };
        assert_eq!(conditional.handle, handle);
        assert_eq!(
            conditional.type_,
            sys::CUgraphConditionalNodeType::CU_GRAPH_COND_TYPE_WHILE
        );
        assert_eq!(conditional.size, 1);
        assert!(conditional.phGraph_out.is_null());
        assert_eq!(conditional.ctx, ctx);
    }

    #[test]
    fn conditional_body_exposes_device_setter_handle_and_context() {
        let graph = 0x1234_usize as sys::CUgraph;
        let handle = 0x5678_u64;
        let context = 0x9abc_usize as sys::CUcontext;
        let body = ConditionalCudaGraphBody {
            graph,
            handle,
            context,
        };

        assert_eq!(body.graph(), graph);
        assert_eq!(body.handle(), handle);
        assert_eq!(body.context(), context);
    }

    #[test]
    fn real_conditional_while_graph_creates_instantiates_and_launches() {
        let context = match CudaContext::new(0) {
            Ok(context) => context,
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA unavailable: {error}");
                return;
            }
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let body_buffer = stream.alloc_zeros::<u32>(1).expect("body buffer");
        let (body_ptr, _body_sync) = body_buffer.device_ptr(&stream);
        let ptx = compile_ptx(
            r#"
            extern "C" __device__ void cudaGraphSetConditional(
                unsigned long long handle,
                unsigned int value
            );

            extern "C" __global__ void run_once(
                unsigned long long handle,
                unsigned int *counter
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    *counter += 1;
                    cudaGraphSetConditional(handle, 0);
                }
            }
            "#,
        )
        .expect("compile conditional setter kernel");
        let module = context.load_module(ptx).expect("load setter module");
        let run_once = module
            .load_function("run_once")
            .expect("load setter function");
        let graph = CapturedCudaGraph::conditional_while_on_stream(&stream, 1, true, |body| {
            body.capture_on_stream(&stream, || {
                let handle = body.handle();
                let mut launch = stream.launch_builder(&run_once);
                launch.arg(&handle).arg(&body_ptr);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })
        });
        let graph = match graph {
            Ok(graph) => graph,
            Err(error) if error.is_unsupported() => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!("CUDA conditional graphs are required: {error}");
                }
                eprintln!("Skipping test: {error}");
                return;
            }
            Err(error) => panic!("conditional graph construction failed: {error}"),
        };

        assert_eq!(graph.node_count().expect("node count"), 1);
        assert_eq!(
            graph.nodes().expect("nodes")[0].kind,
            CudaGraphNodeKind::Conditional
        );
        graph.launch(&stream).expect("conditional graph launch");
        stream.synchronize().expect("conditional graph completion");
        let mut observed = [0_u32; 1];
        stream
            .memcpy_dtoh(&body_buffer, &mut observed)
            .expect("read body effect");
        assert_eq!(observed, [1], "WHILE body must execute exactly once");
    }

    #[test]
    fn real_conditional_sequence_orders_segments_around_multiple_while_capability() {
        let context = match CudaContext::new(0) {
            Ok(context) => context,
            Err(error) if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("XLOG_REQUIRE_CUDA=1 but CUDA setup failed: {error}")
            }
            Err(error) => {
                eprintln!("Skipping test: CUDA unavailable: {error}");
                return;
            }
        };
        let stream = context.new_stream().expect("non-default CUDA stream");
        let buffer = stream.alloc_zeros::<u32>(1).expect("sequence buffer");
        let (buffer_ptr, _buffer_sync) = buffer.device_ptr(&stream);
        let ptx = compile_ptx(
            r#"
            extern "C" __device__ void cudaGraphSetConditional(
                unsigned long long handle,
                unsigned int value
            );

            extern "C" __global__ void add_value(
                unsigned int *counter,
                unsigned int value
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) *counter += value;
            }

            extern "C" __global__ void add_once(
                unsigned long long handle,
                unsigned int *counter
            ) {
                if (blockIdx.x == 0 && threadIdx.x == 0) {
                    *counter += 2;
                    cudaGraphSetConditional(handle, 0);
                }
            }
            "#,
        )
        .expect("compile sequence kernels");
        let module = context.load_module(ptx).expect("load sequence module");
        let add_value = module
            .load_function("add_value")
            .expect("load add function");
        let add_once = module
            .load_function("add_once")
            .expect("load conditional function");

        let sequence = ConditionalCudaGraphSequenceBuilder::new(&stream).and_then(|mut builder| {
            builder.capture_segment_on_stream(&stream, || {
                let value = 1_u32;
                let mut launch = stream.launch_builder(&add_value);
                launch.arg(&buffer_ptr).arg(&value);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })?;
            builder.add_conditional_while(1, true, |body| {
                body.capture_on_stream(&stream, || {
                    let handle = body.handle();
                    let mut launch = stream.launch_builder(&add_once);
                    launch.arg(&handle).arg(&buffer_ptr);
                    unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                        .map(|_| ())
                        .map_err(|error| XlogError::Kernel(error.to_string()))
                })
            })?;
            builder.capture_segment_on_stream(&stream, || {
                let value = 4_u32;
                let mut launch = stream.launch_builder(&add_value);
                launch.arg(&buffer_ptr).arg(&value);
                unsafe { launch.launch(LaunchConfig::for_num_elems(1)) }
                    .map(|_| ())
                    .map_err(|error| XlogError::Kernel(error.to_string()))
            })?;
            builder.instantiate()
        });
        let graph = match sequence {
            Ok(graph) => graph,
            Err(error) if error.is_unsupported() => {
                if std::env::var("XLOG_REQUIRE_CUDA").as_deref() == Ok("1") {
                    panic!("CUDA conditional graphs are required: {error}");
                }
                eprintln!("Skipping test: {error}");
                return;
            }
            Err(error) => panic!("conditional graph sequence construction failed: {error}"),
        };
        assert_eq!(graph.node_count().expect("sequence node count"), 3);
        graph.launch(&stream).expect("sequence graph launch");
        stream.synchronize().expect("sequence graph completion");
        let mut observed = [0_u32; 1];
        stream
            .memcpy_dtoh(&buffer, &mut observed)
            .expect("read sequence effect");
        assert_eq!(observed, [7]);
    }

    #[test]
    fn scan_topology_matches_recursive_multiblock_shape() {
        assert_eq!(
            scan_topology_u32(0),
            ScanTopology {
                input_len: 0,
                block_size: 256,
                scratch_lengths: vec![],
                kernel_node_count: 0,
            }
        );
        assert_eq!(scan_topology_u32(256).scratch_lengths, Vec::<u32>::new());
        assert_eq!(scan_topology_u32(256).kernel_node_count, 1);
        assert_eq!(scan_topology_u32(257).scratch_lengths, vec![2]);
        assert_eq!(scan_topology_u32(257).kernel_node_count, 3);
        assert_eq!(scan_topology_u32(65_537).scratch_lengths, vec![257, 2]);
        assert_eq!(scan_topology_u32(65_537).kernel_node_count, 5);
    }

    #[test]
    fn csm_key_uses_capacity_classes_and_layout_version() {
        let key = CsmCudaGraphKey::inner(2, 16, 257, 513).expect("key");
        assert_eq!(key.join_kind, CsmCudaGraphJoinKind::Inner);
        assert_eq!(key.key_arity, 2);
        assert_eq!(key.key_bytes, 16);
        assert_eq!(key.probe_capacity_class, 512);
        assert_eq!(key.output_capacity_class, 1024);
        assert_eq!(key.scan_topology.scratch_lengths, vec![2]);
        assert_eq!(key.node_layout_version, CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION);
    }
}
