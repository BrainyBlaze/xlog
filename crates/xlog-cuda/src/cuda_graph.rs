//! CUDA Graph RAII helpers for production graph capture/replay.
//!
//! This module intentionally stays close to the CUDA driver API. W66's CSM
//! path needs explicit graph lifetime ownership and node inventory before it can
//! safely update graph-exec parameters for runtime pointers and capacity
//! classes.

use std::{mem, ptr};

use cudarc::driver::{sys, CudaStream};
use xlog_core::{Result, XlogError};

pub const CSM_CUDA_GRAPH_NODE_LAYOUT_VERSION: u32 = 1;

/// Instantiated CUDA Graph with owned graph + exec handles.
pub struct CapturedCudaGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
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
    /// Capture work submitted by `record` on `stream`, instantiate it, and take
    /// ownership of the resulting graph handles.
    pub fn capture_on_stream<F>(stream: &CudaStream, record: F) -> Result<Self>
    where
        F: FnOnce() -> Result<()>,
    {
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

        Ok(Self { graph, exec })
    }

    /// Replay the instantiated graph on `stream`.
    pub fn launch(&self, stream: &CudaStream) -> Result<()> {
        unsafe {
            cuda_graph_check(
                "cuGraphLaunch",
                sys::cuGraphLaunch(self.exec, stream.cu_stream()),
            )
        }
    }

    /// Number of nodes in the captured graph. Used by W66 cache-key and node
    /// inventory certs to prove topology stability.
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

    /// Return graph nodes in CUDA's captured graph order with their node type.
    pub fn nodes(&self) -> Result<Vec<CudaGraphNode>> {
        let count = self.node_count()?;
        if count == 0 {
            return Ok(Vec::new());
        }
        let mut raw_nodes = vec![ptr::null_mut(); count];
        let mut count_again = count;
        unsafe {
            cuda_graph_check(
                "cuGraphGetNodes(nodes)",
                sys::cuGraphGetNodes(self.graph, raw_nodes.as_mut_ptr(), &mut count_again),
            )?;
        }
        raw_nodes.truncate(count_again);

        let mut nodes = Vec::with_capacity(raw_nodes.len());
        for (index, raw) in raw_nodes.into_iter().enumerate() {
            let mut ty = sys::CUgraphNodeType::CU_GRAPH_NODE_TYPE_EMPTY;
            unsafe {
                cuda_graph_check("cuGraphNodeGetType", sys::cuGraphNodeGetType(raw, &mut ty))?;
            }
            nodes.push(CudaGraphNode {
                index,
                raw,
                kind: CudaGraphNodeKind::from_sys(ty),
            });
        }
        Ok(nodes)
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
