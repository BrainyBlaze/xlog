//! CUDA Graph RAII helpers for production graph capture/replay.
//!
//! This module intentionally stays close to the CUDA driver API. W66's CSM
//! path needs explicit graph lifetime ownership and node inventory before it can
//! safely update graph-exec parameters for runtime pointers and capacity
//! classes.

use std::ptr;

use cudarc::driver::{sys, CudaStream};
use xlog_core::{Result, XlogError};

/// Instantiated CUDA Graph with owned graph + exec handles.
pub struct CapturedCudaGraph {
    graph: sys::CUgraph,
    exec: sys::CUgraphExec,
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

fn cuda_graph_check(label: &str, code: sys::CUresult) -> Result<()> {
    if code == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(XlogError::Kernel(format!("{label} failed: {code:?}")))
    }
}
