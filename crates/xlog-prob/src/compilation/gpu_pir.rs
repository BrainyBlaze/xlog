//! GPU-resident Provenance IR (PIR) representation.
//!
//! Mirrors `crate::pir::PirGraph` in a structure-of-arrays layout on device.

use std::sync::Arc;

use cudarc::driver::DeviceSlice;
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::CudaKernelProvider;

use crate::pir::{PirGraph, PirNode, PirNodeId};

/// Node type tags matching `PirNode` variants.
pub const PIR_CONST: u8 = 0;
pub const PIR_LIT: u8 = 1;
pub const PIR_NEG_LIT: u8 = 2;
pub const PIR_AND: u8 = 3;
pub const PIR_OR: u8 = 4;
pub const PIR_DECISION: u8 = 5;

/// GPU-resident PIR graph (device-side mirror of `pir::PirGraph`).
pub struct GpuPirGraph {
    pub node_type: TrackedCudaSlice<u8>,
    pub child_offsets: TrackedCudaSlice<u32>,
    pub children: TrackedCudaSlice<u32>,
    pub leaf_id: TrackedCudaSlice<u32>,
    pub decision_var: TrackedCudaSlice<u32>,
    pub decision_child_false: TrackedCudaSlice<u32>,
    pub decision_child_true: TrackedCudaSlice<u32>,
}

/// GPU-resident PIR root list.
pub struct GpuPirRoots {
    pub roots: TrackedCudaSlice<u32>,
}

impl GpuPirGraph {
    /// Upload a host PIR graph to device buffers.
    ///
    /// This is intended for tests and tooling. Production GPU-native paths
    /// should construct PIR directly on device.
    pub fn from_host(pir: &PirGraph, provider: &Arc<CudaKernelProvider>) -> Result<Self> {
        let num_nodes = pir.len();
        let num_nodes_u32 = u32::try_from(num_nodes).map_err(|_| {
            XlogError::Compilation("GpuPirGraph::from_host: node count overflow".to_string())
        })?;

        let mut node_type: Vec<u8> = Vec::with_capacity(num_nodes);
        let mut child_offsets: Vec<u32> = Vec::with_capacity(num_nodes + 1);
        let mut children: Vec<u32> = Vec::new();
        let mut leaf_id: Vec<u32> = Vec::with_capacity(num_nodes);
        let mut decision_var: Vec<u32> = Vec::with_capacity(num_nodes);
        let mut decision_child_false: Vec<u32> = Vec::with_capacity(num_nodes);
        let mut decision_child_true: Vec<u32> = Vec::with_capacity(num_nodes);

        child_offsets.push(0);

        for (idx, node) in pir.nodes().iter().enumerate() {
            let node_id = u32::try_from(idx).map_err(|_| {
                XlogError::Compilation("GpuPirGraph::from_host: node id overflow".to_string())
            })?;

            match node {
                PirNode::Const(_) => {
                    node_type.push(PIR_CONST);
                    leaf_id.push(0);
                    decision_var.push(0);
                    decision_child_false.push(0);
                    decision_child_true.push(0);
                }
                PirNode::Lit { leaf } => {
                    node_type.push(PIR_LIT);
                    leaf_id.push(leaf.as_u32());
                    decision_var.push(0);
                    decision_child_false.push(0);
                    decision_child_true.push(0);
                }
                PirNode::NegLit { leaf } => {
                    node_type.push(PIR_NEG_LIT);
                    leaf_id.push(leaf.as_u32());
                    decision_var.push(0);
                    decision_child_false.push(0);
                    decision_child_true.push(0);
                }
                PirNode::And { children: kids } => {
                    validate_children_sorted(node_id, kids, num_nodes_u32)?;
                    node_type.push(PIR_AND);
                    leaf_id.push(0);
                    decision_var.push(0);
                    decision_child_false.push(0);
                    decision_child_true.push(0);
                    for &child in kids {
                        children.push(child.as_u32());
                    }
                }
                PirNode::Or { children: kids } => {
                    validate_children_sorted(node_id, kids, num_nodes_u32)?;
                    node_type.push(PIR_OR);
                    leaf_id.push(0);
                    decision_var.push(0);
                    decision_child_false.push(0);
                    decision_child_true.push(0);
                    for &child in kids {
                        children.push(child.as_u32());
                    }
                }
                PirNode::Decision {
                    var,
                    child_false,
                    child_true,
                } => {
                    validate_child_id(node_id, *child_false, num_nodes_u32)?;
                    validate_child_id(node_id, *child_true, num_nodes_u32)?;
                    node_type.push(PIR_DECISION);
                    leaf_id.push(0);
                    decision_var.push(var.as_u32());
                    decision_child_false.push(child_false.as_u32());
                    decision_child_true.push(child_true.as_u32());
                }
            }

            let next_off = u32::try_from(children.len()).map_err(|_| {
                XlogError::Compilation(
                    "GpuPirGraph::from_host: children count exceeds u32".to_string(),
                )
            })?;
            child_offsets.push(next_off);
        }

        if child_offsets.len() != num_nodes + 1 {
            return Err(XlogError::Compilation(
                "GpuPirGraph::from_host: child_offsets length mismatch".to_string(),
            ));
        }

        let memory = provider.memory();
        let device = provider.device().inner();

        let mut d_node_type = memory.alloc::<u8>(node_type.len())?;
        let mut d_child_offsets = memory.alloc::<u32>(child_offsets.len())?;
        let mut d_children = memory.alloc::<u32>(children.len())?;
        let mut d_leaf_id = memory.alloc::<u32>(leaf_id.len())?;
        let mut d_decision_var = memory.alloc::<u32>(decision_var.len())?;
        let mut d_decision_child_false = memory.alloc::<u32>(decision_child_false.len())?;
        let mut d_decision_child_true = memory.alloc::<u32>(decision_child_true.len())?;

        device
            .htod_sync_copy_into(&node_type, &mut d_node_type)
            .map_err(|e| XlogError::Kernel(format!("GpuPirGraph upload node_type: {}", e)))?;
        device
            .htod_sync_copy_into(&child_offsets, &mut d_child_offsets)
            .map_err(|e| XlogError::Kernel(format!("GpuPirGraph upload child_offsets: {}", e)))?;
        device
            .htod_sync_copy_into(&children, &mut d_children)
            .map_err(|e| XlogError::Kernel(format!("GpuPirGraph upload children: {}", e)))?;
        device
            .htod_sync_copy_into(&leaf_id, &mut d_leaf_id)
            .map_err(|e| XlogError::Kernel(format!("GpuPirGraph upload leaf_id: {}", e)))?;
        device
            .htod_sync_copy_into(&decision_var, &mut d_decision_var)
            .map_err(|e| XlogError::Kernel(format!("GpuPirGraph upload decision_var: {}", e)))?;
        device
            .htod_sync_copy_into(&decision_child_false, &mut d_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("GpuPirGraph upload decision_child_false: {}", e))
            })?;
        device
            .htod_sync_copy_into(&decision_child_true, &mut d_decision_child_true)
            .map_err(|e| {
                XlogError::Kernel(format!("GpuPirGraph upload decision_child_true: {}", e))
            })?;

        Ok(Self {
            node_type: d_node_type,
            child_offsets: d_child_offsets,
            children: d_children,
            leaf_id: d_leaf_id,
            decision_var: d_decision_var,
            decision_child_false: d_decision_child_false,
            decision_child_true: d_decision_child_true,
        })
    }

    pub fn num_nodes(&self) -> usize {
        self.node_type.len()
    }
}

impl GpuPirRoots {
    pub fn from_host(roots: &[PirNodeId], provider: &Arc<CudaKernelProvider>) -> Result<Self> {
        let mut host: Vec<u32> = Vec::with_capacity(roots.len());
        for &r in roots {
            host.push(r.as_u32());
        }

        let memory = provider.memory();
        let device = provider.device().inner();
        let mut d_roots = memory.alloc::<u32>(host.len())?;
        device
            .htod_sync_copy_into(&host, &mut d_roots)
            .map_err(|e| XlogError::Kernel(format!("GpuPirRoots upload: {}", e)))?;

        Ok(Self { roots: d_roots })
    }
}

fn validate_child_id(parent: u32, child: PirNodeId, num_nodes: u32) -> Result<()> {
    let id = child.as_u32();
    if id >= num_nodes {
        return Err(XlogError::Compilation(format!(
            "GpuPirGraph::from_host: child {:?} out of bounds for parent {}",
            child, parent
        )));
    }
    Ok(())
}

fn validate_children_sorted(parent: u32, children: &[PirNodeId], num_nodes: u32) -> Result<()> {
    let mut prev: Option<u32> = None;
    for &child in children {
        let id = child.as_u32();
        if id >= num_nodes {
            return Err(XlogError::Compilation(format!(
                "GpuPirGraph::from_host: child {:?} out of bounds for parent {}",
                child, parent
            )));
        }
        if let Some(p) = prev {
            if id <= p {
                return Err(XlogError::Compilation(format!(
                    "GpuPirGraph::from_host: children of {} must be sorted and unique",
                    parent
                )));
            }
        }
        prev = Some(id);
    }
    Ok(())
}
