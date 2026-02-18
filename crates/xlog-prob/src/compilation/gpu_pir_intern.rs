//! GPU PIR interner (device-side hash-consing).
//!
//! Implements deterministic, memory-bounded interning of PIR node batches on GPU.

use std::ffi::c_void;
use std::sync::Arc;

use cudarc::driver::{DeviceRepr, DeviceSlice, LaunchAsync, LaunchConfig};
use xlog_core::{Result, XlogError};
use xlog_cuda::memory::TrackedCudaSlice;
use xlog_cuda::provider::{pir_kernels, scan_kernels, RadixSortScratch, PIR_MODULE, SCAN_MODULE};
use xlog_cuda::CudaKernelProvider;

use crate::compilation::gpu_pir::{GpuPirGraph, PIR_AND, PIR_CONST};

/// Host-side PIR batch (for tests and host-driven workflows).
#[derive(Debug, Clone)]
pub struct PirBatch {
    pub node_type: Vec<u8>,
    pub leaf_id: Vec<u32>,
    pub decision_var: Vec<u32>,
    pub decision_child_false: Vec<u32>,
    pub decision_child_true: Vec<u32>,
    pub child_offsets: Vec<u32>,
    pub children: Vec<u32>,
}

impl PirBatch {
    pub fn len(&self) -> usize {
        self.node_type.len()
    }

    pub fn is_empty(&self) -> bool {
        self.node_type.is_empty()
    }

    /// Build a batch of AND nodes with the given child lists.
    ///
    /// This helper is primarily used by tests.
    pub fn and_or_batch(children: Vec<Vec<u32>>) -> Self {
        let num_nodes = children.len();
        let mut node_type = vec![PIR_AND; num_nodes];
        let leaf_id = vec![0u32; num_nodes];
        let decision_var = vec![0u32; num_nodes];
        let decision_child_false = vec![0u32; num_nodes];
        let decision_child_true = vec![0u32; num_nodes];

        let mut child_offsets = Vec::with_capacity(num_nodes + 1);
        let mut flat_children = Vec::new();
        child_offsets.push(0);
        for kids in children {
            flat_children.extend(kids);
            child_offsets.push(flat_children.len() as u32);
        }

        if node_type.is_empty() {
            node_type = Vec::new();
        }

        Self {
            node_type,
            leaf_id,
            decision_var,
            decision_child_false,
            decision_child_true,
            child_offsets,
            children: flat_children,
        }
    }

    pub fn to_device(&self, provider: &Arc<CudaKernelProvider>) -> Result<GpuPirBatch> {
        let num_nodes = self.node_type.len();
        if self.leaf_id.len() != num_nodes
            || self.decision_var.len() != num_nodes
            || self.decision_child_false.len() != num_nodes
            || self.decision_child_true.len() != num_nodes
        {
            return Err(XlogError::Compilation(
                "PirBatch: array length mismatch".to_string(),
            ));
        }
        if self.child_offsets.len() != num_nodes + 1 {
            return Err(XlogError::Compilation(
                "PirBatch: child_offsets must be len num_nodes+1".to_string(),
            ));
        }
        if let Some(&last) = self.child_offsets.last() {
            if last as usize != self.children.len() {
                return Err(XlogError::Compilation(
                    "PirBatch: child_offsets last entry must equal children len".to_string(),
                ));
            }
        }

        let memory = provider.memory();
        let device = provider.device().inner();

        let mut d_node_type = memory.alloc::<u8>(num_nodes)?;
        let mut d_leaf_id = memory.alloc::<u32>(num_nodes)?;
        let mut d_decision_var = memory.alloc::<u32>(num_nodes)?;
        let mut d_decision_child_false = memory.alloc::<u32>(num_nodes)?;
        let mut d_decision_child_true = memory.alloc::<u32>(num_nodes)?;
        let mut d_child_offsets = memory.alloc::<u32>(self.child_offsets.len())?;
        let mut d_children = memory.alloc::<u32>(self.children.len())?;

        device
            .htod_sync_copy_into(&self.node_type, &mut d_node_type)
            .map_err(|e| XlogError::Kernel(format!("PirBatch upload node_type: {}", e)))?;
        device
            .htod_sync_copy_into(&self.leaf_id, &mut d_leaf_id)
            .map_err(|e| XlogError::Kernel(format!("PirBatch upload leaf_id: {}", e)))?;
        device
            .htod_sync_copy_into(&self.decision_var, &mut d_decision_var)
            .map_err(|e| XlogError::Kernel(format!("PirBatch upload decision_var: {}", e)))?;
        device
            .htod_sync_copy_into(&self.decision_child_false, &mut d_decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("PirBatch upload decision_child_false: {}", e))
            })?;
        device
            .htod_sync_copy_into(&self.decision_child_true, &mut d_decision_child_true)
            .map_err(|e| {
                XlogError::Kernel(format!("PirBatch upload decision_child_true: {}", e))
            })?;
        device
            .htod_sync_copy_into(&self.child_offsets, &mut d_child_offsets)
            .map_err(|e| XlogError::Kernel(format!("PirBatch upload child_offsets: {}", e)))?;
        device
            .htod_sync_copy_into(&self.children, &mut d_children)
            .map_err(|e| XlogError::Kernel(format!("PirBatch upload children: {}", e)))?;

        Ok(GpuPirBatch {
            node_type: d_node_type,
            leaf_id: d_leaf_id,
            decision_var: d_decision_var,
            decision_child_false: d_decision_child_false,
            decision_child_true: d_decision_child_true,
            child_offsets: d_child_offsets,
            children: d_children,
        })
    }
}

/// Device-resident PIR batch.
pub struct GpuPirBatch {
    pub node_type: TrackedCudaSlice<u8>,
    pub leaf_id: TrackedCudaSlice<u32>,
    pub decision_var: TrackedCudaSlice<u32>,
    pub decision_child_false: TrackedCudaSlice<u32>,
    pub decision_child_true: TrackedCudaSlice<u32>,
    pub child_offsets: TrackedCudaSlice<u32>,
    pub children: TrackedCudaSlice<u32>,
}

impl GpuPirBatch {
    pub fn num_nodes(&self) -> usize {
        self.node_type.len()
    }

    pub fn num_children(&self) -> usize {
        self.children.len()
    }
}

/// GPU PIR interner (device-side).
pub struct GpuPirInterner {
    provider: Arc<CudaKernelProvider>,
    node_cap: u32,
    child_cap: u32,
    graph: GpuPirGraph,
    graph_hashes: TrackedCudaSlice<u64>,
    num_nodes: TrackedCudaSlice<u32>,
    num_children: TrackedCudaSlice<u32>,
}

impl GpuPirInterner {
    pub fn new(provider: &Arc<CudaKernelProvider>, node_cap: u32, child_cap: u32) -> Result<Self> {
        if node_cap < 2 {
            return Err(XlogError::Compilation(
                "GpuPirInterner requires node_cap >= 2 for const nodes".to_string(),
            ));
        }
        let memory = provider.memory();
        let device = provider.device().inner();

        let mut node_type = memory.alloc::<u8>(node_cap as usize)?;
        let mut child_offsets = memory.alloc::<u32>((node_cap as usize) + 1)?;
        let mut children = memory.alloc::<u32>(child_cap as usize)?;
        let mut leaf_id = memory.alloc::<u32>(node_cap as usize)?;
        let mut decision_var = memory.alloc::<u32>(node_cap as usize)?;
        let mut decision_child_false = memory.alloc::<u32>(node_cap as usize)?;
        let mut decision_child_true = memory.alloc::<u32>(node_cap as usize)?;
        let mut graph_hashes = memory.alloc::<u64>(node_cap as usize)?;

        device
            .memset_zeros(&mut node_type)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init node_type: {}", e)))?;
        device
            .memset_zeros(&mut child_offsets)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init child_offsets: {}", e)))?;
        device
            .memset_zeros(&mut children)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init children: {}", e)))?;
        device
            .memset_zeros(&mut decision_var)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init decision_var: {}", e)))?;
        device
            .memset_zeros(&mut decision_child_false)
            .map_err(|e| {
                XlogError::Kernel(format!("GpuPirInterner init decision_child_false: {}", e))
            })?;
        device.memset_zeros(&mut decision_child_true).map_err(|e| {
            XlogError::Kernel(format!("GpuPirInterner init decision_child_true: {}", e))
        })?;
        device
            .memset_zeros(&mut graph_hashes)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init graph_hashes: {}", e)))?;

        let mut leaf_id_host = vec![0u32; node_cap as usize];
        if node_cap > 1 {
            leaf_id_host[1] = 1;
        }
        device
            .htod_sync_copy_into(&leaf_id_host, &mut leaf_id)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init leaf_id: {}", e)))?;

        let hash_fn = device
            .get_func(PIR_MODULE, pir_kernels::PIR_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pir_hash_keys not found".to_string()))?;
        let num_const = 2u32;
        let block_size = 256u32;
        let grid_const = (num_const + block_size - 1) / block_size;
        unsafe {
            hash_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_const.max(1), 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &node_type,
                    &leaf_id,
                    &decision_var,
                    &decision_child_false,
                    &decision_child_true,
                    &child_offsets,
                    &children,
                    num_const,
                    &mut graph_hashes,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init hash: {}", e)))?;

        let mut num_nodes = memory.alloc::<u32>(1)?;
        let mut num_children = memory.alloc::<u32>(1)?;
        device
            .htod_sync_copy_into(&[2u32], &mut num_nodes)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init num_nodes: {}", e)))?;
        device
            .htod_sync_copy_into(&[0u32], &mut num_children)
            .map_err(|e| XlogError::Kernel(format!("GpuPirInterner init num_children: {}", e)))?;

        Ok(Self {
            provider: Arc::clone(provider),
            node_cap,
            child_cap,
            graph: GpuPirGraph {
                node_type,
                child_offsets,
                children,
                leaf_id,
                decision_var,
                decision_child_false,
                decision_child_true,
            },
            graph_hashes,
            num_nodes,
            num_children,
        })
    }

    pub fn graph(&self) -> &GpuPirGraph {
        &self.graph
    }

    pub fn intern_batch(&mut self, batch: &PirBatch) -> Result<TrackedCudaSlice<u32>> {
        if batch.node_type.iter().any(|&t| t == PIR_CONST) {
            return Err(XlogError::Compilation(
                "GpuPirInterner does not accept PIR_CONST in batches".to_string(),
            ));
        }
        let mut device_batch = batch.to_device(&self.provider)?;
        self.intern_device_batch(&mut device_batch)
    }

    pub fn intern_device_batch(
        &mut self,
        batch: &mut GpuPirBatch,
    ) -> Result<TrackedCudaSlice<u32>> {
        let num_nodes = batch.num_nodes();
        if num_nodes == 0 {
            return self.provider.memory().alloc::<u32>(0);
        }
        let num_nodes_u32 = u32::try_from(num_nodes).map_err(|_| {
            XlogError::Compilation("GpuPirInterner: num_nodes overflow".to_string())
        })?;
        let num_children = batch.num_children();
        let num_children_u32 = u32::try_from(num_children).map_err(|_| {
            XlogError::Compilation("GpuPirInterner: num_children overflow".to_string())
        })?;

        let device = self.provider.device().inner();
        let memory = self.provider.memory();
        let block_size = 256u32;

        // Canonicalize AND/OR children (sort + dedup) into new buffers.
        let mut canon_child_offsets = memory.alloc::<u32>((num_nodes as usize) + 1)?;
        let mut canon_children = memory.alloc::<u32>(num_children as usize)?;

        if num_children_u32 == 0 {
            device.memset_zeros(&mut canon_child_offsets).map_err(|e| {
                XlogError::Kernel(format!("GpuPirInterner zero child_offsets: {}", e))
            })?;
        } else {
            let mut parent_ids = memory.alloc::<u32>(num_children as usize)?;
            let fill_fn = device
                .get_func(PIR_MODULE, pir_kernels::PIR_FILL_CHILD_PARENTS)
                .ok_or_else(|| XlogError::Kernel("pir_fill_child_parents not found".to_string()))?;
            let grid_nodes = (num_nodes_u32 + block_size - 1) / block_size;
            unsafe {
                fill_fn.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&batch.child_offsets, num_nodes_u32, &mut parent_ids),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("pir_fill_child_parents failed: {}", e)))?;

            let mut sort_scratch = RadixSortScratch::new(&self.provider, num_children_u32)?;
            self.provider.radix_sort_u32_pairs(
                &mut batch.children,
                &mut parent_ids,
                num_children_u32,
                &mut sort_scratch,
            )?;
            self.provider.radix_sort_u32_pairs(
                &mut parent_ids,
                &mut batch.children,
                num_children_u32,
                &mut sort_scratch,
            )?;

            let mut pair_unique_mask = memory.alloc::<u8>(num_children as usize)?;
            let mark_pairs = device
                .get_func(PIR_MODULE, pir_kernels::PIR_MARK_UNIQUE_PAIRS)
                .ok_or_else(|| XlogError::Kernel("pir_mark_unique_pairs not found".to_string()))?;
            let grid_pairs = (num_children_u32 + block_size - 1) / block_size;
            unsafe {
                mark_pairs.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_pairs, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &parent_ids,
                        &batch.children,
                        num_children_u32,
                        &mut pair_unique_mask,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("pir_mark_unique_pairs failed: {}", e)))?;

            let pair_prefix = self
                .provider
                .scan_u8_mask_device(&pair_unique_mask, num_children_u32)?;

            let mut unique_pairs_total = memory.alloc::<u32>(1)?;
            device
                .memset_zeros(&mut unique_pairs_total)
                .map_err(|e| XlogError::Kernel(format!("zero unique_pairs_total: {}", e)))?;
            let count_mask = device
                .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
                .ok_or_else(|| XlogError::Kernel("count_mask not found".to_string()))?;
            unsafe {
                count_mask.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_pairs, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&pair_unique_mask, num_children_u32, &mut unique_pairs_total),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("count_mask (pairs) failed: {}", e)))?;

            let mut canon_parent = memory.alloc::<u32>(num_children as usize)?;
            let compact_pairs = device
                .get_func(PIR_MODULE, pir_kernels::PIR_COMPACT_PAIRS)
                .ok_or_else(|| XlogError::Kernel("pir_compact_pairs not found".to_string()))?;
            unsafe {
                compact_pairs.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_pairs, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &parent_ids,
                        &batch.children,
                        &pair_unique_mask,
                        &pair_prefix,
                        num_children_u32,
                        &mut canon_parent,
                        &mut canon_children,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("pir_compact_pairs failed: {}", e)))?;

            let mut child_counts = memory.alloc::<u32>(num_nodes as usize)?;
            device
                .memset_zeros(&mut child_counts)
                .map_err(|e| XlogError::Kernel(format!("zero child_counts: {}", e)))?;
            let count_children = device
                .get_func(PIR_MODULE, pir_kernels::PIR_COUNT_CHILDREN)
                .ok_or_else(|| XlogError::Kernel("pir_count_children not found".to_string()))?;
            unsafe {
                count_children.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_pairs, 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &canon_parent,
                        &unique_pairs_total,
                        num_nodes_u32,
                        &mut child_counts,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("pir_count_children failed: {}", e)))?;

            self.provider
                .exclusive_scan_u32_inplace(&mut child_counts, num_nodes_u32)?;

            let write_offsets = device
                .get_func(PIR_MODULE, pir_kernels::PIR_WRITE_CHILD_OFFSETS)
                .ok_or_else(|| {
                    XlogError::Kernel("pir_write_child_offsets not found".to_string())
                })?;
            let grid_nodes = (num_nodes_u32 + block_size - 1) / block_size;
            unsafe {
                write_offsets.clone().launch(
                    LaunchConfig {
                        grid_dim: (grid_nodes.max(1), 1, 1),
                        block_dim: (block_size, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &child_counts,
                        num_nodes_u32,
                        &unique_pairs_total,
                        &mut canon_child_offsets,
                    ),
                )
            }
            .map_err(|e| XlogError::Kernel(format!("pir_write_child_offsets failed: {}", e)))?;
        }

        // Hash and pack keys for nodes.
        let mut hashes = memory.alloc::<u64>(num_nodes as usize)?;
        let hash_fn = device
            .get_func(PIR_MODULE, pir_kernels::PIR_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pir_hash_keys not found".to_string()))?;
        let grid_nodes = (num_nodes_u32 + block_size - 1) / block_size;
        unsafe {
            hash_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &batch.node_type,
                    &batch.leaf_id,
                    &batch.decision_var,
                    &batch.decision_child_false,
                    &batch.decision_child_true,
                    &canon_child_offsets,
                    &canon_children,
                    num_nodes_u32,
                    &mut hashes,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_hash_keys failed: {}", e)))?;

        let mut key_tag = memory.alloc::<u32>(num_nodes as usize)?;
        let mut key_payload = memory.alloc::<u32>(num_nodes as usize)?;
        let mut key_child_len = memory.alloc::<u32>(num_nodes as usize)?;
        let pack_fn = device
            .get_func(PIR_MODULE, pir_kernels::PIR_PACK_KEYS)
            .ok_or_else(|| XlogError::Kernel("pir_pack_keys not found".to_string()))?;
        unsafe {
            pack_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &batch.node_type,
                    &batch.leaf_id,
                    &batch.decision_var,
                    &batch.decision_child_false,
                    &batch.decision_child_true,
                    &canon_child_offsets,
                    num_nodes_u32,
                    &mut key_tag,
                    &mut key_payload,
                    &mut key_child_len,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_pack_keys failed: {}", e)))?;

        // Sort indices by (hash, tag, payload, len).
        let mut indices = memory.alloc::<u32>(num_nodes as usize)?;
        self.provider.init_indices(&mut indices, num_nodes_u32)?;
        let mut keys = memory.alloc::<u32>(num_nodes as usize)?;
        let mut node_sort = RadixSortScratch::new(&self.provider, num_nodes_u32)?;

        self.provider
            .gather_u32_by_indices(&key_child_len, &indices, &mut keys, num_nodes_u32)?;
        self.provider.radix_sort_u32_pairs(
            &mut keys,
            &mut indices,
            num_nodes_u32,
            &mut node_sort,
        )?;

        self.provider
            .gather_u32_by_indices(&key_payload, &indices, &mut keys, num_nodes_u32)?;
        self.provider.radix_sort_u32_pairs(
            &mut keys,
            &mut indices,
            num_nodes_u32,
            &mut node_sort,
        )?;

        self.provider
            .gather_u32_by_indices(&key_tag, &indices, &mut keys, num_nodes_u32)?;
        self.provider.radix_sort_u32_pairs(
            &mut keys,
            &mut indices,
            num_nodes_u32,
            &mut node_sort,
        )?;

        self.provider
            .gather_u64_lo_by_indices(&hashes, &indices, &mut keys, num_nodes_u32)?;
        self.provider.radix_sort_u32_pairs(
            &mut keys,
            &mut indices,
            num_nodes_u32,
            &mut node_sort,
        )?;

        self.provider
            .gather_u64_hi_by_indices(&hashes, &indices, &mut keys, num_nodes_u32)?;
        self.provider.radix_sort_u32_pairs(
            &mut keys,
            &mut indices,
            num_nodes_u32,
            &mut node_sort,
        )?;

        // Gather sorted node arrays.
        let mut sorted_node_type = memory.alloc::<u8>(num_nodes as usize)?;
        let mut sorted_leaf_id = memory.alloc::<u32>(num_nodes as usize)?;
        let mut sorted_decision_var = memory.alloc::<u32>(num_nodes as usize)?;
        let mut sorted_decision_child_false = memory.alloc::<u32>(num_nodes as usize)?;
        let mut sorted_decision_child_true = memory.alloc::<u32>(num_nodes as usize)?;

        self.provider.gather_u8_by_indices(
            &batch.node_type,
            &indices,
            &mut sorted_node_type,
            num_nodes_u32,
        )?;
        self.provider.gather_u32_by_indices(
            &batch.leaf_id,
            &indices,
            &mut sorted_leaf_id,
            num_nodes_u32,
        )?;
        self.provider.gather_u32_by_indices(
            &batch.decision_var,
            &indices,
            &mut sorted_decision_var,
            num_nodes_u32,
        )?;
        self.provider.gather_u32_by_indices(
            &batch.decision_child_false,
            &indices,
            &mut sorted_decision_child_false,
            num_nodes_u32,
        )?;
        self.provider.gather_u32_by_indices(
            &batch.decision_child_true,
            &indices,
            &mut sorted_decision_child_true,
            num_nodes_u32,
        )?;

        // Build sorted child offsets/children.
        let mut sorted_child_len = memory.alloc::<u32>(num_nodes as usize)?;
        self.provider.gather_u32_by_indices(
            &key_child_len,
            &indices,
            &mut sorted_child_len,
            num_nodes_u32,
        )?;
        self.provider
            .exclusive_scan_u32_inplace(&mut sorted_child_len, num_nodes_u32)?;

        let mut sorted_child_offsets = memory.alloc::<u32>((num_nodes as usize) + 1)?;
        let write_offsets = device
            .get_func(PIR_MODULE, pir_kernels::PIR_WRITE_CHILD_OFFSETS)
            .ok_or_else(|| XlogError::Kernel("pir_write_child_offsets not found".to_string()))?;
        let total_children_view = canon_child_offsets.slice(num_nodes..(num_nodes + 1));
        unsafe {
            write_offsets.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes.max(1), 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &sorted_child_len,
                    num_nodes_u32,
                    &total_children_view,
                    &mut sorted_child_offsets,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_write_child_offsets(sorted) failed: {}", e)))?;

        let mut sorted_children = memory.alloc::<u32>(num_children as usize)?;
        let gather_children = device
            .get_func(PIR_MODULE, pir_kernels::PIR_GATHER_CHILDREN)
            .ok_or_else(|| XlogError::Kernel("pir_gather_children not found".to_string()))?;
        unsafe {
            gather_children.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &indices,
                    &canon_child_offsets,
                    &canon_children,
                    &sorted_child_offsets,
                    num_nodes_u32,
                    &mut sorted_children,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_gather_children failed: {}", e)))?;

        // Recompute hashes in sorted order for uniqueness checks.
        let mut sorted_hashes = memory.alloc::<u64>(num_nodes as usize)?;
        let hash_fn = device
            .get_func(PIR_MODULE, pir_kernels::PIR_HASH_KEYS)
            .ok_or_else(|| XlogError::Kernel("pir_hash_keys not found".to_string()))?;
        unsafe {
            hash_fn.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &sorted_node_type,
                    &sorted_leaf_id,
                    &sorted_decision_var,
                    &sorted_decision_child_false,
                    &sorted_decision_child_true,
                    &sorted_child_offsets,
                    &sorted_children,
                    num_nodes_u32,
                    &mut sorted_hashes,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_hash_keys(sorted) failed: {}", e)))?;

        let hash_table = self
            .provider
            .build_hash_table_u64(&self.graph_hashes, self.node_cap)?;
        let mut existing_id = memory.alloc::<u32>(num_nodes as usize)?;
        let find_existing = device
            .get_func(PIR_MODULE, pir_kernels::PIR_FIND_EXISTING)
            .ok_or_else(|| XlogError::Kernel("pir_find_existing not found".to_string()))?;
        let mut find_params: Vec<*mut c_void> = vec![
            (&sorted_hashes).as_kernel_param(),
            (&sorted_node_type).as_kernel_param(),
            (&sorted_leaf_id).as_kernel_param(),
            (&sorted_decision_var).as_kernel_param(),
            (&sorted_decision_child_false).as_kernel_param(),
            (&sorted_decision_child_true).as_kernel_param(),
            (&sorted_child_offsets).as_kernel_param(),
            (&sorted_children).as_kernel_param(),
            num_nodes_u32.as_kernel_param(),
            (&self.graph.node_type).as_kernel_param(),
            (&self.graph.child_offsets).as_kernel_param(),
            (&self.graph.children).as_kernel_param(),
            (&self.graph.leaf_id).as_kernel_param(),
            (&self.graph.decision_var).as_kernel_param(),
            (&self.graph.decision_child_false).as_kernel_param(),
            (&self.graph.decision_child_true).as_kernel_param(),
            (&self.num_nodes).as_kernel_param(),
            (&hash_table.bucket_offsets).as_kernel_param(),
            (&hash_table.bucket_counts).as_kernel_param(),
            (&hash_table.bucket_entries).as_kernel_param(),
            (&hash_table.bucket_entry_hashes).as_kernel_param(),
            hash_table.bucket_mask.as_kernel_param(),
            (&mut existing_id).as_kernel_param(),
        ];
        unsafe {
            find_existing.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut find_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_find_existing failed: {}", e)))?;

        // Mark unique nodes in sorted order.
        let mut unique_mask = memory.alloc::<u8>(num_nodes as usize)?;
        let mark_unique = device
            .get_func(PIR_MODULE, pir_kernels::PIR_MARK_UNIQUE)
            .ok_or_else(|| XlogError::Kernel("pir_mark_unique not found".to_string()))?;
        unsafe {
            mark_unique.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &sorted_hashes,
                    &sorted_node_type,
                    &sorted_leaf_id,
                    &sorted_decision_var,
                    &sorted_decision_child_false,
                    &sorted_decision_child_true,
                    &sorted_child_offsets,
                    &sorted_children,
                    num_nodes_u32,
                    &mut unique_mask,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_mark_unique failed: {}", e)))?;

        let mut new_mask = memory.alloc::<u8>(num_nodes as usize)?;
        let mark_new = device
            .get_func(PIR_MODULE, pir_kernels::PIR_MARK_NEW_GROUPS)
            .ok_or_else(|| XlogError::Kernel("pir_mark_new_groups not found".to_string()))?;
        unsafe {
            mark_new.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&unique_mask, &existing_id, num_nodes_u32, &mut new_mask),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_mark_new_groups failed: {}", e)))?;

        let unique_prefix = self
            .provider
            .scan_u8_mask_device(&unique_mask, num_nodes_u32)?;

        let new_prefix = self
            .provider
            .scan_u8_mask_device(&new_mask, num_nodes_u32)?;

        let mut new_nodes_total = memory.alloc::<u32>(1)?;
        device
            .memset_zeros(&mut new_nodes_total)
            .map_err(|e| XlogError::Kernel(format!("zero new_nodes_total: {}", e)))?;
        let count_mask = device
            .get_func(SCAN_MODULE, scan_kernels::COUNT_MASK)
            .ok_or_else(|| XlogError::Kernel("count_mask not found".to_string()))?;
        unsafe {
            count_mask.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&new_mask, num_nodes_u32, &mut new_nodes_total),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("count_mask (new nodes) failed: {}", e)))?;

        let mut group_node_id = memory.alloc::<u32>(num_nodes as usize)?;
        let build_group = device
            .get_func(PIR_MODULE, pir_kernels::PIR_BUILD_GROUP_IDS)
            .ok_or_else(|| XlogError::Kernel("pir_build_group_ids not found".to_string()))?;
        unsafe {
            build_group.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &unique_mask,
                    &unique_prefix,
                    &existing_id,
                    &new_prefix,
                    &self.num_nodes,
                    num_nodes_u32,
                    &mut group_node_id,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_build_group_ids failed: {}", e)))?;

        let mut graph_child_counts = memory.alloc::<u32>(num_nodes as usize)?;
        let build_counts = device
            .get_func(PIR_MODULE, pir_kernels::PIR_BUILD_GRAPH_CHILD_COUNTS)
            .ok_or_else(|| {
                XlogError::Kernel("pir_build_graph_child_counts not found".to_string())
            })?;
        unsafe {
            build_counts.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &sorted_node_type,
                    &sorted_child_offsets,
                    &new_mask,
                    num_nodes_u32,
                    &mut graph_child_counts,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_build_graph_child_counts failed: {}", e)))?;

        let mut graph_children_total = memory.alloc::<u32>(1)?;
        device
            .memset_zeros(&mut graph_children_total)
            .map_err(|e| XlogError::Kernel(format!("zero graph_children_total: {}", e)))?;
        let sum_counts = device
            .get_func(PIR_MODULE, pir_kernels::PIR_SUM_COUNTS)
            .ok_or_else(|| XlogError::Kernel("pir_sum_counts not found".to_string()))?;
        unsafe {
            sum_counts.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &graph_child_counts,
                    num_nodes_u32,
                    &mut graph_children_total,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_sum_counts failed: {}", e)))?;

        self.provider
            .exclusive_scan_u32_inplace(&mut graph_child_counts, num_nodes_u32)?;

        let mut out_ids = memory.alloc::<u32>(num_nodes as usize)?;
        let emit = device
            .get_func(PIR_MODULE, pir_kernels::PIR_EMIT_NODES_AND_IDS)
            .ok_or_else(|| XlogError::Kernel("pir_emit_nodes_and_ids not found".to_string()))?;
        let mut emit_params: Vec<*mut c_void> = vec![
            (&sorted_node_type).as_kernel_param(),
            (&sorted_leaf_id).as_kernel_param(),
            (&sorted_decision_var).as_kernel_param(),
            (&sorted_decision_child_false).as_kernel_param(),
            (&sorted_decision_child_true).as_kernel_param(),
            (&sorted_child_offsets).as_kernel_param(),
            (&sorted_children).as_kernel_param(),
            (&unique_mask).as_kernel_param(),
            (&unique_prefix).as_kernel_param(),
            (&group_node_id).as_kernel_param(),
            (&graph_child_counts).as_kernel_param(),
            (&indices).as_kernel_param(),
            num_nodes_u32.as_kernel_param(),
            (&self.num_nodes).as_kernel_param(),
            (&self.num_children).as_kernel_param(),
            self.node_cap.as_kernel_param(),
            self.child_cap.as_kernel_param(),
            (&mut self.graph.node_type).as_kernel_param(),
            (&mut self.graph.child_offsets).as_kernel_param(),
            (&mut self.graph.children).as_kernel_param(),
            (&mut self.graph.leaf_id).as_kernel_param(),
            (&mut self.graph.decision_var).as_kernel_param(),
            (&mut self.graph.decision_child_false).as_kernel_param(),
            (&mut self.graph.decision_child_true).as_kernel_param(),
            (&new_mask).as_kernel_param(),
            (&sorted_hashes).as_kernel_param(),
            (&mut self.graph_hashes).as_kernel_param(),
            (&mut out_ids).as_kernel_param(),
        ];
        unsafe {
            emit.clone().launch(
                LaunchConfig {
                    grid_dim: (grid_nodes, 1, 1),
                    block_dim: (block_size, 1, 1),
                    shared_mem_bytes: 0,
                },
                &mut emit_params,
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_emit_nodes_and_ids failed: {}", e)))?;

        let update_counts = device
            .get_func(PIR_MODULE, pir_kernels::PIR_UPDATE_COUNTS)
            .ok_or_else(|| XlogError::Kernel("pir_update_counts not found".to_string()))?;
        unsafe {
            update_counts.clone().launch(
                LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (1, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &new_nodes_total,
                    &graph_children_total,
                    self.node_cap,
                    self.child_cap,
                    &mut self.num_nodes,
                    &mut self.num_children,
                ),
            )
        }
        .map_err(|e| XlogError::Kernel(format!("pir_update_counts failed: {}", e)))?;
        // No device synchronize: returns device-resident IDs; same-stream ordering suffices.
        Ok(out_ids)
    }
}
