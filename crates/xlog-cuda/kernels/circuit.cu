// kernels/circuit.cu
#include <cstdint>
#include <cstddef>
#include <cmath>
#include "semantic_measure.cuh"

/**
 * XGCF Circuit Evaluation Kernels
 *
 * This module evaluates Decision-DNNF circuits lowered to XGCF in log-space.
 *
 * Node type encoding is owned by xlog-ir/src/semantic_measure.rs:
 * 0 = Const0
 * 1 = Const1
 * 2 = Lit
 * 3 = And
 * 4 = Or
 * 5 = Decision
 */

using semantic_measure::XGCF_CONST0;
using semantic_measure::XGCF_CONST1;
using semantic_measure::XGCF_LIT;
using semantic_measure::XGCF_AND;
using semantic_measure::XGCF_OR;
using semantic_measure::XGCF_DECISION;
using semantic_measure::logsumexp2;

#define BLOCK_SIZE 256

__device__ __forceinline__ size_t slot_offset(uint32_t slot, uint32_t stride) {
    return static_cast<size_t>(slot) * static_cast<size_t>(stride);
}

/**
 * Evaluate one topological level of an XGCF circuit.
 *
 * Each thread computes one node value for the provided list of node indices.
 * Nodes in the same level have all dependencies in earlier levels.
 *
 * Inputs:
 * - node_type: u8 node kind per node
 * - child_offsets/child_indices: CSR adjacency for AND/OR nodes
 * - lit: i32 literal per node (for LIT nodes, +/- var id)
 * - decision_var/decision_child_{false,true}: decision node fields
 * - level_nodes: node indices belonging to this level (length = num_level_nodes)
 * - var_log_true/var_log_false: log-space weights per CNF variable (1-based, index 0 unused)
 *
 * Output:
 * - values: per-node log-space value array
 */
extern "C" __global__ void xgcf_forward_level(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    double* __restrict__ values
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = level_offsets[level];
    uint32_t num_level_nodes = level_offsets[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes[level_offset + tid];
    const semantic_measure::CircuitView circuit{
        node_type, child_offsets, child_indices, lit,
        decision_var, decision_child_false, decision_child_true};
    values[node] = semantic_measure::forward_node(
        circuit, node, var_log_true, var_log_false, values);
}

/**
 * Backward pass: propagate reverse-mode adjoints to children for one level.
 *
 * This kernel performs only structural propagation for:
 * - AND: adj[child] += adj[parent]
 * - OR:  adj[child] += adj[parent] * exp(values[child] - values[parent])
 * - DECISION: adj[child_f] += adj[parent] * exp(log_f + values[child_f] - values[parent])
 *             adj[child_t] += adj[parent] * exp(log_t + values[child_t] - values[parent])
 *
 * Weight gradients for DECISION/LIT are handled by separate kernels to stay within
 * the host launch argument tuple limit.
 */
extern "C" __global__ void xgcf_backward_level_propagate(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    double* __restrict__ adj
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = level_offsets[level];
    uint32_t num_level_nodes = level_offsets[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes[level_offset + tid];
    const semantic_measure::CircuitView circuit{
        node_type, child_offsets, child_indices, nullptr, decision_var,
        decision_child_false, decision_child_true
    };
    semantic_measure::propagate_node_adjoint(
        circuit, node, adj[node], var_log_true, var_log_false, values,
        semantic_measure::AtomicNodeAdjoints{adj});
}

/**
 * Backward pass: accumulate gradients for DECISION nodes (per-level).
 *
 * For each DECISION node, compute the normalized branch probabilities and
 * accumulate:
 *   grad_false[var] += adj[node] * p_false
 *   grad_true[var]  += adj[node] * p_true
 *
 * Where p_false = exp(log_f + values[child_f] - values[node]) and similarly for true.
 */
extern "C" __global__ void xgcf_backward_level_decision_grad(
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = level_offsets[level];
    uint32_t num_level_nodes = level_offsets[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes[level_offset + tid];
    if (node_type[node] != XGCF_DECISION) {
        return;
    }
    const semantic_measure::CircuitView circuit{
        node_type, nullptr, nullptr, nullptr, decision_var,
        decision_child_false, decision_child_true
    };
    semantic_measure::accumulate_node_weight_gradients(
        circuit, node, adj[node], var_log_true, var_log_false, values,
        semantic_measure::AtomicWeightGradients{grad_true, grad_false});
}

/**
 * Backward pass: accumulate gradients for LIT nodes (per-level).
 *
 * For each LIT node, accumulate:
 *   grad_true[var]  += adj[node]  (if lit > 0)
 *   grad_false[var] += adj[node]  (if lit < 0)
 */
extern "C" __global__ void xgcf_backward_level_lit_grad(
    const uint8_t* __restrict__ node_type,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = level_offsets[level];
    uint32_t num_level_nodes = level_offsets[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes[level_offset + tid];
    if (node_type[node] != XGCF_LIT) {
        return;
    }
    const semantic_measure::CircuitView circuit{
        node_type, nullptr, nullptr, lit, nullptr, nullptr, nullptr
    };
    const double* unused_values = nullptr;
    semantic_measure::accumulate_node_weight_gradients(
        circuit, node, adj[node], nullptr, nullptr, unused_values,
        semantic_measure::AtomicWeightGradients{grad_true, grad_false});
}

/**
 * Apply free-variable gradient corrections on device.
 *
 * For each var with free_var_mask[var] != 0:
 *   grad_true[var]  += softmax_true(log_true[var], log_false[var])
 *   grad_false[var] += softmax_false(log_true[var], log_false[var])
 */
extern "C" __global__ void xgcf_free_var_apply_grad(
    const uint8_t* __restrict__ free_var_mask,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    uint32_t n,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t var = tid; var < n; var += blockDim.x * gridDim.x) {
        if (var == 0u || free_var_mask[var] == 0u) {
            continue;
        }
        double t = var_log_true[var];
        double f = var_log_false[var];
        double m = (t > f) ? t : f;
        if (isinf(m) && m < 0.0) {
            continue;
        }
        double et = exp(t - m);
        double ef = exp(f - m);
        double sum = et + ef;
        if (sum == 0.0) {
            continue;
        }
        double pt = et / sum;
        double pf = ef / sum;
        grad_true[var] += pt;
        grad_false[var] += pf;
    }
}

/**
 * Deterministic pairwise reduction for free-variable logZ correction.
 *
 * mode = 0: compute terms from (mask, weights), ignore in_vals.
 * mode = 1: reduce in_vals pairwise, ignore mask/weights.
 */
extern "C" __global__ void xgcf_free_var_reduce_stage(
    const uint8_t* __restrict__ free_var_mask,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ in_vals,
    uint32_t n,
    uint32_t mode,
    double* __restrict__ out_vals
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_len = (n + 1u) / 2u;
    if (tid >= out_len) {
        return;
    }

    uint32_t i0 = tid * 2u;
    uint32_t i1 = i0 + 1u;

    double a = 0.0;
    double b = 0.0;

    if (mode == 0u) {
        if (i0 < n && i0 != 0u && free_var_mask[i0] != 0u) {
            a = logsumexp2(var_log_true[i0], var_log_false[i0]);
        }
        if (i1 < n && i1 != 0u && free_var_mask[i1] != 0u) {
            b = logsumexp2(var_log_true[i1], var_log_false[i1]);
        }
    } else {
        if (i0 < n) {
            a = in_vals[i0];
        }
        if (i1 < n) {
            b = in_vals[i1];
        }
    }

    out_vals[tid] = a + b;
}

/**
 * Add a device scalar to a single values[index].
 */
extern "C" __global__ void xgcf_add_scalar(
    double* __restrict__ values,
    uint32_t index,
    const double* __restrict__ scalar
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    values[index] += scalar[0];
}

// Cached variants (device-slot addressing).

extern "C" __global__ void xgcf_forward_level_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    double* __restrict__ values
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* var_log_true_s = var_log_true + var_base;
    const double* var_log_false_s = var_log_false + var_base;
    double* values_s = values + node_base;

    uint32_t level_offset = level_offsets_s[level];
    uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes_s[level_offset + tid];
    const semantic_measure::CircuitView circuit{
        node_type_s, child_offsets_s, child_indices_s, lit_s,
        decision_var_s, decision_child_false_s, decision_child_true_s};
    values_s[node] = semantic_measure::forward_node(
        circuit, node, var_log_true_s, var_log_false_s, values_s);
}

// Evaluate all levels in a single kernel (cached layout).
extern "C" __global__ void xgcf_eval_all_levels_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    double* __restrict__ values,
    const uint32_t* __restrict__ meta_num_levels
) {
    if (blockIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* var_log_true_s = var_log_true + var_base;
    const double* var_log_false_s = var_log_false + var_base;
    double* values_s = values + node_base;

    uint32_t num_levels = meta_num_levels[slot];
    for (uint32_t level = 0; level < num_levels; ++level) {
        uint32_t level_offset = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_offset + idx];
            const semantic_measure::CircuitView circuit{
                node_type_s, child_offsets_s, child_indices_s, lit_s,
                decision_var_s, decision_child_false_s, decision_child_true_s};
            values_s[node] = semantic_measure::forward_node(
                circuit, node, var_log_true_s, var_log_false_s, values_s);
        }
        __syncthreads();
    }
}

// Evaluate all levels for a batch of query-specific weight tables while reusing
// the same cached circuit topology slot.
// - One CUDA block handles one query (blockIdx.x = query index).
// - Topology and level metadata come from active_slot[0].
// - var/value buffers are laid out as [batch][stride].
extern "C" __global__ void xgcf_eval_all_levels_cached_batched(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    const uint32_t* __restrict__ meta_num_levels,
    const double* __restrict__ var_log_true_batch,
    const double* __restrict__ var_log_false_batch,
    uint32_t var_stride,
    double* __restrict__ values_batch,
    uint32_t value_stride,
    uint32_t batch_size
) {
    uint32_t q = blockIdx.x;
    if (q >= batch_size) {
        return;
    }

    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;

    size_t q_var_base = static_cast<size_t>(q) * static_cast<size_t>(var_stride);
    size_t q_value_base = static_cast<size_t>(q) * static_cast<size_t>(value_stride);
    const double* var_log_true_q = var_log_true_batch + q_var_base;
    const double* var_log_false_q = var_log_false_batch + q_var_base;
    double* values_q = values_batch + q_value_base;

    uint32_t num_levels = meta_num_levels[slot];
    for (uint32_t level = 0; level < num_levels; ++level) {
        uint32_t level_offset = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_offset + idx];
            const semantic_measure::CircuitView circuit{
                node_type_s, child_offsets_s, child_indices_s, lit_s,
                decision_var_s, decision_child_false_s, decision_child_true_s};
            values_q[node] = semantic_measure::forward_node(
                circuit, node, var_log_true_q, var_log_false_q, values_q);
        }
        __syncthreads();
    }
}

// Set adj[root] = 1.0 for each query batch row.
extern "C" __global__ void xgcf_set_root_adj_cached_batched(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    const uint32_t* __restrict__ meta_root,
    double* __restrict__ adj_batch,
    uint32_t adj_stride,
    uint32_t batch_size
) {
    uint32_t q = blockIdx.x;
    if (q >= batch_size || threadIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    uint32_t root = meta_root[slot];
    size_t q_adj_base = static_cast<size_t>(q) * static_cast<size_t>(adj_stride);
    (void)node_cap;
    adj_batch[q_adj_base + root] = 1.0;
}

// Fused backward pass over all levels for batched query rows.
extern "C" __global__ void xgcf_backward_all_levels_cached_batched(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    const uint32_t* __restrict__ meta_num_levels,
    const double* __restrict__ var_log_true_batch,
    const double* __restrict__ var_log_false_batch,
    uint32_t var_stride,
    const double* __restrict__ values_batch,
    uint32_t value_stride,
    double* __restrict__ adj_batch,
    uint32_t adj_stride,
    double* __restrict__ grad_true_batch,
    double* __restrict__ grad_false_batch,
    uint32_t grad_stride,
    uint32_t batch_size
) {
    uint32_t q = blockIdx.x;
    if (q >= batch_size) {
        return;
    }

    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;

    size_t q_var_base = static_cast<size_t>(q) * static_cast<size_t>(var_stride);
    size_t q_value_base = static_cast<size_t>(q) * static_cast<size_t>(value_stride);
    size_t q_adj_base = static_cast<size_t>(q) * static_cast<size_t>(adj_stride);
    size_t q_grad_base = static_cast<size_t>(q) * static_cast<size_t>(grad_stride);
    const double* var_log_true_q = var_log_true_batch + q_var_base;
    const double* var_log_false_q = var_log_false_batch + q_var_base;
    const double* values_q = values_batch + q_value_base;
    double* adj_q = adj_batch + q_adj_base;
    double* grad_true_q = grad_true_batch + q_grad_base;
    double* grad_false_q = grad_false_batch + q_grad_base;

    const semantic_measure::CircuitView circuit{
        node_type_s, child_offsets_s, child_indices_s, lit_s, decision_var_s,
        decision_child_false_s, decision_child_true_s
    };
    uint32_t num_levels = meta_num_levels[slot];
    for (int32_t level = static_cast<int32_t>(num_levels) - 1; level >= 0; --level) {
        uint32_t level_off = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_off;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_off + idx];
            double adjoint = adj_q[node];
            semantic_measure::propagate_node_adjoint(
                circuit, node, adjoint, var_log_true_q, var_log_false_q, values_q,
                semantic_measure::AtomicNodeAdjoints{adj_q});
            semantic_measure::accumulate_node_weight_gradients(
                circuit, node, adjoint, var_log_true_q, var_log_false_q, values_q,
                semantic_measure::AtomicWeightGradients{grad_true_q, grad_false_q});
        }
        __syncthreads();
    }
}

extern "C" __global__ void xgcf_copy_root_cached_meta_batched(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    const uint32_t* __restrict__ meta_root,
    const double* __restrict__ values_batch,
    uint32_t value_stride,
    double* __restrict__ out_log_z,
    uint32_t batch_size
) {
    uint32_t q = blockIdx.x;
    if (q >= batch_size || threadIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    uint32_t root = meta_root[slot];
    size_t q_value_base = static_cast<size_t>(q) * static_cast<size_t>(value_stride);
    (void)node_cap;
    out_log_z[q] = values_batch[q_value_base + root];
}

extern "C" __global__ void xgcf_copy_root_cached_meta(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    const double* __restrict__ values,
    const uint32_t* __restrict__ meta_root,
    double* __restrict__ out_log_z
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    uint32_t root = meta_root[slot];
    size_t node_base = slot_offset(slot, node_cap);
    out_log_z[0] = values[node_base + root];
}

extern "C" __global__ void xgcf_backward_level_propagate_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    double* __restrict__ adj,
    const uint32_t* __restrict__ meta_num_levels
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t slot = active_slot[0];
    uint32_t num_levels = meta_num_levels[slot];
    if (level >= num_levels) {
        return;
    }
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* var_log_true_s = var_log_true + var_base;
    const double* var_log_false_s = var_log_false + var_base;
    const double* values_s = values + node_base;
    double* adj_s = adj + node_base;

    uint32_t level_offset = level_offsets_s[level];
    uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes_s[level_offset + tid];
    const semantic_measure::CircuitView circuit{
        node_type_s, child_offsets_s, child_indices_s, nullptr, decision_var_s,
        decision_child_false_s, decision_child_true_s
    };
    semantic_measure::propagate_node_adjoint(
        circuit, node, adj_s[node], var_log_true_s, var_log_false_s, values_s,
        semantic_measure::AtomicNodeAdjoints{adj_s});
}

extern "C" __global__ void xgcf_backward_level_decision_grad_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false,
    const uint32_t* __restrict__ meta_num_levels
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t slot = active_slot[0];
    uint32_t num_levels = meta_num_levels[slot];
    if (level >= num_levels) {
        return;
    }
    size_t node_base = slot_offset(slot, node_cap);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* var_log_true_s = var_log_true + var_base;
    const double* var_log_false_s = var_log_false + var_base;
    const double* values_s = values + node_base;
    const double* adj_s = adj + node_base;
    double* grad_true_s = grad_true + var_base;
    double* grad_false_s = grad_false + var_base;

    uint32_t level_offset = level_offsets_s[level];
    uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes_s[level_offset + tid];
    if (node_type_s[node] != XGCF_DECISION) {
        return;
    }
    const semantic_measure::CircuitView circuit{
        node_type_s, nullptr, nullptr, nullptr, decision_var_s,
        decision_child_false_s, decision_child_true_s
    };
    semantic_measure::accumulate_node_weight_gradients(
        circuit, node, adj_s[node], var_log_true_s, var_log_false_s, values_s,
        semantic_measure::AtomicWeightGradients{grad_true_s, grad_false_s});
}

extern "C" __global__ void xgcf_backward_level_lit_grad_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    uint32_t level,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false,
    const uint32_t* __restrict__ meta_num_levels
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t slot = active_slot[0];
    uint32_t num_levels = meta_num_levels[slot];
    if (level >= num_levels) {
        return;
    }
    size_t node_base = slot_offset(slot, node_cap);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* adj_s = adj + node_base;
    double* grad_true_s = grad_true + var_base;
    double* grad_false_s = grad_false + var_base;

    uint32_t level_offset = level_offsets_s[level];
    uint32_t num_level_nodes = level_offsets_s[level + 1] - level_offset;
    if (tid >= num_level_nodes) {
        return;
    }

    uint32_t node = level_nodes_s[level_offset + tid];
    if (node_type_s[node] != XGCF_LIT) {
        return;
    }
    const semantic_measure::CircuitView circuit{
        node_type_s, nullptr, nullptr, lit_s, nullptr, nullptr, nullptr
    };
    const double* unused_values = nullptr;
    semantic_measure::accumulate_node_weight_gradients(
        circuit, node, adj_s[node], nullptr, nullptr, unused_values,
        semantic_measure::AtomicWeightGradients{grad_true_s, grad_false_s});
}

// Fused backward pass: all levels in a single kernel (cached layout).
// Replaces the host-side per-level loop that launches three separate kernels
// (propagate, decision_grad, lit_grad) for each level.
extern "C" __global__ void xgcf_backward_all_levels_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    uint32_t edge_cap,
    uint32_t level_cap,
    uint32_t var_cap,
    const uint8_t* __restrict__ node_type,
    const uint32_t* __restrict__ child_offsets,
    const uint32_t* __restrict__ child_indices,
    const uint32_t* __restrict__ decision_var,
    const uint32_t* __restrict__ decision_child_false,
    const uint32_t* __restrict__ decision_child_true,
    const int32_t* __restrict__ lit,
    const uint32_t* __restrict__ level_nodes,
    const uint32_t* __restrict__ level_offsets,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false,
    const uint32_t* __restrict__ meta_num_levels
) {
    if (blockIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    size_t edge_base = slot_offset(slot, edge_cap);
    size_t offset_base = slot_offset(slot, node_cap + 1u);
    size_t level_base = slot_offset(slot, node_cap);
    size_t level_offset_base = slot_offset(slot, level_cap + 1u);
    size_t var_base = slot_offset(slot, var_cap + 1u);

    const uint8_t* node_type_s = node_type + node_base;
    const uint32_t* child_offsets_s = child_offsets + offset_base;
    const uint32_t* child_indices_s = child_indices + edge_base;
    const uint32_t* decision_var_s = decision_var + node_base;
    const uint32_t* decision_child_false_s = decision_child_false + node_base;
    const uint32_t* decision_child_true_s = decision_child_true + node_base;
    const int32_t* lit_s = lit + node_base;
    const uint32_t* level_nodes_s = level_nodes + level_base;
    const uint32_t* level_offsets_s = level_offsets + level_offset_base;
    const double* var_log_true_s = var_log_true + var_base;
    const double* var_log_false_s = var_log_false + var_base;
    const double* values_s = values + node_base;
    double* adj_s = adj + node_base;
    double* grad_true_s = grad_true + var_base;
    double* grad_false_s = grad_false + var_base;

    const semantic_measure::CircuitView circuit{
        node_type_s, child_offsets_s, child_indices_s, lit_s, decision_var_s,
        decision_child_false_s, decision_child_true_s
    };
    uint32_t num_levels = meta_num_levels[slot];
    for (int32_t level = static_cast<int32_t>(num_levels) - 1; level >= 0; --level) {
        uint32_t level_off = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_off;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_off + idx];
            double adjoint = adj_s[node];
            semantic_measure::propagate_node_adjoint(
                circuit, node, adjoint, var_log_true_s, var_log_false_s, values_s,
                semantic_measure::AtomicNodeAdjoints{adj_s});
            semantic_measure::accumulate_node_weight_gradients(
                circuit, node, adjoint, var_log_true_s, var_log_false_s, values_s,
                semantic_measure::AtomicWeightGradients{grad_true_s, grad_false_s});
        }
        __syncthreads();
    }
}

extern "C" __global__ void xgcf_free_var_apply_grad_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t var_cap,
    const uint8_t* __restrict__ free_var_mask,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    uint32_t n,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t slot = active_slot[0];
    size_t var_base = slot_offset(slot, var_cap + 1u);
    const uint8_t* mask_s = free_var_mask + var_base;
    const double* log_true_s = var_log_true + var_base;
    const double* log_false_s = var_log_false + var_base;
    double* grad_true_s = grad_true + var_base;
    double* grad_false_s = grad_false + var_base;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t var = tid; var < n; var += blockDim.x * gridDim.x) {
        if (var == 0u || mask_s[var] == 0u) {
            continue;
        }
        double t = log_true_s[var];
        double f = log_false_s[var];
        double m = (t > f) ? t : f;
        if (isinf(m) && m < 0.0) {
            continue;
        }
        double et = exp(t - m);
        double ef = exp(f - m);
        double sum = et + ef;
        if (sum == 0.0) {
            continue;
        }
        double pt = et / sum;
        double pf = ef / sum;
        grad_true_s[var] += pt;
        grad_false_s[var] += pf;
    }
}

extern "C" __global__ void xgcf_free_var_reduce_stage_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t var_cap,
    const uint8_t* __restrict__ free_var_mask,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ in_vals,
    uint32_t n,
    uint32_t mode,
    double* __restrict__ out_vals
) {
    uint32_t slot = active_slot[0];
    size_t var_base = slot_offset(slot, var_cap + 1u);
    const uint8_t* mask_s = free_var_mask + var_base;
    const double* log_true_s = var_log_true + var_base;
    const double* log_false_s = var_log_false + var_base;

    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t out_len = (n + 1u) / 2u;
    if (tid >= out_len) {
        return;
    }

    uint32_t i0 = tid * 2u;
    uint32_t i1 = i0 + 1u;

    double a = 0.0;
    double b = 0.0;

    if (mode == 0u) {
        if (i0 < n && i0 != 0u && mask_s[i0] != 0u) {
            a = logsumexp2(log_true_s[i0], log_false_s[i0]);
        }
        if (i1 < n && i1 != 0u && mask_s[i1] != 0u) {
            b = logsumexp2(log_true_s[i1], log_false_s[i1]);
        }
    } else {
        if (i0 < n) {
            a = in_vals[i0];
        }
        if (i1 < n) {
            b = in_vals[i1];
        }
    }

    out_vals[tid] = a + b;
}

extern "C" __global__ void xgcf_add_scalar_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    double* __restrict__ values,
    const uint32_t* __restrict__ meta_root,
    const double* __restrict__ scalar
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    uint32_t index = meta_root[slot];
    size_t node_base = slot_offset(slot, node_cap);
    values[node_base + index] += scalar[0];
}

extern "C" __global__ void xgcf_copy_root_cached(
    const uint32_t* __restrict__ active_slot,
    uint32_t node_cap,
    const double* __restrict__ values,
    uint32_t root,
    double* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    uint32_t slot = active_slot[0];
    size_t node_base = slot_offset(slot, node_cap);
    out[0] = values[node_base + root];
}
