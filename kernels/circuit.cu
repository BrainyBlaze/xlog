// kernels/circuit.cu
#include <cstdint>
#include <cstddef>
#include <cmath>

/**
 * XGCF Circuit Evaluation Kernels (Phase 4)
 *
 * This module evaluates Decision-DNNF circuits lowered to XGCF in log-space.
 *
 * Node type encoding must match crates/xlog-prob/src/xgcf.rs:
 * 0 = Const0
 * 1 = Const1
 * 2 = Lit
 * 3 = And
 * 4 = Or
 * 5 = Decision
 */

#define XGCF_CONST0 0
#define XGCF_CONST1 1
#define XGCF_LIT 2
#define XGCF_AND 3
#define XGCF_OR 4
#define XGCF_DECISION 5

#define BLOCK_SIZE 256

__device__ __forceinline__ double logsumexp2(double a, double b) {
    double m = (a > b) ? a : b;
    if (isinf(m) && m < 0.0) {
        return m;
    }
    return m + log(exp(a - m) + exp(b - m));
}

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
    uint8_t ty = node_type[node];

    double v;
    switch (ty) {
        case XGCF_CONST0:
            v = -INFINITY;
            break;
        case XGCF_CONST1:
            v = 0.0;
            break;
        case XGCF_LIT: {
            int32_t l = lit[node];
            uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
            v = (l > 0) ? var_log_true[var] : var_log_false[var];
            break;
        }
        case XGCF_AND: {
            uint32_t c0 = child_offsets[node];
            uint32_t c1 = child_offsets[node + 1];
            double acc = 0.0;
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices[i];
                acc += values[child];
            }
            v = acc;
            break;
        }
        case XGCF_OR: {
            uint32_t c0 = child_offsets[node];
            uint32_t c1 = child_offsets[node + 1];
            double maxv = -INFINITY;
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices[i];
                double cv = values[child];
                if (cv > maxv) {
                    maxv = cv;
                }
            }
            if (isinf(maxv) && maxv < 0.0) {
                v = maxv;
            } else {
                double sum = 0.0;
                for (uint32_t i = c0; i < c1; i++) {
                    uint32_t child = child_indices[i];
                    sum += exp(values[child] - maxv);
                }
                v = maxv + log(sum);
            }
            break;
        }
        case XGCF_DECISION: {
            uint32_t var = decision_var[node];
            uint32_t child_f = decision_child_false[node];
            uint32_t child_t = decision_child_true[node];
            double t = var_log_true[var];
            double f = var_log_false[var];
            v = logsumexp2(f + values[child_f], t + values[child_t]);
            break;
        }
        default:
            v = -INFINITY;
            break;
    }

    values[node] = v;
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
    double a = adj[node];
    if (a == 0.0) {
        return;
    }

    uint8_t ty = node_type[node];
    switch (ty) {
        case XGCF_AND: {
            uint32_t c0 = child_offsets[node];
            uint32_t c1 = child_offsets[node + 1];
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices[i];
                atomicAdd(&adj[child], a);
            }
            break;
        }
        case XGCF_OR: {
            double parent_v = values[node];
            if (isinf(parent_v) && parent_v < 0.0) {
                return;
            }
            uint32_t c0 = child_offsets[node];
            uint32_t c1 = child_offsets[node + 1];
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices[i];
                double child_v = values[child];
                if (isinf(child_v) && child_v < 0.0) {
                    continue;
                }
                double ratio = exp(child_v - parent_v);
                if (ratio != 0.0) {
                    atomicAdd(&adj[child], a * ratio);
                }
            }
            break;
        }
        case XGCF_DECISION: {
            double parent_v = values[node];
            if (isinf(parent_v) && parent_v < 0.0) {
                return;
            }
            uint32_t var = decision_var[node];
            uint32_t child_f = decision_child_false[node];
            uint32_t child_t = decision_child_true[node];
            double log_t = var_log_true[var];
            double log_f = var_log_false[var];

            double vf = values[child_f];
            double vt = values[child_t];
            if (!(isinf(vf) && vf < 0.0)) {
                double ratio_f = exp((log_f + vf) - parent_v);
                if (ratio_f != 0.0) {
                    atomicAdd(&adj[child_f], a * ratio_f);
                }
            }
            if (!(isinf(vt) && vt < 0.0)) {
                double ratio_t = exp((log_t + vt) - parent_v);
                if (ratio_t != 0.0) {
                    atomicAdd(&adj[child_t], a * ratio_t);
                }
            }
            break;
        }
        default:
            break;
    }
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

    double a = adj[node];
    if (a == 0.0) {
        return;
    }

    double parent_v = values[node];
    if (isinf(parent_v) && parent_v < 0.0) {
        return;
    }

    uint32_t var = decision_var[node];
    uint32_t child_f = decision_child_false[node];
    uint32_t child_t = decision_child_true[node];
    double log_t = var_log_true[var];
    double log_f = var_log_false[var];

    double vf = values[child_f];
    double vt = values[child_t];

    double p_false = 0.0;
    double p_true = 0.0;
    if (!(isinf(vf) && vf < 0.0)) {
        p_false = exp((log_f + vf) - parent_v);
    }
    if (!(isinf(vt) && vt < 0.0)) {
        p_true = exp((log_t + vt) - parent_v);
    }

    if (p_false != 0.0) {
        atomicAdd(&grad_false[var], a * p_false);
    }
    if (p_true != 0.0) {
        atomicAdd(&grad_true[var], a * p_true);
    }
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

    double a = adj[node];
    if (a == 0.0) {
        return;
    }

    int32_t l = lit[node];
    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
    if (l > 0) {
        atomicAdd(&grad_true[var], a);
    } else if (l < 0) {
        atomicAdd(&grad_false[var], a);
    }
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
    uint8_t ty = node_type_s[node];

    double v;
    switch (ty) {
        case XGCF_CONST0:
            v = -INFINITY;
            break;
        case XGCF_CONST1:
            v = 0.0;
            break;
        case XGCF_LIT: {
            int32_t l = lit_s[node];
            uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
            v = (l > 0) ? var_log_true_s[var] : var_log_false_s[var];
            break;
        }
        case XGCF_AND: {
            uint32_t c0 = child_offsets_s[node];
            uint32_t c1 = child_offsets_s[node + 1];
            double acc = 0.0;
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices_s[i];
                acc += values_s[child];
            }
            v = acc;
            break;
        }
        case XGCF_OR: {
            uint32_t c0 = child_offsets_s[node];
            uint32_t c1 = child_offsets_s[node + 1];
            double maxv = -INFINITY;
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices_s[i];
                double cv = values_s[child];
                if (cv > maxv) {
                    maxv = cv;
                }
            }
            if (isinf(maxv) && maxv < 0.0) {
                v = maxv;
            } else {
                double sum = 0.0;
                for (uint32_t i = c0; i < c1; i++) {
                    uint32_t child = child_indices_s[i];
                    sum += exp(values_s[child] - maxv);
                }
                v = maxv + log(sum);
            }
            break;
        }
        case XGCF_DECISION: {
            uint32_t var = decision_var_s[node];
            uint32_t child_f = decision_child_false_s[node];
            uint32_t child_t = decision_child_true_s[node];
            double t = var_log_true_s[var];
            double f = var_log_false_s[var];
            v = logsumexp2(f + values_s[child_f], t + values_s[child_t]);
            break;
        }
        default:
            v = -INFINITY;
            break;
    }

    values_s[node] = v;
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
            uint8_t ty = node_type_s[node];

            double v;
            switch (ty) {
                case XGCF_CONST0:
                    v = -INFINITY;
                    break;
                case XGCF_CONST1:
                    v = 0.0;
                    break;
                case XGCF_LIT: {
                    int32_t l = lit_s[node];
                    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
                    v = (l > 0) ? var_log_true_s[var] : var_log_false_s[var];
                    break;
                }
                case XGCF_AND: {
                    uint32_t c0 = child_offsets_s[node];
                    uint32_t c1 = child_offsets_s[node + 1];
                    double acc = 0.0;
                    for (uint32_t i = c0; i < c1; i++) {
                        uint32_t child = child_indices_s[i];
                        acc += values_s[child];
                    }
                    v = acc;
                    break;
                }
                case XGCF_OR: {
                    uint32_t c0 = child_offsets_s[node];
                    uint32_t c1 = child_offsets_s[node + 1];
                    double maxv = -INFINITY;
                    for (uint32_t i = c0; i < c1; i++) {
                        uint32_t child = child_indices_s[i];
                        double cv = values_s[child];
                        if (cv > maxv) {
                            maxv = cv;
                        }
                    }
                    if (isinf(maxv) && maxv < 0.0) {
                        v = maxv;
                    } else {
                        double sum = 0.0;
                        for (uint32_t i = c0; i < c1; i++) {
                            uint32_t child = child_indices_s[i];
                            sum += exp(values_s[child] - maxv);
                        }
                        v = maxv + log(sum);
                    }
                    break;
                }
                case XGCF_DECISION: {
                    uint32_t var = decision_var_s[node];
                    uint32_t child_f = decision_child_false_s[node];
                    uint32_t child_t = decision_child_true_s[node];
                    double t = var_log_true_s[var];
                    double f = var_log_false_s[var];
                    v = logsumexp2(f + values_s[child_f], t + values_s[child_t]);
                    break;
                }
                default:
                    v = -INFINITY;
                    break;
            }

            values_s[node] = v;
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
            uint8_t ty = node_type_s[node];

            double v;
            switch (ty) {
                case XGCF_CONST0:
                    v = -INFINITY;
                    break;
                case XGCF_CONST1:
                    v = 0.0;
                    break;
                case XGCF_LIT: {
                    int32_t l = lit_s[node];
                    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
                    v = (l > 0) ? var_log_true_q[var] : var_log_false_q[var];
                    break;
                }
                case XGCF_AND: {
                    uint32_t c0 = child_offsets_s[node];
                    uint32_t c1 = child_offsets_s[node + 1];
                    double acc = 0.0;
                    for (uint32_t i = c0; i < c1; i++) {
                        uint32_t child = child_indices_s[i];
                        acc += values_q[child];
                    }
                    v = acc;
                    break;
                }
                case XGCF_OR: {
                    uint32_t c0 = child_offsets_s[node];
                    uint32_t c1 = child_offsets_s[node + 1];
                    double maxv = -INFINITY;
                    for (uint32_t i = c0; i < c1; i++) {
                        uint32_t child = child_indices_s[i];
                        double cv = values_q[child];
                        if (cv > maxv) {
                            maxv = cv;
                        }
                    }
                    if (isinf(maxv) && maxv < 0.0) {
                        v = maxv;
                    } else {
                        double sum = 0.0;
                        for (uint32_t i = c0; i < c1; i++) {
                            uint32_t child = child_indices_s[i];
                            sum += exp(values_q[child] - maxv);
                        }
                        v = maxv + log(sum);
                    }
                    break;
                }
                case XGCF_DECISION: {
                    uint32_t var = decision_var_s[node];
                    uint32_t child_f = decision_child_false_s[node];
                    uint32_t child_t = decision_child_true_s[node];
                    double t = var_log_true_q[var];
                    double f = var_log_false_q[var];
                    v = logsumexp2(f + values_q[child_f], t + values_q[child_t]);
                    break;
                }
                default:
                    v = -INFINITY;
                    break;
            }

            values_q[node] = v;
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

    uint32_t num_levels = meta_num_levels[slot];
    for (int32_t level = static_cast<int32_t>(num_levels) - 1; level >= 0; --level) {
        uint32_t level_off = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_off;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_off + idx];
            double a = adj_q[node];

            if (a != 0.0) {
                uint8_t ty = node_type_s[node];
                switch (ty) {
                    case XGCF_AND: {
                        uint32_t c0 = child_offsets_s[node];
                        uint32_t c1 = child_offsets_s[node + 1];
                        for (uint32_t i = c0; i < c1; i++) {
                            uint32_t child = child_indices_s[i];
                            atomicAdd(&adj_q[child], a);
                        }
                        break;
                    }
                    case XGCF_OR: {
                        double parent_v = values_q[node];
                        if (!(isinf(parent_v) && parent_v < 0.0)) {
                            uint32_t c0 = child_offsets_s[node];
                            uint32_t c1 = child_offsets_s[node + 1];
                            for (uint32_t i = c0; i < c1; i++) {
                                uint32_t child = child_indices_s[i];
                                double child_v = values_q[child];
                                if (isinf(child_v) && child_v < 0.0) {
                                    continue;
                                }
                                double ratio = exp(child_v - parent_v);
                                if (ratio != 0.0) {
                                    atomicAdd(&adj_q[child], a * ratio);
                                }
                            }
                        }
                        break;
                    }
                    case XGCF_DECISION: {
                        double parent_v = values_q[node];
                        if (!(isinf(parent_v) && parent_v < 0.0)) {
                            uint32_t var = decision_var_s[node];
                            uint32_t child_f = decision_child_false_s[node];
                            uint32_t child_t = decision_child_true_s[node];
                            double log_t = var_log_true_q[var];
                            double log_f = var_log_false_q[var];

                            double vf = values_q[child_f];
                            double vt = values_q[child_t];
                            if (!(isinf(vf) && vf < 0.0)) {
                                double ratio_f = exp((log_f + vf) - parent_v);
                                if (ratio_f != 0.0) {
                                    atomicAdd(&adj_q[child_f], a * ratio_f);
                                }
                            }
                            if (!(isinf(vt) && vt < 0.0)) {
                                double ratio_t = exp((log_t + vt) - parent_v);
                                if (ratio_t != 0.0) {
                                    atomicAdd(&adj_q[child_t], a * ratio_t);
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }

                if (ty == XGCF_DECISION) {
                    double parent_v = values_q[node];
                    if (!(isinf(parent_v) && parent_v < 0.0)) {
                        uint32_t var = decision_var_s[node];
                        uint32_t child_f = decision_child_false_s[node];
                        uint32_t child_t = decision_child_true_s[node];
                        double log_t = var_log_true_q[var];
                        double log_f = var_log_false_q[var];

                        double vf = values_q[child_f];
                        double vt = values_q[child_t];

                        double p_false = 0.0;
                        double p_true = 0.0;
                        if (!(isinf(vf) && vf < 0.0)) {
                            p_false = exp((log_f + vf) - parent_v);
                        }
                        if (!(isinf(vt) && vt < 0.0)) {
                            p_true = exp((log_t + vt) - parent_v);
                        }

                        if (p_false != 0.0) {
                            atomicAdd(&grad_false_q[var], a * p_false);
                        }
                        if (p_true != 0.0) {
                            atomicAdd(&grad_true_q[var], a * p_true);
                        }
                    }
                }

                if (ty == XGCF_LIT) {
                    int32_t l = lit_s[node];
                    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
                    if (l > 0) {
                        atomicAdd(&grad_true_q[var], a);
                    } else if (l < 0) {
                        atomicAdd(&grad_false_q[var], a);
                    }
                }
            }
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
    double a = adj_s[node];
    if (a == 0.0) {
        return;
    }

    uint8_t ty = node_type_s[node];
    switch (ty) {
        case XGCF_AND: {
            uint32_t c0 = child_offsets_s[node];
            uint32_t c1 = child_offsets_s[node + 1];
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices_s[i];
                atomicAdd(&adj_s[child], a);
            }
            break;
        }
        case XGCF_OR: {
            double parent_v = values_s[node];
            if (isinf(parent_v) && parent_v < 0.0) {
                return;
            }
            uint32_t c0 = child_offsets_s[node];
            uint32_t c1 = child_offsets_s[node + 1];
            for (uint32_t i = c0; i < c1; i++) {
                uint32_t child = child_indices_s[i];
                double child_v = values_s[child];
                if (isinf(child_v) && child_v < 0.0) {
                    continue;
                }
                double ratio = exp(child_v - parent_v);
                if (ratio != 0.0) {
                    atomicAdd(&adj_s[child], a * ratio);
                }
            }
            break;
        }
        case XGCF_DECISION: {
            double parent_v = values_s[node];
            if (isinf(parent_v) && parent_v < 0.0) {
                return;
            }
            uint32_t var = decision_var_s[node];
            uint32_t child_f = decision_child_false_s[node];
            uint32_t child_t = decision_child_true_s[node];
            double log_t = var_log_true_s[var];
            double log_f = var_log_false_s[var];

            double vf = values_s[child_f];
            double vt = values_s[child_t];
            if (!(isinf(vf) && vf < 0.0)) {
                double ratio_f = exp((log_f + vf) - parent_v);
                if (ratio_f != 0.0) {
                    atomicAdd(&adj_s[child_f], a * ratio_f);
                }
            }
            if (!(isinf(vt) && vt < 0.0)) {
                double ratio_t = exp((log_t + vt) - parent_v);
                if (ratio_t != 0.0) {
                    atomicAdd(&adj_s[child_t], a * ratio_t);
                }
            }
            break;
        }
        default:
            break;
    }
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

    double a = adj_s[node];
    if (a == 0.0) {
        return;
    }

    double parent_v = values_s[node];
    if (isinf(parent_v) && parent_v < 0.0) {
        return;
    }

    uint32_t var = decision_var_s[node];
    uint32_t child_f = decision_child_false_s[node];
    uint32_t child_t = decision_child_true_s[node];
    double log_t = var_log_true_s[var];
    double log_f = var_log_false_s[var];

    double vf = values_s[child_f];
    double vt = values_s[child_t];

    double p_false = 0.0;
    double p_true = 0.0;
    if (!(isinf(vf) && vf < 0.0)) {
        p_false = exp((log_f + vf) - parent_v);
    }
    if (!(isinf(vt) && vt < 0.0)) {
        p_true = exp((log_t + vt) - parent_v);
    }

    if (p_false != 0.0) {
        atomicAdd(&grad_false_s[var], a * p_false);
    }
    if (p_true != 0.0) {
        atomicAdd(&grad_true_s[var], a * p_true);
    }
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

    double a = adj_s[node];
    if (a == 0.0) {
        return;
    }

    int32_t l = lit_s[node];
    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
    if (l > 0) {
        atomicAdd(&grad_true_s[var], a);
    } else if (l < 0) {
        atomicAdd(&grad_false_s[var], a);
    }
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

    uint32_t num_levels = meta_num_levels[slot];
    for (int32_t level = static_cast<int32_t>(num_levels) - 1; level >= 0; --level) {
        uint32_t level_off = level_offsets_s[level];
        uint32_t num_level_nodes = level_offsets_s[level + 1] - level_off;
        for (uint32_t idx = threadIdx.x; idx < num_level_nodes; idx += blockDim.x) {
            uint32_t node = level_nodes_s[level_off + idx];
            double a = adj_s[node];

            // --- propagate (from xgcf_backward_level_propagate_cached) ---
            if (a != 0.0) {
                uint8_t ty = node_type_s[node];
                switch (ty) {
                    case XGCF_AND: {
                        uint32_t c0 = child_offsets_s[node];
                        uint32_t c1 = child_offsets_s[node + 1];
                        for (uint32_t i = c0; i < c1; i++) {
                            uint32_t child = child_indices_s[i];
                            atomicAdd(&adj_s[child], a);
                        }
                        break;
                    }
                    case XGCF_OR: {
                        double parent_v = values_s[node];
                        if (!(isinf(parent_v) && parent_v < 0.0)) {
                            uint32_t c0 = child_offsets_s[node];
                            uint32_t c1 = child_offsets_s[node + 1];
                            for (uint32_t i = c0; i < c1; i++) {
                                uint32_t child = child_indices_s[i];
                                double child_v = values_s[child];
                                if (isinf(child_v) && child_v < 0.0) {
                                    continue;
                                }
                                double ratio = exp(child_v - parent_v);
                                if (ratio != 0.0) {
                                    atomicAdd(&adj_s[child], a * ratio);
                                }
                            }
                        }
                        break;
                    }
                    case XGCF_DECISION: {
                        double parent_v = values_s[node];
                        if (!(isinf(parent_v) && parent_v < 0.0)) {
                            uint32_t var = decision_var_s[node];
                            uint32_t child_f = decision_child_false_s[node];
                            uint32_t child_t = decision_child_true_s[node];
                            double log_t = var_log_true_s[var];
                            double log_f = var_log_false_s[var];

                            double vf = values_s[child_f];
                            double vt = values_s[child_t];
                            if (!(isinf(vf) && vf < 0.0)) {
                                double ratio_f = exp((log_f + vf) - parent_v);
                                if (ratio_f != 0.0) {
                                    atomicAdd(&adj_s[child_f], a * ratio_f);
                                }
                            }
                            if (!(isinf(vt) && vt < 0.0)) {
                                double ratio_t = exp((log_t + vt) - parent_v);
                                if (ratio_t != 0.0) {
                                    atomicAdd(&adj_s[child_t], a * ratio_t);
                                }
                            }
                        }
                        break;
                    }
                    default:
                        break;
                }

                // --- decision_grad (from xgcf_backward_level_decision_grad_cached) ---
                if (ty == XGCF_DECISION) {
                    double parent_v = values_s[node];
                    if (!(isinf(parent_v) && parent_v < 0.0)) {
                        uint32_t var = decision_var_s[node];
                        uint32_t child_f = decision_child_false_s[node];
                        uint32_t child_t = decision_child_true_s[node];
                        double log_t = var_log_true_s[var];
                        double log_f = var_log_false_s[var];

                        double vf = values_s[child_f];
                        double vt = values_s[child_t];

                        double p_false = 0.0;
                        double p_true = 0.0;
                        if (!(isinf(vf) && vf < 0.0)) {
                            p_false = exp((log_f + vf) - parent_v);
                        }
                        if (!(isinf(vt) && vt < 0.0)) {
                            p_true = exp((log_t + vt) - parent_v);
                        }

                        if (p_false != 0.0) {
                            atomicAdd(&grad_false_s[var], a * p_false);
                        }
                        if (p_true != 0.0) {
                            atomicAdd(&grad_true_s[var], a * p_true);
                        }
                    }
                }

                // --- lit_grad (from xgcf_backward_level_lit_grad_cached) ---
                if (ty == XGCF_LIT) {
                    int32_t l = lit_s[node];
                    uint32_t var = (l > 0) ? static_cast<uint32_t>(l) : static_cast<uint32_t>(-l);
                    if (l > 0) {
                        atomicAdd(&grad_true_s[var], a);
                    } else if (l < 0) {
                        atomicAdd(&grad_false_s[var], a);
                    }
                }
            }
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
