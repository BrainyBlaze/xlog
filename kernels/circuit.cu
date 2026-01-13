// kernels/circuit.cu
#include <cstdint>
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
    uint64_t level_range,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    double* __restrict__ values
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = static_cast<uint32_t>(level_range & 0xFFFFFFFFull);
    uint32_t num_level_nodes = static_cast<uint32_t>(level_range >> 32);
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
    uint64_t level_range,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    double* __restrict__ adj
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = static_cast<uint32_t>(level_range & 0xFFFFFFFFull);
    uint32_t num_level_nodes = static_cast<uint32_t>(level_range >> 32);
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
    uint64_t level_range,
    const double* __restrict__ var_log_true,
    const double* __restrict__ var_log_false,
    const double* __restrict__ values,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = static_cast<uint32_t>(level_range & 0xFFFFFFFFull);
    uint32_t num_level_nodes = static_cast<uint32_t>(level_range >> 32);
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
    uint64_t level_range,
    const double* __restrict__ adj,
    double* __restrict__ grad_true,
    double* __restrict__ grad_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t level_offset = static_cast<uint32_t>(level_range & 0xFFFFFFFFull);
    uint32_t num_level_nodes = static_cast<uint32_t>(level_range >> 32);
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
