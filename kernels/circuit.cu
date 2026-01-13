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
