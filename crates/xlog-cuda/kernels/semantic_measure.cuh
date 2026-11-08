#pragma once

#include <stdint.h>
#include <cmath>

namespace semantic_measure {

// The device encoding is shared with xlog-ir/src/semantic_measure.rs.
enum NodeType : uint8_t {
    XGCF_CONST0 = 0,
    XGCF_CONST1 = 1,
    XGCF_LIT = 2,
    XGCF_AND = 3,
    XGCF_OR = 4,
    XGCF_DECISION = 5,
};

// This view owns no roots, counts, weights, allocation or publication selector.
// A caller supplies the arrays of its acquired circuit representation.
struct CircuitView {
    const uint8_t* node_types;
    const uint32_t* child_offsets;
    const uint32_t* child_indices;
    const int32_t* literals;
    const uint32_t* decision_variables;
    const uint32_t* false_children;
    const uint32_t* true_children;

    __device__ __forceinline__ uint8_t type(uint32_t node) const {
        return node_types[node];
    }
    __device__ __forceinline__ int32_t literal(uint32_t node) const {
        return literals[node];
    }
    __device__ __forceinline__ uint32_t child_begin(uint32_t node) const {
        return child_offsets[node];
    }
    __device__ __forceinline__ uint32_t child_end(uint32_t node) const {
        return child_offsets[node + 1];
    }
    __device__ __forceinline__ uint32_t child(uint32_t edge) const {
        return child_indices[edge];
    }
    __device__ __forceinline__ uint32_t decision_variable(uint32_t node) const {
        return decision_variables[node];
    }
    __device__ __forceinline__ uint32_t false_child(uint32_t node) const {
        return false_children[node];
    }
    __device__ __forceinline__ uint32_t true_child(uint32_t node) const {
        return true_children[node];
    }
};

__device__ __forceinline__ double logsumexp2(double a, double b) {
    if (isnan(a) || isnan(b)) {
        return NAN;
    }
    double m = (a > b) ? a : b;
    if (isinf(m)) {
        return m;
    }
    return m + log(exp(a - m) + exp(b - m));
}

template<class Circuit, class Values>
__device__ __forceinline__ double logsumexp_children(
    const Circuit& circuit, uint32_t node, const Values& values
) {
    const uint32_t child_start = circuit.child_begin(node);
    const uint32_t child_end = circuit.child_end(node);
    double maxv = -INFINITY;
    for (uint32_t i = child_start; i < child_end; i++) {
        double child_value = values[circuit.child(i)];
        if (isnan(child_value)) {
            return NAN;
        }
        if (child_value > maxv) {
            maxv = child_value;
        }
    }
    if (isinf(maxv)) {
        return maxv;
    }
    double sum = 0.0;
    for (uint32_t i = child_start; i < child_end; i++) {
        sum += exp(values[circuit.child(i)] - maxv);
    }
    return maxv + log(sum);
}

// The caller establishes topological order and validates its acquired node view.
// Evaluation reads the topology, weights and values of that same representation.
template<class Circuit, class Values>
__device__ __forceinline__ double forward_node(
    const Circuit& circuit, uint32_t node,
    const double* var_log_true, const double* var_log_false,
    const Values& values
) {
    double value;
    switch (circuit.type(node)) {
        case XGCF_CONST0:
            value = -INFINITY;
            break;
        case XGCF_CONST1:
            value = 0.0;
            break;
        case XGCF_LIT: {
            int32_t literal = circuit.literal(node);
            uint32_t variable = literal > 0
                ? static_cast<uint32_t>(literal) : static_cast<uint32_t>(-literal);
            value = literal > 0 ? var_log_true[variable] : var_log_false[variable];
            break;
        }
        case XGCF_AND: {
            uint32_t start = circuit.child_begin(node);
            uint32_t end = circuit.child_end(node);
            double acc = 0.0;
            for (uint32_t i = start; i < end; i++) {
                uint32_t child = circuit.child(i);
                acc += values[child];
            }
            value = acc;
            break;
        }
        case XGCF_OR:
            value = logsumexp_children(circuit, node, values);
            break;
        case XGCF_DECISION: {
            uint32_t variable = circuit.decision_variable(node);
            uint32_t child_false = circuit.false_child(node);
            uint32_t child_true = circuit.true_child(node);
            double true_weight = var_log_true[variable];
            double false_weight = var_log_false[variable];
            value = logsumexp2(false_weight + values[child_false],
                              true_weight + values[child_true]);
            break;
        }
        default:
            value = -INFINITY;
            break;
    }
    return value;
}

} // namespace semantic_measure
