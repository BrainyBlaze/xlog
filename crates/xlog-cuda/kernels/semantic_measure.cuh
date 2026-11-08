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

// These sinks borrow the original device buffers. They allocate nothing and
// retain neither an independent circuit nor a mutable publication selector.
struct AtomicNodeAdjoints {
    double* values;

    __device__ __forceinline__ void add(uint32_t node, double value) const {
        atomicAdd(values + node, value);
    }
};

struct AtomicWeightGradients {
    double* true_values;
    double* false_values;

    __device__ __forceinline__ void add_true(uint32_t variable, double value) const {
        atomicAdd(true_values + variable, value);
    }
    __device__ __forceinline__ void add_false(uint32_t variable, double value) const {
        atomicAdd(false_values + variable, value);
    }
};

// Pull back the supplied node cotangent through the same log-space operation
// used by forward_node. The caller owns reverse topological order, seeding and
// completion/numerical status; a cotangent need not be a unit root seed.
template<class Circuit, class Values, class Adjoints>
__device__ __forceinline__ void propagate_node_adjoint(
    const Circuit& circuit, uint32_t node, double adjoint,
    const double* var_log_true, const double* var_log_false,
    const Values& values, const Adjoints& output
) {
    if (adjoint == 0.0) {
        return;
    }
    switch (circuit.type(node)) {
        case XGCF_AND: {
            uint32_t begin = circuit.child_begin(node);
            uint32_t end = circuit.child_end(node);
            for (uint32_t edge = begin; edge < end; ++edge) {
                output.add(circuit.child(edge), adjoint);
            }
            break;
        }
        case XGCF_OR: {
            double parent = values[node];
            if (isinf(parent) && parent < 0.0) {
                return;
            }
            uint32_t begin = circuit.child_begin(node);
            uint32_t end = circuit.child_end(node);
            for (uint32_t edge = begin; edge < end; ++edge) {
                uint32_t child = circuit.child(edge);
                double value = values[child];
                if (isinf(value) && value < 0.0) {
                    continue;
                }
                double ratio = exp(value - parent);
                if (ratio != 0.0) {
                    output.add(child, adjoint * ratio);
                }
            }
            break;
        }
        case XGCF_DECISION: {
            double parent = values[node];
            if (isinf(parent) && parent < 0.0) {
                return;
            }
            uint32_t variable = circuit.decision_variable(node);
            uint32_t child_false = circuit.false_child(node);
            uint32_t child_true = circuit.true_child(node);
            double false_weight = var_log_false[variable];
            double true_weight = var_log_true[variable];
            double false_value = values[child_false];
            double true_value = values[child_true];
            if (!(isinf(false_value) && false_value < 0.0)) {
                double ratio = exp((false_weight + false_value) - parent);
                if (ratio != 0.0) {
                    output.add(child_false, adjoint * ratio);
                }
            }
            if (!(isinf(true_value) && true_value < 0.0)) {
                double ratio = exp((true_weight + true_value) - parent);
                if (ratio != 0.0) {
                    output.add(child_true, adjoint * ratio);
                }
            }
            break;
        }
        default:
            break;
    }
}

// These are derivatives with respect to the original log literal potentials.
// Parameterization and any chain rule into authored coefficients remain with
// their actual producer; a discrete coefficient code has no implied Jacobian.
template<class Circuit, class Values, class Gradients>
__device__ __forceinline__ void accumulate_node_weight_gradients(
    const Circuit& circuit, uint32_t node, double adjoint,
    const double* var_log_true, const double* var_log_false,
    const Values& values, const Gradients& output
) {
    if (adjoint == 0.0) {
        return;
    }
    if (circuit.type(node) == XGCF_DECISION) {
        double parent = values[node];
        if (isinf(parent) && parent < 0.0) {
            return;
        }
        uint32_t variable = circuit.decision_variable(node);
        uint32_t child_false = circuit.false_child(node);
        uint32_t child_true = circuit.true_child(node);
        double false_weight = var_log_false[variable];
        double true_weight = var_log_true[variable];
        double false_value = values[child_false];
        double true_value = values[child_true];
        double false_probability = 0.0;
        double true_probability = 0.0;
        if (!(isinf(false_value) && false_value < 0.0)) {
            false_probability = exp((false_weight + false_value) - parent);
        }
        if (!(isinf(true_value) && true_value < 0.0)) {
            true_probability = exp((true_weight + true_value) - parent);
        }
        if (false_probability != 0.0) {
            output.add_false(variable, adjoint * false_probability);
        }
        if (true_probability != 0.0) {
            output.add_true(variable, adjoint * true_probability);
        }
    } else if (circuit.type(node) == XGCF_LIT) {
        int32_t literal = circuit.literal(node);
        uint32_t variable = literal > 0
            ? static_cast<uint32_t>(literal) : static_cast<uint32_t>(-literal);
        if (literal > 0) {
            output.add_true(variable, adjoint);
        } else if (literal < 0) {
            output.add_false(variable, adjoint);
        }
    }
}

} // namespace semantic_measure
