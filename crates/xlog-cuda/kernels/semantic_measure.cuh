#pragma once

#include <stdint.h>
#include <cmath>

namespace semantic_measure {

// These bounds certify the same Boolean DAG without assuming an error bound for
// libdevice log/exp. Normalize each original variable's two nonnegative weights:
// the common scale cancels in every conditioned mass. Skipped variables then
// integrate to one. The original log-space primal and reverse remain unchanged.
//
// A nonzero endpoint is fraction * 2^exponent, fraction in [1/2,1). Separate
// endpoint exponents retain positive lower bounds even for very rare conditions.
// Original U64 ratios are >= 2^-65 when positive. Every subsequent mantissa
// product, sum and quotient is normal; at depth <= UINT32_MAX, exponent magnitude
// is bounded by 67 * (depth + 1), well inside int64_t.
struct ScaledMassBound {
    double fraction;
    int64_t exponent;

    // Nonzero inputs are normal: integer ratios and mantissa arithmetic never
    // approach binary64 underflow. Biased exponent 1022 encodes [1/2,1).
    __device__ static ScaledMassBound normalize(double value,int64_t exponent=0) {
        if(value==0.0)return {0.0,0};
        const uint64_t bits=uint64_t(__double_as_longlong(value));
        return {__longlong_as_double((bits&0x000fffffffffffffULL)|(uint64_t(1022)<<52)),
                exponent+int64_t(bits>>52)-1022};
    }
    __device__ bool positive() const { return fraction>0.0; }
};
struct MassInterval {
    ScaledMassBound lower,upper;
};
static_assert(sizeof(ScaledMassBound)==2*sizeof(uint64_t),"scaled mass endpoint layout");
static_assert(sizeof(MassInterval)==4*sizeof(uint64_t),"mass interval layout");
static_assert(alignof(MassInterval)==alignof(uint64_t),"mass interval alignment");

__device__ __forceinline__ MassInterval exact_mass_interval(double value) {
    const auto bound=ScaledMassBound::normalize(value);
    return {bound,bound};
}
__device__ __forceinline__ MassInterval rational_mass_interval(
        uint64_t numerator,double denominator_lower,double denominator_upper) {
    return {ScaledMassBound::normalize(__ddiv_rd(__ull2double_rd(numerator),denominator_upper)),
            ScaledMassBound::normalize(fmin(1.0,__ddiv_ru(__ull2double_ru(numerator),denominator_lower)))};
}
__device__ __forceinline__ ScaledMassBound mass_product(
        ScaledMassBound a,ScaledMassBound b,bool upper) {
    if(!a.positive() || !b.positive())return {0.0,0};
    return ScaledMassBound::normalize(upper ? __dmul_ru(a.fraction,b.fraction) :
        __dmul_rd(a.fraction,b.fraction),a.exponent+b.exponent);
}
__device__ __forceinline__ ScaledMassBound mass_sum(
        ScaledMassBound a,ScaledMassBound b,bool upper) {
    if(!a.positive())return b;
    if(!b.positive())return a;
    if(a.exponent<b.exponent) { const auto swap=a;a=b;b=swap; }
    const int64_t gap=a.exponent-b.exponent;
    double value;
    if(gap>=53) {
        // The smaller positive term is strictly less than one ULP of a.fraction.
        // Increment its positive IEEE encoding for the upward neighbor, including
        // the crossing through one; normalize that neighbor below.
        value=upper ? __longlong_as_double(__double_as_longlong(a.fraction)+1) : a.fraction;
    } else {
        const double scale=__longlong_as_double(uint64_t(1023-gap)<<52);
        const double smaller=__dmul_rn(b.fraction,scale); // Exact normal power-of-two scaling.
        value=upper ? __dadd_ru(a.fraction,smaller) : __dadd_rd(a.fraction,smaller);
    }
    return ScaledMassBound::normalize(value,a.exponent);
}
__device__ __forceinline__ MassInterval decision_mass_interval(
        const MassInterval& false_weight,const MassInterval& low,
        const MassInterval& true_weight,const MassInterval& high) {
    return {mass_sum(mass_product(false_weight.lower,low.lower,false),
                     mass_product(true_weight.lower,high.lower,false),false),
            mass_sum(mass_product(false_weight.upper,low.upper,true),
                     mass_product(true_weight.upper,high.upper,true),true)};
}
__device__ __forceinline__ ScaledMassBound mass_quotient(
        ScaledMassBound numerator,ScaledMassBound denominator,bool upper) {
    if(!numerator.positive())return {0.0,0};
    return ScaledMassBound::normalize(upper ?
        __ddiv_ru(numerator.fraction,denominator.fraction) :
        __ddiv_rd(numerator.fraction,denominator.fraction),
        numerator.exponent-denominator.exponent);
}
__device__ __forceinline__ double probability_endpoint(ScaledMassBound value,bool upper) {
    if(!value.positive())return 0.0;
    // Intersect the conditional enclosure with its proven probability domain.
    // Branch on the wide exponent before narrowing or forming a bit shift.
    if(value.exponent>=1)return 1.0;
    if(value.exponent<-1073)return upper ? 0x1p-1074 : 0.0;
    const uint64_t fraction_bits=uint64_t(__double_as_longlong(value.fraction))&0x000fffffffffffffULL;
    if(value.exponent>=-1021)
        return __longlong_as_double(fraction_bits|(uint64_t(1022+value.exponent)<<52));
    const double scale=__longlong_as_double(uint64_t(1)<<unsigned(value.exponent+1074));
    return upper ? __dmul_ru(value.fraction,scale) : __dmul_rd(value.fraction,scale);
}
// The numerator is an event intersected with this same conditioning event.
// Original ROBDD identity establishes the exact unit-mass endpoint.
__device__ __forceinline__ bool probability_error_bound(
        const MassInterval& numerator,const MassInterval& normalizer,bool identical,
        double value,double* error) {
    if(!normalizer.lower.positive() || !isfinite(value) || value<0.0 || value>1.0)return false;
    const double lower=identical ? 1.0 :
        probability_endpoint(mass_quotient(numerator.lower,normalizer.upper,false),false);
    const double upper=identical ? 1.0 :
        probability_endpoint(mass_quotient(numerator.upper,normalizer.lower,true),true);
    if(!(lower>=0.0 && lower<=upper && upper<=1.0))return false;
    const double projected=double(__double2float_rn(value));
    *error=fmax(fmax(__dadd_ru(value,-lower),__dadd_ru(upper,-value)),
                fmax(__dadd_ru(projected,-lower),__dadd_ru(upper,-projected)));
    return isfinite(*error) && *error>=0.0;
}

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

template<class Values>
__device__ __forceinline__ double forward_decision(
    uint32_t variable, uint32_t child_false, uint32_t child_true,
    const double* var_log_true, const double* var_log_false, const Values& values
) {
    return logsumexp2(var_log_false[variable] + values[child_false],
                      var_log_true[variable] + values[child_true]);
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
            value = forward_decision(variable, child_false, child_true,
                                     var_log_true, var_log_false, values);
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

template<class Values, class Adjoints>
__device__ __forceinline__ void propagate_decision_adjoint(
    uint32_t variable, uint32_t child_false, uint32_t child_true,
    double parent, double adjoint,
    const double* var_log_true, const double* var_log_false,
    const Values& values, const Adjoints& output
) {
    if (adjoint == 0.0 || (isinf(parent) && parent < 0.0)) {
        return;
    }
    double false_value = values[child_false];
    double true_value = values[child_true];
    if (!(isinf(false_value) && false_value < 0.0)) {
        double ratio = exp((var_log_false[variable] + false_value) - parent);
        if (ratio != 0.0) {
            output.add(child_false, adjoint * ratio);
        }
    }
    if (!(isinf(true_value) && true_value < 0.0)) {
        double ratio = exp((var_log_true[variable] + true_value) - parent);
        if (ratio != 0.0) {
            output.add(child_true, adjoint * ratio);
        }
    }
}

template<class Values, class Gradients>
__device__ __forceinline__ void accumulate_decision_weight_gradients(
    uint32_t variable, uint32_t child_false, uint32_t child_true,
    double parent, double adjoint,
    const double* var_log_true, const double* var_log_false,
    const Values& values, const Gradients& output
) {
    if (adjoint == 0.0 || (isinf(parent) && parent < 0.0)) {
        return;
    }
    double false_value = values[child_false];
    double true_value = values[child_true];
    double false_probability = 0.0;
    double true_probability = 0.0;
    if (!(isinf(false_value) && false_value < 0.0)) {
        false_probability = exp((var_log_false[variable] + false_value) - parent);
    }
    if (!(isinf(true_value) && true_value < 0.0)) {
        true_probability = exp((var_log_true[variable] + true_value) - parent);
    }
    if (false_probability != 0.0) {
        output.add_false(variable, adjoint * false_probability);
    }
    if (true_probability != 0.0) {
        output.add_true(variable, adjoint * true_probability);
    }
}

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
            propagate_decision_adjoint(circuit.decision_variable(node),
                circuit.false_child(node), circuit.true_child(node), values[node],
                adjoint, var_log_true, var_log_false, values, output);
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
        accumulate_decision_weight_gradients(circuit.decision_variable(node),
            circuit.false_child(node), circuit.true_child(node), values[node],
            adjoint, var_log_true, var_log_false, values, output);
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
