// kernels/neural.cu
//
// GPU neural fast-path helpers for XLOG:
// - Fill annotated-disjunction (AD) conditional-chain log weights from device-resident probabilities.
// - Scatter XGCF gradients (w.r.t. ln(q) and ln(1-q)) back into probability-space gradients (chain rule).
//
// These kernels are designed to be called from the XLOG training loop where neural network outputs
// are already device-resident (e.g., Torch CUDA tensors exchanged via DLPack).
//
// Notes:
// - The AD encoding uses a Bernoulli conditional chain (see xlog-prob provenance::compile_annotated_disjunction).
// - We intentionally scale probabilities by (1 - eps) to guarantee a non-zero "none" branch so the
//   compiled CNF/XGCF always uses exactly L choice variables for L labels.

#include <cstdint>
#include <cmath>

static __device__ __forceinline__ double clamp_f64(double x, double lo, double hi) {
    return (x < lo) ? lo : ((x > hi) ? hi : x);
}

extern "C" __global__ void neural_fill_ad_chain_f32(
    const float* __restrict__ p,
    uint32_t n,
    const uint32_t* __restrict__ slot_var,
    double eps,
    double min_p,
    double* __restrict__ var_log_true,
    double* __restrict__ var_log_false
) {
    // Single-threaded per (group, call) kernel: n is typically small (e.g., 10–1000),
    // and this is called per neural predicate instance (few per query). The dominant
    // cost is circuit eval/backprop; this keeps logic simple and deterministic.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    const double scale = 1.0 - eps;
    double remaining = 1.0;

    for (uint32_t i = 0; i < n; i++) {
        double pi = static_cast<double>(p[i]);
        // Treat NaN/negative as 0; caller should pass softmax outputs.
        if (!(pi >= 0.0) || isnan(pi)) {
            pi = 0.0;
        }
        pi = (pi < min_p) ? min_p : pi;

        double p_out = scale * pi;
        double denom = (remaining > min_p) ? remaining : min_p;
        double q = p_out / denom;
        q = clamp_f64(q, min_p, 1.0 - min_p);

        uint32_t var = slot_var[i];
        var_log_true[var] = log(q);
        var_log_false[var] = log(1.0 - q);

        remaining -= p_out;
    }
}

extern "C" __global__ void neural_scatter_ad_chain_grads_f32(
    const float* __restrict__ p,
    uint32_t n,
    const uint32_t* __restrict__ slot_var,
    double eps,
    double min_p,
    const double* __restrict__ grad_true,
    const double* __restrict__ grad_false,
    uint8_t mode,
    float* __restrict__ out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    // Shared scratch (3 * n doubles): p_out[i], remaining[i], dlogZ/dq[i]
    extern __shared__ double sh[];
    double* p_out = sh;
    double* remaining_at = sh + (static_cast<size_t>(n) * 1);
    double* dlogdq = sh + (static_cast<size_t>(n) * 2);

    const double scale = 1.0 - eps;

    // Forward pass: compute p_out, remaining, q, and dlogZ/dq for each link.
    double remaining = 1.0;
    for (uint32_t i = 0; i < n; i++) {
        double pi = static_cast<double>(p[i]);
        if (!(pi >= 0.0) || isnan(pi)) {
            pi = 0.0;
        }
        pi = (pi < min_p) ? min_p : pi;

        double po = scale * pi;
        double denom = (remaining > min_p) ? remaining : min_p;
        double q = po / denom;
        q = clamp_f64(q, min_p, 1.0 - min_p);

        uint32_t var = slot_var[i];
        double gT = grad_true[var];
        double gF = grad_false[var];

        // dlogZ/dq = dlogZ/dln(q) * dln(q)/dq + dlogZ/dln(1-q) * dln(1-q)/dq
        //          = gT*(1/q) + gF*(-1/(1-q))
        double inv_q = 1.0 / q;
        double inv_1mq = 1.0 / (1.0 - q);
        double d = gT * inv_q + gF * (-inv_1mq);

        p_out[i] = po;
        remaining_at[i] = denom; // remaining before consuming i
        dlogdq[i] = d;

        remaining -= po;
    }

    // Backward sweep: compute dlogZ/dp_out and accumulate suffix term.
    double S = 0.0;
    for (int64_t i = static_cast<int64_t>(n) - 1; i >= 0; i--) {
        double rem = remaining_at[i];
        double d = dlogdq[i];

        double inv_rem = 1.0 / rem;
        double dlogdp_out = d * inv_rem + S;
        double grad = scale * dlogdp_out; // dlogZ/dp = (1-eps) * dlogZ/dp_out

        if (mode == 0) {
            out[i] = static_cast<float>(grad);
        } else {
            out[i] -= static_cast<float>(grad);
        }

        double inv_rem2 = inv_rem * inv_rem;
        S += d * (p_out[i] * inv_rem2);
    }
}

