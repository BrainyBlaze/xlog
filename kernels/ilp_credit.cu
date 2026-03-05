#include <stdint.h>
#include <math.h>

/// Write COO pairs from compacted fact indices.
/// For each compacted fact index, writes (fact_index, cidx) pairs at the given offset.
extern "C" __global__ void ilp_coo_fill(
    const uint32_t* compacted_fact_indices,  // [count]
    uint32_t cidx,                            // broadcast candidate index
    uint32_t count,                           // number of matching facts
    uint32_t offset,                          // write offset in COO arrays
    uint32_t* coo_fact,                       // output: fact indices [nnz_total]
    uint32_t* coo_cand                        // output: cand indices [nnz_total]
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= count) return;
    coo_fact[offset + tid] = compacted_fact_indices[tid];
    coo_cand[offset + tid] = cidx;
}

/// CSR credit gather + clamp + NLL loss (float).
/// For each fact, sums candidate probabilities of covering candidates,
/// then computes clamped credit and NLL loss contribution.
extern "C" __global__ void ilp_credit_forward_f32(
    const uint32_t* row_offsets,    // [num_facts + 1]
    const uint32_t* col_indices,    // [nnz]
    const float* cand_probs,        // [num_candidates]
    const uint8_t* is_positive,     // [num_facts] — 1=positive, 0=negative
    uint32_t num_facts,
    float eps,
    float* credit_out,              // [num_facts] — clamped value used in derivative
    float* loss_contrib             // [num_facts] — per-fact NLL contribution
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;
    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];
    float credit = 0.0f;
    for (uint32_t p = start; p < end; p++) {
        credit += cand_probs[col_indices[p]];
    }
    if (is_positive[f]) {
        float clamped = fmaxf(credit, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -logf(clamped);
    } else {
        float one_minus = 1.0f - credit;
        float clamped = fmaxf(one_minus, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -logf(clamped);
    }
}

/// CSR credit gather + clamp + NLL loss (double).
/// Same as ilp_credit_forward_f32 but in double precision.
extern "C" __global__ void ilp_credit_forward_f64(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const double* cand_probs,
    const uint8_t* is_positive,
    uint32_t num_facts,
    double eps,
    double* credit_out,
    double* loss_contrib
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;
    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];
    double credit = 0.0;
    for (uint32_t p = start; p < end; p++) {
        credit += cand_probs[col_indices[p]];
    }
    if (is_positive[f]) {
        double clamped = fmax(credit, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -log(clamped);
    } else {
        double one_minus = 1.0 - credit;
        double clamped = fmax(one_minus, eps);
        credit_out[f] = clamped;
        loss_contrib[f] = -log(clamped);
    }
}

/// Gradient scatter via CSR + atomicAdd (float).
/// For each fact, computes the gradient of the NLL loss w.r.t. each covering
/// candidate's probability, and atomically accumulates into d_cand_probs.
///   positive: d_loss/d_cand = -1/credit_out[f]
///   negative: d_loss/d_cand = +1/credit_out[f]
extern "C" __global__ void ilp_credit_backward_f32(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const float* credit_out,        // [num_facts] — clamped value from forward
    const uint8_t* is_positive,
    uint32_t num_facts,
    float* d_cand_probs             // [num_candidates] — accumulated gradients (MUST be zeroed before call)
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;
    // d_loss/d_cand_probs[c]:
    //   positive: -1/credit_out[f]  (credit_out = clamp(sum, eps))
    //   negative: +1/credit_out[f]  (credit_out = clamp(1-sum, eps))
    float grad;
    if (is_positive[f]) {
        grad = -1.0f / credit_out[f];
    } else {
        grad = 1.0f / credit_out[f];
    }
    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];
    for (uint32_t p = start; p < end; p++) {
        atomicAdd(&d_cand_probs[col_indices[p]], grad);
    }
}

/// Gradient scatter via CSR + atomicAdd (double).
/// Same as ilp_credit_backward_f32 but in double precision.
extern "C" __global__ void ilp_credit_backward_f64(
    const uint32_t* row_offsets,
    const uint32_t* col_indices,
    const double* credit_out,
    const uint8_t* is_positive,
    uint32_t num_facts,
    double* d_cand_probs
) {
    uint32_t f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= num_facts) return;
    double grad;
    if (is_positive[f]) {
        grad = -1.0 / credit_out[f];
    } else {
        grad = 1.0 / credit_out[f];
    }
    uint32_t start = row_offsets[f];
    uint32_t end   = row_offsets[f + 1];
    for (uint32_t p = start; p < end; p++) {
        atomicAdd(&d_cand_probs[col_indices[p]], grad);
    }
}
