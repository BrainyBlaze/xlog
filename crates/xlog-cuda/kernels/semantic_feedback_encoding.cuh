#pragma once
#include <stdint.h>
#include <string.h>
#include <math.h>
#include "semantic_value_schema.cuh"

#ifdef __CUDACC__
#define XLOG_FEEDBACK_INLINE __host__ __device__ inline
#else
#define XLOG_FEEDBACK_INLINE inline
#endif

namespace semantic_feedback {
struct Statement { const uint8_t* bytes; uint64_t length; };

// Invalid records do not name a statement or a support value. In particular,
// none of their remaining bytes can influence address arithmetic or features.
template<class Record>
XLOG_FEEDBACK_INLINE uint64_t validate(const Record& record, const Statement& statement,
        uint64_t statement_count, uint64_t capacity,const uint64_t* value) {
    if (record.valid == 0) return 0;
    if (record.valid != 1 || record.statement_index >= statement_count ||
        record.pro > 1 || record.contra > 1 || record.statement_length_bytes == 0 ||
        record.statement_length_bytes > capacity || !value) return 1;
    if (!statement.bytes || record.statement_offset_bytes > statement.length ||
        record.statement_length_bytes > statement.length - record.statement_offset_bytes) return 1;
    namespace r=semantic_value::receipt;
    if(value[r::status]!=0 || value[r::completion]!=1 || value[r::query_kind]>1 ||
       value[r::estimator_kind]>=semantic_value::feature::estimators ||
       value[r::assignment_defined]!=(value[r::query_kind]==0) ||
       value[r::mass_defined]!=(value[r::query_kind]==1) || value[r::error_bound_defined]>1)return 1;
    return 0;
}

XLOG_FEEDBACK_INLINE float numeric_coordinate(uint64_t bits,bool outward=false) {
    double value;
    memcpy(&value,&bits,sizeof(value));
#ifdef __CUDA_ARCH__
    const float rounded=__double2float_rn(value);
#else
    const float rounded=static_cast<float>(value);
#endif
    return outward && double(rounded)<value ? nextafterf(rounded,INFINITY) : rounded;
}

// Statement bytes are learned content. Native identities, generations, receipts
// and dependencies remain exclusively in the separate unlearned service tensors.
template<class Record>
XLOG_FEEDBACK_INLINE float coordinate(const Record& record, const Statement& statement,
        uint64_t capacity, uint64_t column,const uint64_t* value) {
    if (record.valid == 0) return 0.0f;
    if (column < 9 * capacity) {
        const uint64_t offset = column / 9, bit = column % 9;
        if (offset >= record.statement_length_bytes) return 0.0f;
        if (bit == 8) return 1.0f;
        const uint8_t value = statement.bytes[record.statement_offset_bytes + offset];
        return float((value >> bit) & 1);
    }
    const uint64_t tail = column - 9 * capacity;
    namespace r=semantic_value::receipt;
    namespace f=semantic_value::feature;
    if(tail<f::joint_masses) {
        if(!value[r::assignment_defined])return 0.0f;
        const uint64_t truth=record.pro|(record.contra<<1);
        const uint64_t one_hot=truth==1 ? 0 : truth==2 ? 1 : truth==3 ? 2 : 3;
        return float(tail==one_hot);
    }
    if(tail<f::absolute_error_bounds)return value[r::mass_defined] ?
        numeric_coordinate(value[r::joint_mass_00_f64_bits+tail-f::joint_masses]) : 0.0f;
    if(tail<f::query_kind)return value[r::error_bound_defined] ?
        numeric_coordinate(value[r::error_bound_00_f64_bits+tail-f::absolute_error_bounds],true) : 0.0f;
    if(tail<f::estimator_kind)return float(tail-f::query_kind==value[r::query_kind]);
    if(tail<f::value_availability)return float(tail-f::estimator_kind==value[r::estimator_kind]);
    if(tail<f::sample_count)return float(value[r::assignment_defined+tail-f::value_availability]);
    // The exact-structure producer performs no sampling. These fields are
    // explicitly undefined in the sole estimator's authoritative descriptor.
    return 0.0f;
}
}

#undef XLOG_FEEDBACK_INLINE
