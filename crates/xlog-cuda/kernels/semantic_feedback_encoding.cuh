#pragma once
#include <stdint.h>

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
XLOG_FEEDBACK_INLINE uint64_t validate(const Record& record, const Statement* statements, uint64_t capacity) {
    if (record.valid == 0) return 0;
    if (record.valid != 1 || record.statement_index >= 2 || !statements ||
        record.pro > 1 || record.contra > 1 || record.statement_length_bytes == 0 ||
        record.statement_length_bytes > capacity) return 1;
    const auto& statement = statements[record.statement_index];
    if (!statement.bytes || record.statement_offset_bytes > statement.length ||
        record.statement_length_bytes > statement.length - record.statement_offset_bytes) return 1;
    return 0;
}

// Call only after validate succeeds; coordinate is in [0, 9*capacity+4).
// Each byte has eight LSB-first bits then presence. Required independent
// support fields each have their value bit then presence, in PRO/CONTRA order.
template<class Record>
XLOG_FEEDBACK_INLINE float coordinate(const Record& record, const Statement* statements,
        uint64_t capacity, uint64_t column) {
    if (record.valid == 0) return 0.0f;
    if (column < 9 * capacity) {
        const uint64_t offset = column / 9, bit = column % 9;
        if (offset >= record.statement_length_bytes) return 0.0f;
        if (bit == 8) return 1.0f;
        const uint8_t value = statements[record.statement_index].bytes[record.statement_offset_bytes + offset];
        return float((value >> bit) & 1);
    }
    const uint64_t tail = column - 9 * capacity;
    if (tail == 0) return float(record.pro);
    if (tail == 2) return float(record.contra);
    return 1.0f;
}
}

#undef XLOG_FEEDBACK_INLINE
