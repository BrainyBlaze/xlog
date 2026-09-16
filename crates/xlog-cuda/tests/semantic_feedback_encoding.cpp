#include "semantic_feedback_encoding.cuh"
#include <cassert>
#include <cstdint>
#include <limits>

// The same coordinate encoder is instantiated by the CUDA producer. These
// integral fixtures test its learned-value contract without a device runtime.
struct Record {
    uint64_t valid, statement_index, statement_offset_bytes, statement_length_bytes, pro, contra;
};

int main() {
    const uint8_t bytes[] = {0x00, 0x01, 0x80, 0xa5};
    const semantic_feedback::Statement statements[] = {{bytes, 4}, {bytes, 1}, {bytes + 1, 2}};
    for (uint64_t pro = 0; pro != 2; ++pro) {
        for (uint64_t contra = 0; contra != 2; ++contra) {
            Record record{1, 0, 0, 4, pro, contra};
            assert(semantic_feedback::validate(record, statements, 4) == 0);
            for (uint64_t i = 0; i != 4; ++i) {
                for (uint64_t bit = 0; bit != 8; ++bit)
                    assert(semantic_feedback::coordinate(record, statements, 4, 9*i+bit) == float((bytes[i] >> bit) & 1));
                assert(semantic_feedback::coordinate(record, statements, 4, 9*i+8) == 1.0f);
            }
            const float tail[] = {float(pro), 1.0f, float(contra), 1.0f};
            for (uint64_t i = 0; i != 4; ++i)
                assert(semantic_feedback::coordinate(record, statements, 4, 36+i) == tail[i]);
        }
    }
    Record short_record{1, 1, 0, 1, 0, 0};
    assert(semantic_feedback::validate(short_record, statements, 4) == 0);
    for (uint64_t i = 0; i != 8; ++i)
        assert(semantic_feedback::coordinate(short_record, statements, 4, i) == 0.0f);
    assert(semantic_feedback::coordinate(short_record, statements, 4, 8) == 1.0f);
    for (uint64_t i = 9; i != 36; ++i)
        assert(semantic_feedback::coordinate(short_record, statements, 4, i) == 0.0f);

    // Invalid payload is never followed, including forged pointers/indices.
    const auto garbage = std::numeric_limits<uint64_t>::max();
    Record invalid{0, garbage, garbage, garbage, garbage, garbage};
    assert(semantic_feedback::validate(invalid, nullptr, 4) == 0);
    for (uint64_t i = 0; i != 40; ++i)
        assert(semantic_feedback::coordinate(invalid, nullptr, 4, i) == 0.0f);

    for (int field = 0; field != 6; ++field) {
        Record bad{1, 0, 0, 4, 0, 0};
        if (field == 0) bad.valid = 2;
        if (field == 1) bad.statement_index = 3;
        if (field == 2) bad.statement_offset_bytes = garbage;
        if (field == 3) bad.statement_length_bytes = 5;
        if (field == 4) bad.pro = 2;
        if (field == 5) bad.contra = 2;
        assert(semantic_feedback::validate(bad, statements, 4) != 0);
    }
    Record offset{1, 0, 1, 3, 1, 0};
    assert(semantic_feedback::validate(offset, statements, 4) == 0);
    assert(semantic_feedback::coordinate(offset, statements, 4, 0) == 1.0f);
    assert(semantic_feedback::coordinate(offset, statements, 4, 9+7) == 1.0f);
    assert(semantic_feedback::coordinate(offset, statements, 4, 27+8) == 0.0f);
}
