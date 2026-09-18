#pragma once
#include <cstdint>

// Declarations exercise the supplied numerical interface without supplying a
// second implementation of policy arithmetic.
namespace supplied_policy::numeric {
inline constexpr std::uint32_t abi_version = 1;
inline constexpr std::uint32_t width = 128;
inline constexpr std::uint32_t field_count = 18;
struct FieldView {
    const float* embeddings;
    const float* biases;
    std::uint32_t cardinality;
    std::int32_t null_category;
};
struct FieldAdjoint { float* embeddings; float* biases; };
void readout_hidden(const float*, const float*, const float*, float*, std::uint32_t, std::uint32_t);
void readout_scores(const float*, FieldView, float*, std::uint32_t, std::uint32_t);
void advance(const float*, const float*, const float*, FieldView, std::uint32_t, float*,
             std::uint32_t, std::uint32_t);
void readout_vjp(const float*, FieldView, const float*, float*, float*, float*, FieldAdjoint,
                 std::uint32_t, std::uint32_t);
void advance_vjp(const float*, const float*, const float*, FieldView, std::uint32_t, const float*,
                 float*, float*, float*, FieldAdjoint, std::uint32_t, std::uint32_t);
}
