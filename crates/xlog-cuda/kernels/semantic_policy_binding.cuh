#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "semantic_policy_selection.cuh"

// The retained policy tape and parameter layout implement numerical ABI 1.
// The supplied producer owns its arithmetic; this boundary checks the layout
// and operations consumed by the resident forward and derivative paths.
namespace semantic_policy_abi {
static_assert(model_policy::abi_version == 1, "unsupported semantic policy ABI version");
static_assert(model_policy::width == 128, "unsupported semantic policy vector width");
static_assert(model_policy::field_count == 18, "unsupported semantic policy field count");
using Field = model_policy::FieldView;
using Adjoint = model_policy::FieldAdjoint;
static_assert(std::is_standard_layout_v<Field> && sizeof(Field) == 24 && alignof(Field) == 8,
              "invalid semantic policy field layout");
static_assert(offsetof(Field, embeddings) == 0 && offsetof(Field, biases) == 8 &&
              offsetof(Field, cardinality) == 16 && offsetof(Field, null_category) == 20,
              "invalid semantic policy field offsets");
static_assert(std::is_same_v<decltype(Field::embeddings), const float*> &&
              std::is_same_v<decltype(Field::biases), const float*> &&
              std::is_same_v<decltype(Field::cardinality), std::uint32_t> &&
              std::is_same_v<decltype(Field::null_category), std::int32_t>,
              "invalid semantic policy field types");
static_assert(std::is_standard_layout_v<Adjoint> && sizeof(Adjoint) == 16 && alignof(Adjoint) == 8 &&
              offsetof(Adjoint, embeddings) == 0 && offsetof(Adjoint, biases) == 8,
              "invalid semantic policy adjoint layout");
static_assert(std::is_same_v<decltype(Adjoint::embeddings), float*> &&
              std::is_same_v<decltype(Adjoint::biases), float*>,
              "invalid semantic policy adjoint types");
using ReadoutHidden = void (*)(const float*, const float*, const float*, float*,
                               std::uint32_t, std::uint32_t);
using ReadoutScores = void (*)(const float*, Field, float*, std::uint32_t, std::uint32_t);
using Advance = void (*)(const float*, const float*, const float*, Field, std::uint32_t, float*,
                         std::uint32_t, std::uint32_t);
using ReadoutVjp = void (*)(const float*, Field, const float*, float*, float*, float*, Adjoint,
                            std::uint32_t, std::uint32_t);
using AdvanceVjp = void (*)(const float*, const float*, const float*, Field, std::uint32_t,
                            const float*, float*, float*, float*, Adjoint,
                            std::uint32_t, std::uint32_t);
static_assert(std::is_same_v<decltype(&model_policy::readout_hidden), ReadoutHidden>,
              "invalid semantic policy readout_hidden signature");
static_assert(std::is_same_v<decltype(&model_policy::readout_scores), ReadoutScores>,
              "invalid semantic policy readout_scores signature");
static_assert(std::is_same_v<decltype(&model_policy::advance), Advance>,
              "invalid semantic policy advance signature");
static_assert(std::is_same_v<decltype(&model_policy::readout_vjp), ReadoutVjp>,
              "invalid semantic policy readout_vjp signature");
static_assert(std::is_same_v<decltype(&model_policy::advance_vjp), AdvanceVjp>,
              "invalid semantic policy advance_vjp signature");
}
