#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include "semantic_policy_selection.cuh"

// The supplied producer owns its arithmetic. Check both the value operations
// and their numerical transfer interfaces before compiling resident kernels.
namespace semantic_policy_abi {
static_assert(model_policy::abi_version == 3, "unsupported semantic policy ABI version");
static_assert(model_policy::width == 128, "unsupported semantic policy vector width");
static_assert(model_policy::field_count == 18, "unsupported semantic policy field count");
using Field = model_policy::FieldView;
using Adjoint = model_policy::FieldAdjoint;
using Error = model_policy::ErrorEnvelope;
using Domain = model_policy::DomainValue;
using FieldError = model_policy::FieldErrorView;
using FieldErrorAdjoint = model_policy::FieldErrorAdjoint;
using FieldDomain = model_policy::FieldDomainView;
using FieldDomainAdjoint = model_policy::FieldDomainAdjoint;
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
static_assert(std::is_standard_layout_v<Error> && sizeof(Error) == 16 && alignof(Error) == 8 &&
              offsetof(Error, ordinary) == 0 && offsetof(Error, underflow) == 8 &&
              std::is_same_v<decltype(Error::ordinary), double> &&
              std::is_same_v<decltype(Error::underflow), double>,
              "invalid semantic policy error envelope");
static_assert(std::is_standard_layout_v<Domain> && sizeof(Domain) == 24 && alignof(Domain) == 8 &&
              offsetof(Domain, magnitude) == 0 && offsetof(Domain, error) == 8 &&
              std::is_same_v<decltype(Domain::magnitude), double> &&
              std::is_same_v<decltype(Domain::error), Error>,
              "invalid semantic policy domain value");
static_assert(std::is_standard_layout_v<FieldError> && sizeof(FieldError) == 16 &&
              std::is_same_v<decltype(FieldError::embeddings), const Error*> &&
              std::is_same_v<decltype(FieldError::biases), const Error*> &&
              std::is_standard_layout_v<FieldErrorAdjoint> && sizeof(FieldErrorAdjoint) == 16 &&
              std::is_same_v<decltype(FieldErrorAdjoint::embeddings), Error*> &&
              std::is_same_v<decltype(FieldErrorAdjoint::biases), Error*>,
              "invalid semantic policy field error views");
static_assert(std::is_standard_layout_v<FieldDomain> && sizeof(FieldDomain) == 24 &&
              std::is_same_v<decltype(FieldDomain::embeddings), const Domain*> &&
              std::is_same_v<decltype(FieldDomain::biases), const Domain*> &&
              std::is_same_v<decltype(FieldDomain::cardinality), std::uint32_t> &&
              std::is_same_v<decltype(FieldDomain::null_category), std::int32_t> &&
              std::is_standard_layout_v<FieldDomainAdjoint> && sizeof(FieldDomainAdjoint) == 16 &&
              std::is_same_v<decltype(FieldDomainAdjoint::embeddings), Domain*> &&
              std::is_same_v<decltype(FieldDomainAdjoint::biases), Domain*>,
              "invalid semantic policy field domain views");
static_assert(std::is_same_v<decltype(model_policy::ReadoutHiddenErrors::hidden), Error*> &&
              std::is_same_v<decltype(model_policy::ReadoutScoreErrors::field), FieldError> &&
              std::is_same_v<decltype(model_policy::AdvanceErrors::field), FieldError> &&
              std::is_same_v<decltype(model_policy::ReadoutVjpErrors::field_adjoint),
                             FieldErrorAdjoint> &&
              std::is_same_v<decltype(model_policy::AdvanceVjpErrors::field_adjoint),
                             FieldErrorAdjoint>,
              "invalid semantic policy numerical contexts");
using ReadoutHidden = void (*)(const float*, const float*, const float*, float*,
                               std::uint32_t, std::uint32_t,
                               const model_policy::ReadoutHiddenErrors*);
using ReadoutScores = void (*)(const float*, Field, float*, std::uint32_t, std::uint32_t,
                               const model_policy::ReadoutScoreErrors*);
using Advance = void (*)(const float*, const float*, const float*, Field, std::uint32_t, float*,
                         std::uint32_t, std::uint32_t, const model_policy::AdvanceErrors*);
using ReadoutVjp = void (*)(const float*, Field, const float*, float*, float*, float*, Adjoint,
                            std::uint32_t, std::uint32_t, const model_policy::ReadoutVjpErrors*);
using AdvanceVjp = void (*)(const float*, const float*, const float*, Field, std::uint32_t,
                            const float*, float*, float*, float*, Adjoint,
                            std::uint32_t, std::uint32_t, const model_policy::AdvanceVjpErrors*);
using DomainReadoutHidden = void (*)(const Domain*, const Domain*, const Domain*, Domain*,
                                    std::uint32_t, std::uint32_t);
using DomainReadoutScores = void (*)(const Domain*, FieldDomain, Domain*,
                                    std::uint32_t, std::uint32_t);
using DomainAdvance = void (*)(const Domain*, const Domain*, const Domain*, FieldDomain,
                               std::uint32_t, Domain*, std::uint32_t, std::uint32_t);
using DomainReadoutVjp = void (*)(const Domain*, FieldDomain, const Domain*, Domain*, Domain*,
                                 Domain*, FieldDomainAdjoint, std::uint32_t, std::uint32_t);
using DomainAdvanceVjp = void (*)(const Domain*, const Domain*, const Domain*, FieldDomain,
                                  std::uint32_t, const Domain*, Domain*, Domain*, Domain*,
                                  FieldDomainAdjoint, std::uint32_t, std::uint32_t);
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
static_assert(std::is_same_v<decltype(&model_policy::domain_readout_hidden), DomainReadoutHidden>,
              "invalid semantic policy domain_readout_hidden signature");
static_assert(std::is_same_v<decltype(&model_policy::domain_readout_scores), DomainReadoutScores>,
              "invalid semantic policy domain_readout_scores signature");
static_assert(std::is_same_v<decltype(&model_policy::domain_advance), DomainAdvance>,
              "invalid semantic policy domain_advance signature");
static_assert(std::is_same_v<decltype(&model_policy::domain_readout_vjp), DomainReadoutVjp>,
              "invalid semantic policy domain_readout_vjp signature");
static_assert(std::is_same_v<decltype(&model_policy::domain_advance_vjp), DomainAdvanceVjp>,
              "invalid semantic policy domain_advance_vjp signature");
}
