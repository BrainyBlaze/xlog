#pragma once
#include <cstdint>

// Declarations exercise the supplied numerical interface without supplying a
// second implementation of policy arithmetic.
namespace supplied_policy::numeric {
inline constexpr std::uint32_t abi_version = 3;
inline constexpr std::uint32_t width = 128;
inline constexpr std::uint32_t field_count = 18;
struct FieldView {
    const float* embeddings;
    const float* biases;
    std::uint32_t cardinality;
    std::int32_t null_category;
};
struct FieldAdjoint { float* embeddings; float* biases; };
struct ErrorEnvelope { double ordinary; double underflow; };
struct DomainValue { double magnitude; ErrorEnvelope error; };
struct FieldErrorView { const ErrorEnvelope* embeddings; const ErrorEnvelope* biases; };
struct FieldErrorAdjoint { ErrorEnvelope* embeddings; ErrorEnvelope* biases; };
struct ReadoutHiddenErrors {
    const ErrorEnvelope* z; const ErrorEnvelope* recurrent; const ErrorEnvelope* position;
    ErrorEnvelope* hidden;
};
struct ReadoutScoreErrors {
    const ErrorEnvelope* hidden; FieldErrorView field; ErrorEnvelope* scores;
};
struct AdvanceErrors {
    const ErrorEnvelope* recurrent; const ErrorEnvelope* recurrence; const ErrorEnvelope* position;
    FieldErrorView field; ErrorEnvelope* successor;
};
struct ReadoutVjpErrors {
    const ErrorEnvelope* hidden; FieldErrorView field; const ErrorEnvelope* score_cotangent;
    ErrorEnvelope* z_adjoint; ErrorEnvelope* recurrent_adjoint; ErrorEnvelope* position_adjoint;
    FieldErrorAdjoint field_adjoint;
};
struct AdvanceVjpErrors {
    const ErrorEnvelope* recurrent; const ErrorEnvelope* recurrence; const ErrorEnvelope* successor;
    FieldErrorView field; const ErrorEnvelope* successor_cotangent;
    ErrorEnvelope* recurrent_adjoint; ErrorEnvelope* recurrence_adjoint;
    ErrorEnvelope* position_adjoint; FieldErrorAdjoint field_adjoint;
};
struct FieldDomainView {
    const DomainValue* embeddings; const DomainValue* biases;
    std::uint32_t cardinality; std::int32_t null_category;
};
struct FieldDomainAdjoint { DomainValue* embeddings; DomainValue* biases; };
void readout_hidden(const float*, const float*, const float*, float*, std::uint32_t,
                    std::uint32_t, const ReadoutHiddenErrors* = nullptr);
void readout_scores(const float*, FieldView, float*, std::uint32_t, std::uint32_t,
                    const ReadoutScoreErrors* = nullptr);
void advance(const float*, const float*, const float*, FieldView, std::uint32_t, float*,
             std::uint32_t, std::uint32_t, const AdvanceErrors* = nullptr);
void readout_vjp(const float*, FieldView, const float*, float*, float*, float*, FieldAdjoint,
                 std::uint32_t, std::uint32_t, const ReadoutVjpErrors* = nullptr);
void advance_vjp(const float*, const float*, const float*, FieldView, std::uint32_t, const float*,
                 float*, float*, float*, FieldAdjoint, std::uint32_t, std::uint32_t,
                 const AdvanceVjpErrors* = nullptr);
void domain_readout_hidden(const DomainValue*, const DomainValue*, const DomainValue*,
                           DomainValue*, std::uint32_t, std::uint32_t);
void domain_readout_scores(const DomainValue*, FieldDomainView, DomainValue*,
                           std::uint32_t, std::uint32_t);
void domain_advance(const DomainValue*, const DomainValue*, const DomainValue*, FieldDomainView,
                    std::uint32_t, DomainValue*, std::uint32_t, std::uint32_t);
void domain_readout_vjp(const DomainValue*, FieldDomainView, const DomainValue*, DomainValue*,
                        DomainValue*, DomainValue*, FieldDomainAdjoint, std::uint32_t,
                        std::uint32_t);
void domain_advance_vjp(const DomainValue*, const DomainValue*, const DomainValue*, FieldDomainView,
                        std::uint32_t, const DomainValue*, DomainValue*, DomainValue*,
                        DomainValue*, FieldDomainAdjoint, std::uint32_t, std::uint32_t);
}
