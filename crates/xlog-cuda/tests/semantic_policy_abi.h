#pragma once
#include <cstdint>

// Declarations exercise the supplied numerical interface without supplying a
// second implementation of policy arithmetic.
namespace supplied_policy::numeric {
inline constexpr std::uint32_t abi_version = 4;
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
struct PrimalEnvelope { ErrorEnvelope error; double ideal_magnitude; };
struct DomainValue { double magnitude; ErrorEnvelope error; double ideal_magnitude; };
struct CoherentCotangent { DomainValue actor; DomainValue full; };
struct FieldErrorView { const PrimalEnvelope* embeddings; const PrimalEnvelope* biases; };
struct FieldCotangentAdjoint { CoherentCotangent* embeddings; CoherentCotangent* biases; };
struct ReadoutHiddenErrors {
    const PrimalEnvelope* z; const PrimalEnvelope* recurrent; const PrimalEnvelope* position;
    PrimalEnvelope* hidden;
};
struct ReadoutScoreErrors {
    const PrimalEnvelope* hidden; FieldErrorView field; PrimalEnvelope* scores;
};
struct AdvanceErrors {
    const PrimalEnvelope* recurrent; const PrimalEnvelope* recurrence; const PrimalEnvelope* position;
    FieldErrorView field; PrimalEnvelope* successor;
};
struct ReadoutVjpErrors {
    const PrimalEnvelope* hidden; FieldErrorView field; const CoherentCotangent* score_cotangent;
    CoherentCotangent* z_adjoint; CoherentCotangent* recurrent_adjoint; CoherentCotangent* position_adjoint;
    FieldCotangentAdjoint field_adjoint;
};
struct AdvanceVjpErrors {
    const PrimalEnvelope* recurrent; const PrimalEnvelope* recurrence; const PrimalEnvelope* successor;
    FieldErrorView field; const CoherentCotangent* successor_cotangent;
    CoherentCotangent* recurrent_adjoint; CoherentCotangent* recurrence_adjoint;
    CoherentCotangent* position_adjoint; FieldCotangentAdjoint field_adjoint;
};
struct FieldDomainView {
    const DomainValue* embeddings; const DomainValue* biases;
    std::uint32_t cardinality; std::int32_t null_category;
};
struct FieldDomainAdjoint { CoherentCotangent* embeddings; CoherentCotangent* biases; };
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
void domain_readout_vjp(const DomainValue*, FieldDomainView, const CoherentCotangent*, CoherentCotangent*,
                        CoherentCotangent*, CoherentCotangent*, FieldDomainAdjoint, std::uint32_t,
                        std::uint32_t);
void domain_advance_vjp(const DomainValue*, const DomainValue*, const DomainValue*, FieldDomainView,
                        std::uint32_t, const CoherentCotangent*, CoherentCotangent*, CoherentCotangent*,
                        CoherentCotangent*, FieldDomainAdjoint, std::uint32_t, std::uint32_t);
}
