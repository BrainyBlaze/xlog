#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#define __device__
#include "../kernels/semantic_program.cuh"
#include "../kernels/semantic_rule_action.cuh"

static void require(bool condition, const char* message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

static void composition_preserves_controls_and_existing_consequences() {
    using namespace semantic_program;
    // All predicate and variable indices are structural, not task-specific names.
    const Fact input[] = {
        {0, 1, 2}, {0, 4, 5}, {1, 2, 3}, {1, 1, 4}, {2, 6, 7},
    };
    const Rule rules[] = {
        {3, 0, 0, 1, {0, 1}, {0, 1}, {0, 0}},
        {3, 2, 0, 1, {0, 1}, {0, 1}, {0, 0}},
        {4, 2, 0, 1, {0, 1}, {0, 1}, {0, 0}},
        {4, 0, 1, 2, {0, 2}, {0, 1}, {1, 2}},
    };
    Fact output[64]{};
    const auto baseline = materialize(input, 5, rules, 3, 5, output, 64, 100000);
    require(baseline.status == Status::Ok, "base program must materialize");
    require(!contains(output, baseline.fact_count, {4, 1, 3}),
            "target must be absent before the structural insertion");
    require(contains(output, baseline.fact_count, {4, 6, 7}),
            "base consequence must not depend on the insertion");
    const auto result = materialize(input, 5, rules, 4, 5, output, 64, 100000);
    require(result.status == Status::Ok, "positive program must materialize");
    require(result.derived_count == baseline.derived_count + 1,
            "one new consequence must follow from the inserted rule");
    require(contains(output, result.fact_count, {4, 1, 3}), "join must derive target");
    require(!contains(output, result.fact_count, {4, 1, 4}), "join must preserve first control");
    require(!contains(output, result.fact_count, {4, 4, 3}), "join must preserve second control");
    require(contains(output, result.fact_count, {3, 1, 2}), "base rule must execute");
    require(contains(output, result.fact_count, {4, 6, 7}), "base head must execute");
}

static void cycles_reach_fixpoint_and_overflow_refuses() {
    using namespace semantic_program;
    const Fact input[] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}};
    const Rule rules[] = {
        {1, 0, 0, 1, {0, 1}, {0, 1}, {0, 0}},
        {1, 1, 0, 2, {0, 2}, {0, 1}, {1, 2}},
    };
    Fact output[32]{};
    auto result = materialize(input, 3, rules, 2, 2, output, 32, 100000);
    require(result.status == Status::Ok, "recursive program must reach fixpoint");
    require(contains(output, result.fact_count, {1, 1, 4}), "recursive closure missing");
    result = materialize(input, 3, rules, 2, 2, output, 4, 100000);
    require(result.status == Status::FactCapacity, "capacity must refuse incomplete closure");
    require(result.fact_count == 0, "capacity refusal must expose no partial relation");
    result = materialize(input, 3, rules, 2, 2, output, 32, 1);
    require(result.status == Status::FuelExhausted, "fuel must refuse incomplete closure");
    require(result.fact_count == 0, "fuel refusal must expose no partial relation");
}

static void one_rule_can_materialize_more_facts_than_edit_commands() {
    using namespace semantic_program;
    const Fact input[] = {{0, 1, 2}, {0, 2, 3}, {0, 3, 4}, {0, 4, 5}};
    const Rule insertion = {1, 0, 0, 1, {0, 1}, {0, 1}, {0, 0}};
    Fact output[16]{};
    const auto baseline = materialize(input, 4, nullptr, 0, 2, output, 16, 1000);
    const auto edited = materialize(input, 4, &insertion, 1, 2, output, 16, 1000);
    require(baseline.status == Status::Ok && baseline.derived_count == 0 &&
                edited.status == Status::Ok && edited.derived_count == 4,
            "one structural edit must account for all newly materialized facts");
}

static void repeated_variables_and_rule_sequence_are_executable() {
    using namespace semantic_program;
    const Fact input[] = {{0, 1, 1}, {0, 1, 2}};
    const Rule rules[] = {
        {1, 0, 0, 1, {0, 0}, {0, 0}, {0, 0}},
        {2, 1, 0, 2, {0, 1}, {0, 0}, {0, 1}},
    };
    Fact output[16]{};
    const auto result = materialize(input, 2, rules, 2, 3, output, 16, 1000);
    require(result.status == Status::Ok, "ordered rules must materialize");
    require(contains(output, result.fact_count, {1, 1, 1}), "equal arguments must match");
    require(!contains(output, result.fact_count, {1, 1, 2}), "repeated variable was ignored");
    require(contains(output, result.fact_count, {2, 1, 2}),
            "second rule must consume first rule consequence");
    const Rule unsafe = {1, 0, 0, 1, {2, 2}, {0, 1}, {0, 0}};
    require(materialize(input, 2, &unsafe, 1, 3, output, 16, 1000).status ==
                Status::InvalidProgram,
            "unbound head variable must be refused");
    const Rule malformed = {1, 0, 2, 1, {0, 1}, {0, 1}, {0, 0}};
    require(materialize(input, 2, &malformed, 1, 3, output, 16, 1000).status ==
                Status::InvalidProgram,
            "unused body fields must be canonical");
}

static void canonical_bank_decodes_without_truncation_or_extra_words() {
    using namespace semantic_program;
    const uint64_t words[] = {
        2, 1, 1,
        1, 1, 3, 1, 2, 3, 1, 3, 4,
        0, 1, 3,
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0,
    };
    Bank bank{};
    require(decode_bank(words, 25, &bank), "canonical program bank must decode");
    require(bank.predicate_count == 2 && bank.initial_count == 1 && bank.rule_count == 1,
            "program bank counts changed");
    require(bank_fact(bank, 3).predicate == 0 && bank_fact(bank, 3).second == 3,
            "input fact changed during bank decoding");
    require(bank_rule(bank, 0).head_predicate == 1,
            "rule changed during bank decoding");
    Fact input[8]{};
    Rule rules[8]{};
    Fact output[16]{};
    auto materialized = materialize_bank(bank, nullptr, 0, input, 8, rules, 8,
                                         output, 16, 1000);
    require(materialized.status == Status::Ok &&
                contains(output, materialized.fact_count, {1, 1, 3}),
            "base program bank must execute");
    require(!contains(output, materialized.fact_count, {0, 1, 1}),
            "uninserted rule must not affect the base program");
    const Rule inserted = {0, 1, 0, 1, {0, 0}, {0, 1}, {0, 0}};
    materialized = materialize_bank(bank, &inserted, 1, input, 8, rules, 8,
                                    output, 16, 1000);
    require(materialized.status == Status::Ok &&
                contains(output, materialized.fact_count, {0, 1, 1}),
            "inserted rule must alter the native closure");
    require(materialize_bank(bank, &inserted, 1, input, 8, rules, 1,
                             output, 16, 1000).status == Status::RuleCapacity,
            "insufficient rule storage must refuse the edit");
    require(!decode_bank(words, 24, &bank), "truncated bank must be refused");
    uint64_t extra[26]{};
    for (unsigned index = 0; index < 25; ++index) extra[index] = words[index];
    require(!decode_bank(extra, 26, &bank), "extra bank words must be refused");
    extra[0] = UINT64_C(1) << 32;
    require(!decode_bank(extra, 25, &bank), "u32 predicate truncation must be refused");
}

static void sampled_rule_categories_decode_exactly() {
    std::vector<uint64_t> books(25, 0);
    books[0] = books.size();
    books[1] = 5; // NULL, one ordinary graph target, three declared predicates.
    books.insert(books.end(), 10, 0);
    books.insert(books.end(), {7, 2, 1, 1, 0, 0, 0, 0, 0, 0});
    for (uint64_t predicate = 0; predicate < 3; ++predicate)
        books.insert(books.end(), {predicate, UINT64_MAX, 0, 0, 0, 0, 0, 0, 0, 0});
    books[2] = books.size();
    books[3] = 9; // NULL, one ordinary operand, three predicates, four variables.
    books.insert(books.end(), 6, 0);
    books.insert(books.end(), {0, 4, 1, 1, 0, 0});
    for (uint64_t predicate = 0; predicate < 3; ++predicate)
        books.insert(books.end(), {0, 0, 0, 1, predicate, 0});
    for (uint64_t variable = 0; variable < 4; ++variable)
        books.insert(books.end(), {0, 0, 0, 2, variable, 0});

    uint32_t choices[18] = {1, 2, 4, 2, 3, 5, 7, 5, 6, 6, 7};
    semantic_program::Rule rule{};
    require(semantic_rule_action::decode(books.data(), choices, 3, &rule),
            "typed rule categories must decode");
    require(rule.head_predicate == 2 && rule.left_predicate == 0 &&
                rule.right_predicate == 1 && rule.body_count == 2 &&
                rule.head_variables[0] == 0 && rule.head_variables[1] == 2 &&
                rule.left_variables[0] == 0 && rule.left_variables[1] == 1 &&
                rule.right_variables[0] == 1 && rule.right_variables[1] == 2,
            "sampled rule changed predicate or variable meaning");
    choices[6] = 8;
    require(!semantic_rule_action::decode(books.data(), choices, 3, &rule),
            "unbound head variable must be refused");
    choices[6] = 7;
    choices[3] = 1;
    require(!semantic_rule_action::decode(books.data(), choices, 3, &rule),
            "ordinary operand is not a rule predicate");
    choices[3] = 2;
    choices[11] = 1;
    require(!semantic_rule_action::decode(books.data(), choices, 3, &rule),
            "inactive support fields must remain NULL");
    choices[11] = 0;
    books[books[0] + 10 * choices[2]] = 3;
    require(!semantic_rule_action::decode(books.data(), choices, 3, &rule),
            "head predicate outside admitted schema must be refused");
}

static void rule_prefixes_preserve_a_legal_completion() {
    struct Field {
        uint32_t offset;
        uint32_t cardinality;
    };
    std::vector<uint64_t> books(25, 0);
    books[0] = books.size();
    books[1] = 5;
    books.insert(books.end(), 10, 0);
    books.insert(books.end(), {7, 2, 1, 1, 0, 0, 0, 0, 0, 0});
    for (uint64_t predicate = 0; predicate < 3; ++predicate)
        books.insert(books.end(), {predicate, UINT64_MAX, 0, 0, 0, 0, 0, 0, 0, 0});
    books[2] = books.size();
    books[3] = 9;
    books.insert(books.end(), 6, 0);
    books.insert(books.end(), {0, 4, 1, 1, 0, 0});
    for (uint64_t predicate = 0; predicate < 3; ++predicate)
        books.insert(books.end(), {0, 0, 0, 1, predicate, 0});
    for (uint64_t variable = 0; variable < 4; ++variable)
        books.insert(books.end(), {0, 0, 0, 2, variable, 0});
    Field fields[18]{};
    uint8_t support[18 * 9]{};
    for (uint32_t field = 0; field < 18; ++field) {
        fields[field] = {field * 9, field == 2 ? 5U : 9U};
        if (field >= 11) support[field * 9] = 1;
    }
    for (uint32_t field = 2; field <= 4; ++field)
        for (uint32_t category = 2; category <= 4; ++category)
            support[field * 9 + category] = 1;
    for (uint32_t field = 5; field <= 10; ++field)
        for (uint32_t category = 5; category <= 8; ++category)
            support[field * 9 + category] = 1;
    uint32_t choices[18] = {1, 2, 4, 2, 3, 5, 7, 5, 6, 6, 7};
    require(semantic_rule_action::prefix_completes(books.data(), fields, support,
                0, choices, 1, 2, 3), "rule opcode must have a legal completion");
    for (uint32_t field = 7; field <= 10; ++field)
        support[field * 9 + 7] = support[field * 9 + 8] = 0;
    require(!semantic_rule_action::prefix_completes(books.data(), fields, support,
                0, choices, 6, 7, 3), "unbound head prefix must not be sampled");
    support[10 * 9 + 7] = 1;
    require(semantic_rule_action::prefix_completes(books.data(), fields, support,
                0, choices, 6, 7, 3), "later body field may bind a head variable");
    support[10 * 9 + 7] = 0;
    choices[6] = 5;
    require(semantic_rule_action::prefix_completes(books.data(), fields, support,
                0, choices, 6, 5, 3), "repeated head variable needs one body binding");
    require(!semantic_rule_action::prefix_completes(books.data(), fields, support,
                0, choices, 3, 1, 3), "ordinary operand must not complete a body predicate");
}

static void program_queries_follow_the_inserted_rule_sequence() {
    using namespace semantic_program;
    const uint64_t words[] = {
        3, 3, 1,
        2, 1, 3, 2, 1, 4, 2, 4, 3,
        0, 1, 2, 0, 4, 5, 1, 2, 3,
        2, 0, 0, 1, 0, 1, 0, 1, 0, 0,
    };
    Bank bank{};
    require(decode_bank(words, sizeof(words) / sizeof(words[0]), &bank),
            "query fixture bank must decode");
    Fact input[8]{}, output[32]{};
    Rule rules[8]{};
    auto baseline = semantic_rule_action::evaluate(bank, nullptr, 0, input, 8,
                                                     rules, 8, output, 32, 10000);
    require(baseline.closure.status == Status::Ok &&
                baseline.truth[0] == 0 && baseline.truth[1] == 0 && baseline.truth[2] == 0,
            "absent program facts must be Neither, not False");
    const Rule insertion = {2, 0, 1, 2, {0, 2}, {0, 1}, {1, 2}};
    auto edited = semantic_rule_action::evaluate(bank, &insertion, 1, input, 8,
                                                   rules, 8, output, 32, 10000);
    require(edited.closure.status == Status::Ok && edited.truth[0] == 1 &&
                edited.truth[1] == 0 && edited.truth[2] == 0,
            "one join rule must change only the target query");
    const Rule second = {2, 0, 0, 2, {0, 2}, {0, 1}, {1, 2}};
    const Rule sequence[] = {insertion, second};
    edited = semantic_rule_action::evaluate(bank, sequence, 2, input, 8,
                                             rules, 8, output, 32, 10000);
    require(edited.closure.status == Status::Ok && edited.truth[0] == 1,
            "second edit must execute with the first edit present");
    edited = semantic_rule_action::evaluate(bank, &insertion, 1, input, 8,
                                             rules, 8, output, 3, 10000);
    require(edited.closure.status == Status::FactCapacity &&
                edited.closure.fact_count == 0,
            "capacity refusal must not expose partial query truth");
}

int main() {
    composition_preserves_controls_and_existing_consequences();
    cycles_reach_fixpoint_and_overflow_refuses();
    one_rule_can_materialize_more_facts_than_edit_commands();
    repeated_variables_and_rule_sequence_are_executable();
    canonical_bank_decodes_without_truncation_or_extra_words();
    sampled_rule_categories_decode_exactly();
    rule_prefixes_preserve_a_legal_completion();
    program_queries_follow_the_inserted_rule_sequence();
}
