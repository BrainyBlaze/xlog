#pragma once

#include <stdint.h>

namespace semantic_program {

constexpr uint32_t kRuleCapacity = 256;

// The cold compiler assigns predicate indices from the admitted source schema.
// Body positions share variable indices, so equalities are retained verbatim.
struct Fact {
    uint32_t predicate;
    uint32_t first;
    uint32_t second;
};

struct Rule {
    uint32_t head_predicate;
    uint32_t left_predicate;
    uint32_t right_predicate;
    uint32_t body_count;
    uint32_t head_variables[2];
    uint32_t left_variables[2];
    uint32_t right_variables[2];
};

static_assert(sizeof(Fact) == 12, "semantic program fact ABI drift");
static_assert(sizeof(Rule) == 40, "semantic program rule ABI drift");

enum class Status : uint32_t {
    Ok,
    InvalidProgram,
    FactCapacity,
    FuelExhausted,
    RuleCapacity,
    ProtectedObservation,
};

struct Result {
    Status status;
    uint32_t fact_count;
    uint32_t derived_count;
    uint64_t charged_pairs;
};

struct Bank {
    const uint64_t* words;
    uint32_t predicate_count;
    uint32_t initial_count;
    uint32_t rule_count;
    uint32_t query_count;
};

struct DualSupport {
    uint8_t* truth;
    uint64_t* lineage;
    uint32_t words;
};

struct Observation { Fact fact;uint64_t truth,present; };
struct NoObservations {
    __device__ uint32_t size() const { return 0; }
    __device__ Observation operator[](uint32_t) const { return {}; }
    __device__ bool protected_fact(Fact) const { return false; }
};

__device__ uint32_t find_fact(const Fact* facts,uint32_t count,Fact needle) {
    for(uint32_t index=0;index<count;++index) {
        const Fact fact=facts[index];
        if(fact.predicate==needle.predicate && fact.first==needle.first && fact.second==needle.second)return index;
    }
    return count;
}

__device__ bool contains(const Fact* facts, uint32_t count, Fact needle) {
    return find_fact(facts,count,needle)!=count;
}

__device__ bool valid_rule(const Rule& rule, uint32_t predicate_count) {
    if (rule.body_count < 1 || rule.body_count > 2 ||
        rule.head_predicate >= predicate_count ||
        rule.left_predicate >= predicate_count ||
        (rule.body_count == 2 && rule.right_predicate >= predicate_count) ||
        (rule.body_count == 1 &&
         (rule.right_predicate || rule.right_variables[0] || rule.right_variables[1]))) {
        return false;
    }
    bool bound[4] = {};
    for (uint32_t position = 0; position < 2; ++position) {
        if (rule.left_variables[position] > 3 ||
            (rule.body_count == 2 && rule.right_variables[position] > 3) ||
            rule.head_variables[position] > 3) {
            return false;
        }
        bound[rule.left_variables[position]] = true;
        if (rule.body_count == 2) bound[rule.right_variables[position]] = true;
    }
    return bound[rule.head_variables[0]] && bound[rule.head_variables[1]];
}

__device__ Fact bank_fact(const Bank& bank, uint64_t index) {
    const uint64_t* values = bank.words + 4 + 3 * uint64_t(index);
    return {uint32_t(values[0]), uint32_t(values[1]), uint32_t(values[2])};
}

__device__ Rule bank_rule(const Bank& bank, uint32_t index) {
    const uint64_t* values = bank.words + 4 + 3 * (uint64_t(bank.query_count) + bank.initial_count) +
                             10 * uint64_t(index);
    return {uint32_t(values[0]), uint32_t(values[1]), uint32_t(values[2]),
            uint32_t(values[3]), {uint32_t(values[4]), uint32_t(values[5])},
            {uint32_t(values[6]), uint32_t(values[7])},
            {uint32_t(values[8]), uint32_t(values[9])}};
}

// Decode only the exact canonical word bank emitted by the cold compiler.
// No caller may dereference a field until the supplied word count is checked.
__device__ bool decode_bank(const uint64_t* words, uint64_t word_count, Bank* result) {
    if (!words || !result || word_count < 7 || !words[0] || !words[3] ||
        words[0] > UINT32_MAX || words[1] > UINT32_MAX || words[2] > UINT32_MAX ||
        words[3] > UINT32_MAX) {
        return false;
    }
    const uint64_t initial_count = words[1];
    const uint64_t rule_count = words[2];
    const uint64_t query_count = words[3];
    const uint64_t fact_words = 3 * (query_count + initial_count);
    if (rule_count > (UINT64_MAX - 4 - fact_words) / 10 ||
        word_count != 4 + fact_words + 10 * rule_count) {
        return false;
    }
    Bank bank = {words, uint32_t(words[0]), uint32_t(initial_count),
                 uint32_t(rule_count), uint32_t(query_count)};
    for (uint64_t index = 0; index < query_count + initial_count; ++index) {
        const uint64_t* values = words + 4 + 3 * index;
        if (values[0] >= bank.predicate_count || values[1] > UINT32_MAX ||
            values[2] > UINT32_MAX) return false;
    }
    for (uint32_t index = 0; index < bank.rule_count; ++index) {
        const uint64_t* values = words + 4 + fact_words + 10 * uint64_t(index);
        for (uint32_t field = 0; field < 10; ++field)
            if (values[field] > UINT32_MAX) return false;
        if (!valid_rule(bank_rule(bank, index), bank.predicate_count)) return false;
    }
    *result = bank;
    return true;
}

__device__ bool bind_pair(const Fact& fact, const uint32_t variables[2],
                          uint32_t values[4], bool bound[4]) {
    const uint32_t terms[2] = {fact.first, fact.second};
    for (uint32_t position = 0; position < 2; ++position) {
        const uint32_t variable = variables[position];
        if (bound[variable] && values[variable] != terms[position]) return false;
        values[variable] = terms[position];
        bound[variable] = true;
    }
    return true;
}

// Materialize a least fixpoint into caller-owned resident scratch. A non-Ok
// result has zero visible facts; the caller must not publish partial scratch.
// Each body-pair inspection consumes fuel, including unsuccessful joins.
template<class Observations>
__device__ Result materialize(const Fact* initial, uint32_t initial_count,
                              const Rule* rules, uint32_t rule_count,
                              uint32_t predicate_count, Fact* output,
                              uint32_t capacity, uint64_t fuel,
                              DualSupport* support,const Observations& observations,
                              uint32_t protected_rule_begin) {
    if (!predicate_count || (initial_count && (!initial || !output)) ||
        (rule_count && (!rules || !output))) {
        return {Status::InvalidProgram, 0, 0, 0};
    }
    for (uint32_t index = 0; index < rule_count; ++index) {
        if (!valid_rule(rules[index], predicate_count)) {
            return {Status::InvalidProgram, 0, 0, 0};
        }
    }
    if(support && (!support->truth || (support->words && !support->lineage)))
        return {Status::InvalidProgram,0,0,0};
    uint64_t charged=0;
    if(support)for(uint32_t fact=0;fact<capacity;++fact) {
        support->truth[fact]=0;
        for(uint32_t channel=0;channel<2;++channel)
            for(uint32_t word=0;word<support->words;++word) {
                if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                ++charged;support->lineage[(uint64_t(fact)*2+channel)*support->words+word]=0;
            }
    }
    uint32_t count = 0;
    for (uint32_t index = 0; index < initial_count; ++index) {
        if (initial[index].predicate >= predicate_count) {
            return {Status::InvalidProgram, 0, 0, 0};
        }
        uint32_t slot=find_fact(output,count,initial[index]);
        if (slot==count) {
            if (count == capacity) return {Status::FactCapacity, 0, 0, 0};
            output[count++] = initial[index];
        }
        if(support)support->truth[slot]|=1;
    }
    for(uint32_t binding=0;binding<observations.size();++binding) {
        const Observation observation=observations[binding];
        if(observation.present>1 || observation.truth>3 || observation.fact.predicate>=predicate_count ||
           !support || binding/64>=support->words)return {Status::InvalidProgram,0,0,charged};
        if(!observation.present)continue;
        uint32_t slot=find_fact(output,count,observation.fact);
        if(slot==count) {
            if(count==capacity)return {Status::FactCapacity,0,0,charged};
            output[count++]=observation.fact;
        }
        support->truth[slot]|=uint8_t(observation.truth);
        for(uint32_t channel=0;channel<2;++channel)if(observation.truth&(uint64_t(1)<<channel)) {
            if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
            ++charged;
            support->lineage[(uint64_t(slot)*2+channel)*support->words+binding/64]|=uint64_t(1)<<(binding%64);
        }
    }
    const uint32_t base_count = count;
    for (;;) {
        const uint32_t before = count;
        bool changed=false;
        for (uint32_t rule_index = 0; rule_index < rule_count; ++rule_index) {
            const Rule& rule = rules[rule_index];
            for (uint32_t left_index = 0; left_index < before; ++left_index) {
                if (charged == fuel) return {Status::FuelExhausted, 0, 0, charged};
                ++charged;
                const Fact left = output[left_index];
                if (left.predicate != rule.left_predicate || (support && !(support->truth[left_index]&1))) continue;
                const uint32_t right_count = rule.body_count == 2 ? before : 1;
                for (uint32_t right_index = 0; right_index < right_count; ++right_index) {
                    if (rule.body_count == 2) {
                        if (charged == fuel) return {Status::FuelExhausted, 0, 0, charged};
                        ++charged;
                    }
                    const Fact right = rule.body_count == 2 ? output[right_index] : left;
                    if (rule.body_count == 2 && (right.predicate != rule.right_predicate ||
                        (support && !(support->truth[right_index]&1)))) continue;
                    uint32_t values[4] = {};
                    bool bound[4] = {};
                    if (!bind_pair(left, rule.left_variables, values, bound) ||
                        (rule.body_count == 2 &&
                         !bind_pair(right, rule.right_variables, values, bound))) {
                        continue;
                    }
                    const Fact head = {rule.head_predicate,
                                       values[rule.head_variables[0]],
                                       values[rule.head_variables[1]]};
                    if(rule_index>=protected_rule_begin && observations.protected_fact(head))
                        return {Status::ProtectedObservation,0,0,charged};
                    uint32_t slot=find_fact(output,count,head);
                    if (slot==count) {
                        if (count == capacity) {
                            return {Status::FactCapacity, 0, 0, charged};
                        }
                        output[count++] = head;
                        changed=true;
                    }
                    if(support) {
                        changed|=!(support->truth[slot]&1);support->truth[slot]|=1;
                        for(uint32_t word=0;word<support->words;++word) {
                            if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                            ++charged;
                            uint64_t sources=support->lineage[uint64_t(left_index)*2*support->words+word];
                            if(rule.body_count==2)sources|=support->lineage[uint64_t(right_index)*2*support->words+word];
                            auto& retained=support->lineage[uint64_t(slot)*2*support->words+word];
                            changed|=(retained|sources)!=retained;retained|=sources;
                        }
                    }
                }
            }
        }
        if (!changed && count==before) break;
    }
    return {Status::Ok, count, count - base_count, charged};
}

__device__ Result materialize(const Fact* initial,uint32_t initial_count,
        const Rule* rules,uint32_t rule_count,uint32_t predicate_count,Fact* output,
        uint32_t capacity,uint64_t fuel) {
    return materialize(initial,initial_count,rules,rule_count,predicate_count,
        output,capacity,fuel,nullptr,NoObservations{},rule_count);
}

// The device bank is an integer word stream, not a packed Fact/Rule array.
// Decode into caller-owned resident storage, append the proposed rule edits,
// then execute the same least-fixpoint path as the cold initial program.
template<class Observations>
__device__ Result materialize_bank(const Bank& bank, const Rule* inserted,
                                   uint32_t inserted_count, Fact* input,
                                   uint32_t input_capacity, Rule* rules,
                                   uint32_t rule_capacity, Fact* output,
                                   uint32_t output_capacity, uint64_t fuel,
                                   const Rule* selected,uint32_t selected_count,
                                   DualSupport* support,const Observations& observations) {
    if (!bank.words || (inserted_count && !inserted) ||
        (bank.initial_count && !input) ||
        (bank.rule_count + uint64_t(inserted_count) && !rules)) {
        return {Status::InvalidProgram, 0, 0, 0};
    }
    if (bank.initial_count > input_capacity) return {Status::FactCapacity, 0, 0, 0};
    if (bank.rule_count + uint64_t(selected_count) + inserted_count > rule_capacity)
        return {Status::RuleCapacity, 0, 0, 0};
    for (uint32_t index = 0; index < bank.initial_count; ++index)
        input[index] = bank_fact(bank, uint64_t(index) + bank.query_count);
    for (uint32_t index = 0; index < bank.rule_count; ++index)
        rules[index] = bank_rule(bank, index);
    for (uint32_t index = 0; index < selected_count; ++index) {
        if(!selected || !valid_rule(selected[index],bank.predicate_count))return {Status::InvalidProgram,0,0,0};
        rules[bank.rule_count+index]=selected[index];
    }
    for (uint32_t index = 0; index < inserted_count; ++index)
        rules[bank.rule_count + selected_count + index] = inserted[index];
    return materialize(input, bank.initial_count, rules,
                       bank.rule_count + selected_count + inserted_count, bank.predicate_count,
                       output, output_capacity, fuel,support,observations,bank.rule_count+selected_count);
}

__device__ Result materialize_bank(const Bank& bank,const Rule* inserted,uint32_t inserted_count,
        Fact* input,uint32_t input_capacity,Rule* rules,uint32_t rule_capacity,
        Fact* output,uint32_t output_capacity,uint64_t fuel) {
    return materialize_bank(bank,inserted,inserted_count,input,input_capacity,rules,
        rule_capacity,output,output_capacity,fuel,nullptr,0,nullptr,NoObservations{});
}

}  // namespace semantic_program
