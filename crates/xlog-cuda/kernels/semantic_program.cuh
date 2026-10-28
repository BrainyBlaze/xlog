#pragma once

#include <stdint.h>

namespace semantic_program {

constexpr uint32_t kRuleCapacity = 256;
constexpr uint32_t kBodyCapacity = 3;
constexpr uint32_t kVariableCapacity = 6;
constexpr uint32_t kRuleWords = 25;

// The cold compiler assigns predicate indices and arities from the admitted schema.
// Inactive unary argument storage is canonical zero, not another semantic argument.
struct Fact {
    uint32_t predicate;
    uint32_t arity;
    uint32_t first;
    uint32_t second;
};

struct Term {
    uint32_t kind;  // zero: variable, one: u32 constant
    uint32_t value;
};

struct Atom {
    uint32_t predicate;
    uint32_t arity;
    Term terms[2];
};

struct Rule {
    Atom head;
    Atom body[kBodyCapacity];
    uint32_t body_count;
};

static_assert(sizeof(Fact) == 16, "semantic program fact ABI drift");
static_assert(sizeof(Term) == 8, "semantic program term ABI drift");
static_assert(sizeof(Atom) == 24, "semantic program atom ABI drift");
static_assert(sizeof(Rule) == 100, "semantic program rule ABI drift");

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
    const uint64_t* predicate_arities;
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
};

__device__ uint32_t bank_predicate_arity(const Bank& bank, uint32_t predicate) {
    if (!bank.predicate_arities || predicate >= bank.predicate_count) return 0;
    const uint64_t arity = bank.predicate_arities[predicate];
    return arity == 1 || arity == 2 ? uint32_t(arity) : 0;
}

__device__ bool valid_fact(const Fact& fact, const Bank& bank) {
    return (fact.arity == 1 || fact.arity == 2) &&
           bank_predicate_arity(bank, fact.predicate) == fact.arity &&
           (fact.arity == 2 || fact.second == 0);
}

__device__ uint32_t find_fact(const Fact* facts,uint32_t count,Fact needle,
        uint64_t* charged=nullptr,uint64_t fuel=UINT64_MAX) {
    for(uint32_t index=0;index<count;++index) {
        if(charged) {
            if(*charged==fuel)return UINT32_MAX;
            ++*charged;
        }
        const Fact fact=facts[index];
        if(fact.predicate==needle.predicate && fact.arity==needle.arity &&
           fact.first==needle.first && fact.second==needle.second)return index;
    }
    return count;
}

__device__ bool contains(const Fact* facts, uint32_t count, Fact needle) {
    return find_fact(facts,count,needle)!=count;
}

__device__ bool zero_term(const Term& term) {
    return term.kind == 0 && term.value == 0;
}

__device__ bool zero_atom(const Atom& atom) {
    return atom.predicate == 0 && atom.arity == 0 &&
           zero_term(atom.terms[0]) && zero_term(atom.terms[1]);
}

__device__ bool valid_atom(const Atom& atom, const Bank& bank) {
    if ((atom.arity != 1 && atom.arity != 2) ||
        bank_predicate_arity(bank, atom.predicate) != atom.arity) return false;
    for (uint32_t position = 0; position < atom.arity; ++position) {
        const Term term = atom.terms[position];
        if (term.kind > 1 || (term.kind == 0 && term.value >= kVariableCapacity))
            return false;
    }
    return atom.arity == 2 || zero_term(atom.terms[1]);
}

__device__ bool valid_rule(const Rule& rule, const Bank& bank) {
    if (rule.body_count < 1 || rule.body_count > kBodyCapacity ||
        !valid_atom(rule.head, bank)) return false;
    bool bound[kVariableCapacity] = {};
    for (uint32_t index = 0; index < kBodyCapacity; ++index) {
        const Atom& atom = rule.body[index];
        if (index >= rule.body_count) {
            if (!zero_atom(atom)) return false;
            continue;
        }
        if (!valid_atom(atom, bank)) return false;
        for (uint32_t position = 0; position < atom.arity; ++position) {
            const Term term = atom.terms[position];
            if (term.kind == 0) bound[term.value] = true;
        }
    }
    for (uint32_t position = 0; position < rule.head.arity; ++position) {
        const Term term = rule.head.terms[position];
        if (term.kind == 0 && !bound[term.value]) return false;
    }
    return true;
}

__device__ Fact bank_fact(const Bank& bank, uint64_t index) {
    const uint64_t* values = bank.words + 5 + bank.predicate_count + 4 * index;
    return {uint32_t(values[0]), uint32_t(values[1]),
            uint32_t(values[2]), uint32_t(values[3])};
}

__device__ Atom bank_atom(const uint64_t* values) {
    return {uint32_t(values[0]), uint32_t(values[1]),
            {{uint32_t(values[2]), uint32_t(values[3])},
             {uint32_t(values[4]), uint32_t(values[5])}}};
}

__device__ Rule bank_rule(const Bank& bank, uint32_t index) {
    const uint64_t* values = bank.words + 5 + bank.predicate_count +
        4 * (uint64_t(bank.query_count) + bank.initial_count) + kRuleWords * uint64_t(index);
    return {bank_atom(values + 1),
            {bank_atom(values + 7), bank_atom(values + 13), bank_atom(values + 19)},
            uint32_t(values[0])};
}

// Decode only the exact versioned canonical word bank emitted by the cold compiler.
// No caller may dereference a field until the supplied word count is checked.
__device__ bool decode_bank(const uint64_t* words, uint64_t word_count, Bank* result) {
    if (!words || !result || word_count < 5 || words[0] != 2 || !words[1] || !words[4] ||
        words[1] > UINT32_MAX || words[2] > UINT32_MAX || words[3] > UINT32_MAX ||
        words[4] > UINT32_MAX) return false;
    const uint64_t predicate_count = words[1];
    const uint64_t initial_count = words[2];
    const uint64_t rule_count = words[3];
    const uint64_t query_count = words[4];
    // Each count is at most u32::MAX, so these products and their sum fit u64.
    const uint64_t fact_words = 4 * (query_count + initial_count);
    if (word_count != 5 + predicate_count + fact_words + kRuleWords * rule_count)
        return false;
    Bank bank = {words, words + 5, uint32_t(predicate_count), uint32_t(initial_count),
                 uint32_t(rule_count), uint32_t(query_count)};
    for (uint32_t predicate = 0; predicate < bank.predicate_count; ++predicate)
        if (!bank_predicate_arity(bank, predicate)) return false;
    for (uint64_t index = 0; index < query_count + initial_count; ++index) {
        const uint64_t* values = words + 5 + predicate_count + 4 * index;
        for (uint32_t field = 0; field < 4; ++field)
            if (values[field] > UINT32_MAX) return false;
        if (!valid_fact(bank_fact(bank, index), bank)) return false;
    }
    for (uint32_t index = 0; index < bank.rule_count; ++index) {
        const uint64_t* values = words + 5 + predicate_count + fact_words +
                                 kRuleWords * uint64_t(index);
        for (uint32_t field = 0; field < kRuleWords; ++field)
            if (values[field] > UINT32_MAX) return false;
        if (!valid_rule(bank_rule(bank, index), bank)) return false;
    }
    *result = bank;
    return true;
}

// The caller validates the atom and fact before binding. Only real arguments
// participate; equal variables unify and distinct variables may have equal values.
__device__ bool bind_atom(const Fact& fact, const Atom& atom,
                          uint32_t values[kVariableCapacity], bool bound[kVariableCapacity]) {
    if (fact.predicate != atom.predicate || fact.arity != atom.arity) return false;
    const uint32_t arguments[2] = {fact.first, fact.second};
    for (uint32_t position = 0; position < atom.arity; ++position) {
        const Term term = atom.terms[position];
        if (term.kind == 1) {
            if (term.value != arguments[position]) return false;
        } else {
            if (bound[term.value] && values[term.value] != arguments[position]) return false;
            values[term.value] = arguments[position];
            bound[term.value] = true;
        }
    }
    return true;
}

__device__ Fact substitute_head(const Atom& head, const uint32_t values[kVariableCapacity]) {
    Fact fact = {head.predicate, head.arity, 0, 0};
    const Term first = head.terms[0];
    fact.first = first.kind == 1 ? first.value : values[first.value];
    if (head.arity == 2) {
        const Term second = head.terms[1];
        fact.second = second.kind == 1 ? second.value : values[second.value];
    }
    return fact;
}

// Materialize a least fixpoint into caller-owned resident scratch. A non-Ok
// result has zero visible facts; the caller must not publish partial scratch.
// Every body-fact inspection and retained lineage-word operation consumes fuel.
template<class Observations>
__device__ Result materialize(const Fact* initial, uint32_t initial_count,
                              const Rule* rules, uint32_t rule_count,
                              const Bank& bank, Fact* output,
                              uint32_t capacity, uint64_t fuel,
                              DualSupport* support,const Observations& observations,
                              uint32_t protected_rule_begin) {
    if (!bank.predicate_count || !bank.predicate_arities ||
        (initial_count && (!initial || !output)) ||
        ((rule_count || observations.size()) && !output) || (rule_count && !rules) ||
        (support && ((!support->truth && capacity) ||
                     (support->words && !support->lineage))))
        return {Status::InvalidProgram, 0, 0, 0};
    for (uint32_t predicate = 0; predicate < bank.predicate_count; ++predicate)
        if (!bank_predicate_arity(bank, predicate)) return {Status::InvalidProgram, 0, 0, 0};
    for (uint32_t index = 0; index < rule_count; ++index)
        if (!valid_rule(rules[index], bank)) return {Status::InvalidProgram, 0, 0, 0};
    for (uint32_t index = 0; index < initial_count; ++index)
        if (!valid_fact(initial[index], bank)) return {Status::InvalidProgram, 0, 0, 0};
    uint64_t charged=0;
    for (uint32_t binding = 0; binding < observations.size(); ++binding) {
        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
        ++charged;
        const Observation observation = observations[binding];
        if (observation.present > 1 || observation.truth > 3 ||
            !valid_fact(observation.fact, bank) || !support || binding / 64 >= support->words)
            return {Status::InvalidProgram, 0, 0, 0};
    }
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
        const uint32_t slot=find_fact(output,count,initial[index]);
        if (slot==count) {
            if (count == capacity) return {Status::FactCapacity, 0, 0, charged};
            output[count++] = initial[index];
        }
        if(support)support->truth[slot]|=1;
    }
    for(uint32_t binding=0;binding<observations.size();++binding) {
        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
        ++charged;
        const Observation observation=observations[binding];
        if(!observation.present)continue;
        const uint32_t slot=find_fact(output,count,observation.fact,&charged,fuel);
        if(slot==UINT32_MAX)return {Status::FuelExhausted,0,0,charged};
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
            uint32_t next[kBodyCapacity] = {};
            uint32_t matched[kBodyCapacity] = {};
            uint32_t values[kBodyCapacity + 1][kVariableCapacity] = {};
            bool bound[kBodyCapacity + 1][kVariableCapacity] = {};
            uint32_t depth = 0;
            // Each depth extends the same substitution with one authored body atom.
            // Backtracking restores the retained prefix rather than losing equalities.
            for (;;) {
                if (next[depth] == before) {
                    if (depth == 0) break;
                    --depth;
                    continue;
                }
                const uint32_t fact_index = next[depth]++;
                if (charged == fuel) return {Status::FuelExhausted, 0, 0, charged};
                ++charged;
                if (support && !(support->truth[fact_index] & 1)) continue;
                for (uint32_t variable = 0; variable < kVariableCapacity; ++variable) {
                    values[depth + 1][variable] = values[depth][variable];
                    bound[depth + 1][variable] = bound[depth][variable];
                }
                if (!bind_atom(output[fact_index], rule.body[depth],
                               values[depth + 1], bound[depth + 1])) continue;
                matched[depth] = fact_index;
                if (depth + 1 < rule.body_count) {
                    ++depth;
                    next[depth] = 0;
                    continue;
                }
                const Fact head = substitute_head(rule.head, values[depth + 1]);
                if(rule_index>=protected_rule_begin) {
                    for(uint32_t binding=0;binding<observations.size();++binding) {
                        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                        ++charged;
                        const Fact target=observations[binding].fact;
                        if(target.predicate==head.predicate && target.arity==head.arity &&
                           target.first==head.first && target.second==head.second)
                            return {Status::ProtectedObservation,0,0,charged};
                    }
                }
                const uint32_t slot=find_fact(output,count,head);
                if (slot==count) {
                    if (count == capacity) return {Status::FactCapacity, 0, 0, charged};
                    output[count++] = head;
                    changed=true;
                }
                if(support) {
                    changed|=!(support->truth[slot]&1);support->truth[slot]|=1;
                    for(uint32_t word=0;word<support->words;++word) {
                        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                        ++charged;
                        uint64_t sources = 0;
                        for (uint32_t body = 0; body < rule.body_count; ++body)
                            sources |= support->lineage[uint64_t(matched[body])*2*support->words+word];
                        auto& retained=support->lineage[uint64_t(slot)*2*support->words+word];
                        changed|=(retained|sources)!=retained;retained|=sources;
                    }
                }
            }
        }
        if (!changed && count==before) break;
    }
    return {Status::Ok, count, count - base_count, charged};
}

__device__ Result materialize(const Fact* initial,uint32_t initial_count,
        const Rule* rules,uint32_t rule_count,const Bank& bank,Fact* output,
        uint32_t capacity,uint64_t fuel) {
    return materialize(initial,initial_count,rules,rule_count,bank,
        output,capacity,fuel,nullptr,NoObservations{},rule_count);
}

// Decode the word bank into caller-owned resident storage, append selected and
// proposed rules, and execute the same substitution and least-fixpoint engine.
template<class Observations>
__device__ Result materialize_bank(const Bank& bank, const Rule* inserted,
                                   uint32_t inserted_count, Fact* input,
                                   uint32_t input_capacity, Rule* rules,
                                   uint32_t rule_capacity, Fact* output,
                                   uint32_t output_capacity, uint64_t fuel,
                                   const Rule* selected,uint32_t selected_count,
                                   DualSupport* support,const Observations& observations) {
    const uint64_t total_rules = uint64_t(bank.rule_count) + selected_count + inserted_count;
    if (!bank.words || !bank.predicate_arities || (inserted_count && !inserted) ||
        (selected_count && !selected) || (bank.initial_count && !input) ||
        (total_rules && !rules)) return {Status::InvalidProgram, 0, 0, 0};
    if (bank.initial_count > input_capacity) return {Status::FactCapacity, 0, 0, 0};
    if (total_rules > rule_capacity) return {Status::RuleCapacity, 0, 0, 0};
    for (uint32_t index = 0; index < bank.initial_count; ++index)
        input[index] = bank_fact(bank, uint64_t(index) + bank.query_count);
    for (uint32_t index = 0; index < bank.rule_count; ++index)
        rules[index] = bank_rule(bank, index);
    for (uint32_t index = 0; index < selected_count; ++index)
        rules[bank.rule_count + index] = selected[index];
    for (uint32_t index = 0; index < inserted_count; ++index)
        rules[bank.rule_count + selected_count + index] = inserted[index];
    return materialize(input, bank.initial_count, rules, uint32_t(total_rules), bank,
                       output, output_capacity, fuel,support,observations,bank.rule_count);
}

__device__ Result materialize_bank(const Bank& bank,const Rule* inserted,uint32_t inserted_count,
        Fact* input,uint32_t input_capacity,Rule* rules,uint32_t rule_capacity,
        Fact* output,uint32_t output_capacity,uint64_t fuel) {
    return materialize_bank(bank,inserted,inserted_count,input,input_capacity,rules,
        rule_capacity,output,output_capacity,fuel,nullptr,0,nullptr,NoObservations{});
}

}  // namespace semantic_program
