#pragma once

#include <stdint.h>
#include "semantic_world_measure.cuh"

namespace semantic_program {

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
    CyclicImage,
    UnsupportedStatement,
};

struct Result {
    Status status;
    uint32_t fact_count;
    uint32_t derived_count;
    uint64_t charged_pairs;
};

struct Image {
    const uint64_t *words,*roots,*nodes,*guards,*leaves,*events,*blocks,*references;
    uint32_t root_count,node_count,guard_count,leaf_count,block_count;
    uint32_t event_count,reference_count,mutation_capacity;
    uint32_t local_root_begin,local_node_begin,local_guard_begin,local_leaf_begin,local_block_begin;
    uint32_t local_mutation_capacity;
    __device__ Fact guard(uint32_t index) const {
        const auto* row=guards+uint64_t(index)*5;
        return {uint32_t(row[0]),uint32_t(row[1]),uint32_t(row[2]),uint32_t(row[3])};
    }
};

struct Bank {
    const uint64_t* words;
    const uint64_t* predicate_arities;
    uint32_t predicate_count;
    uint32_t initial_count;
    uint32_t rule_count;
    uint32_t query_count;
    uint32_t fact_capacity,statement_count;
    const uint64_t* statement_rows;
    uint64_t word_count;
    Image image;
};

struct DualSupport {
    uint8_t* truth;
    uint64_t* lineage;
    uint32_t words;
    uint64_t* leaf_refs;
    semantic_world::Measure* world=nullptr;
    uint64_t* events=nullptr;
    uint64_t* lineage_events=nullptr;
};

// Retain the first proof separately in each part of the original sample space.
// A later alternative expands support by Boolean union, without overwriting the
// provenance of worlds already proved by an earlier derivation.
template<class Lineage>
__device__ bool merge_world_support(DualSupport& support,uint32_t fact,uint32_t channel,
        uint32_t event,const Lineage& lineage,uint64_t leaf=0) {
    auto& world=*support.world;
    if(fact>=world.fact_capacity || channel>1) {
        world.fail(semantic_world::MeasureStatus::InvalidSource);return false;
    }
    const uint64_t cell=uint64_t(fact)*2+channel;
    const uint32_t before=uint32_t(world.read(support.events,cell));
    const bool was_visible=world.visible(before);
    const uint32_t added=world.intersection(event,world.complement(before));
    if(!added || world.status!=semantic_world::MeasureStatus::Ok)return false;
    const uint32_t combined=world.unite(before,event);
    if(!world.charge(2))return false;
    support.events[cell]=combined;
    const bool now_visible=world.visible(combined);
    if(now_visible)support.truth[fact]|=uint8_t(1U<<channel);
    if(!was_visible && now_visible && support.leaf_refs) {
        if(!world.charge())return false;support.leaf_refs[cell]=leaf;
    }
    for(uint32_t origin=0;origin<world.lineage_events;++origin) {
        const uint32_t proof=world.intersection(added,lineage(origin));
        const uint64_t offset=cell*world.lineage_events+origin;
        const uint32_t retained=uint32_t(world.read(support.lineage_events,offset));
        const uint32_t merged=world.unite(retained,proof);
        if(!world.charge())return false;support.lineage_events[offset]=merged;
        if(world.visible(merged)) {
            if(!world.charge(2))return false;
            support.lineage[cell*support.words+origin/64]|=uint64_t(1)<<(origin%64);
        }
    }
    return world.status==semantic_world::MeasureStatus::Ok;
}

struct Observation { Fact fact;uint64_t truth,present; };
struct NoObservations {
    __device__ uint32_t size() const { return 0; }
    __device__ uint32_t protected_size() const { return 0; }
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
    const uint64_t* values = bank.words + 7 + bank.predicate_count + 4 * index;
    return {uint32_t(values[0]), uint32_t(values[1]),
            uint32_t(values[2]), uint32_t(values[3])};
}

__device__ Atom bank_atom(const uint64_t* values) {
    return {uint32_t(values[0]), uint32_t(values[1]),
            {{uint32_t(values[2]), uint32_t(values[3])},
             {uint32_t(values[4]), uint32_t(values[5])}}};
}

__device__ Rule rule_words(const uint64_t* values) {
    return {bank_atom(values + 1),
            {bank_atom(values + 7), bank_atom(values + 13), bank_atom(values + 19)},
            uint32_t(values[0])};
}

__device__ Rule bank_rule(const Bank& bank, uint32_t index) {
    return rule_words(bank.words + 7 + bank.predicate_count +
        4 * (uint64_t(bank.query_count) + bank.initial_count) + kRuleWords * uint64_t(index));
}

__device__ bool decode_image(const uint64_t* words,uint64_t count,const Bank& bank,Image* result) {
    if(!words || !result || count<16 || words[0]!=0x584c4f47464f5253ULL || words[1]!=1)return false;
    for(uint32_t i=2;i<16;++i)if(words[i]>UINT32_MAX)return false;
    if(!words[2] || !words[3] || !words[4] || !words[5] || !words[6] ||
       words[10]>=words[2] || words[11]>=words[3] || words[12]>=words[4] ||
       words[13]>=words[5] || words[14]>=words[6] || words[15]>words[9])return false;
    const uint64_t expected=16+words[2]+8*words[3]+5*words[4]+4*words[5]+2*words[7]+2*words[6]+words[8];
    if(count!=expected)return false;
    Image image{};
    image.words=words;
    image.root_count=uint32_t(words[2]);image.node_count=uint32_t(words[3]);
    image.guard_count=uint32_t(words[4]);image.leaf_count=uint32_t(words[5]);
    image.block_count=uint32_t(words[6]);image.event_count=uint32_t(words[7]);
    image.reference_count=uint32_t(words[8]);image.mutation_capacity=uint32_t(words[9]);
    image.local_root_begin=uint32_t(words[10]);image.local_node_begin=uint32_t(words[11]);
    image.local_guard_begin=uint32_t(words[12]);image.local_leaf_begin=uint32_t(words[13]);
    image.local_block_begin=uint32_t(words[14]);image.local_mutation_capacity=uint32_t(words[15]);
    image.roots=words+16;image.nodes=image.roots+image.root_count;
    image.guards=image.nodes+uint64_t(image.node_count)*8;
    image.leaves=image.guards+uint64_t(image.guard_count)*5;
    image.events=image.leaves+uint64_t(image.leaf_count)*4;
    image.blocks=image.events+uint64_t(image.event_count)*2;
    image.references=image.blocks+uint64_t(image.block_count)*2;
    bool active=false;
    for(uint32_t i=0;i<image.root_count;++i) {
        const uint64_t root=image.roots[i];
        if(root==UINT32_MAX)continue;
        if(root>=image.node_count || !image.nodes[root*8])return false;
        active=true;
    }
    if(!active)return false;
    for(uint32_t i=0;i<image.node_count;++i) {
        const auto* node=image.nodes+uint64_t(i)*8;
        for(uint32_t field=0;field<8;++field)if(node[field]>UINT32_MAX)return false;
        if(node[0]==0) { for(uint32_t field=1;field<8;++field)if(node[field])return false; }
        else if(node[0]==1) {
            if(node[1]>=image.guard_count || node[6] || node[7])return false;
            for(uint32_t edge=0;edge<4;++edge)
                if(node[2+edge]>=image.node_count || !image.nodes[node[2+edge]*8])return false;
        } else if(node[0]==2) {
            if(node[6]>=image.leaf_count || node[1] || node[2] || node[3] || node[4] || node[5] || node[7])return false;
        } else return false;
    }
    for(uint32_t i=0;i<image.guard_count;++i) {
        const auto* guard=image.guards+uint64_t(i)*5;
        for(uint32_t field=0;field<5;++field)if(guard[field]>UINT32_MAX)return false;
        if(!valid_fact(image.guard(i),bank))return false;
    }
    uint64_t events=0;
    for(uint32_t i=0;i<image.leaf_count;++i) {
        const auto* leaf=image.leaves+uint64_t(i)*4;
        if(leaf[3])return false;
        if(leaf[0]==0) { if(leaf[1] || leaf[2])return false; }
        else if(leaf[0]==1) {
            if(leaf[1]!=events || !leaf[2] || leaf[2]>image.event_count-events)return false;
            events+=leaf[2];
        } else if(leaf[0]==2) { if(leaf[1]>=image.block_count || leaf[2])return false; }
        else return false;
    }
    if(events!=image.event_count)return false;
    for(uint32_t i=0;i<image.event_count;++i)
        if(image.events[uint64_t(i)*2]>=image.guard_count ||
           image.events[uint64_t(i)*2+1]<1 || image.events[uint64_t(i)*2+1]>2)return false;
    uint64_t references=0;
    for(uint32_t i=0;i<image.block_count;++i) {
        const auto* block=image.blocks+uint64_t(i)*2;
        if(block[0]!=references || block[1]>image.reference_count-references)return false;
        references+=block[1];
        for(uint64_t j=0;j<block[1];++j) {
            const uint64_t rule=image.references[block[0]+j];
            if(rule>=bank.rule_count)return false;
            for(uint64_t before=0;before<j;++before)
                if(image.references[block[0]+before]==rule)return false;
        }
    }
    if(references!=image.reference_count)return false;
    *result=image;return true;
}

// Decode only the exact versioned canonical word bank emitted by the cold compiler.
// No caller may dereference a field until the supplied word count is checked.
__device__ bool decode_bank(const uint64_t* words, uint64_t word_count, Bank* result) {
    if (!words || !result || word_count < 7 || words[0] != 3 || !words[1] || !words[4] ||
        words[1] > UINT32_MAX || words[2] > UINT32_MAX || words[3] > UINT32_MAX ||
        words[4] > UINT32_MAX || !words[5] || words[5] > UINT32_MAX || words[6]>UINT32_MAX) return false;
    const uint64_t predicate_count = words[1];
    const uint64_t initial_count = words[2];
    const uint64_t rule_count = words[3];
    const uint64_t query_count = words[4];
    // Each count is at most u32::MAX, so these products and their sum fit u64.
    const uint64_t fact_words = 4 * (query_count + initial_count);
    const uint64_t statement_offset=7+predicate_count+fact_words+kRuleWords*rule_count;
    const uint64_t image_offset=statement_offset+5*words[6];
    if(word_count<image_offset+16)return false;
    Bank bank{};
    bank.words=words;bank.predicate_arities=words+7;
    bank.predicate_count=uint32_t(predicate_count);bank.initial_count=uint32_t(initial_count);
    bank.rule_count=uint32_t(rule_count);bank.query_count=uint32_t(query_count);
    bank.fact_capacity=uint32_t(words[5]);bank.word_count=word_count;
    bank.statement_count=uint32_t(words[6]);bank.statement_rows=words+statement_offset;
    for (uint32_t predicate = 0; predicate < bank.predicate_count; ++predicate)
        if (!bank_predicate_arity(bank, predicate)) return false;
    for (uint64_t index = 0; index < query_count + initial_count; ++index) {
        const uint64_t* values = words + 7 + predicate_count + 4 * index;
        for (uint32_t field = 0; field < 4; ++field)
            if (values[field] > UINT32_MAX) return false;
        if (!valid_fact(bank_fact(bank, index), bank)) return false;
    }
    for (uint32_t index = 0; index < bank.rule_count; ++index) {
        const uint64_t* values = words + 7 + predicate_count + fact_words +
                                 kRuleWords * uint64_t(index);
        for (uint32_t field = 0; field < kRuleWords; ++field)
            if (values[field] > UINT32_MAX) return false;
        if (!valid_rule(bank_rule(bank, index), bank)) return false;
    }
    for(uint32_t index=0;index<bank.statement_count;++index) {
        const auto* row=bank.statement_rows+uint64_t(index)*5;
        for(uint32_t field=0;field<5;++field)if(row[field]>UINT32_MAX)return false;
        if(index && bank.statement_rows[uint64_t(index-1)*5]>=row[0])return false;
        if(!valid_fact({uint32_t(row[1]),uint32_t(row[2]),uint32_t(row[3]),uint32_t(row[4])},bank))return false;
    }
    if(!decode_image(words+image_offset,word_count-image_offset,bank,&bank.image))return false;
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
                              uint32_t protected_rule_begin,uint32_t active_event=1,
                              uint32_t appended_rules=0) {
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
    auto* world=support ? support->world : nullptr;
    semantic_world::MeasureFuel measure_fuel(world,&charged,fuel);
    if(world && (!support->events || (world->lineage_events && !support->lineage_events)))
        return {Status::InvalidProgram,0,0,charged};
    for (uint32_t binding = 0; binding < observations.size(); ++binding) {
        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
        ++charged;
        const Observation observation = observations[binding];
        if (observation.present > 1 || observation.truth > 3 ||
            !valid_fact(observation.fact, bank) || !support || (uint64_t(binding)*2+1) / 64 >= support->words)
            return {Status::InvalidProgram, 0, 0, charged};
    }
    if(support)for(uint32_t fact=0;fact<capacity;++fact) {
        support->truth[fact]=0;
        if(support->leaf_refs) {
            if(fuel-charged<2)return {Status::FuelExhausted,0,0,charged};
            charged+=2;support->leaf_refs[uint64_t(fact)*2]=0;support->leaf_refs[uint64_t(fact)*2+1]=0;
        }
        for(uint32_t channel=0;channel<2;++channel)
            for(uint32_t word=0;word<support->words;++word) {
                if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                ++charged;support->lineage[(uint64_t(fact)*2+channel)*support->words+word]=0;
            }
        if(world)for(uint32_t channel=0;channel<2;++channel) {
            if(!world->charge())return {Status::FuelExhausted,0,0,charged};
            const uint64_t cell=uint64_t(fact)*2+channel;support->events[cell]=0;
            for(uint32_t origin=0;origin<world->lineage_events;++origin) {
                if(!world->charge())return {Status::FuelExhausted,0,0,charged};
                support->lineage_events[cell*world->lineage_events+origin]=0;
            }
        }
    }
    uint32_t count = 0;
    for (uint32_t index = 0; index < initial_count; ++index) {
        const uint32_t slot=find_fact(output,count,initial[index]);
        if (slot==count) {
            if (count == capacity) return {Status::FactCapacity, 0, 0, charged};
            output[count++] = initial[index];
        }
        if(world) {
            for(uint32_t channel=0;channel<2;++channel)
                merge_world_support(*support,slot,channel,world->fact_event(index,channel),
                    [](uint32_t) { return uint32_t(0); });
            if(world->status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
        } else if(support)support->truth[slot]|=1;
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
        const uint8_t added=uint8_t(observation.truth)&~support->truth[slot];
        if(world) {
            for(uint32_t channel=0;channel<2;++channel)if(observation.truth&(uint64_t(1)<<channel))
                merge_world_support(*support,slot,channel,world->view_condition,
                    [=](uint32_t origin) { return uint32_t(origin==binding*2+channel); });
            if(world->status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
            continue;
        }
        support->truth[slot]|=uint8_t(observation.truth);
        for(uint32_t channel=0;channel<2;++channel)if(added&(uint64_t(1)<<channel)) {
            if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
            ++charged;
            support->lineage[(uint64_t(slot)*2+channel)*support->words+(uint64_t(binding)*2+channel)/64]|=uint64_t(1)<<((uint64_t(binding)*2+channel)%64);
        }
    }
    const uint32_t base_count = count;
    for (;;) {
        const uint32_t before = count;
        bool changed=false;
        for (uint32_t rule_index = 0; rule_index < rule_count; ++rule_index) {
            const Rule& rule = rules[rule_index];
            uint32_t head_channel=0,body_channels=0,gate=1;
            if(world) {
                if(rule_index>=world->rule_capacity)return {Status::InvalidProgram,0,0,charged};
                gate=world->rule_gate(uint32_t(world->read(world->rule_ordinals,rule_index)),
                    appended_rules,&head_channel,&body_channels);
                gate=world->intersection(gate,active_event);
                if(world->status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
                if(!gate)continue;
            }
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
                const uint32_t channel=(body_channels>>depth)&1;
                if (support && (world ? !world->read(support->events,uint64_t(fact_index)*2+channel) :
                    !(support->truth[fact_index] & (uint32_t(1)<<channel)))) continue;
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
                uint32_t effect=gate;
                if(world) {
                    for(uint32_t body=0;body<rule.body_count;++body)
                        effect=world->intersection(effect,uint32_t(world->read(support->events,
                            uint64_t(matched[body])*2+((body_channels>>body)&1))));
                    if(world->status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
                    if(!effect)continue;
                }
                if(rule_index>=protected_rule_begin) {
                    for(uint32_t binding=0;binding<observations.protected_size();++binding) {
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
                    if(world) {
                        changed|=merge_world_support(*support,slot,head_channel,effect,
                            [&](uint32_t origin) {
                                uint32_t sources=0;
                                for(uint32_t body=0;body<rule.body_count;++body) {
                                    const uint64_t cell=uint64_t(matched[body])*2+((body_channels>>body)&1);
                                    sources=world->unite(sources,uint32_t(world->read(support->lineage_events,
                                        cell*world->lineage_events+origin)));
                                }
                                return sources;
                            });
                        if(world->status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
                        continue;
                    }
                    if(support->truth[slot]&1)continue;
                    changed=true;support->truth[slot]|=1;
                    for(uint32_t word=0;word<support->words;++word) {
                        if(charged==fuel)return {Status::FuelExhausted,0,0,charged};
                        ++charged;
                        uint64_t sources = 0;
                        for (uint32_t body = 0; body < rule.body_count; ++body)
                            sources |= support->lineage[uint64_t(matched[body])*2*support->words+word];
                        auto& retained=support->lineage[uint64_t(slot)*2*support->words+word];
                        retained=sources;
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

}  // namespace semantic_program
