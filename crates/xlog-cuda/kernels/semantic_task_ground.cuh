#pragma once

#include <stdint.h>
#include "semantic_program.cuh"

namespace semantic_task_ground {

constexpr uint64_t kMagic = 0x584c4f4754475244ULL;
constexpr uint64_t kPublicationRole = 58;
constexpr uint64_t kQueryWords = 44;
constexpr uint64_t kObservationWords = 18;
constexpr uint64_t kBindingWords = 36;
constexpr uint64_t kInheritedMagic = 0x584c4f47494e484fULL;

struct Task {
    const uint64_t* words;
    uint32_t queries, bindings;
    uint64_t coding, support_count, level_count, goal_count;
    const uint64_t *scoring, *goal_witness, *query_rows, *supports, *levels, *goals, *binding_rows;
    const uint64_t* observation_rows;
    const uint64_t *requirements, *requirement_offsets, *attainment_order;
    uint64_t actor_eligible;
    semantic_program::Bank program;
    bool editable;
    __device__ const uint64_t* query(uint32_t index) const { return query_rows + uint64_t(index)*7; }
    __device__ const uint64_t* binding(uint32_t index) const { return binding_rows + uint64_t(index)*kBindingWords; }
    __device__ const uint64_t* observation(uint32_t index) const { return observation_rows + uint64_t(index)*8; }
    __device__ bool target(uint32_t query, uint64_t* truth) const {
        for(uint64_t index=0;index<goal_count;++index) {
            if(goals[index*3]==query) { *truth=goals[index*3+1];return true; }
        }
        return false;
    }
    __device__ bool observation_fact(uint32_t binding_index,semantic_program::Fact* fact) const {
        if(binding_index>=bindings || !editable)return false;
        const uint64_t* row=observation(binding_index);
        if(!(row[0]|row[1]|row[2]|row[3]) || row[4]>=program.predicate_count ||
           row[5]>UINT32_MAX || row[6]>UINT32_MAX || row[7]>UINT32_MAX)return false;
        *fact={uint32_t(row[4]),uint32_t(row[5]),uint32_t(row[6]),uint32_t(row[7])};
        return semantic_program::valid_fact(*fact,program);
    }
};

// One exact decoder for the native owner's variable-length task bank. No
// consumer derives an extent from a public pointer or an unbounded sentinel.
__device__ bool decode(const uint64_t* words,uint64_t count,Task* output) {
    if(!words || !output || count<39 || words[0]!=9 || !words[1] ||
       !words[6] || words[6]>UINT32_MAX || words[7]>UINT32_MAX || words[8]>1 ||
       (words[8]==0 ? words[7]!=0 : words[7]==0) || words[10]>words[6] ||
       words[11]>words[6] || words[37]>1)return false;
    Task task{};
    task.words=words;task.queries=uint32_t(words[6]);task.bindings=uint32_t(words[7]);
    task.coding=words[8];task.support_count=words[9];task.level_count=words[10];task.goal_count=words[11];
    task.scoring=words+13;task.goal_witness=words+19;task.actor_eligible=words[37];
    uint64_t offset=38;
    auto take=[&](uint64_t entries,uint64_t width,const uint64_t** pointer) {
        if(entries>(count-offset)/width)return false;
        *pointer=words+offset;offset+=entries*width;return true;
    };
    if(!take(task.queries,7,&task.query_rows) || !take(task.support_count,5,&task.supports) ||
       !take(task.level_count,1,&task.levels) || !take(task.goal_count,3,&task.goals) ||
       !take(task.bindings,kBindingWords,&task.binding_rows) ||
       !take(task.bindings,8,&task.observation_rows) || offset==count)return false;
    const uint64_t requirement_words=words[offset++];
    if(requirement_words>count-offset)return false;
    if(task.coding) {
        if(task.goal_count!=task.queries || requirement_words<1+uint64_t(task.queries)*5 ||
           words[offset]!=task.queries)return false;
        task.requirements=words+offset;
        task.requirement_offsets=task.requirements+1;
        task.attainment_order=task.requirement_offsets+task.queries;
        uint64_t row_offset=1+uint64_t(task.queries)*2;
        for(uint32_t query=0;query<task.queries;++query) {
            if(task.requirement_offsets[query]!=row_offset || requirement_words-row_offset<3)return false;
            const auto* row=task.requirements+row_offset;
            if(row[0]!=query || row[1]>task.queries || row[2]>task.bindings ||
               (!row[1]&&!row[2]) || row[1]+row[2]>requirement_words-row_offset-3)return false;
            for(uint64_t index=0;index<row[1];++index)
                if(row[3+index]>=task.queries || row[3+index]==query ||
                   (index && row[3+index-1]>=row[3+index]))return false;
            for(uint64_t index=0;index<row[2];++index) {
                const uint64_t binding=row[3+row[1]+index];
                if(binding>=task.bindings || (index && row[3+row[1]+index-1]>=binding) ||
                   task.binding(uint32_t(binding))[1]!=task.query(query)[4])return false;
            }
            row_offset+=3+row[1]+row[2];
        }
        if(row_offset!=requirement_words)return false;
        for(uint32_t rank=0;rank<task.queries;++rank) {
            const uint64_t query=task.attainment_order[rank];
            if(query>=task.queries)return false;
            for(uint32_t before=0;before<rank;++before)
                if(task.attainment_order[before]==query)return false;
            const auto* row=task.requirements+task.requirement_offsets[query];
            for(uint64_t index=0;index<row[1];++index) {
                bool earlier=false;
                for(uint32_t before=0;before<rank;++before)
                    earlier|=task.attainment_order[before]==row[3+index];
                if(!earlier)return false;
            }
        }
    } else if(requirement_words)return false;
    offset+=requirement_words;
    if(words[12]!=count-offset)return false;
    task.editable=words[12]!=0;
    if(task.editable && (!semantic_program::decode_bank(words+offset,words[12],&task.program) ||
       task.program.query_count!=task.queries || task.program.rule_count>semantic_program::kRuleCapacity))return false;
    if(task.coding && !task.editable)return false;
    for(uint32_t query=0;query<task.queries;++query) {
        const uint64_t* row=task.query(query);
        if(!(row[0]|row[1]|row[2]|row[3]) || row[4]>UINT32_MAX || row[5]>3 || !row[6] || row[6]>15)return false;
    }
    uint64_t goals=0;
    for(uint64_t level=0;level<task.level_count;++level) {
        if(!task.levels[level] || task.levels[level]>task.goal_count-goals)return false;
        goals+=task.levels[level];
    }
    if(goals!=task.goal_count)return false;
    for(uint64_t goal=0;goal<goals;++goal) {
        const uint64_t* row=task.goals+goal*3;
        if(row[0]>=task.queries || row[1]>3 || !row[2] || row[2]>UINT32_MAX ||
           !(task.query(uint32_t(row[0]))[6]&(uint64_t(1)<<row[1])))return false;
        for(uint64_t before=0;before<goal;++before)if(task.goals[before*3]==row[0])return false;
    }
    for(uint32_t binding=0;binding<task.bindings;++binding) {
        const uint64_t* row=task.binding(binding);
        semantic_program::Fact fact{};
        if(row[0]>=task.queries || row[1]>UINT32_MAX || row[18]<1 || row[18]>5 ||
           row[19]>1 || (row[18]==4)!=(row[19]==1) || !task.observation_fact(binding,&fact))return false;
    }
    *output=task;return true;
}

struct Query { uint64_t receipt[42],truth,attained; };
// The first word is an origin tag: zero is absent, one is Action, two is Intent.
// The historical action word group holds the origin identity for either tag.
struct Observation { uint64_t present,truth,receipt[4],action[4],intent[4],support[4]; };
struct InheritedObservation {
    // Immutable donor evidence, separate from the active task's verifier
    // policy. Only the canonical closure reads this plane after restoration.
    uint64_t binding, statement_record;
    semantic_program::Fact fact;
    Observation observation;
};
static_assert(sizeof(Query)==352,"task query record ABI");
static_assert(sizeof(Observation)==144,"task observation record ABI");
static_assert(sizeof(InheritedObservation)==176,"inherited observation record ABI");

struct Plane {
    uint64_t* header;
    Observation* observations;
    Query* query_rows;
    semantic_program::Rule* selected_rules;
    const InheritedObservation* inherited;
    uint32_t inherited_capacity;
    __device__ Query& query(uint32_t candidate,uint32_t query) const {
        return query_rows[uint64_t(candidate)*header[2]+query];
    }
};

__device__ bool decode_plane(uint64_t address,uint64_t bytes,const Task& task,Plane* output) {
    if(!address || address%alignof(uint64_t) || bytes>UINT64_MAX-address || bytes<64)return false;
    auto* header=reinterpret_cast<uint64_t*>(address);
    const uint64_t capacity=task.editable ? semantic_program::kRuleCapacity-uint64_t(task.program.rule_count) : 0;
    const uint64_t query_offset=64+uint64_t(task.bindings)*sizeof(Observation);
    const uint64_t rule_offset=query_offset+3*uint64_t(task.queries)*sizeof(Query);
    if(header[0]!=kMagic || header[1]!=4 || header[2]!=task.queries || header[3]!=task.bindings ||
       header[4]!=kQueryWords || header[5]!=kObservationWords || header[6]>capacity || header[7]!=capacity ||
       bytes<rule_offset+capacity*sizeof(semantic_program::Rule)+16)return false;
    const uint64_t inherited_offset=((rule_offset+capacity*sizeof(semantic_program::Rule)+7)&~uint64_t(7))+16;
    if(bytes<inherited_offset)return false;
    const auto* inherited_header=reinterpret_cast<const uint64_t*>(address+inherited_offset-16);
    if(inherited_header[0]!=kInheritedMagic || inherited_header[1]>UINT32_MAX-task.bindings ||
       inherited_header[1]>(bytes-inherited_offset)/sizeof(InheritedObservation) ||
       bytes!=inherited_offset+inherited_header[1]*sizeof(InheritedObservation))return false;
    Plane plane{header,reinterpret_cast<Observation*>(address+64),
        reinterpret_cast<Query*>(address+query_offset),reinterpret_cast<semantic_program::Rule*>(address+rule_offset),
        reinterpret_cast<const InheritedObservation*>(address+inherited_offset),uint32_t(inherited_header[1])};
    for(uint32_t index=0;index<task.bindings+plane.inherited_capacity;++index) {
        const bool inherited=index>=task.bindings;
        if(inherited) {
            const auto& row=plane.inherited[index-task.bindings];
            if(row.binding!=index-task.bindings || row.statement_record>UINT32_MAX ||
               !semantic_program::valid_fact(row.fact,task.program))return false;
        }
        const auto& observation=inherited ? plane.inherited[index-task.bindings].observation : plane.observations[index];
        const auto* words=reinterpret_cast<const uint64_t*>(&observation);
        if(observation.present>2 || observation.truth>3)return false;
        if(observation.present==2) {
            // The plane unions real verifier truths; each signed support is
            // authenticated separately with one truth bit before publication.
            for(uint32_t word=0;word<4;++word)
                if(observation.action[word]!=observation.intent[word])return false;
        }
        if(!observation.present) {
            for(uint32_t field=0;field<18;++field)if(words[field])return false;
        } else for(uint32_t identity=0;identity<4;++identity)
            if(!(words[2+identity*4]|words[3+identity*4]|words[4+identity*4]|words[5+identity*4]))return false;
    }
    for(uint64_t index=0;index<header[6];++index)
        if(!semantic_program::valid_rule(plane.selected_rules[index],task.program))return false;
    *output=plane;return true;
}

struct Observations {
    Task task;
    Plane plane;
    __device__ uint32_t size() const { return task.bindings+plane.inherited_capacity; }
    __device__ semantic_program::Observation operator[](uint32_t index) const {
        if(index>=task.bindings) {
            const auto& row=plane.inherited[index-task.bindings];
            return {row.fact,row.observation.truth,uint64_t(row.observation.present!=0)};
        }
        semantic_program::Fact fact{};
        task.observation_fact(index,&fact);
        const auto& observation=plane.observations[index];
        // The generic program evaluator consumes presence, not an origin tag.
        return {fact,observation.truth,uint64_t(observation.present!=0)};
    }
};

// Raw derivability is unchanged. Attainment consumes the original protected
// dependency graph after every query truth has been materialized.
__device__ bool attained(const Task& task,const Plane& plane,uint32_t slot,uint32_t query,
        uint64_t truth,uint32_t fact_index,uint32_t fact_count,
        const semantic_program::DualSupport& support,uint64_t* inspected) {
    ++*inspected;
    uint64_t target=0;
    if(!task.target(query,&target) || truth!=target)return false;
    const auto* row=task.requirements+task.requirement_offsets[query];
    for(uint64_t index=0;index<row[1];++index) {
        ++*inspected;
        if(plane.query(slot,uint32_t(row[3+index])).attained!=1)return false;
    }
    const auto query_fact=semantic_program::bank_fact(task.program,query);
    for(uint64_t index=0;index<row[2];++index) {
        ++*inspected;
        const uint32_t binding=uint32_t(row[3+row[1]+index]);
        const auto& observation=plane.observations[binding];
        if((observation.present!=1 && observation.present!=2) || observation.truth!=1 || fact_index==fact_count)return false;
        semantic_program::Fact observed{};task.observation_fact(binding,&observed);
        if(observed.predicate!=query_fact.predicate || observed.arity!=query_fact.arity ||
           observed.first!=query_fact.first ||
           (query_fact.arity==2 && observed.second!=query_fact.second))return false;
        const uint64_t sources=support.lineage[(uint64_t(fact_index)*2)*support.words+binding/64];
        if(!(sources&(uint64_t(1)<<(binding%64))))return false;
    }
    return true;
}

} // namespace semantic_task_ground
