#pragma once
#include <stdint.h>
#include "semantic_program.cuh"

namespace semantic_program_image {
constexpr uint64_t kGeometryMagic=0x584c4f47494d4747ULL;
constexpr uint64_t kFrameMagic=0x584c4f47494d4746ULL;
constexpr uint64_t kGeometryWords=32;
constexpr uint64_t kFrameHeaderWords=48;
constexpr uint64_t kLineageWords=56;

struct Geometry {
    uint32_t roots,nodes,guards,leaves,blocks,mutations;
    uint32_t local_roots,local_nodes,local_guards,local_leaves,local_blocks,local_mutations;
    uint32_t facts,library,statements,entry_supports;
    uint64_t root_offset,node_offset,rule_offset,block_offset,activity_offset,lineage_offset,frame_bytes;
};
__device__ bool decode_geometry(const uint64_t* words,const semantic_program::Bank* bank,
        uint32_t library,Geometry* output) {
    if(!words || !output || words[0]!=kGeometryMagic || words[1]!=1)return false;
    for(uint32_t i=2;i<16;++i)if(words[i]>UINT32_MAX)return false;
    for(uint32_t i=19;i<31;++i)if(words[i])return false;
    Geometry g{};
    g.roots=uint32_t(words[2]);g.nodes=uint32_t(words[3]);g.guards=uint32_t(words[4]);
    g.leaves=uint32_t(words[5]);g.blocks=uint32_t(words[6]);g.mutations=uint32_t(words[7]);
    g.local_roots=uint32_t(words[8]);g.local_nodes=uint32_t(words[9]);g.local_guards=uint32_t(words[10]);
    g.local_leaves=uint32_t(words[11]);g.local_blocks=uint32_t(words[12]);g.local_mutations=uint32_t(words[13]);
    g.facts=uint32_t(words[14]);g.library=uint32_t(words[15]);
    if(words[17]>UINT32_MAX || words[18]>UINT32_MAX)return false;
    g.statements=uint32_t(words[17]);g.entry_supports=uint32_t(words[18]);
    if(g.library!=library)return false;
    if(bank) {
        const auto& image=bank->image;
        if(g.roots!=image.root_count || g.nodes!=image.node_count || g.guards!=image.guard_count ||
           g.leaves!=image.leaf_count || g.blocks!=image.block_count || g.mutations!=image.mutation_capacity ||
           g.local_roots!=image.local_root_begin || g.local_nodes!=image.local_node_begin ||
           g.local_guards!=image.local_guard_begin || g.local_leaves!=image.local_leaf_begin ||
           g.local_blocks!=image.local_block_begin || g.local_mutations!=image.local_mutation_capacity ||
           g.facts!=bank->fact_capacity || g.statements!=bank->statement_count)return false;
    } else if(g.roots || g.nodes || g.guards || g.leaves || g.blocks || g.mutations ||
        g.local_roots || g.local_nodes || g.local_guards || g.local_leaves || g.local_blocks ||
        g.local_mutations || g.facts || g.statements || g.entry_supports || library)return false;
    g.root_offset=kFrameHeaderWords*8;
    g.node_offset=g.root_offset+uint64_t(g.roots)*8;
    g.rule_offset=g.node_offset+uint64_t(g.nodes)*64;
    g.block_offset=g.rule_offset+uint64_t(g.mutations)*sizeof(semantic_program::Rule);
    g.activity_offset=g.block_offset+uint64_t(g.mutations)*4;
    g.lineage_offset=g.activity_offset+uint64_t(g.blocks)*8;
    g.frame_bytes=g.lineage_offset+uint64_t(g.mutations)*kLineageWords*8;
    if(words[16]!=g.frame_bytes)return false;
    *output=g;return true;
}
struct Frame {
    uint64_t *header,*roots,*nodes,*activity,*lineage;
    semantic_program::Rule* rules;
    uint32_t* rule_blocks;
    Geometry geometry;
    __device__ uint64_t* node(uint32_t index) const { return nodes+uint64_t(index)*8; }
    __device__ uint64_t* history(uint32_t index) const { return lineage+uint64_t(index)*kLineageWords; }
};
__device__ bool decode_frame(uint64_t address,const Geometry& g,
        const semantic_program::Bank* bank,Frame* output) {
    if(!address || address%8 || !output || g.frame_bytes>UINT64_MAX-address)return false;
    auto* bytes=reinterpret_cast<uint8_t*>(address);
    Frame f{};
    f.header=reinterpret_cast<uint64_t*>(bytes);f.geometry=g;
    f.roots=reinterpret_cast<uint64_t*>(bytes+g.root_offset);
    f.nodes=reinterpret_cast<uint64_t*>(bytes+g.node_offset);
    f.rules=reinterpret_cast<semantic_program::Rule*>(bytes+g.rule_offset);
    f.rule_blocks=reinterpret_cast<uint32_t*>(bytes+g.block_offset);
    f.activity=reinterpret_cast<uint64_t*>(bytes+g.activity_offset);
    f.lineage=reinterpret_cast<uint64_t*>(bytes+g.lineage_offset);
    if(f.header[0]!=kFrameMagic || f.header[1]!=1 || f.header[3]>g.mutations ||
       f.header[4]>g.mutations || f.header[5]>f.header[3] ||
       f.header[2]<f.header[3]-f.header[5])return false;
    for(uint32_t i=34;i<48;++i)if(f.header[i])return false;
    if(bank) {
        if(!(f.header[6]|f.header[7]|f.header[8]|f.header[9]) ||
           !(f.header[10]|f.header[11]|f.header[12]|f.header[13]))return false;
        bool active=false;
        for(uint32_t root=0;root<g.roots;++root) {
            const uint64_t node=f.roots[root];
            if(node==UINT32_MAX)continue;
            if(node>=g.nodes || !f.node(uint32_t(node))[0])return false;
            active=true;
        }
        if(!active)return false;
        for(uint32_t i=0;i<g.nodes;++i) {
            const auto* node=f.node(i);
            for(uint32_t field=0;field<8;++field)if(node[field]>UINT32_MAX)return false;
            if(node[0]==0) { for(uint32_t field=1;field<8;++field)if(node[field])return false; }
            else if(node[0]==1) {
                if(node[1]>=g.guards || node[6] || node[7])return false;
                for(uint32_t edge=0;edge<4;++edge)
                    if(node[2+edge]>=g.nodes || !f.node(uint32_t(node[2+edge]))[0])return false;
            } else if(node[0]==2) {
                if(node[6]>=g.leaves || node[1] || node[2] || node[3] || node[4] || node[5] || node[7])return false;
            } else return false;
        }
        for(uint32_t i=0;i<f.header[4];++i)
            if(f.rule_blocks[i]>=g.blocks || !semantic_program::valid_rule(f.rules[i],*bank))return false;
        const uint64_t permitted=g.library==64 ? UINT64_MAX : (uint64_t(1)<<g.library)-1;
        for(uint32_t block=0;block<g.blocks;++block)
            if(f.activity[block]&~permitted)return false;
    }
    *output=f;return true;
}

struct DecodedEdit {
    uint32_t opcode,target,guard,children[4],effect,block,library_index;
    semantic_program::Rule rule;
};
// A backward prefix is a read-only view of the acquired entry and the one
// preceding, already admitted edit. It neither forks truth nor republishes it.
struct Prefix {
    Frame base;
    DecodedEdit prior;
    bool has_prior;
    uint64_t replacement[8];
    Geometry geometry;
    __device__ uint64_t generation() const { return base.header[2]+uint64_t(has_prior); }
    __device__ uint64_t mutations() const { return base.header[3]+uint64_t(has_prior); }
    __device__ uint64_t inherited_mutations() const { return base.header[5]; }
    __device__ uint64_t selected_rules() const { return base.header[4]+uint64_t(has_prior && prior.opcode==2 && prior.library_index==UINT32_MAX); }
    __device__ const uint64_t* node(uint32_t index) const {
        return has_prior && (prior.opcode==3 || prior.opcode==4) && prior.target==index
            ? replacement : base.node(index);
    }
    __device__ uint64_t root(uint32_t index) const {
        return has_prior && prior.opcode==5 && prior.target==index ? prior.children[0] : base.roots[index];
    }
    __device__ void reset(Frame frame) {
        base=frame;geometry=frame.geometry;has_prior=false;prior={};
        for(uint32_t i=0;i<8;++i)replacement[i]=0;
    }
    __device__ void after(const DecodedEdit& edit) {
        prior=edit;has_prior=edit.opcode!=0;
        for(uint32_t i=0;i<8;++i)replacement[i]=0;
        if(edit.opcode==3) {
            replacement[0]=1;replacement[1]=edit.guard;
            for(uint32_t edge=0;edge<4;++edge)replacement[2+edge]=edit.children[edge];
        } else if(edit.opcode==4) { replacement[0]=2;replacement[6]=edit.effect; }
    }
};

// One native content identity for roots, edges, rules, block membership and
// activity. Lineage binds successive identities; it does not redefine them.
struct GenomeBytes {
    Frame frame;
    __device__ uint8_t operator[](uint64_t index) const {
        constexpr char domain[]="xlog.program-genome.v1";
        if(index<sizeof(domain)-1)return uint8_t(domain[index]);
        index-=sizeof(domain)-1;
        if(index<32)return reinterpret_cast<const uint8_t*>(frame.header+6)[index];
        if(index<40)return reinterpret_cast<const uint8_t*>(frame.header+4)[index-32];
        return reinterpret_cast<const uint8_t*>(frame.header)[frame.geometry.root_offset+index-40];
    }
    __device__ uint64_t size() const {
        return sizeof("xlog.program-genome.v1")-1+40+frame.geometry.lineage_offset-frame.geometry.root_offset;
    }
};
// Every initialized node must remain acyclic, including disconnected reserved
// subgraphs that a later root edit can make reachable. No device recursion.
template<class Work>
__device__ semantic_program::Status acyclic(const Frame& frame,uint8_t* colors,
        uint32_t* stack,uint8_t* cursors,uint64_t* charged,uint64_t fuel,Work work) {
    const uint32_t count=frame.geometry.nodes;
    if(count && (!colors || !stack || !cursors))return semantic_program::Status::InvalidProgram;
    auto tick=[&]() {
        if(*charged==fuel)return false;
        ++*charged;work();return true;
    };
    for(uint32_t i=0;i<count;++i) { if(!tick())return semantic_program::Status::FuelExhausted;colors[i]=0; }
    for(uint32_t start=0;start<count;++start) {
        if(!tick())return semantic_program::Status::FuelExhausted;
        if(colors[start] || !frame.node(start)[0])continue;
        uint32_t depth=1;stack[0]=start;cursors[0]=0;colors[start]=1;
        while(depth) {
            if(!tick())return semantic_program::Status::FuelExhausted;
            const uint32_t node=stack[depth-1];
            const auto* words=frame.node(node);
            if(words[0]==2 || cursors[depth-1]==4) { colors[node]=2;--depth;continue; }
            if(words[0]!=1 || words[1]>=frame.geometry.guards)return semantic_program::Status::InvalidProgram;
            const uint64_t child=words[2+cursors[depth-1]++];
            if(child>=count || !frame.node(uint32_t(child))[0] || colors[child]==1)
                return semantic_program::Status::CyclicImage;
            if(colors[child]==2)continue;
            if(depth==count)return semantic_program::Status::InvalidProgram;
            colors[child]=1;stack[depth]=uint32_t(child);cursors[depth++]=0;
        }
    }
    return semantic_program::Status::Ok;
}
// Execute every active root against one immutable entry. A block materializer
// receives that same entry, never a preceding tree's staged result.
__device__ bool event_order(const Frame& frame,semantic_world::Measure& world,
        uint32_t* ordered) {
    if(frame.geometry.nodes!=world.image_nodes)return false;
    uint32_t active=0,count=0;
    for(uint32_t node=0;node<frame.geometry.nodes;++node) {
        if(!world.charge(2))return false;
        world.image_degree[node]=0;
        active+=frame.node(node)[0]!=0;
    }
    for(uint32_t node=0;node<frame.geometry.nodes;++node) {
        if(!world.charge())return false;
        const auto* row=frame.node(node);
        if(row[0]!=1)continue;
        for(uint32_t edge=0;edge<4;++edge) {
            if(!world.charge(3))return false;
            if(row[2+edge]>=frame.geometry.nodes)return false;
            ++world.image_degree[row[2+edge]];
        }
    }
    for(uint32_t node=0;node<frame.geometry.nodes;++node) {
        if(!world.charge(2))return false;
        if(frame.node(node)[0] && !world.image_degree[node])world.image_order[count++]=node;
    }
    for(uint32_t next=0;next<count;++next) {
        if(!world.charge(2))return false;
        const auto* row=frame.node(uint32_t(world.image_order[next]));
        if(row[0]!=1)continue;
        for(uint32_t edge=0;edge<4;++edge) {
            if(!world.charge(3))return false;
            const uint32_t child=uint32_t(row[2+edge]);
            if(!world.image_degree[child])return false;
            if(!--world.image_degree[child]) {
                if(count==frame.geometry.nodes)return false;
                world.image_order[count++]=child;
            }
        }
    }
    *ordered=count;
    return count==active;
}

template<class Observations>
__device__ semantic_program::Result execute(const semantic_program::Bank& bank,const Frame& frame,
        semantic_program::Fact* input,semantic_program::Rule* rules,uint32_t rule_capacity,
        semantic_program::Fact* entry,semantic_program::DualSupport entry_support,
        semantic_program::Fact* block_output,semantic_program::DualSupport block_support,
        semantic_program::Fact* output,semantic_program::DualSupport output_support,
        uint32_t capacity,uint64_t fuel,const Observations& observations,
        const uint64_t* library,uint32_t library_count) {
    using semantic_program::Status;using semantic_program::Result;
    if(!bank.words || !frame.header || capacity!=bank.fact_capacity || !input || !entry ||
       !block_output || !output || !output_support.truth || !entry_support.truth ||
       !block_support.truth || entry_support.words!=output_support.words ||
       block_support.words!=output_support.words || bank.initial_count>capacity ||
        library_count>64 || (library_count && !library) || !output_support.world ||
        entry_support.world!=output_support.world || block_support.world!=output_support.world)
        return {Status::InvalidProgram,0,0,0};
    uint64_t charged=0;
    auto& world=*output_support.world;
    semantic_world::MeasureFuel measure_fuel(&world,&charged,fuel);
    auto tick=[&](uint64_t count=1) {
        if(count>fuel-charged)return false;charged+=count;return true;
    };
    for(uint32_t index=0;index<bank.initial_count;++index) {
        if(!tick())return {Status::FuelExhausted,0,0,charged};
        input[index]=semantic_program::bank_fact(bank,uint64_t(bank.query_count)+index);
    }
    auto base=semantic_program::materialize(input,bank.initial_count,nullptr,0,bank,entry,capacity,
        fuel-charged,&entry_support,observations,0);
    charged+=base.charged_pairs;
    if(base.status!=Status::Ok)return {base.status,0,0,charged};
    for(uint32_t index=0;index<capacity;++index) {
        if(!tick())return {Status::FuelExhausted,0,0,charged};
        output_support.truth[index]=index<base.fact_count ? entry_support.truth[index] : 0;
        if(index<base.fact_count)output[index]=entry[index];
        for(uint32_t channel=0;channel<2;++channel) {
            const uint64_t cell=uint64_t(index)*2+channel;
            if(!world.charge())return {Status::FuelExhausted,0,0,charged};
            output_support.events[cell]=index<base.fact_count ? entry_support.events[cell] : 0;
            for(uint32_t origin=0;origin<world.lineage_events;++origin) {
                if(!world.charge())return {Status::FuelExhausted,0,0,charged};
                output_support.lineage_events[cell*world.lineage_events+origin]=index<base.fact_count ?
                    entry_support.lineage_events[cell*world.lineage_events+origin] : 0;
            }
            if(output_support.leaf_refs)output_support.leaf_refs[uint64_t(index)*2+channel]=0;
            for(uint32_t word=0;word<output_support.words;++word) {
                if(!tick())return {Status::FuelExhausted,0,0,charged};
                output_support.lineage[(uint64_t(index)*2+channel)*output_support.words+word]=
                    index<base.fact_count ? entry_support.lineage[(uint64_t(index)*2+channel)*entry_support.words+word] : 0;
            }
        }
    }
    uint32_t count=base.fact_count,derived=0;
    uint32_t current=0,path_event=1;
    auto merge=[&](semantic_program::Fact fact,uint8_t truth,uint32_t source,
            const semantic_program::DualSupport* provenance,uint64_t leaf_ref) {
        const uint32_t index=semantic_program::find_fact(output,count,fact,&charged,fuel);
        if(index==UINT32_MAX)return Status::FuelExhausted;
        if(index==count) {
            if(count==capacity)return Status::FactCapacity;
            output[count++]=fact;
        }
        for(uint32_t channel=0;channel<2;++channel) {
            if(!tick())return Status::FuelExhausted;
            const uint8_t bit=uint8_t(1U<<channel);
            const uint64_t source_cell=uint64_t(source)*2+channel;
            const uint32_t event=world.intersection(path_event,provenance ?
                uint32_t(world.read(provenance->events,source_cell)) : truth&bit ? world.view_condition : 0);
            if(!event)continue;
            const bool absent=!(output_support.truth[index]&bit);
            semantic_program::merge_world_support(output_support,index,channel,event,
                [&](uint32_t origin) {
                    const uint32_t inherited=provenance ? uint32_t(world.read(provenance->lineage_events,
                        source_cell*world.lineage_events+origin)) : 0;
                    return world.unite(inherited,uint32_t(world.read(world.image_lineage_events,
                        uint64_t(current)*world.lineage_events+origin)));
                },leaf_ref);
            derived+=uint32_t(absent && (output_support.truth[index]&bit));
            if(world.status!=semantic_world::MeasureStatus::Ok)return Status::InvalidProgram;
        }
        return Status::Ok;
    };
    uint32_t ordered=0;
    if(!event_order(frame,world,&ordered))return {world.status==semantic_world::MeasureStatus::FuelExhausted ?
        Status::FuelExhausted : Status::CyclicImage,0,0,charged};
    bool active=false;
    for(uint32_t root=0;root<frame.geometry.roots;++root) {
        if(!tick())return {Status::FuelExhausted,0,0,charged};
        const uint64_t initial=frame.roots[root];
        if(initial==UINT32_MAX)continue;
        if(initial>=frame.geometry.nodes)return {Status::InvalidProgram,0,0,charged};
        active=true;
        for(uint32_t node=0;node<frame.geometry.nodes;++node) {
            if(!tick())return {Status::FuelExhausted,0,0,charged};
            world.image_paths[node]=node==initial ? world.view_condition : 0;
            for(uint32_t origin=0;origin<world.lineage_events;++origin) {
                if(!tick())return {Status::FuelExhausted,0,0,charged};
                world.image_lineage_events[uint64_t(node)*world.lineage_events+origin]=0;
            }
        }
        for(uint32_t position=0;position<ordered;++position) {
            if(!tick(2))return {Status::FuelExhausted,0,0,charged};
            current=uint32_t(world.image_order[position]);path_event=uint32_t(world.image_paths[current]);
            if(!path_event)continue;
            const auto* node=frame.node(current);
            if(node[0]!=2) {
                if(node[0]!=1 || node[1]>=bank.image.guard_count)return {Status::InvalidProgram,0,0,charged};
                const auto fact=bank.image.guard(uint32_t(node[1]));
                const uint32_t found=semantic_program::find_fact(entry,base.fact_count,fact,&charged,fuel);
                if(found==UINT32_MAX)return {Status::FuelExhausted,0,0,charged};
                const uint32_t pro=found==base.fact_count ? 0 : uint32_t(world.read(entry_support.events,uint64_t(found)*2));
                const uint32_t contra=found==base.fact_count ? 0 : uint32_t(world.read(entry_support.events,uint64_t(found)*2+1));
                const uint32_t not_pro=world.complement(pro),not_contra=world.complement(contra);
                const uint32_t outcomes[4]={world.intersection(pro,not_contra),world.intersection(not_pro,contra),
                    world.intersection(pro,contra),world.intersection(not_pro,not_contra)};
                for(uint32_t edge=0;edge<4;++edge) {
                    const uint32_t child=uint32_t(node[2+edge]);
                    const uint32_t event=world.intersection(path_event,outcomes[edge]);
                    if(!event)continue;
                    const uint32_t prior=uint32_t(world.read(world.image_paths,child));
                    const uint32_t added=world.intersection(event,world.complement(prior));
                    const uint32_t combined=world.unite(prior,event);
                    if(!tick())return {Status::FuelExhausted,0,0,charged};world.image_paths[child]=combined;
                    for(uint32_t origin=0;origin<world.lineage_events;++origin) {
                        uint32_t inherited=uint32_t(world.read(world.image_lineage_events,
                            uint64_t(current)*world.lineage_events+origin));
                        if(found<base.fact_count)for(uint32_t channel=0;channel<2;++channel)
                            inherited=world.unite(inherited,uint32_t(world.read(entry_support.lineage_events,
                                (uint64_t(found)*2+channel)*world.lineage_events+origin)));
                        const uint64_t offset=uint64_t(child)*world.lineage_events+origin;
                        const uint32_t proof=world.unite(uint32_t(world.read(world.image_lineage_events,offset)),
                            world.intersection(added,inherited));
                        if(!tick())return {Status::FuelExhausted,0,0,charged};world.image_lineage_events[offset]=proof;
                    }
                }
                if(world.status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
                continue;
            }
            const uint64_t effect=frame.node(current)[6];
            if(effect>=bank.image.leaf_count)return {Status::InvalidProgram,0,0,charged};
            const auto* leaf=bank.image.leaves+effect*4;
            const uint64_t leaf_ref=(uint64_t(root)<<32)|uint64_t(current);
            if(leaf_ref==UINT64_MAX)return {Status::InvalidProgram,0,0,charged};
            if(leaf[0]==0)continue;
            if(leaf[0]==1) {
                for(uint64_t event=0;event<leaf[2];++event) {
                    if(!tick())return {Status::FuelExhausted,0,0,charged};
                    const auto* row=bank.image.events+(leaf[1]+event)*2;
                    if(row[0]>=bank.image.guard_count || row[1]<1 || row[1]>2)return {Status::InvalidProgram,0,0,charged};
                    const auto fact=bank.image.guard(uint32_t(row[0]));
                    for(uint32_t binding=0;binding<observations.protected_size();++binding) {
                        if(!tick())return {Status::FuelExhausted,0,0,charged};
                        const auto protected_fact=observations[binding].fact;
                        if(fact.predicate==protected_fact.predicate && fact.arity==protected_fact.arity &&
                           fact.first==protected_fact.first && fact.second==protected_fact.second)
                            return {Status::ProtectedObservation,0,0,charged};
                    }
                    const Status status=merge(fact,uint8_t(row[1]),0,nullptr,leaf_ref+1);
                    if(status!=Status::Ok)return {status,0,0,charged};
                }
                continue;
            }
            if(leaf[0]!=2 || leaf[1]>=bank.image.block_count)return {Status::InvalidProgram,0,0,charged};
            const uint32_t block=uint32_t(leaf[1]);
            const auto* block_row=bank.image.blocks+uint64_t(block)*2;
            uint32_t rule_count=0;
            for(uint64_t index=0;index<block_row[1];++index) {
                if(!tick(semantic_program::kRuleWords))return {Status::FuelExhausted,0,0,charged};
                if(rule_count==rule_capacity)return {Status::RuleCapacity,0,0,charged};
                const uint32_t ordinal=uint32_t(bank.image.references[block_row[0]+index]);
                world.rule_ordinals[rule_count]=ordinal;
                rules[rule_count++]=semantic_program::bank_rule(bank,ordinal);
            }
            const uint32_t protected_begin=rule_count;
            for(uint32_t index=0;index<frame.header[4];++index) {
                if(!tick())return {Status::FuelExhausted,0,0,charged};
                if(frame.rule_blocks[index]!=block)continue;
                if(rule_count==rule_capacity)return {Status::RuleCapacity,0,0,charged};
                world.rule_ordinals[rule_count]=uint64_t(bank.rule_count)+index;
                rules[rule_count++]=frame.rules[index];
            }
            const uint64_t mask=frame.activity[block];
            const uint64_t permitted=library_count==64 ? UINT64_MAX : (uint64_t(1)<<library_count)-1;
            if(mask&~permitted)return {Status::InvalidProgram,0,0,charged};
            for(uint32_t index=0;index<library_count;++index) {
                if(!tick())return {Status::FuelExhausted,0,0,charged};
                if(!(mask&(uint64_t(1)<<index)))continue;
                if(rule_count==rule_capacity)return {Status::RuleCapacity,0,0,charged};
                world.rule_ordinals[rule_count]=UINT32_MAX;
                rules[rule_count++]=semantic_program::rule_words(library+uint64_t(index)*semantic_program::kRuleWords);
            }
            auto result=semantic_program::materialize(input,bank.initial_count,rules,rule_count,bank,
                block_output,capacity,fuel-charged,&block_support,observations,protected_begin,path_event,uint32_t(frame.header[4]));
            charged+=result.charged_pairs;
            if(result.status!=Status::Ok)return {result.status,0,0,charged};
            for(uint32_t index=0;index<result.fact_count;++index) {
                const Status status=merge(block_output[index],block_support.truth[index],index,&block_support,leaf_ref+1);
                if(status!=Status::Ok)return {status,0,0,charged};
            }
        }
    }
    if(world.status!=semantic_world::MeasureStatus::Ok)return {Status::InvalidProgram,0,0,charged};
    return active ? Result{Status::Ok,count,derived,charged} : Result{Status::InvalidProgram,0,0,charged};
}
} // namespace semantic_program_image
