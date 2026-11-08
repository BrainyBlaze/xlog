#pragma once
#include "semantic_program_image.cuh"
#include "semantic_rule_action.cuh"

namespace semantic_program_image_action {
// Descriptor tags share the original four cold codebook domains. Generation
// and owner are acquired from the selected native image, never sampled as data.
template<class Charge>
__device__ bool target(const uint64_t* books,uint32_t category,uint64_t tag,
        uint32_t begin,uint32_t count,uint32_t* index,Charge charge) {
    if(!category || category>=books[1] ||
       !semantic_rule_action::context_category(books,2,category,charge))return false;
    const auto* row=books+books[0]+uint64_t(category)*10;
    charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
    if(row[1]!=tag || begin>count || row[0]>=count-begin)return false;
    for(uint32_t i=2;i<10;++i) {
        charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
        if(row[i])return false;
    }
    *index=begin+uint32_t(row[0]);return true;
}
template<class Charge>
__device__ bool operand(const uint64_t* books,uint32_t field,uint32_t category,uint64_t tag,
        uint32_t begin,uint32_t count,uint32_t* index,Charge charge) {
    if(!category || category>=books[3] ||
       !semantic_rule_action::context_category(books,field,category,charge))return false;
    const auto* row=books+books[2]+uint64_t(category)*6;
    charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
    if(row[0] || row[1] || row[2] || row[3]!=tag || begin>count || row[4]>=count-begin || row[5])return false;
    *index=begin+uint32_t(row[4]);return true;
}
template<class Charge>
__device__ bool leaf(const uint64_t* books,uint32_t category,const semantic_program::Bank& bank,
        uint32_t* effect,Charge charge,bool block_only=false) {
    if(!category || category>=books[7] ||
       !semantic_rule_action::context_category(books,17,category,charge))return false;
    const auto* row=books+books[6]+uint64_t(category)*18;
    charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
    if(row[0]!=UINT64_MAX || row[1]>=bank.image.leaf_count-bank.image.local_leaf_begin)return false;
    for(uint32_t i=2;i<18;++i) {
        charge(semantic_rule_action::PrefixWorkEvent::TableSlot);if(row[i])return false;
    }
    if(block_only) {
        const auto* value=bank.image.leaves+(bank.image.local_leaf_begin+row[1])*4;
        charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
        if(value[0]!=2 || value[1]<bank.image.local_block_begin || value[1]>=bank.image.block_count)return false;
    }
    *effect=bank.image.local_leaf_begin+uint32_t(row[1]);return true;
}
template<class Component,class Charge>
__device__ bool prefix_completes(const uint64_t* books,const Component* components,
        const uint8_t* support,uint32_t start,const uint32_t choices[18],uint32_t field,
        uint32_t category,uint32_t opcode,uint32_t active,
        const semantic_program::Bank& bank,const semantic_program_image::Prefix& frame,Charge charge) {
    if(opcode<3 || opcode>5 || !frame.base.header ||
       frame.mutations()>=frame.geometry.mutations ||
       frame.mutations()-frame.inherited_mutations()>=frame.geometry.local_mutations ||
       frame.generation()==UINT64_MAX)return false;
    auto admitted=[&](uint32_t position,uint32_t value) {
        charge(semantic_rule_action::PrefixWorkEvent::Category);
        const auto& component=components[start+position];
        return value<component.cardinality &&
            semantic_rule_action::context_category(books,position,value,charge) &&
            support[component.offset+value] &&
            (position>field || value==(position==field ? category : choices[position]));
    };
    for(uint32_t position=3;position<18;++position)
        if(!(active&(uint32_t(1)<<position)) && !admitted(position,0))return false;
    // Future fields are independent typed roles. Summarize their actual
    // supported values once, then compare each target with that same summary.
    uint32_t first[6]={};bool varied[6]={};
    for(uint32_t position=12;position<18;++position) {
        if(!(active&(uint32_t(1)<<position)))continue;
        bool found=false;
        const uint32_t count=position==17 ? uint32_t(books[7]) : uint32_t(books[3]);
        const uint32_t begin=position<=field ? (position==field ? category : choices[position]) : 1;
        const uint32_t end=position<=field ? begin+uint32_t(begin<count) : count;
        for(uint32_t value=begin;value<end;++value) {
            if(!admitted(position,value))continue;
            uint32_t decoded=0;bool valid=false;
            if(position==17)valid=leaf(books,value,bank,&decoded,charge);
            else if(position==12)valid=operand(books,position,value,3,frame.geometry.local_guards,
                frame.geometry.guards,&decoded,charge);
            else {
                valid=operand(books,position,value,4,frame.geometry.local_nodes,
                    frame.geometry.nodes,&decoded,charge);
                if(valid) {
                    charge(semantic_rule_action::PrefixWorkEvent::TableSlot);
                    valid=frame.node(decoded)[0]!=0;
                }
            }
            if(!valid)continue;
            if(!found)first[position-12]=decoded;
            else if(first[position-12]!=decoded)varied[position-12]=true;
            found=true;
            if(varied[position-12])break;
        }
        if(!found)return false;
    }
    const uint32_t begin=field>=2 ? (field==2 ? category : choices[2]) : 1;
    const uint32_t end=field>=2 ? begin+uint32_t(begin<books[1]) : uint32_t(books[1]);
    for(uint32_t candidate=begin;candidate<end;++candidate) {
        uint32_t index=0;
        if(!admitted(2,candidate) || !target(books,candidate,opcode==5 ? UINT64_MAX-2 : UINT64_MAX-1,
                opcode==5 ? frame.geometry.local_roots : frame.geometry.local_nodes,
                opcode==5 ? frame.geometry.roots : frame.geometry.nodes,&index,charge))continue;
        const auto* previous=opcode==5 ? nullptr : frame.node(index);
        bool changed=opcode==3 ? previous[0]!=1 : opcode==4 ? previous[0]!=2 : false;
        for(uint32_t position=12;position<18;++position) {
            if(!(active&(uint32_t(1)<<position)))continue;
            const uint64_t old=opcode==5 ? frame.root(index) :
                position==12 ? previous[1] : position==17 ? previous[6] : previous[position-11];
            changed|=varied[position-12] || old!=first[position-12];
        }
        if(changed)return true;
    }
    return false;
}
template<class Charge>
__device__ bool decode(const uint64_t* books,const uint32_t choices[18],
        const semantic_program::Bank& bank,const semantic_program_image::Prefix& image,
        uint32_t active,uint32_t null_fields,bool fixed,
        const uint64_t* library_rules,const uint64_t* library_actions,uint32_t library_count,
        semantic_program_image::DecodedEdit* output,Charge charge) {
    if(!output || !choices || !books || !image.base.header)return false;
    semantic_program_image::DecodedEdit edit{};
    edit.opcode=choices[1];edit.library_index=UINT32_MAX;
    if(choices[0]==0) {
        for(uint32_t i=1;i<18;++i)if(choices[i])return false;
        *output=edit;return true;
    }
    if(choices[0]!=1 || edit.opcode<2 || edit.opcode>5 || !active ||
       (active|null_fields)!=0x3ffff || (active&null_fields))return false;
    for(uint32_t i=0;i<18;++i)
        if((null_fields&(uint32_t(1)<<i)) && choices[i])return false;
    if(image.mutations()>=image.geometry.mutations ||
       image.mutations()-image.inherited_mutations()>=image.geometry.local_mutations ||
       image.generation()==UINT64_MAX)return false;
    if(edit.opcode==2) {
        if(!leaf(books,choices[17],bank,&edit.effect,charge,true))return false;
        edit.block=uint32_t(bank.image.leaves[uint64_t(edit.effect)*4+1]);
        if(fixed) {
            if(!semantic_rule_action::library_member(books,choices,bank,library_rules,
                    library_actions,library_count,&edit.library_index,&edit.rule,charge))return false;
        } else if(image.selected_rules()>=image.geometry.mutations ||
            !semantic_rule_action::decode(books,choices,bank,&edit.rule,charge))return false;
    } else {
        if(fixed || !target(books,choices[2],edit.opcode==5 ? UINT64_MAX-2 : UINT64_MAX-1,
                edit.opcode==5 ? image.geometry.local_roots : image.geometry.local_nodes,
                edit.opcode==5 ? image.geometry.roots : image.geometry.nodes,&edit.target,charge))return false;
        if(edit.opcode==3) {
            if(!operand(books,12,choices[12],3,image.geometry.local_guards,image.geometry.guards,
                    &edit.guard,charge))return false;
            for(uint32_t edge=0;edge<4;++edge) {
                if(!operand(books,13+edge,choices[13+edge],4,image.geometry.local_nodes,image.geometry.nodes,
                        &edit.children[edge],charge) || !image.node(edit.children[edge])[0])return false;
            }
        } else if(edit.opcode==4) {
            if(!leaf(books,choices[17],bank,&edit.effect,charge))return false;
        } else if(!operand(books,13,choices[13],4,image.geometry.local_nodes,image.geometry.nodes,
                &edit.children[0],charge) || !image.node(edit.children[0])[0])return false;
    }
    *output=edit;return true;
}
} // namespace semantic_program_image_action
