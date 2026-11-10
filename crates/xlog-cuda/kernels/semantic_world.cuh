#pragma once

#include <stdint.h>

namespace semantic_world {

constexpr uint64_t kSourceHeaderWords = 36;
constexpr uint64_t kContextWords = 20;
constexpr uint64_t kEventWords = 4;
constexpr uint64_t kFactorWords = 4;

// The immutable source is a range in the same PublicationBank as its selected
// rule-cell ledger. Offsets are words relative to this acquired original range.
struct Source {
    const uint64_t* words;
    uint64_t word_count;
    uint32_t context_count,event_count,factor_count;
    uint64_t context_offset,event_offset,factor_offset,payload_offset;
    uint32_t nodes,edges,unique_slots,apply_entries,dependency_edges,frontier_entries;
    uint32_t particles,variables,base_variables,selectors;
};

template<class Read>
__device__ bool decode_source(const uint64_t* words,uint64_t bytes,
        Source* output,Read read) {
    if(!words || !output || bytes%8 || bytes/8<kSourceHeaderWords)return false;
    uint64_t h[kSourceHeaderWords];
    for(uint32_t index=0;index<kSourceHeaderWords;++index)h[index]=read(words,index);
    if(h[0]!=3 || !h[9] || h[9]>UINT32_MAX || h[13]<2 || h[13]>UINT32_MAX ||
       !h[14] || h[14]>UINT32_MAX || h[19]!=bytes/8 || h[15]!=kSourceHeaderWords ||
       h[30]!=kEventWords || h[31]!=kContextWords || h[11]<h[12])return false;
    const uint64_t events=kSourceHeaderWords+h[9]*kContextWords;
    const uint64_t factors=events+h[13]*kEventWords;
    const uint64_t payload=factors+h[14]*kFactorWords;
    if(h[16]!=events || h[17]!=factors || h[18]!=payload || payload>h[19])return false;
    for(uint32_t index=20;index<30;++index)if(h[index]>UINT32_MAX)return false;
    if(h[20]<h[13] || !h[22] || (h[22]&(h[22]-1)) || h[22]<=h[20] ||
       !h[23] || !h[25] || h[24]<h[9] || h[28]+h[29]>h[27])return false;
    *output={words,h[19],uint32_t(h[9]),uint32_t(h[13]),uint32_t(h[14]),
        h[15],h[16],h[17],h[18],uint32_t(h[20]),uint32_t(h[21]),uint32_t(h[22]),
        uint32_t(h[23]),uint32_t(h[24]),uint32_t(h[25]),uint32_t(h[26]),uint32_t(h[27]),
        uint32_t(h[28]),uint32_t(h[29])};
    return true;
}

} // namespace semantic_world
