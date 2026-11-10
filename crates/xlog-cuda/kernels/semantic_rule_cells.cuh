#pragma once
#include <stdint.h>

namespace semantic_rule_cells {

static constexpr uint32_t kReplaceWeight=6;
static constexpr uint64_t kTargetTag=UINT64_MAX-3;
static constexpr uint64_t kCoefficientSort=5;
static constexpr uint64_t kEditReceiptTag=0x584c4f47494d4544ULL;

struct View {
    const uint64_t* data;
    uint64_t contexts,context,count,generation;
};
struct Handle {
    uint64_t context,rule,birth_low,birth_high,generation,code,ordinal;
};
struct Update {
    Handle original;
    uint32_t coefficient;
};

// Exact zero and unit endpoints of the authored absolute coefficient grid.
__device__ bool endpoint(uint64_t code) { return code==1 || code==32769; }

template<class Read>
__device__ bool bind(const uint64_t* data,uint64_t bytes,View* output,Read read) {
    if(!data || !output || bytes%8 || bytes<16*8 || read(data,0)!=1)return false;
    const uint64_t words=bytes/8,contexts=read(data,9),context=read(data,15);
    const uint64_t count=read(data,11),capacity=read(data,12),generation=read(data,10);
    if(!contexts || context>=contexts || !generation || contexts>words-16 ||
       count>capacity || count>(words-16-contexts)/6 || words!=16+contexts+6*count)return false;
    *output={data,contexts,context,count,generation};
    return true;
}

// The descriptor names a logical rule. Its authority is issued here from the
// original selected ledger, including the never-reused birth and generation.
template<class Read>
__device__ bool lookup(const View& view,uint64_t rule,Handle* output,Read read) {
    if(!view.data || !output || rule>=read(view.data,16+view.context))return false;
    for(uint64_t ordinal=0;ordinal<view.count;++ordinal) {
        const uint64_t offset=16+view.contexts+ordinal*6;
        if(read(view.data,offset)!=view.context || read(view.data,offset+1)!=rule)continue;
        Handle handle{view.context,rule,read(view.data,offset+2),read(view.data,offset+3),
            read(view.data,offset+4),read(view.data,offset+5),ordinal};
        if(!(handle.birth_low|handle.birth_high) || !handle.generation ||
           handle.generation>view.generation ||
           handle.code<1 || handle.code>32769)return false;
        *output=handle;return true;
    }
    return false;
}

template<class Read>
__device__ bool update_for_handle(const View& view,const Handle& handle,
        const uint64_t* receipt,Update* output,Read read) {
    if(!receipt || !output || read(receipt,0)!=0 || read(receipt,1)!=kEditReceiptTag ||
       read(receipt,12)!=kReplaceWeight || read(receipt,40)!=0 ||
       read(receipt,31)!=handle.rule || read(receipt,32)!=handle.code ||
       read(receipt,34)!=handle.birth_low || read(receipt,35)!=handle.birth_high ||
       read(receipt,36)!=handle.generation || read(receipt,37)!=handle.context ||
       read(receipt,38)!=handle.ordinal || read(receipt,39)!=view.generation ||
       handle.context!=view.context || handle.generation==UINT64_MAX ||
       view.generation==UINT64_MAX)return false;
    const uint64_t code=read(receipt,33);
    if(!endpoint(handle.code) || !endpoint(code) || code==handle.code)return false;
    *output={handle,uint32_t(code)};
    return true;
}

template<class Read>
__device__ bool read_update(const View& view,const uint64_t* receipt,Update* output,Read read) {
    Handle original{};
    return receipt && lookup(view,read(receipt,31),&original,read) &&
        update_for_handle(view,original,receipt,output,read);
}

// Candidate receipts describe private proposed writes. They never mutate the
// acquired ledger; only the selected publication may apply these same writes.
template<class Read>
__device__ bool candidate_coefficient(const View& view,const Handle& handle,
        const uint64_t* receipts,uint32_t count,uint64_t* coefficient,Read read) {
    if(!coefficient || count>2 || (count && !receipts))return false;
    *coefficient=handle.code;
    bool changed=false;
    for(uint32_t slot=0;slot<count;++slot) {
        const auto* receipt=receipts+uint64_t(slot)*42;
        if(read(receipt,0)!=0 || read(receipt,1)!=kEditReceiptTag ||
           read(receipt,12)!=kReplaceWeight || read(receipt,40)!=0 ||
           read(receipt,31)!=handle.rule)continue;
        Update update{};
        if(changed || !update_for_handle(view,handle,receipt,&update,read))return false;
        *coefficient=update.coefficient;changed=true;
    }
    return true;
}

} // namespace semantic_rule_cells
