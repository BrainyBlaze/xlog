#include <stdint.h>
#include <math.h>
#ifdef __CUDACC__
#include <cuda/atomic>
#else
#include <atomic>
#endif
#include "semantic_transition_catalogue.cuh"
#include "semantic_feedback_encoding.cuh"
#ifdef XLOG_SEMANTIC_POLICY
#include "semantic_policy_binding.cuh"
#endif
#define XLOG_SEMANTIC_GRAPH_DEVICE_ONLY
#include "semantic_hypergraph.cu"

// Three little-endian 64-bit limbs. Signed projection terms are evaluated as
// differences of nonnegative 192-bit quantities, comparing before subtraction.
struct U192 { uint64_t w[3]; };
__device__ double u192_to_double_rn(U192 value) {
    int limb=2;
    while(limb>=0 && value.w[limb]==0)--limb;
    if(limb<0)return 0.0;
    const unsigned top=unsigned(limb)*64+63-__clzll(value.w[limb]);
    if(top<=52)return double(value.w[0]);

    // Retain the leading 53 bits without rounding any limb separately.
    unsigned discarded=top-52;
    const unsigned word=discarded/64, shift=discarded%64;
    uint64_t significand=value.w[word]>>shift;
    if(shift && word+1<3)significand |= value.w[word+1]<<(64-shift);
    const unsigned guard_word=(discarded-1)/64, guard_shift=(discarded-1)%64;
    const bool guard=(value.w[guard_word]>>guard_shift)&1;
    uint64_t sticky=value.w[guard_word]&((uint64_t(1)<<guard_shift)-1);
    for(unsigned i=0;i<guard_word;++i)sticky |= value.w[i];
    if(guard && (sticky || (significand&1)))++significand;
    if(significand==(uint64_t(1)<<53)) {
        significand>>=1;
        ++discarded;
    }
    // The significand conversion and power-of-two scaling are both exact.
    return ldexp(double(significand),int(discarded));
}
__device__ U192 add(U192 a, U192 b) {
    uint64_t carry = 0;
    for (int i=0;i<3;++i) { uint64_t s=a.w[i]+b.w[i]; uint64_t c=s<a.w[i];
        uint64_t t=s+carry; carry=c|(t<s); a.w[i]=t; }
    return a;
}
__device__ U192 sub(U192 a, U192 b) {
    uint64_t borrow=0;
    for(int i=0;i<3;++i) { uint64_t s=a.w[i]-b.w[i]; uint64_t c=a.w[i]<b.w[i];
        uint64_t t=s-borrow; borrow=c|(s<borrow); a.w[i]=t; }
    return a;
}
__device__ int compare(U192 a,U192 b) {
    for(int i=2;i>=0;--i) { if(a.w[i]<b.w[i])return -1; if(a.w[i]>b.w[i])return 1; }
    return 0;
}
__device__ U192 shifted(uint64_t n,unsigned bits) {
    U192 a={{0,0,0}}; unsigned i=bits/64, shift=bits%64;
    if(i<3)a.w[i]=n<<shift;
    if(shift && i+1<3)a.w[i+1]=n>>(64-shift);
    return a;
}
__device__ U192 multiply(U192 a,uint64_t b) {
    U192 out={{0,0,0}}; uint64_t carry=0;
    for(int i=0;i<3;++i) {
        uint64_t lo=a.w[i]*b, hi=__umul64hi(a.w[i],b);
        out.w[i]=lo+carry; carry=hi+(out.w[i]<lo);
    }
    return out;
}
__device__ U192 fp32_magnitude_scaled(float z) {
    uint32_t bits=__float_as_uint(z)&0x7fffffff, exponent=bits>>23;
    uint64_t mantissa=(bits&0x7fffff)|(exponent ? 0x800000 : 0);
    return shifted(mantissa,exponent ? exponent-1 : 0);
}
__device__ U192 distance_scaled(float maximum,float z) {
    // Double is only a coarse saturation guard. Below 32, every unequal FP32
    // pair has magnitude <2^29 and fits the exact scaled 192-bit subtraction.
    if(z==maximum)return U192{{0,0,0}};
    U192 cap=shifted(16,149);
    if(double(maximum)-double(z)>=32.0)return cap;
    U192 a=fp32_magnitude_scaled(maximum), b=fp32_magnitude_scaled(z), d;
    if(z<0 && maximum>=0)d=add(a,b);
    else if(maximum<0)d=sub(b,a);
    else d=sub(a,b);
    return compare(d,cap)>0 ? cap : d;
}
__device__ uint64_t boundary(U192 cumulative,uint32_t h) {
    // Divide B by 2^86*h: first divide the high 106 bits by h using base 2^32.
    // The discarded low 86 bits participate in the exact ties-to-even test.
    uint64_t high_lo=(cumulative.w[1]>>22)|(cumulative.w[2]<<42);
    uint64_t high_hi=cumulative.w[2]>>22;
    uint64_t rem=high_hi%h;
    uint64_t q_hi=((rem<<32)|(high_lo>>32))/h;
    rem=((rem<<32)|(high_lo>>32))%h;
    uint64_t q_lo=((rem<<32)|(high_lo&0xffffffff))/h;
    rem=((rem<<32)|(high_lo&0xffffffff))%h;
    uint64_t q=(q_hi<<32)|q_lo;
    U192 remainder={{cumulative.w[0], cumulative.w[1]&((uint64_t(1)<<22)-1),0}};
    remainder=add(remainder,shifted(rem,86));
    int tie=compare(multiply(remainder,2),shifted(h,86));
    return q+uint64_t(tie>0 || (tie==0 && (q&1)));
}
__device__ void philox(uint32_t out[4],const uint32_t counter[4],const uint32_t key[2],
        semantic_graph::NativeWorkTally* work=nullptr) {
    for(int j=0;j<4;++j)out[j]=counter[j];
    uint32_t k0=key[0],k1=key[1];
    for(int round=0;round<10;++round) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::PhiloxRound,1);
        uint64_t a=uint64_t(0xd2511f53U)*out[0], b=uint64_t(0xcd9e8d57U)*out[2];
        uint32_t next[4]={uint32_t(b>>32)^out[1]^k0,uint32_t(b),uint32_t(a>>32)^out[3]^k1,uint32_t(a)};
        for(int j=0;j<4;++j)out[j]=next[j];
        k0+=0x9e3779b9U; k1+=0xbb67ae85U;
    }
}
struct Receipt {
    uint64_t proposal,catalogue_generation;
    uint8_t catalogue_digest[32];
    uint8_t admission_binding[32];
    uint32_t ordinal,lane,slot,field,kind,choice,legal_count,active_count;
    uint32_t key[2],counter[4],random_words[4];
    uint64_t draw,cdf_start,cdf_end,mass;
    U192 p,q,factor_denominator;
    uint32_t active_fields,null_fields;
};
__device__ double receipt_importance_weight(const Receipt* receipts) {
    double weight=1.0;
    // Preserve forward ordinal order, including canonical singleton 1/1 factors.
    for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        const Receipt& receipt=receipts[ordinal];
        const double factor=__ddiv_rn(u192_to_double_rn(receipt.p),
            u192_to_double_rn(receipt.factor_denominator));
        weight=__dmul_rn(weight,factor);
    }
    return weight;
}
struct Work {
    uint64_t edit_commands,added_supports,defined_truth_changes;
};
struct ModelWorkEvent { uint64_t kind,witness,tensor,rank,dimensions[8],actual; };
struct ModelWorkInput { uint64_t events,count,bound; };
struct ExecutionWork {
    uint64_t raw,model_once,native_attempt,model_events,model_bound,overflow;
    uint64_t native_events[9],candidates[3],active_candidate;
};
__device__ void semantic_content_integrity_trap();
static_assert(sizeof(ModelWorkEvent)==104 && sizeof(ModelWorkInput)==24 && sizeof(ExecutionWork)==152,
    "original execution work ABI");

// Native owns this scratch bank and records a reset before each actual producer
// occurrence. The immutable descriptor and its upper geometry are never exported.
extern "C" __global__ void semantic_model_work_reset(uint64_t actual,uint64_t words) {
    auto* values=reinterpret_cast<uint64_t*>(actual);
    for(uint64_t i=threadIdx.x;i<words;i+=blockDim.x)values[i]=0;
}

// One owning thread appends actual logical work. Failure never wraps either
// the contribution or the common accumulator and is checked before publication.
__device__ bool execution_work_add(ExecutionWork& work,uint64_t units,bool model) {
    uint64_t& contribution=model ? work.model_once : work.native_attempt;
    if(work.overflow || units>UINT64_MAX-work.raw || units>UINT64_MAX-contribution) {
        work.overflow=1;return false;
    }
    contribution+=units;work.raw+=units;return true;
}

// Called only by the original serial owner, after the reached native operation.
// Candidate values attribute existing attempt units; they are not extra charges.
__device__ void execution_work_merge_native(ExecutionWork& work,const semantic_graph::NativeWorkTally& actual) {
    if(work.overflow)return;
    if(actual.units>UINT64_MAX-work.raw || actual.units>UINT64_MAX-work.native_attempt ||
       work.active_candidate>3 || (work.active_candidate &&
       actual.units>UINT64_MAX-work.candidates[work.active_candidate-1])) { work.overflow=1;return; }
    for(uint32_t i=0;i<9;++i)
        if(actual.events[i]>UINT64_MAX-work.native_events[i]) { work.overflow=1;return; }
    work.raw+=actual.units;work.native_attempt+=actual.units;
    for(uint32_t i=0;i<9;++i)work.native_events[i]+=actual.events[i];
    if(work.active_candidate)work.candidates[work.active_candidate-1]+=actual.units;
    if(actual.overflow)work.overflow=1;
}

// Every thread reaches this barrier at the same original phase boundary. Only
// the serial owner mutates the attempt; integer reduction ordering cannot affect
// an accepted sum, and any reduction overflow prevents publication.
__device__ void execution_work_merge_parallel(ExecutionWork& work,semantic_graph::NativeWorkTally& local) {
    __shared__ unsigned long long totals[9];
    __shared__ unsigned overflow;
    if(threadIdx.x==0) { for(uint32_t i=0;i<9;++i)totals[i]=0;overflow=0; }
    __syncthreads();
    if(local.overflow)atomicOr(&overflow,1U);
    for(uint32_t i=0;i<9;++i)if(local.events[i]) {
        const auto count=static_cast<unsigned long long>(local.events[i]);
        if(atomicAdd(&totals[i],count)>UINT64_MAX-count)atomicOr(&overflow,1U);
    }
    __syncthreads();
    if(threadIdx.x==0) {
        semantic_graph::NativeWorkTally actual{};
        if(overflow)work.overflow=1;
        else {
            for(uint32_t i=0;i<9;++i)
                semantic_graph::charge_native(&actual,static_cast<semantic_graph::NativeWorkEvent>(i),totals[i]);
            execution_work_merge_native(work,actual);
        }
    }
    local=semantic_graph::NativeWorkTally{};
    __syncthreads();
}

__device__ void consume_model_work(const ModelWorkInput& input,ExecutionWork& work) {
    if(!input.events && !input.count && !input.bound)return;
    // A prepared state is cold-initialized and submitted once. Preserve native
    // charges from preceding input/guard work; never reset the attempt here.
    if(work.model_once || work.model_events || work.model_bound || work.raw!=work.native_attempt) {
        semantic_content_integrity_trap();return;
    }
    work.model_bound=input.bound;
    if(work.overflow)return;
    // The count is frozen in the same native launch descriptor as the original
    // step. The private input allocation is never exported to the model.
    if(!input.events || !input.count || !input.bound || input.events%alignof(ModelWorkEvent) ||
       input.count>UINT64_MAX/sizeof(ModelWorkEvent) ||
       input.events>UINT64_MAX-input.count*sizeof(ModelWorkEvent)) {
        semantic_content_integrity_trap();return;
    }
    const auto* events=reinterpret_cast<const ModelWorkEvent*>(input.events);
    uint64_t declared_bound=0;
    for(uint64_t i=0;i<input.count;++i) {
        const auto& event=events[i];
        const bool saved=event.kind==15;
        if(event.kind<1 || event.kind>15 || event.rank>8 ||
           (saved && (event.witness==UINT64_MAX || event.tensor==UINT64_MAX || !event.rank || event.actual)) ||
           (!saved && (event.witness!=UINT64_MAX || event.tensor!=UINT64_MAX))) { semantic_content_integrity_trap();return; }
        bool empty=false;
        for(uint64_t axis=0;axis<8;++axis) {
            if(axis>=event.rank && event.dimensions[axis]) { semantic_content_integrity_trap();return; }
            if(axis<event.rank && !event.dimensions[axis])empty=true;
        }
        if(saved) {
            const uint64_t bytes=event.dimensions[event.rank-1];
            if(bytes!=1 && bytes!=2 && bytes!=4 && bytes!=8) { semantic_content_integrity_trap();return; }
        }
        uint64_t units=empty ? 0 : 1;
        if(!empty)for(uint64_t axis=0;axis<event.rank;++axis) {
            if(units>UINT64_MAX/event.dimensions[axis]) { work.overflow=1;return; }
            units*=event.dimensions[axis];
        }
        if(units>UINT64_MAX-declared_bound) { semantic_content_integrity_trap();return; }
        declared_bound+=units;
        if(event.actual) {
            if(event.actual%alignof(uint64_t) || event.actual>UINT64_MAX-3*sizeof(uint64_t)) {
                semantic_content_integrity_trap();return;
            }
            // This pointer was bound to the original native-owned occurrence
            // before capture ended. Stream/graph dependencies join all writers.
            const auto* actual=reinterpret_cast<const uint64_t*>(event.actual);
            if(actual[1]>1) { semantic_content_integrity_trap();return; }
            if(actual[1]) { work.overflow=1;return; }
            if(!actual[2] || actual[0]>units) { semantic_content_integrity_trap();return; }
            units=actual[0];
        }
        if(!execution_work_add(work,units,true))return;
        ++work.model_events;
        if(work.model_once>input.bound) { work.overflow=1;return; }
    }
    if(declared_bound!=input.bound) { semantic_content_integrity_trap();return; }
}
struct TaskFacts {
    uint64_t correct[3],g,p,c;
    int64_t v;
    uint64_t eligible;
};
struct TaskEvaluation {
    uint64_t lane_refusal[2]; // 0 accepted, 1 scope, 2 hard constraint.
    semantic_graph::Receipt query_receipts[3][3];
    uint64_t query_count,winner;
    int64_t return_value;
    TaskFacts facts[3]; // Base, first learned lane, second learned lane.
};
struct State {
    uint32_t model_generation,family_id;
    uint64_t stream_serial,proposal,next_proposal,blocks,status,catalogue_generation;
    uint8_t catalogue_digest[32];
    uint8_t binding_digest[32];
    semantic_graph::Receipt semantic_receipts[7];
    Work work[2];
    double importance_weight;
    uint64_t retired_roots_mask;
    semantic_graph::Receipt cleanup_receipts[3];
    TaskEvaluation task_evaluation;
    ExecutionWork execution_work;
};
struct SourceSlot {
    uint64_t token,logical_position,kind,provenance,valid,committed,recomputed,provenance_record;
};
struct TextRow { uint64_t source_slot,logical_position; };
struct TextBinding { uint64_t rows,count,selected; };
struct ModelUpdateBinding { uint64_t source,bytes,slots[2]; };
struct ContinuationInputs {
    TextBinding text;
    uint64_t active_rows,active_row_count,numerical_admissibility,transition_kind,authority_bytes;
    uint64_t training_selection,model_update_bindings,model_update_binding_count,model_update_admissibility;
};
struct PublicationStorageEntry { uint64_t pointer,bytes,generation; };
struct PublicationRange {
    uint64_t role,index,storage_slot,generation,offset_bytes,length_bytes,logical_begin,logical_end;
    uint64_t digest[4];
    uint64_t backing_digest[4];
};
struct PublicationHeader {
    uint64_t abi,instance[4],recovered_instance[4],sealed_epoch,base_word,publication_word;
    uint64_t ring_head,prefix_extent,semantic_owner,semantic_slot,semantic_generation;
    uint64_t semantic_digest[4],semantic_extents[3];
    uint64_t model_generation,neural_bank,neural_generation,cache_generation,authority_generation;
    uint64_t terminal,terminal_position,fuel,proposal,stream_serial,family_id,training_cursor,training_rng[4];
    uint64_t range_count,model_geometry_digest[4],model_numerical_digest[4];
    uint64_t logical_digest[4],state_digest[4],descriptor_digest[4];
};
struct PublicationBank {
    PublicationHeader header;
    SourceSlot source[32];
    State state;
    Receipt receipts[136];
};
struct PublicationControl {
    uint64_t abi,word,instance[4],banks[2],directories[2],storage,storage_count,contract,continuation;
    uint64_t reader_counts[2],reader_gate,refusal;
};
struct PublicationRoleCount { uint64_t role,count; };
struct ModelContractLayout {
    uint64_t schema_begin,schema_bytes,schema_digest_offset,generation_offset,numerical_digest_offset,identity_offset;
};
struct PublicationContract {
    uint64_t abi,range_capacity,prefix_capacity,window_capacity,feedback_capacity,max_position,pad_token;
    uint64_t terminal_tokens,terminal_token_count,final_intent_payload_bytes;
    uint64_t model_generation,policy_generation,authority_generation,semantic_owner;
    uint64_t task_identity[4],topology_identity[4],table_identity[4],role_counts,role_count;
    ModelContractLayout model_contract_layout;
};
struct PendingContinuation {
    uint64_t abi,instance[4],base_word,model_generation,authority_generation;
    uint64_t topology_identity[4],table_identity[4],prefix_identity[4],ranges,range_count,transition_kind;
    TextBinding text;
    uint64_t numerical_admissibility,training_selection,model_update_bindings,model_update_binding_count;
    uint64_t model_update_admissibility;
};
struct PublicationCommand { uint64_t control,lease,operation; };
struct PublicationLease { uint64_t abi,status,instance[4],word,bank,epoch,active,transition_kind; };
struct PublicationStepResult { uint64_t abi,word,refusal;PublicationHeader header;uint64_t advanced; };
struct SemanticTrainingViewOriginRecord {
    uint64_t present,transition;
    uint64_t lineage_instance[4];
    uint64_t predecessor_instance[4],predecessor_word,predecessor_logical[4],predecessor_state[4];
    uint64_t successor_instance[4],successor_word,successor_logical[4],successor_state[4];
    uint64_t model_generation,stream_serial,family_id,proposal;
    uint64_t model_geometry_digest[4],model_numerical_digest[4];
};
struct PublicationTensorLayout {
    uint64_t role,index,element_bytes,scalar_type,rank,logical_axis,dimensions[4],strides_bytes[4];
};
struct TensorLayoutTableHeader { uint64_t abi,layout_count,active_computed,active_row_count; };
struct PublicationStepInput {
    uint64_t role,index,destination,capacity_bytes,backing,backing_bytes;
    PublicationTensorLayout layout;
};
static_assert(sizeof(PublicationStepInput)==160,"publication step input ABI");
struct ActiveRow { uint64_t physical_row,source_slot,logical_position,kind; };
static_assert(sizeof(TensorLayoutTableHeader)==32 && sizeof(ActiveRow)==32,"active row table ABI");
struct RawFeedbackRecord {
    uint64_t valid,statement_index,statement_offset_bytes,statement_length_bytes,pro,contra;
    semantic_graph::Receipt query_receipt;
};
struct CompletionCoverage {
    uint64_t abi,instance[4],base_word,model_generation,topology_identity[4],table_identity[4],prefix_identity[4];
    uint64_t prefix_begin,prefix_end,source_digest[4],attention_digest[4],linear_digest[4];
    uint64_t value_digest[4],edit_digest[4],receipt_digest[4];
};
struct AttemptReceipt {
    uint64_t abi,instance[4],base_word,next_word,logical_digest[4],action_receipts_digest[4];
    uint64_t semantic_receipts_digest[4],coverage_digest[4],replay_head_digest[4],intent_head_digest[4];
    uint64_t acknowledgement_head_digest[4],previous_attempt_digest[4],receipt_digest[4];
};
struct TokenProvenanceRecord {
    uint64_t source_slot,logical_position,token,base_word,proposal,ordinal,action_receipt_digest[4];
    uint64_t authority_generation,origin_logical_digest[4],authority_closure_digest[4],record_digest[4];
};
struct IntentQueueHeader {
    uint64_t abi,count,capacity,payload_used_bytes,payload_capacity_bytes,effect_offset_bytes,effect_length_bytes,chain_head[4];
};
struct IntentEntry {
    uint64_t stable_identity[4],checkpoint_lineage[4],base_logical[4],effective_delta[4],result_logical[4];
    uint64_t effect_digest[4],payload_digest[4],payload_offset,payload_len,audit_instance[4],audit_epoch;
    uint64_t previous_chain[4],chain[4];
};
struct PolicyField { uint64_t embeddings,biases; uint32_t cardinality; int32_t null_category; };
struct PolicyDescriptor {
    uint64_t z,recurrence,positions,hidden,scores,recurrent;
    PolicyField fields[18];
};
struct SemanticTrainingViewSelection {
    uint64_t status,row_count,capacity,ordinal,basis,window,source_length,block_size,prefix_extent,answer_start;
    uint64_t identity[4],source_identity[4],content_identity[4];
    SemanticTrainingViewOriginRecord origin;
    uint64_t origin_candidate,training_rng[4];
};
struct SemanticTrainingObjectiveRecord {
    uint64_t identity[4],row_count,capacity,group_count,group_member_count,canary_count;
    uint64_t evaluator_min_bits,evaluator_max_bits,coefficient_bits[9],cost_unit[4],cost_cap;
};
struct SemanticTrainingObjectiveGroupRecord {
    uint64_t kind,denominator,member_offset,member_count;
};
struct PolicyBackward {
    uint64_t cotangents,parameters,text,baselines,recurrent,scores,status,parameter_cells,text_cells;
    uint64_t selection,objective,objective_groups,objective_group_members,origin_candidate,mode;
};
struct Descriptor {
    uint64_t logits,support,scratch,receipts,state,components,codebooks,arena[7];
    PolicyDescriptor policy;
    PolicyBackward backward;
    uint64_t task;
    PublicationCommand publication;
    TextBinding text;
    ModelWorkInput model_work;
};
static_assert(sizeof(U192)==24,"integer ABI");
static_assert(sizeof(Receipt)==264,"receipt ABI");
static_assert(sizeof(Work)==24,"semantic work ABI");
static_assert(sizeof(TaskFacts)==64,"task facts ABI");
static_assert(sizeof(TaskEvaluation)==3256,"task evaluation ABI");
static_assert(sizeof(State)==6952,"state ABI");
static_assert(sizeof(PolicyField)==24,"policy field ABI");
static_assert(sizeof(PolicyDescriptor)==480,"policy ABI");
static_assert(sizeof(SemanticTrainingViewOriginRecord)==352,"training view origin ABI");
static_assert(sizeof(SemanticTrainingViewSelection)==568,"training view selection ABI");
static_assert(sizeof(SemanticTrainingObjectiveRecord)==200,"training objective ABI");
static_assert(sizeof(SemanticTrainingObjectiveGroupRecord)==32,"training objective group ABI");
static_assert(sizeof(PolicyBackward)==120,"policy backward ABI");
static_assert(sizeof(SourceSlot)==64,"source slot ABI");
static_assert(sizeof(PublicationStorageEntry)==24,"owned storage ABI");
static_assert(sizeof(PublicationRange)==128,"publication range ABI");
static_assert(sizeof(PublicationHeader)==488,"publication header ABI");
static_assert(sizeof(PublicationControl)==144,"publication control ABI");
static_assert(sizeof(PublicationBank)==45392,"publication bank ABI");
static_assert(sizeof(ModelContractLayout)==48,"model contract byte layout ABI");
static_assert(sizeof(PublicationContract)==272,"publication contract ABI");
static_assert(sizeof(TextRow)==16,"compact text row ABI");
static_assert(sizeof(TextBinding)==24,"compact text binding ABI");
static_assert(sizeof(ModelUpdateBinding)==32,"model update binding ABI");
static_assert(sizeof(ContinuationInputs)==96,"resident continuation inputs ABI");
static_assert(sizeof(PendingContinuation)==248,"pending continuation ABI");
static_assert(sizeof(PublicationCommand)==24,"publication command ABI");
static_assert(sizeof(PublicationLease)==88,"publication lease ABI");
static_assert(sizeof(PublicationStepResult)==sizeof(PublicationHeader)+32,"publication step result ABI");
static_assert(sizeof(PublicationTensorLayout)==112,"tensor layout ABI");
static_assert(sizeof(RawFeedbackRecord)==384,"raw feedback ABI");
static_assert(sizeof(CompletionCoverage)==360,"completion coverage ABI");
static_assert(sizeof(AttemptReceipt)==344,"attempt receipt ABI");
static_assert(sizeof(TokenProvenanceRecord)==184,"token provenance ABI");
static_assert(sizeof(IntentQueueHeader)==88,"intent queue ABI");
static_assert(sizeof(IntentEntry)==344,"intent entry ABI");
static_assert(sizeof(Descriptor)==792,"launch ABI");
static_assert((2*262144+4*65536)*sizeof(uint32_t)<=SCRATCH_BYTES,"serial scratch and completion witnesses");

__device__ uint64_t text_row_count(TextBinding binding) {
    return binding.count && !(binding.count%alignof(uint64_t)) && binding.count<=UINT64_MAX-sizeof(uint64_t)
        ? *reinterpret_cast<const uint64_t*>(binding.count) : UINT64_MAX;
}
__device__ bool text_selected(TextBinding binding,uint32_t slot) {
    return slot<32 && binding.selected && binding.selected<=UINT64_MAX-32 &&
        reinterpret_cast<const uint8_t*>(binding.selected)[slot]==1;
}
__device__ bool text_any_selected(TextBinding binding) {
    for(uint32_t slot=0;slot<32;++slot)if(text_selected(binding,slot))return true;
    return false;
}
__device__ uint64_t text_row_index(TextBinding binding,uint32_t slot) {
    const auto* rows=reinterpret_cast<const TextRow*>(binding.rows);
    const uint64_t count=text_row_count(binding);
    if(count>32)return UINT64_MAX;
    for(uint64_t i=0;i<count;++i)if(rows[i].source_slot==slot)return i;
    return UINT64_MAX;
}
__device__ bool validate_text_binding(const SourceSlot* source,TextBinding binding) {
    const uint64_t count=text_row_count(binding);
    if(!source || count>32 || !binding.rows || binding.rows%alignof(TextRow) ||
       binding.rows>UINT64_MAX-32*sizeof(TextRow) || !binding.selected || binding.selected>UINT64_MAX-32)return false;
    const auto* rows=reinterpret_cast<const TextRow*>(binding.rows);
    uint64_t seen=0,expected=0;
    for(uint32_t i=0;i<32;++i) {
        if(source[i].valid==1 && source[i].kind==2)expected|=uint64_t(1)<<i;
        const uint8_t selected=reinterpret_cast<const uint8_t*>(binding.selected)[i];
        if(selected>1 || (selected && !(expected&(uint64_t(1)<<i))))return false;
    }
    for(uint64_t i=0;i<count;++i) {
        const TextRow row=rows[i];if(row.source_slot>=32)return false;
        const uint64_t bit=uint64_t(1)<<row.source_slot;
        if((seen&bit) || !(expected&bit) || source[row.source_slot].logical_position!=row.logical_position)return false;
        seen|=bit;
    }
    return seen==expected;
}
__device__ bool text_null_receipt(const Receipt& r) {
    return r.kind==COMPONENT_KIND_TEXT && r.choice==TEXT_NULL && r.legal_count==1 && r.active_count==1 &&
        !r.active_fields && !r.null_fields && !r.cdf_start && r.cdf_end==(uint64_t(1)<<63) && r.mass==r.cdf_end &&
        r.p.w[0]==1 && !r.p.w[1] && !r.p.w[2] && r.q.w[0]==1 && !r.q.w[1] && !r.q.w[2] &&
        r.factor_denominator.w[0]==1 && !r.factor_denominator.w[1] && !r.factor_denominator.w[2];
}
__device__ Receipt component_receipt(const Component& c,const State& state,
        semantic_graph::NativeWorkTally* work=nullptr) {
    Receipt r={};r.proposal=state.proposal;r.catalogue_generation=CATALOGUE_GENERATION;
    for(int j=0;j<32;++j){r.catalogue_digest[j]=CATALOGUE_DIGEST[j];r.admission_binding[j]=state.binding_digest[j];}
    r.ordinal=c.ordinal;r.lane=c.lane;r.slot=c.slot;r.field=c.field;r.kind=c.kind;
    r.active_fields=c.kind==COMPONENT_KIND_EDIT ? NO_EDIT_ACTIVE_MASK : 0;
    r.null_fields=c.kind==COMPONENT_KIND_EDIT ? NO_EDIT_NULL_MASK : 0;
    r.key[0]=uint32_t(CATALOGUE_GENERATION);r.key[1]=state.model_generation;
    uint64_t block=uint64_t(COMPONENT_COUNT)*state.proposal+c.ordinal;
    uint64_t stream=(state.stream_serial<<8)|state.family_id;
    r.counter[0]=uint32_t(block);r.counter[1]=uint32_t(block>>32);
    r.counter[2]=uint32_t(stream);r.counter[3]=uint32_t(stream>>32);
    philox(r.random_words,r.counter,r.key,work);
    r.draw=(uint64_t(r.random_words[0])|(uint64_t(r.random_words[1])<<32))&0x7fffffffffffffffULL;
    return r;
}

// Source indices are ring identities, not the model's packed physical rows.
// Only FILLED rows that predate this invocation contribute to its coverage.
__device__ bool publication_text_position_limit(const PublicationContract& contract,uint64_t* limit) {
    return contract.window_capacity==32 &&
        semantic_graph::checked_add(contract.prefix_capacity,contract.window_capacity,limit);
}
__device__ uint64_t publication_validate_source(const PublicationBank& bank,
        const PublicationContract& contract,uint64_t* structural_end) {
    uint64_t text_limit=0;
    if(!publication_text_position_limit(contract,&text_limit) || bank.header.ring_head>=32 ||
       bank.header.prefix_extent>contract.prefix_capacity ||
       bank.header.prefix_extent>contract.max_position)return 1;
    const uint64_t* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    uint64_t terminal_position=contract.max_position;
    for(uint32_t i=0;i<32;++i) {
        const SourceSlot& s=bank.source[i];
        if(s.kind>2 || s.valid!=(s.kind!=0) || s.committed!=0 || s.recomputed>1)return 1;
        if(s.kind==0) {
            if(s.token!=contract.pad_token || s.logical_position || s.provenance ||
               s.recomputed || s.provenance_record)return 1;
            continue;
        }
        if(s.logical_position<bank.header.prefix_extent || s.logical_position>=text_limit ||
           s.logical_position>=contract.max_position)return 1;
        for(uint32_t j=0;j<i;++j)
            if(bank.source[j].valid && bank.source[j].logical_position==s.logical_position)return 1;
        if(s.kind==2) {
            if(s.token!=MASK_TOKEN || s.provenance || s.provenance_record || s.recomputed)return 1;
        } else {
            if(s.token>=TEXT_CARDINALITY || (s.provenance!=1 && s.provenance!=2))return 1;
            for(uint64_t j=0;j<contract.terminal_token_count;++j)
                if(s.token==terminals[j] && s.logical_position<terminal_position)
                    terminal_position=s.logical_position;
        }
    }
    for(uint32_t i=0;i<32;++i)
        if(bank.source[i].kind==1 && bank.source[i].logical_position>terminal_position)return 1;
    uint64_t end=bank.header.prefix_extent;
    for(uint32_t step=0;step<32;++step) {
        bool filled=false;
        for(uint32_t i=0;i<32;++i)
            filled|=bank.source[i].kind==1 && bank.source[i].logical_position==end;
        if(!filled)break;
        ++end;
    }
    if(end>contract.prefix_capacity)return 1;
    *structural_end=end;
    return 0;
}

// These operations use device-scope acquire/release ordering. The host branch
// compiles this exact source for the serial diagnostic, not a CUDA substitute.
__device__ uint64_t publication_load(uint64_t& value) {
#ifdef __CUDACC__
    return cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(value).load(cuda::memory_order_acquire);
#else
    return std::atomic_ref<uint64_t>(value).load(std::memory_order_acquire);
#endif
}
__device__ bool publication_compare_exchange(uint64_t& value,uint64_t expected,uint64_t next) {
#ifdef __CUDACC__
    return cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(value).compare_exchange_strong(
        expected,next,cuda::memory_order_acq_rel,cuda::memory_order_acquire);
#else
    return std::atomic_ref<uint64_t>(value).compare_exchange_strong(
        expected,next,std::memory_order_acq_rel,std::memory_order_acquire);
#endif
}
__device__ void publication_store(uint64_t& value,uint64_t next) {
#ifdef __CUDACC__
    cuda::atomic_ref<uint64_t,cuda::thread_scope_device>(value).store(next,cuda::memory_order_release);
#else
    std::atomic_ref<uint64_t>(value).store(next,std::memory_order_release);
#endif
}
__device__ bool publication_identity_equal(const uint64_t* a,const uint64_t* b) {
    for(uint32_t i=0;i<4;++i)if(a[i]!=b[i])return false;
    return true;
}
__device__ bool publication_header_equal(const PublicationHeader& a,const PublicationHeader& b) {
    const auto* left=reinterpret_cast<const uint64_t*>(&a);
    const auto* right=reinterpret_cast<const uint64_t*>(&b);
    for(uint64_t i=0;i<sizeof(PublicationHeader)/sizeof(uint64_t);++i)
        if(left[i]!=right[i])return false;
    return true;
}
__device__ PublicationBank* publication_bank(PublicationControl& control,uint64_t word) {
    return reinterpret_cast<PublicationBank*>(control.banks[word&1]);
}
__device__ bool publication_bank_matches(const PublicationControl& control,
        const PublicationBank& bank,uint64_t word) {
    return bank.header.abi==1 && bank.header.sealed_epoch==word>>1 &&
        bank.header.publication_word==word && publication_identity_equal(bank.header.instance,control.instance);
}
__device__ PublicationBank* publication_acquired_bank(const PublicationControl& control,
        const PublicationLease& lease) {
    if(control.abi!=1 || lease.abi!=1 || lease.status || lease.active!=1 || lease.bank>1 ||
       lease.bank!=(lease.word&1) || lease.epoch!=lease.word>>1 || !control.reader_counts[lease.bank] ||
       !publication_identity_equal(lease.instance,control.instance) || !control.banks[lease.bank])return nullptr;
    auto* bank=reinterpret_cast<PublicationBank*>(control.banks[lease.bank]);
    return publication_bank_matches(control,*bank,lease.word) ? bank : nullptr;
}
// The gate only closes the acquire-load/count-increment race. The PublicationWord
// remains the sole canonical selector; no reader derives a bank from this gate.
__device__ uint64_t publication_header_eligibility(const PublicationControl& control,
        const PublicationBank& bank,uint64_t word,uint64_t transition_kind) {
    if(!publication_bank_matches(control,bank,word))return 3;
    if(control.reader_counts[(word&1)^1])return 2;
    if((word>>1)==(UINT64_MAX>>1))return 5;
    if(bank.header.terminal>1)return 4;
    if(transition_kind<1 || transition_kind>4 ||
       (bank.header.terminal==1)!=(transition_kind==3))return 1;
    return bank.header.fuel<(transition_kind==1 ? 2 : 1) ? 5 : 0;
}
__device__ uint64_t publication_acquire(PublicationControl& control,PublicationLease& lease,
        uint64_t requested_kind=0,uint64_t expected_word=UINT64_MAX) {
    if(control.abi!=1 || lease.active || !control.banks[0] || !control.banks[1])return 1;
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    const uint64_t word=publication_load(control.word),bank=word&1;
    uint64_t status=0;
    const auto& parent=*publication_bank(control,word);
    const uint64_t kind=requested_kind>=1 && requested_kind<=4 && parent.header.terminal==1 ? 3 : requested_kind;
    if(requested_kind)status=expected_word!=UINT64_MAX && word!=expected_word ? 3 : publication_header_eligibility(control,parent,word,kind);
    else if(!publication_bank_matches(control,parent,word) ||
       (parent.header.terminal==1 && !parent.header.fuel))status=1;
    if(!status &&
       control.reader_counts[bank]==UINT64_MAX)status=1;
    if(!status) {
        ++control.reader_counts[bank];
        lease=PublicationLease{};lease.abi=1;lease.word=word;lease.bank=bank;lease.epoch=word>>1;
        lease.transition_kind=kind;
        for(uint32_t i=0;i<4;++i)lease.instance[i]=control.instance[i];
        publication_store(lease.active,1);
    }
    publication_store(control.reader_gate,0);
    lease.status=status;
    return status;
}
// Session schedules this only after every consumer's stream join. A replayed or
// foreign lease cannot decrement a later reader's count, including after reuse.
__device__ uint64_t publication_release(PublicationControl& control,PublicationLease& lease) {
    if(control.abi!=1 || lease.abi!=1 || lease.bank>1 || lease.bank!=(lease.word&1) ||
       lease.epoch!=lease.word>>1 || !publication_identity_equal(lease.instance,control.instance))return 1;
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    uint64_t status=0;
    if(!publication_bank_matches(control,*publication_bank(control,lease.word),lease.word) ||
       !control.reader_counts[lease.bank] || !publication_compare_exchange(lease.active,1,0))status=1;
    else --control.reader_counts[lease.bank];
    publication_store(control.reader_gate,0);
    lease.status=status;
    return status;
}
__device__ uint64_t publication_prepare_text(const PublicationBank& base,PublicationBank& next,
        const PublicationContract& contract,const Receipt* receipts,uint64_t winner,uint64_t structural_end,TextBinding text) {
    if(winner>2 || structural_end<base.header.prefix_extent)return 1;
    next.header=base.header;next.header.prefix_extent=structural_end;
    for(uint32_t i=0;i<32;++i) {
        next.source[i]=base.source[i];
        SourceSlot& s=next.source[i];
        if(s.kind==1 && s.logical_position<structural_end) {
            s=SourceSlot{};s.token=contract.pad_token;
        } else if(s.kind==2 && winner && text_selected(text,i)) {
            if(receipts[(winner-1)*68+i].choice>=TEXT_CARDINALITY)return 1;
            s.token=receipts[(winner-1)*68+i].choice;s.kind=1;s.provenance=2;
            s.committed=0;s.recomputed=0;
        }
    }
    uint64_t next_end=0;
    if(publication_validate_source(next,contract,&next_end))return 1;
    const uint64_t* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    for(uint32_t i=0;i<32;++i)if(next.source[i].kind==1)
        for(uint64_t j=0;j<contract.terminal_token_count;++j)
            if(next.source[i].token==terminals[j]) {
                // Every preceding logical position must already be contiguous.
                if(next_end<=next.source[i].logical_position)return 1;
                next.header.terminal=1; // Drain required; not an output intent.
                next.header.terminal_position=next.source[i].logical_position;
            }
    return 0;
}

__device__ bool publication_resolve_range(const PublicationControl& control,const PublicationRange& range,
        uint8_t** bytes) {
    *bytes=nullptr;
    if(!control.storage || range.storage_slot>=control.storage_count || range.logical_end<range.logical_begin)return false;
    const auto& storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage)[range.storage_slot];
    if(!storage.generation || storage.generation!=range.generation ||
       range.offset_bytes>storage.bytes || range.length_bytes>storage.bytes-range.offset_bytes ||
       storage.pointer>UINT64_MAX-storage.bytes)return false;
    // A genuine empty model tensor has an owned zero-byte storage entry. Null
    // is its data pointer, not an invalid owner and not an alias-class identity.
    if(!storage.bytes)return !storage.pointer && range.role>=18 && range.role<=20 &&
        !range.offset_bytes && !range.length_bytes && !range.logical_begin && !range.logical_end;
    if(!storage.pointer)return false;
    *bytes=reinterpret_cast<uint8_t*>(storage.pointer+range.offset_bytes);
    return true;
}
__device__ uint8_t* publication_range_bytes(const PublicationControl& control,const PublicationRange& range) {
    uint8_t* bytes=nullptr;
    return publication_resolve_range(control,range,&bytes) ? bytes : nullptr;
}
__device__ const PublicationRange* publication_find_range(const PublicationRange* ranges,uint64_t count,
        uint64_t role,uint64_t index=0) {
    for(uint64_t i=0;i<count;++i)if(ranges[i].role==role && ranges[i].index==index)return &ranges[i];
    return nullptr;
}
struct TensorLayoutTableView {
    const TensorLayoutTableHeader* header;
    const PublicationTensorLayout* layouts;
    const ActiveRow* rows;
};
__device__ bool publication_tensor_table(const PublicationControl& control,
        const PublicationRange* ranges,uint64_t count,TensorLayoutTableView* view) {
    const auto* table=publication_find_range(ranges,count,55);
    if(!table || table->index || table->logical_begin || table->logical_end ||
       table->length_bytes<sizeof(TensorLayoutTableHeader))return false;
    const auto* bytes=publication_range_bytes(control,*table);
    if(!bytes || reinterpret_cast<uintptr_t>(bytes)%alignof(TensorLayoutTableHeader))return false;
    const auto* header=reinterpret_cast<const TensorLayoutTableHeader*>(bytes);
    uint64_t layout_bytes=0,row_bytes=0,length=0;
    if(header->abi!=1 || header->active_computed>1 || (!header->active_computed && header->active_row_count) ||
       !semantic_graph::checked_mul(header->layout_count,sizeof(PublicationTensorLayout),&layout_bytes) ||
       !semantic_graph::checked_mul(header->active_row_count,sizeof(ActiveRow),&row_bytes) ||
       !semantic_graph::checked_add(sizeof(TensorLayoutTableHeader),layout_bytes,&length) ||
       !semantic_graph::checked_add(length,row_bytes,&length) || length!=table->length_bytes)return false;
    view->header=header;
    view->layouts=reinterpret_cast<const PublicationTensorLayout*>(bytes+sizeof(TensorLayoutTableHeader));
    view->rows=reinterpret_cast<const ActiveRow*>(bytes+sizeof(TensorLayoutTableHeader)+layout_bytes);
    return true;
}
__device__ const PublicationTensorLayout* publication_find_layout(const PublicationControl& control,
        const PublicationRange* ranges,uint64_t count,uint64_t role,uint64_t index) {
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,count,&table))return nullptr;
    for(uint64_t i=0;i<table.header->layout_count;++i)
        if(table.layouts[i].role==role && table.layouts[i].index==index)return &table.layouts[i];
    return nullptr;
}

__device__ bool publication_active_map_shape(const TensorLayoutTableView& table,const PublicationContract& contract) {
    uint64_t capacity=0,text_limit=0,position_end=0;
    if(!publication_text_position_limit(contract,&text_limit) ||
       !semantic_graph::checked_add(contract.window_capacity,contract.feedback_capacity,&capacity) ||
       !semantic_graph::checked_add(text_limit,contract.feedback_capacity,&position_end) ||
       table.header->active_row_count>capacity)return false;
    uint64_t text_slots=0,last_kind=0,last_position=0;
    for(uint64_t i=0;i<table.header->active_row_count;++i) {
        const auto& row=table.rows[i];
        const uint64_t order=row.kind==1 ? 1 : row.kind==3 ? 2 : row.kind==2 ? 3 : 0;
        if(!order || order<last_kind || row.physical_row>=capacity ||
           (i && row.physical_row<=table.rows[i-1].physical_row))return false;
        if(row.kind==3) {
            if(row.source_slot<32 || row.source_slot>=capacity ||
               row.logical_position!=contract.prefix_capacity+row.source_slot)return false;
        } else {
            if(row.source_slot>=32 || (text_slots&(1ULL<<row.source_slot)) || row.logical_position>=text_limit ||
               (order==last_kind && row.logical_position<=last_position))return false;
            text_slots|=1ULL<<row.source_slot;
        }
        last_kind=order;last_position=row.logical_position;
    }
    return true;
}
__device__ bool publication_tensor_role(uint64_t role) {
    return (role>=4 && role<=13) || (role>=18 && role<=25) ||
        (role>=51 && role<=54);
}
struct PublicationTensorBytes {
    const uint8_t* bytes;
    const PublicationTensorLayout* layout;
    const ActiveRow* active_rows;
    uint64_t dimensions[4],prefix[16];
    __device__ uint64_t offset(uint64_t cell) const {
        uint64_t result=0;
        for(int i=int(layout->rank)-1;i>=0;--i) {
            uint64_t coordinate=cell%dimensions[i];cell/=dimensions[i];
            if(active_rows && uint64_t(i)==layout->logical_axis)coordinate=active_rows[coordinate].physical_row;
            result+=coordinate*layout->strides_bytes[i];
        }
        return result;
    }
    __device__ uint8_t operator[](uint64_t index) const {
        if(index<sizeof(prefix))return reinterpret_cast<const uint8_t*>(prefix)[index];
        index-=sizeof(prefix);
        return layout ? bytes[offset(index/layout->element_bytes)+index%layout->element_bytes] : bytes[index];
    }
};
// Publication seals and transient witnesses traverse the same logical bytes.
// Storage addresses, padding and strides do not enter the content identity.
__device__ uint64_t publication_content_view(const uint8_t* data,uint64_t length,const PublicationRange& range,
        const PublicationTensorLayout* layout,PublicationTensorBytes* view,uint64_t* content_bytes) {
    const uint64_t pointer=reinterpret_cast<uint64_t>(data);
    if(range.logical_end<range.logical_begin || range.length_bytes!=length ||
       (length && !data) || pointer>UINT64_MAX-length)return 1;
    view->bytes=data;view->layout=layout;view->active_rows=nullptr;
    for(uint32_t i=0;i<16;++i)view->prefix[i]=0;
    view->prefix[0]=0x786c6f6772616e31ULL;view->prefix[1]=range.role;view->prefix[2]=range.index;
    view->prefix[3]=range.logical_begin;view->prefix[4]=range.logical_end;
    *content_bytes=length;
    if(!layout)return 0;
    // U8, U32, U64, F16, BF16, F32, I64 and Bool8 retain distinct type codes.
    const uint64_t sizes[9]={0,1,4,8,2,2,4,8,1};
    if(layout->role!=range.role || layout->index!=range.index || layout->rank>4 ||
       layout->scalar_type<1 || layout->scalar_type>8 || layout->element_bytes!=sizes[layout->scalar_type] ||
       (layout->logical_axis!=UINT64_MAX && layout->logical_axis>=layout->rank) || pointer%layout->element_bytes)return 1;
    view->prefix[5]=layout->element_bytes;view->prefix[6]=layout->scalar_type;
    view->prefix[7]=layout->rank;view->prefix[8]=layout->logical_axis;
    bool empty=false;
    uint32_t axes[4],axis_count=0;
    for(uint32_t i=0;i<layout->rank;++i) {
        if(layout->strides_bytes[i]%layout->element_bytes)return 1;
        uint64_t dimension=layout->dimensions[i];
        if(uint64_t(i)==layout->logical_axis) {
            dimension=range.logical_end-range.logical_begin;
            if(dimension>layout->dimensions[i])return 1;
        }
        view->dimensions[i]=dimension;view->prefix[9+i]=dimension;
        empty|=dimension==0;
        if(layout->dimensions[i]>1)axes[axis_count++]=i;
    }
    for(uint64_t i=layout->rank;i<4;++i)if(layout->dimensions[i] || layout->strides_bytes[i])return 1;
    // Sort physical axes only for the non-overlap check. The hash continues to
    // enumerate logical coordinates, including real transposed producer views.
    for(uint32_t i=1;i<axis_count;++i) {
        const uint32_t axis=axes[i];uint32_t j=i;
        while(j && layout->strides_bytes[axes[j-1]]>layout->strides_bytes[axis]) {
            axes[j]=axes[j-1];--j;
        }
        axes[j]=axis;
    }
    uint64_t span=layout->element_bytes;
    for(uint32_t i=0;!empty && i<axis_count;++i) {
        const uint32_t axis=axes[i];uint64_t extent=0;
        if(layout->strides_bytes[axis]<span ||
           !semantic_graph::checked_mul(layout->strides_bytes[axis],layout->dimensions[axis]-1,&extent) ||
           !semantic_graph::checked_add(span,extent,&span))return 1;
    }
    uint64_t cells=empty ? 0 : 1,last=0;
    for(uint32_t i=0;!empty && i<layout->rank;++i) {
        const uint64_t dimension=view->dimensions[i];
        if(!semantic_graph::checked_mul(cells,dimension,&cells))return 1;
        uint64_t extent=0;
        if(!semantic_graph::checked_mul(dimension-1,layout->strides_bytes[i],&extent) ||
           !semantic_graph::checked_add(last,extent,&last))return 1;
    }
    if(!semantic_graph::checked_mul(cells,layout->element_bytes,content_bytes) ||
       (*content_bytes && (last>length || layout->element_bytes>length-last)))return 1;
    return 0;
}
__device__ uint64_t publication_tensor_view(const PublicationControl& control,const PublicationRange& range,
        const PublicationTensorLayout* layout,PublicationTensorBytes* view,uint64_t* content_bytes,
        const TensorLayoutTableView* active_table=nullptr) {
    uint8_t* bytes=nullptr;
    if(!publication_resolve_range(control,range,&bytes) ||
       publication_content_view(bytes,range.length_bytes,range,layout,view,content_bytes))return 1;
    if(!layout)return bytes ? 0 : 1;
    const bool empty_model=range.role>=18 && range.role<=25 &&
        layout->logical_axis==UINT64_MAX && !*content_bytes;
    if(empty_model) {
        if(range.length_bytes || range.logical_begin || range.logical_end)return 1;
    } else if(!bytes)return 1;
    // Native publication arenas retain their canonical capacity-stride and
    // logical-axis constraints even when transient producer views are general.
    uint64_t minimum_stride=layout->element_bytes;
    for(int i=int(layout->rank)-1;!(range.role>=18 && range.role<=25) && i>=0;--i) {
        if((!layout->dimensions[i] && uint64_t(i)!=layout->logical_axis && !empty_model) || layout->strides_bytes[i]<minimum_stride ||
           !semantic_graph::checked_mul(layout->strides_bytes[i],layout->dimensions[i] ? layout->dimensions[i] : 1,&minimum_stride))return 1;
    }
    if(range.role>=51 && range.role<=54) {
        // Semantic row count is not a dense physical extent. The same sealed
        // map selects source, destination and hashed cells; holes are padding.
        // A pending full-capacity producer may have zero semantic rows, while
        // its published range uses zero length. Directory validation owns that
        // distinction and the computed/uncomputed flag.
        if(!active_table || !active_table->header || range.logical_begin ||
           layout->logical_axis>=layout->rank ||
           range.logical_end!=active_table->header->active_row_count ||
           range.logical_end>layout->dimensions[layout->logical_axis])return 1;
        for(uint64_t row=0;row<range.logical_end;++row)
            if(active_table->rows[row].physical_row>=layout->dimensions[layout->logical_axis] ||
               (row && active_table->rows[row-1].physical_row>=active_table->rows[row].physical_row))return 1;
        view->active_rows=active_table->rows;
        if(*content_bytes) {
            uint64_t last=0;
            for(uint64_t axis=0;axis<layout->rank;++axis) {
                const uint64_t coordinate=axis==layout->logical_axis ?
                    active_table->rows[range.logical_end-1].physical_row : view->dimensions[axis]-1;
                uint64_t extent=0;
                if(!semantic_graph::checked_mul(coordinate,layout->strides_bytes[axis],&extent) ||
                   !semantic_graph::checked_add(last,extent,&last))return 1;
            }
            if(last>range.length_bytes || layout->element_bytes>range.length_bytes-last)return 1;
        }
    }
    return 0;
}
__device__ uint64_t publication_content_digest(const PublicationTensorBytes& view,uint64_t bytes,uint64_t* digest) {
    if(bytes>UINT64_MAX/8-sizeof(view.prefix))return 1;
    semantic_graph::sha256(view,sizeof(view.prefix)+bytes,digest);
    return 0;
}
__device__ uint64_t publication_range_digest(const PublicationControl& control,const PublicationRange& range,
        const PublicationTensorLayout* layout,uint64_t* digest,const TensorLayoutTableView* active_table=nullptr) {
    PublicationTensorBytes view{};uint64_t bytes=0;
    if(publication_tensor_view(control,range,layout,&view,&bytes,active_table))return 1;
    return publication_content_digest(view,bytes,digest);
}

// Full model backing bytes are sealed separately from typed logical content.
// The descriptor retains both seals, so stride gaps and sibling storage slices
// survive export/restoration without changing the tensor-witness hash format.
__device__ uint64_t publication_backing_digest(const PublicationControl& control,
        const PublicationRange& range,uint64_t* digest) {
    if(range.role<18 || range.role>25 || range.storage_slot>=control.storage_count || !control.storage)return 1;
    const auto& allocation=reinterpret_cast<const PublicationStorageEntry*>(control.storage)[range.storage_slot];
    if(allocation.generation!=range.generation || allocation.pointer>UINT64_MAX-allocation.bytes ||
       allocation.bytes>UINT64_MAX/8 || (!allocation.pointer && allocation.bytes))return 1;
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(allocation.pointer),allocation.bytes,digest);
    return 0;
}

__device__ void semantic_content_integrity_trap() {
#ifdef __CUDACC__
    __trap();
#else
    __builtin_trap();
#endif
}

// The admission contract fixes the initial model and storage geometry, not the
// generation selected by a later publication. Resolve that generation from the
// acquired bank and verify its original record and numerical seals together.
__device__ uint64_t publication_validate_selected_model(const PublicationControl& control,
        const PublicationBank& bank);

// Resolve the range in the current device lease, not a host-selected bank.
// The native owner retains both directories and the original sealing layout.
// Expected digest bytes never enter launch arguments.
// PrefixSource uses has_layout=0; its typed export is not its sealing format.
// All writes precede this kernel and every consumer follows it in stream order.
extern "C" __global__ void semantic_publication_content_guard(uint64_t control_ptr,
        uint64_t role,uint64_t index,PublicationTensorLayout layout,uint64_t has_layout,uint64_t lease_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!control_ptr || control_ptr%alignof(PublicationControl) || control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) || lease_ptr>UINT64_MAX-sizeof(PublicationLease) ||
       !role || role>55 || has_layout>1 || has_layout!=uint64_t(publication_tensor_role(role))) {
        semantic_content_integrity_trap();return;
    }
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.bank>1 || !control.contract || control.contract%alignof(PublicationContract) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract) ||
       !control.banks[lease.bank] || control.banks[lease.bank]%alignof(PublicationBank) ||
       control.banks[lease.bank]>UINT64_MAX-sizeof(PublicationBank) ||
       !control.directories[lease.bank] || control.directories[lease.bank]%alignof(PublicationRange)) {
        semantic_content_integrity_trap();return;
    }
    const auto* bank=publication_acquired_bank(control,lease);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(!bank || contract.abi!=1 || bank->header.range_count>contract.range_capacity ||
       bank->header.range_count>(UINT64_MAX-control.directories[lease.bank])/sizeof(PublicationRange)) {
        semantic_content_integrity_trap();return;
    }
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease.bank]);
    const uint64_t count=bank->header.range_count;
    const auto* selected=publication_find_range(ranges,count,role,index);
    if(!selected) { semantic_content_integrity_trap();return; }
    // Authenticate the complete model binding at its contract boundary; tensor
    // guards below keep checking individual payloads against their own seals.
    if(role==44 && publication_validate_selected_model(control,*bank)) {
        semantic_content_integrity_trap();return;
    }
    const auto& range=*selected;
    uint64_t digest[4];
    TensorLayoutTableView table{};
    if(range.role>=51 && range.role<=54) {
        const auto* table_range=publication_find_range(ranges,count,55);
        if(!table_range || publication_range_digest(control,*table_range,nullptr,digest) ||
           !publication_identity_equal(digest,table_range->digest) ||
           !publication_tensor_table(control,ranges,count,&table) || table.header->active_computed!=1 ||
           !publication_active_map_shape(table,contract)) { semantic_content_integrity_trap();return; }
        const auto* sealed_layout=publication_find_layout(control,ranges,count,range.role,range.index);
        if(!sealed_layout) { semantic_content_integrity_trap();return; }
        const auto* actual=reinterpret_cast<const uint64_t*>(sealed_layout);
        const auto* expected=reinterpret_cast<const uint64_t*>(&layout);
        for(uint32_t i=0;i<sizeof(layout)/sizeof(uint64_t);++i)
            if(actual[i]!=expected[i]) { semantic_content_integrity_trap();return; }
        uint64_t capacity=0;
        if(!semantic_graph::checked_add(32,contract.feedback_capacity,&capacity) ||
           layout.logical_axis!=((range.role==51 || range.role==52) ? 2 : 0) ||
           layout.logical_axis>=layout.rank || layout.dimensions[layout.logical_axis]!=capacity ||
           range.logical_begin || range.logical_end!=table.header->active_row_count ||
           ((range.length_bytes==0)!=(table.header->active_row_count==0))) { semantic_content_integrity_trap();return; }
    }
    if(publication_range_digest(control,range,has_layout ? &layout : nullptr,digest,&table) ||
       !publication_identity_equal(digest,range.digest))semantic_content_integrity_trap();
    if(range.role>=18 && range.role<=25 &&
       (publication_backing_digest(control,range,digest) ||
        !publication_identity_equal(digest,range.backing_digest)))semantic_content_integrity_trap();
}

// Native ownership retains the original ModelContract range and a copy of its
// raw payload interval before releasing its publication reader. The copied range
// keeps its original offset; data_ptr already starts at the copied interval.
// Expected digest bytes remain in that private range, independent of bank reuse.
extern "C" __global__ void semantic_retained_model_contract_guard(uint64_t data_ptr,
        uint64_t length_bytes,uint64_t range_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!data_ptr || !length_bytes || data_ptr>UINT64_MAX-length_bytes ||
       !range_ptr || range_ptr%alignof(PublicationRange) || range_ptr>UINT64_MAX-sizeof(PublicationRange) ||
       (data_ptr<range_ptr+sizeof(PublicationRange) && range_ptr<data_ptr+length_bytes)) {
        semantic_content_integrity_trap();return;
    }
    const auto& range=*reinterpret_cast<const PublicationRange*>(range_ptr);
    PublicationTensorBytes view{};uint64_t bytes=0,digest[4];
    if(range.role!=44 || range.index || range.generation!=1 ||
       range.offset_bytes>UINT64_MAX-range.length_bytes ||
       publication_content_view(reinterpret_cast<const uint8_t*>(data_ptr),length_bytes,range,nullptr,&view,&bytes) ||
       publication_content_digest(view,bytes,digest) || !publication_identity_equal(digest,range.digest))
        semantic_content_integrity_trap();
}

// The expected digest is a private native allocation retained with the actual
// producer. Capture precedes hooks; verify precedes consumption on its stream.
// A mismatch is fatal to this CUDA context and has no host status/readback path.
extern "C" __global__ void semantic_tensor_content_witness(uint64_t data,uint64_t length,
        PublicationRange range,PublicationTensorLayout layout,uint64_t expected_digest_ptr,uint64_t verify) {
    if(blockIdx.x || threadIdx.x)return;
    if(verify>1 || !expected_digest_ptr || expected_digest_ptr%alignof(uint64_t) ||
       expected_digest_ptr>UINT64_MAX-4*sizeof(uint64_t) || data>UINT64_MAX-length ||
       (length && data<expected_digest_ptr+4*sizeof(uint64_t) && expected_digest_ptr<data+length)) {
        semantic_content_integrity_trap();return;
    }
    PublicationTensorBytes view{};uint64_t bytes=0,digest[4];
    if(publication_content_view(reinterpret_cast<const uint8_t*>(data),length,range,&layout,&view,&bytes) ||
       publication_content_digest(view,bytes,digest)) {
        semantic_content_integrity_trap();return;
    }
    auto* expected=reinterpret_cast<uint64_t*>(expected_digest_ptr);
    if(verify) {
        if(!publication_identity_equal(digest,expected))semantic_content_integrity_trap();
    } else semantic_graph::copy_identity(expected,digest);
}
__device__ uint64_t publication_validate_directory(const PublicationControl& control,
        const PublicationContract& contract,PublicationRange* ranges,uint64_t count,bool seal) {
    if(!ranges || count>contract.range_capacity || contract.role_count!=55 || !contract.role_counts)return 1;
    const auto* roles=reinterpret_cast<const PublicationRoleCount*>(contract.role_counts);
    uint64_t expected=0,tensor_count=0;
    for(uint64_t i=0;i<55;++i) {
        if(roles[i].role!=i+1 || !semantic_graph::checked_add(expected,roles[i].count,&expected))return 1;
        if(publication_tensor_role(i+1))tensor_count+=roles[i].count;
        else if(roles[i].count!=(i+1==16 ? 3 : 1))return 1;
    }
    if(expected!=count || roles[3].count!=roles[4].count || roles[5].count!=roles[6].count ||
       roles[50].count!=roles[3].count || roles[51].count!=roles[4].count ||
       roles[52].count!=roles[5].count || roles[53].count!=roles[6].count || !roles[17].count)return 1;
    for(uint64_t role=8;role<=13;++role)if(roles[role-1].count!=1)return 1;
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,count,&table) || table.header->layout_count!=tensor_count ||
       !publication_active_map_shape(table,contract))return 1;
    const auto* layouts=table.layouts;
    for(uint64_t i=0;i<tensor_count;++i) {
        if(!publication_tensor_role(layouts[i].role) || !publication_find_range(ranges,count,layouts[i].role,layouts[i].index))return 1;
        for(uint64_t j=0;j<i;++j)
            if(layouts[j].role==layouts[i].role && layouts[j].index==layouts[i].index)return 1;
    }
    uint64_t cursor=0;
    for(uint64_t role=1;role<=55;++role)for(uint64_t index=0;index<roles[role-1].count;++index) {
        PublicationRange& range=ranges[cursor++];
        if(range.role!=role || range.index!=index)return 1;
        const auto* layout=publication_find_layout(control,ranges,count,role,index);
        if(publication_tensor_role(role)!=bool(layout))return 1;
        if(!publication_tensor_role(role) && role!=1 && role!=2 && !range.length_bytes)return 1;
        if(role==14 && range.length_bytes!=sizeof(CompletionCoverage))return 1;
        if(role==15 && (contract.feedback_capacity<3 || contract.feedback_capacity>UINT64_MAX/sizeof(RawFeedbackRecord) ||
            range.length_bytes!=contract.feedback_capacity*sizeof(RawFeedbackRecord)))return 1;
        if(role==33 && range.length_bytes!=sizeof(AttemptReceipt))return 1;
        if(layout && (role==4 || role==5 || role==51 || role==52) &&
            (layout->rank!=4 || layout->logical_axis!=2 || layout->dimensions[0]!=1 ||
             (layout->scalar_type!=4 && layout->scalar_type!=5 && layout->scalar_type!=6)))return 1;
        if(layout && role>=6 && role<=13 && layout->logical_axis!=UINT64_MAX)return 1;
        if(layout && role>=8 && role<=13 && layout->scalar_type!=6)return 1;
        if(layout && role>=51 && role<=54 &&
           (layout->logical_axis!=((role==51 || role==52) ? 2 : 0) ||
            layout->dimensions[layout->logical_axis]!=32+contract.feedback_capacity ||
            range.logical_begin || range.logical_end!=table.header->active_row_count ||
            ((range.length_bytes==0)!=(table.header->active_row_count==0))))return 1;
        PublicationTensorBytes view{};uint64_t bytes=0;
        if(publication_tensor_view(control,range,layout,&view,&bytes,&table))return 1;
        if(seal && publication_range_digest(control,range,layout,range.digest,&table))return 1;
    }
    return 0;
}
__device__ bool publication_mutable_role(uint64_t role) {
    return role==1 || role==2 || role==3 || (role>=4 && role<=15) || (role>=18 && role<=25) || role==30 || role==31 || role==33 || role==39 || role==44 ||
        (role>=51 && role<=55);
}
__device__ uint64_t publication_validate_storage(const PublicationControl& control) {
    if(!control.storage || !control.storage_count)return 1;
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    for(uint64_t i=0;i<control.storage_count;++i) {
        if((!storage[i].pointer != !storage[i].bytes) || !storage[i].generation ||
           storage[i].pointer>UINT64_MAX-storage[i].bytes)return 1;
        for(uint64_t j=0;j<i;++j)
            if(storage[i].pointer<storage[j].pointer+storage[j].bytes &&
               storage[j].pointer<storage[i].pointer+storage[i].bytes)return 1;
    }
    return 0;
}
__device__ uint64_t publication_validate_destinations(const PublicationControl& control,
        const PublicationRange* base,const PublicationRange* next,uint64_t count) {
    for(uint64_t i=0;i<count;++i) {
        uint8_t *base_bytes=nullptr,*next_bytes=nullptr;
        if(base[i].role!=next[i].role || base[i].index!=next[i].index ||
           !publication_resolve_range(control,base[i],&base_bytes) ||
           !publication_resolve_range(control,next[i],&next_bytes))return 1;
        if(!publication_mutable_role(base[i].role))continue;
        // Prefix and provenance arenas are append-only. Model views may share
        // a declared backing within their bank, never with the acquired bank
        // or a native control/cache owner.
        if(base[i].role==1 || base[i].role==2 || base[i].role==31) {
            if(base[i].storage_slot!=next[i].storage_slot || base[i].offset_bytes!=next[i].offset_bytes)return 1;
        } else if(!(base[i].role>=18 && base[i].role<=25))for(uint64_t j=0;j<count;++j)
            if(next[i].storage_slot==base[j].storage_slot)return 1;
        for(uint64_t j=0;j<i;++j)
            if(publication_mutable_role(next[j].role) && next[i].storage_slot==next[j].storage_slot &&
               !((next[i].role>=18 && next[i].role<=25) && (next[j].role>=18 && next[j].role<=25)))return 1;
    }
    return 0;
}
__device__ bool publication_match_active_row(const TensorLayoutTableView& table,uint64_t* cursor,
        uint64_t physical,uint64_t slot,uint64_t position,uint64_t kind) {
    if(*cursor>=table.header->active_row_count)return false;
    const auto& row=table.rows[(*cursor)++];
    return row.physical_row==physical && row.source_slot==slot && row.logical_position==position && row.kind==kind;
}
__device__ bool publication_validate_active_origin(const PublicationControl& control,const PublicationContract& contract,
        const PublicationBank& base,const PublicationRange* bank_ranges,const TensorLayoutTableView& table) {
    if(!table.header->active_computed || !publication_active_map_shape(table,contract))return false;
    const auto* feedback_range=publication_find_range(bank_ranges,base.header.range_count,15);
    uint64_t feedback_bytes=0;
    if(!feedback_range || !semantic_graph::checked_mul(contract.feedback_capacity,sizeof(RawFeedbackRecord),&feedback_bytes) ||
       feedback_range->length_bytes!=feedback_bytes)return false;
    const auto* feedback=reinterpret_cast<const RawFeedbackRecord*>(publication_range_bytes(control,*feedback_range));
    if(!feedback)return false;
    uint64_t cursor=0,physical=0;
    // Bounded source roster; sorting by logical position never trims FILLED
    // rows at the first gap. Physical feedback slots stay reserved in between.
    for(uint64_t kind=1;kind<=2;++kind) {
        uint32_t used=0;
        for(uint32_t n=0;n<32;++n) {
            uint32_t chosen=32;
            for(uint32_t slot=0;slot<32;++slot)if(!(used&(1u<<slot)) && base.source[slot].valid==1 && base.source[slot].kind==kind &&
                (chosen==32 || base.source[slot].logical_position<base.source[chosen].logical_position))chosen=slot;
            if(chosen==32)break;
            used|=1u<<chosen;
            if(!publication_match_active_row(table,&cursor,physical++,chosen,base.source[chosen].logical_position,kind))return false;
        }
        if(kind==1)for(uint64_t slot=0;slot<contract.feedback_capacity;++slot,++physical) {
            if(feedback[slot].valid>1)return false;
            if(feedback[slot].valid && !publication_match_active_row(table,&cursor,physical,32+slot,
                    contract.prefix_capacity+32+slot,3))return false;
        }
    }
    return cursor==table.header->active_row_count;
}
__device__ bool publication_model_spans_overlap(uint64_t left,uint64_t left_bytes,
        uint64_t right,uint64_t right_bytes) {
    return left_bytes && right_bytes && left<right+right_bytes && right<left+left_bytes;
}
__device__ uint64_t publication_validate_model_update(const PublicationControl& control,
        const PublicationBank& base,const PendingContinuation& pending) {
    if(pending.transition_kind!=4)
        return pending.model_update_bindings || pending.model_update_binding_count ||
            pending.model_update_admissibility;
    if(!pending.model_update_bindings ||
       pending.model_update_bindings%alignof(ModelUpdateBinding) ||
       !pending.model_update_binding_count ||
       pending.model_update_binding_count>base.header.range_count ||
       pending.model_update_binding_count>
           (UINT64_MAX-pending.model_update_bindings)/sizeof(ModelUpdateBinding))return 1;
    if(!pending.model_update_admissibility || pending.model_update_admissibility==UINT64_MAX ||
       *reinterpret_cast<const uint8_t*>(pending.model_update_admissibility)>1)return 1;
    if(base.header.neural_bank>1)return 1;
    const uint64_t bank=base.header.neural_bank;
    const auto* bindings=reinterpret_cast<const ModelUpdateBinding*>(pending.model_update_bindings);
    const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t allocations=0;
    for(uint64_t i=0;i<base.header.range_count;++i)if(old[i].role>=18 && old[i].role<=25) {
        bool first=true;
        for(uint64_t prior=0;prior<i;++prior)
            if(old[prior].role>=18 && old[prior].role<=25 &&
               old[prior].storage_slot==old[i].storage_slot)first=false;
        if(!first)continue;
        ++allocations;bool found=false;
        for(uint64_t item=0;item<pending.model_update_binding_count;++item)
            if(bindings[item].slots[bank]==old[i].storage_slot)found=true;
        if(!found)return 1;
    }
    if(allocations!=pending.model_update_binding_count)return 1;
    for(uint64_t item=0;item<pending.model_update_binding_count;++item) {
        const auto& binding=bindings[item];
        if(binding.slots[0]>=control.storage_count || binding.slots[1]>=control.storage_count ||
           binding.slots[0]==binding.slots[1] || storage[binding.slots[0]].bytes!=binding.bytes ||
           storage[binding.slots[1]].bytes!=binding.bytes ||
           (binding.bytes && (!binding.source || binding.source>UINT64_MAX-binding.bytes ||
            storage[binding.slots[0]].pointer==storage[binding.slots[1]].pointer)))return 1;
        bool used=false;
        for(uint64_t i=0;i<base.header.range_count;++i)
            if(old[i].role>=18 && old[i].role<=25 &&
               old[i].storage_slot==binding.slots[bank])used=true;
        if(!used)return 1;
        for(uint64_t slot=0;slot<control.storage_count;++slot) {
            const auto& owned=storage[slot];
            if(binding.bytes && owned.bytes &&
               publication_model_spans_overlap(binding.source,binding.bytes,owned.pointer,owned.bytes))return 1;
        }
        for(uint64_t prior=0;prior<item;++prior)
            if(binding.slots[0]==bindings[prior].slots[0] ||
               binding.slots[0]==bindings[prior].slots[1] ||
               binding.slots[1]==bindings[prior].slots[0] ||
               binding.slots[1]==bindings[prior].slots[1] ||
               publication_model_spans_overlap(binding.source,binding.bytes,
                   bindings[prior].source,bindings[prior].bytes))return 1;
    }
    return 0;
}
__device__ uint64_t publication_validate_continuation(const PublicationControl& control,
        const PublicationContract& contract,const PublicationBank& base,const PendingContinuation& pending,uint64_t end) {
    if(pending.abi!=1 || !publication_identity_equal(pending.instance,base.header.instance) ||
       pending.base_word!=base.header.publication_word || pending.model_generation!=base.header.model_generation ||
       pending.authority_generation!=base.header.authority_generation ||
       !publication_identity_equal(pending.topology_identity,contract.topology_identity) ||
       !publication_identity_equal(pending.table_identity,contract.table_identity) || !pending.ranges ||
       pending.range_count>contract.range_capacity ||
       publication_validate_model_update(control,base,pending))return 1;
    if(!validate_text_binding(base.source,pending.text) || (pending.transition_kind!=1 && text_any_selected(pending.text)))return 1;
    const auto* bank_ranges=reinterpret_cast<const PublicationRange*>(control.directories[pending.base_word&1]);
    const auto* prefix=publication_find_range(bank_ranges,base.header.range_count,3);
    if(!prefix || prefix->length_bytes!=32 || !publication_range_bytes(control,*prefix) ||
       !publication_identity_equal(pending.prefix_identity,
           reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix))))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(pending.ranges);
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,pending.range_count,&table) ||
       !publication_validate_active_origin(control,contract,base,bank_ranges,table))return 1;
    uint64_t expected=3;
    for(uint64_t i=0;i<base.header.range_count;++i)
        if((bank_ranges[i].role>=4 && bank_ranges[i].role<=13) || (bank_ranges[i].role>=51 && bank_ranges[i].role<=54))++expected;
    if(pending.range_count!=expected)return 1;
    if(table.header->layout_count!=expected-3)return 1;
    for(uint64_t i=0;i<table.header->layout_count;++i) {
        const auto& layout=table.layouts[i];
        if(!((layout.role>=4 && layout.role<=13) || (layout.role>=51 && layout.role<=54)) ||
           !publication_find_range(ranges,pending.range_count,layout.role,layout.index))return 1;
        for(uint64_t j=0;j<i;++j)if(layout.role==table.layouts[j].role && layout.index==table.layouts[j].index)return 1;
    }
    for(uint64_t i=0;i<pending.range_count;++i) {
        const PublicationRange& r=ranges[i];
        const bool active=r.role>=51 && r.role<=54;
        if((r.role!=1 && r.role!=39 && r.role!=55 && (r.role<4 || r.role>13) && !active) || !publication_range_bytes(control,r))return 1;
        for(uint64_t j=0;j<i;++j)if(ranges[j].role==r.role && ranges[j].index==r.index)return 1;
        if(r.role==55) { if(r.index)return 1;continue; }
        if(r.role==39) {
            if(r.index || !r.length_bytes || r.logical_begin || r.logical_end)return 1;
            const auto* original=publication_find_range(bank_ranges,base.header.range_count,39);
            if(!original)return 1;
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            if(r.length_bytes>storage[original->storage_slot].bytes-original->offset_bytes)return 1;
            continue;
        }
        if(r.role==1 && (r.logical_begin!=base.header.prefix_extent || r.logical_end!=end))return 1;
        if((r.role==4 || r.role==5) &&
           (r.logical_begin!=(pending.transition_kind==4 ? 0 : base.header.prefix_extent) ||
            r.logical_end!=end))return 1;
        if(r.role>=6 && r.role<=13 && (r.logical_begin || r.logical_end))return 1;
        if(active && (r.logical_begin!=0 || r.logical_end!=table.header->active_row_count))return 1;
        const auto* original=publication_find_range(bank_ranges,base.header.range_count,r.role,r.index);
        if(!original)return 1;
        if(r.role==1) {
            if(r.index || r.length_bytes!=(end-base.header.prefix_extent)*sizeof(SourceSlot))return 1;
            const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,r));
            for(uint64_t k=0;k<end-base.header.prefix_extent;++k) {
                const SourceSlot& s=source[k];const SourceSlot* acquired=nullptr;
                for(uint32_t slot=0;slot<32;++slot)
                    if(base.source[slot].kind==1 && base.source[slot].logical_position==base.header.prefix_extent+k)
                        acquired=&base.source[slot];
                if(!acquired || s.kind!=1 || s.valid!=1 || s.committed!=1 || s.recomputed!=1 ||
                   s.logical_position!=acquired->logical_position || s.token!=acquired->token ||
                   s.provenance!=acquired->provenance || s.provenance_record!=acquired->provenance_record)return 1;
            }
        } else {
            const auto* from=publication_find_layout(control,ranges,pending.range_count,r.role,r.index);
            const auto* to=publication_find_layout(control,bank_ranges,base.header.range_count,r.role,r.index);
            if(!from || !to || from->rank!=to->rank || from->scalar_type!=to->scalar_type ||
               from->element_bytes!=to->element_bytes || from->logical_axis!=to->logical_axis)return 1;
            if(r.role>=6 && r.role<=13 && from->logical_axis!=UINT64_MAX)return 1;
            for(uint64_t axis=0;axis<to->rank;++axis)
                if(axis!=to->logical_axis && from->dimensions[axis]!=to->dimensions[axis])return 1;
            if(active && to->logical_axis!=UINT64_MAX && r.logical_end>to->dimensions[to->logical_axis])return 1;
            if(active && (from->logical_axis!=((r.role==51 || r.role==52) ? 2 : 0) ||
                from->dimensions[from->logical_axis]!=32+contract.feedback_capacity))return 1;
            PublicationTensorBytes view{};uint64_t bytes=0;
            if(publication_tensor_view(control,r,from,&view,&bytes,&table))return 1;
            if(r.role==4 || r.role==5) {
                const uint64_t expected_rows=pending.transition_kind==4 ? contract.prefix_capacity : 32;
                if(from->logical_axis!=2 || from->dimensions[2]!=expected_rows ||
                   to->logical_axis!=2 || to->dimensions[2]!=contract.prefix_capacity ||
                   original->logical_begin || original->logical_end!=base.header.prefix_extent)return 1;
            }
        }
    }
    return 0;
}
__device__ uint64_t publication_prepare_continuation(PublicationControl& control,
        const PublicationLease& lease,ContinuationInputs inputs) {
    const uint64_t requested_kind=inputs.transition_kind;
    if((requested_kind==4)!=(inputs.training_selection!=0) ||
       (inputs.training_selection && (inputs.training_selection%alignof(uint64_t) ||
        inputs.training_selection>UINT64_MAX-sizeof(uint64_t))) ||
       (inputs.model_update_bindings==0)!=(inputs.model_update_binding_count==0) ||
       (requested_kind==4)!=(inputs.model_update_bindings!=0) ||
       (requested_kind==4)!=(inputs.model_update_admissibility!=0) ||
       (inputs.model_update_bindings &&
        (inputs.model_update_bindings%alignof(ModelUpdateBinding) ||
         inputs.model_update_binding_count>
             (UINT64_MAX-inputs.model_update_bindings)/sizeof(ModelUpdateBinding))))return 1;
    if(lease.transition_kind)inputs.transition_kind=lease.transition_kind;
    if(!publication_acquired_bank(control,lease) ||
       !control.directories[lease.bank] || !control.contract || !control.continuation ||
       control.contract%alignof(PublicationContract) || control.continuation%alignof(PendingContinuation) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract) ||
       control.continuation>UINT64_MAX-sizeof(PendingContinuation) || publication_validate_storage(control))return 1;
    const auto& base=*reinterpret_cast<const PublicationBank*>(control.banks[lease.bank]);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    auto& pending=*reinterpret_cast<PendingContinuation*>(control.continuation);
    uint64_t capacity=0,active_bytes=0,end=0;
    if(!publication_bank_matches(control,base,lease.word) || contract.abi!=1 ||
       base.header.range_count>contract.range_capacity || publication_validate_selected_model(control,base) ||
       base.header.authority_generation!=contract.authority_generation || pending.abi!=1 ||
       inputs.transition_kind<1 || inputs.transition_kind>4 || !inputs.authority_bytes || !pending.ranges ||
       pending.ranges%alignof(PublicationRange) || pending.range_count>contract.range_capacity ||
       pending.range_count>(UINT64_MAX-pending.ranges)/sizeof(PublicationRange) ||
       !semantic_graph::checked_add(32,contract.feedback_capacity,&capacity) ||
       !semantic_graph::checked_mul(capacity,sizeof(ActiveRow),&active_bytes) ||
       !inputs.active_rows || inputs.active_rows%alignof(ActiveRow) || inputs.active_rows>UINT64_MAX-active_bytes ||
       !inputs.active_row_count || inputs.active_row_count%alignof(uint64_t) ||
       inputs.active_row_count>UINT64_MAX-sizeof(uint64_t) ||
       !inputs.numerical_admissibility || inputs.numerical_admissibility==UINT64_MAX ||
       *reinterpret_cast<const uint8_t*>(inputs.numerical_admissibility)>1 ||
       !validate_text_binding(base.source,inputs.text) ||
       (inputs.transition_kind!=1 && text_any_selected(inputs.text)) ||
       publication_validate_source(base,contract,&end))return 1;
    const uint64_t active_count=*reinterpret_cast<const uint64_t*>(inputs.active_row_count);
    if(active_count>capacity)return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(pending.ranges);
    const auto* bank_ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease.bank]);
    const auto* prefix=publication_find_range(bank_ranges,base.header.range_count,3);
    if(!prefix || prefix->length_bytes!=32 || !publication_range_bytes(control,*prefix))return 1;
    auto* authority=const_cast<PublicationRange*>(publication_find_range(ranges,pending.range_count,39));
    if(!authority)return 1;
    PublicationRange actual_authority=*authority;actual_authority.length_bytes=inputs.authority_bytes;
    if(!publication_range_bytes(control,actual_authority))return 1;
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,pending.range_count,&table) ||
       table.header->active_computed!=1 || table.header->active_row_count>capacity)return 1;
    auto* table_range=const_cast<PublicationRange*>(publication_find_range(ranges,pending.range_count,55));
    uint64_t layout_bytes=0,table_bytes=0;
    if(!semantic_graph::checked_mul(table.header->layout_count,sizeof(PublicationTensorLayout),&layout_bytes) ||
       !semantic_graph::checked_add(sizeof(TensorLayoutTableHeader),layout_bytes,&table_bytes))return 1;
    PublicationRange full_table=*table_range;
    if(!semantic_graph::checked_add(table_bytes,active_bytes,&full_table.length_bytes) ||
       !publication_range_bytes(control,full_table))return 1;
    uint64_t row_bytes=0;
    if(!semantic_graph::checked_mul(active_count,sizeof(ActiveRow),&row_bytes) ||
       !semantic_graph::checked_add(table_bytes,row_bytes,&table_range->length_bytes))return 1;
    auto* header=const_cast<TensorLayoutTableHeader*>(table.header);
    auto* rows=const_cast<ActiveRow*>(table.rows);
    header->active_row_count=active_count;
    const auto* actual_rows=reinterpret_cast<const ActiveRow*>(inputs.active_rows);
    for(uint64_t i=0;i<active_count;++i)rows[i]=actual_rows[i];
    if(!publication_validate_active_origin(control,contract,base,bank_ranges,table))return 1;
    authority->length_bytes=inputs.authority_bytes;
    pending.transition_kind=inputs.transition_kind;
    pending.base_word=lease.word;
    semantic_graph::copy_identity(pending.instance,base.header.instance);
    pending.model_generation=base.header.model_generation;
    pending.authority_generation=base.header.authority_generation;
    semantic_graph::copy_identity(pending.topology_identity,contract.topology_identity);
    semantic_graph::copy_identity(pending.table_identity,contract.table_identity);
    semantic_graph::copy_identity(pending.prefix_identity,
        reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix)));
    pending.text=inputs.text;pending.numerical_admissibility=inputs.numerical_admissibility;
    pending.training_selection=inputs.transition_kind==4 ? inputs.training_selection : 0;
    pending.model_update_bindings=inputs.transition_kind==4 ? inputs.model_update_bindings : 0;
    pending.model_update_binding_count=inputs.transition_kind==4 ? inputs.model_update_binding_count : 0;
    pending.model_update_admissibility=inputs.transition_kind==4 ? inputs.model_update_admissibility : 0;
    for(uint64_t i=0;i<pending.range_count;++i) {
        auto& range=ranges[i];
        if(range.role==1 || range.role==4 || range.role==5) {
            range.logical_begin=(inputs.transition_kind==4 && (range.role==4 || range.role==5)) ?
                0 : base.header.prefix_extent;
            range.logical_end=end;
        } else if(range.role>=51 && range.role<=54) {
            range.logical_begin=0;range.logical_end=active_count;
        } else range.logical_begin=range.logical_end=0;
        if(range.role!=1)continue;
        PublicationRange full_source=range;full_source.length_bytes=32*sizeof(SourceSlot);
        if(!publication_range_bytes(control,full_source))return 1;
        const uint64_t count=end-base.header.prefix_extent;
        range.length_bytes=count*sizeof(SourceSlot);
        auto* source=reinterpret_cast<SourceSlot*>(publication_range_bytes(control,range));
        if(!source)return 1;
        for(uint64_t row=0;row<count;++row) {
            const SourceSlot* original=nullptr;
            for(uint32_t slot=0;slot<32;++slot)if(base.source[slot].kind==1 &&
                base.source[slot].logical_position==base.header.prefix_extent+row)original=&base.source[slot];
            if(!original)return 1;
            source[row]=*original;source[row].committed=source[row].recomputed=1;
        }
    }
    return publication_validate_continuation(control,contract,base,pending,end);
}
extern "C" __global__ void semantic_publication_prepare_continuation(uint64_t control_ptr,
        uint64_t lease_ptr,ContinuationInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    if(!control_ptr || control_ptr%alignof(PublicationControl) || control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) || lease_ptr>UINT64_MAX-sizeof(PublicationLease) ||
       publication_prepare_continuation(*reinterpret_cast<PublicationControl*>(control_ptr),
           *reinterpret_cast<const PublicationLease*>(lease_ptr),inputs))semantic_content_integrity_trap();
}
extern "C" __global__ void semantic_publication_prepare_model_update_admissibility(
        uint64_t source_ptr,uint64_t destination_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!source_ptr || source_ptr==UINT64_MAX || !destination_ptr || destination_ptr==UINT64_MAX) {
        semantic_content_integrity_trap();return;
    }
    const uint8_t value=*reinterpret_cast<const uint8_t*>(source_ptr);
    if(value>1) { semantic_content_integrity_trap();return; }
    *reinterpret_cast<uint8_t*>(destination_ptr)=value;
}
extern "C" __global__ void semantic_publication_apply_model_update(uint64_t control_ptr,
        uint64_t lease_ptr) {
    if(!control_ptr || control_ptr%alignof(PublicationControl) ||
       control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) ||
       lease_ptr>UINT64_MAX-sizeof(PublicationLease)) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    const auto* base=publication_acquired_bank(control,lease);
    if(!base || lease.transition_kind!=4 || !control.continuation ||
       control.continuation%alignof(PendingContinuation) || !control.storage ||
       base->header.neural_bank>1) {
        semantic_content_integrity_trap();return;
    }
    const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
    if(pending.base_word!=lease.word || pending.transition_kind!=4 ||
       !pending.model_update_bindings ||
       pending.model_update_bindings%alignof(ModelUpdateBinding) ||
       !pending.model_update_admissibility || pending.model_update_admissibility==UINT64_MAX ||
       blockIdx.y>=pending.model_update_binding_count) {
        semantic_content_integrity_trap();return;
    }
    const uint8_t admissibility=
        *reinterpret_cast<const uint8_t*>(pending.model_update_admissibility);
    if(admissibility>1) { semantic_content_integrity_trap();return; }
    if(!admissibility)return;
    const auto& binding=reinterpret_cast<const ModelUpdateBinding*>(pending.model_update_bindings)[blockIdx.y];
    const uint64_t destination_slot=binding.slots[base->header.neural_bank^1];
    if(destination_slot>=control.storage_count ||
       (binding.bytes && (!binding.source || binding.source>UINT64_MAX-binding.bytes))) {
        semantic_content_integrity_trap();return;
    }
    const auto& destination=reinterpret_cast<const PublicationStorageEntry*>(control.storage)[destination_slot];
    if(destination.bytes!=binding.bytes ||
       (binding.bytes && (!destination.pointer || destination.pointer>UINT64_MAX-binding.bytes ||
        destination.pointer==binding.source))) {
        semantic_content_integrity_trap();return;
    }
    auto* output=reinterpret_cast<uint8_t*>(destination.pointer);
    const auto* source=reinterpret_cast<const uint8_t*>(binding.source);
    const uint64_t start=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const uint64_t stride=uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t offset=start;offset<binding.bytes;offset+=stride)output[offset]=source[offset];
}
extern "C" __global__ void semantic_publication_prepare_drain(uint64_t control_ptr,uint64_t lease_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!control_ptr || control_ptr%alignof(PublicationControl) ||
       control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) ||
       lease_ptr>UINT64_MAX-sizeof(PublicationLease)) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    const auto* base=publication_acquired_bank(control,lease);
    if(!base || lease.transition_kind!=3 || !control.contract || !control.continuation ||
       control.contract%alignof(PublicationContract) || control.continuation%alignof(PendingContinuation) ||
       publication_validate_storage(control)) {
        semantic_content_integrity_trap();return;
    }
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    auto& pending=*reinterpret_cast<PendingContinuation*>(control.continuation);
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease.bank]);
    const auto* prefix=publication_find_range(ranges,base->header.range_count,3);
    const auto* prefix_bytes=prefix ? publication_range_bytes(control,*prefix) : nullptr;
    if(contract.abi!=1 || pending.abi!=1 || !pending.ranges ||
       pending.range_count>contract.range_capacity || !prefix || prefix->length_bytes!=32 || !prefix_bytes ||
       publication_validate_selected_model(control,*base) ||
       base->header.authority_generation!=contract.authority_generation) {
        semantic_content_integrity_trap();return;
    }
    pending.base_word=lease.word;
    pending.model_generation=base->header.model_generation;
    pending.authority_generation=base->header.authority_generation;
    pending.transition_kind=3;
    pending.text={};
    pending.numerical_admissibility=0;
    pending.training_selection=0;
    pending.model_update_bindings=0;
    pending.model_update_binding_count=0;
    pending.model_update_admissibility=0;
    semantic_graph::copy_identity(pending.instance,base->header.instance);
    semantic_graph::copy_identity(pending.topology_identity,contract.topology_identity);
    semantic_graph::copy_identity(pending.table_identity,contract.table_identity);
    semantic_graph::copy_identity(pending.prefix_identity,
        reinterpret_cast<const uint64_t*>(prefix_bytes));
}
__device__ void publication_copy_bytes(uint8_t* to,const uint8_t* from,uint64_t count) {
    for(uint64_t i=0;i<count;++i)to[i]=from[i];
}
__device__ uint64_t publication_apply_continuation(const PublicationControl& control,
        const PublicationBank& base,PublicationBank& next,const PendingContinuation& pending) {
    const auto* from=reinterpret_cast<const PublicationRange*>(pending.ranges);
    const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[pending.base_word&1]);
    auto* to=reinterpret_cast<PublicationRange*>(control.directories[(pending.base_word&1)^1]);
    if(publication_validate_model_update(control,base,pending))return 1;
    if(pending.transition_kind==3) {
        const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
        for(uint64_t i=0;i<next.header.range_count;++i) {
            if((old[i].role!=to[i].role || old[i].index!=to[i].index) ||
               old[i].storage_slot>=control.storage_count || to[i].storage_slot>=control.storage_count)return 1;
            if(old[i].role>=18 && old[i].role<=25) {
                to[i]=old[i];
                continue;
            }
            const uint64_t destination_slot=to[i].storage_slot;
            const uint64_t destination_generation=to[i].generation;
            const uint64_t source_slot=old[i].storage_slot;
            to[i]=old[i];to[i].storage_slot=destination_slot;to[i].generation=destination_generation;
            if(destination_slot==source_slot)continue;
            bool copied=false;
            for(uint64_t j=0;j<i;++j)if(to[j].storage_slot==destination_slot) { copied=true;break; }
            if(copied)continue;
            const auto& source=storage[source_slot];
            const auto& destination=storage[destination_slot];
            if(source.bytes!=destination.bytes || (source.bytes && source.pointer==destination.pointer))return 1;
            publication_copy_bytes(reinterpret_cast<uint8_t*>(destination.pointer),
                reinterpret_cast<const uint8_t*>(source.pointer),source.bytes);
        }
        return 0;
    }
    TensorLayoutTableView pending_table{};
    if(!publication_tensor_table(control,from,pending.range_count,&pending_table))return 1;
    for(uint64_t i=0;i<next.header.range_count;++i) {
        if(!publication_mutable_role(old[i].role))to[i]=old[i];
        if(old[i].role==1 || old[i].role==2 || old[i].role==31)to[i]=old[i];
        if((old[i].role==4 || old[i].role==5) && pending.transition_kind!=4) {
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            if(old[i].storage_slot>=control.storage_count || to[i].storage_slot>=control.storage_count ||
               old[i].storage_slot==to[i].storage_slot)return 1;
            const auto& source=storage[old[i].storage_slot];
            const auto& destination=storage[to[i].storage_slot];
            if(source.bytes!=destination.bytes || (source.bytes && source.pointer==destination.pointer))return 1;
            publication_copy_bytes(reinterpret_cast<uint8_t*>(destination.pointer),
                reinterpret_cast<const uint8_t*>(source.pointer),source.bytes);
        }
        if(old[i].role==44) {
            // Ordinary continuations preserve the model schema. The selected
            // record stays read-leased; only this bank's private copy is sealed.
            uint64_t digest[4];
            if(to[i].storage_slot==old[i].storage_slot || publication_range_digest(control,old[i],nullptr,digest) ||
               !publication_identity_equal(digest,old[i].digest))return 1;
            to[i].length_bytes=old[i].length_bytes;
            to[i].logical_begin=old[i].logical_begin;to[i].logical_end=old[i].logical_end;
            auto* destination=publication_range_bytes(control,to[i]);
            const auto* source=publication_range_bytes(control,old[i]);
            if(!destination || !source)return 1;
            publication_copy_bytes(destination,source,old[i].length_bytes);
        }
        if(old[i].role>=18 && old[i].role<=25) {
            if(pending.transition_kind!=4) {
                to[i]=old[i];
                continue;
            }
            const auto* bindings=reinterpret_cast<const ModelUpdateBinding*>(pending.model_update_bindings);
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            bool found=false;
            for(uint64_t item=0;item<pending.model_update_binding_count;++item)
                if(bindings[item].slots[base.header.neural_bank]==old[i].storage_slot) {
                    const uint64_t destination_slot=bindings[item].slots[base.header.neural_bank^1];
                    if(destination_slot>=control.storage_count)return 1;
                    to[i]=old[i];
                    to[i].storage_slot=destination_slot;
                    to[i].generation=storage[destination_slot].generation;
                    found=true;
                    break;
                }
            if(!found)return 1;
        }
    }
    for(uint64_t i=0;i<pending.range_count;++i) {
        const PublicationRange& source=from[i];if(source.role==55)continue;
        auto* destination=const_cast<PublicationRange*>(publication_find_range(to,next.header.range_count,source.role,source.index));
        if(!destination)return 1;
        const uint8_t* bytes=publication_range_bytes(control,source);
        if(source.role==1) {
            destination->length_bytes=next.header.prefix_extent*sizeof(SourceSlot);
            destination->logical_begin=0;destination->logical_end=next.header.prefix_extent;
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            publication_copy_bytes(output+base.header.prefix_extent*sizeof(SourceSlot),bytes,source.length_bytes);
        } else if(source.role==39) {
            destination->length_bytes=source.length_bytes;
            destination->logical_begin=source.logical_begin;destination->logical_end=source.logical_end;
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            publication_copy_bytes(output,bytes,source.length_bytes);
        } else {
            const auto* source_layout=publication_find_layout(control,from,pending.range_count,source.role,source.index);
            const auto* destination_layout=publication_find_layout(control,to,next.header.range_count,source.role,source.index);
            PublicationTensorBytes view{};uint64_t byte_count=0;
            if(publication_tensor_view(control,source,source_layout,&view,&byte_count,&pending_table))return 1;
            if(source.role>=51 && source.role<=54)destination->length_bytes=byte_count ?
                reinterpret_cast<const PublicationStorageEntry*>(control.storage)[destination->storage_slot].bytes-destination->offset_bytes : 0;
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            for(uint64_t cell=0;cell<byte_count/source_layout->element_bytes;++cell) {
                uint64_t rest=cell,offset=0;
                for(int axis=int(source_layout->rank)-1;axis>=0;--axis) {
                    uint64_t coordinate=rest%view.dimensions[axis];rest/=view.dimensions[axis];
                    if((source.role==4 || source.role==5) && pending.transition_kind!=4 &&
                       uint64_t(axis)==source_layout->logical_axis)coordinate+=base.header.prefix_extent;
                    if(source.role>=51 && source.role<=54 && uint64_t(axis)==source_layout->logical_axis)
                        coordinate=pending_table.rows[coordinate].physical_row;
                    offset+=coordinate*destination_layout->strides_bytes[axis];
                }
                publication_copy_bytes(output+offset,bytes+view.offset(cell),source_layout->element_bytes);
            }
            destination->logical_begin=(source.role==4 || source.role==5) ? 0 : source.logical_begin;
            destination->logical_end=source.logical_end;
        }
    }
    TensorLayoutTableView original{};
    if(!publication_tensor_table(control,old,base.header.range_count,&original))return 1;
    auto* destination=const_cast<PublicationRange*>(publication_find_range(to,next.header.range_count,55));
    const auto* original_range=publication_find_range(old,base.header.range_count,55);
    if(!destination || !original_range || destination->storage_slot==original_range->storage_slot)return 1;
    uint64_t layout_bytes=0,row_bytes=0,length=0;
    if(!semantic_graph::checked_mul(original.header->layout_count,sizeof(PublicationTensorLayout),&layout_bytes) ||
       !semantic_graph::checked_mul(pending_table.header->active_row_count,sizeof(ActiveRow),&row_bytes) ||
       !semantic_graph::checked_add(sizeof(TensorLayoutTableHeader),layout_bytes,&length) ||
       !semantic_graph::checked_add(length,row_bytes,&length))return 1;
    destination->length_bytes=length;destination->logical_begin=0;destination->logical_end=0;
    auto* output=publication_range_bytes(control,*destination);if(!output)return 1;
    // The pending layout roster contains only outputs. Preserve the full cold
    // destination roster (and its capacity strides) in the inaccessible bank.
    auto* header=reinterpret_cast<TensorLayoutTableHeader*>(output);
    *header=*original.header;
    header->active_computed=pending_table.header->active_computed;
    header->active_row_count=pending_table.header->active_row_count;
    publication_copy_bytes(output+sizeof(TensorLayoutTableHeader),reinterpret_cast<const uint8_t*>(original.layouts),layout_bytes);
    publication_copy_bytes(output+sizeof(TensorLayoutTableHeader)+layout_bytes,reinterpret_cast<const uint8_t*>(pending_table.rows),row_bytes);
    return 0;
}
__device__ bool publication_receipt_runtime_word(uint64_t word) {
    // Owner receipt 3..12 are root/fork/statement/support/version slot+generation
    // pairs. 34..41 carry the resident handle, owner and acquisition coordinates.
    return (word>=3 && word<=12) || (word>=34 && word<=41);
}
struct PublicationCodebookBytes {
    const uint64_t* words;
    uint8_t prefix[80];uint64_t prefix_bytes;
    __device__ uint8_t operator[](uint64_t index) const {
        if(index<prefix_bytes)return prefix[index];
        index-=prefix_bytes;const uint64_t word=index/8;
        const uint64_t value=(word>=14 && word<=16) || (word>=21 && word<=24) ? 0 : words[word];
        return uint8_t(value>>(8*(index%8)));
    }
};
__device__ uint64_t publication_logical_codebook_digest(const uint64_t* words,uint64_t count,uint64_t* digest) {
    if(!words || count<25 || count>UINT64_MAX/8 || words[8]%8 || words[8]>count*8 ||
       words[9]>count*8-words[8] || (words[8]+words[9]+7)/8!=count)return 1;
    PublicationCodebookBytes bytes{};bytes.words=words;
    const char domain[]="xlog.semantic.logical-codebooks.v1";
    for(uint32_t i=0;i<sizeof(domain);++i)bytes.prefix[bytes.prefix_bytes++]=domain[i];
    for(uint32_t i=0;i<32;++i)bytes.prefix[bytes.prefix_bytes++]=CATALOGUE_DIGEST[i];
    for(uint32_t i=0;i<8;++i)bytes.prefix[bytes.prefix_bytes++]=uint8_t(count>>(8*i));
    if(count>(UINT64_MAX/8-bytes.prefix_bytes)/8)return 1;
    semantic_graph::sha256(bytes,bytes.prefix_bytes+count*8,digest);
    return 0;
}
struct PublicationActionReceiptBytes {
    const Receipt* receipts;
    uint64_t logical_binding[4];
    __device__ uint8_t operator[](uint64_t index) const {
        const uint64_t ordinal=index/sizeof(Receipt),offset=index%sizeof(Receipt);
        if(offset>=48 && offset<80 && receipts[ordinal].catalogue_generation==CATALOGUE_GENERATION)
            return reinterpret_cast<const uint8_t*>(logical_binding)[offset-48];
        return reinterpret_cast<const uint8_t*>(receipts)[index];
    }
};
__device__ uint64_t publication_action_digest(const PublicationControl& control,const PublicationBank& bank,
        const Receipt* receipts,uint64_t count,uint64_t* digest) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* codebook=publication_find_range(ranges,bank.header.range_count,48);
    if(!codebook || codebook->length_bytes%8)return 1;
    PublicationActionReceiptBytes bytes{};bytes.receipts=receipts;
    if(publication_logical_codebook_digest(reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*codebook)),
        codebook->length_bytes/8,bytes.logical_binding))return 1;
    semantic_graph::sha256(bytes,count*sizeof(Receipt),digest);
    return 0;
}
struct PublicationLogicalBytes {
    PublicationTensorBytes view;
    uint64_t role;
    __device__ uint8_t operator[](uint64_t index) const {
        if(index<128)return view[index];
        const uint64_t byte=index-128,word=byte/8;
        // The stable record seal binds the normalized originating receipt;
        // retain it while excluding the separately preserved audit receipt hash.
        if(role==2 && (word%23==3 || (word%23>=6 && word%23<=9)))return 0;
        if(role==14 && ((word>=1 && word<=5) || word>=41))return 0;
        if(role==15 && word%48>=6 && publication_receipt_runtime_word(word%48-6))return 0;
        if(role==33 && ((word>=1 && word<=6) || word>=39))return 0;
        if(role==30 && byte>=sizeof(IntentQueueHeader)) {
            const uint64_t entry_word=(byte-sizeof(IntentQueueHeader))/8%43;
            if(entry_word>=30 && entry_word<=34)return 0;
        }
        return view[index];
    }
};
__device__ uint64_t publication_logical_range_digest(const PublicationControl& control,
        const PublicationRange& range,const PublicationTensorLayout* layout,uint64_t* digest) {
    if(range.role==48) {
        if(range.length_bytes%8)return 1;
        return publication_logical_codebook_digest(reinterpret_cast<const uint64_t*>(publication_range_bytes(control,range)),range.length_bytes/8,digest);
    }
    PublicationLogicalBytes input{};uint64_t bytes=0;input.role=range.role;
    if(publication_tensor_view(control,range,layout,&input.view,&bytes) || bytes>UINT64_MAX/8-128)return 1;
    semantic_graph::sha256(input,128+bytes,digest);
    return 0;
}
__device__ void publication_fold(uint64_t* digest,uint64_t role,uint64_t index,const uint64_t* content) {
    uint64_t words[10];semantic_graph::copy_identity(words,digest);words[4]=role;words[5]=index;
    semantic_graph::copy_identity(words+6,content);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(words),sizeof(words),digest);
}
struct PublicationSemanticReceiptBytes {
    const semantic_graph::Receipt* receipts;
    __device__ uint8_t operator[](uint64_t index) const {
        return publication_receipt_runtime_word(index/8%42) ? 0 : reinterpret_cast<const uint8_t*>(receipts)[index];
    }
};

__device__ bool publication_model_contract_layout(const ModelContractLayout& layout,uint64_t bytes) {
    const uint64_t begin[]={layout.schema_begin,layout.schema_digest_offset,layout.generation_offset,
        layout.numerical_digest_offset,layout.identity_offset};
    const uint64_t length[]={layout.schema_bytes,32,8,32,32};
    for(uint32_t i=0;i<5;++i) {
        if(!length[i] || begin[i]>bytes || length[i]>bytes-begin[i])return false;
        for(uint32_t j=0;j<i;++j)
            if(begin[i]<begin[j]+length[j] && begin[j]<begin[i]+length[i])return false;
    }
    return layout.schema_bytes<=UINT64_MAX/8;
}

__device__ uint64_t publication_finalize_model_contract(const PublicationControl& control,
        const PublicationHeader& header,const PublicationRange& record,bool compare_only) {
    const auto& layout=reinterpret_cast<const PublicationContract*>(control.contract)->model_contract_layout;
    auto* bytes=publication_range_bytes(control,record);
    if(record.role!=44 || record.index || record.logical_begin || record.logical_end || !bytes ||
       !publication_model_contract_layout(layout,record.length_bytes))return 1;
    uint64_t schema_digest[4],identity[4];uint8_t generation[8];
    semantic_graph::sha256(bytes+layout.schema_begin,layout.schema_bytes,schema_digest);
    for(uint32_t i=0;i<8;++i)generation[i]=uint8_t(header.model_generation>>(8*i));
    const char domain[]="xlog.semantic.model-identity.v1";
    uint8_t input[sizeof(domain)+32+8+32];
    publication_copy_bytes(input,reinterpret_cast<const uint8_t*>(domain),sizeof(domain));
    publication_copy_bytes(input+sizeof(domain),reinterpret_cast<const uint8_t*>(schema_digest),32);
    publication_copy_bytes(input+sizeof(domain)+32,generation,8);
    publication_copy_bytes(input+sizeof(domain)+40,reinterpret_cast<const uint8_t*>(header.model_numerical_digest),32);
    semantic_graph::sha256(input,sizeof(input),identity);
    const uint64_t offsets[]={layout.schema_digest_offset,layout.generation_offset,layout.numerical_digest_offset,layout.identity_offset};
    const uint64_t lengths[]={32,8,32,32};
    const uint8_t* values[]={reinterpret_cast<const uint8_t*>(schema_digest),generation,
        reinterpret_cast<const uint8_t*>(header.model_numerical_digest),reinterpret_cast<const uint8_t*>(identity)};
    // All bounds and overlaps were checked before the first write. Producer
    // offsets need not be aligned. Restore never rewrites a saved baseline.
    for(uint32_t field=0;field<4;++field) {
        if(compare_only) {
            for(uint64_t i=0;i<lengths[field];++i)if(bytes[offsets[field]+i]!=values[field][i])return 1;
        } else publication_copy_bytes(bytes+offsets[field],values[field],lengths[field]);
    }
    return 0;
}

__device__ uint64_t publication_seal_ranges(const PublicationControl& control,PublicationBank& bank,
        const PublicationBank* previous,bool restored=false) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,bank.header.range_count,&table))return 1;
    const auto* old=previous ? reinterpret_cast<const PublicationRange*>(control.directories[previous->header.publication_word&1]) : nullptr;
    // The cold owner derives geometry from the complete, retained producer
    // storage/view map. Only actual device range and full-backing seals enter
    // this numerical fold; ModelContract remains in the enclosing parent seal.
    const char numerical_domain[]="xlog.semantic.model-numerics.v1";
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(numerical_domain),sizeof(numerical_domain),
        bank.header.model_numerical_digest);
    publication_fold(bank.header.model_numerical_digest,0,0,bank.header.model_geometry_digest);
    for(uint64_t i=0;i<bank.header.range_count;++i) {
        // Record 44 includes N, so its first seal must follow the complete
        // numerical fold, not participate in that fold or be sealed early.
        if(ranges[i].role==44)continue;
        if(old && !publication_mutable_role(ranges[i].role)) {
            // The native-owned immutable storage generation remains read-leased.
            // This seal was computed at initialization, not accepted from input.
            ranges[i]=old[i];
        } else if(publication_range_digest(control,ranges[i],
            publication_find_layout(control,ranges,bank.header.range_count,ranges[i].role,ranges[i].index),ranges[i].digest,&table))return 1;
        if(ranges[i].role>=18 && ranges[i].role<=25) {
            bool copied=false;
            for(uint64_t j=0;j<i;++j)if(ranges[j].storage_slot==ranges[i].storage_slot) {
                if(ranges[j].role<18 || ranges[j].role>25)return 1;
                semantic_graph::copy_identity(ranges[i].backing_digest,ranges[j].backing_digest);
                copied=true;break;
            }
            if(!copied && publication_backing_digest(control,ranges[i],ranges[i].backing_digest))return 1;
            publication_fold(bank.header.model_numerical_digest,ranges[i].role,ranges[i].index,ranges[i].digest);
            publication_fold(bank.header.model_numerical_digest,ranges[i].role,ranges[i].index,ranges[i].backing_digest);
        } else for(uint32_t word=0;word<4;++word)ranges[i].backing_digest[word]=0;
    }
    auto* model=const_cast<PublicationRange*>(publication_find_range(ranges,bank.header.range_count,44));
    if(!model || publication_finalize_model_contract(control,bank.header,*model,restored))return 1;
    for(uint32_t word=0;word<4;++word)model->backing_digest[word]=0;
    return publication_range_digest(control,*model,nullptr,model->digest,&table);
}
__device__ uint64_t publication_logical_digest(const PublicationControl& control,PublicationBank& bank) {
    PublicationHeader logical=bank.header;
    // Cold initialization hashes the sealed representation before exposing ABI 1.
    logical.abi=1;
    for(uint32_t i=0;i<4;++i)logical.instance[i]=logical.recovered_instance[i]=0;
    logical.sealed_epoch=logical.base_word=logical.publication_word=0;
    logical.semantic_owner=logical.semantic_slot=logical.semantic_generation=logical.neural_bank=0;
    for(uint32_t i=0;i<4;++i)logical.logical_digest[i]=logical.state_digest[i]=logical.descriptor_digest[i]=0;
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&logical),sizeof(logical),bank.header.logical_digest);
    uint64_t digest[4];
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(bank.source),sizeof(bank.source),digest);
    publication_fold(bank.header.logical_digest,0,1,digest);
    if(publication_action_digest(control,bank,bank.receipts,136,digest))return 1;
    publication_fold(bank.header.logical_digest,0,2,digest);
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    for(uint64_t i=0;i<bank.header.range_count;++i) {
        const auto& range=ranges[i];
        if(range.role==27 || range.role==28 || range.role==30 || range.role==31 || range.role==32 || range.role==33)continue;
        if(range.role==2 || range.role==14 || range.role==15 || range.role==48) {
            if(publication_logical_range_digest(control,range,nullptr,digest))return 1;
        } else semantic_graph::copy_identity(digest,range.digest);
        publication_fold(bank.header.logical_digest,range.role,range.index,digest);
        if(range.role>=18 && range.role<=25)
            publication_fold(bank.header.logical_digest,range.role,range.index,range.backing_digest);
    }
    return 0;
}
struct PublicationSealedBankBytes {
    const PublicationBank* bank;
    __device__ uint8_t operator[](uint64_t index) const {
        if(index<sizeof(uint64_t))return index==0 ? 1 : 0;
        const uint64_t seal_offset=uint64_t(reinterpret_cast<const uint8_t*>(bank->header.descriptor_digest)-
            reinterpret_cast<const uint8_t*>(bank));
        if(index>=seal_offset && index<seal_offset+sizeof(bank->header.descriptor_digest))return 0;
        return reinterpret_cast<const uint8_t*>(bank)[index];
    }
};
__device__ void publication_original_descriptor_digest(const PublicationControl& control,
        const PublicationBank& bank,uint64_t* output) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    semantic_graph::sha256(PublicationSealedBankBytes{&bank},sizeof(bank),output);
    for(uint64_t i=0;i<bank.header.range_count;++i) {
        uint64_t digest[4];
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&ranges[i]),sizeof(PublicationRange),digest);
        publication_fold(output,ranges[i].role,ranges[i].index,digest);
    }
}
__device__ uint64_t publication_validate_selected_model(const PublicationControl& control,
        const PublicationBank& bank) {
    if(!control.contract || control.contract%alignof(PublicationContract) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract))return 1;
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& header=bank.header;
    const uint64_t directory=control.directories[header.publication_word&1];
    if(contract.abi!=1 || !contract.model_generation || header.model_generation<contract.model_generation ||
       header.model_generation>UINT32_MAX || header.range_count>contract.range_capacity ||
       !directory || directory%alignof(PublicationRange) ||
       header.range_count>(UINT64_MAX-directory)/sizeof(PublicationRange) ||
       !publication_bank_matches(control,bank,header.publication_word))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(directory);
    uint64_t digest[4];publication_original_descriptor_digest(control,bank,digest);
    if(!publication_identity_equal(digest,header.descriptor_digest))return 1;
    const auto* model=publication_find_range(ranges,header.range_count,44);
    if(!model || publication_range_digest(control,*model,nullptr,digest) ||
       !publication_identity_equal(digest,model->digest) ||
       publication_finalize_model_contract(control,header,*model,true))return 1;
    // Fold the original typed and full-backing seals in their canonical order.
    // Individual consumers still verify actual payloads against those seals;
    // neither admission nor this check reseals a supplied consumer tensor.
    const char domain[]="xlog.semantic.model-numerics.v1";
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(domain),sizeof(domain),digest);
    publication_fold(digest,0,0,header.model_geometry_digest);
    for(uint64_t i=0;i<header.range_count;++i)if(ranges[i].role>=18 && ranges[i].role<=25) {
        publication_fold(digest,ranges[i].role,ranges[i].index,ranges[i].digest);
        publication_fold(digest,ranges[i].role,ranges[i].index,ranges[i].backing_digest);
    }
    return publication_identity_equal(digest,header.model_numerical_digest) ? 0 : 1;
}
__device__ uint64_t publication_descriptor_digest(const PublicationControl& control,PublicationBank& bank) {
    semantic_graph::copy_identity(bank.header.state_digest,bank.header.logical_digest);
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    uint64_t digest[4];
    for(uint64_t i=0;i<bank.header.range_count;++i) {
        if(ranges[i].role!=27 && ranges[i].role!=28 && ranges[i].role!=30 && ranges[i].role!=31 &&
           ranges[i].role!=32 && ranges[i].role!=33)continue;
        if(publication_logical_range_digest(control,ranges[i],nullptr,digest))return 1;
        publication_fold(bank.header.state_digest,ranges[i].role,ranges[i].index,digest);
    }
    publication_original_descriptor_digest(control,bank,bank.header.descriptor_digest);
    return 0;
}

__device__ bool publication_pointer_span(uint64_t pointer,uint64_t bytes,uint64_t alignment=1) {
    return pointer && !(pointer%alignment) && bytes && pointer<=UINT64_MAX-bytes;
}
__device__ bool publication_spans_overlap(uint64_t a,uint64_t a_bytes,uint64_t b,uint64_t b_bytes) {
    return a<b+b_bytes && b<a+a_bytes;
}
__device__ bool publication_step_output_span(const PublicationControl& control,uint64_t lease,
        uint64_t bindings,uint64_t binding_bytes,uint64_t pointer,uint64_t bytes) {
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(!publication_pointer_span(pointer,bytes) ||
       publication_spans_overlap(pointer,bytes,reinterpret_cast<uint64_t>(&control),sizeof(control)) ||
       publication_spans_overlap(pointer,bytes,lease,sizeof(PublicationLease)) ||
       publication_spans_overlap(pointer,bytes,bindings,binding_bytes) ||
       publication_spans_overlap(pointer,bytes,control.contract,sizeof(contract)) ||
       publication_spans_overlap(pointer,bytes,contract.role_counts,55*sizeof(PublicationRoleCount)) ||
       publication_spans_overlap(pointer,bytes,control.storage,control.storage_count*sizeof(PublicationStorageEntry)))return false;
    for(uint64_t bank=0;bank<2;++bank)
        if(publication_spans_overlap(pointer,bytes,control.banks[bank],sizeof(PublicationBank)) ||
           publication_spans_overlap(pointer,bytes,control.directories[bank],contract.range_capacity*sizeof(PublicationRange)))return false;
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    for(uint64_t i=0;i<control.storage_count;++i)
        if(publication_spans_overlap(pointer,bytes,storage[i].pointer,storage[i].bytes))return false;
    return true;
}
__device__ bool publication_step_layout_capacity(const PublicationTensorLayout& layout,uint64_t* capacity) {
    if(layout.rank>4 || !layout.element_bytes)return false;
    *capacity=layout.element_bytes;
    for(int axis=int(layout.rank)-1;axis>=0;--axis)
        if(!layout.dimensions[axis] || layout.strides_bytes[axis]!=*capacity ||
           !semantic_graph::checked_mul(*capacity,layout.dimensions[axis],capacity))return false;
    for(uint64_t axis=layout.rank;axis<4;++axis)
        if(layout.dimensions[axis] || layout.strides_bytes[axis])return false;
    return true;
}

__device__ uint64_t publication_step_metadata_digest(const PublicationHeader& header,const SourceSlot* source,
        uint64_t binding_count,uint64_t output[2][4]) {
    for(uint64_t index=0;index<2;++index) {
        PublicationRange range{};range.index=index;
        range.length_bytes=index ? 32*sizeof(SourceSlot) : sizeof(PublicationHeader);
        PublicationTensorLayout layout{};layout.index=index;layout.element_bytes=8;layout.scalar_type=3;
        layout.logical_axis=UINT64_MAX;layout.rank=index ? 2 : 1;
        layout.dimensions[0]=index ? 32 : sizeof(PublicationHeader)/8;
        layout.dimensions[1]=index ? 8 : 0;
        layout.strides_bytes[0]=index ? sizeof(SourceSlot) : 8;
        layout.strides_bytes[1]=index ? 8 : 0;
        const auto* bytes=index ? reinterpret_cast<const uint8_t*>(source) : reinterpret_cast<const uint8_t*>(&header);
        PublicationTensorBytes view{};uint64_t content_bytes=0;
        if(publication_content_view(bytes,range.length_bytes,range,&layout,&view,&content_bytes) ||
           publication_content_digest(view,content_bytes,output[index]))return 1;
    }
    const uint64_t roster_extent[4]={binding_count,0,0,0};
    publication_fold(output[0],0,0,roster_extent);
    return 0;
}

// One admission selects one actual parent and one mode for the whole captured
// body. Refusal clears the IF handle before acquiring any publication reader.
extern "C" __global__ void semantic_publication_step_admit(uint64_t control_ptr,uint64_t lease_ptr,
        uint64_t requested_kind,uint64_t conditional_handle,uint64_t previous_result_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!conditional_handle) { semantic_content_integrity_trap();return; }
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),0);
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       publication_spans_overlap(control_ptr,sizeof(PublicationControl),lease_ptr,sizeof(PublicationLease))) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    auto& lease=*reinterpret_cast<PublicationLease*>(lease_ptr);
    uint64_t expected_word=UINT64_MAX;
    if(previous_result_ptr) {
        if(!publication_pointer_span(previous_result_ptr,sizeof(PublicationStepResult),alignof(PublicationStepResult))) {
            semantic_content_integrity_trap();return;
        }
        const auto& previous=*reinterpret_cast<const PublicationStepResult*>(previous_result_ptr);
        if(previous.abi>1 || previous.advanced>1 || (!previous.abi && previous.advanced)) {
            semantic_content_integrity_trap();return;
        }
        // Status 7 means the preceding slot did not publish. Do not repeat its
        // unchanged RNG invocation in the remaining bounded slots.
        if(!previous.advanced) { lease.status=control.refusal=7;return; }
        if(previous.refusal || previous.header.abi!=1 || previous.header.publication_word!=previous.word ||
           !publication_identity_equal(previous.header.instance,control.instance)) {
            semantic_content_integrity_trap();return;
        }
        expected_word=previous.word;
    }
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank))) {
            semantic_content_integrity_trap();return;
        }
    const uint64_t status=requested_kind>=1 && requested_kind<=4 ? publication_acquire(control,lease,requested_kind,expected_word) : 1;
    lease.status=status;control.refusal=status;
    if(!status)cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),1);
}

extern "C" __global__ void semantic_publication_step_kind_gate(uint64_t lease_ptr,
        uint64_t expected_kind,uint64_t conditional_handle) {
    if(blockIdx.x || threadIdx.x)return;
    if(!conditional_handle) { semantic_content_integrity_trap();return; }
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),0);
    if(!publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease))) {
        semantic_content_integrity_trap();return;
    }
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.abi!=1 || lease.status || expected_kind<1 || expected_kind>4)return;
    if(lease.active==1 && lease.transition_kind==expected_kind)
        cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),1);
}

extern "C" __global__ void semantic_publication_step_kind_bank_gate(uint64_t lease_ptr,
        uint64_t expected_kind,uint64_t expected_bank,uint64_t conditional_handle) {
    if(blockIdx.x || threadIdx.x)return;
    if(!conditional_handle) { semantic_content_integrity_trap();return; }
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),0);
    if(!publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease))) {
        semantic_content_integrity_trap();return;
    }
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.abi!=1 || lease.status || expected_kind<1 || expected_kind>4 || expected_bank>1)return;
    if(lease.active==1 && lease.transition_kind==expected_kind && lease.bank==expected_bank)
        cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),1);
}

extern "C" __global__ void semantic_publication_step_active_gate(uint64_t lease_ptr,
        uint64_t conditional_handle) {
    if(blockIdx.x || threadIdx.x)return;
    if(!conditional_handle) { semantic_content_integrity_trap();return; }
    cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),0);
    if(!publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease))) {
        semantic_content_integrity_trap();return;
    }
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.abi==1 && lease.active==1)
        cudaGraphSetConditional(static_cast<cudaGraphConditionalHandle>(conditional_handle),1);
}

// Preserve the actual result before release permits either physical bank to be
// reused. Ordinary refusals retain the original parent and its refusal status.
extern "C" __global__ void semantic_publication_step_result(uint64_t control_ptr,uint64_t lease_ptr,
        uint64_t result_ptr,uint64_t parent_header_ptr,uint64_t training_origin_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !publication_pointer_span(result_ptr,sizeof(PublicationStepResult),alignof(PublicationStepResult)) ||
       (training_origin_ptr &&
        !publication_pointer_span(parent_header_ptr,sizeof(PublicationHeader),alignof(PublicationHeader)))) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    uint64_t storage_bytes=0,directory_bytes=0;
    if(!publication_pointer_span(control.contract,sizeof(PublicationContract),alignof(PublicationContract)) ||
       !semantic_graph::checked_mul(control.storage_count,sizeof(PublicationStorageEntry),&storage_bytes) ||
       !publication_pointer_span(control.storage,storage_bytes,alignof(PublicationStorageEntry))) {
        semantic_content_integrity_trap();return;
    }
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(contract.abi!=1 || contract.role_count!=55 ||
       !publication_pointer_span(contract.role_counts,55*sizeof(PublicationRoleCount),alignof(PublicationRoleCount)) ||
       !semantic_graph::checked_mul(contract.range_capacity,sizeof(PublicationRange),&directory_bytes)) {
        semantic_content_integrity_trap();return;
    }
    if(training_origin_ptr &&
       (!publication_step_output_span(control,lease_ptr,parent_header_ptr,sizeof(PublicationHeader),
            training_origin_ptr,sizeof(SemanticTrainingViewOriginRecord)) ||
        publication_spans_overlap(result_ptr,sizeof(PublicationStepResult),training_origin_ptr,
            sizeof(SemanticTrainingViewOriginRecord)))) {
        semantic_content_integrity_trap();return;
    }
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank)) ||
           !publication_pointer_span(control.directories[bank],directory_bytes,alignof(PublicationRange))) {
            semantic_content_integrity_trap();return;
        }
    if(!publication_acquired_bank(control,lease) || publication_load(control.reader_gate) ||
       publication_validate_storage(control) ||
       !publication_step_output_span(control,lease_ptr,0,0,result_ptr,sizeof(PublicationStepResult))) {
        semantic_content_integrity_trap();return;
    }
    if(training_origin_ptr &&
       !publication_header_equal(*reinterpret_cast<const PublicationHeader*>(parent_header_ptr),
           publication_acquired_bank(control,lease)->header)) {
        semantic_content_integrity_trap();return;
    }
    const uint64_t word=publication_load(control.word);
    const auto& bank=*publication_bank(control,word);
    if(!publication_bank_matches(control,bank,word) || (word!=lease.word &&
       ((lease.word>>1)==(UINT64_MAX>>1) || word!=(((lease.word>>1)+1)<<1 | ((lease.word^1)&1)) ||
        bank.header.base_word!=lease.word))) { semantic_content_integrity_trap();return; }
    const uint64_t advanced=uint64_t(word!=lease.word && !control.refusal);
    *reinterpret_cast<PublicationStepResult*>(result_ptr)={1,word,control.refusal,bank.header,advanced};
    if(training_origin_ptr) {
        auto& origin=*reinterpret_cast<SemanticTrainingViewOriginRecord*>(training_origin_ptr);
        origin={};
        if(advanced) {
            const auto& parent=*reinterpret_cast<const PublicationHeader*>(parent_header_ptr);
            origin.present=1;
            origin.transition=lease.transition_kind;
            uint64_t recovered=0;
            for(uint32_t i=0;i<4;++i)recovered|=parent.recovered_instance[i];
            origin.predecessor_word=parent.publication_word;
            origin.successor_word=bank.header.publication_word;
            origin.model_generation=parent.model_generation;
            origin.stream_serial=parent.stream_serial;
            origin.family_id=parent.family_id;
            origin.proposal=parent.proposal;
            for(uint32_t i=0;i<4;++i) {
                origin.lineage_instance[i]=recovered ? parent.recovered_instance[i] : parent.instance[i];
                origin.predecessor_instance[i]=parent.instance[i];
                origin.predecessor_logical[i]=parent.logical_digest[i];
                origin.predecessor_state[i]=parent.state_digest[i];
                origin.successor_instance[i]=bank.header.instance[i];
                origin.successor_logical[i]=bank.header.logical_digest[i];
                origin.successor_state[i]=bank.header.state_digest[i];
                origin.model_geometry_digest[i]=parent.model_geometry_digest[i];
                origin.model_numerical_digest[i]=parent.model_numerical_digest[i];
            }
        }
    }
}

// The graph places release after every admitted consumer. Failure traps instead
// of silently retrying or abandoning an acquired reader; mode remains retained.
extern "C" __global__ void semantic_publication_step_release(uint64_t control_ptr,uint64_t lease_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease))) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank))) {
            semantic_content_integrity_trap();return;
        }
    if(publication_release(control,*reinterpret_cast<PublicationLease*>(lease_ptr)))semantic_content_integrity_trap();
}

// Retain original model range seals and raw contract bytes through the actual
// reader. The cold roster contains role/index pairs in the model's binding order;
// output ranges preserve that order and append the original contract range.
extern "C" __global__ void semantic_publication_model_seals(uint64_t control_ptr,uint64_t lease_ptr,
        uint64_t roster_ptr,uint64_t count,uint64_t output_ranges_ptr,uint64_t output_contract_ptr,
        uint64_t contract_capacity) {
    if(blockIdx.x || threadIdx.x)return;
    uint64_t roster_bytes=0,output_count=0,output_bytes=0;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !semantic_graph::checked_mul(count,2*sizeof(uint64_t),&roster_bytes) ||
       (count && !publication_pointer_span(roster_ptr,roster_bytes,alignof(uint64_t))) ||
       !semantic_graph::checked_add(count,1,&output_count) ||
       !semantic_graph::checked_mul(output_count,sizeof(PublicationRange),&output_bytes) ||
       !publication_pointer_span(output_ranges_ptr,output_bytes,alignof(PublicationRange)) ||
       !publication_pointer_span(output_contract_ptr,contract_capacity)) {
        semantic_content_integrity_trap();return;
    }
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    uint64_t storage_bytes=0,directory_bytes=0;
    if(lease.bank>1 || !publication_pointer_span(control.contract,sizeof(PublicationContract),alignof(PublicationContract)) ||
       !semantic_graph::checked_mul(control.storage_count,sizeof(PublicationStorageEntry),&storage_bytes) ||
       !publication_pointer_span(control.storage,storage_bytes,alignof(PublicationStorageEntry))) {
        semantic_content_integrity_trap();return;
    }
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(contract.abi!=1 || contract.role_count!=55 ||
       !publication_pointer_span(contract.role_counts,55*sizeof(PublicationRoleCount),alignof(PublicationRoleCount)) ||
       !semantic_graph::checked_mul(contract.range_capacity,sizeof(PublicationRange),&directory_bytes)) {
        semantic_content_integrity_trap();return;
    }
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank)) ||
           !publication_pointer_span(control.directories[bank],directory_bytes,alignof(PublicationRange))) {
            semantic_content_integrity_trap();return;
        }
    const auto* bank=publication_acquired_bank(control,lease);
    if(!bank || bank->header.range_count>contract.range_capacity || count>bank->header.range_count ||
       publication_validate_storage(control) ||
       !publication_step_output_span(control,lease_ptr,roster_ptr,roster_bytes,output_ranges_ptr,output_bytes) ||
       !publication_step_output_span(control,lease_ptr,roster_ptr,roster_bytes,output_contract_ptr,contract_capacity) ||
       publication_spans_overlap(output_ranges_ptr,output_bytes,output_contract_ptr,contract_capacity)) {
        semantic_content_integrity_trap();return;
    }
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[lease.bank]);
    uint64_t digest[4];
    if(publication_validate_selected_model(control,*bank)) { semantic_content_integrity_trap();return; }
    const auto* table=publication_find_range(ranges,bank->header.range_count,55);
    const auto* original_contract=publication_find_range(ranges,bank->header.range_count,44);
    if(!table || publication_range_digest(control,*table,nullptr,digest) ||
       !publication_identity_equal(digest,table->digest) ||
       publication_validate_directory(control,contract,ranges,bank->header.range_count,false) ||
       !original_contract || original_contract->length_bytes!=contract_capacity ||
       publication_range_digest(control,*original_contract,nullptr,digest) ||
       !publication_identity_equal(digest,original_contract->digest)) {
        semantic_content_integrity_trap();return;
    }
    const auto* roster=reinterpret_cast<const uint64_t*>(roster_ptr);
    for(uint64_t i=0;i<count;++i) {
        const uint64_t role=roster[2*i],index=roster[2*i+1];
        if(role<18 || role>20) { semantic_content_integrity_trap();return; }
        const auto* range=publication_find_range(ranges,bank->header.range_count,role,index);
        const auto* layout=publication_find_layout(control,ranges,bank->header.range_count,role,index);
        if(!range || !layout || publication_range_digest(control,*range,layout,digest) ||
           !publication_identity_equal(digest,range->digest)) { semantic_content_integrity_trap();return; }
    }
    auto* output=reinterpret_cast<PublicationRange*>(output_ranges_ptr);
    for(uint64_t i=0;i<count;++i)
        output[i]=*publication_find_range(ranges,bank->header.range_count,roster[2*i],roster[2*i+1]);
    output[count]=*original_contract;
    publication_copy_bytes(reinterpret_cast<uint8_t*>(output_contract_ptr),publication_range_bytes(control,*original_contract),contract_capacity);
}

// Capture binds both resident input banks once. Every execution resolves the
// actual acquired parent and validates all originals before mutating any private
// output. Prefix, attention caches and model tensors remain direct device aliases;
// Source, feedback records, header and seals are retained private snapshots.
extern "C" __global__ void semantic_publication_step_inputs(uint64_t control_ptr,uint64_t lease_ptr,
        uint64_t bindings0_ptr,uint64_t bindings1_ptr,uint64_t binding_count,uint64_t header_output_ptr,
        uint64_t source_output_ptr,uint64_t range_output_ptr,uint64_t metadata_digests_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    uint64_t binding_bytes=0,range_bytes=0;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationStepInput),&binding_bytes) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationRange),&range_bytes) ||
       !publication_pointer_span(bindings0_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(bindings1_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(header_output_ptr,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
       !publication_pointer_span(source_output_ptr,32*sizeof(SourceSlot),alignof(SourceSlot)) ||
       !publication_pointer_span(range_output_ptr,range_bytes,alignof(PublicationRange)) ||
       !publication_pointer_span(metadata_digests_ptr,8*sizeof(uint64_t),alignof(uint64_t))) {
        semantic_content_integrity_trap();return;
    }
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    const uint64_t bindings_ptr=lease.bank ? bindings1_ptr : bindings0_ptr;
    uint64_t storage_bytes=0;
    if(lease.bank>1 || !publication_pointer_span(control.contract,sizeof(PublicationContract),alignof(PublicationContract)) ||
       !semantic_graph::checked_mul(control.storage_count,sizeof(PublicationStorageEntry),&storage_bytes) ||
       !publication_pointer_span(control.storage,storage_bytes,alignof(PublicationStorageEntry))) {
        semantic_content_integrity_trap();return;
    }
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    uint64_t directory_bytes=0;
    if(contract.abi!=1 || contract.role_count!=55 ||
       !publication_pointer_span(contract.role_counts,55*sizeof(PublicationRoleCount),alignof(PublicationRoleCount)) ||
       !semantic_graph::checked_mul(contract.range_capacity,sizeof(PublicationRange),&directory_bytes)) {
        semantic_content_integrity_trap();return;
    }
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank)) ||
           !publication_pointer_span(control.directories[bank],directory_bytes,alignof(PublicationRange))) {
            semantic_content_integrity_trap();return;
        }
    const auto* bank=publication_acquired_bank(control,lease);
    if(!bank || bank->header.range_count>contract.range_capacity || publication_validate_storage(control)) {
        semantic_content_integrity_trap();return;
    }
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[lease.bank]);
    const uint64_t count=bank->header.range_count;
    uint64_t digest[4];
    if(publication_validate_selected_model(control,*bank)) { semantic_content_integrity_trap();return; }
    const auto* table_range=publication_find_range(ranges,count,55);
    if(!table_range || publication_range_digest(control,*table_range,nullptr,digest) ||
       !publication_identity_equal(digest,table_range->digest) || publication_validate_directory(control,contract,ranges,count,false)) {
        semantic_content_integrity_trap();return;
    }
    const auto* roles=reinterpret_cast<const PublicationRoleCount*>(contract.role_counts);
    uint64_t expected=3;
    for(uint64_t role=4;role<=25;++role)
        if(role!=14 && !semantic_graph::checked_add(expected,roles[role-1].count,&expected)) {
            semantic_content_integrity_trap();return;
        }
    if(roles[43].count!=1 || binding_count!=expected) { semantic_content_integrity_trap();return; }
    const uint64_t output_pointers[]={header_output_ptr,source_output_ptr,range_output_ptr,metadata_digests_ptr};
    const uint64_t output_bytes[]={sizeof(PublicationHeader),32*sizeof(SourceSlot),range_bytes,8*sizeof(uint64_t)};
    for(uint64_t i=0;i<4;++i) {
        if(!publication_step_output_span(control,lease_ptr,bindings0_ptr,binding_bytes,output_pointers[i],output_bytes[i]) ||
           !publication_step_output_span(control,lease_ptr,bindings1_ptr,binding_bytes,output_pointers[i],output_bytes[i])) {
            semantic_content_integrity_trap();return;
        }
        for(uint64_t j=0;j<i;++j)
            if(publication_spans_overlap(output_pointers[i],output_bytes[i],output_pointers[j],output_bytes[j])) {
                semantic_content_integrity_trap();return;
            }
    }
    const auto* bindings=reinterpret_cast<const PublicationStepInput*>(bindings_ptr);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t cursor=0;
    for(uint64_t role=1;role<=44;++role) {
        if(role==2 || role==14 || (role>25 && role!=44))continue;
        const bool model=role>=18 && role<=25;
        for(uint64_t index=0;index<roles[role-1].count;++index) {
            const auto& binding=bindings[cursor++];
            const auto* range=publication_find_range(ranges,count,role,index);
            const auto* layout=publication_find_layout(control,ranges,count,role,index);
            if(!range || binding.role!=role || binding.index!=index ||
               ((!model || binding.capacity_bytes) && !publication_pointer_span(binding.destination,binding.capacity_bytes)) ||
               publication_range_digest(control,*range,layout,digest) || !publication_identity_equal(digest,range->digest)) {
                semantic_content_integrity_trap();return;
            }
            const auto* original=publication_range_bytes(control,*range);
            const bool alias=role==1 || role==4 || role==5 || model;
            if(model) {
                uint64_t destination=0;
                if(binding.backing!=storage[range->storage_slot].pointer ||
                   binding.backing_bytes!=storage[range->storage_slot].bytes ||
                   !semantic_graph::checked_add(binding.backing,range->offset_bytes,&destination) ||
                   binding.destination!=destination ||
                   (binding.backing_bytes && !publication_pointer_span(binding.backing,binding.backing_bytes))) {
                    semantic_content_integrity_trap();return;
                }
                bool sealed=false;
                for(uint64_t i=0;i+1<cursor;++i) {
                    const auto* previous=publication_find_range(ranges,count,bindings[i].role,bindings[i].index);
                    if(previous->storage_slot==range->storage_slot) {
                        if(bindings[i].backing!=binding.backing || bindings[i].backing_bytes!=binding.backing_bytes ||
                           !publication_identity_equal(previous->backing_digest,range->backing_digest)) {
                            semantic_content_integrity_trap();return;
                        }
                        sealed=true;break;
                    }
                }
                if(!sealed && (publication_backing_digest(control,*range,digest) ||
                   !publication_identity_equal(digest,range->backing_digest))) { semantic_content_integrity_trap();return; }
            } else if(binding.backing!=binding.destination || binding.backing_bytes!=binding.capacity_bytes) {
                semantic_content_integrity_trap();return;
            }
            if(role==3 || (role>=15 && role<=17) || role==44) {
                const auto* words=reinterpret_cast<const uint64_t*>(&binding.layout);
                for(uint64_t word=0;word<sizeof(binding.layout)/8;++word)
                    if(words[word]) { semantic_content_integrity_trap();return; }
                if(range->logical_begin || (role!=15 && range->logical_end) || binding.capacity_bytes!=range->length_bytes ||
                   (role==15 && (range->length_bytes%sizeof(RawFeedbackRecord) ||
                       range->logical_end>range->length_bytes/sizeof(RawFeedbackRecord))) ||
                   (role==3 && (range->length_bytes!=32 || binding.destination%alignof(uint64_t))) ||
                   (role==44 && !publication_model_contract_layout(contract.model_contract_layout,range->length_bytes))) {
                    semantic_content_integrity_trap();return;
                }
            } else {
                PublicationTensorLayout original_layout{};
                if(role==1) {
                    original_layout.role=1;original_layout.element_bytes=8;original_layout.scalar_type=3;
                    original_layout.rank=2;original_layout.logical_axis=0;
                    original_layout.dimensions[0]=contract.prefix_capacity;original_layout.dimensions[1]=8;
                    original_layout.strides_bytes[0]=sizeof(SourceSlot);original_layout.strides_bytes[1]=8;
                } else original_layout=*layout;
                const auto* actual=reinterpret_cast<const uint64_t*>(&binding.layout);
                const auto* sealed=reinterpret_cast<const uint64_t*>(&original_layout);
                for(uint64_t word=0;word<sizeof(original_layout)/8;++word)
                    if(actual[word]!=sealed[word]) { semantic_content_integrity_trap();return; }
                uint64_t capacity=0;
                if(model)capacity=range->length_bytes;
                if((!model && !publication_step_layout_capacity(original_layout,&capacity)) || binding.capacity_bytes!=capacity ||
                   binding.destination%original_layout.element_bytes ||
                   capacity>storage[range->storage_slot].bytes-range->offset_bytes ||
                   (!alias && (range->length_bytes!=capacity || range->logical_begin || range->logical_end))) {
                    semantic_content_integrity_trap();return;
                }
                if(alias && (binding.destination!=reinterpret_cast<uint64_t>(original) || range->logical_begin ||
                   range->logical_end!=bank->header.prefix_extent ||
                   (role==1 && range->length_bytes!=bank->header.prefix_extent*sizeof(SourceSlot)) ||
                   (role!=1 && (range->length_bytes!=capacity || original_layout.dimensions[2]!=contract.prefix_capacity)))) {
                    semantic_content_integrity_trap();return;
                }
            }
            if(alias)continue;
            if(binding.backing_bytes &&
               (!publication_step_output_span(control,lease_ptr,bindings0_ptr,binding_bytes,binding.backing,binding.backing_bytes) ||
                !publication_step_output_span(control,lease_ptr,bindings1_ptr,binding_bytes,binding.backing,binding.backing_bytes))) {
                semantic_content_integrity_trap();return;
            }
            for(uint64_t i=0;i<4;++i)
                if(publication_spans_overlap(binding.backing,binding.backing_bytes,output_pointers[i],output_bytes[i])) {
                    semantic_content_integrity_trap();return;
                }
            for(uint64_t i=0;i+1<cursor;++i) {
                const auto* previous=publication_find_range(ranges,count,bindings[i].role,bindings[i].index);
                const bool same_model=model && bindings[i].role>=18 && previous->storage_slot==range->storage_slot;
                if(!same_model && publication_spans_overlap(binding.backing,binding.backing_bytes,bindings[i].backing,bindings[i].backing_bytes)) {
                    semantic_content_integrity_trap();return;
                }
            }
        }
    }
    uint64_t metadata[2][4];
    if(publication_step_metadata_digest(bank->header,bank->source,binding_count,metadata)) { semantic_content_integrity_trap();return; }
    publication_copy_bytes(reinterpret_cast<uint8_t*>(metadata_digests_ptr),reinterpret_cast<const uint8_t*>(metadata),sizeof(metadata));
    *reinterpret_cast<PublicationHeader*>(header_output_ptr)=bank->header;
    publication_copy_bytes(reinterpret_cast<uint8_t*>(source_output_ptr),reinterpret_cast<const uint8_t*>(bank->source),sizeof(bank->source));
    auto* output_ranges=reinterpret_cast<PublicationRange*>(range_output_ptr);
    for(uint64_t i=0;i<binding_count;++i) {
        const auto& binding=bindings[i];const auto* range=publication_find_range(ranges,count,binding.role,binding.index);
        output_ranges[i]=*range;
        if(binding.role==3 || (binding.role>=6 && (binding.role<18 || binding.role>25)))
            publication_copy_bytes(reinterpret_cast<uint8_t*>(binding.destination),publication_range_bytes(control,*range),range->length_bytes);
    }
}

// This verification runs before release against the device-selected resident
// aliases and retained private copies. Shared append-only arenas are hashed only
// through the original logical interval.
extern "C" __global__ void semantic_publication_step_input_guard(uint64_t lease_ptr,uint64_t header_ptr,uint64_t source_ptr,
        uint64_t bindings0_ptr,uint64_t bindings1_ptr,uint64_t binding_count,uint64_t ranges_ptr,uint64_t metadata_digests_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    uint64_t binding_bytes=0,range_bytes=0;
    if(!publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !publication_pointer_span(header_ptr,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
       !publication_pointer_span(source_ptr,32*sizeof(SourceSlot),alignof(SourceSlot)) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationStepInput),&binding_bytes) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationRange),&range_bytes) ||
       !publication_pointer_span(bindings0_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(bindings1_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(ranges_ptr,range_bytes,alignof(PublicationRange)) ||
       !publication_pointer_span(metadata_digests_ptr,8*sizeof(uint64_t),alignof(uint64_t))) {
        semantic_content_integrity_trap();return;
    }
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.abi!=1 || lease.status || lease.active!=1 || lease.bank>1) {
        semantic_content_integrity_trap();return;
    }
    const auto& header=*reinterpret_cast<const PublicationHeader*>(header_ptr);
    uint64_t actual[2][4];
    const auto* expected=reinterpret_cast<const uint64_t*>(metadata_digests_ptr);
    if(header.abi!=1 || binding_count<9 || binding_count>header.range_count ||
       publication_step_metadata_digest(header,reinterpret_cast<const SourceSlot*>(source_ptr),binding_count,actual) ||
       !publication_identity_equal(actual[0],expected) || !publication_identity_equal(actual[1],expected+4)) {
        semantic_content_integrity_trap();return;
    }
    const auto* bindings=reinterpret_cast<const PublicationStepInput*>(lease.bank ? bindings1_ptr : bindings0_ptr);
    const auto* ranges=reinterpret_cast<const PublicationRange*>(ranges_ptr);
    uint64_t fixed_role=8;
    for(uint64_t i=0;i<binding_count;++i) {
        const auto& binding=bindings[i];const auto& range=ranges[i];
        if(range.role>=8 && range.role<=13 && (range.role!=fixed_role++ || range.index)) { semantic_content_integrity_trap();return; }
        const bool alias=range.role==1 || range.role==4 || range.role==5;
        const bool model=range.role>=18 && range.role<=25;
        if(range.role!=binding.role || range.index!=binding.index || !range.generation ||
           range.offset_bytes>UINT64_MAX-range.length_bytes || range.logical_end<range.logical_begin ||
           ((!model || binding.capacity_bytes) && !publication_pointer_span(binding.destination,binding.capacity_bytes)) || range.length_bytes>binding.capacity_bytes ||
           (i==0 && (range.role!=1 || range.index)) || (i==1 && (range.role!=3 || range.index)) ||
           (i>1 && ((!model && range.role!=44 && !(range.role>=15 && range.role<=17) && (range.role<4 || range.role>13)) ||
               (range.role==ranges[i-1].role ? range.index!=ranges[i-1].index+1 :
                   (range.role<ranges[i-1].role || range.index))))) {
            semantic_content_integrity_trap();return;
        }
        if(alias && (range.logical_begin || range.logical_end!=header.prefix_extent ||
           (range.role==1 && (header.prefix_extent>UINT64_MAX/sizeof(SourceSlot) ||
               range.length_bytes!=header.prefix_extent*sizeof(SourceSlot))))) {
            semantic_content_integrity_trap();return;
        }
        if(!alias && (range.logical_begin || (range.role!=15 && range.logical_end) || range.length_bytes!=binding.capacity_bytes ||
           (range.role==15 && (range.length_bytes%sizeof(RawFeedbackRecord) ||
               range.logical_end>range.length_bytes/sizeof(RawFeedbackRecord))) ||
           (range.role==3 && range.length_bytes!=32))) { semantic_content_integrity_trap();return; }
        if(model) {
            uint64_t destination=0;
            if(range.offset_bytes>binding.backing_bytes || range.length_bytes>binding.backing_bytes-range.offset_bytes ||
               !semantic_graph::checked_add(binding.backing,range.offset_bytes,&destination) || destination!=binding.destination ||
               (binding.backing_bytes && !publication_pointer_span(binding.backing,binding.backing_bytes))) {
                semantic_content_integrity_trap();return;
            }
            bool sealed=false;
            for(uint64_t previous=0;previous<i;++previous)
                if(ranges[previous].storage_slot==range.storage_slot) {
                    if(bindings[previous].backing!=binding.backing || bindings[previous].backing_bytes!=binding.backing_bytes ||
                       !publication_identity_equal(ranges[previous].backing_digest,range.backing_digest)) {
                        semantic_content_integrity_trap();return;
                    }
                    sealed=true;break;
                }
            if(!sealed) {
                uint64_t digest[4];
                semantic_graph::sha256(reinterpret_cast<const uint8_t*>(binding.backing),binding.backing_bytes,digest);
                if(!publication_identity_equal(digest,range.backing_digest)) { semantic_content_integrity_trap();return; }
            }
        } else if(binding.backing!=binding.destination || binding.backing_bytes!=binding.capacity_bytes) {
            semantic_content_integrity_trap();return;
        }
        PublicationTensorBytes view{};uint64_t bytes=0,digest[4];
        const auto* layout=publication_tensor_role(range.role) ? &binding.layout : nullptr;
        if(publication_content_view(reinterpret_cast<const uint8_t*>(binding.destination),range.length_bytes,range,layout,&view,&bytes) ||
           publication_content_digest(view,bytes,digest) || !publication_identity_equal(digest,range.digest)) {
           semantic_content_integrity_trap();return;
        }
    }
    if(fixed_role!=14 || ranges[binding_count-1].role!=44 || ranges[binding_count-1].index)semantic_content_integrity_trap();
}
__device__ void publication_intent_identity(const IntentEntry& entry,uint64_t* identity) {
    uint64_t words[25];words[0]=0x786c6f67696e7431ULL;
    const uint64_t* content=entry.checkpoint_lineage;
    for(uint32_t i=0;i<24;++i)words[i+1]=content[i];
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(words),sizeof(words),identity);
}
__device__ uint64_t publication_validate_intents(const PublicationControl& control,
        const PublicationRange* ranges,uint64_t count,uint64_t terminal_position=UINT64_MAX) {
    const auto* entry_range=publication_find_range(ranges,count,30);
    const auto* payload_range=publication_find_range(ranges,count,31);
    if(!entry_range || !payload_range || entry_range->length_bytes<sizeof(IntentQueueHeader))return 1;
    const auto* queue=reinterpret_cast<const IntentQueueHeader*>(publication_range_bytes(control,*entry_range));
    const auto* payload=publication_range_bytes(control,*payload_range);
    if(!queue || !payload || queue->abi!=1 || queue->count>queue->capacity ||
       queue->capacity>(UINT64_MAX-sizeof(IntentQueueHeader))/sizeof(IntentEntry) ||
       entry_range->length_bytes!=sizeof(IntentQueueHeader)+queue->count*sizeof(IntentEntry) ||
       payload_range->length_bytes!=queue->payload_used_bytes || !queue->effect_length_bytes ||
       queue->effect_offset_bytes!=0 || queue->effect_length_bytes>queue->payload_used_bytes ||
       queue->payload_used_bytes>queue->payload_capacity_bytes)return 1;
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    if(storage[entry_range->storage_slot].bytes-entry_range->offset_bytes<sizeof(IntentQueueHeader)+queue->capacity*sizeof(IntentEntry) ||
       storage[payload_range->storage_slot].bytes-payload_range->offset_bytes!=queue->payload_capacity_bytes)return 1;
    const auto* entries=reinterpret_cast<const IntentEntry*>(queue+1);
    uint64_t chain[4]={},effect[4],cursor=queue->effect_length_bytes;
    semantic_graph::sha256(payload,queue->effect_length_bytes,effect);
    for(uint64_t i=0;i<queue->count;++i) {
        const IntentEntry& entry=entries[i];uint64_t digest[4];
        if(entry.payload_offset!=cursor || entry.payload_len>queue->payload_used_bytes-cursor ||
           !publication_identity_equal(entry.effect_digest,effect) || !publication_identity_equal(entry.previous_chain,chain))return 1;
        semantic_graph::sha256(payload+cursor,entry.payload_len,digest);
        if(!publication_identity_equal(digest,entry.payload_digest))return 1;
        publication_intent_identity(entry,digest);
        if(!publication_identity_equal(digest,entry.stable_identity))return 1;
        publication_fold(chain,30,i,digest);
        if(!publication_identity_equal(chain,entry.chain))return 1;
        cursor+=entry.payload_len;
    }
    if(cursor!=queue->payload_used_bytes || !publication_identity_equal(queue->chain_head,chain))return 1;
    if(terminal_position!=UINT64_MAX) {
        if(queue->count==queue->capacity || terminal_position>(UINT64_MAX-16)/8)return 5;
        const uint64_t needed=16+terminal_position*8;
        if(needed>queue->payload_capacity_bytes-queue->payload_used_bytes)return 5;
    }
    return 0;
}

__device__ bool before(uint32_t a,uint32_t b,const float* logits) {
    if(a==0xffffffffU)return false;
    if(b==0xffffffffU)return true;
    return logits[a]>logits[b] || (logits[a]==logits[b] && a<b);
}

// The only semantic dispatcher remains the hypergraph owner's complete resident
// admission/validation/mutation/retirement path, shared by both public kernels.
__device__ void semantic_call(const Descriptor& d, uint64_t admission,
        const semantic_graph::ResidentHandle* handle, const semantic_graph::Receipt* source,
        semantic_graph::Receipt* output, const semantic_graph::DecodedStatement* statement=nullptr,
        const semantic_graph::DecodedSupport* support=nullptr,ExecutionWork* work=nullptr) {
    semantic_graph::LaunchDescriptor launch={};
    launch.expected_owner=d.arena[1];launch.root_capacity=d.arena[2];
    launch.statement_capacity=d.arena[3];launch.support_capacity=d.arena[4];
    launch.version_capacity=d.arena[5];launch.arena_words=d.arena[6];
    launch.handle_ptr=reinterpret_cast<uint64_t>(handle);
    launch.source_receipt_ptr=reinterpret_cast<uint64_t>(source);
    launch.receipt_ptr=reinterpret_cast<uint64_t>(output);
    launch.decoded_statement_ptr=reinterpret_cast<uint64_t>(statement);
    launch.decoded_support_ptr=reinterpret_cast<uint64_t>(support);
    launch.admission=admission;launch.abi_generation=semantic_graph::kAbiGeneration;
    semantic_graph::NativeWorkTally actual{};
    semantic_graph::execute(reinterpret_cast<uint64_t*>(d.arena[0]),launch,work ? &actual : nullptr);
    if(work)execution_work_merge_native(*work,actual);
}

// Reclaim only roots freshly sealed by this pending tuple. Original receipts
// remain intact for refusal diagnostics; the mask makes cleanup idempotent.
// Unchanged seals alias the protected base and confer no retirement authority.
__device__ bool retire_pending_roots(const Descriptor& d,
        const semantic_graph::ResidentHandle& base, State* state,uint64_t retained_slot=UINT64_MAX,
        const semantic_graph::ResidentHandle* protected_root=nullptr) {
    for(uint32_t lane=0;lane<2;++lane) {
        if(lane+1==retained_slot)continue;
        if(state->retired_roots_mask & (1ULL<<lane))continue;
        const auto& sealed=state->semantic_receipts[3+lane*3];
        if(sealed.words[0]!=semantic_graph::kOk ||
           sealed.words[33]!=semantic_graph::kRootKind)continue;
        if(sealed.words[1]==semantic_graph::kOutcomeUnchanged) {
            if(sealed.words[39]!=base.owner || sealed.words[34]!=base.slot ||
               sealed.words[35]!=base.generation)return false;
        } else {
            if(sealed.words[39]!=base.owner ||
               sealed.words[1]!=semantic_graph::kOutcomeInserted)return false;
            semantic_graph::Command command={};
            command.words[0]=semantic_graph::kRetireRoot;
            command.words[1]=base.owner;
            command.words[8]=sealed.words[34];command.words[9]=sealed.words[35];
            command.words[10]=protected_root ? protected_root->slot : base.slot;
            command.words[11]=protected_root ? protected_root->generation : base.generation;
            auto cleanup=&state->cleanup_receipts[1+lane];
            semantic_graph::LaunchDescriptor launch={};
            launch.expected_owner=d.arena[1];launch.root_capacity=d.arena[2];
            launch.statement_capacity=d.arena[3];launch.support_capacity=d.arena[4];
            launch.version_capacity=d.arena[5];launch.arena_words=d.arena[6];
            launch.command_ptr=reinterpret_cast<uint64_t>(&command);
            launch.receipt_ptr=reinterpret_cast<uint64_t>(cleanup);
            launch.admission=semantic_graph::kHostCommandAdmission;
            launch.abi_generation=semantic_graph::kAbiGeneration;
            semantic_graph::NativeWorkTally actual{};
            semantic_graph::execute(reinterpret_cast<uint64_t*>(d.arena[0]),launch,&actual);
            execution_work_merge_native(state->execution_work,actual);
            if(cleanup->words[0]!=semantic_graph::kOk)return false;
        }
        state->retired_roots_mask|=1ULL<<lane;
    }
    return true;
}

__device__ bool supported(const Component* components,const uint8_t* support,uint32_t start,uint32_t field,uint32_t category,
        semantic_graph::NativeWorkTally* work=nullptr) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
    const Component c=components[start+field];
    return category<c.cardinality && support[c.offset+category];
}
__device__ bool operand_matches(const uint64_t* books,uint32_t target,uint32_t position,uint32_t operand,
        semantic_graph::NativeWorkTally* work=nullptr) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
    const uint64_t* t=books+books[0]+10*target;
    if(position>=t[1])return operand==0;
    if(!operand)return false;
    const uint64_t* o=books+books[2]+6*operand;
    return o[2]==t[2+position] && o[3]==t[6+position];
}
__device__ bool target_completes(const uint64_t* books,const Component* components,const uint8_t* support,uint32_t start,uint32_t target,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!target || !supported(components,support,start,2,target,work))return false;
    for(uint32_t position=0;position<4;++position) {
        bool found=false;
        for(uint32_t operand=0;operand<books[3];++operand)
            if(supported(components,support,start,3+position,operand,work) && operand_matches(books,target,position,operand,work)){found=true;break;}
        if(!found)return false;
    }
    return true;
}
__device__ bool action_completes(const uint64_t* books,const Component* components,const uint8_t* support,
                               const uint32_t* viable,uint32_t start,uint32_t action,
                               semantic_graph::NativeWorkTally* work=nullptr) {
    if(!supported(components,support,start,0,action,work))return false;
    if(action==ACTION_NO_EDIT) {
        for(uint32_t f=1;f<18;++f)if(!supported(components,support,start,f,0,work))return false;
        return true;
    }
    if(!supported(components,support,start,1,1,work))return false;
    for(uint32_t f=1;f<18;++f)if((INSERT_SUPPORT_NULL_MASK>>f)&1)
        if(!supported(components,support,start,f,0,work))return false;
    for(uint32_t f=11;f<=17;f+=6) {
        bool found=false;
        for(uint32_t i=1;i<components[start+f].cardinality;++i)found |= supported(components,support,start,f,i,work);
        if(!found)return false;
    }
    for(uint32_t i=1;i<books[1];++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
        if(viable[i])return true;
    }
    return false;
}
__device__ bool legal_category(const Component c,uint32_t category,const uint64_t* books,
        const uint32_t* viable,const uint32_t* choices,bool no_edit,bool apply,
        semantic_graph::NativeWorkTally* work=nullptr) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
    if(c.kind==COMPONENT_KIND_TEXT)return true;
    if(c.field==0)return category==0 ? no_edit : category==1 && apply;
    if(choices[0]==ACTION_NO_EDIT)return category==0;
    if((INSERT_SUPPORT_NULL_MASK>>c.field)&1)return category==0;
    if(c.field==1)return category==1;
    if(c.field==2)return category && viable[category];
    if(c.field>=3 && c.field<=6)return operand_matches(books,choices[2],c.field-3,category,work);
    return category!=0;
}

// Segmented canonical bytes avoid a size-limited temporary for accepted symbols
// and ordered qualifier bundles. SHA-256 is the same implementation as the owner.
struct Segments {
    const uint8_t* data[5];uint64_t lengths[5];uint32_t count;
    __device__ uint8_t operator[](uint64_t index) const {
        for(uint32_t i=0;i<count;++i){if(index<lengths[i])return data[i][index];index-=lengths[i];}
        return 0; // Padding is handled by sha256; no out-of-range source is requested.
    }
};
__device__ void decode_action(const uint64_t* books,const uint32_t* choices,
                             semantic_graph::DecodedStatement* statement,semantic_graph::DecodedSupport* support) {
    // A task-free Cartesian construction need not name an original statement
    // occurrence. Only the matched task query can supply that original index.
    statement->record=UINT32_MAX;
    for(uint32_t i=0;i<10;++i)statement->reconstruction[i]=0;
    const char atom_domain[]="xlog.semantic.record.v2\0";
    uint8_t prefix[sizeof(atom_domain)-1+32+8];
    uint32_t cursor=0;
    for(uint32_t i=0;i<sizeof(atom_domain)-1;++i)prefix[cursor++]=atom_domain[i];
    semantic_graph::words_to_bytes(books+10,prefix+cursor);cursor+=32;
    const uint64_t* target=books+books[0]+10*choices[2];
    statement->reconstruction[0]=uint32_t(target[0]);
    for(uint32_t i=0;i<4;++i)prefix[cursor++]=uint8_t(target[0]>>(8*i));
    for(uint32_t i=0;i<4;++i)prefix[cursor++]=uint8_t(target[1]>>(8*i));
    Segments atom={};atom.data[0]=prefix;atom.lengths[0]=cursor;atom.count=1;
    uint64_t length=cursor;
    const uint8_t* bytes=reinterpret_cast<const uint8_t*>(books)+books[8];
    for(uint32_t i=0;i<target[1];++i) {
        const uint64_t* operand=books+books[2]+6*choices[3+i];
        statement->reconstruction[1+2*i]=uint32_t(operand[4]);
        statement->reconstruction[2+2*i]=uint32_t(operand[5]);
        atom.data[atom.count]=bytes+operand[0];atom.lengths[atom.count++]=operand[1];length+=operand[1];
    }
    uint64_t atom_digest[4];semantic_graph::sha256(atom,length,atom_digest);
    const char statement_domain[]="xlog.semantic.statement.v2\0";
    uint8_t statement_prefix[sizeof(statement_domain)-1+32];cursor=0;
    for(uint32_t i=0;i<sizeof(statement_domain)-1;++i)statement_prefix[cursor++]=statement_domain[i];
    semantic_graph::words_to_bytes(atom_digest,statement_prefix+cursor);cursor+=32;
    const uint64_t* qualifier=books+books[4]+3*choices[11];
    statement->reconstruction[9]=uint32_t(qualifier[2]);
    Segments complete={};complete.count=2;complete.data[0]=statement_prefix;complete.lengths[0]=cursor;
    complete.data[1]=bytes+qualifier[0];complete.lengths[1]=qualifier[1];
    uint64_t digest[4];semantic_graph::sha256(complete,cursor+qualifier[1],digest);
    for(uint32_t i=0;i<8;++i)statement->identity_words[i]=uint32_t(digest[i/2]>>(32*(i%2)));
    const uint64_t* leaf=books+books[6]+18*choices[17];support->polarity=uint32_t(leaf[0]);
    support->record=uint32_t(leaf[17]);
    uint32_t* destinations[4]={support->provenance_words,support->source_words,support->context_words,support->scope_words};
    for(uint32_t m=0;m<4;++m)for(uint32_t i=0;i<8;++i)
        destinations[m][i]=uint32_t(leaf[1+m*4+i/2]>>(32*(i%2)));
}

// The private cold task bank binds full qualified keys and statement-specific
// support identities. It is not part of the sampler's categorical codebooks.
__device__ bool task_allows(const uint64_t* task,
        semantic_graph::DecodedStatement& statement,
        const semantic_graph::DecodedSupport& support) {
    if(!task || task[0]!=4)return false;
    uint64_t identity[4];semantic_graph::decode_identity(statement.identity_words,identity);
    uint32_t query_index=3;
    for(uint32_t query=0;query<3 && query_index==3;++query) {
        bool equal=true;
        for(uint32_t i=0;i<4;++i)equal &= identity[i]==task[6+4*query+i];
        if(equal)query_index=query;
    }
    if(query_index==3)return false;
    uint64_t digest[4];semantic_graph::support_digest(identity,support,digest);
    for(uint64_t event=0;event<task[21];++event) {
        bool equal=task[34+5*event]==support.record;
        for(uint32_t i=0;i<4;++i)equal &= digest[i]==task[35+5*event+i];
        if(equal) {
            statement.record=uint32_t(task[22+query_index]);
            for(uint32_t i=0;i<10;++i)statement.reconstruction[i]=0;
            return true;
        }
    }
    return false;
}

__device__ void task_cost(State* state,uint32_t slot,const uint64_t* task) {
    auto& facts=state->task_evaluation.facts[slot];
    if(slot) {
        const auto& work=state->work[slot-1];
        facts.c=work.edit_commands+work.added_supports+work.defined_truth_changes;
    }
    facts.v=int64_t(task[25]*facts.g+task[26]*facts.p)-int64_t(task[27]*facts.c);
}

__device__ bool task_queries(const Descriptor& d,const uint64_t* task,uint32_t slot,
        const semantic_graph::ResidentHandle* root,
        const semantic_graph::Receipt* candidate,State* state) {
    auto& evaluation=state->task_evaluation;
    auto& facts=evaluation.facts[slot];
    bool valid=true,hard=true;
    for(uint32_t query=0;query<3;++query) {
        semantic_graph::DecodedStatement statement{};
        statement.record=uint32_t(task[22+query]);
        for(uint32_t i=0;i<8;++i)
            statement.identity_words[i]=uint32_t(task[6+4*query+i/2]>>(32*(i%2)));
        auto output=&evaluation.query_receipts[slot][query];
        ++evaluation.query_count;
        // Both queries use the original view receipt, never a preceding truth result.
        semantic_call(d,slot ? semantic_graph::kResidentForkTruthAdmission :
            semantic_graph::kResidentRootTruthAdmission,root,candidate,output,&statement,nullptr,&state->execution_work);
        valid &= output->words[0]==semantic_graph::kOk && output->words[2]<=3;
        hard &= output->words[2]<=3 && (task[31+query] & (uint64_t(1)<<output->words[2]))!=0;
        facts.correct[query]=output->words[0]==semantic_graph::kOk &&
            output->words[2]==task[18+query];
    }
    facts.g=facts.correct[0]+facts.correct[1]+facts.correct[2];
    facts.p=facts.correct[0] && facts.correct[1] && facts.correct[2];
    facts.eligible=valid && hard;
    task_cost(state,slot,task);
    return valid;
}

__device__ void select_task(State* state,const uint64_t* task) {
    auto& evaluation=state->task_evaluation;
    uint64_t winner=0,refusals=0,spent=evaluation.query_count;
    for(uint32_t slot=1;slot<3;++slot) {
        const auto& current=evaluation.facts[slot];
        const auto& best=evaluation.facts[winner];
        if(current.eligible && (!best.eligible || current.v>best.v))winner=slot;
    }
    for(uint32_t lane=0;lane<2;++lane) {
        refusals+=evaluation.lane_refusal[lane]!=0;
        if(lane+1!=winner)spent+=state->work[lane].edit_commands+state->work[lane].defined_truth_changes;
    }
    evaluation.winner=winner;
    evaluation.return_value=int64_t(task[28])*(evaluation.facts[winner].v-evaluation.facts[0].v)
        -int64_t(task[29])*int64_t(refusals)-int64_t(task[30])*int64_t(spent);
}
__device__ void semantic_policy_backward_status_guard(uint64_t status_ptr) {
    if(!status_ptr || status_ptr%alignof(uint64_t) || status_ptr>UINT64_MAX-sizeof(uint64_t) ||
       *reinterpret_cast<const uint64_t*>(status_ptr)!=0)semantic_content_integrity_trap();
}
#ifdef XLOG_SEMANTIC_POLICY
__device__ bool semantic_policy_identity_present(const uint64_t identity[4]) {
    return identity[0] || identity[1] || identity[2] || identity[3];
}

// Late differentiation reads the original receipt and tape. It never enters
// semantic admission, generates random words, or writes forward state.
__device__ void semantic_policy_backward(const Descriptor& descriptor) {
    const auto p=descriptor.policy;
    const auto b=descriptor.backward;
    auto status=reinterpret_cast<uint64_t*>(b.status);
    auto coefficients=reinterpret_cast<double*>(b.cotangents);
    auto gradients=reinterpret_cast<float*>(b.parameters);
    auto text_gradients=reinterpret_cast<float*>(b.text);
    auto baseline_gradients=reinterpret_cast<float*>(b.baselines);
    auto recurrent_gradients=reinterpret_cast<float*>(b.recurrent);
    auto components=reinterpret_cast<const Component*>(descriptor.components);
    auto receipts=reinterpret_cast<const Receipt*>(descriptor.receipts);
    auto support=reinterpret_cast<const uint8_t*>(descriptor.support);
    auto books=reinterpret_cast<const uint64_t*>(descriptor.codebooks);
    auto viability=reinterpret_cast<uint32_t*>(descriptor.scratch)+2*262144;
    __shared__ uint32_t actions[4][2],choices[18];
    __shared__ uint32_t legal_count,active_count,selected_active,apply_selected;
    __shared__ uint64_t actor_denominator;
    __shared__ float maximum,reward_f32,critic_scale;
    __shared__ U192 threshold;
    __shared__ double scale,reward,actor_scale,cost_scale;
    if(threadIdx.x==0) {
        *status=0;
        apply_selected=1;
        actor_denominator=0;
        reward=actor_scale=cost_scale=0.0;
        reward_f32=critic_scale=0.0f;
        if(b.mode==0) {
            for(uint32_t i=0;i<COMPONENT_COUNT;++i)
                if(!isfinite(coefficients[i]))*status=1;
        } else if(b.mode==1) {
            if(!b.selection || b.selection%alignof(SemanticTrainingViewSelection) ||
               !b.objective || b.objective%alignof(SemanticTrainingObjectiveRecord) ||
               !b.objective_groups || b.objective_groups%alignof(SemanticTrainingObjectiveGroupRecord) ||
               !b.objective_group_members || b.objective_group_members%alignof(uint64_t)) {
                *status=1;
            } else {
                const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(b.selection);
                const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(b.objective);
                const auto* groups=reinterpret_cast<const SemanticTrainingObjectiveGroupRecord*>(b.objective_groups);
                const auto* members=reinterpret_cast<const uint64_t*>(b.objective_group_members);
                if(selection.status || selection.basis!=1 || selection.origin.present!=1 ||
                   selection.origin.transition!=1 || selection.origin_candidate==UINT64_MAX ||
                   selection.ordinal>=selection.row_count || selection.row_count!=objective.row_count ||
                   selection.capacity!=objective.capacity || objective.group_count!=8 ||
                   !objective.group_member_count || !objective.cost_cap ||
                   !semantic_policy_identity_present(objective.identity) ||
                   !semantic_policy_identity_present(objective.cost_unit)) {
                    *status=1;
                }
                const double evaluator_min=__longlong_as_double((long long)objective.evaluator_min_bits);
                const double evaluator_max=__longlong_as_double((long long)objective.evaluator_max_bits);
                float objective_coefficients[9];
                for(uint32_t i=0;i<9;++i) {
                    objective_coefficients[i]=__uint_as_float(uint32_t(objective.coefficient_bits[i]));
                    if((objective.coefficient_bits[i]>>32) || !isfinite(objective_coefficients[i]) ||
                       objective_coefficients[i]<=0.0f)*status=1;
                }
                if(!isfinite(evaluator_min) || !isfinite(evaluator_max) || evaluator_min>evaluator_max)*status=1;
                uint64_t actor_groups=0;
                bool selected_member=false;
                for(uint64_t i=0;i<objective.group_count && !*status;++i) {
                    const auto group=groups[i];
                    if(!group.denominator || !group.member_count ||
                       group.member_offset>objective.group_member_count ||
                       group.member_count>objective.group_member_count-group.member_offset) {
                        *status=1;break;
                    }
                    if(group.kind==8) {
                        ++actor_groups;
                        actor_denominator=group.denominator;
                        for(uint64_t member=0;member<group.member_count;++member)
                            selected_member |= members[group.member_offset+member]==selection.ordinal;
                    }
                }
                if(actor_groups!=1 || !selected_member)*status=1;
                apply_selected=b.origin_candidate==selection.origin_candidate;
                if(apply_selected && !*status) {
                    const auto* state=reinterpret_cast<const State*>(descriptor.state);
                    if(!state || state->status || state->model_generation!=selection.origin.model_generation ||
                       state->stream_serial!=selection.origin.stream_serial ||
                       state->family_id!=selection.origin.family_id ||
                       state->proposal!=selection.origin.proposal ||
                       !isfinite(state->importance_weight) || state->importance_weight<=0.0 ||
                       state->task_evaluation.winner>=3 ||
                       state->task_evaluation.facts[state->task_evaluation.winner].c>objective.cost_cap) {
                        *status=1;
                    } else {
                        reward=__ll2double_rn((long long)state->task_evaluation.return_value);
                        reward_f32=__ll2float_rn((long long)state->task_evaluation.return_value);
                        if(reward<evaluator_min || reward>evaluator_max || !isfinite(reward))*status=1;
                        const double denominator=__ull2double_rn(actor_denominator);
                        const double weight=state->importance_weight;
                        const double cost=__ddiv_rn(
                            __ull2double_rn(state->task_evaluation.facts[state->task_evaluation.winner].c),
                            __ull2double_rn(objective.cost_cap));
                        actor_scale=__ddiv_rn(__dmul_rn(double(objective_coefficients[6]),weight),denominator);
                        cost_scale=__ddiv_rn(__dmul_rn(__dmul_rn(double(objective_coefficients[8]),weight),cost),denominator);
                        critic_scale=__fdiv_rn(__fmul_rn(2.0f,objective_coefficients[7]),
                            __ull2float_rn(actor_denominator));
                        if(!isfinite(actor_scale) || !isfinite(cost_scale) || !isfinite(critic_scale))*status=1;
                    }
                }
            }
        } else {
            *status=1;
        }
    }
    __syncthreads();
    if(*status)return;
    for(uint64_t i=threadIdx.x;i<b.parameter_cells;i+=blockDim.x)gradients[i]=0.0f;
    for(uint64_t i=threadIdx.x;i<b.text_cells;i+=blockDim.x)text_gradients[i]=0.0f;
    for(uint32_t i=threadIdx.x;i<2*37*128;i+=blockDim.x)recurrent_gradients[i]=0.0f;
    for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
        if(b.mode==0) {
            baseline_gradients[i]=0.0f;
        } else {
            const float baseline=baseline_gradients[i];
            coefficients[i]=0.0;
            baseline_gradients[i]=0.0f;
            if(apply_selected) {
                const float difference=__fsub_rn(baseline,reward_f32);
                const float raw_square=__fmul_rn(difference,difference);
                const double advantage=__dsub_rn(reward,double(baseline));
                coefficients[i]=__dadd_rn(-__dmul_rn(actor_scale,advantage),cost_scale);
                if(receipts[i].legal_count>1)
                    baseline_gradients[i]=__fmul_rn(critic_scale,difference);
                if(!isfinite(baseline) || !isfinite(raw_square) ||
                   !isfinite(coefficients[i]) || !isfinite(baseline_gradients[i]))
                    atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
            }
        }
    }
    __syncthreads();
    if(*status || (b.mode==1 && !apply_selected))return;
    for(uint32_t edit=0;edit<4;++edit) {
        uint32_t start=32+(edit/2)*68+(edit%2)*18;
        uint32_t* viable=viability+edit*books[1];
        for(uint32_t t=threadIdx.x;t<books[1];t+=blockDim.x)
            viable[t]=target_completes(books,components,support,start,t);
        __syncthreads();
        if(threadIdx.x==0)for(uint32_t action=0;action<2;++action)
            actions[edit][action]=action_completes(books,components,support,viable,start,action);
        __syncthreads();
    }
    // Lanes are serial here because R, positions and field parameters are shared.
    for(int ordinal=COMPONENT_COUNT-1;ordinal>=0;--ordinal) {
        const Component c=components[ordinal];
        const Receipt receipt=receipts[ordinal];
        const uint32_t lane=uint32_t(ordinal)/68;
        const uint32_t edit=lane*2+c.slot;
        const uint32_t* viable=viability+edit*books[1];
        const bool structural=c.kind==COMPONENT_KIND_EDIT;
        if(!structural && !text_selected(descriptor.text,c.field)) {
            if(threadIdx.x==0 && !text_null_receipt(receipt))*status=2;
            __syncthreads();if(*status)return;
            continue;
        }
        const uint32_t step=c.slot*18+c.field;
        const uint64_t text_index=structural ? 0 : text_row_index(descriptor.text,c.field);
        if(!structural && text_index>=text_row_count(descriptor.text)) {
            if(threadIdx.x==0)*status=2;__syncthreads();return;
        }
        const float* row=structural ? nullptr : reinterpret_cast<const float*>(descriptor.logits)+text_index*TEXT_CARDINALITY;
        float* score_gradients=structural ? reinterpret_cast<float*>(b.scores)
            : text_gradients+text_index*TEXT_CARDINALITY;
        model_policy::FieldView field={};
        model_policy::FieldAdjoint field_gradients={};
        if(structural) {
            const auto f=p.fields[c.field];
            field={reinterpret_cast<const float*>(f.embeddings),reinterpret_cast<const float*>(f.biases),
                f.cardinality,f.null_category};
            field_gradients={f.embeddings ? gradients+(f.embeddings-p.z)/4 : nullptr,
                f.biases ? gradients+(f.biases-p.z)/4 : nullptr};
            if(threadIdx.x==0)for(uint32_t j=0;j<18;++j)
                choices[j]=j<c.field ? receipts[ordinal-c.field+j].choice : 0;
            model_policy::readout_hidden(reinterpret_cast<const float*>(p.z)+(lane*2+c.slot)*128,
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.positions)+c.field*128,
                reinterpret_cast<float*>(p.hidden),threadIdx.x,blockDim.x);
            __syncthreads();
            model_policy::readout_scores(reinterpret_cast<const float*>(p.hidden),field,
                reinterpret_cast<float*>(p.scores),threadIdx.x,blockDim.x);
            __syncthreads();
            row=reinterpret_cast<const float*>(p.scores);
        }
        if(threadIdx.x==0) {
            legal_count=0;active_count=0;selected_active=0;maximum=-INFINITY;scale=0.0;
            for(uint32_t j=0;j<c.cardinality;++j)
                if(support[c.offset+j] && legal_category(c,j,books,viable,choices,actions[edit][0],actions[edit][1])) {
                    if(!isfinite(row[j]))*status=2;
                    if(row[j]>maximum)maximum=row[j];
                    ++legal_count;
                }
            if(legal_count!=receipt.legal_count || !receipt.active_count || receipt.active_count>legal_count
                || receipt.choice>=c.cardinality || !support[c.offset+receipt.choice]
                || !legal_category(c,receipt.choice,books,viable,choices,actions[edit][0],actions[edit][1]))*status=2;
            if(!*status && legal_count>1 && receipt.active_count>1) {
                U192 floor=shifted(receipt.active_count,118);
                const int positive=compare(receipt.p,floor);
                if(positive<0)*status=2;
                if(positive>0) {
                    selected_active=1;
                    threshold=add(sub(receipt.p,floor),multiply(distance_scaled(maximum,row[receipt.choice]),receipt.active_count));
                    // Active membership stays exact U192. Round its numerator
                    // once before the FP64 cotangent arithmetic.
                    const double divisor=ldexp(u192_to_double_rn(receipt.p),-149);
                    scale=__ddiv_rn(coefficients[ordinal],divisor);
                }
            }
        }
        __syncthreads();
        if(*status)return;
        for(uint32_t j=threadIdx.x;j<c.cardinality;j+=blockDim.x) {
            bool active=selected_active && support[c.offset+j]
                && legal_category(c,j,books,viable,choices,actions[edit][0],actions[edit][1])
                && compare(multiply(distance_scaled(maximum,row[j]),receipt.active_count),threshold)<0;
            if(active)atomicAdd(&active_count,1U);
            const float contribution=active ? __double2float_rn(__dmul_rn(scale,
                j==receipt.choice ? double(receipt.active_count-1) : -1.0)) : 0.0f;
            if(structural)score_gradients[j]=contribution;
            else score_gradients[j]=__fadd_rn(score_gradients[j],contribution);
        }
        __syncthreads();
        if(threadIdx.x==0 && selected_active && active_count!=receipt.active_count)*status=2;
        __syncthreads();
        if(*status)return;
        if(structural) {
            float* position_gradient=gradients+(p.positions-p.z)/4+c.field*128;
            model_policy::advance_vjp(reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.recurrence),
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step+1)*128,
                field,receipt.choice,recurrent_gradients+(lane*37+step+1)*128,
                recurrent_gradients+(lane*37+step)*128,gradients+(p.recurrence-p.z)/4,
                position_gradient,field_gradients,threadIdx.x,blockDim.x);
            __syncthreads();
            model_policy::readout_vjp(reinterpret_cast<const float*>(p.hidden),field,score_gradients,
                gradients+(lane*2+c.slot)*128,recurrent_gradients+(lane*37+step)*128,
                position_gradient,field_gradients,threadIdx.x,blockDim.x);
            __syncthreads();
        }
    }
    for(uint64_t i=threadIdx.x;i<b.parameter_cells;i+=blockDim.x)
        if(!isfinite(gradients[i]))atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
    for(uint64_t i=threadIdx.x;i<b.text_cells;i+=blockDim.x)
        if(!isfinite(text_gradients[i]))atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
    for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x)
        if(!isfinite(baseline_gradients[i]))atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
}
#endif

__device__ uint64_t publication_source_records(const PublicationControl& control,const PublicationBank& bank) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* prefix=publication_find_range(ranges,bank.header.range_count,1);
    const auto* provenance=publication_find_range(ranges,bank.header.range_count,2);
    if(!prefix || !provenance || prefix->logical_begin || prefix->logical_end!=bank.header.prefix_extent ||
       prefix->length_bytes!=bank.header.prefix_extent*sizeof(SourceSlot) ||
       provenance->length_bytes%sizeof(TokenProvenanceRecord))return 1;
    const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,*prefix));
    const auto* records=reinterpret_cast<const TokenProvenanceRecord*>(publication_range_bytes(control,*provenance));
    if(!source || !records)return 1;
    for(uint64_t i=0;i<bank.header.prefix_extent+32;++i) {
        const SourceSlot& s=i<bank.header.prefix_extent ? source[i] : bank.source[i-bank.header.prefix_extent];
        if(i<bank.header.prefix_extent && (s.logical_position!=i || s.kind!=1 || s.valid!=1 ||
            s.committed!=1 || s.recomputed!=1))return 1;
        if(s.kind!=1)continue;
        if(s.provenance_record>=provenance->length_bytes/sizeof(TokenProvenanceRecord))return 1;
        const auto& record=records[s.provenance_record];
        if(record.logical_position!=s.logical_position || record.token!=s.token ||
           !(record.origin_logical_digest[0]|record.origin_logical_digest[1]|record.origin_logical_digest[2]|record.origin_logical_digest[3]) ||
           !(record.authority_closure_digest[0]|record.authority_closure_digest[1]|record.authority_closure_digest[2]|record.authority_closure_digest[3]))return 1;
    }
    return 0;
}
__device__ uint64_t publication_write_feedback(const Descriptor& descriptor,PublicationControl& control,
        PublicationBank& bank,const semantic_graph::ResidentHandle& root,bool restored=false,ExecutionWork* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    auto* output_range=const_cast<PublicationRange*>(publication_find_range(ranges,bank.header.range_count,15));
    if(!output_range)return 1;
    auto* output=reinterpret_cast<RawFeedbackRecord*>(publication_range_bytes(control,*output_range));
    if(!output || output_range->length_bytes<2*sizeof(RawFeedbackRecord) ||
       output_range->length_bytes%sizeof(RawFeedbackRecord))return 1;
    if(restored) {
        if(output_range->logical_begin || output_range->logical_end!=2)return 1;
        for(uint64_t i=2;i<output_range->length_bytes/sizeof(RawFeedbackRecord);++i) {
            const auto* bytes=reinterpret_cast<const uint8_t*>(&output[i]);
            for(uint64_t byte=0;byte<sizeof(RawFeedbackRecord);++byte)if(bytes[byte])return 1;
        }
    } else {
        for(uint64_t i=0;i<output_range->length_bytes/sizeof(RawFeedbackRecord);++i)output[i]=RawFeedbackRecord{};
    }
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    if(!task)return 1;
    for(uint32_t index=0;index<3;++index) {
        const auto* statement_range=publication_find_range(ranges,bank.header.range_count,16,index);
        if(!statement_range || !publication_range_bytes(control,*statement_range))return 1;
        semantic_graph::DecodedStatement statement{};
        statement.record=uint32_t(task[22+index]);
        for(uint32_t word=0;word<8;++word)statement.identity_words[word]=uint32_t(task[6+index*4+word/2]>>(32*(word%2)));
        RawFeedbackRecord expected{};
        semantic_call(descriptor,semantic_graph::kResidentRootTruthAdmission,&root,nullptr,&expected.query_receipt,&statement,nullptr,work);
        if(expected.query_receipt.words[0]!=semantic_graph::kOk || expected.query_receipt.words[2]>3)return 6;
        expected.valid=1;expected.statement_index=index;
        expected.statement_length_bytes=statement_range->length_bytes;
        expected.pro=expected.query_receipt.words[2]&1;
        expected.contra=(expected.query_receipt.words[2]>>1)&1;
        if(restored) {
            const auto& saved=output[index];
            if(saved.valid!=expected.valid || saved.statement_index!=expected.statement_index ||
               saved.statement_offset_bytes || saved.statement_length_bytes!=expected.statement_length_bytes ||
               saved.pro!=expected.pro || saved.contra!=expected.contra)return 1;
            for(uint32_t word=0;word<42;++word)
                if(!publication_receipt_runtime_word(word) &&
                   saved.query_receipt.words[word]!=expected.query_receipt.words[word])return 6;
        } else output[index]=expected;
    }
    output_range->logical_begin=0;output_range->logical_end=3;
    return 0;
}
__device__ uint64_t publication_write_coverage(const PublicationControl& control,const PublicationContract& contract,
        PublicationBank& bank,const PendingContinuation* pending) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* range=publication_find_range(ranges,bank.header.range_count,14);
    if(!range)return 1;
    auto* coverage=reinterpret_cast<CompletionCoverage*>(publication_range_bytes(control,*range));
    if(!coverage || range->length_bytes!=sizeof(CompletionCoverage))return 1;
    *coverage=CompletionCoverage{};coverage->abi=1;
    semantic_graph::copy_identity(coverage->instance,bank.header.instance);
    coverage->base_word=pending ? pending->base_word : bank.header.base_word;
    coverage->model_generation=bank.header.model_generation;
    semantic_graph::copy_identity(coverage->topology_identity,contract.topology_identity);
    semantic_graph::copy_identity(coverage->table_identity,contract.table_identity);
    const auto* prefix=publication_find_range(ranges,bank.header.range_count,3);
    if(!prefix || prefix->length_bytes!=32)return 1;
    semantic_graph::copy_identity(coverage->prefix_identity,pending ? pending->prefix_identity :
        reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix)));
    const auto* inputs=pending ? reinterpret_cast<const PublicationRange*>(pending->ranges) : ranges;
    const uint64_t count=pending ? pending->range_count : bank.header.range_count;
    for(uint64_t i=0;i<count;++i) {
        const auto& source=inputs[i];uint64_t* group=nullptr;
        if(source.role==1)group=coverage->source_digest;
        if(source.role==4 || source.role==5)group=coverage->attention_digest;
        if(source.role==6 || source.role==7)group=coverage->linear_digest;
        if(source.role>=8 && source.role<=10)group=coverage->value_digest;
        if(source.role>=11 && source.role<=13)group=coverage->edit_digest;
        if(!group)continue;
        uint64_t digest[4];
        if(publication_range_digest(control,source,publication_find_layout(control,inputs,count,source.role,source.index),digest))return 1;
        publication_fold(group,source.role,source.index,digest);
        if(source.role==1){coverage->prefix_begin=source.logical_begin;coverage->prefix_end=source.logical_end;}
    }
    return publication_logical_range_digest(control,*range,nullptr,coverage->receipt_digest);
}
__device__ uint64_t publication_write_attempt(const PublicationControl& control,PublicationBank& bank,
        const PublicationBank& base,bool drain=false) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* target=publication_find_range(ranges,bank.header.range_count,33);
    if(!target)return 1;
    auto* attempt=reinterpret_cast<AttemptReceipt*>(publication_range_bytes(control,*target));
    if(!attempt || target->length_bytes!=sizeof(AttemptReceipt))return 1;
    *attempt=AttemptReceipt{};attempt->abi=1;
    semantic_graph::copy_identity(attempt->instance,bank.header.instance);
    attempt->base_word=base.header.publication_word;attempt->next_word=bank.header.publication_word;
    semantic_graph::copy_identity(attempt->logical_digest,bank.header.logical_digest);
    if(publication_action_digest(control,bank,bank.receipts,drain ? 0 : 136,attempt->action_receipts_digest))return 1;
    semantic_graph::sha256(PublicationSemanticReceiptBytes{bank.state.semantic_receipts},sizeof(bank.state.semantic_receipts),attempt->semantic_receipts_digest);
    const uint64_t role[5]={14,27,30,32,33};
    uint64_t* output[5]={attempt->coverage_digest,attempt->replay_head_digest,attempt->intent_head_digest,
        attempt->acknowledgement_head_digest,attempt->previous_attempt_digest};
    for(uint32_t i=0;i<5;++i) {
        const auto* source=publication_find_range(i==4 ? previous : ranges,bank.header.range_count,role[i]);
        if(!source || publication_logical_range_digest(control,*source,nullptr,output[i]))return 1;
    }
    return publication_logical_range_digest(control,*target,nullptr,attempt->receipt_digest);
}
__device__ uint64_t publication_initialize(const Descriptor& descriptor,PublicationControl& control,bool restored=false) {
    if(control.abi!=1 || control.word || control.reader_counts[0] || control.reader_counts[1] ||
       !control.contract || !control.banks[0] || !control.banks[1] || control.banks[0]==control.banks[1] ||
       !control.directories[0] || !control.directories[1] || !descriptor.task || !descriptor.state ||
       !descriptor.codebooks || publication_validate_storage(control))return 1;
    auto& bank=*publication_bank(control,0);auto& inactive=*publication_bank(control,1);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    if(bank.header.abi || inactive.header.abi || bank.header.sealed_epoch || bank.header.base_word ||
       bank.header.publication_word || bank.header.neural_bank || contract.abi!=1 || contract.window_capacity!=32 ||
       contract.semantic_owner!=descriptor.arena[1] || bank.header.semantic_owner!=descriptor.arena[1] ||
       !contract.model_generation || bank.header.model_generation>UINT32_MAX ||
       (restored ? bank.header.model_generation<contract.model_generation : bank.header.model_generation!=contract.model_generation) ||
       bank.header.authority_generation!=contract.authority_generation ||
        !publication_identity_equal(contract.task_identity,task+2) || task[0]!=4 || task[1]!=descriptor.arena[1] ||
       !publication_identity_equal(control.instance,bank.header.instance) || bank.header.range_count!=inactive.header.range_count ||
       (contract.terminal_token_count && !contract.terminal_tokens))return 1;
    uint64_t end=0;
    if(publication_validate_source(bank,contract,&end) || publication_source_records(control,bank))return 1;
    for(uint32_t index=0;index<2;++index)
        if(publication_validate_directory(control,contract,reinterpret_cast<PublicationRange*>(control.directories[index]),bank.header.range_count,false))return 1;
    if(publication_validate_destinations(control,reinterpret_cast<const PublicationRange*>(control.directories[0]),
        reinterpret_cast<const PublicationRange*>(control.directories[1]),bank.header.range_count) ||
       publication_validate_intents(control,reinterpret_cast<const PublicationRange*>(control.directories[0]),bank.header.range_count))return 1;
    const auto* book_range=publication_find_range(reinterpret_cast<const PublicationRange*>(control.directories[0]),bank.header.range_count,48);
    if(!book_range || book_range->length_bytes%8 || book_range->length_bytes<25*8)return 1;
    const auto* actual_books=reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*book_range));
    const auto* descriptor_books=reinterpret_cast<const uint64_t*>(descriptor.codebooks);
    for(uint64_t i=0;i<book_range->length_bytes/8;++i)if(actual_books[i]!=descriptor_books[i])return 1;
    semantic_graph::ResidentHandle root={bank.header.semantic_owner,semantic_graph::kRootKind,bank.header.semantic_slot,bank.header.semantic_generation};
    uint64_t expected_logical[4],expected_state[4];
    semantic_graph::copy_identity(expected_logical,bank.header.logical_digest);
    semantic_graph::copy_identity(expected_state,bank.header.state_digest);
    // Restored receipts are historical evidence, never fresh runtime handles.
    // Validate the rematerialized root without rewriting that saved history.
    State preflight{};
    State* checked=restored ? &preflight : &bank.state;
    if(!restored)bank.state=*reinterpret_cast<const State*>(descriptor.state);
    semantic_call(descriptor,semantic_graph::kResidentPreflightTransitionAdmission,&root,nullptr,&checked->semantic_receipts[0]);
    if(checked->semantic_receipts[0].words[0]!=semantic_graph::kOk ||
       !publication_identity_equal(checked->semantic_receipts[0].words+16,bank.header.semantic_digest))return 6;
    for(uint32_t i=0;i<3;++i) {
        const uint64_t extent=checked->semantic_receipts[0].words[13+i];
        if(restored && bank.header.semantic_extents[i]!=extent)return 6;
        if(!restored)bank.header.semantic_extents[i]=extent;
    }
    if(!task_queries(descriptor,task,0,&root,nullptr,checked))return 6;
    if(!checked->task_evaluation.facts[0].eligible)return 1;
    if(!restored) {
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[0]);
        const auto* prefix=publication_find_range(ranges,bank.header.range_count,3);
        const auto* source=publication_find_range(ranges,bank.header.range_count,1);
        if(!prefix || !source || prefix->length_bytes!=32)return 1;
        auto* identity=publication_range_bytes(control,*prefix);
        // The validated directory owns this output until the first seal. Use
        // the successor publication's source digest, never caller identity bytes.
        if(!identity || publication_range_digest(control,*source,nullptr,
            reinterpret_cast<uint64_t*>(identity)))return 1;
    }
    if(publication_write_feedback(descriptor,control,bank,root,restored) ||
       (!restored && publication_write_coverage(control,contract,bank,nullptr)))return 1;
    // The reader gate is held throughout initialization. Hash the valid ABI
    // representation, then invalidate it on any refusal before releasing that
    // gate; no reader can acquire this intermediate bank.
    bank.header.abi=1;
    if(publication_seal_ranges(control,bank,nullptr,restored) || publication_logical_digest(control,bank) ||
       publication_descriptor_digest(control,bank)){bank.header.abi=0;return 1;}
    if(restored && (!publication_identity_equal(expected_logical,bank.header.logical_digest) ||
        !publication_identity_equal(expected_state,bank.header.state_digest))){bank.header.abi=0;return 1;}
    return 0;
}
__device__ void publication_command(const Descriptor& descriptor) {
    if(!descriptor.publication.control)return;
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    uint64_t status=1;
    if(descriptor.publication.operation==1 || descriptor.publication.operation==4) {
        if(!publication_compare_exchange(control.reader_gate,0,1))status=2;
        else {status=publication_initialize(descriptor,control,descriptor.publication.operation==4);publication_store(control.reader_gate,0);}
    } else if(descriptor.publication.lease) {
        auto& lease=*reinterpret_cast<PublicationLease*>(descriptor.publication.lease);
        if(descriptor.publication.operation==2)status=publication_acquire(control,lease);
        if(descriptor.publication.operation==3)status=publication_release(control,lease);
        lease.status=status;
    }
    control.refusal=status;
}
__device__ uint64_t publication_begin(const Descriptor& descriptor,State* state,PublicationBank** acquired,
        uint64_t* word,uint64_t* end,bool* held,bool* staged) {
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    *held=false;*staged=false;*acquired=nullptr;
    if(control.abi!=1 || !control.contract || !control.continuation || !descriptor.task)return 1;
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    *held=true;*word=publication_load(control.word);
    const uint64_t index=*word&1;
    if(!control.banks[0] || !control.banks[1])return 1;
    PublicationBank& bank=*publication_bank(control,*word);
    PublicationBank& next=*publication_bank(control,*word^1);
    if(!publication_bank_matches(control,bank,*word))return 3;
    *acquired=&bank;
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
    const uint64_t eligibility=publication_header_eligibility(control,bank,*word,pending.transition_kind);
    if(eligibility)return eligibility;
    if(descriptor.publication.lease) {
        const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
        if(lease.transition_kind && (publication_acquired_bank(control,lease)!=&bank ||
           lease.word!=*word || lease.transition_kind!=pending.transition_kind))return 3;
    }
    if(pending.transition_kind==1 && (!descriptor.policy.z || descriptor.text.rows!=pending.text.rows || descriptor.text.count!=pending.text.count ||
       descriptor.text.selected!=pending.text.selected || !validate_text_binding(bank.source,descriptor.text)))return 1;
    if(pending.base_word!=*word)return 3;
    if(contract.abi!=1 || bank.header.semantic_owner!=descriptor.arena[1] ||
       publication_validate_selected_model(control,bank) || bank.header.authority_generation!=contract.authority_generation ||
       bank.header.model_generation>UINT32_MAX || bank.header.neural_bank>1 || bank.header.proposal>UINT32_MAX ||
       bank.header.stream_serial>(UINT64_MAX>>8) || bank.header.family_id>255 ||
       bank.header.cache_generation==UINT64_MAX || !publication_identity_equal(contract.task_identity,
           reinterpret_cast<const uint64_t*>(descriptor.task)+2))return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[index]);
    auto* destinations=reinterpret_cast<PublicationRange*>(control.directories[index^1]);
    if(publication_validate_source(bank,contract,end) || publication_source_records(control,bank) ||
       publication_validate_directory(control,contract,ranges,bank.header.range_count,false) ||
       publication_validate_directory(control,contract,destinations,bank.header.range_count,false) ||
       publication_validate_destinations(control,ranges,destinations,bank.header.range_count) ||
       (pending.transition_kind!=3 &&
        publication_validate_continuation(control,contract,bank,pending,*end)))return 1;
    if(bank.header.terminal==1 && *end!=bank.header.terminal_position+1)return 1;
    const auto* source=publication_find_range(ranges,bank.header.range_count,1);
    const auto* provenance=publication_find_range(ranges,bank.header.range_count,2);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t masks=0,terminal_bound=0;bool terminal_possible=false;
    const auto* components=reinterpret_cast<const Component*>(descriptor.components);
    const auto* support=reinterpret_cast<const uint8_t*>(descriptor.support);
    const auto* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    for(uint32_t i=0;i<32;++i)if(pending.transition_kind==1 && text_selected(pending.text,i)) {
        ++masks;
        for(uint32_t lane=0;lane<2;++lane)for(uint64_t t=0;t<contract.terminal_token_count;++t)
            if(terminals[t]<components[lane*68+i].cardinality && support[components[lane*68+i].offset+terminals[t]]) {
                terminal_possible=true;
                if(bank.source[i].logical_position>terminal_bound)terminal_bound=bank.source[i].logical_position;
            }
    }
    if(!source || !provenance || *end>(storage[source->storage_slot].bytes-source->offset_bytes)/sizeof(SourceSlot) ||
       provenance->length_bytes>storage[provenance->storage_slot].bytes-provenance->offset_bytes ||
       masks>(storage[provenance->storage_slot].bytes-provenance->offset_bytes-provenance->length_bytes)/sizeof(TokenProvenanceRecord))return 5;
    uint64_t intent_status=publication_validate_intents(control,ranges,bank.header.range_count,
        bank.header.terminal==1 ? bank.header.terminal_position : (terminal_possible ? terminal_bound : UINT64_MAX));
    if(intent_status)return intent_status;
    // The inactive bank and its exclusive semantic root can be reused only after
    // its reader count is zero. Remaining descendants are traced by the owner.
    if(next.header.abi==1 && next.header.semantic_slot!=0 &&
       (next.header.semantic_slot!=bank.header.semantic_slot || next.header.semantic_generation!=bank.header.semantic_generation)) {
        semantic_graph::Command command{};command.words[0]=semantic_graph::kRetireRoot;
        command.words[1]=descriptor.arena[1];command.words[8]=next.header.semantic_slot;command.words[9]=next.header.semantic_generation;
        command.words[10]=bank.header.semantic_slot;command.words[11]=bank.header.semantic_generation;
        semantic_graph::LaunchDescriptor launch{};
        launch.expected_owner=descriptor.arena[1];launch.root_capacity=descriptor.arena[2];
        launch.statement_capacity=descriptor.arena[3];launch.support_capacity=descriptor.arena[4];
        launch.version_capacity=descriptor.arena[5];launch.arena_words=descriptor.arena[6];
        launch.command_ptr=reinterpret_cast<uint64_t>(&command);
        launch.receipt_ptr=reinterpret_cast<uint64_t>(&state->cleanup_receipts[0]);
        launch.abi_generation=semantic_graph::kAbiGeneration;
        semantic_graph::NativeWorkTally actual{};
        semantic_graph::execute(reinterpret_cast<uint64_t*>(descriptor.arena[0]),launch,&actual);
        execution_work_merge_native(state->execution_work,actual);
        if(state->cleanup_receipts[0].words[0]!=semantic_graph::kOk)return 6;
    }
    next.header.abi=0;*staged=true;
    return 0;
}
__device__ uint64_t publication_append_provenance(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next,const Receipt* receipts,uint64_t winner,TextBinding text) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    auto* range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,2));
    const auto* closure=publication_find_range(ranges,next.header.range_count,17);
    if(!range || !closure || range->length_bytes%sizeof(TokenProvenanceRecord))return 1;
    const uint64_t old_count=range->length_bytes/sizeof(TokenProvenanceRecord);
    uint64_t count=old_count;
    for(uint32_t i=0;i<32;++i)if(winner && text_selected(text,i))++count;
    range->length_bytes=count*sizeof(TokenProvenanceRecord);range->logical_begin=0;range->logical_end=count;
    auto* records=reinterpret_cast<TokenProvenanceRecord*>(publication_range_bytes(control,*range));
    if(!records)return 5;
    count=old_count;
    for(uint32_t i=0;i<32;++i)if(winner && text_selected(text,i)) {
        auto& record=records[count];record=TokenProvenanceRecord{};
        record.source_slot=i;record.logical_position=next.source[i].logical_position;record.token=next.source[i].token;
        record.base_word=base.header.publication_word;record.proposal=next.state.proposal;record.ordinal=(winner-1)*68+i;
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipts[record.ordinal]),sizeof(Receipt),record.action_receipt_digest);
        record.authority_generation=base.header.authority_generation;
        semantic_graph::copy_identity(record.origin_logical_digest,base.header.logical_digest);
        semantic_graph::copy_identity(record.authority_closure_digest,closure->digest);
        TokenProvenanceRecord logical=record;logical.base_word=0;
        if(publication_action_digest(control,next,receipts+record.ordinal,1,logical.action_receipt_digest))return 1;
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&logical),sizeof(logical),record.record_digest);
        next.source[i].provenance_record=count++;
    }
    return 0;
}
__device__ uint64_t publication_append_final_intent(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    auto* entry_range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,30));
    auto* payload_range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,31));
    const auto* prefix=publication_find_range(ranges,next.header.range_count,1);
    const auto* lineage=publication_find_range(ranges,next.header.range_count,40);
    if(!entry_range || !payload_range || !prefix || !lineage || next.header.terminal!=2 ||
       next.header.prefix_extent!=next.header.terminal_position+1)return 1;
    auto* queue=reinterpret_cast<IntentQueueHeader*>(publication_range_bytes(control,*entry_range));
    const uint64_t payload_bytes=8+next.header.prefix_extent*8;
    if(!queue || queue->count>=queue->capacity || payload_bytes>queue->payload_capacity_bytes-queue->payload_used_bytes)return 5;
    const uint64_t payload_offset=queue->payload_used_bytes;
    entry_range->length_bytes=sizeof(IntentQueueHeader)+(queue->count+1)*sizeof(IntentEntry);
    payload_range->length_bytes=payload_offset+payload_bytes;
    if(!publication_range_bytes(control,*entry_range))return 5;
    auto* payload=publication_range_bytes(control,*payload_range);
    const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,*prefix));
    if(!payload || !source)return 1;
    // Authorized effect descriptors have arbitrary byte lengths; the following
    // token payload is explicitly little-endian and need not be aligned.
    for(uint32_t b=0;b<8;++b)payload[payload_offset+b]=uint8_t(next.header.prefix_extent>>(8*b));
    for(uint64_t i=0;i<next.header.prefix_extent;++i) {
        if(source[i].kind!=1 || source[i].logical_position!=i || !source[i].committed || !source[i].recomputed)return 1;
        for(uint32_t b=0;b<8;++b)payload[payload_offset+8*(i+1)+b]=uint8_t(source[i].token>>(8*b));
    }
    auto& entry=reinterpret_cast<IntentEntry*>(queue+1)[queue->count];entry=IntentEntry{};
    semantic_graph::copy_identity(entry.checkpoint_lineage,lineage->digest);
    semantic_graph::copy_identity(entry.base_logical,base.header.logical_digest);
    semantic_graph::copy_identity(entry.result_logical,next.header.logical_digest);
    uint64_t delta[9];delta[0]=0x786c6f6764656c31ULL;
    semantic_graph::copy_identity(delta+1,base.header.logical_digest);semantic_graph::copy_identity(delta+5,next.header.logical_digest);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(delta),sizeof(delta),entry.effective_delta);
    semantic_graph::sha256(payload+queue->effect_offset_bytes,queue->effect_length_bytes,entry.effect_digest);
    semantic_graph::sha256(payload+payload_offset,payload_bytes,entry.payload_digest);
    entry.payload_offset=payload_offset;entry.payload_len=payload_bytes;
    semantic_graph::copy_identity(entry.audit_instance,next.header.instance);entry.audit_epoch=next.header.sealed_epoch;
    semantic_graph::copy_identity(entry.previous_chain,queue->chain_head);
    publication_intent_identity(entry,entry.stable_identity);
    publication_fold(queue->chain_head,30,queue->count,entry.stable_identity);
    semantic_graph::copy_identity(entry.chain,queue->chain_head);
    ++queue->count;queue->payload_used_bytes+=payload_bytes;
    entry_range->logical_begin=0;entry_range->logical_end=queue->count;
    payload_range->logical_begin=0;payload_range->logical_end=queue->payload_used_bytes;
    if(publication_range_digest(control,*entry_range,nullptr,entry_range->digest) ||
       publication_range_digest(control,*payload_range,nullptr,payload_range->digest))return 1;
    return 0;
}
__device__ void publication_snapshot(const Descriptor& descriptor,const semantic_graph::ResidentHandle& root,
        semantic_graph::Receipt* receipt,ExecutionWork* work=nullptr) {
    semantic_graph::Command command{};command.words[0]=semantic_graph::kSnapshot;
    command.words[1]=root.owner;command.words[7]=1;command.words[8]=root.slot;command.words[9]=root.generation;
    semantic_graph::LaunchDescriptor launch{};
    launch.expected_owner=descriptor.arena[1];launch.root_capacity=descriptor.arena[2];
    launch.statement_capacity=descriptor.arena[3];launch.support_capacity=descriptor.arena[4];
    launch.version_capacity=descriptor.arena[5];launch.arena_words=descriptor.arena[6];
    launch.command_ptr=reinterpret_cast<uint64_t>(&command);launch.receipt_ptr=reinterpret_cast<uint64_t>(receipt);
    launch.abi_generation=semantic_graph::kAbiGeneration;
    semantic_graph::NativeWorkTally actual{};
    semantic_graph::execute(reinterpret_cast<uint64_t*>(descriptor.arena[0]),launch,work ? &actual : nullptr);
    if(work)execution_work_merge_native(*work,actual);
}
__device__ void publication_refusal_snapshot(const Descriptor& descriptor,
        const semantic_graph::ResidentHandle& root,const PublicationBank* bank,State* state) {
    auto& receipt=state->semantic_receipts[0];
    publication_snapshot(descriptor,root,&receipt,&state->execution_work);
    if(receipt.words[0]!=semantic_graph::kOk || receipt.words[39]!=root.owner ||
       receipt.words[3]!=root.slot || receipt.words[4]!=root.generation) {
        semantic_content_integrity_trap();return;
    }
    if(bank) {
        if(!publication_identity_equal(receipt.words+16,bank->header.semantic_digest)) {
            semantic_content_integrity_trap();return;
        }
        for(uint32_t i=0;i<3;++i)if(receipt.words[13+i]!=bank->header.semantic_extents[i]) {
            semantic_content_integrity_trap();return;
        }
    }
}
__device__ uint64_t publication_publish(const Descriptor& descriptor,PublicationBank& base,uint64_t word,uint64_t end,
        const semantic_graph::ResidentHandle& base_root,State* state,const Receipt* receipts,bool no_draw=false,bool drain=false) {
    if(state->execution_work.overflow) { state->status=11;return 1; }
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
    auto& next=*publication_bank(control,word^1);
    const uint64_t winner=no_draw ? 0 : state->task_evaluation.winner;
    if(state->blocks!=(no_draw ? 0 : 136) || winner>2 || publication_prepare_text(base,next,contract,receipts,winner,end,descriptor.text))return 1;
    semantic_graph::ResidentHandle root=base_root;
    const semantic_graph::Receipt* seal=&state->semantic_receipts[0];
    if(winner) {
        seal=&state->semantic_receipts[3+(winner-1)*3];
        if(seal->words[0]!=semantic_graph::kOk || seal->words[33]!=semantic_graph::kRootKind)return 6;
        root={seal->words[39],semantic_graph::kRootKind,seal->words[34],seal->words[35]};
    }
    if(!retire_pending_roots(descriptor,base_root,state,winner,&root))return 6;
    next.header.abi=1;next.header.sealed_epoch=(word>>1)+1;next.header.base_word=word;
    next.header.publication_word=(next.header.sealed_epoch<<1)|((word&1)^1);
    next.header.semantic_owner=root.owner;next.header.semantic_slot=root.slot;next.header.semantic_generation=root.generation;
    semantic_graph::copy_identity(next.header.semantic_digest,seal->words+16);
    for(uint32_t i=0;i<3;++i)next.header.semantic_extents[i]=seal->words[13+i];
    next.header.proposal=state->proposal+uint64_t(!no_draw);next.header.cache_generation=base.header.cache_generation+1;
    next.header.fuel=base.header.fuel-1;
    if(pending.transition_kind==4) {
        if(base.header.model_generation>=UINT32_MAX || base.header.neural_bank>1 ||
           base.header.neural_generation==UINT64_MAX || base.header.training_cursor==UINT64_MAX)return 1;
        next.header.model_generation=base.header.model_generation+1;
        next.header.neural_bank=base.header.neural_bank^1;
        next.header.neural_generation=base.header.neural_generation+1;
        next.header.training_cursor=base.header.training_cursor+1;
    }
    if(drain)next.header.terminal=2;
    next.state=*state;next.state.next_proposal=next.header.proposal;
    for(uint32_t i=0;i<136;++i)next.receipts[i]=no_draw ? base.receipts[i] : receipts[i];
    if(publication_apply_continuation(control,base,next,pending) ||
       publication_append_provenance(control,base,next,receipts,winner,descriptor.text))return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[word&1]);
    auto* intent=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,30));
    const auto* old_intent=publication_find_range(previous,base.header.range_count,30);
    if(!intent || !old_intent)return 1;
    intent->length_bytes=old_intent->length_bytes;intent->logical_begin=old_intent->logical_begin;intent->logical_end=old_intent->logical_end;
    uint8_t* intent_bytes=publication_range_bytes(control,*intent);
    if(!intent_bytes)return 1;
    publication_copy_bytes(intent_bytes,publication_range_bytes(control,*old_intent),old_intent->length_bytes);
    const auto* prefix=publication_find_range(ranges,next.header.range_count,3);
    const auto* source=publication_find_range(ranges,next.header.range_count,1);
    if(!prefix || !source || prefix->length_bytes!=32)return 1;
    if(no_draw) {
        auto* feedback=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,15));
        const auto* old_feedback=publication_find_range(previous,base.header.range_count,15);
        if(!feedback || !old_feedback || feedback->length_bytes!=old_feedback->length_bytes)return 1;
        publication_copy_bytes(publication_range_bytes(control,*feedback),publication_range_bytes(control,*old_feedback),feedback->length_bytes);
        feedback->logical_begin=old_feedback->logical_begin;feedback->logical_end=old_feedback->logical_end;
    } else if(publication_write_feedback(descriptor,control,next,root,false,&state->execution_work))return 1;
    // Feedback queries and root retirement belong to this attempt even when
    // they follow candidate selection. Snapshot their actual accumulated work.
    next.state.execution_work=state->execution_work;
    if(publication_range_digest(control,*source,nullptr,reinterpret_cast<uint64_t*>(publication_range_bytes(control,*prefix))) ||
       publication_write_coverage(control,contract,next,drain ? nullptr : &pending) || publication_seal_ranges(control,next,&base) ||
       publication_logical_digest(control,next))return 1;
    if(drain && publication_append_final_intent(control,base,next))return 1;
    if(publication_write_attempt(control,next,base,no_draw))return 1;
    auto* attempt=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,33));
    if(!attempt || publication_range_digest(control,*attempt,nullptr,attempt->digest) ||
       publication_descriptor_digest(control,next))return 1;
    // No host selector, retry, rebase, or secondary canonical success flag. All
    // winning text, caches, semantics, receipts and RNG become visible together.
    if(state->execution_work.overflow) { state->status=11;return 1; }
    if(!publication_compare_exchange(control.word,word,next.header.publication_word))return 3;
    state->next_proposal=next.header.proposal;
    return 0;
}
// Service provenance is separate from the learned feedback coordinates. Rows
// are [original target, original support, support slot, support generation,
// version slot, version generation, insertion ordinal, polarity, digest[4]].
// They are copied from the actual current chain, never from an action roster.
struct FeedbackSupportCollector {
    uint64_t* rows;
    uint64_t capacity, count, refusal;
    __device__ bool operator()(uint64_t slot,const uint64_t* version,const uint64_t* support) {
        if(count>=capacity || support[11]==UINT32_MAX) { refusal=1;return false; }
        if(support[10]==UINT32_MAX) { refusal=2;return false; }
        uint64_t* row=rows+12*count++;
        row[0]=support[10];row[1]=support[11];row[2]=version[5];row[3]=support[1];
        row[4]=slot;row[5]=version[1];row[6]=version[13];row[7]=support[5];
        semantic_graph::copy_identity(row+8,support+6);
        return true;
    }
};

struct PublicationFeedbackInputs {
    const PublicationBank* bank;
    const RawFeedbackRecord* records;
    semantic_feedback::Statement statements[3];
    uint64_t statement_capacity;
};

// Resolve every source through the actual acquired reader. The publication
// word may advance while this reader retains its original bank and seals.
__device__ uint64_t publication_feedback_inputs(uint64_t control_ptr,
        const PublicationLease* lease,uint64_t slots,PublicationFeedbackInputs* inputs) {
    const uint64_t lease_ptr=reinterpret_cast<uint64_t>(lease);
    if(!control_ptr || control_ptr%alignof(PublicationControl) || control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) || lease_ptr>UINT64_MAX-sizeof(PublicationLease) ||
        slots<3 || slots>UINT64_MAX/sizeof(RawFeedbackRecord))return 1;
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    if(lease->bank>1 || !control.contract || control.contract%alignof(PublicationContract) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract) ||
       !control.banks[lease->bank] || control.banks[lease->bank]%alignof(PublicationBank) ||
       control.banks[lease->bank]>UINT64_MAX-sizeof(PublicationBank) ||
       !control.directories[lease->bank] || control.directories[lease->bank]%alignof(PublicationRange) ||
       !control.storage || control.storage%alignof(PublicationStorageEntry) || !control.storage_count ||
       control.storage_count>(UINT64_MAX-control.storage)/sizeof(PublicationStorageEntry))return 1;
    const auto* bank=publication_acquired_bank(control,*lease);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(!bank || contract.abi!=1 || contract.feedback_capacity!=slots ||
       bank->header.range_count>contract.range_capacity ||
       bank->header.range_count>(UINT64_MAX-control.directories[lease->bank])/sizeof(PublicationRange))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease->bank]);
    const auto* records=publication_find_range(ranges,bank->header.range_count,15);
    if(!records || records->length_bytes!=slots*sizeof(RawFeedbackRecord) ||
       records->logical_begin || records->logical_end>slots)return 1;
    const auto* bytes=publication_range_bytes(control,*records);
    uint64_t digest[4];
    if(!bytes || reinterpret_cast<uint64_t>(bytes)%alignof(RawFeedbackRecord) ||
       publication_range_digest(control,*records,nullptr,digest) ||
       !publication_identity_equal(digest,records->digest))return 1;
    inputs->bank=bank;
    inputs->records=reinterpret_cast<const RawFeedbackRecord*>(bytes);
    inputs->statement_capacity=0;
    for(uint64_t index=0;index<3;++index) {
        const auto* statement=publication_find_range(ranges,bank->header.range_count,16,index);
        if(!statement || !statement->length_bytes)return 1;
        const auto* data=publication_range_bytes(control,*statement);
        if(!data || publication_range_digest(control,*statement,nullptr,digest) ||
           !publication_identity_equal(digest,statement->digest))return 1;
        inputs->statements[index]={data,statement->length_bytes};
        if(statement->length_bytes>inputs->statement_capacity)inputs->statement_capacity=statement->length_bytes;
    }
    return 0;
}

__device__ uint64_t publication_feedback_lineage(const Descriptor& descriptor,
        const PublicationLease* lease,uint64_t slots,
        semantic_graph::Receipt* queries,uint64_t* original_statements,uint64_t* offsets,
        uint64_t* supports,uint64_t support_capacity) {
    for(uint64_t slot=0;slot<slots;++slot) {
        queries[slot]=semantic_graph::Receipt{};original_statements[slot]=UINT64_MAX;offsets[slot]=0;
    }
    offsets[slots]=0;
    for(uint64_t i=0;i<support_capacity*12;++i)supports[i]=0;
    PublicationFeedbackInputs inputs{};
    if(publication_feedback_inputs(descriptor.publication.control,lease,slots,&inputs) || !descriptor.task)return 1;
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    if(task[0]!=4 || task[1]!=descriptor.arena[1] ||
       task[22]>=UINT32_MAX || task[23]>=UINT32_MAX || task[24]>=UINT32_MAX)return 1;
    const auto& bank=*inputs.bank;
    semantic_graph::Layout layout{};
    if(!semantic_graph::build_layout(reinterpret_cast<uint64_t*>(descriptor.arena[0]),
        descriptor.arena[2],descriptor.arena[3],descriptor.arena[4],descriptor.arena[5],
        descriptor.arena[6],&layout) || support_capacity!=layout.support_capacity)return 1;
    const semantic_graph::ResidentHandle root{bank.header.semantic_owner,semantic_graph::kRootKind,
        bank.header.semantic_slot,bank.header.semantic_generation};
    FeedbackSupportCollector collector{supports,support_capacity,0,0};
    for(uint64_t slot=0;slot<slots;++slot) {
        offsets[slot]=collector.count;
        const auto& saved=inputs.records[slot];
        if(saved.valid==0) { offsets[slot+1]=collector.count;continue; }
        if(saved.valid!=1 || saved.statement_index>2 || saved.pro>1 || saved.contra>1)return 1;
        semantic_graph::DecodedStatement statement{};
        statement.record=uint32_t(task[22+saved.statement_index]);
        for(uint32_t i=0;i<8;++i)
            statement.identity_words[i]=uint32_t(task[6+4*saved.statement_index+i/2]>>(32*(i%2)));
        auto& query=queries[slot];
        semantic_call(descriptor,semantic_graph::kResidentRootTruthAdmission,&root,nullptr,&query,&statement);
        if(query.words[0]!=semantic_graph::kOk || query.words[2]!=(saved.pro|(saved.contra<<1)) ||
           query.words[39]!=bank.header.semantic_owner || query.words[33]!=semantic_graph::kRootKind ||
           query.words[34]!=bank.header.semantic_slot || query.words[35]!=bank.header.semantic_generation ||
           !publication_identity_equal(query.words+16,bank.header.semantic_digest))return 1;
        for(uint32_t i=0;i<3;++i)if(query.words[13+i]!=bank.header.semantic_extents[i])return 1;
        // Cold restoration deliberately preserves archived runtime handles.
        // Stable semantic fields still have to agree with the fresh query.
        for(uint32_t i=0;i<semantic_graph::kReceiptWords;++i)
            if(!publication_receipt_runtime_word(i) && saved.query_receipt.words[i]!=query.words[i])return 1;
        if((query.words[8]!=0)!=(query.words[12]!=0) ||
           (query.words[2]!=0)!=(query.words[8]!=0))return 1;
        original_statements[slot]=statement.record;
        if(query.words[12]) {
            semantic_graph::Receipt checked{};
            if(!semantic_graph::visit_support_chain(layout,query.words[7],query.words[11]+1,
                    false,collector,&checked))return collector.refusal ? collector.refusal : 1;
        }
        offsets[slot+1]=collector.count;
    }
    return 0;
}

extern "C" __global__ void semantic_feedback_project(Descriptor descriptor,
        const PublicationLease* lease,uint64_t slots,
        semantic_graph::Receipt* queries,uint64_t* original_statements,uint64_t* offsets,
        uint64_t* supports,uint64_t support_capacity,uint64_t* status) {
    if(blockIdx.x || threadIdx.x)return;
    // The preceding encoder initializes every status on this same stream.
    // Consumers follow the projection's event; rejected input must never
    // become usable merely because no host reads its private status buffer.
    for(uint64_t slot=0;slot<slots;++slot)if(status[slot]) {
        semantic_content_integrity_trap();return;
    }
    const uint64_t error=publication_feedback_lineage(descriptor,lease,slots,queries,
        original_statements,offsets,supports,support_capacity);
    // Keep the original private diagnostic while stopping every dependent
    // consumer, including when an actual contributor has no original target.
    if(error) {
        for(uint64_t slot=0;slot<slots;++slot)status[slot]=(uint64_t(1)<<32)|error;
        semantic_content_integrity_trap();
    }
}

// Inputs are original sealed ranges selected by the device-acquired reader.
// Retained output capacity preserves every physical slot, including invalid gaps.
extern "C" __global__ void semantic_feedback_encode(uint64_t control_ptr,
        const PublicationLease* lease,uint64_t capacity,
        uint64_t slots, float* features, uint8_t* validity, uint64_t* status) {
    if(!capacity || capacity>(UINT64_MAX-4)/9 || !slots ||
       slots>UINT64_MAX/(9*capacity+4)/sizeof(float)) { semantic_content_integrity_trap();return; }
    const uint64_t width = 9 * capacity + 4;
    __shared__ PublicationFeedbackInputs inputs;
    __shared__ uint64_t input_error;
    if(threadIdx.x==0) {
        input_error=publication_feedback_inputs(control_ptr,lease,slots,&inputs);
        if(!input_error && inputs.statement_capacity!=capacity)input_error=1;
    }
    __syncthreads();
    for (uint64_t slot = blockIdx.x; slot < slots; slot += gridDim.x) {
        const uint64_t error = input_error ? (uint64_t(1)<<32)|input_error :
            semantic_feedback::validate(inputs.records[slot], inputs.statements, capacity);
        if (threadIdx.x == 0) {
            status[slot] = error;
            validity[slot] = error ? 0 : uint8_t(inputs.records[slot].valid);
        }
        for (uint64_t column = threadIdx.x; column < width; column += blockDim.x)
            features[slot * width + column] = error ? 0.0f :
                semantic_feedback::coordinate(inputs.records[slot], inputs.statements, capacity, column);
    }
}

extern "C" __global__ void semantic_transition_execute(Descriptor descriptor) {
    if(descriptor.publication.operation) {
        if(threadIdx.x==0)publication_command(descriptor);
        return;
    }
#ifdef XLOG_SEMANTIC_POLICY
    if(descriptor.backward.cotangents) {
        semantic_policy_backward(descriptor);
        __syncthreads();
        if(threadIdx.x==0)semantic_policy_backward_status_guard(descriptor.backward.status);
        return;
    }
#endif
    semantic_graph::NativeWorkTally sampler_work{};
    auto logits=reinterpret_cast<const float*>(descriptor.logits);
    auto support=reinterpret_cast<const uint8_t*>(descriptor.support);
    auto sorted=reinterpret_cast<uint32_t*>(descriptor.scratch);
    // Reuse the arena for one component only; no per-component CDF banks.
    auto legal=sorted+262144;
    auto receipts=reinterpret_cast<Receipt*>(descriptor.receipts);
    auto state=reinterpret_cast<State*>(descriptor.state);
    auto components=reinterpret_cast<const Component*>(descriptor.components);
    auto books=reinterpret_cast<const uint64_t*>(descriptor.codebooks);
    auto task=reinterpret_cast<const uint64_t*>(descriptor.task);
    auto viability=legal+262144;
    __shared__ semantic_graph::ResidentHandle base;
    __shared__ PublicationBank* acquired_bank;
    __shared__ uint64_t acquired_word,structural_end,transition_kind;
    __shared__ bool publication_held,publication_staged,numerical_refused;
    __shared__ uint32_t count,invalid,width,failed;
    __shared__ uint32_t choices[18],actions[4][2],lane_live;
    __shared__ semantic_graph::Receipt candidate;
    if(threadIdx.x==0) {
        base={books[14],semantic_graph::kRootKind,books[15],books[16]};
        acquired_bank=nullptr;publication_held=publication_staged=false;acquired_word=structural_end=0;
        numerical_refused=false;
        transition_kind=1;
        state->blocks=0; state->status=0; state->proposal=state->next_proposal;
        state->importance_weight=0.0;
        state->retired_roots_mask=0;
        state->task_evaluation=TaskEvaluation{};
        // Drain never executes the model body. Other modes consume the common
        // original forward, including numerical refusals.
        state->execution_work=ExecutionWork{};
        if(!descriptor.publication.control)consume_model_work(descriptor.model_work,state->execution_work);
        for(uint32_t i=0;i<3;++i)semantic_graph::clear_receipt(&state->cleanup_receipts[i]);
        lane_live=0;
        for(uint32_t i=0;i<7;++i)semantic_graph::clear_receipt(&state->semantic_receipts[i]);
        for(uint32_t lane=0;lane<2;++lane)state->work[lane]=Work{};
        // Published execution validates its proposal on the acquired bank below.
        failed=state->catalogue_generation!=CATALOGUE_GENERATION ||
            (!descriptor.publication.control && state->next_proposal>0xffffffffULL);
        for(int j=0;j<32;++j)failed |= state->catalogue_digest[j]!=CATALOGUE_DIGEST[j];
        for(int j=0;j<32;++j)failed |= state->binding_digest[j]!=reinterpret_cast<const uint8_t*>(books+21)[j];
        if(failed)state->status=1;
        if(!failed && descriptor.publication.control) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(!control.continuation) { semantic_content_integrity_trap();return; }
            const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
            transition_kind=pending.transition_kind;
            uint8_t numerical=1;
            if(transition_kind==3) {
                if(pending.numerical_admissibility || pending.training_selection ||
                   pending.model_update_bindings || pending.model_update_binding_count ||
                   pending.model_update_admissibility) {
                    semantic_content_integrity_trap();return;
                }
            } else {
                if(!pending.numerical_admissibility || pending.numerical_admissibility==UINT64_MAX) {
                    semantic_content_integrity_trap();return;
                }
                numerical=*reinterpret_cast<const uint8_t*>(pending.numerical_admissibility);
                if(numerical>1) { semantic_content_integrity_trap();return; }
                consume_model_work(descriptor.model_work,state->execution_work);
            }
            uint64_t training_status=0;
            if(pending.transition_kind==4) {
                if(!pending.training_selection || pending.training_selection%alignof(uint64_t) ||
                    pending.training_selection>UINT64_MAX-sizeof(uint64_t) ||
                    !pending.model_update_bindings || !pending.model_update_binding_count ||
                    !pending.model_update_admissibility ||
                    pending.model_update_admissibility==UINT64_MAX) {
                    semantic_content_integrity_trap();return;
                }
                training_status=*reinterpret_cast<const uint64_t*>(pending.training_selection);
                if(training_status>3) { semantic_content_integrity_trap();return; }
                const uint8_t update_admissibility=
                    *reinterpret_cast<const uint8_t*>(pending.model_update_admissibility);
                if(update_admissibility>1) { semantic_content_integrity_trap();return; }
                numerical &= update_admissibility;
            } else if(pending.training_selection || pending.model_update_bindings ||
                      pending.model_update_binding_count || pending.model_update_admissibility) {
                semantic_content_integrity_trap();return;
            }
            if(training_status || !numerical || state->execution_work.overflow) {
                if(!descriptor.publication.lease || !control.contract) {
                    semantic_content_integrity_trap();return;
                }
                const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
                acquired_bank=publication_acquired_bank(control,lease);
                const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
                if(!acquired_bank || publication_load(control.word)!=lease.word ||
                   (lease.transition_kind && pending.transition_kind!=lease.transition_kind) ||
                   pending.abi!=1 || pending.base_word!=lease.word ||
                   !publication_identity_equal(pending.instance,lease.instance) || contract.abi!=1 ||
                   !publication_identity_equal(pending.topology_identity,contract.topology_identity) ||
                   !publication_identity_equal(pending.table_identity,contract.table_identity) ||
                    !task || task[0]!=4 || task[1]!=descriptor.arena[1] ||
                   !publication_identity_equal(contract.task_identity,task+2)) {
                    semantic_content_integrity_trap();return;
                }
                const auto& header=acquired_bank->header;
                if(header.semantic_owner!=descriptor.arena[1] || publication_validate_selected_model(control,*acquired_bank) ||
                   header.authority_generation!=contract.authority_generation ||
                   pending.model_generation!=header.model_generation || pending.authority_generation!=header.authority_generation ||
                   pending.transition_kind<1 || pending.transition_kind>4 ||
                   header.model_generation>UINT32_MAX || header.proposal>UINT32_MAX ||
                   header.stream_serial>(UINT64_MAX>>8) || header.family_id>255) {
                    semantic_content_integrity_trap();return;
                }
                base={header.semantic_owner,semantic_graph::kRootKind,header.semantic_slot,header.semantic_generation};
                state->model_generation=uint32_t(header.model_generation);state->family_id=uint32_t(header.family_id);
                state->stream_serial=header.stream_serial;state->proposal=state->next_proposal=header.proposal;
                publication_refusal_snapshot(descriptor,base,acquired_bank,state);
                numerical_refused=true;failed=1;
                state->status=state->execution_work.overflow ? 11 : (training_status ? 11+training_status : 5);
            }
        }
        if(!failed && descriptor.publication.control) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            control.refusal=publication_begin(descriptor,state,&acquired_bank,&acquired_word,&structural_end,&publication_held,&publication_staged);
            if(control.refusal){failed=1;state->status=10;}
            else {
                base={acquired_bank->header.semantic_owner,semantic_graph::kRootKind,
                    acquired_bank->header.semantic_slot,acquired_bank->header.semantic_generation};
                state->model_generation=uint32_t(acquired_bank->header.model_generation);
                state->family_id=uint32_t(acquired_bank->header.family_id);
                state->stream_serial=acquired_bank->header.stream_serial;
                state->proposal=state->next_proposal=acquired_bank->header.proposal;
                transition_kind=reinterpret_cast<const PendingContinuation*>(control.continuation)->transition_kind;
            }
        }
        if(!failed && task && (task[0]!=4 || task[1]!=descriptor.arena[1] ||
           task[18]>3 || task[19]>3 || task[20]>3)) {
            failed=1;state->status=7;
        }
        if(!failed) {
            if(acquired_bank && transition_kind!=1)
                publication_snapshot(descriptor,base,&state->semantic_receipts[0],&state->execution_work);
            else semantic_call(descriptor,semantic_graph::kResidentPreflightTransitionAdmission,&base,nullptr,&state->semantic_receipts[0],nullptr,nullptr,&state->execution_work);
            failed=state->semantic_receipts[0].words[0]!=0;
            if(!failed)for(uint32_t i=0;i<4;++i)failed |= state->semantic_receipts[0].words[16+i]!=
                (acquired_bank ? acquired_bank->header.semantic_digest[i] : books[17+i]);
            if(failed)state->status=4;
        }
        if(!failed && task && transition_kind==1) {
            state->execution_work.active_candidate=1;
            if(!task_queries(descriptor,task,0,&base,nullptr,state)) {
                failed=1;state->status=9;
            } else if(!state->task_evaluation.facts[0].eligible) {
                failed=1;state->status=8;
            }
            state->execution_work.active_candidate=0;
        }
    }
    __syncthreads();
    if(numerical_refused)return;
    if(acquired_bank && transition_kind!=1) {
        if(threadIdx.x==0) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(!failed) {
                state->importance_weight=1.0;
                control.refusal=publication_publish(descriptor,*acquired_bank,acquired_word,structural_end,base,state,receipts,true,transition_kind==3);
                if(control.refusal){failed=1;state->status=state->execution_work.overflow ? 11 : 10;}
            }
            if(failed && publication_staged)publication_bank(control,acquired_word^1)->header.abi=0;
            if(publication_held)publication_store(control.reader_gate,0);
        }
        return;
    }
    // Validate the full immutable input bank before any component draw. Live
    // conditional logits are checked at their own factor, after the unchanged
    // numerical producer has evaluated the actual selected prefix.
    if(threadIdx.x==0)invalid=0;
    __syncthreads();
    for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        const Component c=components[ordinal];
        if(failed)continue;
        const bool active_text=c.kind==COMPONENT_KIND_TEXT &&
            (!descriptor.policy.z || text_selected(descriptor.text,c.field));
        const uint64_t offset=descriptor.policy.z && active_text
            ? text_row_index(descriptor.text,c.field)*TEXT_CARDINALITY : c.offset;
        for(uint32_t i=threadIdx.x;i<c.cardinality;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            uint8_t value=support[c.offset+i];
            if(value>1)atomicOr(&invalid,4U);
            if(value && active_text && !isfinite(logits[offset+i]))atomicOr(&invalid,2U);
        }
        if(threadIdx.x==0 && active_text) {
            bool present=false;
            for(uint32_t i=0;i<c.cardinality;++i) {
                semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
                present |= support[c.offset+i]!=0;
            }
            if(!present)atomicOr(&invalid,1U);
        }
    }
#ifdef XLOG_SEMANTIC_POLICY
    if(descriptor.policy.z) {
        const auto p=descriptor.policy;
        for(uint32_t i=threadIdx.x;i<4*128;i+=blockDim.x)
            if(!isfinite(reinterpret_cast<const float*>(p.z)[i]))atomicOr(&invalid,2U);
        for(uint32_t i=threadIdx.x;i<128*128;i+=blockDim.x)
            if(!isfinite(reinterpret_cast<const float*>(p.recurrence)[i]))atomicOr(&invalid,2U);
        for(uint32_t i=threadIdx.x;i<18*128;i+=blockDim.x)
            if(!isfinite(reinterpret_cast<const float*>(p.positions)[i]))atomicOr(&invalid,2U);
        for(uint32_t f=0;f<18;++f) {
            const auto field=p.fields[f];
            const uint32_t rows=field.cardinality-uint32_t(field.null_category>=0);
            for(uint32_t i=threadIdx.x;i<rows*128;i+=blockDim.x)
                if(!isfinite(reinterpret_cast<const float*>(field.embeddings)[i]))atomicOr(&invalid,2U);
            if(field.biases)for(uint32_t i=threadIdx.x;i<field.cardinality;i+=blockDim.x)
                if(!isfinite(reinterpret_cast<const float*>(field.biases)[i]))atomicOr(&invalid,2U);
        }
    }
#endif
    __syncthreads();
    execution_work_merge_parallel(state->execution_work,sampler_work);
    if(invalid&4U) { semantic_content_integrity_trap();return; }
    if(threadIdx.x==0 && invalid && !failed) {
        failed=1;state->status=(invalid&2) ? 5 : 2;
    }
    __syncthreads();
#ifdef XLOG_SEMANTIC_POLICY
    if (descriptor.policy.z) {
        // Initial states and every successor have separate addresses. No advance
        // aliases its input and the second slot continues the first slot's tape.
        auto tape = reinterpret_cast<float*>(descriptor.policy.recurrent);
        for (uint32_t lane=0;lane<2;++lane)
            for (uint32_t d=threadIdx.x;d<model_policy::width;d+=blockDim.x)
                tape[lane*37*model_policy::width+d]=0.0f;
    }
    __syncthreads();
#endif
    // Cold-bound product support is immutable over this invocation. Prove all
    // four edits completable before consuming the first Philox block.
    for(uint32_t edit=0;edit<4;++edit) {
        uint32_t start=32+(edit/2)*68+(edit%2)*18;
        uint32_t* viable=viability+edit*books[1];
        for(uint32_t t=threadIdx.x;t<books[1];t+=blockDim.x)
            viable[t]=target_completes(books,components,support,start,t,&sampler_work);
        __syncthreads();
        if(threadIdx.x==0) {
            actions[edit][0]=action_completes(books,components,support,viable,start,0,&sampler_work);
            actions[edit][1]=action_completes(books,components,support,viable,start,1,&sampler_work);
            if(!actions[edit][0] && !actions[edit][1] && !failed){failed=1;state->status=2;}
        }
        __syncthreads();
    }
    execution_work_merge_parallel(state->execution_work,sampler_work);
    for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        if(failed)continue;
        const Component component=components[ordinal];
        const uint32_t lane=ordinal/68;
        const uint32_t edit=lane*2+component.slot;
        const uint32_t* viable=viability+edit*books[1];
        if(threadIdx.x==0 && ordinal%68==0) {
            state->execution_work.active_candidate=lane+2;
            semantic_call(descriptor,semantic_graph::kResidentForkAdmission,&base,nullptr,&candidate,nullptr,nullptr,&state->execution_work);
            lane_live=candidate.words[41]!=0;
            if(task && candidate.words[0]!=semantic_graph::kOk){failed=1;state->status=9;}
        }
        if(threadIdx.x==0 && component.kind==COMPONENT_KIND_EDIT && component.field==0)
            for(uint32_t i=0;i<18;++i)choices[i]=0;
        __syncthreads();
        if(failed)continue;
        const bool inactive_text=descriptor.policy.z && component.kind==COMPONENT_KIND_TEXT && !text_selected(descriptor.text,component.field);
        if(inactive_text) {
            if(threadIdx.x==0) {
                Receipt result=component_receipt(component,*state,&sampler_work);
                result.choice=TEXT_NULL;result.legal_count=result.active_count=1;
                result.p.w[0]=result.q.w[0]=result.factor_denominator.w[0]=1;
                result.cdf_end=result.mass=uint64_t(1)<<63;
                receipts[ordinal]=result;++state->blocks;
                semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(Receipt));
            }
            execution_work_merge_parallel(state->execution_work,sampler_work);
            continue;
        }
        const uint64_t offset=descriptor.policy.z && component.kind==COMPONENT_KIND_TEXT
            ? text_row_index(descriptor.text,component.field)*TEXT_CARDINALITY : component.offset;
        const float* row=descriptor.policy.z && component.kind==COMPONENT_KIND_EDIT ? nullptr : logits+offset;
#ifdef XLOG_SEMANTIC_POLICY
        if (descriptor.policy.z && component.kind==COMPONENT_KIND_EDIT) {
            const auto p=descriptor.policy;
            const auto f=p.fields[component.field];
            model_policy::FieldView field={reinterpret_cast<const float*>(f.embeddings),
                reinterpret_cast<const float*>(f.biases),f.cardinality,f.null_category};
            const uint32_t step=component.slot*18+component.field;
            model_policy::readout_hidden(
                reinterpret_cast<const float*>(p.z)+(lane*2+component.slot)*128,
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.positions)+component.field*128,
                reinterpret_cast<float*>(p.hidden),threadIdx.x,blockDim.x);
            __syncthreads();
            model_policy::readout_scores(reinterpret_cast<const float*>(p.hidden),field,
                reinterpret_cast<float*>(p.scores),threadIdx.x,blockDim.x);
            __syncthreads();
            row=reinterpret_cast<const float*>(p.scores);
        }
#endif
        const uint8_t* mask=support+component.offset;
        if(threadIdx.x==0){count=0;invalid=0;}
        __syncthreads();
        for(uint32_t i=threadIdx.x;i<component.cardinality;i+=blockDim.x) {
            bool admitted=mask[i]!=0 && legal_category(component,i,books,viable,choices,actions[edit][0],actions[edit][1],&sampler_work);
            if(admitted) {
                if(!isfinite(row[i]))atomicExch(&invalid,1U);
                uint32_t index=atomicAdd(&count,1U); sorted[index]=i;
            }
        }
        __syncthreads();
        if(threadIdx.x==0) {
            if(!count || invalid) { failed=1;state->status=invalid ? 5 : 2; }
            width=1;while(width<count)width<<=1;
        }
        __syncthreads();
        execution_work_merge_parallel(state->execution_work,sampler_work);
        if(failed)continue;
        for(uint32_t i=count+threadIdx.x;i<width;i+=blockDim.x)sorted[i]=0xffffffffU;
        __syncthreads();
        // Bounded stable numeric order: ordinal breaks all FP32 ties, including ±0.
        for(uint32_t k=2;k<=width;k<<=1)for(uint32_t j=k>>1;j;j>>=1) {
            for(uint32_t i=threadIdx.x;i<width;i+=blockDim.x) {
                uint32_t other=i^j;
                if(other>i) {
                    semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::SortComparison,1);
                    uint32_t a=sorted[i],b=sorted[other];
                    bool swap=(i&k) ? before(a,b,row) : before(b,a,row);
                    if(swap){sorted[i]=b;sorted[other]=a;}
                }
            }
            __syncthreads();
        }
        if(threadIdx.x==0) {
            Receipt result=component_receipt(component,*state,&sampler_work);result.legal_count=count;
            ++state->blocks;
            if(count==1) {
                result.choice=sorted[0];result.p.w[0]=result.q.w[0]=result.factor_denominator.w[0]=1;
                result.active_count=1;result.cdf_end=result.mass=uint64_t(1)<<63;
            } else {
                float maximum=row[sorted[0]];
                U192 g=sub(shifted(1,149),shifted(count,118));
                U192 prefix={{0,0,0}},active_sum={{0,0,0}};
                uint32_t h=0;
                for(uint32_t i=0;i<count;++i) {
                    semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CdfStep,1);
                    U192 d=distance_scaled(maximum,row[sorted[i]]);
                    prefix=add(prefix,d);
                    if(compare(multiply(d,i+1),add(prefix,g))<0){h=i+1;active_sum=prefix;}
                }
                result.active_count=h;result.q=shifted(h,149);
                U192 cumulative={{0,0,0}};
                uint64_t lower=0;
                // Restore original categorical order, independently of compaction.
                uint32_t n=0;
                for(uint32_t i=0;i<component.cardinality;++i)if(mask[i] && legal_category(component,i,books,viable,choices,actions[edit][0],actions[edit][1],&sampler_work))legal[n++]=i;
                for(uint32_t index=0;index<count;++index) {
                    semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CdfStep,1);
                    uint32_t i=legal[index];
                    U192 base=add(active_sum,g), hd=multiply(distance_scaled(maximum,row[i]),h);
                    U192 p=shifted(h,118);
                    if(compare(base,hd)>0)p=add(p,sub(base,hd));
                    cumulative=add(cumulative,p);
                    uint64_t upper=boundary(cumulative,h);
                    if(result.draw>=lower && result.draw<upper) {
                        result.choice=i;result.p=p;result.cdf_start=lower;result.cdf_end=upper;
                        result.mass=upper-lower;result.factor_denominator=multiply(shifted(h,86),result.mass);
                    }
                    lower=upper;
                }
                if(compare(cumulative,result.q)!=0 || lower!=(uint64_t(1)<<63) || result.mass==0) {
                    failed=1;state->status=3;
                }
            }
            if(component.kind==COMPONENT_KIND_EDIT) {
                choices[component.field]=result.choice;
                if(choices[0]==ACTION_APPLY){result.active_fields=INSERT_SUPPORT_ACTIVE_MASK;result.null_fields=INSERT_SUPPORT_NULL_MASK;}
            }
            receipts[ordinal]=result;
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(Receipt));
            if(!failed && component.kind==COMPONENT_KIND_EDIT && component.field==17) {
                auto output=&state->semantic_receipts[1+lane*3+component.slot];
                if(choices[0]==ACTION_APPLY && candidate.words[0]==0) {
                    semantic_graph::DecodedStatement statement;
                    semantic_graph::DecodedSupport event;
                    decode_action(books,choices,&statement,&event);
                    if(task && !state->task_evaluation.lane_refusal[lane] &&
                       !task_allows(task,statement,event))state->task_evaluation.lane_refusal[lane]=1;
                    if(!task || !state->task_evaluation.lane_refusal[lane]) {
                        ++state->work[lane].edit_commands;
                        semantic_call(descriptor,semantic_graph::kResidentInsertSupportAdmission,nullptr,&candidate,output,&statement,&event,&state->execution_work);
                        state->work[lane].added_supports+=semantic_graph::inserted_support_count(output);
                        state->work[lane].defined_truth_changes+=semantic_graph::defined_truth_changed(output);
                        semantic_graph::copy_receipt(&candidate,output);
                        if(task && output->words[0]!=semantic_graph::kOk){failed=1;state->status=9;}
                    } else {
                        semantic_graph::copy_receipt(output,&candidate);output->words[1]=0;
                    }
                } else {
                    semantic_graph::copy_receipt(output,&candidate);
                    if(output->words[0]==0)output->words[1]=0;
                }
            }
            if(!failed && ordinal%68==67) {
                uint64_t admission=semantic_graph::kResidentSealAdmission;
                if(task) {
                    if(state->task_evaluation.lane_refusal[lane]) {
                        task_cost(state,lane+1,task);admission=semantic_graph::kResidentDiscardAdmission;
                    } else if(!task_queries(descriptor,task,lane+1,nullptr,&candidate,state)) {
                        failed=1;state->status=9;
                    } else if(!state->task_evaluation.facts[lane+1].eligible) {
                        state->task_evaluation.lane_refusal[lane]=2;
                        admission=semantic_graph::kResidentDiscardAdmission;
                    }
                }
                if(!failed) {
                    auto output=&state->semantic_receipts[3+lane*3];
                    semantic_call(descriptor,admission,nullptr,&candidate,output,nullptr,nullptr,&state->execution_work);
                    if(task && output->words[0]!=semantic_graph::kOk){failed=1;state->status=9;}
                    else lane_live=0;
                }
            }
        }
        execution_work_merge_parallel(state->execution_work,sampler_work);
#ifdef XLOG_SEMANTIC_POLICY
        if (!failed && descriptor.policy.z && component.kind==COMPONENT_KIND_EDIT) {
            const auto p=descriptor.policy;
            const auto f=p.fields[component.field];
            model_policy::FieldView field={reinterpret_cast<const float*>(f.embeddings),
                reinterpret_cast<const float*>(f.biases),f.cardinality,f.null_category};
            const uint32_t step=component.slot*18+component.field;
            model_policy::advance(
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.recurrence),
                reinterpret_cast<const float*>(p.positions)+component.field*128,
                field,receipts[ordinal].choice,
                reinterpret_cast<float*>(p.recurrent)+(lane*37+step+1)*128,
                threadIdx.x,blockDim.x);
        }
        __syncthreads();
#endif
    }
    if(threadIdx.x==0) {
        if(failed && lane_live) {
            auto cleanup=&state->cleanup_receipts[0];
            semantic_call(descriptor,semantic_graph::kResidentDiscardAdmission,nullptr,&candidate,cleanup,nullptr,nullptr,&state->execution_work);
            if(cleanup->words[0]!=semantic_graph::kOk)state->status=6;
        }
        state->execution_work.active_candidate=0;
        if(state->execution_work.overflow){failed=1;state->status=11;}
        if(!failed) {
            const double weight=receipt_importance_weight(receipts);
            if(!isfinite(weight) || weight<=0.0) {
                failed=1;state->status=3;
            } else {
                if(task && acquired_bank) {
                    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
                    auto& text_candidate=*publication_bank(control,acquired_word^1);
                    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
                    for(uint64_t lane=0;lane<2;++lane)if(!state->task_evaluation.lane_refusal[lane] &&
                        publication_prepare_text(*acquired_bank,text_candidate,contract,receipts,lane+1,structural_end,descriptor.text)) {
                        state->task_evaluation.lane_refusal[lane]=2;
                        state->task_evaluation.facts[lane+1].eligible=0;
                    }
                    text_candidate.header.abi=0;
                }
                if(task)select_task(state,task);
                state->importance_weight=weight;
                if(acquired_bank) {
                    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
                    control.refusal=publication_publish(descriptor,*acquired_bank,acquired_word,structural_end,base,state,receipts);
                    if(control.refusal){failed=1;state->status=state->execution_work.overflow ? 11 : 10;}
                } else state->next_proposal=state->proposal+1;
            }
        }
        if(failed && !retire_pending_roots(descriptor,base,state))state->status=6;
        if(state->execution_work.overflow)state->status=11;
        if(state->status==2 || state->status==5)
            publication_refusal_snapshot(descriptor,base,acquired_bank,state);
        if(publication_held) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(failed && publication_staged)publication_bank(control,acquired_word^1)->header.abi=0;
            publication_store(control.reader_gate,0);
        }
    }
}
