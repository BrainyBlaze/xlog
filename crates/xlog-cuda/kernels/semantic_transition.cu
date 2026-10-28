#include <stdint.h>
#include <stddef.h>
#include <math.h>
#ifdef __CUDACC__
#include <cuda/atomic>
#else
#include <atomic>
#endif
#include "semantic_transition_catalogue.cuh"
#include "semantic_feedback_encoding.cuh"
#include "semantic_program.cuh"
#include "semantic_task_ground.cuh"
#include "semantic_rule_action.cuh"
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
    uint32_t ordinal,lane,slot,field,kind,choice,legal_count,active_count,selected_rank,reserved;
    uint32_t key[2],counter[4],random_words[4];
    uint64_t draw,cdf_start,cdf_end,mass;
    U192 p,q,factor_denominator,cumulative_mass_below;
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
struct PublicationPolicyWorkEntry { uint64_t tally,ceiling[9]; };
struct PublicationPolicyWorkInput { uint64_t entries,count; };
static_assert(sizeof(PublicationPolicyWorkEntry)==80 && alignof(PublicationPolicyWorkEntry)==8 &&
    offsetof(PublicationPolicyWorkEntry,ceiling)==8 && sizeof(PublicationPolicyWorkInput)==16 &&
    alignof(PublicationPolicyWorkInput)==8 && offsetof(PublicationPolicyWorkInput,count)==8,
    "original policy work roster ABI");
struct ActorRefreshWorkInput { uint64_t abi,entries,count; };
struct ActorRefreshWorkEntry {
    uint64_t member_ordinal,proof,native_work,states[2],readers[2],results[2],parents[2];
    ModelWorkInput model_work[2];
    uint64_t native_ceiling[9];
    uint64_t transition_ceilings[2][9];
};
static_assert(sizeof(ActorRefreshWorkInput)==24 && sizeof(ActorRefreshWorkEntry)==352,
    "original actor work roster ABI");
struct ExecutionWork {
    uint64_t raw,model_once,native_attempt,model_events,model_bound,overflow;
    uint64_t native_events[9],candidates[3],active_candidate;
};
__device__ void semantic_content_integrity_trap();
__device__ bool publication_pointer_span(uint64_t pointer,uint64_t bytes,uint64_t alignment=1);
static_assert(sizeof(ModelWorkEvent)==104 && sizeof(ModelWorkInput)==24 && sizeof(ExecutionWork)==152,
    "original execution work ABI");

// Recorded at the original model-call site. CUDA graph recording alone never
// writes a call: only execution of the selected branch reaches this node.
extern "C" __global__ void semantic_model_invocation(uint64_t actual,uint64_t native_work) {
    if(blockIdx.x || threadIdx.x)return;
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(native_work);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    auto* values=reinterpret_cast<uint64_t*>(actual);
    values[0]=semantic_graph::written_word(1,work);
    values[1]=semantic_graph::written_word(0,work);
    values[2]=semantic_graph::written_word(1,work);
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
__device__ semantic_graph::NativeWorkTally native_work_reduce_parallel(semantic_graph::NativeWorkTally& local) {
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
    semantic_graph::NativeWorkTally actual{};
    if(threadIdx.x==0) {
        if(overflow)actual.overflow=1;
        else {
            for(uint32_t i=0;i<9;++i)
                semantic_graph::charge_native(&actual,static_cast<semantic_graph::NativeWorkEvent>(i),totals[i]);
        }
    }
    local=semantic_graph::NativeWorkTally{};
    __syncthreads();
    return actual;
}
struct PublicationAttemptWork {
    ExecutionWork& execution;
    semantic_graph::NativeWorkTally actual{};
    __device__ explicit PublicationAttemptWork(ExecutionWork& original):execution(original) {}
    __device__ void flush() { execution_work_merge_native(execution,actual);actual={}; }
    __device__ ~PublicationAttemptWork() { flush(); }
    __device__ ExecutionWork completed(const semantic_graph::NativeWorkTally* suffix=nullptr) const {
        ExecutionWork result=execution;execution_work_merge_native(result,actual);
        if(suffix)execution_work_merge_native(result,*suffix);
        return result;
    }
};

__device__ void execution_work_merge_parallel(ExecutionWork& work,semantic_graph::NativeWorkTally& local) {
    const auto actual=native_work_reduce_parallel(local);
    if(threadIdx.x==0) {
        if(actual.overflow)work.overflow=1;
        else execution_work_merge_native(work,actual);
    }
    __syncthreads();
}

// This is the original NativeWorkTally, never an ExecutionWork reinterpretation.
// Parallel writers reduce only their reached stores; capacity is not a charge.
extern "C" __global__ void semantic_model_work_reset(uint64_t actual,uint64_t words,
        uint64_t native_work) {
    if(blockIdx.x)return;
    auto* values=reinterpret_cast<uint64_t*>(actual);
    semantic_graph::NativeWorkTally local{};
    if(threadIdx.x==0)
        semantic_graph::charge_native(&local,semantic_graph::NativeWorkEvent::Command,1);
    for(uint64_t i=threadIdx.x;i<words;i+=blockDim.x)
        values[i]=semantic_graph::written_word(0,&local);
    const auto reached=native_work_reduce_parallel(local);
    if(threadIdx.x==0 && native_work) {
        auto* target=reinterpret_cast<semantic_graph::NativeWorkTally*>(native_work);
        for(uint32_t i=0;i<9;++i)
            semantic_graph::charge_native(target,static_cast<semantic_graph::NativeWorkEvent>(i),reached.events[i]);
        if(reached.overflow)target->overflow=1;
    }
}

__device__ void consume_model_work(const ModelWorkInput& input,ExecutionWork& work,
        uint64_t* model_calls=nullptr) {
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
    if(!input.events || !input.count || input.events%alignof(ModelWorkEvent) ||
       input.count>UINT64_MAX/sizeof(ModelWorkEvent) ||
       input.events>UINT64_MAX-input.count*sizeof(ModelWorkEvent)) {
        semantic_content_integrity_trap();return;
    }
    const auto* events=reinterpret_cast<const ModelWorkEvent*>(input.events);
    uint64_t declared_bound=0;
    for(uint64_t i=0;i<input.count;++i) {
        const auto& event=events[i];
        const bool saved=event.kind==15;
        const bool invocation=event.kind==16;
        if(event.kind<1 || event.kind>16 || event.rank>8 ||
           (invocation && (event.rank || !event.actual)) ||
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
        const uint64_t work_bound=invocation ? 0 : units;
        if(work_bound>UINT64_MAX-declared_bound) { semantic_content_integrity_trap();return; }
        declared_bound+=work_bound;
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
            if(invocation && (actual[0]!=1 || actual[2]!=1)) {
                semantic_content_integrity_trap();return;
            }
            units=actual[0];
        }
        if(invocation) {
            if(model_calls)++*model_calls;
            units=0;
        }
        if(!execution_work_add(work,units,true))return;
        ++work.model_events;
        if(work.model_once>input.bound) { work.overflow=1;return; }
    }
    if(declared_bound!=input.bound) { semantic_content_integrity_trap();return; }
}
// The original cold operation joins model work and its actually reached semantic
// commands. CPU codec is not CUDA model work; no old resident attempt is charged.
// Empty model input requires the original recorder to have explicitly closed.
extern "C" __global__ void semantic_cold_model_work_result(
    uint64_t events,uint64_t count,uint64_t bound,uint64_t semantic_work,uint64_t result) {
    if(blockIdx.x || threadIdx.x)return;
    ExecutionWork work{};
    uint64_t calls=0;
    execution_work_merge_native(work,
        *reinterpret_cast<const semantic_graph::NativeWorkTally*>(semantic_work));
    consume_model_work(ModelWorkInput{events,count,bound},work,&calls);
    auto* output=reinterpret_cast<uint64_t*>(result);
    output[0]=work.overflow || work.model_events!=count || work.model_bound!=bound ? 1 : 0;
    output[1]=work.model_once;output[2]=work.model_events;
    output[3]=work.model_bound;output[4]=calls;
    output[5]=work.native_attempt;
    for(uint32_t i=0;i<9;++i)output[6+i]=work.native_events[i];
}
// Cold observation only: no Update, RNG, optimizer or publication side effect.
extern "C" __global__ void semantic_model_evaluation_result(
    uint64_t events,uint64_t count,uint64_t bound,uint64_t selection,
    uint64_t losses,uint64_t admissible,uint64_t retained_bytes,uint64_t result) {
    if(blockIdx.x || threadIdx.x)return;
    auto* output=reinterpret_cast<uint64_t*>(result);
    ExecutionWork work{};
    uint64_t model_calls=0;
    consume_model_work(ModelWorkInput{events,count,bound},work,&model_calls);
    const auto* values=reinterpret_cast<const float*>(losses);
    const auto* selected=reinterpret_cast<const uint64_t*>(selection);
    uint64_t status=selected[0] || !*reinterpret_cast<const uint8_t*>(admissible) ? 1 : 0;
    if(work.overflow || work.model_events!=count || work.model_bound!=bound)status=2;
    for(uint32_t i=0;i<6;++i) {
        const float value=values[i];
        // Numerical refusal must not hide an incomplete work certificate.
        if(!isfinite(value) && status==0)status=1;
        output[i+1]=__float_as_uint(value);
    }
    output[0]=status;output[7]=work.model_once;output[8]=work.model_events;
    output[9]=work.model_bound;output[10]=retained_bytes;output[11]=model_calls;
}
// Cancellation observes only the actual original work roster. It emits no
// losses, selection status or successful model observation.
extern "C" __global__ void semantic_model_cancellation_work_result(
    uint64_t events,uint64_t count,uint64_t bound,uint64_t result) {
    if(blockIdx.x || threadIdx.x)return;
    ExecutionWork work{};
    uint64_t calls=0;
    consume_model_work(ModelWorkInput{events,count,bound},work,&calls);
    auto* output=reinterpret_cast<uint64_t*>(result);
    output[0]=work.overflow || work.model_events!=count || work.model_bound!=bound ? 1 : 0;
    output[1]=work.model_once;output[2]=work.model_events;
    output[3]=work.model_bound;output[4]=calls;
}
struct TaskFacts {
    uint64_t g,p,c;
    int64_t v;
    uint64_t eligible;
};
struct TaskEvaluation {
    uint64_t lane_refusal[2]; // 0 accepted, 1 scope, 2 hard constraint, 3 fact cap, 4 fuel.
    uint64_t query_count,winner;
    int64_t return_value;
    TaskFacts facts[3]; // Base, first learned lane, second learned lane.
};
struct PolicyPwlCell {
    uint64_t abi,ordinal,cardinality,legal_count,active_count,selected_rank,maximum_bits,singleton;
    U192 g,active_sum,selected_probability,probability_denominator;
};
struct PolicySelectedScoreVjp {
    uint64_t abi,ordinal,choice,active_count,selected_active,singleton;
    U192 probability_numerator,probability_denominator;
};
struct ActionBatchReceipt {
    uint64_t abi,proposal,stream_serial,family_id,action_law_generation,model_generation;
    uint64_t actor_eligible,component_count,candidate_count,winner,total_return_bits;
    uint64_t base_word,next_word,rng_base,rng_span,rng_successor;
    uint64_t catalogue_digest[4],component_receipts_digest[4],semantic_receipts_digest[4];
    uint64_t candidate_roots[3][4],base_logical_digest[4];
    ExecutionWork execution_work;
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
    uint64_t actor_eligible;
    ExecutionWork execution_work;
    ActionBatchReceipt action_batch;
    uint64_t action_batch_root[4];
};
struct SourceSlot {
    uint64_t token,logical_position,kind,provenance,valid,committed,recomputed,provenance_record;
};
struct TextRow { uint64_t source_slot,logical_position; };
struct TextBinding { uint64_t rows,count,selected; };
// slots name the immutable source and this Update's exact pending generation.
struct ModelUpdateBinding { uint64_t candidate,bytes,slots[2]; };
struct SemanticCapacityRefusalRecord {
    uint64_t abi,reason,allocation_ordinal,requested_bytes,used_bytes,capacity_bytes;
};
struct ContinuationInputs {
    TextBinding text;
    uint64_t active_rows,active_row_count,numerical_admissibility,transition_kind,authority_bytes;
    uint64_t training_selection,model_update_bindings,model_update_binding_count,model_update_admissibility;
    uint64_t model_update_refusal,model_update_canary_results,model_update_canaries;
    uint64_t numerical_blocks,physical_blocks,model_schema_digest[4],model_identity[4];
    uint64_t model_contract,model_contract_bytes;
    uint64_t capacity_refusal_source,capacity_refusal,capacity_allocation_ordinal,capacity_bytes;
};
struct PublicationStorageEntry { uint64_t pointer,bytes,generation; };
struct PublicationRange {
    uint64_t role,index,storage_slot,generation,offset_bytes,length_bytes,logical_begin,logical_end;
};
struct PublicationHeader {
    uint64_t abi,instance[4],recovered_instance[4],original_instance[4],sealed_epoch,base_word,publication_word;
    uint64_t ring_head,prefix_extent,semantic_owner,semantic_slot,semantic_generation;
    uint64_t semantic_digest[4],semantic_extents[3];
    uint64_t model_generation,neural_bank,neural_generation,cache_generation,authority_generation;
    uint64_t terminal,terminal_position,fuel,proposal,stream_serial,family_id,training_cursor,training_rng[4];
    uint64_t range_count,model_geometry_digest[4];
    uint64_t logical_digest[4];
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
    uint64_t schema_begin,schema_bytes,schema_digest_offset,generation_offset,identity_offset;
};
struct PublicationContract {
    uint64_t abi,range_capacity,prefix_capacity,window_capacity,feedback_capacity,max_position,pad_token;
    uint64_t terminal_tokens,terminal_token_count,final_intent_payload_bytes;
    uint64_t nonterminal_intent_mode,nonterminal_intent_plan_digest[4];
    uint64_t nonterminal_intent_prefix_begin,nonterminal_intent_prefix_end;
    uint64_t replay_row_capacity,replay_original_slot_boundary,replay_raw_capacity_bytes,replay_payload_capacity_bytes;
    uint64_t replay_metadata_bytes,replay_max_material_bytes,replay_total_material_bytes,replay_max_evidence_bytes;
    uint64_t replay_reservation_identity[4];
    uint64_t model_generation,policy_generation,authority_generation,semantic_owner;
    uint64_t task_identity[4],topology_identity[4],table_identity[4],role_counts,role_count;
    ModelContractLayout model_contract_layout;
};
struct PendingContinuation {
    uint64_t abi,instance[4],base_word,model_generation,authority_generation;
    uint64_t topology_identity[4],table_identity[4],prefix_identity[4],ranges,range_count,transition_kind;
    TextBinding text;
    uint64_t numerical_admissibility,training_selection,model_update_bindings,model_update_binding_count;
    uint64_t model_update_admissibility,model_update_refusal,model_update_canary_results,model_update_canaries;
    uint64_t numerical_blocks,physical_blocks,model_schema_digest[4],model_identity[4];
    uint64_t capacity_refusal;
};
struct PublicationCommand { uint64_t control,lease,operation,native_work; };
struct PublicationDeliveryInput {
    uint64_t lease,receipt,receipt_len,recipient_id[4],receipt_digest[4];
};
struct PublicationLease { uint64_t abi,status,instance[4],word,bank,epoch,active,transition_kind; };
struct SemanticTrainingViewOriginRecord {
    uint64_t present,transition;
    uint64_t lineage_instance[4];
    uint64_t predecessor_instance[4],predecessor_word,predecessor_logical[4];
    uint64_t successor_instance[4],successor_word,successor_logical[4];
    uint64_t model_generation,stream_serial,family_id,proposal;
    uint64_t model_geometry_digest[4];
};
struct TrainingViewRowDescriptor {
    uint64_t ordinal,basis,raw_offset,raw_bytes,window,source_length,block_size,prefix_extent,answer_start;
    uint64_t identity[4],source_identity[4],content_identity[4],task_query_identity[4];
    uint64_t task_theory_program_identity[4],task_result_identity[4];
    SemanticTrainingViewOriginRecord origin;
};
struct SemanticTrainingReplayAppendHeader {
    uint64_t abi,count,capacity,payload_used_bytes,payload_capacity_bytes,eligible_count;
    uint64_t original_stage,original_count,original_prefix_identity[4],chain_head[4];
};
struct SemanticTrainingReplayAppendEntry {
    uint64_t row_ordinal,content_identity[4];
    SemanticTrainingViewOriginRecord original_origin;
    uint64_t stable_intent[4],result_receipt_digest[4],payload_offset_bytes,payload_length_bytes,disposition;
};
struct PublicationReplayDeliveryRow {
    SemanticTrainingReplayAppendEntry entry;
    uint64_t payload,payload_bytes;
    TrainingViewRowDescriptor row_descriptor;
    uint64_t row_bytes,row_bytes_len;
};
struct PublicationReplayDeliveryInput {
    PublicationDeliveryInput delivery;
    uint64_t receiver_outcome,receiver_outcome_bytes,receiver_outcome_digest[4],receiver_disposition;
    uint64_t rows,row_count,arena_descriptors,arena_descriptor_count,arena_raw,arena_raw_bytes;
    uint64_t observations,observation_count,statements,supports,native_work;
};
struct TrainingOriginalMaterializationInput {
    uint64_t lease,source,source_bytes,destinations[7],lengths[7],original_count,original_prefix_identity[4];
};
static_assert(sizeof(TrainingOriginalMaterializationInput)==176,"original training materialization ABI");
static_assert(sizeof(SemanticTrainingReplayAppendHeader)==128 && sizeof(SemanticTrainingReplayAppendEntry)==384 &&
    sizeof(PublicationReplayDeliveryRow)==936 && sizeof(PublicationReplayDeliveryInput)==232,
    "replay append queue ABI");
static_assert(offsetof(PublicationReplayDeliveryInput,receiver_outcome)==88 &&
    offsetof(PublicationReplayDeliveryInput,rows)==144 &&
    offsetof(PublicationReplayDeliveryInput,observations)==192 &&
    offsetof(PublicationReplayDeliveryInput,native_work)==224,"replay delivery field ABI");
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
    uint64_t prefix_begin,prefix_end,source_origin[4],attention_origin[4],linear_origin[4];
    uint64_t value_origin[4],edit_origin[4],receipt_digest[4];
};
struct AttemptReceipt {
    uint64_t abi,transition_kind,instance[4],base_word,next_word,logical_digest[4],action_receipts_digest[4];
    uint64_t semantic_receipts_digest[4],coverage_digest[4],replay_head_digest[4],intent_head_digest[4];
    uint64_t acknowledgement_head_digest[4],previous_attempt_digest[4],receipt_digest[4];
};
struct PublicationStepResult {
    uint64_t abi,word,refusal;
    PublicationHeader header;
    uint64_t advanced;
    AttemptReceipt attempt;
};
struct TokenProvenanceRecord {
    uint64_t source_slot,logical_position,token,base_word,proposal,ordinal,action_receipt_digest[4];
    uint64_t authority_generation,origin_logical_digest[4],authority_closure_origin[4],record_digest[4];
    uint64_t successor_logical_digest[4];
};
struct IntentQueueHeader {
    uint64_t abi,count,capacity,payload_used_bytes,payload_capacity_bytes,effect_offset_bytes,effect_length_bytes,chain_head[4];
};
struct IntentEntry {
    uint64_t stable_identity[4],checkpoint_lineage[4],base_logical[4],effective_delta[4],result_logical[4];
    uint64_t effect_digest[4],payload_digest[4],payload_offset,payload_len,audit_instance[4],audit_epoch;
    uint64_t previous_chain[4],chain[4];
};
struct AcknowledgementQueueHeader {
    uint64_t abi,count,capacity,payload_used_bytes,payload_capacity_bytes,chain_head[4];
};
struct AcknowledgementEntry {
    uint64_t stable_identity[4],receipt_digest[4],receipt_len,receiver_outcome_digest[4],receiver_outcome_len;
    uint64_t previous_chain[4],chain[4];
};
struct PolicyField { uint64_t embeddings,biases; uint32_t cardinality; int32_t null_category; uint64_t retained_offset; };
struct PolicyNumericalDescriptor {
    uint64_t parameter_errors,baseline_errors,parameter_domains,text_domains,baseline_domains;
    uint64_t hidden_errors,score_errors,recurrent_errors,parameter_cells;
    uint64_t parameter_envelopes,baseline_envelopes;
    uint64_t uniform;
};
struct PolicyDescriptor {
    uint64_t z,recurrence,positions,hidden,scores,recurrent,retained_scores,retained_score_stride;
    uint64_t final_masks,active_sets,pwl_cells,selected_score_vjps,component_baselines;
    PolicyField fields[18];
    PolicyNumericalDescriptor numerical;
};
#ifdef XLOG_SEMANTIC_POLICY
__device__ const model_policy::PrimalEnvelope* policy_parameter_errors(
        const PolicyDescriptor& policy,uint64_t values) {
    return values ? reinterpret_cast<const model_policy::PrimalEnvelope*>(
        policy.numerical.parameter_envelopes)+(values-policy.z)/sizeof(float) : nullptr;
}
__device__ model_policy::FieldErrorView policy_field_errors(
        const PolicyDescriptor& policy,const PolicyField& field) {
    return {policy_parameter_errors(policy,field.embeddings),
            policy_parameter_errors(policy,field.biases)};
}
__device__ void policy_readout_hidden(const PolicyDescriptor& policy,
        uint32_t lane,uint32_t slot,uint32_t step,uint32_t field,
        semantic_graph::NativeWorkTally* work) {
    const auto* recurrent_errors=reinterpret_cast<const model_policy::PrimalEnvelope*>(
        policy.numerical.recurrent_errors)+(lane*37+step)*128;
    const model_policy::ReadoutHiddenErrors errors{
        policy_parameter_errors(policy,policy.z)+(lane*2+slot)*128,
        recurrent_errors,policy_parameter_errors(policy,policy.positions)+field*128,
        reinterpret_cast<model_policy::PrimalEnvelope*>(policy.numerical.hidden_errors)};
    model_policy::readout_hidden(reinterpret_cast<const float*>(policy.z)+(lane*2+slot)*128,
        reinterpret_cast<const float*>(policy.recurrent)+(lane*37+step)*128,
        reinterpret_cast<const float*>(policy.positions)+field*128,
        reinterpret_cast<float*>(policy.hidden),threadIdx.x,blockDim.x,&errors);
    if(work)for(uint32_t d=threadIdx.x;d<128;d+=blockDim.x)
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
}
__device__ void policy_readout_scores(const PolicyDescriptor& policy,uint32_t field_index,
        semantic_graph::NativeWorkTally* work) {
    const auto field=policy.fields[field_index];
    const model_policy::FieldView values{reinterpret_cast<const float*>(field.embeddings),
        reinterpret_cast<const float*>(field.biases),field.cardinality,field.null_category};
    const model_policy::ReadoutScoreErrors errors{
        reinterpret_cast<const model_policy::PrimalEnvelope*>(policy.numerical.hidden_errors),
        policy_field_errors(policy,field),
        reinterpret_cast<model_policy::PrimalEnvelope*>(policy.numerical.score_errors)};
    model_policy::readout_scores(reinterpret_cast<const float*>(policy.hidden),values,
        reinterpret_cast<float*>(policy.scores),threadIdx.x,blockDim.x,&errors);
    if(work)for(uint32_t category=threadIdx.x;category<field.cardinality;category+=blockDim.x) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(model_policy::embedding_row(values,category))
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,128);
        if(values.biases)semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    }
}
__device__ bool policy_error_finite(model_policy::ErrorEnvelope error) {
    return model_policy::finite_bound(error.ordinary) && model_policy::finite_bound(error.underflow);
}
__device__ bool policy_primal_finite(model_policy::PrimalEnvelope input) {
    return policy_error_finite(input.error) && model_policy::finite_bound(input.ideal_magnitude);
}
__device__ bool policy_cotangent_finite(model_policy::CoherentCotangent input) {
    return model_policy::finite_domain(input.actor) && model_policy::finite_domain(input.full);
}
__device__ bool policy_domain_covers(float value,model_policy::ErrorEnvelope error,
        model_policy::DomainValue domain) {
    return isfinite(value) && policy_error_finite(error) &&
        model_policy::finite_domain(domain) && domain.magnitude>=fabs(double(value)) &&
        domain.error.ordinary>=error.ordinary && domain.error.underflow>=error.underflow;
}

// Native rational coefficients execute in RN64 before the original RN32
// cotangent narrowing. Ideal magnitude intervals follow the original exact
// integer/rational formula, independently of actual values and error radii.
struct PolicyBoundedScalar {
    double value;
    model_policy::ErrorEnvelope error;
    double ideal_magnitude,ideal_minimum;
};
__device__ PolicyBoundedScalar policy_exact_scalar(double value) {
    return {value,{0.0,0.0},fabs(value),fabs(value)};
}
__device__ model_policy::ErrorEnvelope policy_round_error(
        double lower,double upper,double unit,double tiny,double allowance) {
    if(!isfinite(lower) || !isfinite(upper))return {INFINITY,INFINITY};
    if(lower==upper)return {0.0,0.0};
    const double magnitude=fmax(fabs(lower),fabs(upper));
    // Directed neighbors determine whether the exact result is tiny, even
    // when RN crosses the smallest-normal boundary.
    if(magnitude<=tiny)return {0.0,allowance};
    return {model_policy::upper_multiply(unit,magnitude),0.0};
}
__device__ model_policy::ErrorEnvelope policy_error_add(
        model_policy::ErrorEnvelope a,model_policy::ErrorEnvelope b) {
    return {model_policy::upper_add(a.ordinary,b.ordinary),
            model_policy::upper_add(a.underflow,b.underflow)};
}
__device__ PolicyBoundedScalar policy_scalar_negate(PolicyBoundedScalar a) {
    return {-a.value,a.error,a.ideal_magnitude,a.ideal_minimum};
}
__device__ PolicyBoundedScalar policy_scalar_add(PolicyBoundedScalar a,PolicyBoundedScalar b) {
    const auto local=policy_round_error(__dadd_rd(a.value,b.value),__dadd_ru(a.value,b.value),
        0x1p-53,0x1p-1022,0x1p-1074);
    return {__dadd_rn(a.value,b.value),policy_error_add(policy_error_add(a.error,b.error),local),
        model_policy::upper_add(a.ideal_magnitude,b.ideal_magnitude),0.0};
}
__device__ PolicyBoundedScalar policy_scalar_multiply(PolicyBoundedScalar a,PolicyBoundedScalar b) {
    const double aa=fabs(a.value),bb=fabs(b.value);
    const auto local=policy_round_error(__dmul_rd(a.value,b.value),__dmul_ru(a.value,b.value),
        0x1p-53,0x1p-1022,0x1p-1074);
    const auto& x=a.error;const auto& y=b.error;
    const double ordinary=model_policy::upper_add(
        model_policy::upper_add(model_policy::upper_multiply(aa,y.ordinary),
                                model_policy::upper_multiply(bb,x.ordinary)),
        model_policy::upper_multiply(x.ordinary,y.ordinary));
    const double underflow=model_policy::upper_add(
        model_policy::upper_add(model_policy::upper_multiply(aa,y.underflow),
                                model_policy::upper_multiply(bb,x.underflow)),
        model_policy::upper_add(
            model_policy::upper_add(model_policy::upper_multiply(x.ordinary,y.underflow),
                                    model_policy::upper_multiply(x.underflow,y.ordinary)),
            model_policy::upper_multiply(x.underflow,y.underflow)));
    return {__dmul_rn(a.value,b.value),policy_error_add({ordinary,underflow},local),
        model_policy::upper_multiply(a.ideal_magnitude,b.ideal_magnitude),
        __dmul_rd(a.ideal_minimum,b.ideal_minimum)};
}
__device__ model_policy::ErrorEnvelope policy_quotient_error(
        PolicyBoundedScalar a,PolicyBoundedScalar b,model_policy::ErrorEnvelope local) {
    const double lower=__dsub_rd(fabs(b.value),model_policy::upper_add(b.error.ordinary,b.error.underflow));
    if(!(lower>0.0) || !isfinite(a.value) || !isfinite(b.value))return {INFINITY,INFINITY};
    const double quotient=__ddiv_ru(fabs(a.value),fabs(b.value));
    return policy_error_add({
        __ddiv_ru(model_policy::upper_add(a.error.ordinary,
            model_policy::upper_multiply(quotient,b.error.ordinary)),lower),
        __ddiv_ru(model_policy::upper_add(a.error.underflow,
            model_policy::upper_multiply(quotient,b.error.underflow)),lower)},local);
}
__device__ PolicyBoundedScalar policy_scalar_divide(PolicyBoundedScalar a,PolicyBoundedScalar b) {
    const auto local=policy_round_error(__ddiv_rd(a.value,b.value),__ddiv_ru(a.value,b.value),
        0x1p-53,0x1p-1022,0x1p-1074);
    return {__ddiv_rn(a.value,b.value),policy_quotient_error(a,b,local),
        b.ideal_minimum>0.0 ? __ddiv_ru(a.ideal_magnitude,b.ideal_minimum) : INFINITY,
        b.ideal_magnitude>0.0 ? __ddiv_rd(a.ideal_minimum,b.ideal_magnitude) : 0.0};
}
__device__ PolicyBoundedScalar policy_u192_bound(U192 value) {
    int limb=2;while(limb>=0 && !value.w[limb])--limb;
    const double rounded=u192_to_double_rn(value);
    // Directed limb conversion supplies an independent interval for the exact
    // unsigned integer, including carries beyond the RN64 significand.
    double ideal_upper=0.0,ideal_lower=0.0;
    for(int i=2;i>=0;--i) {
        ideal_upper=__dadd_ru(ideal_upper,ldexp(__ull2double_ru(value.w[i]),64*i));
        ideal_lower=__dadd_rd(ideal_lower,ldexp(__ull2double_rd(value.w[i]),64*i));
    }
    if(limb<0)return {rounded,{0.0,0.0},ideal_upper,ideal_lower};
    const unsigned top=unsigned(limb)*64+63-__clzll(value.w[limb]);
    if(top<=52)return {rounded,{0.0,0.0},ideal_upper,ideal_lower};
    const unsigned discarded=top-52,word=discarded/64,shift=discarded%64;
    uint64_t tail=shift ? value.w[word]&((uint64_t(1)<<shift)-1) : 0;
    for(unsigned i=0;i<word;++i)tail|=value.w[i];
    return {rounded,{tail ? ldexp(1.0,int(top)-53) : 0.0,0.0},ideal_upper,ideal_lower};
}
__device__ PolicyBoundedScalar policy_uint64_bound(uint64_t value) {
    return policy_u192_bound(U192{{value,0,0}});
}
__device__ PolicyBoundedScalar policy_int64_bound(int64_t value) {
    const uint64_t magnitude=value<0 ? uint64_t(-(value+1))+1 : uint64_t(value);
    const auto bound=policy_uint64_bound(magnitude);
    return value<0 ? policy_scalar_negate(bound) : bound;
}
__device__ model_policy::BoundedValue policy_uint64_float_bound(uint64_t value) {
    const float rounded=__ull2float_rn(value);
    const double ideal=__ull2double_ru(value);
    if(!value)return {rounded,{0.0,0.0},ideal};
    const unsigned top=63-__clzll(value);
    if(top<=23)return {rounded,{0.0,0.0},ideal};
    const unsigned discarded=top-23;
    const bool inexact=(value&((uint64_t(1)<<discarded)-1))!=0;
    return {rounded,{inexact ? ldexp(1.0,int(top)-24) : 0.0,0.0},ideal};
}
__device__ model_policy::BoundedValue policy_int64_float_bound(int64_t value) {
    const uint64_t magnitude=value<0 ? uint64_t(-(value+1))+1 : uint64_t(value);
    const auto bound=policy_uint64_float_bound(magnitude);
    return value<0 ? model_policy::bounded_negate(bound) : bound;
}
__device__ model_policy::BoundedValue policy_float_divide(
        model_policy::BoundedValue a,model_policy::BoundedValue b,double ideal_denominator_minimum) {
    const auto local=policy_round_error(double(__fdiv_rd(a.value,b.value)),
        double(__fdiv_ru(a.value,b.value)),0x1p-24,0x1p-126,0x1p-150);
    return {__fdiv_rn(a.value,b.value),policy_quotient_error(
        {double(a.value),a.error,a.ideal_magnitude,0.0},
        {double(b.value),b.error,b.ideal_magnitude,0.0},local),
        ideal_denominator_minimum>0.0 ? __ddiv_ru(a.ideal_magnitude,ideal_denominator_minimum) : INFINITY};
}
__device__ PolicyBoundedScalar policy_importance_bound(const Receipt* receipts,
        semantic_graph::NativeWorkTally* work) {
    PolicyBoundedScalar weight=policy_exact_scalar(1.0);
    for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& receipt=receipts[ordinal];
        weight=policy_scalar_multiply(weight,policy_scalar_divide(
            policy_u192_bound(receipt.p),policy_u192_bound(receipt.factor_denominator)));
    }
    return weight;
}

// ONE actual RN64 coefficient accompanies both labels. Cases are classified
// from directed neighbors of the exact FULL operation, never from actor M or
// a rounded quotient y/t. The shared header owns every label transfer.
struct PolicyCoherentScalar {
    double value;
    model_policy::CoherentCotangent envelope;
};
__device__ uint32_t policy_scalar_rounding(double lower,double upper,double tiny) {
    if(lower==upper && isfinite(lower))return model_policy::exact_rounding;
    return fmax(fabs(lower),fabs(upper))<=tiny
        ? model_policy::inexact_tiny_rounding : model_policy::normal_rounding;
}
__device__ PolicyCoherentScalar policy_cotangent_seed(PolicyBoundedScalar input,bool actor) {
    const model_policy::DomainValue full{fabs(input.value),input.error,input.ideal_magnitude};
    return {input.value,{actor ? full : model_policy::zero_cotangent().actor,full}};
}
__device__ PolicyCoherentScalar policy_cotangent_negate(PolicyCoherentScalar input) {
    return {-input.value,input.envelope};
}
__device__ PolicyCoherentScalar policy_cotangent_add(PolicyCoherentScalar a,PolicyCoherentScalar b) {
    const auto cases=policy_scalar_rounding(__dadd_rd(a.value,b.value),__dadd_ru(a.value,b.value),0x1p-1022);
    return {__dadd_rn(a.value,b.value),model_policy::add_cotangent_labels(
        a.envelope,b.envelope,cases,model_policy::fp64_unit_roundoff)};
}
__device__ PolicyCoherentScalar policy_cotangent_multiply(PolicyCoherentScalar a,PolicyBoundedScalar b) {
    const auto cases=policy_scalar_rounding(__dmul_rd(a.value,b.value),__dmul_ru(a.value,b.value),0x1p-1022);
    return {__dmul_rn(a.value,b.value),model_policy::multiply_cotangent_labels(
        a.envelope,{fabs(b.value),b.error,b.ideal_magnitude},cases,model_policy::fp64_unit_roundoff)};
}
__device__ PolicyCoherentScalar policy_cotangent_divide(PolicyCoherentScalar a,PolicyBoundedScalar b) {
    const auto cases=policy_scalar_rounding(__ddiv_rd(a.value,b.value),__ddiv_ru(a.value,b.value),0x1p-1022);
    const double actual=fabs(b.value);
    model_policy::DomainValue reciprocal=model_policy::invalid_domain();
    if(actual>0.0 && b.ideal_minimum>0.0) {
        // This is an enclosure of the exact reciprocal factor, NOT another
        // executed reciprocal or rounding step. F-F* uses the original ideal
        // denominator lower bound, independent of its actual error envelope.
        reciprocal={__ddiv_ru(1.0,actual),
            {__ddiv_ru(__ddiv_ru(b.error.ordinary,actual),b.ideal_minimum),
             __ddiv_ru(__ddiv_ru(b.error.underflow,actual),b.ideal_minimum)},
            __ddiv_ru(1.0,b.ideal_minimum)};
    }
    return {__ddiv_rn(a.value,b.value),model_policy::multiply_cotangent_labels(
        a.envelope,reciprocal,cases,model_policy::fp64_unit_roundoff)};
}
__device__ model_policy::BoundedCotangent policy_cotangent_narrow(PolicyCoherentScalar input) {
    const float value=__double2float_rn(input.value);
    const auto cases=policy_scalar_rounding(double(__double2float_rd(input.value)),
        double(__double2float_ru(input.value)),0x1p-126);
    return {value,model_policy::round_cotangent(input.envelope,cases,model_policy::unit_roundoff)};
}
__device__ model_policy::BoundedCotangent policy_full_cotangent(model_policy::BoundedValue input) {
    return {input.value,{model_policy::zero_cotangent().actor,
        {fabs(double(input.value)),input.error,input.ideal_magnitude}}};
}
#endif
struct SemanticTrainingViewSelection {
    uint64_t status,row_count,capacity,ordinal,basis,window,source_length,block_size,prefix_extent,answer_start;
    uint64_t identity[4],source_identity[4],content_identity[4];
    SemanticTrainingViewOriginRecord origin;
    uint64_t origin_candidate,training_rng[4];
};
struct SemanticTrainingObjectiveRecord {
    uint64_t evaluator_abi,identity[4],task_identity[4];
    uint64_t row_count,capacity,group_count,group_member_count,canary_count,protected_member_count;
    uint64_t evaluator_min_bits,evaluator_max_bits,coefficient_bits[9],cost_unit[4],cost_cap;
    uint64_t truth_tokens[4];
    uint64_t canary_source_offset,canary_source_words;
};
struct SemanticTrainingObjectiveGroupRecord {
    uint64_t kind,denominator,member_offset,member_count;
};
struct SemanticTrainingRosterRow {
    uint64_t ordinal,basis,window,source_length,block_size,prefix_extent,answer_start,origin_candidate;
    uint64_t identity[4],source_identity[4],content_identity[4];
    SemanticTrainingViewOriginRecord origin;
};
struct SemanticTrainingCanaryRecord {
    uint64_t evaluator_abi,kind,row_ordinal,lower_bound_bits,upper_bound_bits,memory_limit,work_limit;
    uint64_t obligation_member_offset,obligation_member_count,protected_member_offset,protected_member_count;
    uint64_t row_identity[4],row_content_identity[4],task_identity[4],identity[4];
};
struct SemanticTrainingCanaryResultRecord {
    uint64_t abi,kind,row_ordinal,availability,reason,measurement_bits,memory_used,work_used,identity[4];
};
struct SemanticTrainingCanaryRefusalRecord {
    uint64_t abi,reason,kind,row_ordinal,measurement_bits,lower_bound_bits,upper_bound_bits;
    uint64_t memory_used,memory_limit,work_used,work_limit,identity[4],selection_identity[4];
};
struct ModelUpdateEvidenceSource {
    uint64_t lease,parent_header,result,origin,states[2];
};

struct ModelForwardReceipt {
    uint64_t abi,availability,role,generation,work_sequence,stream_state,scalar_type,row_count,capacity,vocabulary;
    uint64_t block_positions,row_seals,finite,model_geometry_digest[4];
    uint64_t logits_digest[4],identity[4];
};
struct ModelUpdateCanaryInputs {
    uint64_t control,lease,selection,roster_rows,objective,canaries,canary_count;
    uint64_t protected_members,protected_member_count,task,task_words,evidence,evidence_count,forward_receipts;
    uint64_t retention_labels,kinds;
    uint64_t baseline_state,candidate_state,baseline_row_seals,candidate_row_seals,comparison_state,comparison_receipt;
    uint64_t scalar_type,row_count,capacity,vocabulary,block_positions;
    ModelWorkInput model_work;
    uint64_t accounted_reserved_bytes,copy_bytes,admissibility_source,results;
    uint64_t admissibility_destination,refusal_destination,native_work;
};
struct PolicyAdjointNumericalDescriptor {
    uint64_t parameter_errors,text_errors,baseline_errors;
    uint64_t parameter_domains,text_domains,baseline_domains;
    uint64_t recurrent_errors,score_errors,coefficient_errors;
    uint64_t parameter_cotangents,text_cotangents,baseline_cotangents;
};
struct PreparedActorRefreshDevice {
    uint64_t abi,status,target_control,target_lease,target_header,target_bank,child_control,member_ordinal;
    uint64_t minimum_generation,stream_serial,family_id,proposal,program[4],phase[4];
    SemanticTrainingViewOriginRecord original_origin;
    uint64_t acquired_generation,model_geometry_digest[4];
    uint64_t role,demonstrated_lane,demonstrated_choices[COMPONENT_COUNT];
};
static_assert(sizeof(PreparedActorRefreshDevice)==1560,"prepared policy basis ABI");
struct PreparedModelContextContinuationDevice {
    uint64_t abi,status,source_control,source_lease,target_control,minimum_generation,stream_serial,family_id,proposal;
    uint64_t source_instance[4],target_instance[4],model_geometry_digest[4];
    uint64_t acquired_generation,source_word,target_word,training_cursor,training_rng[4];
};
static_assert(sizeof(PreparedModelContextContinuationDevice)==232,"prepared model context continuation ABI");
struct PolicyBackward {
    uint64_t cotangents,parameters,text,baselines,recurrent,scores,status,parameter_cells,text_cells;
    uint64_t selection,objective,objective_groups,objective_group_members,roster_rows,origin_candidate;
    uint64_t origin_lease,origin_bank,mode,member_ordinal,critic_term,target_header,derived_origin;
    uint64_t work,baseline_snapshot,score_cotangent_snapshot;
    PolicyAdjointNumericalDescriptor numerical;
};
static_assert(sizeof(PolicyBackward)==296 && offsetof(PolicyBackward,work)==176 &&
    offsetof(PolicyBackward,baseline_snapshot)==184 &&
    offsetof(PolicyBackward,score_cotangent_snapshot)==192 && offsetof(PolicyBackward,numerical)==200,
    "original policy backward work ABI");
struct Descriptor {
    uint64_t logits,support,scratch,receipts,state,components,codebooks,arena[7];
    PolicyDescriptor policy;
    PolicyBackward backward;
    uint64_t task,task_words,task_return_bound;
    uint64_t task_ground,task_ground_bytes,task_support_truths,task_support_lineage,task_support_lineage_words;
    PublicationCommand publication;
    TextBinding text;
    ModelWorkInput model_work;
    struct PreparedReplayDescriptor {
        struct Copy {
            uint64_t rows,count,bank,directory,directory_count,arena,arena_words,actual;
        } parent,successor;
    } replay;
    uint64_t policy_work,policy_score;
};
static_assert(sizeof(Descriptor)==1472 && offsetof(Descriptor,backward)==888 &&
    offsetof(Descriptor,policy_work)==1456 && offsetof(Descriptor,policy_score)==1464,
    "original descriptor policy work ABI");

using ReplayCopyDescriptor=Descriptor::PreparedReplayDescriptor::Copy;
__device__ PublicationBank* publication_acquired_bank(const PublicationControl&,const PublicationLease&,
        semantic_graph::NativeWorkTally* work=nullptr);
__device__ bool publication_model_cache_ready(const PublicationControl&,const PublicationBank&,
        semantic_graph::NativeWorkTally* work=nullptr);
struct ReplayCopyRow {
    uint64_t role,index,model,slots[2],offsets[2],bytes[2],destination,capacity;
};
static_assert(sizeof(ReplayCopyRow)==88 && sizeof(ReplayCopyDescriptor)==64,
    "private replay custody ABI");

// A logical model allocation can select a different bank from PublicationWord.
// Resolve its original directory references, never the most recently used bank.
__device__ uint64_t replay_copy_source(const PublicationControl& control,
        const PublicationBank& bank,const ReplayCopyRow& row,uint64_t* count,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(row.model>2 || (row.capacity && !row.destination) || row.capacity<row.bytes[0] || row.capacity<row.bytes[1] ||
       row.slots[0]>=control.storage_count || row.slots[1]>=control.storage_count)
        { semantic_content_integrity_trap();return 0; }
    const auto* directory=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    uint64_t selected=bank.header.publication_word&1;
    bool found=false;
    for(uint64_t i=0;i<bank.header.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& range=directory[i];
        const bool match=row.model ? range.role>=18 && range.role<=25 &&
            (range.storage_slot==row.slots[0] || range.storage_slot==row.slots[1]) :
            range.role==row.role && range.index==row.index;
        if(!match)continue;
        uint64_t candidate;
        if(range.storage_slot==row.slots[0] && (row.model || range.offset_bytes==row.offsets[0]))candidate=0;
        else if(range.storage_slot==row.slots[1] && (row.model || range.offset_bytes==row.offsets[1]))candidate=1;
        else { semantic_content_integrity_trap();return 0; }
        if(found && candidate!=selected) { semantic_content_integrity_trap();return 0; }
        selected=candidate;found=true;
        if(!row.model && range.length_bytes>row.bytes[selected]) { semantic_content_integrity_trap();return 0; }
    }
    if(!row.model && !found) { semantic_content_integrity_trap();return 0; }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    const auto& allocation=reinterpret_cast<const PublicationStorageEntry*>(control.storage)[row.slots[selected]];
    if(row.offsets[selected]>allocation.bytes || row.bytes[selected]!=allocation.bytes-row.offsets[selected] ||
       (allocation.bytes && !allocation.pointer) || allocation.pointer>UINT64_MAX-allocation.bytes ||
       row.destination>UINT64_MAX-row.capacity) { semantic_content_integrity_trap();return 0; }
    // Only native immutable replay owners create a borrowed model row. Both
    // descriptor banks name the same full original allocation; it is retained
    // and sealed, never copied onto itself or made writable by this snapshot.
    if(row.model==2 && (!found || row.slots[0]!=row.slots[1] || row.offsets[0] || row.offsets[1] ||
       row.bytes[0]!=row.bytes[1] || row.destination!=allocation.pointer || row.capacity!=allocation.bytes))
        { semantic_content_integrity_trap();return 0; }
    *count=row.bytes[selected];
    return allocation.pointer+row.offsets[selected];
}

__device__ uint64_t replay_copy_extent(const PublicationControl& control,
        const PublicationBank& bank,const ReplayCopyDescriptor& copy,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!copy.rows || !copy.count || !copy.bank || !copy.directory || !copy.actual ||
       copy.count>UINT64_MAX/sizeof(ReplayCopyRow) || bank.header.range_count!=copy.directory_count ||
       copy.directory_count>UINT64_MAX/sizeof(PublicationRange) || copy.arena_words>UINT64_MAX/8 ||
       (copy.arena_words && !copy.arena)) { semantic_content_integrity_trap();return 0; }
    uint64_t bytes=sizeof(PublicationBank)+3*sizeof(uint64_t);
    const uint64_t directory_bytes=copy.directory_count*sizeof(PublicationRange),arena_bytes=copy.arena_words*8;
    if(directory_bytes>UINT64_MAX-bytes || arena_bytes>UINT64_MAX-bytes-directory_bytes)
        { semantic_content_integrity_trap();return 0; }
    bytes+=directory_bytes+arena_bytes;
    const auto* rows=reinterpret_cast<const ReplayCopyRow*>(copy.rows);
    for(uint64_t i=0;i<copy.count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        uint64_t count=0;replay_copy_source(control,bank,rows[i],&count,work);
        if(rows[i].model==2)continue;
        if(count>UINT64_MAX-bytes) { semantic_content_integrity_trap();return 0; }
        bytes+=count;
    }
    return bytes;
}

__device__ void replay_copy_snapshot(const PublicationControl& control,
        const PublicationBank& bank,const ReplayCopyDescriptor& copy,uint64_t source_arena,
        uint64_t first,uint64_t stride,semantic_graph::NativeWorkTally* work=nullptr) {
    __shared__ uint64_t resolved_source,resolved_count,resolved_borrowed;
    const auto* rows=reinterpret_cast<const ReplayCopyRow*>(copy.rows);
    for(uint64_t row=0;row<copy.count;++row) {
        uint64_t count=0;
        uint64_t pointer=0,borrowed=0;
        if(!first) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            pointer=replay_copy_source(control,bank,rows[row],&count,work);
            borrowed=rows[row].model==2;
            if(stride>1) { resolved_source=pointer;resolved_count=count;resolved_borrowed=borrowed; }
        }
        if(stride>1) {
            __syncthreads();
            pointer=resolved_source;count=resolved_count;borrowed=resolved_borrowed;
        }
        if(!borrowed) {
            const auto* source=reinterpret_cast<const uint8_t*>(pointer);
            auto* destination=reinterpret_cast<uint8_t*>(rows[row].destination);
            for(uint64_t i=first;i<count;i+=stride)destination[i]=source[i];
        }
        if(stride>1)__syncthreads();
    }
    const auto* source=reinterpret_cast<const uint8_t*>(&bank);
    auto* destination=reinterpret_cast<uint8_t*>(copy.bank);
    for(uint64_t i=first;i<sizeof(PublicationBank);i+=stride)destination[i]=source[i];
    source=reinterpret_cast<const uint8_t*>(control.directories[bank.header.publication_word&1]);
    destination=reinterpret_cast<uint8_t*>(copy.directory);
    for(uint64_t i=first;i<copy.directory_count*sizeof(PublicationRange);i+=stride)destination[i]=source[i];
    if(copy.arena_words) {
        if(!source_arena) { semantic_content_integrity_trap();return; }
        const auto* arena=reinterpret_cast<const uint64_t*>(source_arena);
        auto* saved=reinterpret_cast<uint64_t*>(copy.arena);
        for(uint64_t i=first;i<copy.arena_words;i+=stride)saved[i]=arena[i];
    }
}

extern "C" __global__ void semantic_prepared_replay_parent(uint64_t control_ptr,
        uint64_t lease_ptr,uint64_t arena,ReplayCopyDescriptor copy,uint64_t native_work0,uint64_t native_work1) {
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(lease_ptr);
    if(lease.bank>1) { semantic_content_integrity_trap();return; }
    auto* work=threadIdx.x ? nullptr : reinterpret_cast<semantic_graph::NativeWorkTally*>(
        lease.bank ? native_work1 : native_work0);
    if(!threadIdx.x)semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    __shared__ PublicationBank* bank;
    __shared__ uint64_t bytes;
    if(threadIdx.x==0) {
        bank=publication_acquired_bank(control,lease,work);
        if(!bank) { semantic_content_integrity_trap();return; }
        bytes=replay_copy_extent(control,*bank,copy,work);
    }
    __syncthreads();
    replay_copy_snapshot(control,*bank,copy,arena,threadIdx.x,blockDim.x,work);
    __syncthreads();
    if(threadIdx.x==0) {
        auto* actual=reinterpret_cast<uint64_t*>(copy.actual);
        actual[0]=bytes;actual[1]=bank->header.publication_word;actual[2]=1;
    }
}

struct TaskProgramBank {
    semantic_program::Bank program;
    bool editable;
    semantic_task_ground::Task task;
    uint32_t selected_rule_count;
};

static constexpr uint32_t PROGRAM_FACT_CAPACITY=4096;
static constexpr uint32_t PROGRAM_RULE_CAPACITY=semantic_program::kRuleCapacity;
static constexpr uint64_t PROGRAM_FUEL=1000000;
static constexpr uint64_t PROGRAM_SCRATCH_OFFSET=4*1024*1024;
static_assert(PROGRAM_SCRATCH_OFFSET+2*PROGRAM_FACT_CAPACITY*sizeof(semantic_program::Fact)+
    PROGRAM_RULE_CAPACITY*sizeof(semantic_program::Rule)<=SCRATCH_BYTES,
    "resident program scratch exceeds the transition reservation");

__device__ bool task_program_bank(const uint64_t* task,uint64_t word_count,
        TaskProgramBank* result) {
    semantic_task_ground::Task view{};
    if(!result || !semantic_task_ground::decode(task,word_count,&view))return false;
    *result={view.program,view.editable,view};
    return true;
}

// The original task binding owns the complete signed return enclosure. Cover
// the actual raw RN32 critic reduction over every admitted baseline and legal
// count before any draw, denominator or coefficient can conceal overflow.
__device__ bool task_critic_admissible(uint64_t return_bound,const PolicyDescriptor& policy,
        semantic_graph::NativeWorkTally* work) {
#ifdef XLOG_SEMANTIC_POLICY
    if(return_bound>uint64_t(INT64_MAX) || !policy.component_baselines ||
       policy.component_baselines%alignof(float) || !policy.numerical.baseline_domains ||
       policy.numerical.baseline_domains%alignof(model_policy::DomainValue))return false;
    const auto* baselines=reinterpret_cast<const float*>(policy.component_baselines);
    const auto* domains=reinterpret_cast<const model_policy::DomainValue*>(policy.numerical.baseline_domains);
    // R is converted to RN32 by the original backward before subtraction.
    // The upward conversion covers both signs throughout [-bound,+bound].
    const float reward_magnitude=__ull2float_ru(return_bound);
    float raw=0.0f;
    for(uint32_t i=0;i<COMPONENT_COUNT;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
        const auto domain=domains[i];
        if(!isfinite(baselines[i]) || !model_policy::finite_domain(domain) ||
           domain.magnitude<fabs(double(baselines[i])))return false;
        const float difference=__double2float_ru(__dadd_ru(domain.magnitude,double(reward_magnitude)));
        const float square=__fmul_ru(difference,difference);
        // Legal-count masks can only omit nonnegative terms. Keeping all 136
        // encloses every prefix without borrowing a future selected receipt.
        raw=__fadd_ru(raw,square);
        if(!isfinite(raw))return false;
    }
    return true;
#else
    return false;
#endif
}

__device__ void retain_policy_cell(const PolicyDescriptor& policy,const Component& component,
        const Receipt& receipt,float maximum,U192 g,U192 active_sum,bool singleton) {
    if(!policy.z)return;
    auto& cell=reinterpret_cast<PolicyPwlCell*>(policy.pwl_cells)[component.ordinal];
    cell=PolicyPwlCell{};cell.abi=1;cell.ordinal=component.ordinal;
    cell.cardinality=component.cardinality;cell.legal_count=receipt.legal_count;
    cell.active_count=receipt.active_count;cell.selected_rank=receipt.selected_rank;
    cell.maximum_bits=uint64_t(__float_as_uint(maximum));cell.singleton=uint64_t(singleton);
    cell.g=g;cell.active_sum=active_sum;cell.selected_probability=receipt.p;
    cell.probability_denominator=receipt.q;
    auto& vjp=reinterpret_cast<PolicySelectedScoreVjp*>(policy.selected_score_vjps)[component.ordinal];
    vjp=PolicySelectedScoreVjp{};vjp.abi=1;vjp.ordinal=component.ordinal;
    vjp.choice=receipt.choice;vjp.active_count=receipt.active_count;vjp.selected_active=1;
    vjp.singleton=uint64_t(singleton);vjp.probability_numerator=receipt.p;
    vjp.probability_denominator=receipt.q;
}
static_assert(sizeof(U192)==24,"integer ABI");
static_assert(sizeof(Receipt)==296,"receipt ABI");
static_assert(sizeof(Work)==24,"semantic work ABI");
static_assert(sizeof(TaskFacts)==40,"task facts ABI");
static_assert(sizeof(TaskEvaluation)==160,"task evaluation ABI");
static_assert(sizeof(PolicyPwlCell)==160,"policy PWL cell ABI");
static_assert(sizeof(PolicySelectedScoreVjp)==96,"selected-score VJP ABI");
static_assert(sizeof(ActionBatchReceipt)==504,"action batch receipt ABI");
static_assert(sizeof(State)==4400,"state ABI");
static_assert(sizeof(PolicyField)==32,"policy field ABI");
static_assert(sizeof(PolicyNumericalDescriptor)==96 && sizeof(PolicyDescriptor)==776,"policy ABI");
static_assert(sizeof(SemanticTrainingViewOriginRecord)==256,"training view origin ABI");
static_assert(sizeof(SemanticTrainingViewSelection)==472,"training view selection ABI");
static_assert(sizeof(SemanticTrainingObjectiveRecord)==296,"training objective ABI");
static_assert(sizeof(SemanticTrainingObjectiveGroupRecord)==32,"training objective group ABI");
static_assert(sizeof(SemanticTrainingRosterRow)==416,"training roster row ABI");
static_assert(sizeof(SemanticTrainingCanaryRecord)==216,"training canary ABI");
static_assert(sizeof(SemanticTrainingCanaryResultRecord)==96,"training canary result ABI");
static_assert(sizeof(SemanticTrainingCanaryRefusalRecord)==152,"training canary refusal ABI");
static_assert(sizeof(ModelUpdateEvidenceSource)==48,"model update evidence ABI");
static_assert(sizeof(ModelForwardReceipt)==200,"model forward receipt ABI");
static_assert(offsetof(ModelForwardReceipt,identity)==168,"model forward receipt seal ABI");
static_assert(sizeof(ModelUpdateCanaryInputs)==296,"training canary input ABI");
static_assert(sizeof(PolicyAdjointNumericalDescriptor)==96,"policy adjoint numerical ABI");
static_assert(sizeof(PolicyBackward)==296,"policy backward ABI");

__device__ bool semantic_training_canary_refusal_valid(
        const SemanticTrainingCanaryRefusalRecord& refusal,
        const SemanticTrainingCanaryResultRecord* results,semantic_graph::NativeWorkTally* work) {
    const uint64_t pointer=reinterpret_cast<uint64_t>(results);
    if(!pointer || pointer%alignof(SemanticTrainingCanaryResultRecord) ||
       pointer>UINT64_MAX-5*sizeof(SemanticTrainingCanaryResultRecord) ||
       refusal.abi!=1 || refusal.reason>9)return false;
    const SemanticTrainingCanaryResultRecord* winner=nullptr;
    for(uint64_t item=0;item<5;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& result=results[item];
        if(result.abi!=1 || result.kind!=item+1 || result.availability>2 || result.reason>9 ||
           (!result.reason && result.availability!=2))return false;
        if(result.reason && (!winner || result.reason<winner->reason ||
           (result.reason==winner->reason && result.kind<winner->kind)))winner=&result;
    }
    if(!refusal.reason) {
        if(winner || refusal.kind || refusal.row_ordinal || refusal.measurement_bits ||
           refusal.lower_bound_bits || refusal.upper_bound_bits || refusal.memory_used ||
           refusal.memory_limit || refusal.work_used || refusal.work_limit)return false;
        for(uint32_t word=0;word<4;++word) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(refusal.identity[word] || refusal.selection_identity[word])return false;
        }
        return true;
    }
    if(!winner || refusal.kind!=winner->kind || refusal.reason!=winner->reason ||
       refusal.row_ordinal!=winner->row_ordinal ||
       refusal.measurement_bits!=winner->measurement_bits ||
       refusal.memory_used!=winner->memory_used || refusal.work_used!=winner->work_used)return false;
    for(uint32_t word=0;word<4;++word) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(refusal.identity[word]!=winner->identity[word])return false;
    }
    const double lower=__longlong_as_double((long long)refusal.lower_bound_bits);
    const double upper=__longlong_as_double((long long)refusal.upper_bound_bits);
    const double measurement=__longlong_as_double((long long)refusal.measurement_bits);
    if(!isfinite(lower) || !isfinite(upper) || lower>upper ||
       !refusal.memory_limit || !refusal.work_limit)return false;
    const uint64_t availability=winner->availability;
    if(refusal.reason==1)return availability==1 || (availability==2 && !isfinite(measurement));
    if(refusal.reason==2)return availability==2 && isfinite(measurement) &&
        (measurement<lower || measurement>upper);
    if(refusal.reason==3)return availability==2 && isfinite(measurement) &&
        measurement>=lower && measurement<=upper &&
        refusal.memory_used>refusal.memory_limit;
    // The final work-limit pass may outrank a calculation that was incomplete
    // or refused retention/goal checks before checking bounds and memory.
    if(refusal.reason==4)return refusal.work_used>refusal.work_limit;
    if(refusal.reason==5)return availability!=2;
    if(refusal.reason==6)return availability==2 && refusal.kind==2;
    if(refusal.reason==7)return (availability==1 && (refusal.kind==1 || refusal.kind==4)) ||
        (availability==2 && refusal.kind==4);
    if(refusal.reason==9)return availability==0;
    return refusal.reason==8;
}
static_assert(sizeof(SourceSlot)==64,"source slot ABI");
static_assert(sizeof(PublicationStorageEntry)==24,"owned storage ABI");
static_assert(sizeof(PublicationRange)==64,"publication range ABI");
static_assert(sizeof(PublicationHeader)==424,"publication header ABI");
static_assert(sizeof(PublicationControl)==144,"publication control ABI");
static_assert(sizeof(PublicationBank)==47128,"publication bank ABI");
static_assert(sizeof(ModelContractLayout)==40,"model contract byte layout ABI");
static_assert(sizeof(PublicationContract)==416,"publication contract ABI");
static_assert(sizeof(PendingContinuation)==360 && alignof(PendingContinuation)==8 &&
    offsetof(PendingContinuation,capacity_refusal)==352,"pending continuation ABI");
static_assert(sizeof(ContinuationInputs)==248 && alignof(ContinuationInputs)==8 &&
    offsetof(ContinuationInputs,capacity_refusal_source)==216 &&
    offsetof(ContinuationInputs,capacity_refusal)==224 &&
    offsetof(ContinuationInputs,capacity_allocation_ordinal)==232 &&
    offsetof(ContinuationInputs,capacity_bytes)==240,"continuation input ABI");
static_assert(sizeof(TextRow)==16,"compact text row ABI");
static_assert(sizeof(TextBinding)==24,"compact text binding ABI");
static_assert(sizeof(ModelUpdateBinding)==32,"model update binding ABI");
static_assert(sizeof(PublicationCommand)==32,"publication command ABI");
static_assert(sizeof(PublicationDeliveryInput)==88,"publication delivery input ABI");
static_assert(sizeof(PublicationLease)==88,"publication lease ABI");
static_assert(sizeof(PublicationStepResult)==sizeof(PublicationHeader)+32+sizeof(AttemptReceipt),"publication step result ABI");
static_assert(sizeof(PublicationTensorLayout)==112,"tensor layout ABI");
static_assert(sizeof(RawFeedbackRecord)==384,"raw feedback ABI");
static_assert(sizeof(CompletionCoverage)==360,"completion coverage ABI");
static_assert(sizeof(AttemptReceipt)==352,"attempt receipt ABI");
static_assert(sizeof(TokenProvenanceRecord)==216,"token provenance ABI");
static_assert(offsetof(TokenProvenanceRecord,successor_logical_digest)==184,"token provenance successor ABI");
static_assert(sizeof(IntentQueueHeader)==88,"intent queue ABI");
static_assert(sizeof(IntentEntry)==344,"intent entry ABI");
static_assert(sizeof(AcknowledgementQueueHeader)==72,"acknowledgement queue ABI");
static_assert(sizeof(AcknowledgementEntry)==176,"acknowledgement entry ABI");
static_assert(sizeof(Descriptor)==1472,"launch ABI");
static_assert((2*262144+4*65536)*sizeof(uint32_t)<=SCRATCH_BYTES,"serial scratch and completion witnesses");
#ifdef XLOG_SEMANTIC_POLICY
#include "semantic_policy_domains.cuh"
#endif

__device__ uint64_t text_row_count(TextBinding binding) {
    return binding.count && !(binding.count%alignof(uint64_t)) && binding.count<=UINT64_MAX-sizeof(uint64_t)
        ? *reinterpret_cast<const uint64_t*>(binding.count) : UINT64_MAX;
}
__device__ bool text_selected(TextBinding binding,uint32_t slot) {
    return slot<32 && binding.selected && binding.selected<=UINT64_MAX-32 &&
        reinterpret_cast<const uint8_t*>(binding.selected)[slot]==1;
}
__device__ bool text_any_selected(TextBinding binding,semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint32_t slot=0;slot<32;++slot) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(text_selected(binding,slot))return true;
    }
    return false;
}
__device__ uint64_t text_row_index(TextBinding binding,uint32_t slot) {
    const auto* rows=reinterpret_cast<const TextRow*>(binding.rows);
    const uint64_t count=text_row_count(binding);
    if(count>32)return UINT64_MAX;
    for(uint64_t i=0;i<count;++i)if(rows[i].source_slot==slot)return i;
    return UINT64_MAX;
}
__device__ bool validate_text_binding(const SourceSlot* source,TextBinding binding,
        semantic_graph::NativeWorkTally* work) {
    const uint64_t count=text_row_count(binding);
    if(!source || count>32 || !binding.rows || binding.rows%alignof(TextRow) ||
       binding.rows>UINT64_MAX-32*sizeof(TextRow) || !binding.selected || binding.selected>UINT64_MAX-32)return false;
    const auto* rows=reinterpret_cast<const TextRow*>(binding.rows);
    uint64_t seen=0,expected=0;
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(source[i].valid==1 && source[i].kind==2)expected|=uint64_t(1)<<i;
        const uint8_t selected=reinterpret_cast<const uint8_t*>(binding.selected)[i];
        if(selected>1 || (selected && !(expected&(uint64_t(1)<<i))))return false;
    }
    for(uint64_t i=0;i<count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(TextRow));
        const TextRow row=rows[i];if(row.source_slot>=32)return false;
        const uint64_t bit=uint64_t(1)<<row.source_slot;
        if((seen&bit) || !(expected&bit) || source[row.source_slot].logical_position!=row.logical_position)return false;
        seen|=bit;
    }
    return seen==expected;
}
__device__ bool text_null_receipt(const Receipt& r) {
    return r.kind==COMPONENT_KIND_TEXT && r.choice==TEXT_NULL && r.legal_count==1 && r.active_count==1 &&
        !r.selected_rank && !r.cumulative_mass_below.w[0] && !r.cumulative_mass_below.w[1] &&
        !r.cumulative_mass_below.w[2] && !r.active_fields && !r.null_fields && !r.cdf_start &&
        r.cdf_end==(uint64_t(1)<<63) && r.mass==r.cdf_end &&
        r.p.w[0]==1 && !r.p.w[1] && !r.p.w[2] && r.q.w[0]==1 && !r.q.w[1] && !r.q.w[2] &&
        r.factor_denominator.w[0]==1 && !r.factor_denominator.w[1] && !r.factor_denominator.w[2];
}
__device__ Receipt component_receipt(const Component& c,const State& state,
        semantic_graph::NativeWorkTally* work=nullptr,bool sample=true) {
    Receipt r={};r.proposal=state.proposal;r.catalogue_generation=CATALOGUE_GENERATION;
    for(int j=0;j<32;++j){r.catalogue_digest[j]=CATALOGUE_DIGEST[j];r.admission_binding[j]=state.binding_digest[j];}
    r.ordinal=c.ordinal;r.lane=c.lane;r.slot=c.slot;r.field=c.field;r.kind=c.kind;
    r.active_fields=c.kind==COMPONENT_KIND_EDIT ? NO_EDIT_ACTIVE_MASK : 0;
    r.null_fields=c.kind==COMPONENT_KIND_EDIT ? NO_EDIT_NULL_MASK : 0;
    if(!sample)return r;
    r.key[0]=uint32_t(CATALOGUE_GENERATION);r.key[1]=state.model_generation;
    uint64_t block=uint64_t(COMPONENT_COUNT)*state.proposal+c.ordinal;
    uint64_t stream=(state.stream_serial<<8)|state.family_id;
    r.counter[0]=uint32_t(block);r.counter[1]=uint32_t(block>>32);
    r.counter[2]=uint32_t(stream);r.counter[3]=uint32_t(stream>>32);
    philox(r.random_words,r.counter,r.key,work);
    r.draw=(uint64_t(r.random_words[0])|(uint64_t(r.random_words[1])<<32))&0x7fffffffffffffffULL;
    return r;
}

#include "semantic_conditional_policy.cuh"

// Source indices are ring identities, not the model's packed physical rows.
// Only FILLED rows that predate this invocation contribute to its coverage.
__device__ bool publication_text_position_limit(const PublicationContract& contract,uint64_t* limit) {
    return contract.window_capacity==32 &&
        semantic_graph::checked_add(contract.prefix_capacity,contract.window_capacity,limit);
}
__device__ uint64_t publication_validate_source(const PublicationBank& bank,
        const PublicationContract& contract,uint64_t* structural_end,
        semantic_graph::NativeWorkTally* work=nullptr) {
    uint64_t text_limit=0;
    if(!publication_text_position_limit(contract,&text_limit) || bank.header.ring_head>=32 ||
       bank.header.prefix_extent>contract.prefix_capacity ||
       bank.header.prefix_extent>contract.max_position)return 1;
    const uint64_t* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    uint64_t terminal_position=contract.max_position;
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const SourceSlot& s=bank.source[i];
        if(s.kind>2 || s.valid!=(s.kind!=0) || s.committed!=0 || s.recomputed>1)return 1;
        if(s.kind==0) {
            if(s.token!=contract.pad_token || s.logical_position || s.provenance ||
               s.recomputed || s.provenance_record)return 1;
            continue;
        }
        if(s.logical_position<bank.header.prefix_extent || s.logical_position>=text_limit ||
           s.logical_position>=contract.max_position)return 1;
        for(uint32_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(bank.source[j].valid && bank.source[j].logical_position==s.logical_position)return 1;
        }
        if(s.kind==2) {
            if(s.token!=MASK_TOKEN || s.provenance || s.provenance_record || s.recomputed)return 1;
        } else {
            if(s.token>=TEXT_CARDINALITY || (s.provenance!=1 && s.provenance!=2))return 1;
            for(uint64_t j=0;j<contract.terminal_token_count;++j) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(s.token==terminals[j] && s.logical_position<terminal_position)
                    terminal_position=s.logical_position;
            }
        }
    }
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(bank.source[i].kind==1 && bank.source[i].logical_position>terminal_position)return 1;
    }
    uint64_t end=bank.header.prefix_extent;
    for(uint32_t step=0;step<32;++step) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        bool filled=false;
        for(uint32_t i=0;i<32;++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            filled|=bank.source[i].kind==1 && bank.source[i].logical_position==end;
        }
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
struct LearningFoldAssignment {
    uint64_t original,scratch,expected,source,up,down,count,mode,phase,scale_bits,serial_scatter;
    PublicationTensorLayout layout,source_layout,up_layout,down_layout;
};
static_assert(sizeof(LearningFoldAssignment)==536,"learning absorption launch ABI");
__device__ uint64_t learning_fold_offset(PublicationTensorLayout layout,uint64_t ordinal) {
    uint64_t offset=0;
    for(uint64_t axis=layout.rank;axis>0;--axis) {
        offset+=(ordinal%layout.dimensions[axis-1])*layout.strides_bytes[axis-1];
        ordinal/=layout.dimensions[axis-1];
    }
    return offset;
}
__device__ float learning_fold_float(uint64_t pointer,PublicationTensorLayout layout,uint64_t ordinal) {
    const auto* bytes=reinterpret_cast<const uint8_t*>(pointer)+learning_fold_offset(layout,ordinal);
    uint32_t bits=0;
    for(uint64_t i=0;i<layout.element_bytes;++i)bits|=uint32_t(bytes[i])<<(8*i);
    return __uint_as_float(layout.scalar_type==5 ? bits<<16 : bits);
}
// A cold operation on original native-owned allocations. Numerical expectations
// precede all candidate writes; full physical aliases are checked byte-for-byte.
extern "C" __global__ void semantic_learning_fold(LearningFoldAssignment assignment,
        uint64_t stage,uint64_t status_ptr) {
    auto& status=*reinterpret_cast<uint64_t*>(status_ptr);
    if(publication_load(status))return;
    const uint64_t thread=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    const bool serial=stage==2 && assignment.serial_scatter;
    if(serial && thread)return;
    const uint64_t stride=serial ? 1 : uint64_t(gridDim.x)*blockDim.x;
    for(uint64_t ordinal=serial ? 0 : thread;ordinal<assignment.count;ordinal+=stride) {
        const uint64_t offset=learning_fold_offset(assignment.layout,ordinal);
        const uint64_t width=assignment.layout.element_bytes;
        auto* expected=reinterpret_cast<uint8_t*>(assignment.expected+ordinal*width);
        if(stage<=1) {
            uint64_t bits=0;
            if(assignment.mode==2) {
                const float value=learning_fold_float(assignment.source,assignment.source_layout,ordinal);
                if(!isfinite(value)) { publication_compare_exchange(status,0,1);return; }
                bits=__float_as_uint(value);
            } else if(assignment.mode==3)bits=assignment.phase;
            else if(assignment.mode==4) {
                const uint64_t columns=assignment.layout.dimensions[1];
                const uint64_t rank=assignment.down_layout.dimensions[0];
                const uint64_t row=ordinal/columns,column=ordinal%columns;
                const float weight=learning_fold_float(assignment.original,assignment.layout,ordinal);
                float sum=__uint_as_float(0);
                if(!isfinite(weight)) { publication_compare_exchange(status,0,1);return; }
                for(uint64_t j=0;j<rank;++j) {
                    const float up=learning_fold_float(assignment.up,assignment.up_layout,row*rank+j);
                    const float down=learning_fold_float(assignment.down,assignment.down_layout,j*columns+column);
                    if(!isfinite(up) || !isfinite(down)) { publication_compare_exchange(status,0,1);return; }
                    const float product=__fmul_rn(up,down);
                    sum=__fadd_rn(sum,product);
                    if(!isfinite(product) || !isfinite(sum)) { publication_compare_exchange(status,0,1);return; }
                }
                const float delta=__fmul_rn(__uint_as_float(uint32_t(assignment.scale_bits)),sum);
                const float value=__fadd_rn(weight,delta);
                if(!isfinite(delta) || !isfinite(value)) { publication_compare_exchange(status,0,1);return; }
                const uint32_t fp32=__float_as_uint(value);
                bits=assignment.layout.scalar_type==5 ? uint64_t((fp32+0x7fff+((fp32>>16)&1))>>16) : fp32;
                if(assignment.layout.scalar_type==5 && (bits&0x7f80)==0x7f80) {
                    publication_compare_exchange(status,0,1);return;
                }
            } else if(assignment.mode==5) {
                const auto* source=reinterpret_cast<const uint8_t*>(assignment.source)+learning_fold_offset(assignment.source_layout,ordinal);
                if(!isfinite(learning_fold_float(assignment.source,assignment.source_layout,ordinal))) {
                    publication_compare_exchange(status,0,1);return;
                }
                for(uint64_t i=0;i<width;++i)bits|=uint64_t(source[i])<<(8*i);
            }
            for(uint64_t i=0;i<width;++i)expected[i]=uint8_t(bits>>(8*i));
        } else if(stage==2) {
            auto* scratch=reinterpret_cast<uint8_t*>(assignment.scratch)+offset;
            for(uint64_t i=0;i<width;++i)scratch[i]=expected[i];
        } else {
            const auto* original=reinterpret_cast<const uint8_t*>(assignment.original)+offset;
            const auto* scratch=reinterpret_cast<const uint8_t*>(assignment.scratch)+offset;
            for(uint64_t i=0;i<width;++i)if(scratch[i]!=(assignment.mode ? expected[i] : original[i])) {
                publication_compare_exchange(status,0,1);return;
            }
        }
    }
}
__device__ bool publication_identity_equal(const uint64_t* a,const uint64_t* b,
        semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint32_t i=0;i<4;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(a[i]!=b[i])return false;
    }
    return true;
}
__device__ bool publication_header_equal(const PublicationHeader& a,const PublicationHeader& b,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* left=reinterpret_cast<const uint64_t*>(&a);
    const auto* right=reinterpret_cast<const uint64_t*>(&b);
    for(uint64_t i=0;i<sizeof(PublicationHeader)/sizeof(uint64_t);++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(left[i]!=right[i])return false;
    }
    return true;
}
__device__ PublicationBank* publication_bank(PublicationControl& control,uint64_t word) {
    return reinterpret_cast<PublicationBank*>(control.banks[word&1]);
}
__device__ bool publication_bank_matches(const PublicationControl& control,
        const PublicationBank& bank,uint64_t word,semantic_graph::NativeWorkTally* work=nullptr) {
    return bank.header.abi==1 && bank.header.sealed_epoch==word>>1 &&
        bank.header.publication_word==word && publication_identity_equal(bank.header.instance,control.instance,work);
}
__device__ PublicationBank* publication_acquired_bank(const PublicationControl& control,
        const PublicationLease& lease,semantic_graph::NativeWorkTally* work) {
    if(control.abi!=1 || lease.abi!=1 || lease.status || lease.active!=1 || lease.bank>1 ||
       lease.bank!=(lease.word&1) || lease.epoch!=lease.word>>1 || !control.reader_counts[lease.bank] ||
       !publication_identity_equal(lease.instance,control.instance,work) || !control.banks[lease.bank])return nullptr;
    auto* bank=reinterpret_cast<PublicationBank*>(control.banks[lease.bank]);
    return publication_bank_matches(control,*bank,lease.word,work) ? bank : nullptr;
}
// The gate only closes the acquire-load/count-increment race. The PublicationWord
// remains the sole canonical selector; no reader derives a bank from this gate.
__device__ uint64_t publication_header_eligibility(const PublicationControl& control,
        const PublicationBank& bank,uint64_t word,uint64_t transition_kind,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!publication_bank_matches(control,bank,word,work))return 3;
    if(control.reader_counts[(word&1)^1])return 2;
    if((word>>1)==(UINT64_MAX>>1))return 5;
    if(bank.header.terminal>1)return 4;
    if(transition_kind<1 || transition_kind>4 ||
       (bank.header.terminal==1)!=(transition_kind==3))return 1;
    // Absorption changes the effective model generation without pretending to
    // recompute old caches. Only the real Recompute can make them usable again.
    if(transition_kind!=2 && !publication_model_cache_ready(control,bank,work))return 1;
    return bank.header.fuel<((transition_kind==1 || transition_kind==3) ? 2 : 1) ? 5 : 0;
}
__device__ uint64_t publication_acquire(PublicationControl& control,PublicationLease& lease,
        uint64_t requested_kind=0,uint64_t expected_word=UINT64_MAX,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(control.abi!=1 || lease.active || !control.banks[0] || !control.banks[1])return 1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    const uint64_t word=publication_load(control.word),bank=word&1;
    uint64_t status=0;
    const auto& parent=*publication_bank(control,word);
    const uint64_t kind=requested_kind>=1 && requested_kind<=4 && parent.header.terminal==1 ? 3 : requested_kind;
    if(requested_kind)status=expected_word!=UINT64_MAX && word!=expected_word ? 3 : publication_header_eligibility(control,parent,word,kind);
    else if(!publication_bank_matches(control,parent,word,work) ||
       (parent.header.terminal==1 && !parent.header.fuel))status=1;
    if(!status &&
       control.reader_counts[bank]==UINT64_MAX)status=1;
    if(!status) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(PublicationLease)+88);
        ++control.reader_counts[bank];
        lease=PublicationLease{};lease.abi=1;lease.word=word;lease.bank=bank;lease.epoch=word>>1;
        lease.transition_kind=kind;
        for(uint32_t i=0;i<4;++i)lease.instance[i]=control.instance[i];
        publication_store(lease.active,1);
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,16);
    publication_store(control.reader_gate,0);
    lease.status=status;
    return status;
}
// Session schedules this only after every consumer's stream join. A replayed or
// foreign lease cannot decrement a later reader's count, including after reuse.
__device__ uint64_t publication_release(PublicationControl& control,PublicationLease& lease,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(control.abi!=1 || lease.abi!=1 || lease.bank>1 || lease.bank!=(lease.word&1) ||
       lease.epoch!=lease.word>>1 || !publication_identity_equal(lease.instance,control.instance,work))return 1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    uint64_t status=0;
    if(!publication_bank_matches(control,*publication_bank(control,lease.word),lease.word,work) ||
       !control.reader_counts[lease.bank])status=1;
    else {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
        if(!publication_compare_exchange(lease.active,1,0))status=1;
        else {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
            --control.reader_counts[lease.bank];
        }
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,16);
    publication_store(control.reader_gate,0);
    lease.status=status;
    return status;
}
__device__ uint64_t publication_prepare_text(const PublicationBank& base,PublicationBank& next,
        const PublicationContract& contract,const Receipt* receipts,uint64_t winner,uint64_t structural_end,TextBinding text,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(winner>2 || structural_end<base.header.prefix_extent)return 1;
    next.header=base.header;next.header.prefix_extent=structural_end;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationHeader)+sizeof(uint64_t));
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        next.source[i]=base.source[i];
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(SourceSlot));
        SourceSlot& s=next.source[i];
        if(s.kind==1 && s.logical_position<structural_end) {
            s=SourceSlot{};s.token=contract.pad_token;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(SourceSlot)+sizeof(uint64_t));
        } else if(s.kind==2 && winner && text_selected(text,i)) {
            if(receipts[(winner-1)*68+i].choice>=TEXT_CARDINALITY)return 1;
            s.token=receipts[(winner-1)*68+i].choice;s.kind=1;s.provenance=2;
            s.committed=0;s.recomputed=0;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,5*sizeof(uint64_t));
        }
    }
    uint64_t next_end=0;
    if(publication_validate_source(next,contract,&next_end,work))return 1;
    const uint64_t* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    for(uint32_t i=0;i<32;++i) {
      semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
      if(next.source[i].kind==1)
        for(uint64_t j=0;j<contract.terminal_token_count;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(next.source[i].token==terminals[j]) {
                // Every preceding logical position must already be contiguous.
                if(next_end<=next.source[i].logical_position)return 1;
                next.header.terminal=1; // Drain required; not an output intent.
                next.header.terminal_position=next.source[i].logical_position;
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
            }
        }
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
    // Genuine empty model tensors and an unreserved replay payload retain an
    // owned zero-byte entry. Null is their data pointer, not a missing owner.
    if(!storage.bytes) {
        const bool empty_model=range.role>=18 && range.role<=20;
        const bool empty_payload=range.role==57 && !range.index && control.contract &&
            !reinterpret_cast<const PublicationContract*>(control.contract)->replay_payload_capacity_bytes;
        return !storage.pointer && (empty_model || empty_payload) &&
            !range.offset_bytes && !range.length_bytes && !range.logical_begin && !range.logical_end;
    }
    if(!storage.pointer)return false;
    *bytes=reinterpret_cast<uint8_t*>(storage.pointer+range.offset_bytes);
    return true;
}
__device__ uint8_t* publication_range_bytes(const PublicationControl& control,const PublicationRange& range) {
    uint8_t* bytes=nullptr;
    return publication_resolve_range(control,range,&bytes) ? bytes : nullptr;
}
__device__ const PublicationRange* publication_find_range(const PublicationRange* ranges,uint64_t count,
        uint64_t role,uint64_t index=0,semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint64_t i=0;i<count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(ranges[i].role==role && ranges[i].index==index)return &ranges[i];
    }
    return nullptr;
}
__device__ bool publication_model_cache_ready(const PublicationControl& control,const PublicationBank& bank,
        semantic_graph::NativeWorkTally* work) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* range=publication_find_range(ranges,bank.header.range_count,14,0,work);
    if(!range || range->length_bytes!=sizeof(CompletionCoverage))return false;
    const auto* bytes=publication_range_bytes(control,*range);
    if(!bytes)return false;
    const auto& coverage=*reinterpret_cast<const CompletionCoverage*>(bytes);
    return coverage.abi==1 && coverage.model_generation==bank.header.model_generation;
}
__device__ bool publication_model_block_identity(const PublicationControl& control,
        const PublicationBank& bank,const PublicationContract& contract,
        const uint64_t* schema_digest,const uint64_t* identity,semantic_graph::NativeWorkTally* work) {
    if(!control.directories[bank.header.publication_word&1])return false;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(
        control.directories[bank.header.publication_word&1]);
    const auto* model=publication_find_range(ranges,bank.header.range_count,44,0,work);
    if(!model || !model->length_bytes)return false;
    const auto* bytes=publication_range_bytes(control,*model);
    const auto& layout=contract.model_contract_layout;
    if(!bytes || layout.schema_digest_offset>model->length_bytes ||
       32>model->length_bytes-layout.schema_digest_offset ||
       layout.identity_offset>model->length_bytes ||
       32>model->length_bytes-layout.identity_offset)return false;
    const auto* schema_bytes=reinterpret_cast<const uint8_t*>(schema_digest);
    const auto* identity_bytes=reinterpret_cast<const uint8_t*>(identity);
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(bytes[layout.schema_digest_offset+i]!=schema_bytes[i] ||
           bytes[layout.identity_offset+i]!=identity_bytes[i])return false;
    }
    return true;
}
struct TensorLayoutTableView {
    const TensorLayoutTableHeader* header;
    const PublicationTensorLayout* layouts;
    const ActiveRow* rows;
};
__device__ bool publication_tensor_table(const PublicationControl& control,
        const PublicationRange* ranges,uint64_t count,TensorLayoutTableView* view,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* table=publication_find_range(ranges,count,55,0,work);
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
        const PublicationRange* ranges,uint64_t count,uint64_t role,uint64_t index,
        semantic_graph::NativeWorkTally* work=nullptr) {
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,count,&table,work))return nullptr;
    for(uint64_t i=0;i<table.header->layout_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(table.layouts[i].role==role && table.layouts[i].index==index)return &table.layouts[i];
    }
    return nullptr;
}

__device__ bool publication_active_map_shape(const TensorLayoutTableView& table,const PublicationContract& contract,
        semantic_graph::NativeWorkTally* work=nullptr) {
    uint64_t capacity=0,text_limit=0,position_end=0;
    if(!publication_text_position_limit(contract,&text_limit) ||
       !semantic_graph::checked_add(contract.window_capacity,contract.feedback_capacity,&capacity) ||
       !semantic_graph::checked_add(text_limit,contract.feedback_capacity,&position_end) ||
       table.header->active_row_count>capacity)return false;
    uint64_t text_slots=0,last_kind=0,last_position=0;
    for(uint64_t i=0;i<table.header->active_row_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
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
struct PublicationTensorView { uint64_t dimensions[4]; };
// Resolve logical tensor geometry without reading its numerical payload.
// The resulting view also frames bounded control-record identities.
__device__ uint64_t publication_tensor_geometry(const uint8_t* data,uint64_t length,const PublicationRange& range,
        const PublicationTensorLayout* layout,PublicationTensorView* view,uint64_t* content_bytes,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const uint64_t pointer=reinterpret_cast<uint64_t>(data);
    if(range.logical_end<range.logical_begin || range.length_bytes!=length ||
       (length && !data) || pointer>UINT64_MAX-length)return 1;
    *content_bytes=length;
    if(!layout)return 0;
    // U8, U32, U64, F16, BF16, F32, I64, Bool8 and F64 retain distinct type codes.
    const uint64_t sizes[10]={0,1,4,8,2,2,4,8,1,8};
    if(layout->role!=range.role || layout->index!=range.index || layout->rank>4 ||
       layout->scalar_type<1 || layout->scalar_type>9 || layout->element_bytes!=sizes[layout->scalar_type] ||
       (layout->logical_axis!=UINT64_MAX && layout->logical_axis>=layout->rank) || pointer%layout->element_bytes)return 1;
    bool empty=false;
    uint32_t axes[4],axis_count=0;
    for(uint32_t i=0;i<layout->rank;++i) {
        if(layout->strides_bytes[i]%layout->element_bytes)return 1;
        uint64_t dimension=layout->dimensions[i];
        if(uint64_t(i)==layout->logical_axis) {
            dimension=range.logical_end-range.logical_begin;
            if(dimension>layout->dimensions[i])return 1;
        }
        view->dimensions[i]=dimension;
        empty|=dimension==0;
        if(layout->dimensions[i]>1)axes[axis_count++]=i;
    }
    for(uint64_t i=layout->rank;i<4;++i)if(layout->dimensions[i] || layout->strides_bytes[i])return 1;
    // Sort physical axes only for the non-overlap check. Logical traversal
    // retains the coordinates of real transposed producer views.
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
        const PublicationTensorLayout* layout,PublicationTensorView* view,uint64_t* content_bytes,
        const TensorLayoutTableView* active_table=nullptr,semantic_graph::NativeWorkTally* work=nullptr) {
    uint8_t* bytes=nullptr;
    if(!publication_resolve_range(control,range,&bytes) ||
       publication_tensor_geometry(bytes,range.length_bytes,range,layout,view,content_bytes,work))return 1;
    if(!layout)return bytes || (range.role==57 && !range.length_bytes) ? 0 : 1;
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
        // map selects source and destination cells; holes are padding.
        // A pending full-capacity producer may have zero semantic rows, while
        // its published range uses zero length. Directory validation owns that
        // distinction and the computed/uncomputed flag.
        if(!active_table || !active_table->header || range.logical_begin ||
           layout->logical_axis>=layout->rank ||
           range.logical_end!=active_table->header->active_row_count ||
           range.logical_end>layout->dimensions[layout->logical_axis])return 1;
        for(uint64_t row=0;row<range.logical_end;++row) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(active_table->rows[row].physical_row>=layout->dimensions[layout->logical_axis] ||
               (row && active_table->rows[row-1].physical_row>=active_table->rows[row].physical_row))return 1;
        }
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


__device__ void semantic_content_integrity_trap() {
#ifdef __CUDACC__
    __trap();
#else
    __builtin_trap();
#endif
}

// The admission contract fixes the initial model and storage geometry, not the
// generation selected by a later publication. Resolve its structural schema,
// generation and original branch from the acquired bank.
__device__ uint64_t publication_validate_selected_model(const PublicationControl& control,
        const PublicationBank& bank,semantic_graph::NativeWorkTally* work=nullptr);


__device__ uint64_t publication_validate_directory(const PublicationControl& control,
        const PublicationContract& contract,PublicationRange* ranges,uint64_t count,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!ranges || count>contract.range_capacity || contract.role_count!=semantic_task_ground::kPublicationRole || !contract.role_counts)return 1;
    const auto* roles=reinterpret_cast<const PublicationRoleCount*>(contract.role_counts);
    uint64_t expected=0,tensor_count=0;
    for(uint64_t i=0;i<semantic_task_ground::kPublicationRole;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(roles[i].role!=i+1 || !semantic_graph::checked_add(expected,roles[i].count,&expected))return 1;
        if(publication_tensor_role(i+1))tensor_count+=roles[i].count;
        else if(i+1==16 ? !roles[i].count : roles[i].count!=1)return 1;
    }
    if(expected!=count || roles[3].count!=roles[4].count || roles[5].count!=roles[6].count ||
       roles[50].count!=roles[3].count || roles[51].count!=roles[4].count ||
       roles[52].count!=roles[5].count || roles[53].count!=roles[6].count || !roles[17].count)return 1;
    for(uint64_t role=8;role<=13;++role)if(roles[role-1].count!=1)return 1;
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,count,&table,work) || table.header->layout_count!=tensor_count ||
       !publication_active_map_shape(table,contract,work))return 1;
    const auto* layouts=table.layouts;
    for(uint64_t i=0;i<tensor_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(!publication_tensor_role(layouts[i].role) || !publication_find_range(ranges,count,layouts[i].role,layouts[i].index,work))return 1;
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(layouts[j].role==layouts[i].role && layouts[j].index==layouts[i].index)return 1;
        }
    }
    uint64_t cursor=0;
    for(uint64_t role=1;role<=semantic_task_ground::kPublicationRole;++role)for(uint64_t index=0;index<roles[role-1].count;++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        PublicationRange& range=ranges[cursor++];
        if(range.role!=role || range.index!=index)return 1;
        const auto* layout=publication_find_layout(control,ranges,count,role,index,work);
        if(publication_tensor_role(role)!=bool(layout))return 1;
        if(!publication_tensor_role(role) && role!=1 && role!=2 && role!=57 && !range.length_bytes)return 1;
        if(role==14 && range.length_bytes!=sizeof(CompletionCoverage))return 1;
        if(role==15 && (contract.feedback_capacity<roles[15].count || contract.feedback_capacity>UINT64_MAX/sizeof(RawFeedbackRecord) ||
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
        PublicationTensorView view{};uint64_t bytes=0;
        if(publication_tensor_view(control,range,layout,&view,&bytes,&table,work))return 1;
    }
    return 0;
}
__device__ bool publication_mutable_role(uint64_t role) {
    return role==1 || role==2 || role==3 || (role>=4 && role<=15) || (role>=18 && role<=25) || role==30 || role==31 || role==32 || role==33 || role==39 || role==44 ||
        (role>=51 && role<=semantic_task_ground::kPublicationRole);
}
__device__ uint64_t publication_validate_storage(const PublicationControl& control,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!control.storage || !control.storage_count)return 1;
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    for(uint64_t i=0;i<control.storage_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if((!storage[i].pointer != !storage[i].bytes) || !storage[i].generation ||
           storage[i].pointer>UINT64_MAX-storage[i].bytes)return 1;
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(storage[i].pointer<storage[j].pointer+storage[j].bytes &&
               storage[j].pointer<storage[i].pointer+storage[i].bytes)return 1;
        }
    }
    return 0;
}
__device__ uint64_t publication_validate_destinations(const PublicationControl& control,
        const PublicationRange* base,const PublicationRange* next,uint64_t count,
        semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint64_t i=0;i<count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
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
        } else if(!(base[i].role>=18 && base[i].role<=25))for(uint64_t j=0;j<count;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(next[i].storage_slot==base[j].storage_slot)return 1;
        }
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(publication_mutable_role(next[j].role) && next[i].storage_slot==next[j].storage_slot &&
               !((next[i].role>=18 && next[i].role<=25) && (next[j].role>=18 && next[j].role<=25)))return 1;
        }
    }
    return 0;
}
__device__ bool publication_match_active_row(const TensorLayoutTableView& table,uint64_t* cursor,
        uint64_t physical,uint64_t slot,uint64_t position,uint64_t kind,semantic_graph::NativeWorkTally* work) {
    if(*cursor>=table.header->active_row_count)return false;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    const auto& row=table.rows[(*cursor)++];
    return row.physical_row==physical && row.source_slot==slot && row.logical_position==position && row.kind==kind;
}
__device__ bool publication_validate_active_origin(const PublicationControl& control,const PublicationContract& contract,
        const PublicationBank& base,const PublicationRange* bank_ranges,const TensorLayoutTableView& table,
        semantic_graph::NativeWorkTally* work) {
    if(!table.header->active_computed || !publication_active_map_shape(table,contract,work))return false;
    const auto* feedback_range=publication_find_range(bank_ranges,base.header.range_count,15,0,work);
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
            for(uint32_t slot=0;slot<32;++slot) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(!(used&(1u<<slot)) && base.source[slot].valid==1 && base.source[slot].kind==kind &&
                    (chosen==32 || base.source[slot].logical_position<base.source[chosen].logical_position))chosen=slot;
            }
            if(chosen==32)break;
            used|=1u<<chosen;
            if(!publication_match_active_row(table,&cursor,physical++,chosen,base.source[chosen].logical_position,kind,work))return false;
        }
        if(kind==1)for(uint64_t slot=0;slot<contract.feedback_capacity;++slot,++physical) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(feedback[slot].valid>1)return false;
            if(feedback[slot].valid && !publication_match_active_row(table,&cursor,physical,32+slot,
                    contract.prefix_capacity+32+slot,3,work))return false;
        }
    }
    return cursor==table.header->active_row_count;
}
__device__ bool publication_model_spans_overlap(uint64_t left,uint64_t left_bytes,
        uint64_t right,uint64_t right_bytes) {
    return left_bytes && right_bytes && left<right+right_bytes && right<left+left_bytes;
}
__device__ bool publication_capacity_refusal_valid(const SemanticCapacityRefusalRecord& refusal,
        semantic_graph::NativeWorkTally* work) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,6);
    if(refusal.abi!=1 || refusal.reason>1)return false;
    if(!refusal.reason)return !refusal.allocation_ordinal && !refusal.requested_bytes &&
        !refusal.used_bytes && !refusal.capacity_bytes;
    return refusal.requested_bytes && refusal.used_bytes<=refusal.capacity_bytes &&
        refusal.requested_bytes>refusal.capacity_bytes-refusal.used_bytes;
}

__device__ uint64_t publication_validate_model_update(const PublicationControl& control,
        const PublicationBank& base,const PendingContinuation& pending,semantic_graph::NativeWorkTally* work) {
    if(pending.transition_kind!=4)
        return pending.model_update_bindings || pending.model_update_binding_count ||
            pending.model_update_admissibility || pending.model_update_refusal ||
            pending.model_update_canary_results || pending.model_update_canaries || pending.capacity_refusal;
    if(!pending.model_update_bindings ||
       pending.model_update_bindings%alignof(ModelUpdateBinding) ||
       !pending.model_update_binding_count ||
       pending.model_update_binding_count>base.header.range_count ||
       pending.model_update_binding_count>
           (UINT64_MAX-pending.model_update_bindings)/sizeof(ModelUpdateBinding))return 1;
    if(!pending.model_update_admissibility || pending.model_update_admissibility==UINT64_MAX ||
       *reinterpret_cast<const uint8_t*>(pending.model_update_admissibility)>1)return 1;
    if(!pending.model_update_refusal ||
       pending.model_update_refusal%alignof(SemanticTrainingCanaryRefusalRecord) ||
       pending.model_update_refusal>UINT64_MAX-sizeof(SemanticTrainingCanaryRefusalRecord))return 1;
    if(!pending.model_update_canary_results ||
       pending.model_update_canary_results%alignof(SemanticTrainingCanaryResultRecord) ||
       pending.model_update_canary_results>UINT64_MAX-5*sizeof(SemanticTrainingCanaryResultRecord))return 1;
    if(!publication_pointer_span(pending.model_update_canaries,5*sizeof(SemanticTrainingCanaryRecord),
        alignof(SemanticTrainingCanaryRecord)))return 1;
    if(pending.capacity_refusal && !publication_pointer_span(pending.capacity_refusal,
        sizeof(SemanticCapacityRefusalRecord),alignof(SemanticCapacityRefusalRecord)))return 1;
    if(base.header.neural_bank>1)return 1;
    const auto* bindings=reinterpret_cast<const ModelUpdateBinding*>(pending.model_update_bindings);
    const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t allocations=0;
    for(uint64_t i=0;i<base.header.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(old[i].role<18 || old[i].role>25)continue;
        bool first=true;
        for(uint64_t prior=0;prior<i;++prior) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(old[prior].role>=18 && old[prior].role<=25 &&
               old[prior].storage_slot==old[i].storage_slot)first=false;
        }
        if(!first)continue;
        ++allocations;bool found=false;
        for(uint64_t item=0;item<pending.model_update_binding_count;++item) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(bindings[item].slots[0]==old[i].storage_slot)found=true;
        }
        if(!found)return 1;
    }
    if(allocations!=pending.model_update_binding_count)return 1;
    for(uint64_t item=0;item<pending.model_update_binding_count;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& binding=bindings[item];
        if(binding.slots[0]>=control.storage_count || binding.slots[1]>=control.storage_count ||
           binding.slots[0]==binding.slots[1])return 1;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& original=storage[binding.slots[0]];
        if(original.bytes!=binding.bytes)return 1;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& candidate=storage[binding.slots[1]];
        if(candidate.bytes!=binding.bytes || binding.candidate!=candidate.pointer ||
           (binding.bytes && (!binding.candidate || binding.candidate>UINT64_MAX-binding.bytes ||
            original.pointer==candidate.pointer)))return 1;
        bool used=false;
        for(uint64_t i=0;i<base.header.range_count;++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(old[i].role>=18 && old[i].role<=25 &&
               old[i].storage_slot==binding.slots[0])used=true;
        }
        if(!used)return 1;
        for(uint64_t slot=0;slot<control.storage_count;++slot) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const auto& owned=storage[slot];
            if(slot!=binding.slots[1] && binding.bytes && owned.bytes &&
               publication_model_spans_overlap(binding.candidate,binding.bytes,owned.pointer,owned.bytes))return 1;
        }
        for(uint64_t prior=0;prior<item;++prior) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(binding.slots[0]==bindings[prior].slots[0] ||
               binding.slots[0]==bindings[prior].slots[1] ||
               binding.slots[1]==bindings[prior].slots[0] ||
               binding.slots[1]==bindings[prior].slots[1] ||
               publication_model_spans_overlap(binding.candidate,binding.bytes,
                   bindings[prior].candidate,bindings[prior].bytes))return 1;
        }
    }
    return 0;
}
__device__ uint64_t publication_validate_continuation(const PublicationControl& control,
        const PublicationContract& contract,const PublicationBank& base,const PendingContinuation& pending,uint64_t end,
        semantic_graph::NativeWorkTally* work) {
    if(pending.abi!=1 || !publication_identity_equal(pending.instance,base.header.instance,work) ||
       pending.base_word!=base.header.publication_word || pending.model_generation!=base.header.model_generation ||
       pending.authority_generation!=base.header.authority_generation ||
       !publication_identity_equal(pending.topology_identity,contract.topology_identity,work) ||
       !publication_identity_equal(pending.table_identity,contract.table_identity,work) || !pending.ranges ||
       pending.range_count>contract.range_capacity ||
       publication_validate_model_update(control,base,pending,work))return 1;
    if(!validate_text_binding(base.source,pending.text,work) || (pending.transition_kind!=1 && text_any_selected(pending.text,work)))return 1;
    const auto* bank_ranges=reinterpret_cast<const PublicationRange*>(control.directories[pending.base_word&1]);
    const auto* prefix=publication_find_range(bank_ranges,base.header.range_count,3,0,work);
    if(!prefix || prefix->length_bytes!=32 || !publication_range_bytes(control,*prefix) ||
       !publication_identity_equal(pending.prefix_identity,
           reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix)),work))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(pending.ranges);
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,pending.range_count,&table,work) ||
       !publication_validate_active_origin(control,contract,base,bank_ranges,table,work))return 1;
    uint64_t expected=3;
    for(uint64_t i=0;i<base.header.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if((bank_ranges[i].role>=4 && bank_ranges[i].role<=13) || (bank_ranges[i].role>=51 && bank_ranges[i].role<=54))++expected;
    }
    if(pending.range_count!=expected)return 1;
    if(table.header->layout_count!=expected-3)return 1;
    for(uint64_t i=0;i<table.header->layout_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& layout=table.layouts[i];
        if(!((layout.role>=4 && layout.role<=13) || (layout.role>=51 && layout.role<=54)) ||
           !publication_find_range(ranges,pending.range_count,layout.role,layout.index,work))return 1;
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(layout.role==table.layouts[j].role && layout.index==table.layouts[j].index)return 1;
        }
    }
    for(uint64_t i=0;i<pending.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const PublicationRange& r=ranges[i];
        const bool active=r.role>=51 && r.role<=54;
        if((r.role!=1 && r.role!=39 && r.role!=55 && (r.role<4 || r.role>13) && !active) || !publication_range_bytes(control,r))return 1;
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(ranges[j].role==r.role && ranges[j].index==r.index)return 1;
        }
        if(r.role==55) { if(r.index)return 1;continue; }
        if(r.role==39) {
            if(r.index || !r.length_bytes || r.logical_begin || r.logical_end)return 1;
            const auto* original=publication_find_range(bank_ranges,base.header.range_count,39,0,work);
            if(!original)return 1;
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(r.length_bytes>storage[original->storage_slot].bytes-original->offset_bytes)return 1;
            continue;
        }
        if(r.role==1 && (r.logical_begin!=base.header.prefix_extent || r.logical_end!=end))return 1;
        if((r.role==4 || r.role==5) &&
           (r.logical_begin!=(pending.transition_kind==4 ? 0 : base.header.prefix_extent) ||
            r.logical_end!=end))return 1;
        if(r.role>=6 && r.role<=13 && (r.logical_begin || r.logical_end))return 1;
        if(active && (r.logical_begin!=0 || r.logical_end!=table.header->active_row_count))return 1;
        const auto* original=publication_find_range(bank_ranges,base.header.range_count,r.role,r.index,work);
        if(!original)return 1;
        if(r.role==1) {
            if(r.index || r.length_bytes!=(end-base.header.prefix_extent)*sizeof(SourceSlot))return 1;
            const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,r));
            for(uint64_t k=0;k<end-base.header.prefix_extent;++k) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const SourceSlot& s=source[k];const SourceSlot* acquired=nullptr;
                for(uint32_t slot=0;slot<32;++slot) {
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    if(base.source[slot].kind==1 && base.source[slot].logical_position==base.header.prefix_extent+k)
                        acquired=&base.source[slot];
                }
                if(!acquired || s.kind!=1 || s.valid!=1 || s.committed!=1 || s.recomputed!=1 ||
                   s.logical_position!=acquired->logical_position || s.token!=acquired->token ||
                   s.provenance!=acquired->provenance || s.provenance_record!=acquired->provenance_record)return 1;
            }
        } else {
            const auto* from=publication_find_layout(control,ranges,pending.range_count,r.role,r.index,work);
            const auto* to=publication_find_layout(control,bank_ranges,base.header.range_count,r.role,r.index,work);
            if(!from || !to || from->rank!=to->rank || from->scalar_type!=to->scalar_type ||
               from->element_bytes!=to->element_bytes || from->logical_axis!=to->logical_axis)return 1;
            if(r.role>=6 && r.role<=13 && from->logical_axis!=UINT64_MAX)return 1;
            for(uint64_t axis=0;axis<to->rank;++axis) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(axis!=to->logical_axis && from->dimensions[axis]!=to->dimensions[axis])return 1;
            }
            if(active && to->logical_axis!=UINT64_MAX && r.logical_end>to->dimensions[to->logical_axis])return 1;
            if(active && (from->logical_axis!=((r.role==51 || r.role==52) ? 2 : 0) ||
                from->dimensions[from->logical_axis]!=32+contract.feedback_capacity))return 1;
            PublicationTensorView view{};uint64_t bytes=0;
            if(publication_tensor_view(control,r,from,&view,&bytes,&table,work))return 1;
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
        const PublicationLease& lease,ContinuationInputs inputs,semantic_graph::NativeWorkTally* work) {
    const uint64_t requested_kind=inputs.transition_kind;
    if((requested_kind==4)!=(inputs.training_selection!=0) ||
       (inputs.capacity_refusal_source==0)!=(inputs.capacity_refusal==0) ||
       (!inputs.capacity_refusal_source && (inputs.capacity_allocation_ordinal || inputs.capacity_bytes)) ||
       (inputs.capacity_refusal_source && (requested_kind!=4 ||
        !publication_pointer_span(inputs.capacity_refusal_source,sizeof(SemanticCapacityRefusalRecord),
            alignof(SemanticCapacityRefusalRecord)) ||
        !publication_pointer_span(inputs.capacity_refusal,sizeof(SemanticCapacityRefusalRecord),
            alignof(SemanticCapacityRefusalRecord)))) ||
       (inputs.training_selection && (inputs.training_selection%alignof(uint64_t) ||
        inputs.training_selection>UINT64_MAX-sizeof(uint64_t))) ||
       (inputs.model_update_bindings==0)!=(inputs.model_update_binding_count==0) ||
        (requested_kind==4)!=(inputs.model_update_bindings!=0) ||
        (requested_kind==4)!=(inputs.model_update_admissibility!=0) ||
        (requested_kind==4)!=(inputs.model_update_refusal!=0) ||
        (requested_kind==4)!=(inputs.model_update_canary_results!=0) ||
        (requested_kind==4)!=(inputs.model_update_canaries!=0) ||
       (inputs.model_update_bindings &&
        (inputs.model_update_bindings%alignof(ModelUpdateBinding) ||
         inputs.model_update_binding_count>
             (UINT64_MAX-inputs.model_update_bindings)/sizeof(ModelUpdateBinding))))return 1;
    if(lease.transition_kind)inputs.transition_kind=lease.transition_kind;
    if(inputs.capacity_refusal_source && inputs.transition_kind!=4)return 1;
    if(!publication_acquired_bank(control,lease,work) ||
       !control.directories[lease.bank] || !control.contract || !control.continuation ||
       control.contract%alignof(PublicationContract) || control.continuation%alignof(PendingContinuation) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract) ||
       control.continuation>UINT64_MAX-sizeof(PendingContinuation) || publication_validate_storage(control,work))return 1;
    const auto& base=*reinterpret_cast<const PublicationBank*>(control.banks[lease.bank]);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    auto& pending=*reinterpret_cast<PendingContinuation*>(control.continuation);
    uint64_t capacity=0,active_bytes=0,end=0;
    if(!publication_bank_matches(control,base,lease.word,work) || contract.abi!=1 ||
       base.header.range_count>contract.range_capacity || publication_validate_selected_model(control,base,work) ||
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
       !validate_text_binding(base.source,inputs.text,work) ||
       (inputs.transition_kind!=1 && text_any_selected(inputs.text,work)) ||
       publication_validate_source(base,contract,&end,work))return 1;
    if(inputs.capacity_refusal_source) {
        const auto& refusal=*reinterpret_cast<const SemanticCapacityRefusalRecord*>(inputs.capacity_refusal_source);
        if(!publication_capacity_refusal_valid(refusal,work) ||
           (refusal.reason && (refusal.allocation_ordinal!=inputs.capacity_allocation_ordinal ||
            refusal.capacity_bytes!=inputs.capacity_bytes ||
            *reinterpret_cast<const uint8_t*>(inputs.numerical_admissibility)!=0)))return 1;
        *reinterpret_cast<SemanticCapacityRefusalRecord*>(inputs.capacity_refusal)=refusal;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(SemanticCapacityRefusalRecord));
    }
    const uint64_t active_count=*reinterpret_cast<const uint64_t*>(inputs.active_row_count);
    if(active_count>capacity)return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(pending.ranges);
    const auto* bank_ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease.bank]);
    if(inputs.model_contract || inputs.model_contract_bytes) {
        // The native-acquired role44 snapshot carries this invocation's
        // structural schema, model generation and original branch identity.
        const auto& layout=contract.model_contract_layout;
        if(inputs.transition_kind!=1 || !inputs.numerical_blocks ||
           !inputs.model_contract || !inputs.model_contract_bytes ||
           inputs.model_contract>UINT64_MAX-inputs.model_contract_bytes ||
           layout.schema_digest_offset>inputs.model_contract_bytes ||
           32>inputs.model_contract_bytes-layout.schema_digest_offset ||
           layout.identity_offset>inputs.model_contract_bytes ||
           32>inputs.model_contract_bytes-layout.identity_offset ||
           inputs.model_schema_digest[0] || inputs.model_schema_digest[1] ||
           inputs.model_schema_digest[2] || inputs.model_schema_digest[3] ||
           inputs.model_identity[0] || inputs.model_identity[1] ||
           inputs.model_identity[2] || inputs.model_identity[3])return 1;
        const auto* original=reinterpret_cast<const uint8_t*>(inputs.model_contract);
        auto* schema_bytes=reinterpret_cast<uint8_t*>(inputs.model_schema_digest);
        auto* identity_bytes=reinterpret_cast<uint8_t*>(inputs.model_identity);
        for(uint32_t i=0;i<32;++i) {
            schema_bytes[i]=original[layout.schema_digest_offset+i];
            identity_bytes[i]=original[layout.identity_offset+i];
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2);
        }
        if(!(inputs.model_schema_digest[0] || inputs.model_schema_digest[1] ||
             inputs.model_schema_digest[2] || inputs.model_schema_digest[3]) ||
           !(inputs.model_identity[0] || inputs.model_identity[1] ||
             inputs.model_identity[2] || inputs.model_identity[3]))return 1;
    }
    // Cold binding requires this slot for a versioned physical roster; other
    // model schemas retain their separate ordinary numerical predicate.
    if(inputs.transition_kind==1 && inputs.numerical_blocks) {
        uint64_t block_bytes=0;
        if(!inputs.physical_blocks ||
           inputs.numerical_blocks%alignof(double) ||
           !semantic_graph::checked_mul(inputs.physical_blocks,6*sizeof(double),&block_bytes) ||
           inputs.numerical_blocks>UINT64_MAX-block_bytes ||
           !publication_model_block_identity(control,base,contract,
               inputs.model_schema_digest,inputs.model_identity,work))return 1;
    } else if(inputs.numerical_blocks || inputs.physical_blocks ||
              inputs.model_schema_digest[0] || inputs.model_schema_digest[1] ||
              inputs.model_schema_digest[2] || inputs.model_schema_digest[3] ||
              inputs.model_identity[0] || inputs.model_identity[1] ||
              inputs.model_identity[2] || inputs.model_identity[3])return 1;
    const auto* prefix=publication_find_range(bank_ranges,base.header.range_count,3,0,work);
    if(!prefix || prefix->length_bytes!=32 || !publication_range_bytes(control,*prefix))return 1;
    auto* authority=const_cast<PublicationRange*>(publication_find_range(ranges,pending.range_count,39,0,work));
    if(!authority)return 1;
    PublicationRange actual_authority=*authority;actual_authority.length_bytes=inputs.authority_bytes;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange)+sizeof(uint64_t));
    if(!publication_range_bytes(control,actual_authority))return 1;
    TensorLayoutTableView table{};
    if(!publication_tensor_table(control,ranges,pending.range_count,&table,work) ||
       table.header->active_computed!=1 || table.header->active_row_count>capacity)return 1;
    auto* table_range=const_cast<PublicationRange*>(publication_find_range(ranges,pending.range_count,55,0,work));
    uint64_t layout_bytes=0,table_bytes=0;
    if(!semantic_graph::checked_mul(table.header->layout_count,sizeof(PublicationTensorLayout),&layout_bytes) ||
       !semantic_graph::checked_add(sizeof(TensorLayoutTableHeader),layout_bytes,&table_bytes))return 1;
    PublicationRange full_table=*table_range;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
    if(!semantic_graph::checked_add(table_bytes,active_bytes,&full_table.length_bytes) ||
       !publication_range_bytes(control,full_table))return 1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    uint64_t row_bytes=0;
    if(!semantic_graph::checked_mul(active_count,sizeof(ActiveRow),&row_bytes) ||
       !semantic_graph::checked_add(table_bytes,row_bytes,&table_range->length_bytes))return 1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    auto* header=const_cast<TensorLayoutTableHeader*>(table.header);
    auto* rows=const_cast<ActiveRow*>(table.rows);
    header->active_row_count=active_count;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    const auto* actual_rows=reinterpret_cast<const ActiveRow*>(inputs.active_rows);
    for(uint64_t i=0;i<active_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        rows[i]=actual_rows[i];
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(ActiveRow));
    }
    if(!publication_validate_active_origin(control,contract,base,bank_ranges,table,work))return 1;
    authority->length_bytes=inputs.authority_bytes;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    pending.transition_kind=inputs.transition_kind;
    pending.base_word=lease.word;
    semantic_graph::copy_identity(pending.instance,base.header.instance);
    pending.model_generation=base.header.model_generation;
    pending.authority_generation=base.header.authority_generation;
    semantic_graph::copy_identity(pending.topology_identity,contract.topology_identity);
    semantic_graph::copy_identity(pending.table_identity,contract.table_identity);
    semantic_graph::copy_identity(pending.prefix_identity,
        reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix)));
    pending.text=inputs.text;
    pending.numerical_admissibility=inputs.transition_kind==3 ? 0 : inputs.numerical_admissibility;
    pending.numerical_blocks=inputs.numerical_blocks;
    pending.physical_blocks=inputs.physical_blocks;
    semantic_graph::copy_identity(pending.model_schema_digest,inputs.model_schema_digest);
    semantic_graph::copy_identity(pending.model_identity,inputs.model_identity);
    pending.training_selection=inputs.transition_kind==4 ? inputs.training_selection : 0;
    pending.model_update_bindings=inputs.transition_kind==4 ? inputs.model_update_bindings : 0;
    pending.model_update_binding_count=inputs.transition_kind==4 ? inputs.model_update_binding_count : 0;
    pending.model_update_admissibility=inputs.transition_kind==4 ? inputs.model_update_admissibility : 0;
    pending.model_update_refusal=inputs.transition_kind==4 ? inputs.model_update_refusal : 0;
    pending.model_update_canary_results=inputs.transition_kind==4 ? inputs.model_update_canary_results : 0;
    pending.model_update_canaries=inputs.transition_kind==4 ? inputs.model_update_canaries : 0;
    pending.capacity_refusal=inputs.transition_kind==4 ? inputs.capacity_refusal : 0;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PendingContinuation)-3*sizeof(uint64_t));
    for(uint64_t i=0;i<pending.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        auto& range=ranges[i];
        if(range.role==1 || range.role==4 || range.role==5) {
            range.logical_begin=(inputs.transition_kind==4 && (range.role==4 || range.role==5)) ?
                0 : base.header.prefix_extent;
            range.logical_end=end;
        } else if(range.role>=51 && range.role<=54) {
            range.logical_begin=0;range.logical_end=active_count;
        } else range.logical_begin=range.logical_end=0;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
        if(range.role!=1)continue;
        PublicationRange full_source=range;full_source.length_bytes=32*sizeof(SourceSlot);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange)+sizeof(uint64_t));
        if(!publication_range_bytes(control,full_source))return 1;
        const uint64_t count=end-base.header.prefix_extent;
        range.length_bytes=count*sizeof(SourceSlot);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
        auto* source=reinterpret_cast<SourceSlot*>(publication_range_bytes(control,range));
        if(!source)return 1;
        for(uint64_t row=0;row<count;++row) {
            const SourceSlot* original=nullptr;
            for(uint32_t slot=0;slot<32;++slot) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(base.source[slot].kind==1 &&
                    base.source[slot].logical_position==base.header.prefix_extent+row)original=&base.source[slot];
            }
            if(!original)return 1;
            source[row]=*original;source[row].committed=source[row].recomputed=1;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(SourceSlot)+2*sizeof(uint64_t));
        }
    }
    return publication_validate_continuation(control,contract,base,pending,end,work);
}
extern "C" __global__ void semantic_publication_prepare_continuation(uint64_t control_ptr,
        uint64_t lease_ptr,ContinuationInputs inputs,uint64_t native_work_ptr,
        uint64_t copied_bytes,uint64_t copy_commands) {
    if(blockIdx.x || threadIdx.x)return;
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(native_work_ptr);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    // Reaching this ordered command proves the original fixed D2D copy list
    // preceding it on this stream completed. Capacity is never an actual charge.
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,copy_commands);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,copied_bytes);
    if(!control_ptr || control_ptr%alignof(PublicationControl) || control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) || lease_ptr>UINT64_MAX-sizeof(PublicationLease) ||
       publication_prepare_continuation(*reinterpret_cast<PublicationControl*>(control_ptr),
           *reinterpret_cast<const PublicationLease*>(lease_ptr),inputs,work))semantic_content_integrity_trap();
}
__device__ double semantic_canary_logit(uint64_t pointer,uint64_t scalar_type,uint64_t offset) {
    const auto* bytes=reinterpret_cast<const uint8_t*>(pointer+offset);
    if(scalar_type==6)return double(*reinterpret_cast<const float*>(bytes));
    const uint16_t bits=*reinterpret_cast<const uint16_t*>(bytes);
    if(scalar_type==5)return double(__uint_as_float(uint32_t(bits)<<16));
    const uint64_t sign=bits>>15,exponent=(bits>>10)&31,fraction=bits&1023;
    double value=0.0;
    if(exponent==31)value=fraction ? NAN : INFINITY;
    else if(exponent)value=ldexp(double(1024+fraction),int(exponent)-25);
    else value=ldexp(double(fraction),-24);
    return sign ? -value : value;
}

__device__ bool semantic_canary_charge(ExecutionWork& work,semantic_graph::NativeWorkTally* native,
        uint64_t count,uint64_t units,
        semantic_graph::NativeWorkEvent event=semantic_graph::NativeWorkEvent::TableSlot) {
    if(units && count>UINT64_MAX/units) { native->overflow=work.overflow=1;return false; }
    semantic_graph::NativeWorkTally reached{};
    semantic_graph::charge_native(&reached,event,count*units);
    semantic_graph::charge_native(native,event,count*units);
    execution_work_merge_native(work,reached);
    if(native->overflow)work.overflow=1;
    return !work.overflow;
}
__device__ void semantic_canary_merge(ExecutionWork& work,semantic_graph::NativeWorkTally* native,
        const semantic_graph::NativeWorkTally& reached) {
    for(uint32_t i=0;i<9;++i)
        semantic_graph::charge_native(native,static_cast<semantic_graph::NativeWorkEvent>(i),reached.events[i]);
    execution_work_merge_native(work,reached);
    if(reached.overflow)native->overflow=work.overflow=1;
}

__device__ void publication_fold(uint64_t* digest,uint64_t role,uint64_t index,const uint64_t* content,
        semantic_graph::NativeWorkTally* work=nullptr);
__device__ uint64_t publication_validate_selected_model(const PublicationControl& control,
        const PublicationBank& bank,semantic_graph::NativeWorkTally* work);

struct ModelCanaryLogitLayout {
    uint64_t scalar_type,vocabulary,position_stride_bytes,vocabulary_stride_bytes;
};

__device__ bool semantic_canary_argmax(const ModelCanaryLogitLayout& inputs,uint64_t pointer,
        uint64_t position,uint64_t& token,ExecutionWork& work,
        semantic_graph::NativeWorkTally* native_work) {
    const uint64_t base=position*inputs.position_stride_bytes;
    double maximum=-INFINITY;token=0;
    for(uint64_t vocabulary=0;vocabulary<inputs.vocabulary;++vocabulary) {
        semantic_canary_charge(work,native_work,1,1);
        const double value=semantic_canary_logit(pointer,inputs.scalar_type,
            base+vocabulary*inputs.vocabulary_stride_bytes);
        if(!isfinite(value))return false;
        if(value>maximum) { maximum=value;token=vocabulary; }
    }
    return true;
}

__device__ bool semantic_canary_log_normalizer(const ModelCanaryLogitLayout& inputs,uint64_t pointer,
        uint64_t position,double& maximum,double& logarithm,ExecutionWork& work,
        semantic_graph::NativeWorkTally* native_work) {
    const uint64_t base=position*inputs.position_stride_bytes;
    maximum=-INFINITY;
    for(uint64_t vocabulary=0;vocabulary<inputs.vocabulary;++vocabulary) {
        semantic_canary_charge(work,native_work,1,1);
        const double value=semantic_canary_logit(pointer,inputs.scalar_type,
            base+vocabulary*inputs.vocabulary_stride_bytes);
        if(!isfinite(value))return false;
        if(value>maximum)maximum=value;
    }
    double sum=0.0;
    for(uint64_t vocabulary=0;vocabulary<inputs.vocabulary;++vocabulary) {
        semantic_canary_charge(work,native_work,1,2);
        const double value=semantic_canary_logit(pointer,inputs.scalar_type,
            base+vocabulary*inputs.vocabulary_stride_bytes);
        sum=__dadd_rn(sum,exp(__dsub_rn(value,maximum)));
    }
    if(!isfinite(sum) || !(sum>0.0))return false;
    logarithm=log(sum);
    return isfinite(logarithm);
}

#include "semantic_model_forward_stream.cuh"
extern "C" __global__ void semantic_publication_prepare_model_update_admissibility(
        ModelUpdateCanaryInputs inputs) {
    if(blockIdx.x || threadIdx.x)return;
    if(!publication_pointer_span(inputs.native_work,sizeof(semantic_graph::NativeWorkTally),
            alignof(semantic_graph::NativeWorkTally))) { semantic_content_integrity_trap();return; }
    auto* native_work=reinterpret_cast<semantic_graph::NativeWorkTally*>(inputs.native_work);
    // The original canary counter retains this replay's reached work. Binding
    // work remains in its original model scratch and joins at final publication.
    semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::Command,1);
    ExecutionWork work{};
    execution_work_merge_native(work,*native_work);
    if(!publication_pointer_span(inputs.baseline_state,sizeof(ModelForwardStreamState),alignof(ModelForwardStreamState)) ||
       !publication_pointer_span(inputs.candidate_state,sizeof(ModelForwardStreamState),alignof(ModelForwardStreamState)) ||
       !publication_pointer_span(inputs.comparison_state,sizeof(ModelCanaryStreamState),alignof(ModelCanaryStreamState))) {
        semantic_content_integrity_trap();return;
    }
    const auto& baseline=*reinterpret_cast<const ModelForwardStreamState*>(inputs.baseline_state);
    const auto& candidate=*reinterpret_cast<const ModelForwardStreamState*>(inputs.candidate_state);
    const auto& comparison=*reinterpret_cast<const ModelCanaryStreamState*>(inputs.comparison_state);
    // The three original producers finish before this one ordered gate. Their
    // reached events enter the selected model scratch once, also on refusal.
    semantic_stream_merge_work(baseline,candidate,comparison,work,native_work);
    if(!inputs.control || inputs.control%alignof(PublicationControl) ||
       !inputs.lease || inputs.lease%alignof(PublicationLease) ||
       !inputs.selection || inputs.selection%alignof(SemanticTrainingViewSelection) ||
       !inputs.roster_rows || inputs.roster_rows%alignof(SemanticTrainingRosterRow) ||
       !inputs.objective || inputs.objective%alignof(SemanticTrainingObjectiveRecord) ||
       !inputs.canaries || inputs.canaries%alignof(SemanticTrainingCanaryRecord) || inputs.canary_count!=5 ||
       !inputs.task || inputs.task%alignof(uint64_t) || inputs.task_words<39 ||
       (inputs.protected_member_count && (!inputs.protected_members ||
        inputs.protected_members%alignof(uint64_t))) ||
       !publication_pointer_span(inputs.forward_receipts,2*sizeof(ModelForwardReceipt),alignof(ModelForwardReceipt)) ||
       !publication_pointer_span(inputs.comparison_receipt,sizeof(ModelCanaryStreamReceipt),alignof(ModelCanaryStreamReceipt)) ||
       !inputs.retention_labels || inputs.retention_labels%alignof(int64_t) ||
       !inputs.kinds || inputs.kinds%alignof(int64_t) ||
       !inputs.admissibility_source || !inputs.results ||
       inputs.results%alignof(SemanticTrainingCanaryResultRecord) ||
       !inputs.admissibility_destination || !inputs.refusal_destination ||
       inputs.refusal_destination%alignof(SemanticTrainingCanaryRefusalRecord)) {
        semantic_content_integrity_trap();return;
    }
    if((inputs.scalar_type!=4 && inputs.scalar_type!=5 && inputs.scalar_type!=6) ||
       !inputs.row_count || !inputs.capacity || !inputs.vocabulary || !inputs.block_positions ||
       inputs.row_count>UINT64_MAX/(4*sizeof(uint64_t)) ||
       !publication_pointer_span(inputs.baseline_row_seals,inputs.row_count*4*sizeof(uint64_t),alignof(uint64_t)) ||
       !publication_pointer_span(inputs.candidate_row_seals,inputs.row_count*4*sizeof(uint64_t),alignof(uint64_t)) ||
       inputs.row_count>UINT64_MAX/sizeof(SemanticTrainingRosterRow) ||
       inputs.capacity>UINT64_MAX/inputs.row_count) {
        semantic_content_integrity_trap();return;
    }
    const auto& control=*reinterpret_cast<const PublicationControl*>(inputs.control);
    const auto& lease=*reinterpret_cast<const PublicationLease*>(inputs.lease);
    semantic_graph::NativeWorkTally input_work{};
    const auto* base=publication_acquired_bank(control,lease,&input_work);
    const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(inputs.selection);
    const auto* rows=reinterpret_cast<const SemanticTrainingRosterRow*>(inputs.roster_rows);
    const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(inputs.objective);
    const auto* canaries=reinterpret_cast<const SemanticTrainingCanaryRecord*>(inputs.canaries);
    const auto* task=reinterpret_cast<const uint64_t*>(inputs.task);
    const auto* protected_members=reinterpret_cast<const uint64_t*>(inputs.protected_members);
    const auto* forward_receipts=reinterpret_cast<const ModelForwardReceipt*>(inputs.forward_receipts);
    const auto& comparison_receipt=*reinterpret_cast<const ModelCanaryStreamReceipt*>(inputs.comparison_receipt);
    const auto* baseline_row_seals=reinterpret_cast<const uint64_t*>(inputs.baseline_row_seals);
    const auto* candidate_row_seals=reinterpret_cast<const uint64_t*>(inputs.candidate_row_seals);
    const auto* retention_labels=reinterpret_cast<const int64_t*>(inputs.retention_labels);
    auto* results=reinterpret_cast<SemanticTrainingCanaryResultRecord*>(inputs.results);
    const uint8_t numerical=*reinterpret_cast<const uint8_t*>(inputs.admissibility_source);
    TaskProgramBank program_bank{};
    const bool valid_inputs=!(numerical>1 || objective.evaluator_abi!=3 || objective.canary_count!=5 ||
       objective.row_count!=selection.row_count || objective.capacity!=inputs.capacity ||
       objective.protected_member_count!=inputs.protected_member_count ||
       !selection.row_count || selection.row_count>inputs.row_count || selection.capacity!=inputs.capacity ||
        task[0]!=9 || !task[1] ||
        !task_program_bank(task,inputs.task_words,&program_bank) ||
       !publication_identity_equal(objective.task_identity,task+2,&input_work));
    semantic_canary_merge(work,native_work,input_work);
    if(!valid_inputs) { semantic_content_integrity_trap();return; }
    const double evaluator_min=__longlong_as_double((long long)objective.evaluator_min_bits);
    const double evaluator_max=__longlong_as_double((long long)objective.evaluator_max_bits);
    if(!isfinite(evaluator_min) || !isfinite(evaluator_max) || evaluator_min>evaluator_max) {
        semantic_content_integrity_trap();return;
    }
    for(uint32_t left=0;left<4;++left) {
        semantic_canary_charge(work,native_work,1,1);
        if(objective.truth_tokens[left]>=inputs.vocabulary) { semantic_content_integrity_trap();return; }
        for(uint32_t right=left+1;right<4;++right) {
            semantic_canary_charge(work,native_work,1,1);
            if(objective.truth_tokens[left]==objective.truth_tokens[right]) {
                semantic_content_integrity_trap();return;
            }
        }
    }
    for(uint32_t query=0;query<program_bank.task.queries;++query) {
        semantic_canary_charge(work,native_work,1,1);
        const auto* row=program_bank.task.query(query);
        if(!(row[6]&(uint64_t(1)<<row[5]))) {
            semantic_content_integrity_trap();return;
        }
    }
    consume_model_work(inputs.model_work,work);
    semantic_canary_charge(work,native_work,inputs.copy_bytes,1,semantic_graph::NativeWorkEvent::CanonicalByte);
    uint64_t global_reason=0;
    if(!base || selection.status || inputs.model_work.count==0 || inputs.model_work.bound==0 ||
       work.model_events!=inputs.model_work.count || work.model_bound!=inputs.model_work.bound)
        global_reason=5;
    semantic_graph::NativeWorkTally receipt_work{};
    if(!global_reason && (!semantic_stream_forward_valid(baseline,forward_receipts[0],inputs.baseline_state,
                inputs.baseline_row_seals,0,base->header,&receipt_work) ||
             !semantic_stream_forward_valid(candidate,forward_receipts[1],inputs.candidate_state,
                inputs.candidate_row_seals,1,base->header,&receipt_work) ||
             baseline.row_count!=inputs.row_count || candidate.row_count!=inputs.row_count ||
             baseline.capacity!=inputs.capacity || candidate.capacity!=inputs.capacity ||
             baseline.vocabulary!=inputs.vocabulary || candidate.vocabulary!=inputs.vocabulary ||
             baseline.scalar_type!=inputs.scalar_type || candidate.scalar_type!=inputs.scalar_type ||
             baseline.finite>1 || candidate.finite>1 ||
             baseline.block_positions!=inputs.block_positions || candidate.block_positions!=inputs.block_positions ||
             !publication_identity_equal(baseline.selection_identity,selection.identity,&receipt_work) ||
             !publication_identity_equal(candidate.selection_identity,selection.identity,&receipt_work) ||
             comparison.publication_word!=base->header.publication_word ||
             comparison.scalar_type!=inputs.scalar_type || comparison.block_positions!=inputs.block_positions ||
             !semantic_stream_comparison_valid(comparison,comparison_receipt,forward_receipts,
                selection,canaries,inputs.row_count,inputs.capacity,inputs.vocabulary,&receipt_work) ||
             forward_receipts[0].work_sequence>=forward_receipts[1].work_sequence ||
             forward_receipts[1].work_sequence>inputs.model_work.count ||
            publication_identity_equal(forward_receipts[0].identity,forward_receipts[1].identity,&receipt_work)))
        global_reason=9;
    for(uint64_t item=0;!global_reason && item<5;++item) {
        semantic_graph::charge_native(&receipt_work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& row=comparison.rows[item];
        if(row.row_ordinal>=selection.row_count || row.row_ordinal>=inputs.row_count ||
           !publication_identity_equal(row.baseline_digest,baseline_row_seals+row.row_ordinal*4,&receipt_work) ||
           !publication_identity_equal(row.candidate_digest,candidate_row_seals+row.row_ordinal*4,&receipt_work))
            global_reason=9;
    }
    if(!global_reason && !semantic_stream_full_finite(baseline,candidate,&receipt_work))global_reason=1;
    semantic_canary_merge(work,native_work,receipt_work);
    uint64_t seen=0;
    SemanticTrainingCanaryRefusalRecord refusal{};refusal.abi=1;
    semantic_canary_charge(work,native_work,sizeof(refusal)+sizeof(uint64_t),1,semantic_graph::NativeWorkEvent::CanonicalByte);
    uint8_t admissible=numerical && !global_reason;
    for(uint64_t item=0;item<inputs.canary_count;++item) {
        semantic_graph::NativeWorkTally canary_work{};
        const auto& canary=canaries[item];
        if(canary.evaluator_abi!=3 || canary.kind!=item+1 ||
           canary.row_ordinal>=selection.row_count || (seen&(uint64_t(1)<<(canary.kind-1))) ||
           !publication_identity_equal(canary.task_identity,objective.task_identity,&canary_work)) {
            semantic_content_integrity_trap();return;
        }
        seen|=uint64_t(1)<<(canary.kind-1);
        const auto& row=rows[canary.row_ordinal];
        if(row.ordinal!=canary.row_ordinal || row.window>inputs.capacity ||
           !publication_identity_equal(row.identity,canary.row_identity,&canary_work) ||
           !publication_identity_equal(row.content_identity,canary.row_content_identity,&canary_work)) {
            semantic_content_integrity_trap();return;
        }
        semantic_canary_merge(work,native_work,canary_work);
        if(canary.kind!=1 && canary.kind!=4 && canary.obligation_member_count) {
            semantic_content_integrity_trap();return;
        }
        if(canary.kind==2) {
            if(canary.protected_member_offset>inputs.protected_member_count ||
               canary.protected_member_count>
                   inputs.protected_member_count-canary.protected_member_offset) {
                semantic_content_integrity_trap();return;
            }
            uint64_t previous=UINT64_MAX;
            for(uint64_t member=0;member<canary.protected_member_count;++member) {
                semantic_canary_charge(work,native_work,1,1);
                const uint64_t position=protected_members[canary.protected_member_offset+member];
                if(position>=row.window || (member && position<=previous) ||
                   retention_labels[canary.row_ordinal*inputs.capacity+position]<0) {
                    semantic_content_integrity_trap();return;
                }
                previous=position;
            }
        } else if(canary.protected_member_count) {
            semantic_content_integrity_trap();return;
        }
        const double lower=__longlong_as_double((long long)canary.lower_bound_bits);
        const double upper=__longlong_as_double((long long)canary.upper_bound_bits);
        if(!isfinite(lower) || !isfinite(upper) || lower>upper ||
           !canary.memory_limit || !canary.work_limit) {
            semantic_content_integrity_trap();return;
        }
        semantic_canary_charge(work,native_work,1,1);
        uint64_t reason=global_reason;
        // Availability describes this calculation, not whole-Update acceptance.
        uint64_t availability=reason && reason!=1 ? 0 : 1;
        double measurement=0.0;
        if(!reason && canary.kind!=5) {
            if(!semantic_stream_canary_measurement(comparison,item,canary,availability,reason,measurement)) {
                semantic_content_integrity_trap();return;
            }
        } else if(!reason) {
            const double memory_ratio=__ddiv_rn(double(inputs.accounted_reserved_bytes),double(canary.memory_limit));
            const double work_ratio=__ddiv_rn(double(work.raw),double(canary.work_limit));
            measurement=memory_ratio>work_ratio ? memory_ratio : work_ratio;
            availability=2;
        }
        if(!reason && !isfinite(measurement))reason=1;
        // The resource ratio is only a prefix here. Its original bounds and
        // memory decision belong to the complete-attempt publication gate.
        if(!reason && canary.kind!=5 && (measurement<lower || measurement>upper))reason=2;
        if(!reason && canary.kind!=5 && inputs.accounted_reserved_bytes>canary.memory_limit)reason=3;
        semantic_canary_charge(work,native_work,3*sizeof(SemanticTrainingCanaryResultRecord),1,semantic_graph::NativeWorkEvent::CanonicalByte);
        SemanticTrainingCanaryResultRecord result{};
        result.abi=1;result.kind=canary.kind;result.row_ordinal=canary.row_ordinal;
        result.availability=availability;result.reason=reason;
        result.measurement_bits=(uint64_t)__double_as_longlong(measurement);
        result.memory_used=inputs.accounted_reserved_bytes;result.work_used=work.raw;
        for(uint32_t word=0;word<4;++word)result.identity[word]=canary.identity[word];
        results[item]=result;
        if(reason && (!refusal.reason || reason<refusal.reason ||
           (reason==refusal.reason && result.kind<refusal.kind))) {
            semantic_canary_charge(work,native_work,sizeof(refusal)-sizeof(uint64_t),1,semantic_graph::NativeWorkEvent::CanonicalByte);
            refusal.reason=reason;refusal.kind=result.kind;refusal.row_ordinal=result.row_ordinal;
            refusal.measurement_bits=result.measurement_bits;
            refusal.lower_bound_bits=canary.lower_bound_bits;refusal.upper_bound_bits=canary.upper_bound_bits;
            refusal.memory_used=result.memory_used;refusal.memory_limit=canary.memory_limit;
            refusal.work_used=result.work_used;refusal.work_limit=canary.work_limit;
            for(uint32_t word=0;word<4;++word) {
                refusal.identity[word]=result.identity[word];refusal.selection_identity[word]=selection.identity[word];
            }
        }
        admissible &= !reason;
    }
    for(uint64_t item=0;item<inputs.canary_count;++item) {
        semantic_canary_charge(work,native_work,1,1);
        const auto& canary=canaries[item];
        auto& result=results[item];
        const uint64_t reason=work.overflow ? 8 : work.raw>canary.work_limit ? 4 : 0;
        if(reason && (!result.reason || reason<result.reason)) {
            semantic_canary_charge(work,native_work,sizeof(uint64_t),1,semantic_graph::NativeWorkEvent::CanonicalByte);
            result.reason=reason;
        }
        if(reason && (!refusal.reason || reason<refusal.reason ||
           (reason==refusal.reason && result.kind<refusal.kind))) {
            semantic_canary_charge(work,native_work,sizeof(refusal)-sizeof(uint64_t),1,semantic_graph::NativeWorkEvent::CanonicalByte);
            refusal.reason=reason;refusal.kind=result.kind;refusal.row_ordinal=result.row_ordinal;
            refusal.measurement_bits=result.measurement_bits;
            refusal.lower_bound_bits=canary.lower_bound_bits;refusal.upper_bound_bits=canary.upper_bound_bits;
            refusal.memory_used=result.memory_used;refusal.memory_limit=canary.memory_limit;
            refusal.work_used=result.work_used;refusal.work_limit=canary.work_limit;
            for(uint32_t word=0;word<4;++word) {
                refusal.identity[word]=result.identity[word];refusal.selection_identity[word]=selection.identity[word];
            }
        }
        admissible &= !reason;
    }
    semantic_canary_charge(work,native_work,sizeof(refusal)+sizeof(uint8_t)+5*sizeof(uint64_t)+
        (refusal.reason ? sizeof(uint64_t) : 0),1,semantic_graph::NativeWorkEvent::CanonicalByte);
    for(uint32_t item=0;item<5;++item)results[item].work_used=work.raw;
    if(refusal.reason)refusal.work_used=work.raw;
    semantic_graph::NativeWorkTally validation_work{};
    const bool valid_refusal=seen==31 && semantic_training_canary_refusal_valid(refusal,results,&validation_work);
    semantic_canary_merge(work,native_work,validation_work);
    if(!valid_refusal) {
        semantic_content_integrity_trap();return;
    }
    semantic_canary_charge(work,native_work,5*sizeof(uint64_t)+
        (refusal.reason ? sizeof(uint64_t) : 0),1,semantic_graph::NativeWorkEvent::CanonicalByte);
    for(uint32_t item=0;item<5;++item)results[item].work_used=work.raw;
    if(refusal.reason)refusal.work_used=work.raw;
    *reinterpret_cast<SemanticTrainingCanaryRefusalRecord*>(inputs.refusal_destination)=refusal;
    *reinterpret_cast<uint8_t*>(inputs.admissibility_destination)=admissible;
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
    pending.numerical_blocks=0;
    pending.physical_blocks=0;
    for(uint32_t i=0;i<4;++i) {
        pending.model_schema_digest[i]=0;
        pending.model_identity[i]=0;
    }
    pending.training_selection=0;
    pending.model_update_bindings=0;
    pending.model_update_binding_count=0;
    pending.model_update_admissibility=0;
    pending.model_update_refusal=0;
    pending.model_update_canary_results=0;
    pending.model_update_canaries=0;
    pending.capacity_refusal=0;
    semantic_graph::copy_identity(pending.instance,base->header.instance);
    semantic_graph::copy_identity(pending.topology_identity,contract.topology_identity);
    semantic_graph::copy_identity(pending.table_identity,contract.table_identity);
    semantic_graph::copy_identity(pending.prefix_identity,
        reinterpret_cast<const uint64_t*>(prefix_bytes));
}
__device__ void publication_copy_bytes(uint8_t* to,const uint8_t* from,uint64_t count,
        semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint64_t i=0;i<count;++i)to[i]=semantic_graph::written_byte(from[i],work);
}
__device__ uint64_t publication_preserve_range(const PublicationControl& control,
        const PublicationRange* original,uint64_t range_count,
        const PublicationRange& source,PublicationRange& destination,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(source.role!=destination.role || source.index!=destination.index ||
       source.storage_slot>=control.storage_count || destination.storage_slot>=control.storage_count ||
       source.storage_slot==destination.storage_slot || source.offset_bytes!=destination.offset_bytes)return 1;
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,2);
    const auto& from=storage[source.storage_slot];
    const auto& to=storage[destination.storage_slot];
    if(from.bytes!=to.bytes || source.offset_bytes>from.bytes ||
       destination.generation!=to.generation || (from.bytes && from.pointer==to.pointer))return 1;
    const auto* layout=publication_find_layout(control,original,range_count,source.role,source.index,work);
    TensorLayoutTableView table{};
    const bool tensor=(source.role>=4 && source.role<=13) ||
        (source.role>=18 && source.role<=25) || (source.role>=51 && source.role<=54);
    if(tensor && (!layout || !publication_tensor_table(control,original,range_count,&table,work)))return 1;
    const uint64_t slot=destination.storage_slot,generation=destination.generation;
    destination=source;destination.storage_slot=slot;destination.generation=generation;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange)+2*sizeof(uint64_t));
    uint8_t *source_bytes=nullptr,*destination_bytes=nullptr;
    if(!publication_resolve_range(control,source,&source_bytes) ||
       !publication_resolve_range(control,destination,&destination_bytes))return 1;
    // Retain padding and unused capacity as well as the visible logical bytes;
    // future native producers own the same frozen backing, not a short prefix.
    publication_copy_bytes(destination_bytes,source_bytes,from.bytes-source.offset_bytes,work);
    return 0;
}
__device__ uint64_t publication_apply_continuation(const PublicationControl& control,
        const PublicationBank& base,PublicationBank& next,const PendingContinuation& pending,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* from=reinterpret_cast<const PublicationRange*>(pending.ranges);
    const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[pending.base_word&1]);
    auto* to=reinterpret_cast<PublicationRange*>(control.directories[(pending.base_word&1)^1]);
    if(publication_validate_model_update(control,base,pending,work))return 1;
    TensorLayoutTableView pending_table{};
    if(!publication_tensor_table(control,from,pending.range_count,&pending_table,work))return 1;
    for(uint64_t i=0;i<next.header.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(!publication_mutable_role(old[i].role)) {
            to[i]=old[i];semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
        }
        if(old[i].role==1 || old[i].role==2 || old[i].role==31) {
            to[i]=old[i];semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
        }
        if((old[i].role==4 || old[i].role==5) && pending.transition_kind!=4) {
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            if(old[i].storage_slot>=control.storage_count || to[i].storage_slot>=control.storage_count ||
               old[i].storage_slot==to[i].storage_slot)return 1;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,2);
            const auto& source=storage[old[i].storage_slot];
            const auto& destination=storage[to[i].storage_slot];
            if(source.bytes!=destination.bytes || (source.bytes && source.pointer==destination.pointer))return 1;
            publication_copy_bytes(reinterpret_cast<uint8_t*>(destination.pointer),
                reinterpret_cast<const uint8_t*>(source.pointer),source.bytes,work);
        }
        if(old[i].role==32 || old[i].role==44 || old[i].role==56 || old[i].role==57 ||
           old[i].role==semantic_task_ground::kPublicationRole)
            if(publication_preserve_range(control,old,base.header.range_count,old[i],to[i],work))return 1;
        if(old[i].role>=18 && old[i].role<=25) {
            if(pending.transition_kind!=4) {
                to[i]=old[i];
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
                continue;
            }
            const auto* bindings=reinterpret_cast<const ModelUpdateBinding*>(pending.model_update_bindings);
            const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
            bool found=false;
            for(uint64_t item=0;item<pending.model_update_binding_count;++item) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(bindings[item].slots[0]==old[i].storage_slot) {
                    const uint64_t destination_slot=bindings[item].slots[1];
                    if(destination_slot>=control.storage_count)return 1;
                    to[i]=old[i];
                    to[i].storage_slot=destination_slot;
                    to[i].generation=storage[destination_slot].generation;
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange)+2*sizeof(uint64_t));
                    found=true;
                    break;
                }
            }
            if(!found)return 1;
        }
    }
    for(uint64_t i=0;i<pending.range_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const PublicationRange& source=from[i];if(source.role==55)continue;
        auto* destination=const_cast<PublicationRange*>(publication_find_range(to,next.header.range_count,source.role,source.index,work));
        if(!destination)return 1;
        const uint8_t* bytes=publication_range_bytes(control,source);
        if(source.role==1) {
            destination->length_bytes=next.header.prefix_extent*sizeof(SourceSlot);
            destination->logical_begin=0;destination->logical_end=next.header.prefix_extent;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            publication_copy_bytes(output+base.header.prefix_extent*sizeof(SourceSlot),bytes,source.length_bytes,work);
        } else if(source.role==39) {
            destination->length_bytes=source.length_bytes;
            destination->logical_begin=source.logical_begin;destination->logical_end=source.logical_end;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            publication_copy_bytes(output,bytes,source.length_bytes,work);
        } else {
            const auto* source_layout=publication_find_layout(control,from,pending.range_count,source.role,source.index,work);
            const auto* destination_layout=publication_find_layout(control,to,next.header.range_count,source.role,source.index,work);
            PublicationTensorView view{};uint64_t byte_count=0;
            if(publication_tensor_view(control,source,source_layout,&view,&byte_count,&pending_table,work))return 1;
            if(source.role>=51 && source.role<=54) {
                destination->length_bytes=byte_count ?
                    reinterpret_cast<const PublicationStorageEntry*>(control.storage)[destination->storage_slot].bytes-destination->offset_bytes : 0;
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
                if(byte_count)semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            }
            uint8_t* output=publication_range_bytes(control,*destination);if(!output)return 1;
            for(uint64_t cell=0;cell<byte_count/source_layout->element_bytes;++cell) {
                uint64_t rest=cell,offset=0,source_offset=0;
                for(int axis=int(source_layout->rank)-1;axis>=0;--axis) {
                    uint64_t coordinate=rest%view.dimensions[axis];rest/=view.dimensions[axis];
                    if(source.role>=51 && source.role<=54 && uint64_t(axis)==source_layout->logical_axis)
                        coordinate=pending_table.rows[coordinate].physical_row;
                    source_offset+=coordinate*source_layout->strides_bytes[axis];
                    if((source.role==4 || source.role==5) && pending.transition_kind!=4 &&
                       uint64_t(axis)==source_layout->logical_axis)coordinate+=base.header.prefix_extent;
                    offset+=coordinate*destination_layout->strides_bytes[axis];
                }
                publication_copy_bytes(output+offset,bytes+source_offset,source_layout->element_bytes,work);
            }
            destination->logical_begin=(source.role==4 || source.role==5) ? 0 : source.logical_begin;
            destination->logical_end=source.logical_end;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
        }
    }
    TensorLayoutTableView original{};
    if(!publication_tensor_table(control,old,base.header.range_count,&original,work))return 1;
    auto* destination=const_cast<PublicationRange*>(publication_find_range(to,next.header.range_count,55,0,work));
    const auto* original_range=publication_find_range(old,base.header.range_count,55,0,work);
    if(!destination || !original_range || destination->storage_slot==original_range->storage_slot)return 1;
    uint64_t layout_bytes=0,row_bytes=0,length=0;
    if(!semantic_graph::checked_mul(original.header->layout_count,sizeof(PublicationTensorLayout),&layout_bytes) ||
       !semantic_graph::checked_mul(pending_table.header->active_row_count,sizeof(ActiveRow),&row_bytes) ||
       !semantic_graph::checked_add(sizeof(TensorLayoutTableHeader),layout_bytes,&length) ||
       !semantic_graph::checked_add(length,row_bytes,&length))return 1;
    destination->length_bytes=length;destination->logical_begin=0;destination->logical_end=0;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
    auto* output=publication_range_bytes(control,*destination);if(!output)return 1;
    // The pending layout roster contains only outputs. Preserve the full cold
    // destination roster (and its capacity strides) in the inaccessible bank.
    auto* header=reinterpret_cast<TensorLayoutTableHeader*>(output);
    *header=*original.header;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(*header));
    header->active_computed=pending_table.header->active_computed;
    header->active_row_count=pending_table.header->active_row_count;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    publication_copy_bytes(output+sizeof(TensorLayoutTableHeader),reinterpret_cast<const uint8_t*>(original.layouts),layout_bytes,work);
    publication_copy_bytes(output+sizeof(TensorLayoutTableHeader)+layout_bytes,reinterpret_cast<const uint8_t*>(pending_table.rows),row_bytes,work);
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
__device__ uint64_t publication_logical_codebook_digest(const uint64_t* words,uint64_t count,uint64_t* digest,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!words || count<42 || count>UINT64_MAX/8 || words[8]%8 || words[8]>count*8 ||
       words[9]>count*8-words[8] || (words[8]+words[9]+7)/8!=count)return 1;
    PublicationCodebookBytes bytes{};bytes.words=words;
    const char domain[]="xlog.semantic.logical-codebooks.v1";
    for(uint32_t i=0;i<sizeof(domain);++i)bytes.prefix[bytes.prefix_bytes++]=semantic_graph::written_byte(domain[i],work);
    for(uint32_t i=0;i<32;++i)bytes.prefix[bytes.prefix_bytes++]=semantic_graph::written_byte(CATALOGUE_DIGEST[i],work);
    for(uint32_t i=0;i<8;++i)bytes.prefix[bytes.prefix_bytes++]=semantic_graph::written_byte(uint8_t(count>>(8*i)),work);
    if(count>(UINT64_MAX/8-bytes.prefix_bytes)/8)return 1;
    semantic_graph::sha256(bytes,bytes.prefix_bytes+count*8,digest,work);
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
        const Receipt* receipts,uint64_t count,uint64_t* digest,semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* codebook=publication_find_range(ranges,bank.header.range_count,48,0,work);
    if(!codebook || codebook->length_bytes%8)return 1;
    PublicationActionReceiptBytes bytes{};bytes.receipts=receipts;
    if(publication_logical_codebook_digest(reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*codebook)),
        codebook->length_bytes/8,bytes.logical_binding,work))return 1;
    semantic_graph::sha256(bytes,count*sizeof(Receipt),digest,work);
    return 0;
}
struct PublicationRecordBytes {
    const uint8_t* bytes;
    uint64_t prefix[16];
    uint64_t role;
    __device__ uint8_t operator[](uint64_t index) const {
        if(index<128)return reinterpret_cast<const uint8_t*>(prefix)[index];
        const uint64_t byte=index-128,word=byte/8;
        if(role==14 && ((word>=1 && word<=5) || word>=41))return 0;
        if(role==33 && ((word>=1 && word<=6) || word>=39))return 0;
        if(role==30 && byte>=sizeof(IntentQueueHeader)) {
            const uint64_t entry_word=(byte-sizeof(IntentQueueHeader))/8%43;
            if(entry_word>=30 && entry_word<=34)return 0;
        }
        return bytes[index-128];
    }
};
__device__ uint64_t publication_record_identity(const PublicationControl& control,
        const PublicationRange& range,uint64_t* digest,semantic_graph::NativeWorkTally* work=nullptr) {
    if(publication_tensor_role(range.role) || range.length_bytes>UINT64_MAX/8-128)return 1;
    PublicationRecordBytes input;
    input.bytes=publication_range_bytes(control,range);input.role=range.role;
    if(!input.bytes && range.length_bytes)return 1;
    for(uint32_t i=0;i<16;++i)input.prefix[i]=semantic_graph::written_word(0,work);
    input.prefix[0]=semantic_graph::written_word(0x786c6f6772616e31ULL,work);
    input.prefix[1]=semantic_graph::written_word(range.role,work);
    input.prefix[2]=semantic_graph::written_word(range.index,work);
    input.prefix[3]=semantic_graph::written_word(range.logical_begin,work);
    input.prefix[4]=semantic_graph::written_word(range.logical_end,work);
    semantic_graph::sha256(input,128+range.length_bytes,digest,work);
    return 0;
}
__device__ void publication_range_origin(const PublicationHeader& header,const PublicationRange& range,
        uint64_t* identity,semantic_graph::NativeWorkTally* work=nullptr) {
    uint64_t coordinates[8];
    semantic_graph::copy_identity(coordinates,header.logical_digest,work);
    coordinates[4]=range.role;coordinates[5]=range.index;
    coordinates[6]=range.logical_begin;coordinates[7]=range.logical_end;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,4*sizeof(uint64_t));
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(coordinates),sizeof(coordinates),identity,work);
}
__device__ void publication_fold(uint64_t* digest,uint64_t role,uint64_t index,const uint64_t* content,
        semantic_graph::NativeWorkTally* work) {
    uint64_t words[10];semantic_graph::copy_identity(words,digest,work);
    words[4]=semantic_graph::written_word(role,work);
    words[5]=semantic_graph::written_word(index,work);
    semantic_graph::copy_identity(words+6,content,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(words),sizeof(words),digest,work);
}
struct PublicationSemanticReceiptBytes {
    const semantic_graph::Receipt* receipts;
    __device__ uint8_t operator[](uint64_t index) const {
        return publication_receipt_runtime_word(index/8%42) ? 0 : reinterpret_cast<const uint8_t*>(receipts)[index];
    }
};

__device__ uint64_t publication_seal_action_batch(const PublicationControl& control,
        const PublicationBank& base,const PublicationBank& bank,State& state,
        semantic_graph::NativeWorkTally* work=nullptr,const ExecutionWork* completed_work=nullptr) {
    ActionBatchReceipt receipt{};receipt.abi=1;receipt.proposal=state.proposal;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(receipt)+16*sizeof(uint64_t)+32);
    receipt.stream_serial=state.stream_serial;receipt.family_id=state.family_id;
    receipt.action_law_generation=state.catalogue_generation;
    receipt.model_generation=state.model_generation;receipt.actor_eligible=state.actor_eligible;
    receipt.component_count=COMPONENT_COUNT;receipt.candidate_count=3;
    receipt.winner=state.task_evaluation.winner;
    receipt.total_return_bits=uint64_t(state.task_evaluation.return_value);
    receipt.base_word=base.header.publication_word;receipt.next_word=bank.header.publication_word;
    receipt.rng_base=state.proposal*COMPONENT_COUNT;receipt.rng_span=COMPONENT_COUNT;
    receipt.rng_successor=bank.header.proposal*COMPONENT_COUNT;
    auto* catalogue_digest_bytes=reinterpret_cast<uint8_t*>(receipt.catalogue_digest);
    for(uint32_t byte=0;byte<32;++byte)
        catalogue_digest_bytes[byte]=state.catalogue_digest[byte];
    if(publication_action_digest(control,bank,bank.receipts,COMPONENT_COUNT,
            receipt.component_receipts_digest,work))return 1;
    semantic_graph::sha256(PublicationSemanticReceiptBytes{state.semantic_receipts},
        sizeof(state.semantic_receipts),receipt.semantic_receipts_digest,work);
    const uint32_t candidate_receipts[3]={0,3,6};
    for(uint32_t candidate=0;candidate<3;++candidate)
        semantic_graph::copy_identity(receipt.candidate_roots[candidate],
            state.semantic_receipts[candidate_receipts[candidate]].words+16,work);
    semantic_graph::copy_identity(receipt.base_logical_digest,base.header.logical_digest,work);
    receipt.execution_work=completed_work ? *completed_work : state.execution_work;
    state.action_batch=receipt;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(ExecutionWork)+sizeof(receipt));
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipt),sizeof(receipt),
        state.action_batch_root,work);
    return 0;
}

__device__ bool publication_model_contract_layout(const ModelContractLayout& layout,uint64_t bytes) {
    const uint64_t begin[]={layout.schema_begin,layout.schema_digest_offset,layout.generation_offset,layout.identity_offset};
    const uint64_t length[]={layout.schema_bytes,32,8,32};
    for(uint32_t i=0;i<4;++i) {
        if(!length[i] || begin[i]>bytes || length[i]>bytes-begin[i])return false;
        for(uint32_t j=0;j<i;++j)
            if(begin[i]<begin[j]+length[j] && begin[j]<begin[i]+length[i])return false;
    }
    return layout.schema_bytes<=UINT64_MAX/8;
}

__device__ uint64_t publication_finalize_model_contract(const PublicationControl& control,
        const PublicationHeader& header,const PublicationRange& record,bool compare_only,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto& layout=reinterpret_cast<const PublicationContract*>(control.contract)->model_contract_layout;
    auto* bytes=publication_range_bytes(control,record);
    if(record.role!=44 || record.index || record.logical_begin || record.logical_end || !bytes ||
       !publication_model_contract_layout(layout,record.length_bytes))return 1;
    uint64_t schema_digest[4],identity[4];uint8_t generation[8];
    semantic_graph::sha256(bytes+layout.schema_begin,layout.schema_bytes,schema_digest,work);
    for(uint32_t i=0;i<8;++i)
        generation[i]=semantic_graph::written_byte(uint8_t(header.model_generation>>(8*i)),work);
    const char domain[]="xlog.semantic.model-identity.v2";
    uint8_t input[sizeof(domain)+32+8+32];
    publication_copy_bytes(input,reinterpret_cast<const uint8_t*>(domain),sizeof(domain),work);
    publication_copy_bytes(input+sizeof(domain),reinterpret_cast<const uint8_t*>(schema_digest),32,work);
    publication_copy_bytes(input+sizeof(domain)+32,generation,8,work);
    publication_copy_bytes(input+sizeof(domain)+40,reinterpret_cast<const uint8_t*>(header.original_instance),32,work);
    semantic_graph::sha256(input,sizeof(input),identity,work);
    const uint64_t offsets[]={layout.schema_digest_offset,layout.generation_offset,layout.identity_offset};
    const uint64_t lengths[]={32,8,32};
    const uint8_t* values[]={reinterpret_cast<const uint8_t*>(schema_digest),generation,reinterpret_cast<const uint8_t*>(identity)};
    for(uint32_t field=0;field<3;++field) {
        if(compare_only) {
            for(uint64_t i=0;i<lengths[field];++i) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(bytes[offsets[field]+i]!=values[field][i])return 1;
            }
        } else publication_copy_bytes(bytes+offsets[field],values[field],lengths[field],work);
    }
    return 0;
}

__device__ uint64_t publication_finalize_model_origin(const PublicationControl& control,PublicationBank& bank,
        bool restored=false,semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* model=publication_find_range(ranges,bank.header.range_count,44,0,work);
    return !model || publication_finalize_model_contract(control,bank.header,*model,restored,work);
}
enum class PublicationOriginKind : uint64_t {
    Proposal=1,Recompute=2,Drain=3,Update=4,DeliveryAcknowledgement=5,
    ReplayAcknowledgement=6,TrainingMaterialization=7,ModelContext=8,ActorRefresh=9,TrainingDemonstration=10,
    LearningPhaseTransition=11,LearningFold=12
};
struct PublicationOriginCoordinates {
    uint64_t parent[4],branch[4],operation,proposal,next_proposal,stream_serial,family_id;
    uint64_t model_generation,neural_generation,cache_generation,authority_generation,training_cursor,training_rng[4];
};
__device__ void publication_derive_origin(PublicationBank& bank,const PublicationHeader& parent,
        PublicationOriginKind operation,semantic_graph::NativeWorkTally* work=nullptr) {
    PublicationOriginCoordinates coordinates;
    semantic_graph::copy_identity(coordinates.parent,parent.logical_digest,work);
    semantic_graph::copy_identity(coordinates.branch,bank.header.original_instance,work);
    coordinates.operation=static_cast<uint64_t>(operation);
    coordinates.proposal=parent.proposal;coordinates.next_proposal=bank.header.proposal;
    coordinates.stream_serial=bank.header.stream_serial;coordinates.family_id=bank.header.family_id;
    coordinates.model_generation=bank.header.model_generation;coordinates.neural_generation=bank.header.neural_generation;
    coordinates.cache_generation=bank.header.cache_generation;coordinates.authority_generation=bank.header.authority_generation;
    coordinates.training_cursor=bank.header.training_cursor;
    semantic_graph::copy_identity(coordinates.training_rng,bank.header.training_rng,work);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,10*sizeof(uint64_t));
    const char domain[]="xlog.semantic.publication-origin.v1";
    uint8_t input[sizeof(domain)+sizeof(coordinates)+sizeof(uint64_t)];
    publication_copy_bytes(input,reinterpret_cast<const uint8_t*>(domain),sizeof(domain),work);
    publication_copy_bytes(input+sizeof(domain),reinterpret_cast<const uint8_t*>(&coordinates),sizeof(coordinates),work);
    uint64_t length=sizeof(domain)+sizeof(coordinates);
    if(operation==PublicationOriginKind::Proposal) {
        const uint64_t action=semantic_graph::written_word(bank.state.task_evaluation.winner,work);
        publication_copy_bytes(input+length,reinterpret_cast<const uint8_t*>(&action),sizeof(action),work);
        length+=sizeof(action);
    }
    semantic_graph::sha256(input,length,bank.header.logical_digest,work);
}


__device__ uint64_t publication_validate_selected_model(const PublicationControl& control,
        const PublicationBank& bank,semantic_graph::NativeWorkTally* work) {
    if(!control.contract || control.contract%alignof(PublicationContract) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract))return 1;
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& header=bank.header;
    const uint64_t directory=control.directories[header.publication_word&1];
    if(contract.abi!=1 || !contract.model_generation || header.model_generation<contract.model_generation ||
       header.model_generation>UINT32_MAX || header.range_count>contract.range_capacity ||
       !directory || directory%alignof(PublicationRange) ||
       header.range_count>(UINT64_MAX-directory)/sizeof(PublicationRange) ||
       !publication_bank_matches(control,bank,header.publication_word,work))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(directory);
    const auto* model=publication_find_range(ranges,header.range_count,44,0,work);
    return !model || publication_finalize_model_contract(control,header,*model,true,work);
}


__device__ bool publication_pointer_span(uint64_t pointer,uint64_t bytes,uint64_t alignment) {
    return pointer && !(pointer%alignment) && bytes && pointer<=UINT64_MAX-bytes;
}
__device__ bool publication_spans_overlap(uint64_t a,uint64_t a_bytes,uint64_t b,uint64_t b_bytes) {
    return a<b+b_bytes && b<a+a_bytes;
}
extern "C" __global__ void semantic_numerical_block_snapshot(uint64_t source_ptr,
        uint64_t destination_ptr,uint64_t bytes,uint64_t native_work_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(native_work_ptr);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    if(!publication_pointer_span(source_ptr,bytes,alignof(double)) ||
       !publication_pointer_span(destination_ptr,bytes,alignof(double)) ||
       publication_spans_overlap(source_ptr,bytes,destination_ptr,bytes)) {
        semantic_content_integrity_trap();return;
    }
    publication_copy_bytes(reinterpret_cast<uint8_t*>(destination_ptr),
        reinterpret_cast<const uint8_t*>(source_ptr),bytes,work);
}
__device__ bool publication_step_output_span(const PublicationControl& control,uint64_t lease,
        uint64_t bindings,uint64_t binding_bytes,uint64_t pointer,uint64_t bytes,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(!publication_pointer_span(pointer,bytes) ||
       publication_spans_overlap(pointer,bytes,reinterpret_cast<uint64_t>(&control),sizeof(control)) ||
       publication_spans_overlap(pointer,bytes,lease,sizeof(PublicationLease)) ||
       publication_spans_overlap(pointer,bytes,bindings,binding_bytes) ||
       publication_spans_overlap(pointer,bytes,control.contract,sizeof(contract)) ||
       publication_spans_overlap(pointer,bytes,contract.role_counts,semantic_task_ground::kPublicationRole*sizeof(PublicationRoleCount)) ||
       publication_spans_overlap(pointer,bytes,control.storage,control.storage_count*sizeof(PublicationStorageEntry)))return false;
    for(uint64_t bank=0;bank<2;++bank) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(publication_spans_overlap(pointer,bytes,control.banks[bank],sizeof(PublicationBank)) ||
           publication_spans_overlap(pointer,bytes,control.directories[bank],contract.range_capacity*sizeof(PublicationRange)))return false;
    }
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    for(uint64_t i=0;i<control.storage_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(publication_spans_overlap(pointer,bytes,storage[i].pointer,storage[i].bytes))return false;
    }
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


// Cold catalogue replacement is not publication. Live entries keep their exact
// address and generation; expired, unselected slots never retain reused addresses.
extern "C" __global__ void semantic_publication_replace_model_storage(uint64_t control_ptr,
        uint64_t original_ptr,uint64_t original_count,uint64_t expanded_ptr,uint64_t expanded_count) {
    if(blockIdx.x || threadIdx.x)return;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       original_count>UINT64_MAX/sizeof(PublicationStorageEntry) ||
       expanded_count>UINT64_MAX/sizeof(PublicationStorageEntry) || expanded_count<original_count ||
       !publication_pointer_span(original_ptr,original_count*sizeof(PublicationStorageEntry),alignof(PublicationStorageEntry)) ||
       !publication_pointer_span(expanded_ptr,expanded_count*sizeof(PublicationStorageEntry),alignof(PublicationStorageEntry))) {
        semantic_content_integrity_trap();return;
    }
    auto& control=*reinterpret_cast<PublicationControl*>(control_ptr);
    if(control.abi!=1 || control.storage!=original_ptr || control.storage_count!=original_count ||
       control.reader_gate || control.reader_counts[0] || control.reader_counts[1]) {
        semantic_content_integrity_trap();return;
    }
    const auto* original=reinterpret_cast<const PublicationStorageEntry*>(original_ptr);
    const auto* expanded=reinterpret_cast<const PublicationStorageEntry*>(expanded_ptr);
    const auto& selected=*reinterpret_cast<const PublicationBank*>(control.banks[control.word&1]);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(selected.header.abi!=1 || selected.header.publication_word!=control.word ||
       selected.header.range_count>contract.range_capacity) { semantic_content_integrity_trap();return; }
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[control.word&1]);
    for(uint64_t i=0;i<original_count;++i) {
        if(original[i].pointer!=expanded[i].pointer || original[i].bytes!=expanded[i].bytes ||
           original[i].generation!=expanded[i].generation) {
            if(expanded[i].pointer || expanded[i].bytes || original[i].generation!=expanded[i].generation) {
                semantic_content_integrity_trap();return;
            }
            for(uint64_t j=0;j<selected.header.range_count;++j)
                if(ranges[j].storage_slot==i) { semantic_content_integrity_trap();return; }
        }
    }
    for(uint64_t i=original_count;i<expanded_count;++i) {
        if(!expanded[i].generation || (expanded[i].bytes &&
           (!expanded[i].pointer || expanded[i].pointer>UINT64_MAX-expanded[i].bytes))) {
            semantic_content_integrity_trap();return;
        }
        for(uint64_t j=0;j<i;++j)
            if(publication_spans_overlap(expanded[i].pointer,expanded[i].bytes,
                    expanded[j].pointer,expanded[j].bytes)) { semantic_content_integrity_trap();return; }
    }
    // The next transition validates both directories before applying its
    // continuation. An unleased inactive bank cannot keep a numerical range
    // pointing at a retired slot: ordinary continuations share the selected
    // generation, while a model update installs its own prepared destination.
    auto* inactive=reinterpret_cast<PublicationRange*>(control.directories[(control.word&1)^1]);
    for(uint64_t i=0;i<selected.header.range_count;++i) {
        if(ranges[i].role<18 || ranges[i].role>25)continue;
        if(!publication_find_range(inactive,selected.header.range_count,ranges[i].role,ranges[i].index)) {
            semantic_content_integrity_trap();return;
        }
    }
    for(uint64_t i=0;i<selected.header.range_count;++i) {
        if(ranges[i].role<18 || ranges[i].role>25)continue;
        auto* destination=const_cast<PublicationRange*>(publication_find_range(
            inactive,selected.header.range_count,ranges[i].role,ranges[i].index));
        *destination=ranges[i];
    }
    control.storage=expanded_ptr;control.storage_count=expanded_count;
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
        uint64_t result_ptr,uint64_t parent_header_ptr,uint64_t training_origin_ptr,
        uint64_t score_ptr,uint64_t score_state0,uint64_t score_state1) {
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
    if(contract.abi!=1 || contract.role_count!=semantic_task_ground::kPublicationRole ||
       !publication_pointer_span(contract.role_counts,semantic_task_ground::kPublicationRole*sizeof(PublicationRoleCount),alignof(PublicationRoleCount)) ||
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
    if(score_ptr) {
        const uint64_t state_ptr=lease.bank ? score_state1 : score_state0;
        if(training_origin_ptr || !publication_pointer_span(score_ptr,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice)) ||
           !publication_pointer_span(state_ptr,sizeof(State),alignof(State)) ||
           !publication_pointer_span(parent_header_ptr,sizeof(PublicationHeader),alignof(PublicationHeader))) {
            semantic_content_integrity_trap();return;
        }
        const auto& proof=*reinterpret_cast<const PreparedActorRefreshDevice*>(score_ptr);
        const auto& state=*reinterpret_cast<const State*>(state_ptr);
        const auto& parent=*reinterpret_cast<const PublicationHeader*>(parent_header_ptr);
        if(proof.abi!=1 || proof.status || proof.role!=1 || proof.child_control!=control_ptr ||
           lease.transition_kind!=1 || word!=lease.word || control.refusal || state.blocks ||
           state.next_proposal!=parent.proposal || state.proposal!=parent.proposal ||
           state.model_generation!=parent.model_generation || state.stream_serial!=parent.stream_serial ||
           state.family_id!=parent.family_id || state.actor_eligible || state.action_batch.abi ||
           proof.acquired_generation!=parent.model_generation ||
           !publication_bank_matches(control,bank,word) || !publication_header_equal(parent,bank.header) ||
           !publication_identity_equal(proof.model_geometry_digest,parent.model_geometry_digest)) {
            semantic_content_integrity_trap();return;
        }
        *reinterpret_cast<PublicationStepResult*>(result_ptr)={2,word,state.status,bank.header,0,{}};
        return;
    }
    if(score_state0 || score_state1) { semantic_content_integrity_trap();return; }
    if(!publication_bank_matches(control,bank,word) || (word!=lease.word &&
       ((lease.word>>1)==(UINT64_MAX>>1) || word!=(((lease.word>>1)+1)<<1 | ((lease.word^1)&1)) ||
        bank.header.base_word!=lease.word))) { semantic_content_integrity_trap();return; }
    const uint64_t advanced=uint64_t(word!=lease.word && !control.refusal);
    PublicationStepResult retained={1,word,control.refusal,bank.header,advanced,{}};
    if(advanced) {
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[word&1]);
        const auto* range=publication_find_range(ranges,bank.header.range_count,33);
        const auto* bytes=range ? publication_range_bytes(control,*range) : nullptr;
        if(!range || range->length_bytes!=sizeof(AttemptReceipt) || !bytes ||
           reinterpret_cast<uintptr_t>(bytes)%alignof(AttemptReceipt)) {
            semantic_content_integrity_trap();return;
        }
        retained.attempt=*reinterpret_cast<const AttemptReceipt*>(bytes);
    }
    *reinterpret_cast<PublicationStepResult*>(result_ptr)=retained;
    if(training_origin_ptr) {
        auto& origin=*reinterpret_cast<SemanticTrainingViewOriginRecord*>(training_origin_ptr);
        origin={};
        if(advanced) {
            const auto& parent=*reinterpret_cast<const PublicationHeader*>(parent_header_ptr);
            origin.present=1;
            origin.transition=lease.transition_kind;
            origin.predecessor_word=parent.publication_word;
            origin.successor_word=bank.header.publication_word;
            origin.model_generation=parent.model_generation;
            origin.stream_serial=parent.stream_serial;
            origin.family_id=parent.family_id;
            origin.proposal=parent.proposal;
            for(uint32_t i=0;i<4;++i) {
                origin.lineage_instance[i]=parent.original_instance[i];
                origin.predecessor_instance[i]=parent.instance[i];
                origin.predecessor_logical[i]=parent.logical_digest[i];
                origin.successor_instance[i]=bank.header.instance[i];
                origin.successor_logical[i]=bank.header.logical_digest[i];
                origin.model_geometry_digest[i]=parent.model_geometry_digest[i];
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


// Capture binds both resident input banks once. Every execution resolves the
// actual acquired parent and validates all originals before mutating any private
// output. Prefix, attention caches and model tensors remain direct device aliases;
// Source, feedback records, header and range coordinates remain private snapshots.
extern "C" __global__ void semantic_publication_step_inputs(uint64_t control_ptr,uint64_t lease_ptr,
        uint64_t bindings0_ptr,uint64_t bindings1_ptr,uint64_t binding_count,uint64_t header_output_ptr,
        uint64_t source_output_ptr,uint64_t range_output_ptr,uint64_t cold_work_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(cold_work_ptr);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    uint64_t binding_bytes=0,range_bytes=0;
    if(!publication_pointer_span(control_ptr,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(lease_ptr,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationStepInput),&binding_bytes) ||
       !semantic_graph::checked_mul(binding_count,sizeof(PublicationRange),&range_bytes) ||
       !publication_pointer_span(bindings0_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(bindings1_ptr,binding_bytes,alignof(PublicationStepInput)) ||
       !publication_pointer_span(header_output_ptr,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
       !publication_pointer_span(source_output_ptr,32*sizeof(SourceSlot),alignof(SourceSlot)) ||
       !publication_pointer_span(range_output_ptr,range_bytes,alignof(PublicationRange))) {
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
    if(contract.abi!=1 || contract.role_count!=semantic_task_ground::kPublicationRole ||
       !publication_pointer_span(contract.role_counts,semantic_task_ground::kPublicationRole*sizeof(PublicationRoleCount),alignof(PublicationRoleCount)) ||
       !semantic_graph::checked_mul(contract.range_capacity,sizeof(PublicationRange),&directory_bytes)) {
        semantic_content_integrity_trap();return;
    }
    for(uint64_t bank=0;bank<2;++bank)
        if(!publication_pointer_span(control.banks[bank],sizeof(PublicationBank),alignof(PublicationBank)) ||
           !publication_pointer_span(control.directories[bank],directory_bytes,alignof(PublicationRange))) {
            semantic_content_integrity_trap();return;
        }
    const auto* bank=publication_acquired_bank(control,lease);
    if(!bank || bank->header.range_count>contract.range_capacity || publication_validate_storage(control,work)) {
        semantic_content_integrity_trap();return;
    }
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[lease.bank]);
    const uint64_t count=bank->header.range_count;
    if(publication_validate_selected_model(control,*bank,work)) { semantic_content_integrity_trap();return; }
    const auto* table_range=publication_find_range(ranges,count,55,0,work);
    if(!table_range || publication_validate_directory(control,contract,ranges,count,work)) {
        semantic_content_integrity_trap();return;
    }
    const auto* roles=reinterpret_cast<const PublicationRoleCount*>(contract.role_counts);
    uint64_t expected=3;
    for(uint64_t role=4;role<=25;++role) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(role!=14 && !semantic_graph::checked_add(expected,roles[role-1].count,&expected)) {
            semantic_content_integrity_trap();return;
        }
    }
    if(roles[43].count!=1 || binding_count!=expected) { semantic_content_integrity_trap();return; }
    const uint64_t output_pointers[]={header_output_ptr,source_output_ptr,range_output_ptr};
    const uint64_t output_bytes[]={sizeof(PublicationHeader),32*sizeof(SourceSlot),range_bytes};
    for(uint64_t i=0;i<3;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(!publication_step_output_span(control,lease_ptr,bindings0_ptr,binding_bytes,output_pointers[i],output_bytes[i],work) ||
           !publication_step_output_span(control,lease_ptr,bindings1_ptr,binding_bytes,output_pointers[i],output_bytes[i],work)) {
            semantic_content_integrity_trap();return;
        }
        for(uint64_t j=0;j<i;++j) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(publication_spans_overlap(output_pointers[i],output_bytes[i],output_pointers[j],output_bytes[j])) {
                semantic_content_integrity_trap();return;
            }
        }
    }
    const auto* bindings=reinterpret_cast<const PublicationStepInput*>(bindings_ptr);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t cursor=0;
    for(uint64_t role=1;role<=44;++role) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(role==2 || role==14 || (role>25 && role!=44))continue;
        const bool model=role>=18 && role<=25;
        for(uint64_t index=0;index<roles[role-1].count;++index) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const auto& binding=bindings[cursor++];
            const auto* range=publication_find_range(ranges,count,role,index,work);
            const auto* layout=publication_find_layout(control,ranges,count,role,index,work);
            if(!range || binding.role!=role || binding.index!=index ||
               ((!model || binding.capacity_bytes) && !publication_pointer_span(binding.destination,binding.capacity_bytes))) {
                semantic_content_integrity_trap();return;
            }
            const auto* original=publication_range_bytes(control,*range);
            const bool prefix_cache=role==4 || role==5;
            const bool alias=role==1 || prefix_cache || model;
            if(model) {
                uint64_t destination=0;
                if(binding.backing!=storage[range->storage_slot].pointer ||
                   binding.backing_bytes!=storage[range->storage_slot].bytes ||
                   !semantic_graph::checked_add(binding.backing,range->offset_bytes,&destination) ||
                   binding.destination!=destination ||
                   (binding.backing_bytes && !publication_pointer_span(binding.backing,binding.backing_bytes))) {
                    semantic_content_integrity_trap();return;
                }
                for(uint64_t i=0;i+1<cursor;++i) {
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    const auto* previous=publication_find_range(ranges,count,bindings[i].role,bindings[i].index,work);
                    if(previous->storage_slot==range->storage_slot) {
                        if(bindings[i].backing!=binding.backing || bindings[i].backing_bytes!=binding.backing_bytes) {
                            semantic_content_integrity_trap();return;
                        }
                        break;
                    }
                }
            } else if(binding.backing!=binding.destination || binding.backing_bytes!=binding.capacity_bytes) {
                semantic_content_integrity_trap();return;
            }
            if(role==3 || (role>=15 && role<=17) || role==44) {
                const auto* words=reinterpret_cast<const uint64_t*>(&binding.layout);
                for(uint64_t word=0;word<sizeof(binding.layout)/8;++word) {
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    if(words[word]) { semantic_content_integrity_trap();return; }
                }
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
                for(uint64_t word=0;word<sizeof(original_layout)/8;++word) {
                    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    if(actual[word]!=sealed[word]) { semantic_content_integrity_trap();return; }
                }
                uint64_t capacity=0;
                if(model)capacity=range->length_bytes;
                if((!model && !publication_step_layout_capacity(original_layout,&capacity)) || binding.capacity_bytes!=capacity ||
                   binding.destination%original_layout.element_bytes ||
                   capacity>storage[range->storage_slot].bytes-range->offset_bytes ||
                   (!alias && (range->length_bytes!=capacity || range->logical_begin || range->logical_end))) {
                    semantic_content_integrity_trap();return;
                }
                const uint64_t alias_logical_end=model ? 0 : bank->header.prefix_extent;
                if(alias && (binding.destination!=reinterpret_cast<uint64_t>(original) || range->logical_begin ||
                   range->logical_end!=alias_logical_end ||
                   (role==1 && range->length_bytes!=bank->header.prefix_extent*sizeof(SourceSlot)) ||
                   (prefix_cache && (range->length_bytes!=capacity || original_layout.dimensions[2]!=contract.prefix_capacity)))) {
                    semantic_content_integrity_trap();return;
                }
            }
            if(alias)continue;
            if(binding.backing_bytes &&
               (!publication_step_output_span(control,lease_ptr,bindings0_ptr,binding_bytes,binding.backing,binding.backing_bytes,work) ||
                !publication_step_output_span(control,lease_ptr,bindings1_ptr,binding_bytes,binding.backing,binding.backing_bytes,work))) {
                semantic_content_integrity_trap();return;
            }
            for(uint64_t i=0;i<3;++i) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(publication_spans_overlap(binding.backing,binding.backing_bytes,output_pointers[i],output_bytes[i])) {
                    semantic_content_integrity_trap();return;
                }
            }
            for(uint64_t i=0;i+1<cursor;++i) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const auto* previous=publication_find_range(ranges,count,bindings[i].role,bindings[i].index,work);
                const bool same_model=model && bindings[i].role>=18 && previous->storage_slot==range->storage_slot;
                if(!same_model && publication_spans_overlap(binding.backing,binding.backing_bytes,bindings[i].backing,bindings[i].backing_bytes)) {
                    semantic_content_integrity_trap();return;
                }
            }
        }
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationHeader));
    *reinterpret_cast<PublicationHeader*>(header_output_ptr)=bank->header;
    publication_copy_bytes(reinterpret_cast<uint8_t*>(source_output_ptr),reinterpret_cast<const uint8_t*>(bank->source),sizeof(bank->source),work);
    auto* output_ranges=reinterpret_cast<PublicationRange*>(range_output_ptr);
    for(uint64_t i=0;i<binding_count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& binding=bindings[i];const auto* range=publication_find_range(ranges,count,binding.role,binding.index,work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
        output_ranges[i]=*range;
        if(binding.role==3 || (binding.role>=6 && (binding.role<18 || binding.role>25)))
            publication_copy_bytes(reinterpret_cast<uint8_t*>(binding.destination),publication_range_bytes(control,*range),range->length_bytes,work);
    }
}


__device__ void publication_intent_identity(const IntentEntry& entry,uint64_t* identity,
        semantic_graph::NativeWorkTally* work=nullptr) {
    uint64_t words[25];words[0]=0x786c6f67696e7431ULL;
    const uint64_t* content=entry.checkpoint_lineage;
    for(uint32_t i=0;i<24;++i)words[i+1]=content[i];
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(words));
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(words),sizeof(words),identity,work);
}
__device__ bool publication_nonterminal_intent_binding_valid(const PublicationContract& contract) {
    const bool plan=contract.nonterminal_intent_plan_digest[0] || contract.nonterminal_intent_plan_digest[1] ||
        contract.nonterminal_intent_plan_digest[2] || contract.nonterminal_intent_plan_digest[3];
    if(!contract.nonterminal_intent_mode)return !plan && !contract.nonterminal_intent_prefix_begin &&
        !contract.nonterminal_intent_prefix_end;
    return contract.nonterminal_intent_mode==1 && plan &&
        contract.nonterminal_intent_prefix_begin<contract.nonterminal_intent_prefix_end &&
        contract.nonterminal_intent_prefix_end<=contract.prefix_capacity &&
        contract.nonterminal_intent_prefix_end-contract.nonterminal_intent_prefix_begin<=(UINT64_MAX-48)/8;
}
__device__ uint64_t publication_validate_intents(const PublicationControl& control,
        const PublicationRange* ranges,uint64_t count,uint64_t terminal_position=UINT64_MAX,
        semantic_graph::NativeWorkTally* work=nullptr) {
    if(!publication_nonterminal_intent_binding_valid(
            *reinterpret_cast<const PublicationContract*>(control.contract)))return 1;
    const auto* entry_range=publication_find_range(ranges,count,30,0,work);
    const auto* payload_range=publication_find_range(ranges,count,31,0,work);
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
    semantic_graph::sha256(payload,queue->effect_length_bytes,effect,work);
    for(uint64_t i=0;i<queue->count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const IntentEntry& entry=entries[i];uint64_t digest[4];
        if(entry.payload_offset!=cursor || entry.payload_len>queue->payload_used_bytes-cursor ||
           !publication_identity_equal(entry.effect_digest,effect,work) || !publication_identity_equal(entry.previous_chain,chain,work))return 1;
        semantic_graph::sha256(payload+cursor,entry.payload_len,digest,work);
        if(!publication_identity_equal(digest,entry.payload_digest,work))return 1;
        publication_intent_identity(entry,digest,work);
        if(!publication_identity_equal(digest,entry.stable_identity,work))return 1;
        publication_fold(chain,30,i,digest,work);
        if(!publication_identity_equal(chain,entry.chain,work))return 1;
        cursor+=entry.payload_len;
    }
    if(cursor!=queue->payload_used_bytes || !publication_identity_equal(queue->chain_head,chain,work))return 1;
    if(terminal_position!=UINT64_MAX) {
        const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
        uint64_t tokens=terminal_position,header_bytes=16;
        if(contract.nonterminal_intent_mode==1) {
            if(terminal_position<contract.nonterminal_intent_prefix_begin)return 1;
            const uint64_t end=terminal_position<contract.nonterminal_intent_prefix_end ? terminal_position : contract.nonterminal_intent_prefix_end;
            tokens=end-contract.nonterminal_intent_prefix_begin;header_bytes=48;
        }
        if(queue->count==queue->capacity || tokens>(UINT64_MAX-header_bytes)/8)return 5;
        const uint64_t needed=header_bytes+tokens*8;
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
        const semantic_graph::DecodedSupport* support=nullptr,ExecutionWork* work=nullptr,
        semantic_graph::NativeWorkTally* native_work=nullptr) {
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
    // An operation has one meter owner. ExecutionWork already merges its native
    // tally; the explicit original cold tally is used only without that owner.
    semantic_graph::execute(reinterpret_cast<uint64_t*>(d.arena[0]),launch,work ? &actual : native_work);
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

struct NativePrefixWork {
    semantic_graph::NativeWorkTally* tally;
    __device__ void operator()(semantic_rule_action::PrefixWorkEvent event) const {
        semantic_graph::charge_native(tally,event==semantic_rule_action::PrefixWorkEvent::Category
            ? semantic_graph::NativeWorkEvent::Category : semantic_graph::NativeWorkEvent::TableSlot,1);
    }
};
__device__ bool supported(const uint64_t* books,const Component* components,const uint8_t* support,uint32_t start,uint32_t field,uint32_t category,
        semantic_graph::NativeWorkTally* work=nullptr) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
    const Component c=components[start+field];
    return category<c.cardinality &&
        semantic_rule_action::context_category(books,field,category,NativePrefixWork{work}) &&
        support[c.offset+category];
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
    if(!target || !supported(books,components,support,start,2,target,work))return false;
    if((books+books[0]+10*target)[1]==UINT64_MAX)return false;
    for(uint32_t position=0;position<4;++position) {
        bool found=false;
        for(uint32_t operand=0;operand<books[3];++operand)
            if(supported(books,components,support,start,3+position,operand,work) && operand_matches(books,target,position,operand,work)){found=true;break;}
        if(!found)return false;
    }
    return true;
}
__device__ bool action_completes(const uint64_t* books,const Component* components,const uint8_t* support,
                               const uint32_t* viable,uint32_t start,uint32_t opcode,
                               const TaskProgramBank& program_bank,
                               semantic_graph::NativeWorkTally* work=nullptr) {
    if(!supported(books,components,support,start,0,opcode==0 ? ACTION_NO_EDIT : ACTION_APPLY,work))return false;
    if(opcode==0) {
        for(uint32_t f=1;f<18;++f)if(!supported(books,components,support,start,f,0,work))return false;
        return true;
    }
    if(opcode==2) {
        if(!program_bank.editable ||
           uint64_t(program_bank.program.rule_count)+program_bank.selected_rule_count>=PROGRAM_RULE_CAPACITY ||
           !supported(books,components,support,start,1,2,work))return false;
        uint32_t empty_choices[18]={};
        empty_choices[1]=2;
        return semantic_rule_action::prefix_completes(books,components,support,start,
            empty_choices,1,2,program_bank.program,
            NativePrefixWork{work});
    }
    if(opcode!=1 || program_bank.editable)return false;
    if(!supported(books,components,support,start,1,1,work))return false;
    for(uint32_t f=1;f<18;++f)if((INSERT_SUPPORT_NULL_MASK>>f)&1)
        if(!supported(books,components,support,start,f,0,work))return false;
    for(uint32_t f=11;f<=17;f+=6) {
        bool found=false;
        for(uint32_t i=1;i<components[start+f].cardinality;++i)found |= supported(books,components,support,start,f,i,work);
        if(!found)return false;
    }
    for(uint32_t i=1;i<books[1];++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
        if(viable[i])return true;
    }
    return false;
}
__device__ bool legal_category(const Component c,uint32_t category,const uint64_t* books,
        const Component* components,const uint8_t* support,uint32_t start,
        const TaskProgramBank& program_bank,const uint32_t* viable,const uint32_t* choices,
        bool no_edit,bool apply_support,bool apply_rule,
        semantic_graph::NativeWorkTally* work=nullptr) {
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Category,1);
    if(c.kind==COMPONENT_KIND_TEXT)return true;
    if(!semantic_rule_action::context_category(books,c.field,category,NativePrefixWork{work}))return false;
    if(c.field==0)return category==0 ? no_edit : category==1 && (apply_support || apply_rule);
    if(choices[0]==ACTION_NO_EDIT)return category==0;
    if(c.field==1)return category==1 ? apply_support : category==2 && apply_rule;
    if(choices[1]==2)return program_bank.editable &&
        semantic_rule_action::prefix_completes(books,components,support,start,choices,c.field,
            category,program_bank.program,
            NativePrefixWork{work});
    if(program_bank.editable)return false;
    if((INSERT_SUPPORT_NULL_MASK>>c.field)&1)return category==0;
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
__device__ bool task_allows(const semantic_task_ground::Task& task,
        semantic_graph::DecodedStatement& statement,
        const semantic_graph::DecodedSupport& support) {
    uint64_t identity[4];semantic_graph::decode_identity(statement.identity_words,identity);
    uint32_t query_index=task.queries;
    for(uint32_t query=0;query<task.queries && query_index==task.queries;++query) {
        bool equal=true;
        for(uint32_t i=0;i<4;++i)equal &= identity[i]==task.query(query)[i];
        if(equal)query_index=query;
    }
    if(query_index==task.queries)return false;
    if(task.coding)for(uint32_t binding=0;binding<task.bindings;++binding)
        if(task.binding(binding)[1]==task.query(query_index)[4])return false;
    uint64_t digest[4];semantic_graph::support_digest(identity,support,digest);
    for(uint64_t event=0;event<task.support_count;++event) {
        bool equal=task.supports[5*event]==support.record;
        for(uint32_t i=0;i<4;++i)equal &= digest[i]==task.supports[5*event+1+i];
        if(equal) {
            statement.record=uint32_t(task.query(query_index)[4]);
            for(uint32_t i=0;i<10;++i)statement.reconstruction[i]=0;
            return true;
        }
    }
    return false;
}

__device__ void task_cost(State* state,uint32_t slot,const semantic_task_ground::Task& task) {
    auto& facts=state->task_evaluation.facts[slot];
    if(slot) {
        const auto& work=state->work[slot-1];
        facts.c=work.edit_commands+work.added_supports+work.defined_truth_changes;
    }
    facts.v=int64_t(task.scoring[0]*facts.g+task.scoring[1]*facts.p)-int64_t(task.scoring[2]*facts.c);
}

__device__ bool selected_task_cost_matches(const State* state,uint64_t cap) {
    if(!state || state->task_evaluation.winner>=3)return false;
    const uint64_t winner=state->task_evaluation.winner;
    uint64_t measured=0;
    if(winner) {
        const auto& work=state->work[winner-1];
        if(work.edit_commands>2 || work.added_supports>cap ||
           work.defined_truth_changes>3)return false;
        measured=work.edit_commands+work.added_supports+work.defined_truth_changes;
    }
    return measured<=cap && state->task_evaluation.facts[winner].c==measured;
}

__device__ bool task_queries(const Descriptor& d,const uint64_t* task,uint32_t slot,
        const semantic_graph::ResidentHandle* root,
        const semantic_graph::Receipt* candidate,State* state) {
    auto& evaluation=state->task_evaluation;
    auto& facts=evaluation.facts[slot];
    semantic_task_ground::Task view{};semantic_task_ground::Plane plane{};
    if(!semantic_task_ground::decode(task,d.task_words,&view) || view.coding ||
       !semantic_task_ground::decode_plane(d.task_ground,d.task_ground_bytes,view,&plane))return false;
    bool valid=true,hard=true;
    for(uint32_t query=0;query<view.queries;++query) {
        semantic_graph::DecodedStatement statement{};
        const uint64_t* row=view.query(query);
        statement.record=uint32_t(row[4]);
        for(uint32_t i=0;i<8;++i)
            statement.identity_words[i]=uint32_t(row[i/2]>>(32*(i%2)));
        auto& record=plane.query(slot,query);
        auto output=reinterpret_cast<semantic_graph::Receipt*>(record.receipt);
        ++evaluation.query_count;
        // All three queries use the original view receipt, never a preceding truth result.
        semantic_call(d,slot ? semantic_graph::kResidentForkTruthAdmission :
            semantic_graph::kResidentRootTruthAdmission,root,candidate,output,&statement,nullptr,&state->execution_work);
        valid &= output->words[0]==semantic_graph::kOk && output->words[2]<=3;
        hard &= output->words[2]<=3 && (row[6] & (uint64_t(1)<<output->words[2]))!=0;
        record.truth=output->words[2];
        record.attained=output->words[0]==semantic_graph::kOk && record.truth==row[5];
        facts.g+=record.attained;
    }
    facts.p=facts.g==view.queries;
    facts.eligible=valid && hard;
    task_cost(state,slot,view);
    return valid;
}

enum class TaskProgramQueryOutcome : uint32_t {
    Ok,
    FactCapacity,
    FuelExhausted,
    Scope,
    Failure,
};

// Query execution and feedback lineage use the same original program
// materializer and the same finite native support scratch.
__device__ semantic_program::Result task_program_closure(const Descriptor& descriptor,
        const TaskProgramBank& bank,const semantic_task_ground::Plane& plane,uint32_t slot,
        const semantic_program::Rule* inserted,uint32_t inserted_count,
        semantic_program::DualSupport* support,semantic_program::Fact** output,
        ExecutionWork* work=nullptr,semantic_graph::NativeWorkTally* native_work=nullptr) {
    const uint64_t lineage_words=(uint64_t(bank.task.bindings)+plane.inherited_capacity+63)/64;
    if(slot>2 || !descriptor.task_support_truths ||
       descriptor.task_support_lineage_words!=3*PROGRAM_FACT_CAPACITY*2*lineage_words ||
       (lineage_words && !descriptor.task_support_lineage))
        return {semantic_program::Status::InvalidProgram,0,0,0};
    *support={reinterpret_cast<uint8_t*>(descriptor.task_support_truths)+uint64_t(slot)*PROGRAM_FACT_CAPACITY,
        lineage_words ? reinterpret_cast<uint64_t*>(descriptor.task_support_lineage)+
            uint64_t(slot)*PROGRAM_FACT_CAPACITY*2*lineage_words : nullptr,uint32_t(lineage_words)};
    auto* scratch=reinterpret_cast<uint8_t*>(descriptor.scratch)+PROGRAM_SCRATCH_OFFSET;
    auto* input=reinterpret_cast<semantic_program::Fact*>(scratch);
    auto* rules=reinterpret_cast<semantic_program::Rule*>(input+PROGRAM_FACT_CAPACITY);
    *output=reinterpret_cast<semantic_program::Fact*>(rules+PROGRAM_RULE_CAPACITY);
    const auto closure=semantic_program::materialize_bank(bank.program,inserted,inserted_count,
        input,PROGRAM_FACT_CAPACITY,rules,PROGRAM_RULE_CAPACITY,*output,PROGRAM_FACT_CAPACITY,
        PROGRAM_FUEL,plane.selected_rules,uint32_t(plane.header[6]),support,
        semantic_task_ground::Observations{bank.task,plane});
    if(work) {
        semantic_graph::NativeWorkTally native{};
        semantic_graph::charge_native(&native,semantic_graph::NativeWorkEvent::TableSlot,closure.charged_pairs);
        execution_work_merge_native(*work,native);
    } else semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,closure.charged_pairs);
    return closure;
}

__device__ TaskProgramQueryOutcome task_program_queries(const Descriptor& descriptor,const uint64_t* task,
        const TaskProgramBank& bank,uint32_t slot,const semantic_program::Rule* inserted,
        uint32_t inserted_count,uint32_t baseline_derived,State* state,
        uint32_t* derived_count) {
    if(!bank.editable || inserted_count>2 || !derived_count)return TaskProgramQueryOutcome::Failure;
    const auto& view=bank.task;
    semantic_task_ground::Plane plane{};
    if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,view,&plane))
        return TaskProgramQueryOutcome::Failure;
    semantic_program::DualSupport support{};
    semantic_program::Fact* output=nullptr;
    const auto closure=task_program_closure(descriptor,bank,plane,slot,inserted,inserted_count,
        &support,&output,&state->execution_work);
    semantic_graph::NativeWorkTally work{};
    if(state->execution_work.overflow)return TaskProgramQueryOutcome::Failure;
    if(closure.status!=semantic_program::Status::Ok) {
        if(slot && closure.status==semantic_program::Status::ProtectedObservation)
            return TaskProgramQueryOutcome::Scope;
        if(slot && (closure.status==semantic_program::Status::FactCapacity || closure.status==semantic_program::Status::RuleCapacity))
            return TaskProgramQueryOutcome::FactCapacity;
        if(slot && closure.status==semantic_program::Status::FuelExhausted)
            return TaskProgramQueryOutcome::FuelExhausted;
        return TaskProgramQueryOutcome::Failure;
    }
    work=semantic_graph::NativeWorkTally{};
    *derived_count=closure.derived_count;
    auto& facts=state->task_evaluation.facts[slot];
    facts.g=0;facts.p=0;
    bool hard=true;
    for(uint32_t query=0;query<view.queries;++query) {
        const auto query_fact=semantic_program::bank_fact(bank.program,query);
        const uint32_t fact_index=semantic_program::find_fact(output,closure.fact_count,query_fact);
        const uint64_t truth=fact_index==closure.fact_count ? 0 : support.truth[fact_index];
        auto& record=plane.query(slot,query);
        auto* receipt=reinterpret_cast<semantic_graph::Receipt*>(record.receipt);
        for(uint32_t word=0;word<42;++word)receipt->words[word]=0;
        receipt->words[0]=semantic_graph::kOk;
        receipt->words[1]=0x5850524f47515532ULL;
        receipt->words[2]=truth;
        receipt->words[3]=slot;
        receipt->words[4]=query;
        for(uint32_t word=0;word<4;++word) {
            receipt->words[5+word]=task[2+word];
            uint64_t binding=0;
            for(uint32_t byte=0;byte<8;++byte)
                binding|=uint64_t(state->binding_digest[word*8+byte])<<(8*byte);
            receipt->words[9+word]=binding;
        }
        receipt->words[13]=inserted_count;
        for(uint32_t edit=0;edit<inserted_count;++edit) {
            const auto& rule=inserted[edit];
            uint64_t fields[10]{};
            if(!semantic_rule_action::receipt_words(bank.program,rule,fields))
                return TaskProgramQueryOutcome::Failure;
            for(uint32_t word=0;word<10;++word)receipt->words[14+edit*10+word]=fields[word];
        }
        receipt->words[34]=query_fact.predicate;
        receipt->words[35]=query_fact.first;
        receipt->words[36]=query_fact.second;
        receipt->words[37]=closure.charged_pairs;
        receipt->words[38]=closure.fact_count;
        receipt->words[39]=closure.derived_count;
        receipt->words[40]=bank.program.rule_count+plane.header[6];
        receipt->words[41]=bank.program.initial_count;
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(semantic_graph::Receipt));
        ++state->task_evaluation.query_count;
        hard&=(view.query(query)[6] & (uint64_t(1)<<truth))!=0;
        record.truth=truth;
        record.attained=view.coding ? 0 : truth==view.query(query)[5];
        if(!view.coding)facts.g+=record.attained;
    }
    if(view.coding)for(uint32_t rank=0;rank<view.queries;++rank) {
        const uint32_t query=uint32_t(view.attainment_order[rank]);
        auto& record=plane.query(slot,query);
        const auto query_fact=semantic_program::bank_fact(bank.program,query);
        uint32_t fact_index=0;
        for(;fact_index<closure.fact_count;++fact_index) {
            semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(output[fact_index].predicate==query_fact.predicate &&
               output[fact_index].arity==query_fact.arity &&
               output[fact_index].first==query_fact.first &&
               (query_fact.arity==1 || output[fact_index].second==query_fact.second))break;
        }
        uint64_t inspected=0;
        record.attained=semantic_task_ground::attained(view,plane,slot,query,record.truth,
            fact_index,closure.fact_count,support,&inspected);
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::TableSlot,inspected);
        facts.g+=record.attained;
    }
    execution_work_merge_native(state->execution_work,work);
    if(state->execution_work.overflow)return TaskProgramQueryOutcome::Failure;
    if(slot) {
        // Candidate rules only extend the same initial fact/rule bank; a
        // successful least-fixpoint closure cannot remove baseline facts.
        if(*derived_count<baseline_derived)return TaskProgramQueryOutcome::Failure;
        auto& lane_work=state->work[slot-1];
        lane_work.added_supports=*derived_count-baseline_derived;
        for(uint32_t query=0;query<view.queries;++query)
            lane_work.defined_truth_changes+=plane.query(slot,query).truth!=plane.query(0,query).truth;
    }
    facts.p=view.coding ? view.goal_count && facts.g==view.goal_count : facts.g==view.queries;
    facts.eligible=hard;
    task_cost(state,slot,view);
    return TaskProgramQueryOutcome::Ok;
}

__device__ bool program_rule_receipt(semantic_graph::Receipt* receipt,const uint64_t* task,
        const State* state,const TaskProgramBank& bank,uint32_t lane,uint32_t slot,
        const semantic_program::Rule* inserted,uint32_t inserted_count) {
    for(uint32_t word=0;word<42;++word)receipt->words[word]=0;
    receipt->words[0]=semantic_graph::kOk;
    receipt->words[1]=3;
    receipt->words[2]=lane+1;
    receipt->words[3]=slot;
    for(uint32_t word=0;word<4;++word) {
        receipt->words[4+word]=task[2+word];
        uint64_t binding=0;
        for(uint32_t byte=0;byte<8;++byte)
            binding|=uint64_t(state->binding_digest[word*8+byte])<<(8*byte);
        receipt->words[8+word]=binding;
    }
    const auto& current=inserted[inserted_count-1];
    uint64_t fields[10]{};
    if(!semantic_rule_action::receipt_words(bank.program,current,fields))return false;
    for(uint32_t word=0;word<10;++word)receipt->words[12+word]=fields[word];
    receipt->words[22]=bank.program.initial_count;
    receipt->words[23]=bank.program.rule_count;
    receipt->words[24]=inserted_count-1;
    if(inserted_count==2) {
        const auto& prior=inserted[0];
        uint64_t previous[10]{};
        if(!semantic_rule_action::receipt_words(bank.program,prior,previous))return false;
        for(uint32_t word=0;word<10;++word)receipt->words[25+word]=previous[word];
    }
    return true;
}

__device__ bool task_candidate_better(const semantic_task_ground::Task& task,
        const semantic_task_ground::Plane& plane,uint32_t candidate_index,uint32_t incumbent_index,
        const TaskFacts& candidate,const TaskFacts& incumbent) {
    uint64_t goal_offset=0;
    for(uint64_t level=0;level<task.level_count;++level) {
        int64_t candidate_score=0,incumbent_score=0;
        for(uint64_t i=0;i<task.levels[level];++i) {
            const uint64_t* goal=task.goals+(goal_offset+i)*3;
            const uint32_t query=uint32_t(goal[0]);
            const uint64_t target=goal[1];
            const auto& base=plane.query(0,query);
            const auto& current=plane.query(candidate_index,query);
            const auto& previous=plane.query(incumbent_index,query);
            const int64_t base_value=task.coding ? base.attained : base.truth==target;
            candidate_score+=int64_t(goal[2])*(int64_t(task.coding ? current.attained : current.truth==target)-base_value);
            incumbent_score+=int64_t(goal[2])*(int64_t(task.coding ? previous.attained : previous.truth==target)-base_value);
        }
        if(candidate_score!=incumbent_score)return candidate_score>incumbent_score;
        goal_offset+=task.levels[level];
    }
    return candidate.g>incumbent.g ||
        (candidate.g==incumbent.g && candidate.c<incumbent.c);
}

__device__ bool select_task(const Descriptor& descriptor,State* state,const semantic_task_ground::Task& task) {
    semantic_task_ground::Plane plane{};
    if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,task,&plane))return false;
    auto& evaluation=state->task_evaluation;
    uint64_t winner=0,refusals=0,spent=evaluation.query_count;
    for(uint32_t slot=1;slot<3;++slot) {
        const auto& current=evaluation.facts[slot];
        const auto& best=evaluation.facts[winner];
        if(current.eligible && (!best.eligible ||
           task_candidate_better(task,plane,slot,uint32_t(winner),current,best)))winner=slot;
    }
    for(uint32_t lane=0;lane<2;++lane) {
        refusals+=evaluation.lane_refusal[lane]!=0;
        if(lane+1!=winner)spent+=state->work[lane].edit_commands+state->work[lane].defined_truth_changes;
    }
    evaluation.winner=winner;
    evaluation.return_value=int64_t(task.scoring[3])*(evaluation.facts[winner].v-evaluation.facts[0].v)
        -int64_t(task.scoring[4])*int64_t(refusals)-int64_t(task.scoring[5])*int64_t(spent);
    return true;
}
// Zero means both original operations published, one is an authenticated
// reached refusal, and two is inconsistent original operation custody.
__device__ uint64_t publication_actor_refresh_outcome(const ActorRefreshWorkEntry& entry) {
    if(!publication_pointer_span(entry.proof,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice)))return 2;
    const auto& proof=*reinterpret_cast<const PreparedActorRefreshDevice*>(entry.proof);
    if(proof.abi!=1 || proof.member_ordinal!=entry.member_ordinal || proof.role>1 ||
       (proof.role==0 ? proof.demonstrated_lane!=0 :
        (proof.demonstrated_lane<1 || proof.demonstrated_lane>2)) ||
       (proof.status!=0 && proof.status!=2) ||
       !publication_pointer_span(proof.child_control,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(proof.target_control,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(proof.target_lease,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !publication_pointer_span(proof.target_header,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
       proof.target_bank>1 || !proof.minimum_generation || proof.minimum_generation>UINT32_MAX ||
       proof.stream_serial>=(1ULL<<56) || proof.family_id>UINT8_MAX || proof.proposal>=UINT32_MAX)return 2;
    const auto& target_control=*reinterpret_cast<const PublicationControl*>(proof.target_control);
    const auto& target_lease=*reinterpret_cast<const PublicationLease*>(proof.target_lease);
    const auto* target=publication_acquired_bank(target_control,target_lease);
    const auto& copied=*reinterpret_cast<const PublicationHeader*>(proof.target_header);
    if(!target || target_control.word!=target_lease.word || target_lease.transition_kind!=4 ||
       target_lease.bank!=proof.target_bank || !publication_header_equal(target->header,copied) ||
       copied.model_generation<proof.minimum_generation || copied.model_generation>UINT32_MAX ||
       (proof.status==0 && (proof.acquired_generation!=copied.model_generation ||
        !publication_identity_equal(proof.model_geometry_digest,copied.model_geometry_digest))))return 2;
    const auto& control=*reinterpret_cast<const PublicationControl*>(proof.child_control);
    bool refused=proof.status==2;
    uint64_t expected_word=0;
    for(uint32_t operation=0;operation<2;++operation) {
        if(!publication_pointer_span(entry.readers[operation],sizeof(PublicationLease),alignof(PublicationLease)) ||
           !publication_pointer_span(entry.results[operation],sizeof(PublicationStepResult),alignof(PublicationStepResult)) ||
           !publication_pointer_span(entry.parents[operation],sizeof(PublicationHeader),alignof(PublicationHeader)) ||
           !publication_pointer_span(entry.states[operation],sizeof(State),alignof(State)))return 2;
        const auto& lease=*reinterpret_cast<const PublicationLease*>(entry.readers[operation]);
        const auto& result=*reinterpret_cast<const PublicationStepResult*>(entry.results[operation]);
        if(!result.abi) {
            // Admission leaves an unentered slot at its original zero ABI and
            // writes a refusal status. No unexecuted header/state is projected.
            if(lease.abi || lease.active || !lease.status || (lease.status>5 && lease.status!=7) || lease.word || lease.bank ||
               lease.epoch || lease.transition_kind)return 2;
            for(uint32_t word=0;word<sizeof(result)/sizeof(uint64_t);++word)
                if(reinterpret_cast<const uint64_t*>(&result)[word])return 2;
            for(uint32_t word=0;word<4;++word)if(lease.instance[word])return 2;
            refused=true;
            continue;
        }
        if(refused)return 2;
        const auto& parent=*reinterpret_cast<const PublicationHeader*>(entry.parents[operation]);
        const uint64_t kind=operation==0 ? 2 : 1;
        const bool score=proof.role==1 && operation==1;
        if(lease.abi!=1 || lease.status || lease.active || lease.transition_kind!=kind ||
           lease.word!=expected_word || lease.bank!=(lease.word&1) || lease.epoch!=(lease.word>>1) ||
           !publication_identity_equal(lease.instance,control.instance) ||
           parent.abi!=1 || !publication_identity_equal(parent.instance,control.instance) ||
           parent.publication_word!=lease.word || parent.sealed_epoch!=lease.epoch ||
           parent.model_generation!=proof.acquired_generation ||
           !publication_identity_equal(parent.model_geometry_digest,proof.model_geometry_digest) ||
           result.abi!=(score ? 2 : 1) || result.header.abi!=1 ||
           !publication_identity_equal(result.header.instance,control.instance) ||
           result.header.publication_word!=result.word || result.header.sealed_epoch!=(result.word>>1) ||
           result.advanced>1 || result.advanced!=uint64_t(result.word!=lease.word && !result.refusal))return 2;
        if(score) {
            const auto& state=*reinterpret_cast<const State*>(entry.states[operation]);
            if(result.word!=lease.word || result.advanced || !publication_header_equal(result.header,parent) ||
               result.refusal!=state.status || state.blocks || state.next_proposal!=proof.proposal ||
               state.proposal!=proof.proposal || state.model_generation!=proof.acquired_generation ||
               state.stream_serial!=proof.stream_serial || state.family_id!=proof.family_id ||
               state.actor_eligible || state.action_batch.abi)return 2;
            for(uint32_t word=0;word<sizeof(result.attempt)/sizeof(uint64_t);++word)
                if(reinterpret_cast<const uint64_t*>(&result.attempt)[word])return 2;
            refused=result.refusal!=0;
            continue;
        }
        if(result.advanced) {
            const uint64_t next=((lease.epoch+1)<<1)|((lease.word^1)&1);
            if(lease.epoch==UINT64_MAX>>1 || result.word!=next || result.header.base_word!=lease.word ||
               result.attempt.abi!=2 || result.attempt.transition_kind!=kind ||
               !publication_identity_equal(result.attempt.instance,control.instance) ||
               result.attempt.base_word!=lease.word || result.attempt.next_word!=result.word ||
               !publication_identity_equal(result.attempt.logical_digest,result.header.logical_digest))return 2;
            expected_word=result.word;
        } else {
            if(result.word!=lease.word || !result.refusal)return 2;
            for(uint32_t word=0;word<sizeof(result.attempt)/sizeof(uint64_t);++word)
                if(reinterpret_cast<const uint64_t*>(&result.attempt)[word])return 2;
            refused=true;
        }
    }
    if(proof.status==0 && (control.abi!=1 || control.word!=expected_word))return 2;
    return refused ? 1 : 0;
}

__device__ void semantic_policy_backward_status_guard(uint64_t status_ptr,uint64_t mode) {
    if(!status_ptr || status_ptr%alignof(uint64_t) || status_ptr>UINT64_MAX-sizeof(uint64_t) ||
       (*reinterpret_cast<const uint64_t*>(status_ptr)!=0 &&
        !((mode==4 || mode==5) && *reinterpret_cast<const uint64_t*>(status_ptr)==4)))semantic_content_integrity_trap();
}
#ifdef XLOG_SEMANTIC_POLICY
__device__ bool semantic_policy_identity_present(const uint64_t identity[4],
        semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint32_t word=0;word<4;++word) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(identity[word])return true;
    }
    return false;
}

// Late differentiation reads the original receipt and tape. It never enters
// semantic admission, generates random words, or writes forward state.
// The selected-score reference is anchored at the actual canonical FP32
// scores and saved piecewise cell, including structural readouts. Primal
// envelopes still enter the ideal neural Jacobian; they do not move that
// score anchor or select a different active set on idealized scores.
__device__ void semantic_policy_backward(const Descriptor& descriptor,
        semantic_graph::NativeWorkTally& work,model_policy::ScalarWorkTally& scalar_work) {
    const auto slot=[&](uint64_t count=1) {
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::TableSlot,count);
    };
    const auto p=descriptor.policy;
    const auto b=descriptor.backward;
    auto status=reinterpret_cast<uint64_t*>(b.status);
    auto coefficients=reinterpret_cast<double*>(b.cotangents);
    auto gradients=reinterpret_cast<float*>(b.parameters);
    auto text_gradients=reinterpret_cast<float*>(b.text);
    auto baseline_gradients=reinterpret_cast<float*>(b.baselines);
    auto recurrent_gradients=reinterpret_cast<float*>(b.recurrent);
    auto parameter_cotangents=reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.parameter_cotangents);
    auto text_cotangents=reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.text_cotangents);
    auto baseline_cotangents=reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.baseline_cotangents);
    auto recurrent_cotangents=reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.recurrent_errors);
    auto coefficient_cotangents=reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.coefficient_errors);
    const auto* original_baseline_envelopes=reinterpret_cast<const model_policy::PrimalEnvelope*>(p.numerical.baseline_envelopes);
    auto components=reinterpret_cast<const Component*>(descriptor.components);
    auto receipts=reinterpret_cast<const Receipt*>(descriptor.receipts);
    auto support=reinterpret_cast<const uint8_t*>(descriptor.support);
    auto books=reinterpret_cast<const uint64_t*>(descriptor.codebooks);
    auto viability=reinterpret_cast<uint32_t*>(descriptor.scratch)+2*262144;
    __shared__ uint32_t actions[4][3],choices[18];
    __shared__ TaskProgramBank backward_program_bank;
    __shared__ uint32_t legal_count,active_count,selected_active,apply_selected,apply_actor,apply_edit,edit_lane;
    __shared__ uint64_t actor_denominator,edit_denominator;
    __shared__ float maximum,reward_f32,critic_scale;
    __shared__ U192 threshold;
    __shared__ double reward,actor_scale,edit_scale,cost_scale;
    __shared__ PolicyBoundedScalar bounded_reward,bounded_edit_scale,bounded_cost_scale;
    __shared__ PolicyCoherentScalar bounded_scale,bounded_actor_scale;
    __shared__ model_policy::BoundedValue bounded_reward_f32,bounded_critic_scale;
    if(threadIdx.x==0) {
        *status=0;
        if(b.mode==4 || b.mode==5) {
            const uint64_t address=descriptor.publication.native_work;
            if(!publication_pointer_span(address,sizeof(ActorRefreshWorkEntry),alignof(ActorRefreshWorkEntry)) ||
               !publication_pointer_span(b.objective,sizeof(SemanticTrainingObjectiveRecord),alignof(SemanticTrainingObjectiveRecord)) ||
               !publication_pointer_span(b.derived_origin,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice))) *status=1;
            else {
                const auto& entry=*reinterpret_cast<const ActorRefreshWorkEntry*>(address);
                const auto& proof=*reinterpret_cast<const PreparedActorRefreshDevice*>(b.derived_origin);
                const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(b.objective);
                uint64_t row_bytes=0;
                if(entry.proof!=b.derived_origin || entry.member_ordinal!=b.member_ordinal ||
                   proof.member_ordinal!=b.member_ordinal || proof.target_header!=b.target_header ||
                   proof.role!=uint64_t(b.mode==5) ||
                   b.origin_bank!=1 ||
                   (b.mode==4 ? !publication_pointer_span(b.critic_term,sizeof(float),alignof(float)) : b.critic_term!=0) ||
                   entry.states[1]!=descriptor.state || entry.readers[1]!=b.origin_lease ||
                   b.member_ordinal>=objective.row_count ||
                   !semantic_graph::checked_mul(objective.row_count,sizeof(SemanticTrainingRosterRow),&row_bytes) ||
                   !publication_pointer_span(b.roster_rows,row_bytes,alignof(SemanticTrainingRosterRow))) *status=1;
                else {
                    const auto& row=reinterpret_cast<const SemanticTrainingRosterRow*>(b.roster_rows)[b.member_ordinal];
                    if(row.ordinal!=b.member_ordinal || row.basis!=1 || row.origin.present!=1 ||
                       row.origin.transition!=1 || row.origin_candidate==UINT64_MAX-1)*status=1;
                    for(uint32_t word=0;word<sizeof(row.origin)/sizeof(uint64_t);++word) {
                        slot(2);
                        if(reinterpret_cast<const uint64_t*>(&row.origin)[word]!=
                           reinterpret_cast<const uint64_t*>(&proof.original_origin)[word])*status=1;
                    }
                    if(!*status) {
                        const uint64_t outcome=publication_actor_refresh_outcome(entry);
                        if(outcome==2)*status=1;
                        else if(outcome==1)*status=4;
                    }
                }
            }
        }
    }
    __syncthreads();
    if(*status==4) {
        // A genuine reached refusal has no actor derivative or critic term.
        // Clear the original output buffers without reading unexecuted inputs.
        for(uint64_t i=threadIdx.x;i<b.parameter_cells;i+=blockDim.x) {
            slot(2);
            gradients[i]=0.0f;parameter_cotangents[i]=model_policy::zero_cotangent();
        }
        for(uint64_t i=threadIdx.x;i<b.text_cells;i+=blockDim.x) {
            slot(2);
            text_gradients[i]=0.0f;text_cotangents[i]=model_policy::zero_cotangent();
        }
        for(uint32_t i=threadIdx.x;i<2*37*128;i+=blockDim.x) {
            slot(2);
            recurrent_gradients[i]=0.0f;recurrent_cotangents[i]=model_policy::zero_cotangent();
        }
        for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
            slot(4);
            coefficients[i]=0.0;baseline_gradients[i]=0.0f;
            baseline_cotangents[i]=model_policy::zero_cotangent();
            coefficient_cotangents[i]=model_policy::zero_cotangent();
        }
        if(threadIdx.x==0 && b.critic_term) { slot();*reinterpret_cast<float*>(b.critic_term)=0.0f; }
        return;
    }
    if(*status)return;
    if(threadIdx.x==0) {
        backward_program_bank=TaskProgramBank{};
        if(descriptor.task || descriptor.task_words) {
            if(!descriptor.task || descriptor.task%alignof(uint64_t) ||
               !task_program_bank(reinterpret_cast<const uint64_t*>(descriptor.task),
                                  descriptor.task_words,&backward_program_bank))*status=1;
            else if(backward_program_bank.editable && b.mode!=0) {
                semantic_task_ground::Plane ground{};
                if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,
                        backward_program_bank.task,&ground)) *status=1;
                else {
                    const auto& baseline=ground.query(0,0);
                    if(b.mode==5)backward_program_bank.selected_rule_count=uint32_t(ground.header[6]);
                    else if(baseline.receipt[0]!=semantic_graph::kOk ||
                       baseline.receipt[1]!=0x5850524f47515532ULL ||
                       baseline.receipt[40]<backward_program_bank.program.rule_count ||
                       baseline.receipt[40]-backward_program_bank.program.rule_count>ground.header[7]) *status=1;
                    else backward_program_bank.selected_rule_count=uint32_t(
                        baseline.receipt[40]-backward_program_bank.program.rule_count);
                }
            }
        }
        apply_selected=1;
        apply_actor=apply_edit=0;
        edit_lane=UINT32_MAX;
        actor_denominator=edit_denominator=0;
        reward=actor_scale=edit_scale=cost_scale=0.0;
        reward_f32=critic_scale=0.0f;
        bounded_reward=bounded_edit_scale=bounded_cost_scale=policy_exact_scalar(0.0);
        bounded_scale=bounded_actor_scale=policy_cotangent_seed(policy_exact_scalar(0.0),false);
        bounded_reward_f32=bounded_critic_scale={0.0f,{0.0,0.0},0.0};
        if(b.mode==0) {
            for(uint32_t i=0;i<COMPONENT_COUNT;++i) {
                slot();
                if(!isfinite(coefficients[i]))*status=1;
            }
        } else if(b.mode==4 || b.mode==5) {
            if(!b.selection || b.selection%alignof(SemanticTrainingViewSelection) ||
               !b.objective || b.objective%alignof(SemanticTrainingObjectiveRecord) ||
               !b.objective_groups || b.objective_groups%alignof(SemanticTrainingObjectiveGroupRecord) ||
               !b.objective_group_members || b.objective_group_members%alignof(uint64_t) ||
               !b.roster_rows || b.roster_rows%alignof(SemanticTrainingRosterRow)) {
                *status=1;
            } else {
                const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(b.selection);
                const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(b.objective);
                const auto* groups=reinterpret_cast<const SemanticTrainingObjectiveGroupRecord*>(b.objective_groups);
                const auto* members=reinterpret_cast<const uint64_t*>(b.objective_group_members);
                const auto* roster=reinterpret_cast<const SemanticTrainingRosterRow*>(b.roster_rows);
                constexpr uint64_t committed_append_origin=UINT64_MAX-1;
                const bool committed_append=selection.origin_candidate==committed_append_origin;
                if(selection.status || selection.basis!=1 || selection.origin.present!=1 ||
                   (!committed_append && selection.origin.transition!=1) || selection.origin_candidate==UINT64_MAX ||
                   selection.ordinal>=selection.row_count ||
                   selection.row_count!=objective.row_count ||
                   selection.capacity!=objective.capacity || objective.group_count!=8 ||
                   !objective.group_member_count || !objective.cost_cap ||
                   !semantic_policy_identity_present(objective.identity,&work) ||
                   !semantic_policy_identity_present(objective.cost_unit,&work)) {
                    *status=1;
                }
                const double evaluator_min=__longlong_as_double((long long)objective.evaluator_min_bits);
                const double evaluator_max=__longlong_as_double((long long)objective.evaluator_max_bits);
                float objective_coefficients[9];
                for(uint32_t i=0;i<9;++i) {
                    slot();
                    objective_coefficients[i]=__uint_as_float(uint32_t(objective.coefficient_bits[i]));
                    if((objective.coefficient_bits[i]>>32) || !isfinite(objective_coefficients[i]) ||
                       objective_coefficients[i]<=0.0f)*status=1;
                }
                const uint32_t fixed_coefficients[]={0,1,2,3,4,5,8};
                for(uint32_t i=0;i<7;++i) {
                    slot();
                    if(objective_coefficients[fixed_coefficients[i]]!=1.0f)*status=1;
                }
                if(!isfinite(evaluator_min) || !isfinite(evaluator_max) || evaluator_min>evaluator_max)*status=1;
                uint64_t actor_groups=0,edit_groups=0,matching_edit_rows=0;
                bool selected_member=false,group_member=false;
                for(uint64_t i=0;i<objective.group_count && !*status;++i) {
                    slot();
                    const auto group=groups[i];
                    if(!group.denominator || group.denominator!=group.member_count ||
                       group.member_offset>objective.group_member_count ||
                       group.member_count>objective.group_member_count-group.member_offset) {
                        *status=1;break;
                    }
                    if(group.kind==8) {
                        ++actor_groups;
                        actor_denominator=group.denominator;
                        for(uint64_t member=0;member<group.member_count;++member) {
                            slot();
                            const uint64_t ordinal=members[group.member_offset+member];
                            if(ordinal>=objective.row_count) {*status=1;break;}
                            if(roster[ordinal].origin_candidate==committed_append_origin) {*status=1;break;}
                            selected_member |= ordinal==selection.ordinal;
                            group_member |= b.mode==4 && ordinal==b.member_ordinal;
                        }
                    } else if(group.kind==4) {
                        ++edit_groups;
                        edit_denominator=group.denominator;
                        for(uint64_t member=0;member<group.member_count;++member) {
                            slot();
                            const uint64_t ordinal=members[group.member_offset+member];
                            if(ordinal>=objective.row_count) {*status=1;break;}
                            const auto& row=roster[ordinal];
                            if(row.origin_candidate==committed_append_origin) {
                                if(row.ordinal!=ordinal || row.basis!=1 || row.origin.present!=1)*status=1;
                                continue;
                            }
                            if(row.ordinal!=ordinal || row.basis!=1 || row.origin.present!=1 ||
                               row.origin.transition!=1) {
                                *status=1;break;
                            }
                            if(b.mode==5 && ordinal==b.member_ordinal) {
                                ++matching_edit_rows;
                            }
                        }
                    }
                }
                if(actor_groups!=1 || edit_groups!=1 ||
                   (!selected_member && !committed_append) || (selected_member && committed_append) ||
                   (b.mode==4 && (!group_member || b.member_ordinal>=objective.row_count ||
                    !b.critic_term || b.critic_term%alignof(float) ||
                    !publication_pointer_span(b.critic_term,sizeof(float),alignof(float)))) ||
                   (b.mode==5 && (b.member_ordinal>=objective.row_count ||
                    matching_edit_rows!=1)) ||
                   matching_edit_rows>1)*status=1;
                const auto* origin_lease=reinterpret_cast<const PublicationLease*>(b.origin_lease);
                if(!origin_lease || b.origin_bank>1 ||
                   !publication_pointer_span(b.origin_lease,sizeof(PublicationLease),alignof(PublicationLease)) ||
                   origin_lease->abi!=1 || origin_lease->bank>1)
                    *status=1;
                const auto* state=reinterpret_cast<const State*>(descriptor.state);
                const bool owned=origin_lease && origin_lease->bank==b.origin_bank;
                apply_actor=owned && b.mode==4 && group_member;
                apply_edit=b.mode==5 && owned && matching_edit_rows==1;
                apply_selected=apply_actor || apply_edit;
                if(apply_selected && !*status) {
                    const auto* target=reinterpret_cast<const PublicationHeader*>(b.target_header);
                    const auto& origin=roster[b.member_ordinal].origin;
                    if(!publication_pointer_span(b.target_header,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
                       target->abi!=1 || origin.present!=1)*status=1;
                }
                if((apply_actor || apply_edit) && (!state || state->status))*status=1;
                if((b.mode==4 || b.mode==5) && !*status) {
                    const auto* proof=reinterpret_cast<const PreparedActorRefreshDevice*>(b.derived_origin);
                    if(!owned || !state || state->status || (b.mode==4 ? !apply_actor : !apply_edit) ||
                       !publication_pointer_span(b.derived_origin,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice)) ||
                       proof->abi!=1 || proof->status || proof->member_ordinal!=b.member_ordinal ||
                       proof->target_header!=b.target_header || proof->target_bank>1 ||
                       proof->acquired_generation>UINT32_MAX || !proof->acquired_generation ||
                       proof->stream_serial>=(1ULL<<56) || proof->family_id>UINT8_MAX ||
                       proof->proposal>=UINT32_MAX || b.origin_bank!=1 ||
                       origin_lease->status || origin_lease->transition_kind!=1 ||
                       origin_lease->word!=3 || origin_lease->epoch!=1 ||
                       !publication_pointer_span(proof->child_control,sizeof(PublicationControl),alignof(PublicationControl))) {
                        *status=1;
                    } else {
                        const auto& row=roster[b.member_ordinal];
                        const auto& child=*reinterpret_cast<const PublicationControl*>(proof->child_control);
                        const auto& target=*reinterpret_cast<const PublicationHeader*>(b.target_header);
                        const auto* actual=reinterpret_cast<const uint64_t*>(&row.origin);
                        const auto* original=reinterpret_cast<const uint64_t*>(&proof->original_origin);
                        if(row.ordinal!=b.member_ordinal || row.basis!=1 || row.origin.present!=1 ||
                           row.origin.transition!=1 || row.origin_candidate==committed_append_origin ||
                           child.abi!=1 || child.word!=(b.mode==5 ? 3 : 4) ||
                           !publication_pointer_span(child.banks[1],sizeof(PublicationBank),alignof(PublicationBank)) ||
                           !publication_identity_equal(child.instance,origin_lease->instance) ||
                           state->model_generation!=proof->acquired_generation ||
                           state->stream_serial!=proof->stream_serial || state->family_id!=proof->family_id ||
                           state->proposal!=proof->proposal || state->next_proposal!=proof->proposal+uint64_t(b.mode==4) ||
                           (b.mode==4 ? state->actor_eligible!=1 : (state->blocks!=0 || state->actor_eligible || state->action_batch.abi)) ||
                           target.model_generation!=proof->acquired_generation ||
                           !publication_identity_equal(proof->model_geometry_digest,target.model_geometry_digest)) *status=1;
                        if(!*status) {
                            const auto& fresh=*reinterpret_cast<const PublicationBank*>(child.banks[1]);
                            if(!publication_bank_matches(child,fresh,origin_lease->word) ||
                               fresh.header.model_generation!=proof->acquired_generation ||
                               !publication_identity_equal(fresh.header.model_geometry_digest,proof->model_geometry_digest)) *status=1;
                        }
                        for(uint32_t word=0;word<sizeof(row.origin)/sizeof(uint64_t);++word) {
                            slot(2);
                            if(actual[word]!=original[word])*status=1;
                        }
                        const auto& batch=state->action_batch;
                        if(b.mode==4 && (batch.abi!=1 || batch.actor_eligible!=1 || batch.model_generation!=proof->acquired_generation ||
                           batch.stream_serial!=proof->stream_serial || batch.family_id!=proof->family_id ||
                           batch.proposal!=proof->proposal || batch.base_word!=3 || batch.next_word!=4 ||
                           batch.rng_base!=COMPONENT_COUNT*proof->proposal || batch.rng_span!=COMPONENT_COUNT ||
                           batch.rng_successor!=COMPONENT_COUNT*(proof->proposal+1))) *status=1;
                        if(b.mode==5) {
                            if(proof->demonstrated_lane<1 || proof->demonstrated_lane>2)*status=1;
                            for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
                                const uint32_t choice=receipts[ordinal].choice;slot();
                                const uint64_t original=proof->demonstrated_choices[ordinal];slot();
                                const uint64_t draw=receipts[ordinal].draw;slot();
                                if(choice!=original || draw)*status=1;
                                for(uint32_t word=0;word<4;++word) {
                                    const uint32_t counter=receipts[ordinal].counter[word];slot();
                                    const uint32_t random=receipts[ordinal].random_words[word];slot();
                                    if(counter || random)*status=1;
                                }
                                for(uint32_t word=0;word<2;++word) {
                                    const uint32_t key=receipts[ordinal].key[word];slot();
                                    if(key)*status=1;
                                }
                            }
                        }
                    }
                }
                if(apply_edit && !*status) {
                    const auto& proof=*reinterpret_cast<const PreparedActorRefreshDevice*>(b.derived_origin);
                    edit_lane=uint32_t(proof.demonstrated_lane-1);
                    bounded_edit_scale=policy_scalar_divide(policy_exact_scalar(double(objective_coefficients[3])),
                        policy_uint64_bound(edit_denominator));
                    edit_scale=bounded_edit_scale.value;
                    if(!isfinite(edit_scale))*status=1;
                }
                if(apply_actor && !*status) {
                    if(!state || state->status ||
                       !isfinite(state->importance_weight) || state->importance_weight<=0.0 ||
                       !selected_task_cost_matches(state,objective.cost_cap)) {
                        *status=1;
                    } else {
                        bounded_reward=policy_int64_bound((int64_t)state->task_evaluation.return_value);
                        bounded_reward_f32=policy_int64_float_bound((int64_t)state->task_evaluation.return_value);
                        reward=bounded_reward.value;
                        reward_f32=bounded_reward_f32.value;
                        if(reward<evaluator_min || reward>evaluator_max || !isfinite(reward))*status=1;
                        const auto denominator=policy_uint64_bound(actor_denominator);
                        const auto weight=policy_importance_bound(receipts,&work);
                        if(weight.value!=state->importance_weight)*status=1;
                        const auto cost=policy_scalar_divide(
                            policy_uint64_bound(state->task_evaluation.facts[state->task_evaluation.winner].c),
                            policy_uint64_bound(objective.cost_cap));
                        // The actor label starts at its ORIGINAL loss term,
                        // before any coefficient operation or edit/cost mix.
                        bounded_actor_scale=policy_cotangent_divide(policy_cotangent_multiply(
                            policy_cotangent_seed(policy_exact_scalar(double(objective_coefficients[6])),true),
                            weight),denominator);
                        bounded_cost_scale=policy_scalar_divide(policy_scalar_multiply(policy_scalar_multiply(
                            policy_exact_scalar(double(objective_coefficients[8])),weight),cost),denominator);
                        // Reverse the original outer coefficient and episode
                        // normalization before the square's analytic adjoint.
                        // Moving 2 across the RN32 division can change the result.
                        bounded_critic_scale=model_policy::bounded_multiply(policy_float_divide(
                            {objective_coefficients[7],{0.0,0.0},double(objective_coefficients[7])},
                            policy_uint64_float_bound(actor_denominator),denominator.ideal_minimum),
                            {2.0f,{0.0,0.0},2.0});
                        actor_scale=bounded_actor_scale.value;cost_scale=bounded_cost_scale.value;
                        critic_scale=bounded_critic_scale.value;
                        if(!isfinite(actor_scale) || !isfinite(cost_scale) || !isfinite(critic_scale))*status=1;
                    }
                }
            }
        } else {
            *status=1;
        }
        if(b.mode==4 && b.critic_term && b.critic_term%alignof(float)==0 && !*status) {
            float raw=0.0f;
            const auto* baseline=reinterpret_cast<const float*>(p.component_baselines);
            for(uint32_t i=0;i<COMPONENT_COUNT;++i) {
                slot();
                if(receipts[i].legal_count>1) {
                    slot();
                    const float difference=__fsub_rn(baseline[i],reward_f32);
                    raw=__fadd_rn(raw,__fmul_rn(difference,difference));
                }
            }
            // Keep the raw episode sum representable, then normalize it.
            // The frozen coefficient belongs after the ordered group sum.
            const float term=__fdiv_rn(raw,__ull2float_rn(actor_denominator));
            if(!isfinite(term))*status=3;
            else { slot();*reinterpret_cast<float*>(b.critic_term)=term; }
        }
    }
    __syncthreads();
    if(*status)return;
    for(uint64_t i=threadIdx.x;i<b.parameter_cells;i+=blockDim.x) {
        slot(2);
        gradients[i]=0.0f;parameter_cotangents[i]=model_policy::zero_cotangent();
    }
    for(uint64_t i=threadIdx.x;i<b.text_cells;i+=blockDim.x) {
        slot(2);
        text_gradients[i]=0.0f;text_cotangents[i]=model_policy::zero_cotangent();
    }
    for(uint32_t i=threadIdx.x;i<2*37*128;i+=blockDim.x) {
        slot(2);
        recurrent_gradients[i]=0.0f;recurrent_cotangents[i]=model_policy::zero_cotangent();
    }
    for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
        slot(2);
        baseline_cotangents[i]=model_policy::zero_cotangent();
        coefficient_cotangents[i]=model_policy::zero_cotangent();
        if(b.mode==0) {
            slot(3);
            baseline_gradients[i]=0.0f;
            coefficient_cotangents[i]=policy_cotangent_seed(policy_exact_scalar(coefficients[i]),false).envelope;
        } else {
            slot(3);
            const float baseline=baseline_gradients[i];
            coefficients[i]=0.0;
            baseline_gradients[i]=0.0f;
            if(apply_selected) {
                slot(2);
                auto coefficient=policy_cotangent_seed(policy_exact_scalar(0.0),false);
                if(apply_edit && i>=edit_lane*68+32 && i<edit_lane*68+68)
                    coefficient=policy_cotangent_seed(policy_scalar_negate(bounded_edit_scale),false);
                if(apply_actor) {
                    slot(2);
                    const auto bounded_baseline=model_policy::bound(baseline,original_baseline_envelopes,i);
                    const auto bounded_difference=model_policy::bounded_add(bounded_baseline,
                        model_policy::bounded_negate(bounded_reward_f32));
                    const float difference=bounded_difference.value;
                    const float raw_square=__fmul_rn(difference,difference);
                    // The actor reference fixes the original FP32 baseline
                    // snapshot exactly. Its live model envelope belongs to the
                    // critic difference above, not this detached coefficient.
                    const auto advantage=policy_scalar_add(bounded_reward,
                        policy_scalar_negate(policy_exact_scalar(double(baseline))));
                    coefficient=policy_cotangent_add(coefficient,policy_cotangent_add(policy_cotangent_negate(
                        policy_cotangent_multiply(bounded_actor_scale,advantage)),
                        policy_cotangent_seed(bounded_cost_scale,false)));
                    if(receipts[i].legal_count>1) {
                        slot(2);
                        const auto gradient=model_policy::cotangent_multiply(
                            policy_full_cotangent(bounded_critic_scale),bounded_difference);
                        baseline_gradients[i]=gradient.value;baseline_cotangents[i]=gradient.envelope;
                    }
                    if(!isfinite(baseline) || !isfinite(raw_square) ||
                       !isfinite(coefficient.value) || !policy_cotangent_finite(coefficient.envelope) ||
                       !isfinite(baseline_gradients[i]) || !policy_cotangent_finite(baseline_cotangents[i]))
                        atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
                }
                coefficients[i]=coefficient.value;coefficient_cotangents[i]=coefficient.envelope;
            }
        }
    }
    __syncthreads();
    if(*status)return;
    for(uint32_t edit=0;edit<4 && (b.mode!=1 || apply_selected);++edit) {
        uint32_t start=32+(edit/2)*68+(edit%2)*18;
        uint32_t* viable=viability+edit*books[1];
        for(uint32_t t=threadIdx.x;t<books[1];t+=blockDim.x) {
            slot();
            viable[t]=target_completes(books,components,support,start,t,&work);
        }
        __syncthreads();
        if(threadIdx.x==0)for(uint32_t opcode=0;opcode<3;++opcode) {
            slot();
            actions[edit][opcode]=action_completes(books,components,support,viable,start,
                                                   opcode,backward_program_bank,&work);
        }
        __syncthreads();
    }
    // Lanes are serial here because R, positions and field parameters are shared.
    for(int ordinal=COMPONENT_COUNT-1;ordinal>=0 && (b.mode!=1 || apply_selected);--ordinal) {
        if(threadIdx.x==0)slot(2);
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
        auto* score_cotangents=structural ? reinterpret_cast<model_policy::CoherentCotangent*>(b.numerical.score_errors)
            : text_cotangents+text_index*TEXT_CARDINALITY;
        model_policy::FieldView field={};
        model_policy::FieldAdjoint field_gradients={};
        if(structural) {
            const auto f=p.fields[c.field];
            field={reinterpret_cast<const float*>(f.embeddings),reinterpret_cast<const float*>(f.biases),
                f.cardinality,f.null_category};
            field_gradients={f.embeddings ? gradients+(f.embeddings-p.z)/4 : nullptr,
                f.biases ? gradients+(f.biases-p.z)/4 : nullptr};
            if(threadIdx.x==0)for(uint32_t j=0;j<18;++j) {
                slot();
                choices[j]=j<c.field ? receipts[ordinal-c.field+j].choice : 0;
            }
            policy_readout_hidden(p,lane,c.slot,step,c.field,&work);
            __syncthreads();
            policy_readout_scores(p,c.field,&work);
            __syncthreads();
            row=reinterpret_cast<const float*>(p.scores);
        }
        if(threadIdx.x==0) {
            legal_count=0;active_count=0;selected_active=0;maximum=-INFINITY;
            bounded_scale=policy_cotangent_seed(policy_exact_scalar(0.0),false);
            for(uint32_t j=0;j<c.cardinality;++j) {
                slot();
                if((support[c.offset+j] ||
                    (semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::Category,1),false)) &&
                    legal_category(c,j,books,components,support,
                    32+(edit/2)*68+(edit%2)*18,backward_program_bank,viable,choices,
                    actions[edit][0],actions[edit][1],actions[edit][2],&work)) {
                    slot();
                    if(!isfinite(row[j]))*status=2;
                    if(row[j]>maximum)maximum=row[j];
                    ++legal_count;
                }
            }
            if(legal_count!=receipt.legal_count || !receipt.active_count || receipt.active_count>legal_count
                || receipt.choice>=c.cardinality || !(slot(),support[c.offset+receipt.choice] ||
                    (semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::Category,1),false))
                 || !legal_category(c,receipt.choice,books,components,support,
                    32+(edit/2)*68+(edit%2)*18,backward_program_bank,viable,choices,
                    actions[edit][0],actions[edit][1],actions[edit][2],&work))*status=2;
            if(!*status && legal_count>1 && receipt.active_count>1) {
                U192 floor=shifted(receipt.active_count,118);
                const int positive=compare(receipt.p,floor);
                if(positive<0)*status=2;
                if(positive>0) {
                    slot();
                    selected_active=1;
                    threshold=add(sub(receipt.p,floor),multiply(distance_scaled(maximum,row[receipt.choice]),receipt.active_count));
                    // Active membership stays exact U192. Round its numerator
                    // once before the FP64 cotangent arithmetic.
                    auto divisor=policy_u192_bound(receipt.p);
                    divisor.value=ldexp(divisor.value,-149);
                    divisor.error.ordinary=ldexp(divisor.error.ordinary,-149);
                    divisor.ideal_magnitude=ldexp(divisor.ideal_magnitude,-149);
                    divisor.ideal_minimum=ldexp(divisor.ideal_minimum,-149);
                    bounded_scale=policy_cotangent_divide(
                        {coefficients[ordinal],coefficient_cotangents[ordinal]},divisor);
                }
            }
        }
        __syncthreads();
        if(*status)return;
        for(uint32_t j=threadIdx.x;j<c.cardinality;j+=blockDim.x) {
            slot(2);
            bool active=selected_active && (slot(),support[c.offset+j] ||
                (semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::Category,1),false))
                && legal_category(c,j,books,components,support,
                    32+(edit/2)*68+(edit%2)*18,backward_program_bank,viable,choices,
                    actions[edit][0],actions[edit][1],actions[edit][2],&work)
                && compare(multiply(distance_scaled(maximum,(slot(),row[j])),receipt.active_count),threshold)<0;
            if(active)atomicAdd(&active_count,1U);
            const auto contribution=active ? policy_cotangent_narrow(policy_cotangent_multiply(bounded_scale,
                policy_exact_scalar(j==receipt.choice ? double(receipt.active_count-1) : -1.0)))
                : model_policy::BoundedCotangent{0.0f,model_policy::zero_cotangent()};
            if(structural) {
                score_gradients[j]=contribution.value;score_cotangents[j]=contribution.envelope;
            } else {
                slot(2);
                model_policy::store_cotangent(model_policy::cotangent_add(
                    model_policy::cotangent_bound(score_gradients[j],score_cotangents,j),contribution),
                    score_gradients,score_cotangents,j);
            }
        }
        __syncthreads();
        if(threadIdx.x==0 && selected_active && active_count!=receipt.active_count)*status=2;
        __syncthreads();
        if(*status)return;
        if(structural) {
            float* position_gradient=gradients+(p.positions-p.z)/4+c.field*128;
            auto* position_cotangent=parameter_cotangents+(p.positions-p.z)/4+c.field*128;
            const auto f=p.fields[c.field];
            const model_policy::FieldCotangentAdjoint field_cotangent_adjoint{
                f.embeddings ? parameter_cotangents+(f.embeddings-p.z)/4 : nullptr,
                f.biases ? parameter_cotangents+(f.biases-p.z)/4 : nullptr};
            const auto* forward_recurrent_errors=reinterpret_cast<const model_policy::PrimalEnvelope*>(p.numerical.recurrent_errors);
            const model_policy::AdvanceVjpErrors advance_errors{
                forward_recurrent_errors+(lane*37+step)*128,policy_parameter_errors(p,p.recurrence),
                forward_recurrent_errors+(lane*37+step+1)*128,policy_field_errors(p,f),
                recurrent_cotangents+(lane*37+step+1)*128,recurrent_cotangents+(lane*37+step)*128,
                parameter_cotangents+(p.recurrence-p.z)/4,position_cotangent,field_cotangent_adjoint,&scalar_work};
            model_policy::advance_vjp(reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.recurrence),
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step+1)*128,
                field,receipt.choice,recurrent_gradients+(lane*37+step+1)*128,
                recurrent_gradients+(lane*37+step)*128,gradients+(p.recurrence-p.z)/4,
                position_gradient,field_gradients,threadIdx.x,blockDim.x,&advance_errors);
            __syncthreads();
            const model_policy::ReadoutVjpErrors readout_errors{
                reinterpret_cast<const model_policy::PrimalEnvelope*>(p.numerical.hidden_errors),
                policy_field_errors(p,f),score_cotangents,parameter_cotangents+(lane*2+c.slot)*128,
                recurrent_cotangents+(lane*37+step)*128,position_cotangent,field_cotangent_adjoint,&scalar_work};
            model_policy::readout_vjp(reinterpret_cast<const float*>(p.hidden),field,score_gradients,
                gradients+(lane*2+c.slot)*128,recurrent_gradients+(lane*37+step)*128,
                position_gradient,field_gradients,threadIdx.x,blockDim.x,&readout_errors);
            __syncthreads();
        }
    }
    for(uint32_t root=0;root<3;++root) {
        const auto* values=root==0 ? gradients : root==1 ? text_gradients : baseline_gradients;
        const auto* cotangents=root==0 ? parameter_cotangents : root==1 ? text_cotangents : baseline_cotangents;
        const uint64_t count=root==0 ? b.parameter_cells : root==1 ? b.text_cells : COMPONENT_COUNT;
        auto* domains=reinterpret_cast<model_policy::DomainValue*>(root==0 ? b.numerical.parameter_domains
            : root==1 ? b.numerical.text_domains : b.numerical.baseline_domains);
        auto* errors=reinterpret_cast<model_policy::ErrorEnvelope*>(root==0 ? b.numerical.parameter_errors
            : root==1 ? b.numerical.text_errors : b.numerical.baseline_errors);
        for(uint64_t i=threadIdx.x;i<count;i+=blockDim.x) {
            slot(5);
            const auto envelope=cotangents[i];
            if(!isfinite(values[i]) || !policy_cotangent_finite(envelope) ||
               envelope.full.magnitude<fabs(double(values[i])))
                atomicExch(reinterpret_cast<unsigned long long*>(status),3ULL);
            // Internal [N,2,4] is NOT the external [2,...,4] layout. Pack
            // explicitly and derive full E from this SAME coherent result.
            domains[i]=envelope.actor;
            domains[count+i]=envelope.full;
            errors[i]=envelope.full.error;
        }
    }
}
#endif

#ifdef XLOG_SEMANTIC_POLICY
extern "C" __global__ void semantic_group_critic_reduce(
    uint64_t terms_ptr,uint64_t total_ptr,uint64_t count,uint64_t objective_ptr,uint64_t work_ptr) {
    if(threadIdx.x || blockIdx.x)return;
    if(!publication_pointer_span(work_ptr,sizeof(semantic_graph::NativeWorkTally),
            alignof(semantic_graph::NativeWorkTally))) {
        semantic_content_integrity_trap();return;
    }
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(work_ptr);
    *work=semantic_graph::NativeWorkTally{};
    if(!count || !terms_ptr || !total_ptr ||
       terms_ptr%alignof(float) || total_ptr%alignof(float) ||
       count>UINT64_MAX/sizeof(float) ||
       !publication_pointer_span(terms_ptr,count*sizeof(float),alignof(float)) ||
       !publication_pointer_span(total_ptr,sizeof(float),alignof(float)) ||
       !publication_pointer_span(objective_ptr,sizeof(SemanticTrainingObjectiveRecord),
           alignof(SemanticTrainingObjectiveRecord)))
        semantic_content_integrity_trap();
    const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(objective_ptr);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    const float coefficient=__uint_as_float(uint32_t(objective.coefficient_bits[7]));
    if((semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1),objective.evaluator_abi!=3) ||
       (semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1),objective.group_count!=8) ||
       !semantic_policy_identity_present(objective.identity,work) ||
       (objective.coefficient_bits[7]>>32) || !isfinite(coefficient) || coefficient<=0.0f)
        semantic_content_integrity_trap();
    const auto* terms=reinterpret_cast<const float*>(terms_ptr);
    float total=0.0f;
    for(uint64_t ordinal=0;ordinal<count;++ordinal) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(!isfinite(terms[ordinal]))semantic_content_integrity_trap();
        total=__fadd_rn(total,terms[ordinal]);
    }
    total=__fmul_rn(coefficient,total);
    if(!isfinite(total))semantic_content_integrity_trap();
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
    *reinterpret_cast<float*>(total_ptr)=total;
}
#endif

__device__ uint64_t publication_source_records(const PublicationControl& control,const PublicationBank& bank,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* prefix=publication_find_range(ranges,bank.header.range_count,1,0,work);
    const auto* provenance=publication_find_range(ranges,bank.header.range_count,2,0,work);
    if(!prefix || !provenance || prefix->logical_begin || prefix->logical_end!=bank.header.prefix_extent ||
       prefix->length_bytes!=bank.header.prefix_extent*sizeof(SourceSlot) ||
       provenance->length_bytes%sizeof(TokenProvenanceRecord))return 1;
    const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,*prefix));
    const auto* records=reinterpret_cast<const TokenProvenanceRecord*>(publication_range_bytes(control,*provenance));
    if(!source || !records)return 1;
    for(uint64_t i=0;i<bank.header.prefix_extent+32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const SourceSlot& s=i<bank.header.prefix_extent ? source[i] : bank.source[i-bank.header.prefix_extent];
        if(i<bank.header.prefix_extent && (s.logical_position!=i || s.kind!=1 || s.valid!=1 ||
            s.committed!=1 || s.recomputed!=1))return 1;
        if(s.kind!=1)continue;
        if(s.provenance_record>=provenance->length_bytes/sizeof(TokenProvenanceRecord))return 1;
        const auto& record=records[s.provenance_record];
        const uint64_t successor=record.successor_logical_digest[0]|record.successor_logical_digest[1]|
            record.successor_logical_digest[2]|record.successor_logical_digest[3];
        if(record.logical_position!=s.logical_position || record.token!=s.token ||
           (s.provenance!=1 && s.provenance!=2) || (s.provenance==1 && successor) || (s.provenance==2 && !successor) ||
           !(record.origin_logical_digest[0]|record.origin_logical_digest[1]|record.origin_logical_digest[2]|record.origin_logical_digest[3]) ||
           !(record.authority_closure_origin[0]|record.authority_closure_origin[1]|record.authority_closure_origin[2]|record.authority_closure_origin[3]))return 1;
    }
    return 0;
}
__device__ uint64_t publication_write_feedback(const Descriptor& descriptor,PublicationControl& control,
        PublicationBank& bank,const semantic_graph::ResidentHandle& root,bool restored=false,ExecutionWork* work=nullptr,
        semantic_graph::NativeWorkTally* native_work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    auto* output_range=const_cast<PublicationRange*>(publication_find_range(ranges,bank.header.range_count,15,0,native_work));
    if(!output_range)return 1;
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    TaskProgramBank program{};
    if(!task_program_bank(task,descriptor.task_words,&program))return 1;
    const auto& view=program.task;
    const auto* ground_range=publication_find_range(ranges,bank.header.range_count,semantic_task_ground::kPublicationRole,0,native_work);
    semantic_task_ground::Plane plane{};
    if(!ground_range || !semantic_task_ground::decode_plane(
        reinterpret_cast<uint64_t>(publication_range_bytes(control,*ground_range)),ground_range->length_bytes,view,&plane) ||
        bank.state.task_evaluation.winner>2)return 1;
    const uint32_t selected=uint32_t(bank.state.task_evaluation.winner);
    auto* output=reinterpret_cast<RawFeedbackRecord*>(publication_range_bytes(control,*output_range));
    if(!output || output_range->length_bytes<uint64_t(view.queries)*sizeof(RawFeedbackRecord) ||
       output_range->length_bytes%sizeof(RawFeedbackRecord))return 1;
    if(restored) {
        if(output_range->logical_begin || output_range->logical_end!=view.queries)return 1;
        for(uint64_t i=view.queries;i<output_range->length_bytes/sizeof(RawFeedbackRecord);++i) {
            const auto* bytes=reinterpret_cast<const uint8_t*>(&output[i]);
            for(uint64_t byte=0;byte<sizeof(RawFeedbackRecord);++byte) {
                semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::CanonicalByte,1);
                if(bytes[byte])return 1;
            }
        }
    } else {
        for(uint64_t i=0;i<output_range->length_bytes/sizeof(RawFeedbackRecord);++i) {
            output[i]=RawFeedbackRecord{};
            semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(RawFeedbackRecord));
        }
    }
    for(uint32_t index=0;index<view.queries;++index) {
        semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto* statement_range=publication_find_range(ranges,bank.header.range_count,16,index,native_work);
        if(!statement_range || !publication_range_bytes(control,*statement_range))return 1;
        semantic_graph::DecodedStatement statement{};
        statement.record=uint32_t(view.query(index)[4]);
        for(uint32_t word=0;word<8;++word) {
            semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            statement.identity_words[word]=uint32_t(view.query(index)[word/2]>>(32*(word%2)));
        }
        RawFeedbackRecord expected{};
        if(program.editable) {
            const auto& actual=plane.query(selected,index);
            if(actual.truth>3 || actual.receipt[1]!=0x5850524f47515532ULL ||
               actual.receipt[2]!=actual.truth || actual.receipt[3]!=selected || actual.receipt[4]!=index)return 6;
            for(uint32_t word=0;word<42;++word) {
                semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                expected.query_receipt.words[word]=actual.receipt[word];
            }
        } else {
            semantic_call(descriptor,semantic_graph::kResidentRootTruthAdmission,&root,nullptr,&expected.query_receipt,&statement,nullptr,work);
        }
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
            for(uint32_t word=0;word<42;++word) {
                semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if((program.editable || !publication_receipt_runtime_word(word)) &&
                   saved.query_receipt.words[word]!=expected.query_receipt.words[word])return 6;
            }
        } else { output[index]=expected;
            semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(RawFeedbackRecord)); }
    }
    output_range->logical_begin=semantic_graph::written_word(0,native_work);
    output_range->logical_end=semantic_graph::written_word(view.queries,native_work);
    return 0;
}
__device__ uint64_t publication_write_coverage(const PublicationControl& control,const PublicationContract& contract,
        PublicationBank& bank,const PendingContinuation* pending,semantic_graph::NativeWorkTally* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* range=publication_find_range(ranges,bank.header.range_count,14,0,work);
    if(!range)return 1;
    auto* coverage=reinterpret_cast<CompletionCoverage*>(publication_range_bytes(control,*range));
    if(!coverage || range->length_bytes!=sizeof(CompletionCoverage))return 1;
    *coverage=CompletionCoverage{};coverage->abi=1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(CompletionCoverage)+sizeof(uint64_t));
    semantic_graph::copy_identity(coverage->instance,bank.header.instance,work);
    coverage->base_word=pending ? pending->base_word : bank.header.base_word;
    coverage->model_generation=bank.header.model_generation;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    semantic_graph::copy_identity(coverage->topology_identity,contract.topology_identity,work);
    semantic_graph::copy_identity(coverage->table_identity,contract.table_identity,work);
    const auto* prefix=publication_find_range(ranges,bank.header.range_count,3,0,work);
    if(!prefix || prefix->length_bytes!=32)return 1;
    semantic_graph::copy_identity(coverage->prefix_identity,pending ? pending->prefix_identity :
        reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*prefix)),work);
    const auto* inputs=pending ? reinterpret_cast<const PublicationRange*>(pending->ranges) : ranges;
    const uint64_t count=pending ? pending->range_count : bank.header.range_count;
    for(uint64_t i=0;i<count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& source=inputs[i];uint64_t* group=nullptr;
        if(source.role==1)group=coverage->source_origin;
        if(source.role==4 || source.role==5)group=coverage->attention_origin;
        if(source.role==6 || source.role==7)group=coverage->linear_origin;
        if(source.role>=8 && source.role<=10)group=coverage->value_origin;
        if(source.role>=11 && source.role<=13)group=coverage->edit_origin;
        if(!group)continue;
        uint64_t origin[4];
        publication_range_origin(bank.header,source,origin,work);
        publication_fold(group,source.role,source.index,origin,work);
        if(source.role==1){coverage->prefix_begin=source.logical_begin;coverage->prefix_end=source.logical_end;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));}
    }
    return publication_record_identity(control,*range,coverage->receipt_digest,work);
}
__device__ uint64_t publication_write_attempt(const PublicationControl& control,PublicationBank& bank,
        const PublicationBank& base,uint64_t transition_kind,semantic_graph::NativeWorkTally* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* target=publication_find_range(ranges,bank.header.range_count,33,0,work);
    if(!target)return 1;
    auto* attempt=reinterpret_cast<AttemptReceipt*>(publication_range_bytes(control,*target));
    if(!attempt || target->length_bytes!=sizeof(AttemptReceipt))return 1;
    if(transition_kind<1 || transition_kind>5)return 1;
    *attempt=AttemptReceipt{};
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(*attempt));
    attempt->abi=semantic_graph::written_word(2,work);
    attempt->transition_kind=semantic_graph::written_word(transition_kind,work);
    semantic_graph::copy_identity(attempt->instance,bank.header.instance,work);
    attempt->base_word=semantic_graph::written_word(base.header.publication_word,work);
    attempt->next_word=semantic_graph::written_word(bank.header.publication_word,work);
    semantic_graph::copy_identity(attempt->logical_digest,bank.header.logical_digest,work);
    if(transition_kind!=1) {
        if(publication_action_digest(control,bank,bank.receipts,0,
                attempt->action_receipts_digest,work))return 1;
    } else {
        uint64_t root[4];
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&bank.state.action_batch),
            sizeof(bank.state.action_batch),root,work);
        if(bank.state.action_batch.abi!=1 || bank.state.action_batch.component_count!=COMPONENT_COUNT ||
           !publication_identity_equal(root,bank.state.action_batch_root,work))return 1;
        semantic_graph::copy_identity(attempt->action_receipts_digest,root,work);
    }
    semantic_graph::sha256(PublicationSemanticReceiptBytes{bank.state.semantic_receipts},sizeof(bank.state.semantic_receipts),attempt->semantic_receipts_digest,work);
    const uint64_t role[5]={14,27,30,32,33};
    uint64_t* output[5]={attempt->coverage_digest,attempt->replay_head_digest,attempt->intent_head_digest,
        attempt->acknowledgement_head_digest,attempt->previous_attempt_digest};
    for(uint32_t i=0;i<5;++i) {
        const auto* source=publication_find_range(i==4 ? previous : ranges,bank.header.range_count,role[i],0,work);
        if(!source || publication_record_identity(control,*source,output[i],work))return 1;
    }
    return publication_record_identity(control,*target,attempt->receipt_digest,work);
}
__device__ bool publication_charge_native_work(semantic_graph::NativeWorkTally* target,
        const ExecutionWork& actual,uint64_t* charged);
__device__ uint64_t publication_initialize(const Descriptor& descriptor,PublicationControl& control,uint64_t operation,
        semantic_graph::NativeWorkTally* native_work=nullptr) {
    const bool restored=operation!=1;
    if(control.abi!=1 || control.word || control.reader_counts[0] || control.reader_counts[1] ||
       !control.contract || !control.banks[0] || !control.banks[1] || control.banks[0]==control.banks[1] ||
       !control.directories[0] || !control.directories[1] || !descriptor.task || !descriptor.state ||
       !descriptor.codebooks || publication_validate_storage(control,native_work))return 1;
    auto& bank=*publication_bank(control,0);auto& inactive=*publication_bank(control,1);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    if(bank.header.abi || inactive.header.abi || bank.header.sealed_epoch || bank.header.base_word ||
       bank.header.publication_word || bank.header.neural_bank || contract.abi!=1 || contract.window_capacity!=32 ||
       contract.semantic_owner!=descriptor.arena[1] || bank.header.semantic_owner!=descriptor.arena[1] ||
       !contract.model_generation || bank.header.model_generation>UINT32_MAX ||
       (restored ? bank.header.model_generation<contract.model_generation : bank.header.model_generation!=contract.model_generation) ||
       bank.header.authority_generation!=contract.authority_generation ||
        !publication_identity_equal(contract.task_identity,task+2) || task[0]!=9 || task[1]!=descriptor.arena[1] ||
       !publication_identity_equal(control.instance,bank.header.instance) || bank.header.range_count!=inactive.header.range_count ||
       (contract.terminal_token_count && !contract.terminal_tokens))return 1;
    uint64_t end=0;
    if(publication_validate_source(bank,contract,&end,native_work) || publication_source_records(control,bank,native_work))return 1;
    for(uint32_t index=0;index<2;++index) {
        semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(publication_validate_directory(control,contract,reinterpret_cast<PublicationRange*>(control.directories[index]),bank.header.range_count,native_work))return 1;
    }
    if(publication_validate_destinations(control,reinterpret_cast<const PublicationRange*>(control.directories[0]),
        reinterpret_cast<const PublicationRange*>(control.directories[1]),bank.header.range_count,native_work) ||
       publication_validate_intents(control,reinterpret_cast<const PublicationRange*>(control.directories[0]),bank.header.range_count,UINT64_MAX,native_work))return 1;
    const auto* book_range=publication_find_range(reinterpret_cast<const PublicationRange*>(control.directories[0]),bank.header.range_count,48,0,native_work);
    if(!book_range || book_range->length_bytes%8 || book_range->length_bytes<25*8)return 1;
    const auto* actual_books=reinterpret_cast<const uint64_t*>(publication_range_bytes(control,*book_range));
    const auto* descriptor_books=reinterpret_cast<const uint64_t*>(descriptor.codebooks);
    for(uint64_t i=0;i<book_range->length_bytes/8;++i) {
        semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(actual_books[i]!=descriptor_books[i])return 1;
    }
    semantic_graph::ResidentHandle root={bank.header.semantic_owner,semantic_graph::kRootKind,bank.header.semantic_slot,bank.header.semantic_generation};
    const PublicationHeader original_header=bank.header;
    if(operation!=4)semantic_graph::copy_identity(bank.header.original_instance,control.instance,native_work);
    if(operation==1)semantic_graph::copy_identity(bank.header.logical_digest,bank.header.original_instance,native_work);
    // Restored receipts are historical evidence, never fresh runtime handles.
    // Validate the rematerialized root without rewriting that saved history.
    State preflight{};
    State* checked=restored ? &preflight : &bank.state;
    if(!restored) { bank.state=*reinterpret_cast<const State*>(descriptor.state);
        semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(State)); }
    ExecutionWork semantic_work{};uint64_t charged[9]={};
    semantic_call(descriptor,semantic_graph::kResidentPreflightTransitionAdmission,&root,nullptr,&checked->semantic_receipts[0],
        nullptr,nullptr,native_work ? &semantic_work : nullptr);
    if(!publication_charge_native_work(native_work,semantic_work,charged))return 6;
    if(checked->semantic_receipts[0].words[0]!=semantic_graph::kOk ||
       !publication_identity_equal(checked->semantic_receipts[0].words+16,bank.header.semantic_digest))return 6;
    for(uint32_t i=0;i<3;++i) {
        semantic_graph::charge_native(native_work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const uint64_t extent=checked->semantic_receipts[0].words[13+i];
        if(restored && bank.header.semantic_extents[i]!=extent)return 6;
        if(!restored)bank.header.semantic_extents[i]=semantic_graph::written_word(extent,native_work);
    }
    TaskProgramBank program{};
    if(!task_program_bank(task,descriptor.task_words,&program))return 6;
    if(program.editable) {
        if(restored) {
            // Historical query receipts retain their original full invocation.
            // Restored external supports are authenticated before hot execution.
            checked->task_evaluation=bank.state.task_evaluation;
        } else {
            const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[0]);
            const auto* ground=publication_find_range(ranges,bank.header.range_count,semantic_task_ground::kPublicationRole);
            auto* bytes=ground ? publication_range_bytes(control,*ground) : nullptr;
            if(!ground || !bytes || ground->length_bytes!=descriptor.task_ground_bytes)return 1;
            Descriptor initial=descriptor;initial.task_ground=reinterpret_cast<uint64_t>(bytes);
            initial.task_ground_bytes=ground->length_bytes;
            uint32_t derived=0;
            if(native_work)checked->execution_work=semantic_work;
            if(task_program_queries(initial,task,program,0,nullptr,0,0,checked,&derived)!=TaskProgramQueryOutcome::Ok)return 6;
            if(native_work)semantic_work=checked->execution_work;
            if(!publication_charge_native_work(native_work,semantic_work,charged))return 6;
        }
    } else {
        if(native_work)checked->execution_work=semantic_work;
        const bool queried=task_queries(descriptor,task,0,&root,nullptr,checked);
        if(native_work)semantic_work=checked->execution_work;
        if(!publication_charge_native_work(native_work,semantic_work,charged) || !queried)return 6;
    }
    if(!checked->task_evaluation.facts[0].eligible)return 1;
    if(!restored) {
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[0]);
        const auto* prefix=publication_find_range(ranges,bank.header.range_count,3);
        const auto* source=publication_find_range(ranges,bank.header.range_count,1);
        if(!prefix || !source || prefix->length_bytes!=32)return 1;
        auto* identity=publication_range_bytes(control,*prefix);
        // This native output starts at the original publication's root key.
        if(!identity)return 1;
        semantic_graph::copy_identity(reinterpret_cast<uint64_t*>(identity),bank.header.original_instance,native_work);
    }
    const uint64_t feedback=publication_write_feedback(descriptor,control,bank,root,restored,
        native_work ? &semantic_work : nullptr,native_work);
    if(!publication_charge_native_work(native_work,semantic_work,charged) || feedback ||
       (!restored && publication_write_coverage(control,contract,bank,nullptr,native_work)))return 1;
    // The reader gate retains the original coordinates until initialization
    // succeeds. Any refusal invalidates this private bank before gate release.
    bank.header.abi=semantic_graph::written_word(1,native_work);
    if(operation!=1 && operation!=4 && operation!=10)
        publication_derive_origin(bank,original_header,operation==7 ? PublicationOriginKind::LearningFold :
            PublicationOriginKind::LearningPhaseTransition,native_work);
    if(publication_finalize_model_origin(control,bank,operation==4,native_work)){
        bank.header.abi=semantic_graph::written_word(0,native_work);return 1;
    }
    return 0;
}
// These destinations remain inert until the original publication word commits.
// The same transaction publishes the acknowledgement and the complete replay payload.
__device__ uint64_t publication_append_replay(PublicationControl& control,const PublicationBank& base,
        const PublicationRange* old,PublicationRange* next_ranges,const PublicationRange& entries_template,
        const PublicationRange& payload_template,const PublicationReplayDeliveryInput& input,
        const IntentEntry& intent,semantic_graph::NativeWorkTally* work=nullptr,bool commit=true) {
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto* prior_entries=publication_find_range(old,base.header.range_count,56,0,work);
    const auto* prior_payload=publication_find_range(old,base.header.range_count,57,0,work);
    auto* next_entries=const_cast<PublicationRange*>(publication_find_range(next_ranges,base.header.range_count,56,0,work));
    auto* next_payload=const_cast<PublicationRange*>(publication_find_range(next_ranges,base.header.range_count,57,0,work));
    if(!prior_entries || !prior_payload || !next_entries || !next_payload ||
       prior_entries->length_bytes<sizeof(SemanticTrainingReplayAppendHeader) ||
       entries_template.storage_slot>=control.storage_count || payload_template.storage_slot>=control.storage_count ||
       entries_template.storage_slot==prior_entries->storage_slot || payload_template.storage_slot==prior_payload->storage_slot ||
       entries_template.storage_slot==payload_template.storage_slot)return 1;
    const auto* old_bytes=publication_range_bytes(control,*prior_entries);
    const auto* old_payload=publication_range_bytes(control,*prior_payload);
    if(!old_bytes || (prior_payload->length_bytes && !old_payload))return 1;
    const auto& header=*reinterpret_cast<const SemanticTrainingReplayAppendHeader*>(old_bytes);
    if(header.abi!=1 || header.original_stage>1 || header.original_count>contract.replay_original_slot_boundary ||
       (input.row_count && (header.original_stage!=1 ||
        !(header.original_prefix_identity[0]|header.original_prefix_identity[1]|header.original_prefix_identity[2]|header.original_prefix_identity[3]))) ||
       header.eligible_count>header.count || header.count>header.capacity ||
       input.row_count>header.capacity-header.count || contract.replay_original_slot_boundary>contract.replay_row_capacity ||
       header.capacity!=contract.replay_row_capacity-contract.replay_original_slot_boundary ||
       header.capacity>(UINT64_MAX-sizeof(header))/sizeof(SemanticTrainingReplayAppendEntry) ||
       prior_entries->length_bytes!=sizeof(header)+header.count*sizeof(SemanticTrainingReplayAppendEntry) ||
       prior_payload->length_bytes!=header.payload_used_bytes || header.payload_used_bytes>header.payload_capacity_bytes ||
       header.payload_capacity_bytes!=contract.replay_payload_capacity_bytes ||
       input.row_count>UINT64_MAX/sizeof(PublicationReplayDeliveryRow) ||
       (input.row_count && !publication_pointer_span(input.rows,input.row_count*sizeof(PublicationReplayDeliveryRow),alignof(PublicationReplayDeliveryRow))))return 1;
    const auto* items=reinterpret_cast<const SemanticTrainingReplayAppendEntry*>(old_bytes+sizeof(header));
    uint64_t chain[4]={},used=0,eligible_seen=0;
    for(uint64_t i=0;i<header.count;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::ChainLink,1);
        const auto& item=items[i];
        if((item.disposition!=1 && item.disposition!=2) ||
           item.row_ordinal!=(item.disposition==1 ? header.original_count+eligible_seen : UINT64_MAX) ||
           item.payload_offset_bytes!=used || used>header.payload_used_bytes ||
           item.payload_length_bytes>header.payload_used_bytes-used ||
           (input.row_count && publication_identity_equal(item.stable_intent,intent.stable_identity,work)) ||
           !(item.result_receipt_digest[0]|item.result_receipt_digest[1]|item.result_receipt_digest[2]|item.result_receipt_digest[3]))return 1;
        uint64_t digest[4];
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&item),sizeof(item),digest,work);
        publication_fold(chain,56,i,digest,work);
        semantic_graph::sha256(old_payload+used,item.payload_length_bytes,digest,work);
        publication_fold(chain,57,i,digest,work);
        used+=item.payload_length_bytes;eligible_seen+=item.disposition==1;
    }
    if(used!=header.payload_used_bytes || eligible_seen!=header.eligible_count ||
       !publication_identity_equal(chain,header.chain_head,work))return 1;
    const auto* rows=reinterpret_cast<const PublicationReplayDeliveryRow*>(input.rows);
    uint64_t eligible_added=0;
    for(uint64_t index=0;index<input.row_count;++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& value=rows[index];const auto& added=value.entry;const auto& row=value.row_descriptor;
        const bool eligible=added.disposition==1;
        uint64_t physical=0;
        if(!semantic_graph::checked_add(contract.replay_original_slot_boundary,header.eligible_count,&physical) ||
           !semantic_graph::checked_add(physical,eligible_added,&physical))return 1;
        if((added.disposition!=1 && added.disposition!=2) ||
           added.row_ordinal!=(eligible ? header.original_count+header.eligible_count+eligible_added : UINT64_MAX) ||
           input.arena_descriptor_count!=contract.replay_row_capacity || input.arena_raw_bytes!=contract.replay_raw_capacity_bytes ||
           row.ordinal!=added.row_ordinal || row.basis!=1 || row.origin.present!=1 || row.origin.transition!=1 ||
           row.raw_bytes!=value.row_bytes_len || !row.raw_bytes ||
           (eligible && (physical>=input.arena_descriptor_count || physical>UINT64_MAX/row.raw_bytes ||
             row.raw_offset!=physical*row.raw_bytes || row.raw_offset>input.arena_raw_bytes ||
             row.raw_bytes>input.arena_raw_bytes-row.raw_offset)) || (!eligible && row.raw_offset!=0) ||
           !publication_identity_equal(row.content_identity,added.content_identity,work) ||
           !publication_identity_equal(added.stable_intent,intent.stable_identity,work) ||
           !(row.origin.predecessor_logical[0]|row.origin.predecessor_logical[1]|row.origin.predecessor_logical[2]|row.origin.predecessor_logical[3]) ||
           !(row.origin.successor_logical[0]|row.origin.successor_logical[1]|row.origin.successor_logical[2]|row.origin.successor_logical[3]) ||
           !(added.result_receipt_digest[0]|added.result_receipt_digest[1]|added.result_receipt_digest[2]|added.result_receipt_digest[3]) ||
           added.payload_offset_bytes!=used || added.payload_length_bytes!=value.payload_bytes ||
           used>header.payload_capacity_bytes || value.payload_bytes>header.payload_capacity_bytes-used ||
           !publication_pointer_span(value.payload,value.payload_bytes,1) ||
           !publication_pointer_span(value.row_bytes,value.row_bytes_len,1) ||
           value.payload_bytes<40+sizeof(TrainingViewRowDescriptor) ||
           value.row_bytes!=value.payload+40+sizeof(TrainingViewRowDescriptor) ||
           row.raw_bytes>value.payload_bytes-40-sizeof(TrainingViewRowDescriptor) ||
           input.arena_descriptor_count>UINT64_MAX/sizeof(TrainingViewRowDescriptor) ||
           !publication_pointer_span(input.arena_descriptors,input.arena_descriptor_count*sizeof(TrainingViewRowDescriptor),alignof(TrainingViewRowDescriptor)) ||
           !publication_pointer_span(input.arena_raw,input.arena_raw_bytes,1))return 1;
        const auto* origin_words=reinterpret_cast<const uint64_t*>(&row.origin);
        const auto* entry_origin=reinterpret_cast<const uint64_t*>(&added.original_origin);
        for(uint32_t word=0;word<sizeof(row.origin)/8;++word) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(origin_words[word]!=entry_origin[word])return 1;
        }
        for(uint64_t prior=0;prior<index;++prior) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(publication_identity_equal(rows[prior].entry.original_origin.predecessor_logical,row.origin.predecessor_logical,work) &&
               publication_identity_equal(rows[prior].entry.original_origin.successor_logical,row.origin.successor_logical,work))return 1;
        }
        used+=value.payload_bytes;eligible_added+=eligible;
    }
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    const uint64_t entries_length=sizeof(header)+(header.count+input.row_count)*sizeof(SemanticTrainingReplayAppendEntry);
    if(entries_template.offset_bytes>storage[entries_template.storage_slot].bytes ||
       entries_length>storage[entries_template.storage_slot].bytes-entries_template.offset_bytes ||
       payload_template.offset_bytes>storage[payload_template.storage_slot].bytes ||
       used>storage[payload_template.storage_slot].bytes-payload_template.offset_bytes)return 1;
    PublicationRange staged_entries=entries_template,staged_payload=payload_template;
    staged_entries.length_bytes=entries_length;staged_payload.length_bytes=used;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        2*sizeof(PublicationRange)+2*sizeof(uint64_t));
    uint8_t* staged=nullptr;
    if(!publication_resolve_range(control,staged_entries,&staged) ||
       !publication_resolve_range(control,staged_payload,&staged))return 1;
    if(!commit || !input.row_count)return 0;
    next_entries->storage_slot=entries_template.storage_slot;
    next_entries->length_bytes=entries_length;next_entries->logical_begin=0;next_entries->logical_end=header.count+input.row_count;
    next_payload->storage_slot=payload_template.storage_slot;
    next_payload->length_bytes=used;next_payload->logical_begin=0;next_payload->logical_end=used;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8*sizeof(uint64_t));
    auto* target_entries=publication_range_bytes(control,*next_entries);
    auto* target_payload=publication_range_bytes(control,*next_payload);
    if(!target_entries || !target_payload)return 1;
    publication_copy_bytes(target_entries,old_bytes,prior_entries->length_bytes,work);
    if(header.payload_used_bytes)publication_copy_bytes(target_payload,old_payload,header.payload_used_bytes,work);
    uint64_t physical=contract.replay_original_slot_boundary+header.eligible_count;
    for(uint64_t index=0;index<input.row_count;++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& value=rows[index];const auto& added=value.entry;const auto& row=value.row_descriptor;
        publication_copy_bytes(target_payload+added.payload_offset_bytes,reinterpret_cast<const uint8_t*>(value.payload),value.payload_bytes,work);
        reinterpret_cast<SemanticTrainingReplayAppendEntry*>(target_entries+sizeof(header))[header.count+index]=added;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(added));
        uint64_t digest[4];semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&added),sizeof(added),digest,work);
        publication_fold(chain,56,header.count+index,digest,work);
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(value.payload),value.payload_bytes,digest,work);
        publication_fold(chain,57,header.count+index,digest,work);
        if(added.disposition==1) {
            publication_copy_bytes(reinterpret_cast<uint8_t*>(input.arena_raw+row.raw_offset),
                reinterpret_cast<const uint8_t*>(value.row_bytes),row.raw_bytes,work);
            reinterpret_cast<TrainingViewRowDescriptor*>(input.arena_descriptors)[physical++]=row;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(row));
        }
    }
    auto* next_header=reinterpret_cast<SemanticTrainingReplayAppendHeader*>(target_entries);
    next_header->count+=input.row_count;next_header->payload_used_bytes=used;next_header->eligible_count+=eligible_added;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
    semantic_graph::copy_identity(next_header->chain_head,chain,work);
    return 0;
}
// Reuse the semantic owner's retirement operation only after the publication
// reader gate proves this backing is inactive. The live base remains protected.
__device__ bool publication_retire_root(const Descriptor& descriptor,
        const semantic_graph::ResidentHandle& root,const semantic_graph::ResidentHandle& protected_root,
        ExecutionWork* work) {
    if(root.slot==0 || (root.slot==protected_root.slot && root.generation==protected_root.generation))return true;
    if(root.owner!=protected_root.owner || root.owner!=descriptor.arena[1])return false;
    semantic_graph::Command command{};command.words[0]=semantic_graph::kRetireRoot;
    command.words[1]=root.owner;command.words[8]=root.slot;command.words[9]=root.generation;
    command.words[10]=protected_root.slot;command.words[11]=protected_root.generation;
    semantic_graph::Receipt receipt{};
    semantic_graph::LaunchDescriptor launch{};
    launch.expected_owner=descriptor.arena[1];launch.root_capacity=descriptor.arena[2];
    launch.statement_capacity=descriptor.arena[3];launch.support_capacity=descriptor.arena[4];
    launch.version_capacity=descriptor.arena[5];launch.arena_words=descriptor.arena[6];
    launch.command_ptr=reinterpret_cast<uint64_t>(&command);
    launch.receipt_ptr=reinterpret_cast<uint64_t>(&receipt);
    launch.abi_generation=semantic_graph::kAbiGeneration;
    semantic_graph::NativeWorkTally actual{};
    semantic_graph::execute(reinterpret_cast<uint64_t*>(descriptor.arena[0]),launch,work ? &actual : nullptr);
    if(work)execution_work_merge_native(*work,actual);
    return receipt.words[0]==semantic_graph::kOk;
}

__device__ bool publication_charge_native_work(semantic_graph::NativeWorkTally* target,
        const ExecutionWork& actual,uint64_t* charged) {
    if(!target)return !actual.overflow;
    for(uint32_t event=0;event<9;++event) {
        if(actual.native_events[event]<charged[event])return false;
        semantic_graph::charge_native(target,static_cast<semantic_graph::NativeWorkEvent>(event),
            actual.native_events[event]-charged[event]);
        charged[event]=actual.native_events[event];
    }
    if(actual.overflow)target->overflow=1;
    return !target->overflow;
}

__device__ uint64_t publication_acknowledgement_chain(const uint8_t* bytes,uint64_t length,
        const AcknowledgementQueueHeader& queue,const IntentEntry* selected,const uint64_t* selected_digest,
        const IntentQueueHeader* all_intents,uint64_t* receipt_used,uint64_t* chain,
        semantic_graph::NativeWorkTally* work=nullptr,const uint64_t* selected_outcome=nullptr) {
    if(!bytes || length<sizeof(AcknowledgementQueueHeader) || queue.abi!=1 || queue.count>queue.capacity ||
       queue.payload_used_bytes!=length-sizeof(AcknowledgementQueueHeader) ||
       queue.capacity>(UINT64_MAX-sizeof(AcknowledgementQueueHeader))/sizeof(AcknowledgementEntry) ||
       queue.payload_capacity_bytes>UINT64_MAX-sizeof(AcknowledgementQueueHeader)-queue.capacity*sizeof(AcknowledgementEntry) ||
       (all_intents && (!all_intents->count || queue.count!=all_intents->count)))return 1;
    uint64_t cursor=sizeof(AcknowledgementQueueHeader);*receipt_used=0;
    for(uint32_t word=0;word<4;++word)chain[word]=0;
    for(uint64_t ordinal=0;ordinal<queue.count;++ordinal) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::ChainLink,1);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(cursor>length || length-cursor<sizeof(AcknowledgementEntry))return 1;
        const auto& item=*reinterpret_cast<const AcknowledgementEntry*>(bytes+cursor);
        if(item.receipt_len>length-cursor-sizeof(AcknowledgementEntry) ||
           item.receiver_outcome_len>length-cursor-sizeof(AcknowledgementEntry)-item.receipt_len ||
           !publication_identity_equal(item.previous_chain,chain,work))return 1;
        uint64_t digest[4];
        semantic_graph::sha256(bytes+cursor+sizeof(AcknowledgementEntry),item.receipt_len,digest,work);
        if(!publication_identity_equal(digest,item.receipt_digest,work))return 1;
        uint64_t outcome[4]={};
        if(item.receiver_outcome_len)
            semantic_graph::sha256(bytes+cursor+sizeof(AcknowledgementEntry)+item.receipt_len,item.receiver_outcome_len,outcome,work);
        if(!publication_identity_equal(outcome,item.receiver_outcome_digest,work))return 1;
        publication_fold(chain,32,ordinal,item.stable_identity,work);
        publication_fold(chain,32,ordinal,item.receipt_digest,work);
        publication_fold(chain,32,ordinal,item.receiver_outcome_digest,work);
        if(!publication_identity_equal(chain,item.chain,work))return 1;
        if(selected && publication_identity_equal(item.stable_identity,selected->stable_identity,work))
            return publication_identity_equal(item.receipt_digest,selected_digest,work) &&
                (selected_outcome ? publication_identity_equal(item.receiver_outcome_digest,selected_outcome,work) :
                 !(outcome[0]|outcome[1]|outcome[2]|outcome[3])) ? 7 : 1;
        if(all_intents) {
            const auto* entries=reinterpret_cast<const IntentEntry*>(all_intents+1);
            uint64_t matches=0;
            for(uint64_t i=0;i<all_intents->count;++i) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                matches+=publication_identity_equal(item.stable_identity,entries[i].stable_identity,work);
            }
            if(matches!=1)return 1;
            uint64_t prior_cursor=sizeof(AcknowledgementQueueHeader);
            for(uint64_t prior=0;prior<ordinal;++prior) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const auto& previous=*reinterpret_cast<const AcknowledgementEntry*>(bytes+prior_cursor);
                if(publication_identity_equal(previous.stable_identity,item.stable_identity,work))return 1;
                prior_cursor+=sizeof(AcknowledgementEntry)+previous.receipt_len+previous.receiver_outcome_len;
            }
        }
        if(*receipt_used>queue.payload_capacity_bytes || item.receipt_len>queue.payload_capacity_bytes-*receipt_used)return 1;
        *receipt_used+=item.receipt_len;
        if(item.receiver_outcome_len>queue.payload_capacity_bytes-*receipt_used)return 1;
        *receipt_used+=item.receiver_outcome_len;
        cursor+=sizeof(AcknowledgementEntry)+item.receipt_len+item.receiver_outcome_len;
    }
    return cursor==length && publication_identity_equal(chain,queue.chain_head,work) ? 0 : 1;
}

__device__ uint64_t publication_acknowledge(const Descriptor& descriptor,PublicationControl& control,
        const PublicationDeliveryInput& input,const PublicationReplayDeliveryInput* replay=nullptr) {
    auto* cold_work=replay && replay->native_work
        ? reinterpret_cast<semantic_graph::NativeWorkTally*>(replay->native_work) : nullptr;
    semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::Command,1);
    if(!input.lease || !input.receipt || input.receipt_len<169 || !control.contract ||
       !publication_compare_exchange(control.reader_gate,0,1))return 2;
    semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.reader_gate));
    uint64_t status=1;
    semantic_graph::ResidentHandle protected_root{};
    semantic_graph::Receipt candidate{},sealed{};
    ExecutionWork work{};
    bool candidate_live=false,sealed_live=false;
    PublicationBank* staged_bank=nullptr;
    uint64_t charged_events[9]={};
    do {
        const auto& lease=*reinterpret_cast<const PublicationLease*>(input.lease);
        const uint64_t word=publication_load(control.word);
        const auto* base=publication_acquired_bank(control,lease);
        if(!replay && reinterpret_cast<const PublicationContract*>(control.contract)->nonterminal_intent_mode==1)break;
        if(!base || lease.word!=word || (replay ? (base->header.terminal!=0 && base->header.terminal!=2) : base->header.terminal!=2) || !base->header.fuel ||
           control.reader_counts[(word&1)^1] || (word>>1)==(UINT64_MAX>>1) ||
           publication_validate_selected_model(control,*base,cold_work)) { status=3;break; }
        protected_root={base->header.semantic_owner,semantic_graph::kRootKind,
            base->header.semantic_slot,base->header.semantic_generation};
        const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[word&1]);
        auto* next_ranges=reinterpret_cast<PublicationRange*>(control.directories[(word&1)^1]);
        if(!old || !next_ranges || publication_validate_intents(control,old,base->header.range_count,UINT64_MAX,cold_work))break;
        const auto* intent_range=publication_find_range(old,base->header.range_count,30,0,cold_work);
        const auto* payload_range=publication_find_range(old,base->header.range_count,31,0,cold_work);
        const auto* prior_ack=publication_find_range(old,base->header.range_count,32,0,cold_work);
        auto* ack_target=const_cast<PublicationRange*>(publication_find_range(next_ranges,base->header.range_count,32,0,cold_work));
        auto* attempt_target=const_cast<PublicationRange*>(publication_find_range(next_ranges,base->header.range_count,33,0,cold_work));
        if(!intent_range || !payload_range || !prior_ack || !ack_target || !attempt_target ||
           prior_ack->length_bytes<sizeof(AcknowledgementQueueHeader))break;
        const PublicationRange ack_template=*ack_target,attempt_template=*attempt_target;
        PublicationRange append_entries_template{},append_payload_template{};
        if(replay) {
            const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
            const auto* entry_target=publication_find_range(next_ranges,base->header.range_count,56,0,cold_work);
            const auto* payload_target=publication_find_range(next_ranges,base->header.range_count,57,0,cold_work);
            if(contract.nonterminal_intent_mode!=1 || !entry_target || !payload_target)break;
            append_entries_template=*entry_target;append_payload_template=*payload_target;
        }
        const auto* intents=reinterpret_cast<const IntentQueueHeader*>(publication_range_bytes(control,*intent_range));
        const auto* payload=publication_range_bytes(control,*payload_range);
        const auto* old_queue=reinterpret_cast<const AcknowledgementQueueHeader*>(publication_range_bytes(control,*prior_ack));
        if(!intents || !payload || !old_queue || old_queue->abi!=1 ||
           old_queue->count>old_queue->capacity ||
           old_queue->payload_used_bytes!=prior_ack->length_bytes-sizeof(AcknowledgementQueueHeader) ||
           old_queue->capacity>(UINT64_MAX-sizeof(AcknowledgementQueueHeader))/sizeof(AcknowledgementEntry) ||
           old_queue->payload_capacity_bytes>UINT64_MAX-sizeof(AcknowledgementQueueHeader)-
               old_queue->capacity*sizeof(AcknowledgementEntry))break;
        const auto* receipt=reinterpret_cast<const uint8_t*>(input.receipt);
        const char domain[]="xlog.delivery.receipt.v1";
        bool exact=true;
        for(uint32_t i=0;i<sizeof(domain);++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            exact &= receipt[i]==uint8_t(domain[i]);
        }
        for(uint32_t i=0;i<32;++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            exact &= receipt[57+i]==reinterpret_cast<const uint8_t*>(input.recipient_id)[i];
        }
        if(!exact)break;
        uint64_t effect_len=0;
        for(uint32_t i=0;i<8;++i)effect_len|=uint64_t(receipt[89+i])<<(8*i);
        if(effect_len!=intents->effect_length_bytes || effect_len>input.receipt_len-169)break;
        const uint64_t payload_length_offset=97+effect_len;
        uint64_t output_len=0;
        for(uint32_t i=0;i<8;++i)output_len|=uint64_t(receipt[payload_length_offset+i])<<(8*i);
        if(output_len>input.receipt_len-169-effect_len ||
           input.receipt_len!=169+effect_len+output_len)break;
        for(uint64_t i=0;i<effect_len;++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(receipt[97+i]!=payload[i]){exact=false;break;}
        }
        if(!exact)break;
        const IntentEntry* selected=nullptr;
        const auto* entries=reinterpret_cast<const IntentEntry*>(intents+1);
        for(uint64_t i=0;i<intents->count;++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const auto& entry=entries[i];
            if(entry.payload_len!=output_len)continue;
            bool same=true;
            for(uint32_t b=0;b<32;++b) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                same &= receipt[25+b]==reinterpret_cast<const uint8_t*>(entry.stable_identity)[b];
            }
            if(same){selected=&entry;break;}
        }
        if(!selected || output_len>intents->payload_used_bytes ||
           selected->payload_offset>intents->payload_used_bytes-output_len)break;
        for(uint64_t i=0;i<output_len;++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(receipt[payload_length_offset+8+i]!=payload[selected->payload_offset+i]){exact=false;break;}
        }
        if(!exact)break;
        const uint64_t outcome_bytes=replay ? replay->receiver_outcome_bytes : 0;
        uint64_t joint_bytes=0,retained_bytes=0;
        if(old_queue->count>old_queue->payload_used_bytes/sizeof(AcknowledgementEntry) ||
           !semantic_graph::checked_add(input.receipt_len,outcome_bytes,&joint_bytes))break;
        retained_bytes=old_queue->payload_used_bytes-old_queue->count*sizeof(AcknowledgementEntry);
        if(retained_bytes>old_queue->payload_capacity_bytes ||
           joint_bytes>old_queue->payload_capacity_bytes)break;
        semantic_task_ground::Task result_task{};
        if(replay) {
            if(!semantic_task_ground::decode(reinterpret_cast<const uint64_t*>(descriptor.task),
                    descriptor.task_words,&result_task) || !result_task.coding ||
               !result_task.bindings || (replay->receiver_disposition!=1 && replay->receiver_disposition!=2) ||
               replay->receiver_outcome_bytes<265 ||
               !publication_pointer_span(replay->receiver_outcome,replay->receiver_outcome_bytes,1))break;
            const bool completed=replay->receiver_disposition==1;
            uint64_t expected_observations=0,batches=0;
            if((!completed && replay->row_count) ||
               !semantic_graph::checked_add(replay->row_count,1,&batches) ||
               (completed && !semantic_graph::checked_mul(batches,result_task.bindings,&expected_observations)) ||
               replay->observation_count!=expected_observations ||
               replay->observation_count>UINT64_MAX/sizeof(semantic_task_ground::Observation) ||
               replay->observation_count>UINT64_MAX/sizeof(semantic_graph::DecodedStatement) ||
               replay->observation_count>UINT64_MAX/sizeof(semantic_graph::DecodedSupport) ||
               (replay->observation_count && (
                !publication_pointer_span(replay->observations,replay->observation_count*sizeof(semantic_task_ground::Observation),alignof(semantic_task_ground::Observation)) ||
                !publication_pointer_span(replay->statements,replay->observation_count*sizeof(semantic_graph::DecodedStatement),alignof(semantic_graph::DecodedStatement)) ||
                !publication_pointer_span(replay->supports,replay->observation_count*sizeof(semantic_graph::DecodedSupport),alignof(semantic_graph::DecodedSupport)))))break;
            const auto* terminal=reinterpret_cast<const uint8_t*>(replay->receiver_outcome);
            const char outcome_domain[]="xlog.receiver.outcome.v1";
            for(uint32_t index=0;index<sizeof(outcome_domain);++index) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(terminal[index]!=uint8_t(outcome_domain[index]))exact=false;
            }
            uint64_t terminal_status=0,content_bytes=0;
            for(uint32_t index=0;index<8;++index) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,2);
                terminal_status|=uint64_t(terminal[217+index])<<(8*index);
                content_bytes|=uint64_t(terminal[225+index])<<(8*index);
            }
            if(!exact || terminal_status!=replay->receiver_disposition ||
               content_bytes!=replay->receiver_outcome_bytes-265 || !content_bytes ||
               (!completed && content_bytes>4096))break;
            const uint64_t* expected_ids[]={selected->stable_identity,input.recipient_id,selected->effect_digest,
                selected->payload_digest,input.receipt_digest};
            for(uint32_t identity=0;identity<5;++identity)for(uint32_t byte=0;byte<32;++byte) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                if(terminal[25+identity*32+byte]!=reinterpret_cast<const uint8_t*>(expected_ids[identity])[byte])exact=false;
            }
            uint8_t checkpoint=0;
            for(uint32_t byte=0;byte<32;++byte) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                checkpoint|=terminal[185+byte];
            }
            uint64_t terminal_digest[4];semantic_graph::sha256(terminal,replay->receiver_outcome_bytes,terminal_digest,cold_work);
            if(!exact || !checkpoint || !publication_identity_equal(terminal_digest,replay->receiver_outcome_digest,cold_work))break;
            const auto* observations=reinterpret_cast<const semantic_task_ground::Observation*>(replay->observations);
            const auto* statements=reinterpret_cast<const semantic_graph::DecodedStatement*>(replay->statements);
            const auto* supports=reinterpret_cast<const semantic_graph::DecodedSupport*>(replay->supports);
            for(uint64_t ordinal=0;ordinal<replay->observation_count && exact;++ordinal) {
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const auto& observation=observations[ordinal];
                const auto& statement=statements[ordinal];const auto& support=supports[ordinal];
                const uint32_t binding=uint32_t(ordinal%result_task.bindings);
                const auto& common=observations[binding];
                if(observation.present!=(ordinal<result_task.bindings ? 2 : 1) ||
                   (observation.truth!=1 && observation.truth!=2) ||
                   (ordinal<result_task.bindings && !publication_identity_equal(observation.action,observation.intent,cold_work)) ||
                   (ordinal>=result_task.bindings && (observation.truth!=common.truth ||
                    !publication_identity_equal(observation.receipt,common.receipt,cold_work))) ||
                   statement.record!=result_task.binding(binding)[1] || support.record!=UINT32_MAX ||
                   support.polarity!=observation.truth ||
                   !publication_identity_equal(observation.intent,selected->stable_identity,cold_work)) { exact=false;break; }
                for(uint32_t field=0;field<10;++field) {
                    semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    if(statement.reconstruction[field])exact=false;
                }
                const uint64_t* identities[]={observation.receipt,observation.action,observation.intent,observation.support};
                const uint32_t* decoded[]={support.provenance_words,support.source_words,support.context_words,support.scope_words};
                for(uint32_t identity=0;identity<4;++identity) {
                    if(!(identities[identity][0]|identities[identity][1]|identities[identity][2]|identities[identity][3]))exact=false;
                    for(uint32_t word=0;word<4;++word) {
                        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                        if(identities[identity][word]!=(uint64_t(decoded[identity][word*2])|
                            (uint64_t(decoded[identity][word*2+1])<<32)))exact=false;
                    }
                }
                const auto* original=result_task.observation(binding);
                for(uint32_t word=0;word<4;++word) {
                    semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
                    if(original[word]!=(uint64_t(statement.identity_words[word*2])|
                        (uint64_t(statement.identity_words[word*2+1])<<32)))exact=false;
                }
            }
            if(!exact)break;
        }
        uint64_t digest[4];semantic_graph::sha256(receipt,input.receipt_len,digest,cold_work);
        if(!publication_identity_equal(digest,input.receipt_digest,cold_work))break;
        const uint8_t* old_bytes=publication_range_bytes(control,*prior_ack);
        uint64_t chain[4]={},receipt_used=0;
        const uint64_t acknowledged=publication_acknowledgement_chain(old_bytes,prior_ack->length_bytes,
            *old_queue,selected,digest,nullptr,&receipt_used,chain,cold_work,replay ? replay->receiver_outcome_digest : nullptr);
        if(acknowledged) { status=acknowledged;break; }
        const uint64_t cursor=prior_ack->length_bytes;
        if(old_queue->count==old_queue->capacity ||
           joint_bytes>old_queue->payload_capacity_bytes-receipt_used)break;
        const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
        if(ack_template.storage_slot>=control.storage_count || attempt_template.storage_slot>=control.storage_count ||
           ack_template.storage_slot==prior_ack->storage_slot ||
           ack_template.storage_slot==attempt_template.storage_slot ||
           joint_bytes>UINT64_MAX-sizeof(AcknowledgementEntry) ||
           cursor>storage[ack_template.storage_slot].bytes ||
           sizeof(AcknowledgementEntry)+joint_bytes>storage[ack_template.storage_slot].bytes-cursor)break;
        if(replay && publication_append_replay(control,*base,old,next_ranges,
            append_entries_template,append_payload_template,*replay,*selected,cold_work,false))break;
        for(uint64_t i=0;i<base->header.range_count;++i) {
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::TableSlot,1);
            PublicationRange target=next_ranges[i];
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
            const uint64_t role=old[i].role;
            next_ranges[i]=old[i];
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
            // Keep every unchanged owned mutable range in its original inactive
            // backing. Aliasing the acquired directory would strand the next
            // ordinary continuation or overwrite its still-read-leased source.
            if(publication_mutable_role(role) && role!=1 && role!=2 && role!=31 &&
               !(role>=18 && role<=25) && role!=32 && role!=33 &&
               !(replay && replay->row_count && (role==56 || role==57))) {
                if(publication_preserve_range(control,old,base->header.range_count,old[i],target,cold_work)) { exact=false;break; }
                next_ranges[i]=target;
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
            }
        }
        if(!exact)break;
        auto* next_ack=const_cast<PublicationRange*>(publication_find_range(next_ranges,base->header.range_count,32,0,cold_work));
        auto* next_attempt=const_cast<PublicationRange*>(publication_find_range(next_ranges,base->header.range_count,33,0,cold_work));
        next_ack->storage_slot=ack_template.storage_slot;
        next_attempt->storage_slot=attempt_template.storage_slot;
        next_ack->length_bytes=cursor+sizeof(AcknowledgementEntry)+joint_bytes;
        next_ack->logical_begin=0;next_ack->logical_end=old_queue->count+1;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,5*sizeof(uint64_t));
        auto* next_bytes=publication_range_bytes(control,*next_ack);
        if(!next_bytes)break;
        publication_copy_bytes(next_bytes,old_bytes,cursor,cold_work);
        auto* queue=reinterpret_cast<AcknowledgementQueueHeader*>(next_bytes);
        auto* added=reinterpret_cast<AcknowledgementEntry*>(next_bytes+cursor);
        *added=AcknowledgementEntry{};
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(*added));
        semantic_graph::copy_identity(added->stable_identity,selected->stable_identity,cold_work);
        semantic_graph::copy_identity(added->receipt_digest,digest,cold_work);
        added->receipt_len=input.receipt_len;added->receiver_outcome_len=outcome_bytes;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(added->receiver_outcome_len));
        if(replay)semantic_graph::copy_identity(added->receiver_outcome_digest,replay->receiver_outcome_digest,cold_work);
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(added->receipt_len));
        semantic_graph::copy_identity(added->previous_chain,chain,cold_work);
        publication_fold(chain,32,old_queue->count,added->stable_identity,cold_work);
        publication_fold(chain,32,old_queue->count,added->receipt_digest,cold_work);
        publication_fold(chain,32,old_queue->count,added->receiver_outcome_digest,cold_work);
        semantic_graph::copy_identity(added->chain,chain,cold_work);
        publication_copy_bytes(next_bytes+cursor+sizeof(AcknowledgementEntry),receipt,input.receipt_len,cold_work);
        if(replay)publication_copy_bytes(next_bytes+cursor+sizeof(AcknowledgementEntry)+input.receipt_len,
            reinterpret_cast<const uint8_t*>(replay->receiver_outcome),outcome_bytes,cold_work);
        ++queue->count;queue->payload_used_bytes+=sizeof(AcknowledgementEntry)+joint_bytes;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
        semantic_graph::copy_identity(queue->chain_head,chain,cold_work);
        if(replay && publication_append_replay(control,*base,old,next_ranges,
            append_entries_template,append_payload_template,*replay,*selected,cold_work))break;
        auto& next=*publication_bank(control,word^1);
        if(replay && next.header.abi==1) {
            const semantic_graph::ResidentHandle retired{next.header.semantic_owner,semantic_graph::kRootKind,
                next.header.semantic_slot,next.header.semantic_generation};
            if(!publication_retire_root(descriptor,retired,protected_root,&work)){status=6;break;}
            // The same inactive backing no longer owns a root even if a later
            // pre-publication check refuses this append.
            next.header.abi=0;
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(next.header.abi));
        }
        next=*base;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(next));
        if(replay && replay->receiver_disposition==1) {
            staged_bank=&next;
            auto* ground_range=const_cast<PublicationRange*>(publication_find_range(next_ranges,
                base->header.range_count,semantic_task_ground::kPublicationRole,0,cold_work));
            auto* ground_bytes=ground_range ? publication_range_bytes(control,*ground_range) : nullptr;
            semantic_task_ground::Plane plane{};
            if(!ground_bytes || !semantic_task_ground::decode_plane(reinterpret_cast<uint64_t>(ground_bytes),
                ground_range->length_bytes,result_task,&plane))break;
            const auto* observations=reinterpret_cast<const semantic_task_ground::Observation*>(replay->observations);
            for(uint32_t ordinal=0;ordinal<result_task.bindings;++ordinal) {
                const uint64_t accumulated=plane.observations[ordinal].truth|observations[ordinal].truth;
                plane.observations[ordinal]=observations[ordinal];
                plane.observations[ordinal].truth=accumulated;
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,
                    sizeof(semantic_task_ground::Observation));
            }
            semantic_call(descriptor,semantic_graph::kResidentForkAdmission,&protected_root,nullptr,&candidate,
                nullptr,nullptr,&work);
            candidate_live=candidate.words[41]!=0;
            if(candidate.words[0]!=semantic_graph::kOk)break;
            const auto* statements=reinterpret_cast<const semantic_graph::DecodedStatement*>(replay->statements);
            const auto* supports=reinterpret_cast<const semantic_graph::DecodedSupport*>(replay->supports);
            for(uint64_t ordinal=0;ordinal<replay->observation_count;++ordinal) {
                semantic_call(descriptor,semantic_graph::kResidentInsertSupportAdmission,nullptr,&candidate,&candidate,
                    statements+ordinal,supports+ordinal,&work);
                if(candidate.words[0]!=semantic_graph::kOk)break;
            }
            if(candidate.words[0]!=semantic_graph::kOk)break;
            semantic_call(descriptor,semantic_graph::kResidentSealAdmission,nullptr,&candidate,&sealed,
                nullptr,nullptr,&work);
            if(sealed.words[0]!=semantic_graph::kOk || sealed.words[33]!=semantic_graph::kRootKind)break;
            candidate_live=false;sealed_live=sealed.words[1]!=semantic_graph::kOutcomeUnchanged;
            next.header.semantic_owner=sealed.words[39];next.header.semantic_slot=sealed.words[34];
            next.header.semantic_generation=sealed.words[35];
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
            semantic_graph::copy_identity(next.header.semantic_digest,sealed.words+16,cold_work);
            for(uint32_t extent=0;extent<3;++extent)next.header.semantic_extents[extent]=sealed.words[13+extent];
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(next.header.semantic_extents));
            // Refresh the actual baseline through the same full program and
            // authenticated dual-support engine used by ordinary execution.
            // The other two candidate banks did not execute in this ingress.
            next.state.task_evaluation={};next.state.execution_work=work;
            semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(next.state.task_evaluation)+sizeof(next.state.execution_work));
            for(uint64_t query=0;query<3*uint64_t(result_task.queries);++query) {
                plane.query_rows[query]={};
                semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,
                    sizeof(semantic_task_ground::Query));
            }
            Descriptor result_descriptor=descriptor;
            result_descriptor.task_ground=reinterpret_cast<uint64_t>(ground_bytes);
            result_descriptor.task_ground_bytes=ground_range->length_bytes;
            TaskProgramBank program_bank{};uint32_t derived=0;
            if(!task_program_bank(reinterpret_cast<const uint64_t*>(descriptor.task),descriptor.task_words,&program_bank) ||
               task_program_queries(result_descriptor,reinterpret_cast<const uint64_t*>(descriptor.task),
                    program_bank,0,nullptr,0,0,&next.state,&derived)!=TaskProgramQueryOutcome::Ok ||
               next.state.execution_work.overflow)break;
            work=next.state.execution_work;
        }
        next.header.sealed_epoch=(word>>1)+1;
        next.header.base_word=word;
        next.header.publication_word=(next.header.sealed_epoch<<1)|((word&1)^1);
        --next.header.fuel;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,4*sizeof(uint64_t));
        publication_derive_origin(next,base->header,replay ? PublicationOriginKind::ReplayAcknowledgement :
            PublicationOriginKind::DeliveryAcknowledgement,cold_work);
        if(publication_write_attempt(control,next,*base,5,cold_work) || !next_attempt)break;
        if(!publication_charge_native_work(cold_work,work,charged_events))break;
        // The selector CAS, guaranteed gate retirement and caller's refusal
        // record are the remaining canonical writes on success. Check their
        // combined extent before exposing the successor, without charging an
        // operation that has not yet reached its original write boundary.
        if(cold_work && (cold_work->units>UINT64_MAX-3*sizeof(uint64_t) ||
           cold_work->events[unsigned(semantic_graph::NativeWorkEvent::CanonicalByte)]>UINT64_MAX-3*sizeof(uint64_t))) {
            cold_work->overflow=1;break;
        }
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.word));
        status=publication_compare_exchange(control.word,word,next.header.publication_word) ? 0 : 3;
    } while(false);
    if(status!=0 && sealed_live) {
        const semantic_graph::ResidentHandle refused{sealed.words[39],semantic_graph::kRootKind,
            sealed.words[34],sealed.words[35]};
        if(!publication_retire_root(descriptor,refused,protected_root,&work))status=6;
    } else if(status!=0 && candidate_live) {
        semantic_graph::Receipt cleanup{};
        semantic_call(descriptor,semantic_graph::kResidentDiscardAdmission,nullptr,&candidate,&cleanup,
            nullptr,nullptr,&work);
        if(cleanup.words[0]!=semantic_graph::kOk)status=6;
    }
    if(!publication_charge_native_work(cold_work,work,charged_events))status=6;
    if(status!=0 && staged_bank) {
        staged_bank->header.abi=0;
        semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(staged_bank->header.abi));
    }
    publication_store(control.reader_gate,0);
    semantic_graph::charge_native(cold_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.reader_gate));
    return status;
}
__device__ uint64_t publication_materialize_training(PublicationControl& control,
        const TrainingOriginalMaterializationInput& input,semantic_graph::NativeWorkTally* work) {
    if(!input.lease || !control.contract || !publication_compare_exchange(control.reader_gate,0,1))return 2;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
    uint64_t status=1;
    do {
        const auto& lease=*reinterpret_cast<const PublicationLease*>(input.lease);
        const uint64_t word=publication_load(control.word);
        const auto* base=publication_acquired_bank(control,lease,work);
        const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
        if(!base || word!=lease.word || base->header.terminal!=0 ||
           control.reader_counts[word&1]!=1 || control.reader_counts[(word&1)^1] ||
           (word>>1)==(UINT64_MAX>>1) || !contract.replay_row_capacity ||
           contract.replay_original_slot_boundary>contract.replay_row_capacity ||
           !input.original_count || input.original_count>contract.replay_original_slot_boundary ||
           !(input.original_prefix_identity[0]|input.original_prefix_identity[1]|input.original_prefix_identity[2]|input.original_prefix_identity[3]) ||
           publication_validate_selected_model(control,*base,work))break;
        const auto* old=reinterpret_cast<const PublicationRange*>(control.directories[word&1]);
        auto* next_ranges=reinterpret_cast<PublicationRange*>(control.directories[(word&1)^1]);
        const auto* old_entries=publication_find_range(old,base->header.range_count,56,0,work);
        const auto* target=publication_find_range(next_ranges,base->header.range_count,56,0,work);
        if(!old_entries || !target || old_entries->length_bytes!=sizeof(SemanticTrainingReplayAppendHeader) ||
           target->storage_slot==old_entries->storage_slot || target->storage_slot>=control.storage_count)break;
        const auto* bytes=publication_range_bytes(control,*old_entries);
        if(!bytes)break;
        const auto& original=*reinterpret_cast<const SemanticTrainingReplayAppendHeader*>(bytes);
        if(original.abi!=1 || original.original_stage || original.original_count || original.count || original.eligible_count ||
           original.payload_used_bytes || original.capacity!=contract.replay_row_capacity-contract.replay_original_slot_boundary ||
           original.payload_capacity_bytes!=contract.replay_payload_capacity_bytes ||
           (original.original_prefix_identity[0]|original.original_prefix_identity[1]|original.original_prefix_identity[2]|original.original_prefix_identity[3]) ||
           (original.chain_head[0]|original.chain_head[1]|original.chain_head[2]|original.chain_head[3]))break;
        uint64_t total=0;bool valid=true;
        for(uint32_t i=0;i<7;++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(input.lengths[i]>UINT64_MAX-total ||
               (input.lengths[i] && !publication_pointer_span(input.destinations[i],input.lengths[i],8))) {valid=false;break;}
            total+=input.lengths[i];
        }
        if(!valid || total!=input.source_bytes || !publication_pointer_span(input.source,total,8) ||
           contract.replay_row_capacity>UINT64_MAX/sizeof(TrainingViewRowDescriptor) ||
           input.lengths[0]!=contract.replay_row_capacity*sizeof(TrainingViewRowDescriptor) ||
           input.lengths[1]!=contract.replay_raw_capacity_bytes || input.lengths[2]!=sizeof(SemanticTrainingObjectiveRecord) ||
           input.lengths[3]!=8*sizeof(SemanticTrainingObjectiveGroupRecord) ||
           contract.replay_row_capacity>UINT64_MAX/64 || input.lengths[4]!=contract.replay_row_capacity*64 ||
           input.lengths[5]!=5*sizeof(SemanticTrainingCanaryRecord))break;
        const auto* source=reinterpret_cast<const uint8_t*>(input.source);
        const auto& objective=*reinterpret_cast<const SemanticTrainingObjectiveRecord*>(source+input.lengths[0]+input.lengths[1]);
        if(objective.row_count!=input.original_count || objective.group_count!=8 || objective.canary_count!=5 ||
           !objective.capacity || objective.capacity>UINT64_MAX/8 ||
           (input.lengths[6] && input.lengths[6]!=objective.capacity*8) ||
           !publication_identity_equal(objective.task_identity,contract.task_identity,work))break;
        const PublicationRange target_template=*target;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
        valid=true;
        for(uint64_t i=0;i<base->header.range_count;++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const uint64_t role=old[i].role;
            if(!publication_mutable_role(role) || role==56 || role==1 || role==2 || role==31 || (role>=18 && role<=25)) {
                next_ranges[i]=old[i];
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
            } else if(publication_preserve_range(control,old,base->header.range_count,old[i],next_ranges[i],work)) {
                valid=false;break;
            }
        }
        if(!valid)break;
        auto* next_entries=const_cast<PublicationRange*>(publication_find_range(next_ranges,base->header.range_count,56,0,work));
        next_entries->storage_slot=target_template.storage_slot;
        next_entries->generation=target_template.generation;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
        auto* next_bytes=publication_range_bytes(control,*next_entries);
        if(!next_bytes)break;
        uint64_t offset=0;
        for(uint32_t i=0;i<7;++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(input.lengths[i])publication_copy_bytes(reinterpret_cast<uint8_t*>(input.destinations[i]),source+offset,input.lengths[i],work);
            offset+=input.lengths[i];
        }
        auto sealed=original;
        sealed.original_stage=1;sealed.original_count=input.original_count;
        semantic_graph::copy_identity(sealed.original_prefix_identity,input.original_prefix_identity,work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
            2*sizeof(SemanticTrainingReplayAppendHeader)+2*sizeof(uint64_t));
        *reinterpret_cast<SemanticTrainingReplayAppendHeader*>(next_bytes)=sealed;
        auto& next=*publication_bank(control,word^1);next=*base;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationBank)+3*sizeof(uint64_t));
        next.header.sealed_epoch=(word>>1)+1;next.header.base_word=word;
        next.header.publication_word=(next.header.sealed_epoch<<1)|((word&1)^1);
        publication_derive_origin(next,base->header,PublicationOriginKind::TrainingMaterialization,work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
        status=publication_compare_exchange(control.word,word,next.header.publication_word) ? 0 : 3;
    } while(false);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
    publication_store(control.reader_gate,0);
    return status;
}

__device__ bool publication_context_record_equal(const PublicationControl& source,const PublicationBank& source_bank,
        const PublicationControl& target,const PublicationBank& target_bank,uint64_t role,
        semantic_graph::NativeWorkTally* work) {
    const auto* source_ranges=reinterpret_cast<const PublicationRange*>(source.directories[source_bank.header.publication_word&1]);
    const auto* target_ranges=reinterpret_cast<const PublicationRange*>(target.directories[target_bank.header.publication_word&1]);
    const auto* a=publication_find_range(source_ranges,source_bank.header.range_count,role,0,work);
    const auto* b=publication_find_range(target_ranges,target_bank.header.range_count,role,0,work);
    if(!a || !b || a->length_bytes!=b->length_bytes || a->logical_begin!=b->logical_begin || a->logical_end!=b->logical_end)return false;
    const auto* left=publication_range_bytes(source,*a);const auto* right=publication_range_bytes(target,*b);
    if(!left || !right)return false;
    for(uint64_t i=0;i<a->length_bytes;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(left[i]!=right[i])return false;
    }
    return true;
}

__device__ bool publication_context_catalogue_equal(const PublicationControl& source,const PublicationBank& source_bank,
        const PublicationControl& target,const PublicationBank& target_bank,semantic_graph::NativeWorkTally* work) {
    const auto* source_ranges=reinterpret_cast<const PublicationRange*>(source.directories[source_bank.header.publication_word&1]);
    const auto* target_ranges=reinterpret_cast<const PublicationRange*>(target.directories[target_bank.header.publication_word&1]);
    const auto* a=publication_find_range(source_ranges,source_bank.header.range_count,48,0,work);
    const auto* b=publication_find_range(target_ranges,target_bank.header.range_count,48,0,work);
    if(!a || !b || a->length_bytes<42*sizeof(uint64_t) || b->length_bytes<42*sizeof(uint64_t) ||
       a->length_bytes%sizeof(uint64_t) || b->length_bytes%sizeof(uint64_t))return false;
    const auto* left=reinterpret_cast<const uint64_t*>(publication_range_bytes(source,*a));
    const auto* right=reinterpret_cast<const uint64_t*>(publication_range_bytes(target,*b));
    if(!left || !right)return false;
    // Both members share the frozen model rows and original frame; their
    // context resolution digests, local slices and task rights are distinct.
    const uint64_t shared[]={1,3,5,7,25,26,27,28,37,40,41};
    for(uint32_t i=0;i<sizeof(shared)/sizeof(shared[0]);++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(left[shared[i]]!=right[shared[i]])return false;
    }
    if(!left[41] || left[40]>a->length_bytes || left[41]>a->length_bytes-left[40] ||
       right[40]>b->length_bytes || right[41]>b->length_bytes-right[40])return false;
    const auto* source_frame=reinterpret_cast<const uint8_t*>(left)+left[40];
    const auto* target_frame=reinterpret_cast<const uint8_t*>(right)+right[40];
    for(uint64_t i=0;i<left[41];++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(source_frame[i]!=target_frame[i])return false;
    }
    return true;
}

__device__ uint64_t publication_initialize_model_context_continuation(const Descriptor& descriptor,PublicationControl& control) {
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(descriptor.publication.native_work);
    const uint64_t address=descriptor.publication.lease;
    if(!publication_pointer_span(descriptor.publication.native_work,sizeof(semantic_graph::NativeWorkTally),alignof(semantic_graph::NativeWorkTally)) ||
       !publication_pointer_span(address,sizeof(PreparedModelContextContinuationDevice),alignof(PreparedModelContextContinuationDevice)))return 1;
    auto& proof=*reinterpret_cast<PreparedModelContextContinuationDevice*>(address);
    if(proof.abi!=1 || proof.status!=1)return 1;
    // Once entered, an unknown owner must recover this original command rather
    // than submit another initialization or rebuild a captured continuation.
    proof.status=semantic_graph::written_word(2,work);
    if(proof.target_control!=descriptor.publication.control || proof.source_control==proof.target_control ||
       !publication_pointer_span(proof.source_control,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(proof.source_lease,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !proof.minimum_generation || proof.minimum_generation>UINT32_MAX || proof.proposal>=UINT32_MAX ||
       proof.stream_serial>=(1ULL<<56) || proof.family_id>UINT8_MAX || !descriptor.state)return 1;
    auto& source=*reinterpret_cast<PublicationControl*>(proof.source_control);
    if(!publication_compare_exchange(source.reader_gate,0,1))return 2;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(source.reader_gate));
    uint64_t status=1;PublicationBank* initialized=nullptr;
    do {
        const auto& lease=*reinterpret_cast<const PublicationLease*>(proof.source_lease);
        const auto* acquired=publication_acquired_bank(source,lease,work);
        if(!acquired || source.word!=lease.word || source.reader_counts[lease.bank]!=1 || source.reader_counts[lease.bank^1] ||
           !publication_identity_equal(source.instance,proof.source_instance,work) ||
           !publication_identity_equal(control.instance,proof.target_instance,work) ||
           acquired->header.terminal!=2 || acquired->header.model_generation!=proof.minimum_generation ||
           acquired->header.proposal!=proof.proposal || acquired->header.stream_serial!=proof.stream_serial ||
           acquired->header.family_id!=proof.family_id ||
           !publication_identity_equal(acquired->header.model_geometry_digest,proof.model_geometry_digest,work) ||
           publication_validate_selected_model(source,*acquired,work) || publication_validate_storage(source,work) ||
           !publication_pointer_span(source.continuation,sizeof(PendingContinuation),alignof(PendingContinuation)) ||
           !publication_pointer_span(control.continuation,sizeof(PendingContinuation),alignof(PendingContinuation)))break;
        const auto& source_pending=*reinterpret_cast<const PendingContinuation*>(source.continuation);
        const auto& target_pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
        // Published continuation metadata is historical; only an unconsumed
        // current or future base can still authorize another transition.
        if((source_pending.abi!=0 && (source_pending.abi!=1 || source_pending.base_word>=source.word ||
            !publication_identity_equal(source_pending.instance,source.instance,work))) || target_pending.abi ||
           control.abi!=1 || control.word || control.reader_counts[0] || control.reader_counts[1] ||
           !control.banks[0] || !control.banks[1] || !control.contract || !source.contract ||
           control.contract==source.contract || control.continuation==source.continuation ||
           control.storage==source.storage || publication_validate_storage(control,work))break;
        const auto& source_contract=*reinterpret_cast<const PublicationContract*>(source.contract);
        const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
        if(contract.abi!=1 || source_contract.abi!=1 || contract.model_generation!=proof.minimum_generation ||
           source_contract.model_generation!=proof.minimum_generation ||
           contract.policy_generation!=source_contract.policy_generation ||
           contract.authority_generation!=source_contract.authority_generation ||
           !publication_identity_equal(contract.topology_identity,source_contract.topology_identity,work) ||
           !publication_identity_equal(contract.table_identity,source_contract.table_identity,work))break;
        const auto* source_ranges=reinterpret_cast<const PublicationRange*>(source.directories[lease.bank]);
        const auto* intents_range=publication_find_range(source_ranges,acquired->header.range_count,30,0,work);
        const auto* ack_range=publication_find_range(source_ranges,acquired->header.range_count,32,0,work);
        if(!intents_range || !ack_range || ack_range->length_bytes<sizeof(AcknowledgementQueueHeader) ||
           publication_validate_intents(source,source_ranges,acquired->header.range_count,UINT64_MAX,work))break;
        const auto* intents=reinterpret_cast<const IntentQueueHeader*>(publication_range_bytes(source,*intents_range));
        const auto* ack_bytes=publication_range_bytes(source,*ack_range);uint64_t chain[4],receipt_used=0;
        if(!intents || !ack_bytes ||
           publication_acknowledgement_chain(ack_bytes,ack_range->length_bytes,
                *reinterpret_cast<const AcknowledgementQueueHeader*>(ack_bytes),nullptr,nullptr,intents,&receipt_used,chain,work))break;
        const auto* source_storage=reinterpret_cast<const PublicationStorageEntry*>(source.storage);
        const auto* target_storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
        bool valid=true;
        for(uint32_t index=0;index<2 && valid;++index) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            auto& bank=*publication_bank(control,index);
            if(bank.header.abi || control.banks[index]==source.banks[0] || control.banks[index]==source.banks[1] ||
               !control.directories[index] || control.directories[index]==source.directories[0] ||
               control.directories[index]==source.directories[1] ||
               publication_validate_directory(control,contract,reinterpret_cast<PublicationRange*>(control.directories[index]),
                    bank.header.range_count,work)) { valid=false;break; }
            const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[index]);
            for(uint64_t i=0;i<bank.header.range_count && valid;++i) {
                semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                const auto& range=ranges[i];
                if(range.role>=18 && range.role<=25) {
                    if(index)continue;
                    const auto* original=publication_find_range(source_ranges,acquired->header.range_count,range.role,range.index,work);
                    const auto* a=original ? &source_storage[original->storage_slot] : nullptr;
                    const auto& b=target_storage[range.storage_slot];
                    if(!a || a->pointer!=b.pointer || a->bytes!=b.bytes || a->generation!=b.generation ||
                       original->offset_bytes!=range.offset_bytes || original->length_bytes!=range.length_bytes ||
                       original->logical_begin!=range.logical_begin || original->logical_end!=range.logical_end) { valid=false;break; }
                    const auto* a_layout=publication_find_layout(source,source_ranges,acquired->header.range_count,range.role,range.index,work);
                    const auto* b_layout=publication_find_layout(control,ranges,bank.header.range_count,range.role,range.index,work);
                    if(!a_layout || !b_layout) { valid=false;break; }
                    const auto* left=reinterpret_cast<const uint64_t*>(a_layout);
                    const auto* right=reinterpret_cast<const uint64_t*>(b_layout);
                    for(uint64_t word=0;word<sizeof(PublicationTensorLayout)/sizeof(uint64_t);++word) {
                        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                        if(left[word]!=right[word]) { valid=false;break; }
                    }
                } else if(publication_mutable_role(range.role) || range.role==17 || range.role==semantic_task_ground::kPublicationRole) {
                    const auto& backing=target_storage[range.storage_slot];
                    for(uint64_t j=0;j<source.storage_count;++j) {
                        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
                        if(backing.pointer<source_storage[j].pointer+source_storage[j].bytes &&
                           source_storage[j].pointer<backing.pointer+backing.bytes) { valid=false;break; }
                    }
                }
            }
        }
        if(!valid)break;
        initialized=publication_bank(control,0);
        const uint64_t records[]={42,45,46,47,50};
        for(uint32_t i=0;i<sizeof(records)/sizeof(records[0]);++i) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(!publication_context_record_equal(source,*acquired,control,*initialized,records[i],work)) { valid=false;break; }
        }
        if(!valid || !publication_context_catalogue_equal(source,*acquired,control,*initialized,work))break;
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[0]);
        const auto* target_intent=publication_find_range(ranges,initialized->header.range_count,30,0,work);
        const auto* target_ack=publication_find_range(ranges,initialized->header.range_count,32,0,work);
        const auto* queue=target_intent ? reinterpret_cast<const IntentQueueHeader*>(publication_range_bytes(control,*target_intent)) : nullptr;
        const auto* target_ack_bytes=target_ack ? publication_range_bytes(control,*target_ack) : nullptr;
        const auto* acknowledgements=reinterpret_cast<const AcknowledgementQueueHeader*>(target_ack_bytes);
        if(!queue || queue->count || !acknowledgements || acknowledgements->abi!=1 || acknowledgements->count ||
           target_ack->length_bytes!=sizeof(AcknowledgementQueueHeader) || acknowledgements->payload_used_bytes ||
           publication_acknowledgement_chain(target_ack_bytes,target_ack->length_bytes,*acknowledgements,
                nullptr,nullptr,nullptr,&receipt_used,chain,work))break;
        for(uint32_t index=0;index<2;++index) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            auto& bank=*publication_bank(control,index);
            bank.header.model_generation=semantic_graph::written_word(acquired->header.model_generation,work);
            bank.header.neural_generation=semantic_graph::written_word(acquired->header.neural_generation,work);
            bank.header.proposal=semantic_graph::written_word(acquired->header.proposal,work);
            bank.header.stream_serial=semantic_graph::written_word(acquired->header.stream_serial,work);
            bank.header.family_id=semantic_graph::written_word(acquired->header.family_id,work);
            bank.header.training_cursor=semantic_graph::written_word(acquired->header.training_cursor,work);
            semantic_graph::copy_identity(bank.header.training_rng,acquired->header.training_rng,work);
            bank.state.model_generation=uint32_t(acquired->header.model_generation);
            bank.state.family_id=uint32_t(acquired->header.family_id);
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint32_t));
            bank.state.stream_serial=semantic_graph::written_word(acquired->header.stream_serial,work);
            bank.state.proposal=semantic_graph::written_word(acquired->header.proposal,work);
            bank.state.next_proposal=semantic_graph::written_word(acquired->header.proposal,work);
        }
        *reinterpret_cast<State*>(descriptor.state)=initialized->state;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(State));
        status=publication_initialize(descriptor,control,1,work);
        if(status)break;
        // The held gate keeps this image private through the final activation.
        initialized->header.abi=semantic_graph::written_word(0,work);
        if(!publication_identity_equal(initialized->header.model_geometry_digest,acquired->header.model_geometry_digest,work)) { status=1;break; }
        auto* coverage_range=const_cast<PublicationRange*>(publication_find_range(ranges,initialized->header.range_count,14,0,work));
        auto* coverage=coverage_range ? reinterpret_cast<CompletionCoverage*>(publication_range_bytes(control,*coverage_range)) : nullptr;
        if(!coverage || coverage_range->length_bytes!=sizeof(CompletionCoverage)) { status=1;break; }
        // TARGET owns its cold outputs. Its captured first Recompute must
        // authenticate them; SOURCE cache coverage never enters this context.
        coverage->model_generation=semantic_graph::written_word(0,work);
        initialized->header.abi=semantic_graph::written_word(1,work);
        publication_derive_origin(*initialized,acquired->header,PublicationOriginKind::ModelContext,work);
        if(publication_record_identity(control,*coverage_range,coverage->receipt_digest,work)) { status=1;break; }
        initialized->header.abi=semantic_graph::written_word(0,work);
        proof.acquired_generation=semantic_graph::written_word(acquired->header.model_generation,work);
        proof.source_word=semantic_graph::written_word(lease.word,work);
        proof.target_word=semantic_graph::written_word(initialized->header.publication_word,work);
        proof.training_cursor=semantic_graph::written_word(acquired->header.training_cursor,work);
        semantic_graph::copy_identity(proof.training_rng,acquired->header.training_rng,work);
        // Activation, status and both gate retirements plus the dispatcher's
        // refusal word are the only remaining canonical writes on success.
        const uint64_t final_bytes=5*sizeof(uint64_t);
        if(work->overflow || work->units>UINT64_MAX-final_bytes ||
           work->events[unsigned(semantic_graph::NativeWorkEvent::CanonicalByte)]>UINT64_MAX-final_bytes) { work->overflow=1;status=1;break; }
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(initialized->header.abi));
        if(!publication_compare_exchange(initialized->header.abi,0,1)) { status=3;break; }
        proof.status=semantic_graph::written_word(0,work);
        status=0;
    } while(false);
    if(status && initialized)initialized->header.abi=semantic_graph::written_word(0,work);
    publication_store(source.reader_gate,0);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(source.reader_gate));
    return status;
}

__device__ uint64_t publication_initialize_actor_refresh(const Descriptor& descriptor,PublicationControl& control) {
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(descriptor.publication.native_work);
    if(!publication_pointer_span(descriptor.publication.native_work,sizeof(semantic_graph::NativeWorkTally),alignof(semantic_graph::NativeWorkTally)))return 1;
    const uint64_t address=descriptor.publication.lease;
    if(!publication_pointer_span(address,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice)))return 1;
    auto& proof=*reinterpret_cast<PreparedActorRefreshDevice*>(address);
    if(proof.abi!=1 || proof.status!=1 || proof.role>1 ||
       (proof.role==0 ? proof.demonstrated_lane!=0 :
        (proof.demonstrated_lane<1 || proof.demonstrated_lane>2)))return 1;
    // Entered refusal is not interchangeable with the initial never-entered
    // marker. No failure below permits a second initialization or a new draw.
    proof.status=semantic_graph::written_word(2,work);
    if(proof.child_control!=descriptor.publication.control || proof.target_bank>1 ||
       !publication_pointer_span(proof.target_control,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(proof.target_lease,sizeof(PublicationLease),alignof(PublicationLease)) ||
       !publication_pointer_span(proof.target_header,sizeof(PublicationHeader),alignof(PublicationHeader)) ||
       !proof.minimum_generation || proof.minimum_generation>UINT32_MAX ||
       proof.stream_serial>=(1ULL<<56) || proof.family_id>UINT8_MAX || proof.proposal>=UINT32_MAX ||
       !(proof.program[0]|proof.program[1]|proof.program[2]|proof.program[3]) ||
       !(proof.phase[0]|proof.phase[1]|proof.phase[2]|proof.phase[3]))return 1;
    const auto& target_control=*reinterpret_cast<const PublicationControl*>(proof.target_control);
    const auto& target_lease=*reinterpret_cast<const PublicationLease*>(proof.target_lease);
    const auto* target=publication_acquired_bank(target_control,target_lease,work);
    const auto& copied=*reinterpret_cast<const PublicationHeader*>(proof.target_header);
    if(!target || target_control.word!=target_lease.word ||
       target_lease.transition_kind!=4 || target_lease.bank!=proof.target_bank ||
       target->header.model_generation<proof.minimum_generation || target->header.model_generation>UINT32_MAX ||
       !target->header.model_generation || copied.abi!=1 ||
       !publication_header_equal(target->header,copied,work))return 1;
    if(!control.banks[0] || !control.banks[1] || control.word || !descriptor.state)return 1;
    for(uint32_t index=0;index<2;++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        auto& bank=*publication_bank(control,index);
        if(bank.header.abi)return 1;
        bank.header.model_generation=semantic_graph::written_word(target->header.model_generation,work);
        // Both child directory banks retain the exact same CURRENT model
        // owners. Its own canonical initializer starts at neural bank zero;
        // the target's physical publication bank is not a child selector.
        bank.header.neural_bank=semantic_graph::written_word(0,work);
        bank.header.neural_generation=semantic_graph::written_word(target->header.neural_generation,work);
        bank.header.proposal=semantic_graph::written_word(proof.proposal,work);
        bank.header.stream_serial=semantic_graph::written_word(proof.stream_serial,work);
        bank.header.family_id=semantic_graph::written_word(proof.family_id,work);
        bank.state.model_generation=uint32_t(target->header.model_generation);
        bank.state.stream_serial=semantic_graph::written_word(proof.stream_serial,work);
        bank.state.family_id=uint32_t(proof.family_id);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint32_t));
        bank.state.proposal=semantic_graph::written_word(proof.proposal,work);
        bank.state.next_proposal=semantic_graph::written_word(proof.proposal,work);
    }
    auto& initialized=*publication_bank(control,0);
    *reinterpret_cast<State*>(descriptor.state)=initialized.state;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(State));
    semantic_graph::copy_identity(initialized.header.original_instance,control.instance,work);
    publication_derive_origin(initialized,target->header,proof.role==0 ? PublicationOriginKind::ActorRefresh :
        PublicationOriginKind::TrainingDemonstration,work);
    const uint64_t result=publication_initialize(descriptor,control,10,work);
    if(result || work->overflow || publication_model_cache_ready(control,initialized,work) ||
       !publication_identity_equal(initialized.header.model_geometry_digest,target->header.model_geometry_digest,work)) {
        initialized.header.abi=semantic_graph::written_word(0,work);
        return result ? result : 1;
    }
    proof.acquired_generation=semantic_graph::written_word(target->header.model_generation,work);
    semantic_graph::copy_identity(proof.model_geometry_digest,target->header.model_geometry_digest,work);
    if(work->overflow) {
        initialized.header.abi=0;
        return 1;
    }
    proof.status=semantic_graph::written_word(0,work);
    if(work->overflow) {
        proof.status=2;initialized.header.abi=0;
        return 1;
    }
    return 0;
}

// Child work is already produced by the genuine embedded Recompute/Proposal.
// This joins those original certificates; it never reruns a model-work consumer
// or treats a frozen upper bound as actual work. Child candidate attribution is
// local to that child and therefore is shared work in the enclosing Update.
__device__ bool publication_merge_actor_refresh_work(const Descriptor& descriptor,ExecutionWork& target) {
    const uint64_t address=descriptor.publication.native_work;
    if(!address)return true;
    if(!publication_pointer_span(address,sizeof(ActorRefreshWorkInput),alignof(ActorRefreshWorkInput))) {
        semantic_content_integrity_trap();return false;
    }
    const auto& input=*reinterpret_cast<const ActorRefreshWorkInput*>(address);
    uint64_t bytes=0;
    if(input.abi!=1 || !input.count ||
       !semantic_graph::checked_mul(input.count,sizeof(ActorRefreshWorkEntry),&bytes) ||
       !publication_pointer_span(input.entries,bytes,alignof(ActorRefreshWorkEntry))) {
        semantic_content_integrity_trap();return false;
    }
    const auto* entries=reinterpret_cast<const ActorRefreshWorkEntry*>(input.entries);
    bool admissible=true;
    for(uint64_t member=0;member<input.count;++member) {
        const auto& entry=entries[member];
        if(!publication_pointer_span(entry.proof,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice)) ||
           !publication_pointer_span(entry.native_work,sizeof(semantic_graph::NativeWorkTally),alignof(semantic_graph::NativeWorkTally))) {
            semantic_content_integrity_trap();return false;
        }
        const auto& proof=*reinterpret_cast<const PreparedActorRefreshDevice*>(entry.proof);
        const auto& bridge=*reinterpret_cast<const semantic_graph::NativeWorkTally*>(entry.native_work);
        if(proof.abi!=1 || proof.member_ordinal!=entry.member_ordinal ||
           (proof.status!=0 && proof.status!=2)) { semantic_content_integrity_trap();return false; }
        const uint64_t outcome=publication_actor_refresh_outcome(entry);
        if(outcome==2) { semantic_content_integrity_trap();return false; }
        admissible &= outcome==0;
        uint64_t native_units=0;
        for(uint32_t kind=0;kind<9;++kind) {
            if(!semantic_graph::checked_add(native_units,bridge.events[kind],&native_units)) {
                target.overflow=1;return false;
            }
            if(bridge.events[kind]>entry.native_ceiling[kind]) { target.overflow=1;return false; }
        }
        if(native_units!=bridge.units) { semantic_content_integrity_trap();return false; }
        execution_work_merge_native(target,bridge);
        for(uint32_t operation=0;operation<2;++operation) {
            if(!publication_pointer_span(entry.states[operation],sizeof(State),alignof(State))) {
                semantic_content_integrity_trap();return false;
            }
            const auto& child=reinterpret_cast<const State*>(entry.states[operation])->execution_work;
            uint64_t child_native=0,child_raw=0;
            for(uint32_t kind=0;kind<9;++kind) {
                if(!semantic_graph::checked_add(child_native,child.native_events[kind],&child_native)) {
                    target.overflow=1;return false;
                }
                if(child.native_events[kind]>entry.transition_ceilings[operation][kind]) {
                    target.overflow=1;return false;
                }
            }
            if(child.active_candidate || child_native!=child.native_attempt ||
               !semantic_graph::checked_add(child.model_once,child.native_attempt,&child_raw) || child_raw!=child.raw ||
               (child.model_bound && (child.model_bound!=entry.model_work[operation].bound ||
                child.model_events>entry.model_work[operation].count || child.model_once>child.model_bound)) ||
               (!child.model_bound && (child.model_once || child.model_events))) {
                semantic_content_integrity_trap();return false;
            }
            semantic_graph::NativeWorkTally native{};
            for(uint32_t kind=0;kind<9;++kind)
                semantic_graph::charge_native(&native,static_cast<semantic_graph::NativeWorkEvent>(kind),child.native_events[kind]);
            execution_work_merge_native(target,native);
            if(child.model_events>UINT64_MAX-target.model_events || child.model_bound>UINT64_MAX-target.model_bound) {
                target.overflow=1;return false;
            }
            target.model_events+=child.model_events;target.model_bound+=child.model_bound;
            execution_work_add(target,child.model_once,true);
            if(child.overflow)target.overflow=1;
        }
    }
    return admissible;
}
__device__ bool publication_merge_policy_work(const Descriptor& descriptor,ExecutionWork& target) {
    if(!descriptor.policy_work)return true;
    if(!publication_pointer_span(descriptor.policy_work,sizeof(PublicationPolicyWorkInput),
            alignof(PublicationPolicyWorkInput))) { semantic_content_integrity_trap();return false; }
    const auto& input=*reinterpret_cast<const PublicationPolicyWorkInput*>(descriptor.policy_work);
    uint64_t bytes=0;
    if(!input.count || !semantic_graph::checked_mul(input.count,sizeof(PublicationPolicyWorkEntry),&bytes) ||
       !publication_pointer_span(input.entries,bytes,alignof(PublicationPolicyWorkEntry))) {
        semantic_content_integrity_trap();return false;
    }
    const auto* entries=reinterpret_cast<const PublicationPolicyWorkEntry*>(input.entries);
    for(uint64_t index=0;index<input.count;++index) {
        const auto& entry=entries[index];
        if(!publication_pointer_span(entry.tally,sizeof(semantic_graph::NativeWorkTally),
                alignof(semantic_graph::NativeWorkTally))) { semantic_content_integrity_trap();return false; }
        for(uint64_t prior=0;prior<index;++prior)
            if(entries[prior].tally==entry.tally) { semantic_content_integrity_trap();return false; }
        const auto& actual=*reinterpret_cast<const semantic_graph::NativeWorkTally*>(entry.tally);
        if(actual.overflow>1) { semantic_content_integrity_trap();return false; }
        if(actual.overflow) { target.overflow=1;return false; }
        uint64_t units=0;
        for(uint32_t event=0;event<9;++event) {
            if(!semantic_graph::checked_add(units,actual.events[event],&units) ||
               actual.events[event]>entry.ceiling[event]) { target.overflow=1;return false; }
        }
        if(units!=actual.units) { semantic_content_integrity_trap();return false; }
        execution_work_merge_native(target,actual);
    }
    return !target.overflow;
}
__device__ void publication_command(const Descriptor& descriptor) {
    if(!descriptor.publication.control)return;
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    if(descriptor.publication.operation==11 && !publication_pointer_span(descriptor.publication.native_work,
        sizeof(semantic_graph::NativeWorkTally),alignof(semantic_graph::NativeWorkTally))) { control.refusal=1;return; }
    auto* work=descriptor.publication.operation==2 || descriptor.publication.operation==3 || descriptor.publication.operation==10 || descriptor.publication.operation==11 ?
        reinterpret_cast<semantic_graph::NativeWorkTally*>(descriptor.publication.native_work) : nullptr;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    uint64_t status=1;
    if(descriptor.publication.operation==1 || descriptor.publication.operation==4 ||
       descriptor.publication.operation==5 || descriptor.publication.operation==7) {
        if(!publication_compare_exchange(control.reader_gate,0,1))status=2;
        else {status=publication_initialize(descriptor,control,descriptor.publication.operation);publication_store(control.reader_gate,0);}
    } else if(descriptor.publication.operation==11 && descriptor.publication.lease) {
        if(!publication_compare_exchange(control.reader_gate,0,1)) {
            status=2;
            if(publication_pointer_span(descriptor.publication.lease,sizeof(PreparedModelContextContinuationDevice),alignof(PreparedModelContextContinuationDevice))) {
                auto& proof=*reinterpret_cast<PreparedModelContextContinuationDevice*>(descriptor.publication.lease);
                if(proof.abi==1 && proof.status==1)proof.status=semantic_graph::written_word(2,work);
            }
        } else {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.reader_gate));
            status=publication_initialize_model_context_continuation(descriptor,control);
            publication_store(control.reader_gate,0);
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.reader_gate));
        }
    } else if(descriptor.publication.operation==10 && descriptor.publication.lease) {
        if(!publication_compare_exchange(control.reader_gate,0,1)) {
            status=2;
            if(publication_pointer_span(descriptor.publication.lease,sizeof(PreparedActorRefreshDevice),alignof(PreparedActorRefreshDevice))) {
                auto& proof=*reinterpret_cast<PreparedActorRefreshDevice*>(descriptor.publication.lease);
                if(proof.abi==1 && proof.status==1)proof.status=semantic_graph::written_word(2,work);
            }
        }
        else {
            status=publication_initialize_actor_refresh(descriptor,control);
            publication_store(control.reader_gate,0);
        }
    } else if(descriptor.publication.operation==6 && descriptor.publication.lease) {
        status=publication_acknowledge(descriptor,control,
            *reinterpret_cast<const PublicationDeliveryInput*>(descriptor.publication.lease));
    } else if(descriptor.publication.operation==8 && descriptor.publication.lease) {
        const auto& input=*reinterpret_cast<const PublicationReplayDeliveryInput*>(descriptor.publication.lease);
        status=publication_acknowledge(descriptor,control,input.delivery,&input);
    } else if(descriptor.publication.operation==9 && descriptor.publication.lease) {
        status=publication_materialize_training(control,
            *reinterpret_cast<const TrainingOriginalMaterializationInput*>(descriptor.publication.lease),work);
    } else if(descriptor.publication.lease) {
        auto& lease=*reinterpret_cast<PublicationLease*>(descriptor.publication.lease);
        if(descriptor.publication.operation==2)status=publication_acquire(control,lease,0,UINT64_MAX,work);
        if(descriptor.publication.operation==3)status=publication_release(control,lease,work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
        lease.status=status;
    }
    if(descriptor.publication.operation==8 && descriptor.publication.lease) {
        const auto& replay=*reinterpret_cast<const PublicationReplayDeliveryInput*>(descriptor.publication.lease);
        auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(replay.native_work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(control.refusal));
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,8);
    control.refusal=status;
}

extern "C" __global__ void semantic_publication_continue_model_context(Descriptor descriptor,uint64_t frozen_proof_ptr,
        uint64_t source_lease_ptr,uint64_t native_work_ptr) {
    if(blockIdx.x || threadIdx.x)return;
    if(!publication_pointer_span(frozen_proof_ptr,sizeof(PreparedModelContextContinuationDevice),alignof(PreparedModelContextContinuationDevice)) ||
       !publication_pointer_span(native_work_ptr,sizeof(semantic_graph::NativeWorkTally),alignof(semantic_graph::NativeWorkTally)))return;
    auto& proof=*reinterpret_cast<PreparedModelContextContinuationDevice*>(frozen_proof_ptr);
    if(proof.abi!=1 || proof.status!=1)return;
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(native_work_ptr);
    if(!publication_pointer_span(proof.target_control,sizeof(PublicationControl),alignof(PublicationControl)) ||
       !publication_pointer_span(source_lease_ptr,sizeof(PublicationLease),alignof(PublicationLease))) {
        proof.status=semantic_graph::written_word(2,work);return;
    }
    proof.source_lease=semantic_graph::written_word(source_lease_ptr,work);
    descriptor.publication={proof.target_control,frozen_proof_ptr,11,native_work_ptr};
    publication_command(descriptor);
}
__device__ uint64_t publication_begin(const Descriptor& descriptor,State* state,PublicationBank** acquired,
        uint64_t* word,uint64_t* end,bool* held,bool* staged) {
    PublicationAttemptWork attempt(state->execution_work);auto* work=&attempt.actual;
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    *held=false;*staged=false;*acquired=nullptr;
    // The caller always stores this admission's returned refusal.
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    if(control.abi!=1 || !control.contract || !control.continuation || !descriptor.task)return 1;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    if(!publication_compare_exchange(control.reader_gate,0,1))return 2;
    // A held gate is released by the original caller on every remaining path.
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    *held=true;*word=publication_load(control.word);
    const uint64_t index=*word&1;
    if(!control.banks[0] || !control.banks[1])return 1;
    PublicationBank& bank=*publication_bank(control,*word);
    PublicationBank& next=*publication_bank(control,*word^1);
    if(!publication_bank_matches(control,bank,*word,work))return 3;
    *acquired=&bank;
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
    const uint64_t eligibility=publication_header_eligibility(control,bank,*word,pending.transition_kind,work);
    if(eligibility)return eligibility;
    if(descriptor.publication.lease) {
        const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
        if(lease.transition_kind && (publication_acquired_bank(control,lease,work)!=&bank ||
           lease.word!=*word || lease.transition_kind!=pending.transition_kind))return 3;
    }
    if(pending.transition_kind==1 && (!descriptor.policy.z || descriptor.text.rows!=pending.text.rows || descriptor.text.count!=pending.text.count ||
       descriptor.text.selected!=pending.text.selected || !validate_text_binding(bank.source,descriptor.text,work)))return 1;
    if(pending.base_word!=*word)return 3;
    if(contract.abi!=1 || bank.header.semantic_owner!=descriptor.arena[1] ||
       publication_validate_selected_model(control,bank,work) || bank.header.authority_generation!=contract.authority_generation ||
       bank.header.model_generation>UINT32_MAX || bank.header.neural_bank>1 || bank.header.proposal>UINT32_MAX ||
       bank.header.stream_serial>(UINT64_MAX>>8) || bank.header.family_id>255 ||
       bank.header.cache_generation==UINT64_MAX || !publication_identity_equal(contract.task_identity,
           reinterpret_cast<const uint64_t*>(descriptor.task)+2,work))return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[index]);
    auto* destinations=reinterpret_cast<PublicationRange*>(control.directories[index^1]);
    if(publication_validate_source(bank,contract,end,work) || publication_source_records(control,bank,work) ||
       publication_validate_directory(control,contract,ranges,bank.header.range_count,work) ||
       publication_validate_directory(control,contract,destinations,bank.header.range_count,work) ||
       publication_validate_destinations(control,ranges,destinations,bank.header.range_count,work) ||
       (pending.transition_kind!=3 &&
        publication_validate_continuation(control,contract,bank,pending,*end,work)))return 1;
    if(bank.header.terminal==1 && *end!=bank.header.terminal_position+1)return 1;
    const auto* source=publication_find_range(ranges,bank.header.range_count,1,0,work);
    const auto* provenance=publication_find_range(ranges,bank.header.range_count,2,0,work);
    const auto* storage=reinterpret_cast<const PublicationStorageEntry*>(control.storage);
    uint64_t masks=0,terminal_bound=0;bool terminal_possible=false;
    const auto* components=reinterpret_cast<const Component*>(descriptor.components);
    const auto* support=reinterpret_cast<const uint8_t*>(descriptor.support);
    const auto* terminals=reinterpret_cast<const uint64_t*>(contract.terminal_tokens);
    for(uint32_t i=0;i<32;++i) {
      semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
      if(pending.transition_kind==1 && text_selected(pending.text,i)) {
        ++masks;
        for(uint32_t lane=0;lane<2;++lane)for(uint64_t t=0;t<contract.terminal_token_count;++t) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            if(terminals[t]<components[lane*68+i].cardinality && support[components[lane*68+i].offset+terminals[t]]) {
                terminal_possible=true;
                if(bank.source[i].logical_position>terminal_bound)terminal_bound=bank.source[i].logical_position;
            }
        }
      }
    }
    if(!source || !provenance || *end>(storage[source->storage_slot].bytes-source->offset_bytes)/sizeof(SourceSlot) ||
       provenance->length_bytes>storage[provenance->storage_slot].bytes-provenance->offset_bytes ||
       masks>(storage[provenance->storage_slot].bytes-provenance->offset_bytes-provenance->length_bytes)/sizeof(TokenProvenanceRecord))return 5;
    uint64_t intent_status=publication_validate_intents(control,ranges,bank.header.range_count,
        bank.header.terminal==1 ? bank.header.terminal_position : (terminal_possible ? terminal_bound : UINT64_MAX),work);
    if(intent_status)return intent_status;
    // The inactive bank and its exclusive semantic root can be reused only after
    // its reader count is zero. Remaining descendants are traced by the owner.
    if(next.header.abi==1 && next.header.semantic_slot!=0 &&
       (next.header.semantic_slot!=bank.header.semantic_slot || next.header.semantic_generation!=bank.header.semantic_generation)) {
        const semantic_graph::ResidentHandle retired{next.header.semantic_owner,semantic_graph::kRootKind,
            next.header.semantic_slot,next.header.semantic_generation};
        const semantic_graph::ResidentHandle protected_root{bank.header.semantic_owner,semantic_graph::kRootKind,
            bank.header.semantic_slot,bank.header.semantic_generation};
        if(!publication_retire_root(descriptor,retired,protected_root,&state->execution_work))return 6;
    }
    next.header.abi=semantic_graph::written_word(0,work);*staged=true;
    return 0;
}
__device__ uint64_t publication_append_provenance(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next,const Receipt* receipts,uint64_t winner,TextBinding text,
        semantic_graph::NativeWorkTally* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    auto* range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,2,0,work));
    const auto* closure=publication_find_range(ranges,next.header.range_count,17,0,work);
    if(!range || !closure || range->length_bytes%sizeof(TokenProvenanceRecord))return 1;
    const uint64_t old_count=range->length_bytes/sizeof(TokenProvenanceRecord);
    uint64_t count=old_count;
    for(uint32_t i=0;i<32;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(winner && text_selected(text,i))++count;
    }
    range->length_bytes=count*sizeof(TokenProvenanceRecord);range->logical_begin=0;range->logical_end=count;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
    auto* records=reinterpret_cast<TokenProvenanceRecord*>(publication_range_bytes(control,*range));
    if(!records)return 5;
    count=old_count;
    for(uint32_t i=0;i<32;++i) {
      semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
      if(winner && text_selected(text,i)) {
        auto& record=records[count];record=TokenProvenanceRecord{};
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(record)+7*sizeof(uint64_t));
        record.source_slot=i;record.logical_position=next.source[i].logical_position;record.token=next.source[i].token;
        record.base_word=base.header.publication_word;record.proposal=next.state.proposal;record.ordinal=(winner-1)*68+i;
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&receipts[record.ordinal]),sizeof(Receipt),record.action_receipt_digest,work);
        record.authority_generation=base.header.authority_generation;
        semantic_graph::copy_identity(record.origin_logical_digest,base.header.logical_digest,work);
        publication_range_origin(base.header,*closure,record.authority_closure_origin,work);
        TokenProvenanceRecord logical=record;logical.base_word=0;
        for(uint32_t w=0;w<4;++w)logical.successor_logical_digest[w]=0;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,32);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(logical)+sizeof(uint64_t));
        if(publication_action_digest(control,next,receipts+record.ordinal,1,logical.action_receipt_digest,work))return 1;
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&logical),sizeof(logical),record.record_digest,work);
        next.source[i].provenance_record=count++;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
      }
    }
    return 0;
}
__device__ uint64_t publication_annotate_provenance(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next,semantic_graph::NativeWorkTally* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    auto* range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,2,0,work));
    const auto* old=publication_find_range(previous,base.header.range_count,2,0,work);
    if(!range || !old || old->length_bytes>range->length_bytes ||
       range->length_bytes%sizeof(TokenProvenanceRecord) || old->length_bytes%sizeof(TokenProvenanceRecord))return 1;
    if(range->length_bytes==old->length_bytes)return 0;
    auto* records=reinterpret_cast<TokenProvenanceRecord*>(publication_range_bytes(control,*range));
    if(!records)return 1;
    for(uint64_t index=old->length_bytes/sizeof(TokenProvenanceRecord);index<range->length_bytes/sizeof(TokenProvenanceRecord);++index) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        auto& record=records[index];
        if(record.successor_logical_digest[0]|record.successor_logical_digest[1]|
           record.successor_logical_digest[2]|record.successor_logical_digest[3])return 1;
        semantic_graph::copy_identity(record.successor_logical_digest,next.header.logical_digest,work);
    }
    return 0;
}
__device__ uint64_t publication_append_prefix_intent(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next,uint64_t begin,uint64_t end,const uint64_t* plan_digest=nullptr,
        semantic_graph::NativeWorkTally* work=nullptr) {
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    auto* entry_range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,30,0,work));
    auto* payload_range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,31,0,work));
    const auto* prefix=publication_find_range(ranges,next.header.range_count,1,0,work);
    const auto* lineage=publication_find_range(ranges,next.header.range_count,40,0,work);
    const uint64_t header_bytes=plan_digest ? 48 : 8;
    if(!entry_range || !payload_range || !prefix || !lineage || begin>end || end>next.header.prefix_extent ||
       end-begin>(UINT64_MAX-header_bytes)/8 || end>prefix->length_bytes/sizeof(SourceSlot))return 1;
    auto* queue=reinterpret_cast<IntentQueueHeader*>(publication_range_bytes(control,*entry_range));
    const uint64_t token_count=end-begin,payload_bytes=header_bytes+token_count*8;
    if(!queue || queue->count>=queue->capacity || payload_bytes>queue->payload_capacity_bytes-queue->payload_used_bytes)return 5;
    const uint64_t payload_offset=queue->payload_used_bytes;
    entry_range->length_bytes=sizeof(IntentQueueHeader)+(queue->count+1)*sizeof(IntentEntry);
    payload_range->length_bytes=payload_offset+payload_bytes;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    if(!publication_range_bytes(control,*entry_range))return 5;
    auto* payload=publication_range_bytes(control,*payload_range);
    const auto* source=reinterpret_cast<const SourceSlot*>(publication_range_bytes(control,*prefix));
    if(!payload || !source)return 1;
    // Authorized effect descriptors have arbitrary byte lengths; the following
    // token payload is explicitly little-endian and need not be aligned.
    uint64_t count_offset=payload_offset;
    if(plan_digest) {
        for(uint32_t w=0;w<4;++w)for(uint32_t b=0;b<8;++b)
            payload[payload_offset+8*w+b]=semantic_graph::written_byte(uint8_t(plan_digest[w]>>(8*b)),work);
        for(uint32_t b=0;b<8;++b)payload[payload_offset+32+b]=semantic_graph::written_byte(uint8_t(begin>>(8*b)),work);
        count_offset+=40;
    }
    for(uint32_t b=0;b<8;++b)payload[count_offset+b]=semantic_graph::written_byte(uint8_t(token_count>>(8*b)),work);
    for(uint64_t i=begin;i<end;++i) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(source[i].kind!=1 || source[i].logical_position!=i || !source[i].committed || !source[i].recomputed)return 1;
        for(uint32_t b=0;b<8;++b)payload[payload_offset+header_bytes+8*(i-begin)+b]=semantic_graph::written_byte(uint8_t(source[i].token>>(8*b)),work);
    }
    auto& entry=reinterpret_cast<IntentEntry*>(queue+1)[queue->count];entry=IntentEntry{};
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(entry));
    publication_range_origin(base.header,*lineage,entry.checkpoint_lineage,work);
    semantic_graph::copy_identity(entry.base_logical,base.header.logical_digest,work);
    semantic_graph::copy_identity(entry.result_logical,next.header.logical_digest,work);
    uint64_t delta[9];delta[0]=0x786c6f6764656c31ULL;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    semantic_graph::copy_identity(delta+1,base.header.logical_digest,work);semantic_graph::copy_identity(delta+5,next.header.logical_digest,work);
    semantic_graph::sha256(reinterpret_cast<const uint8_t*>(delta),sizeof(delta),entry.effective_delta,work);
    semantic_graph::sha256(payload+queue->effect_offset_bytes,queue->effect_length_bytes,entry.effect_digest,work);
    semantic_graph::sha256(payload+payload_offset,payload_bytes,entry.payload_digest,work);
    entry.payload_offset=payload_offset;entry.payload_len=payload_bytes;
    semantic_graph::copy_identity(entry.audit_instance,next.header.instance,work);entry.audit_epoch=next.header.sealed_epoch;
    semantic_graph::copy_identity(entry.previous_chain,queue->chain_head,work);
    publication_intent_identity(entry,entry.stable_identity,work);
    publication_fold(queue->chain_head,30,queue->count,entry.stable_identity,work);
    semantic_graph::copy_identity(entry.chain,queue->chain_head,work);
    ++queue->count;queue->payload_used_bytes+=payload_bytes;
    entry_range->logical_begin=0;entry_range->logical_end=queue->count;
    payload_range->logical_begin=0;payload_range->logical_end=queue->payload_used_bytes;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,9*sizeof(uint64_t));
    return 0;
}
__device__ uint64_t publication_append_final_intent(const PublicationControl& control,const PublicationBank& base,
        PublicationBank& next,semantic_graph::NativeWorkTally* work=nullptr) {
    if(next.header.terminal!=2 || next.header.prefix_extent!=next.header.terminal_position+1)return 1;
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(contract.nonterminal_intent_mode==1) {
        if(next.header.terminal_position<contract.nonterminal_intent_prefix_begin ||
           next.header.terminal_position>contract.nonterminal_intent_prefix_end)return 1;
        return publication_append_prefix_intent(control,base,next,contract.nonterminal_intent_prefix_begin,
            next.header.terminal_position,contract.nonterminal_intent_plan_digest,work);
    }
    return publication_append_prefix_intent(control,base,next,0,next.header.prefix_extent,nullptr,work);
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
__device__ bool task_ground_acquire(const Descriptor& descriptor,const PublicationBank* bank,
        const semantic_task_ground::Task& task,uint64_t transition_kind,State* state) {
    semantic_graph::NativeWorkTally work{};
    if(bank) {
        const auto& control=*reinterpret_cast<const PublicationControl*>(descriptor.publication.control);
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank->header.publication_word&1]);
        const auto* original=publication_find_range(ranges,bank->header.range_count,semantic_task_ground::kPublicationRole,0,&work);
        if(!original || original->length_bytes!=descriptor.task_ground_bytes)return false;
        const auto* bytes=publication_range_bytes(control,*original);
        if(!bytes)return false;
        publication_copy_bytes(reinterpret_cast<uint8_t*>(descriptor.task_ground),bytes,original->length_bytes,&work);
    }
    semantic_task_ground::Plane plane{};
    if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,task,&plane))return false;
    if(transition_kind==1) {
        auto* words=reinterpret_cast<uint64_t*>(plane.query_rows);
        for(uint64_t word=0;word<3*uint64_t(task.queries)*semantic_task_ground::kQueryWords;++word)
            words[word]=semantic_graph::written_word(0,&work);
    }
    execution_work_merge_native(state->execution_work,work);
    return !state->execution_work.overflow;
}

__device__ bool task_ground_accept_rules(const Descriptor& descriptor,const TaskProgramBank& bank,
        const semantic_program::Rule* inserted,uint32_t inserted_count,State* state) {
    if(!bank.editable)return inserted_count==0;
    semantic_task_ground::Plane plane{};
    if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,bank.task,&plane))return false;
    semantic_graph::NativeWorkTally work{};
    semantic_program::Rule additions[2]{};uint32_t count=0;
    for(uint32_t proposed=0;proposed<inserted_count;++proposed) {
        bool duplicate=false;
        for(uint64_t prior=0;prior<uint64_t(bank.program.rule_count)+plane.header[6]+count;++prior) {
            semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const auto rule=prior<bank.program.rule_count ? semantic_program::bank_rule(bank.program,uint32_t(prior)) :
                prior<bank.program.rule_count+plane.header[6] ? plane.selected_rules[prior-bank.program.rule_count] :
                additions[prior-bank.program.rule_count-plane.header[6]];
            const auto* a=reinterpret_cast<const uint32_t*>(&rule);
            const auto* b=reinterpret_cast<const uint32_t*>(&inserted[proposed]);
            bool equal=true;
            for(uint32_t field=0;field<sizeof(semantic_program::Rule)/sizeof(uint32_t);++field)
                equal&=a[field]==b[field];
            duplicate|=equal;
        }
        if(!duplicate)additions[count++]=inserted[proposed];
    }
    if(count>plane.header[7]-plane.header[6])return false;
    for(uint32_t added=0;added<count;++added) {
        publication_copy_bytes(reinterpret_cast<uint8_t*>(&plane.selected_rules[plane.header[6]+added]),
            reinterpret_cast<const uint8_t*>(&additions[added]),sizeof(semantic_program::Rule),&work);
    }
    plane.header[6]=semantic_graph::written_word(plane.header[6]+count,&work);
    execution_work_merge_native(state->execution_work,work);
    return !state->execution_work.overflow;
}

// Only accounting geometry is predicted. No payload is hashed, sealed or
// copied here. These scratch counts never become an observed execution tally.
// The actual suffix must reproduce every event before its one publication CAS.
struct PublicationSealPrediction {
    semantic_graph::NativeWorkTally work{};
    semantic_graph::NativeWorkTally overhead{};
    ExecutionWork& original;
    __device__ explicit PublicationSealPrediction(ExecutionWork& original):original(original) {}
    __device__ void flush() { execution_work_merge_native(original,overhead);overhead={}; }
    __device__ ~PublicationSealPrediction() { flush(); }
    __device__ void geometry(const semantic_graph::NativeWorkTally& reached) {
        for(uint32_t i=0;i<9;++i) {
            semantic_graph::charge_native(&work,static_cast<semantic_graph::NativeWorkEvent>(i),reached.events[i]);
            semantic_graph::charge_native(&overhead,static_cast<semantic_graph::NativeWorkEvent>(i),reached.events[i]);
        }
        if(reached.overflow)work.overflow=overhead.overflow=1;
    }
    __device__ void bytes(uint64_t count) {
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::CanonicalByte,count);
    }
    __device__ void slots(uint64_t count=1) {
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::TableSlot,count);
    }
    __device__ void inspect() {
        slots();semantic_graph::charge_native(&overhead,semantic_graph::NativeWorkEvent::TableSlot,1);
    }
    __device__ bool sha(uint64_t bytes) {
        uint64_t blocks=0;
        if(!semantic_graph::sha256_block_count(bytes,&blocks)) { work.overflow=1;return false; }
        semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::ShaBlock,blocks);
        this->bytes(32);return !work.overflow;
    }
    __device__ bool fold() { bytes(80);return sha(80); }
    __device__ const PublicationRange* find(const PublicationRange* ranges,uint64_t count,uint64_t role) {
        semantic_graph::NativeWorkTally reached{};
        const auto* result=publication_find_range(ranges,count,role,0,&reached);
        geometry(reached);return result;
    }
    __device__ bool codebook(const PublicationRange& range) {
        const uint64_t prefix=sizeof("xlog.semantic.logical-codebooks.v1")+32+8;
        uint64_t length=0;
        if(range.length_bytes%8 || !semantic_graph::checked_add(prefix,range.length_bytes,&length))return false;
        bytes(prefix);return sha(length);
    }
    __device__ bool record(const PublicationControl& control,const PublicationRange& range) {
        if(publication_tensor_role(range.role) || range.length_bytes>UINT64_MAX-128)return false;
        bytes(21*sizeof(uint64_t));
        return sha(128+range.length_bytes);
    }
    __device__ bool action(const PublicationControl& control,const PublicationBank& bank,uint64_t count) {
        const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
        const auto* codebooks=find(ranges,bank.header.range_count,48);uint64_t bytes=0;
        return codebooks && codebook(*codebooks) &&
            semantic_graph::checked_mul(count,sizeof(Receipt),&bytes) && sha(bytes);
    }
    __device__ bool action_batch(const PublicationControl& control,const PublicationBank& bank) {
        bytes(2*sizeof(ActionBatchReceipt)+sizeof(ExecutionWork)+16*sizeof(uint64_t)+32+4*32);
        return action(control,bank,COMPONENT_COUNT) && sha(sizeof(bank.state.semantic_receipts)) &&
            sha(sizeof(ActionBatchReceipt));
    }

};

__device__ uint64_t publication_update_work_record_bytes(bool refused) {
    return 5*2*sizeof(uint64_t)+sizeof(uint64_t)+2*sizeof(SemanticTrainingCanaryRefusalRecord)+
        sizeof(uint64_t)+sizeof(uint8_t)+(refused ? sizeof(SemanticTrainingCanaryRefusalRecord)-sizeof(uint64_t) : 0);
}
__device__ bool publication_update_work_admits(const PendingContinuation& pending,const ExecutionWork& whole,
        semantic_graph::NativeWorkTally* work) {
    if(pending.transition_kind!=4)return true;
    const auto* canaries=reinterpret_cast<const SemanticTrainingCanaryRecord*>(pending.model_update_canaries);
    const auto* results=reinterpret_cast<const SemanticTrainingCanaryResultRecord*>(pending.model_update_canary_results);
    bool admitted=!whole.overflow;
    for(uint32_t item=0;item<5;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& canary=canaries[item];const auto& result=results[item];
        admitted&=!result.reason && whole.raw<=canary.work_limit && result.memory_used<=canary.memory_limit;
        if(item==4 && result.availability==2) {
            const double memory=__ddiv_rn(double(result.memory_used),double(canary.memory_limit));
            const double compute=__ddiv_rn(double(whole.raw),double(canary.work_limit));
            const double measurement=memory>compute ? memory : compute;
            admitted&=isfinite(measurement) && measurement>=__longlong_as_double((long long)canary.lower_bound_bits) &&
                measurement<=__longlong_as_double((long long)canary.upper_bound_bits);
        }
    }
    return admitted;
}
__device__ void publication_write_update_work(const PendingContinuation& pending,const ExecutionWork& whole,
        semantic_graph::NativeWorkTally* work) {
    if(pending.transition_kind!=4)return;
    const auto* canaries=reinterpret_cast<const SemanticTrainingCanaryRecord*>(pending.model_update_canaries);
    auto* results=reinterpret_cast<SemanticTrainingCanaryResultRecord*>(pending.model_update_canary_results);
    SemanticTrainingCanaryRefusalRecord refusal{};refusal.abi=1;
    const SemanticTrainingCanaryResultRecord* winner=nullptr;const SemanticTrainingCanaryRecord* law=nullptr;
    for(uint32_t item=0;item<5;++item) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        const auto& canary=canaries[item];auto& result=results[item];result.work_used=whole.raw;
        uint64_t reason=whole.overflow ? 8 : whole.raw>canary.work_limit ? 4 : 0;
        uint64_t measurement_bits=result.measurement_bits;
        if(item==4 && result.availability==2) {
            const double memory=__ddiv_rn(double(result.memory_used),double(canary.memory_limit));
            const double compute=__ddiv_rn(double(whole.raw),double(canary.work_limit));
            const double value=memory>compute ? memory : compute;
            measurement_bits=(uint64_t)__double_as_longlong(value);
            const uint64_t numerical=!isfinite(value) ? 1 :
                (value<__longlong_as_double((long long)canary.lower_bound_bits) ||
                 value>__longlong_as_double((long long)canary.upper_bound_bits)) ? 2 : 0;
            if(numerical && (!reason || numerical<reason))reason=numerical;
            if(!numerical && result.memory_used>canary.memory_limit && (!reason || 3<reason))reason=3;
        }
        if(item==4)result.measurement_bits=measurement_bits;
        result.reason=reason && (!result.reason || reason<result.reason) ? reason : result.reason;
        if(result.reason && (!winner || result.reason<winner->reason ||
            (result.reason==winner->reason && result.kind<winner->kind))) { winner=&result;law=&canary; }
    }
    if(winner) {
        refusal.reason=winner->reason;refusal.kind=winner->kind;refusal.row_ordinal=winner->row_ordinal;
        refusal.measurement_bits=winner->measurement_bits;refusal.lower_bound_bits=law->lower_bound_bits;
        refusal.upper_bound_bits=law->upper_bound_bits;refusal.memory_used=winner->memory_used;
        refusal.memory_limit=law->memory_limit;refusal.work_used=whole.raw;refusal.work_limit=law->work_limit;
        const auto& selection=*reinterpret_cast<const SemanticTrainingViewSelection*>(pending.training_selection);
        for(uint32_t i=0;i<4;++i) { refusal.identity[i]=winner->identity[i];refusal.selection_identity[i]=selection.identity[i]; }
    }
    *reinterpret_cast<SemanticTrainingCanaryRefusalRecord*>(pending.model_update_refusal)=refusal;
    auto* admissible=reinterpret_cast<uint8_t*>(pending.model_update_admissibility);
    *admissible=uint8_t(*admissible && !winner);
}

__device__ void publication_finish_refused_update(const PendingContinuation& pending,ExecutionWork& execution) {
    if(pending.transition_kind!=4)return;
    semantic_graph::NativeWorkTally actual{},remaining{};
    semantic_graph::charge_native(&remaining,semantic_graph::NativeWorkEvent::TableSlot,10);
    semantic_graph::charge_native(&remaining,semantic_graph::NativeWorkEvent::CanonicalByte,publication_update_work_record_bytes(false));
    auto decision=execution;execution_work_merge_native(decision,remaining);
    const bool refused=!publication_update_work_admits(pending,decision,&actual);
    semantic_graph::charge_native(&actual,semantic_graph::NativeWorkEvent::CanonicalByte,publication_update_work_record_bytes(refused));
    remaining={};semantic_graph::charge_native(&remaining,semantic_graph::NativeWorkEvent::TableSlot,5);
    auto reached=execution;execution_work_merge_native(reached,actual);execution_work_merge_native(reached,remaining);
    publication_write_update_work(pending,reached,&actual);
    execution_work_merge_native(execution,actual);
}

__device__ bool publication_predict_provenance_annotation(const PublicationControl& control,const PublicationBank& base,
        const PublicationBank& bank,PublicationSealPrediction& cost) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* range=cost.find(ranges,bank.header.range_count,2);
    const auto* old=cost.find(previous,base.header.range_count,2);
    if(!range || !old || old->length_bytes>range->length_bytes ||
       range->length_bytes%sizeof(TokenProvenanceRecord) || old->length_bytes%sizeof(TokenProvenanceRecord))return false;
    const uint64_t count=(range->length_bytes-old->length_bytes)/sizeof(TokenProvenanceRecord);
    if(!count)return true;
    cost.slots(count);cost.bytes(count*32);
    return !cost.work.overflow;
}
struct PublicationIntentPrediction {
    bool append=false,planned=false;
    uint64_t begin=0,end=0,payload_bytes=0;
    PublicationRange entries{},payload{};
    __device__ PublicationRange after(const PublicationRange& range,semantic_graph::NativeWorkTally* overhead) const {
        semantic_graph::charge_native(overhead,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(PublicationRange));
        return append && range.role==30 ? entries : append && range.role==31 ? payload : range;
    }
};

__device__ bool publication_predict_intent(const PublicationControl& control,
        const PublicationBank& bank,bool drain,PublicationSealPrediction& cost,PublicationIntentPrediction& result) {
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    result.append=drain;
    if(!result.append)return true;
    result.planned=contract.nonterminal_intent_mode==1;
    result.begin=result.planned ? contract.nonterminal_intent_prefix_begin : 0;
    result.end=result.planned ? bank.header.terminal_position : bank.header.prefix_extent;
    if(result.planned && result.end>contract.nonterminal_intent_prefix_end)return false;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* entries=cost.find(ranges,bank.header.range_count,30);
    const auto* payload=cost.find(ranges,bank.header.range_count,31);
    const auto* prefix=cost.find(ranges,bank.header.range_count,1);
    const auto* lineage=cost.find(ranges,bank.header.range_count,40);
    if(!entries || !payload || !prefix || !lineage || result.begin>result.end)return false;
    const auto* queue=reinterpret_cast<const IntentQueueHeader*>(publication_range_bytes(control,*entries));
    uint64_t token_bytes=0,count=0;
    if(!queue || !semantic_graph::checked_mul(result.end-result.begin,8,&token_bytes) ||
       !semantic_graph::checked_add(result.planned ? 48 : 8,token_bytes,&result.payload_bytes) ||
       !semantic_graph::checked_add(queue->count,1,&count))return false;
    result.entries=*entries;result.payload=*payload;
    semantic_graph::charge_native(&cost.overhead,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(PublicationRange));
    if(!semantic_graph::checked_mul(count,sizeof(IntentEntry),&result.entries.length_bytes) ||
       !semantic_graph::checked_add(sizeof(IntentQueueHeader),result.entries.length_bytes,&result.entries.length_bytes) ||
       !semantic_graph::checked_add(queue->payload_used_bytes,result.payload_bytes,&result.payload.length_bytes))return false;
    result.entries.logical_begin=result.payload.logical_begin=0;
    result.entries.logical_end=count;result.payload.logical_end=result.payload.length_bytes;
    cost.slots(result.end-result.begin);
    cost.bytes(2*sizeof(uint64_t));
    cost.bytes(result.payload_bytes);cost.bytes(sizeof(IntentEntry)+8*32+10*sizeof(uint64_t)+64);
    if(!cost.sha(64))return false;
    if(!cost.sha(9*sizeof(uint64_t)) || !cost.sha(queue->effect_length_bytes) || !cost.sha(result.payload_bytes))return false;
    cost.bytes(25*sizeof(uint64_t));
    return cost.sha(25*sizeof(uint64_t)) && cost.fold();
}

__device__ bool publication_predict_attempt(const PublicationControl& control,const PublicationBank& base,
        const PublicationBank& bank,uint64_t kind,const PublicationIntentPrediction& intent,PublicationSealPrediction& cost) {
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[bank.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[base.header.publication_word&1]);
    const auto* target=cost.find(ranges,bank.header.range_count,33);
    if(!target)return false;
    cost.bytes(sizeof(AttemptReceipt)+4*sizeof(uint64_t)+2*32);
    if(kind!=1) { if(!cost.action(control,bank,0))return false; }
    else { if(!cost.sha(sizeof(ActionBatchReceipt)))return false;cost.slots(4);cost.bytes(32); }
    if(!cost.sha(sizeof(bank.state.semantic_receipts)))return false;
    const uint64_t roles[]={14,27,30,32,33};
    for(uint32_t i=0;i<5;++i) {
        const auto* item=cost.find(i==4 ? previous : ranges,bank.header.range_count,roles[i]);
        if(!item)return false;
        if(i==4) { if(!cost.record(control,*item))return false; }
        else if(!cost.record(control,intent.after(*item,&cost.overhead)))return false;
    }
    return cost.record(control,*target);
}


__device__ uint64_t publication_publish(const Descriptor& descriptor,PublicationBank& base,uint64_t word,uint64_t end,
        const semantic_graph::ResidentHandle& base_root,State* state,const Receipt* receipts,bool no_draw=false,bool drain=false) {
    PublicationAttemptWork attempt(state->execution_work);
    auto* work=&attempt.actual;
    // Both callers unconditionally write this publication's returned refusal.
    // Admission already charged its guaranteed release of the original gate.
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    if(state->execution_work.overflow) { state->status=11;return 1; }
    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
    auto& next=*publication_bank(control,word^1);
    const uint64_t winner=no_draw ? 0 : state->task_evaluation.winner;
    if(state->blocks!=(no_draw ? 0 : 136) || winner>2 || publication_prepare_text(base,next,contract,receipts,winner,end,descriptor.text,work))return 1;
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
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,13*sizeof(uint64_t)+32);
    if(pending.transition_kind==4) {
        if(base.header.model_generation>=UINT32_MAX || base.header.neural_bank>1 ||
           base.header.neural_generation==UINT64_MAX || base.header.training_cursor==UINT64_MAX)return 1;
        next.header.model_generation=base.header.model_generation+1;
        next.header.neural_bank=base.header.neural_bank^1;
        next.header.neural_generation=base.header.neural_generation+1;
        next.header.training_cursor=base.header.training_cursor+1;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,4*sizeof(uint64_t));
    }
    if(drain)next.header.terminal=semantic_graph::written_word(2,work);
    next.state=*state;next.state.next_proposal=next.header.proposal;
    for(uint32_t i=0;i<136;++i)next.receipts[i]=no_draw ? base.receipts[i] : receipts[i];
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
        sizeof(State)+sizeof(uint64_t)+sizeof(next.receipts));
    semantic_graph::NativeWorkTally preserved_work{};
    const uint64_t preservation_status=publication_apply_continuation(control,base,next,pending,&preserved_work);
    execution_work_merge_native(state->execution_work,preserved_work);
    if(preservation_status ||
       publication_append_provenance(control,base,next,receipts,winner,descriptor.text,work))return 1;
    auto* ranges=reinterpret_cast<PublicationRange*>(control.directories[next.header.publication_word&1]);
    const auto* previous=reinterpret_cast<const PublicationRange*>(control.directories[word&1]);
    auto* ground=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,semantic_task_ground::kPublicationRole,0,work));
    if(!ground || ground->length_bytes!=descriptor.task_ground_bytes)return 1;
    auto* ground_bytes=publication_range_bytes(control,*ground);if(!ground_bytes)return 1;
    semantic_graph::NativeWorkTally ground_work{};
    publication_copy_bytes(ground_bytes,reinterpret_cast<const uint8_t*>(descriptor.task_ground),ground->length_bytes,&ground_work);
    execution_work_merge_native(state->execution_work,ground_work);
    if(state->execution_work.overflow)return 1;
    auto* intent=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,30,0,work));
    const auto* old_intent=publication_find_range(previous,base.header.range_count,30,0,work);
    if(!intent || !old_intent)return 1;
    intent->length_bytes=old_intent->length_bytes;intent->logical_begin=old_intent->logical_begin;intent->logical_end=old_intent->logical_end;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,3*sizeof(uint64_t));
    uint8_t* intent_bytes=publication_range_bytes(control,*intent);
    if(!intent_bytes)return 1;
    publication_copy_bytes(intent_bytes,publication_range_bytes(control,*old_intent),old_intent->length_bytes,work);
    const auto* prefix=publication_find_range(ranges,next.header.range_count,3,0,work);
    const auto* source=publication_find_range(ranges,next.header.range_count,1,0,work);
    if(!prefix || !source || prefix->length_bytes!=32)return 1;
    if(no_draw) {
        auto* feedback=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,15,0,work));
        const auto* old_feedback=publication_find_range(previous,base.header.range_count,15,0,work);
        if(!feedback || !old_feedback || feedback->length_bytes!=old_feedback->length_bytes)return 1;
        publication_copy_bytes(publication_range_bytes(control,*feedback),publication_range_bytes(control,*old_feedback),feedback->length_bytes,work);
        feedback->logical_begin=old_feedback->logical_begin;feedback->logical_end=old_feedback->logical_end;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    } else if(publication_write_feedback(descriptor,control,next,root,false,&state->execution_work,work))return 1;
    // Derive this operation's origin before the bounded action and provenance
    // receipt suffix; no numerical payload contributes to the origin.
    publication_derive_origin(next,base.header,static_cast<PublicationOriginKind>(pending.transition_kind),work);
    semantic_graph::copy_identity(reinterpret_cast<uint64_t*>(publication_range_bytes(control,*prefix)),next.header.logical_digest,work);
    if(publication_write_coverage(control,contract,next,drain ? nullptr : &pending,work) ||
       publication_finalize_model_origin(control,next,false,work))return 1;
    attempt.flush();
    PublicationSealPrediction prediction(state->execution_work);PublicationIntentPrediction intent_prediction{};
    prediction.bytes(sizeof(ExecutionWork));
    if(!no_draw) {
        if(!prediction.action_batch(control,next))return 1;
        prediction.bytes(sizeof(ActionBatchReceipt)+32);
    }
    const uint64_t replay_bytes=descriptor.replay.successor.rows ?
        replay_copy_extent(control,next,descriptor.replay.successor,&prediction.overhead) : 0;
    if(!publication_predict_provenance_annotation(control,base,next,prediction) ||
       !publication_predict_intent(control,next,drain,prediction,intent_prediction) ||
       !publication_predict_attempt(control,base,next,pending.transition_kind,intent_prediction,prediction))return 1;
    if(!prediction.find(ranges,next.header.range_count,33))return 1;
    if(descriptor.replay.successor.rows) {
        prediction.bytes(replay_bytes);
        uint64_t replay_visits=0;
        if(!semantic_graph::checked_add(next.header.range_count,2,&replay_visits) ||
           !semantic_graph::checked_mul(descriptor.replay.successor.count,replay_visits,&replay_visits))return 1;
        prediction.slots(replay_visits);
    }
    if(pending.transition_kind==4) { prediction.slots(10);prediction.bytes(publication_update_work_record_bytes(false)); }
    prediction.slots(sizeof(semantic_graph::NativeWorkTally)/sizeof(uint64_t));
    prediction.bytes(2*sizeof(uint64_t)); // original next-proposal write and the one selector CAS
    if(prediction.work.overflow)return 1;
    prediction.flush();
    const ExecutionWork completed=attempt.completed(&prediction.work);
    if(completed.overflow)return 1;
    next.state.execution_work=completed;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(ExecutionWork));
    if(!no_draw) {
        if(publication_seal_action_batch(control,base,next,*state,work,&completed))return 1;
        next.state.action_batch=state->action_batch;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(ActionBatchReceipt));
        semantic_graph::copy_identity(next.state.action_batch_root,state->action_batch_root,work);
    }
    if(publication_annotate_provenance(control,base,next,work))return 1;
    if(drain && publication_append_final_intent(control,base,next,work))return 1;
    if(publication_write_attempt(control,next,base,pending.transition_kind,work))return 1;
    auto* attempt_range=const_cast<PublicationRange*>(publication_find_range(ranges,next.header.range_count,33,0,work));
    if(!attempt_range)return 1;
    // No host selector, retry, rebase, or secondary canonical success flag. All
    // winning text, caches, semantics, receipts and RNG become visible together.
    if(state->execution_work.overflow) { state->status=11;return 1; }
    if(descriptor.replay.successor.rows) {
        replay_copy_snapshot(control,next,descriptor.replay.successor,0,0,1,work);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,replay_bytes);
        auto* actual=reinterpret_cast<uint64_t*>(descriptor.replay.successor.actual);
        actual[0]=replay_bytes;actual[1]=next.header.publication_word;actual[2]=1;
    }
    if(!publication_update_work_admits(pending,completed,work)) {
        // The reached refusal writes replace the unexecuted selector tail.
        // The returned record contains actual refusal work, never its prediction.
        next.header.abi=0;state->status=10;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
        return 1;
    }
    if(pending.transition_kind==4) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,publication_update_work_record_bytes(false));
        publication_write_update_work(pending,completed,work);
    }
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,
        sizeof(semantic_graph::NativeWorkTally)/sizeof(uint64_t));
    auto closed_actual=attempt.actual;
    // Only the two straight-line, bounds-checked publication writes remain.
    // They are included in the comparison and every resource limit; only their
    // actual call sites charge the original counter, including a failed CAS.
    semantic_graph::charge_native(&closed_actual,semantic_graph::NativeWorkEvent::CanonicalByte,2*sizeof(uint64_t));
    const auto* reached=reinterpret_cast<const uint64_t*>(&closed_actual);
    const auto* expected=reinterpret_cast<const uint64_t*>(&prediction.work);
    bool exact=true;
    for(uint32_t i=0;i<sizeof(closed_actual)/sizeof(uint64_t);++i)exact&=reached[i]==expected[i];
    if(!exact || attempt.actual.overflow)return 1;
    const uint64_t old_proposal=state->next_proposal;
    state->next_proposal=next.header.proposal;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    if(!publication_compare_exchange(control.word,word,next.header.publication_word)) {
        state->next_proposal=old_proposal;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
        return 3;
    }
    return 0;
}
// Service provenance is separate from the learned feedback coordinates. Rows
// are [original target, original support, support slot, support generation,
// version slot, version generation, insertion ordinal, polarity, digest[4]].
// They are copied from the actual current chain, never from an action roster.
struct FeedbackSupportCollector {
    uint64_t* rows;
    uint64_t capacity, count, refusal, begin=0, channels=3;
    semantic_graph::NativeWorkTally* work=nullptr;
    __device__ bool operator()(uint64_t slot,const uint64_t* version,const uint64_t* support) {
        if(!(channels&support[5]))return true;
        for(uint64_t index=begin;index<count;++index) {
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
            const auto* prior=rows+12*index;
            if(prior[2]==version[5] && prior[3]==support[1] && prior[4]==slot && prior[5]==version[1])return true;
        }
        if(count>=capacity) { refusal=1;return false; }
        if(support[10]==UINT32_MAX) { refusal=2;return false; }
        uint64_t* row=rows+12*count++;
        row[0]=support[10];row[1]=support[11];row[2]=version[5];row[3]=support[1];
        row[4]=slot;row[5]=version[1];row[6]=version[13];row[7]=support[5];
        semantic_graph::copy_identity(row+8,support+6);
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,12*sizeof(uint64_t));
        return true;
    }
};

struct PublicationFeedbackInputs {
    const PublicationBank* bank;
    const RawFeedbackRecord* records;
    const PublicationControl* control;
    const PublicationRange* ranges;
    uint64_t statement_count,statement_capacity;
};

// Resolve every source through the actual acquired reader. The publication
// word may advance while this reader retains its original bank and seals.
__device__ uint64_t publication_feedback_inputs(uint64_t control_ptr,
        const PublicationLease* lease,uint64_t slots,PublicationFeedbackInputs* inputs,
        semantic_graph::NativeWorkTally* work=nullptr) {
    const uint64_t lease_ptr=reinterpret_cast<uint64_t>(lease);
    if(!control_ptr || control_ptr%alignof(PublicationControl) || control_ptr>UINT64_MAX-sizeof(PublicationControl) ||
       !lease_ptr || lease_ptr%alignof(PublicationLease) || lease_ptr>UINT64_MAX-sizeof(PublicationLease) ||
        !slots || slots>UINT64_MAX/sizeof(RawFeedbackRecord))return 1;
    const auto& control=*reinterpret_cast<const PublicationControl*>(control_ptr);
    if(lease->bank>1 || !control.contract || control.contract%alignof(PublicationContract) ||
       control.contract>UINT64_MAX-sizeof(PublicationContract) ||
       !control.banks[lease->bank] || control.banks[lease->bank]%alignof(PublicationBank) ||
       control.banks[lease->bank]>UINT64_MAX-sizeof(PublicationBank) ||
       !control.directories[lease->bank] || control.directories[lease->bank]%alignof(PublicationRange) ||
       !control.storage || control.storage%alignof(PublicationStorageEntry) || !control.storage_count ||
       control.storage_count>(UINT64_MAX-control.storage)/sizeof(PublicationStorageEntry))return 1;
    const auto* bank=publication_acquired_bank(control,*lease,work);
    const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
    if(!bank || contract.abi!=1 || contract.feedback_capacity!=slots ||
       bank->header.range_count>contract.range_capacity ||
       bank->header.range_count>(UINT64_MAX-control.directories[lease->bank])/sizeof(PublicationRange))return 1;
    const auto* ranges=reinterpret_cast<const PublicationRange*>(control.directories[lease->bank]);
    const auto* records=publication_find_range(ranges,bank->header.range_count,15,0,work);
    if(!records || records->length_bytes!=slots*sizeof(RawFeedbackRecord) ||
       records->logical_begin || records->logical_end>slots)return 1;
    const auto* bytes=publication_range_bytes(control,*records);
    if(!bytes || reinterpret_cast<uint64_t>(bytes)%alignof(RawFeedbackRecord))return 1;
    inputs->bank=bank;
    inputs->records=reinterpret_cast<const RawFeedbackRecord*>(bytes);
    inputs->control=&control;inputs->ranges=ranges;
    if(contract.role_count!=semantic_task_ground::kPublicationRole || !contract.role_counts)return 1;
    const auto* roles=reinterpret_cast<const PublicationRoleCount*>(contract.role_counts);
    if(roles[15].role!=16 || !roles[15].count || roles[15].count>UINT32_MAX || roles[15].count>slots)return 1;
    inputs->statement_count=roles[15].count;
    inputs->statement_capacity=0;
    for(uint64_t index=0;index<inputs->statement_count;++index) {
        const auto* statement=publication_find_range(ranges,bank->header.range_count,16,index,work);
        if(!statement || !statement->length_bytes)return 1;
        const auto* data=publication_range_bytes(control,*statement);
        if(!data)return 1;
        if(statement->length_bytes>inputs->statement_capacity)inputs->statement_capacity=statement->length_bytes;
    }
    return 0;
}

__device__ uint64_t publication_feedback_lineage(const Descriptor& descriptor,
        const PublicationLease* lease,uint64_t slots,
        semantic_graph::Receipt* queries,uint64_t* original_statements,uint64_t* offsets,
        uint64_t* supports,uint64_t support_capacity,semantic_graph::NativeWorkTally* work=nullptr) {
    for(uint64_t slot=0;slot<slots;++slot) {
        queries[slot]=semantic_graph::Receipt{};original_statements[slot]=UINT64_MAX;offsets[slot]=0;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
            sizeof(semantic_graph::Receipt)+2*sizeof(uint64_t));
    }
    offsets[slots]=0;
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    for(uint64_t i=0;i<support_capacity*12;++i) {
        supports[i]=0;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    }
    PublicationFeedbackInputs inputs{};
    if(publication_feedback_inputs(descriptor.publication.control,lease,slots,&inputs,work) || !descriptor.task)return 1;
    const auto* task=reinterpret_cast<const uint64_t*>(descriptor.task);
    TaskProgramBank program{};
    if(!task_program_bank(task,descriptor.task_words,&program) || task[1]!=descriptor.arena[1])return 1;
    const auto& view=program.task;
    const auto& bank=*inputs.bank;
    semantic_graph::Layout layout{};
    if(!semantic_graph::build_layout(reinterpret_cast<uint64_t*>(descriptor.arena[0]),
        descriptor.arena[2],descriptor.arena[3],descriptor.arena[4],descriptor.arena[5],
        descriptor.arena[6],&layout) || (slots && layout.support_capacity>UINT64_MAX/slots) ||
        support_capacity!=slots*layout.support_capacity)return 1;
    layout.work=work;
    const semantic_graph::ResidentHandle root{bank.header.semantic_owner,semantic_graph::kRootKind,
        bank.header.semantic_slot,bank.header.semantic_generation};
    semantic_task_ground::Plane plane{};
    semantic_program::DualSupport lineage{};
    semantic_program::Fact* facts=nullptr;
    semantic_program::Result closure{};
    if(program.editable) {
        const auto* ground=publication_find_range(inputs.ranges,bank.header.range_count,semantic_task_ground::kPublicationRole,0,work);
        if(!ground || !semantic_task_ground::decode_plane(
            reinterpret_cast<uint64_t>(publication_range_bytes(*inputs.control,*ground)),ground->length_bytes,view,&plane))return 1;
        closure=task_program_closure(descriptor,program,plane,0,nullptr,0,&lineage,&facts,nullptr,work);
        if(closure.status!=semantic_program::Status::Ok)return 1;
    }
    FeedbackSupportCollector collector{supports,support_capacity,0,0};
    collector.work=work;
    for(uint64_t slot=0;slot<slots;++slot) {
        offsets[slot]=collector.count;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
        const auto& saved=inputs.records[slot];
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(saved.valid==0) {
            offsets[slot+1]=collector.count;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
            continue;
        }
        if(saved.valid!=1 || saved.statement_index>=view.queries || saved.pro>1 || saved.contra>1)return 1;
        collector.begin=collector.count;
        semantic_graph::DecodedStatement statement{};
        statement.record=uint32_t(view.query(uint32_t(saved.statement_index))[4]);
        for(uint32_t i=0;i<8;++i)
            statement.identity_words[i]=uint32_t(view.query(uint32_t(saved.statement_index))[i/2]>>(32*(i%2)));
        auto& query=queries[slot];
        if(program.editable) {
            const auto query_fact=semantic_program::bank_fact(program.program,saved.statement_index);
            const uint32_t fact_index=semantic_program::find_fact(facts,closure.fact_count,query_fact);
            const uint64_t truth=fact_index==closure.fact_count ? 0 : lineage.truth[fact_index];
            if(truth!=(saved.pro|(saved.contra<<1)) || saved.query_receipt.words[1]!=0x5850524f47515532ULL ||
               saved.query_receipt.words[2]!=truth || saved.query_receipt.words[4]!=saved.statement_index)return 1;
            query=saved.query_receipt;
            original_statements[slot]=statement.record;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,
                sizeof(semantic_graph::Receipt)+sizeof(uint64_t));
            for(uint32_t binding=0;binding<view.bindings && fact_index<closure.fact_count;++binding) {
                uint64_t channels=0;
                for(uint32_t channel=0;channel<2;++channel) {
                    if(lineage.lineage[(uint64_t(fact_index)*2+channel)*lineage.words+binding/64] &
                       (uint64_t(1)<<(binding%64)))channels|=uint64_t(1)<<channel;
                }
                if(!channels)continue;
                if(!plane.observations[binding].present)return 1;
                semantic_graph::DecodedStatement protected_statement{};
                protected_statement.record=uint32_t(view.binding(binding)[1]);
                for(uint32_t word=0;word<8;++word)
                    protected_statement.identity_words[word]=uint32_t(view.observation(binding)[word/2]>>(32*(word%2)));
                semantic_graph::Receipt actual{};
                semantic_call(descriptor,semantic_graph::kResidentRootTruthAdmission,&root,nullptr,&actual,&protected_statement,nullptr,nullptr,work);
                if(actual.words[0]!=semantic_graph::kOk || (actual.words[2]&channels)!=channels ||
                   actual.words[39]!=bank.header.semantic_owner || actual.words[33]!=semantic_graph::kRootKind ||
                   actual.words[34]!=bank.header.semantic_slot || actual.words[35]!=bank.header.semantic_generation ||
                   !publication_identity_equal(actual.words+16,bank.header.semantic_digest,work))return 1;
                for(uint32_t word=0;word<3;++word)if(actual.words[13+word]!=bank.header.semantic_extents[word])return 1;
                collector.channels=channels;
                semantic_graph::Receipt checked{};
                if(!actual.words[12] || !semantic_graph::visit_support_chain(layout,actual.words[7],actual.words[11]+1,
                    false,collector,&checked))return collector.refusal ? collector.refusal : 1;
            }
            offsets[slot+1]=collector.count;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
            continue;
        }
        collector.channels=3;
        semantic_call(descriptor,semantic_graph::kResidentRootTruthAdmission,&root,nullptr,&query,&statement,nullptr,nullptr,work);
        if(query.words[0]!=semantic_graph::kOk || query.words[2]!=(saved.pro|(saved.contra<<1)) ||
           query.words[39]!=bank.header.semantic_owner || query.words[33]!=semantic_graph::kRootKind ||
           query.words[34]!=bank.header.semantic_slot || query.words[35]!=bank.header.semantic_generation ||
           !publication_identity_equal(query.words+16,bank.header.semantic_digest,work))return 1;
        for(uint32_t i=0;i<3;++i)if(query.words[13+i]!=bank.header.semantic_extents[i])return 1;
        // Cold restoration deliberately preserves archived runtime handles.
        // Stable semantic fields still have to agree with the fresh query.
        for(uint32_t i=0;i<semantic_graph::kReceiptWords;++i)
            if(!publication_receipt_runtime_word(i) && saved.query_receipt.words[i]!=query.words[i])return 1;
        if((query.words[8]!=0)!=(query.words[12]!=0) ||
           (query.words[2]!=0)!=(query.words[8]!=0))return 1;
        original_statements[slot]=statement.record;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
        if(query.words[12]) {
            semantic_graph::Receipt checked{};
            if(!semantic_graph::visit_support_chain(layout,query.words[7],query.words[11]+1,
                    false,collector,&checked))return collector.refusal ? collector.refusal : 1;
        }
        offsets[slot+1]=collector.count;
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
    }
    return 0;
}

extern "C" __global__ void semantic_feedback_project(Descriptor descriptor,
        const PublicationLease* lease,uint64_t slots,
        semantic_graph::Receipt* queries,uint64_t* original_statements,uint64_t* offsets,
        uint64_t* supports,uint64_t support_capacity,uint64_t* status) {
    if(blockIdx.x || threadIdx.x)return;
    const uint64_t cold_work_ptr=descriptor.publication.native_work;
    if(cold_work_ptr && !publication_pointer_span(cold_work_ptr,sizeof(semantic_graph::NativeWorkTally),
            alignof(semantic_graph::NativeWorkTally))) { semantic_content_integrity_trap();return; }
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(cold_work_ptr);
    semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::Command,1);
    // The preceding encoder initializes every status on this same stream.
    // Consumers follow the projection's event; rejected input must never
    // become usable merely because no host reads its private status buffer.
    for(uint64_t slot=0;slot<slots;++slot) {
        semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::TableSlot,1);
        if(status[slot]) { semantic_content_integrity_trap();return; }
    }
    const uint64_t error=publication_feedback_lineage(descriptor,lease,slots,queries,
        original_statements,offsets,supports,support_capacity,work);
    // Keep the original private diagnostic while stopping every dependent
    // consumer, including when an actual contributor has no original target.
    if(error) {
        for(uint64_t slot=0;slot<slots;++slot) {
            status[slot]=(uint64_t(1)<<32)|error;
            semantic_graph::charge_native(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
        }
        semantic_content_integrity_trap();
    }
}

// Inputs are original sealed ranges selected by the device-acquired reader.
// Retained output capacity preserves every physical slot, including invalid gaps.
extern "C" __global__ void semantic_feedback_encode(uint64_t control_ptr,
        const PublicationLease* lease,uint64_t capacity,
        uint64_t slots, float* features, uint8_t* validity, uint64_t* status,uint64_t cold_work_ptr=0) {
    if(cold_work_ptr && !publication_pointer_span(cold_work_ptr,sizeof(semantic_graph::NativeWorkTally),
            alignof(semantic_graph::NativeWorkTally))) { semantic_content_integrity_trap();return; }
    auto* work=reinterpret_cast<semantic_graph::NativeWorkTally*>(cold_work_ptr);
    if(blockIdx.x==0 && threadIdx.x==0)
        semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::Command,1);
    if(!capacity || capacity>(UINT64_MAX-4)/9 || !slots ||
       slots>UINT64_MAX/(9*capacity+4)/sizeof(float)) { semantic_content_integrity_trap();return; }
    const uint64_t width = 9 * capacity + 4;
    __shared__ PublicationFeedbackInputs inputs;
    __shared__ uint64_t input_error;
    __shared__ semantic_feedback::Statement statement;
    __shared__ uint64_t statement_error;
    if(threadIdx.x==0) {
        // Every block resolves the actual reader and its input geometry. Its
        // reached work is merged atomically without deduplicating execution.
        semantic_graph::NativeWorkTally local{};
        input_error=publication_feedback_inputs(control_ptr,lease,slots,&inputs,work ? &local : nullptr);
        if(!input_error && inputs.statement_capacity!=capacity)input_error=1;
        for(unsigned event=0;event<9;++event)
            semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent(event),local.events[event]);
        if(work && local.overflow)atomicExch(reinterpret_cast<unsigned long long*>(&work->overflow),1ULL);
    }
    __syncthreads();
    for (uint64_t slot = blockIdx.x; slot < slots; slot += gridDim.x) {
        if(threadIdx.x==0) {
            semantic_graph::NativeWorkTally local{};
            statement={};statement_error=0;
            if(!input_error && inputs.records[slot].valid) {
                const auto index=inputs.records[slot].statement_index;
                const auto* range=index<inputs.statement_count ? publication_find_range(inputs.ranges,
                    inputs.bank->header.range_count,16,index,work ? &local : nullptr) : nullptr;
                if(!range)statement_error=1;
                else statement={publication_range_bytes(*inputs.control,*range),range->length_bytes};
            }
            for(unsigned event=0;event<9;++event)
                semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent(event),local.events[event]);
            if(work && local.overflow)atomicExch(reinterpret_cast<unsigned long long*>(&work->overflow),1ULL);
        }
        __syncthreads();
        const uint64_t error = input_error ? (uint64_t(1)<<32)|input_error : statement_error ? statement_error :
            semantic_feedback::validate(inputs.records[slot], statement, inputs.statement_count, capacity);
        if (threadIdx.x == 0) {
            status[slot] = error;
            validity[slot] = error ? 0 : uint8_t(inputs.records[slot].valid);
            semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::CanonicalByte,
                sizeof(uint64_t)+sizeof(uint8_t));
        }
        for (uint64_t column = threadIdx.x; column < width; column += blockDim.x) {
            features[slot * width + column] = error ? 0.0f :
                semantic_feedback::coordinate(inputs.records[slot], statement, capacity, column);
            semantic_graph::charge_native_parallel(work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(float));
        }
        __syncthreads();
    }
}

extern "C" __global__ void semantic_transition_execute(Descriptor descriptor) {
    if(descriptor.publication.operation) {
        if(threadIdx.x==0)publication_command(descriptor);
        return;
    }
#ifdef XLOG_SEMANTIC_POLICY
    if(descriptor.backward.cotangents) {
        if(!publication_pointer_span(descriptor.backward.work,sizeof(semantic_graph::NativeWorkTally),
                alignof(semantic_graph::NativeWorkTally))) {
            if(threadIdx.x==0)semantic_content_integrity_trap();
            return;
        }
        auto* retained_work=reinterpret_cast<semantic_graph::NativeWorkTally*>(descriptor.backward.work);
        if(threadIdx.x==0)*retained_work=semantic_graph::NativeWorkTally{};
        __syncthreads();
        semantic_graph::NativeWorkTally local_work{};
        model_policy::ScalarWorkTally scalar_work{};
        // The original snapshots and their reached bytes share this one launch
        // boundary. A refused derivative still retains the copies it executed;
        // a rejected launch cannot leave an independently submitted copy prefix.
        if(descriptor.backward.baseline_snapshot) {
            const auto* source=reinterpret_cast<const uint32_t*>(descriptor.backward.baseline_snapshot);
            auto* destination=reinterpret_cast<uint32_t*>(descriptor.backward.baselines);
            for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
                destination[i]=source[i];
                semantic_graph::charge_native(&local_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint32_t));
            }
        }
        if(descriptor.backward.score_cotangent_snapshot) {
            const auto* source=reinterpret_cast<const uint64_t*>(descriptor.backward.score_cotangent_snapshot);
            auto* destination=reinterpret_cast<uint64_t*>(descriptor.backward.cotangents);
            for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
                destination[i]=source[i];
                semantic_graph::charge_native(&local_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
            }
        }
        __syncthreads();
        semantic_policy_backward(descriptor,local_work,scalar_work);
        __syncthreads();
        semantic_graph::charge_native(&local_work,semantic_graph::NativeWorkEvent::TableSlot,scalar_work.slots);
        if(scalar_work.overflow)local_work.overflow=1;
        const auto actual=native_work_reduce_parallel(local_work);
        if(threadIdx.x==0) {
            for(uint32_t event=0;event<9;++event)
                semantic_graph::charge_native(retained_work,
                    static_cast<semantic_graph::NativeWorkEvent>(event),actual.events[event]);
            if(actual.overflow)retained_work->overflow=1;
        }
        __syncthreads();
        if(threadIdx.x==0)semantic_policy_backward_status_guard(descriptor.backward.status,descriptor.backward.mode);
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
    const bool fixed_score=descriptor.policy_score!=0;
    const auto* demonstration=reinterpret_cast<const PreparedActorRefreshDevice*>(descriptor.policy_score);
    if(fixed_score && !publication_pointer_span(descriptor.policy_score,sizeof(PreparedActorRefreshDevice),
            alignof(PreparedActorRefreshDevice))) { semantic_content_integrity_trap();return; }
    auto viability=legal+262144;
    __shared__ semantic_graph::ResidentHandle base;
    __shared__ PublicationBank* acquired_bank;
    __shared__ uint64_t acquired_word,structural_end,transition_kind;
    __shared__ bool publication_held,publication_staged,numerical_refused,input_bank_admitted;
    __shared__ uint32_t count,invalid,width,failed;
    __shared__ uint32_t choices[18],actions[4][3],lane_live;
    __shared__ TaskProgramBank program_bank;
    __shared__ semantic_program::Rule inserted_rules[2][2];
    __shared__ uint32_t inserted_counts[2],baseline_derived,candidate_derived[2];
    __shared__ semantic_graph::Receipt candidate;
    if(threadIdx.x==0)state->execution_work=ExecutionWork{};
    __syncthreads();
    if(threadIdx.x==0 && descriptor.replay.parent.actual) {
        const auto* actual=reinterpret_cast<const uint64_t*>(descriptor.replay.parent.actual);
        const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
        if(actual[2]!=1 || actual[1]!=lease.word) { semantic_content_integrity_trap();return; }
        semantic_graph::NativeWorkTally copy_work{};
        semantic_graph::charge_native(&copy_work,semantic_graph::NativeWorkEvent::CanonicalByte,actual[0]);
        execution_work_merge_native(state->execution_work,copy_work);
    }
    __syncthreads();
    bool uniform_admissible=true;
#ifdef XLOG_SEMANTIC_POLICY
    if(descriptor.policy.z)uniform_admissible=policy_uniform_admissible(descriptor.policy,&sampler_work);
#endif
    execution_work_merge_parallel(state->execution_work,sampler_work);
    if(threadIdx.x==0) {
        base={books[14],semantic_graph::kRootKind,books[15],books[16]};
        acquired_bank=nullptr;publication_held=publication_staged=false;acquired_word=structural_end=0;
        numerical_refused=input_bank_admitted=false;
        transition_kind=1;
        state->blocks=0; state->status=0; state->proposal=state->next_proposal;
        state->importance_weight=0.0;
        state->retired_roots_mask=0;
        state->task_evaluation=TaskEvaluation{};
        state->actor_eligible=0;
        state->action_batch=ActionBatchReceipt{};
        for(uint32_t i=0;i<4;++i)state->action_batch_root[i]=0;
        // Drain never executes the model body. Other modes consume the common
        // original forward, including numerical refusals.
        if(!descriptor.publication.control)consume_model_work(descriptor.model_work,state->execution_work);
        for(uint32_t i=0;i<3;++i)semantic_graph::clear_receipt(&state->cleanup_receipts[i]);
        lane_live=0;
        baseline_derived=0;
        for(uint32_t lane=0;lane<2;++lane) {
            inserted_counts[lane]=0;
            candidate_derived[lane]=0;
        }
        for(uint32_t i=0;i<7;++i)semantic_graph::clear_receipt(&state->semantic_receipts[i]);
        for(uint32_t lane=0;lane<2;++lane)state->work[lane]=Work{};
        // Published execution validates its proposal on the acquired bank below.
        failed=state->catalogue_generation!=CATALOGUE_GENERATION ||
            (!descriptor.publication.control && state->next_proposal>0xffffffffULL);
        for(int j=0;j<32;++j)failed |= state->catalogue_digest[j]!=CATALOGUE_DIGEST[j];
        for(int j=0;j<32;++j)failed |= state->binding_digest[j]!=reinterpret_cast<const uint8_t*>(books+21)[j];
        if(failed)state->status=1;
        if(!failed && (task || descriptor.task_words)) {
            if(!task || descriptor.task%alignof(uint64_t) ||
                !task_program_bank(task,descriptor.task_words,&program_bank)) {
                failed=1;state->status=7;
            } else if(program_bank.editable &&
                (program_bank.program.initial_count>PROGRAM_FACT_CAPACITY ||
                 program_bank.program.rule_count>PROGRAM_RULE_CAPACITY-2)) {
                failed=1;state->status=7;
            }
        } else program_bank=TaskProgramBank{};
        const bool critic_admissible=fixed_score || !task || !descriptor.policy.z ||
            (task && !failed && task_critic_admissible(descriptor.task_return_bound,descriptor.policy,&sampler_work));
        execution_work_merge_native(state->execution_work,sampler_work);
        sampler_work=semantic_graph::NativeWorkTally{};
        if(!failed && descriptor.publication.control) {
            PublicationAttemptWork admission_work(state->execution_work);
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(!control.continuation) { semantic_content_integrity_trap();return; }
            const auto& pending=*reinterpret_cast<const PendingContinuation*>(control.continuation);
            transition_kind=pending.transition_kind;
            if(fixed_score && (transition_kind!=1 || demonstration->abi!=1 || demonstration->status ||
               demonstration->role!=1 || demonstration->demonstrated_lane<1 || demonstration->demonstrated_lane>2 ||
               demonstration->child_control!=descriptor.publication.control)) {
                semantic_content_integrity_trap();return;
            }
            uint8_t numerical=1;
            if(transition_kind==3) {
                if(pending.numerical_admissibility || pending.training_selection ||
                    pending.model_update_bindings || pending.model_update_binding_count ||
                    pending.model_update_admissibility || pending.model_update_refusal ||
                    pending.model_update_canary_results || pending.model_update_canaries || pending.capacity_refusal) {
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
            const uint8_t late_numerical=numerical;
            if(transition_kind==4) {
                numerical &= publication_merge_actor_refresh_work(descriptor,state->execution_work);
            } else if(descriptor.publication.native_work) {
                semantic_content_integrity_trap();return;
            }
            // Every non-Drain model branch owns its scratch producer component;
            // Update additionally owns derivative, critic and actor components.
            if(transition_kind!=3)
                numerical &= publication_merge_policy_work(descriptor,state->execution_work);
            else if(descriptor.policy_work) { semantic_content_integrity_trap();return; }
            uint64_t training_status=0;
            uint8_t original_numerical=numerical;
            if(transition_kind==1)numerical &= critic_admissible && uniform_admissible;
            if(transition_kind==1 && pending.numerical_blocks) {
                if(!descriptor.publication.lease || !control.contract ||
                   !pending.physical_blocks ||
                   pending.numerical_blocks%alignof(double)) {
                    semantic_content_integrity_trap();return;
                }
                const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
                const auto* selected=publication_acquired_bank(control,lease,&admission_work.actual);
                const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
                if(!selected || publication_validate_selected_model(control,*selected,&admission_work.actual) ||
                   !publication_model_block_identity(control,*selected,contract,
                       pending.model_schema_digest,pending.model_identity,&admission_work.actual) ||
                   selected->header.model_generation!=pending.model_generation) {
                    semantic_content_integrity_trap();return;
                }
                const auto* blocks=reinterpret_cast<const double*>(pending.numerical_blocks);
                for(uint64_t block=0;block<pending.physical_blocks;++block)
                    for(uint32_t area=0;area<2;++area) {
                        const uint64_t offset=(block*2+area)*3;
                        const double budget=blocks[offset];
                        const double ordinary=blocks[offset+1];
                        const double underflow=blocks[offset+2];
                        if(!isfinite(budget) || !isfinite(ordinary) || !isfinite(underflow) ||
                           budget<0.0 || ordinary<0.0 || underflow<0.0 ||
                           (budget==0.0 ? ordinary!=0.0 : __dmul_ru(100.0,ordinary)>budget))
                            numerical=0;
                    }
            } else if(pending.numerical_blocks || pending.physical_blocks ||
                      pending.model_schema_digest[0] || pending.model_schema_digest[1] ||
                      pending.model_schema_digest[2] || pending.model_schema_digest[3] ||
                      pending.model_identity[0] || pending.model_identity[1] ||
                      pending.model_identity[2] || pending.model_identity[3]) {
                semantic_content_integrity_trap();return;
            }
            const SemanticTrainingCanaryRefusalRecord* update_refusal=nullptr;
            bool capacity_refused=false;
            if(pending.transition_kind==4) {
                if(!pending.training_selection || pending.training_selection%alignof(uint64_t) ||
                    pending.training_selection>UINT64_MAX-sizeof(uint64_t) ||
                    !pending.model_update_bindings || !pending.model_update_binding_count ||
                    !pending.model_update_admissibility ||
                    pending.model_update_admissibility==UINT64_MAX ||
                    !pending.model_update_refusal ||
                    pending.model_update_refusal%alignof(SemanticTrainingCanaryRefusalRecord) ||
                    pending.model_update_refusal>UINT64_MAX-sizeof(SemanticTrainingCanaryRefusalRecord) ||
                    !publication_pointer_span(pending.model_update_canaries,5*sizeof(SemanticTrainingCanaryRecord),alignof(SemanticTrainingCanaryRecord)) ||
                    !publication_pointer_span(pending.model_update_canary_results,5*sizeof(SemanticTrainingCanaryResultRecord),alignof(SemanticTrainingCanaryResultRecord))) {
                    semantic_content_integrity_trap();return;
                }
                training_status=*reinterpret_cast<const uint64_t*>(pending.training_selection);
                if(training_status>3) { semantic_content_integrity_trap();return; }
                if(pending.capacity_refusal) {
                    if(!publication_pointer_span(pending.capacity_refusal,sizeof(SemanticCapacityRefusalRecord),
                            alignof(SemanticCapacityRefusalRecord))) {
                        semantic_content_integrity_trap();return;
                    }
                    const auto& refusal=*reinterpret_cast<const SemanticCapacityRefusalRecord*>(pending.capacity_refusal);
                    if(!publication_capacity_refusal_valid(refusal,&admission_work.actual) ||
                       (refusal.reason && late_numerical!=0)) {
                        semantic_content_integrity_trap();return;
                    }
                    capacity_refused=refusal.reason==1;
                }
                const uint8_t update_admissibility=
                    *reinterpret_cast<const uint8_t*>(pending.model_update_admissibility);
                if(update_admissibility>1) { semantic_content_integrity_trap();return; }
                update_refusal=reinterpret_cast<const SemanticTrainingCanaryRefusalRecord*>(
                    pending.model_update_refusal);
                if(!semantic_training_canary_refusal_valid(*update_refusal,
                       reinterpret_cast<const SemanticTrainingCanaryResultRecord*>(
                           pending.model_update_canary_results),&admission_work.actual) ||
                   (update_admissibility && update_refusal->reason)) {
                    semantic_content_integrity_trap();return;
                }
                numerical &= update_admissibility;
            } else if(pending.training_selection || pending.model_update_bindings ||
                      pending.model_update_binding_count || pending.model_update_admissibility ||
                      pending.model_update_refusal || pending.model_update_canary_results || pending.model_update_canaries ||
                      pending.capacity_refusal) {
                semantic_content_integrity_trap();return;
            }
            admission_work.flush();
            if(training_status || !numerical || state->execution_work.overflow) {
                if(!descriptor.publication.lease || !control.contract) {
                    semantic_content_integrity_trap();return;
                }
                const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
                acquired_bank=publication_acquired_bank(control,lease,&admission_work.actual);
                const auto& contract=*reinterpret_cast<const PublicationContract*>(control.contract);
                if(!acquired_bank || publication_load(control.word)!=lease.word ||
                   (lease.transition_kind && pending.transition_kind!=lease.transition_kind) ||
                   pending.abi!=1 || pending.base_word!=lease.word ||
                   !publication_identity_equal(pending.instance,lease.instance) || contract.abi!=1 ||
                   !publication_identity_equal(pending.topology_identity,contract.topology_identity) ||
                   !publication_identity_equal(pending.table_identity,contract.table_identity) ||
                    !task || task[0]!=9 || task[1]!=descriptor.arena[1] ||
                   !publication_identity_equal(contract.task_identity,task+2)) {
                    semantic_content_integrity_trap();return;
                }
                const auto& header=acquired_bank->header;
                if(header.semantic_owner!=descriptor.arena[1] || publication_validate_selected_model(control,*acquired_bank,&admission_work.actual) ||
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
                if(!fixed_score)publication_refusal_snapshot(descriptor,base,acquired_bank,state);
                numerical_refused=true;failed=1;
                state->status=state->execution_work.overflow ? 11 :
                    training_status ? 11+training_status :
                    capacity_refused ? 16 :
                    !original_numerical ? 5 :
                    (update_refusal && update_refusal->reason) ? 15 : 5;
            }
        }
        if(!failed && !descriptor.publication.control && (!critic_admissible || !uniform_admissible)) {
            failed=1;state->status=5;
        }
        input_bank_admitted=!failed && transition_kind==1;
    }
    __syncthreads();
    if(numerical_refused) {
        if(threadIdx.x==0 && transition_kind==4) {
            const auto& control=*reinterpret_cast<const PublicationControl*>(descriptor.publication.control);
            publication_finish_refused_update(
                *reinterpret_cast<const PendingContinuation*>(control.continuation),state->execution_work);
        }
        return;
    }
    // publication_begin reads selected support while deriving terminal intent
    // and may retire an inactive semantic root. Prove the entire immutable
    // proposal support bank binary before either semantic effect can occur.
    if(threadIdx.x==0)invalid=0;
    __syncthreads();
    if(input_bank_admitted)for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        const Component component=components[ordinal];
        for(uint32_t i=threadIdx.x;i<component.cardinality;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            if(support[component.offset+i]>1)atomicExch(&invalid,1U);
        }
    }
    execution_work_merge_parallel(state->execution_work,sampler_work);
    if(invalid) { semantic_content_integrity_trap();return; }
    if(threadIdx.x==0) {
        if(!failed && descriptor.publication.control) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(fixed_score) {
                PublicationAttemptWork admission_work(state->execution_work);
                const auto& lease=*reinterpret_cast<const PublicationLease*>(descriptor.publication.lease);
                acquired_bank=publication_acquired_bank(control,lease,&admission_work.actual);
                acquired_word=lease.word;
                if(!acquired_bank || publication_load(control.word)!=lease.word || lease.transition_kind!=1 ||
                   publication_validate_selected_model(control,*acquired_bank,&admission_work.actual) ||
                   !publication_model_cache_ready(control,*acquired_bank,&admission_work.actual) ||
                   acquired_bank->header.model_generation!=demonstration->acquired_generation ||
                   !publication_identity_equal(acquired_bank->header.model_geometry_digest,
                       demonstration->model_geometry_digest,&admission_work.actual)) {
                    semantic_content_integrity_trap();return;
                }
            } else {
                control.refusal=publication_begin(descriptor,state,&acquired_bank,&acquired_word,&structural_end,&publication_held,&publication_staged);
                if(control.refusal){failed=1;state->status=10;}
            }
            if(!failed) {
                base={acquired_bank->header.semantic_owner,semantic_graph::kRootKind,
                    acquired_bank->header.semantic_slot,acquired_bank->header.semantic_generation};
                state->model_generation=uint32_t(acquired_bank->header.model_generation);
                state->family_id=uint32_t(acquired_bank->header.family_id);
                state->stream_serial=acquired_bank->header.stream_serial;
                state->proposal=state->next_proposal=acquired_bank->header.proposal;
                transition_kind=reinterpret_cast<const PendingContinuation*>(control.continuation)->transition_kind;
            }
        }
        if(!failed && task && (task[0]!=9 || task[1]!=descriptor.arena[1])) {
            failed=1;state->status=7;
        }
        if(!failed && task && !fixed_score)state->actor_eligible=program_bank.task.actor_eligible;
        if(!failed && task && !task_ground_acquire(descriptor,acquired_bank,program_bank.task,fixed_score ? 2 : transition_kind,state)) {
            failed=1;state->status=9;
        }
        if(!failed && program_bank.editable) {
            semantic_task_ground::Plane plane{};
            if(!semantic_task_ground::decode_plane(descriptor.task_ground,descriptor.task_ground_bytes,
                    program_bank.task,&plane)) { failed=1;state->status=9; }
            else program_bank.selected_rule_count=uint32_t(plane.header[6]);
        }
        input_bank_admitted=!failed && transition_kind==1;
    }
    __syncthreads();
    if(acquired_bank && transition_kind!=1) {
        if(threadIdx.x==0) {
            auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
            if(!failed) {
                publication_snapshot(descriptor,base,&state->semantic_receipts[0],&state->execution_work);
                failed=state->semantic_receipts[0].words[0]!=0;
                if(!failed)for(uint32_t i=0;i<4;++i)failed |=
                    state->semantic_receipts[0].words[16+i]!=acquired_bank->header.semantic_digest[i];
                if(failed)state->status=4;
            }
            if(!failed) {
                state->importance_weight=1.0;
                control.refusal=publication_publish(descriptor,*acquired_bank,acquired_word,structural_end,base,state,receipts,true,transition_kind==3);
                if(control.refusal){failed=1;state->status=state->execution_work.overflow ? 11 : 10;}
            }
            if(failed && publication_staged) {
                publication_bank(control,acquired_word^1)->header.abi=0;
                semantic_graph::NativeWorkTally invalidation{};
                semantic_graph::charge_native(&invalidation,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
                execution_work_merge_native(state->execution_work,invalidation);
            }
            if(failed && transition_kind==4)publication_finish_refused_update(
                *reinterpret_cast<const PendingContinuation*>(control.continuation),state->execution_work);
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
        if(!input_bank_admitted)continue;
        const bool active_text=c.kind==COMPONENT_KIND_TEXT &&
            (!descriptor.policy.z || text_selected(descriptor.text,c.field));
        const uint64_t offset=descriptor.policy.z && active_text
            ? text_row_index(descriptor.text,c.field)*TEXT_CARDINALITY : c.offset;
        for(uint32_t i=threadIdx.x;i<c.cardinality;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            uint8_t value=support[c.offset+i];
            if(value && active_text && !isfinite(logits[offset+i]))atomicOr(&invalid,2U);
#ifdef XLOG_SEMANTIC_POLICY
            if(value && active_text && descriptor.policy.z && !policy_domain_covers(
                    logits[offset+i],{0.0,0.0},
                    reinterpret_cast<const model_policy::DomainValue*>(descriptor.policy.numerical.text_domains)[offset+i]))
                atomicOr(&invalid,2U);
#endif
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
        const auto n=p.numerical;
        const auto* parameter_errors=reinterpret_cast<const model_policy::ErrorEnvelope*>(n.parameter_errors);
        const auto* parameter_domains=reinterpret_cast<const model_policy::DomainValue*>(n.parameter_domains);
        const auto* baseline_errors=reinterpret_cast<const model_policy::ErrorEnvelope*>(n.baseline_errors);
        const auto* baseline_domains=reinterpret_cast<const model_policy::DomainValue*>(n.baseline_domains);
        const auto* text_domains=reinterpret_cast<const model_policy::DomainValue*>(n.text_domains);
        auto* parameter_envelopes=reinterpret_cast<model_policy::PrimalEnvelope*>(n.parameter_envelopes);
        auto* baseline_envelopes=reinterpret_cast<model_policy::PrimalEnvelope*>(n.baseline_envelopes);
        for(uint64_t i=threadIdx.x;i<n.parameter_cells;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            if(!policy_domain_covers(reinterpret_cast<const float*>(p.z)[i],parameter_errors[i],parameter_domains[i]))
                atomicOr(&invalid,2U);
            // Pointwise E and independently derived ideal I have different
            // source strides. Assemble the actual 24-byte primal explicitly.
            parameter_envelopes[i]={parameter_errors[i],parameter_domains[i].ideal_magnitude};
        }
        for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            if(!policy_domain_covers(reinterpret_cast<const float*>(p.component_baselines)[i],
                    baseline_errors[i],baseline_domains[i]))atomicOr(&invalid,2U);
            baseline_envelopes[i]={baseline_errors[i],baseline_domains[i].ideal_magnitude};
        }
        // Validate the numerical storage without reading inactive TEXT primal
        // rows. Active score coverage was checked with the original mapping
        // above; its canonical FP32 score anchor has no neural input error.
        for(uint64_t i=threadIdx.x;i<32ULL*TEXT_CARDINALITY;i+=blockDim.x) {
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1);
            if(!model_policy::finite_domain(text_domains[i]))atomicOr(&invalid,2U);
        }
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
    if(threadIdx.x==0) {
        if(!failed && !fixed_score) {
            if(acquired_bank && transition_kind!=1)
                publication_snapshot(descriptor,base,&state->semantic_receipts[0],&state->execution_work);
            else semantic_call(descriptor,semantic_graph::kResidentPreflightTransitionAdmission,&base,nullptr,&state->semantic_receipts[0],nullptr,nullptr,&state->execution_work);
            failed=state->semantic_receipts[0].words[0]!=0;
            if(!failed)for(uint32_t i=0;i<4;++i)failed |= state->semantic_receipts[0].words[16+i]!=
                (acquired_bank ? acquired_bank->header.semantic_digest[i] : books[17+i]);
            if(failed)state->status=4;
        }
        if(!failed && !fixed_score && task && transition_kind==1) {
            state->execution_work.active_candidate=1;
            if(!(program_bank.editable
                ? task_program_queries(descriptor,task,program_bank,0,nullptr,0,0,
                    state,&baseline_derived)==TaskProgramQueryOutcome::Ok
                : task_queries(descriptor,task,0,&base,nullptr,state))) {
                failed=1;state->status=9;
            } else if(!state->task_evaluation.facts[0].eligible) {
                failed=1;state->status=8;
            }
            state->execution_work.active_candidate=0;
        }
        if(invalid && !failed) {
            failed=1;state->status=(invalid&2) ? 5 : 2;
        }
    }
    __syncthreads();
#ifdef XLOG_SEMANTIC_POLICY
    if (descriptor.policy.z) {
        // Initial states and every successor have separate addresses. No advance
        // aliases its input and the second slot continues the first slot's tape.
        auto tape = reinterpret_cast<float*>(descriptor.policy.recurrent);
        auto errors = reinterpret_cast<model_policy::PrimalEnvelope*>(descriptor.policy.numerical.recurrent_errors);
        for (uint32_t lane=0;lane<2;++lane)
            for (uint32_t d=threadIdx.x;d<model_policy::width;d+=blockDim.x)
                {
                    tape[lane*37*model_policy::width+d]=0.0f;
                    errors[lane*37*model_policy::width+d]={{0.0,0.0},0.0};
                }
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
            for(uint32_t opcode=0;opcode<3;++opcode)
                actions[edit][opcode]=action_completes(books,components,support,viable,start,
                                                      opcode,program_bank,&sampler_work);
            if(!actions[edit][0] && !actions[edit][1] && !actions[edit][2] && !failed){failed=1;state->status=2;}
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
        uint8_t* final_masks=descriptor.policy.z
            ? reinterpret_cast<uint8_t*>(descriptor.policy.final_masks)+component.offset : nullptr;
        uint8_t* active_sets=descriptor.policy.z
            ? reinterpret_cast<uint8_t*>(descriptor.policy.active_sets)+component.offset : nullptr;
        if(threadIdx.x==0 && !fixed_score && ordinal%68==0) {
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
            for(uint32_t i=threadIdx.x;i<component.cardinality;i+=blockDim.x) {
                final_masks[i]=uint8_t(i==TEXT_NULL);
                active_sets[i]=uint8_t(i==TEXT_NULL);
            }
            __syncthreads();
            if(threadIdx.x==0) {
                Receipt result=component_receipt(component,*state,&sampler_work,!fixed_score);
                if(fixed_score && demonstration->demonstrated_choices[ordinal]!=TEXT_NULL) {
                    failed=1;state->status=2;
                }
                result.choice=TEXT_NULL;result.legal_count=result.active_count=1;
                result.p.w[0]=result.q.w[0]=result.factor_denominator.w[0]=1;
                result.cdf_end=result.mass=uint64_t(1)<<63;
                retain_policy_cell(descriptor.policy,component,result,0.0f,U192{},U192{},true);
                receipts[ordinal]=result;if(!fixed_score)++state->blocks;
                semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(Receipt));
            }
            execution_work_merge_parallel(state->execution_work,sampler_work);
            continue;
        }
        const uint64_t offset=descriptor.policy.z && component.kind==COMPONENT_KIND_TEXT
            ? text_row_index(descriptor.text,component.field)*TEXT_CARDINALITY : component.offset;
        const float* row=descriptor.policy.z && component.kind==COMPONENT_KIND_EDIT ? nullptr : logits+offset;
        if(threadIdx.x==0){count=0;invalid=0;}
        __syncthreads();
#ifdef XLOG_SEMANTIC_POLICY
        if (descriptor.policy.z && component.kind==COMPONENT_KIND_EDIT) {
            const auto p=descriptor.policy;
            const auto f=p.fields[component.field];
            model_policy::FieldView field={reinterpret_cast<const float*>(f.embeddings),
                reinterpret_cast<const float*>(f.biases),f.cardinality,f.null_category};
            const uint32_t step=component.slot*18+component.field;
            policy_readout_hidden(p,lane,component.slot,step,component.field,&sampler_work);
            __syncthreads();
            policy_readout_scores(p,component.field,&sampler_work);
            __syncthreads();
            row=reinterpret_cast<const float*>(p.scores);
            float* retained=reinterpret_cast<float*>(p.retained_scores)
                +(lane*2+component.slot)*p.retained_score_stride+f.retained_offset;
            for(uint32_t i=threadIdx.x;i<component.cardinality;i+=blockDim.x) {
                retained[i]=row[i];
                if(!isfinite(row[i]) || !policy_primal_finite(
                        reinterpret_cast<const model_policy::PrimalEnvelope*>(p.numerical.score_errors)[i]))
                    atomicExch(&invalid,1U);
            }
            __syncthreads();
        }
#endif
        const uint8_t* mask=support+component.offset;
        for(uint32_t i=threadIdx.x;i<component.cardinality;i+=blockDim.x) {
            bool admitted=(mask[i]!=0 ||
                (semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1),false)) &&
                legal_category(component,i,books,components,support,
                32+(edit/2)*68+(edit%2)*18,program_bank,viable,choices,
                actions[edit][0],actions[edit][1],actions[edit][2],&sampler_work);
            if(final_masks) {
                final_masks[i]=uint8_t(admitted);
                active_sets[i]=0;
            }
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
            Receipt result=component_receipt(component,*state,&sampler_work,!fixed_score);
            float maximum=0.0f;
            U192 g={{0,0,0}},active_sum={{0,0,0}};
            if(!fixed_score)++state->blocks;
            bool supported=semantic_conditional_policy::prepare_conditional_cell(
                row,sorted,count,active_sets,result,maximum,g,active_sum,&sampler_work);
            if(count>1) {
                // Restore original categorical order, independently of compaction.
                uint32_t n=0;
                for(uint32_t i=0;i<component.cardinality;++i)if((mask[i] ||
                    (semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::Category,1),false)) &&
                    legal_category(component,i,books,components,support,
                        32+(edit/2)*68+(edit%2)*18,program_bank,viable,choices,
                        actions[edit][0],actions[edit][1],actions[edit][2],&sampler_work))legal[n++]=i;
            }
            supported &= semantic_conditional_policy::retain_selected_probability(
                row,legal,count,fixed_score,fixed_score ? uint32_t(demonstration->demonstrated_choices[ordinal]) : 0,
                result,maximum,g,active_sum,&sampler_work);
            if(!supported) { failed=1;state->status=fixed_score ? 2 : 3; }
            if(component.kind==COMPONENT_KIND_EDIT) {
                choices[component.field]=result.choice;
                if(choices[0]==ACTION_APPLY && component.field>=1) {
                    result.active_fields=choices[1]==2 ? INSERT_RULE_ACTIVE_MASK : INSERT_SUPPORT_ACTIVE_MASK;
                    result.null_fields=choices[1]==2 ? INSERT_RULE_NULL_MASK : INSERT_SUPPORT_NULL_MASK;
                    if(component.field==1) {
                        receipts[ordinal-1].active_fields=result.active_fields;
                        receipts[ordinal-1].null_fields=result.null_fields;
                    }
                }
            }
            retain_policy_cell(descriptor.policy,component,result,maximum,g,active_sum,count==1);
            receipts[ordinal]=result;
            semantic_graph::charge_native(&sampler_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(Receipt));
            if(!failed && !fixed_score && component.kind==COMPONENT_KIND_EDIT && component.field==17) {
                auto output=&state->semantic_receipts[1+lane*3+component.slot];
                if(choices[0]==ACTION_APPLY && choices[1]==2 && program_bank.editable &&
                   candidate.words[0]==semantic_graph::kOk) {
                    semantic_program::Rule rule{};
                    if(inserted_counts[lane]>=2 ||
                       !semantic_rule_action::decode(books,choices,
                           program_bank.program,&rule,NativePrefixWork{&sampler_work})) {
                        failed=1;state->status=9;
                    } else {
                        inserted_rules[lane][inserted_counts[lane]++]=rule;
                        ++state->work[lane].edit_commands;
                        if(!program_rule_receipt(output,task,state,program_bank,lane,component.slot,
                            inserted_rules[lane],inserted_counts[lane])) {
                            failed=1;state->status=9;
                        }
                    }
                } else if(choices[0]==ACTION_APPLY && candidate.words[0]==0) {
                    semantic_graph::DecodedStatement statement;
                    semantic_graph::DecodedSupport event;
                    decode_action(books,choices,&statement,&event);
                    if(task && !state->task_evaluation.lane_refusal[lane] &&
                       !task_allows(program_bank.task,statement,event))state->task_evaluation.lane_refusal[lane]=1;
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
            if(!failed && !fixed_score && ordinal%68==67) {
                uint64_t admission=semantic_graph::kResidentSealAdmission;
                if(task) {
                    if(state->task_evaluation.lane_refusal[lane]) {
                        task_cost(state,lane+1,program_bank.task);admission=semantic_graph::kResidentDiscardAdmission;
                    } else {
                        const auto query=program_bank.editable
                            ? task_program_queries(descriptor,task,program_bank,lane+1,
                                inserted_rules[lane],inserted_counts[lane],baseline_derived,state,
                                &candidate_derived[lane])
                            : (task_queries(descriptor,task,lane+1,nullptr,&candidate,state)
                                ? TaskProgramQueryOutcome::Ok : TaskProgramQueryOutcome::Failure);
                        if(query==TaskProgramQueryOutcome::Failure) {
                            failed=1;state->status=9;
                        } else if(query==TaskProgramQueryOutcome::Scope || query==TaskProgramQueryOutcome::FactCapacity ||
                                  query==TaskProgramQueryOutcome::FuelExhausted) {
                            state->task_evaluation.lane_refusal[lane]=
                                query==TaskProgramQueryOutcome::Scope ? 1 : query==TaskProgramQueryOutcome::FactCapacity ? 3 : 4;
                            task_cost(state,lane+1,program_bank.task);
                            admission=semantic_graph::kResidentDiscardAdmission;
                        } else if(!state->task_evaluation.facts[lane+1].eligible) {
                            state->task_evaluation.lane_refusal[lane]=2;
                            admission=semantic_graph::kResidentDiscardAdmission;
                        }
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
            const model_policy::AdvanceErrors errors{
                reinterpret_cast<const model_policy::PrimalEnvelope*>(p.numerical.recurrent_errors)+(lane*37+step)*128,
                policy_parameter_errors(p,p.recurrence),
                policy_parameter_errors(p,p.positions)+component.field*128,
                policy_field_errors(p,f),
                reinterpret_cast<model_policy::PrimalEnvelope*>(p.numerical.recurrent_errors)+(lane*37+step+1)*128};
            model_policy::advance(
                reinterpret_cast<const float*>(p.recurrent)+(lane*37+step)*128,
                reinterpret_cast<const float*>(p.recurrence),
                reinterpret_cast<const float*>(p.positions)+component.field*128,
                field,receipts[ordinal].choice,
                reinterpret_cast<float*>(p.recurrent)+(lane*37+step+1)*128,
                threadIdx.x,blockDim.x,&errors);
            for(uint32_t d=threadIdx.x;d<128;d+=blockDim.x)
                if(!policy_primal_finite(errors.successor[d])) {
                    atomicExch(&failed,1U);atomicExch(reinterpret_cast<unsigned long long*>(&state->status),5ULL);
                }
        }
        __syncthreads();
#endif
    }
    if(fixed_score) {
        if(threadIdx.x==0) {
            state->execution_work.active_candidate=0;
            if(state->execution_work.overflow)state->status=11;
        }
        return;
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
                    semantic_graph::NativeWorkTally text_work{};
                    for(uint64_t lane=0;lane<2;++lane)if(!state->task_evaluation.lane_refusal[lane]) {
                        if(publication_prepare_text(*acquired_bank,text_candidate,contract,receipts,lane+1,structural_end,descriptor.text,&text_work)) {
                            state->task_evaluation.lane_refusal[lane]=2;
                            state->task_evaluation.facts[lane+1].eligible=0;
                        }
                    }
                    text_candidate.header.abi=0;
                    semantic_graph::charge_native(&text_work,semantic_graph::NativeWorkEvent::CanonicalByte,sizeof(uint64_t));
                    execution_work_merge_native(state->execution_work,text_work);
                }
                if(task && !select_task(descriptor,state,program_bank.task)) {failed=1;state->status=9;}
                if(!failed && task && program_bank.editable && state->task_evaluation.winner) {
                    const uint32_t lane=uint32_t(state->task_evaluation.winner-1);
                    if(!task_ground_accept_rules(descriptor,program_bank,inserted_rules[lane],inserted_counts[lane],state)) {
                        failed=1;state->status=9;
                    }
                }
                state->importance_weight=weight;
                if(!failed && acquired_bank) {
                    auto& control=*reinterpret_cast<PublicationControl*>(descriptor.publication.control);
                    control.refusal=publication_publish(descriptor,*acquired_bank,acquired_word,structural_end,base,state,receipts);
                    if(control.refusal){failed=1;state->status=state->execution_work.overflow ? 11 : 10;}
                } else if(!failed)state->next_proposal=state->proposal+1;
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
        if(failed && transition_kind==4 && descriptor.publication.control) {
            const auto& control=*reinterpret_cast<const PublicationControl*>(descriptor.publication.control);
            publication_finish_refused_update(
                *reinterpret_cast<const PendingContinuation*>(control.continuation),state->execution_work);
        }
    }
}
