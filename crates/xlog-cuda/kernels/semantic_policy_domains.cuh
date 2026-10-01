#pragma once

// Private pre-draw storage, reserved with the original policy owner. Nothing
// here selects an action, reads a receipt, or evaluates another model tape.
struct PolicyUniformDescriptor {
    uint64_t outputs[3],roots[3];
    uint64_t recurrent,hidden,scores,recurrent_cotangents,score_cotangents,score_roots;
    uint64_t status,work,training_rows,return_bound;
    double loss_coefficients[4]; // edit, actor, critic, cost from the cold law
};
static_assert(sizeof(PolicyUniformDescriptor)==160,"uniform policy domain ABI");

namespace policy_uniform {
using model_policy::DomainValue;
using model_policy::CoherentCotangent;
constexpr uint32_t general64=model_policy::exact_rounding | model_policy::normal_rounding |
    model_policy::inexact_tiny_rounding;
constexpr uint32_t normal64=model_policy::exact_rounding | model_policy::normal_rounding;
constexpr double unit64=0x1p-53;

__device__ DomainValue exact(double magnitude) { return {magnitude,{0.0,0.0},magnitude}; }
__device__ DomainValue join(DomainValue a,DomainValue b) { return model_policy::domain_join(a,b); }
__device__ CoherentCotangent join(CoherentCotangent a,CoherentCotangent b) {
    return {join(a.actor,b.actor),join(a.full,b.full)};
}
__device__ CoherentCotangent seed(DomainValue value,bool actor) {
    return {actor ? value : exact(0.0),value};
}
__device__ CoherentCotangent multiply64(CoherentCotangent input,DomainValue factor,uint32_t cases) {
    return model_policy::multiply_cotangent_labels(input,factor,cases,unit64);
}
__device__ CoherentCotangent add64(CoherentCotangent a,CoherentCotangent b) {
    return model_policy::add_cotangent_labels(a,b,general64,unit64);
}
__device__ DomainValue sum64(DomainValue a,DomainValue b) {
    return model_policy::rounded_cotangent_label(
        model_policy::unrounded_cotangent_sum(a,b),normal64,unit64);
}

__device__ DomainValue importance() {
    // Closed nearest-even CDF error, including attainable opposite half ties.
    // Singleton factors are exactly one and are included in this interval.
    const double eta=0x1p-32;
    const double upper_factor=__ddiv_ru(1.0,__dadd_rd(1.0,-eta));
    // TWO correctly rounded U192 conversions, one divide, one multiply per
    // original stage. Directed construction must not round 1+u back to one.
    const double plus=__dadd_ru(1.0,unit64),minus=__dadd_rd(1.0,-unit64);
    const double upper_round=__ddiv_ru(__dmul_ru(__dmul_ru(plus,plus),plus),minus);
    const double lower_round=__ddiv_rd(__dmul_rd(__dmul_rd(minus,minus),minus),plus);
    double ideal_upper=1.0,round_upper=1.0,round_lower=1.0;
    for(uint32_t ordinal=0;ordinal<COMPONENT_COUNT;++ordinal) {
        ideal_upper=__dmul_ru(ideal_upper,upper_factor);
        round_upper=__dmul_ru(round_upper,upper_round);
        round_lower=__dmul_rd(round_lower,lower_round);
    }
    const double relative=fmax(__dadd_ru(round_upper,-1.0),__dadd_ru(1.0,-round_lower));
    // Every intermediate weight/factor remains normal and positive. No tiny
    // operation is inferred from an observed receipt or a future action.
    return {__dmul_ru(ideal_upper,round_upper),
            {__dmul_ru(ideal_upper,relative),0.0},ideal_upper};
}

__device__ DomainValue denominator_factor(double unit) {
    // Mathematical 1/RN(N), NOT an executed rounded reciprocal. The bound
    // covers EACH integer N in 1..R, including U64-to-FP32/RN64 conversion.
    // Its exact ideal bound is one; the original division rounds only once.
    return {1.0,{__ddiv_ru(unit,__dadd_rd(1.0,-unit)),0.0},1.0};
}
__device__ CoherentCotangent divide_by_rows(CoherentCotangent input) {
    return multiply64(input,denominator_factor(unit64),normal64);
}
__device__ DomainValue reward(uint64_t bound,double unit) {
    const double ideal=__ull2double_ru(bound);
    return {__dmul_ru(ideal,__dadd_ru(1.0,unit)),{__dmul_ru(unit,ideal),0.0},ideal};
}
__device__ DomainValue normalized_cost() {
    // C/cap is in [0,1]. Nonzero original integers and both conversions keep
    // the original RN64 division normal even at the largest U64 cap.
    const double plus=__dadd_ru(1.0,unit64),minus=__dadd_rd(1.0,-unit64);
    const double factor=__ddiv_ru(__dmul_ru(plus,plus),minus);
    return {factor,{__dadd_ru(factor,-1.0),0.0},1.0};
}
__device__ CoherentCotangent coefficient(const PolicyUniformDescriptor& u,
        const PolicyDescriptor& p,uint32_t ordinal,DomainValue weight) {
    const double baseline=fabs(double(reinterpret_cast<const float*>(p.component_baselines)[ordinal]));
    // The actor reference fixes the original detached FP32 baseline EXACTLY;
    // live baseline E/D is consumed separately by the critic root below.
    const auto advantage=sum64(reward(u.return_bound,unit64),exact(baseline));
    auto actor=divide_by_rows(multiply64(seed(exact(u.loss_coefficients[1]),true),weight,normal64));
    actor=multiply64(actor,advantage,normal64);
    const auto cost=divide_by_rows(multiply64(multiply64(seed(exact(u.loss_coefficients[3]),false),
        weight,normal64),normalized_cost(),normal64));
    const auto edit=divide_by_rows(seed(exact(u.loss_coefficients[0]),false));
    // Negation is exact. Zero/ineligible/floor alternatives are already in
    // each magnitude domain. Preserve edit+(actor+cost), including shared κ.
    return add64(edit,add64(actor,cost));
}
__device__ CoherentCotangent score(CoherentCotangent input) {
    // P>=h*2^118: |h*delta-1|/(P*2^-149)<=2^31, including ALL h and
    // categories, not a selected active set. P's one U192 conversion is the
    // only coefficient error here; its power-of-two scaling is exact.
    const double conversion=__ddiv_ru(unit64,__dadd_rd(1.0,-unit64));
    const DomainValue factor{__ddiv_ru(0x1p31,__dadd_rd(1.0,-unit64)),
        {__dmul_ru(0x1p31,conversion),0.0},0x1p31};
    const auto unrounded=CoherentCotangent{
        model_policy::unrounded_cotangent_product(input.actor,factor),
        model_policy::unrounded_cotangent_product(input.full,factor)};
    // The original division and subsequent integer multiplication round
    // separately. Transporting the first error through the exact integer
    // factor is homogeneous in both labels; this does not execute a fused or
    // reordered score formula. GENERAL cases cover cancellation in full.
    auto result=model_policy::round_cotangent(unrounded,general64,unit64);
    result=model_policy::round_cotangent(result,general64,unit64);
    result=model_policy::round_cotangent(result,general64,model_policy::unit_roundoff);
    return result.full.magnitude<=0x1.fffffep127 ? result : model_policy::invalid_cotangent();
}
__device__ CoherentCotangent critic(const PolicyUniformDescriptor& u,DomainValue baseline) {
    const double lambda=u.loss_coefficients[2];
    const auto reciprocal=denominator_factor(model_policy::unit_roundoff);
    auto scale=model_policy::unrounded_cotangent_product(exact(lambda),reciprocal);
    // Original FP32 lambda/RN32(N), possibly tiny, then exact power-of-two
    // scaling by two. Ordinary and underflow errors remain separate.
    const double roundoff=__dmul_ru(model_policy::unit_roundoff,scale.magnitude);
    scale={__dadd_ru(__dadd_ru(scale.magnitude,roundoff),model_policy::half_min_subnormal),
        {__dadd_ru(scale.error.ordinary,roundoff),model_policy::half_min_subnormal},lambda};
    scale={__dmul_ru(2.0,scale.magnitude),{__dmul_ru(2.0,scale.error.ordinary),
        __dmul_ru(2.0,scale.error.underflow)},__dmul_ru(2.0,scale.ideal_magnitude)};
    const auto difference=model_policy::domain_add(baseline,reward(u.return_bound,model_policy::unit_roundoff));
    return model_policy::domain_cotangent_multiply(seed(scale,false),difference);
}
__device__ const DomainValue* parameter(const PolicyDescriptor& p,uint64_t pointer) {
    return pointer ? reinterpret_cast<const DomainValue*>(p.numerical.parameter_domains)+(pointer-p.z)/4 : nullptr;
}
__device__ model_policy::FieldDomainView field(const PolicyDescriptor& p,uint32_t index) {
    const auto f=p.fields[index];
    return {parameter(p,f.embeddings),parameter(p,f.biases),f.cardinality,f.null_category};
}
__device__ void invalid(uint64_t pointer) {
    atomicOr(reinterpret_cast<unsigned long long*>(pointer),1ULL);
}
__device__ void charge(semantic_graph::NativeWorkTally& work,uint64_t count=1) {
    semantic_graph::charge_native(&work,semantic_graph::NativeWorkEvent::Category,count);
}
__device__ void merge_work(uint64_t pointer,const semantic_graph::NativeWorkTally& local) {
    auto& total=*reinterpret_cast<semantic_graph::NativeWorkTally*>(pointer);
    if(local.overflow)atomicOr(reinterpret_cast<unsigned long long*>(&total.overflow),1ULL);
    if(atomicAdd(reinterpret_cast<unsigned long long*>(&total.units),local.units)>UINT64_MAX-local.units)
        atomicOr(reinterpret_cast<unsigned long long*>(&total.overflow),1ULL);
    for(uint32_t i=0;i<9;++i)
        if(atomicAdd(reinterpret_cast<unsigned long long*>(&total.events[i]),local.events[i])>UINT64_MAX-local.events[i])
            atomicOr(reinterpret_cast<unsigned long long*>(&total.overflow),1ULL);
}
}

extern "C" __global__ void semantic_policy_uniform_domains(PolicyDescriptor p,uint64_t components_pointer,
        uint64_t support_pointer) {
    using namespace policy_uniform;
    const auto& u=*reinterpret_cast<const PolicyUniformDescriptor*>(p.numerical.uniform);
    const auto* components=reinterpret_cast<const Component*>(components_pointer);
    const auto* support=reinterpret_cast<const uint8_t*>(support_pointer);
    auto* parameters=reinterpret_cast<CoherentCotangent*>(u.roots[0]);
    auto* text=reinterpret_cast<CoherentCotangent*>(u.roots[1]);
    auto* baselines=reinterpret_cast<CoherentCotangent*>(u.roots[2]);
    auto* recurrent=reinterpret_cast<DomainValue*>(u.recurrent);
    auto* hidden=reinterpret_cast<DomainValue*>(u.hidden);
    auto* scores=reinterpret_cast<DomainValue*>(u.scores);
    auto* recurrent_cotangents=reinterpret_cast<CoherentCotangent*>(u.recurrent_cotangents);
    auto* score_cotangents=reinterpret_cast<CoherentCotangent*>(u.score_cotangents);
    auto* score_roots=reinterpret_cast<CoherentCotangent*>(u.score_roots);
    __shared__ DomainValue selected_embedding[128],null_successor[128];
    __shared__ CoherentCotangent embedding_delta[128];
    __shared__ DomainValue weight;
    __shared__ uint32_t legal_count,has_null;
    semantic_graph::NativeWorkTally work{};
    if(threadIdx.x==0) {
        *reinterpret_cast<uint64_t*>(u.status)=0;
        *reinterpret_cast<semantic_graph::NativeWorkTally*>(u.work)={};
        weight=importance();
        charge(work,COMPONENT_COUNT);
        if(u.return_bound>uint64_t(INT64_MAX))invalid(u.status);
    }
    __syncthreads();
    for(uint64_t i=threadIdx.x;i<p.numerical.parameter_cells;i+=blockDim.x) {
        parameters[i]=model_policy::zero_cotangent();charge(work);
        const auto value=reinterpret_cast<const float*>(p.z)[i];
        const auto error=reinterpret_cast<const model_policy::ErrorEnvelope*>(p.numerical.parameter_errors)[i];
        const auto domain=reinterpret_cast<const DomainValue*>(p.numerical.parameter_domains)[i];
        if(!policy_domain_covers(value,error,domain))invalid(u.status);
    }
    for(uint32_t i=threadIdx.x;i<2*37*128;i+=blockDim.x) {
        recurrent[i]=exact(0.0);recurrent_cotangents[i]=model_policy::zero_cotangent();charge(work);
    }
    for(uint32_t i=threadIdx.x;i<COMPONENT_COUNT;i+=blockDim.x) {
        const auto baseline=reinterpret_cast<const DomainValue*>(p.numerical.baseline_domains)[i];
        const auto error=reinterpret_cast<const model_policy::ErrorEnvelope*>(p.numerical.baseline_errors)[i];
        if(!policy_domain_covers(reinterpret_cast<const float*>(p.component_baselines)[i],error,baseline))invalid(u.status);
        const auto component=components[i];uint32_t count=0;
        if(component.offset!=UINT64_MAX)for(uint32_t c=0;c<component.cardinality;++c) {
            const uint8_t present=support[component.offset+c];
            if(present>1)invalid(u.status);
            count+=present!=0;charge(work);
        }
        // Singleton and inactive TEXT have EXACT zero score/critic roots.
        // No prospective training domain means no admissible native training
        // group, not a fabricated denominator or a future use grant.
        score_roots[i]=count>1 && u.training_rows ? score(coefficient(u,p,i,weight)) : model_policy::zero_cotangent();
        baselines[i]=count>1 && u.training_rows ? critic(u,baseline) : model_policy::zero_cotangent();charge(work);
    }
    __syncthreads();
    // Abstract recurrent prefixes preserve lane/slot order and original
    // primitive ordering. Support is an over-approximation of semantic legal
    // masks; no observed choice, clamp cell or receipt narrows this domain.
    for(uint32_t lane=0;lane<2;++lane)for(uint32_t step=0;step<36;++step) {
        const uint32_t ordinal=lane*68+32+step,field_index=step%18,slot=step/18;
        const auto component=components[ordinal];const auto f=field(p,field_index);
        const auto* prefix=recurrent+(lane*37+step)*128;
        model_policy::domain_readout_hidden(parameter(p,p.z)+(lane*2+slot)*128,prefix,
            parameter(p,p.positions)+field_index*128,hidden,threadIdx.x,blockDim.x);
        __syncthreads();
        model_policy::domain_readout_scores(hidden,f,scores,threadIdx.x,blockDim.x);
        __syncthreads();
        if(threadIdx.x==0) {
            legal_count=0;has_null=0;
            for(uint32_t c=0;c<f.cardinality;++c)if(support[component.offset+c]) {
                ++legal_count;has_null|=int32_t(c)==f.null_category;
            }
        }
        for(uint32_t d=threadIdx.x;d<128;d+=blockDim.x) {
            auto joined=exact(0.0);
            for(uint32_t c=0;c<f.cardinality;++c)if(support[component.offset+c]) {
                charge(work);const auto* row=model_policy::domain_embedding_row(f,c);
                if(row)joined=join(joined,row[d]);
            }
            selected_embedding[d]=joined;
        }
        for(uint32_t c=threadIdx.x;c<f.cardinality;c+=blockDim.x)
            if(support[component.offset+c] && !model_policy::finite_domain(scores[c]))invalid(u.status);
        __syncthreads();
        // A single abstract embedding row joins alternatives, not a synthetic
        // action. NULL omits the embedding addition in the original primitive.
        const model_policy::FieldDomainView abstract_field{selected_embedding,nullptr,1,-1};
        model_policy::domain_advance(prefix,parameter(p,p.recurrence),parameter(p,p.positions)+field_index*128,
            abstract_field,0,recurrent+(lane*37+step+1)*128,threadIdx.x,blockDim.x);
        if(has_null) {
            const model_policy::FieldDomainView null_field{nullptr,nullptr,1,0};
            model_policy::domain_advance(prefix,parameter(p,p.recurrence),parameter(p,p.positions)+field_index*128,
                null_field,0,null_successor,threadIdx.x,blockDim.x);
        }
        __syncthreads();
        for(uint32_t d=threadIdx.x;d<128;d+=blockDim.x) {
            auto& next=recurrent[(lane*37+step+1)*128+d];
            if(has_null)next=join(next,null_successor[d]);
            if(!model_policy::finite_domain(next) || !legal_count)invalid(u.status);
        }
        __syncthreads();
    }
    // The unknown text mapping is injective within each lane. A physical row
    // receives at most one text component per lane; join component alternatives
    // first, then add the two lane contributions in original reverse order.
    for(uint32_t category=threadIdx.x;category<TEXT_CARDINALITY;category+=blockDim.x) {
        auto result=model_policy::zero_cotangent();
        for(int lane=1;lane>=0;--lane) {
            auto alternatives=model_policy::zero_cotangent();
            for(uint32_t slot=0;slot<32;++slot) {
                const auto c=components[lane*68+slot];
                if(c.offset!=UINT64_MAX && support[c.offset+category])
                    alternatives=join(alternatives,score_roots[lane*68+slot]);
                charge(work);
            }
            result=model_policy::domain_cotangent_add(result,alternatives);
        }
        text[category]=result;
    }
    __syncthreads();
    for(uint64_t i=threadIdx.x;i<32ULL*TEXT_CARDINALITY;i+=blockDim.x) {
        if(!model_policy::finite_domain(reinterpret_cast<const DomainValue*>(p.numerical.text_domains)[i]))invalid(u.status);
        charge(work);
    }
    __syncthreads();
    // Same reverse primitive order and shared-parameter accumulation as the
    // sole original backward. Alternatives are joined, NEVER summed as if all
    // mutually exclusive embedding selections had actually occurred.
    for(int lane=1;lane>=0;--lane)for(int step=35;step>=0;--step) {
        const uint32_t ordinal=lane*68+32+step,field_index=step%18,slot=step/18;
        const auto c=components[ordinal];const auto f=field(p,field_index);
        const auto pf=p.fields[field_index];
        auto* field_embeddings=pf.embeddings ? parameters+(pf.embeddings-p.z)/4 : nullptr;
        auto* field_biases=pf.biases ? parameters+(pf.biases-p.z)/4 : nullptr;
        const auto* prefix=recurrent+(lane*37+step)*128;
        const auto* successor=recurrent+(lane*37+step+1)*128;
        model_policy::domain_readout_hidden(parameter(p,p.z)+(lane*2+slot)*128,prefix,
            parameter(p,p.positions)+field_index*128,hidden,threadIdx.x,blockDim.x);
        for(uint32_t d=threadIdx.x;d<128;d+=blockDim.x)embedding_delta[d]=model_policy::zero_cotangent();
        __syncthreads();
        const model_policy::FieldDomainView abstract_field{selected_embedding,nullptr,1,-1};
        model_policy::domain_advance_vjp(prefix,parameter(p,p.recurrence),successor,abstract_field,0,
            recurrent_cotangents+(lane*37+step+1)*128,recurrent_cotangents+(lane*37+step)*128,
            parameters+(p.recurrence-p.z)/4,parameters+(p.positions-p.z)/4+field_index*128,
            {embedding_delta,nullptr},threadIdx.x,blockDim.x);
        __syncthreads();
        for(uint32_t category=threadIdx.x;category<f.cardinality;category+=blockDim.x) {
            const bool supported=support[c.offset+category]!=0;
            score_cotangents[category]=supported ? score_roots[ordinal] : model_policy::zero_cotangent();
            if(supported && int32_t(category)!=f.null_category) {
                const uint32_t row=category-uint32_t(f.null_category>=0 && category>uint32_t(f.null_category));
                for(uint32_t d=0;d<128;++d) {
                    auto& root=field_embeddings[uint64_t(row)*128+d];
                    root=model_policy::domain_cotangent_add(root,embedding_delta[d]);charge(work);
                }
            }
            charge(work);
        }
        __syncthreads();
        model_policy::domain_readout_vjp(hidden,f,score_cotangents,
            parameters+(lane*2+slot)*128,recurrent_cotangents+(lane*37+step)*128,
            parameters+(p.positions-p.z)/4+field_index*128,{field_embeddings,field_biases},threadIdx.x,blockDim.x);
        __syncthreads();
    }
    const uint64_t cells[3]={p.numerical.parameter_cells,32ULL*TEXT_CARDINALITY,COMPONENT_COUNT};
    for(uint32_t root=0;root<3;++root) {
        const auto* labels=reinterpret_cast<const CoherentCotangent*>(u.roots[root]);
        auto* output=reinterpret_cast<DomainValue*>(u.outputs[root]);
        for(uint64_t i=threadIdx.x;i<cells[root];i+=blockDim.x) {
            // The unknown injective text mapping gives every physical row
            // the same category domain. Retain that domain ONCE, not 32 dense
            // duplicate private rows; only the public wire format is expanded.
            const auto value=labels[root==1 ? i%TEXT_CARDINALITY : i];charge(work);
            output[i]=value.actor;output[cells[root]+i]=value.full;
            if(!policy_cotangent_finite(value))invalid(u.status);
        }
    }
    merge_work(u.work,work);
}

// Export aliases never become authoritative. Compare their exact bits with
// the retained producer labels before draw; invalid bounds are a numerical
// refusal, whereas a changed alias is content corruption, not a new budget.
__device__ bool policy_uniform_admissible(const PolicyDescriptor& p,semantic_graph::NativeWorkTally* work) {
    if(!p.numerical.uniform)return false;
    const auto& u=*reinterpret_cast<const PolicyUniformDescriptor*>(p.numerical.uniform);
    const uint64_t cells[3]={p.numerical.parameter_cells,32ULL*TEXT_CARDINALITY,COMPONENT_COUNT};
    for(uint32_t root=0;root<3;++root) {
        const auto* labels=reinterpret_cast<const model_policy::CoherentCotangent*>(u.roots[root]);
        const auto* output=reinterpret_cast<const model_policy::DomainValue*>(u.outputs[root]);
        for(uint64_t i=threadIdx.x;i<cells[root];i+=blockDim.x) {
            policy_uniform::charge(*work);
            const auto& value=labels[root==1 ? i%TEXT_CARDINALITY : i];
            for(uint32_t word=0;word<4;++word)
                if(reinterpret_cast<const uint64_t*>(output+i)[word]!=reinterpret_cast<const uint64_t*>(&value.actor)[word] ||
                   reinterpret_cast<const uint64_t*>(output+cells[root]+i)[word]!=reinterpret_cast<const uint64_t*>(&value.full)[word]) {
                    semantic_content_integrity_trap();return false;
                }
        }
    }
    if(threadIdx.x==0) {
        const auto& produced=*reinterpret_cast<const semantic_graph::NativeWorkTally*>(u.work);
        for(uint32_t event=0;event<9;++event)
            semantic_graph::charge_native(work,static_cast<semantic_graph::NativeWorkEvent>(event),produced.events[event]);
        if(produced.overflow)work->overflow=1;
    }
    return *reinterpret_cast<const uint64_t*>(u.status)==0;
}
