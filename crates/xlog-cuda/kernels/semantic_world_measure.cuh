#pragma once

#include <math.h>
#include "semantic_world.cuh"
#include "semantic_rule_cells.cuh"
#include "semantic_measure.cuh"

namespace semantic_world {

enum class MeasureStatus : uint64_t {
    Ok, InvalidSource, NodeCapacity, ApplyCapacity, FrontierCapacity,
    FuelExhausted, UndefinedCondition, UnsupportedCoefficient, InvalidNumerics, CertificateUnavailable,
    ScopeRefused, FactCapacity, RuleCapacity, InvalidProgram,
};

struct Context {
    uint32_t ordinal,kind,condition,facts,rules,queries,assignments;
    uint64_t fact_offset,rule_offset,query_offset,assignment_offset;
};

template<class Read>
__device__ bool decode_context(const Source& source,uint32_t ordinal,
        Context* output,Read read) {
    if(!output || ordinal>=source.context_count)return false;
    const auto* row=source.words+source.context_offset+uint64_t(ordinal)*kContextWords;
    uint64_t c[kContextWords];
    for(uint32_t index=0;index<kContextWords;++index)c[index]=read(row,index);
    if(c[1]>1 || c[2]>=source.event_count || c[11]>1)return false;
    for(uint32_t index=3;index<7;++index)if(c[index]>UINT32_MAX)return false;
    const uint64_t widths[4]={2,3,2,1};
    for(uint32_t index=0;index<4;++index)
        if(c[7+index]<source.payload_offset || c[7+index]>source.word_count ||
           c[3+index]*widths[index]>source.word_count-c[7+index])return false;
    if((c[1]==0 && c[6]!=source.base_variables) || (c[1]==1 && c[6]))return false;
    *output={ordinal,uint32_t(c[1]),uint32_t(c[2]),uint32_t(c[3]),uint32_t(c[4]),
        uint32_t(c[5]),uint32_t(c[6]),c[7],c[8],c[9],c[10]};
    return true;
}

// One original query worker owns this scratch. Its Boolean representation uses
// reduced ordered decisions, so intersections and unions retain shared events
// and never multiply marginal support probabilities.
struct Measure {
    static constexpr uint64_t kHeaderWords=32;
    Source source{};
    Context context{};
    uint64_t *header=nullptr,*nodes=nullptr,*unique=nullptr,*memo=nullptr,*stack=nullptr;
    uint64_t *events=nullptr,*support=nullptr,*image_degree=nullptr,*image_order=nullptr;
    uint64_t *image_paths=nullptr,*assigned=nullptr,*query_roots=nullptr;
    uint64_t *support_lineage=nullptr,*image_lineage_events=nullptr,*attainment_roots=nullptr;
    const uint64_t* ledger=nullptr;
    semantic_rule_cells::View rule_cells{};
    const uint64_t* candidate_updates=nullptr;
    uint32_t candidate_update_count=0;
    uint64_t ledger_words=0,ledger_cells=0,next_rule=0;
    uint64_t* rule_ordinals=nullptr;
    double *values=nullptr,*adjoints=nullptr,*true_weights=nullptr,*false_weights=nullptr;
    double *true_gradients=nullptr,*false_gradients=nullptr,*partitions=nullptr;
    double *masses=nullptr,*factor_gradients=nullptr,*log_potentials=nullptr;
    uint32_t count=2,variables=0,condition=1,view_condition=1,rule_capacity=0;
    uint32_t fact_capacity=0,image_nodes=0,lineage_events=0;
    bool has_measure=false;
    MeasureStatus status=MeasureStatus::Ok;
    uint64_t* spent=nullptr;
    uint64_t fuel=0;

    __device__ void fail(MeasureStatus reason) {
        if(status==MeasureStatus::Ok)status=reason;
    }
    __device__ bool charge(uint64_t amount=1) {
        if(status!=MeasureStatus::Ok)return false;
        if(!spent || *spent>fuel || amount>fuel-*spent) {
            fail(MeasureStatus::FuelExhausted);return false;
        }
        *spent+=amount;return true;
    }
    __device__ uint64_t read(const uint64_t* words,uint64_t index) {
        return charge() ? words[index] : 0;
    }
    __device__ bool bind(const Source& original,const Context& view,uint64_t* storage,
            uint64_t words,uint32_t facts,uint32_t image_count,uint32_t query_count,
            uint32_t rules,uint32_t observation_count) {
        source=original;context=view;header=storage;
        if(!storage || words<kHeaderWords || query_count!=view.queries)return false;
        variables=source.base_variables+source.selectors;
        if(variables==UINT32_MAX)return false;
        if(observation_count>UINT32_MAX/2)return false;
        fact_capacity=facts;image_nodes=image_count;
        lineage_events=observation_count*2;
        rule_capacity=rules;
        uint64_t cursor=kHeaderWords;
        auto take=[&](uint64_t amount) -> uint64_t* {
            if(cursor>words || amount>words-cursor) { cursor=UINT64_MAX;return nullptr; }
            auto* result=storage+cursor;cursor+=amount;return result;
        };
        nodes=take(uint64_t(source.nodes)*4);
        unique=take(source.unique_slots);
        memo=take(uint64_t(source.apply_entries)*4);
        stack=take(uint64_t(source.frontier_entries)*8);
        events=take(source.event_count);
        support=take(uint64_t(facts)*10);
        image_degree=take(image_count);image_order=take(image_count);image_paths=take(image_count);
        support_lineage=take(uint64_t(facts)*10*lineage_events);
        image_lineage_events=take(uint64_t(image_count)*lineage_events);
        values=reinterpret_cast<double*>(take(source.nodes));
        adjoints=reinterpret_cast<double*>(take(source.nodes));
        true_weights=reinterpret_cast<double*>(take(uint64_t(source.variables)+1));
        false_weights=reinterpret_cast<double*>(take(uint64_t(source.variables)+1));
        true_gradients=reinterpret_cast<double*>(take(uint64_t(source.variables)+1));
        false_gradients=reinterpret_cast<double*>(take(uint64_t(source.variables)+1));
        partitions=reinterpret_cast<double*>(take(uint64_t(source.variables)+1));
        assigned=take(uint64_t(source.variables)+1);
        masses=reinterpret_cast<double*>(take(uint64_t(query_count)*12));
        query_roots=take(uint64_t(query_count)*4);
        attainment_roots=take(query_count);
        factor_gradients=reinterpret_cast<double*>(take(uint64_t(source.factor_count)*6));
        log_potentials=reinterpret_cast<double*>(take(uint64_t(source.factor_count)*2));
        rule_ordinals=take(rules);
        return cursor==words;
    }
    __device__ static uint64_t hash(uint64_t a,uint64_t b,uint64_t c) {
        uint64_t value=a*0x9e3779b97f4a7c15ULL ^ b*0xbf58476d1ce4e5b9ULL ^ c;
        value=(value^(value>>30))*0xbf58476d1ce4e5b9ULL;
        value=(value^(value>>27))*0x94d049bb133111ebULL;
        return value^(value>>31);
    }
    __device__ uint32_t variable(uint32_t node) const {
        return node<2 ? variables+1 : uint32_t(nodes[uint64_t(node)*4]);
    }
    __device__ uint32_t child(uint32_t node,uint32_t top,bool high) const {
        return variable(node)==top ? uint32_t(nodes[uint64_t(node)*4+1+uint32_t(high)]) : node;
    }
    __device__ uint32_t decision(uint32_t top,uint32_t low,uint32_t high) {
        if(status!=MeasureStatus::Ok)return 0;
        if(low==high)return low;
        if(!top || top>variables || low>=count || high>=count ||
           variable(low)<=top || variable(high)<=top) {
            fail(MeasureStatus::InvalidSource);return 0;
        }
        uint64_t slot=hash(top,low,high)%source.unique_slots;
        for(uint32_t probe=0;probe<source.unique_slots;++probe) {
            if(!charge())return 0;
            const uint64_t stored=unique[slot];
            if(!stored) {
                if(count==source.nodes || uint64_t(count-1)*2>source.edges) {
                    fail(MeasureStatus::NodeCapacity);return 0;
                }
                const uint32_t result=count++;
                auto* row=nodes+uint64_t(result)*4;
                if(!charge(5))return 0;
                row[0]=top;row[1]=low;row[2]=high;row[3]=0;unique[slot]=uint64_t(result)+1;
                return result;
            }
            if(stored-1>=count) { fail(MeasureStatus::InvalidSource);return 0; }
            const auto* row=nodes+(stored-1)*4;
            if(!charge(3))return 0;
            if(row[0]==top && row[1]==low && row[2]==high)return uint32_t(stored-1);
            slot=(slot+1)%source.unique_slots;
        }
        fail(MeasureStatus::NodeCapacity);return 0;
    }
    __device__ bool lookup(uint32_t op,uint32_t left,uint32_t right,
            uint32_t* result,uint64_t* empty) {
        uint64_t slot=hash(op,left,right)%source.apply_entries;
        for(uint32_t probe=0;probe<source.apply_entries;++probe) {
            if(!charge(4))return false;
            const auto* row=memo+slot*4;
            if(!row[3]) { *empty=slot;return false; }
            if(row[0]==op && row[1]==left && row[2]==right) {
                *result=uint32_t(row[3]-1);return true;
            }
            slot=(slot+1)%source.apply_entries;
        }
        fail(MeasureStatus::ApplyCapacity);return false;
    }
    // AND, OR and XOR share one bounded iterative Shannon APPLY. The stack and
    // memo are the original declaration's arenas; there is no device recursion.
    __device__ uint32_t apply(uint32_t op,uint32_t left,uint32_t right) {
        if(status!=MeasureStatus::Ok)return 0;
        if(op>2 || left>=count || right>=count) { fail(MeasureStatus::InvalidSource);return 0; }
        uint32_t depth=1,result=0;
        stack[0]=op;stack[1]=left;stack[2]=right;stack[3]=0;
        while(depth && status==MeasureStatus::Ok) {
            if(!charge())break;
            auto* frame=stack+uint64_t(depth-1)*8;
            op=uint32_t(frame[0]);left=uint32_t(frame[1]);right=uint32_t(frame[2]);
            if(left>right) { const uint32_t swap=left;left=right;right=swap;frame[1]=left;frame[2]=right; }
            if(!frame[3]) {
                bool complete=true;
                if(left==right)result=op==2 ? 0 : left;
                else if(op==0 && left==0)result=0;
                else if(op==0 && left==1)result=right;
                else if(op==1 && left==0)result=right;
                else if(op==1 && left==1)result=1;
                else if(op==2 && left==0)result=right;
                else complete=false;
                if(complete) { --depth;continue; }
                uint64_t empty=0;
                if(lookup(op,left,right,&result,&empty)) { --depth;continue; }
                if(status!=MeasureStatus::Ok)break;
                const uint32_t a=variable(left),b=variable(right),top=a<b ? a : b;
                if(top>variables) { fail(MeasureStatus::InvalidSource);break; }
                frame[3]=1;frame[4]=top;
                if(depth==source.frontier_entries) { fail(MeasureStatus::FrontierCapacity);break; }
                auto* next=stack+uint64_t(depth++)*8;
                next[0]=op;next[1]=child(left,top,false);next[2]=child(right,top,false);next[3]=0;
            } else if(frame[3]==1) {
                frame[3]=2;frame[5]=result;
                const uint32_t top=uint32_t(frame[4]);
                if(depth==source.frontier_entries) { fail(MeasureStatus::FrontierCapacity);break; }
                auto* next=stack+uint64_t(depth++)*8;
                next[0]=op;next[1]=child(left,top,true);next[2]=child(right,top,true);next[3]=0;
            } else {
                result=decision(uint32_t(frame[4]),uint32_t(frame[5]),result);
                uint32_t cached=0;uint64_t empty=0;
                if(!lookup(op,left,right,&cached,&empty) && status==MeasureStatus::Ok) {
                    if(!charge(4))break;
                    auto* row=memo+empty*4;
                    row[0]=op;row[1]=left;row[2]=right;row[3]=uint64_t(result)+1;
                }
                --depth;
            }
        }
        return status==MeasureStatus::Ok ? result : 0;
    }
    __device__ uint32_t intersection(uint32_t left,uint32_t right) { return apply(0,left,right); }
    __device__ uint32_t unite(uint32_t left,uint32_t right) { return apply(1,left,right); }
    __device__ uint32_t complement(uint32_t root) { return apply(2,root,1); }
    __device__ uint32_t literal(uint32_t variable,bool positive) {
        return decision(variable,positive ? 0 : 1,positive ? 1 : 0);
    }
    __device__ uint32_t restrict_view(uint32_t event) {
        return intersection(event,view_condition);
    }
    __device__ uint32_t query_kind(uint32_t query) {
        if(query>=context.queries) { fail(MeasureStatus::InvalidSource);return 0; }
        const uint64_t kind=read(source.words,context.query_offset+uint64_t(query)*2);
        if(kind>1 || (!kind && context.kind!=0))fail(MeasureStatus::InvalidSource);
        return uint32_t(kind);
    }
    __device__ bool visible(uint32_t event) {
        return context.kind==0 && has_measure ? assignment(event) : event!=0;
    }
    __device__ bool bind_ledger(const uint64_t* original,uint64_t bytes) {
        if(!original || bytes%8 || bytes/8<16+source.context_count) {
            fail(MeasureStatus::InvalidSource);return false;
        }
        ledger=original;ledger_words=bytes/8;
        if(read(ledger,0)!=1 || read(ledger,9)!=source.context_count ||
           read(ledger,15)!=context.ordinal || !read(ledger,10)) {
            fail(MeasureStatus::InvalidSource);return false;
        }
        for(uint32_t word=1;word<9;++word)
            if(read(ledger,word)!=read(source.words,word))fail(MeasureStatus::InvalidSource);
        ledger_cells=read(ledger,11);
        const uint64_t cell_capacity=read(ledger,12);
        if(cell_capacity!=read(source.words,11) || ledger_cells>cell_capacity ||
           ledger_cells>(ledger_words-16-source.context_count)/6 ||
           ledger_words!=16+source.context_count+ledger_cells*6) {
            fail(MeasureStatus::InvalidSource);return false;
        }
        next_rule=read(ledger,16+context.ordinal);
        if(next_rule<context.rules || next_rule>UINT32_MAX)fail(MeasureStatus::InvalidSource);
        if(status==MeasureStatus::Ok &&
           (!semantic_rule_cells::bind(ledger,bytes,&rule_cells,
                [&](const uint64_t* data,uint64_t index) { return read(data,index); }) ||
            rule_cells.context!=context.ordinal))fail(MeasureStatus::InvalidSource);
        return status==MeasureStatus::Ok;
    }
    __device__ uint32_t fact_event(uint32_t fact,uint32_t channel) {
        if(fact>=context.facts || channel>1) { fail(MeasureStatus::InvalidSource);return 0; }
        const uint64_t event=read(source.words,context.fact_offset+uint64_t(fact)*2+channel);
        if(event>=source.event_count) { fail(MeasureStatus::InvalidSource);return 0; }
        return restrict_view(uint32_t(events[event]));
    }
    // The categorical cell has no continuous Jacobian. Only its exact zero and
    // unit endpoints have a settled Boolean meaning; intermediate codes need an
    // authored coefficient law before any value or adjoint can be published.
    __device__ uint32_t rule_gate(uint32_t rule,uint32_t appended_rules,
            uint32_t* head,uint32_t* body_mask) {
        *head=0;*body_mask=0;
        if(rule==UINT32_MAX)return 1; // Original fixed library activation.
        uint32_t gate=1;
        if(rule<context.rules) {
            const uint64_t offset=context.rule_offset+uint64_t(rule)*3;
            const uint64_t event=read(source.words,offset),head_channel=read(source.words,offset+1),
                body=read(source.words,offset+2);
            if(event>=source.event_count || head_channel>1 || body>7) {
                fail(MeasureStatus::InvalidSource);return 0;
            }
            gate=uint32_t(events[event]);*head=uint32_t(head_channel);*body_mask=uint32_t(body);
        }
        if(next_rule>uint64_t(context.rules)+appended_rules ||
           rule>=uint64_t(context.rules)+appended_rules) { fail(MeasureStatus::InvalidSource);return 0; }
        if(rule>=next_rule)return gate; // Native append creates a unit cell at the sole publication.
        semantic_rule_cells::Handle cell{};uint64_t code=0;
        auto read_cell=[&](const uint64_t* data,uint64_t index) { return read(data,index); };
        if(!semantic_rule_cells::lookup(rule_cells,rule,&cell,read_cell) ||
           !semantic_rule_cells::candidate_coefficient(rule_cells,cell,candidate_updates,
                candidate_update_count,&code,read_cell)) {
            fail(MeasureStatus::InvalidSource);return 0;
        }
        if(code==1)return 0;
        if(code==32769)return gate;
        fail(code>1 && code<32769 ? MeasureStatus::UnsupportedCoefficient : MeasureStatus::InvalidSource);
        return 0;
    }
    __device__ bool assignment(uint32_t root) {
        while(root>=2 && status==MeasureStatus::Ok) {
            if(root>=count || !charge()) { fail(MeasureStatus::InvalidSource);return false; }
            const uint32_t top=variable(root);
            root=uint32_t(nodes[uint64_t(root)*4+1+uint32_t(assigned[top]!=0)]);
        }
        return root==1;
    }
    __device__ bool initialize() {
        count=2;condition=1;view_condition=1;has_measure=false;
        for(uint32_t query=0;query<context.queries && status==MeasureStatus::Ok;++query)
            has_measure|=query_kind(query)==1;
        if(status!=MeasureStatus::Ok)return false;
        for(uint64_t index=0;index<source.unique_slots;++index) {
            if(!charge())return false;unique[index]=0;
        }
        for(uint64_t index=0;index<source.apply_entries;++index) {
            if(!charge())return false;memo[index*4+3]=0;
        }
        for(uint64_t index=0;index<=variables;++index) {
            if(!charge(3))return false;
            true_weights[index]=0.0;false_weights[index]=0.0;assigned[index]=0;
        }
        if(context.kind==0)for(uint32_t index=0;index<context.assignments;++index) {
            const uint64_t bit=read(source.words,context.assignment_offset+index);
            if(status!=MeasureStatus::Ok)return false;
            if(bit>1) { fail(MeasureStatus::InvalidSource);return false; }
            assigned[index+1]=bit;
        }
        for(uint32_t index=0;index<source.event_count && status==MeasureStatus::Ok;++index) {
            const auto* row=source.words+source.event_offset+uint64_t(index)*kEventWords;
            const uint64_t type=read(row,0),argument=read(row,1),length=read(row,2),reserved=read(row,3);
            uint32_t root=0;
            if(reserved) { fail(MeasureStatus::InvalidSource);break; }
            if(type==0 || type==1)root=uint32_t(type);
            else if(type==2) {
                const int64_t signed_variable=int64_t(argument);
                if(!signed_variable || signed_variable==INT64_MIN ||
                   (signed_variable>0 ? uint64_t(signed_variable) : uint64_t(-signed_variable))>source.base_variables) {
                    fail(MeasureStatus::InvalidSource);break;
                }
                root=literal(uint32_t(signed_variable>0 ? signed_variable : -signed_variable),signed_variable>0);
            } else if(type==3 || type==4) {
                if(!length || argument<source.payload_offset || argument>source.word_count ||
                   length>source.word_count-argument) { fail(MeasureStatus::InvalidSource);break; }
                root=type==3 ? 1 : 0;
                for(uint64_t edge=0;edge<length && status==MeasureStatus::Ok;++edge) {
                    const uint64_t child=read(source.words,argument+edge);
                    if(child>=index) { fail(MeasureStatus::InvalidSource);break; }
                    root=apply(type==3 ? 0 : 1,root,uint32_t(events[child]));
                }
            } else if(type==5 && argument<index)root=complement(uint32_t(events[argument]));
            else { fail(MeasureStatus::InvalidSource);break; }
            if(!charge())break;events[index]=root;
        }
        uint32_t selector=source.base_variables;
        for(uint32_t index=0;index<source.factor_count && status==MeasureStatus::Ok;++index) {
            const auto* row=source.words+source.factor_offset+uint64_t(index)*kFactorWords;
            const uint64_t type=read(row,0),target=read(row,1),numerator=read(row,2),denominator=read(row,3);
            if(!denominator || type>2) { fail(MeasureStatus::InvalidSource);break; }
            if(type==0)continue;
            const uint32_t variable=type==1 ? uint32_t(target) : ++selector;
            if(!variable || variable>variables || (type==1 && (target>source.base_variables || numerator>denominator)) ||
               (type==2 && target>=source.event_count)) { fail(MeasureStatus::InvalidSource);break; }
            const double positive=numerator ? log(double(numerator)/double(denominator)) : -INFINITY;
            const uint64_t negative=type==1 ? denominator-numerator : denominator;
            const double negative_weight=negative ? log(double(negative)/double(denominator)) : -INFINITY;
            if(!charge(2))break;
            true_weights[variable]=positive;false_weights[variable]=negative_weight;
            if(has_measure) {
                if(!numerator)condition=intersection(condition,literal(variable,false));
                if(!negative)condition=intersection(condition,literal(variable,true));
            }
            if(type==2) {
                const uint32_t authored=uint32_t(events[target]);
                if(context.kind==0)assigned[variable]=assignment(authored);
                const uint32_t equivalent=complement(apply(2,literal(variable,true),authored));
                condition=intersection(condition,equivalent);
            }
        }
        if(selector!=variables)fail(MeasureStatus::InvalidSource);
        const uint32_t restriction=uint32_t(events[context.condition]);
        condition=intersection(condition,restriction);
        // The authored condition restricts the common measure, not point-truth.
        // A mixed roster needs the original realization AND the conditioned
        // sample space; their union is the single closure's execution domain.
        view_condition=restriction;
        if(context.kind==0) {
            uint32_t realization=1;
            for(uint32_t variable=1;variable<=source.base_variables && status==MeasureStatus::Ok;++variable)
                realization=intersection(realization,literal(variable,assigned[variable]!=0));
            view_condition=has_measure ? unite(realization,restriction) : realization;
        }
        if(!view_condition || (has_measure && !condition))fail(MeasureStatus::UndefinedCondition);
        partitions[0]=0.0;
        for(uint32_t variable=1;variable<=variables && status==MeasureStatus::Ok;++variable) {
            if(!charge(4))break;
            const double partition=semantic_measure::logsumexp2(true_weights[variable],false_weights[variable]);
            partitions[variable]=partitions[variable-1]+partition;
            if(!isfinite(partitions[variable]))fail(MeasureStatus::InvalidNumerics);
        }
        return status==MeasureStatus::Ok;
    }

    __device__ bool begin_results(uint32_t slot) {
        if(slot>2 || !header)return false;
        if(slot==0) {
            for(uint32_t word=0;word<kHeaderWords;++word) {
                if(!charge())return false;header[word]=0;
            }
            for(uint64_t index=0;index<uint64_t(context.queries)*12;++index) {
                if(!charge())return false;masses[index]=0.0;
            }
            for(uint64_t index=0;index<uint64_t(source.factor_count)*6;++index) {
                if(!charge())return false;factor_gradients[index]=0.0;
            }
            for(uint64_t index=0;index<uint64_t(source.factor_count)*2;++index) {
                if(!charge())return false;log_potentials[index]=0.0;
            }
            if(!charge(12))return false;
            header[0]=1;header[1]=context.queries;header[2]=source.factor_count;
            header[3]=context.kind;header[4]=read(source.words,source.context_offset+uint64_t(context.ordinal)*kContextWords);
            header[5]=context.ordinal;header[6]=read(ledger,10);header[26]=variables;
            for(uint32_t word=0;word<4;++word)header[27+word]=read(source.words,word+1);
            if(!charge())return false;header[31]=uint64_t(has_measure);
        }
        if(!charge())return false;header[8+slot]=0;
        uint32_t selector=source.base_variables;
        for(uint32_t factor=0;factor<source.factor_count && status==MeasureStatus::Ok;++factor) {
            const auto* row=source.words+source.factor_offset+uint64_t(factor)*kFactorWords;
            const uint64_t kind=read(row,0);
            if(!kind)continue;
            const uint32_t variable=kind==1 ? uint32_t(read(row,1)) : ++selector;
            if(!charge(4))return false;
            log_potentials[uint64_t(factor)*2]=false_weights[variable];
            log_potentials[uint64_t(factor)*2+1]=true_weights[variable];
        }
        return status==MeasureStatus::Ok;
    }
    __device__ uint32_t support_outcome(uint32_t pro,uint32_t contra,uint32_t truth) {
        if(truth>3) { fail(MeasureStatus::InvalidSource);return 0; }
        return restrict_view(intersection(truth&1 ? pro : complement(pro),truth&2 ? contra : complement(contra)));
    }
    __device__ uint32_t query_event(uint32_t query,uint32_t pro,uint32_t contra,uint32_t truth) {
        const uint32_t event=support_outcome(pro,contra,truth);
        return query_kind(query)==1 ? intersection(event,condition) : event;
    }
    __device__ double probability(uint32_t root,double normalizer,uint32_t kind) {
        if(!kind)return assignment(root) ? 1.0 : 0.0;
        const double numerator=root_value(root);
        if(numerator==-INFINITY)return 0.0;
        const double value=exp(numerator-normalizer);
        if(!isfinite(value) || value<0.0 || value>1.0+0x1p-40)fail(MeasureStatus::InvalidNumerics);
        return value;
    }
    __device__ void seed_probability(uint32_t root,double probability,double coefficient) {
        const double cotangent=probability*coefficient;
        if(cotangent!=0.0) { seed(root,cotangent);seed(condition,-cotangent); }
    }
    __device__ bool finish_results(uint32_t slot,uint32_t joint,double correct_weight,
            double all_correct_weight,double work_cost) {
        if(slot>2)return false;
        double normalizer=0.0;
        if(has_measure) {
            if(!forward())return false;
            normalizer=root_value(condition);
            if(!isfinite(normalizer)) { fail(MeasureStatus::UndefinedCondition);return false; }
        }
        double correct=0.0;
        for(uint32_t query=0;query<context.queries;++query) {
            const uint64_t declaration=context.query_offset+uint64_t(query)*2;
            const uint32_t kind=query_kind(query);
            const uint64_t certificate=read(source.words,declaration+1);
            for(uint32_t truth=0;truth<4;++truth) {
                const double value=probability(uint32_t(read(query_roots,uint64_t(query)*4+truth)),normalizer,kind);
                if(!charge())return false;masses[(uint64_t(slot)*context.queries+query)*4+truth]=value;
            }
            correct+=probability(uint32_t(read(attainment_roots,query)),normalizer,kind);
            if(certificate && kind==1)fail(MeasureStatus::CertificateUnavailable);
        }
        if(status!=MeasureStatus::Ok)return false;
        const double all_correct=probability(joint,normalizer,uint32_t(has_measure));
        const double value=correct_weight*correct+all_correct_weight*all_correct-work_cost;
        if(!isfinite(value)) { fail(MeasureStatus::InvalidNumerics);return false; }
        if(!charge(3))return false;
        header[14+slot]=__double_as_longlong(correct);header[17+slot]=__double_as_longlong(all_correct);
        header[20+slot]=__double_as_longlong(value);
        if(has_measure) {
            if(!clear_reverse())return false;
            for(uint32_t query=0;query<context.queries;++query) {
                const uint32_t root=uint32_t(read(attainment_roots,query));
                if(query_kind(query)==1)seed_probability(root,probability(root,normalizer,1),correct_weight);
            }
            seed_probability(joint,all_correct,all_correct_weight);
            if(!reverse())return false;
        }
        uint32_t selector=source.base_variables;
        for(uint32_t factor=0;factor<source.factor_count;++factor) {
            const auto* row=source.words+source.factor_offset+uint64_t(factor)*kFactorWords;
            const uint64_t kind=read(row,0);
            const uint32_t variable=kind==1 ? uint32_t(read(row,1)) : kind==2 ? ++selector : 0;
            const uint64_t offset=(uint64_t(slot)*source.factor_count+factor)*2;
            if(!charge(2))return false;
            factor_gradients[offset]=has_measure && variable ? false_gradients[variable] : 0.0;
            factor_gradients[offset+1]=has_measure && variable ? true_gradients[variable] : 0.0;
        }
        if(!charge(2))return false;header[8+slot]=1;header[11+slot]=count;
        return status==MeasureStatus::Ok;
    }
    __device__ void record_failure(uint32_t slot) {
        if(header && slot<3)header[8+slot]=uint64_t(status==MeasureStatus::Ok ? MeasureStatus::InvalidSource : status)+1;
    }

    struct GapValues {
        const Measure* owner;
        uint32_t parent;
        __device__ double operator[](uint32_t node) const {
            const double value=owner->values[node];
            if(node==parent)return value;
            return value+(owner->partitions[owner->variable(node)-1]-owner->partitions[owner->variable(parent)]);
        }
    };
    __device__ double root_value(uint32_t root) const {
        return values[root]+partitions[variable(root)-1];
    }
    __device__ bool forward() {
        for(uint32_t node=0;node<count && status==MeasureStatus::Ok;++node) {
            if(!charge(5))break;
            values[node]=node<2 ? (node==0 ? -INFINITY : 0.0) :
                semantic_measure::forward_decision(variable(node),uint32_t(nodes[uint64_t(node)*4+1]),
                    uint32_t(nodes[uint64_t(node)*4+2]),true_weights,false_weights,GapValues{this,node});
            if(isnan(values[node]) || values[node]==INFINITY)fail(MeasureStatus::InvalidNumerics);
        }
        return status==MeasureStatus::Ok;
    }
    __device__ void gap_gradient(uint32_t begin,uint32_t end,double cotangent) {
        if(cotangent==0.0)return;
        for(uint32_t variable=begin;variable<end && status==MeasureStatus::Ok;++variable) {
            if(!charge(6))return;
            const double partition=semantic_measure::logsumexp2(true_weights[variable],false_weights[variable]);
            true_gradients[variable]+=cotangent*exp(true_weights[variable]-partition);
            false_gradients[variable]+=cotangent*exp(false_weights[variable]-partition);
        }
    }
    struct NodeAdjoints {
        Measure* owner;
        uint32_t parent;
        __device__ void add(uint32_t node,double value) const {
            if(!owner->charge())return;
            owner->adjoints[node]+=value;
            owner->gap_gradient(owner->variable(parent)+1,owner->variable(node),value);
        }
    };
    struct WeightGradients {
        Measure* owner;
        __device__ void add_true(uint32_t variable,double value) const {
            if(owner->charge())owner->true_gradients[variable]+=value;
        }
        __device__ void add_false(uint32_t variable,double value) const {
            if(owner->charge())owner->false_gradients[variable]+=value;
        }
    };
    __device__ bool clear_reverse() {
        for(uint32_t node=0;node<count;++node) {
            if(!charge())return false;adjoints[node]=0.0;
        }
        for(uint32_t variable=0;variable<=variables;++variable) {
            if(!charge(2))return false;true_gradients[variable]=0.0;false_gradients[variable]=0.0;
        }
        return true;
    }
    __device__ void seed(uint32_t root,double cotangent) {
        if(!isfinite(cotangent) || root>=count) { fail(MeasureStatus::InvalidNumerics);return; }
        if(!charge())return;adjoints[root]+=cotangent;
        gap_gradient(1,variable(root),cotangent);
    }
    __device__ bool reverse() {
        for(uint32_t next=count;next && status==MeasureStatus::Ok;) {
            const uint32_t node=--next;
            if(!charge(4))break;
            const double cotangent=adjoints[node];
            if(!isfinite(cotangent)) { fail(MeasureStatus::InvalidNumerics);break; }
            if(node<2)continue;
            const uint32_t top=variable(node),low=uint32_t(nodes[uint64_t(node)*4+1]),
                high=uint32_t(nodes[uint64_t(node)*4+2]);
            semantic_measure::propagate_decision_adjoint(top,low,high,values[node],cotangent,
                true_weights,false_weights,GapValues{this,node},NodeAdjoints{this,node});
            semantic_measure::accumulate_decision_weight_gradients(top,low,high,values[node],cotangent,
                true_weights,false_weights,GapValues{this,node},WeightGradients{this});
        }
        for(uint32_t variable=1;variable<=variables && status==MeasureStatus::Ok;++variable) {
            if(!charge(2))return false;
            if(!isfinite(true_gradients[variable]) || !isfinite(false_gradients[variable]))
                fail(MeasureStatus::InvalidNumerics);
        }
        return status==MeasureStatus::Ok;
    }
};

// Nested original program operations charge their own live budget and restore
// the enclosing owner's counter on every success or failure return.
struct MeasureFuel {
    Measure* measure;
    uint64_t* original;
    uint64_t limit;
    __device__ MeasureFuel(Measure* owner,uint64_t* counter,uint64_t ceiling):
        measure(owner),original(owner ? owner->spent : nullptr),limit(owner ? owner->fuel : 0) {
        if(owner) { owner->spent=counter;owner->fuel=ceiling; }
    }
    __device__ ~MeasureFuel() { if(measure) { measure->spent=original;measure->fuel=limit; } }
};

} // namespace semantic_world
