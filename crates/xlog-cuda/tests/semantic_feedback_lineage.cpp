// Host execution of the production sequential admission/query code. CUDA launch,
// parallel scheduling, and stream lifetime are not exercised by this test.
#include <algorithm>
#include <array>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <execinfo.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <sys/resource.h>
#include <sys/mman.h>
#include <sys/wait.h>
#include <unistd.h>

#define __device__
#define __global__
#define __shared__
#define __constant__
struct HostIndex { unsigned x = 0, y = 0; };
static HostIndex blockIdx, threadIdx, blockDim{1}, gridDim{1};
static void __syncthreads() {}
static unsigned __clzll(uint64_t value) { return __builtin_clzll(value); }
static uint64_t __umul64hi(uint64_t a, uint64_t b) {
    return uint64_t((static_cast<unsigned __int128>(a) * b) >> 64);
}
static uint32_t __float_as_uint(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
static float __uint_as_float(uint32_t bits) {
    float value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
static double __longlong_as_double(long long bits) {
    double value; std::memcpy(&value, &bits, sizeof(value)); return value;
}
static long long __double_as_longlong(double value) {
    long long bits; std::memcpy(&bits, &value, sizeof(bits)); return bits;
}
static double __dadd_rn(double a, double b) { return a+b; }
static double __dsub_rn(double a, double b) { return a-b; }
static double __ddiv_rn(double a, double b) { return a/b; }
static double __dmul_rn(double a, double b) { return a*b; }
static double __ull2double_rn(uint64_t value) { return static_cast<double>(value); }
#ifdef XLOG_SEMANTIC_POLICY
static double __ll2double_rn(long long value) { return static_cast<double>(value); }
static float __ll2float_rn(long long value) { return static_cast<float>(value); }
static float __ull2float_rn(uint64_t value) { return static_cast<float>(value); }
static float __double2float_rn(double value) { return static_cast<float>(value); }
static float __fadd_rn(float a, float b) { return a+b; }
static float __fsub_rn(float a, float b) { return a-b; }
static float __fmul_rn(float a, float b) { return a*b; }
static float __fdiv_rn(float a, float b) { return a/b; }
#endif
template<typename T> T atomicAdd(T* target, T value) { T old=*target; *target+=value; return old; }
template<typename T> T atomicOr(T* target, T value) { T old=*target; *target|=value; return old; }
template<typename T> T atomicExch(T* target, T value) { T old=*target; *target=value; return old; }
using cudaGraphConditionalHandle = uint64_t;
static void cudaGraphSetConditional(cudaGraphConditionalHandle handle, unsigned value) {
    *reinterpret_cast<uint64_t*>(handle)=value;
}

#include "../kernels/semantic_transition.cu"

static void require(bool condition, const char* message) {
    if (!condition) { std::cerr << message << '\n'; std::exit(1); }
}

static void print_fatal_signal_backtrace(int signal) {
    static constexpr char heading[] = "native feedback fatal signal backtrace:\n";
    (void)::write(STDERR_FILENO, heading, sizeof(heading) - 1);
    void* frames[64];
    int frame_count = ::backtrace(frames, 64);
    ::backtrace_symbols_fd(frames, frame_count, STDERR_FILENO);
    std::raise(signal);
    _exit(128 + signal);
}

static void install_fatal_signal_backtraces() {
    struct sigaction action {};
    action.sa_handler = print_fatal_signal_backtrace;
    sigemptyset(&action.sa_mask);
    action.sa_flags = SA_NODEFER | SA_RESETHAND;
    for (int signal : {SIGABRT, SIGBUS, SIGFPE, SIGILL, SIGSEGV}) {
        require(sigaction(signal, &action, nullptr) == 0,
                "could not install native fatal signal backtrace");
    }
}

static void device_step_admission_preserves_refused_publications() {
    for(uint64_t scenario=0;scenario<11;++scenario) {
        PublicationBank banks[2]{};PublicationControl control{};PublicationLease lease{};
        control.abi=1;control.word=2;control.instance[0]=41;
        for(uint64_t bank=0;bank<2;++bank) {
            control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
            banks[bank].header.abi=1;banks[bank].header.instance[0]=41;
            banks[bank].header.publication_word=bank ? 1 : 2;
            banks[bank].header.sealed_epoch=bank ? 0 : 1;banks[bank].header.fuel=3;
        }
        uint64_t requested_kind=1,expected=0,conditional=1;
        if(scenario==0){control.reader_gate=1;expected=2;}
        if(scenario==1){control.reader_counts[1]=1;expected=2;}
        if(scenario==2){banks[0].header.sealed_epoch=0;expected=3;}
        if(scenario==3){banks[0].header.terminal=2;expected=4;}
        if(scenario==4){control.word=UINT64_MAX-1;banks[0].header.publication_word=control.word;
            banks[0].header.sealed_epoch=control.word>>1;expected=5;}
        if(scenario==5){banks[0].header.fuel=1;expected=5;}
        if(scenario==6){banks[0].header.terminal=1;banks[0].header.fuel=0;expected=5;}
        if(scenario==7){requested_kind=5;expected=1;}
        if(scenario==8){requested_kind=3;expected=1;}
        if(scenario==9){requested_kind=0;expected=1;}
        if(scenario==10){control.reader_counts[0]=UINT64_MAX;expected=1;}
        const auto before_control=control;std::array<PublicationBank,2> before_banks{banks[0],banks[1]};
        semantic_publication_step_admit(reinterpret_cast<uint64_t>(&control),reinterpret_cast<uint64_t>(&lease),
            requested_kind,reinterpret_cast<uint64_t>(&conditional),0);
        require(conditional==0 && control.refusal==expected && lease.status==expected,
            "refused device step did not clear its conditional and preserve its typed refusal");
        require(!lease.active && !lease.transition_kind && control.word==before_control.word &&
            control.reader_gate==before_control.reader_gate &&
            control.reader_counts[0]==before_control.reader_counts[0] &&
            control.reader_counts[1]==before_control.reader_counts[1] &&
            std::memcmp(banks,before_banks.data(),sizeof(banks))==0,
            "refused device step acquired a reader or mutated publication content");
    }
}

static void device_step_lease_tracks_current_parent_and_drain() {
    PublicationBank banks[2]{};PublicationControl control{};PublicationLease lease{};
    control.abi=1;control.instance[0]=73;
    for(uint64_t bank=0;bank<2;++bank) {
        control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
        banks[bank].header.abi=1;banks[bank].header.instance[0]=73;
        banks[bank].header.publication_word=bank ? 3 : 0;
        banks[bank].header.sealed_epoch=bank ? 1 : 0;banks[bank].header.fuel=2;
    }
    for(uint64_t invocation=0;invocation<4;++invocation) {
        if(invocation){control.word=3;banks[1].header.terminal=invocation==2 ? 1 : 0;}
        if(invocation>=2)banks[1].header.fuel=1;
        uint64_t conditional=0;
        semantic_publication_step_admit(reinterpret_cast<uint64_t>(&control),reinterpret_cast<uint64_t>(&lease),
            invocation==3 ? 2 : 1,reinterpret_cast<uint64_t>(&conditional),0);
        const uint64_t bank=control.word&1,kind=invocation==2 ? 3 : (invocation==3 ? 2 : 1);
        require(conditional==1 && !control.refusal && !lease.status && lease.active==1 &&
            lease.word==control.word && lease.bank==bank && lease.transition_kind==kind &&
            control.reader_counts[bank]==1 && !control.reader_counts[bank^1],
            "device step failed to acquire the actual parent and effective transition kind");
        require(publication_acquired_bank(control,lease)==&banks[bank],
            "device step lease is not a real canonical publication reader");
        semantic_publication_step_release(reinterpret_cast<uint64_t>(&control),reinterpret_cast<uint64_t>(&lease));
        require(!lease.active && !lease.status && !control.reader_counts[0] && !control.reader_counts[1] &&
            lease.transition_kind==kind,"device step release lost its retained mode or leaked ownership");
    }
}

static void publication_rejects_a_mode_different_from_device_admission() {
    PublicationBank banks[2]{};PublicationControl control{};PublicationLease lease{};
    PublicationContract contract{};PendingContinuation pending{};Descriptor descriptor{};State state{};
    std::array<uint64_t,27> task{};uint64_t conditional=0;
    control.abi=1;control.contract=reinterpret_cast<uint64_t>(&contract);control.continuation=reinterpret_cast<uint64_t>(&pending);
    for(uint64_t bank=0;bank<2;++bank)control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
    banks[0].header.abi=1;banks[0].header.fuel=3;
    semantic_publication_step_admit(reinterpret_cast<uint64_t>(&control),reinterpret_cast<uint64_t>(&lease),
        1,reinterpret_cast<uint64_t>(&conditional),0);
    require(conditional==1,"mode mismatch fixture did not admit the original proposal");
    pending.transition_kind=2;descriptor.task=reinterpret_cast<uint64_t>(task.data());
    descriptor.publication.control=reinterpret_cast<uint64_t>(&control);descriptor.publication.lease=reinterpret_cast<uint64_t>(&lease);
    PublicationBank* acquired=nullptr;uint64_t word=0,end=0;bool held=false,staged=false;
    require(publication_begin(descriptor,&state,&acquired,&word,&end,&held,&staged)==3 && held && !staged,
        "late publication accepted a continuation mode different from its admitted lease");
    publication_store(control.reader_gate,0);
    semantic_publication_step_release(reinterpret_cast<uint64_t>(&control),reinterpret_cast<uint64_t>(&lease));
}

template<typename Operation>
static void require_content_trap(Operation operation,
        const char* message="changed sealed content did not reach the actual integrity trap") {
    const pid_t child=fork();
    require(child>=0,"content integrity child could not start");
    if(child==0) {
        const rlimit no_core{0,0};
        if(setrlimit(RLIMIT_CORE,&no_core)!=0)_exit(91);
        operation();
        _exit(0);
    }
    int status=0;
    pid_t waited;
    do { waited=waitpid(child,&status,0); } while(waited<0 && errno==EINTR);
    require(waited==child,"content integrity child could not be reaped");
    require(WIFSIGNALED(status) && WTERMSIG(status)==SIGILL,message);
}

static void content_guard_follows_each_acquired_publication() {
    std::array<SourceSlot,2> values{};
    values[0].token=17;values[1].token=29;
    PublicationStorageEntry allocations[2]{};
    PublicationRange ranges[2]{};
    PublicationBank banks[2]{};
    PublicationContract contract{};contract.abi=1;contract.range_capacity=1;
    PublicationControl control{};control.abi=1;control.word=2;
    control.contract=reinterpret_cast<uint64_t>(&contract);
    control.storage=reinterpret_cast<uint64_t>(allocations);control.storage_count=2;
    for(uint64_t bank=0;bank<2;++bank) {
        allocations[bank]={reinterpret_cast<uint64_t>(&values[bank]),sizeof(SourceSlot),1};
        ranges[bank].role=1;ranges[bank].generation=1;ranges[bank].storage_slot=bank;
        ranges[bank].length_bytes=sizeof(SourceSlot);ranges[bank].logical_end=1;
        banks[bank].header.abi=1;banks[bank].header.publication_word=2+bank;
        banks[bank].header.sealed_epoch=1;banks[bank].header.range_count=1;
        control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
        control.directories[bank]=reinterpret_cast<uint64_t>(&ranges[bank]);
        require(publication_range_digest(control,ranges[bank],nullptr,ranges[bank].digest)==0,
            "acquired source content seal failed");
    }
    PublicationLease lease{};
    require(publication_acquire(control,lease)==0,"first content reader acquisition failed");
    const auto consume=[&] { semantic_publication_content_guard(reinterpret_cast<uint64_t>(&control),
        1,0,PublicationTensorLayout{},0,reinterpret_cast<uint64_t>(&lease)); };
    consume();
    // A retained reader keeps its original parent even if publication advances.
    control.word=3;values[1].token^=1;consume();values[1].token^=1;
    require(publication_release(control,lease)==0 && publication_acquire(control,lease)==0,
        "next content reader acquisition failed");
    consume();
    require_content_trap([&] { values[1].token^=1;consume(); },
        "content guard followed the captured bank instead of the acquired publication");
    values[0].token^=1;consume();values[0].token^=1;
    require_content_trap([&] { lease.active=0;consume(); },
        "content guard accepted an inactive acquired reader");
    require_content_trap([&] { ++lease.epoch;consume(); },
        "content guard accepted a stale acquired generation");
    require_content_trap([&] { lease.instance[0]^=1;consume(); },
        "content guard accepted a foreign acquired instance");
    require_content_trap([&] { control.reader_counts[1]=0;consume(); },
        "content guard accepted an unretained acquired bank");
    require_content_trap([&] { ranges[1].index=1;consume(); },
        "content guard accepted an absent original range key");
    require_content_trap([&] { banks[1].header.range_count=2;consume(); },
        "content guard traversed beyond the admitted directory capacity");
    require(publication_release(control,lease)==0 && !control.reader_counts[0] && !control.reader_counts[1],
        "content guard changed acquired reader lifetime");
}

static void retained_model_contract_preserves_original_seal_after_bank_reuse() {
    std::array<uint8_t,48> published_payload{};
    for(uint64_t i=0;i<published_payload.size();++i)published_payload[i]=uint8_t(3*i+1);
    PublicationStorageEntry storage{reinterpret_cast<uint64_t>(published_payload.data()),
        sizeof(published_payload),1};
    PublicationRange published_range{};published_range.role=44;published_range.generation=1;
    published_range.offset_bytes=8;published_range.length_bytes=32;
    PublicationContract contract{};contract.abi=1;contract.range_capacity=1;
    PublicationBank banks[2]{};
    PublicationControl control{};control.abi=1;control.word=2;
    control.contract=reinterpret_cast<uint64_t>(&contract);
    control.storage=reinterpret_cast<uint64_t>(&storage);control.storage_count=1;
    for(uint64_t bank=0;bank<2;++bank) {
        banks[bank].header.abi=1;banks[bank].header.publication_word=2+bank;
        banks[bank].header.sealed_epoch=1;banks[bank].header.range_count=1;
        control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
        control.directories[bank]=reinterpret_cast<uint64_t>(&published_range);
    }
    require(publication_range_digest(control,published_range,nullptr,published_range.digest)==0,
        "original raw model contract seal failed");
    PublicationLease original_lease{};
    require(publication_acquire(control,original_lease)==0,"original model contract reader failed");
    // This fixture retains the raw role-44 interval and its original seal. Full
    // selected-model publication authentication requires the descriptor and
    // numerical owners built by the complete publication fixture.
    std::array<uint8_t,32> retained_payload{};
    std::memcpy(retained_payload.data(),publication_range_bytes(control,published_range),retained_payload.size());
    PublicationRange retained_range=published_range;
    const auto original_payload=retained_payload;
    const auto original_range=retained_range;
    require(publication_release(control,original_lease)==0,"original model contract reader release failed");

    // Reuse the original physical directory and payload for a later epoch.
    control.word=4;banks[0].header.publication_word=4;banks[0].header.sealed_epoch=2;
    published_payload.fill(197);++storage.generation;++published_range.generation;
    require(publication_range_digest(control,published_range,nullptr,published_range.digest)==0 &&
        !publication_identity_equal(published_range.digest,retained_range.digest),
        "reused bank did not establish distinct model contract content");
    PublicationLease next_lease{};
    require(publication_acquire(control,next_lease)==0,"reused model contract reader failed");
    require(publication_release(control,next_lease)==0,"reused model contract reader release failed");

    const auto consume=[&] { semantic_retained_model_contract_guard(reinterpret_cast<uint64_t>(retained_payload.data()),
        retained_payload.size(),reinterpret_cast<uint64_t>(&retained_range)); };
    consume();
    for(uint64_t byte : {uint64_t(0),uint64_t(retained_payload.size()-1)})
        require_content_trap([&] { retained_payload[byte]^=1;consume(); },
            "retained model contract accepted changed original payload");
    for(uint64_t word=0;word<4;++word)
        require_content_trap([&] { retained_range.digest[word]^=1;consume(); },
            "retained model contract accepted a changed original seal");
    require_content_trap([&] { ++retained_range.logical_end;consume(); },
        "retained model contract ignored its original logical interval");
    require_content_trap([&] { --retained_range.length_bytes;consume(); },
        "retained model contract accepted a mismatched retained extent");
    require_content_trap([&] { retained_range.offset_bytes=UINT64_MAX;consume(); },
        "retained model contract accepted an overflowing original range");
    require_content_trap([&] { retained_range.logical_begin=1;consume(); },
        "retained model contract accepted a reversed original interval");
    require_content_trap([&] { retained_range.generation=0;consume(); },
        "retained model contract accepted an invalid original storage generation");
    require_content_trap([&] { retained_range.generation=2;consume(); },
        "retained model contract accepted a changed original storage generation");
    require_content_trap([&] { retained_range.role=43;consume(); },
        "retained model contract accepted another publication role");
    require_content_trap([&] { retained_range.index=1;consume(); },
        "retained model contract accepted another publication index");
    const auto rejects_arguments=[&](uint64_t data,uint64_t length,uint64_t range,const char* message) {
        require_content_trap([&] { semantic_retained_model_contract_guard(data,length,range); },message);
    };
    const uint64_t data=reinterpret_cast<uint64_t>(retained_payload.data());
    const uint64_t range=reinterpret_cast<uint64_t>(&retained_range);
    rejects_arguments(data,retained_payload.size()-1,range,"retained model contract accepted truncated payload bytes");
    rejects_arguments(data,retained_payload.size()+1,range,"retained model contract accepted excess payload bytes");
    rejects_arguments(0,retained_payload.size(),range,"retained model contract accepted a null payload");
    rejects_arguments(UINT64_MAX-7,retained_payload.size(),range,"retained model contract accepted an overflowing payload");
    rejects_arguments(data,0,range,"retained model contract accepted an empty payload");
    rejects_arguments(data,retained_payload.size(),0,"retained model contract accepted a null retained range");
    rejects_arguments(data,retained_payload.size(),range+1,"retained model contract accepted an unaligned retained range");
    rejects_arguments(data,retained_payload.size(),UINT64_MAX-7,"retained model contract accepted an overflowing retained range");
    rejects_arguments(range,retained_payload.size(),range,"retained model contract accepted payload overlapping its seal");
    consume();
    require(retained_payload==original_payload && std::memcmp(&retained_range,&original_range,sizeof(retained_range))==0,
        "retained model contract verification replaced original bytes or metadata");
    require(!control.reader_counts[0] && !control.reader_counts[1],
        "retained model contract verification reacquired a publication bank");
}

static void active_content_requires_computed_acquired_bank() {
    PublicationTensorLayout layout{};layout.role=53;layout.element_bytes=4;
    layout.scalar_type=6;layout.rank=2;layout.logical_axis=0;
    layout.dimensions[0]=34;layout.dimensions[1]=2;
    layout.strides_bytes[0]=8;layout.strides_bytes[1]=4;
    struct Table { TensorLayoutTableHeader header; PublicationTensorLayout layout; };
    Table table{{1,1,0,0},layout};
    std::array<float,68> values{};
    PublicationStorageEntry allocations[2]={{reinterpret_cast<uint64_t>(values.data()),sizeof(values),1},
        {reinterpret_cast<uint64_t>(&table),sizeof(table),1}};
    PublicationRange ranges[2]{};
    ranges[0].role=53;ranges[0].generation=1;
    ranges[1].role=55;ranges[1].generation=1;ranges[1].storage_slot=1;ranges[1].length_bytes=sizeof(table);
    PublicationContract contract{};contract.abi=1;contract.window_capacity=32;
    contract.feedback_capacity=2;contract.prefix_capacity=64;contract.max_position=128;contract.range_capacity=2;
    PublicationBank bank{};bank.header.abi=1;bank.header.publication_word=2;
    bank.header.sealed_epoch=1;bank.header.range_count=2;
    PublicationLease lease{};lease.abi=1;lease.active=1;lease.word=2;lease.epoch=1;
    PublicationControl control{};control.abi=1;control.word=3;
    control.banks[0]=reinterpret_cast<uint64_t>(&bank);control.reader_counts[0]=1;
    control.directories[0]=reinterpret_cast<uint64_t>(ranges);control.contract=reinterpret_cast<uint64_t>(&contract);
    control.storage=reinterpret_cast<uint64_t>(allocations);control.storage_count=2;
    TensorLayoutTableView active_table{};
    require(publication_tensor_table(control,ranges,2,&active_table),"empty active table refused");
    require(publication_range_digest(control,ranges[0],&layout,ranges[0].digest,&active_table)==0,"empty active seal failed");
    const auto seal_table=[&] { require(publication_range_digest(control,ranges[1],nullptr,ranges[1].digest)==0,
        "active table seal failed"); };
    const auto consume=[&] { semantic_publication_content_guard(reinterpret_cast<uint64_t>(&control),
        ranges[0].role,ranges[0].index,layout,1,reinterpret_cast<uint64_t>(&lease)); };
    seal_table();
    require_content_trap(consume);
    table.header.active_computed=1;seal_table();
    consume(); // Computed-empty is valid, even after the current bank changes.
    require_content_trap([&] { lease.active=0;consume(); });
    require_content_trap([&] { table.header.active_computed=0;consume(); });
    require_content_trap([&] { ranges[0].logical_end=1;consume(); });
    require_content_trap([&] { table.layout.dimensions[1]=3;seal_table();consume(); });
}

struct ResidentTextServices {
    std::array<TextRow,32> rows{};
    uint64_t count=0;
    std::array<uint8_t,32> selected{};
    TextBinding binding() { return {reinterpret_cast<uint64_t>(rows.data()),
        reinterpret_cast<uint64_t>(&count),reinterpret_cast<uint64_t>(selected.data())}; }
};

static void resident_text_services_validate_original_rows() {
    std::array<SourceSlot,32> source{};
    source[7].kind=2;source[7].valid=1;source[7].logical_position=19;
    ResidentTextServices services;services.count=1;services.rows[0]={7,19};services.selected[7]=1;
    require(validate_text_binding(source.data(),services.binding()),"resident text service pointers were interpreted as host scalars");
    services.selected[7]=2;
    require(!validate_text_binding(source.data(),services.binding()),"non-boolean device selector was accepted");
    services.selected[7]=0;services.selected[8]=1;
    require(!validate_text_binding(source.data(),services.binding()),"device selector chose a non-MASK source");
    services.selected[8]=0;services.count=33;
    require(!validate_text_binding(source.data(),services.binding()),"oversized resident text count was accepted");
    services.count=0;
    require(!validate_text_binding(source.data(),services.binding()),"resident text mapping omitted an actual MASK source");
}

struct RefusalExecution {
    static constexpr uint64_t roots=4,statements=4,supports=8,versions=8;
    std::vector<uint64_t> arena;
    std::array<uint64_t,32> books{};
    std::array<Component,136> components{};
    std::array<Receipt,136> receipts{};
    std::array<uint8_t,136> support{};
    std::array<float,136> logits{};
    std::vector<uint32_t> scratch;
    State state{};
    Descriptor descriptor{};
    semantic_graph::Receipt initial{};
    RefusalExecution() : arena(semantic_graph::kControlWords+roots*semantic_graph::kRootWords+
        semantic_graph::kCandidateWords+statements*semantic_graph::kStatementWords+
        supports*semantic_graph::kSupportWords+versions*semantic_graph::kVersionWords+
        roots*statements+statements),scratch(SCRATCH_BYTES/sizeof(uint32_t)) {
        descriptor.arena[0]=reinterpret_cast<uint64_t>(arena.data());descriptor.arena[1]=91;
        descriptor.arena[2]=roots;descriptor.arena[3]=statements;descriptor.arena[4]=supports;
        descriptor.arena[5]=versions;descriptor.arena[6]=arena.size();
        semantic_graph::Command command{};command.words[0]=semantic_graph::kInitialize;command.words[1]=91;
        semantic_graph::LaunchDescriptor launch{};
        launch.expected_owner=91;launch.root_capacity=roots;launch.statement_capacity=statements;
        launch.support_capacity=supports;launch.version_capacity=versions;launch.arena_words=arena.size();
        launch.command_ptr=reinterpret_cast<uint64_t>(&command);launch.receipt_ptr=reinterpret_cast<uint64_t>(&initial);
        launch.abi_generation=semantic_graph::kAbiGeneration;
        semantic_graph::execute(arena.data(),launch);
        require(initial.words[0]==semantic_graph::kOk,"refusal fixture initialization failed");
        publication_snapshot(descriptor,{91,semantic_graph::kRootKind,0,1},&initial);
        require(initial.words[0]==semantic_graph::kOk,"refusal fixture root snapshot failed");
        books[1]=books[3]=1;books[14]=91;books[16]=1;
        std::copy(initial.words+16,initial.words+20,books.begin()+17);
        for(uint32_t i=0;i<136;++i) {
            components[i]=COMPONENTS[i];components[i].offset=i;components[i].cardinality=1;support[i]=1;
        }
        state.catalogue_generation=CATALOGUE_GENERATION;state.next_proposal=7;
        std::copy(CATALOGUE_DIGEST,CATALOGUE_DIGEST+32,state.catalogue_digest);
        descriptor.state=reinterpret_cast<uint64_t>(&state);descriptor.codebooks=reinterpret_cast<uint64_t>(books.data());
        descriptor.components=reinterpret_cast<uint64_t>(components.data());
        descriptor.receipts=reinterpret_cast<uint64_t>(receipts.data());descriptor.scratch=reinterpret_cast<uint64_t>(scratch.data());
        descriptor.support=reinterpret_cast<uint64_t>(support.data());descriptor.logits=reinterpret_cast<uint64_t>(logits.data());
    }
    void check_protected_root() {
        const auto before=state.execution_work;
        semantic_graph::Receipt actual{};
        publication_snapshot(descriptor,{91,semantic_graph::kRootKind,0,1},&actual);
        require(std::memcmp(&before,&state.execution_work,sizeof(before))==0,
            "a separate root observation changed the original attempt work");
        require(actual.words[0]==semantic_graph::kOk && actual.words[3]==initial.words[3] &&
            actual.words[4]==initial.words[4] && std::equal(actual.words+13,actual.words+20,initial.words+13),
            "ordinary refusal changed the protected root");
        require(std::equal(actual.words+13,actual.words+20,state.semantic_receipts[0].words+13),
            "completed refusal did not retain its actual protected-root snapshot");
        semantic_graph::Layout layout{};
        require(semantic_graph::build_layout(arena.data(),roots,statements,supports,versions,arena.size(),&layout) &&
            layout.candidate[0]==semantic_graph::kFree,"ordinary refusal retained a live candidate");
        require(state.next_proposal==7 && state.proposal==7 && state.importance_weight==0.0,
            "ordinary refusal advanced its proposal or manufactured an importance weight");
    }
};

static void malformed_support_is_not_an_ordinary_refusal() {
    RefusalExecution execution;
    for(bool nonfinite : {false,true})require_content_trap([&] {
        execution.support[0]=2;if(nonfinite)execution.logits[0]=NAN;
        semantic_transition_execute(execution.descriptor);
    },"malformed Bool8 support returned as an ordinary refusal");
}

static void ordinary_refusals_finish_actual_candidate_cleanup() {
    for(uint32_t cause=0;cause<4;++cause) {
        RefusalExecution execution;
        if(cause==0)execution.support[0]=0;
        if(cause==1)execution.support[32]=0;
        if(cause==2)execution.logits[0]=NAN;
        if(cause==3)execution.logits[100]=NAN;
        semantic_transition_execute(execution.descriptor);
        require(execution.state.status==(cause<2 ? 2 : 5),"ordinary refusal acquired an integrity classification");
        const auto& work=execution.state.execution_work;
        uint64_t native=0;for(auto value:work.native_events)native+=value;
        require(!work.overflow && work.raw==work.native_attempt && native==work.native_attempt &&
            native>0 && !work.active_candidate,
            "native refusal lost the original checked event accumulator");
        require(work.native_events[unsigned(semantic_graph::NativeWorkEvent::PhiloxRound)]==
            10*execution.state.blocks,
            "sampler lost or repeated the original Philox rounds on refusal");
        execution.check_protected_root();
        if(cause==3) {
            require(work.candidates[1]>0 && work.candidates[2]>0 &&
                work.candidates[1]+work.candidates[2]<=work.native_attempt,
                "late refusal lost either candidate or charged attribution twice");
            const auto& seal=execution.state.semantic_receipts[3];
            const auto& cleanup=execution.state.cleanup_receipts[0];
            require(execution.state.blocks==100 && execution.state.retired_roots_mask==1 &&
                seal.words[0]==semantic_graph::kOk && seal.words[1]==semantic_graph::kOutcomeUnchanged &&
                seal.words[34]==0 && seal.words[35]==1 && cleanup.words[0]==semantic_graph::kOk &&
                cleanup.words[5]==0 && cleanup.words[6]>1 && cleanup.words[39]==91,
                "late numerical refusal did not consume the sealed first lane and discard the second candidate");
        } else require(!execution.state.blocks && !execution.state.retired_roots_mask,
            "pre-draw ordinary refusal acquired or retired a lane");
    }
}

static void native_work_preserves_one_model_charge_through_refusal() {
    RefusalExecution execution;
    ModelWorkEvent original{};original.kind=4;original.witness=original.tensor=UINT64_MAX;
    original.rank=1;original.dimensions[0]=7;
    execution.descriptor.model_work={reinterpret_cast<uint64_t>(&original),1,7};
    execution.logits[100]=NAN;
    semantic_transition_execute(execution.descriptor);
    const auto& work=execution.state.execution_work;
    require(execution.state.status==5 && work.model_events==1 && work.model_once==7 &&
        work.model_bound==7 && work.native_attempt>0 && work.raw==7+work.native_attempt &&
        !work.overflow && !work.active_candidate,
        "refusal reset, duplicated or dropped the original model work");
    execution.check_protected_root();

    semantic_graph::Receipt query{};
    semantic_graph::DecodedStatement statement{};
    const semantic_graph::ResidentHandle root{91,semantic_graph::kRootKind,0,1};
    const auto completed=work;
    semantic_call(execution.descriptor,semantic_graph::kResidentRootTruthAdmission,
        &root,nullptr,&query,&statement);
    require(query.words[0]==semantic_graph::kOk &&
        std::memcmp(&completed,&work,sizeof(work))==0,
        "a separate semantic query charged a completed attempt through its descriptor");
    ExecutionWork owner{};
    semantic_call(execution.descriptor,semantic_graph::kResidentRootTruthAdmission,
        &root,nullptr,&query,&statement,nullptr,&owner);
    require(query.words[0]==semantic_graph::kOk && owner.native_attempt>0 &&
        owner.raw==owner.native_attempt && std::memcmp(&completed,&work,sizeof(work))==0,
        "semantic work did not remain with its explicit original owner");

    semantic_graph::NativeWorkTally reached{};
    semantic_graph::charge_native(&reached,semantic_graph::NativeWorkEvent::TableSlot,2);
    ExecutionWork full{};full.raw=full.native_attempt=UINT64_MAX;
    full.native_events[0]=UINT64_MAX;
    execution_work_merge_native(full,reached);
    require(full.overflow==1 && full.raw==UINT64_MAX && full.native_attempt==UINT64_MAX &&
        full.native_events[1]==0,"native merge overflow partially changed the common sum");
}

static void task_preflight_accepts_all_four_truth_states() {
    for(uint64_t first=0;first<4;++first)for(uint64_t second=0;second<4;++second) {
        RefusalExecution execution;
        std::array<uint64_t,34> task{};
        task[0]=4;task[1]=91;task[6]=11;task[10]=21;
        task[18]=first;task[19]=second;task[20]=0;
        task[25]=task[26]=15;task[31]=task[32]=task[33]=15;
        execution.descriptor.task=reinterpret_cast<uint64_t>(task.data());
        // Stop at the existing ordinary support refusal after the actual task
        // preflight and root truth queries, before any categorical draw.
        execution.support[0]=0;
        semantic_transition_execute(execution.descriptor);
        require(execution.state.status==2 && execution.state.task_evaluation.query_count==3,
            "valid four-valued task truth was rejected before native query execution");
        const auto& facts=execution.state.task_evaluation.facts[0];
        require(facts.correct[0]==uint64_t(first==0) && facts.correct[1]==uint64_t(second==0),
            "task agreement did not compare the actual empty-root truth receipts");
        execution.check_protected_root();
    }
    for(uint32_t slot=0;slot<3;++slot)for(uint64_t invalid : {uint64_t(4),UINT64_MAX}) {
        RefusalExecution execution;
        std::array<uint64_t,34> task{};
        task[0]=4;task[1]=91;task[18+slot]=invalid;task[25]=task[26]=15;
        execution.descriptor.task=reinterpret_cast<uint64_t>(task.data());
        semantic_transition_execute(execution.descriptor);
        require(execution.state.status==7 && execution.state.task_evaluation.query_count==0 &&
            execution.state.blocks==0,
            "out-of-range task truth reached native queries or categorical draws");
    }
}

static void resident_numerical_refusal_precedes_publication();

static void ordinary_publication_refusal_releases_gate_and_preserves_base() {
    static constexpr uint64_t model_schema_begin=104;
    static constexpr std::array<uint8_t,8> model_schema{44,0,0x80,0xff,0,0,0,0};
    static constexpr uint64_t model_contract_bytes=model_schema_begin+model_schema.size();
    for(bool nonfinite : {false,true}) {
        RefusalExecution execution;
        auto& descriptor=execution.descriptor;
        semantic_graph::ResidentHandle base_root{91,semantic_graph::kRootKind,0,1};
        semantic_graph::Receipt candidate{},inserted{},sealed{};
        semantic_call(descriptor,semantic_graph::kResidentForkAdmission,&base_root,nullptr,&candidate);
        semantic_graph::DecodedStatement statement{};statement.identity_words[0]=11;
        semantic_graph::DecodedSupport event{};event.polarity=1;event.source_words[0]=31;
        semantic_call(descriptor,semantic_graph::kResidentInsertSupportAdmission,nullptr,&candidate,&inserted,&statement,&event);
        semantic_call(descriptor,semantic_graph::kResidentSealAdmission,nullptr,&inserted,&sealed);
        require(sealed.words[0]==semantic_graph::kOk && sealed.words[1]==semantic_graph::kOutcomeInserted,
            "publication refusal fixture did not create an actual exclusive inactive root");

        std::vector<std::vector<uint64_t>> allocations;
        std::vector<PublicationStorageEntry> storage;
        const auto allocate=[&](uint64_t bytes) {
            allocations.emplace_back((bytes+7)/8);
            auto& cells=allocations.back();
            storage.push_back({reinterpret_cast<uint64_t>(cells.data()),cells.size()*8,1});
            return uint64_t(storage.size()-1);
        };
        const auto bytes=[&](uint64_t slot) { return reinterpret_cast<uint64_t*>(storage[slot].pointer); };
        std::array<PublicationRoleCount,55> roles{};
        std::vector<PublicationTensorLayout> layouts;
        std::vector<PublicationRange> ranges,destinations;
        for(uint64_t role=1;role<=55;++role) {
            const bool tensor=publication_tensor_role(role);
            const uint64_t count=tensor ? uint64_t((role>=8 && role<=13) || role==18) : (role==16 ? 3 : 1);
            roles[role-1]={role,count};
            for(uint64_t index=0;index<count;++index) {
                uint64_t length=tensor ? 4 : 8;
                if(role==1 || role==2)length=0;
                if(role==3)length=32;
                if(role==14)length=sizeof(CompletionCoverage);
                if(role==15)length=3*sizeof(RawFeedbackRecord);
                if(role==30)length=sizeof(IntentQueueHeader);
                if(role==31)length=1;
                if(role==33)length=sizeof(AttemptReceipt);
                if(role==44)length=model_contract_bytes;
                if(role==55)length=sizeof(TensorLayoutTableHeader)+7*sizeof(PublicationTensorLayout);
                PublicationRange range{};range.role=role;range.index=index;range.generation=1;
                range.length_bytes=length;range.storage_slot=allocate(length ? length : 8);
                ranges.push_back(range);
                if(publication_mutable_role(role) && role!=1 && role!=2 && role!=31)
                    range.storage_slot=allocate(length ? length : 8);
                destinations.push_back(range);
                if(tensor) {
                    PublicationTensorLayout layout{};layout.role=role;layout.index=index;
                    layout.element_bytes=4;layout.scalar_type=6;layout.rank=1;layout.logical_axis=UINT64_MAX;
                    layout.dimensions[0]=1;layout.strides_bytes[0]=4;layouts.push_back(layout);
                }
            }
        }
        const auto initialize_table=[&](const std::vector<PublicationRange>& directory) {
            const auto* range=publication_find_range(directory.data(),directory.size(),55);
            auto* header=reinterpret_cast<TensorLayoutTableHeader*>(bytes(range->storage_slot));
            *header={1,layouts.size(),1,0};
            std::copy(layouts.begin(),layouts.end(),reinterpret_cast<PublicationTensorLayout*>(header+1));
        };
        initialize_table(ranges);initialize_table(destinations);
        const auto* model_range=publication_find_range(ranges.data(),ranges.size(),44);
        std::copy(model_schema.begin(),model_schema.end(),
            reinterpret_cast<uint8_t*>(bytes(model_range->storage_slot))+model_schema_begin);
        const auto* intent_range=publication_find_range(ranges.data(),ranges.size(),30);
        auto* queue=reinterpret_cast<IntentQueueHeader*>(bytes(intent_range->storage_slot));
        queue->abi=1;queue->payload_used_bytes=queue->effect_length_bytes=1;queue->payload_capacity_bytes=8;

        std::vector<PublicationRange> pending_ranges;
        std::vector<PublicationTensorLayout> pending_layouts;
        for(const auto& range:ranges)if(range.role==1 || range.role==39 || (range.role>=8 && range.role<=13)) {
            auto copy=range;copy.storage_slot=allocate(range.length_bytes ? range.length_bytes : 32*sizeof(SourceSlot));
            pending_ranges.push_back(copy);
            if(range.role>=8 && range.role<=13)
                pending_layouts.push_back(layouts[range.role-8]);
        }
        PublicationRange pending_table{};pending_table.role=55;pending_table.generation=1;
        pending_table.length_bytes=sizeof(TensorLayoutTableHeader)+pending_layouts.size()*sizeof(PublicationTensorLayout);
        pending_table.storage_slot=allocate(pending_table.length_bytes);
        auto* table=reinterpret_cast<TensorLayoutTableHeader*>(bytes(pending_table.storage_slot));
        *table={1,pending_layouts.size(),1,0};
        std::copy(pending_layouts.begin(),pending_layouts.end(),reinterpret_cast<PublicationTensorLayout*>(table+1));
        pending_ranges.push_back(pending_table);

        std::array<uint64_t,34> task{};task[0]=4;task[1]=91;task[6]=11;task[10]=task[14]=21;
        task[18]=1;task[19]=task[20]=2;
        task[25]=14;task[26]=7;task[27]=1;task[28]=45;task[29]=15;task[30]=1;
        task[31]=task[32]=task[33]=7;
        descriptor.task=reinterpret_cast<uint64_t>(task.data());
        PublicationContract contract{};contract.abi=1;contract.range_capacity=ranges.size();
        contract.prefix_capacity=64;contract.window_capacity=32;contract.feedback_capacity=3;contract.max_position=128;
        contract.model_generation=3;contract.authority_generation=4;contract.semantic_owner=91;
        contract.model_contract_layout={model_schema_begin,model_schema.size(),0,32,40,72};
        contract.role_counts=reinterpret_cast<uint64_t>(roles.data());contract.role_count=roles.size();
        PublicationBank bank{},inactive{};
        bank.header.abi=1;bank.header.publication_word=2;bank.header.sealed_epoch=1;
        bank.header.semantic_owner=91;bank.header.semantic_generation=1;bank.header.range_count=ranges.size();
        bank.header.model_generation=3;bank.header.authority_generation=4;bank.header.fuel=2;bank.header.proposal=7;
        std::copy(execution.initial.words+13,execution.initial.words+16,bank.header.semantic_extents);
        std::copy(execution.initial.words+16,execution.initial.words+20,bank.header.semantic_digest);
        inactive.header=bank.header;inactive.header.semantic_slot=sealed.words[34];inactive.header.semantic_generation=sealed.words[35];
        ResidentTextServices services;uint8_t numerical=1;
        PendingContinuation pending{};pending.abi=1;pending.base_word=2;pending.model_generation=3;pending.authority_generation=4;
        pending.transition_kind=1;pending.text=services.binding();pending.numerical_admissibility=reinterpret_cast<uint64_t>(&numerical);
        pending.ranges=reinterpret_cast<uint64_t>(pending_ranges.data());pending.range_count=pending_ranges.size();
        PublicationControl control{};control.abi=1;control.word=2;control.contract=reinterpret_cast<uint64_t>(&contract);
        control.continuation=reinterpret_cast<uint64_t>(&pending);control.storage=reinterpret_cast<uint64_t>(storage.data());control.storage_count=storage.size();
        control.banks[0]=reinterpret_cast<uint64_t>(&bank);control.banks[1]=reinterpret_cast<uint64_t>(&inactive);
        control.directories[0]=reinterpret_cast<uint64_t>(ranges.data());control.directories[1]=reinterpret_cast<uint64_t>(destinations.data());
        require(publication_seal_ranges(control,bank,nullptr)==0,
            "publication refusal fixture could not seal its selected model");
        require(publication_descriptor_digest(control,bank)==0,
            "publication refusal fixture could not authenticate its selected model descriptor");
        PublicationLease lease{};descriptor.publication.control=reinterpret_cast<uint64_t>(&control);
        descriptor.publication.lease=reinterpret_cast<uint64_t>(&lease);
        std::array<float,4*128> z{};std::array<float,128*128> recurrence{};
        std::array<float,18*128> positions{};std::array<float,2*37*128> recurrent{};
        descriptor.text=services.binding();descriptor.policy.z=reinterpret_cast<uint64_t>(z.data());
        descriptor.policy.recurrence=reinterpret_cast<uint64_t>(recurrence.data());
        descriptor.policy.positions=reinterpret_cast<uint64_t>(positions.data());
        descriptor.policy.recurrent=reinterpret_cast<uint64_t>(recurrent.data());
        for(auto& field:descriptor.policy.fields) { field.cardinality=1;field.null_category=0; }
        execution.support[32]=0;
#ifdef XLOG_SEMANTIC_POLICY
        if(nonfinite)z[0]=NAN;
#else
        if(nonfinite)continue;
#endif
        require_content_trap([&] {
            control.reader_gate=1;
            execution.support[0]=2;semantic_transition_execute(descriptor);
        },"malformed inactive TEXT support reached publication begin");
        descriptor.publication.operation=2;semantic_transition_execute(descriptor);descriptor.publication.operation=0;
        require(lease.active==1 && !lease.status,"publication refusal did not acquire its actual reader");
        const auto original_bank=bank;
        semantic_transition_execute(descriptor);
        if(execution.state.status!=(nonfinite ? 5 : 2) || control.refusal) {
            std::cerr<<"ordinary publication refusal did not traverse the validated publication begin path: state.status="
                <<execution.state.status<<", control.refusal="<<control.refusal<<", nonfinite="<<nonfinite<<'\n';
            std::exit(1);
        }
        require(control.word==lease.word && control.reader_gate==0 && lease.active==1 &&
            control.reader_counts[lease.bank]==1 && !inactive.header.abi && std::memcmp(&bank,&original_bank,sizeof(bank))==0,
            "ordinary refusal published effects, leaked its gate, or changed the acquired bank");
        const auto& cleanup=execution.state.cleanup_receipts[0];
        require(cleanup.words[0]==semantic_graph::kOk && cleanup.words[3]==sealed.words[34] &&
            cleanup.words[4]==sealed.words[35]+1 && cleanup.words[39]==91 && !execution.state.retired_roots_mask,
            "ordinary refusal lost the actual exclusive inactive-root retirement receipt");
        execution.check_protected_root();
        descriptor.publication.operation=3;semantic_transition_execute(descriptor);
        require(!lease.status && !lease.active && !control.reader_counts[0] && !control.reader_gate,
            "completed ordinary refusal prevented its actual reader release");
    }
}

static void backward_completion_rejects_every_error_status() {
    uint64_t status=0;
    semantic_policy_backward_status_guard(reinterpret_cast<uint64_t>(&status));
    for(uint64_t failure : {uint64_t(1),uint64_t(2),uint64_t(3),UINT64_MAX}) {
        require_content_trap([&] { status=failure;semantic_policy_backward_status_guard(reinterpret_cast<uint64_t>(&status)); },
            "backward completion accepted a mutated device error status");
    }
}

#ifdef XLOG_SEMANTIC_POLICY
static void backward_nonfinite_cotangent_reaches_integrity_trap() {
    std::array<double,136> coefficients{};coefficients[73]=NAN;
    std::vector<uint32_t> scratch(SCRATCH_BYTES/sizeof(uint32_t));
    uint64_t status=0;
    Descriptor descriptor{};descriptor.scratch=reinterpret_cast<uint64_t>(scratch.data());
    descriptor.backward.cotangents=reinterpret_cast<uint64_t>(coefficients.data());
    descriptor.backward.status=reinterpret_cast<uint64_t>(&status);
    semantic_policy_backward(descriptor);
    require(status==1,"actual backward did not diagnose its nonfinite original cotangent");
    require_content_trap([&] { semantic_transition_execute(descriptor); },
        "actual backward wrapper returned a failed derivative without an integrity trap");
}
#endif

static void active_continuation_preserves_physical_holes(uint64_t role) {
    PublicationTensorLayout layout{};layout.role=role;layout.element_bytes=4;
    layout.scalar_type=6;layout.rank=2;layout.logical_axis=0;
    layout.dimensions[0]=66;layout.dimensions[1]=2;
    layout.strides_bytes[0]=8;layout.strides_bytes[1]=4;
    if(role==51 || role==52) {
        layout.rank=4;layout.logical_axis=2;
        layout.dimensions[0]=1;layout.dimensions[1]=1;layout.dimensions[2]=66;layout.dimensions[3]=2;
        layout.strides_bytes[0]=layout.strides_bytes[1]=528;layout.strides_bytes[2]=8;layout.strides_bytes[3]=4;
    }
    struct Table { TensorLayoutTableHeader header; PublicationTensorLayout layout; ActiveRow row; };
    constexpr uint64_t empty_table_bytes=sizeof(TensorLayoutTableHeader)+sizeof(PublicationTensorLayout);
    Table original{{1,1,0,0},layout,{}}, destination=original;
    Table incoming{{1,1,1,1},layout,{33,65,129,3}};
    std::array<float,132> old_values{},next_values{},input_values{};
    std::array<RawFeedbackRecord,34> feedback{};feedback[33].valid=1;
    next_values.fill(-11.0f);input_values.fill(-7.0f);
    input_values[66]=17.0f;input_values[67]=23.0f;
    PublicationStorageEntry storage[]={
        {reinterpret_cast<uint64_t>(old_values.data()),sizeof(old_values),1},
        {reinterpret_cast<uint64_t>(&original),sizeof(original),1},
        {reinterpret_cast<uint64_t>(next_values.data()),sizeof(next_values),1},
        {reinterpret_cast<uint64_t>(&destination),sizeof(destination),1},
        {reinterpret_cast<uint64_t>(input_values.data()),sizeof(input_values),1},
        {reinterpret_cast<uint64_t>(&incoming),sizeof(incoming),1},
        {reinterpret_cast<uint64_t>(feedback.data()),sizeof(feedback),1}};
    const auto range=[](uint64_t role,uint64_t slot,uint64_t bytes,uint64_t count) {
        PublicationRange value{};value.role=role;value.storage_slot=slot;
        value.generation=1;value.length_bytes=bytes;value.logical_end=count;return value;
    };
    PublicationRange old_ranges[]={range(role,0,0,0),range(55,1,empty_table_bytes,0),range(15,6,sizeof(feedback),1)};
    PublicationRange next_ranges[]={range(role,2,0,0),range(55,3,empty_table_bytes,0),old_ranges[2]};
    PublicationRange inputs[]={range(role,4,sizeof(input_values),1),range(55,5,sizeof(incoming),0)};
    PublicationContract contract{};contract.abi=1;contract.window_capacity=32;
    contract.feedback_capacity=34;contract.prefix_capacity=64;contract.max_position=192;contract.range_capacity=3;
    PublicationBank base{},next{};base.header.abi=next.header.abi=1;
    base.header.publication_word=2;next.header.publication_word=3;
    base.header.sealed_epoch=next.header.sealed_epoch=1;
    base.header.range_count=next.header.range_count=3;
    PublicationControl control{};control.abi=1;control.word=3;
    control.banks[0]=reinterpret_cast<uint64_t>(&base);control.banks[1]=reinterpret_cast<uint64_t>(&next);
    control.directories[0]=reinterpret_cast<uint64_t>(old_ranges);
    control.directories[1]=reinterpret_cast<uint64_t>(next_ranges);
    control.reader_counts[1]=1;control.contract=reinterpret_cast<uint64_t>(&contract);
    control.storage=reinterpret_cast<uint64_t>(storage);control.storage_count=7;
    PendingContinuation pending{};pending.base_word=2;
    pending.ranges=reinterpret_cast<uint64_t>(inputs);pending.range_count=2;
    TensorLayoutTableView incoming_table{};
    require(publication_tensor_table(control,inputs,2,&incoming_table)
        && publication_validate_active_origin(control,contract,base,old_ranges,incoming_table),
        "sparse active rows did not originate from the actual predecessor feedback slots");
    require(publication_apply_continuation(control,base,next,pending)==0,"active continuation copy refused");
    require(next_values[66]==17.0f && next_values[67]==23.0f,
        "active continuation copied a dense prefix instead of its mapped physical row");
    require(next_values[0]==-11.0f,"active continuation copied an unused physical row");
    require(destination.row.physical_row==33 && destination.header.active_row_count==1
        && next_ranges[0].logical_end==1,"active publication changed its sparse row map or semantic count");
    TensorLayoutTableView published_table{};
    const auto seal_active_publication=[&] {
        require(publication_tensor_table(control,next_ranges,3,&published_table),
            "active publication table refused");
        require(publication_range_digest(control,next_ranges[1],nullptr,next_ranges[1].digest)==0,
            "active publication table seal refused");
        require(publication_range_digest(control,next_ranges[0],&layout,next_ranges[0].digest,
            &published_table)==0,"mapped active publication seal refused");
    };
    seal_active_publication();
    PublicationLease lease{};lease.abi=1;lease.active=1;lease.word=3;lease.bank=1;lease.epoch=1;
    const auto consume=[&] { semantic_publication_content_guard(reinterpret_cast<uint64_t>(&control),
        next_ranges[0].role,next_ranges[0].index,layout,1,reinterpret_cast<uint64_t>(&lease)); };
    consume();
    next_values[0]=999.0f;consume();
    require_content_trap([&] { next_values[66]+=1.0f;consume(); },
        "mapped active value mutation did not reach the actual content guard");
    feedback[33].valid=0;
    incoming.header.active_row_count=0;inputs[0].logical_end=0;inputs[1].length_bytes=empty_table_bytes;
    require(publication_tensor_table(control,inputs,2,&incoming_table)
        && publication_validate_active_origin(control,contract,base,old_ranges,incoming_table),
        "computed-empty active rows did not match the empty predecessor inputs");
    require(publication_apply_continuation(control,base,next,pending)==0,
        "computed-empty full-capacity continuation refused");
    require(next_ranges[0].length_bytes==0 && !destination.header.active_row_count,
        "computed-empty publication retained a semantic physical extent");
    seal_active_publication();
    consume();
}

static void fixed_continuation_buffers_keep_semantic_intervals(uint64_t boundary_role) {
    PublicationTensorLayout attention{};attention.role=4;attention.element_bytes=4;
    attention.scalar_type=6;attention.rank=4;attention.logical_axis=2;
    attention.dimensions[0]=1;attention.dimensions[1]=1;attention.dimensions[2]=64;attention.dimensions[3]=1;
    attention.strides_bytes[0]=256;attention.strides_bytes[1]=256;
    attention.strides_bytes[2]=4;attention.strides_bytes[3]=4;
    auto suffix=attention;suffix.dimensions[2]=32;suffix.strides_bytes[0]=suffix.strides_bytes[1]=128;
    PublicationTensorLayout boundary{};boundary.role=boundary_role;boundary.element_bytes=4;
    boundary.scalar_type=6;boundary.rank=2;boundary.logical_axis=UINT64_MAX;
    boundary.dimensions[0]=boundary.dimensions[1]=1;boundary.strides_bytes[0]=boundary.strides_bytes[1]=4;
    struct Table { TensorLayoutTableHeader header; PublicationTensorLayout layouts[2]; ActiveRow rows[34]; };
    constexpr uint64_t empty_table_bytes=sizeof(TensorLayoutTableHeader)+2*sizeof(PublicationTensorLayout);
    constexpr uint64_t two_row_table_bytes=empty_table_bytes+2*sizeof(ActiveRow);
    Table original{{1,2,0,0},{attention,boundary},{}},destination=original;
    Table incoming{{1,2,1,2},{suffix,boundary},{{0,0,7,1},{1,1,8,1}}};
    std::array<SourceSlot,64> prefix{};std::array<SourceSlot,32> completed{};
    std::array<uint64_t,4> identity{};std::array<RawFeedbackRecord,2> feedback{};
    std::array<float,64> cache{},next_cache{};cache.fill(-11.0f);next_cache.fill(-11.0f);
    std::array<float,32> output{};output.fill(-7.0f);output[0]=17.0f;output[1]=23.0f;
    float old_boundary=1.0f,new_boundary=0.0f,output_boundary=42.0f;
    std::array<uint8_t,4> authority{1,0,1,0},next_authority{};
    std::vector<PublicationStorageEntry> storage;
    const auto allocate=[&](auto& value) { storage.push_back({reinterpret_cast<uint64_t>(&value),sizeof(value),1});
        return uint64_t(storage.size()-1); };
    const auto range=[](uint64_t role,uint64_t slot,uint64_t bytes,uint64_t begin=0,uint64_t end=0) {
        PublicationRange value{};value.role=role;value.storage_slot=slot;value.generation=1;
        value.length_bytes=bytes;value.logical_begin=begin;value.logical_end=end;return value;
    };
    PublicationRange old_ranges[]={range(1,allocate(prefix),7*sizeof(SourceSlot),0,7),
        range(3,allocate(identity),sizeof(identity)),range(4,allocate(cache),sizeof(cache),0,7),
        range(boundary_role,allocate(old_boundary),sizeof(old_boundary)),range(15,allocate(feedback),sizeof(feedback)),
        range(39,allocate(authority),sizeof(authority)),range(55,allocate(original),empty_table_bytes)};
    std::array<PublicationRange,7> next_ranges{};
    std::copy(old_ranges,old_ranges+7,next_ranges.begin());
    next_ranges[2].storage_slot=allocate(next_cache);next_ranges[3].storage_slot=allocate(new_boundary);
    next_ranges[5].storage_slot=allocate(next_authority);
    next_ranges[6].storage_slot=allocate(destination);
    PublicationRange inputs[]={range(1,allocate(completed),2*sizeof(SourceSlot),7,9),
        range(4,allocate(output),sizeof(output),7,9),range(boundary_role,allocate(output_boundary),sizeof(output_boundary)),
        old_ranges[5],range(55,allocate(incoming),two_row_table_bytes)};
    PublicationContract contract{};contract.abi=1;contract.window_capacity=32;
    contract.feedback_capacity=2;contract.prefix_capacity=64;contract.max_position=128;contract.range_capacity=7;
    PublicationBank base{},next{};base.header.abi=next.header.abi=1;
    base.header.publication_word=2;next.header.publication_word=3;
    base.header.sealed_epoch=next.header.sealed_epoch=1;base.header.range_count=next.header.range_count=7;
    base.header.prefix_extent=7;next.header.prefix_extent=9;
    for(uint64_t i=0;i<2;++i) {
        auto& slot=base.source[i];slot.token=11+i;slot.logical_position=7+i;slot.kind=slot.valid=slot.provenance=1;
        completed[i]=slot;completed[i].committed=completed[i].recomputed=1;
    }
    PublicationControl control{};control.abi=1;control.contract=reinterpret_cast<uint64_t>(&contract);
    control.directories[0]=reinterpret_cast<uint64_t>(old_ranges);
    control.directories[1]=reinterpret_cast<uint64_t>(next_ranges.data());
    control.storage=reinterpret_cast<uint64_t>(storage.data());control.storage_count=storage.size();
    PendingContinuation pending{};pending.abi=1;pending.base_word=2;pending.transition_kind=1;
    ResidentTextServices text_services;pending.text=text_services.binding();
    pending.ranges=reinterpret_cast<uint64_t>(inputs);pending.range_count=5;
    require(publication_validate_continuation(control,contract,base,pending,9)==0,
        "fixed continuation refused a non-positioned boundary interval");
    incoming.layouts[0].dimensions[2]=2;
    require(publication_validate_continuation(control,contract,base,pending,9)!=0,
        "continuation accepted an extent-narrowed suffix buffer");
    incoming.layouts[0]=suffix;
    original.layouts[1].logical_axis=incoming.layouts[1].logical_axis=0;
    require(publication_validate_continuation(control,contract,base,pending,9)!=0,
        "continuation accepted a positioned recurrent boundary");
    original.layouts[1]=incoming.layouts[1]=boundary;
    require(publication_apply_continuation(control,base,next,pending)==0,"fixed continuation copy refused");
    require(next_cache[6]==-11.0f && next_cache[7]==17.0f && next_cache[8]==23.0f && next_cache[9]==-11.0f,
        "fixed suffix copy included padding or lost its absolute destination interval");
    require(new_boundary==42.0f && !next_ranges[3].logical_begin && !next_ranges[3].logical_end,
        "non-positioned boundary acquired a prefix interval");
}

static void original_content_seal_accepts_equal_strided_producer();

static void scalar_model_content_seals_one_cell_without_reshaping() {
    const uint64_t widths[]={0,1,4,8,2,2,4,8,1};
    for(uint64_t type=1;type<=8;++type) {
        alignas(8) std::array<uint8_t,8> original{1,2,3,4,5,6,7,8};
        PublicationStorageEntry storage{reinterpret_cast<uint64_t>(original.data()),widths[type],1};
        PublicationControl control{};control.abi=1;
        control.storage=reinterpret_cast<uint64_t>(&storage);control.storage_count=1;
        PublicationRange range{};range.role=18;range.generation=1;range.length_bytes=widths[type];
        PublicationTensorLayout scalar{};scalar.role=18;scalar.element_bytes=widths[type];
        scalar.scalar_type=type;scalar.logical_axis=UINT64_MAX;
        require(publication_range_digest(control,range,&scalar,range.digest)==0,
            "native scalar model content could not be sealed");
        uint64_t capacity=0;
        require(publication_step_layout_capacity(scalar,&capacity) && capacity==widths[type],
            "scalar step capacity is not exactly one original element");
        PublicationTensorBytes view{};uint64_t content_bytes=0;
        require(publication_tensor_view(control,range,&scalar,&view,&content_bytes)==0 &&
            content_bytes==widths[type],"scalar publication became empty or acquired an artificial axis");
        for(uint64_t i=0;i<content_bytes;++i)
            require(view[sizeof(view.prefix)+i]==original[i],"scalar traversal missed its original cell");
        semantic_tensor_content_witness(reinterpret_cast<uint64_t>(original.data()),range.length_bytes,
            range,scalar,reinterpret_cast<uint64_t>(range.digest),1);
        require_content_trap([&] {
            original[0]^=1;
            semantic_tensor_content_witness(reinterpret_cast<uint64_t>(original.data()),range.length_bytes,
                range,scalar,reinterpret_cast<uint64_t>(range.digest),1);
        });
        // The mutation above belongs to the isolated trap child. Both rank
        // comparisons below must hash the same unchanged parent payload.
        std::array<uint64_t,4> scalar_digest{};
        require(publication_range_digest(control,range,&scalar,scalar_digest.data())==0 &&
            publication_identity_equal(range.digest,scalar_digest.data()),
            "integrity child changed the parent's scalar content");
        auto vector=scalar;vector.rank=1;vector.dimensions[0]=1;vector.strides_bytes[0]=widths[type];
        std::array<uint64_t,4> vector_digest{};
        require(publication_range_digest(control,range,&vector,vector_digest.data())==0 &&
            !publication_identity_equal(range.digest,vector_digest.data()),
            "scalar content identity lost its distinction from a length-one vector");
        auto invalid=scalar;invalid.dimensions[0]=1;
        require(publication_range_digest(control,range,&invalid,vector_digest.data())!=0,
            "scalar accepted an undeclared dimension");
        invalid=scalar;invalid.strides_bytes[0]=widths[type];
        require(publication_range_digest(control,range,&invalid,vector_digest.data())!=0,
            "scalar accepted an undeclared stride");
        invalid=scalar;invalid.logical_axis=0;
        require(publication_range_digest(control,range,&invalid,vector_digest.data())!=0,
            "scalar accepted a nonexistent logical axis");
        auto truncated=range;truncated.length_bytes=widths[type]-1;
        require(publication_range_digest(control,truncated,&scalar,vector_digest.data())!=0,
            "scalar accepted a truncated original cell");
    }
}

static void empty_model_storage_keeps_metadata_without_device_cells() {
    PublicationStorageEntry storage[2]{{0,0,1},{0,0,1}};
    PublicationControl control{};control.abi=1;
    control.storage=reinterpret_cast<uint64_t>(storage);control.storage_count=2;
    require(publication_validate_storage(control)==0,"genuine zero-byte owners were rejected");
    for(uint64_t role : {18,19,20}) {
        PublicationRange range{};range.role=role;range.generation=1;
        PublicationTensorLayout layout{};layout.role=role;layout.element_bytes=4;layout.scalar_type=6;
        layout.rank=3;layout.logical_axis=UINT64_MAX;
        layout.dimensions[0]=2;layout.dimensions[2]=3;
        layout.strides_bytes[0]=12;layout.strides_bytes[1]=12;layout.strides_bytes[2]=4;
        PublicationTensorBytes view{};uint64_t content_bytes=99;
        require(publication_tensor_view(control,range,&layout,&view,&content_bytes)==0 && content_bytes==0,
            "empty model buffer fabricated cells or lost its exact zero span");
        require(publication_range_digest(control,range,&layout,range.digest)==0,
            "empty model buffer lost its metadata seal");
        semantic_tensor_content_witness(0,0,range,layout,reinterpret_cast<uint64_t>(range.digest),1);
        auto next=range;next.storage_slot=1;
        require(publication_validate_destinations(control,&range,&next,1)==0,
            "zero-byte destination owners were confused with invalid pointers");
        auto changed=layout;changed.dimensions[0]=4;
        std::array<uint64_t,4> digest{};
        require(publication_range_digest(control,range,&changed,digest.data())==0 &&
            !publication_identity_equal(range.digest,digest.data()),"empty shape did not enter its content identity");
        for(uint32_t fault=0;fault<5;++fault) {
            auto invalid=range;
            if(fault==0)invalid.offset_bytes=1;
            if(fault==1)invalid.length_bytes=1;
            if(fault==2)invalid.generation=2;
            if(fault==3)invalid.storage_slot=2;
            if(fault==4)invalid.role=43;
            require(publication_tensor_view(control,invalid,&layout,&view,&content_bytes)!=0,
                "empty range accepted invalid ownership, bounds or role");
        }
        changed=layout;changed.dimensions[1]=1;
        require(publication_tensor_view(control,range,&changed,&view,&content_bytes)!=0,
            "nonempty model tensor borrowed a zero-byte owner");
    }
    storage[0].generation=0;
    require(publication_validate_storage(control)!=0,"empty storage lost its generation guard");
    storage[0].generation=1;storage[0].bytes=4;
    require(publication_validate_storage(control)!=0,"nonnull extent accepted a null allocation");
    storage[0].bytes=0;storage[0].pointer=8;
    require(publication_validate_storage(control)!=0,"zero-byte owner accepted an invented sentinel pointer");
}

struct StepPublicationFixture {
    static constexpr uint64_t kTaskQueryCount=3;
    static constexpr uint64_t kFeedbackCapacity=kTaskQueryCount;
    static constexpr uint64_t kActiveRowCapacity=32+kFeedbackCapacity;
    static constexpr uint64_t kStepInputCount=26;
    static constexpr uint64_t kModelSchemaBegin=104;
    static constexpr std::array<uint8_t,8> kModelSchema{44,0,0x80,0xff,0,0,0,0};
    static constexpr uint64_t kModelContractBytes=kModelSchemaBegin+kModelSchema.size();
    RefusalExecution execution;
    std::vector<std::vector<uint64_t>> allocations;
    std::vector<PublicationStorageEntry> storage;
    std::array<PublicationRoleCount,55> roles{};
    std::vector<PublicationTensorLayout> layouts;
    std::vector<PublicationRange> directories[2],pending_ranges;
    PublicationBank banks[2]{};
    PublicationContract contract{};
    PublicationControl control{};
    PublicationLease lease{};
    PendingContinuation pending{};
    std::array<uint64_t,34> task{};
    uint64_t terminal_token=17;

    uint64_t allocate(uint64_t bytes) {
        allocations.emplace_back((bytes+7)/8);
        storage.push_back({reinterpret_cast<uint64_t>(allocations.back().data()),allocations.back().size()*8,1});
        return storage.size()-1;
    }
    PublicationRange& range(uint64_t bank,uint64_t role) {
        return *const_cast<PublicationRange*>(publication_find_range(directories[bank].data(),directories[bank].size(),role));
    }
    void* data(const PublicationRange& range) { return reinterpret_cast<void*>(storage[range.storage_slot].pointer+range.offset_bytes); }
    explicit StepPublicationFixture(bool drain_required=false, std::vector<uint64_t> prefix_tokens={},
            const PublicationTensorLayout* original_layout=nullptr,const void* original_data=nullptr,
            uint64_t original_bytes=0) {
        require((original_layout && original_data && original_bytes) ||
            (!original_layout && !original_data && !original_bytes),
            "step fixture original model override is incomplete");
        execution.books[8]=execution.books.size()*8;
        uint64_t tensor_count=0;
        for(uint64_t role=1;role<=55;++role)if(publication_tensor_role(role))++tensor_count;
        for(uint64_t role=1;role<=55;++role) {
            roles[role-1]={role,role==16 ? kTaskQueryCount : 1ULL};
            for(uint64_t index=0;index<roles[role-1].count;++index) {
                uint64_t length=8,capacity=8;
                PublicationTensorLayout layout{};
                if(publication_tensor_role(role)) {
                    layout.role=role;layout.index=index;layout.element_bytes=4;layout.scalar_type=6;
                    layout.rank=1;layout.logical_axis=UINT64_MAX;layout.dimensions[0]=2;layout.strides_bytes[0]=4;
                    if(role==4 || role==5 || role==51 || role==52) {
                        layout.rank=4;layout.logical_axis=2;layout.dimensions[0]=layout.dimensions[1]=1;
                        layout.dimensions[2]=role<51 ? 64 : kActiveRowCapacity;layout.dimensions[3]=2;
                        layout.strides_bytes[3]=4;layout.strides_bytes[2]=8;
                        layout.strides_bytes[1]=layout.strides_bytes[0]=layout.dimensions[2]*8;
                        capacity=length=layout.strides_bytes[0];
                    } else if(role==53 || role==54) {
                        layout.logical_axis=0;layout.dimensions[0]=kActiveRowCapacity;
                        capacity=length=kActiveRowCapacity*4;
                    }
                    if(role==18 && original_layout) {
                        layout=*original_layout;capacity=length=original_bytes;
                    }
                    if(role>=51 && role<=54)length=0;
                    layouts.push_back(layout);
                }
                if(role==1){length=0;capacity=64*sizeof(SourceSlot);}
                if(role==2){length=0;capacity=64*sizeof(TokenProvenanceRecord);}
                if(role==3)capacity=length=32;
                if(role==14)capacity=length=sizeof(CompletionCoverage);
                if(role==15)capacity=length=kFeedbackCapacity*sizeof(RawFeedbackRecord);
                if(role==30){length=sizeof(IntentQueueHeader);capacity=length+(drain_required ? sizeof(IntentEntry) : 0);}
                if(role==31){length=1;if(drain_required)capacity=32;}
                if(role==33)capacity=length=sizeof(AttemptReceipt);
                if(role==44)capacity=length=kModelContractBytes;
                if(role==48)capacity=length=sizeof(execution.books);
                if(role==55){length=sizeof(TensorLayoutTableHeader)+tensor_count*sizeof(PublicationTensorLayout);
                    capacity=length+kActiveRowCapacity*sizeof(ActiveRow);}
                PublicationRange item{};item.role=role;item.index=index;item.generation=1;
                item.length_bytes=length;item.storage_slot=allocate(capacity);
                directories[0].push_back(item);
                if(publication_mutable_role(role) && role!=1 && role!=2 && role!=31)
                    item.storage_slot=allocate(capacity);
                directories[1].push_back(item);
            }
        }
        for(uint64_t bank=0;bank<2;++bank) {
            if(original_data)std::memcpy(data(range(bank,18)),original_data,original_bytes);
            range(bank,1).length_bytes=prefix_tokens.size()*sizeof(SourceSlot);
            range(bank,1).logical_end=prefix_tokens.size();
            range(bank,4).logical_end=range(bank,5).logical_end=prefix_tokens.size();
            range(bank,2).length_bytes=(prefix_tokens.size()+1)*sizeof(TokenProvenanceRecord);
            range(bank,2).logical_end=prefix_tokens.size()+1;
            auto* provenance=static_cast<TokenProvenanceRecord*>(data(range(bank,2)));
            for(uint64_t position=0;position<=prefix_tokens.size();++position) {
                const uint64_t token=position<prefix_tokens.size() ? prefix_tokens[position] : 17;
                provenance[position].token=token;provenance[position].logical_position=position;
                provenance[position].origin_logical_digest[0]=1;
                provenance[position].authority_closure_digest[0]=2;
                if(position<prefix_tokens.size())
                    static_cast<SourceSlot*>(data(range(bank,1)))[position]={token,position,1,1,1,1,1,position};
            }
            auto* table=static_cast<TensorLayoutTableHeader*>(data(range(bank,55)));
            *table={1,layouts.size(),1,0};std::copy(layouts.begin(),layouts.end(),reinterpret_cast<PublicationTensorLayout*>(table+1));
            auto* queue=static_cast<IntentQueueHeader*>(data(range(bank,30)));
            queue->abi=1;queue->payload_used_bytes=queue->effect_length_bytes=1;queue->payload_capacity_bytes=8;
            if(drain_required){queue->capacity=1;queue->payload_capacity_bytes=32;}
            std::copy(kModelSchema.begin(),kModelSchema.end(),
                static_cast<uint8_t*>(data(range(bank,44)))+kModelSchemaBegin);
            std::memcpy(data(range(bank,48)),execution.books.data(),sizeof(execution.books));
            banks[bank].header.range_count=directories[bank].size();
            banks[bank].header.semantic_owner=91;banks[bank].header.semantic_generation=1;
            banks[bank].header.model_generation=3;banks[bank].header.authority_generation=4;banks[bank].header.fuel=9;
            banks[bank].header.prefix_extent=prefix_tokens.size();
            banks[bank].source[0]={17,prefix_tokens.size(),1,1,1,0,0,prefix_tokens.size()};
            banks[bank].source[1]={MASK_TOKEN,prefix_tokens.size()+1,2,0,1,0,0,0};
            if(drain_required){banks[bank].header.terminal=1;banks[bank].source[1]={};}
            std::copy(execution.initial.words+13,execution.initial.words+16,banks[bank].header.semantic_extents);
            std::copy(execution.initial.words+16,execution.initial.words+20,banks[bank].header.semantic_digest);
            control.banks[bank]=reinterpret_cast<uint64_t>(&banks[bank]);
            control.directories[bank]=reinterpret_cast<uint64_t>(directories[bank].data());
        }
        task[0]=4;task[1]=91;task[6]=11;task[10]=task[14]=21;
        task[18]=1;task[19]=task[20]=2;
        task[22]=task[23]=task[24]=0;
        task[25]=14;task[26]=7;task[27]=1;task[28]=45;task[29]=15;task[30]=1;
        task[31]=task[32]=task[33]=7;
        execution.descriptor.task=reinterpret_cast<uint64_t>(task.data());
        contract.abi=1;contract.range_capacity=directories[0].size();contract.prefix_capacity=64;
        contract.window_capacity=32;contract.feedback_capacity=kFeedbackCapacity;contract.max_position=96;
        contract.model_generation=3;contract.authority_generation=4;contract.semantic_owner=91;
        contract.model_contract_layout={kModelSchemaBegin,kModelSchema.size(),0,32,40,72};
        if(drain_required){contract.terminal_tokens=reinterpret_cast<uint64_t>(&terminal_token);contract.terminal_token_count=1;}
        contract.role_count=roles.size();contract.role_counts=reinterpret_cast<uint64_t>(roles.data());
        for(const auto& original:directories[0])if(original.role==1 || original.role==39 || original.role==55 ||
            (original.role>=4 && original.role<=13)) {
            auto item=original;item.storage_slot=allocate(storage[original.storage_slot].bytes);
            std::memcpy(data(item),data(original),item.length_bytes);pending_ranges.push_back(item);
        }
        control.abi=1;control.contract=reinterpret_cast<uint64_t>(&contract);
        control.storage=reinterpret_cast<uint64_t>(storage.data());control.storage_count=storage.size();
        control.continuation=reinterpret_cast<uint64_t>(&pending);
        pending.ranges=reinterpret_cast<uint64_t>(pending_ranges.data());pending.range_count=pending_ranges.size();
        execution.descriptor.publication.control=reinterpret_cast<uint64_t>(&control);
        const uint64_t initialization_status=publication_initialize(execution.descriptor,control);
        if(initialization_status)std::cerr<<"initial prefix extent "<<prefix_tokens.size()<<", status "<<initialization_status<<'\n';
        require(initialization_status==0,"step fixture actual publication initialization refused");
        require(publication_acquire(control,lease)==0,"step fixture initial reader refused");
    }
    void publish() {
        const uint64_t word=lease.word;auto& base=banks[lease.bank];pending.base_word=word;
        uint64_t end=0;
        require(publication_validate_source(base,contract,&end)==0,"step fixture source validation failed");
        for(auto& item:pending_ranges)if(item.role==1 || item.role==4 || item.role==5) {
            item.logical_begin=base.header.prefix_extent;item.logical_end=end;
            if(item.role==1) {
                item.length_bytes=(end-base.header.prefix_extent)*sizeof(SourceSlot);
                auto* destination=static_cast<SourceSlot*>(data(item));
                for(uint64_t position=base.header.prefix_extent;position<end;++position)
                    for(const auto& source:base.source)if(source.kind==1 && source.logical_position==position) {
                        destination[position-base.header.prefix_extent]=source;
                        destination[position-base.header.prefix_extent].committed=1;
                        destination[position-base.header.prefix_extent].recomputed=1;
                    }
            }
        }
        for(auto& item:pending_ranges)if(item.role>=6 && item.role<=13) {
            auto* values=static_cast<float*>(data(item));values[0]=float(10+word+item.role);values[1]=-values[0];
        }
        State state=base.state;state.blocks=0;
        const semantic_graph::ResidentHandle root{91,semantic_graph::kRootKind,base.header.semantic_slot,base.header.semantic_generation};
        require(publication_publish(execution.descriptor,base,word,end,root,&state,base.receipts,true)==0,
            "step fixture actual native publication refused");
        require(publication_release(control,lease)==0 && publication_acquire(control,lease)==0,
            "step fixture next reader refused");
    }
};

static void original_content_seal_accepts_equal_strided_producer() {
    PublicationTensorLayout canonical{};canonical.role=18;canonical.element_bytes=4;
    canonical.scalar_type=6;canonical.rank=2;canonical.logical_axis=UINT64_MAX;
    canonical.dimensions[0]=2;canonical.dimensions[1]=3;
    canonical.strides_bytes[0]=12;canonical.strides_bytes[1]=4;
    std::array<float,6> original{1,2,3,4,5,6};
    StepPublicationFixture fixture(false,{},&canonical,original.data(),sizeof(original));
    auto& sealed=fixture.range(fixture.lease.bank,18);
    auto* published=static_cast<float*>(fixture.data(sealed));
    std::array<uint64_t,4> backing{};
    require(publication_backing_digest(fixture.control,sealed,backing.data())==0 &&
        publication_identity_equal(backing.data(),sealed.backing_digest),
        "canonical original model backing could not be sealed");
    semantic_publication_content_guard(reinterpret_cast<uint64_t>(&fixture.control),
        sealed.role,sealed.index,canonical,1,reinterpret_cast<uint64_t>(&fixture.lease));
    const std::array<uint64_t,4> original_digest{
        sealed.digest[0],sealed.digest[1],sealed.digest[2],sealed.digest[3]};

    const auto verify=[&](float* data,uint64_t length,uint64_t column_stride) {
        auto range=sealed;range.length_bytes=length;
        auto layout=canonical;layout.strides_bytes[0]=4;layout.strides_bytes[1]=column_stride;
        semantic_tensor_content_witness(reinterpret_cast<uint64_t>(data),length,range,layout,
            reinterpret_cast<uint64_t>(sealed.digest),1);
        require(publication_identity_equal(sealed.digest,original_digest.data()),
            "verification replaced the original model content seal");
    };
    std::array<float,6> transposed{1,4,2,5,3,6};
    verify(transposed.data(),sizeof(transposed),8);
    std::array<float,12> padded{1,4,-77,-77,2,5,-77,-77,3,6,-77,-77};
    constexpr uint64_t padded_extent=10*sizeof(float);
    verify(padded.data(),padded_extent,16);
    reinterpret_cast<uint8_t*>(padded.data())[2*sizeof(float)]^=1;
    verify(padded.data(),padded_extent,16);

    require_content_trap([&] {
        reinterpret_cast<uint8_t*>(padded.data())[5*sizeof(float)]^=1;
        verify(padded.data(),padded_extent,16);
    });
    require_content_trap([&] {
        reinterpret_cast<uint8_t*>(published)[0]^=1;
        semantic_publication_content_guard(reinterpret_cast<uint64_t>(&fixture.control),
            sealed.role,sealed.index,canonical,1,reinterpret_cast<uint64_t>(&fixture.lease));
    });

    for(uint32_t field=0;field<4;++field) {
        auto range=sealed;auto layout=canonical;
        if(field==0)range.role=layout.role=19;
        if(field==1)range.index=layout.index=1;
        if(field==2)layout.scalar_type=2;
        if(field==3)range.logical_begin=range.logical_end=1;
        std::array<uint64_t,4> changed{};
        require(publication_range_digest(fixture.control,range,&layout,changed.data())==0,
            "changed content descriptor failed its actual digest traversal");
        require(!publication_identity_equal(changed.data(),sealed.digest),
            "original model seal omitted role, index, scalar type, or logical interval");
    }
}

static void resident_numerical_refusal_precedes_publication() {
    StepPublicationFixture fixture;
    auto& execution=fixture.execution;auto& descriptor=execution.descriptor;auto& state=execution.state;
    auto& control=fixture.control;auto& lease=fixture.lease;auto& pending=fixture.pending;
    auto& bank=fixture.banks[lease.bank];
    bank.header.proposal=7;bank.header.stream_serial=11;bank.header.family_id=2;
    require(publication_logical_digest(control,bank)==0 && publication_descriptor_digest(control,bank)==0,
        "resident numerical refusal fixture could not authenticate its selected model coordinates");
    uint8_t numerical=0;
    pending.abi=1;pending.base_word=lease.word;pending.transition_kind=1;
    pending.model_generation=bank.header.model_generation;pending.authority_generation=bank.header.authority_generation;
    pending.numerical_admissibility=reinterpret_cast<uint64_t>(&numerical);
    descriptor.publication.lease=reinterpret_cast<uint64_t>(&lease);
    control.reader_gate=1;control.refusal=77;
    state.next_proposal=uint64_t(UINT32_MAX)+1;
    semantic_transition_execute(descriptor);
    require(state.status==5 && !state.blocks && !state.retired_roots_mask && control.refusal==77 && control.reader_gate==1,
        "resident numerical refusal entered publication or retirement");
    require(state.model_generation==3 && state.family_id==2 && state.stream_serial==11,
        "resident numerical refusal did not authenticate and retain its acquired bank coordinates");
    execution.check_protected_root();
    require_content_trap([&] { ++lease.epoch;semantic_transition_execute(descriptor); },
        "numerical refusal accepted a stale acquired lease");
    require_content_trap([&] { ++bank.header.semantic_generation;semantic_transition_execute(descriptor); },
        "numerical refusal accepted a stale protected root");
    require_content_trap([&] { numerical=2;semantic_transition_execute(descriptor); },
        "non-boolean resident numerical service did not reach the integrity trap");
    require_content_trap([&] { bank.header.proposal=uint64_t(UINT32_MAX)+1;semantic_transition_execute(descriptor); },
        "numerical refusal accepted an exhausted acquired proposal coordinate");

    const State accepted_state=state;
    const auto reject_metadata=[&](auto corrupt) {
        for(bool published : {false,true}) {
            state=accepted_state;corrupt(state);
            descriptor.publication.control=published ? reinterpret_cast<uint64_t>(&control) : 0;
            semantic_transition_execute(descriptor);
            require(state.status==1 && !state.blocks && control.refusal==77 && control.reader_gate==1,
                "invalid cold binding metadata reached proposal execution");
        }
    };
    reject_metadata([](State& value) { ++value.catalogue_generation; });
    reject_metadata([](State& value) { value.catalogue_digest[0]^=1; });
    reject_metadata([](State& value) { value.binding_digest[0]^=1; });
    state=accepted_state;state.next_proposal=uint64_t(UINT32_MAX)+1;
    descriptor.publication.control=0;
    semantic_transition_execute(descriptor);
    require(state.status==1 && !state.blocks,"cold execution accepted an exhausted proposal coordinate");
}

static void recompute_execution_preserves_proposal_state(bool drain_required) {
    StepPublicationFixture fixture(drain_required);
    auto& descriptor=fixture.execution.descriptor;
    auto& state=fixture.execution.state;
    const auto parent=fixture.banks[fixture.lease.bank].header;
    const auto snapshot=[&](uint64_t role) {
        const auto& range=fixture.range(fixture.lease.bank,role);
        const auto* bytes=static_cast<const uint8_t*>(fixture.data(range));
        return std::vector<uint8_t>(bytes,bytes+range.length_bytes);
    };
    const auto replay=snapshot(27),intent=snapshot(30),payload=snapshot(31),feedback=snapshot(15);
    require(publication_release(fixture.control,fixture.lease)==0,"recompute fixture initial reader release refused");
    fixture.lease={};uint64_t conditional=0;
    semantic_publication_step_admit(reinterpret_cast<uint64_t>(&fixture.control),
        reinterpret_cast<uint64_t>(&fixture.lease),2,reinterpret_cast<uint64_t>(&conditional),0);
    require(conditional==1 && !fixture.lease.status && fixture.lease.transition_kind==(drain_required ? 3 : 2),
        "recompute admission lost its actual drain override or original requested mode");

    // The same original fixed-capacity producer roster used by continuation
    // admission supplies suffix, recurrent boundary, and active cache outputs.
    fixture.pending_ranges.clear();
    std::vector<PublicationTensorLayout> layouts;
    for(const auto& original:fixture.directories[fixture.lease.bank]) {
        if(original.role!=1 && original.role!=39 && original.role!=55 &&
           !(original.role>=4 && original.role<=13) && !(original.role>=51 && original.role<=54))continue;
        auto item=original;item.storage_slot=fixture.allocate(fixture.storage[original.storage_slot].bytes);
        std::memcpy(fixture.data(item),fixture.data(original),original.length_bytes);
        if(publication_tensor_role(item.role)) {
            const auto found=std::find_if(fixture.layouts.begin(),fixture.layouts.end(),
                [&](const auto& layout){return layout.role==item.role && layout.index==item.index;});
            require(found!=fixture.layouts.end(),"recompute producer has no original tensor layout");
            auto layout=*found;
            if(item.role==4 || item.role==5) {
                layout.dimensions[2]=32;layout.strides_bytes[0]=layout.strides_bytes[1]=32*8;
            }
            item.length_bytes=fixture.storage[item.storage_slot].bytes;
            layouts.push_back(layout);
            auto* values=static_cast<float*>(fixture.data(item));
            for(uint64_t cell=0;cell<item.length_bytes/sizeof(float);++cell)values[cell]=float(item.role+cell+1);
        }
        fixture.pending_ranges.push_back(item);
    }
    auto* table_range=const_cast<PublicationRange*>(publication_find_range(fixture.pending_ranges.data(),
        fixture.pending_ranges.size(),55));
    require(table_range,"recompute producer has no original layout table");
    table_range->length_bytes=sizeof(TensorLayoutTableHeader)+layouts.size()*sizeof(PublicationTensorLayout);
    auto* table=static_cast<TensorLayoutTableHeader*>(fixture.data(*table_range));
    *table={1,layouts.size(),1,0};
    std::copy(layouts.begin(),layouts.end(),reinterpret_cast<PublicationTensorLayout*>(table+1));
    fixture.control.storage=reinterpret_cast<uint64_t>(fixture.storage.data());
    fixture.control.storage_count=fixture.storage.size();
    fixture.pending={};fixture.pending.abi=1;
    fixture.pending.ranges=reinterpret_cast<uint64_t>(fixture.pending_ranges.data());
    fixture.pending.range_count=fixture.pending_ranges.size();
    ResidentTextServices text;
    std::array<ActiveRow,StepPublicationFixture::kActiveRowCapacity> active_rows{};
    active_rows[0]={0,0,0,1};
    uint64_t active_count=1;uint8_t numerical=1;
    const auto* original_feedback=static_cast<const RawFeedbackRecord*>(fixture.data(fixture.range(fixture.lease.bank,15)));
    for(uint64_t slot=0;slot<fixture.contract.feedback_capacity;++slot)if(original_feedback[slot].valid)
        active_rows[active_count++]={1+slot,32+slot,fixture.contract.prefix_capacity+32+slot,3};
    if(!drain_required){text.count=1;text.rows[0]={1,1};
        active_rows[active_count++]={1+fixture.contract.feedback_capacity,1,1,2};}
    ContinuationInputs inputs{text.binding(),reinterpret_cast<uint64_t>(active_rows.data()),
        reinterpret_cast<uint64_t>(&active_count),reinterpret_cast<uint64_t>(&numerical),2,sizeof(uint64_t)};
    semantic_publication_prepare_continuation(reinterpret_cast<uint64_t>(&fixture.control),
        reinterpret_cast<uint64_t>(&fixture.lease),inputs);
    require(fixture.pending.transition_kind==fixture.lease.transition_kind,
        "recompute continuation ignored the actual admitted drain override");
    for(uint64_t role : {6,7}) {
        const auto* boundary=publication_find_range(fixture.pending_ranges.data(),fixture.pending_ranges.size(),role);
        require(boundary && !boundary->logical_begin && !boundary->logical_end &&
            boundary->length_bytes==fixture.storage[boundary->storage_slot].bytes,
            "resident fixed boundary lost its non-positioned producer extent");
    }
    descriptor.publication.lease=reinterpret_cast<uint64_t>(&fixture.lease);
    descriptor.text=text.binding();descriptor.policy={};descriptor.logits=descriptor.support=0;
    const auto receipts=fixture.execution.receipts;
    semantic_transition_execute(descriptor);
    require(!state.status && !fixture.control.refusal && !state.blocks && state.importance_weight==1.0 &&
        state.proposal==parent.proposal && state.next_proposal==parent.proposal && !state.task_evaluation.query_count &&
        std::memcmp(receipts.data(),fixture.execution.receipts.data(),sizeof(receipts))==0,
        "nonproposal execution consumed policy inputs, drew actions, or changed proposal credit");
    const auto& published=fixture.banks[fixture.control.word&1];
    require(published.header.publication_word!=parent.publication_word && published.header.proposal==parent.proposal &&
        published.header.fuel+1==parent.fuel && published.header.cache_generation==parent.cache_generation+1 &&
        published.header.prefix_extent==1 && published.header.terminal==(drain_required ? 2 : 0),
        "recompute did not publish its actual completed prefix and cache state");
    const auto* boundary=static_cast<const float*>(fixture.data(fixture.range(fixture.control.word&1,6)));
    require(boundary[0]==7.0f && boundary[1]==8.0f,
        "recompute publication lost its original numerical continuation output");
    const auto matches=[&](uint64_t role,const auto& expected) {
        const auto& range=fixture.range(fixture.control.word&1,role);
        return range.length_bytes==expected.size() && std::memcmp(fixture.data(range),expected.data(),expected.size())==0;
    };
    require(matches(27,replay) && matches(15,feedback),"nonproposal execution changed replay or learned feedback state");
    if(!drain_required)require(matches(30,intent) && matches(31,payload),"ordinary recompute appended an external intent");
    else {
        const auto& range=fixture.range(fixture.control.word&1,30);
        const auto* queue=static_cast<const IntentQueueHeader*>(fixture.data(range));
        require(queue->count==1 && range.logical_end==1 && queue->payload_used_bytes==17,
            "drain did not append exactly its completed FINAL token payload");
        std::array<uint8_t,17> final_payload{};final_payload[1]=1;final_payload[9]=17;
        require(matches(31,final_payload),"drain FINAL payload differs from the completed original token");
        require(publication_validate_intents(fixture.control,fixture.directories[fixture.control.word&1].data(),
            fixture.directories[fixture.control.word&1].size())==0,"drain FINAL intent failed the native queue validator");
    }
    PublicationStepResult result{};
    semantic_publication_step_result(reinterpret_cast<uint64_t>(&fixture.control),
        reinterpret_cast<uint64_t>(&fixture.lease),reinterpret_cast<uint64_t>(&result),
        reinterpret_cast<uint64_t>(&parent),0);
    require(result.abi==1 && result.advanced==1 && result.word==fixture.control.word && !result.refusal,
        "recompute completion lost its actual publication outcome");
    semantic_publication_step_release(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&fixture.lease));
    require(!fixture.lease.active && !fixture.control.reader_counts[0] && !fixture.control.reader_counts[1] &&
        !fixture.control.reader_gate,"recompute completion leaked its original reader");
}

static void captured_model_seals_follow_actual_reader() {
    StepPublicationFixture fixture;
    const std::array<std::array<uint64_t,2>,3> roster{{{{18,0}},{{19,0}},{{20,0}}}};
    std::array<PublicationRange,4> retained{};uint64_t contract_bytes=UINT64_MAX;
    const auto capture=[&] {
        semantic_publication_model_seals(reinterpret_cast<uint64_t>(&fixture.control),
            reinterpret_cast<uint64_t>(&fixture.lease),reinterpret_cast<uint64_t>(roster.data()),roster.size(),
            reinterpret_cast<uint64_t>(retained.data()),reinterpret_cast<uint64_t>(&contract_bytes),sizeof(contract_bytes));
    };
    for(uint64_t invocation=0;invocation<3;++invocation) {
        if(invocation)fixture.publish();
        capture();
        for(uint64_t i=0;i<roster.size();++i)
            require(std::memcmp(&retained[i],&fixture.range(fixture.lease.bank,roster[i][0]),sizeof(PublicationRange))==0,
                "captured model retained a range from a different publication reader");
        require(std::memcmp(&retained.back(),&fixture.range(fixture.lease.bank,44),sizeof(PublicationRange))==0 &&
            std::memcmp(&contract_bytes,fixture.data(fixture.range(fixture.lease.bank,44)),sizeof(contract_bytes))==0,
            "captured model did not retain the original raw contract and seal");
    }
    require_content_trap([&] {
        *static_cast<uint64_t*>(fixture.data(fixture.range(fixture.lease.bank,44)))^=1;
        capture();
    },"captured model accepted contract bytes changed after publication sealing");
    require_content_trap([&] {
        *static_cast<float*>(fixture.data(fixture.range(fixture.lease.bank,18)))=123.0f;
        capture();
    },"captured model accepted tensor bytes changed after publication sealing");
    require_content_trap([&] {
        semantic_publication_model_seals(reinterpret_cast<uint64_t>(&fixture.control),
            reinterpret_cast<uint64_t>(&fixture.lease),reinterpret_cast<uint64_t>(roster.data()),roster.size(),
            reinterpret_cast<uint64_t>(retained.data()),reinterpret_cast<uint64_t>(&contract_bytes),sizeof(contract_bytes)-1);
    },"captured model accepted a truncated original contract allocation");
    require(publication_release(fixture.control,fixture.lease)==0,"captured model reader release failed");
    require_content_trap(capture,"captured model reused a released publication reader");
    require_content_trap([&] {
        semantic_publication_step_release(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&fixture.lease));
    },"captured step silently accepted duplicate reader release");
}

static void completed_step_result_survives_publication_bank_reuse() {
    StepPublicationFixture fixture;PublicationStepResult result{};
    PublicationLease original{};
    require(publication_acquire(fixture.control,original)==0,"completion result fixture reader refused");
    fixture.publish();
    semantic_publication_step_result(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&original),
        reinterpret_cast<uint64_t>(&result),reinterpret_cast<uint64_t>(&fixture.banks[original.bank].header),0);
    require(result.abi==1 && !result.refusal && result.word==fixture.control.word &&
        std::memcmp(&result.header,&fixture.banks[result.word&1].header,sizeof(PublicationHeader))==0,
        "step result did not preserve the actual post-transition publication header");
    const auto retained=result;
    require(publication_release(fixture.control,original)==0,"completion result fixture reader release refused");
    fixture.publish();fixture.publish();
    require(std::memcmp(&result,&retained,sizeof(result))==0,
        "step result changed after physical publication bank reuse");
    require_content_trap([&] {
        semantic_publication_step_result(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&original),
            reinterpret_cast<uint64_t>(&result),reinterpret_cast<uint64_t>(&fixture.banks[original.bank].header),0);
    },"step result accepted a released publication reader");
    fixture.control.refusal=5;
    semantic_publication_step_result(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&fixture.lease),
        reinterpret_cast<uint64_t>(&result),reinterpret_cast<uint64_t>(&fixture.banks[fixture.lease.bank].header),0);
    require(result.abi==1 && result.refusal==5 && result.word==fixture.lease.word,
        "step result lost an actual non-publishing refusal");
}

static void next_step_stops_after_actual_nonpublication() {
    StepPublicationFixture fixture;PublicationStepResult result{};PublicationLease next{};uint64_t conditional=1;
    semantic_publication_step_result(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&fixture.lease),
        reinterpret_cast<uint64_t>(&result),reinterpret_cast<uint64_t>(&fixture.banks[fixture.lease.bank].header),0);
    const auto readers=fixture.control.reader_counts[fixture.lease.bank];
    semantic_publication_step_admit(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&next),1,
        reinterpret_cast<uint64_t>(&conditional),reinterpret_cast<uint64_t>(&result));
    require(conditional==0 && next.status==7 && !next.active && !next.abi &&
        fixture.control.reader_counts[fixture.lease.bank]==readers,
        "later step acquired another invocation after an actual non-publishing result");
    const auto original_result=result;
    require(publication_release(fixture.control,fixture.lease)==0,"stopped fixture reader release refused");
    PublicationLease final_reader{};Descriptor command{};
    command.publication.control=reinterpret_cast<uint64_t>(&fixture.control);
    command.publication.lease=reinterpret_cast<uint64_t>(&final_reader);command.publication.operation=2;
    publication_command(command);
    require(final_reader.active && !final_reader.status && !fixture.control.refusal &&
        std::memcmp(&result,&original_result,sizeof(result))==0,
        "final canonical acquisition rejected a stopped parent or changed its original result");
    command.publication.operation=3;publication_command(command);
    result={};conditional=1;
    semantic_publication_step_admit(reinterpret_cast<uint64_t>(&fixture.control),reinterpret_cast<uint64_t>(&next),1,
        reinterpret_cast<uint64_t>(&conditional),reinterpret_cast<uint64_t>(&result));
    require(conditional==0 && next.status==7 && !next.active,
        "later step acquired an invocation after a skipped predecessor");
    StepPublicationFixture progressing;
    PublicationLease previous{};require(publication_acquire(progressing.control,previous)==0,"advance fixture reader refused");
    progressing.publish();
    semantic_publication_step_result(reinterpret_cast<uint64_t>(&progressing.control),reinterpret_cast<uint64_t>(&previous),
        reinterpret_cast<uint64_t>(&result),reinterpret_cast<uint64_t>(&progressing.banks[previous.bank].header),0);
    require(publication_release(progressing.control,previous)==0,"advance fixture reader release refused");
    semantic_publication_step_admit(reinterpret_cast<uint64_t>(&progressing.control),reinterpret_cast<uint64_t>(&next),1,
        reinterpret_cast<uint64_t>(&conditional),reinterpret_cast<uint64_t>(&result));
    require(conditional==1 && next.active && !next.status,"actual publication progress incorrectly stopped the next step");
    semantic_publication_step_release(reinterpret_cast<uint64_t>(&progressing.control),reinterpret_cast<uint64_t>(&next));
}

static void fresh_authority_snapshot_is_published_and_sealed() {
    StepPublicationFixture fixture;
    const auto* pending=publication_find_range(fixture.pending_ranges.data(),fixture.pending_ranges.size(),39);
    require(pending && pending->length_bytes==sizeof(uint64_t),"authority fixture lacks its original pending carrier");
    auto* bytes=static_cast<uint64_t*>(fixture.data(*pending));
    *bytes=11;
    const auto original=fixture.range(fixture.lease.bank,39);
    *bytes=29;
    fixture.publish();
    const auto& published=fixture.range(fixture.lease.bank,39);uint64_t digest[4]{};
    require(*static_cast<const uint64_t*>(fixture.data(published))==29 &&
        publication_range_digest(fixture.control,published,nullptr,digest)==0 &&
        publication_identity_equal(digest,published.digest) &&
        !publication_identity_equal(digest,original.digest),
        "publication copied or sealed old authority bytes instead of the fresh pending snapshot");
}

static void step_inputs_follow_device_selected_resident_bank() {
    StepPublicationFixture fixture;
    {
        auto original=fixture.banks[0];uint64_t expected[4],digest[4];
        std::fill(std::begin(original.header.descriptor_digest),std::end(original.header.descriptor_digest),0);
        semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&original),sizeof(original),expected);
        for(const auto& range:fixture.directories[0]) {
            semantic_graph::sha256(reinterpret_cast<const uint8_t*>(&range),sizeof(range),digest);
            publication_fold(expected,range.role,range.index,digest);
        }
        require(publication_identity_equal(expected,fixture.banks[0].header.descriptor_digest),
            "pure descriptor verification changed the original sealed byte format");
    }
    struct Outputs {
        PublicationHeader header;
        SourceSlot source[32];
        PublicationRange ranges[StepPublicationFixture::kStepInputCount];
        alignas(8) uint8_t copied[StepPublicationFixture::kStepInputCount]
            [StepPublicationFixture::kFeedbackCapacity*sizeof(RawFeedbackRecord)];
        uint64_t metadata_digests[2][4];
    };
    auto* outputs=static_cast<Outputs*>(mmap(nullptr,2*sizeof(Outputs),PROT_READ|PROT_WRITE,MAP_SHARED|MAP_ANONYMOUS,-1,0));
    require(outputs!=MAP_FAILED,"step fixture shared output allocation failed");
    auto* first_outputs=outputs;
    std::array<std::array<PublicationStepInput,StepPublicationFixture::kStepInputCount>,2> bindings{};
    const auto bind_outputs=[&] {
        for(uint64_t bank=0;bank<2;++bank) {
            uint64_t cursor=0;
            for(uint64_t role=1;role<=44;++role) {
                if(role==2 || role==14 || (role>25 && role!=44))continue;
                for(uint64_t index=0;index<fixture.roles[role-1].count;++index) {
                    auto& input=bindings[bank][cursor];
                    const auto* range=publication_find_range(fixture.directories[bank].data(),fixture.directories[bank].size(),role,index);
                    require(range,"step fixture binding range is absent");
                    const bool model=role>=18 && role<=25;
                    const bool alias=role==1 || role==4 || role==5 || model;
                    input={};input.role=role;input.index=index;
                    input.capacity_bytes=model ? range->length_bytes : fixture.storage[range->storage_slot].bytes;
                    input.destination=alias ? reinterpret_cast<uint64_t>(fixture.data(*range)) :
                        reinterpret_cast<uint64_t>(outputs->copied[cursor]);
                    input.backing=model ? fixture.storage[range->storage_slot].pointer : input.destination;
                    input.backing_bytes=model ? fixture.storage[range->storage_slot].bytes : input.capacity_bytes;
                    if(role==1) {
                        input.layout.role=1;input.layout.element_bytes=8;input.layout.scalar_type=3;input.layout.rank=2;
                        input.layout.logical_axis=0;input.layout.dimensions[0]=64;input.layout.dimensions[1]=8;
                        input.layout.strides_bytes[0]=64;input.layout.strides_bytes[1]=8;
                    } else if(publication_tensor_role(role))
                        input.layout=*publication_find_layout(fixture.control,fixture.directories[bank].data(),
                            fixture.directories[bank].size(),role,index);
                    ++cursor;
                }
            }
            require(cursor==bindings[bank].size(),"step fixture binding roster is incomplete");
        }
    };
    bind_outputs();
    uint64_t binding_count=bindings[0].size();
    const auto consume=[&] { semantic_publication_step_inputs(reinterpret_cast<uint64_t>(&fixture.control),
        reinterpret_cast<uint64_t>(&fixture.lease),reinterpret_cast<uint64_t>(bindings[0].data()),
        reinterpret_cast<uint64_t>(bindings[1].data()),binding_count,
        reinterpret_cast<uint64_t>(&outputs->header),reinterpret_cast<uint64_t>(outputs->source),reinterpret_cast<uint64_t>(outputs->ranges),
        reinterpret_cast<uint64_t>(outputs->metadata_digests)); };
    const auto guard=[&] { semantic_publication_step_input_guard(reinterpret_cast<uint64_t>(&fixture.lease),
        reinterpret_cast<uint64_t>(&outputs->header),reinterpret_cast<uint64_t>(outputs->source),
        reinterpret_cast<uint64_t>(bindings[0].data()),reinterpret_cast<uint64_t>(bindings[1].data()),bindings[0].size(),
        reinterpret_cast<uint64_t>(outputs->ranges),reinterpret_cast<uint64_t>(outputs->metadata_digests)); };
    const auto verify=[&] {
        const auto& bank=fixture.banks[fixture.lease.bank];
        const auto& selected=bindings[fixture.lease.bank];
        require(std::memcmp(&outputs->header,&bank.header,sizeof(bank.header))==0 &&
            std::memcmp(outputs->source,bank.source,sizeof(bank.source))==0,"step inputs did not follow the acquired original metadata");
        for(uint64_t i=0;i<selected.size();++i) {
            const auto& original=*publication_find_range(fixture.directories[fixture.lease.bank].data(),
                fixture.directories[fixture.lease.bank].size(),selected[i].role,selected[i].index);
            require(std::memcmp(&outputs->ranges[i],&original,sizeof(original))==0,"step inputs replaced the original range seal");
            const bool copied=selected[i].role==3 ||
                (selected[i].role>=6 && (selected[i].role<18 || selected[i].role>25));
            if(copied)
                require(std::memcmp(reinterpret_cast<void*>(selected[i].destination),fixture.data(original),original.length_bytes)==0,
                    "step cache input did not follow the acquired original values");
            else require(selected[i].destination==reinterpret_cast<uint64_t>(fixture.data(original)),
                "resident step input did not alias the device-selected publication bank");
        }
    };
    consume();verify();guard();const Outputs first=*outputs;
    outputs=first_outputs+1;
    fixture.publish();bind_outputs();consume();verify();guard();const Outputs second=*outputs;
    require(std::memcmp(first.copied,second.copied,sizeof(first.copied))!=0,"distinct real publication retained old step cache values");
    require(first.header.prefix_extent==0 && second.header.prefix_extent==1 && first.source[0].kind==1 && second.source[0].kind==0,
        "real publication did not change the original prefix metadata and Source snapshot");
    fixture.publish();
    consume();verify();guard();const Outputs third=*outputs;
    require(third.header.publication_word==4 && second.header.publication_word==3 && first.header.publication_word==0,
        "step fixture did not traverse three real publication epochs");
    const auto rejects=[&](auto corrupt,const char* message) {
        std::memset(outputs,0xa5,sizeof(Outputs));const Outputs before=*outputs;
        require_content_trap([&] { corrupt();consume(); },message);
        require(std::memcmp(outputs,&before,sizeof(before))==0,"rejected step input changed output bytes before the integrity trap");
    };
    const uint64_t active=fixture.lease.bank;
    auto& selected=bindings[active];
    rejects([&] { fixture.banks[active].source[0].token^=1; },"step input replaced changed original Source bytes");
    rejects([&] { ++fixture.banks[active].header.ring_head; },"step input ignored changed original header");
    rejects([&] { fixture.banks[active].header.descriptor_digest[0]^=1; },"step input ignored original descriptor seal");
    rejects([&] { fixture.range(active,13).digest[0]^=1; },"step input ignored changed original cache seal");
    rejects([&] { static_cast<float*>(fixture.data(fixture.range(active,13)))[1]+=1; },"step input resealed changed original cache values");
    rejects([&] { static_cast<uint64_t*>(fixture.data(fixture.range(active,3)))[0]^=1; },"step input resealed changed original PrefixIdentity");
    rejects([&] { static_cast<SourceSlot*>(fixture.data(fixture.range(active,1)))[0].token^=1; },
        "step input ignored changed original shared Prefix content");
    rejects([&] { static_cast<float*>(fixture.data(fixture.range(active,4)))[0]+=1; },
        "step input ignored changed original shared attention content");
    rejects([&] { static_cast<float*>(fixture.data(fixture.range(active,18)))[0]+=1; },
        "step input ignored changed resident model content");
    rejects([&] { auto* table=static_cast<TensorLayoutTableHeader*>(fixture.data(fixture.range(active,55)));
        reinterpret_cast<PublicationTensorLayout*>(table+1)[0].strides_bytes[0]+=4; },"step input ignored original layout seal");
    rejects([&] { selected[0].destination+=8; },"step input accepted a different Prefix capacity alias");
    rejects([&] { selected[2].destination+=8; },"step input accepted a different attention capacity alias");
    rejects([&] { --selected[2].layout.dimensions[2]; },"step input accepted different attention capacity layout");
    rejects([&] { --selected[4].capacity_bytes; },"step input accepted a truncated cache owner");
    rejects([&] { selected[4].layout.strides_bytes[0]=8; },"step input accepted different cache layout");
    rejects([&] { std::swap(selected[4],selected[5]); },"step input accepted a noncanonical role roster");
    rejects([&] { --binding_count; },"step input accepted an incomplete input roster");
    rejects([&] { binding_count=UINT64_MAX; },"step input accepted an overflowing input roster");
    rejects([&] { fixture.roles[12].count=2; },"step input accepted a changed original role count");
    rejects([&] { selected[4].destination=UINT64_MAX-3; },"step input accepted an overflowing cache output");
    rejects([&] { fixture.range(active,13).storage_slot=fixture.control.storage_count; },
        "step input accepted an original range outside the storage directory");
    rejects([&] { selected[4].destination=selected[5].destination; },"step input accepted overlapping copied destinations");
    rejects([&] { selected[4].destination=reinterpret_cast<uint64_t>(outputs->source); },"step input accepted a copied cache overlapping Source output");
    rejects([&] { selected[4].destination=reinterpret_cast<uint64_t>(fixture.data(fixture.range(active,6))); },
        "step input accepted a copy destination in the original publication");
    const auto model_input=std::find_if(selected.begin(),selected.end(),
        [](const auto& input){return input.role==18 && input.index==0;});
    require(model_input!=selected.end(),"step fixture lacks its first resident model binding");
    const uint64_t model_input_index=uint64_t(model_input-selected.begin());
    rejects([&] { model_input->destination=reinterpret_cast<uint64_t>(outputs->copied[model_input_index]); },
        "step input accepted a copied model in place of the resident alias");
    rejects([&] { fixture.lease.active=0; },"step input accepted a released device lease");
    rejects([&] { fixture.lease.bank=2; },"step input traversed an invalid bank");
    consume();verify();
    const auto rejects_guard=[&](auto corrupt,const char* message) {
        require_content_trap([&] { corrupt();guard(); },message);
    };
    rejects_guard([&] { ++outputs->header.ring_head; },"step guard accepted changed original RingHead");
    rejects_guard([&] { ++outputs->header.prefix_extent; },"step guard accepted changed original PrefixExtent");
    rejects_guard([&] { outputs->source[0].token^=1; },"step guard accepted changed original Source");
    rejects_guard([&] { outputs->metadata_digests[1][0]^=1; },"step guard replaced original Source baseline");
    rejects_guard([&] { outputs->copied[1][0]^=1; },"step guard accepted changed original PrefixIdentity");
    rejects_guard([&] { outputs->copied[11][0]^=1; },"step guard accepted changed original cache copy");
    rejects_guard([&] { outputs->ranges[11].digest[0]^=1; },"step guard replaced original range seal");
    require_content_trap([&] { semantic_publication_step_input_guard(reinterpret_cast<uint64_t>(&fixture.lease),
        reinterpret_cast<uint64_t>(&outputs->header),reinterpret_cast<uint64_t>(outputs->source),
        reinterpret_cast<uint64_t>(bindings[0].data()),reinterpret_cast<uint64_t>(bindings[1].data()),bindings[0].size()-1,
        reinterpret_cast<uint64_t>(outputs->ranges),reinterpret_cast<uint64_t>(outputs->metadata_digests)); },
        "step guard accepted an incomplete fixed cache roster");
    rejects_guard([&] { static_cast<SourceSlot*>(fixture.data(fixture.range(active,1)))[0].token^=1; },
        "step guard accepted changed resident Prefix content");
    rejects_guard([&] { static_cast<float*>(fixture.data(fixture.range(active,4)))[0]+=1; },
        "step guard accepted changed resident attention content");
    rejects_guard([&] { static_cast<float*>(fixture.data(fixture.range(active,18)))[0]+=1; },
        "step guard accepted changed resident model content");
    consume();verify();
    static_cast<SourceSlot*>(fixture.data(fixture.range(0,1)))[63].token=19;
    static_cast<float*>(fixture.data(fixture.range(active,4)))[127]=21;
    guard();
    require(std::memcmp(outputs,&third,sizeof(third))==0,"step input verification replaced original retained content");
    require(publication_release(fixture.control,fixture.lease)==0,"step fixture final reader release refused");
    require_content_trap(guard,"step guard accepted a released device lease");
    require(munmap(first_outputs,2*sizeof(Outputs))==0,"step fixture output release failed");
}

static void exact_task_occurrence() {
    semantic_graph::DecodedStatement statement{};
    statement.identity_words[0]=91;
    semantic_graph::DecodedSupport support{};
    support.polarity=1; support.record=3;
    uint64_t identity[4], digest[4];
    semantic_graph::decode_identity(statement.identity_words,identity);
    semantic_graph::support_digest(identity,support,digest);
    std::array<uint64_t,39> task{};
    task[0]=4; task[1]=91;
    std::copy(identity,identity+4,task.begin()+6);
    task[10]=task[14]=92; task[18]=1; task[19]=task[20]=2; task[21]=1;
    task[22]=7; task[23]=task[24]=8; task[31]=task[32]=task[33]=7; task[34]=3;
    std::copy(digest,digest+4,task.begin()+35);
    require(task_allows(task.data(),statement,support), "exact permitted support occurrence refused");
    require(statement.record==7, "task query lost its original statement occurrence");
    support.record=0;
    require(!task_allows(task.data(),statement,support), "equal event from another original record was permitted");
    support.record=3; statement.identity_words[0]=93;
    require(!task_allows(task.data(),statement,support), "support was permitted for another statement");
}

static void acquired_feedback_history() {
    using namespace semantic_graph;
    constexpr uint64_t roots=4, statements=4, supports=8, versions=8;
    std::vector<uint64_t> arena(kControlWords+roots*kRootWords+kCandidateWords+
        statements*kStatementWords+supports*kSupportWords+versions*kVersionWords+
        roots*statements+statements);
    const auto run=[&](Command command) {
        semantic_graph::Receipt receipt{};
        command.words[1]=91;
        LaunchDescriptor launch{};
        launch.expected_owner=91;launch.root_capacity=roots;launch.statement_capacity=statements;
        launch.support_capacity=supports;launch.version_capacity=versions;launch.arena_words=arena.size();
        launch.command_ptr=reinterpret_cast<uint64_t>(&command);
        launch.receipt_ptr=reinterpret_cast<uint64_t>(&receipt);launch.abi_generation=kAbiGeneration;
        execute(arena.data(),launch);
        require(receipt.words[0]==kOk,"feedback fixture native operation refused");
        return receipt;
    };
    Command command{};command.words[0]=kInitialize;run(command);
    command={};command.words[0]=kFork;command.words[7]=kRootKind;command.words[9]=1;
    const auto fork=run(command);
    command={};command.words[0]=kInsertSupport;command.words[8]=fork.words[5];command.words[9]=fork.words[6];
    command.words[12]=1;command.words[13]=3;command.words[14]=5;command.words[16]=11;command.words[20]=31;
    const auto pro=run(command);
    command.words[14]=7;require(run(command).words[1]==kOutcomeUnchanged,"duplicate changed fixture history");
    command.words[12]=2;command.words[14]=6;command.words[20]=32;run(command);
    command={};command.words[0]=kSeal;command.words[7]=kForkKind;command.words[8]=fork.words[5];command.words[9]=fork.words[6];
    const auto sealed=run(command);
    Descriptor descriptor{};
    descriptor.arena[0]=reinterpret_cast<uint64_t>(arena.data());descriptor.arena[1]=91;
    descriptor.arena[2]=roots;descriptor.arena[3]=statements;descriptor.arena[4]=supports;
    descriptor.arena[5]=versions;descriptor.arena[6]=arena.size();
    std::array<uint64_t,34> task{};
    task[0]=4;task[1]=91;task[6]=11;task[10]=task[14]=21;
    task[22]=3;task[23]=task[24]=8;task[31]=task[32]=task[33]=7;
    descriptor.task=reinterpret_cast<uint64_t>(task.data());
    PublicationBank banks[2]{};auto& bank=banks[0];bank.header.abi=1;bank.header.semantic_owner=91;
    bank.header.semantic_slot=sealed.words[3];bank.header.semantic_generation=sealed.words[4];
    std::copy(sealed.words+13,sealed.words+16,bank.header.semantic_extents);
    std::copy(sealed.words+16,sealed.words+20,bank.header.semantic_digest);
    PublicationControl control{};control.abi=1;
    banks[1]=bank;banks[1].header.publication_word=1;
    for(uint64_t i=0;i<2;++i)control.banks[i]=reinterpret_cast<uint64_t>(&banks[i]);
    descriptor.publication.control=reinterpret_cast<uint64_t>(&control);
    PublicationLease lease{};require(publication_acquire(control,lease)==0,"native reader acquisition refused");
    std::array<RawFeedbackRecord,3> records{};
    for(uint32_t i=0;i<2;++i) {
        auto& record=records[i];record.valid=1;record.statement_index=i;record.statement_length_bytes=1;
        DecodedStatement statement{};statement.identity_words[0]=uint32_t(task[6+4*i]);
        ResidentHandle root{91,kRootKind,bank.header.semantic_slot,bank.header.semantic_generation};
        semantic_call(descriptor,kResidentRootTruthAdmission,&root,nullptr,&record.query_receipt,&statement);
        record.pro=record.query_receipt.words[2]&1;record.contra=record.query_receipt.words[2]>>1;
        // A restored archive names an older allocation. Only the fresh query
        // may supply runtime coordinates to the service projection.
        record.query_receipt.words[3]=999;record.query_receipt.words[35]=999;
    }
    records[2].statement_index=UINT64_MAX;records[2].statement_length_bytes=UINT64_MAX;
    std::swap(records[1],records[2]);
    std::array<semantic_graph::Receipt,3> queries{};
    std::array<uint64_t,3> original{};std::array<uint64_t,4> offsets{};
    std::array<uint64_t,supports*12> rows{};
    std::array<float,3*13> features{};
    std::array<uint8_t,3> validity{};
    std::array<uint64_t,3> status{};
    uint8_t first=0xa5,second=0x3c,third=0x3c;
    auto next_records=records;std::swap(next_records[0],next_records[2]);
    PublicationContract contract{};contract.abi=1;contract.range_capacity=4;contract.feedback_capacity=3;
    PublicationStorageEntry storage[]={{reinterpret_cast<uint64_t>(records.data()),sizeof(records),1},
        {reinterpret_cast<uint64_t>(next_records.data()),sizeof(next_records),1},
        {reinterpret_cast<uint64_t>(&first),1,1},{reinterpret_cast<uint64_t>(&second),1,1},
        {reinterpret_cast<uint64_t>(&third),1,1}};
    PublicationRange ranges[2][4]{};
    control.contract=reinterpret_cast<uint64_t>(&contract);
    control.storage=reinterpret_cast<uint64_t>(storage);control.storage_count=5;
    for(uint64_t parent=0;parent<2;++parent) {
        banks[parent].header.range_count=4;
        control.directories[parent]=reinterpret_cast<uint64_t>(ranges[parent]);
        ranges[parent][0].role=15;ranges[parent][0].generation=1;
        ranges[parent][0].storage_slot=parent;ranges[parent][0].length_bytes=sizeof(records);
        for(uint64_t statement=0;statement<3;++statement) {
            auto& range=ranges[parent][1+statement];range.role=16;range.index=statement;
            range.generation=1;range.storage_slot=2+statement;range.length_bytes=1;
        }
        for(auto& range:ranges[parent])
            require(publication_range_digest(control,range,nullptr,range.digest)==0,
                "original feedback range seal failed");
    }
    std::array<PublicationRange,8> original_ranges{};
    std::memcpy(original_ranges.data(),ranges,sizeof(ranges));
    auto encode=[&]() { semantic_feedback_encode(reinterpret_cast<uint64_t>(&control),&lease,1,3,
        features.data(),validity.data(),status.data()); };
    auto guarded_project=[&]() { semantic_feedback_project(descriptor,&lease,3,
        queries.data(),original.data(),offsets.data(),rows.data(),supports,status.data()); };
    auto project=[&]() { return publication_feedback_lineage(descriptor,&lease,3,
        queries.data(),original.data(),offsets.data(),rows.data(),supports); };
    const uint64_t output_data[]={reinterpret_cast<uint64_t>(features.data()),
        reinterpret_cast<uint64_t>(validity.data()),reinterpret_cast<uint64_t>(queries.data()),
        reinterpret_cast<uint64_t>(original.data()),reinterpret_cast<uint64_t>(offsets.data()),
        reinterpret_cast<uint64_t>(rows.data())};
    const uint64_t output_bytes[]={sizeof(features),sizeof(validity),sizeof(queries),
        sizeof(original),sizeof(offsets),sizeof(rows)};
    const uint64_t output_shapes[][2]={{3,13},{3,0},{3,42},{3,0},{4,0},{supports,12}};
    const uint64_t output_scalar_types[]={6,8,3,3,3,3};
    const uint64_t output_element_bytes[]={4,1,8,8,8,8};
    std::array<PublicationTensorLayout,6> output_layouts{};
    std::array<std::array<uint64_t,4>,6> output_digests{};
    for(uint64_t index=0;index<output_layouts.size();++index) {
        auto& output=output_layouts[index];output.index=index;
        output.element_bytes=output_element_bytes[index];output.scalar_type=output_scalar_types[index];
        output.rank=output_shapes[index][1] ? 2 : 1;output.logical_axis=UINT64_MAX;
        output.dimensions[0]=output_shapes[index][0];output.dimensions[1]=output_shapes[index][1];
        output.strides_bytes[output.rank-1]=output.element_bytes;
        if(output.rank==2)output.strides_bytes[0]=output.dimensions[1]*output.element_bytes;
    }
    const auto output_witness=[&](uint64_t index,uint64_t verify) {
        PublicationRange range{};range.index=index;range.length_bytes=output_bytes[index];
        semantic_tensor_content_witness(output_data[index],output_bytes[index],range,output_layouts[index],
            reinterpret_cast<uint64_t>(output_digests[index].data()),verify);
    };
    encode();
    require(status==std::array<uint64_t,3>{0,0,0},"valid acquired feedback encoding refused");
    require(validity==std::array<uint8_t,3>{1,0,1},"feedback encoder packed or changed validity gaps");
    for(uint32_t i=13;i<26;++i)
        require(features[i]==0.0f,"invalid feedback slot retained learned coordinates");
    guarded_project();
    // Seal the original outputs immediately after the real producer. Every
    // consumer reuses these six private seals; hooks cannot establish a new baseline.
    for(uint64_t index=0;index<output_layouts.size();++index)output_witness(index,0);
    const auto original_output_digests=output_digests;
    for(uint64_t index=0;index<output_layouts.size();++index) {
        require(output_digests[index]!=std::array<uint64_t,4>{},"original feedback output was not sealed");
        output_witness(index,1);
    }
    require(output_digests==original_output_digests,"feedback output verification replaced an original seal");
    require(original[0]==3 && original[1]==UINT64_MAX && original[2]==8,"original query mapping lost");
    require(offsets==std::array<uint64_t,4>{0,2,2,2},"actual Both or Neither contributors changed");
    require(rows[1]==6 && rows[13]==5,"projection used allowed roster or replaced first insertion source");
    require(queries[0].words[3]==bank.header.semantic_slot && queries[0].words[35]==bank.header.semantic_generation,
        "projection reused archived runtime coordinates");
    const char* output_failures[]={"changed learned feedback features did not reach the original output trap",
        "changed feedback validity did not reach the original output trap",
        "changed feedback query lineage did not reach the original output trap",
        "changed feedback statement lineage did not reach the original output trap",
        "changed feedback support offsets did not reach the original output trap",
        "changed feedback contributor lineage did not reach the original output trap"};
    for(uint64_t index=0;index<output_layouts.size();++index)
        require_content_trap([&] {
            reinterpret_cast<uint8_t*>(output_data[index])[0]^=1;
            output_witness(index,1);
        },output_failures[index]);
    for(uint64_t index=0;index<output_layouts.size();++index)output_witness(index,1);
    require(output_digests==original_output_digests,"feedback consumers modified the original producer seals");
    records[0].valid=2;encode();
    require(status[0]!=0,"malformed feedback did not produce the actual encoder error");
    records[0].valid=1;
    require_content_trap(guarded_project,"feedback encoder error did not stop the actual projection wrapper");
    encode();
    Layout layout{};require(build_layout(arena.data(),roots,statements,supports,versions,arena.size(),&layout),"layout failed");
    auto* head=version_record(layout,queries[0].words[11]);const auto head_generation=head[1];head[1]=0;
    require(project()!=0,"feedback projection erased actual contributions behind a zero head generation");
    require_content_trap(guarded_project,"stale head generation did not stop the actual projection wrapper");head[1]=head_generation;
    auto* old=support_record(layout,pro.words[9]);const auto generation=old[1];old[1]++;
    require(project()!=0,"feedback projection accepted stale older support generation");
    require_content_trap(guarded_project,"stale support generation did not stop the actual projection wrapper");old[1]=generation;
    const auto original_target=old[10];old[10]=UINT32_MAX;
    require(project()==2,"feedback did not distinguish a genuine absence of original target provenance");
    require_content_trap(guarded_project,"derived contributor did not stop the actual projection wrapper");old[10]=original_target;
    lease.active=0;require(project()!=0,"feedback projection accepted released reader");
    require_content_trap(guarded_project,"released reader did not stop the actual projection wrapper");lease.active=1;
    bank.header.semantic_digest[0]^=1;require(project()!=0,"feedback projection accepted another acquired root digest");
    require_content_trap(guarded_project,"changed acquired root did not stop the actual projection wrapper");
    bank.header.semantic_digest[0]^=1;
    encode();guarded_project();const auto original_features=features;
    control.word=1;encode();guarded_project();
    require(original[0]==3 && original[2]==8,"retained feedback reader followed a later publication");
    require(publication_release(control,lease)==0 && publication_acquire(control,lease)==0,
        "next feedback reader acquisition refused");
    encode();guarded_project();
    require(original[0]==8 && original[2]==3,
        "feedback projection followed captured raw records instead of acquired publication");
    for(uint64_t column=0;column<13;++column)
        require(features[column]==original_features[26+column] && features[26+column]==original_features[column],
            "feedback encoding followed captured raw records instead of acquired publication");
    const auto rejects_input=[&](auto corrupt,const char* message) {
        require_content_trap([&] {
            corrupt();encode();
            for(uint64_t slot=0;slot<3;++slot)
                require(status[slot] && !validity[slot],"invalid acquired feedback source lost encoder diagnostics");
            for(float value:features)require(value==0.0f,"invalid acquired feedback source retained learned coordinates");
            require(project()!=0,"lineage accepted malformed acquired feedback source");
            guarded_project();
        },message);
    };
    rejects_input([&] { lease.active=0; },"feedback accepted an inactive device reader");
    rejects_input([&] { ++lease.epoch; },"feedback accepted a stale device reader");
    rejects_input([&] { lease.bank=2; },"feedback read outside the publication bank pair");
    rejects_input([&] { control.directories[1]=UINT64_MAX-7; },"feedback accepted an overflowing acquired directory");
    rejects_input([&] { ranges[1][0].storage_slot=control.storage_count; },"feedback accepted an absent raw storage slot");
    rejects_input([&] { ++ranges[1][0].generation; },"feedback accepted a stale raw storage generation");
    rejects_input([&] { ranges[1][0].length_bytes-=sizeof(RawFeedbackRecord); },"feedback accepted an incomplete physical slot array");
    rejects_input([&] { ++storage[1].pointer; },"feedback accepted unaligned raw records");
    rejects_input([&] { ++contract.feedback_capacity; },"feedback accepted another output slot capacity");
    rejects_input([&] { ++next_records[0].valid; },"feedback replaced the original raw record seal");
    rejects_input([&] { first^=1; },"feedback replaced the original statement seal");
    require(std::memcmp(original_ranges.data(),ranges,sizeof(ranges))==0,
        "feedback consumption changed an original publication range or seal");
    require(publication_release(control,lease)==0 && !control.reader_counts[0] && !control.reader_counts[1],
        "feedback projection changed reader ownership");
}

static void initial_prefix_identity_is_native_owned() {
    std::array<uint64_t,4> previous{};
    for(const auto& tokens:std::vector<std::vector<uint64_t>>{{},{7},{7,8,9},{7,8,10}}) {
        StepPublicationFixture fixture(false,tokens);
        std::array<uint64_t,4> expected{};
        require(publication_range_digest(fixture.control,fixture.range(0,1),nullptr,expected.data())==0,
            "actual initial prefix source cannot be hashed");
        const auto* actual=static_cast<const uint64_t*>(fixture.data(fixture.range(0,3)));
        require(publication_identity_equal(actual,expected.data()),
            "fresh publication did not derive identity from its actual prefix source");
        const auto* coverage=static_cast<const CompletionCoverage*>(fixture.data(fixture.range(0,14)));
        require(publication_identity_equal(coverage->prefix_identity,expected.data()),
            "initial coverage does not bind the actual native prefix identity");
        require(!publication_identity_equal(previous.data(),expected.data()),
            "different actual initial prefixes share an identity");
        previous=expected;
        require(publication_release(fixture.control,fixture.lease)==0,"initial identity reader release failed");
        fixture.banks[0].header.abi=0;
        require(publication_initialize(fixture.execution.descriptor,fixture.control,true)==0 &&
            publication_identity_equal(actual,expected.data()),
            "restoration changed the saved initial prefix identity");
        // Saved logical/state digests still name the original identity. Restore
        // must reject altered bytes, never recompute them into an accepted bank.
        auto* changed=static_cast<uint64_t*>(fixture.data(fixture.range(0,3)));
        changed[0]^=1;const uint64_t corrupted=changed[0];
        fixture.banks[0].header.abi=0;
        require(publication_initialize(fixture.execution.descriptor,fixture.control,true)!=0 &&
            fixture.banks[0].header.abi==0 && changed[0]==corrupted,
            "restoration repaired a corrupted prefix identity instead of rejecting it");
    }
}

static void task_selection_uses_supplied_scoring() {
    std::array<uint64_t,34> task{};
    task[25]=3;task[26]=5;task[27]=7;task[28]=2;task[29]=11;task[30]=13;
    State state{};
    auto& evaluation=state.task_evaluation;
    evaluation.facts[0].eligible=1;
    evaluation.facts[1].eligible=1;evaluation.facts[1].g=2;evaluation.facts[1].p=1;
    state.work[0].edit_commands=2;
    evaluation.facts[2].eligible=1;evaluation.facts[2].g=1;
    for(uint32_t slot=0;slot<3;++slot)task_cost(&state,slot,task.data());
    require(evaluation.facts[1].v==-3 && evaluation.facts[2].v==3,
        "task value ignored caller coefficients");
    evaluation.query_count=6;select_task(&state,task.data());
    require(evaluation.winner==2,"selection ignored the supplied cost tradeoff");
    require(evaluation.return_value==6-13*8,"return ignored improvement/spent coefficients");
    task[27]=0;
    for(uint32_t slot=0;slot<3;++slot)task_cost(&state,slot,task.data());
    select_task(&state,task.data());
    require(evaluation.winner==1,"changing the task scoring did not change the selected candidate");
}

int main(int argc, char** argv) {
    if(argc==4 && std::strcmp(argv[1],"--decode")==0) {
        std::ifstream input(argv[2]);std::ofstream output(argv[3]);
        uint64_t count=0;input>>count;
        require(input.good() && count>25 && count<1000000,"invalid decoder test bank");
        std::vector<uint64_t> books(count);std::array<uint32_t,136> choices{};
        for(auto& word:books)input>>word;
        for(auto& choice:choices)input>>choice;
        require(!input.fail(),"truncated decoder test input");
        semantic_graph::DecodedStatement statement{};semantic_graph::DecodedSupport support{};
        decode_action(books.data(),choices.data(),&statement,&support);
        for(auto word:statement.identity_words)output<<word<<' ';
        output<<statement.record<<' ';
        for(auto reference:statement.reconstruction)output<<reference<<' ';
        output<<support.record<<'\n';
        require(output.good(),"decoder test output failed");
        return 0;
    }
    require(argc==1,"unexpected feedback test arguments");
    install_fatal_signal_backtraces();
    task_preflight_accepts_all_four_truth_states();
    task_selection_uses_supplied_scoring();
    initial_prefix_identity_is_native_owned();
    recompute_execution_preserves_proposal_state(false);
    recompute_execution_preserves_proposal_state(true);
    device_step_admission_preserves_refused_publications();
    device_step_lease_tracks_current_parent_and_drain();
    publication_rejects_a_mode_different_from_device_admission();
    backward_completion_rejects_every_error_status();
#ifdef XLOG_SEMANTIC_POLICY
    backward_nonfinite_cotangent_reaches_integrity_trap();
#endif
    content_guard_follows_each_acquired_publication();
    retained_model_contract_preserves_original_seal_after_bank_reuse();
    active_content_requires_computed_acquired_bank();
    for(uint64_t role : {51,52,53,54})active_continuation_preserves_physical_holes(role);
    resident_text_services_validate_original_rows();
    malformed_support_is_not_an_ordinary_refusal();
    ordinary_refusals_finish_actual_candidate_cleanup();
    native_work_preserves_one_model_charge_through_refusal();
    ordinary_publication_refusal_releases_gate_and_preserves_base();
    resident_numerical_refusal_precedes_publication();
    for(uint64_t role : {6,7})fixed_continuation_buffers_keep_semantic_intervals(role);
    original_content_seal_accepts_equal_strided_producer();
    scalar_model_content_seals_one_cell_without_reshaping();
    empty_model_storage_keeps_metadata_without_device_cells();
    step_inputs_follow_device_selected_resident_bank();
    captured_model_seals_follow_actual_reader();
    completed_step_result_survives_publication_bank_reuse();
    next_step_stops_after_actual_nonpublication();
    fresh_authority_snapshot_is_published_and_sealed();
    exact_task_occurrence();
    acquired_feedback_history();
    std::cout << "native feedback occurrence checks passed\n";
}
