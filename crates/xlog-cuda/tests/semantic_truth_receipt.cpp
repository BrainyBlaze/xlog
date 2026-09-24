// Run the actual sequential native semantic producer on the host. This checks
// receipt semantics only; it does not exercise CUDA launch or stream behavior.
#include <algorithm>
#include <array>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

#define __device__
#define XLOG_SEMANTIC_GRAPH_DEVICE_ONLY
#include "../kernels/semantic_hypergraph.cu"

using namespace semantic_graph;

void require(bool condition, const char *message) {
    if (!condition) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

struct NativeGraph {
    static constexpr uint64_t owner = 91;
    static constexpr uint64_t roots = 4, statements = 4, supports = 8, versions = 8;
    std::vector<uint64_t> arena = std::vector<uint64_t>(kControlWords + roots * kRootWords
        + kCandidateWords + statements * kStatementWords + supports * kSupportWords
        + versions * kVersionWords + roots * statements + statements);

    LaunchDescriptor descriptor(Receipt &receipt) {
        LaunchDescriptor value = {};
        value.expected_owner = owner;
        value.root_capacity = roots;
        value.statement_capacity = statements;
        value.support_capacity = supports;
        value.version_capacity = versions;
        value.arena_words = arena.size();
        value.receipt_ptr = reinterpret_cast<uint64_t>(&receipt);
        value.abi_generation = kAbiGeneration;
        return value;
    }

    Receipt run(Command command, NativeWorkTally *work = nullptr) {
        Receipt receipt = {};
        command.words[1] = owner;
        auto launch = descriptor(receipt);
        launch.command_ptr = reinterpret_cast<uint64_t>(&command);
        execute(arena.data(), launch, work);
        return receipt;
    }

    Receipt view(uint64_t operation, uint64_t kind, uint64_t slot, uint64_t generation,
                 const std::array<uint64_t, 4> &identity = {}) {
        Command command = {};
        command.words[0] = operation;
        command.words[7] = kind;
        command.words[8] = slot;
        command.words[9] = generation;
        std::copy(identity.begin(), identity.end(), command.words + 16);
        return run(command);
    }
};

void check_query(const Receipt &query, const Receipt &snapshot, uint64_t kind,
                 uint64_t slot, uint64_t generation, const std::array<uint64_t, 4> &identity,
                 const Receipt *insertion) {
    require(query.words[0] == kOk, "query refused a valid view");
    require(query.words[39] == NativeGraph::owner, "query lost its native owner");
    require(std::equal(identity.begin(), identity.end(), query.words + 20),
            "query lost the requested statement identity");
    require(std::equal(snapshot.words + 13, snapshot.words + 20, query.words + 13),
            "query lost the exact root/fork extents and digest");
    require(query.words[33] == kind && query.words[34] == slot && query.words[35] == generation,
            "query lost its validated typed view handle");
    const auto offset = kind == kRootKind ? 3 : 5;
    require(query.words[offset] == slot && query.words[offset + 1] == generation,
            "query lost its validated root or fork coordinates");
    const auto unused_offset = kind == kRootKind ? 5 : 3;
    require(query.words[unused_offset] == 0 && query.words[unused_offset + 1] == 0,
            "query fabricated a different view kind's coordinates");
    require(query.words[9] == 0 && query.words[10] == 0
            && std::all_of(query.words + 24, query.words + 28, [](uint64_t value) { return value == 0; }),
            "truth query fabricated a complete support proof from one edge");
    if (insertion == nullptr) {
        require(query.words[2] == 0, "absent statement did not remain Neither");
        require(query.words[7] == 0 && query.words[8] == 0 && query.words[11] == 0 && query.words[12] == 0,
                "absent statement fabricated live statement/version handles");
        require(std::all_of(query.words + 28, query.words + 32, [](uint64_t value) { return value == 0; }),
                "absent statement fabricated a version identity");
    } else {
        require(query.words[2] == insertion->words[2], "query changed the native truth");
        require(query.words[7] == insertion->words[7] && query.words[8] == insertion->words[8]
                && query.words[8] != 0, "query lost the real statement generation");
        require(query.words[11] == insertion->words[11] && query.words[12] == insertion->words[12]
                && query.words[12] != 0, "query lost the actual head version generation");
        require(std::equal(insertion->words + 28, insertion->words + 32, query.words + 28),
                "query lost the actual head version identity");
    }
}

struct Contributions {
    std::vector<std::array<uint64_t, 6>> records;
    bool operator()(uint64_t version_slot, const uint64_t* version, const uint64_t* support) {
        records.push_back({support[10], support[11], support[5], version[5], support[1], version_slot});
        return true;
    }
};

void check_native_work() {
    NativeWorkTally sum{};
    charge_native(&sum, NativeWorkEvent::TableSlot, UINT64_MAX);
    charge_native(&sum, NativeWorkEvent::Command, 1);
    require(sum.overflow == 1 && sum.units == UINT64_MAX,
            "native work wrapped instead of retaining overflow");
    require(sum.events[unsigned(NativeWorkEvent::Command)] == 0,
            "overflow partially appended a native event");

    NativeGraph graph;
    Command initialize{}; initialize.words[0] = kInitialize;
    require(graph.run(initialize).words[0] == kOk, "work regression initialization failed");
    const auto original = graph.arena;
    Command query{}; query.words[0] = kTruth; query.words[7] = kRootKind;
    query.words[9] = 1; query.words[16] = 73;
    NativeWorkTally actual{};
    const auto result = graph.run(query, &actual);
    require(result.words[0] == kOk && result.words[2] == 0 && graph.arena == original,
            "metering changed an absent query or its arena");
    require(actual.events[unsigned(NativeWorkEvent::Command)] == 1 &&
            actual.events[unsigned(NativeWorkEvent::TableSlot)] >= NativeGraph::statements &&
            actual.events[unsigned(NativeWorkEvent::CanonicalByte)] >= sizeof(Receipt),
            "actual command, empty slots or receipt writes were not charged");
    const auto before_refusal = actual.units;
    query.words[9] = 2;
    require(graph.run(query, &actual).words[0] == kStaleGeneration &&
            actual.units > before_refusal &&
            actual.events[unsigned(NativeWorkEvent::Command)] == 2 && graph.arena == original,
            "refused command lost its actual work or changed semantic state");

    Layout layout{};
    require(build_layout(graph.arena.data(), NativeGraph::roots, NativeGraph::statements,
            NativeGraph::supports, NativeGraph::versions, graph.arena.size(), &layout),
            "work regression layout failed");
    NativeWorkTally scan{}; layout.work = &scan;
    Receipt receipt{}; const uint64_t identity[4] = {73, 0, 0, 0};
    require(find_statement(layout, layout.root_heads, identity, &receipt) == kNull &&
            scan.events[unsigned(NativeWorkEvent::TableSlot)] == NativeGraph::statements,
            "empty statement heads were skipped by actual scan accounting");

    for (uint64_t bytes : {uint64_t(0), uint64_t(55), uint64_t(56), uint64_t(64)}) {
        const uint8_t input[64] = {};
        uint64_t unmetered[4], metered[4]; NativeWorkTally hash{};
        sha256(input, bytes, unmetered);
        sha256(input, bytes, metered, &hash);
        require(std::equal(unmetered, unmetered + 4, metered) &&
                hash.events[unsigned(NativeWorkEvent::ShaBlock)] == (bytes + 72) / 64 &&
                hash.events[unsigned(NativeWorkEvent::CanonicalByte)] == 32,
                "SHA accounting changed bytes or charged the wrong padding blocks");
    }
}

int main(int argc, char** argv) {
    if (argc == 4 && std::string(argv[1]) == "--commands") {
        // Test transport only: every supplied ABI command still executes the
        // production native owner, including reconstruction tags and generations.
        NativeGraph graph;
        std::ifstream commands(argv[2], std::ios::binary);
        require(commands.good(), "native command input could not be opened");
        Command command{};
        Receipt receipt{};
        while (commands.read(reinterpret_cast<char*>(&command), sizeof(command))) {
            receipt = graph.run(command);
            require(receipt.words[0] == kOk, "canonical restoration command was refused");
        }
        require(commands.eof() && commands.gcount() == 0, "native command input was truncated");
        std::ofstream output(argv[3], std::ios::binary);
        output.write(reinterpret_cast<const char*>(&receipt), sizeof(receipt));
        output.write(reinterpret_cast<const char*>(graph.arena.data()),
                     graph.arena.size() * sizeof(uint64_t));
        require(output.good(), "native restoration result could not be written");
        return 0;
    }
    require(argc == 1, "unexpected native truth regression arguments");
    check_native_work();
    NativeGraph graph;
    Command initialize_command = {};
    initialize_command.words[0] = kInitialize;
    const auto initialized = graph.run(initialize_command);
    require(initialized.words[0] == kOk, "native initialization failed");
    const std::array<uint64_t, 4> statement = {11, 12, 13, 14};
    const std::array<uint64_t, 4> absent = {21, 22, 23, 24};
    const auto empty = graph.view(kSnapshot, kRootKind, initialized.words[3], initialized.words[4]);
    check_query(graph.view(kTruth, kRootKind, 0, 1, statement), empty, kRootKind, 0, 1, statement, nullptr);

    const auto fork = graph.view(kFork, kRootKind, 0, 1);
    require(fork.words[0] == kOk, "native fork failed");
    Command insertion = {};
    insertion.words[0] = kInsertSupport;
    insertion.words[8] = fork.words[5];
    insertion.words[9] = fork.words[6];
    insertion.words[12] = 1;
    insertion.words[13] = 3;
    insertion.words[14] = 5;
    std::copy(statement.begin(), statement.end(), insertion.words + 16);
    insertion.words[20] = 31;
    const auto pro = graph.run(insertion);
    require(pro.words[0] == kOk && pro.words[1] == kOutcomeInserted, "native insertion failed");
    Layout layout{};
    require(build_layout(graph.arena.data(), NativeGraph::roots, NativeGraph::statements,
                         NativeGraph::supports, NativeGraph::versions, graph.arena.size(), &layout),
            "native layout failed");
    const auto* attached = support_record(layout, pro.words[9]);
    require(attached[10] == 3 && attached[11] == 5,
            "native support lost the exact original statement and support occurrences");
    insertion.words[13] = 7;
    insertion.words[14] = 6;
    const auto duplicate = graph.run(insertion);
    require(duplicate.words[0] == kOk && duplicate.words[1] == kOutcomeUnchanged,
            "equal support occurrence changed logical insertion behavior");
    require(attached[10] == 3 && attached[11] == 5,
            "duplicate support rewrote the first successful original occurrences");
    auto candidate = graph.view(kSnapshot, kForkKind, fork.words[5], fork.words[6]);
    check_query(graph.view(kTruth, kForkKind, fork.words[5], fork.words[6], statement),
                candidate, kForkKind, fork.words[5], fork.words[6], statement, &pro);
    check_query(graph.view(kTruth, kForkKind, fork.words[5], fork.words[6], absent),
                candidate, kForkKind, fork.words[5], fork.words[6], absent, nullptr);
    check_query(graph.view(kTruth, kRootKind, 0, 1, statement), empty, kRootKind, 0, 1, statement, nullptr);
    insertion.words[12] = 2;
    insertion.words[20] = 32;
    const auto both = graph.run(insertion);
    require(both.words[0] == kOk && both.words[2] == 3, "native opposite support did not produce Both");
    // Matching the newest event must not hide a corrupt older contribution.
    auto* older_support = support_record(layout, pro.words[9]);
    const auto saved_generation = older_support[1];
    older_support[1] += 1;
    const auto corrupt_duplicate = graph.run(insertion);
    require(corrupt_duplicate.words[0] == kCorruptLineage,
            "matching support head hid a corrupt older contribution");
    older_support[1] = saved_generation;
    Contributions staged;
    Receipt trace{};
    require(visit_support_chain(layout, both.words[7], both.words[11] + 1, true, staged, &trace)
            && staged.records.size() == 2, "candidate trace lost an actual contribution");
    Contributions sealed_only;
    trace = {};
    require(!visit_support_chain(layout, both.words[7], both.words[11] + 1, false, sealed_only, &trace)
            && trace.words[0] == kCorruptLineage,
            "sealed contribution query accepted staged records");
    candidate = graph.view(kSnapshot, kForkKind, fork.words[5], fork.words[6]);
    check_query(graph.view(kTruth, kForkKind, fork.words[5], fork.words[6], statement),
                candidate, kForkKind, fork.words[5], fork.words[6], statement, &both);
    const auto sealed = graph.view(kSeal, kForkKind, fork.words[5], fork.words[6]);
    require(sealed.words[0] == kOk, "native seal failed");
    const auto snapshot = graph.view(kSnapshot, kRootKind, sealed.words[3], sealed.words[4]);
    const auto query = graph.view(kTruth, kRootKind, sealed.words[3], sealed.words[4], statement);
    check_query(query, snapshot, kRootKind, sealed.words[3], sealed.words[4], statement, &both);
    Contributions contributions;
    trace = {};
    require(visit_support_chain(layout, query.words[7], query.words[11] + 1, false, contributions, &trace),
            "native contribution traversal refused a valid sealed chain");
    require(contributions.records.size() == 2 && contributions.records[0][0] == 7
            && contributions.records[0][1] == 6 && contributions.records[0][2] == 2
            && contributions.records[1][0] == 3 && contributions.records[1][1] == 5
            && contributions.records[1][2] == 1,
            "native contribution traversal changed the exact original support history");
    auto* head = version_record(layout, query.words[11]);
    const auto head_generation = head[1];
    head[1] = 0;
    require(graph.view(kTruth, kRootKind, sealed.words[3], sealed.words[4], statement).words[0]
            == kCorruptLineage, "truth query accepted a live head with generation zero");
    head[1] = head_generation;
    auto* queried_statement = statement_record(layout, query.words[7]);
    const auto statement_generation = queried_statement[1];
    queried_statement[1] = 0;
    head[4] = 0;
    require(graph.view(kTruth, kRootKind, sealed.words[3], sealed.words[4], statement).words[0]
            == kCorruptLineage, "truth query accepted a live statement with generation zero");
    queried_statement[1] = statement_generation;
    head[4] = statement_generation;
    const auto previous = head[7];
    head[7] = query.words[11] + 1;
    Contributions cycle;
    trace = {};
    require(!visit_support_chain(layout, query.words[7], query.words[11] + 1, false, cycle, &trace)
            && trace.words[0] == kCorruptLineage, "cyclic contribution history was accepted");
    head[7] = previous;
    const auto support_polarity = older_support[5];
    older_support[5] = 2;
    Contributions inconsistent;
    trace = {};
    require(!visit_support_chain(layout, query.words[7], query.words[11] + 1, false, inconsistent, &trace)
            && trace.words[0] == kCorruptLineage, "inconsistent historical truth was accepted");
    older_support[5] = support_polarity;
    // A support for the same logical statement in another root is not part of
    // this query, and reclaiming that root must preserve every current edge.
    const auto branch = graph.view(kFork, kRootKind, 0, 1);
    require(branch.words[0] == kOk, "unrelated branch fork failed");
    Command unrelated = insertion;
    unrelated.words[8] = branch.words[5];
    unrelated.words[9] = branch.words[6];
    unrelated.words[13] = 8;
    unrelated.words[14] = 2;
    unrelated.words[20] = 33;
    require(graph.run(unrelated).words[0] == kOk, "unrelated support insertion failed");
    const auto other_root = graph.view(kSeal, kForkKind, branch.words[5], branch.words[6]);
    require(other_root.words[0] == kOk, "unrelated root seal failed");
    require(graph.view(kTruth, kRootKind, other_root.words[3], other_root.words[4], statement).words[2] == 2,
            "independent negative support did not remain ContraOnly");
    Contributions isolated;
    trace = {};
    require(visit_support_chain(layout, query.words[7], query.words[11] + 1, false, isolated, &trace)
            && isolated.records == contributions.records, "unrelated root entered the current support proof");
    Command retire{};
    retire.words[0] = kRetireRoot;
    retire.words[8] = other_root.words[3];
    retire.words[9] = other_root.words[4];
    retire.words[10] = sealed.words[3];
    retire.words[11] = sealed.words[4];
    require(graph.run(retire).words[0] == kOk, "unrelated root retirement failed");
    require(graph.view(kTruth, kRootKind, other_root.words[3], other_root.words[4], statement).words[0]
            == kStaleGeneration, "retired root remained queryable");
    Contributions retained;
    trace = {};
    require(visit_support_chain(layout, query.words[7], query.words[11] + 1, false, retained, &trace)
            && retained.records == contributions.records, "root retirement lost current support generations");
    require(graph.view(kTruth, kRootKind, sealed.words[3], sealed.words[4] + 1, statement).words[0]
            == kStaleGeneration, "query accepted a stale root generation");

    // Exercise the resident admission path as well as the host command path.
    ResidentHandle handle = {NativeGraph::owner, kRootKind, sealed.words[3], sealed.words[4]};
    DecodedStatement decoded = {};
    for (size_t index = 0; index < statement.size(); ++index) {
        decoded.identity_words[index * 2] = static_cast<uint32_t>(statement[index]);
        decoded.identity_words[index * 2 + 1] = static_cast<uint32_t>(statement[index] >> 32);
    }
    Receipt resident = {};
    auto launch = graph.descriptor(resident);
    launch.admission = kResidentRootTruthAdmission;
    launch.handle_ptr = reinterpret_cast<uint64_t>(&handle);
    launch.decoded_statement_ptr = reinterpret_cast<uint64_t>(&decoded);
    execute(graph.arena.data(), launch);
    require(std::equal(query.words, query.words + kReceiptWords, resident.words),
            "resident truth admission does not use the canonical query producer");
    ResidentHandle empty_handle{NativeGraph::owner, kRootKind, 0, 1};
    Receipt reused_fork{};
    auto fork_launch = graph.descriptor(reused_fork);
    fork_launch.admission = kResidentForkAdmission;
    fork_launch.handle_ptr = reinterpret_cast<uint64_t>(&empty_handle);
    execute(graph.arena.data(), fork_launch);
    require(reused_fork.words[0] == kOk, "reused native fork failed");
    DecodedInput input{};
    std::copy(decoded.identity_words, decoded.identity_words + 8, input.statement_identity_words);
    input.polarity = 1;
    input.statement_record = 9;
    input.support_record = 4;
    input.provenance_words[0] = 41;
    input.source_words[0] = 42;
    input.context_words[0] = 43;
    input.scope_words[0] = 44;
    DecodedStatement materialized_statement{};
    DecodedSupport materialized_support{};
    Receipt materialized{};
    auto materialization = graph.descriptor(materialized);
    materialization.admission = kResidentMaterializeDecodedAdmission;
    materialization.decoded_input_ptr = reinterpret_cast<uint64_t>(&input);
    materialization.decoded_statement_ptr = reinterpret_cast<uint64_t>(&materialized_statement);
    materialization.decoded_support_ptr = reinterpret_cast<uint64_t>(&materialized_support);
    execute(graph.arena.data(), materialization);
    require(materialized.words[0] == kOk && materialized_statement.record == 9
            && materialized_support.record == 4, "resident materialization lost original occurrence tags");
    Receipt resident_insertion{};
    auto insert_launch = graph.descriptor(resident_insertion);
    insert_launch.admission = kResidentInsertSupportAdmission;
    insert_launch.source_receipt_ptr = reinterpret_cast<uint64_t>(&reused_fork);
    insert_launch.decoded_statement_ptr = reinterpret_cast<uint64_t>(&materialized_statement);
    insert_launch.decoded_support_ptr = reinterpret_cast<uint64_t>(&materialized_support);
    execute(graph.arena.data(), insert_launch);
    require(resident_insertion.words[0] == kOk && resident_insertion.words[1] == kOutcomeInserted,
            "resident insertion refused its actual decoded occurrence");
    const auto* resident_support = support_record(layout, resident_insertion.words[9]);
    require(resident_support[10] == 9 && resident_support[11] == 4 && resident_support[1] > 1,
            "resident insertion did not preserve original indices across slot reuse");
    input.statement_record = UINT32_MAX;
    input.support_record = 7;
    input.provenance_words[0] = 45;
    for (uint32_t field = 0; field < 10; ++field) input.reconstruction[field] = 70 + field;
    execute(graph.arena.data(), materialization);
    require(materialized.words[0] == kOk && materialized_statement.record == UINT32_MAX
            && materialized_support.record == 7
            && std::equal(input.reconstruction, input.reconstruction + 10,
                          materialized_statement.reconstruction),
            "resident materialization lost derived target reconstruction references");
    execute(graph.arena.data(), insert_launch);
    require(resident_insertion.words[0] == kOk && resident_insertion.words[1] == kOutcomeInserted,
            "resident insertion refused its derived decoded occurrence");
    const auto* resident_derived = support_record(layout, resident_insertion.words[9]);
    require(resident_derived[10] == UINT32_MAX && resident_derived[11] == 7,
            "resident insertion relabeled a derived target as original");
    for (uint32_t word = 0; word < 5; ++word) {
        const uint64_t packed = input.reconstruction[2 * word]
            | (static_cast<uint64_t>(input.reconstruction[2 * word + 1]) << 32);
        require(resident_derived[12 + word] == packed,
                "resident insertion lost packed derived reconstruction fields");
    }
    Command derived = insertion;
    derived.words[8] = reused_fork.words[5];
    derived.words[9] = reused_fork.words[6];
    derived.words[13] = UINT32_MAX;
    derived.words[14] = 5;
    derived.words[20] = 51;
    for (uint32_t word = 0; word < 5; ++word) derived.words[24 + word] = 61 + word;
    const auto derived_insertion = graph.run(derived);
    require(derived_insertion.words[0] == kOk && derived_insertion.words[1] == kOutcomeInserted,
            "derived target insertion failed");
    const auto* derived_support = support_record(layout, derived_insertion.words[9]);
    require(derived_support[10] == UINT32_MAX &&
            std::equal(derived.words + 24, derived.words + 29, derived_support + 12),
            "derived target lost its actual decoder reconstruction references");
    auto substituted = derived;
    substituted.words[13] = 0;
    require(graph.run(substituted).words[0] == kInvalidCommand,
            "derived reconstruction references were relabeled as original record zero");
    std::fill(substituted.words + 24, substituted.words + 29, 0);
    substituted.words[14] = 0;
    const auto equal_original = graph.run(substituted);
    require(equal_original.words[0] == kOk && equal_original.words[1] == kOutcomeUnchanged
            && derived_support[10] == UINT32_MAX && derived_support[11] == 5
            && std::equal(derived.words + 24, derived.words + 29, derived_support + 12),
            "equal original insertion replaced the successful derived source");
    std::cout << "native truth receipt fields verified\n";
}
