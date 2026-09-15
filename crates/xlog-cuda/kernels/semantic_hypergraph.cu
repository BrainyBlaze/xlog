#include <stdint.h>

namespace semantic_graph {

// Unit-tariff logical events of the native producer, not GPU instructions.
// A tally is local to its original owning execution. Its accounting arithmetic
// is excluded; no global state, saturation or observation-derived cap is used.
enum class NativeWorkEvent : unsigned {
    Command, TableSlot, ChainLink, Category, SortComparison, CdfStep,
    PhiloxRound, ShaBlock, CanonicalByte
};
struct NativeWorkTally {
    uint64_t units = 0;
    uint64_t events[9] = {};
    uint64_t overflow = 0;
};
__device__ void charge_native(NativeWorkTally *work, NativeWorkEvent event, uint64_t count) {
    if (!work || work->overflow) return;
    const unsigned kind = unsigned(event);
    if (count > UINT64_MAX - work->units || count > UINT64_MAX - work->events[kind]) {
        work->overflow = 1;
        return;
    }
    work->units += count;
    work->events[kind] += count;
}
__device__ uint64_t written_word(uint64_t value, NativeWorkTally *work) {
    charge_native(work, NativeWorkEvent::CanonicalByte, sizeof(value));
    return value;
}
__device__ uint8_t written_byte(uint8_t value, NativeWorkTally *work) {
    charge_native(work, NativeWorkEvent::CanonicalByte, sizeof(value));
    return value;
}


constexpr uint64_t kMagic = 0x584c4f4753484731ULL;
constexpr uint64_t kNull = UINT64_MAX;
constexpr uint64_t kFree = 0;
constexpr uint64_t kLive = 1;
constexpr uint64_t kStaged = 2;
constexpr uint64_t kSealed = 3;

constexpr uint64_t kControlWords = 16;
constexpr uint64_t kRootWords = 16;
constexpr uint64_t kCandidateWords = 16;
constexpr uint64_t kStatementWords = 8;
constexpr uint64_t kSupportWords = 17;
constexpr uint64_t kVersionWords = 16;
constexpr uint32_t kReceiptWords = 42;
// Version word 13 retains the root-local insertion ordinal across slot reuse.
constexpr uint64_t kAbiGeneration = 7;
constexpr uint32_t kAcquiredCandidateSlot = 40;
constexpr uint32_t kAcquiredCandidateGeneration = 41;

constexpr uint64_t kInitialize = 1;
constexpr uint64_t kFork = 2;
constexpr uint64_t kInsertSupport = 3;
constexpr uint64_t kDiscard = 4;
constexpr uint64_t kSeal = 5;
constexpr uint64_t kSnapshot = 6;
constexpr uint64_t kTruth = 7;
constexpr uint64_t kInspectStatement = 8;
constexpr uint64_t kInspectSupport = 9;
constexpr uint64_t kInspectVersion = 10;
constexpr uint64_t kPreflightTransition = 11;
constexpr uint64_t kRetireRoot = 12;

constexpr uint64_t kHostCommandAdmission = 0;
constexpr uint64_t kResidentEmptyRootHandleAdmission = 1;
constexpr uint64_t kResidentForkAdmission = 2;
constexpr uint64_t kResidentInsertSupportAdmission = 3;
constexpr uint64_t kResidentForkTruthAdmission = 4;
constexpr uint64_t kResidentSealAdmission = 5;
constexpr uint64_t kResidentConsumeTruthAdmission = 6;
constexpr uint64_t kResidentMaterializeDecodedAdmission = 7;
constexpr uint64_t kResidentPreflightTransitionAdmission = 8;
constexpr uint64_t kResidentRootTruthAdmission = 9;
constexpr uint64_t kResidentDiscardAdmission = 10;

constexpr uint64_t kOk = 0;
constexpr uint64_t kForeignOwner = 1;
constexpr uint64_t kSlotOutOfRange = 2;
constexpr uint64_t kStaleGeneration = 3;
constexpr uint64_t kInactive = 4;
constexpr uint64_t kNotReachable = 5;
constexpr uint64_t kStatementCapacity = 6;
constexpr uint64_t kSupportCapacity = 7;
constexpr uint64_t kVersionCapacity = 8;
constexpr uint64_t kRootCapacity = 9;
constexpr uint64_t kGenerationExhausted = 10;
constexpr uint64_t kCorruptLineage = 11;
constexpr uint64_t kInvalidCommand = 12;
constexpr uint64_t kArenaMismatch = 13;

constexpr uint64_t kOutcomeInserted = 1;
constexpr uint64_t kOutcomeUnchanged = 2;
constexpr uint64_t kInsertNewStatement = 1;
constexpr uint32_t kInsertPreviousTruthShift = 1;
constexpr uint64_t kInsertPreviousTruthMask = 3ULL << kInsertPreviousTruthShift;

constexpr uint64_t kRootKind = 1;
constexpr uint64_t kForkKind = 2;
constexpr uint64_t kStatementKind = 3;
constexpr uint64_t kSupportKind = 4;
constexpr uint64_t kVersionKind = 5;

struct Command {
    uint64_t words[32];
};

struct Receipt {
    // Insertion word 32: new-statement bit 0, actual previous truth bits 1..2.
    // Outcome word 1 distinguishes a new support attachment from a duplicate.
    // Successful truth queries retain the requested identity in words 20..23.
    // Zero statement/version generations (8/12) mean absent, not live slot zero.
    uint64_t words[kReceiptWords];
};

__device__ uint64_t inserted_support_count(const Receipt *receipt) {
    return receipt->words[0] == kOk && receipt->words[1] == kOutcomeInserted ? 1 : 0;
}

__device__ bool defined_truth_changed(const Receipt *receipt) {
    const uint64_t previous =
        (receipt->words[32] & kInsertPreviousTruthMask) >> kInsertPreviousTruthShift;
    return inserted_support_count(receipt) != 0 && previous != 0 &&
           previous != receipt->words[2];
}

struct DecodedStatement {
    uint32_t identity_words[8];
    uint32_t record;
    uint32_t reconstruction[10];
};

struct DecodedSupport {
    uint32_t polarity;
    uint32_t provenance_words[8];
    uint32_t source_words[8];
    uint32_t context_words[8];
    uint32_t scope_words[8];
    uint32_t record;
};

struct DecodedInput {
    uint32_t statement_identity_words[8];
    uint32_t polarity;
    uint32_t provenance_words[8];
    uint32_t source_words[8];
    uint32_t context_words[8];
    uint32_t scope_words[8];
    uint32_t statement_record;
    uint32_t support_record;
    uint32_t reconstruction[10];
};

struct TruthValue {
    uint64_t status;
    uint64_t truth;
    uint64_t owner;
    uint64_t reserved;
};

struct ResidentHandle {
    uint64_t owner;
    uint64_t kind;
    uint64_t slot;
    uint64_t generation;
};

struct LaunchDescriptor {
    uint64_t expected_owner;
    uint64_t root_capacity;
    uint64_t statement_capacity;
    uint64_t support_capacity;
    uint64_t version_capacity;
    uint64_t arena_words;
    uint64_t command_ptr;
    uint64_t command_index;
    uint64_t handle_ptr;
    uint64_t handle_index;
    uint64_t source_receipt_ptr;
    uint64_t source_receipt_index;
    uint64_t decoded_input_ptr;
    uint64_t decoded_input_index;
    uint64_t decoded_statement_ptr;
    uint64_t decoded_statement_index;
    uint64_t decoded_support_ptr;
    uint64_t decoded_support_index;
    uint64_t output_ptr;
    uint64_t output_index;
    uint64_t receipt_ptr;
    uint64_t receipt_index;
    uint64_t admission;
    uint64_t abi_generation;
};

static_assert(sizeof(Command) == 256, "semantic command ABI drift");
static_assert(sizeof(Receipt) == 336, "semantic receipt ABI drift");
static_assert(sizeof(DecodedStatement) == 76, "decoded statement ABI drift");
static_assert(sizeof(DecodedSupport) == 136, "decoded support ABI drift");
static_assert(sizeof(DecodedInput) == 212, "decoded input ABI drift");
static_assert(sizeof(ResidentHandle) == 32, "resident handle ABI drift");
static_assert(sizeof(TruthValue) == 32, "resident truth value ABI drift");
static_assert(sizeof(LaunchDescriptor) == 192, "semantic launch descriptor ABI drift");

struct Layout {
    NativeWorkTally *work = nullptr;
    uint64_t *control;
    uint64_t *roots;
    uint64_t *candidate;
    uint64_t *statements;
    uint64_t *supports;
    uint64_t *versions;
    uint64_t *root_heads;
    uint64_t *candidate_heads;
    uint64_t root_capacity;
    uint64_t statement_capacity;
    uint64_t support_capacity;
    uint64_t version_capacity;
    uint64_t arena_words;
};

__device__ bool checked_add(uint64_t left, uint64_t right, uint64_t *out) {
    *out = left + right;
    return *out >= left;
}

__device__ bool checked_mul(uint64_t left, uint64_t right, uint64_t *out) {
    if (left != 0 && right > UINT64_MAX / left) {
        return false;
    }
    *out = left * right;
    return true;
}

__device__ bool build_layout(uint64_t *arena, uint64_t roots, uint64_t statements,
                             uint64_t supports, uint64_t versions,
                             uint64_t arena_words, Layout *layout) {
    if (roots == 0 || statements == 0 || supports == 0 || versions == 0 ||
        roots > UINT32_MAX || statements > UINT32_MAX || supports > UINT32_MAX ||
        versions > UINT32_MAX) {
        return false;
    }
    uint64_t cursor = kControlWords;
    uint64_t product = 0;
    if (!checked_mul(roots, kRootWords, &product) || !checked_add(cursor, product, &cursor)) {
        return false;
    }
    uint64_t *candidate = arena + cursor;
    if (!checked_add(cursor, kCandidateWords, &cursor)) {
        return false;
    }
    uint64_t *statement_records = arena + cursor;
    if (!checked_mul(statements, kStatementWords, &product) ||
        !checked_add(cursor, product, &cursor)) {
        return false;
    }
    uint64_t *support_records = arena + cursor;
    if (!checked_mul(supports, kSupportWords, &product) ||
        !checked_add(cursor, product, &cursor)) {
        return false;
    }
    uint64_t *version_records = arena + cursor;
    if (!checked_mul(versions, kVersionWords, &product) ||
        !checked_add(cursor, product, &cursor)) {
        return false;
    }
    uint64_t *root_heads = arena + cursor;
    if (!checked_mul(roots, statements, &product) || !checked_add(cursor, product, &cursor)) {
        return false;
    }
    uint64_t *candidate_heads = arena + cursor;
    if (!checked_add(cursor, statements, &cursor) || cursor != arena_words) {
        return false;
    }
    layout->control = arena;
    layout->roots = arena + kControlWords;
    layout->candidate = candidate;
    layout->statements = statement_records;
    layout->supports = support_records;
    layout->versions = version_records;
    layout->root_heads = root_heads;
    layout->candidate_heads = candidate_heads;
    layout->root_capacity = roots;
    layout->statement_capacity = statements;
    layout->support_capacity = supports;
    layout->version_capacity = versions;
    layout->arena_words = cursor;
    return true;
}

__device__ uint64_t *root_record(const Layout &layout, uint64_t slot) {
    charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
    return layout.roots + slot * kRootWords;
}

__device__ uint64_t *statement_record(const Layout &layout, uint64_t slot) {
    charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
    return layout.statements + slot * kStatementWords;
}

__device__ uint64_t *support_record(const Layout &layout, uint64_t slot) {
    charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
    return layout.supports + slot * kSupportWords;
}

__device__ uint64_t *version_record(const Layout &layout, uint64_t slot) {
    charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
    return layout.versions + slot * kVersionWords;
}

__device__ void clear_receipt(Receipt *receipt, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, sizeof(Receipt));
    for (uint32_t index = 0; index < kReceiptWords; ++index) {
        receipt->words[index] = 0;
    }
}

__device__ void fail(Receipt *receipt, uint64_t status, uint64_t kind, uint64_t slot,
                     uint64_t presented, uint64_t current, uint64_t capacity,
                     uint64_t detail, NativeWorkTally *work = nullptr) {
    receipt->words[0] = written_word(status, work);
    receipt->words[33] = written_word(kind, work);
    receipt->words[34] = written_word(slot, work);
    receipt->words[35] = written_word(presented, work);
    receipt->words[36] = written_word(current, work);
    receipt->words[37] = written_word(capacity, work);
    receipt->words[38] = written_word(detail, work);
}

__device__ void copy_receipt(Receipt *destination, const Receipt *source, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, sizeof(Receipt));
    for (uint32_t index = 0; index < kReceiptWords; ++index) {
        destination->words[index] = source->words[index];
    }
}

__device__ void decode_identity(const uint32_t *source, uint64_t *destination, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, 32);
    for (uint32_t index = 0; index < 4; ++index) {
        destination[index] = static_cast<uint64_t>(source[index * 2]) |
                             (static_cast<uint64_t>(source[index * 2 + 1]) << 32);
    }
}

__device__ void dwords_to_bytes(const uint32_t *words, uint8_t *bytes, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, 32);
    for (uint32_t word = 0; word < 8; ++word) {
        for (uint32_t byte = 0; byte < 4; ++byte) {
            bytes[word * 4 + byte] = static_cast<uint8_t>(words[word] >> (byte * 8));
        }
    }
}

__device__ bool load_resident_handle(const LaunchDescriptor &descriptor,
                                     const Receipt *source_receipt,
                                     uint64_t expected_kind, Receipt *receipt,
                                     uint64_t *slot, uint64_t *generation, NativeWorkTally *work = nullptr) {
    const bool has_bank = descriptor.handle_ptr != 0;
    const bool has_receipt = source_receipt != nullptr;
    if (has_bank == has_receipt) {
        fail(receipt, kInvalidCommand, expected_kind, 0, 0, 0, 0, 1, work);
        return false;
    }
    uint64_t owner = 0;
    uint64_t kind = 0;
    if (has_bank) {
        const ResidentHandle *handle =
            reinterpret_cast<const ResidentHandle *>(descriptor.handle_ptr) +
            descriptor.handle_index;
        owner = handle->owner;
        kind = handle->kind;
        *slot = handle->slot;
        *generation = handle->generation;
    } else {
        if (source_receipt->words[39] != descriptor.expected_owner) {
            fail(receipt, kForeignOwner, expected_kind, 0, source_receipt->words[39],
                 descriptor.expected_owner, 0, 0, work);
            return false;
        }
        if (source_receipt->words[0] != kOk) {
            copy_receipt(receipt, source_receipt, work);
            return false;
        }
        owner = source_receipt->words[39];
        kind = source_receipt->words[33];
        *slot = source_receipt->words[34];
        *generation = source_receipt->words[35];
    }
    if (owner != descriptor.expected_owner) {
        fail(receipt, kForeignOwner, expected_kind, *slot, owner,
             descriptor.expected_owner, 0, 0, work);
        return false;
    }
    if (kind != expected_kind) {
        fail(receipt, kInvalidCommand, expected_kind, *slot, kind, expected_kind, 0, 2, work);
        return false;
    }
    return true;
}

__device__ bool inherit_candidate_acquisition(const Receipt *source, uint64_t slot,
                                              uint64_t generation, Receipt *receipt, NativeWorkTally *work = nullptr) {
    if (source == nullptr || source->words[kAcquiredCandidateGeneration] == 0 ||
        source->words[kAcquiredCandidateSlot] != slot ||
        source->words[kAcquiredCandidateGeneration] != generation) {
        fail(receipt, kInvalidCommand, kForkKind, slot, generation, 0, 0, 11, work);
        return false;
    }
    receipt->words[kAcquiredCandidateSlot] = written_word(slot, work);
    receipt->words[kAcquiredCandidateGeneration] = written_word(generation, work);
    return true;
}

__device__ bool equal_identity(const uint64_t *left, const uint64_t *right) {
    return left[0] == right[0] && left[1] == right[1] && left[2] == right[2] &&
           left[3] == right[3];
}

__device__ void copy_identity(uint64_t *destination, const uint64_t *source, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, 32);
    destination[0] = source[0];
    destination[1] = source[1];
    destination[2] = source[2];
    destination[3] = source[3];
}

__device__ void words_to_bytes(const uint64_t *words, uint8_t *bytes, NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::CanonicalByte, 32);
    for (uint32_t word = 0; word < 4; ++word) {
        for (uint32_t byte = 0; byte < 8; ++byte) {
            bytes[word * 8 + byte] = static_cast<uint8_t>(words[word] >> (byte * 8));
        }
    }
}

__device__ uint32_t rotate_right(uint32_t value, uint32_t count) {
    return (value >> count) | (value << (32 - count));
}

template<class Bytes>
__device__ void sha256(Bytes input, uint64_t length, uint64_t *output, NativeWorkTally *work = nullptr) {
    constexpr uint32_t constants[64] = {
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1,
        0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3,
        0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786,
        0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147,
        0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13,
        0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
        0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208,
        0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    };
    const uint64_t block_count = (length + 9 + 63) / 64;
    const uint64_t bit_length = static_cast<uint64_t>(length) * 8;
    const uint64_t length_offset = block_count * 64 - 8;
    uint32_t state[8] = {0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
                         0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19};
    for (uint64_t block = 0; block < block_count; ++block) {
        charge_native(work, NativeWorkEvent::ShaBlock, 1);
        uint32_t schedule[64];
        for (uint32_t index = 0; index < 16; ++index) {
            uint32_t word = 0;
            for (uint32_t byte = 0; byte < 4; ++byte) {
                uint64_t offset = block * 64 + index * 4 + byte;
                uint8_t value = offset < length ? input[offset] :
                    offset == length ? 0x80 : offset >= length_offset ?
                    uint8_t(bit_length >> ((block_count * 64 - 1 - offset) * 8)) : 0;
                word = (word << 8) | value;
            }
            schedule[index] = word;
        }
        for (uint32_t index = 16; index < 64; ++index) {
            const uint32_t a = schedule[index - 15];
            const uint32_t b = schedule[index - 2];
            const uint32_t s0 = rotate_right(a, 7) ^ rotate_right(a, 18) ^ (a >> 3);
            const uint32_t s1 = rotate_right(b, 17) ^ rotate_right(b, 19) ^ (b >> 10);
            schedule[index] = schedule[index - 16] + s0 + schedule[index - 7] + s1;
        }
        uint32_t a = state[0];
        uint32_t b = state[1];
        uint32_t c = state[2];
        uint32_t d = state[3];
        uint32_t e = state[4];
        uint32_t f = state[5];
        uint32_t g = state[6];
        uint32_t h = state[7];
        for (uint32_t index = 0; index < 64; ++index) {
            const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temp1 = h + sum1 + choice + constants[index] + schedule[index];
            const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temp2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state[0] += a;
        state[1] += b;
        state[2] += c;
        state[3] += d;
        state[4] += e;
        state[5] += f;
        state[6] += g;
        state[7] += h;
    }
    charge_native(work, NativeWorkEvent::CanonicalByte, 32);
    uint8_t digest[32];
    for (uint32_t index = 0; index < 8; ++index) {
        digest[index * 4] = static_cast<uint8_t>(state[index] >> 24);
        digest[index * 4 + 1] = static_cast<uint8_t>(state[index] >> 16);
        digest[index * 4 + 2] = static_cast<uint8_t>(state[index] >> 8);
        digest[index * 4 + 3] = static_cast<uint8_t>(state[index]);
    }
    for (uint32_t word = 0; word < 4; ++word) {
        uint64_t value = 0;
        for (uint32_t byte = 0; byte < 8; ++byte) {
            value |= static_cast<uint64_t>(digest[word * 8 + byte]) << (byte * 8);
        }
        output[word] = value;
    }
}

__device__ void support_digest(const uint64_t *statement,
                               const DecodedSupport &support, uint64_t *output, NativeWorkTally *work = nullptr) {
    uint8_t input[193];
    constexpr uint8_t domain[] = "xlog.semantic.support.v1";
    for (uint32_t index = 0; index < sizeof(domain); ++index) {
        input[index] = written_byte(domain[index], work);
    }
    words_to_bytes(statement, input + 25, work);
    for (uint32_t byte = 0; byte < 4; ++byte) {
        input[57 + byte] = written_byte(static_cast<uint8_t>(support.polarity >> (byte * 8)), work);
        input[61 + byte] = written_byte(static_cast<uint8_t>(1U >> (byte * 8)), work);
    }
    dwords_to_bytes(support.provenance_words, input + 65, work);
    dwords_to_bytes(support.source_words, input + 97, work);
    dwords_to_bytes(support.context_words, input + 129, work);
    dwords_to_bytes(support.scope_words, input + 161, work);
    sha256(input, 193, output, work);
}

__device__ void root_digest(const uint64_t *previous, const uint64_t *version,
                            uint32_t statements, uint32_t supports, uint32_t versions,
                            uint64_t *output, NativeWorkTally *work = nullptr) {
    uint8_t input[108];
    for (uint32_t index = 0; index < 108; ++index) {
        input[index] = written_byte(0, work);
    }
    constexpr uint8_t domain[] = "xlog.semantic.root.v1";
    for (uint32_t index = 0; index < sizeof(domain); ++index) {
        input[index] = written_byte(domain[index], work);
    }
    words_to_bytes(previous, input + 32, work);
    words_to_bytes(version, input + 64, work);
    const uint32_t extents[3] = {statements, supports, versions};
    for (uint32_t extent = 0; extent < 3; ++extent) {
        for (uint32_t byte = 0; byte < 4; ++byte) {
            input[96 + extent * 4 + byte] = written_byte(static_cast<uint8_t>(extents[extent] >> (byte * 8)), work);
        }
    }
    sha256(input, 108, output, work);
}

__device__ void version_digest(const uint64_t *previous, const uint64_t *support,
                               uint64_t truth, uint64_t *output, NativeWorkTally *work = nullptr) {
    uint8_t input[104];
    for (uint32_t index = 0; index < 104; ++index) {
        input[index] = written_byte(0, work);
    }
    constexpr uint8_t domain[] = "xlog.semantic.version.v1";
    for (uint32_t index = 0; index < sizeof(domain); ++index) {
        input[index] = written_byte(domain[index], work);
    }
    words_to_bytes(previous, input + 32, work);
    words_to_bytes(support, input + 64, work);
    for (uint32_t byte = 0; byte < 8; ++byte) {
        input[96 + byte] = written_byte(static_cast<uint8_t>(truth >> (byte * 8)), work);
    }
    sha256(input, 104, output, work);
}

__device__ bool validate_control(const Layout &layout, uint64_t expected_owner,
                                  const Command *command, Receipt *receipt) {
    if (layout.control[0] != kMagic || layout.control[1] != expected_owner) {
        fail(receipt, kForeignOwner, kRootKind, 0, expected_owner, layout.control[1], 0,
             layout.control[0], layout.work);
        return false;
    }
    if (layout.control[2] != layout.root_capacity ||
        layout.control[3] != layout.statement_capacity ||
        layout.control[4] != layout.support_capacity ||
        layout.control[5] != layout.version_capacity ||
        layout.control[6] != layout.arena_words) {
        fail(receipt, kArenaMismatch, kRootKind, 0, layout.arena_words, layout.control[6], 0, 0, layout.work);
        return false;
    }
    if (command->words[1] != expected_owner) {
        fail(receipt, kForeignOwner, kRootKind, 0, command->words[1], expected_owner, 0, 0, layout.work);
        return false;
    }
    return true;
}

__device__ bool validate_root(const Layout &layout, uint64_t slot, uint64_t generation,
                              Receipt *receipt) {
    if (slot >= layout.root_capacity) {
        fail(receipt, kSlotOutOfRange, kRootKind, slot, generation, 0,
             layout.root_capacity, 0, layout.work);
        return false;
    }
    uint64_t *root = root_record(layout, slot);
    if (root[1] != generation) {
        fail(receipt, kStaleGeneration, kRootKind, slot, generation, root[1],
             layout.root_capacity, 0, layout.work);
        return false;
    }
    if (root[0] != kSealed) {
        fail(receipt, kInactive, kRootKind, slot, generation, root[1],
             layout.root_capacity, root[0], layout.work);
        return false;
    }
    return true;
}

__device__ bool validate_candidate(const Layout &layout, uint64_t slot, uint64_t generation,
                                   Receipt *receipt) {
    if (slot != 0) {
        fail(receipt, kSlotOutOfRange, kForkKind, slot, generation, 0, 1, 0, layout.work);
        return false;
    }
    if (layout.candidate[1] != generation) {
        fail(receipt, kStaleGeneration, kForkKind, slot, generation,
             layout.candidate[1], 1, 0, layout.work);
        return false;
    }
    if (layout.candidate[0] != kLive) {
        fail(receipt, kInactive, kForkKind, slot, generation, layout.candidate[1], 1,
             layout.candidate[0], layout.work);
        return false;
    }
    return true;
}

__device__ bool validate_view(const Layout &layout, const Command *command, Receipt *receipt,
                              uint64_t **record, uint64_t **heads) {
    if (command->words[7] == 1) {
        const uint64_t slot = command->words[8];
        if (!validate_root(layout, slot, command->words[9], receipt)) {
            return false;
        }
        *record = root_record(layout, slot);
        *heads = layout.root_heads + slot * layout.statement_capacity;
        return true;
    }
    if (command->words[7] == 2) {
        if (!validate_candidate(layout, command->words[8], command->words[9], receipt)) {
            return false;
        }
        *record = layout.candidate;
        *heads = layout.candidate_heads;
        return true;
    }
    fail(receipt, kInvalidCommand, kRootKind, command->words[8], command->words[9], 0,
         0, command->words[7], layout.work);
    return false;
}

__device__ bool record_is_live(uint64_t state) {
    return state == kStaged || state == kSealed;
}

__device__ bool validate_head(const Layout &layout, uint64_t statement_slot,
                              uint64_t encoded_head, Receipt *receipt) {
    if (encoded_head == 0) {
        return true;
    }
    const uint64_t version_slot = encoded_head - 1;
    if (statement_slot >= layout.statement_capacity || version_slot >= layout.version_capacity) {
        fail(receipt, kCorruptLineage, kVersionKind, version_slot, 0, 0,
             layout.version_capacity, 1, layout.work);
        return false;
    }
    uint64_t *version = version_record(layout, version_slot);
    uint64_t *statement = statement_record(layout, statement_slot);
    if (!record_is_live(statement[0]) || statement[1] == 0 ||
        !record_is_live(version[0]) || version[1] == 0 || version[3] != statement_slot ||
        version[4] != statement[1]) {
        fail(receipt, kCorruptLineage, kVersionKind, version_slot, version[4],
             statement_record(layout, statement_slot)[1], layout.version_capacity, 2, layout.work);
        return false;
    }
    return true;
}

__device__ uint64_t find_statement(const Layout &layout, uint64_t *heads,
                                   const uint64_t *identity, Receipt *receipt) {
    for (uint64_t slot = 0; slot < layout.statement_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        if (heads[slot] == 0) {
            continue;
        }
        uint64_t *statement = statement_record(layout, slot);
        if (!record_is_live(statement[0]) || !validate_head(layout, slot, heads[slot], receipt)) {
            if (receipt->words[0] == kOk) {
                fail(receipt, kCorruptLineage, kStatementKind, slot, statement[1],
                     statement[1], layout.statement_capacity, 3, layout.work);
            }
            return kNull;
        }
        if (equal_identity(statement + 3, identity)) {
            return slot;
        }
    }
    return kNull;
}

__device__ bool validate_version_link(const Layout &layout, uint64_t statement_slot,
                                      uint64_t version_slot, Receipt *receipt) {
    if (statement_slot >= layout.statement_capacity || version_slot >= layout.version_capacity) {
        fail(receipt, kCorruptLineage, kVersionKind, version_slot, 0, 0,
             layout.version_capacity, 4, layout.work);
        return false;
    }
    uint64_t *version = version_record(layout, version_slot);
    uint64_t *statement = statement_record(layout, statement_slot);
    if (!record_is_live(statement[0]) || statement[1] == 0 ||
        !record_is_live(version[0]) || version[1] == 0 || version[3] != statement_slot ||
        version[4] != statement[1] || version[5] >= layout.support_capacity) {
        fail(receipt, kCorruptLineage, kVersionKind, version_slot, version[4],
             statement[1], layout.version_capacity, 5, layout.work);
        return false;
    }
    uint64_t *support = support_record(layout, version[5]);
    if (!record_is_live(support[0]) || support[1] == 0 || support[1] != version[6] ||
        support[3] != statement_slot || support[4] != statement[1]) {
        fail(receipt, kCorruptLineage, kSupportKind, version[5], version[6], support[1],
             layout.support_capacity, 6, layout.work);
        return false;
    }
    return true;
}

// Visit the complete current history. A collector's writes remain provisional
// until this returns true; even a matching head cannot hide a malformed tail.
// Candidate callers permit staged records only from their current fork owner.
template <typename Visitor>
__device__ bool visit_support_chain(const Layout &layout, uint64_t statement_slot,
                                    uint64_t encoded_head, bool candidate,
                                    Visitor &visitor, Receipt *receipt) {
    if (statement_slot >= layout.statement_capacity) {
        fail(receipt, kCorruptLineage, kStatementKind, statement_slot, 0, 0,
             layout.statement_capacity, 15, layout.work);
        return false;
    }
    const uint64_t *statement = statement_record(layout, statement_slot);
    if ((!candidate && statement[0] != kSealed) ||
        (candidate && !record_is_live(statement[0])) || statement[1] == 0 ||
        (statement[0] == kStaged && statement[2] != layout.candidate[1])) {
        fail(receipt, kCorruptLineage, kStatementKind, statement_slot, statement[1],
             statement[1], layout.statement_capacity, 16, layout.work);
        return false;
    }
    uint64_t encoded = encoded_head;
    uint64_t newer_ordinal = UINT64_MAX;
    for (uint64_t steps = 0; encoded != 0 && steps < layout.version_capacity; ++steps) {
        charge_native(layout.work, NativeWorkEvent::ChainLink, 1);
        const uint64_t slot = encoded - 1;
        if (!validate_version_link(layout, statement_slot, slot, receipt)) {
            return false;
        }
        const uint64_t *version = version_record(layout, slot);
        const uint64_t *support = support_record(layout, version[5]);
        if ((!candidate && (version[0] != kSealed || support[0] != kSealed)) ||
            (version[0] == kStaged && version[2] != layout.candidate[1]) ||
            (support[0] == kStaged && support[2] != layout.candidate[1]) ||
            version[1] == 0 || support[1] == 0 || version[13] == 0 ||
            version[13] >= newer_ordinal || (support[5] != 1 && support[5] != 2) ||
            support[10] > UINT32_MAX || support[11] > UINT32_MAX) {
            fail(receipt, kCorruptLineage, kVersionKind, slot, version[1], version[1],
                 layout.version_capacity, 17, layout.work);
            return false;
        }
        if (support[10] != UINT32_MAX) {
            for (uint32_t word = 12; word < kSupportWords; ++word) {
                if (support[word] != 0) {
                    fail(receipt, kCorruptLineage, kSupportKind, version[5], support[1],
                         support[1], layout.support_capacity, 19, layout.work);
                    return false;
                }
            }
        }
        uint64_t previous_identity[4] = {0, 0, 0, 0};
        uint64_t previous_truth = 0;
        if (version[7] != 0) {
            const uint64_t previous_slot = version[7] - 1;
            if (!validate_version_link(layout, statement_slot, previous_slot, receipt)) {
                return false;
            }
            const uint64_t *previous = version_record(layout, previous_slot);
            copy_identity(previous_identity, previous + 9, layout.work);
            previous_truth = previous[8];
        }
        uint64_t identity[4];
        version_digest(previous_identity, support + 6, previous_truth | support[5], identity, layout.work);
        if (previous_truth > 3 || version[8] != (previous_truth | support[5]) ||
            !equal_identity(identity, version + 9)) {
            fail(receipt, kCorruptLineage, kVersionKind, slot, version[1], version[1],
                 layout.version_capacity, 18, layout.work);
            return false;
        }
        if (!visitor(slot, version, support)) {
            return false;
        }
        newer_ordinal = version[13];
        encoded = version[7];
    }
    if (encoded != 0) {
        fail(receipt, kCorruptLineage, kVersionKind, encoded - 1, 0, 0,
             layout.version_capacity, 7, layout.work);
        return false;
    }
    return true;
}

struct FindSupportEvent {
    const uint64_t *identity;
    uint64_t found;
    __device__ bool operator()(uint64_t slot, const uint64_t *, const uint64_t *support) {
        if (equal_identity(support + 6, identity)) found = slot;
        return true;
    }
};

__device__ uint64_t find_event_version(const Layout &layout, uint64_t statement_slot,
                                       uint64_t encoded_head, const uint64_t *support_identity,
                                       Receipt *receipt) {
    FindSupportEvent visitor{support_identity, kNull};
    return visit_support_chain(layout, statement_slot, encoded_head, true, visitor, receipt)
               ? visitor.found : kNull;
}

__device__ uint64_t find_free(const Layout &layout, uint64_t *records, uint64_t capacity,
                              uint64_t width) {
    (void)layout;
    for (uint64_t slot = 0; slot < capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        if (records[slot * width] == kFree) {
            return slot;
        }
    }
    return kNull;
}

__device__ void fill_extents_and_digest(Receipt *receipt, const uint64_t *record, NativeWorkTally *work = nullptr) {
    receipt->words[13] = written_word(record[3], work);
    receipt->words[14] = written_word(record[4], work);
    receipt->words[15] = written_word(record[5], work);
    copy_identity(receipt->words + 16, record + 6, work);
}

__device__ void fill_view(Receipt *receipt, const Command *command, const uint64_t *record, NativeWorkTally *work = nullptr) {
    const uint64_t kind = command->words[7];
    const uint32_t handle_offset = kind == kRootKind ? 3 : 5;
    receipt->words[handle_offset] = written_word(command->words[8], work);
    receipt->words[handle_offset + 1] = written_word(command->words[9], work);
    receipt->words[33] = written_word(kind, work);
    receipt->words[34] = written_word(command->words[8], work);
    receipt->words[35] = written_word(command->words[9], work);
    if (kind == kRootKind) {
        fill_extents_and_digest(receipt, record, work);
    } else {
        receipt->words[13] = written_word(record[4], work);
        receipt->words[14] = written_word(record[5], work);
        receipt->words[15] = written_word(record[6], work);
        copy_identity(receipt->words + 16, record + 7, work);
    }
}

__device__ void fill_statement(Receipt *receipt, const Layout &layout, uint64_t slot) {
    uint64_t *statement = statement_record(layout, slot);
    receipt->words[7] = written_word(slot, layout.work);
    receipt->words[8] = written_word(statement[1], layout.work);
    copy_identity(receipt->words + 20, statement + 3, layout.work);
}

__device__ void fill_support(Receipt *receipt, const Layout &layout, uint64_t slot) {
    uint64_t *support = support_record(layout, slot);
    receipt->words[9] = written_word(slot, layout.work);
    receipt->words[10] = written_word(support[1], layout.work);
    copy_identity(receipt->words + 24, support + 6, layout.work);
}

__device__ void fill_version(Receipt *receipt, const Layout &layout, uint64_t slot) {
    uint64_t *version = version_record(layout, slot);
    receipt->words[2] = written_word(version[8], layout.work);
    receipt->words[11] = written_word(slot, layout.work);
    receipt->words[12] = written_word(version[1], layout.work);
    copy_identity(receipt->words + 28, version + 9, layout.work);
}

__device__ void initialize(Layout &layout, uint64_t expected_owner, Receipt *receipt) {
    for (uint64_t word = 0; word < layout.arena_words; ++word) {
        layout.control[word] = written_word(0, layout.work);
    }
    layout.control[0] = written_word(kMagic, layout.work);
    layout.control[1] = written_word(expected_owner, layout.work);
    layout.control[2] = written_word(layout.root_capacity, layout.work);
    layout.control[3] = written_word(layout.statement_capacity, layout.work);
    layout.control[4] = written_word(layout.support_capacity, layout.work);
    layout.control[5] = written_word(layout.version_capacity, layout.work);
    layout.control[6] = written_word(layout.arena_words, layout.work);
    for (uint64_t slot = 0; slot < layout.root_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        root_record(layout, slot)[1] = 1;
    }
    layout.candidate[1] = written_word(1, layout.work);
    for (uint64_t slot = 0; slot < layout.statement_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        statement_record(layout, slot)[1] = 1;
    }
    for (uint64_t slot = 0; slot < layout.support_capacity; ++slot) {
        support_record(layout, slot)[1] = 1;
    }
    for (uint64_t slot = 0; slot < layout.version_capacity; ++slot) {
        version_record(layout, slot)[1] = 1;
    }
    uint64_t zero[4] = {0, 0, 0, 0};
    uint64_t *root = root_record(layout, 0);
    root[1] = written_word(1, layout.work);
    root[2] = written_word(kNull, layout.work);
    root[3] = written_word(0, layout.work);
    root[4] = written_word(0, layout.work);
    root[5] = written_word(0, layout.work);
    root_digest(zero, zero, 0, 0, 0, root + 6, layout.work);
    root[0] = written_word(kSealed, layout.work);
    receipt->words[3] = written_word(0, layout.work);
    receipt->words[4] = written_word(1, layout.work);
    fill_extents_and_digest(receipt, root, layout.work);
}

__device__ bool preflight_candidate_retirement(const Layout &layout, Receipt *receipt) {
    const uint64_t owner = layout.candidate[1];
    if (layout.candidate[1] == UINT64_MAX) {
        fail(receipt, kGenerationExhausted, kForkKind, 0, owner, owner, 1, 0, layout.work);
        return false;
    }
    return true;
}

// Full worst-case budget for two independent lanes, each with at most two
// insertions. This is read-only: the caller must retain exclusive arena ownership
// from this check through both terminals. It is not a transferable reservation.
__device__ void preflight_transition(const Layout &layout, const Command *command,
                                     Receipt *receipt) {
    const uint64_t base_slot = command->words[8];
    if (!validate_root(layout, base_slot, command->words[9], receipt)) {
        return;
    }
    const uint64_t candidate_generation = layout.candidate[1];
    if (layout.candidate[0] != kFree) {
        fail(receipt, kInactive, kForkKind, 0, candidate_generation,
             candidate_generation, 1, layout.candidate[0], layout.work);
        return;
    }
    if (candidate_generation > UINT64_MAX - 2) {
        fail(receipt, kGenerationExhausted, kForkKind, 0, candidate_generation,
             candidate_generation, 1, 2, layout.work);
        return;
    }
    const uint64_t *base = root_record(layout, base_slot);
    for (uint32_t extent = 3; extent < 6; ++extent) {
        if (base[extent] > UINT32_MAX - 2) {
            fail(receipt, kCorruptLineage, kRootKind, base_slot, base[1], base[1],
                 layout.root_capacity, extent, layout.work);
            return;
        }
    }
    uint64_t free_roots = 0;
    for (uint64_t slot = 0; slot < layout.root_capacity && free_roots < 2; ++slot) {
        const uint64_t *root = root_record(layout, slot);
        if (root[0] != kFree) {
            continue;
        }
        if (root[1] == UINT64_MAX) {
            fail(receipt, kGenerationExhausted, kRootKind, slot, root[1], root[1],
                 layout.root_capacity, 1, layout.work);
            return;
        }
        ++free_roots;
    }
    if (free_roots < 2) {
        fail(receipt, kRootCapacity, kRootKind, 0, 0, 0, layout.root_capacity, 2, layout.work);
        return;
    }
    const struct {
        uint64_t *records;
        uint64_t capacity, width, kind, capacity_status;
    } groups[] = {
        {layout.statements, layout.statement_capacity, kStatementWords, kStatementKind, kStatementCapacity},
        {layout.supports, layout.support_capacity, kSupportWords, kSupportKind, kSupportCapacity},
        {layout.versions, layout.version_capacity, kVersionWords, kVersionKind, kVersionCapacity},
    };
    for (const auto &group : groups) {
        uint32_t free_records = 0;
        for (uint64_t slot = 0; slot < group.capacity && free_records < 4; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            const uint64_t *record = group.records + slot * group.width;
            if (record[0] != kFree) {
                continue;
            }
            // First-fit reuses the first lane's records after a discard. Their
            // generations must survive that retirement AND a second insertion
            // with a possible second discard. Sealing does not bump record gens.
            const uint64_t retirements = free_records < 2 ? 2 : 1;
            if (record[1] > UINT64_MAX - retirements) {
                fail(receipt, kGenerationExhausted, group.kind, slot, record[1],
                     record[1], group.capacity, retirements, layout.work);
                return;
            }
            ++free_records;
        }
        if (free_records < 4) {
            fail(receipt, group.capacity_status, group.kind, 0, 0, 0, group.capacity, 4, layout.work);
            return;
        }
    }
    const uint64_t *heads = layout.root_heads + base_slot * layout.statement_capacity;
    for (uint64_t slot = 0; slot < layout.statement_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        if (heads[slot] == 0) {
            continue;
        }
        const uint64_t *statement = statement_record(layout, slot);
        if (statement[0] != kSealed || !validate_head(layout, slot, heads[slot], receipt)) {
            if (receipt->words[0] == kOk) {
                fail(receipt, kCorruptLineage, kStatementKind, slot, statement[1],
                     statement[1], layout.statement_capacity, statement[0], layout.work);
            }
            return;
        }
        if (statement[1] == UINT64_MAX) {
            fail(receipt, kGenerationExhausted, kStatementKind, slot, statement[1],
                 statement[1], layout.statement_capacity, 1, layout.work);
            return;
        }
    }
    receipt->words[3] = written_word(base_slot, layout.work);
    receipt->words[4] = written_word(base[1], layout.work);
    fill_extents_and_digest(receipt, base, layout.work);
}

__device__ void fork_root(Layout &layout, const Command *command, Receipt *receipt) {
    const uint64_t root_slot = command->words[8];
    if (!validate_root(layout, root_slot, command->words[9], receipt)) {
        return;
    }
    if (layout.candidate[0] != kFree) {
        fail(receipt, kInactive, kForkKind, 0, layout.candidate[1], layout.candidate[1], 1,
             layout.candidate[0], layout.work);
        return;
    }
    if (!preflight_candidate_retirement(layout, receipt)) {
        return;
    }
    uint64_t *root = root_record(layout, root_slot);
    layout.candidate[2] = written_word(root_slot, layout.work);
    layout.candidate[3] = written_word(root[1], layout.work);
    layout.candidate[4] = written_word(root[3], layout.work);
    layout.candidate[5] = written_word(root[4], layout.work);
    layout.candidate[6] = written_word(root[5], layout.work);
    copy_identity(layout.candidate + 7, root + 6, layout.work);
    for (uint64_t statement = 0; statement < layout.statement_capacity; ++statement) {
        layout.candidate_heads[statement] = written_word(layout.root_heads[root_slot * layout.statement_capacity + statement], layout.work);
    }
    layout.candidate[0] = written_word(kLive, layout.work);
    receipt->words[5] = written_word(0, layout.work);
    receipt->words[6] = written_word(layout.candidate[1], layout.work);
    receipt->words[13] = written_word(layout.candidate[4], layout.work);
    receipt->words[14] = written_word(layout.candidate[5], layout.work);
    receipt->words[15] = written_word(layout.candidate[6], layout.work);
    copy_identity(receipt->words + 16, layout.candidate + 7, layout.work);
}

__device__ void insert_support(Layout &layout, const Command *command, Receipt *receipt) {
    if (!validate_candidate(layout, command->words[8], command->words[9], receipt)) {
        return;
    }
    if (command->words[13] > UINT32_MAX || command->words[14] > UINT32_MAX) {
        fail(receipt, kInvalidCommand, kSupportKind, 0, 0, 0, 0, 10, layout.work);
        return;
    }
    if (command->words[13] != UINT32_MAX) {
        for (uint32_t word = 24; word < 29; ++word) {
            if (command->words[word] != 0) {
                fail(receipt, kInvalidCommand, kStatementKind, 0, 0, 0, 0, 19, layout.work);
                return;
            }
        }
    }
    const uint64_t *statement_identity = command->words + 16;
    const uint64_t *support_identity = command->words + 20;
    uint64_t statement_slot =
        find_statement(layout, layout.candidate_heads, statement_identity, receipt);
    if (receipt->words[0] != kOk) {
        return;
    }
    const bool new_statement = statement_slot == kNull;
    uint64_t encoded_head = 0;
    if (!new_statement) {
        encoded_head = layout.candidate_heads[statement_slot];
        const uint64_t existing = find_event_version(
            layout, statement_slot, encoded_head, support_identity, receipt);
        if (receipt->words[0] != kOk) {
            return;
        }
        if (existing != kNull) {
            uint64_t *version = version_record(layout, existing);
            fill_statement(receipt, layout, statement_slot);
            fill_support(receipt, layout, version[5]);
            fill_version(receipt, layout, existing);
            receipt->words[1] = written_word(kOutcomeUnchanged, layout.work);
            // The duplicate version can precede the head; retain current truth
            // separately without attaching a support or changing that version.
            receipt->words[32] = written_word(version_record(layout, encoded_head - 1)[8] << kInsertPreviousTruthShift, layout.work);
            return;
        }
    }
    if (new_statement) {
        statement_slot = find_free(layout, layout.statements, layout.statement_capacity,
                                   kStatementWords);
        if (statement_slot == kNull) {
            fail(receipt, kStatementCapacity, kStatementKind, 0, 0, 0,
                 layout.statement_capacity, 0, layout.work);
            return;
        }
    }
    const uint64_t support_slot =
        find_free(layout, layout.supports, layout.support_capacity, kSupportWords);
    if (support_slot == kNull) {
        fail(receipt, kSupportCapacity, kSupportKind, 0, 0, 0,
             layout.support_capacity, 0, layout.work);
        return;
    }
    const uint64_t version_slot =
        find_free(layout, layout.versions, layout.version_capacity, kVersionWords);
    if (version_slot == kNull) {
        fail(receipt, kVersionCapacity, kVersionKind, 0, 0, 0,
             layout.version_capacity, 0, layout.work);
        return;
    }
    uint64_t *statement = statement_record(layout, statement_slot);
    uint64_t *support = support_record(layout, support_slot);
    uint64_t *version = version_record(layout, version_slot);
    uint64_t *records[3] = {statement, support, version};
    for (uint32_t index = 0; index < 3; ++index) {
        uint64_t *pair = records[index];
        if (pair[1] == UINT64_MAX) {
            const uint64_t kind = pair == statement ? kStatementKind
                                  : pair == support ? kSupportKind
                                                    : kVersionKind;
            const uint64_t slot = pair == statement ? statement_slot
                                  : pair == support ? support_slot
                                                    : version_slot;
            fail(receipt, kGenerationExhausted, kind, slot, pair[1], pair[1], 0, 0, layout.work);
            return;
        }
    }
    if (layout.candidate[4] == UINT32_MAX || layout.candidate[5] == UINT32_MAX ||
        layout.candidate[6] == UINT32_MAX) {
        fail(receipt, kCorruptLineage, kForkKind, 0, layout.candidate[1],
             layout.candidate[1], 1, 8, layout.work);
        return;
    }
    uint64_t previous_identity[4] = {0, 0, 0, 0};
    uint64_t previous_truth = 0;
    if (encoded_head != 0) {
        uint64_t *previous = version_record(layout, encoded_head - 1);
        copy_identity(previous_identity, previous + 9, layout.work);
        previous_truth = previous[8];
    }
    const uint64_t polarity = command->words[12];
    if (polarity != 1 && polarity != 2) {
        fail(receipt, kInvalidCommand, kSupportKind, support_slot, polarity, 0, 0, 9, layout.work);
        return;
    }
    const uint64_t truth = previous_truth | (polarity == 1 ? 1ULL : 2ULL);
    uint64_t new_version_identity[4];
    version_digest(previous_identity, support_identity, truth, new_version_identity, layout.work);
    const uint32_t statement_extent =
        static_cast<uint32_t>(layout.candidate[4] + (new_statement ? 1 : 0));
    const uint32_t support_extent = static_cast<uint32_t>(layout.candidate[5] + 1);
    const uint32_t version_extent = static_cast<uint32_t>(layout.candidate[6] + 1);
    uint64_t new_root_digest[4];
    root_digest(layout.candidate + 7, new_version_identity, statement_extent,
                support_extent, version_extent, new_root_digest, layout.work);

    if (new_statement) {
        statement[2] = written_word(layout.candidate[1], layout.work);
        copy_identity(statement + 3, statement_identity, layout.work);
        statement[0] = written_word(kStaged, layout.work);
    }
    support[2] = written_word(layout.candidate[1], layout.work);
    support[3] = written_word(statement_slot, layout.work);
    support[4] = written_word(statement[1], layout.work);
    support[5] = written_word(polarity, layout.work);
    copy_identity(support + 6, support_identity, layout.work);
    // Original typed occurrences belong to the successful attachment. An equal
    // later insertion returns above without replacing these provenance tags.
    support[10] = written_word(command->words[13], layout.work);
    support[11] = written_word(command->words[14], layout.work);
    for (uint32_t word = 0; word < 5; ++word) support[12 + word] = written_word(command->words[24 + word], layout.work);
    support[0] = written_word(kStaged, layout.work);
    version[2] = written_word(layout.candidate[1], layout.work);
    version[3] = written_word(statement_slot, layout.work);
    version[4] = written_word(statement[1], layout.work);
    version[5] = written_word(support_slot, layout.work);
    version[6] = written_word(support[1], layout.work);
    version[7] = written_word(encoded_head, layout.work);
    version[8] = written_word(truth, layout.work);
    copy_identity(version + 9, new_version_identity, layout.work);
    version[13] = written_word(version_extent, layout.work);
    version[0] = written_word(kStaged, layout.work);
    layout.candidate[4] = written_word(statement_extent, layout.work);
    layout.candidate[5] = written_word(support_extent, layout.work);
    layout.candidate[6] = written_word(version_extent, layout.work);
    copy_identity(layout.candidate + 7, new_root_digest, layout.work);
    layout.candidate_heads[statement_slot] = written_word(version_slot + 1, layout.work);

    receipt->words[1] = written_word(kOutcomeInserted, layout.work);
    receipt->words[32] = written_word((new_statement ? kInsertNewStatement : 0) |
                         (previous_truth << kInsertPreviousTruthShift), layout.work);
    fill_statement(receipt, layout, statement_slot);
    fill_support(receipt, layout, support_slot);
    fill_version(receipt, layout, version_slot);
}

__device__ bool preflight_discard(const Layout &layout, Receipt *receipt) {
    if (!preflight_candidate_retirement(layout, receipt)) {
        return false;
    }
    const uint64_t owner = layout.candidate[1];
    const struct {
        uint64_t *records;
        uint64_t capacity;
        uint64_t width;
        uint64_t kind;
    } groups[] = {
        {layout.statements, layout.statement_capacity, kStatementWords, kStatementKind},
        {layout.supports, layout.support_capacity, kSupportWords, kSupportKind},
        {layout.versions, layout.version_capacity, kVersionWords, kVersionKind},
    };
    for (const auto &group : groups) {
        for (uint64_t slot = 0; slot < group.capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            uint64_t *record = group.records + slot * group.width;
            if (record[0] == kStaged && record[2] == owner && record[1] == UINT64_MAX) {
                fail(receipt, kGenerationExhausted, group.kind, slot, record[1], record[1],
                     group.capacity, 0, layout.work);
                return false;
            }
        }
    }
    return true;
}

__device__ void discard(Layout &layout, const Command *command, Receipt *receipt) {
    if (!validate_candidate(layout, command->words[8], command->words[9], receipt) ||
        !preflight_discard(layout, receipt)) {
        return;
    }
    const uint64_t owner = layout.candidate[1];
    const struct {
        uint64_t *records;
        uint64_t capacity;
        uint64_t width;
    } groups[] = {
        {layout.statements, layout.statement_capacity, kStatementWords},
        {layout.supports, layout.support_capacity, kSupportWords},
        {layout.versions, layout.version_capacity, kVersionWords},
    };
    for (const auto &group : groups) {
        for (uint64_t slot = 0; slot < group.capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            uint64_t *record = group.records + slot * group.width;
            if (record[0] == kStaged && record[2] == owner) {
                record[0] = written_word(kFree, layout.work);
                record[1] += written_word(1, layout.work);
                record[2] = written_word(0, layout.work);
            }
        }
    }
    for (uint64_t slot = 0; slot < layout.statement_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        layout.candidate_heads[slot] = written_word(0, layout.work);
    }
    layout.candidate[0] = written_word(kFree, layout.work);
    layout.candidate[1] += written_word(1, layout.work);
    receipt->words[5] = written_word(0, layout.work);
    receipt->words[6] = written_word(layout.candidate[1], layout.work);
}

__device__ void seal(Layout &layout, const Command *command, Receipt *receipt) {
    if (!validate_candidate(layout, command->words[8], command->words[9], receipt) ||
        !preflight_candidate_retirement(layout, receipt)) {
        return;
    }
    const uint64_t base_slot = layout.candidate[2];
    const uint64_t base_generation = layout.candidate[3];
    if (!validate_root(layout, base_slot, base_generation, receipt)) {
        return;
    }
    const uint64_t *base = root_record(layout, base_slot);
    // Inserts are monotone: a later duplicate cannot undo an earlier insertion.
    const bool unchanged = layout.candidate[4] == base[3] &&
                           layout.candidate[5] == base[4] &&
                           layout.candidate[6] == base[5];
    if (unchanged) {
        for (uint32_t word = 0; word < 4; ++word) {
            if (layout.candidate[7 + word] != base[6 + word]) {
                fail(receipt, kCorruptLineage, kRootKind, base_slot,
                     base_generation, base_generation, 0, 0, layout.work);
                return;
            }
        }
        discard(layout, command, receipt);
        if (receipt->words[0] != kOk) {
            return;
        }
        receipt->words[1] = written_word(kOutcomeUnchanged, layout.work);
        receipt->words[3] = written_word(base_slot, layout.work);
        receipt->words[4] = written_word(base_generation, layout.work);
        fill_extents_and_digest(receipt, base, layout.work);
        return;
    }
    const uint64_t root_slot =
        find_free(layout, layout.roots, layout.root_capacity, kRootWords);
    if (root_slot == kNull) {
        fail(receipt, kRootCapacity, kRootKind, 0, 0, 0, layout.root_capacity, 0, layout.work);
        return;
    }
    const uint64_t owner = layout.candidate[1];
    const struct {
        uint64_t *records;
        uint64_t capacity;
        uint64_t width;
    } groups[] = {
        {layout.statements, layout.statement_capacity, kStatementWords},
        {layout.supports, layout.support_capacity, kSupportWords},
        {layout.versions, layout.version_capacity, kVersionWords},
    };
    for (const auto &group : groups) {
        for (uint64_t slot = 0; slot < group.capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            uint64_t *record = group.records + slot * group.width;
            if (record[0] == kStaged && record[2] == owner) {
                record[0] = written_word(kSealed, layout.work);
                record[2] = written_word(0, layout.work);
            }
        }
    }
    uint64_t *root = root_record(layout, root_slot);
    root[2] = written_word(layout.candidate[2], layout.work);
    root[3] = written_word(layout.candidate[4], layout.work);
    root[4] = written_word(layout.candidate[5], layout.work);
    root[5] = written_word(layout.candidate[6], layout.work);
    copy_identity(root + 6, layout.candidate + 7, layout.work);
    for (uint64_t statement = 0; statement < layout.statement_capacity; ++statement) {
        layout.root_heads[root_slot * layout.statement_capacity + statement] = written_word(layout.candidate_heads[statement], layout.work);
        layout.candidate_heads[statement] = written_word(0, layout.work);
    }
    root[0] = written_word(kSealed, layout.work);
    layout.candidate[0] = written_word(kFree, layout.work);
    layout.candidate[1] += written_word(1, layout.work);
    receipt->words[3] = written_word(root_slot, layout.work);
    receipt->words[4] = written_word(root[1], layout.work);
    receipt->words[5] = written_word(0, layout.work);
    receipt->words[6] = written_word(layout.candidate[1], layout.work);
    fill_extents_and_digest(receipt, root, layout.work);
    receipt->words[1] = written_word(kOutcomeInserted, layout.work);
}

// Sealed records have no staging owner. Under the enclosing Session's exclusive
// arena ownership, word 2 is temporary reachability scratch and is restored to
// zero before this command returns, including every refusal path.
__device__ bool trace_retirement_heads(const Layout &layout, const uint64_t *heads,
                                       bool candidate, bool mark, Receipt *receipt) {
    for (uint64_t slot = 0; slot < layout.statement_capacity; ++slot) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
        uint64_t encoded = heads[slot];
        if (encoded == 0) {
            continue;
        }
        uint64_t *statement = statement_record(layout, slot);
        if ((!candidate && statement[0] != kSealed) ||
            (candidate && !record_is_live(statement[0])) ||
            (statement[0] == kStaged && statement[2] != layout.candidate[1])) {
            fail(receipt, kCorruptLineage, kStatementKind, slot, statement[1],
                 statement[1], layout.statement_capacity, 11, layout.work);
            return false;
        }
        if (mark && statement[0] == kSealed) {
            statement[2] = written_word(1, layout.work);
        }
        for (uint64_t steps = 0; encoded != 0 && steps < layout.version_capacity; ++steps) {
        charge_native(layout.work, NativeWorkEvent::ChainLink, 1);
            const uint64_t version_slot = encoded - 1;
            if (!validate_version_link(layout, slot, version_slot, receipt)) {
                return false;
            }
            uint64_t *version = version_record(layout, version_slot);
            uint64_t *support = support_record(layout, version[5]);
            if ((!candidate && (version[0] != kSealed || support[0] != kSealed)) ||
                (version[0] == kStaged && version[2] != layout.candidate[1]) ||
                (support[0] == kStaged && support[2] != layout.candidate[1])) {
                fail(receipt, kCorruptLineage, kVersionKind, version_slot, version[1],
                     version[1], layout.version_capacity, 12, layout.work);
                return false;
            }
            if (mark) {
                if (version[0] == kSealed) version[2] = written_word(1, layout.work);
                if (support[0] == kSealed) support[2] = written_word(1, layout.work);
            }
            encoded = version[7];
        }
        if (encoded != 0) {
            fail(receipt, kCorruptLineage, kVersionKind, encoded - 1, 0, 0,
                 layout.version_capacity, 13, layout.work);
            return false;
        }
    }
    return true;
}

__device__ bool trace_retirement_roots(const Layout &layout, uint64_t excluded,
                                       bool mark, Receipt *receipt) {
    for (uint64_t slot = 0; slot < layout.root_capacity; ++slot) {
        const uint64_t *root = root_record(layout, slot);
        if (root[0] == kFree || slot == excluded) {
            continue;
        }
        if (!validate_root(layout, slot, root[1], receipt) ||
            !trace_retirement_heads(layout, layout.root_heads + slot * layout.statement_capacity,
                                    false, mark, receipt)) {
            return false;
        }
    }
    if (layout.candidate[0] == kFree) {
        return true;
    }
    return validate_candidate(layout, 0, layout.candidate[1], receipt) &&
           validate_root(layout, layout.candidate[2], layout.candidate[3], receipt) &&
           trace_retirement_heads(layout, layout.candidate_heads, true, mark, receipt);
}

__device__ void retire_root(Layout &layout, const Command *command, Receipt *receipt) {
    const uint64_t slot = command->words[8];
    const uint64_t generation = command->words[9];
    const uint64_t base_slot = command->words[10];
    if (!validate_root(layout, slot, generation, receipt) ||
        !validate_root(layout, base_slot, command->words[11], receipt)) {
        return;
    }
    if (slot == 0 || slot == base_slot ||
        (layout.candidate[0] == kLive && layout.candidate[2] == slot &&
         layout.candidate[3] == generation)) {
        fail(receipt, kInactive, kRootKind, slot, generation, generation,
             layout.root_capacity, 0, layout.work);
        return;
    }
    if (generation == UINT64_MAX) {
        fail(receipt, kGenerationExhausted, kRootKind, slot, generation, generation,
             layout.root_capacity, 1, layout.work);
        return;
    }
    // Validate every current chain before changing even temporary owner words.
    if (!trace_retirement_roots(layout, kNull, false, receipt)) {
        return;
    }
    const struct {
        uint64_t *records;
        uint64_t capacity, width, kind;
    } groups[] = {
        {layout.statements, layout.statement_capacity, kStatementWords, kStatementKind},
        {layout.supports, layout.support_capacity, kSupportWords, kSupportKind},
        {layout.versions, layout.version_capacity, kVersionWords, kVersionKind},
    };
    for (const auto &group : groups) {
        for (uint64_t index = 0; index < group.capacity; ++index) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            const uint64_t *record = group.records + index * group.width;
            if (record[0] == kSealed && record[2] != 0) {
                fail(receipt, kCorruptLineage, group.kind, index, record[1], record[1],
                     group.capacity, 14, layout.work);
                return;
            }
        }
    }
    bool ready = trace_retirement_roots(layout, slot, true, receipt);
    for (const auto &group : groups) {
        for (uint64_t index = 0; ready && index < group.capacity; ++index) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            const uint64_t *record = group.records + index * group.width;
            if (record[0] == kSealed && record[2] == 0 && record[1] == UINT64_MAX) {
                fail(receipt, kGenerationExhausted, group.kind, index, record[1], record[1],
                     group.capacity, 1, layout.work);
                ready = false;
            }
        }
    }
    // All fallible checks precede reclamation. A refusal only clears the marks;
    // root heads, states, generations and all original owner words remain exact.
    for (const auto &group : groups) {
        for (uint64_t index = 0; index < group.capacity; ++index) {
        charge_native(layout.work, NativeWorkEvent::TableSlot, 1);
            uint64_t *record = group.records + index * group.width;
            if (record[0] != kSealed) {
                continue;
            }
            if (ready && record[2] == 0) {
                record[0] = written_word(kFree, layout.work);
                record[1] += written_word(1, layout.work);
            }
            record[2] = written_word(0, layout.work);
        }
    }
    if (!ready) {
        return;
    }
    for (uint64_t statement = 0; statement < layout.statement_capacity; ++statement) {
        layout.root_heads[slot * layout.statement_capacity + statement] = written_word(0, layout.work);
    }
    uint64_t *root = root_record(layout, slot);
    root[0] = written_word(kFree, layout.work);
    root[1] += written_word(1, layout.work);
    receipt->words[3] = written_word(slot, layout.work);
    receipt->words[4] = written_word(root[1], layout.work);
}

__device__ void snapshot(const Layout &layout, const Command *command, Receipt *receipt) {
    uint64_t *record = nullptr;
    uint64_t *heads = nullptr;
    if (!validate_view(layout, command, receipt, &record, &heads)) {
        return;
    }
    (void)heads;
    fill_view(receipt, command, record, layout.work);
}

__device__ void truth(const Layout &layout, const Command *command, Receipt *receipt) {
    uint64_t *record = nullptr;
    uint64_t *heads = nullptr;
    if (!validate_view(layout, command, receipt, &record, &heads)) {
        return;
    }
    fill_view(receipt, command, record, layout.work);
    // An absent key still has a validated view and the actual requested
    // statement identity. Its cleared handle generations remain explicitly zero.
    copy_identity(receipt->words + 20, command->words + 16, layout.work);
    const uint64_t statement = find_statement(layout, heads, command->words + 16, receipt);
    if (receipt->words[0] != kOk || statement == kNull) {
        return;
    }
    const uint64_t encoded = heads[statement];
    if (!validate_head(layout, statement, encoded, receipt)) {
        return;
    }
    fill_statement(receipt, layout, statement);
    fill_version(receipt, layout, encoded - 1);
}

__device__ bool target_statement(const Layout &layout, uint64_t *heads, uint64_t slot,
                                 uint64_t generation, Receipt *receipt) {
    if (slot >= layout.statement_capacity) {
        fail(receipt, kSlotOutOfRange, kStatementKind, slot, generation, 0,
             layout.statement_capacity, 0, layout.work);
        return false;
    }
    uint64_t *statement = statement_record(layout, slot);
    if (statement[1] != generation) {
        fail(receipt, kStaleGeneration, kStatementKind, slot, generation, statement[1],
             layout.statement_capacity, 0, layout.work);
        return false;
    }
    if (!record_is_live(statement[0])) {
        fail(receipt, kInactive, kStatementKind, slot, generation, statement[1],
             layout.statement_capacity, statement[0], layout.work);
        return false;
    }
    if (heads[slot] == 0 || !validate_head(layout, slot, heads[slot], receipt)) {
        if (receipt->words[0] == kOk) {
            fail(receipt, kNotReachable, kStatementKind, slot, generation, statement[1],
                 layout.statement_capacity, 0, layout.work);
        }
        return false;
    }
    return true;
}

__device__ bool chain_contains(const Layout &layout, uint64_t *heads, uint64_t statement_slot,
                               uint64_t target_slot, uint64_t target_generation,
                               bool target_is_support, Receipt *receipt) {
    uint64_t encoded = heads[statement_slot];
    for (uint64_t steps = 0; encoded != 0 && steps < layout.version_capacity; ++steps) {
        charge_native(layout.work, NativeWorkEvent::ChainLink, 1);
        const uint64_t slot = encoded - 1;
        if (!validate_version_link(layout, statement_slot, slot, receipt)) {
            return false;
        }
        uint64_t *version = version_record(layout, slot);
        if ((!target_is_support && slot == target_slot && version[1] == target_generation) ||
            (target_is_support && version[5] == target_slot && version[6] == target_generation)) {
            return true;
        }
        encoded = version[7];
    }
    if (encoded != 0) {
        fail(receipt, kCorruptLineage, kVersionKind, encoded - 1, 0, 0,
             layout.version_capacity, 10, layout.work);
    }
    return false;
}

__device__ void inspect_statement(const Layout &layout, const Command *command,
                                  Receipt *receipt) {
    uint64_t *record = nullptr;
    uint64_t *heads = nullptr;
    if (!validate_view(layout, command, receipt, &record, &heads)) {
        return;
    }
    (void)record;
    const uint64_t slot = command->words[10];
    if (target_statement(layout, heads, slot, command->words[11], receipt)) {
        fill_statement(receipt, layout, slot);
    }
}

__device__ void inspect_support(const Layout &layout, const Command *command,
                                Receipt *receipt) {
    uint64_t *record = nullptr;
    uint64_t *heads = nullptr;
    if (!validate_view(layout, command, receipt, &record, &heads)) {
        return;
    }
    (void)record;
    const uint64_t slot = command->words[10];
    const uint64_t generation = command->words[11];
    if (slot >= layout.support_capacity) {
        fail(receipt, kSlotOutOfRange, kSupportKind, slot, generation, 0,
             layout.support_capacity, 0, layout.work);
        return;
    }
    uint64_t *support = support_record(layout, slot);
    if (support[1] != generation) {
        fail(receipt, kStaleGeneration, kSupportKind, slot, generation, support[1],
             layout.support_capacity, 0, layout.work);
        return;
    }
    if (!record_is_live(support[0])) {
        fail(receipt, kInactive, kSupportKind, slot, generation, support[1],
             layout.support_capacity, support[0], layout.work);
        return;
    }
    const uint64_t statement_slot = support[3];
    if (statement_slot >= layout.statement_capacity || heads[statement_slot] == 0 ||
        !chain_contains(layout, heads, statement_slot, slot, generation, true, receipt)) {
        if (receipt->words[0] == kOk) {
            fail(receipt, kNotReachable, kSupportKind, slot, generation, support[1],
                 layout.support_capacity, 0, layout.work);
        }
        return;
    }
    fill_support(receipt, layout, slot);
}

__device__ void inspect_version(const Layout &layout, const Command *command,
                                Receipt *receipt) {
    uint64_t *record = nullptr;
    uint64_t *heads = nullptr;
    if (!validate_view(layout, command, receipt, &record, &heads)) {
        return;
    }
    (void)record;
    const uint64_t slot = command->words[10];
    const uint64_t generation = command->words[11];
    if (slot >= layout.version_capacity) {
        fail(receipt, kSlotOutOfRange, kVersionKind, slot, generation, 0,
             layout.version_capacity, 0, layout.work);
        return;
    }
    uint64_t *version = version_record(layout, slot);
    if (version[1] != generation) {
        fail(receipt, kStaleGeneration, kVersionKind, slot, generation, version[1],
             layout.version_capacity, 0, layout.work);
        return;
    }
    if (!record_is_live(version[0])) {
        fail(receipt, kInactive, kVersionKind, slot, generation, version[1],
             layout.version_capacity, version[0], layout.work);
        return;
    }
    const uint64_t statement_slot = version[3];
    if (statement_slot >= layout.statement_capacity || heads[statement_slot] == 0 ||
        !chain_contains(layout, heads, statement_slot, slot, generation, false, receipt)) {
        if (receipt->words[0] == kOk) {
            fail(receipt, kNotReachable, kVersionKind, slot, generation, version[1],
                 layout.version_capacity, 0, layout.work);
        }
        return;
    }
    fill_version(receipt, layout, slot);
}

__device__ __forceinline__ void execute(uint64_t *arena, const LaunchDescriptor &descriptor,
                                        NativeWorkTally *work = nullptr) {
    charge_native(work, NativeWorkEvent::Command, 1);
    if (descriptor.abi_generation != kAbiGeneration) {
        if (descriptor.receipt_ptr != 0) {
            Receipt *receipt = reinterpret_cast<Receipt *>(descriptor.receipt_ptr) +
                               descriptor.receipt_index;
            clear_receipt(receipt, work);
            receipt->words[39] = written_word(descriptor.expected_owner, work);
            fail(receipt, kArenaMismatch, kRootKind, 0, descriptor.abi_generation,
                 kAbiGeneration, 0, 0, work);
        }
        return;
    }
    ResidentHandle *empty_root_output = nullptr;
    if (descriptor.admission == kResidentEmptyRootHandleAdmission &&
        descriptor.handle_ptr != 0) {
        empty_root_output =
            reinterpret_cast<ResidentHandle *>(descriptor.handle_ptr) + descriptor.handle_index;
        empty_root_output->owner = descriptor.expected_owner;
        empty_root_output->kind = 0;
        empty_root_output->slot = kNull;
        empty_root_output->generation = 0;
    }
    TruthValue *truth_output = nullptr;
    if (descriptor.admission == kResidentConsumeTruthAdmission &&
        descriptor.output_ptr != 0) {
        truth_output =
            reinterpret_cast<TruthValue *>(descriptor.output_ptr) + descriptor.output_index;
        truth_output->status = kInvalidCommand;
        truth_output->truth = 0;
        truth_output->owner = descriptor.expected_owner;
        truth_output->reserved = 0;
    }
    Receipt source_snapshot;
    const Receipt *source_receipt = nullptr;
    if (descriptor.source_receipt_ptr != 0) {
        const Receipt *source =
            reinterpret_cast<const Receipt *>(descriptor.source_receipt_ptr) +
            descriptor.source_receipt_index;
        copy_receipt(&source_snapshot, source, work);
        source_receipt = &source_snapshot;
    }
    if (descriptor.receipt_ptr == 0) {
        return;
    }
    Receipt *receipt = reinterpret_cast<Receipt *>(descriptor.receipt_ptr) +
                       descriptor.receipt_index;
    clear_receipt(receipt, work);
    receipt->words[39] = written_word(descriptor.expected_owner, work);
    Layout layout{};
    layout.work = work;
    if (!build_layout(arena, descriptor.root_capacity, descriptor.statement_capacity,
                      descriptor.support_capacity, descriptor.version_capacity,
                      descriptor.arena_words, &layout)) {
        fail(receipt, kArenaMismatch, kRootKind, 0, descriptor.arena_words, 0, 0, 0, work);
        return;
    }
    if (descriptor.admission > kResidentDiscardAdmission) {
        fail(receipt, kInvalidCommand, kRootKind, 0, descriptor.admission, 0, 0, 0, work);
        return;
    }

    Command resident_command = {};
    resident_command.words[1] = written_word(descriptor.expected_owner, work);
    const Command *command = &resident_command;
    bool emit_empty_root_handle = false;
    bool consume_truth = false;
    bool materialize_decoded = false;
    bool retain_refusal_after_discard = false;
    uint64_t handle_slot = 0;
    uint64_t handle_generation = 0;

    switch (descriptor.admission) {
    case kHostCommandAdmission:
        if (descriptor.command_ptr == 0) {
            fail(receipt, kInvalidCommand, kRootKind, 0, 0, 0, 0, 3, work);
            return;
        }
        command = reinterpret_cast<const Command *>(descriptor.command_ptr) +
                  descriptor.command_index;
        break;
    case kResidentEmptyRootHandleAdmission:
        if (empty_root_output == nullptr || source_receipt != nullptr) {
            fail(receipt, kInvalidCommand, kRootKind, 0, 0, 0, 0, 4, work);
            return;
        }
        emit_empty_root_handle = true;
        break;
    case kResidentForkAdmission:
    case kResidentPreflightTransitionAdmission:
        resident_command.words[0] = written_word(descriptor.admission == kResidentForkAdmission
                                       ? kFork : kPreflightTransition, work);
        if (!load_resident_handle(descriptor, source_receipt, kRootKind, receipt,
                                  &handle_slot, &handle_generation, work)) {
            // Propagating a refused base is not a successful acquisition.
            receipt->words[kAcquiredCandidateSlot] = written_word(0, work);
            receipt->words[kAcquiredCandidateGeneration] = written_word(0, work);
            return;
        }
        resident_command.words[8] = written_word(handle_slot, work);
        resident_command.words[9] = written_word(handle_generation, work);
        break;
    case kResidentInsertSupportAdmission: {
        resident_command.words[0] = written_word(kInsertSupport, work);
        if (!load_resident_handle(descriptor, source_receipt, kForkKind, receipt,
                                  &handle_slot, &handle_generation, work) ||
            !inherit_candidate_acquisition(source_receipt, handle_slot,
                                           handle_generation, receipt, work)) {
            return;
        }
        if (descriptor.decoded_statement_ptr == 0 || descriptor.decoded_support_ptr == 0) {
            if (receipt->words[0] == kOk) {
                fail(receipt, kInvalidCommand, kSupportKind, 0, 0, 0, 0, 5, work);
            }
            return;
        }
        resident_command.words[8] = written_word(handle_slot, work);
        resident_command.words[9] = written_word(handle_generation, work);
        const DecodedStatement *statement =
            reinterpret_cast<const DecodedStatement *>(descriptor.decoded_statement_ptr) +
            descriptor.decoded_statement_index;
        const DecodedSupport *support =
            reinterpret_cast<const DecodedSupport *>(descriptor.decoded_support_ptr) +
            descriptor.decoded_support_index;
        resident_command.words[12] = written_word(support->polarity, work);
        resident_command.words[13] = written_word(statement->record, work);
        resident_command.words[14] = written_word(support->record, work);
        for (uint32_t word = 0; word < 5; ++word) {
            resident_command.words[24 + word] = written_word(uint64_t(statement->reconstruction[2 * word]) |
                (uint64_t(statement->reconstruction[2 * word + 1]) << 32), work);
        }
        decode_identity(statement->identity_words, resident_command.words + 16, work);
        support_digest(resident_command.words + 16, *support,
                       resident_command.words + 20, work);
        break;
    }
    case kResidentForkTruthAdmission:
    case kResidentRootTruthAdmission: {
        resident_command.words[0] = written_word(kTruth, work);
        const bool root_view = descriptor.admission == kResidentRootTruthAdmission;
        const uint64_t kind = root_view ? kRootKind : kForkKind;
        resident_command.words[7] = written_word(root_view ? 1 : 2, work);
        if (!load_resident_handle(descriptor, source_receipt, kind, receipt,
                                  &handle_slot, &handle_generation, work) ||
            (!root_view && !inherit_candidate_acquisition(
                               source_receipt, handle_slot, handle_generation, receipt, work)) ||
            descriptor.decoded_statement_ptr == 0) {
            if (receipt->words[0] == kOk) {
                fail(receipt, kInvalidCommand, kStatementKind, 0, 0, 0, 0, 6, work);
            }
            return;
        }
        resident_command.words[8] = written_word(handle_slot, work);
        resident_command.words[9] = written_word(handle_generation, work);
        const DecodedStatement *statement =
            reinterpret_cast<const DecodedStatement *>(descriptor.decoded_statement_ptr) +
            descriptor.decoded_statement_index;
        decode_identity(statement->identity_words, resident_command.words + 16, work);
        break;
    }
    case kResidentDiscardAdmission:
    case kResidentSealAdmission:
        if (source_receipt != nullptr && source_receipt->words[0] != kOk) {
            if (source_receipt->words[39] != descriptor.expected_owner) {
                fail(receipt, kForeignOwner, kForkKind, 0, source_receipt->words[39],
                     descriptor.expected_owner, 0, 0, work);
                return;
            }
            if (source_receipt->words[kAcquiredCandidateGeneration] == 0) {
                // The fork itself failed: no authority to retire the live candidate.
                copy_receipt(receipt, source_receipt, work);
                return;
            }
            resident_command.words[0] = written_word(kDiscard, work);
            resident_command.words[8] = written_word(source_receipt->words[kAcquiredCandidateSlot], work);
            resident_command.words[9] = written_word(source_receipt->words[kAcquiredCandidateGeneration], work);
            // Explicit discard reports cleanup independently; sealing a refused
            // lane still returns that original refusal after successful cleanup.
            retain_refusal_after_discard = descriptor.admission == kResidentSealAdmission;
            break;
        }
        resident_command.words[0] = written_word(descriptor.admission == kResidentDiscardAdmission
                                       ? kDiscard : kSeal, work);
        if (!load_resident_handle(descriptor, source_receipt, kForkKind, receipt,
                                  &handle_slot, &handle_generation, work) ||
            !inherit_candidate_acquisition(source_receipt, handle_slot,
                                           handle_generation, receipt, work)) {
            return;
        }
        resident_command.words[8] = written_word(handle_slot, work);
        resident_command.words[9] = written_word(handle_generation, work);
        break;
    case kResidentConsumeTruthAdmission:
        consume_truth = true;
        break;
    case kResidentMaterializeDecodedAdmission:
        if (descriptor.decoded_input_ptr == 0 ||
            descriptor.decoded_statement_ptr == 0 ||
            descriptor.decoded_support_ptr == 0) {
            fail(receipt, kInvalidCommand, kStatementKind, 0, 0, 0, 0, 10, work);
            return;
        }
        materialize_decoded = true;
        break;
    default:
        fail(receipt, kInvalidCommand, kRootKind, 0, descriptor.admission, 0, 0, 7, work);
        return;
    }

    if (command->words[0] == kInitialize) {
        initialize(layout, descriptor.expected_owner, receipt);
        return;
    }
    if (!validate_control(layout, descriptor.expected_owner, command, receipt)) {
        return;
    }
    if (emit_empty_root_handle) {
        uint64_t *root = root_record(layout, 0);
        if (!validate_root(layout, 0, root[1], receipt)) {
            return;
        }
        empty_root_output->owner = descriptor.expected_owner;
        empty_root_output->kind = kRootKind;
        empty_root_output->slot = 0;
        empty_root_output->generation = root[1];
        receipt->words[33] = written_word(kRootKind, work);
        receipt->words[34] = written_word(0, work);
        receipt->words[35] = written_word(root[1], work);
        return;
    }
    if (materialize_decoded) {
        const DecodedInput *input =
            reinterpret_cast<const DecodedInput *>(descriptor.decoded_input_ptr) +
            descriptor.decoded_input_index;
        DecodedStatement *statement =
            reinterpret_cast<DecodedStatement *>(descriptor.decoded_statement_ptr) +
            descriptor.decoded_statement_index;
        DecodedSupport *support =
            reinterpret_cast<DecodedSupport *>(descriptor.decoded_support_ptr) +
            descriptor.decoded_support_index;
        for (uint32_t index = 0; index < 8; ++index) {
            statement->identity_words[index] = input->statement_identity_words[index];
            support->provenance_words[index] = input->provenance_words[index];
            support->source_words[index] = input->source_words[index];
            support->context_words[index] = input->context_words[index];
            support->scope_words[index] = input->scope_words[index];
        }
        support->polarity = input->polarity;
        statement->record = input->statement_record;
        support->record = input->support_record;
        for (uint32_t word = 0; word < 10; ++word) statement->reconstruction[word] = input->reconstruction[word];
        return;
    }
    if (consume_truth) {
        if (source_receipt == nullptr || truth_output == nullptr) {
            fail(receipt, kInvalidCommand, kStatementKind, 0, 0, 0, 0, 8, work);
            return;
        }
        if (source_receipt->words[39] != descriptor.expected_owner) {
            truth_output->status = kForeignOwner;
            fail(receipt, kForeignOwner, kStatementKind, 0, source_receipt->words[39],
                 descriptor.expected_owner, 0, 0, work);
            return;
        }
        if (source_receipt->words[0] != kOk) {
            truth_output->status = source_receipt->words[0];
            copy_receipt(receipt, source_receipt, work);
            return;
        }
        if (source_receipt->words[2] > 3) {
            fail(receipt, kCorruptLineage, kStatementKind, 0,
                 source_receipt->words[2], 0, 0, 9, work);
            truth_output->status = kCorruptLineage;
            return;
        }
        truth_output->status = kOk;
        truth_output->truth = source_receipt->words[2];
        return;
    }
    switch (command->words[0]) {
    case kPreflightTransition:
        preflight_transition(layout, command, receipt);
        break;
    case kFork:
        fork_root(layout, command, receipt);
        break;
    case kInsertSupport:
        insert_support(layout, command, receipt);
        break;
    case kDiscard:
        discard(layout, command, receipt);
        break;
    case kSeal:
        seal(layout, command, receipt);
        break;
    case kRetireRoot:
        retire_root(layout, command, receipt);
        break;
    case kSnapshot:
        snapshot(layout, command, receipt);
        break;
    case kTruth:
        truth(layout, command, receipt);
        break;
    case kInspectStatement:
        inspect_statement(layout, command, receipt);
        break;
    case kInspectSupport:
        inspect_support(layout, command, receipt);
        break;
    case kInspectVersion:
        inspect_version(layout, command, receipt);
        break;
    default:
        fail(receipt, kInvalidCommand, kRootKind, 0, command->words[0], 0, 0, 0, work);
        break;
    }
    if (descriptor.admission == kResidentSealAdmission &&
        !retain_refusal_after_discard && receipt->words[0] != kOk) {
        // A terminal refusal must release its scratch just like an edit refusal.
        Receipt refusal;
        copy_receipt(&refusal, receipt, work);
        Command cleanup = {};
        cleanup.words[0] = written_word(kDiscard, work);
        cleanup.words[1] = written_word(descriptor.expected_owner, work);
        cleanup.words[8] = written_word(handle_slot, work);
        cleanup.words[9] = written_word(handle_generation, work);
        clear_receipt(receipt, work);
        receipt->words[39] = written_word(descriptor.expected_owner, work);
        charge_native(work, NativeWorkEvent::Command, 1);
        discard(layout, &cleanup, receipt);
        if (receipt->words[0] == kOk) {
            copy_receipt(receipt, &refusal, work);
        }
        return;
    }
    if (retain_refusal_after_discard) {
        if (receipt->words[0] == kOk) {
            // Cleanup succeeded; retain the original lane refusal, not a fake seal.
            copy_receipt(receipt, source_receipt, work);
        }
        return;
    }
    if (receipt->words[0] == kOk) {
        if (descriptor.admission == kResidentForkAdmission) {
            receipt->words[33] = written_word(kForkKind, work);
            receipt->words[34] = written_word(receipt->words[5], work);
            receipt->words[35] = written_word(receipt->words[6], work);
            receipt->words[kAcquiredCandidateSlot] = written_word(receipt->words[5], work);
            receipt->words[kAcquiredCandidateGeneration] = written_word(receipt->words[6], work);
        } else if (descriptor.admission == kResidentInsertSupportAdmission) {
            receipt->words[33] = written_word(kForkKind, work);
            receipt->words[34] = written_word(handle_slot, work);
            receipt->words[35] = written_word(handle_generation, work);
        } else if (descriptor.admission == kResidentDiscardAdmission) {
            receipt->words[kAcquiredCandidateSlot] = written_word(0, work);
            receipt->words[kAcquiredCandidateGeneration] = written_word(0, work);
        } else if (descriptor.admission == kResidentSealAdmission ||
                   descriptor.admission == kResidentPreflightTransitionAdmission) {
            receipt->words[33] = written_word(kRootKind, work);
            receipt->words[34] = written_word(receipt->words[3], work);
            receipt->words[35] = written_word(receipt->words[4], work);
        }
    }
}

} // namespace semantic_graph

#ifndef XLOG_SEMANTIC_GRAPH_DEVICE_ONLY
extern "C" __global__ void semantic_hypergraph_execute(uint64_t *arena,
                                    semantic_graph::LaunchDescriptor descriptor) {
    if (blockIdx.x == 0 && threadIdx.x == 0) semantic_graph::execute(arena, descriptor);
}
#endif
