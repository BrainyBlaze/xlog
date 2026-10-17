#ifndef XLOG_RESOURCE_OBSERVER_H
#define XLOG_RESOURCE_OBSERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XLOG_RESOURCE_OBSERVER_ABI_VERSION UINT32_C(6)
#define XLOG_RESOURCE_OBSERVER_CAPSULE_NAME "xlog.resource_observer.v6"
#define XLOG_RESOURCE_MEMORY_UNIT "owned-gpu-backing-peak-bytes"
#define XLOG_RESOURCE_MEMORY_ADMISSION "whole-device-total-physical-bytes"

/* Completion and coverage are independent. COMPLETE means the requested
 * supported trace was delivered; it does not certify owned backing peaks,
 * asynchronous allocator completion, or access through host mappings. */
#define XLOG_RESOURCE_COMPLETE UINT32_C(0)
#define XLOG_RESOURCE_INCOMPLETE UINT32_C(1)
#define XLOG_RESOURCE_UNKNOWN UINT32_C(2)
#define XLOG_RESOURCE_FAILED UINT32_C(3)

#define XLOG_RESOURCE_CAPTURE_ADMISSION UINT32_C(0)
#define XLOG_RESOURCE_CAPTURE_RELEASE UINT32_C(1)
#define XLOG_RESOURCE_CAPTURE_CANCEL_GROUP UINT32_C(2)

#define XLOG_RESOURCE_COVERAGE_COPY UINT64_C(1)
#define XLOG_RESOURCE_COVERAGE_PEER_COPY UINT64_C(2)
#define XLOG_RESOURCE_COVERAGE_UNIFIED_MEMORY UINT64_C(4)
#define XLOG_RESOURCE_COVERAGE_CAPTURE_API UINT64_C(8)
#define XLOG_RESOURCE_COVERAGE_GRAPH_RESOURCE UINT64_C(16)
#define XLOG_RESOURCE_COVERAGE_DEVICE_GRAPH UINT64_C(32)
#define XLOG_RESOURCE_COVERAGE_HOST_MAPPING_ACCESS UINT64_C(64)
#define XLOG_RESOURCE_COVERAGE_OWNED_BACKING UINT64_C(128)

#define XLOG_RESOURCE_START_VALID UINT64_C(1)
#define XLOG_RESOURCE_END_VALID UINT64_C(2)
#define XLOG_RESOURCE_DELIVERY_DRAINED UINT64_C(4)
#define XLOG_RESOURCE_STARTED_BEFORE_CUDA UINT64_C(8)
#define XLOG_RESOURCE_TOTALS_VALID UINT64_C(16)
#define XLOG_RESOURCE_BACKING_PEAK_VALID UINT64_C(32)
#define XLOG_RESOURCE_ALLOCATOR_COMPLETED UINT64_C(64)
#define XLOG_RESOURCE_BASELINE_COMPLETE UINT64_C(128)

#define XLOG_RESOURCE_DEVICE_WHOLE UINT64_C(1)
#define XLOG_RESOURCE_DEVICE_BEFORE_CUDA UINT64_C(2)
#define XLOG_RESOURCE_DEVICE_NVML_MEMORY_V2 UINT32_C(1)

/* A capacity proof, NEVER an observation or C_raw. The producer verifies a
 * whole physical device, not a MIG partition, free/used memory or CUDA's
 * available capacity. Native binds this immutable certificate to its actual
 * CUDA UUID and original process before admission. Every frozen memory ceiling,
 * including cleanup and delivery, must be at least total_physical_bytes.
 * observer_identity is a nonzero 32-byte nonce minted once before CUDA and
 * never reset. Trace.certificate_identity is exactly this nonce, identifying
 * this one immutable certificate (not a hash of a Python projection). Repeated
 * reads must preserve all fields; a second GPU cannot share its identity.
 * driver_version
 * is the original NUL-terminated NVML driver version; reserved must be zero.
 * A fresh process/device needs a fresh certificate without changing history.
 * A bound process ledger covers application roots on that one whole device;
 * another device/context cannot silently contribute an unbounded baseline. */
typedef struct XlogResourceDeviceCertificate {
    uint32_t struct_size;
    uint32_t state;
    uint8_t observer_identity[32];
    uint8_t device_uuid[16];
    uint64_t total_physical_bytes;
    uint64_t flags;
    uint64_t process_id;
    char driver_version[32];
    uint32_t source;
    uint32_t reserved;
} XlogResourceDeviceCertificate;

#define XLOG_RESOURCE_BACKING_ACQUIRED UINT32_C(1)
#define XLOG_RESOURCE_BACKING_RELEASED UINT32_C(2)
#define XLOG_RESOURCE_BACKING_UNKNOWN UINT32_C(3)

/* Original native storage-owner evidence, not a second allocation counter.
 * Installation precedes the first native allocation. ACQUIRED carries the
 * cold cuMemGetAddressRange-confirmed root extent, not its requested bytes.
 * RELEASED follows actual-use fences and successful cuMemFree, once for that
 * same generation. The collector associates these confirmations with original
 * successful driver events in its timestamp scale, never callback arrival time.
 * It deduplicates the same root with driver/Torch records, retains cache roots,
 * and never counts aliases or slab/private-pool suballocations again. UNKNOWN
 * supplies no size/release proof, even if bytes is zero. Loss, unresolved roots,
 * wraparound or uncertain completion forbid a numerical interval witness.
 * No callback enters Python, calls CUDA, throws, polls or waits for GPU work. */
typedef struct XlogResourceBackingRoot {
    uint32_t struct_size;
    uint32_t kind;
    uint64_t owner_generation;
    uint64_t context_handle;
    uint64_t base;
    uint64_t bytes;
    uint8_t device_uuid[16];
    uint32_t device_ordinal;
    uint32_t reserved;
} XlogResourceBackingRoot;

#define XLOG_RESOURCE_MODULE_LOADED UINT32_C(1)
#define XLOG_RESOURCE_MODULE_READ_STATIC UINT32_C(2)
#define XLOG_RESOURCE_MODULE_CONFIRMED UINT32_C(3)
#define XLOG_RESOURCE_MODULE_UNLOADED UINT32_C(4)
#define XLOG_RESOURCE_MODULE_UNKNOWN UINT32_C(5)
#define XLOG_RESOURCE_MODULE_DISCOVER_STATIC UINT32_C(6)

#define XLOG_RESOURCE_MODULE_INPUT_UNAVAILABLE UINT32_C(0)
#define XLOG_RESOURCE_MODULE_INPUT_CUBIN UINT32_C(1)
#define XLOG_RESOURCE_MODULE_INPUT_PTX UINT32_C(2)
#define XLOG_RESOURCE_MODULE_INPUT_FILE UINT32_C(3)
#define XLOG_RESOURCE_MODULE_INPUT_DIGEST_VALID UINT64_C(1)
#define XLOG_RESOURCE_MODULE_ORIGINAL_LOAD_VALID UINT64_C(1)
#define XLOG_RESOURCE_MODULE_ORIGINAL_DELIVERY_COMPLETE UINT64_C(2)
#define XLOG_RESOURCE_MODULE_ORIGINAL_CUBIN_DIGEST_VALID UINT64_C(4)
#define XLOG_RESOURCE_MODULE_ORIGINAL_IMAGE_VALID UINT64_C(8)
#define XLOG_RESOURCE_MODULE_ORIGINAL_LIBRARY_HANDLE_VALID UINT64_C(16)
#define XLOG_RESOURCE_MODULE_ORIGINAL_UNLOAD_OBSERVED UINT64_C(32)
#define XLOG_RESOURCE_MODULE_ORIGINAL_RESET_OBSERVED UINT64_C(64)

#define XLOG_RESOURCE_STATIC_NAME_VALID UINT64_C(1)
#define XLOG_RESOURCE_STATIC_SOURCE_VALID UINT64_C(2)
#define XLOG_RESOURCE_STATIC_CONTEXT_VALID UINT64_C(4)
#define XLOG_RESOURCE_STATIC_CORRELATION_VALID UINT64_C(8)

/* Slots in cuda_status and valid_queries. An unset validity bit gives no
 * evidence. UINT32_MAX means the corresponding query was never attempted. */
#define XLOG_RESOURCE_STATIC_QUERY_GLOBAL UINT32_C(0)
#define XLOG_RESOURCE_STATIC_QUERY_ALLOCATION UINT32_C(1)
#define XLOG_RESOURCE_STATIC_QUERY_CONTEXT UINT32_C(2)
#define XLOG_RESOURCE_STATIC_QUERY_DEVICE UINT32_C(3)
#define XLOG_RESOURCE_STATIC_QUERY_MEMORY_TYPE UINT32_C(4)
#define XLOG_RESOURCE_STATIC_QUERY_MAPPED UINT32_C(5)
#define XLOG_RESOURCE_STATIC_QUERY_MAPPING_BASE UINT32_C(6)
#define XLOG_RESOURCE_STATIC_QUERY_MAPPING_SIZE UINT32_C(7)
#define XLOG_RESOURCE_STATIC_QUERY_PHYSICAL_BLOCK UINT32_C(8)
#define XLOG_RESOURCE_STATIC_QUERY_BUFFER_ID UINT32_C(9)
#define XLOG_RESOURCE_STATIC_QUERY_COUNT UINT32_C(10)

/* Immutable copies of ORIGINAL activity fields, not reconstructed load events.
 * name/source are length-delimited bytes owned by the lease, without an assumed
 * trailing NUL. Absent fields remain absent even if their storage is zero.
 * record_id identifies the original row, never its position or owner. */
typedef struct XlogResourceOriginalStaticRecord {
    uint32_t struct_size;
    uint32_t memory_kind;
    uint32_t operation;
    uint32_t device_id;
    uint32_t context_id;
    uint32_t correlation_id;
    uint64_t record_id;
    uint64_t address;
    uint64_t bytes;
    uint64_t timestamp;
    uint64_t valid_fields;
    const uint8_t *name;
    uint64_t name_size;
    const uint8_t *source;
    uint64_t source_size;
} XlogResourceOriginalStaticRecord;

/* Native confirmation of this original row under a LIVE module owner. GLOBAL
 * validity requires its nonempty, nonoverflowing global address/bytes to equal
 * original.address/bytes. A same-named foreign symbol keeps its original CUDA
 * status and raw outputs but has no GLOBAL validity or dependent query facts.
 * Symbol bytes are not root bytes.
 * allocation_* comes from successful cuMemGetAddressRange, mapping_* from the
 * actual mapping attributes, and physical_block_id from MEMORY_BLOCK_ID.
 * These are distinct facts: neither a mapping nor its ID proves the full size
 * of a physical allocation. No flag in this record claims that completeness.
 * The collector joins original physical-allocation/map provenance and dedups
 * aliases with its native/external allocator ledger before issuing any peak.
 * Values are unavailable unless the exact successful query bit is present. */
typedef struct XlogResourceStaticConfirmation {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t record_id;
    uint64_t valid_queries;
    uint32_t cuda_status[10];
    uint64_t global_address;
    uint64_t global_bytes;
    uint64_t allocation_base;
    uint64_t allocation_bytes;
    uint64_t mapping_base;
    uint64_t mapping_bytes;
    uint64_t physical_block_id;
    uint64_t buffer_id;
    uint64_t context_handle;
    uint32_t device_ordinal;
    uint32_t memory_type;
    uint32_t mapped;
    uint32_t reserved_tail;
} XlogResourceStaticConfirmation;

#define XLOG_RESOURCE_MODULE_FUNCTION_RUNTIME_ARGUMENT_VALID UINT64_C(1)
#define XLOG_RESOURCE_MODULE_FUNCTION_RUNTIME_RESULT_VALID UINT64_C(2)
#define XLOG_RESOURCE_MODULE_FUNCTION_LOOKUP_ARGUMENT_VALID UINT64_C(4)
#define XLOG_RESOURCE_MODULE_FUNCTION_LOOKUP_RESULT_VALID UINT64_C(8)

/* Original typed runtime FUNCTION argument and its nested cuModuleGetFunction,
 * captured synchronously on the same host thread, never joined by callback
 * arrival order, time proximity or symbol name. Variable-symbol arguments are
 * not function arguments. The collector associates the lookup's actual module
 * input and context with this lease's original live load identity, including
 * lookups made after loading; a reused CUmodule is not the same load.
 *
 * ARGUMENT_VALID qualifies the corresponding callback/correlation and original
 * input fields. RESULT_VALID qualifies the matching original API_EXIT status;
 * UINT32_MAX means that status is unavailable, not success. function_handle is
 * the original successful lookup output only when LOOKUP_RESULT_VALID is set
 * and lookup_status is CUDA_SUCCESS. Failed outputs are not read or fabricated.
 * record_id identifies the original lookup, not its array index.
 *
 * A runtime argument can also be cudaKernel_t, not a registered host symbol.
 * These raw facts therefore prove neither registration nor CUDA ownership and
 * never grant a right to dereference runtime_function_argument. An external
 * producer must first hold its actual runtime/object/context owners, exclude
 * permitted unregister/unload/reset throughout queries, establish the SAME
 * runtime instance and cold-confirm the registered function/module identity.
 * Failed getters cannot create ownership or trigger a replacement loader.
 * Multiple original observations are retained, not selected by arrival order.
 * Native module owners retain their existing ownership without runtime rows. */
typedef struct XlogResourceOriginalModuleFunction {
    uint32_t struct_size;
    uint32_t reserved;
    uint64_t record_id;
    uint64_t valid_fields;
    uint32_t runtime_callback_id;
    uint32_t runtime_correlation_id;
    uint32_t runtime_status;
    uint32_t lookup_callback_id;
    uint32_t lookup_correlation_id;
    uint32_t lookup_status;
    uint64_t runtime_function_argument;
    uint64_t function_handle;
    uint64_t module_handle;
} XlogResourceOriginalModuleFunction;

/* The original successful load API output, context and correlation establish
 * the module association. A resource callback's moduleId or delivery order is
 * not a CUmodule. actual_cubin_sha256 is copied from the original loaded image
 * only with a proven original load-frame association; an input PTX digest is
 * NEVER that digest. Cold READ_STATIC drains original activity outside every
 * CUPTI callback, then returns this stable immutable lease. No collector mutex
 * may remain held while native queries execute. The lease pins all row/string
 * storage through known UNLOADED acknowledgement; uncertainty retains it.
 * It neither pins the CUDA module by itself nor grants execution/recovery.
 * original_loader_callback_id is the original driver load callback ID, not a
 * guessed loader kind. IMAGE_VALID qualifies the original API input image/code
 * pointer, never a filename or the collector's copied loaded cubin. It attests
 * that original input, NOT its current lifetime or right to dereference it.
 * original_image_bytes=0 means its size is unavailable. LIBRARY_HANDLE_VALID
 * qualifies the original successful CUlibrary output, separately from CUmodule.
 * UNLOAD_OBSERVED/RESET_OBSERVED preserve original invalidation history; their
 * absence alone does not exclude concurrent destruction during a native query.
 * functions is an immutable array of original runtime/function lookup facts,
 * pinned by this SAME lease through known unload acknowledgement, including on
 * a non-COMPLETE read. function_size is exactly sizeof its record; a zero count
 * means no such facts, not runtime ownership. No borrowed collector lock crosses
 * the cold-query boundary. Changing this view requires the exact new ABI, not
 * reinterpretation of reserved fields in an already published version. */
typedef struct XlogResourceModuleOriginalView {
    uint32_t struct_size;
    uint32_t state;
    uint64_t lease_id;
    uint64_t owner_generation;
    uint64_t context_handle;
    uint64_t module_handle;
    uint64_t original_load_start;
    uint64_t original_load_end;
    uint32_t original_context_id;
    uint32_t original_correlation_id;
    uint32_t original_module_id;
    uint32_t reserved;
    uint64_t flags;
    uint8_t actual_cubin_sha256[32];
    const XlogResourceOriginalStaticRecord *records;
    uint64_t record_count;
    uint64_t record_size;
    uint32_t original_loader_callback_id;
    uint32_t reserved_tail;
    uint64_t library_handle;
    uint64_t original_image_pointer;
    uint64_t original_image_bytes;
    const XlogResourceOriginalModuleFunction *functions;
    uint64_t function_count;
    uint64_t function_size;
} XlogResourceModuleOriginalView;

/* One real module lifecycle, not a second loader or a synthetic allocation.
 * LOADED registers the successful loader's actual owner before fallible symbol
 * initialization. A native loader initially uses lease_id=0 and its own fresh
 * owner_generation. READ_STATIC is called cold after requested function lookup,
 * under that same live owner; it returns the immutable original host lease.
 * CONFIRMED supplies cold driver query results for that exact immutable lease.
 * It does not certify a physical peak. Input digest validity is explicit and
 * remains separate from the original view's actual cubin digest/provenance.
 * UNLOADED is emitted only after last-use joins and successful actual unload,
 * never at UNLOAD_STARTING or registry replacement. UNKNOWN supplies no release
 * proof. All operations retain generation/context/module identity; native must
 * retain the actual CUDA owner and its original observer on uncertainty.
 * Read/known-unload acknowledgement can be idempotent for the same lease;
 * they must never repeat loader work, CUDA destruction or activity callbacks.
 * This module protocol does not reinterpret BackingRoot.RELEASED/cuMemFree. */
typedef struct XlogResourceModuleObservation {
    uint32_t struct_size;
    uint32_t operation;
    uint64_t owner_generation;
    uint64_t context_handle;
    uint64_t module_handle;
    uint8_t device_uuid[16];
    uint32_t device_ordinal;
    uint32_t input_kind;
    uint64_t flags;
    uint8_t input_sha256[32];
    uint64_t lease_id;
    const XlogResourceStaticConfirmation *confirmations;
    uint64_t confirmation_count;
    uint64_t confirmation_size;
} XlogResourceModuleObservation;

/* DISCOVER_STATIC uses this SAME entrypoint at a quiescent cold boundary, never
 * inside a CUPTI callback or capture. Supply the actual context/device identity,
 * owner_generation=module_handle=0, INPUT_UNAVAILABLE, flags=0, zero digest,
 * no confirmations, and lease_id=cursor (0 for the first candidate). It drains
 * original delivery and returns the next observed, unclaimed load in that
 * context. A COMPLETE view with lease_id=0 is the drained end, not a module;
 * INCOMPLETE/UNKNOWN is NOT end or permission to submit. Nonzero lease_id pins
 * immutable HOST copies of the original load/rows/strings, not CUDA ownership.
 * It is also the cursor for continuing past that candidate. Candidate identity
 * is the pair (lease_id, owner_generation), issued from the collector's original
 * load identity and never reused. Do not mint a separate discovery-owner counter
 * or identify a new original load solely by its reusable CUDA handles.
 *
 * After establishing real runtime image/context ownership and excluding unload,
 * the external producer calls LOADED with that SAME lease_id, owner_generation,
 * context and module. The collector atomically verifies the original load and
 * absence of invalidation before attaching that owner; then READ_STATIC and
 * CONFIRMED use exactly that lease. Discovery/attach never loads, unloads or
 * retains a CUDA module on the producer's behalf. Failure retains the original
 * host lease and any actual runtime owners already acquired. External runtime
 * owners cannot emit UNLOADED merely because they released host references;
 * only the actual successful unload plus completed consumers proves release.
 * original_view is non-NULL only for READ_STATIC or DISCOVER_STATIC. All output
 * custody must be preserved even when either read returns non-COMPLETE. */

#define XLOG_RESOURCE_EVENT_COPY UINT32_C(1)
#define XLOG_RESOURCE_EVENT_PEER_COPY UINT32_C(2)
#define XLOG_RESOURCE_EVENT_UNIFIED_MEMORY UINT32_C(3)
#define XLOG_RESOURCE_EVENT_API UINT32_C(4)
#define XLOG_RESOURCE_EVENT_GRAPH_RESOURCE UINT32_C(5)

/* Unavailable fields carry no evidence, even when their storage contains zero.
 * These bits concern availability, not the numerical value of a field. */
#define XLOG_RESOURCE_VALID_START_TIME UINT64_C(1)
#define XLOG_RESOURCE_VALID_END_TIME UINT64_C(2)
#define XLOG_RESOURCE_VALID_CORRELATION UINT64_C(4)
#define XLOG_RESOURCE_VALID_CONTEXT UINT64_C(8)
#define XLOG_RESOURCE_VALID_GRAPH UINT64_C(16)
#define XLOG_RESOURCE_VALID_NODE UINT64_C(32)
#define XLOG_RESOURCE_VALID_GRAPH_EXEC UINT64_C(64)
#define XLOG_RESOURCE_VALID_RESULT UINT64_C(128)
#define XLOG_RESOURCE_VALID_SOURCE UINT64_C(256)
#define XLOG_RESOURCE_VALID_DESTINATION UINT64_C(512)
#define XLOG_RESOURCE_VALID_COPY_COUNT UINT64_C(1024)
#define XLOG_RESOURCE_VALID_BYTES UINT64_C(2048)
#define XLOG_RESOURCE_VALID_MEMORY_KINDS UINT64_C(4096)
#define XLOG_RESOURCE_VALID_ADDRESS UINT64_C(8192)
#define XLOG_RESOURCE_VALID_PROCESS UINT64_C(16384)
#define XLOG_RESOURCE_VALID_DIRECTION UINT64_C(32768)
#define XLOG_RESOURCE_VALID_RUNTIME_CORRELATION UINT64_C(65536)
#define XLOG_RESOURCE_VALID_DEVICE_LAUNCHED UINT64_C(131072)

typedef struct XlogResourceCopyEvent {
    uint64_t bytes;
    uint64_t start;
    uint64_t end;
    uint64_t graph_id;
    uint64_t node_id;
    uint64_t copy_count;
    uint32_t direction;
    uint32_t source_memory_kind;
    uint32_t destination_memory_kind;
    uint32_t device_id;
    uint32_t context_id;
    uint32_t stream_id;
    uint32_t correlation_id;
    uint32_t runtime_correlation_id;
    uint32_t is_device_launched;
    uint32_t source_device_id;
    uint32_t source_context_id;
    uint32_t destination_device_id;
    uint32_t destination_context_id;
    uint32_t flags;
    uint32_t reserved;
} XlogResourceCopyEvent;

typedef struct XlogResourceUnifiedMemoryEvent {
    uint64_t value;
    uint64_t address;
    uint64_t start;
    uint64_t end;
    uint32_t counter_kind;
    uint32_t source_id;
    uint32_t destination_id;
    uint32_t process_id;
    uint32_t flags;
    uint32_t reserved;
} XlogResourceUnifiedMemoryEvent;

typedef struct XlogResourceApiEvent {
    uint64_t timestamp;
    uint64_t graph_id;
    uint64_t node_id;
    uint64_t graph_exec_id;
    int64_t result;
    uint32_t domain;
    uint32_t callback_id;
    uint32_t site;
    uint32_t correlation_id;
    uint32_t context_id;
    uint32_t reserved;
} XlogResourceApiEvent;

typedef struct XlogResourceEvent {
    uint32_t struct_size;
    uint32_t kind;
    uint64_t valid_fields;
    uint64_t delivery_sequence;
    union {
        XlogResourceCopyEvent copy;
        XlogResourceUnifiedMemoryEvent unified_memory;
        XlogResourceApiEvent api;
    } data;
} XlogResourceEvent;

/* All timestamps use the activity producer's original timestamp scale.
 * Preserve whole events overlapping a boundary, not fabricated clipped events.
 * Process loss/error counters are sticky: reading or changing models does not
 * reset them. Copy totals do not prove completeness or mapped-memory coverage;
 * copy_count is retained separately and must not multiply the original bytes. */
/* backing_peak_bytes is the maximum sum of simultaneously retained original
 * application-owned GPU backing roots in this exact interval. Include the full
 * initial baseline, allocated cache/pool blocks and pending asynchronous uses;
 * count each confirmed original root extent once. It is not tensor bytes,
 * a requested/reserved total, a delta or a sum of separately observed maxima.
 * Unattributable driver/profiler memory is excluded from this observation and
 * bounded separately by the whole-device certificate, not renamed as measured.
 * Require BACKING_PEAK_VALID, BASELINE_COMPLETE, ALLOCATOR_COMPLETED, complete
 * OWNED_BACKING coverage, original certificate identity, nonce and boundaries,
 * and no loss/error/overflow. Incomplete coverage or unknown roots must never
 * substitute zero or saturate a number. Boundaries cannot move to get a result. */
typedef struct XlogResourceTraceView {
    uint32_t struct_size;
    uint32_t state;
    uint8_t original_attempt_nonce[32];
    uint64_t operation_ordinal;
    uint64_t start;
    uint64_t end;
    uint64_t boundary_flags;
    uint64_t requested_coverage;
    uint64_t enabled_coverage;
    uint64_t unsupported_coverage;
    uint64_t process_dropped_records;
    uint64_t process_errors;
    uint64_t process_buffer_overflows;
    uint8_t certificate_identity[32];
    uint64_t backing_baseline_bytes;
    uint64_t backing_peak_bytes;
    uint64_t host_to_device_bytes;
    uint64_t device_to_host_bytes;
    uint64_t device_to_device_bytes;
    uint64_t peer_copy_bytes;
    uint64_t unified_host_to_device_bytes;
    uint64_t unified_device_to_host_bytes;
    const XlogResourceEvent *events;
    uint64_t event_count;
    uint64_t event_size;
} XlogResourceTraceView;

/* The capsule owns this immutable table and its original process context.
 * Functions never enter Python or throw through this C ABI. The native creator
 * calls interval control cold. Original backing/module observations can also
 * arrive on their native owners' cold reaper threads and must be thread-safe;
 * callbacks collecting activity never poll or wait for CUDA.
 * begin binds a fresh native nonce minted after durable admission, before any
 * preparation allocation. If a failing begin returns an interval, retain it.
 * Never begin a replacement for an entered or unknown original interval.
 * finish is called once, only after the original native/model stream joins.
 * Its end timestamp precedes cold delivery/flush, not the end of the flush.
 * read only reads a stable sealed view: it never repeats finish, flush or work.
 * Unknown delivery or backing/allocator completion cannot be released. Trace
 * COMPLETE/read success/DELIVERY_DRAINED alone never resolve backing memory.
 * A consumed view may be released only after its backing result is known under
 * the conditions above and DELIVERY_DRAINED proves callback delivery has ended.
 * Missing backing validity/coverage retains even a completed trace: do not
 * consume its stored scalar, record numerical usage or admit subsequent work.
 * Preserve any already returned model result without repeating it or finish.
 * Retain the
 * original capsule and interval on every unknown outcome or failed release.
 * read and release return COMPLETE on success; read preserves view.state. */
typedef struct XlogResourceObserverApi {
    uint32_t abi_version;
    uint32_t struct_size;
    void *context;
    uint32_t (*begin_interval)(void *context, const uint8_t nonce[32],
                               uint64_t operation_ordinal, void **interval);
    uint32_t (*finish_interval)(void *context, void *interval);
    uint32_t (*read_interval)(void *context, void *interval,
                              XlogResourceTraceView *view);
    uint32_t (*release_interval)(void *context, void *interval);
    /* Cold registration immediately AFTER the actual admission kernel (site=0)
     * or release kernel (site=1) is enqueued, inside that original stream capture.
     * capture_stream is the original CUstream handle, encoded as uint64_t. The
     * producer must resolve its exact capture graph/dependency node here and
     * retain that identity for original GPU activity correlation. Admission is
     * unconditional; release belongs to its actual conditional body and may
     * be skipped. Ordinals are frozen schedule coordinates, not measurements.
     * Capture timestamps are NOT execution boundaries. Never place a host
     * callback, readback or synchronization between graph steps to obtain them.
     * Unknown registration retains the original interval and capture owners.
     * site=2 is a single whole-roster non-submission handoff OUTSIDE capture.
     * XLOG invokes it only after a native proof that the entire original group
     * never entered submit, capture ended and all preparation consumers joined;
     * subsequent capture/submit is permanently forbidden. operation_ordinal is
     * the original group start; step_ordinal is its complete PLANNED count and
     * capture_stream remains its original stream. Keep the registered prefix
     * and every attempted start/end status, including a known start without an
     * attempted end. Unknown binding results must refuse cancellation. Only a
     * COMPLETE handoff removes execution-subinterval obligations: it invents no
     * activity/timestamps/peaks or skipped/completed outcome. Whole-interval
     * backing coverage/peak, allocator completion, drained delivery and all
     * error/loss checks remain mandatory before original finish/read/release.
     * Unknown cancellation retains the same interval; never repeat the handoff. */
    uint32_t (*bind_step_capture)(void *context, void *interval,
                                  uint64_t operation_ordinal,
                                  uint64_t step_ordinal, uint64_t capture_stream,
                                  uint32_t site);
    /* Read a stable original step view only after whole-interval finish. The
     * same continuous backing producer authenticates GPU execution boundaries
     * in its original activity timestamp scale. First start equals whole start
     * before group allocations; interiors join contiguously at the next step's
     * real GPU admission; last end equals whole end after retirement/adoption.
     * Every interval includes already-held shared allocations and the absolute
     * backing baseline. All validity/coverage/loss/completion requirements above
     * apply independently to every step. Group maxima, capture-time timestamps,
     * ordinal ordering or a copied scalar never prove a step's backing peak.
     * This read never flushes, registers, finishes or executes anything. */
    uint32_t (*read_step_interval)(void *context, void *interval,
                                   uint64_t operation_ordinal,
                                   XlogResourceTraceView *view);
    uint32_t (*read_device_certificate)(void *context,
                                        const uint8_t cuda_uuid[16],
                                        XlogResourceDeviceCertificate *certificate);
    uint32_t (*record_native_backing_root)(void *context,
                                          const XlogResourceBackingRoot *root);
    /* Join only the original unknown CANCEL_GROUP observation. Require its
     * creator, live interval and exact retained ordinal/count/stream, including
     * the originally attempted ordinal. Without that original attempt, on a
     * mismatch, or after a known failure, return FAILED. A confirmed original
     * result returns COMPLETE idempotently; INCOMPLETE/UNKNOWN may only repeat
     * its read-only capture-state observation. Never bind, capture, submit,
     * create markers or clear original errors here. This is not a native
     * non-submission proof and grants no finish/backing/delivery exemption. */
    uint32_t (*resolve_step_cancellation)(void *context, void *original_handle,
                                          uint64_t original_operation_ordinal,
                                          uint64_t original_planned_count,
                                          uint64_t original_capture_stream);
    /* Thread-safe original-owner entrypoint, including cold retirement reapers.
     * READ_STATIC runs only at the live owner's quiescent cold boundary. It may
     * drain original CUPTI delivery outside activity callbacks, but neither it
     * nor other observations may call CUDA/Python or hold a collector mutex
     * across the return to native queries. Even a non-COMPLETE original read may
     * return a lease: preserve its output/custody before checking the status. */
    uint32_t (*observe_native_module)(void *context,
                                      const XlogResourceModuleObservation *observation,
                                      XlogResourceModuleOriginalView *original_view);
} XlogResourceObserverApi;

#ifdef __cplusplus
}
#endif

#endif
