#ifndef XLOG_RESOURCE_OBSERVER_H
#define XLOG_RESOURCE_OBSERVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define XLOG_RESOURCE_OBSERVER_ABI_VERSION UINT32_C(3)
#define XLOG_RESOURCE_OBSERVER_CAPSULE_NAME "xlog.resource_observer.v3"
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
 * calls them cold; callbacks collecting activity never poll or wait for CUDA.
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
} XlogResourceObserverApi;

#ifdef __cplusplus
}
#endif

#endif
