# Environment variables

Every environment variable the XLOG runtime, build, and probabilistic engines read — tuning knobs, parity kill switches, and build-time controls.

XLOG reads environment variables for tuning, debugging, and parity checking. The
defaults are the production configuration — you do not need to set anything for
correct results.

Most non-default settings fall into two families. *Tuning knobs* trade memory for
speed. *Kill switches* turn off one of XLOG's fast execution routes and force the
work back onto its slower baseline route, so you can run the same query both ways
and compare the answers and timing (an A/B comparison).

Two of those fast routes are named throughout this page. The *worst-case-optimal
join* (WCOJ) computes multi-way graph patterns — triangles, cycles, cliques —
directly, instead of building a large intermediate table first. *Factorized*
execution represents intermediate results in a compressed form instead of listing
every row.

Boolean knobs generally accept `1`/`true` to enable and `0`/`false` to disable;
per-variable parsing is noted where it differs.

<Note>
Variables tagged **(unreleased)** exist on `main` but are not part of the latest
published release. They apply only when building from source.
</Note>

## Kernel artifacts and build

These control where XLOG finds its compiled GPU kernel files (the `.cubin` and
`.ptx` binaries the CUDA code runs from) and how those kernels are compiled when
you build from source. They are read by the `xlog-cuda` build script
(`crates/xlog-cuda/build.rs`) at compile time, or by the kernel loader at startup.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_CUBIN_DIR` | Runtime: directory of staged kernel artifacts (`*.sm_NN.cubin`, `*.portable.ptx`), searched before the binary-adjacent `kernels/` directory and the embedded PTX fallback. The `pyxlog` package sets it automatically to its bundled kernels when unset. | path; unset by default |
| `NVCC_PATH` | Build: explicit path to `nvcc`; falls back to `PATH` lookup. | path |
| `XLOG_PTXAS_PATH` | Build: override the `ptxas` binary used for cubin assembly; the build panics if the path does not exist. | path; unset by default |
| `XLOG_NO_CUBIN` | Build: skip cubin generation, produce portable PTX only. | `1` to enable; default off |
| `XLOG_CUBIN_ARCHS` | Build: comma-separated cubin target architectures. | e.g. `sm_89,sm_120`; default `sm_120` |
| `XLOG_PTX_MAX_VERSION` | Build: rewrite the `.version` directive of the generated portable PTX down to the named ISA, so wheels run on drivers older than the build toolkit (for example `8.4` for pre-CUDA-13 drivers). No-op when unset. | version string; unset by default |
| `XLOG_RUSTDOC_NO_CUDA` | Build: skip kernel compilation entirely and emit empty embedded kernel data — docs-only builds without a CUDA toolkit. `DOCS_RS=1` has the same effect. | `1` to enable; default off |

## Runtime and memory

Knobs for GPU memory budgeting and low-level debugging of the runtime. A *CUDA
stream* is a GPU work queue. The `XLOG_DEBUG_*` probes are diagnostic aids for
catching memory bugs and are off in normal use. One variable, `XLOG_CDCL_TRACE`,
traces the GPU SAT solver (CDCL, conflict-driven clause learning — the search
behind exact probabilistic inference) and exists only in debug builds.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_WCOJ_POOL_MB_PER_STREAM` | Device memory pool budget per CUDA stream, in MiB. | positive integer; default `256` |
| `XLOG_WARMUP_PROFILE` | Diagnostic: collect per-module PTX load timing and compile-warmup profiling stats. | `1` to enable; default off |
| `XLOG_DEBUG_POISON_ALLOC` | Debug probe: fill fresh legacy allocations with `0xDD` so reads of unwritten memory surface deterministically. Read once per process. | `1` to enable; default off |
| `XLOG_DEBUG_POISON_FREE` | Debug probe: poison legacy allocations with `0xDD` at drop so live aliases of freed memory become visually distinct. Read once per process. | `1` to enable; default off |
| `XLOG_DEBUG_ALLOC_GUARD` | Debug probe: track live allocation ranges and panic if the allocator hands out a region overlapping a live one (double-hand-out / use-after-free detector). | `1` to enable; default off |
| `XLOG_DEBUG_VERIFY_CLONES` | Debug probe: byte-compare every on-device buffer clone against its source immediately after the copy, to discriminate transport faults from source corruption. | `1` to enable; default off |
| `XLOG_CDCL_TRACE` | Debug-build-only trace of the GPU CDCL solver (compiled in only with debug assertions; release binaries ignore it). | set to any value; default off |

## Join and optimizer knobs

Controls for how XLOG plans and executes joins — the operations that combine
relations. Several apply to the worst-case-optimal join (WCOJ) routes for
graph-shaped rules: `triangle` (three-node cycles), `4-cycle`, and `chain` name
the rule shapes each dispatch handles. A *cost model* estimates which route will
be cheaper for a given rule, and a *variable-order heuristic* chooses the order
in which join variables are bound.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_CSE` | Enable runtime common-subexpression elimination. | `1`/`true`/`on`/`yes`; default off |
| `XLOG_ADAPTIVE_REOPT` | Enable adaptive runtime re-optimization. | `1`/`true`/`on`/`yes`; default off |
| `XLOG_ADAPTIVE_REOPT_MIN_RATIO` | Mis-plan threshold ratio for adaptive re-optimization; values below `1.0` or non-finite fall back to the default. | float `>= 1.0`; default `1.2` |
| `XLOG_PERSISTENT_HASH_INDEXES` | Reuse of persistent build-side hash indexes across joins. On by default; set `0`/`false`/`off`/`no` to disable. | default on |
| `XLOG_PERSISTENT_HASH_INDEX_BACKGROUND_BUILD` | Background-build telemetry for persistent hash indexes. | `1`/`true`/`on`/`yes`; default off |
| `XLOG_WCOJ_COST_MODEL` | Select the WCOJ cost model: `cardinality` or `skew` (alias `skewclassifier`; any other non-empty value also resolves to the skew classifier). | default `cardinality` |
| `XLOG_USE_WCOJ_TRIANGLE_U32` | Force the WCOJ triangle dispatch on. When unset, the adaptive cost model decides per rule (adaptive is on by default). | `1`/`true`; default off (adaptive) |
| `XLOG_USE_WCOJ_4CYCLE` | Force the WCOJ 4-cycle dispatch on (width-neutral: u32, u64, Symbol). **(unreleased)** | `1`/`true`; default off |
| `XLOG_USE_WCOJ_4CYCLE_ADAPTIVE` | Opt in to adaptive cost-model dispatch for 4-cycles. Unlike triangle, 4-cycle adaptive dispatch is off by default. **(unreleased)** | `1`/`true`; default off |
| `XLOG_WCOJ_CHAIN_ENABLE` | Chain-shaped rule dispatcher. On by default; set `0`/`false` for A/B measurements against the binary-join path. | default on |
| `XLOG_WCOJ_BLOCK_WORK_UNIT` | WCOJ block work-unit size. Out-of-range or unparsable values log a warning and use the default. | `1..=8192`; default `1024` |
| `XLOG_BURIED_SKEW_THRESHOLD` | Skew threshold used by the hypergraph variable-order heuristic; non-finite or non-positive values are ignored. | float `> 0`; default `3.0` |

## Factorized and WCOJ kill switches

These turn off one of XLOG's fused or factorized fast routes and force the work
back onto its plain baseline route, so the two can be run against each other for
correctness (parity) and speed comparison. All default to the production fast
route (switch off).

Two route names appear below. *Free Join* is XLOG's generalized multi-way join,
which combines several relations at once rather than one pair at a time. The
*factorized delta* is the compressed incremental step of a recursive rule: each
round of a recursive query derives only the newly-available facts (the *delta*),
and "factorized" keeps that step compressed instead of listing every row.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_DISABLE_FREE_JOIN` | Force every general multiway body through the embedded binary-join fallback instead of the generalized Free Join dispatch. **(unreleased)** | `1`/`true`; default off |
| `XLOG_DISABLE_FACTORIZED_DELTA` | Force every recursive delta step through the legacy hash-join and diff path instead of the factorized recursive-delta dispatch. **(unreleased)** | `1`/`true`; default off |
| `XLOG_FACTORIZED_DELTA_MAX_DOMAIN` | Raise the dense-domain dispatch cap for the factorized delta, up to the provider hard bound of `2^16`. **(unreleased)** | integer; default `2^14` |
| `XLOG_FACTORIZED_DELTA_MAX_TABLE_BYTES` | Byte ceiling for the sparse route's conservative hash table; above it the sparse entry declines to the legacy path. **(unreleased)** | integer bytes; default half the device budget |
| `XLOG_FACTORIZED_DELTA_WORK_DIVISOR` | Per-iteration work floor: dispatch only when estimated candidate work is at least the bitmap word count divided by this value, protecting sparse long-chain fixpoints from the popcount-and-scan floor. **(unreleased)** | integer `>= 1`; default `8` |
| `XLOG_DISABLE_WCOJ_GROUPBY_FUSION` | Force every group-by over a WCOJ triangle through the materialize-then-groupby path instead of the aggregate-fused dispatch. **(unreleased)** | `1`/`true`; default off |
| `XLOG_DISABLE_WCOJ_4CYCLE` | Kill switch for the 4-cycle dispatch, overriding both force and adaptive gates. **(unreleased)** | `1`/`true`; default off |
| `XLOG_WCOJ_STRICT` | Diagnostic: propagate WCOJ pipeline errors instead of the default counted-and-logged decline to the binary-join fallback. **(unreleased)** | `1`/`true`; default off |

<Note>
A WCOJ layout or kernel error never corrupts the relation store: by default it is
counted, logged to stderr, and the rule falls back to the binary-join path.
`XLOG_WCOJ_STRICT=1` turns that decline into a hard error for diagnosis.
</Note>

## Probabilistic

Controls for XLOG's two probabilistic-inference engines. A *Monte Carlo* engine
estimates probabilities by random sampling, and an exact engine (*D4*) compiles
the program to a logic circuit and counts its satisfying assignments. The exact
engine works over Boolean formulas in CNF (conjunctive normal form — an AND of
OR-clauses) and can check its own result with the built-in SAT solver; the size
bounds below keep that step from overrunning GPU memory.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_MC_RESIDENT_MEMORY_BUDGET_BYTES` | Memory budget for the GPU-resident Monte Carlo engine's launch planning. An unparsable value is a typed error, not a silent default. | integer bytes; unset = provider budget |
| `XLOG_MC_RESIDENT_BLOCKS_PER_WORLD` | Blocks launched per sampled world in the resident MC engine. Zero or unparsable values are typed errors. | integer `>= 1`; default `1` |
| `XLOG_D4_VERIFY_MAX_CONFLICTS` | Per-verify conflict budget for the GPU CDCL equivalence verifier. When the budget runs out, the verify declines with the typed `VerifyBudgetExceeded` error rather than trusting an unfinished search. Read once per process. **(unreleased)** | integer; default `0` = unlimited |
| `XLOG_D4_VERIFY_MAX_VARS` | Size bound on CNF variable capacity before the D4 compile and verify launch; an over-bound program declines with the typed `CompileCapacityExceeded` error instead of risking a context-poisoning CUDA launch failure. **(unreleased)** | integer; default unbounded |
| `XLOG_D4_VERIFY_MAX_CLAUSES` | Same size bound, on CNF clause capacity. **(unreleased)** | integer; default unbounded |
| `XLOG_DEBUG_VERIFY_SIZE` | Diagnostic: log the CNF variable/clause capacities the size bound sees, so safe values for the two bounds above can be read off a real workload. **(unreleased)** | `1` to enable; default off |
| `XLOG_CIRCUIT_CACHE_DIR` | Directory for the compiled-circuit disk cache. | path; default `$XDG_CACHE_HOME/xlog/circuits`, else `$HOME/.cache/xlog/circuits` |
| `XLOG_CIRCUIT_CACHE_MAX_MB` | Disk-cache size limit; oldest entries are evicted beyond it. | integer MiB; default `512` |

## Rule induction

Controls for exact rule induction — learning logical rules from example data by
scoring candidate rules on the GPU. These tune a shared-memory optimization for
chain-shaped rules and gate an optional slow reference implementation used only
for validation.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_ILP_EXACT_CHAIN_SMEM` | Shared-memory tiling for the chain topology in the exact-induction scoring kernel. On by default; set `0`/`false`/`off`/`no` to disable. | default on |
| `XLOG_ILP_EXACT_CHAIN_SMEM_MIN_ROWS` | Minimum row count before the shared-memory chain path engages. | integer; default `256` |
| `XLOG_ALLOW_PYTHON_ILP_REFERENCE` | Permit `induce_exact(backend="python")`, the host-orchestrated reference scorer. Without it the Python backend raises a config error — it exists only for explicit parity or compatibility validation, never as a production path. | `1` to allow; default rejected |

## Device runtime (opt-in)

The *recorded-launch* device-runtime stack is an alternative GPU backend that
records each operation once and replays it, rather than launching kernels the
legacy way. It is opt-in: the legacy launch paths remain the production default
until the runtime stack is certified end-to-end. The per-operator flags below
parse as "set and non-empty and not `0`" and each is also implied by the umbrella
`XLOG_USE_RECORDED_OPS`.

| Variable | Effect | Values / Default |
|---|---|---|
| `XLOG_USE_DEVICE_RUNTIME` | Select the device-runtime backend for a whole test or certification process (read at context construction, cached per process). Production embedders select the runtime through the API instead. | `1`/`true`; default legacy |
| `XLOG_USE_RECORDED_OPS` | Umbrella switch: route all supported operators through recorded launches. | non-empty, not `0`; default off |
| `XLOG_USE_RECORDED_FILTERS` | Recorded filter dispatch. | same; default off |
| `XLOG_USE_RECORDED_SORT` | Recorded sort dispatch (u32 / Symbol keys only). | same; default off |
| `XLOG_USE_RECORDED_DEDUP` | Recorded full-row dedup dispatch (all-u32 / Symbol columns). | same; default off |
| `XLOG_USE_RECORDED_GROUPBY` | Recorded group-by dispatch (u32 / Symbol keys; count, sum, min, max). | same; default off |
| `XLOG_USE_RECORDED_HASH_JOIN` | Recorded hash-join dispatch (all four join types; at most 4 key columns). | same; default off |
| `XLOG_USE_RECORDED_CSM` | Recorded count-scan-materialize hash-join sub-strategy (inner and left-outer joins only; consulted only after the recorded hash-join path is selected). | same; default off |
| `XLOG_USE_CSM_CUDA_GRAPH` | CUDA Graph capture and replay for the bounded CSM path; requires the recorded CSM path to be selected first. | same; default off |
| `XLOG_USE_CUDA_GRAPHS` | Broader graph opt-in that also enables the CSM graph path. | same; default off |
| `XLOG_CSM_CUDA_GRAPH_AUTO_OUTPUT_CAP` | Worst-case output-row cap below which the CSM graph path auto-selects a capacity class. | integer; default `1000000` |

## Test-only variables

A few variables appear only in test binaries, benches, or release validation and
are not production knobs: `XLOG_REQUIRE_CUDA=1` makes CUDA-initialization failures
in the certification suite panic instead of skipping (exported by
`scripts/validate_release_gpu.sh` so a CPU-only machine can never satisfy the
release gate); `XLOG_DETERMINISTIC`, `XLOG_CNF_OUTPUT_PATH`,
`XLOG_GPU_HASH_OUTPUT_PATH`, and the `XLOG_A3`/`XLOG_A3A4` family drive
cross-process determinism and stress tests.

## See also

- [CLI reference](/reference/cli) — the CLI itself reads no environment variables
- [Probabilistic engines](/probabilistic/engines) — exact and Monte Carlo inference
