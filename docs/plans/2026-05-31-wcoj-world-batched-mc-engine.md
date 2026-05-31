# Design Checkpoint — WCOJ / Tensorized World-Batched MC Engine

**Status:** design checkpoint (pre-implementation). Supersedes the dense-boolean
resident engine as the *general* MC execution path; dense-boolean remains a
bounded-fragment precursor (proof that the no-host loop is achievable).

**Goal recap (supervisor):** a GPU-resident MC engine that reuses the real
*sparse* relational/WCOJ execution surface, treats world/sample id as a
first-class relation dimension, removes host count-readback from operator
chaining, and orchestrates recursion/fixpoint device-side — with **zero host
interaction in the measured loop** and **fail-closed** (never silent host
fallback) when device-resident sizing is not provably within budget.

---

## 1. Why the existing operators can't be reused as-is

Every operator in `crates/xlog-cuda/src/provider/relational.rs` follows
**count → host-read → allocate → materialize**:

- `device_row_count(...)` → `dtoh_scalar_untracked` (e.g. lines 241/278/464/574/778)
- WCOJ is two-phase count-then-materialize; the count is read to the host between
  phases (`dtoh_scalar_untracked` at 1245/2908/3265/3423) to size the materialize
  buffer.

Each such read is a per-operator host metadata read + a host allocation decision —
exactly the interactions the goal forbids inside the measured loop. The dense
engine avoided this by having *no dynamic output size* (`domain^arity` bitset),
which is also why it is limited to small bounded domains.

## 2. Core idea — device-resident sizing

Replace `host count→allocate→materialize` with a device-resident sizing protocol
so the host never learns a row count inside the loop:

1. **Worst-case preallocation (preferred).** Before the measured region, compute a
   *static* upper bound on each operator's output from input `row_cap`s and the
   operator's combinatorics (e.g. join ≤ `|A|·|B|`; AGM/WCOJ bound for multiway
   joins). Allocate output arenas to that bound. The kernel writes with an
   **atomic cursor** (`atomicAdd`) into the arena and stores the produced count in
   a **device** `u32` (the buffer's `d_num_rows`) — never read to host in-loop.
2. **Budget gate (fail-closed).** If a worst-case bound exceeds the configured
   memory budget, **fail closed before the measured region** with a typed
   `ResidentResourceError { operator, bound_bytes, budget_bytes }`. Do **not**
   silently fall back to host sizing. (Caveat per supervisor: a worst-case bound
   is not automatically practical.)
3. **Device prefix sums for offsets.** Where per-row fan-out varies (multiway
   join expansion), use an on-device exclusive scan over per-input-row output
   counts to assign write offsets — the existing `exclusive_scan_u32_inplace` /
   `multiblock_scan_u32_inplace` run device-side without host reads.
4. **Sizes stay device-resident.** `CudaBuffer::d_num_rows` is the single source of
   truth across the loop; `row_cap` (host) is only the arena capacity, set at
   allocation time from the static bound — not a per-iteration readback.

## 3. World/sample id as a relation dimension

- Relations become **world-segmented columnar**: a logical relation `R` is stored
  as columns plus a `world_id` column (or, equivalently, a per-world segment
  offset array `[num_worlds+1]`). Sampled probabilistic facts/ADs populate the
  per-world EDB segments from the device Bernoulli matrix (already device-resident).
- Joins are **batched over worlds**: the join key is extended with `world_id` so a
  tuple in world `w` only joins tuples in world `w`. WCOJ/hash-join kernels already
  hash composite keys (`compute_composite_hash`) — adding `world_id` as a leading
  key column is the minimal change. This is the "tensorized" view: the world axis
  is just another join dimension.
- Query/evidence counting reduces per-world on device into `[num_worlds]` /
  `[num_queries]` aggregates (same as the dense engine's atomic counters).

## 4. Device-side fixpoint orchestration

- Recursion (e.g. transitive closure) runs as a **device-orchestrated semi-naive
  loop**: each iteration is a world-batched join of the delta against the EDB,
  appended into the IDB arena via atomic cursor; a **device change flag**
  (`atomicOr` of "did the cursor advance") decides continuation.
- The iteration loop must be **device-side** (no host loop over fixpoint
  iterations): either a **CUDA graph with a conditional/while node** (CUDA 13 is
  available on this toolkit) or a **persistent megakernel** with grid-wide sync
  (cooperative groups). The dense engine used block-per-world `__syncthreads`;
  the sparse engine needs grid-wide coordination because a world's relation may
  exceed one block. Decision deferred to implementation spike (cooperative launch
  vs. CUDA-graph while-node); both keep the loop off the host.
- Iteration count recorded to a device `iter_trace` (per world or global), read
  only after the measured region.

## 5. Instrumentation (extends `McNoHostStats`)

Add two counters so the measured region can prove the stronger contract:
- `host_fixpoint_iterations` — host-side fixpoint loop count (must be 0).
- `per_operator_host_allocations` — device allocations issued *inside* the
  measured region (must be 0; all arenas allocated before the region).

Plus the existing `tracked_htod=0`, `tracked_dtoh=0`,
`untracked_metadata_reads=0`, `host_loop_iterations=0`,
`per_sample_host_launches=0`. A provider allocation counter
(`device_alloc_count`) bracketed around the measured region backs
`per_operator_host_allocations`.

## 6. Fail-closed surface (typed, before execution)

- Over-budget worst-case bound → `ResidentResourceError` (operator + bound + budget).
- Non-monotone / negation, comparison/arithmetic, unbounded/compound terms →
  reuse the existing structural `ResidentRejection` analyzer (these remain hard:
  negation needs device-side stratified/WFS; unbounded terms have no finite arena).

## 7. Build order (incremental, green between steps)

1. **Device-sizing primitive + counters** — `device_alloc_count`,
   `host_fixpoint_iterations` on the provider/result; a world-batched single join
   that worst-case-preallocates, writes via atomic cursor, keeps the count in
   `d_num_rows`, and reads nothing to host. Pilot: 2-relation join over N worlds,
   exact counts, `is_no_host()` constant in N. **Over-budget fail-closed negative.**
2. **World-batched WCOJ multiway join** — reuse `wcoj_*` kernels with `world_id`
   leading key; AGM/worst-case bound for arena; device prefix-sum offsets.
3. **Device-side semi-naive fixpoint** — recursion via device-orchestrated loop +
   change flag; recursive transitive-closure pilot with a non-base derived tuple
   and `iter_trace`>1, `host_fixpoint_iterations=0`.
4. **Rewire + dense fallback classification** — route `evaluate_gpu_device*` to the
   sparse engine for in-fragment programs; dense engine retained only for tiny
   bounded cases (or removed if subsumed). Dense pilots stay green.
5. **Docs + acceptance** — dense = bounded-fragment precursor; WCOJ world-batched =
   real general path.

## 8. Acceptance (supervisor minimum)

- Dense no-host pilots remain green.
- New sparse/WCOJ world-batched pilot: `is_no_host()` (all 7 counters 0) constant in N.
- Recursive transitive closure via device-side fixpoint produces non-base tuples;
  `host_fixpoint_iterations=0`.
- Over-budget WCOJ bound fails closed before measured execution (typed diagnostic).
- Docs state the dense/sparse split clearly.

## 9. Risks / open questions

- **Device-side loop mechanism**: cooperative-groups grid sync vs. CUDA-graph
  while-node. Spike both; pick by reliability under cudarc 0.19.
- **Worst-case bound tightness**: naive `|A|·|B|` per join can blow the budget for
  multi-join recursion; need AGM bound + per-world segmenting to keep arenas sane,
  else fail closed. This is the primary feasibility risk.
- **`d_num_rows` propagation**: every reused operator must be audited to ensure it
  consumes/produces the device count without the `device_row_count` host read on
  the in-loop path (may require no-host variants of the operators).
