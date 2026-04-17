# M8 Phase 1 · `ilp_exact` kernel design

Status: design-locked 2026-04-17
Stage: Task 3 Stage B, sub-step 3B.3
Precedes: 3B.4 (CUDA + manifest plumbing), 3B.5 (parity/D2H gate)

## Scope

One CUDA kernel module named **`ilp_exact`** that scores every
`(topology, left_candidate, right_candidate)` triple for the four DTS
topologies (`chain`, `star`, `fanout`, `fanin`) against one request's
positive + negative query sets.

**In scope.**

- Compute `positives_covered[topology, L, R]` and
  `negatives_covered[topology, L, R]` for every triple.
- Emit the two count arrays to host with a single D2H copy.
- Keep any host-side setup transfers constant-sized, independent of `|C|`
  (the candidate count) and independent of `|P| + |N|` (the query counts).

**Out of scope.**

- Top-K reduction + tie diagnostics (already locked in Rust by Stage A in
  `crates/xlog-induce/src/reduce.rs`; deterministic, host-side).
- Name → `RelId` resolution (pyxlog boundary).
- Support for column types other than `u64`. If DTS ever declares relations
  with a different symbol type, a separate kernel instance can be dispatched
  following the `ilp_mark_selected_ids_{u32,i32,i64,u64}` precedent.

## Engine inputs (what the launcher sees)

Shape expected by the provider launcher method in 3B.4:

- `candidate_buffers: &[&CudaBuffer]` — `C` binary-pair relations; each
  column type `U64`; rows accessible via `cached_row_count()`.
- `positives: &CudaBuffer` — 2 columns, `U64`, `P` rows.
- `negatives: Option<&CudaBuffer>` — 2 columns, `U64`, `N` rows.
- `k_per_topology` — not used by the kernel; only by the host reducer.

## Device-side data layout (built once per `induce_exact` call)

The kernel sees a packed columnar view of the candidate relations. One-time
setup in the launcher builds four device buffers:

| Buffer | Type / length | Contents |
|---|---|---|
| `cand_offsets` | `u32 × (C+1)` | Exclusive-sum prefix of candidate row counts. `offsets[i+1] - offsets[i]` is relation `i`'s row count. |
| `cand_arg0` | `u64 × total_rows` | Concatenation of candidate column-0 (`arg0`) arrays. |
| `cand_arg1` | `u64 × total_rows` | Concatenation of candidate column-1 (`arg1`) arrays. |
| `pos_covered`, `neg_covered` | `u32 × 4·C·C` | Output count arrays. |

`pos_arg0`, `pos_arg1`, `neg_arg0`, `neg_arg1` are direct references to the
existing positive / negative `CudaBuffer` columns — no staging copy.

`cand_arg0` / `cand_arg1` / `cand_offsets` are built by **C device-to-device
memcpys** in the setup phase (one per candidate). These are not per-pair
transfers — they scale with `C`, not with `4·C²`, and they live in setup
rather than the scoring loop. The xlog D2H counter is not incremented by
D2D memcpys.

`cand_offsets` is tiny (~C+1 `u32`s) and is **populated from known host
values**; it is uploaded with a single H2D copy (not counted in the D2H
budget, which tracks D→H only).

## Launch geometry

- **Grid**: `(4 · C · C)` blocks, addressed as `(topology_idx, L_idx, R_idx)`.
  At `C = 20` the total is `1 600` blocks — modest but ample for occupancy
  given the tiny work per block.
- **Block**: `256` threads. Block handles all positive + negative queries
  for its single `(topology, L, R)` triple.
- **Shared mem**: two `__shared__ uint32_t reduce_buf[256]` scratchpads (one
  for positives, one for negatives). ~2 KiB per block.

Block-id decoding inside the kernel:

```cuda
uint32_t block_id = blockIdx.x;          // alternative: 3D grid
uint32_t topology_idx = block_id / (C * C);
uint32_t lr_id       = block_id % (C * C);
uint32_t L = lr_id / C;
uint32_t R = lr_id % C;
```

Equivalently we can launch with a 3D grid `grid_dim = (C, C, 4)` — both map
the same work. 3D grid has the advantage that `L, R, topology` are read
directly from `blockIdx.{x,y,z}` with no division.

## Per-topology coverage predicates

Let `L_buf = (L_arg0, L_arg1)` be the left candidate's rows and `R_buf =
(R_arg0, R_arg1)` the right candidate's rows, each `|L|` / `|R|` rows long.
A query pair is `(qx, qy)`.

| Topology | Head rule | Coverage predicate |
|---|---|---|
| `chain`  | `H(X,Y) :- L(X,Z), R(Z,Y)` | `∃ z . (qx, z) ∈ L ∧ (z, qy) ∈ R` |
| `star`   | `H(X,Y) :- L(X,Y), R(X,Y)` | `(qx, qy) ∈ L ∧ (qx, qy) ∈ R` |
| `fanout` | `H(X,Y) :- L(X,Z), R(X,Y)` | `∃ _ . (qx, _) ∈ L ∧ (qx, qy) ∈ R` |
| `fanin`  | `H(X,Y) :- L(X,Y), R(Z,Y)` | `(qx, qy) ∈ L ∧ ∃ _ . (_, qy) ∈ R` |

Each predicate is implemented as a tight in-kernel loop over the respective
row sets — `chain` is the most expensive at `O(|L| · |R|)` per query,
others are `O(|L| + |R|)`. At our sizes (|L|, |R|, |P|, |N| ≤ ~50) the
worst case is `~2.5·10³` integer compares per query, which is microseconds
per block.

## Thread work inside a block

Pseudocode (positive coverage shown; negatives mirror):

```cuda
__shared__ uint32_t pos_scratch[256];
__shared__ uint32_t neg_scratch[256];
uint32_t tid = threadIdx.x;

// Per-thread accumulator: each thread handles a stripe of the query set.
uint32_t local_pos = 0;
for (uint32_t q = tid; q < num_pos; q += blockDim.x) {
    uint64_t qx = pos_arg0[q];
    uint64_t qy = pos_arg1[q];
    if (topology_matches(topology_idx, L_buf, R_buf, qx, qy)) local_pos += 1;
}
pos_scratch[tid] = local_pos;
// ... same for negatives → neg_scratch[tid] ...
__syncthreads();

// Block reduction → pair-halving; leader writes one slot to global.
for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        pos_scratch[tid] += pos_scratch[tid + s];
        neg_scratch[tid] += neg_scratch[tid + s];
    }
    __syncthreads();
}
if (tid == 0) {
    uint32_t slot = topology_idx * (C * C) + L * C + R;
    pos_covered[slot] = pos_scratch[0];
    neg_covered[slot] = neg_scratch[0];
}
```

Crucially: **each block writes one unique slot** in each output array. No
cross-block atomics. No sort. Determinism follows from the integer-only
counting and from the sequential pair-halving reduction (ordering of
addition is fixed by thread id → no associativity concern).

## D2H budget per `induce_exact` call

| Transfer | Direction | Size | Scales with | Counted by `d2h_transfer_count`? |
|---|---|---|---|---|
| Candidate concat + offsets build | D→D / H→D | `O(total_rows + C)` | `C` (setup, not hot loop) | No (not D→H) |
| Positives / negatives ingest | already device-resident via DLPack | 0 | — | No |
| Kernel launch | — | 0 | — | No |
| Result download | D→H | `2 · 4 · C · C · 4 bytes = 32·C²` bytes | `C²` in bytes; **1 in count** | **Yes (1)** |

Total D2H count per call: **1**. The D2H test
(`test_induce_exact_native_does_not_scale_d2h_with_candidate_pairs`) passes
with any number of candidates.

Optional tightening if the D2H counter turns out stricter than expected:
emit `pos_covered` and `neg_covered` into a single interleaved `u32 × 8·C²`
buffer so the result download is a single unambiguous transfer (not two
adjacent copies that some counter implementations might register as two).

## Determinism

- Integer counts only — no floating-point reduction.
- Each block owns one output slot — no atomics on the hot path.
- Host reducer (`reduce_per_topology`) is already pinned deterministic by
  Stage A's 16 unit tests.

## Error model

All shape / type / index-range validation is host-side in
`crates/xlog-induce/src/lib.rs` (3B.1, done). The kernel itself runs with
pre-validated inputs and performs no defensive checks. Bounds-safe by
construction: each thread guards `q < num_pos` / `q < num_neg`; every block
writes a slot inside `[0, 4·C²)`.

## Open items (deferred — not blocking 3B.4)

- **Type dispatch**: if a future DTS run declares `pred p_X(u32, u32)` or
  uses `Symbol` IDs, add a second kernel entry `ilp_exact_score_u32` and
  pick it via the host launcher based on `buf.schema().column_type(0)`.
  Matches `ilp_mark_selected_ids_{u32,i32,i64,u64}` precedent.
- **Chain-heavy optimization**: if benchmarks show `chain` dominates, a
  shared-memory cache of `L_buf` rows can be added — but only after 3B.5
  proves parity, and only if the microbench warrants it.
- **Arg0 / Arg1 indices**: `fanout` and `fanin` only need to know whether
  a given `qx` or `qy` appears as some arg0 / arg1. A precomputed bitmap
  or sorted-index view could drop those checks from linear to O(1), at the
  cost of an additional setup pass. Microbench first.
