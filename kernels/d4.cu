// kernels/d4.cu
//
// GPU-native D4 compiler support kernels (Phase 1).
//
// Design constraints:
// - GPU-native data-plane: no device->host copies for CNF/circuit data
// - Deterministic: stable behavior and failure modes
// - Memory-bounded: fixed-capacity buffers; overflow => hard trap
//
// Primary spec:
// docs/design/2026-01-22-gpu-native-compilation-design.md (Sections 2, 2.5, 5.2.4)

#include <cstdint>
#include <cstddef>

// Fail-fast trap (matches kernels/sat.cu semantics).
__device__ __forceinline__ void d4_trap() {
    asm volatile("trap;");
}

// ---------------------------------------------------------------------------
// Literal helpers (DIMACS: +/- var_id, 1-based var ids; 0 unused)
// ---------------------------------------------------------------------------

__device__ __forceinline__ uint32_t d4_var(int32_t lit) {
    return (lit > 0) ? static_cast<uint32_t>(lit) : static_cast<uint32_t>(-lit);
}

// ---------------------------------------------------------------------------
// CNF invariant validation
// ---------------------------------------------------------------------------

// Validate basic CSR invariants for a device-resident CNF.
//
// This kernel is used to enforce "no silent success" semantics: invalid inputs
// must fail-fast on device without relying on host reads.
extern "C" __global__ void d4_validate_cnf(
    uint32_t var_cap,
    uint32_t clause_cap,
    uint32_t lit_cap,
    const uint32_t* __restrict__ num_vars,    // len=1
    const uint32_t* __restrict__ num_clauses, // len=1
    const uint32_t* __restrict__ num_lits,    // len=1
    const uint32_t* __restrict__ clause_offsets,
    const int32_t* __restrict__ literals
) {
    // Single-threaded: this is a debug/invariant check, not a hot kernel.
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }

    uint32_t nv = num_vars[0];
    uint32_t nc = num_clauses[0];
    uint32_t nl = num_lits[0];

    if (nv > var_cap || nc > clause_cap || nl > lit_cap) {
        d4_trap();
    }

    // Offsets must start at 0 and terminate at num_lits for the active clause range.
    if (nc > 0u && clause_offsets[0] != 0u) {
        d4_trap();
    }
    if (clause_offsets[nc] != nl) {
        d4_trap();
    }

    // Validate monotone offsets and literal bounds for active clauses.
    for (uint32_t c = 0; c < nc; c++) {
        uint32_t s = clause_offsets[c];
        uint32_t e = clause_offsets[c + 1u];
        if (s > e || e > nl) {
            d4_trap();
        }

        for (uint32_t i = s; i < e; i++) {
            int32_t lit = literals[i];
            if (lit == 0) {
                d4_trap();
            }
            uint32_t v = d4_var(lit);
            if (v == 0u || v > nv) {
                d4_trap();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Levelization helpers
// ---------------------------------------------------------------------------

// Compute per-level node counts from a device-resident `node_level[]` table.
//
// The output `level_counts[level]` is intended to be exclusive-scanned on device
// to form `level_offsets` for XGCF evaluation.
extern "C" __global__ void d4_levelize_counts(
    const uint32_t* __restrict__ node_level,
    uint32_t num_nodes,
    uint32_t num_levels,
    uint32_t* __restrict__ level_counts
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        uint32_t lvl = node_level[node];
        if (lvl >= num_levels) {
            d4_trap();
        }
        atomicAdd(&level_counts[lvl], 1u);
    }
}

// Emit `level_nodes` given `level_offsets` and per-level cursors.
//
// Notes:
// - `level_offsets` must be an exclusive scan of `level_counts` with length `num_levels + 1`.
// - `level_cursors` must be initialized to 0 before launch (len = num_levels).
// - Emission order within a level is unspecified; correctness does not depend on it.
extern "C" __global__ void d4_levelize_emit(
    const uint32_t* __restrict__ node_level,
    uint32_t num_nodes,
    uint32_t num_levels,
    const uint32_t* __restrict__ level_offsets, // len = num_levels + 1
    uint32_t* __restrict__ level_cursors,       // len = num_levels (must start at 0)
    uint32_t* __restrict__ level_nodes          // len = num_nodes
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t node = tid; node < num_nodes; node += blockDim.x * gridDim.x) {
        uint32_t lvl = node_level[node];
        if (lvl >= num_levels) {
            d4_trap();
        }
        uint32_t idx = atomicAdd(&level_cursors[lvl], 1u);
        uint32_t pos = level_offsets[lvl] + idx;
        if (pos >= level_offsets[lvl + 1u] || pos >= num_nodes) {
            d4_trap();
        }
        level_nodes[pos] = node;
    }
}

