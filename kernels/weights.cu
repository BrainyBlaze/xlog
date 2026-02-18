// GPU weight table builders for XLOG.

#include <cstdint>
#include <cmath>

__device__ __forceinline__ void weights_trap() {
    asm("trap;");
}

__device__ __forceinline__ double ln_prob(double p) {
    if (p == 0.0) {
        return -INFINITY;
    }
    return log(p);
}

extern "C" __global__ void weights_fill_leaf(
    const uint32_t* __restrict__ leaf_var,
    const double* __restrict__ leaf_prob,
    uint32_t leaf_count,
    uint32_t var_cap,
    double* __restrict__ log_true,
    double* __restrict__ log_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t leaf = tid; leaf < leaf_count; leaf += blockDim.x * gridDim.x) {
        uint32_t var = leaf_var[leaf];
        if (var == 0u) {
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        double p = leaf_prob[leaf];
        if (!(p >= 0.0 && p <= 1.0)) {
            weights_trap();
        }
        double pf = 1.0 - p;
        log_true[var] = ln_prob(p);
        log_false[var] = ln_prob(pf);
    }
}

extern "C" __global__ void weights_fill_choice(
    const uint32_t* __restrict__ choice_var,
    const double* __restrict__ choice_true,
    const double* __restrict__ choice_false,
    uint32_t choice_count,
    uint32_t var_cap,
    double* __restrict__ log_true,
    double* __restrict__ log_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < choice_count; idx += blockDim.x * gridDim.x) {
        uint32_t var = choice_var[idx];
        if (var == 0u) {
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        double pt = choice_true[idx];
        double pf = choice_false[idx];
        if (!(pt >= 0.0 && pt <= 1.0)) {
            weights_trap();
        }
        if (!(pf >= 0.0 && pf <= 1.0)) {
            weights_trap();
        }
        log_true[var] = ln_prob(pt);
        log_false[var] = ln_prob(pf);
    }
}

extern "C" __global__ void weights_set_evidence_from_nodes(
    const uint32_t* __restrict__ node_var,
    const uint32_t* __restrict__ evidence_nodes,
    const uint8_t* __restrict__ evidence_vals,
    uint32_t evidence_count,
    uint32_t var_cap,
    uint8_t* __restrict__ evidence_out
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    for (uint32_t idx = 0; idx < evidence_count; idx++) {
        uint32_t node = evidence_nodes[idx];
        uint32_t var = node_var[node];
        if (var == 0u || var > var_cap) {
            weights_trap();
        }
        uint8_t v = evidence_vals[idx];
        if (v != 1u && v != 2u) {
            weights_trap();
        }
        uint8_t prev = evidence_out[var];
        if (prev == 0u) {
            evidence_out[var] = v;
        } else if (prev != v) {
            weights_trap();
        }
    }
}

extern "C" __global__ void weights_apply_evidence(
    const uint8_t* __restrict__ evidence_by_var,
    uint32_t var_cap,
    double* __restrict__ log_true,
    double* __restrict__ log_false
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t var = tid; var <= var_cap; var += blockDim.x * gridDim.x) {
        if (var == 0u) {
            continue;
        }
        uint8_t v = evidence_by_var[var];
        if (v == 1u) {
            log_false[var] = -INFINITY;
        } else if (v == 2u) {
            log_true[var] = -INFINITY;
        } else if (v != 0u) {
            weights_trap();
        }
    }
}

extern "C" __global__ void weights_map_nodes_to_vars(
    const uint32_t* __restrict__ node_var,
    const uint32_t* __restrict__ node_ids,
    uint32_t count,
    uint32_t var_cap,
    uint32_t* __restrict__ out_vars
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t node = node_ids[idx];
        uint32_t var = node_var[node];
        if (var == 0u || var > var_cap) {
            weights_trap();
        }
        out_vars[idx] = var;
    }
}

extern "C" __global__ void weights_force_var_false(
    uint32_t var,
    double* __restrict__ log_false,
    double* __restrict__ restore
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    restore[0] = log_false[var];
    log_false[var] = -INFINITY;
}

extern "C" __global__ void weights_restore_var_false(
    uint32_t var,
    double* __restrict__ log_false,
    const double* __restrict__ restore
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    log_false[var] = restore[0];
}

extern "C" __global__ void weights_force_var_true(
    uint32_t var,
    double* __restrict__ log_true,
    double* __restrict__ restore
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    restore[0] = log_true[var];
    log_true[var] = -INFINITY;
}

extern "C" __global__ void weights_restore_var_true(
    uint32_t var,
    double* __restrict__ log_true,
    const double* __restrict__ restore
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    log_true[var] = restore[0];
}

// Broadcast one cached slot's weight vectors into batch-major buffers.
// out_{true,false}_batch layout: [batch][var_stride].
extern "C" __global__ void weights_copy_slot_to_batch(
    const uint32_t* __restrict__ active_slot,
    uint32_t var_cap,
    const double* __restrict__ log_true,
    const double* __restrict__ log_false,
    double* __restrict__ out_true_batch,
    double* __restrict__ out_false_batch,
    uint32_t var_stride,
    uint32_t batch_size
) {
    uint32_t slot = active_slot[0];
    uint32_t vars_per_query = var_cap + 1u;
    size_t src_base = static_cast<size_t>(slot) * static_cast<size_t>(vars_per_query);

    uint64_t total = static_cast<uint64_t>(vars_per_query) * static_cast<uint64_t>(batch_size);
    uint64_t tid = static_cast<uint64_t>(blockIdx.x) * static_cast<uint64_t>(blockDim.x)
        + static_cast<uint64_t>(threadIdx.x);
    uint64_t step = static_cast<uint64_t>(blockDim.x) * static_cast<uint64_t>(gridDim.x);

    for (uint64_t idx = tid; idx < total; idx += step) {
        uint32_t q = static_cast<uint32_t>(idx / vars_per_query);
        uint32_t var = static_cast<uint32_t>(idx % vars_per_query);
        size_t dst_base = static_cast<size_t>(q) * static_cast<size_t>(var_stride);
        out_true_batch[dst_base + var] = log_true[src_base + var];
        out_false_batch[dst_base + var] = log_false[src_base + var];
    }
}

extern "C" __global__ void weights_apply_query_vars(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    double* __restrict__ log_false,
    double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            saved[idx] = 0.0;
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        saved[idx] = log_false[var];
        log_false[var] = -INFINITY;
    }
}

extern "C" __global__ void weights_restore_query_vars(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    double* __restrict__ log_false,
    const double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        log_false[var] = saved[idx];
    }
}

// Apply/restore one query var per batch row in strided weight tables.
// Layout: log_{true,false}_batch[query_idx * var_stride + var].
extern "C" __global__ void weights_apply_query_vars_false_batched(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    uint32_t var_stride,
    double* __restrict__ log_false_batch,
    double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            saved[idx] = 0.0;
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        size_t base = static_cast<size_t>(idx) * static_cast<size_t>(var_stride);
        saved[idx] = log_false_batch[base + var];
        log_false_batch[base + var] = -INFINITY;
    }
}

extern "C" __global__ void weights_restore_query_vars_false_batched(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    uint32_t var_stride,
    double* __restrict__ log_false_batch,
    const double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        size_t base = static_cast<size_t>(idx) * static_cast<size_t>(var_stride);
        log_false_batch[base + var] = saved[idx];
    }
}

extern "C" __global__ void weights_apply_query_vars_true_batched(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    uint32_t var_stride,
    double* __restrict__ log_true_batch,
    double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            saved[idx] = 0.0;
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        size_t base = static_cast<size_t>(idx) * static_cast<size_t>(var_stride);
        saved[idx] = log_true_batch[base + var];
        log_true_batch[base + var] = -INFINITY;
    }
}

extern "C" __global__ void weights_restore_query_vars_true_batched(
    const uint32_t* __restrict__ query_vars,
    uint32_t count,
    uint32_t var_cap,
    uint32_t var_stride,
    double* __restrict__ log_true_batch,
    const double* __restrict__ saved
) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (uint32_t idx = tid; idx < count; idx += blockDim.x * gridDim.x) {
        uint32_t var = query_vars[idx];
        if (var == 0u) {
            continue;
        }
        if (var > var_cap) {
            weights_trap();
        }
        size_t base = static_cast<size_t>(idx) * static_cast<size_t>(var_stride);
        log_true_batch[base + var] = saved[idx];
    }
}
