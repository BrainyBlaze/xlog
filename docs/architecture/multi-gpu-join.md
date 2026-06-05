# Multi-GPU Join Architecture

> **Implementation status (current as of v0.9.2):** **Design only.** Multi-GPU memory management
> (`crates/xlog-cuda/src/multi_gpu_memory.rs`) is implemented, but distributed join
> execution and cross-device partitioning kernels are not shipped. Single-GPU joins
> (`hash_join_v2`) remain the production path. Multi-GPU partitioned evaluation is
> targeted for v0.10.0 (see `ROADMAP.md`).

## Overview

Distributed hash join across multiple GPUs using hash-based partitioning.

## Algorithm

### Phase 1: Partition Both Tables

```
For each row in left_table:
    partition_id = hash(key) % num_devices
    send row to device[partition_id]

For each row in right_table:
    partition_id = hash(key) % num_devices
    send row to device[partition_id]
```

### Phase 2: Local Joins

```
For each device d in parallel:
    result[d] = hash_join(left_partition[d], right_partition[d])
```

### Phase 3: Gather Results

```
final_result = concatenate(result[0], result[1], ..., result[n])
```

## Implementation Notes

1. **Partitioning Kernel**: Need GPU kernel to compute partition IDs and scatter
2. **Cross-Device Copy**: Use P2P if available, else copy through host
3. **Load Balancing**: Hash partitioning may be skewed; consider sampling for better distribution
4. **Memory Management**: Each device needs memory for:
   - Input partition
   - Hash table
   - Output buffer

## API Design

```rust
impl MultiGpuKernelProvider {
    pub fn hash_join_distributed(
        &self,
        left: &DistributedBuffer,
        right: &DistributedBuffer,
        left_keys: &[usize],
        right_keys: &[usize],
    ) -> Result<DistributedBuffer>;
}
```

## Future Work

- Implement `DistributedBuffer` type
- Add partitioning kernels
- Implement P2P copy optimization
- Add skew detection and handling
