use std::collections::HashMap;
use xlog_core::RelId;
use xlog_cuda::{CudaBuffer, JoinIndexV2};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct JoinIndexKey {
    pub(crate) rel: RelId,
    pub(crate) version: u64,
    pub(crate) key_cols: Vec<usize>,
}

struct CachedJoinIndex {
    index: JoinIndexV2,
    bytes: u64,
    last_used: u64,
}

pub(crate) struct JoinIndexCache {
    entries: HashMap<JoinIndexKey, CachedJoinIndex>,
    clock: u64,
    total_bytes: u64,
    pub(crate) max_bytes: u64,
}

/// Estimate the GPU memory footprint of a join index built on `right` with `right_keys`.
///
/// Returns u64::MAX if keys are empty or column types are missing (signals "don't build").
pub(crate) fn estimate_join_index_bytes(right: &CudaBuffer, right_keys: &[usize]) -> u64 {
    if right_keys.is_empty() {
        return u64::MAX;
    }

    let mut key_bytes_per_row: u64 = 0;
    for &k in right_keys {
        let Some(ty) = right.schema().column_type(k) else {
            return u64::MAX;
        };
        key_bytes_per_row = key_bytes_per_row.saturating_add(ty.size_bytes() as u64);
    }

    let num_rows = right.num_rows();
    let packed_bytes = num_rows.saturating_mul(key_bytes_per_row);
    let target = num_rows.saturating_mul(2).max(1024);
    let num_buckets = target.next_power_of_two();

    // Stored index bytes: packed keys + (counts+offsets) + (entry row ids + entry hashes)
    packed_bytes
        .saturating_add(num_buckets.saturating_mul(8))
        .saturating_add(num_rows.saturating_mul(12))
}

impl JoinIndexCache {
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            total_bytes: 0,
            max_bytes,
        }
    }

    /// Decide whether to build a new join index for a relation.
    ///
    /// Heuristic: require higher "heat" for larger indexes, and avoid building under
    /// memory pressure. Always skip if the estimated index cannot fit in the cache budget.
    pub(crate) fn should_build(
        &self,
        est_index_bytes: u64,
        build_heat: f32,
        remaining_device_bytes: u64,
        device_budget_bytes: u64,
    ) -> bool {
        let heat_threshold = if self.max_bytes > 0 && est_index_bytes > self.max_bytes / 2 {
            0.6
        } else {
            0.3
        };
        let has_room =
            remaining_device_bytes >= est_index_bytes.saturating_add(device_budget_bytes / 10);

        build_heat >= heat_threshold && est_index_bytes <= self.max_bytes && has_room
    }

    pub(crate) fn clear(&mut self) {
        self.entries.clear();
        self.clock = 0;
        self.total_bytes = 0;
    }

    pub(crate) fn get(&mut self, key: &JoinIndexKey) -> Option<&JoinIndexV2> {
        let entry = self.entries.get_mut(key)?;
        self.clock = self.clock.saturating_add(1);
        entry.last_used = self.clock;
        Some(&entry.index)
    }

    pub(crate) fn insert(&mut self, key: JoinIndexKey, index: JoinIndexV2) {
        let bytes = index.estimated_bytes();
        if bytes > self.max_bytes {
            return;
        }

        self.evict_until_fits(bytes);

        self.clock = self.clock.saturating_add(1);
        let last_used = self.clock;

        if let Some(prev) = self.entries.remove(&key) {
            self.total_bytes = self.total_bytes.saturating_sub(prev.bytes);
        }

        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.entries.insert(
            key,
            CachedJoinIndex {
                index,
                bytes,
                last_used,
            },
        );
    }

    pub(crate) fn invalidate_rel(&mut self, rel: RelId) {
        let keys: Vec<JoinIndexKey> = self
            .entries
            .keys()
            .filter(|k| k.rel == rel)
            .cloned()
            .collect();
        for key in keys {
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            }
        }
    }

    pub(crate) fn evict_until_fits(&mut self, additional_bytes: u64) {
        while !self.entries.is_empty()
            && self.total_bytes.saturating_add(additional_bytes) > self.max_bytes
        {
            let mut oldest_key: Option<JoinIndexKey> = None;
            let mut oldest_clock = u64::MAX;

            for (k, v) in &self.entries {
                if v.last_used < oldest_clock {
                    oldest_clock = v.last_used;
                    oldest_key = Some(k.clone());
                }
            }

            let Some(key) = oldest_key else {
                break;
            };
            if let Some(entry) = self.entries.remove(&key) {
                self.total_bytes = self.total_bytes.saturating_sub(entry.bytes);
            } else {
                break;
            }
        }
    }
}
