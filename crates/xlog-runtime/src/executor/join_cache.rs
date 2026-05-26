use std::collections::HashMap;
use xlog_core::{RelId, ScalarType, Schema};
use xlog_cuda::{CudaBuffer, JoinIndexV2};

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct JoinIndexKey {
    pub(crate) rel: RelId,
    pub(crate) version: u64,
    pub(crate) key_cols: Vec<usize>,
    pub(crate) schema: JoinIndexSchemaSignature,
    pub(crate) device_ordinal: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct JoinIndexSchemaSignature {
    column_types: Vec<ScalarType>,
    row_size_bytes: usize,
}

impl JoinIndexSchemaSignature {
    fn from_schema(schema: &Schema) -> Self {
        Self {
            column_types: (0..schema.arity())
                .filter_map(|idx| schema.column_type(idx))
                .collect(),
            row_size_bytes: schema.row_size_bytes(),
        }
    }
}

impl JoinIndexKey {
    pub(crate) fn new(
        rel: RelId,
        version: u64,
        key_cols: Vec<usize>,
        schema: &Schema,
        device_ordinal: u32,
    ) -> Self {
        Self {
            rel,
            version,
            key_cols,
            schema: JoinIndexSchemaSignature::from_schema(schema),
            device_ordinal,
        }
    }
}

struct CachedJoinIndex {
    index: CachedJoinIndexPayload,
    bytes: u64,
    last_used: u64,
}

#[allow(clippy::large_enum_variant)]
enum CachedJoinIndexPayload {
    Ready(JoinIndexV2),
    #[cfg(test)]
    Placeholder,
}

/// Persistent join-index manager telemetry.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct JoinIndexCacheStats {
    /// Lookup attempts.
    pub lookups: u64,
    /// Successful index reuses.
    pub hits: u64,
    /// Lookup misses.
    pub misses: u64,
    /// Successful index builds inserted into the cache.
    pub builds: u64,
    /// LRU/budget evictions.
    pub evictions: u64,
    /// Entries invalidated because a relation changed.
    pub invalidations: u64,
    /// Stale entries rejected by provider validation.
    pub stale_rejections: u64,
    /// Background-build mode requests.
    pub background_build_requests: u64,
    /// Background-build mode completions.
    pub background_builds_completed: u64,
    /// Background builds whose indexed reuse was deferred until a later evaluation.
    pub background_builds_deferred: u64,
    /// Current retained index count.
    pub entries: usize,
    /// Current retained index bytes.
    pub total_bytes: u64,
}

pub(crate) struct JoinIndexCache {
    entries: HashMap<JoinIndexKey, CachedJoinIndex>,
    clock: u64,
    total_bytes: u64,
    pub(crate) max_bytes: u64,
    stats: JoinIndexCacheStats,
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
            stats: JoinIndexCacheStats::default(),
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
        let removed = self.entries.len() as u64;
        self.entries.clear();
        self.clock = 0;
        self.total_bytes = 0;
        self.stats.invalidations = self.stats.invalidations.saturating_add(removed);
    }

    pub(crate) fn get(&mut self, key: &JoinIndexKey) -> Option<&JoinIndexV2> {
        self.stats.lookups = self.stats.lookups.saturating_add(1);
        let Some(entry) = self.entries.get_mut(key) else {
            self.stats.misses = self.stats.misses.saturating_add(1);
            return None;
        };
        self.clock = self.clock.saturating_add(1);
        entry.last_used = self.clock;
        match &entry.index {
            CachedJoinIndexPayload::Ready(index) => {
                self.stats.hits = self.stats.hits.saturating_add(1);
                Some(index)
            }
            #[cfg(test)]
            CachedJoinIndexPayload::Placeholder => {
                self.stats.misses = self.stats.misses.saturating_add(1);
                None
            }
        }
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
                index: CachedJoinIndexPayload::Ready(index),
                bytes,
                last_used,
            },
        );
        self.stats.builds = self.stats.builds.saturating_add(1);
    }

    pub(crate) fn remove(&mut self, key: &JoinIndexKey) {
        if let Some(prev) = self.entries.remove(key) {
            self.total_bytes = self.total_bytes.saturating_sub(prev.bytes);
        }
    }

    pub(crate) fn remove_stale(&mut self, key: &JoinIndexKey) {
        let before = self.entries.len();
        self.remove(key);
        if self.entries.len() < before {
            self.stats.stale_rejections = self.stats.stale_rejections.saturating_add(1);
        }
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
                self.stats.invalidations = self.stats.invalidations.saturating_add(1);
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
                self.stats.evictions = self.stats.evictions.saturating_add(1);
            } else {
                break;
            }
        }
    }

    pub(crate) fn record_background_build_request(&mut self) {
        self.stats.background_build_requests =
            self.stats.background_build_requests.saturating_add(1);
    }

    pub(crate) fn record_background_build_complete(&mut self) {
        self.stats.background_builds_completed =
            self.stats.background_builds_completed.saturating_add(1);
    }

    pub(crate) fn record_background_build_deferred(&mut self) {
        self.stats.background_builds_deferred =
            self.stats.background_builds_deferred.saturating_add(1);
    }

    pub(crate) fn stats(&self) -> JoinIndexCacheStats {
        let mut stats = self.stats.clone();
        stats.entries = self.entries.len();
        stats.total_bytes = self.total_bytes;
        stats
    }

    #[cfg(test)]
    fn insert_test_entry(&mut self, key: JoinIndexKey, bytes: u64) {
        if bytes > self.max_bytes {
            return;
        }
        self.evict_until_fits(bytes);
        self.clock = self.clock.saturating_add(1);
        self.total_bytes = self.total_bytes.saturating_add(bytes);
        self.entries.insert(
            key,
            CachedJoinIndex {
                index: CachedJoinIndexPayload::Placeholder,
                bytes,
                last_used: self.clock,
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::{ScalarType, Schema};

    fn schema(cols: Vec<(&str, ScalarType)>) -> Schema {
        Schema::new(
            cols.into_iter()
                .map(|(name, ty)| (name.to_string(), ty))
                .collect(),
        )
    }

    #[test]
    fn persistent_key_includes_schema_generation_key_and_device() {
        let u32_schema = schema(vec![("k", ScalarType::U32)]);
        let u64_schema = schema(vec![("k", ScalarType::U64)]);

        let key = JoinIndexKey::new(RelId(7), 3, vec![0], &u32_schema, 0);
        assert_eq!(key.rel, RelId(7));
        assert_eq!(key.version, 3);
        assert_eq!(key.key_cols, vec![0]);
        assert_eq!(key.device_ordinal, 0);

        assert_ne!(
            key,
            JoinIndexKey::new(RelId(7), 4, vec![0], &u32_schema, 0),
            "generation/version must partition stale indexes"
        );
        assert_ne!(
            key,
            JoinIndexKey::new(RelId(7), 3, vec![0], &u64_schema, 0),
            "schema changes must partition indexes"
        );
        assert_ne!(
            key,
            JoinIndexKey::new(RelId(7), 3, vec![0], &u32_schema, 1),
            "device ordinal must partition indexes"
        );
    }

    #[test]
    fn persistent_cache_budget_evicts_lru_and_records_stats() {
        let schema = schema(vec![("k", ScalarType::U32)]);
        let key_a = JoinIndexKey::new(RelId(1), 1, vec![0], &schema, 0);
        let key_b = JoinIndexKey::new(RelId(2), 1, vec![0], &schema, 0);
        let mut cache = JoinIndexCache::new(100);

        cache.insert_test_entry(key_a, 60);
        cache.insert_test_entry(key_b, 60);

        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.total_bytes, 60);
        assert_eq!(stats.evictions, 1);
    }

    #[test]
    fn persistent_cache_invalidation_records_removed_entries() {
        let schema = schema(vec![("k", ScalarType::U32)]);
        let key = JoinIndexKey::new(RelId(1), 1, vec![0], &schema, 0);
        let mut cache = JoinIndexCache::new(100);

        cache.insert_test_entry(key, 32);
        cache.invalidate_rel(RelId(1));

        let stats = cache.stats();
        assert_eq!(stats.entries, 0);
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.invalidations, 1);
    }
}
