use std::collections::HashMap;
use xlog_core::RelId;
use xlog_cuda::JoinIndexV2;

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

impl JoinIndexCache {
    pub(crate) fn new(max_bytes: u64) -> Self {
        Self {
            entries: HashMap::new(),
            clock: 0,
            total_bytes: 0,
            max_bytes,
        }
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
