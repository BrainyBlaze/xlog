//! Metadata for RIR nodes (cardinality, memory estimates, skew)

use xlog_core::Schema;

/// Hint for physical layout selection
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LayoutHint {
    /// Standard cuDF table (baseline)
    #[default]
    CudfTable,
    /// HISA-style indexed storage for recursion
    HisaIndexed,
    /// VFLog-style columnar for bandwidth workloads
    VflogColumnar,
}

/// Signature of data skew for join optimization
#[derive(Debug, Clone)]
pub struct SkewSignature {
    /// Top-k hot keys
    pub hot_keys: Vec<u64>,
    /// Shannon entropy of key distribution
    pub entropy: f64,
}

impl SkewSignature {
    /// Check if data is considered skewed (entropy below threshold)
    pub fn is_skewed(&self) -> bool {
        self.entropy < 3.0 // bits
    }
}

/// Metadata attached to each RIR node
#[derive(Debug, Clone)]
pub struct RirMeta {
    /// Schema of output relation
    pub schema: Schema,
    /// Estimated row count range (min, max)
    pub est_rows: (u64, u64),
    /// Estimated memory bytes range (min, max)
    pub est_bytes: (u64, u64),
    /// Optional skew signature
    pub skew: Option<SkewSignature>,
    /// Whether this node produces deterministic output
    pub deterministic: bool,
    /// Layout hint for physical storage
    pub layout_hint: LayoutHint,
}

impl Default for RirMeta {
    fn default() -> Self {
        Self {
            schema: Schema::new(vec![]),
            est_rows: (0, 0),
            est_bytes: (0, 0),
            skew: None,
            deterministic: true,
            layout_hint: LayoutHint::default(),
        }
    }
}

impl RirMeta {
    /// Create metadata with schema
    pub fn with_schema(schema: Schema) -> Self {
        Self {
            schema,
            ..Default::default()
        }
    }

    /// Set estimated rows
    pub fn with_rows(mut self, min: u64, max: u64) -> Self {
        self.est_rows = (min, max);
        self.est_bytes = (
            min * self.schema.row_size_bytes() as u64,
            max * self.schema.row_size_bytes() as u64,
        );
        self
    }

    /// Set layout hint
    pub fn with_layout(mut self, hint: LayoutHint) -> Self {
        self.layout_hint = hint;
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use xlog_core::ScalarType;

    #[test]
    fn test_rir_meta_default() {
        let meta = RirMeta::default();
        assert_eq!(meta.est_rows, (0, 0));
        assert!(meta.deterministic);
    }

    #[test]
    fn test_layout_hint_default() {
        let hint = LayoutHint::default();
        assert_eq!(hint, LayoutHint::CudfTable);
    }

    #[test]
    fn test_skew_signature() {
        let sig = SkewSignature {
            hot_keys: vec![42, 100],
            entropy: 2.5,
        };
        assert!(sig.is_skewed());
    }

    #[test]
    fn test_meta_with_rows() {
        let schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U32),
        ]);
        let meta = RirMeta::with_schema(schema).with_rows(100, 200);
        assert_eq!(meta.est_rows, (100, 200));
        assert_eq!(meta.est_bytes, (800, 1600)); // 8 bytes per row
    }
}
