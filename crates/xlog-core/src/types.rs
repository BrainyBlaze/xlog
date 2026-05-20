//! Core types for XLOG schemas and data

/// Supported scalar types in XLOG relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 64-bit integer.
    U64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 64-bit integer.
    I64,
    /// 32-bit IEEE 754 floating point.
    F32,
    /// 64-bit IEEE 754 floating point.
    F64,
    /// Boolean (1 byte on GPU).
    Bool,
    /// Dictionary-encoded string (stored as interned `u32` ID).
    Symbol,
}

impl ScalarType {
    /// Returns the size in bytes of this scalar type
    pub fn size_bytes(&self) -> usize {
        match self {
            ScalarType::U32 | ScalarType::I32 | ScalarType::F32 | ScalarType::Symbol => 4,
            ScalarType::U64 | ScalarType::I64 | ScalarType::F64 => 8,
            ScalarType::Bool => 1,
        }
    }

    /// Returns the kernel type code for this scalar type
    pub fn to_code(&self) -> u8 {
        match self {
            ScalarType::U32 => 0,
            ScalarType::U64 => 1,
            ScalarType::I32 => 2,
            ScalarType::I64 => 3,
            ScalarType::F32 => 4,
            ScalarType::F64 => 5,
            ScalarType::Bool => 6,
            ScalarType::Symbol => 7,
        }
    }

    /// Returns the scalar type for a kernel type code
    pub fn from_code(code: u8) -> Option<Self> {
        match code {
            0 => Some(ScalarType::U32),
            1 => Some(ScalarType::U64),
            2 => Some(ScalarType::I32),
            3 => Some(ScalarType::I64),
            4 => Some(ScalarType::F32),
            5 => Some(ScalarType::F64),
            6 => Some(ScalarType::Bool),
            7 => Some(ScalarType::Symbol),
            _ => None,
        }
    }

    /// Returns true if this is a numeric type
    pub fn is_numeric(&self) -> bool {
        !matches!(self, ScalarType::Bool | ScalarType::Symbol)
    }

    /// Returns true if `other` is DLPack-compatible with `self`.
    ///
    /// Two scalar types are DLPack-compatible when they have the same byte width
    /// and differ only in signedness. This allows importing PyTorch signed integer
    /// tensors (int32/int64) into unsigned schema columns (u32/u64) and vice versa,
    /// since the bit patterns are identical on GPU and xlog kernels operate on raw
    /// column buffers without signedness-dependent semantics. Symbol schemas may
    /// import physical 32-bit tensor IDs while preserving their logical schema type.
    ///
    /// Float and bool types require exact match.
    pub fn dlpack_compatible(&self, other: ScalarType) -> bool {
        if *self == other {
            return true;
        }
        matches!(
            (*self, other),
            (ScalarType::U32, ScalarType::I32)
                | (ScalarType::I32, ScalarType::U32)
                | (ScalarType::U64, ScalarType::I64)
                | (ScalarType::I64, ScalarType::U64)
                | (ScalarType::Symbol, ScalarType::U32)
                | (ScalarType::Symbol, ScalarType::I32)
        )
    }

    /// Convert to Arrow DataType
    pub fn to_arrow_type(&self) -> arrow::datatypes::DataType {
        use arrow::datatypes::DataType;
        match self {
            ScalarType::Bool => DataType::Boolean,
            ScalarType::U32 => DataType::UInt32,
            ScalarType::I32 => DataType::Int32,
            ScalarType::U64 => DataType::UInt64,
            ScalarType::I64 => DataType::Int64,
            ScalarType::F32 => DataType::Float32,
            ScalarType::F64 => DataType::Float64,
            ScalarType::Symbol => {
                DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8))
            }
        }
    }

    /// Create from Arrow DataType
    pub fn from_arrow_type(dt: &arrow::datatypes::DataType) -> Option<Self> {
        use arrow::datatypes::DataType;
        match dt {
            DataType::Boolean => Some(ScalarType::Bool),
            DataType::UInt32 => Some(ScalarType::U32),
            DataType::Int32 => Some(ScalarType::I32),
            DataType::UInt64 => Some(ScalarType::U64),
            DataType::Int64 => Some(ScalarType::I64),
            DataType::Float32 => Some(ScalarType::F32),
            DataType::Float64 => Some(ScalarType::F64),
            DataType::Dictionary(key, value)
                if matches!(key.as_ref(), DataType::UInt32)
                    && matches!(value.as_ref(), DataType::Utf8) =>
            {
                Some(ScalarType::Symbol)
            }
            _ => None,
        }
    }
}

/// Schema describing a relation's columns
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Column names and their types
    pub columns: Vec<(String, ScalarType)>,
    /// Indices of columns that form the key (for dedup/indexing)
    pub key_columns: Vec<usize>,
    /// Per-column sort labels used by consumers that need typed output metadata.
    sort_labels: Vec<String>,
}

impl Schema {
    /// Create a new schema with all columns as keys
    pub fn new(columns: Vec<(String, ScalarType)>) -> Self {
        let key_columns = (0..columns.len()).collect();
        let sort_labels = default_sort_labels(&columns);
        Self {
            columns,
            key_columns,
            sort_labels,
        }
    }

    /// Return a copy of this schema with explicit per-column sort labels.
    pub fn with_sort_labels(
        mut self,
        sort_labels: Vec<String>,
    ) -> std::result::Result<Self, String> {
        if sort_labels.len() != self.columns.len() {
            return Err(format!(
                "sort label arity mismatch: expected {}, got {}",
                self.columns.len(),
                sort_labels.len()
            ));
        }
        self.sort_labels = normalize_sort_labels(&self.columns, sort_labels);
        Ok(self)
    }

    /// Number of columns
    pub fn arity(&self) -> usize {
        self.columns.len()
    }

    /// Total size of one row in bytes
    pub fn row_size_bytes(&self) -> usize {
        self.columns.iter().map(|(_, ty)| ty.size_bytes()).sum()
    }

    /// Get column type by index
    pub fn column_type(&self, index: usize) -> Option<ScalarType> {
        self.columns.get(index).map(|(_, ty)| *ty)
    }

    /// Get column index by name
    pub fn column_index(&self, name: &str) -> Option<usize> {
        self.columns.iter().position(|(n, _)| n == name)
    }

    /// Return per-column sort labels in schema column order.
    pub fn sort_labels(&self) -> &[String] {
        &self.sort_labels
    }

    /// Get the sort label for a column by index.
    pub fn column_sort_label(&self, index: usize) -> Option<&str> {
        self.sort_labels.get(index).map(|s| s.as_str())
    }

    /// Return true when every schema column has a non-empty sort label.
    pub fn has_authoritative_sort_labels(&self) -> bool {
        self.sort_labels.len() == self.columns.len()
            && self
                .sort_labels
                .iter()
                .all(|label| !label.trim().is_empty())
    }
}

fn default_sort_labels(columns: &[(String, ScalarType)]) -> Vec<String> {
    columns
        .iter()
        .enumerate()
        .map(|(idx, (name, _))| fallback_sort_label(name, idx))
        .collect()
}

fn normalize_sort_labels(
    columns: &[(String, ScalarType)],
    sort_labels: Vec<String>,
) -> Vec<String> {
    sort_labels
        .into_iter()
        .enumerate()
        .map(|(idx, label)| {
            if label.trim().is_empty() {
                fallback_sort_label(&columns[idx].0, idx)
            } else {
                label
            }
        })
        .collect()
}

fn fallback_sort_label(column_name: &str, index: usize) -> String {
    if column_name.trim().is_empty() {
        format!("col{}", index)
    } else {
        column_name.to_string()
    }
}

/// Unique identifier for a relation (assigned during compilation).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct RelId(
    /// Numeric relation ID.
    pub u32,
);

/// Aggregation operations supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    /// Count the number of rows in each group.
    Count,
    /// Sum of values in the aggregation column.
    Sum,
    /// Minimum value in the aggregation column.
    Min,
    /// Maximum value in the aggregation column.
    Max,
    /// Log-sum-exp aggregation (numerically stable log of summed exponentials).
    LogSumExp,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_type_size() {
        assert_eq!(ScalarType::U32.size_bytes(), 4);
        assert_eq!(ScalarType::U64.size_bytes(), 8);
        assert_eq!(ScalarType::Bool.size_bytes(), 1);
    }

    #[test]
    fn test_scalar_type_code_roundtrip() {
        for ty in [
            ScalarType::U32,
            ScalarType::U64,
            ScalarType::I32,
            ScalarType::I64,
            ScalarType::F32,
            ScalarType::F64,
            ScalarType::Bool,
            ScalarType::Symbol,
        ] {
            let code = ty.to_code();
            let back = ScalarType::from_code(code).unwrap();
            assert_eq!(ty, back);
        }
    }

    #[test]
    fn test_schema_total_row_size() {
        let mut schema = Schema::new(vec![
            ("a".to_string(), ScalarType::U32),
            ("b".to_string(), ScalarType::U64),
        ]);
        schema.key_columns = vec![0];
        assert_eq!(schema.row_size_bytes(), 12);
    }

    #[test]
    fn test_schema_arity() {
        let schema = Schema::new(vec![
            ("x".to_string(), ScalarType::U32),
            ("y".to_string(), ScalarType::U32),
            ("z".to_string(), ScalarType::U32),
        ]);
        assert_eq!(schema.arity(), 3);
    }

    #[test]
    fn test_schema_sort_labels_default_from_column_names() {
        let schema = Schema::new(vec![
            ("pred".to_string(), ScalarType::I64),
            ("arg0".to_string(), ScalarType::I64),
            ("arg1".to_string(), ScalarType::I64),
        ]);
        assert_eq!(schema.sort_labels(), ["pred", "arg0", "arg1"]);
        assert_eq!(schema.column_sort_label(1), Some("arg0"));
        assert!(schema.has_authoritative_sort_labels());
    }

    #[test]
    fn test_schema_sort_label_arity_is_checked() {
        let schema = Schema::new(vec![("c0".to_string(), ScalarType::U32)]);
        let err = schema
            .with_sort_labels(vec!["a".to_string(), "b".to_string()])
            .unwrap_err();
        assert!(err.contains("sort label arity mismatch"));
    }

    #[test]
    fn test_arrow_type_roundtrip() {
        use arrow::datatypes::DataType;

        // Test all scalar types convert to expected Arrow types
        assert_eq!(ScalarType::Bool.to_arrow_type(), DataType::Boolean);
        assert_eq!(ScalarType::U32.to_arrow_type(), DataType::UInt32);
        assert_eq!(ScalarType::I32.to_arrow_type(), DataType::Int32);
        assert_eq!(ScalarType::U64.to_arrow_type(), DataType::UInt64);
        assert_eq!(ScalarType::I64.to_arrow_type(), DataType::Int64);
        assert_eq!(ScalarType::F32.to_arrow_type(), DataType::Float32);
        assert_eq!(ScalarType::F64.to_arrow_type(), DataType::Float64);
        assert_eq!(
            ScalarType::Symbol.to_arrow_type(),
            DataType::Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8))
        );

        // Test roundtrip conversion
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::Bool.to_arrow_type()),
            Some(ScalarType::Bool)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::U32.to_arrow_type()),
            Some(ScalarType::U32)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::I32.to_arrow_type()),
            Some(ScalarType::I32)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::U64.to_arrow_type()),
            Some(ScalarType::U64)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::I64.to_arrow_type()),
            Some(ScalarType::I64)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::F32.to_arrow_type()),
            Some(ScalarType::F32)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::F64.to_arrow_type()),
            Some(ScalarType::F64)
        );
        assert_eq!(
            ScalarType::from_arrow_type(&ScalarType::Symbol.to_arrow_type()),
            Some(ScalarType::Symbol)
        );

        // Test unsupported Arrow types return None
        assert_eq!(ScalarType::from_arrow_type(&DataType::Utf8), None);
        assert_eq!(ScalarType::from_arrow_type(&DataType::Date32), None);
    }

    #[test]
    fn test_dlpack_compatible() {
        // Same type is always compatible
        assert!(ScalarType::U32.dlpack_compatible(ScalarType::U32));
        assert!(ScalarType::I64.dlpack_compatible(ScalarType::I64));
        assert!(ScalarType::F32.dlpack_compatible(ScalarType::F32));

        // Signed ↔ unsigned at same width is compatible
        assert!(ScalarType::U32.dlpack_compatible(ScalarType::I32));
        assert!(ScalarType::I32.dlpack_compatible(ScalarType::U32));
        assert!(ScalarType::U64.dlpack_compatible(ScalarType::I64));
        assert!(ScalarType::I64.dlpack_compatible(ScalarType::U64));

        // Different widths are NOT compatible
        assert!(!ScalarType::U32.dlpack_compatible(ScalarType::U64));
        assert!(!ScalarType::I32.dlpack_compatible(ScalarType::I64));

        // Float ↔ int is NOT compatible
        assert!(!ScalarType::F32.dlpack_compatible(ScalarType::I32));
        assert!(!ScalarType::F64.dlpack_compatible(ScalarType::I64));

        // Bool requires exact match. Symbol columns preserve logical type in
        // the schema, but import from physical u32 tensors for GPU-resident IDs.
        assert!(!ScalarType::Bool.dlpack_compatible(ScalarType::U32));
        assert!(ScalarType::Symbol.dlpack_compatible(ScalarType::U32));
        assert!(ScalarType::Symbol.dlpack_compatible(ScalarType::I32));
        assert!(!ScalarType::U32.dlpack_compatible(ScalarType::Symbol));
    }
}
