//! Core types for XLOG schemas and data

/// Supported scalar types in XLOG relations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarType {
    U32,
    U64,
    I32,
    I64,
    F32,
    F64,
    Bool,
    /// Dictionary-encoded string
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

    /// Returns true if this is a numeric type
    pub fn is_numeric(&self) -> bool {
        !matches!(self, ScalarType::Bool | ScalarType::Symbol)
    }
}

/// Schema describing a relation's columns
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Schema {
    /// Column names and their types
    pub columns: Vec<(String, ScalarType)>,
    /// Indices of columns that form the key (for dedup/indexing)
    pub key_columns: Vec<usize>,
}

impl Schema {
    /// Create a new schema with all columns as keys
    pub fn new(columns: Vec<(String, ScalarType)>) -> Self {
        let key_columns = (0..columns.len()).collect();
        Self { columns, key_columns }
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
}

/// Unique identifier for a relation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RelId(pub u32);

/// Aggregation operations supported
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AggOp {
    Count,
    Sum,
    Min,
    Max,
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
    fn test_schema_total_row_size() {
        let schema = Schema {
            columns: vec![
                ("a".to_string(), ScalarType::U32),
                ("b".to_string(), ScalarType::U64),
            ],
            key_columns: vec![0],
        };
        assert_eq!(schema.row_size_bytes(), 12);
    }

    #[test]
    fn test_schema_arity() {
        let schema = Schema {
            columns: vec![
                ("x".to_string(), ScalarType::U32),
                ("y".to_string(), ScalarType::U32),
                ("z".to_string(), ScalarType::U32),
            ],
            key_columns: vec![0, 1, 2],
        };
        assert_eq!(schema.arity(), 3);
    }
}
