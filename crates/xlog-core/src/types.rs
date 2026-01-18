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
            ScalarType::Symbol => DataType::UInt32, // Symbols are interned u32
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
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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
        assert_eq!(ScalarType::Symbol.to_arrow_type(), DataType::UInt32);

        // Test roundtrip conversion (except Symbol which maps to UInt32)
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

        // Test unsupported Arrow types return None
        assert_eq!(ScalarType::from_arrow_type(&DataType::Utf8), None);
        assert_eq!(ScalarType::from_arrow_type(&DataType::Date32), None);
    }
}
