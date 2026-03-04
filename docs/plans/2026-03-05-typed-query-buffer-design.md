# Typed Query-Buffer Builder Design

**Date:** 2026-03-05
**Scope:** Fix batch upload paths in `batch_fact_membership` and `batch_tagged_credit`
**Deferred:** F32/F64 download support, full typed Python value API

---

## Problem

`batch_fact_membership` (line 4429) and `batch_tagged_credit` (line 4506) in
`crates/pyxlog/src/lib.rs` blindly cast `i64` input values to `u32` when uploading
facts to GPU. This silently corrupts data for any non-U32/Symbol column type.

**Affected types:** I32, I64, U64, Bool (wrong byte width or bit pattern).
**Accidentally correct:** U32, Symbol (both 4-byte u32 encoding).
**Rejected:** F32, F64 (not supported in batch APIs; matches download-side behavior).

## Solution

A private helper `pack_i64_columns_typed` dispatches on `schema.column_type(col_idx)`
to produce correctly-encoded byte columns, uploaded via `create_buffer_from_slices`.

### Helper

```rust
fn pack_i64_columns_typed(
    relation: &str,
    facts: &[Vec<i64>],
    schema: &Schema,
) -> PyResult<Vec<Vec<u8>>>
```

Includes defensive arity validation.

### Per-type encoding

| ScalarType | Width | Encoding | Range check |
|---|---|---|---|
| U32 | 4 | `u32::try_from(val)` LE | `[0, u32::MAX]` |
| I32 | 4 | `i32::try_from(val)` LE | `[i32::MIN, i32::MAX]` |
| U64 | 8 | `u64::try_from(val)` LE | `[0, i64::MAX]` (i64 input limits range) |
| I64 | 8 | `val.to_le_bytes()` | always valid |
| Bool | 1 | `0u8` or `1u8` | `val == 0 || val == 1` |
| Symbol | 4 | `u32::try_from(val)` LE | `[0, u32::MAX]` (symbol ID) |
| F32 | — | **reject** | error: "F32 not supported in batch APIs" |
| F64 | — | **reject** | error: "F64 not supported in batch APIs" |

### Error format

`"Relation '{relation}' column {col_idx} ({type:?}): value {val} out of range"`

### Changes to batch functions

Both `batch_fact_membership` and `batch_tagged_credit` replace manual
`Vec<Vec<u32>>` transpose + `create_buffer_from_u32_columns` with:

```rust
let col_bytes = pack_i64_columns_typed(relation, &facts, schema)?;
let col_slices: Vec<&[u8]> = col_bytes.iter().map(|c| c.as_slice()).collect();
let query_buf = self.provider.create_buffer_from_slices(&col_slices, schema.clone())?;
```

### U64 limitation

With `i64` Python input, U64 columns can only represent `[0, i64::MAX]`
(not the full `[0, u64::MAX]` range). This is documented but not fixed in this scope.

## Testing

### Positive tests
- Typed relation with `.pred` declared I32/I64/U64/Bool/Symbol columns
- `batch_fact_membership` returns correct membership results for each type
- `batch_tagged_credit` returns correct credit for typed relations

### Negative tests
- U32 column with value > `u32::MAX` → error
- U32/U64/Symbol column with negative value → error
- Bool column with value not in `{0, 1}` → error
- F32/F64 column → explicit "not supported" error

### Regression
- Existing ILP reliability (20/20), GA (50-seed), and performance suites remain green
