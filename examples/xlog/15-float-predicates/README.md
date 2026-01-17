# Float Predicate Examples

These examples demonstrate XLOG's float predicate semantics for filtering
floating-point data with special values (NaN, Infinity, signed zero).

## Semantics

XLOG uses **hybrid semantics** for float comparisons:

| Operator | Semantics | Example |
|----------|-----------|---------|
| `=`, `!=` | IEEE 754 | `NaN = NaN` → FALSE |
| `<`, `<=`, `>`, `>=` | Total Ordering | `NaN > Inf` → TRUE |

### Total Ordering

```
-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN
```

This ordering:
- Places NaN at the extremes (negative NaN smallest, positive NaN largest)
- Distinguishes `-0.0` from `+0.0` (unlike IEEE equality)
- Matches Rust's `f64::total_cmp` behavior

## Examples

1. **01_nan_handling.xlog** - Filtering data with NaN (missing values)
2. **02_infinity_detection.xlog** - Detecting and handling infinite values
3. **03_signed_zero.xlog** - Distinguishing -0.0 from +0.0
4. **04_data_quality_pipeline.xlog** - Complete data cleaning workflow
5. **05_statistical_analysis.xlog** - Combining predicates with aggregations

## Running Examples

```bash
# Run an example
xlog run examples/xlog/15-float-predicates/01_nan_handling.xlog

# With specific GPU device
xlog run examples/xlog/15-float-predicates/01_nan_handling.xlog --device 0
```

## Common Patterns

### Filtering Out NaN
```xlog
// NaN != NaN under IEEE, so V = V is only true for non-NaN values
valid(Id, V) :- data(Id, V), V = V.
```

### Filtering Out Infinity
```xlog
// Bound values to exclude infinity
finite(Id, V) :- data(Id, V), V > -1000000.0, V < 1000000.0.
```

### Detecting Special Values
```xlog
// Values greater than any normal bound are Inf or NaN
special(Id, V) :- data(Id, V), V > 1000000000.0.
special(Id, V) :- data(Id, V), V < -1000000000.0.
```
