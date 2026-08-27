# Float Predicate Examples

These examples demonstrate XLOG's float predicate semantics for filtering
floating-point data with special values (NaN, Infinity, signed zero).

## Semantics

XLOG uses one total order for every float comparison:

| Operator | Semantics | Example |
|----------|-----------|---------|
| `=`, `!=` | Total-order identity | `-0.0 = 0.0` → FALSE |
| `<`, `<=`, `>`, `>=` | Total ordering | positive `NaN > Inf` → TRUE |

### Total Ordering

```
-NaN < -Inf < ... < -0.0 < +0.0 < ... < +Inf < +NaN
```

This ordering:
- Places NaN at the extremes (negative NaN smallest, positive NaN largest)
- Distinguishes `-0.0` from `+0.0`
- Distinguishes NaN sign, signaling state, and payload
- Matches Rust's `f64::total_cmp` behavior

Arithmetic-generated NaNs use one canonical positive quiet-NaN bit pattern.
They therefore compare equal to themselves. Do not use `V != V` to detect NaN.

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

### Preventing invalid arithmetic
```xlog
// Validate the producing operation instead of relying on NaN self-inequality.
valid(Id, V) :- input(Id, Num, Denom), Denom != 0.0, V is Num / Denom.
```

### Filtering Out Infinity
```xlog
// Bound values to exclude infinity
finite(Id, V) :- data(Id, V), V > -1000000.0, V < 1000000.0.
```

### Classifying invalid inputs
```xlog
// Preserve provenance when the invalid operation itself matters.
invalid_divisor(Id) :- input(Id, _, Denom), Denom = 0.0.
```
