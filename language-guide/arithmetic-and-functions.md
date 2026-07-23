# Arithmetic and functions

Compute values with `is`, the arithmetic operators, and five built-in functions — then package reusable logic into user-defined functions.

Rules join and filter data, but real programs also need to **compute** — an area from a
width and height, a score from a weight, a running transform. XLOG does this with the
`is` expression, a small set of operators and built-in functions, and user-defined
functions when you want to name and reuse a computation.

## Binding a result with `is`

The `is` expression evaluates an arithmetic expression and binds the result to a
variable:

```xlog
area(W, H, Z) :- rect(W, H), Z is W * H.
```

Read `Z is W * H` as "let `Z` be `W * H`". The variable on the left must be **fresh** —
unbound at that point in the body. The engine evaluates the right-hand side using
values already bound (here `W` and `H`, supplied by `rect`), then binds the fresh `Z`.

You know it worked when the computed value shows up bound in the results: with a
`rect(3, 4)` fact, running this program derives `area(3, 4, 12)`.

<Warning>
`is` binds; it does not test. The left side must be a new variable. To *compare* an
already-bound value against a computed one, use a comparison operator (`==`, `<`, `>=`,
…) instead — those are covered in [Facts and rules](/language-guide/facts-and-rules).
Every variable read on the right of `is` must already be bound earlier in the body, or
the program is rejected at compile time.
</Warning>

## Operators and precedence

Five binary operators are available: `+`, `-`, `*`, `/`, and `%` (remainder). They
follow ordinary arithmetic precedence — `*`, `/`, and `%` bind tighter than `+` and `-`
— so:

```xlog
result(X, Y, Z, R) :- v(X, Y, Z), R is X + Y * Z.
```

computes `Y * Z` first, then adds `X`. Use parentheses to override the default grouping:

```xlog
result(X, Y, Z, R) :- v(X, Y, Z), R is (X + Y) * Z.
```

### Operands must share a type

A binary operator requires both operands to have the **same** type. You cannot add an
`i64` to an `f64` directly, or a `u32` to an `i32` — there is no implicit promotion.
When operand types differ, convert one explicitly with `cast` (below).

## The five built-in functions

XLOG provides exactly five built-in functions. There are no others:

| Function | Meaning |
|---|---|
| `abs(X)` | Absolute value of `X` |
| `min(X, Y)` | The smaller of `X` and `Y` |
| `max(X, Y)` | The larger of `X` and `Y` |
| `pow(X, Y)` | `X` raised to the power `Y` — **always returns `f64`** |
| `cast(X, type)` | Convert `X` to another scalar type |

`pow` always yields an `f64`, whatever its operands are, because exponentiation is
inherently a floating-point operation. `cast` takes a type name as its second argument
and is how you bridge the same-type rule for operators:

```xlog
scaled(X, Y) :- v(X), Y is pow(X, 2).
promoted(X, Z) :- ints(X), Z is cast(X, f64).
```

## Numeric edge cases

Arithmetic on real hardware has boundaries, and XLOG's are defined rather than
undefined:

- **Division by zero.** Integer division by zero yields `INT64_MAX` (the largest signed
  64-bit value). Float division by zero yields `NaN` or `Inf` per IEEE-754.
- **Integer overflow** wraps around (two's-complement), rather than trapping.

None of these raise an error mid-evaluation — the engine keeps running and produces the
defined sentinel. That means a divisor column containing a stray `0`, or an overflowing
product, silently seeds these values into your results. The idiomatic guard is to
**filter the bad inputs out** in the body before computing:

```xlog
ratio(X, Y, R) :- pair(X, Y), Y != 0, R is X / Y.
```

<Note>
Float literals must be written with **both** an integer and a fractional part: `1.0`,
not `1.`; `0.5`, not `.5`. Scientific notation such as `1e9` is not part of the literal
grammar either. Write the digits out on both sides of the decimal point.
</Note>

## User-defined functions

When a computation recurs, name it. A function declaration is `func`, a name, a
parameter list, a return type, and a body:

```xlog
func abs_diff(X: f64, Y: f64) -> f64 = if X < Y then Y - X else X - Y.
```

Parameters may carry `: type` annotations, and the return type follows `->`. The body
above is a **conditional**: `if cond then A else B` chooses between two arithmetic
expressions based on a comparison. Bodies can also be plain arithmetic:

```xlog
func square(X: f64) -> f64 = X * X.
```

Add the `private` modifier to keep a function local to its module — it will not be
exported when another file does `use`:

```xlog
private func helper(X: i64) -> i64 = X + 1.
```

### Recursion with a base case

A function may call itself, as long as it has a base case that terminates the
recursion:

```xlog
func countdown(N: i64) -> i64 = if N <= 0 then 0 else countdown(N - 1).
```

The `if` guard reaches the base case (`N <= 0`), so the recursion bottoms out rather
than descending forever.

### Calling a function

Call a function anywhere an arithmetic expression is allowed. To bind the result to a
fresh variable, use `is`. To test the result against a value you already have, use a
comparison — remember that `=` is an equality test, not an assignment, so both sides
must already be bound:

```xlog
gap(A, B, D) :- span(A, B), D is abs_diff(A, B).
matched(A, B) :- span(A, B), square(A) = B.
```

In the first rule, `is` binds the fresh variable `D`. In the second, `A` and `B` are
already bound by `span`, so `square(A) = B` keeps only the rows where `B` equals the
square of `A`.

<Card title="Aggregation" icon="sigma" href="/language-guide/aggregation">
  Collapse many rows into one — count, sum, min, max, and logsumexp in a rule head.
</Card>
