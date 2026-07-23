# Pragmas

Set compiler and engine behavior from inside a program with #pragma directives — and let CLI flags override them when you need to.

A pragma lets a program carry its own engine settings. Instead of remembering to
pass the right command-line flags every time you run a program, you write the
choice once, inside the file, and it travels with the code.

This is useful when a program only works correctly with a specific setup — for
example, one that needs approximate (sampling-based) probability inference, or a
particular way of reasoning about uncertainty. The requirement lives next to the
code that depends on it.

## How to write one

A pragma is a directive of the form `#pragma key = value`. Put it at the top of
your program. The setting applies whenever that program is compiled and run:

```xlog
#pragma prob_engine = mc.
#pragma prob_samples = 20000.
```

Here the program declares two things: use the `mc` probability engine (Monte
Carlo — it estimates probabilities by random sampling instead of computing them
exactly), and draw 20000 samples when it does.

**How you know it worked:** the `mc` engine estimates probabilities by sampling,
so by design its results depend on the random seed and become more stable as
`prob_samples` grows. The exact engine (`exact_ddnnf`) is deterministic and
ignores the seed. So if changing `#pragma prob_seed` changes a reported
probability, the `mc` engine is the one in effect.

## CLI flags win over pragmas

The pragma is the program's built-in default. When you pass the matching flag on
the command line, the **CLI flag overrides the pragma**.

So the in-program directive sets what the program wants, and the invocation gets
the final say. This lets you override a program's baked-in choice for one run
without editing the file.

## The ten pragmas

| Pragma | Values | Default |
|---|---|---|
| `prob_engine` | `exact_ddnnf` or `mc` | `exact_ddnnf` |
| `prob_cache` | `on` or `off` | none |
| `epistemic_mode` | `g91` or `faeel` | `faeel` |
| `prob_samples` | `<int>` | `10000` |
| `prob_seed` | `<int>` | `0` |
| `prob_confidence` | `<float>` | `0.95` |
| `prob_method` | `rejection` or `evidence_clamping` | none |
| `prob_max_nonmonotone_iterations` | `<int>` (must be `> 0`) | `1024` |
| `max_recursion_depth` | `<int>` | `1000` |
| `magic_sets` | `auto`, `on`, or `off` | `auto` semantics |

A few of the values are short names worth spelling out:

- **`prob_engine`** picks how probabilities are computed. `exact_ddnnf` (the
  default) computes exact probabilities. `mc` is Monte Carlo — it estimates them
  by random sampling, which is approximate.
- **`epistemic_mode`** picks the semantics for epistemic reasoning — reasoning
  about what the program treats as known versus merely possible. `faeel` is the
  default; `g91` selects an alternative rule for what counts as "known" (the
  classic Gelfond-1991 semantics). See [Epistemic reasoning](/epistemic/overview)
  for the difference.

<Note>
The probabilistic pragmas (`prob_samples`, `prob_seed`, `prob_confidence`,
`prob_method`, `prob_max_nonmonotone_iterations`) shape Monte Carlo inference and take
effect when `prob_engine = mc`. `prob_max_nonmonotone_iterations` is validated at parse
time and must be strictly greater than zero.
</Note>

## Magic sets

The `magic_sets` pragma controls a query-rewriting optimization. Use it when a
recursive query has some arguments already bound to specific values, and you want
the engine to compute only the facts that query can actually reach — rather than
deriving the entire relation and filtering afterward.

When a recursive query has bound arguments, magic sets rewrites the recursion to
push those bindings inward. The engine then derives only the reachable facts.

Setting `on` requests the rewrite and `off` disables it.

The default, `auto`, applies the rewrite only when the compiler can **prove the
rewritten program is equivalent** to the original. Where it cannot establish that
equivalence, `auto` declines and evaluates the program unchanged.

This makes `auto` safe to leave on. It optimizes what it can prove and never risks
changing your program's meaning.
