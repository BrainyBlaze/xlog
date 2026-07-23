# Modules

Split an XLOG program across files, import predicates and functions by path, and control what each module exposes with private and domain.

Every XLOG file is a **module**. Larger programs are assembled by importing modules into
one another, so you can keep predicates, functions, and type aliases where they belong and
pull in only what a given file needs.

## Importing

Bring another module into scope with `use`, giving its path. The `/` character is the path
separator between path segments:

```xlog
use utils/math.
```

To import only specific names, list them in braces after `::`:

```xlog
use utils/math::{abs_diff, clamp}.
```

The selective form imports exactly the named predicates or functions and nothing else,
which keeps a module's dependencies explicit.

## Visibility

Predicates and functions are **public by default** — once a module is imported, its
declarations are visible to the importer. Mark a declaration `private` to hide it, so it
stays internal to its own module and never leaks across a `use`:

```xlog
private pred scratch(u32).

private func normalize(X: f64) -> f64 = X / total().
```

Both `pred` and `func` accept the `private` modifier. Use it for helpers that support a
module's public surface but are not meant to be part of it.

## Domain aliases

A `domain` declaration names a reusable type. Write `domain name : type.` to introduce an
alias you can then use anywhere a type is expected:

```xlog
domain node : u32.

pred edge(src: node, dst: node).
```

Here `node` stands for `u32`, so the intent of each column is visible at a glance and a
later change to the underlying type is made in one place.

<Card title="Pragmas" icon="sliders" href="/language-guide/pragmas">
  Set compiler and engine behavior from inside a program with `#pragma` directives.
</Card>
