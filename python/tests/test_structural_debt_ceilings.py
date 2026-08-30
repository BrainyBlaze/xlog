"""Ceilings on three structural-debt classes that no compiler lint can see.

Each class below is real debt recorded in the architecture-debt ledgers. None of
them can be driven to zero in one change, and none of them is detectable by
rustc or clippy: a `pub fn` in a library crate is "used by downstream" as far as
the compiler is concerned, a `Ok(None)` decline is ordinary control flow, and an
environment variable is a string.

So this file does the only thing that makes such a class stop growing: it
records where each count stood on a named date and fails when it rises. That is
a one-way ratchet, not an amnesty — lowering a ceiling is an ordinary pull
request, raising one has to be argued for in review.

ENGINEERING.md forbids deferring cleanup as technical debt and requires a scope
decision instead of a temporary substitute. A dated, enforced ceiling *is* that
scope decision, written where CI reads it. An undated prose exception list is
the temporary substitute the same rule rejects.

Ledgers: `docs/architecture-debt.md`,
`docs/xlog-architecture-debt-audit2-2026-08-24.md`,
`docs/xlog-architecture-debt-audit3-2026-08-28.md`.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Measured on main @ 404e9219, 2026-08-30. Raise only with a recorded reason.
UNREFERENCED_PUBLIC_FUNCTION_CEILING = 126
WCOJ_SILENT_DECLINE_CEILING = 185
DISTINCT_XLOG_ENV_NAME_CEILING = 115

_PUBLIC_FN = re.compile(r"\bpub fn ([a-z_][a-z_0-9]*)")
_IDENTIFIER = re.compile(r"\b[a-z_][a-z_0-9]*\b")
_XLOG_ENV_NAME = re.compile(r"\bXLOG_[A-Z0-9_]+")


def _rust_sources_under(*relative_globs: str) -> list[Path]:
    paths: list[Path] = []
    for pattern in relative_globs:
        paths.extend(sorted(ROOT.glob(pattern)))
    return [path for path in paths if path.is_file()]


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def test_unreferenced_public_functions_do_not_grow() -> None:
    """`pub fn` whose name appears exactly once in all workspace Rust source.

    One occurrence means the definition and no reference anywhere, including
    tests. The compiler cannot report these: `pub` in a library crate suppresses
    `dead_code` because the item is reachable from outside the crate in
    principle, whether or not anything reaches it in fact.

    Token-level, so it would miss a call assembled by a macro. Every candidate
    chased by hand so far has been genuinely unreferenced; treat a sudden drop
    with the same suspicion as a rise.
    """
    declared: set[str] = set()
    for path in _rust_sources_under("crates/*/src/**/*.rs"):
        declared.update(_PUBLIC_FN.findall(_read(path)))

    occurrences: dict[str, int] = {}
    for path in _rust_sources_under("crates/**/*.rs"):
        for identifier in _IDENTIFIER.findall(_read(path)):
            if identifier in declared:
                occurrences[identifier] = occurrences.get(identifier, 0) + 1

    unreferenced = sorted(
        name for name in declared if occurrences.get(name, 0) == 1
    )
    assert len(unreferenced) <= UNREFERENCED_PUBLIC_FUNCTION_CEILING, (
        f"unreferenced public functions rose to {len(unreferenced)}, ceiling is "
        f"{UNREFERENCED_PUBLIC_FUNCTION_CEILING}. A new public API needs a wired "
        f"consumer (ENGINEERING.md). Newly unreferenced or newly added: "
        f"{unreferenced}"
    )


def test_wcoj_structural_declines_do_not_grow() -> None:
    """`Ok(None)` occurrences in the WCOJ dispatcher.

    Occurrences, not lines: one site reads `Ok(None) => Ok(None)`, so a
    line count reports 184 where the real figure is 185. The ledgers carry
    the line count; this is the number that is enforced.

    A structural decline (gate off, shape mismatch, missing buffer) silently
    falls back to the binary-join path. `wcoj_decline_on_error` counts and logs
    the *error* declines and its own doc comment records that structural ones
    deliberately do not go through it. That is a design choice, not an
    oversight, so the count is capped rather than driven to zero.
    """
    dispatcher = ROOT / "crates/xlog-runtime/src/executor/wcoj_dispatch.rs"
    assert dispatcher.is_file(), f"{dispatcher} moved; update this ceiling test"

    declines = _read(dispatcher).count("Ok(None)")
    assert declines <= WCOJ_SILENT_DECLINE_CEILING, (
        f"silent WCOJ declines rose to {declines}, ceiling is "
        f"{WCOJ_SILENT_DECLINE_CEILING}. A new decline either routes through "
        f"wcoj_decline_on_error or lowers this ceiling elsewhere."
    )


def test_environment_variable_surface_does_not_grow() -> None:
    """Distinct `XLOG_*` names anywhere in the tree.

    Every knob is a branch that has to keep working and a combination somebody
    will eventually hit; the count is how many states one binary has. Reading
    them was centralised on `xlog_core::config_value`, which fixed *how* they
    are parsed and not *how many* exist.
    """
    names: set[str] = set()
    searched = (
        "crates/**/*.rs",
        "crates/**/*.cu",
        "crates/**/*.py",
        "scripts/**/*.py",
        "scripts/**/*.sh",
        "docs/**/*.mdx",
        "docs/**/*.md",
    )
    for path in _rust_sources_under(*searched):
        names.update(_XLOG_ENV_NAME.findall(_read(path)))

    makefile = ROOT / "Makefile"
    if makefile.is_file():
        names.update(_XLOG_ENV_NAME.findall(_read(makefile)))

    assert len(names) <= DISTINCT_XLOG_ENV_NAME_CEILING, (
        f"distinct XLOG_* names rose to {len(names)}, ceiling is "
        f"{DISTINCT_XLOG_ENV_NAME_CEILING}. ENGINEERING.md rejects "
        f"environment-variable escape hatches; a new knob needs an owner and a "
        f"documented default, or it replaces an existing one."
    )
