"""Public Python bindings for XLOG's canonical symbol registry."""

import pytest

pyxlog = pytest.importorskip("pyxlog")


def test_symbol_batches_round_trip_through_the_canonical_registry() -> None:
    symbols = ["authority:maran", "case:milk-meat", "authority:maran", ""]

    symbol_ids = pyxlog.intern_symbols(symbols)

    assert len(symbol_ids) == len(symbols)
    assert symbol_ids[0] == symbol_ids[2]
    assert symbol_ids[0] != symbol_ids[1]
    assert pyxlog.resolve_symbols(symbol_ids) == symbols


def test_resolve_symbols_rejects_unknown_identifiers() -> None:
    with pytest.raises(
        ValueError,
        match=r"^unknown symbol ID 4294967295 at index 1$",
    ):
        pyxlog.resolve_symbols([pyxlog.intern_symbols(["known"])[0], 2**32 - 1])
