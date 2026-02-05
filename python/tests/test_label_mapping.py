import pytest

pyxlog = pytest.importorskip("pyxlog")


def test_label_resolution_uses_declared_labels():
    program = pyxlog.Program.compile(
        """
        nn(net, [X], Y, [heads, tails]) :: coin(X, Y).
        """
    )

    assert program.label_to_index("coin", "heads") == 0
    assert program.label_to_index("coin", "tails") == 1
    with pytest.raises(ValueError):
        program.label_to_index("coin", "edge")
