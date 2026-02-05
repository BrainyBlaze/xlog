import pytest

pyxlog = pytest.importorskip("pyxlog")


def test_registry_resolves_network_and_labels():
    program = pyxlog.Program.compile(
        """
        nn(net1, [X], Y, [heads, tails]) :: coin(X, Y).
        nn(net2, [X], Y, [0,1,2]) :: digit(X, Y).
        """
    )

    info = program.neural_predicate_info("coin")
    assert info["network"] == "net1"
    assert info["labels"] == ["heads", "tails"]

    info2 = program.neural_predicate_info("digit")
    assert info2["network"] == "net2"
    assert info2["labels"] == ["0", "1", "2"]
