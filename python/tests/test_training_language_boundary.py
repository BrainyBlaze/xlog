"""Contract for the Python-only neuro-symbolic training document."""

from pyxlog.ilp.neurosymbolic import _desugar_source, _read_only_source


def test_training_declarations_lower_without_reparsing_native_xlog() -> None:
    source = """
        edge(1, 2).
        pred edge(u64, u64).
        pred reach(u64, u64).
        learnable(W_native) :: native_reach(X, Y) :- edge(X, Z), edge(Z, Y).
        trainable_rule(candidate, weight=0.25) :: reach(X, Y) :- edge(X, Y).
        train(reach, binary_cross_entropy).
    """

    lowered, rules, head, objective = _desugar_source(source)

    assert head == "reach"
    assert objective == "binary_cross_entropy"
    assert [rule.id for rule in rules] == ["candidate"]
    assert "trainable_rule" not in lowered
    assert "train(reach" not in lowered
    assert "learnable(W_native)" in lowered
    assert "edge(1, 2)." in lowered
    assert "nn(nsr_w_candidate" in lowered
    assert "nsr_guard_candidate" in lowered


def test_read_only_program_removes_only_python_training_declarations() -> None:
    source = """
        learnable(W_native) :: native_reach(X, Y) :- edge(X, Z), edge(Z, Y).
        trainable_rule(candidate) :: reach(X, Y) :- edge(X, Y).
        train(reach, binary_cross_entropy).
        query(reach(1, 2)).
    """

    native_source = _read_only_source(source)

    assert "trainable_rule" not in native_source
    assert "train(reach" not in native_source
    assert "learnable(W_native)" in native_source
    assert "query(reach(1, 2))." in native_source
