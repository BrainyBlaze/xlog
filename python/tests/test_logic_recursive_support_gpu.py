import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


def _i64(values):
    return torch.tensor(values, device="cuda", dtype=torch.int64)


def _query_columns(query):
    return [torch.from_dlpack(t).cpu().tolist() for t in query.tensors]


def test_logic_session_recursive_support_4_reports_exact_rows():
    source = """
pred committed_fact(i64, i64, i64).
pred first_body_literal(i64, i64, i64).
pred second_body_literal(i64, i64, i64).
pred third_body_literal(i64, i64, i64).
pred fourth_body_literal(i64, i64, i64).
pred usable(i64, i64, i64).
pred support_4(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

usable(P, A0, A1) :- committed_fact(P, A0, A1).
usable(H, A0, A2) :- support_4(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, MiddleArgument, ThirdBodyPredicate, MiddleArgument, A2, FourthBodyPredicate, A0, MiddleArgument).

support_4(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, MiddleArgument, ThirdBodyPredicate, MiddleArgument, A2, FourthBodyPredicate, A0, MiddleArgument) :-
    first_body_literal(R, H, FirstBodyPredicate),
    second_body_literal(R, H, SecondBodyPredicate),
    third_body_literal(R, H, ThirdBodyPredicate),
    fourth_body_literal(R, H, FourthBodyPredicate),
    usable(FirstBodyPredicate, A0, A1),
    usable(SecondBodyPredicate, A1, MiddleArgument),
    usable(ThirdBodyPredicate, MiddleArgument, A2),
    usable(FourthBodyPredicate, A0, MiddleArgument).

?- support_4(H, A0, A2, R, FirstBodyPredicate, X0, X1, SecondBodyPredicate, X2, X3, ThirdBodyPredicate, X4, X5, FourthBodyPredicate, X6, X7).
?- usable(P, X, Y).
"""

    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=512)
    session = program.session()

    # Rule 1 has a valid 4-body witness and derives usable(300, 1, 4).
    # Rule 2 reuses that derived head in body position 0, but has no valid
    # witness because the final diamond edge does not reach MiddleArgument = 5.
    # The correct support_4 output therefore contains exactly one row.
    session.put_relation(
        "first_body_literal",
        [_i64([1, 2]), _i64([300, 400]), _i64([10, 300])],
    )
    session.put_relation(
        "second_body_literal",
        [_i64([1, 2]), _i64([300, 400]), _i64([20, 40])],
    )
    session.put_relation(
        "third_body_literal",
        [_i64([1, 2]), _i64([300, 400]), _i64([30, 50])],
    )
    session.put_relation(
        "fourth_body_literal",
        [_i64([1, 2]), _i64([300, 400]), _i64([60, 60])],
    )
    session.put_relation(
        "committed_fact",
        [
            _i64([10, 20, 30, 40, 50, 60]),
            _i64([1, 2, 3, 4, 5, 1]),
            _i64([2, 3, 4, 5, 6, 3]),
        ],
    )

    result = session.evaluate()

    support = result.queries[0]
    usable = result.queries[1]

    assert support.num_rows == 1
    assert len(support.tensors) == 16
    assert _query_columns(support) == [
        [300],
        [1],
        [4],
        [1],
        [10],
        [1],
        [2],
        [20],
        [2],
        [3],
        [30],
        [3],
        [4],
        [60],
        [1],
        [3],
    ]

    assert usable.num_rows == 7
    assert _query_columns(usable) == [
        [10, 20, 30, 40, 50, 60, 300],
        [1, 2, 3, 4, 5, 1, 1],
        [2, 3, 4, 5, 6, 3, 4],
    ]


def test_logic_session_recursive_support_3_chain_rows_and_arity():
    source = """
pred committed_fact(i64, i64, i64).
pred first_body_literal(i64, i64, i64).
pred second_body_literal(i64, i64, i64).
pred third_body_literal(i64, i64, i64).
pred usable(i64, i64, i64).
pred support_3(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

usable(P, A0, A1) :- committed_fact(P, A0, A1).
usable(H, A0, A2) :- support_3(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, MiddleArgument, ThirdBodyPredicate, MiddleArgument, A2).

support_3(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, MiddleArgument, ThirdBodyPredicate, MiddleArgument, A2) :-
    first_body_literal(R, H, FirstBodyPredicate),
    second_body_literal(R, H, SecondBodyPredicate),
    third_body_literal(R, H, ThirdBodyPredicate),
    usable(FirstBodyPredicate, A0, A1),
    usable(SecondBodyPredicate, A1, MiddleArgument),
    usable(ThirdBodyPredicate, MiddleArgument, A2).

?- support_3(H, A0, A2, R, FirstBodyPredicate, X0, X1, SecondBodyPredicate, X2, X3, ThirdBodyPredicate, X4, X5).
?- usable(P, X, Y).
"""

    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=512)
    session = program.session()

    session.put_relation(
        "first_body_literal",
        [_i64([1, 2]), _i64([100, 200]), _i64([10, 100])],
    )
    session.put_relation(
        "second_body_literal",
        [_i64([1, 2]), _i64([100, 200]), _i64([20, 40])],
    )
    session.put_relation(
        "third_body_literal",
        [_i64([1, 2]), _i64([100, 200]), _i64([30, 50])],
    )
    session.put_relation(
        "committed_fact",
        [
            _i64([10, 20, 30, 40, 50]),
            _i64([1, 2, 3, 4, 5]),
            _i64([2, 3, 4, 5, 6]),
        ],
    )

    result = session.evaluate()

    support = result.queries[0]
    usable = result.queries[1]

    assert support.num_rows == 2
    assert len(support.tensors) == 13
    assert _query_columns(support) == [
        [100, 200],
        [1, 1],
        [4, 6],
        [1, 2],
        [10, 100],
        [1, 1],
        [2, 4],
        [20, 40],
        [2, 4],
        [3, 5],
        [30, 50],
        [3, 5],
        [4, 6],
    ]

    assert usable.num_rows == 7
    assert _query_columns(usable) == [
        [10, 20, 30, 40, 50, 100, 200],
        [1, 2, 3, 4, 5, 1, 1],
        [2, 3, 4, 5, 6, 4, 6],
    ]


def test_logic_session_recursive_mixed_arity_and_session_reuse():
    source = """
pred committed_fact(i64, i64, i64).
pred first_body_literal(i64, i64, i64).
pred second_body_literal(i64, i64, i64).
pred third_body_literal(i64, i64, i64).
pred fourth_body_literal(i64, i64, i64).
pred one_body_rule(i64).
pred two_body_rule(i64).
pred three_body_rule(i64).
pred four_body_rule(i64).
pred usable(i64, i64, i64).
pred support_1(i64, i64, i64, i64, i64, i64, i64).
pred support_2(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).
pred support_3(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).
pred support_4(i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64, i64).

usable(P, A0, A1) :- committed_fact(P, A0, A1).
usable(H, A0, A1) :- support_1(H, A0, A1, R, FirstBodyPredicate, A0, A1).
usable(H, A0, A2) :- support_2(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2).
usable(H, A0, A3) :- support_3(H, A0, A3, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2, ThirdBodyPredicate, A2, A3).
usable(H, A0, A3) :- support_4(H, A0, A3, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2, ThirdBodyPredicate, A2, A3, FourthBodyPredicate, A0, A2).

support_1(H, A0, A1, R, FirstBodyPredicate, A0, A1) :-
    one_body_rule(R),
    first_body_literal(R, H, FirstBodyPredicate),
    usable(FirstBodyPredicate, A0, A1).

support_2(H, A0, A2, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2) :-
    two_body_rule(R),
    first_body_literal(R, H, FirstBodyPredicate),
    second_body_literal(R, H, SecondBodyPredicate),
    usable(FirstBodyPredicate, A0, A1),
    usable(SecondBodyPredicate, A1, A2).

support_3(H, A0, A3, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2, ThirdBodyPredicate, A2, A3) :-
    three_body_rule(R),
    first_body_literal(R, H, FirstBodyPredicate),
    second_body_literal(R, H, SecondBodyPredicate),
    third_body_literal(R, H, ThirdBodyPredicate),
    usable(FirstBodyPredicate, A0, A1),
    usable(SecondBodyPredicate, A1, A2),
    usable(ThirdBodyPredicate, A2, A3).

support_4(H, A0, A3, R, FirstBodyPredicate, A0, A1, SecondBodyPredicate, A1, A2, ThirdBodyPredicate, A2, A3, FourthBodyPredicate, A0, A2) :-
    four_body_rule(R),
    first_body_literal(R, H, FirstBodyPredicate),
    second_body_literal(R, H, SecondBodyPredicate),
    third_body_literal(R, H, ThirdBodyPredicate),
    fourth_body_literal(R, H, FourthBodyPredicate),
    usable(FirstBodyPredicate, A0, A1),
    usable(SecondBodyPredicate, A1, A2),
    usable(ThirdBodyPredicate, A2, A3),
    usable(FourthBodyPredicate, A0, A2).

?- support_1(H, A0, A1, R, FirstBodyPredicate, X0, X1).
?- support_2(H, A0, A2, R, FirstBodyPredicate, X0, X1, SecondBodyPredicate, X2, X3).
?- support_3(H, A0, A3, R, FirstBodyPredicate, X0, X1, SecondBodyPredicate, X2, X3, ThirdBodyPredicate, X4, X5).
?- support_4(H, A0, A3, R, FirstBodyPredicate, X0, X1, SecondBodyPredicate, X2, X3, ThirdBodyPredicate, X4, X5, FourthBodyPredicate, X6, X7).
?- usable(P, X, Y).
"""

    program = pyxlog.LogicProgram.compile(source, device=0, memory_mb=512)
    session = program.session()

    session.put_relation("one_body_rule", [_i64([11])])
    session.put_relation("two_body_rule", [_i64([12])])
    session.put_relation("three_body_rule", [_i64([13])])
    session.put_relation("four_body_rule", [_i64([14])])
    session.put_relation(
        "first_body_literal",
        [_i64([11, 12, 13, 14]), _i64([101, 102, 103, 104]), _i64([10, 101, 102, 103])],
    )
    session.put_relation(
        "second_body_literal",
        [_i64([12, 13, 14]), _i64([102, 103, 104]), _i64([20, 30, 50])],
    )
    session.put_relation(
        "third_body_literal",
        [_i64([13, 14]), _i64([103, 104]), _i64([40, 60])],
    )
    session.put_relation(
        "fourth_body_literal",
        [_i64([14]), _i64([104]), _i64([70])],
    )

    session.put_relation(
        "committed_fact",
        [
            _i64([10, 20, 30, 40, 50, 60, 70]),
            _i64([1, 2, 3, 4, 5, 6, 1]),
            _i64([2, 3, 4, 5, 6, 7, 6]),
        ],
    )
    first = session.evaluate().queries

    assert [q.relation_name for q in first] == [
        "__xlog_query_0",
        "__xlog_query_1",
        "__xlog_query_2",
        "__xlog_query_3",
        "__xlog_query_4",
    ]
    assert [len(q.tensors) for q in first] == [7, 10, 13, 16, 3]
    assert [q.num_rows for q in first] == [1, 1, 1, 1, 11]
    assert _query_columns(first[0]) == [[101], [1], [2], [11], [10], [1], [2]]
    assert _query_columns(first[1]) == [[102], [1], [3], [12], [101], [1], [2], [20], [2], [3]]
    assert _query_columns(first[2]) == [
        [103],
        [1],
        [5],
        [13],
        [102],
        [1],
        [3],
        [30],
        [3],
        [4],
        [40],
        [4],
        [5],
    ]
    assert _query_columns(first[3]) == [
        [104],
        [1],
        [7],
        [14],
        [103],
        [1],
        [5],
        [50],
        [5],
        [6],
        [60],
        [6],
        [7],
        [70],
        [1],
        [6],
    ]
    assert _query_columns(first[4]) == [
        [10, 20, 30, 40, 50, 60, 70, 101, 102, 103, 104],
        [1, 2, 3, 4, 5, 6, 1, 1, 1, 1, 1],
        [2, 3, 4, 5, 6, 7, 6, 2, 3, 5, 7],
    ]

    session.put_relation(
        "committed_fact",
        [
            _i64([10, 20]),
            _i64([9, 10]),
            _i64([10, 11]),
        ],
    )
    second = session.evaluate().queries

    assert [q.relation_name for q in second] == [
        "__xlog_query_0",
        "__xlog_query_1",
        "__xlog_query_2",
        "__xlog_query_3",
        "__xlog_query_4",
    ]
    assert [len(q.tensors) for q in second] == [7, 10, 13, 16, 3]
    assert [q.num_rows for q in second] == [1, 1, 0, 0, 4]
    assert _query_columns(second[0]) == [[101], [9], [10], [11], [10], [9], [10]]
    assert _query_columns(second[1]) == [[102], [9], [11], [12], [101], [9], [10], [20], [10], [11]]
    assert _query_columns(second[2]) == [[], [], [], [], [], [], [], [], [], [], [], [], []]
    assert _query_columns(second[3]) == [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []]
    assert _query_columns(second[4]) == [
        [10, 20, 101, 102],
        [9, 10, 9, 9],
        [10, 11, 10, 11],
    ]
