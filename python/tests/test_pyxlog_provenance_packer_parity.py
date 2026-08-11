import pytest

pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


SOURCE = """
pred parent(u32, u32).
pred ancestor(u32, u32).

parent(1, 2).
ancestor(Child, Parent) :- parent(Child, Parent).
?- ancestor(Child, Parent).
"""

RULE_KEYS = [
    "rule_id",
    "head",
    "source_kind",
    "source_span",
    "generation_trace_hash",
    "support_relation_ids",
    "counterexample_relation_ids",
]

PROOF_KEYS = [
    "query_id",
    "query",
    "answer_relation",
    "rule_ids",
    "source_facts",
    "rejected_alternatives",
]


def test_provenance_and_proof_packing_matches_across_python_program_objects():
    compiled_logic = pyxlog.LogicProgram.compile(SOURCE, device=0, memory_mb=256)
    relation_session = compiled_logic.session()
    compiled_probabilistic = pyxlog.Program.compile(
        SOURCE,
        device=0,
        memory_mb=256,
        prob_engine="exact_ddnnf",
    )

    objects = [compiled_logic, relation_session, compiled_probabilistic]
    provenance = [program.rule_provenance() for program in objects]
    proofs = [program.proof_traces() for program in objects]

    assert provenance[0]
    assert proofs[0]
    assert provenance[1:] == [provenance[0], provenance[0]]
    assert proofs[1:] == [proofs[0], proofs[0]]

    for records in provenance:
        assert all(list(record) == RULE_KEYS for record in records)
        assert [record["head"] for record in records] == [
            "parent(1, 2)",
            "ancestor(Child, Parent)",
        ]
        assert [record["source_kind"] for record in records] == ["source", "source"]
        assert [record["source_span"] for record in records] == [
            "rule_index:0",
            "rule_index:1",
        ]
        assert [record["support_relation_ids"] for record in records] == [
            [],
            ["parent"],
        ]
        assert [record["counterexample_relation_ids"] for record in records] == [
            [],
            [],
        ]
        for index, record in enumerate(records):
            assert isinstance(record["generation_trace_hash"], str)
            assert record["rule_id"] == (
                f"rule:source:{index}:{record['generation_trace_hash']}"
            )

    for records in proofs:
        assert all(list(record) == PROOF_KEYS for record in records)
        assert records == [
            {
                "query_id": "query:source:0:ancestor(Child, Parent)",
                "query": "ancestor(Child, Parent)",
                "answer_relation": "__xlog_query_0",
                "rule_ids": [provenance[0][1]["rule_id"]],
                "source_facts": ["parent(1, 2)."],
                "rejected_alternatives": [],
            }
        ]
