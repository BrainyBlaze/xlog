import hashlib
import gc
import os
import struct

import pytest
import pyxlog
import torch


if not torch.cuda.is_available():
    if os.environ.get("XLOG_REQUIRE_CUDA") == "1":
        raise RuntimeError("XLOG_REQUIRE_CUDA=1 but PyTorch cannot access CUDA")
    pytest.skip("CUDA is unavailable", allow_module_level=True)


SOURCE = """
domain party: u32.
domain asset: u32.
pred transfer(giver: party, receiver: party, asset: asset, time: i64).
pred positional(u32, i64, symbol).
pred float_value(value: f32).
pred float64_value(value: f64).
pred symbol_value(value: symbol).
pred scalar_values(wide: u64, signed32: i32, signed64: i64, flag: bool).
pred alpha(value: i32).
pred zeta(value: i32).
"""


TRANSFER_ROLES = [
    {"name": "giver", "sort": "party", "type": "u32"},
    {"name": "receiver", "sort": "party", "type": "u32"},
    {"name": "asset", "sort": "asset", "type": "u32"},
    {"name": "time", "type": "i64"},
]


def _session(source: str = SOURCE):
    return pyxlog.LogicProgram.compile(source, device=0, memory_mb=256).session()


def _transfer_columns(rows=((10, 20, 7, 1_700_000_000), (10, 20, 8, 1_700_000_001))):
    columns = list(zip(*rows))
    return [
        torch.tensor(columns[0], device="cuda", dtype=torch.int32),
        torch.tensor(columns[1], device="cuda", dtype=torch.int32),
        torch.tensor(columns[2], device="cuda", dtype=torch.int32),
        torch.tensor(columns[3], device="cuda", dtype=torch.int64),
    ]


def _empty_transfer_columns():
    return [
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int32),
        torch.empty(0, device="cuda", dtype=torch.int64),
    ]


def _exported_rows(session, relation):
    columns = [torch.from_dlpack(column).cpu().tolist() for column in session.export_relation(relation)]
    return sorted(zip(*columns))


def _fact_sources(snapshot, values):
    fact = next(fact for fact in snapshot["facts"] if fact["tuple"] == list(values))
    return [record["source"] for record in fact["provenance"]]


def _record(**overrides):
    record = {
        "source": "extractor-output",
        "document": "document-42",
        "span": {"start": 18, "end": 41},
        "content_hash": "sha256:input",
        "kind": "assertion",
        "polarity": "positive",
    }
    record.update(overrides)
    return record


def _fact(values, *records):
    return {"tuple": list(values), "provenance": list(records or (_record(),))}


class _IteratesAs(list):
    def __init__(self, sized_values, iterated_values):
        super().__init__(sized_values)
        self._iterated_values = iterated_values

    def __iter__(self):
        return iter(self._iterated_values)


def _metadata_error():
    return pyxlog.RelationMetadataError


def _fact_identity(relation: str, typed_cells: list[tuple[int, bytes]]) -> str:
    payload = bytearray(b"xlog.fact.identity.v1\0")
    encoded_name = relation.encode("utf-8")
    payload += struct.pack("<I", len(encoded_name))
    payload += encoded_name
    payload += struct.pack("<I", len(typed_cells))
    for type_code, cell in typed_cells:
        payload += bytes([type_code])
        payload += struct.pack("<I", len(cell))
        payload += cell
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def test_native_ternary_and_quaternary_puts_bind_evidence_to_complete_tuples():
    session = _session()
    records = [
        _record(source="manual-review", document="review-9", span=None, kind="confirmation"),
        _record(),
        _record(),
    ]
    snapshot = session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(),
        roles=TRANSFER_ROLES,
        facts=[
            _fact((10, 20, 8, 1_700_000_001), _record(source="z-source")),
            _fact((10, 20, 7, 1_700_000_000), *records),
        ],
    )

    assert snapshot["relation"] == "transfer"
    assert snapshot["metadata_present"] is True
    assert snapshot["row_count"] == 2
    assert snapshot["roles"] == [
        TRANSFER_ROLES[0],
        TRANSFER_ROLES[1],
        TRANSFER_ROLES[2],
        {"name": "time", "sort": None, "type": "i64"},
    ]
    assert [fact["tuple"] for fact in snapshot["facts"]] == [
        [10, 20, 7, 1_700_000_000],
        [10, 20, 8, 1_700_000_001],
    ]
    assert len(snapshot["facts"][0]["provenance"]) == 2
    assert [record["source"] for record in snapshot["facts"][0]["provenance"]] == [
        "extractor-output",
        "manual-review",
    ]
    assert snapshot["facts"][0]["identity"] == _fact_identity(
        "transfer",
        [
            (0, struct.pack("<I", 10)),
            (0, struct.pack("<I", 20)),
            (0, struct.pack("<I", 7)),
            (3, struct.pack("<q", 1_700_000_000)),
        ],
    )
    assert snapshot["facts"][0]["cells"] == [
        {"type": "u32", "hex": "0a000000"},
        {"type": "u32", "hex": "14000000"},
        {"type": "u32", "hex": "07000000"},
        {"type": "i64", "hex": "00f1536500000000"},
    ]
    assert set(snapshot["facts"][0]["provenance"][0]) == {
        "source",
        "document",
        "span",
        "content_hash",
        "kind",
        "polarity",
    }

    ternary = session.put_relation_with_provenance(
        "positional",
        [
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([-2], device="cuda", dtype=torch.int64),
            torch.tensor([9], device="cuda", dtype=torch.int32),
        ],
        roles=[{"name": "entity"}, {"name": "time"}, {"name": "label"}],
        facts=[_fact((1, -2, 9))],
    )
    assert ternary["roles"] == [
        {"name": "entity", "sort": None, "type": "u32"},
        {"name": "time", "sort": None, "type": "i64"},
        {"name": "label", "sort": None, "type": "symbol"},
    ]


@pytest.mark.parametrize(
    "source, relation, columns, roles, fact_values",
    [
        (
            """
            domain party: u32.
            domain asset: u32.
            pred transfer(giver: party, receiver: party, asset: asset, time: i64).
            pred observed(giver: party, receiver: party, asset: asset, time: i64).
            observed(G, R, A, T) :- transfer(G, R, A, T).
            ?- observed(G, R, A, T).
            """,
            "transfer",
            lambda: _transfer_columns(rows=((10, 20, 7, 1_700_000_000),)),
            TRANSFER_ROLES,
            (10, 20, 7, 1_700_000_000),
        ),
        (
            """
            pred positional(u32, i64, symbol).
            pred observed(u32, i64, symbol).
            observed(Entity, Time, Label) :- positional(Entity, Time, Label).
            ?- observed(Entity, Time, Label).
            """,
            "positional",
            lambda: [
                torch.tensor([1], device="cuda", dtype=torch.int32),
                torch.tensor([-2], device="cuda", dtype=torch.int64),
                torch.tensor([9], device="cuda", dtype=torch.int32),
            ],
            [{"name": "entity"}, {"name": "time"}, {"name": "label"}],
            (1, -2, 9),
        ),
    ],
)
def test_metadata_bearing_high_arity_relations_evaluate_on_cuda(
    source, relation, columns, roles, fact_values
):
    session = _session(source)
    session.put_relation_with_provenance(
        relation,
        columns(),
        roles=roles,
        facts=[_fact(fact_values, _record(source="runtime-input"))],
    )
    session.reset_host_transfer_stats()

    result = session.evaluate()
    assert len(result.queries) == 1
    output = [torch.from_dlpack(tensor) for tensor in result.queries[0].tensors]
    assert all(tensor.device.type == "cuda" for tensor in output)
    assert [tensor.cpu().tolist() for tensor in output] == [
        [value] for value in fact_values
    ]
    assert session.host_transfer_stats()["dtoh_calls"] == 0


def test_duplicate_relation_rows_and_fact_entries_union_records_without_fabrication():
    session = _session()
    first = _record(source="first", document=None, span=None)
    second = _record(source="second", document=None, span=None)
    snapshot = session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4), (1, 2, 3, 4), (1, 2, 3, 5))),
        roles=TRANSFER_ROLES,
        facts=[
            _fact((1, 2, 3, 4), first, first),
            _fact((1, 2, 3, 4), second),
            _fact((1, 2, 3, 4), second),
        ],
    )

    assert snapshot["row_count"] == 3
    assert len(snapshot["facts"]) == 1
    assert snapshot["facts"][0]["tuple"] == [1, 2, 3, 4]
    assert [record["source"] for record in snapshot["facts"][0]["provenance"]] == [
        "first",
        "second",
    ]


@pytest.mark.parametrize(
    "roles, message",
    [
        (TRANSFER_ROLES[:-1], "expected 4 roles"),
        (TRANSFER_ROLES + [{"name": "extra"}], "expected 4 roles"),
        ([TRANSFER_ROLES[1], TRANSFER_ROLES[0], *TRANSFER_ROLES[2:]], "role 0"),
        ([TRANSFER_ROLES[0], TRANSFER_ROLES[0], *TRANSFER_ROLES[2:]], "duplicate"),
        ([{}, *TRANSFER_ROLES[1:]], "name"),
        ([{"name": ""}, *TRANSFER_ROLES[1:]], "non-empty"),
        ([{"name": "giver", "sort": "asset"}, *TRANSFER_ROLES[1:]], "sort"),
        ([{"name": "giver", "type": "i32"}, *TRANSFER_ROLES[1:]], "type"),
        ([{"name": "giver", "unexpected": 1}, *TRANSFER_ROLES[1:]], "unknown"),
    ],
)
def test_role_contract_rejects_invalid_named_roles_before_replacement(roles, message):
    session = _session()
    with pytest.raises(_metadata_error(), match=message):
        session.put_relation_with_provenance(
            "transfer", _transfer_columns(), roles=roles, facts=[]
        )
    assert session.relation("transfer").provenance()["row_count"] == 0


@pytest.mark.parametrize("iterated_count", [3, 0, 5])
def test_role_arity_uses_the_values_actually_iterated(iterated_count):
    session = _session()
    iterated_roles = [*TRANSFER_ROLES, {"name": "extra"}][:iterated_count]
    roles = _IteratesAs(TRANSFER_ROLES, iterated_roles)
    with pytest.raises(
        _metadata_error(), match=rf"expected 4 roles but received {iterated_count}"
    ):
        session.put_relation_with_provenance(
            "transfer", _transfer_columns(), roles=roles, facts=[]
        )
    assert session.relation("transfer").provenance()["row_count"] == 0


def test_positional_roles_are_stable_until_metadata_free_replacement_resets_them():
    session = _session()
    first_roles = [{"name": "entity"}, {"name": "time"}, {"name": "label"}]
    columns = lambda: [
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([2], device="cuda", dtype=torch.int64),
        torch.tensor([3], device="cuda", dtype=torch.int32),
    ]
    session.put_relation_with_provenance(
        "positional", columns(), roles=first_roles, facts=[_fact((1, 2, 3))]
    )

    changed_roles = [{"name": "subject"}, {"name": "time"}, {"name": "label"}]
    with pytest.raises(_metadata_error(), match="registered role"):
        session.put_relation_with_provenance(
            "positional", columns(), roles=changed_roles, facts=[]
        )
    assert session.relation("positional").provenance()["roles"][0]["name"] == "entity"

    session.put_relation("positional", columns())
    cleared = session.relation("positional").provenance()
    assert cleared["metadata_present"] is False
    assert cleared["roles"] == []
    assert cleared["facts"] == []
    replaced = session.put_relation_with_provenance(
        "positional", columns(), roles=changed_roles, facts=[]
    )
    assert replaced["roles"][0]["name"] == "subject"


@pytest.mark.parametrize(
    "fact, message",
    [
        ({"provenance": [_record()]}, "exactly one"),
        (
            {
                "tuple": [10, 20, 7, 1_700_000_000],
                "cells": [
                    {"type": "u32", "hex": "0a000000"},
                    {"type": "u32", "hex": "14000000"},
                    {"type": "u32", "hex": "07000000"},
                    {"type": "i64", "hex": "00f1536500000000"},
                ],
                "provenance": [_record()],
            },
            "exactly one",
        ),
        ({"tuple": [10, 20, 7], "provenance": [_record()]}, "arity"),
        ({"tuple": [True, 20, 7, 1_700_000_000], "provenance": [_record()]}, "bool"),
        ({"tuple": [-1, 20, 7, 1_700_000_000], "provenance": [_record()]}, "u32"),
        ({"tuple": [2**32, 20, 7, 1_700_000_000], "provenance": [_record()]}, "u32"),
        ({"tuple": [10, 20, 7, 2**63], "provenance": [_record()]}, "i64"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [_record()], "extra": 1}, "unknown"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [{"source": None}]}, "non-null"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [{"source": "a", "extra": 1}]}, "unknown"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [{"span": {"start": -1, "end": 2}}]}, "non-negative"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [{"span": {"start": 3, "end": 2}}]}, "start"),
        ({"tuple": [10, 20, 7, 1_700_000_000], "provenance": [{"span": {"start": 1, "end": 2, "extra": 3}}]}, "unknown"),
    ],
)
def test_fact_and_record_validation_is_strict_and_atomic(fact, message):
    session = _session()
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4),)),
        roles=TRANSFER_ROLES,
        facts=[_fact((1, 2, 3, 4))],
    )
    before = session.relation("transfer").provenance()
    with pytest.raises(_metadata_error(), match=message):
        session.put_relation_with_provenance(
            "transfer", _transfer_columns(), roles=TRANSFER_ROLES, facts=[fact]
        )
    assert session.relation("transfer").provenance() == before


@pytest.mark.parametrize("iterated_count", [3, 0, 5])
@pytest.mark.parametrize("representation", ["tuple", "cells"])
def test_fact_arity_uses_the_values_actually_iterated(iterated_count, representation):
    session = _session()
    tuple_values = [10, 20, 7, 1_700_000_000]
    exact_cells = [
        {"type": "u32", "hex": "0a000000"},
        {"type": "u32", "hex": "14000000"},
        {"type": "u32", "hex": "07000000"},
        {"type": "i64", "hex": "00f1536500000000"},
    ]
    values = tuple_values if representation == "tuple" else exact_cells
    iterated_values = [*values, values[-1]][:iterated_count]
    fact = {
        representation: _IteratesAs(values, iterated_values),
        "provenance": [_record()],
    }

    with pytest.raises(
        _metadata_error(), match=rf"arity mismatch: expected 4, received {iterated_count}"
    ):
        session.put_relation_with_provenance(
            "transfer", _transfer_columns(), roles=TRANSFER_ROLES, facts=[fact]
        )
    assert session.relation("transfer").provenance()["row_count"] == 0


def test_friendly_float_conversion_and_exact_nan_cells_preserve_bits():
    session = _session()
    value = 1.1
    rounded = struct.pack("<f", value)
    snapshot = session.put_relation_with_provenance(
        "float_value",
        [torch.tensor([value], device="cuda", dtype=torch.float32)],
        roles=[{"name": "value"}],
        facts=[_fact((value,))],
    )
    assert snapshot["facts"][0]["cells"] == [{"type": "f32", "hex": rounded.hex()}]
    assert snapshot["facts"][0]["tuple"] == [struct.unpack("<f", rounded)[0]]

    nan_bits = 0x7FC01234
    nan_tensor = torch.tensor([nan_bits], device="cuda", dtype=torch.uint32).view(torch.float32)
    exact = session.put_relation_with_provenance(
        "float_value",
        [nan_tensor],
        roles=[{"name": "value"}],
        facts=[
            {
                "cells": [{"type": "f32", "hex": struct.pack("<I", nan_bits).hex()}],
                "provenance": [_record()],
            }
        ],
    )
    assert exact["facts"][0]["cells"][0]["hex"] == "3412c07f"
    assert exact["facts"][0]["identity"] == _fact_identity(
        "float_value", [(4, struct.pack("<I", nan_bits))]
    )


def test_float_infinity_and_f64_signed_zero_preserve_ieee_bits():
    session = _session()
    f32 = session.put_relation_with_provenance(
        "float_value",
        [torch.tensor([float("inf"), float("-inf")], device="cuda", dtype=torch.float32)],
        roles=[{"name": "value"}],
        facts=[_fact((float("inf"),)), _fact((float("-inf"),))],
    )
    assert {fact["cells"][0]["hex"] for fact in f32["facts"]} == {
        struct.pack("<f", float("inf")).hex(),
        struct.pack("<f", float("-inf")).hex(),
    }

    f64_values = [-0.0, float("inf"), float("-inf")]
    f64 = session.put_relation_with_provenance(
        "float64_value",
        [torch.tensor(f64_values, device="cuda", dtype=torch.float64)],
        roles=[{"name": "value"}],
        facts=[_fact((value,)) for value in f64_values],
    )
    assert {fact["cells"][0]["hex"] for fact in f64["facts"]} == {
        struct.pack("<d", value).hex() for value in f64_values
    }
    negative_zero = next(
        fact for fact in f64["facts"] if fact["cells"][0]["hex"] == "0000000000000080"
    )
    assert struct.pack("<d", negative_zero["tuple"][0]) == struct.pack("<d", -0.0)


def test_integer_boundaries_and_bool_cells_round_trip_exactly():
    session = _session()
    rows = [
        (0, -(2**31), -(2**63), False),
        (2**64 - 1, 2**31 - 1, 2**63 - 1, True),
    ]
    snapshot = session.put_relation_with_provenance(
        "scalar_values",
        [
            torch.tensor([row[0] for row in rows], device="cuda", dtype=torch.uint64),
            torch.tensor([row[1] for row in rows], device="cuda", dtype=torch.int32),
            torch.tensor([row[2] for row in rows], device="cuda", dtype=torch.int64),
            torch.tensor([row[3] for row in rows], device="cuda", dtype=torch.bool),
        ],
        roles=[
            {"name": "wide"},
            {"name": "signed32"},
            {"name": "signed64"},
            {"name": "flag"},
        ],
        facts=[_fact(row) for row in rows],
    )
    assert {tuple(fact["tuple"]) for fact in snapshot["facts"]} == set(rows)
    assert {
        tuple(cell["type"] for cell in fact["cells"]) for fact in snapshot["facts"]
    } == {("u64", "i32", "i64", "bool")}


@pytest.mark.parametrize(
    "values, message",
    [
        ((-1, 0, 0, False), "u64"),
        ((2**64, 0, 0, False), "u64"),
        ((True, 0, 0, False), "bool"),
        ((0, -(2**31) - 1, 0, False), "i32"),
        ((0, 2**31, 0, False), "i32"),
        ((0, True, 0, False), "bool"),
        ((0, 0, -(2**63) - 1, False), "i64"),
        ((0, 0, 2**63, False), "i64"),
        ((0, 0, True, False), "bool"),
        ((0, 0, 0, 0), "Python bool"),
    ],
)
def test_integer_and_bool_friendly_values_reject_wrong_types_and_ranges(values, message):
    session = _session()
    columns = [
        torch.tensor([0], device="cuda", dtype=torch.uint64),
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([False], device="cuda", dtype=torch.bool),
    ]
    with pytest.raises(_metadata_error(), match=message):
        session.put_relation_with_provenance(
            "scalar_values",
            columns,
            roles=[
                {"name": "wide"},
                {"name": "signed32"},
                {"name": "signed64"},
                {"name": "flag"},
            ],
            facts=[_fact(values)],
        )


@pytest.mark.parametrize(
    "cell, message",
    [
        ({"type": "f64", "hex": "00000000"}, "type"),
        ({"type": "f32", "hex": "00"}, "4 bytes"),
        ({"type": "f32", "hex": "ABCDEF00"}, "lowercase"),
        ({"type": "f32", "hex": "not-hex!"}, "hex"),
        ({"type": "f32", "hex": "00000000", "extra": 1}, "unknown"),
    ],
)
def test_exact_cells_reject_noncanonical_input(cell, message):
    session = _session()
    with pytest.raises(_metadata_error(), match=message):
        session.put_relation_with_provenance(
            "float_value",
            [torch.tensor([0.0], device="cuda", dtype=torch.float32)],
            roles=[{"name": "value"}],
            facts=[{"cells": [cell], "provenance": [_record()]}],
        )


def test_symbol_ids_are_strict_integers_and_validate_full_row_membership():
    session = _session()
    columns = lambda: [torch.tensor([3], device="cuda", dtype=torch.int32)]
    snapshot = session.put_relation_with_provenance(
        "symbol_value", columns(), roles=[{"name": "value"}], facts=[_fact((3,))]
    )
    assert snapshot["facts"][0]["tuple"] == [3]
    assert snapshot["facts"][0]["cells"] == [{"type": "symbol", "hex": "03000000"}]

    for invalid in (True, -1, 2**32):
        with pytest.raises(_metadata_error(), match="symbol"):
            session.put_relation_with_provenance(
                "symbol_value",
                columns(),
                roles=[{"name": "value"}],
                facts=[_fact((invalid,))],
            )

    before = session.relation("symbol_value").provenance()
    with pytest.raises(_metadata_error(), match="not present"):
        session.put_relation_with_provenance(
            "symbol_value", columns(), roles=[{"name": "value"}], facts=[_fact((4,))]
        )
    assert session.relation("symbol_value").provenance() == before


def test_nullary_metadata_is_rejected_without_changing_existing_nullary_behavior():
    session = _session("pred ready(). ?- ready().")

    for columns in ([], object(), [object()]):
        with pytest.raises(_metadata_error(), match="positive arity"):
            session.put_relation_with_provenance(
                "ready", columns, roles=[], facts=[]
            )

    metadata_delta_calls = (
        lambda: session.insert_relation("ready", object(), facts=[]),
        lambda: session.apply_relation_delta(
            "ready", insert_columns=object(), insert_facts=[]
        ),
        lambda: session.apply_relation_delta_batch(
            [
                {
                    "name": "ready",
                    "insert_columns": object(),
                    "insert_facts": [],
                }
            ]
        ),
        lambda: session.apply_relation_delta_debug(
            [
                {
                    "name": "ready",
                    "insert_columns": object(),
                    "insert_facts": [],
                }
            ],
            check_equivalence=True,
        ),
    )
    for apply_delta in metadata_delta_calls:
        with pytest.raises(_metadata_error(), match="positive arity"):
            apply_delta()

    session.put_relation("ready", [])
    session.insert_relation("ready", [])
    session.apply_relation_delta("ready", insert_columns=[])
    session.apply_relation_delta_batch(
        [{"name": "ready", "insert_columns": []}]
    )
    session.apply_relation_delta_debug(
        [{"name": "ready", "insert_columns": []}], check_equivalence=True
    )
    result = session.evaluate()
    assert result.queries[0].is_true is False


def test_relation_evidence_is_immutable_and_named_reads_are_native_snapshots():
    session = _session()
    returned = session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4),)),
        roles=TRANSFER_ROLES,
        facts=[_fact((1, 2, 3, 4))],
    )
    evidence = session.relation("transfer")
    assert isinstance(evidence, pyxlog._native.RelationEvidence)
    assert pyxlog.RelationEvidence is pyxlog._native.RelationEvidence
    assert pyxlog.RelationMetadataError is pyxlog._native.RelationMetadataError
    assert issubclass(pyxlog.RelationMetadataError, ValueError)
    assert pyxlog.__all__.count("RelationEvidence") == 1
    assert pyxlog.__all__.count("RelationMetadataError") == 1
    first = evidence.provenance()
    assert first == returned
    first["roles"].clear()
    first["facts"][0]["provenance"].clear()
    assert evidence.provenance() == returned

    session.put_relation("transfer", _transfer_columns(rows=((8, 9, 10, 11),)))
    assert evidence.provenance() == returned
    assert session.relation("transfer").provenance()["metadata_present"] is False


def test_session_evidence_is_sorted_stable_and_missing_reads_raise_key_error():
    session = _session()
    session.put_relation_with_provenance(
        "zeta",
        [torch.tensor([2], device="cuda", dtype=torch.int32)],
        roles=[{"name": "value"}],
        facts=[_fact((2,))],
    )
    session.put_relation_with_provenance(
        "alpha",
        [torch.tensor([1], device="cuda", dtype=torch.int32)],
        roles=[{"name": "value"}],
        facts=[_fact((1,))],
    )

    first = session.evidence()
    second = session.evidence()
    assert first == second
    assert first["program_hash"].startswith("sha256:")
    assert len(first["program_hash"]) == 71
    assert list(first["relations"]) == sorted(first["relations"])
    alpha = session.evidence("alpha")
    zeta = session.evidence("zeta")
    assert alpha["relations"] == {"alpha": first["relations"]["alpha"]}
    assert zeta["relations"] == {"zeta": first["relations"]["zeta"]}
    assert alpha["program_hash"] == first["program_hash"]
    assert zeta["program_hash"] == first["program_hash"]

    with pytest.raises(KeyError, match="missing"):
        session.relation("missing")
    with pytest.raises(KeyError, match="missing"):
        session.evidence("missing")


def test_program_hash_distinguishes_compiled_schema_identity():
    named = _session("pred item(value: i32).")
    positional = _session("pred item(i32).")
    for session in (named, positional):
        snapshot = session.put_relation_with_provenance(
            "item",
            [torch.tensor([1], device="cuda", dtype=torch.int32)],
            roles=[{"name": "value"}],
            facts=[_fact((1,))],
        )
        assert snapshot["roles"] == [{"name": "value", "sort": None, "type": "i32"}]

    named_evidence = named.evidence()
    positional_evidence = positional.evidence()
    assert named_evidence["relations"] == positional_evidence["relations"]
    assert named_evidence["program_hash"] != positional_evidence["program_hash"]


def test_absent_evidence_tuple_fails_before_relation_and_metadata_replacement():
    session = _session()
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4),)),
        roles=TRANSFER_ROLES,
        facts=[_fact((1, 2, 3, 4))],
    )
    before = session.relation("transfer").provenance()

    with pytest.raises(_metadata_error(), match="not present"):
        session.put_relation_with_provenance(
            "transfer",
            _transfer_columns(rows=((9, 10, 11, 12),)),
            roles=TRANSFER_ROLES,
            facts=[_fact((9, 10, 11, 13))],
        )

    assert session.relation("transfer").provenance() == before
    exported = session.export_relation("transfer")
    assert [torch.from_dlpack(column).cpu().tolist() for column in exported] == [[1], [2], [3], [4]]


def test_invalid_dlpack_replacement_preserves_rows_metadata_stats_and_callbacks():
    session = _session()
    events = []
    session.register_relation_callback(events.append)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4),)),
        roles=TRANSFER_ROLES,
        facts=[_fact((1, 2, 3, 4))],
    )
    session.insert_relation(
        "transfer", _transfer_columns(rows=((5, 6, 7, 8),))
    )
    before_metadata = session.relation("transfer").provenance()
    before_stats = session.delta_stats()
    before_events = list(events)

    invalid_columns = _transfer_columns(rows=((9, 10, 11, 12),))
    invalid_columns[0] = torch.tensor([9.0], device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError, match="dtype"):
        session.put_relation_with_provenance(
            "transfer",
            invalid_columns,
            roles=TRANSFER_ROLES,
            facts=[_fact((9, 10, 11, 12))],
        )

    assert session.relation("transfer").provenance() == before_metadata
    assert session.delta_stats() == before_stats
    assert events == before_events
    assert [event["generation"] for event in events] == [1]
    exported = session.export_relation("transfer")
    assert [torch.from_dlpack(column).cpu().tolist() for column in exported] == [
        [1, 5],
        [2, 6],
        [3, 7],
        [4, 8],
    ]


def test_insert_relation_unions_existing_and_new_fact_records_without_fabrication():
    session = _session()
    existing = (1, 2, 3, 4)
    added = (5, 6, 7, 8)
    unannotated = (9, 10, 11, 12)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(existing,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(existing, _record(source="original"))],
    )

    session.insert_relation(
        "transfer",
        _transfer_columns(rows=(existing, added, added)),
        facts=[
            _fact(existing, _record(source="later"), _record(source="later")),
            _fact(added, _record(source="new")),
            _fact(added, _record(source="new")),
        ],
    )
    snapshot = session.relation("transfer").provenance()
    assert _fact_sources(snapshot, existing) == ["later", "original"]
    assert _fact_sources(snapshot, added) == ["new"]

    session.insert_relation("transfer", _transfer_columns(rows=(unannotated,)))
    snapshot = session.relation("transfer").provenance()
    assert [fact["tuple"] for fact in snapshot["facts"]] == [list(existing), list(added)]
    assert _exported_rows(session, "transfer") == [existing, added, unannotated]

    session.reset_host_transfer_stats()
    session.insert_relation("transfer", _transfer_columns(rows=(added,)), facts=[])
    assert session.host_transfer_stats()["dtoh_calls"] == 0

    metadata_free = _session()
    metadata_free.put_relation("transfer", _transfer_columns(rows=(existing,)))
    with pytest.raises(_metadata_error(), match="registered role"):
        metadata_free.insert_relation(
            "transfer", _transfer_columns(rows=(added,)), facts=[]
        )
    assert _exported_rows(metadata_free, "transfer") == [existing]


def test_empty_provenance_fact_entries_follow_insert_and_batch_cancellation_semantics():
    session = _session()
    first = (1, 2, 3, 4)
    second = (5, 6, 7, 8)
    canceled = (9, 10, 11, 12)
    session.put_relation_with_provenance(
        "transfer", _empty_transfer_columns(), roles=TRANSFER_ROLES, facts=[]
    )

    session.insert_relation(
        "transfer",
        _transfer_columns(rows=(first,)),
        facts=[{"tuple": list(first), "provenance": []}],
    )
    session.apply_relation_delta_batch(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(second,)),
                "insert_facts": [{"tuple": list(second), "provenance": []}],
            },
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(canceled,)),
                "insert_facts": [{"tuple": list(canceled), "provenance": []}],
            },
            {
                "name": "transfer",
                "delete_columns": _transfer_columns(rows=(canceled,)),
            },
        ]
    )

    snapshot = session.relation("transfer").provenance()
    assert [(fact["tuple"], fact["provenance"]) for fact in snapshot["facts"]] == [
        (list(first), []),
        (list(second), []),
    ]


def test_delete_and_raw_combined_delta_use_complete_tuple_delete_then_insert_semantics():
    session = _session()
    earlier = (1, 2, 3, 4)
    later = (1, 2, 3, 5)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(earlier, later)),
        roles=TRANSFER_ROLES,
        facts=[
            _fact(earlier, _record(source="earlier")),
            _fact(later, _record(source="later")),
        ],
    )

    session.delete_relation("transfer", _transfer_columns(rows=(later,)))
    after_delete = session.relation("transfer").provenance()
    assert [fact["tuple"] for fact in after_delete["facts"]] == [list(earlier)]

    stats = session.apply_relation_delta(
        "transfer",
        insert_columns=_transfer_columns(rows=(earlier,)),
        delete_columns=_transfer_columns(rows=(earlier,)),
        insert_facts=[_fact(earlier, _record(source="replacement"))],
    )
    assert stats["insert_rows"] == 1
    assert stats["delete_rows"] == 1
    assert stats["canceled_rows"] == 0
    assert _fact_sources(session.relation("transfer").provenance(), earlier) == [
        "replacement"
    ]

    session.apply_relation_delta(
        "transfer",
        insert_columns=_transfer_columns(rows=(earlier,)),
        delete_columns=_transfer_columns(rows=(earlier,)),
    )
    final = session.relation("transfer").provenance()
    assert final["metadata_present"] is True
    assert final["facts"] == []
    assert _exported_rows(session, "transfer") == [earlier]


def test_batch_complete_cancellation_is_a_data_metadata_generation_noop():
    session = _session()
    row = (1, 2, 3, 4)
    session.put_relation_with_provenance(
        "transfer",
        _empty_transfer_columns(),
        roles=TRANSFER_ROLES,
        facts=[],
    )
    events = []
    session.register_relation_callback(events.append)
    before = session.evidence()

    stats = session.apply_relation_delta_batch(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="canceled"))],
            },
            {"name": "transfer", "delete_columns": _transfer_columns(rows=(row,))},
        ]
    )

    assert stats["changed_relations"] == 0
    assert stats["insert_rows"] == 0
    assert stats["delete_rows"] == 0
    assert stats["canceled_rows"] == 1
    assert session.evidence() == before
    assert _exported_rows(session, "transfer") == []
    assert events == []

    session.insert_relation("transfer", _transfer_columns(rows=(row,)))
    assert [event["generation"] for event in events] == [1]


def test_batch_occurrence_trace_keeps_only_surviving_insert_lineage():
    row = (1, 2, 3, 4)

    insert_delete_insert = _session()
    insert_delete_insert.put_relation_with_provenance(
        "transfer", _empty_transfer_columns(), roles=TRANSFER_ROLES, facts=[]
    )
    events = []
    insert_delete_insert.register_relation_callback(events.append)
    stats = insert_delete_insert.apply_relation_delta_batch(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="first"))],
            },
            {"name": "transfer", "delete_columns": _transfer_columns(rows=(row,))},
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="last"))],
            },
        ]
    )
    assert stats["canceled_rows"] == 1
    assert _fact_sources(
        insert_delete_insert.relation("transfer").provenance(), row
    ) == ["last"]
    assert [event["generation"] for event in events] == [1]

    delete_insert_insert = _session()
    delete_insert_insert.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(row,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(row, _record(source="original"))],
    )
    delete_insert_insert.apply_relation_delta_batch(
        [
            {"name": "transfer", "delete_columns": _transfer_columns(rows=(row,))},
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="canceled"))],
            },
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="surviving"))],
            },
        ]
    )
    assert _fact_sources(
        delete_insert_insert.relation("transfer").provenance(), row
    ) == ["original", "surviving"]


def test_repeated_batch_inserts_union_lineage_and_emit_one_relation_event():
    session = _session()
    first_row = (1, 2, 3, 4)
    second_row = (5, 6, 7, 8)
    session.put_relation_with_provenance(
        "transfer", _empty_transfer_columns(), roles=TRANSFER_ROLES, facts=[]
    )
    events = []
    session.register_relation_callback(events.append)

    session.apply_relation_delta_batch(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(first_row,)),
                "insert_facts": [_fact(first_row, _record(source="first"))],
            },
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(first_row, second_row)),
                "insert_facts": [
                    _fact(first_row, _record(source="second")),
                    _fact(second_row, _record(source="third")),
                ],
            },
        ]
    )
    snapshot = session.relation("transfer").provenance()
    assert _fact_sources(snapshot, first_row) == ["first", "second"]
    assert _fact_sources(snapshot, second_row) == ["third"]
    assert [(event["relation"], event["generation"]) for event in events] == [
        ("transfer", 1)
    ]


def test_batch_prevalidates_every_entry_and_rejects_unknown_shapes_atomically():
    session = _session()
    original = (1, 2, 3, 4)
    first_insert = (5, 6, 7, 8)
    invalid_insert = (9, 10, 11, 12)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(original,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(original, _record(source="original"))],
    )
    events = []
    session.register_relation_callback(events.append)
    session.insert_relation("transfer", _transfer_columns(rows=(first_insert,)))
    before_rows = _exported_rows(session, "transfer")
    before_evidence = session.evidence()
    before_stats = session.delta_stats()
    before_events = list(events)

    with pytest.raises(_metadata_error(), match="not present"):
        session.apply_relation_delta_batch(
            [
                {
                    "name": "transfer",
                    "insert_columns": _transfer_columns(rows=((13, 14, 15, 16),)),
                    "insert_facts": [
                        _fact((13, 14, 15, 16), _record(source="valid"))
                    ],
                },
                {
                    "name": "transfer",
                    "insert_columns": _transfer_columns(rows=(invalid_insert,)),
                    "insert_facts": [
                        _fact((9, 10, 11, 13), _record(source="invalid"))
                    ],
                },
            ]
        )
    assert _exported_rows(session, "transfer") == before_rows
    assert session.evidence() == before_evidence
    assert session.delta_stats() == before_stats
    assert events == before_events

    with pytest.raises(ValueError, match="unknown"):
        session.apply_relation_delta_batch(
            [
                {
                    "name": "transfer",
                    "insert_columns": _transfer_columns(rows=(invalid_insert,)),
                    "unexpected": True,
                }
            ]
        )
    with pytest.raises(_metadata_error(), match="insert_facts.*insert_columns"):
        session.apply_relation_delta_batch(
            [{"name": "transfer", "insert_facts": []}]
        )
    with pytest.raises(_metadata_error(), match="insert_facts.*insert_columns"):
        session.apply_relation_delta("transfer", insert_facts=[])

    session.insert_relation(
        "transfer", _transfer_columns(rows=(invalid_insert,)), facts=[]
    )
    assert [event["generation"] for event in events] == [1, 2]


def test_strict_deterministic_d2h_fails_metadata_before_mutation_but_allows_plain_delta():
    session = _session()
    original = (1, 2, 3, 4)
    added = (5, 6, 7, 8)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(original,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(original, _record(source="original"))],
    )
    events = []
    session.register_relation_callback(events.append)
    before_rows = _exported_rows(session, "transfer")
    before_evidence = session.evidence()
    before_stats = session.delta_stats()
    session.reset_deterministic_d2h_violations()
    session.set_strict_deterministic_d2h(True)
    assert session.strict_deterministic_d2h_enabled() is True
    try:
        with pytest.raises(RuntimeError, match=r"deterministic D2H gate.*1 bytes"):
            session.insert_relation(
                "transfer",
                _transfer_columns(rows=(added,)),
                facts=[_fact(added, _record(source="blocked"))],
            )
        assert session.deterministic_d2h_violation_count() == 1
        assert _exported_rows(session, "transfer") == before_rows
        assert session.evidence() == before_evidence
        assert session.delta_stats() == before_stats
        assert events == []
    finally:
        session.set_strict_deterministic_d2h(False)
    assert session.strict_deterministic_d2h_enabled() is False

    plain = _session()
    plain.reset_deterministic_d2h_violations()
    plain.set_strict_deterministic_d2h(True)
    try:
        plain.insert_relation("transfer", _transfer_columns(rows=(original,)))
    finally:
        plain.set_strict_deterministic_d2h(False)
    assert plain.deterministic_d2h_violation_count() == 0
    assert _exported_rows(plain, "transfer") == [original]


def test_empty_typed_delete_preserves_delta_publication_without_provenance_d2h():
    def columns(values):
        return [torch.tensor(values, device="cuda", dtype=torch.int32)]

    session = _session()
    session.put_relation_with_provenance(
        "alpha",
        columns([7]),
        roles=[{"name": "value"}],
        facts=[_fact((7,), _record(source="original"))],
    )
    events = []
    session.register_relation_callback(events.append)
    before_evidence = session.evidence()
    session.reset_host_transfer_stats()
    session.reset_deterministic_d2h_violations()
    session.set_strict_deterministic_d2h(True)
    try:
        stats = session.delete_relation("alpha", columns([]))
    finally:
        session.set_strict_deterministic_d2h(False)

    assert session.host_transfer_stats()["dtoh_calls"] == 0
    assert session.host_transfer_stats()["dtoh_bytes"] == 0
    assert session.deterministic_d2h_violation_count() == 0
    assert session.evidence() == before_evidence
    assert _exported_rows(session, "alpha") == [(7,)]
    assert session.delta_stats() == stats

    control = _session()
    control.put_relation("alpha", columns([7]))
    control_events = []
    control.register_relation_callback(control_events.append)
    control_stats = control.delete_relation("alpha", columns([]))
    assert stats["insert_rows"] == 0
    assert stats["delete_rows"] == 0
    assert len(events) == 1
    assert stats == control_stats
    assert events == control_events


def test_debug_uses_precommit_equivalence_for_updates_and_fully_canceled_batches():
    row = (1, 2, 3, 4)
    session = _session()
    session.put_relation_with_provenance(
        "transfer", _empty_transfer_columns(), roles=TRANSFER_ROLES, facts=[]
    )
    events = []
    session.register_relation_callback(events.append)

    canceled = session.apply_relation_delta_debug(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="canceled"))],
            },
            {"name": "transfer", "delete_columns": _transfer_columns(rows=(row,))},
        ],
        check_equivalence=True,
    )
    assert canceled["changed_relations"] == 0
    assert canceled["equivalent_to_full_recompute"] is True
    assert session.relation("transfer").provenance()["facts"] == []
    assert events == []

    updated = session.apply_relation_delta_debug(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(row,)),
                "insert_facts": [_fact(row, _record(source="surviving"))],
            }
        ],
        check_equivalence=True,
    )
    assert updated["equivalent_to_full_recompute"] is True
    assert _fact_sources(session.relation("transfer").provenance(), row) == [
        "surviving"
    ]

    second = (5, 6, 7, 8)
    unchecked = session.apply_relation_delta_debug(
        [
            {
                "name": "transfer",
                "insert_columns": _transfer_columns(rows=(second,)),
                "insert_facts": [_fact(second, _record(source="unchecked"))],
            }
        ],
        check_equivalence=False,
    )
    assert unchecked["equivalent_to_full_recompute"] is None
    assert _fact_sources(session.relation("transfer").provenance(), second) == [
        "unchecked"
    ]


def test_late_full_recompute_budget_failure_rolls_back_all_session_state():
    source = """
    pred alpha(value: i32).
    pred stable(value: i32).
    pred out(value: i32).
    out(X) :- alpha(X).
    ?- out(X).
    """

    def prepared_session():
        session = pyxlog.LogicProgram.compile(
            source, device=0, memory_mb=1
        ).session()
        stable = torch.arange(53_000, device="cuda", dtype=torch.int32)
        session.put_relation("stable", [stable])
        session.put_relation_with_provenance(
            "alpha",
            [torch.empty(0, device="cuda", dtype=torch.int32)],
            roles=[{"name": "value"}],
            facts=[],
        )
        initial = session.evaluate()
        assert initial.queries[0].num_rows == 0
        return session, stable

    def update():
        return [
            {
                "name": "alpha",
                "insert_columns": [torch.tensor([1], device="cuda", dtype=torch.int32)],
                "insert_facts": [_fact((1,), _record(source="prepared"))],
            }
        ]

    control, control_stable = prepared_session()
    control_stats = control.apply_relation_delta_debug(
        update(), check_equivalence=False
    )
    assert control_stats["insert_rows"] == 1
    assert _fact_sources(control.relation("alpha").provenance(), (1,)) == [
        "prepared"
    ]
    del control, control_stable
    gc.collect()
    torch.cuda.empty_cache()

    session, stable = prepared_session()
    events = []
    session.register_relation_callback(events.append)
    before_evidence = session.evidence()
    before_stats = session.delta_stats()
    before_memory = session.memory_stats()["allocated_bytes"]

    with pytest.raises(
        RuntimeError,
        match=r"Resource exhausted: GPU memory allocation, estimated 212000 bytes",
    ):
        session.apply_relation_delta_debug(update(), check_equivalence=True)

    assert session.evidence() == before_evidence
    assert session.delta_stats() == before_stats
    assert session.relation("alpha").provenance()["row_count"] == 0
    assert session.relation("stable").provenance()["row_count"] == 53_000
    assert session.memory_stats()["allocated_bytes"] <= before_memory
    assert events == []

    after_failure = session.evaluate()
    assert after_failure.queries[0].num_rows == 0
    committed = session.apply_relation_delta_debug(update(), check_equivalence=False)
    assert committed["insert_rows"] == 1
    assert _fact_sources(session.relation("alpha").provenance(), (1,)) == [
        "prepared"
    ]
    assert [event["generation"] for event in events] == [1]
    assert stable.numel() == 53_000


def test_runtime_constraint_failure_discards_prepared_metadata_and_callback_state():
    session = _session(
        """
        pred forbidden(value: i32).
        pred allowed(value: i32).
        pred out(value: i32).
        out(X) :- allowed(X).
        :- forbidden(X).
        ?- out(X).
        """
    )
    session.put_relation_with_provenance(
        "forbidden",
        [torch.empty(0, device="cuda", dtype=torch.int32)],
        roles=[{"name": "value"}],
        facts=[],
    )
    events = []
    session.register_relation_callback(events.append)
    before = session.evidence()
    before_stats = session.delta_stats()

    with pytest.raises(RuntimeError, match="[Cc]onstraint"):
        session.insert_relation(
            "forbidden",
            [torch.tensor([1], device="cuda", dtype=torch.int32)],
            facts=[_fact((1,), _record(source="rejected"))],
        )
    assert session.evidence() == before
    assert session.delta_stats() == before_stats
    assert events == []
    assert _exported_rows(session, "forbidden") == []


def test_callback_exception_observes_already_committed_data_metadata_and_stats():
    session = _session()
    row = (1, 2, 3, 4)
    session.put_relation_with_provenance(
        "transfer", _empty_transfer_columns(), roles=TRANSFER_ROLES, facts=[]
    )

    def reject_event(_payload):
        raise LookupError("observer failed")

    callback_id = session.register_relation_callback(reject_event)
    with pytest.raises(LookupError, match="observer failed"):
        session.insert_relation(
            "transfer",
            _transfer_columns(rows=(row,)),
            facts=[_fact(row, _record(source="committed"))],
        )

    assert _exported_rows(session, "transfer") == [row]
    assert _fact_sources(session.relation("transfer").provenance(), row) == [
        "committed"
    ]
    assert session.delta_stats()["insert_rows"] == 1

    assert session.unregister_relation_callback(callback_id) is True
    events = []
    session.register_relation_callback(events.append)
    session.insert_relation("transfer", _transfer_columns(rows=((5, 6, 7, 8),)))
    assert [event["generation"] for event in events] == [2]


def test_remove_clear_and_replacement_keep_metadata_and_generations_consistent():
    session = _session()
    transfer = (1, 2, 3, 4)
    session.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(transfer,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(transfer, _record(source="transfer"))],
    )
    session.put_relation_with_provenance(
        "alpha",
        [torch.tensor([9], device="cuda", dtype=torch.int32)],
        roles=[{"name": "value"}],
        facts=[_fact((9,), _record(source="alpha"))],
    )
    alpha_before = session.relation("alpha").provenance()

    events = []
    session.register_relation_callback(events.append)
    session.delete_relation("transfer", _transfer_columns(rows=(transfer,)))
    assert events[-1]["generation"] == 1
    assert session.relation("alpha").provenance() == alpha_before
    session.put_relation("transfer", _transfer_columns(rows=((5, 6, 7, 8),)))
    assert session.relation("transfer").provenance()["metadata_present"] is False

    session.insert_relation("transfer", _transfer_columns(rows=((9, 10, 11, 12),)))
    assert events[-1]["generation"] == 2
    assert session.remove_relation("transfer") is True
    assert session.delta_stats()["status"] == "unavailable"
    with pytest.raises(KeyError, match="transfer"):
        session.relation("transfer")
    assert session.relation("alpha").provenance() == alpha_before

    session.clear_relations()
    assert session.delta_stats()["status"] == "unavailable"
    with pytest.raises(KeyError, match="alpha"):
        session.relation("alpha")
    session.insert_relation("transfer", _transfer_columns(rows=((13, 14, 15, 16),)))
    assert events[-1]["generation"] == 3
