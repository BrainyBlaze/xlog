import copy
import gc
import hashlib
import math
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


class _LiesAboutLength(list):
    def __len__(self):
        return (1 << 63) - 1


class _HugeLengthHintIterator:
    def __init__(self, values):
        self._values = iter(values)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._values)

    def __length_hint__(self):
        return (1 << 63) - 1


class _LiesAboutIteratorLength(list):
    def __iter__(self):
        return _HugeLengthHintIterator(super().__iter__())


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


def _schema_identity(
    relation: str,
    columns: list[tuple[str, int, str | None]],
) -> str:
    payload = bytearray(b"xlog.relation.schema.v1\0")
    encoded_name = relation.encode("utf-8")
    payload += struct.pack("<I", len(encoded_name))
    payload += encoded_name
    payload += struct.pack("<I", len(columns))
    for name, type_code, sort in columns:
        encoded_column = name.encode("utf-8")
        payload += struct.pack("<I", len(encoded_column))
        payload += encoded_column
        payload += bytes([type_code])
        if sort is None:
            payload += b"\0"
        else:
            encoded_sort = sort.encode("utf-8")
            payload += b"\1"
            payload += struct.pack("<I", len(encoded_sort))
            payload += encoded_sort
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


def _manifest_node(manifest, path):
    node = manifest
    for component in path:
        node = node[component]
    return node


def _transfer_manifest_export(rows=((1, 2, 3, 4),)):
    source = _session()
    snapshot = source.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=rows),
        roles=TRANSFER_ROLES,
        facts=[
            _fact(
                rows[0],
                _record(source="first-source", document=None, span=None),
                _record(source="second-source", document="review", span={"start": 2, "end": 5}),
            )
        ],
    )
    exported = source.export_relation_with_provenance("transfer")
    return source, snapshot, exported


def _seed_manifest_import_target():
    target = _session()
    original = (90, 91, 92, 93)
    target.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=(original,)),
        roles=TRANSFER_ROLES,
        facts=[_fact(original, _record(source="preserved"))],
    )
    events = []
    target.register_relation_callback(events.append)
    return target, events


def _assert_manifest_failure_is_atomic(manifest, columns, match):
    target, events = _seed_manifest_import_target()
    before_rows = _exported_rows(target, "transfer")
    before_evidence = target.evidence()
    before_stats = target.delta_stats()

    with pytest.raises(_metadata_error(), match=match):
        target.put_relation_from_manifest("transfer", columns, manifest)

    assert _exported_rows(target, "transfer") == before_rows
    assert target.evidence() == before_evidence
    assert target.delta_stats() == before_stats
    assert events == []
    target.insert_relation(
        "transfer", _transfer_columns(rows=((94, 95, 96, 97),))
    )
    assert [event["generation"] for event in events] == [1]


def test_manifest_quaternary_direct_dlpack_round_trip_is_exact_and_lifetime_safe():
    source_program = SOURCE + "\n?- transfer(Giver, Receiver, Asset, Time).\n"
    source = _session(source_program)
    rows = ((1, 2, 3, 4), (5, 6, 7, 8))
    snapshot = source.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=rows),
        roles=TRANSFER_ROLES,
        facts=[
            _fact(
                rows[0],
                _record(source="manual", document=None, span=None, kind="confirmation"),
                _record(source="extractor", document="document", span={"start": 1, "end": 9}),
            )
        ],
    )

    exported = source.export_relation_with_provenance("transfer")
    assert set(exported) == {"columns", "manifest"}
    assert isinstance(exported["columns"], list)
    assert len(exported["columns"]) == 4
    manifest = exported["manifest"]
    assert set(manifest) == {
        "format",
        "version",
        "predicate",
        "row_count",
        "metadata_present",
        "roles",
        "facts",
    }
    assert manifest["format"] == "xlog.relation-provenance"
    assert manifest["version"] == 1
    assert manifest["predicate"] == {
        "name": "transfer",
        "arity": 4,
        "schema_sha256": _schema_identity(
            "transfer",
            [
                ("giver", 0, "party"),
                ("receiver", 0, "party"),
                ("asset", 0, "asset"),
                ("time", 3, None),
            ],
        ),
    }
    assert manifest["row_count"] == 2
    assert manifest["metadata_present"] is True
    assert manifest["roles"] == snapshot["roles"]
    assert len(manifest["facts"]) == 1
    assert set(manifest["facts"][0]) == {"identity", "cells", "provenance"}
    assert "tuple" not in manifest["facts"][0]
    assert manifest["facts"][0]["identity"] == snapshot["facts"][0]["identity"]
    assert manifest["facts"][0]["cells"] == snapshot["facts"][0]["cells"]
    assert manifest["facts"][0]["provenance"] == snapshot["facts"][0]["provenance"]
    assert "program_hash" not in manifest

    fresh = _session(source_program)
    reconstructed = fresh.put_relation_from_manifest(
        "transfer", exported["columns"], manifest
    )
    assert reconstructed == snapshot
    assert fresh.relation("transfer").provenance() == snapshot
    assert _exported_rows(source, "transfer") == sorted(rows)

    del exported
    del source
    gc.collect()
    torch.cuda.synchronize()
    fresh.reset_host_transfer_stats()
    query = fresh.evaluate().queries[0]
    assert all(pyxlog.dlpack_is_cuda(column) for column in query.tensors)
    assert fresh.host_transfer_stats()["dtoh_bytes"] == 0
    query_columns = [torch.from_dlpack(column).cpu().tolist() for column in query.tensors]
    assert sorted(zip(*query_columns)) == sorted(rows)


def test_manifest_metadata_free_ternary_round_trip_has_no_membership_transfer():
    program = "pred positional(u32, i64, symbol). ?- positional(A, B, C)."
    source = _session(program)
    columns = [
        torch.tensor([1, 2], device="cuda", dtype=torch.int32),
        torch.tensor([-3, 4], device="cuda", dtype=torch.int64),
        torch.tensor([7, 9], device="cuda", dtype=torch.int32),
    ]
    source.put_relation("positional", columns)
    exported = source.export_relation_with_provenance("positional")
    manifest = exported["manifest"]
    assert manifest == {
        "format": "xlog.relation-provenance",
        "version": 1,
        "predicate": {
            "name": "positional",
            "arity": 3,
            "schema_sha256": _schema_identity(
                "positional",
                [("c0", 0, None), ("c1", 3, None), ("c2", 7, None)],
            ),
        },
        "row_count": 2,
        "metadata_present": False,
        "roles": [],
        "facts": [],
    }

    fresh = _session(program)
    fresh.reset_host_transfer_stats()
    fresh.reset_deterministic_d2h_violations()
    fresh.set_strict_deterministic_d2h(True)
    try:
        snapshot = fresh.put_relation_from_manifest(
            "positional", exported["columns"], manifest
        )
    finally:
        fresh.set_strict_deterministic_d2h(False)
    assert snapshot == {
        "relation": "positional",
        "metadata_present": False,
        "row_count": 2,
        "roles": [],
        "facts": [],
    }
    assert fresh.host_transfer_stats()["dtoh_calls"] == 0
    assert fresh.host_transfer_stats()["dtoh_bytes"] == 0
    assert fresh.deterministic_d2h_violation_count() == 0
    query = fresh.evaluate().queries[0]
    query_columns = [torch.from_dlpack(column).cpu().tolist() for column in query.tensors]
    assert sorted(zip(*query_columns)) == [(1, -3, 7), (2, 4, 9)]


def test_manifest_reconstruction_is_row_order_independent_sparse_and_symbol_numeric():
    program = "pred positional(u32, i64, symbol). ?- positional(A, B, C)."
    source = _session(program)
    rows = ((1, 10, 7), (2, 20, 9), (3, 30, 11))
    source_snapshot = source.put_relation_with_provenance(
        "positional",
        [
            torch.tensor([row[0] for row in rows], device="cuda", dtype=torch.int32),
            torch.tensor([row[1] for row in rows], device="cuda", dtype=torch.int64),
            torch.tensor([row[2] for row in rows], device="cuda", dtype=torch.int32),
        ],
        roles=[{"name": "entity"}, {"name": "time"}, {"name": "label"}],
        facts=[_fact(rows[1], _record(source="sparse"))],
    )
    exported = source.export_relation_with_provenance("positional")
    tensors = [torch.from_dlpack(column) for column in exported["columns"]]
    order = torch.tensor([2, 0, 1], device="cuda")
    reordered = [column.index_select(0, order) for column in tensors]

    fresh = _session(program)
    snapshot = fresh.put_relation_from_manifest(
        "positional", reordered, exported["manifest"]
    )
    assert snapshot == source_snapshot
    assert snapshot["row_count"] == 3
    assert snapshot["roles"] == [
        {"name": "entity", "sort": None, "type": "u32"},
        {"name": "time", "sort": None, "type": "i64"},
        {"name": "label", "sort": None, "type": "symbol"},
    ]
    assert [fact["tuple"] for fact in snapshot["facts"]] == [[2, 20, 9]]
    assert snapshot["facts"][0]["cells"][2] == {
        "type": "symbol",
        "hex": "09000000",
    }
    query = fresh.evaluate().queries[0]
    query_columns = [torch.from_dlpack(column).cpu().tolist() for column in query.tensors]
    assert sorted(zip(*query_columns)) == sorted(rows)


@pytest.mark.parametrize(
    "path",
    [
        (),
        ("predicate",),
        ("roles", 0),
        ("facts", 0),
        ("facts", 0, "cells", 0),
        ("facts", 0, "provenance", 1),
        ("facts", 0, "provenance", 1, "span"),
    ],
)
def test_manifest_unknown_keys_are_rejected_at_every_dictionary_level(path):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    _manifest_node(manifest, path)["unexpected"] = "value"
    _assert_manifest_failure_is_atomic(manifest, columns, "unknown key")


@pytest.mark.parametrize(
    "path, key",
    [
        ((), "format"),
        ((), "version"),
        ((), "predicate"),
        ((), "row_count"),
        ((), "metadata_present"),
        ((), "roles"),
        ((), "facts"),
        (("predicate",), "name"),
        (("predicate",), "arity"),
        (("predicate",), "schema_sha256"),
        (("roles", 0), "name"),
        (("roles", 0), "sort"),
        (("roles", 0), "type"),
        (("facts", 0), "identity"),
        (("facts", 0), "cells"),
        (("facts", 0), "provenance"),
        (("facts", 0, "cells", 0), "type"),
        (("facts", 0, "cells", 0), "hex"),
        (("facts", 0, "provenance", 1), "source"),
        (("facts", 0, "provenance", 1), "document"),
        (("facts", 0, "provenance", 1), "span"),
        (("facts", 0, "provenance", 1), "content_hash"),
        (("facts", 0, "provenance", 1), "kind"),
        (("facts", 0, "provenance", 1), "polarity"),
        (("facts", 0, "provenance", 1, "span"), "start"),
        (("facts", 0, "provenance", 1, "span"), "end"),
    ],
)
def test_manifest_required_keys_are_rejected_at_every_dictionary_level(path, key):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    del _manifest_node(manifest, path)[key]
    _assert_manifest_failure_is_atomic(manifest, columns, "missing required")


@pytest.mark.parametrize(
    "case, match",
    [
        ("wrong_format", "format"),
        ("unsupported_version", "version"),
        ("boolean_version", "version"),
        ("wrong_predicate", "predicate"),
        ("wrong_arity", "arity"),
        ("boolean_arity", "arity"),
        ("wrong_schema", "schema"),
        ("wrong_row_count", "row count"),
        ("boolean_row_count", "row_count"),
        ("integer_metadata_present", "metadata_present"),
        ("metadata_absent_with_roles", "metadata_present"),
        ("changed_identity", "identity"),
        ("uppercase_cell", "lowercase"),
        ("changed_role_name", "role 0"),
        ("reordered_roles", "role 0"),
        ("changed_role_sort", "sort"),
        ("changed_role_type", "type"),
        ("null_role_sort", "sort"),
        ("null_role_type", "type"),
        ("changed_cell_type", "type"),
        ("short_cell", "4 bytes"),
        ("oversized_cell", "4 bytes"),
        ("nonhex_cell", "hex"),
        ("record_value_type", "source"),
        ("span_negative", "non-negative"),
        ("span_bool", "non-negative"),
        ("span_reversed", "start"),
    ],
)
def test_manifest_static_mismatches_are_atomic(case, match):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    if case == "wrong_format":
        manifest["format"] = "other"
    elif case == "unsupported_version":
        manifest["version"] = 2
    elif case == "boolean_version":
        manifest["version"] = True
    elif case == "wrong_predicate":
        manifest["predicate"]["name"] = "other"
    elif case == "wrong_arity":
        manifest["predicate"]["arity"] = 3
    elif case == "boolean_arity":
        manifest["predicate"]["arity"] = True
    elif case == "wrong_schema":
        manifest["predicate"]["schema_sha256"] = "sha256:" + "0" * 64
    elif case == "wrong_row_count":
        manifest["row_count"] += 1
    elif case == "boolean_row_count":
        manifest["row_count"] = True
    elif case == "integer_metadata_present":
        manifest["metadata_present"] = 1
    elif case == "metadata_absent_with_roles":
        manifest["metadata_present"] = False
    elif case == "changed_identity":
        manifest["facts"][0]["identity"] = "sha256:" + "f" * 64
    elif case == "uppercase_cell":
        manifest["facts"][0]["cells"][0]["hex"] = "0100000A"
    elif case == "changed_role_name":
        manifest["roles"][0]["name"] = "other"
    elif case == "reordered_roles":
        manifest["roles"][0], manifest["roles"][1] = (
            manifest["roles"][1],
            manifest["roles"][0],
        )
    elif case == "changed_role_sort":
        manifest["roles"][0]["sort"] = "asset"
    elif case == "changed_role_type":
        manifest["roles"][0]["type"] = "i32"
    elif case == "null_role_sort":
        manifest["roles"][0]["sort"] = None
    elif case == "null_role_type":
        manifest["roles"][0]["type"] = None
    elif case == "changed_cell_type":
        manifest["facts"][0]["cells"][0]["type"] = "i32"
    elif case == "short_cell":
        manifest["facts"][0]["cells"][0]["hex"] = "01"
    elif case == "oversized_cell":
        manifest["facts"][0]["cells"][0]["hex"] = "00" * (1024 * 1024)
    elif case == "nonhex_cell":
        manifest["facts"][0]["cells"][0]["hex"] = "nothex00"
    elif case == "record_value_type":
        manifest["facts"][0]["provenance"][0]["source"] = 7
    elif case == "span_negative":
        manifest["facts"][0]["provenance"][1]["span"]["start"] = -1
    elif case == "span_bool":
        manifest["facts"][0]["provenance"][1]["span"]["start"] = True
    elif case == "span_reversed":
        manifest["facts"][0]["provenance"][1]["span"] = {"start": 6, "end": 5}
    else:
        raise AssertionError(f"unhandled case {case}")
    _assert_manifest_failure_is_atomic(manifest, columns, match)


@pytest.mark.parametrize(
    "case, match",
    [
        ("top", "dictionary"),
        ("predicate", "predicate.*dictionary"),
        ("roles", "roles.*sequence"),
        ("facts", "facts.*sequence"),
        ("cells", "cells.*sequence"),
        ("provenance", "provenance.*sequence"),
        ("span", "span.*dictionary"),
    ],
)
def test_manifest_wrong_container_types_are_rejected_atomically(case, match):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    if case == "top":
        manifest = []
    elif case == "predicate":
        manifest["predicate"] = []
    elif case == "roles":
        manifest["roles"] = {}
    elif case == "facts":
        manifest["facts"] = {}
    elif case == "cells":
        manifest["facts"][0]["cells"] = {}
    elif case == "provenance":
        manifest["facts"][0]["provenance"] = {}
    elif case == "span":
        manifest["facts"][0]["provenance"][1]["span"] = []
    else:
        raise AssertionError(f"unhandled case {case}")
    _assert_manifest_failure_is_atomic(manifest, columns, match)


@pytest.mark.parametrize(
    "case, match",
    [
        ("missing_role", "expected 4 roles"),
        ("extra_role", "expected 4 roles"),
        ("role_element", "role 0.*dictionary"),
        ("missing_cell", "cells arity mismatch"),
        ("extra_cell", "cells arity mismatch"),
        ("cell_element", "cell 0.*dictionary"),
        ("fact_element", "fact 0.*dictionary"),
        ("record_element", "record 0.*dictionary"),
    ],
)
def test_manifest_sequence_arities_and_element_types_are_strict(case, match):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    if case == "missing_role":
        manifest["roles"].pop()
    elif case == "extra_role":
        manifest["roles"].append(copy.deepcopy(manifest["roles"][-1]))
    elif case == "role_element":
        manifest["roles"][0] = 1
    elif case == "missing_cell":
        manifest["facts"][0]["cells"].pop()
    elif case == "extra_cell":
        manifest["facts"][0]["cells"].append(
            copy.deepcopy(manifest["facts"][0]["cells"][-1])
        )
    elif case == "cell_element":
        manifest["facts"][0]["cells"][0] = 1
    elif case == "fact_element":
        manifest["facts"][0] = 1
    elif case == "record_element":
        manifest["facts"][0]["provenance"][0] = 1
    else:
        raise AssertionError(f"unhandled case {case}")
    _assert_manifest_failure_is_atomic(manifest, columns, match)


def test_unsupported_manifest_version_takes_precedence_over_future_shape():
    _source, _snapshot, exported = _transfer_manifest_export()
    future_manifest = {
        "format": "xlog.relation-provenance",
        "version": 2,
        "future_only": object(),
    }
    target = _session()
    with pytest.raises(_metadata_error(), match="unsupported.*version 2"):
        target.put_relation_from_manifest(
            "transfer", exported["columns"], future_manifest
        )
    recovered = [torch.from_dlpack(column).cpu().tolist() for column in exported["columns"]]
    assert list(zip(*recovered)) == [(1, 2, 3, 4)]


def test_manifest_import_normalizes_fact_and_record_order_and_exact_duplicates():
    source = _session()
    rows = ((1, 2, 3, 4), (5, 6, 7, 8))
    expected = source.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=rows),
        roles=TRANSFER_ROLES,
        facts=[
            _fact(rows[0], _record(source="a"), _record(source="b")),
            _fact(rows[1], _record(source="c"), _record(source="d")),
        ],
    )
    exported = source.export_relation_with_provenance("transfer")
    manifest = copy.deepcopy(exported["manifest"])
    manifest["facts"].reverse()
    for fact in manifest["facts"]:
        fact["provenance"].reverse()
        fact["provenance"].append(copy.deepcopy(fact["provenance"][0]))
    manifest["facts"].append(copy.deepcopy(manifest["facts"][0]))

    fresh = _session()
    fresh.reset_host_transfer_stats()
    reconstructed = fresh.put_relation_from_manifest(
        "transfer", exported["columns"], manifest
    )
    assert reconstructed == expected
    assert fresh.host_transfer_stats()["dtoh_calls"] == 1
    assert fresh.host_transfer_stats()["dtoh_bytes"] == 2


def test_manifest_static_validation_does_not_consume_dlpack_capsules():
    _source, _snapshot, exported = _transfer_manifest_export()
    manifest = copy.deepcopy(exported["manifest"])
    manifest["version"] = 2
    target = _session()
    with pytest.raises(_metadata_error(), match="version"):
        target.put_relation_from_manifest("transfer", exported["columns"], manifest)
    recovered = [torch.from_dlpack(column).cpu().tolist() for column in exported["columns"]]
    assert list(zip(*recovered)) == [(1, 2, 3, 4)]


@pytest.mark.parametrize("case", ["missing", "extra"])
def test_manifest_column_arity_failures_do_not_consume_dlpack_capsules(case):
    _source, expected, exported = _transfer_manifest_export()
    columns = list(exported["columns"])
    if case == "missing":
        invalid_columns = columns[:-1]
    elif case == "extra":
        invalid_columns = [*columns, columns[0]]
    else:
        raise AssertionError(f"unhandled case {case}")

    target = _session()
    with pytest.raises(RuntimeError, match="tensor count"):
        target.put_relation_from_manifest(
            "transfer", invalid_columns, exported["manifest"]
        )
    assert (
        target.put_relation_from_manifest(
            "transfer", exported["columns"], exported["manifest"]
        )
        == expected
    )
    assert _exported_rows(target, "transfer") == [(1, 2, 3, 4)]


@pytest.mark.parametrize("case", ["role", "identity", "nested_unknown"])
def test_manifest_late_static_failures_do_not_consume_dlpack_capsules(case):
    _source, _snapshot, exported = _transfer_manifest_export()
    manifest = copy.deepcopy(exported["manifest"])
    if case == "role":
        manifest["roles"][0]["name"] = "other"
    elif case == "identity":
        manifest["facts"][0]["identity"] = "sha256:" + "f" * 64
    elif case == "nested_unknown":
        manifest["facts"][0]["provenance"][0]["unexpected"] = 1
    else:
        raise AssertionError(f"unhandled case {case}")

    target = _session()
    with pytest.raises(_metadata_error()):
        target.put_relation_from_manifest("transfer", exported["columns"], manifest)
    recovered = [torch.from_dlpack(column).cpu().tolist() for column in exported["columns"]]
    assert list(zip(*recovered)) == [(1, 2, 3, 4)]


def test_manifest_schema_identity_mismatch_is_rejected_before_dlpack_import():
    source = _session("pred item(value: i32).")
    source.put_relation_with_provenance(
        "item",
        [torch.tensor([1], device="cuda", dtype=torch.int32)],
        roles=[{"name": "value"}],
        facts=[_fact((1,))],
    )
    exported = source.export_relation_with_provenance("item")
    target = _session("pred item(other: i32).")
    with pytest.raises(_metadata_error(), match="schema"):
        target.put_relation_from_manifest(
            "item", exported["columns"], exported["manifest"]
        )
    assert torch.from_dlpack(exported["columns"][0]).cpu().tolist() == [1]
    assert target.relation("item").provenance()["row_count"] == 0


@pytest.mark.parametrize("case", ["dtype", "column_count", "row_lengths"])
def test_manifest_dlpack_failures_preserve_target_state(case):
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = _transfer_columns(rows=((1, 2, 3, 4),))
    if case == "dtype":
        columns[0] = torch.tensor([1.0], device="cuda", dtype=torch.float32)
    elif case == "column_count":
        columns.pop()
    elif case == "row_lengths":
        columns[0] = torch.tensor([1, 2], device="cuda", dtype=torch.int32)
    else:
        raise AssertionError(f"unhandled case {case}")

    target, events = _seed_manifest_import_target()
    before_rows = _exported_rows(target, "transfer")
    before_evidence = target.evidence()
    before_stats = target.delta_stats()
    with pytest.raises(RuntimeError):
        target.put_relation_from_manifest(
            "transfer", columns, exported["manifest"]
        )
    assert _exported_rows(target, "transfer") == before_rows
    assert target.evidence() == before_evidence
    assert target.delta_stats() == before_stats
    assert events == []
    target.insert_relation(
        "transfer", _transfer_columns(rows=((94, 95, 96, 97),))
    )
    assert [event["generation"] for event in events] == [1]


def test_manifest_absent_fact_membership_is_rejected_atomically():
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    manifest = copy.deepcopy(exported["manifest"])
    manifest["facts"][0]["cells"][3]["hex"] = struct.pack("<q", 5).hex()
    manifest["facts"][0]["identity"] = _fact_identity(
        "transfer",
        [
            (0, struct.pack("<I", 1)),
            (0, struct.pack("<I", 2)),
            (0, struct.pack("<I", 3)),
            (3, struct.pack("<q", 5)),
        ],
    )
    _assert_manifest_failure_is_atomic(manifest, columns, "not present")


def test_manifest_strict_membership_gate_fails_before_any_session_mutation():
    _source, _snapshot, exported = _transfer_manifest_export()
    columns = [torch.from_dlpack(column) for column in exported["columns"]]
    target, events = _seed_manifest_import_target()
    before_rows = _exported_rows(target, "transfer")
    before_evidence = target.evidence()
    before_stats = target.delta_stats()
    target.reset_deterministic_d2h_violations()
    target.set_strict_deterministic_d2h(True)
    try:
        with pytest.raises(RuntimeError, match=r"deterministic D2H gate.*1 bytes"):
            target.put_relation_from_manifest(
                "transfer", columns, exported["manifest"]
            )
    finally:
        target.set_strict_deterministic_d2h(False)
    assert target.deterministic_d2h_violation_count() == 1
    assert _exported_rows(target, "transfer") == before_rows
    assert target.evidence() == before_evidence
    assert target.delta_stats() == before_stats
    assert events == []


def test_manifest_role_only_and_empty_provenance_fact_round_trip_without_fabrication():
    role_only = _session()
    role_only.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((1, 2, 3, 4),)),
        roles=TRANSFER_ROLES,
        facts=[],
    )
    exported = role_only.export_relation_with_provenance("transfer")
    fresh = _session()
    fresh.reset_host_transfer_stats()
    fresh.reset_deterministic_d2h_violations()
    fresh.set_strict_deterministic_d2h(True)
    try:
        snapshot = fresh.put_relation_from_manifest(
            "transfer", exported["columns"], exported["manifest"]
        )
    finally:
        fresh.set_strict_deterministic_d2h(False)
    assert snapshot["metadata_present"] is True
    assert snapshot["roles"] == [
        *TRANSFER_ROLES[:3],
        {"name": "time", "sort": None, "type": "i64"},
    ]
    assert snapshot["facts"] == []
    assert fresh.host_transfer_stats()["dtoh_bytes"] == 0
    assert fresh.deterministic_d2h_violation_count() == 0

    empty_records = _session()
    empty_records.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=((5, 6, 7, 8),)),
        roles=TRANSFER_ROLES,
        facts=[{"tuple": [5, 6, 7, 8], "provenance": []}],
    )
    exported = empty_records.export_relation_with_provenance("transfer")
    reconstructed = _session().put_relation_from_manifest(
        "transfer", exported["columns"], exported["manifest"]
    )
    assert reconstructed["facts"][0]["tuple"] == [5, 6, 7, 8]
    assert reconstructed["facts"][0]["provenance"] == []


def test_manifest_success_replaces_old_rows_and_metadata_free_import_clears_evidence():
    source, expected, exported = _transfer_manifest_export(rows=((1, 2, 3, 4),))
    target, events = _seed_manifest_import_target()
    reconstructed = target.put_relation_from_manifest(
        "transfer", exported["columns"], exported["manifest"]
    )
    assert reconstructed == expected
    assert _exported_rows(target, "transfer") == [(1, 2, 3, 4)]
    assert target.relation("transfer").provenance() == expected
    assert events == []
    assert _exported_rows(source, "transfer") == [(1, 2, 3, 4)]

    plain = _session()
    plain.put_relation("transfer", _transfer_columns(rows=((20, 21, 22, 23),)))
    plain_export = plain.export_relation_with_provenance("transfer")
    cleared = target.put_relation_from_manifest(
        "transfer", plain_export["columns"], plain_export["manifest"]
    )
    assert cleared["metadata_present"] is False
    assert cleared["roles"] == []
    assert cleared["facts"] == []
    assert _exported_rows(target, "transfer") == [(20, 21, 22, 23)]


def test_manifest_positional_import_respects_existing_role_contract_atomically():
    program = "pred positional(u32, i64, symbol)."
    target = _session(program)
    columns = [
        torch.tensor([1], device="cuda", dtype=torch.int32),
        torch.tensor([2], device="cuda", dtype=torch.int64),
        torch.tensor([3], device="cuda", dtype=torch.int32),
    ]
    target.put_relation_with_provenance(
        "positional",
        columns,
        roles=[{"name": "entity"}, {"name": "time"}, {"name": "label"}],
        facts=[],
    )
    before = target.relation("positional").provenance()

    source = _session(program)
    source.put_relation_with_provenance(
        "positional",
        columns,
        roles=[{"name": "subject"}, {"name": "instant"}, {"name": "category"}],
        facts=[],
    )
    exported = source.export_relation_with_provenance("positional")
    with pytest.raises(_metadata_error(), match="registered role contract"):
        target.put_relation_from_manifest(
            "positional", exported["columns"], exported["manifest"]
        )
    assert target.relation("positional").provenance() == before
    assert [torch.from_dlpack(column).cpu().tolist() for column in exported["columns"]] == [
        [1],
        [2],
        [3],
    ]


def test_manifest_repeated_exports_have_identical_canonical_ordering():
    source = _session()
    rows = ((5, 6, 7, 8), (1, 2, 3, 4))
    source.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=rows),
        roles=TRANSFER_ROLES,
        facts=[
            _fact(rows[0], _record(source="z"), _record(source="a")),
            _fact(rows[1], _record(source="y"), _record(source="b")),
        ],
    )
    first = source.export_relation_with_provenance("transfer")["manifest"]
    second = source.export_relation_with_provenance("transfer")["manifest"]
    assert first == second
    assert [fact["cells"] for fact in first["facts"]] == [
        fact["cells"] for fact in second["facts"]
    ]
    assert [record["source"] for record in first["facts"][0]["provenance"]] == [
        "b",
        "y",
    ]
    assert _exported_rows(source, "transfer") == sorted(rows)


def test_manifest_export_capsules_outlive_their_source_session_until_import():
    source, expected, exported = _transfer_manifest_export()
    del source
    gc.collect()
    torch.cuda.synchronize()
    fresh = _session()
    assert (
        fresh.put_relation_from_manifest(
            "transfer", exported["columns"], exported["manifest"]
        )
        == expected
    )
    assert _exported_rows(fresh, "transfer") == [(1, 2, 3, 4)]


def test_manifest_and_legacy_put_ignore_untrusted_sequence_length_hints():
    source, _snapshot, exported = _transfer_manifest_export()
    exported_tensors = [torch.from_dlpack(column) for column in exported["columns"]]
    fresh = _session()
    reconstructed = fresh.put_relation_from_manifest(
        "transfer",
        _LiesAboutLength(exported_tensors),
        exported["manifest"],
    )
    assert reconstructed["row_count"] == 1
    assert _exported_rows(fresh, "transfer") == [(1, 2, 3, 4)]

    legacy = _session()
    legacy.put_relation(
        "transfer",
        _LiesAboutLength(_transfer_columns(rows=((5, 6, 7, 8),))),
    )
    assert _exported_rows(legacy, "transfer") == [(5, 6, 7, 8)]
    assert _exported_rows(source, "transfer") == [(1, 2, 3, 4)]

    batch = _session()
    batch.apply_relation_delta_batch(
        _LiesAboutLength(
            [
                {
                    "name": "transfer",
                    "insert_columns": _transfer_columns(rows=((9, 10, 11, 12),)),
                }
            ]
        )
    )
    assert _exported_rows(batch, "transfer") == [(9, 10, 11, 12)]

    debug = _session()
    result = debug.apply_relation_delta_debug(
        _LiesAboutLength(
            [
                {
                    "name": "transfer",
                    "insert_columns": _transfer_columns(rows=((13, 14, 15, 16),)),
                }
            ]
        ),
        check_equivalence=True,
    )
    assert result["equivalent_to_full_recompute"] is True
    assert _exported_rows(debug, "transfer") == [(13, 14, 15, 16)]


def test_structured_and_manifest_sequences_ignore_untrusted_iterator_length_hints():
    row = (1, 2, 3, 4)
    structured = _session()
    structured.put_relation_with_provenance(
        "transfer",
        _LiesAboutIteratorLength(_transfer_columns(rows=(row,))),
        roles=_LiesAboutIteratorLength(TRANSFER_ROLES),
        facts=_LiesAboutIteratorLength(
            [
                {
                    "tuple": _LiesAboutIteratorLength(row),
                    "provenance": _LiesAboutIteratorLength([_record()]),
                }
            ]
        ),
    )
    assert _exported_rows(structured, "transfer") == [row]

    source, expected, exported = _transfer_manifest_export()
    manifest = copy.deepcopy(exported["manifest"])
    manifest["roles"] = _LiesAboutIteratorLength(manifest["roles"])
    manifest["facts"] = _LiesAboutIteratorLength(manifest["facts"])
    for fact in manifest["facts"]:
        fact["cells"] = _LiesAboutIteratorLength(fact["cells"])
        fact["provenance"] = _LiesAboutIteratorLength(fact["provenance"])
    restored = _session()
    assert (
        restored.put_relation_from_manifest(
            "transfer",
            _LiesAboutIteratorLength(exported["columns"]),
            manifest,
        )
        == expected
    )
    assert _exported_rows(restored, "transfer") == [row]
    assert _exported_rows(source, "transfer") == [row]

    metadata_free_source = _session()
    metadata_free_source.put_relation(
        "transfer", _transfer_columns(rows=((5, 6, 7, 8),))
    )
    metadata_free = metadata_free_source.export_relation_with_provenance("transfer")
    metadata_free["manifest"]["roles"] = _LiesAboutIteratorLength([])
    metadata_free["manifest"]["facts"] = _LiesAboutIteratorLength([])
    metadata_free_target = _session()
    metadata_free_target.put_relation_from_manifest(
        "transfer",
        _LiesAboutIteratorLength(metadata_free["columns"]),
        metadata_free["manifest"],
    )
    assert _exported_rows(metadata_free_target, "transfer") == [(5, 6, 7, 8)]


def test_manifest_import_feeds_a_derived_high_arity_gpu_rule_without_host_rows():
    program = """
    domain party: u32.
    domain asset: u32.
    pred transfer(giver: party, receiver: party, asset: asset, time: i64).
    pred observed(giver: party, receiver: party, asset: asset, time: i64).
    observed(Giver, Receiver, Asset, Time) :- transfer(Giver, Receiver, Asset, Time).
    ?- observed(Giver, Receiver, Asset, Time).
    """
    source = _session(program)
    rows = ((1, 2, 3, 4), (5, 6, 7, 8))
    source.put_relation_with_provenance(
        "transfer",
        _transfer_columns(rows=rows),
        roles=TRANSFER_ROLES,
        facts=[_fact(rows[0])],
    )
    exported = source.export_relation_with_provenance("transfer")
    fresh = _session(program)
    fresh.put_relation_from_manifest(
        "transfer", exported["columns"], exported["manifest"]
    )
    fresh.reset_host_transfer_stats()
    query = fresh.evaluate().queries[0]
    assert query.relation_name == "__xlog_query_0"
    assert all(pyxlog.dlpack_is_cuda(column) for column in query.tensors)
    assert fresh.host_transfer_stats()["dtoh_bytes"] == 0
    values = [torch.from_dlpack(column).cpu().tolist() for column in query.tensors]
    assert sorted(zip(*values)) == sorted(rows)


def test_manifest_exact_cells_round_trip_nan_signed_zero_u64_and_bool():
    program = "pred exact_value(single: f32, double: f64, wide: u64, flag: bool)."
    nan_bits = 0x7FC01234
    wide = 2**63 + 5
    columns = [
        torch.tensor([nan_bits], device="cuda", dtype=torch.uint32).view(torch.float32),
        torch.tensor([-0.0], device="cuda", dtype=torch.float64),
        torch.tensor([wide], device="cuda", dtype=torch.uint64),
        torch.tensor([True], device="cuda", dtype=torch.bool),
    ]
    cells = [
        {"type": "f32", "hex": struct.pack("<I", nan_bits).hex()},
        {"type": "f64", "hex": struct.pack("<d", -0.0).hex()},
        {"type": "u64", "hex": struct.pack("<Q", wide).hex()},
        {"type": "bool", "hex": "01"},
    ]
    source = _session(program)
    expected = source.put_relation_with_provenance(
        "exact_value",
        columns,
        roles=[
            {"name": "single"},
            {"name": "double"},
            {"name": "wide"},
            {"name": "flag"},
        ],
        facts=[{"cells": cells, "provenance": [_record(source="exact")]}],
    )
    exported = source.export_relation_with_provenance("exact_value")
    assert exported["manifest"]["facts"][0]["cells"] == cells
    fresh = _session(program)
    reconstructed = fresh.put_relation_from_manifest(
        "exact_value", exported["columns"], exported["manifest"]
    )
    assert reconstructed["relation"] == expected["relation"]
    assert reconstructed["metadata_present"] == expected["metadata_present"]
    assert reconstructed["row_count"] == expected["row_count"]
    assert reconstructed["roles"] == expected["roles"]
    reconstructed_fact = reconstructed["facts"][0]
    expected_fact = expected["facts"][0]
    assert reconstructed_fact["identity"] == expected_fact["identity"]
    assert reconstructed_fact["cells"] == expected_fact["cells"] == cells
    assert reconstructed_fact["provenance"] == expected_fact["provenance"]
    assert math.isnan(reconstructed_fact["tuple"][0])
    assert struct.pack("<d", reconstructed_fact["tuple"][1]) == struct.pack("<d", -0.0)
    assert reconstructed_fact["tuple"][2:] == [wide, True]

    relation_columns = [
        torch.from_dlpack(column) for column in fresh.export_relation("exact_value")
    ]
    assert relation_columns[0].view(torch.uint32).cpu().tolist() == [nan_bits]
    assert struct.pack("<d", relation_columns[1].cpu().item()) == struct.pack("<d", -0.0)
    assert relation_columns[2].cpu().tolist() == [wide]
    assert relation_columns[3].cpu().tolist() == [True]


def test_manifest_nullary_export_and_import_are_explicitly_rejected():
    session = _session("pred ready().")
    session.put_relation("ready", [])
    with pytest.raises(_metadata_error(), match="positive arity"):
        session.export_relation_with_provenance("ready")
    with pytest.raises(_metadata_error(), match="positive arity"):
        session.put_relation_from_manifest("ready", [], {})
