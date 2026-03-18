# python/tests/test_ilp_d2h_gate.py
"""Tests for D2H transfer counter exposed via PyO3."""
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")
import pyxlog.ilp as ilp

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W_reach) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
"""


def _compile_reach_prog_with_active_mask():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    mask_hard = torch.zeros(n, n, n, device="cuda")
    mask_soft = torch.zeros(n, n, n, device="cuda")
    mask_hard[i_edge, i_edge, k_reach] = 1.0
    mask_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", mask_hard.view(-1), mask_soft.view(-1), n)
    prog.evaluate()
    return prog


def test_d2h_counter_accessible():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    assert prog.d2h_transfer_count() == 0


def test_compiled_ilp_program_does_not_expose_public_read_device_helpers():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    assert not hasattr(prog, "read_device_i64_scalar")
    assert not hasattr(prog, "read_device_bool_scalar")
    assert not hasattr(prog, "read_device_i64_list")
    assert not hasattr(prog, "set_rule_mask_sparse_selected_device_trusted")


def test_d2h_counter_reset():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    # fact_exists triggers D2H transfers
    prog.evaluate()
    prog.fact_exists("edge", [1, 2])
    assert prog.d2h_transfer_count() > 0

    prog.reset_d2h_transfer_count()
    assert prog.d2h_transfer_count() == 0


def test_strict_zero_dtoh_rejects_host_sparse_selected_api():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.set_strict_zero_dtoh(True)
    soft = torch.tensor([1.0], device="cuda", dtype=torch.float64)

    with pytest.raises(RuntimeError, match="strict_zero_dtoh"):
        prog.set_rule_mask_sparse_selected("W_reach", [0], soft, False)


def test_batch_fact_membership_basic():
    """batch_fact_membership returns correct bool mask."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    # edge relation has: (1,2), (2,3), (3,4)
    facts = [[1, 2], [5, 6], [2, 3]]
    mask = prog.batch_fact_membership("edge", facts)
    assert mask == [True, False, True]


def test_batch_fact_membership_empty_facts():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()
    mask = prog.batch_fact_membership("edge", [])
    assert mask == []


def test_batch_fact_membership_no_d2h_columns():
    """batch_fact_membership must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 2], [5, 6], [2, 3]]
    _ = prog.batch_fact_membership("edge", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_fact_membership triggered {prog.d2h_transfer_count()} column downloads"
    )


def test_batch_tagged_credit_basic():
    """batch_tagged_credit returns per-fact lists of (i,j,k) contributors."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    # Set mask so that edge+edge->reach is active
    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    # reach should derive (1,3), (2,4) via edge(X,Z) join edge(Z,Y)
    facts = [[1, 3], [2, 4], [99, 99]]
    credits = prog.batch_tagged_credit("reach", facts)
    assert len(credits) == 3

    # First two should have at least one contributing entry
    assert len(credits[0]) > 0, f"(1,3) should be derived, got empty credits"
    assert len(credits[1]) > 0, f"(2,4) should be derived, got empty credits"
    # The contributing entry should be (edge_idx, edge_idx, reach_idx)
    assert credits[0][0] == (i_edge, i_edge, k_reach)
    # (99,99) is not derived -- no contributors
    assert credits[2] == []


def test_batch_tagged_credit_no_d2h_columns():
    """batch_tagged_credit must NOT use download_column_*."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    M_hard = torch.zeros(n, n, n, device="cuda")
    M_soft = torch.zeros(n, n, n, device="cuda")
    M_hard[i_edge, i_edge, k_reach] = 1.0
    M_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", M_hard.view(-1), M_soft.view(-1), n)
    prog.evaluate()

    prog.reset_d2h_transfer_count()
    facts = [[1, 3], [2, 4], [99, 99]]
    _ = prog.batch_tagged_credit("reach", facts)
    assert prog.d2h_transfer_count() == 0, (
        f"batch_tagged_credit triggered {prog.d2h_transfer_count()} column downloads"
    )


from pyxlog.ilp import train_only, TrainConfig


def test_zero_d2h_in_training_step_loop():
    """The training step loop must have zero download_column_* calls."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4])],
        negatives=[],
        config=config,
    )
    assert isinstance(result.total_steps, int)
    assert result.total_steps > 0


def test_full_training_zero_d2h_gate():
    """Full training run must complete without D2H gate violation."""
    source = """
        edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
        learnable(W_reach) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=True,
    )
    result = train_only(
        source=source, mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4]),
                   ("reach", [3, 5]), ("reach", [4, 6])],
        negatives=[],
        config=config,
    )
    # If we got here without IlpTrainingError("d2h_gate_violation"),
    # the hard gate passed.
    strict_result_type = getattr(ilp, "StrictTrainResult", None)
    strict_artifact_type = getattr(ilp, "StrictLearnedArtifact", None)
    assert strict_result_type is not None
    assert strict_artifact_type is not None
    assert isinstance(result, strict_result_type)
    assert isinstance(result.artifact, strict_artifact_type)
    assert result.strict_gpu_native is True
    assert result.compat_materialized is False
    assert not hasattr(result, "converged")
    assert not hasattr(result, "precision")
    assert not hasattr(result, "recall")
    assert not hasattr(result, "rule_frequency")
    assert not hasattr(result, "discovered_rule")
    compat_artifact = result.artifact.export_compat_artifact()
    compat = result.export_compat_result()
    assert compat.converged
    assert compat.discovered_rule is not None
    assert "edge" in compat.discovered_rule
    assert compat_artifact.discovered_rule is not None
    assert "edge" in compat_artifact.discovered_rule


def test_compiled_relation_training_zero_d2h_gate():
    train_on_compiled_relations = getattr(ilp, "train_on_compiled_relations", None)
    assert train_on_compiled_relations is not None, (
        "pyxlog.ilp.train_on_compiled_relations must be exported for strict relation-native training"
    )

    source = """
        pred edge(u32, u32).
        pred reach(u32, u32).
        learnable(W_reach) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=64)
    prog.put_relation(
        "edge",
        [
            torch.tensor([1, 2, 3, 4, 5], device="cuda", dtype=torch.int32),
            torch.tensor([2, 3, 4, 5, 6], device="cuda", dtype=torch.int32),
        ],
    )

    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        tau_start=2.0,
        tau_floor=0.05,
        seed=42,
        strict_gpu_native=True,
    )
    result = train_on_compiled_relations(
        prog,
        "W_reach",
        {
            "reach": [
                torch.tensor([1, 2, 3, 4], device="cuda", dtype=torch.int32),
                torch.tensor([3, 4, 5, 6], device="cuda", dtype=torch.int32),
            ],
        },
        {},
        config,
    )

    strict_result_type = getattr(ilp, "StrictTrainResult", None)
    strict_artifact_type = getattr(ilp, "StrictLearnedArtifact", None)
    assert strict_result_type is not None
    assert strict_artifact_type is not None
    assert isinstance(result, strict_result_type)
    assert isinstance(result.artifact, strict_artifact_type)
    assert result.strict_gpu_native is True
    assert result.compat_materialized is False
    with pytest.raises(RuntimeError, match="no compatibility exporter"):
        result.export_compat_result()


def test_d2h_gate_with_negatives():
    """D2H gate holds even with negative examples."""
    source = """
        parent(1, 2). parent(2, 3). parent(3, 4). parent(4, 5).
        gender(1, 0). gender(2, 1). gender(3, 0). gender(4, 1).
        sibling(1, 3). sibling(3, 1).
        learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    config = TrainConfig(
        step_budget_per_attempt=60, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=True,
    )
    result = train_only(
        source=source, mask_name="W_gp",
        positives=[("grandparent", [1, 3]), ("grandparent", [2, 4])],
        negatives=[("grandparent", [1, 2]), ("grandparent", [3, 1])],
        config=config,
    )
    # No d2h_gate_violation raised
    strict_result_type = getattr(ilp, "StrictTrainResult", None)
    strict_artifact_type = getattr(ilp, "StrictLearnedArtifact", None)
    assert strict_result_type is not None
    assert strict_artifact_type is not None
    assert isinstance(result, strict_result_type)
    assert isinstance(result.artifact, strict_artifact_type)
    assert result.strict_gpu_native is True
    assert result.compat_materialized is False
    assert not hasattr(result, "converged")
    assert not hasattr(result, "precision")
    assert not hasattr(result, "recall")
    assert not hasattr(result, "discovered_rule")
    compat_artifact = result.artifact.export_compat_artifact()
    compat = result.export_compat_result()
    assert compat.converged
    assert compat.discovered_rule is not None
    assert "parent" in compat.discovered_rule
    assert compat_artifact.discovered_rule is not None
    assert "parent" in compat_artifact.discovered_rule


def test_host_transfer_stats_methods_accessible():
    """host_transfer_stats and reset_host_transfer_stats should be available and reset correctly."""
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    keys = prog.host_transfer_stats().keys()
    assert set(keys) >= {"dtoh_bytes", "htod_bytes", "dtoh_calls", "htod_calls"}

    prog.evaluate()
    prog.reset_host_transfer_stats()
    post_reset = prog.host_transfer_stats()
    assert post_reset["dtoh_bytes"] == 0
    assert post_reset["htod_bytes"] == 0
    assert post_reset["dtoh_calls"] == 0
    assert post_reset["htod_calls"] == 0

    _ = prog.batch_fact_membership("edge", [[1, 2], [99, 99]])
    after = prog.host_transfer_stats()
    assert after["htod_calls"] > 0, "batch_fact_membership should track host transfers"


def test_ilp_registry_routes_row_count_reads_through_provider_api():
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "crates/xlog-runtime/src/ilp_registry.rs").read_text()

    assert "provider.device_row_count(buffer)" in src
    assert ".dtoh_sync_copy_into(buffer.num_rows_device()" not in src


@pytest.mark.parametrize(
    ("label", "invoker"),
    [
        ("fact_exists", lambda prog: prog.fact_exists("edge", [1, 2])),
        ("relation_facts", lambda prog: prog.relation_facts("edge")),
        ("batch_fact_membership", lambda prog: prog.batch_fact_membership("edge", [[1, 2], [9, 9]])),
        ("batch_tagged_credit", lambda prog: prog.batch_tagged_credit("reach", [[1, 3], [9, 9]])),
        (
            "sample_false_positives",
            lambda prog: prog.sample_false_positives("reach", [("reach", [1, 3])], 1),
        ),
    ],
)
def test_strict_zero_dtoh_rejects_host_semantic_runtime_apis(label, invoker):
    prog = _compile_reach_prog_with_active_mask()
    prog.set_strict_zero_dtoh(True)

    with pytest.raises(RuntimeError, match="strict_zero_dtoh"):
        invoker(prog)
