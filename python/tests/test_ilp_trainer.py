# python/tests/test_ilp_trainer.py
"""Integration tests for train_only()."""
import inspect
import math

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

import pyxlog.ilp as ilp
from pyxlog.ilp import (
    train_only,
    TrainConfig,
    TrainResult,
    LearnedArtifact,
    IlpConfigError,
)
from pyxlog.ilp import trainer as trainer_mod

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
REACH_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
REACH_NEG: list[tuple[str, list[int]]] = []
REACH_SESSION_SOURCE = """
    pred edge(u32, u32).
    pred reach(u32, u32).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def _u32_columns(left: list[int], right: list[int]) -> "list[torch.Tensor]":  # type: ignore[name-defined]
    return [
        torch.tensor(left, device="cuda", dtype=torch.int32),
        torch.tensor(right, device="cuda", dtype=torch.int32),
    ]


def _reach_relation_batches() -> "tuple[dict[str, list[torch.Tensor]], dict[str, list[torch.Tensor]]]":  # type: ignore[name-defined]
    positives = {
        "reach": _u32_columns([1, 2, 3, 4], [3, 4, 5, 6]),
    }
    negatives = {
        "reach": _u32_columns([1, 1], [4, 5]),
    }
    return positives, negatives


def test_grouped_membership_mask_device_stays_on_cuda():
    class _FakeProg:
        def __init__(self):
            self.calls = []

        def batch_fact_membership_device(self, relation, facts):
            self.calls.append((relation, facts))
            if relation == "reach":
                return torch.tensor([True, False], device="cuda", dtype=torch.bool)
            if relation == "edge":
                return torch.tensor([False], device="cuda", dtype=torch.bool)
            raise AssertionError(f"unexpected relation {relation}")

    positives = [
        ("reach", [1, 3]),
        ("edge", [1, 2]),
        ("reach", [2, 4]),
    ]
    device = torch.device("cuda")
    facts_by_rel, indices_by_rel = trainer_mod._prepare_relation_groups(positives, device)
    prog = _FakeProg()

    mask = trainer_mod._compute_grouped_membership_mask_device(
        prog, facts_by_rel, indices_by_rel, len(positives), device,
    )

    assert prog.calls == [
        ("reach", [[1, 3], [2, 4]]),
        ("edge", [[1, 2]]),
    ]
    assert mask.device.type == "cuda"
    assert mask.dtype == torch.bool
    assert mask.cpu().tolist() == [True, False, False]


def test_check_convergence_uses_device_membership_api_only():
    class _FakeProg:
        def batch_fact_membership(self, relation, facts):
            raise AssertionError("legacy host membership API must not be used")

        def batch_fact_membership_device(self, relation, facts):
            if relation == "reach" and facts == [[1, 3], [2, 4]]:
                return torch.tensor([True, True], device="cuda", dtype=torch.bool)
            if relation == "reach" and facts == [[9, 9]]:
                return torch.tensor([False], device="cuda", dtype=torch.bool)
            raise AssertionError(f"unexpected relation/facts: {relation} {facts}")

        def set_rule_mask(self, *_args, **_kwargs):
            return None

        def evaluate(self):
            return None

        def reset_d2h_transfer_count(self):
            return None

    prog = _FakeProg()
    W = torch.zeros((4, 4, 4), device="cuda")
    converged = trainer_mod._check_convergence(
        prog=prog,
        W=W,
        mask_name="W_reach",
        positives=[("reach", [1, 3]), ("reach", [2, 4])],
        negatives=[("reach", [9, 9])],
        rel_names=["edge", "reach", "foo", "bar"],
        n=4,
        argmax_ijk=(0, 0, 1),
        perform_mask_checks=True,
    )
    assert converged is True


def test_trainer_uses_device_membership_helper_in_hot_loop():
    src = inspect.getsource(trainer_mod._compute_grouped_membership_mask_device)
    assert ".batch_fact_membership_device(" in src
    assert ".batch_fact_membership(" not in src


def test_run_single_attempt_avoids_item_scalar_syncs_in_hot_loop():
    src = inspect.getsource(trainer_mod._run_single_attempt)
    assert ".item()" not in src


def test_train_only_strict_gpu_native_rejects_host_negative_mining():
    config = TrainConfig(
        step_budget_per_attempt=10,
        max_attempts=1,
        max_mined_negatives=1,
        strict_gpu_native=True,
        seed=42,
    )
    with pytest.raises(IlpConfigError, match="host negative mining"):
        train_only(
            source=REACH_SOURCE,
            mask_name="W_reach",
            positives=REACH_POS,
            negatives=REACH_NEG,
            config=config,
        )


def test_train_on_compiled_relations_strict_gpu_native_accepts_uploaded_relations_without_host_sync_traps(monkeypatch):
    train_on_compiled_relations = getattr(ilp, "train_on_compiled_relations", None)
    assert train_on_compiled_relations is not None, (
        "pyxlog.ilp.train_on_compiled_relations must be exported for strict relation-native training"
    )

    prog = pyxlog.IlpProgramFactory.compile(REACH_SESSION_SOURCE, device=0, memory_mb=256)
    assert hasattr(prog, "put_relation"), (
        "CompiledIlpProgram.put_relation must exist for strict relation-native training"
    )
    prog.put_relation("edge", _u32_columns([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))

    positives, negatives = _reach_relation_batches()
    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        tau_start=2.0,
        tau_floor=0.05,
        strict_gpu_native=True,
        seed=42,
    )

    orig_cpu = torch.Tensor.cpu
    orig_tolist = torch.Tensor.tolist
    orig_item = torch.Tensor.item

    def _trap_cpu(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_cpu(self, *args, **kwargs)
        raise AssertionError("unexpected host sync via Tensor.cpu on relation/example tensors")

    def _trap_tolist(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_tolist(self, *args, **kwargs)
        raise AssertionError("unexpected host sync via Tensor.tolist on relation/example tensors")

    def _trap_item(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_item(self, *args, **kwargs)
        raise AssertionError("unexpected host sync via Tensor.item on relation/example tensors")

    monkeypatch.setattr(torch.Tensor, "cpu", _trap_cpu, raising=False)
    monkeypatch.setattr(torch.Tensor, "tolist", _trap_tolist, raising=False)
    monkeypatch.setattr(torch.Tensor, "item", _trap_item, raising=False)

    result = train_on_compiled_relations(
        prog,
        "W_reach",
        positives,
        negatives,
        config,
    )

    strict_result_type = getattr(ilp, "StrictTrainResult", None)
    assert strict_result_type is not None
    assert isinstance(result, strict_result_type)
    assert result.strict_gpu_native is True
    assert result.compat_materialized is False
    assert result.total_steps > 0
    assert len(result.artifact.candidate_map) > 0
    assert isinstance(result.winner_candidate_id, int)
    assert isinstance(result.discovered_rule, str)


def test_train_on_compiled_relations_matches_inline_fact_candidate_surface():
    train_on_compiled_relations = getattr(ilp, "train_on_compiled_relations", None)
    assert train_on_compiled_relations is not None, (
        "pyxlog.ilp.train_on_compiled_relations must be exported for strict relation-native training"
    )

    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        tau_start=2.0,
        tau_floor=0.05,
        strict_gpu_native=True,
        seed=42,
    )

    prog = pyxlog.IlpProgramFactory.compile(REACH_SESSION_SOURCE, device=0, memory_mb=256)
    prog.put_relation("edge", _u32_columns([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    positives, negatives = _reach_relation_batches()
    relation_result = train_on_compiled_relations(
        prog,
        "W_reach",
        positives,
        negatives,
        config,
    )

    inline_source = """
        pred edge(u32, u32).
        pred reach(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
        learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """
    inline_result = train_only(
        inline_source,
        "W_reach",
        REACH_POS,
        [("reach", [1, 4]), ("reach", [1, 5])],
        config,
    )

    assert relation_result.strict_gpu_native is True
    assert inline_result.strict_gpu_native is True
    assert len(relation_result.artifact.candidate_map) == len(inline_result.artifact.candidate_map)
    assert relation_result.attempt_count == inline_result.attempt_count
    assert relation_result.total_steps == inline_result.total_steps


def test_train_on_compiled_relations_strict_result_exposes_winner_metadata():
    train_on_compiled_relations = getattr(ilp, "train_on_compiled_relations", None)
    assert train_on_compiled_relations is not None

    prog = pyxlog.IlpProgramFactory.compile(REACH_SESSION_SOURCE, device=0, memory_mb=256)
    prog.put_relation("edge", _u32_columns([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
    positives, negatives = _reach_relation_batches()
    result = train_on_compiled_relations(
        prog,
        "W_reach",
        positives,
        negatives,
        TrainConfig(
            step_budget_per_attempt=20,
            max_attempts=1,
            tau_start=2.0,
            tau_floor=0.05,
            strict_gpu_native=True,
            seed=42,
        ),
    )

    winner_entry = next(
        entry for entry in result.artifact.candidate_map
        if entry.id == result.winner_candidate_id
    )

    assert isinstance(result.winner_candidate_id, int)
    assert result.discovered_rule == trainer_mod._format_rule(
        winner_entry.left_name,
        winner_entry.right_name,
        winner_entry.head_name,
    )


def test_train_on_compiled_relations_winner_metadata_is_deterministic_under_fixed_seed():
    train_on_compiled_relations = getattr(ilp, "train_on_compiled_relations", None)
    assert train_on_compiled_relations is not None

    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        tau_start=2.0,
        tau_floor=0.05,
        strict_gpu_native=True,
        seed=42,
        deterministic=True,
    )

    def _run_once():
        prog = pyxlog.IlpProgramFactory.compile(REACH_SESSION_SOURCE, device=0, memory_mb=256)
        prog.put_relation("edge", _u32_columns([1, 2, 3, 4, 5], [2, 3, 4, 5, 6]))
        positives, negatives = _reach_relation_batches()
        return train_on_compiled_relations(prog, "W_reach", positives, negatives, config)

    a = _run_once()
    b = _run_once()

    assert a.winner_candidate_id == b.winner_candidate_id
    assert a.discovered_rule == b.discovered_rule


def test_train_only_converges_on_reach():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert isinstance(result, TrainResult)
    assert result.converged
    assert "edge" in result.discovered_rule
    assert "reach" in result.discovered_rule
    assert result.attempt_count >= 1
    assert result.total_steps > 0


def test_train_only_returns_telemetry_level_1():
    config = TrainConfig(
        step_budget_per_attempt=50, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        telemetry_level=1,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert len(result.artifact.telemetry.steps) > 0
    step0 = result.artifact.telemetry.steps[0]
    assert step0.step == 0
    assert isinstance(step0.loss, float)
    assert isinstance(step0.temperature, float)


def test_train_only_forward_p95_telemetry_populated():
    """Telemetry must include forward-pass timing in microseconds."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=13,
        telemetry_level=1,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    steps = result.artifact.telemetry.steps
    assert len(steps) > 0
    step_forward = [s.forward_p95_us for s in steps]
    assert all(v >= 0.0 for v in step_forward)
    assert max(step_forward) >= 0.0

    timing = result.artifact.telemetry.step_timings
    assert isinstance(timing, dict)
    assert "forward_p95_us" in timing
    sorted_forward = sorted(step_forward)
    p95_idx = min(len(sorted_forward) - 1, max(0, math.ceil(0.95 * len(sorted_forward)) - 1))
    assert timing["forward_p95_us"] >= sorted_forward[p95_idx] - 1e-6


def test_train_only_telemetry_step_timings_population():
    """Telemetry step summary should include forward and memory stats."""
    config = TrainConfig(
        step_budget_per_attempt=40, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=13,
        telemetry_level=1,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    timing = result.artifact.telemetry.step_timings
    assert isinstance(timing, dict)
    assert "forward_p95_us" in timing
    assert timing["forward_p95_us"] >= 0.0
    assert "allocated_bytes_p95" in timing
    assert "allocated_bytes_max" in timing


def test_train_only_empty_positives_raises():
    config = TrainConfig(max_attempts=1)
    with pytest.raises(IlpConfigError, match="positives"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=[], negatives=[], config=config,
        )


def test_train_only_contradictory_examples_raises():
    config = TrainConfig(max_attempts=1)
    pos = [("reach", [1, 3])]
    neg = [("reach", [1, 3])]
    with pytest.raises(IlpConfigError, match="contradict"):
        train_only(
            source=REACH_SOURCE, mask_name="W_reach",
            positives=pos, negatives=neg, config=config,
        )


def test_train_only_precision_recall():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert result.precision > 0.0
        assert result.recall > 0.0


def test_train_only_confidence_metrics():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        assert 0.0 <= result.confidence_margin <= 1.0
        assert 0.0 <= result.top_k_concentration <= 1.0 + 1e-6
        assert 0.0 <= result.rule_frequency <= 1.0


def test_train_only_global_step_limit():
    config = TrainConfig(
        global_step_limit=50, step_budget_per_attempt=30,
        max_attempts=10, tau_start=2.0, tau_floor=0.05, seed=42,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    assert result.total_steps <= 50


def test_train_only_nonconverged_has_partial_recall():
    """Non-converged result should report partial recall, not 0.0."""
    # Mix reachable + impossible positive → forces non-convergence
    pos_mixed = [("reach", [1, 3]), ("reach", [99, 99])]
    config = TrainConfig(
        step_budget_per_attempt=30, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=pos_mixed, negatives=REACH_NEG, config=config,
    )
    assert not result.converged
    # reach(1,3) is derivable from edge joins, so recall should be > 0
    assert result.recall > 0.0, "Non-converged recall should reflect partial witness coverage"


def test_train_only_reproducibility_selected_hard_and_probs():
    """Deterministic mode stabilizes learned artifacts across runs."""
    config = TrainConfig(
        step_budget_per_attempt=120,
        max_attempts=4,
        tau_start=2.0,
        tau_floor=0.05,
        seed=123,
        deterministic=True,
        debug_dense_mask=False,
        strict_gpu_native=False,
    )
    a = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=config,
    )
    b = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=config,
    )

    assert a.discovered_rule == b.discovered_rule
    assert a.artifact.selected_hard == b.artifact.selected_hard
    assert len(a.artifact.soft_probs) == len(b.artifact.soft_probs)
    assert a.converged is True
    assert b.converged is True
    for p, q in zip(a.artifact.soft_probs, b.artifact.soft_probs):
        assert abs(p - q) <= max(1e-6, abs(q) * 1e-5 + 1e-6)


def test_train_only_telemetry_entropy_not_zero():
    """Telemetry entropy should reflect actual candidate distribution."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=1,
        tau_start=2.0, tau_floor=0.05, seed=42,
        telemetry_level=1,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    steps = result.artifact.telemetry.steps
    assert len(steps) > 0
    # Early steps with high tau should have non-zero entropy
    assert any(s.entropy > 0.0 for s in steps), \
        "At least some telemetry steps should have non-zero entropy"


def test_train_only_phase_timing_keys():
    """Phase timing breakdown must include all expected keys."""
    config = TrainConfig(
        step_budget_per_attempt=20, max_attempts=2,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    timing = result.artifact.telemetry.step_timings
    expected_p95 = [
        "apply_mask_p95_us", "loss_credit_p95_us", "loss_reduce_p95_us",
        "backward_step_p95_us", "membership_p95_us", "convergence_p95_us",
    ]
    expected_total = [
        "apply_mask_total_ms", "loss_credit_total_ms", "loss_reduce_total_ms",
        "backward_step_total_ms", "membership_total_ms", "convergence_total_ms",
    ]
    for key in expected_p95 + expected_total:
        assert key in timing, f"Missing timing key: {key}"
        assert timing[key] >= 0.0, f"Negative timing: {key}={timing[key]}"


def test_train_only_rule_frequency_multi_attempt():
    """rule_frequency reflects how many attempts found the winning rule."""
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=3,
        tau_start=2.0, tau_floor=0.05, seed=42,
        strict_gpu_native=False,
    )
    result = train_only(
        source=REACH_SOURCE, mask_name="W_reach",
        positives=REACH_POS, negatives=REACH_NEG, config=config,
    )
    if result.converged:
        # At least the winning attempt found this rule
        assert result.rule_frequency >= 1.0 / result.attempt_count


def test_train_only_strict_gpu_native_rejects_telemetry_collection():
    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        telemetry_level=1,
        strict_gpu_native=True,
        seed=42,
    )
    with pytest.raises(IlpConfigError, match="telemetry"):
        train_only(
            source=REACH_SOURCE,
            mask_name="W_reach",
            positives=REACH_POS,
            negatives=REACH_NEG,
            config=config,
        )


def test_train_only_strict_gpu_native_rejects_dense_mask_backend():
    config = TrainConfig(
        step_budget_per_attempt=20,
        max_attempts=1,
        debug_dense_mask=True,
        strict_gpu_native=True,
        seed=42,
    )
    with pytest.raises(IlpConfigError, match="dense"):
        train_only(
            source=REACH_SOURCE,
            mask_name="W_reach",
            positives=REACH_POS,
            negatives=REACH_NEG,
            config=config,
        )


def test_strict_single_attempt_helper_avoids_host_sync_primitives():
    src = inspect.getsource(trainer_mod._run_single_attempt_strict)
    assert ".cpu(" not in src
    assert ".item()" not in src
    assert "torch.cuda.synchronize()" not in src


def test_strict_finalization_helpers_avoid_python_host_sync_primitives():
    helper_sources = [
        inspect.getsource(trainer_mod._finalize_strict_attempt),
        inspect.getsource(trainer_mod._build_strict_train_result),
    ]
    strict_src = "\n".join(helper_sources)
    assert ".cpu(" not in strict_src
    assert ".item()" not in strict_src
    assert ".tolist()" not in strict_src
    assert ".numpy()" not in strict_src


def test_select_better_strict_attempt_updates_argmax_device_selection():
    entry = trainer_mod.CandidateMapEntry(
        id=0, i=0, j=0, k=0,
        left_name="edge", right_name="edge", head_name="reach",
    )
    best = trainer_mod._StrictAttemptState(
        candidate_map=[entry],
        telemetry_steps=[],
        telemetry_timings={"winner": "best"},
        steps_used=1,
        W_device=torch.tensor([9.0, 1.0]),
        cand_probs_device=torch.tensor([0.9, 0.1]),
        score_loss_device=torch.tensor(2.0),
        score_top1_device=torch.tensor(0.9),
        selected_candidate_ids_device=torch.tensor([0], dtype=torch.int64),
        argmax_candidate_id_device=torch.tensor([0], dtype=torch.int64),
        attempt_id_device=torch.tensor([0], dtype=torch.int64),
        rel_names=["edge", "reach"],
        candidates=[{"i": 0, "j": 0, "k": 0}, {"i": 1, "j": 1, "k": 1}],
        n=2,
        allow_recursive=False,
    )
    candidate = trainer_mod._StrictAttemptState(
        candidate_map=[entry],
        telemetry_steps=[
            trainer_mod.StepRecord(
                step=4,
                loss=1.0,
                argmax_rule="edge+edge->reach",
                discreteness=0.9,
                temperature=0.5,
                entropy=0.1,
                stable_count=2,
            )
        ],
        telemetry_timings={"winner": "candidate"},
        steps_used=5,
        W_device=torch.tensor([1.0, 10.0]),
        cand_probs_device=torch.tensor([0.1, 0.9]),
        score_loss_device=torch.tensor(1.0),
        score_top1_device=torch.tensor(0.9),
        selected_candidate_ids_device=torch.tensor([1], dtype=torch.int64),
        argmax_candidate_id_device=torch.tensor([1], dtype=torch.int64),
        attempt_id_device=torch.tensor([1], dtype=torch.int64),
        rel_names=["edge", "reach"],
        candidates=[{"i": 0, "j": 0, "k": 0}, {"i": 1, "j": 1, "k": 1}],
        n=2,
        allow_recursive=False,
    )

    merged = trainer_mod._select_better_strict_attempt(best, candidate)

    assert merged.selected_candidate_ids_device.tolist() == [1]
    assert merged.argmax_candidate_id_device.tolist() == [1]
    assert merged.attempt_id_device.tolist() == [1]


def test_export_compat_attempt_from_strict_state_uses_explicit_winner_metadata(monkeypatch):
    class _FakeProg:
        def reset_runtime(self):
            return None

        def set_rule_mask_sparse_selected(self, *_args, **_kwargs):
            return None

        def evaluate(self):
            return None

    def _noop_metrics(result, *_args, **_kwargs):
        result.confidence_margin = None
        result.top_k_concentration = None
        result.soft_probs = []
        result.logits = []

    monkeypatch.setattr(trainer_mod, "_materialize_compat_witness_coverage", lambda *_args, **_kwargs: 0.5)
    monkeypatch.setattr(trainer_mod, "_check_convergence", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(trainer_mod, "_materialize_compat_metrics", _noop_metrics)

    state = trainer_mod._StrictAttemptState(
        candidate_map=[],
        telemetry_steps=[],
        telemetry_timings={"winner": "stale"},
        steps_used=1,
        W_device=torch.tensor([0.1, 0.9]),
        cand_probs_device=torch.tensor([0.1, 0.9]),
        score_loss_device=torch.tensor(1.0),
        score_top1_device=torch.tensor(0.9),
        selected_candidate_ids_device=torch.tensor([1], dtype=torch.int64),
        argmax_candidate_id_device=torch.tensor([1], dtype=torch.int64),
        attempt_id_device=torch.tensor([7], dtype=torch.int64),
        rel_names=["edge", "reach"],
        candidates=[{"i": 0, "j": 0, "k": 0}, {"i": 1, "j": 1, "k": 1}],
        n=2,
        allow_recursive=False,
    )
    winner_metadata = {
        "steps_used": 5,
        "telemetry_steps": [
            trainer_mod.StepRecord(
                step=4,
                loss=1.0,
                argmax_rule="edge+edge->reach",
                discreteness=0.9,
                temperature=0.5,
                entropy=0.1,
                stable_count=2,
            )
        ],
        "telemetry_timings": {"winner": "candidate"},
    }

    compat_attempt = trainer_mod._export_compat_attempt_from_strict_state(
        _FakeProg(),
        state,
        winner_metadata,
        "W_reach",
        REACH_POS,
        REACH_NEG,
        trainer_mod.TrainConfig(strict_gpu_native=True),
    )

    assert compat_attempt.steps_used == 5
    assert compat_attempt.telemetry_timings == {"winner": "candidate"}
    assert len(compat_attempt.telemetry_steps) == 1
    assert compat_attempt.telemetry_steps[0].step == 4


def test_relation_native_strict_result_uses_actual_argmax_candidate_id():
    best_state = trainer_mod._StrictAttemptState(
        candidate_map=[
            trainer_mod.CandidateMapEntry(
                id=7,
                i=0,
                j=0,
                k=1,
                left_name="edge",
                right_name="edge",
                head_name="reach",
            ),
            trainer_mod.CandidateMapEntry(
                id=3,
                i=1,
                j=1,
                k=2,
                left_name="path",
                right_name="path",
                head_name="reach",
            ),
        ],
        telemetry_steps=[],
        telemetry_timings={},
        steps_used=4,
        W_device=torch.tensor([0.1, 0.9]),
        cand_probs_device=torch.tensor([0.1, 0.9]),
        score_loss_device=torch.tensor(0.2),
        score_top1_device=torch.tensor(0.9),
        selected_candidate_ids_device=torch.tensor([3], dtype=torch.int64),
        argmax_candidate_id_device=torch.tensor([3], dtype=torch.int64),
        attempt_id_device=torch.tensor([0], dtype=torch.int64),
        rel_names=["edge", "path", "reach"],
        candidates=[{"i": 0, "j": 0, "k": 1}, {"i": 1, "j": 1, "k": 2}],
        n=3,
        allow_recursive=False,
    )

    result = trainer_mod._build_relation_native_strict_train_result(
        best_state=best_state,
        config=TrainConfig(strict_gpu_native=True, seed=42),
        global_steps=4,
        attempt_count=1,
    )

    assert result.winner_candidate_id == 3
    assert result.discovered_rule == trainer_mod._format_rule("path", "path", "reach")


def _assert_strict_result_shape(result) -> None:
    strict_result_type = getattr(ilp, "StrictTrainResult", None)
    strict_artifact_type = getattr(ilp, "StrictLearnedArtifact", None)
    assert strict_result_type is not None, "StrictTrainResult must be exported"
    assert strict_artifact_type is not None, "StrictLearnedArtifact must be exported"
    assert isinstance(result, strict_result_type)
    assert isinstance(result.artifact, strict_artifact_type)
    assert result.strict_gpu_native is True
    assert result.compat_materialized is False
    assert hasattr(result, "winner_candidate_id")
    assert hasattr(result, "discovered_rule")
    assert not hasattr(result, "converged")
    assert not hasattr(result, "precision")
    assert not hasattr(result, "recall")
    assert not hasattr(result, "holdout_f1")
    assert not hasattr(result, "holdout_variance")
    assert not hasattr(result, "confidence_margin")
    assert not hasattr(result, "top_k_concentration")
    assert not hasattr(result, "rule_frequency")
    assert not hasattr(result, "ambiguous_alternatives")


def _assert_strict_artifact_shape(artifact) -> None:
    strict_artifact_type = getattr(ilp, "StrictLearnedArtifact", None)
    assert strict_artifact_type is not None, "StrictLearnedArtifact must be exported"
    assert isinstance(artifact, strict_artifact_type)
    assert artifact.strict_gpu_native is True
    assert artifact.compat_materialized is False
    assert not hasattr(artifact, "logits")
    assert not hasattr(artifact, "soft_probs")
    assert not hasattr(artifact, "selected_hard")
    assert not hasattr(artifact, "discovered_rule")


def test_train_only_strict_gpu_native_passes_with_host_sync_traps(monkeypatch):
    orig_cpu = torch.Tensor.cpu
    orig_tolist = torch.Tensor.tolist
    orig_item = torch.Tensor.item

    def _trap_cpu(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_cpu(self, *args, **kwargs)
        raise AssertionError("host materialization called via torch.Tensor.cpu")

    def _trap_item(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_item(self, *args, **kwargs)
        raise AssertionError("host materialization called via torch.Tensor.item")

    def _trap_tolist(self, *args, **kwargs):
        if self.numel() == 1:
            return orig_tolist(self, *args, **kwargs)
        raise AssertionError("host materialization called via torch.Tensor.tolist")

    def _forbid_numpy(*_args, **_kwargs):
        raise AssertionError("host materialization called via torch.Tensor.numpy")

    monkeypatch.setattr(torch.Tensor, "cpu", _trap_cpu)
    monkeypatch.setattr(torch.Tensor, "item", _trap_item)
    monkeypatch.setattr(torch.Tensor, "tolist", _trap_tolist)
    monkeypatch.setattr(torch.Tensor, "numpy", _forbid_numpy)

    result = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=TrainConfig(
            step_budget_per_attempt=40,
            max_attempts=1,
            seed=42,
            strict_gpu_native=True,
        ),
    )

    _assert_strict_result_shape(result)
    _assert_strict_artifact_shape(result.artifact)


def test_train_only_strict_gpu_native_compat_export_materializes_host_shapes():
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=REACH_POS,
        negatives=REACH_NEG,
        config=TrainConfig(
            step_budget_per_attempt=60,
            max_attempts=2,
            seed=42,
            strict_gpu_native=True,
        ),
    )

    compat_artifact = result.artifact.export_compat_artifact()
    compat_result = result.export_compat_result()

    _assert_strict_result_shape(result)
    _assert_strict_artifact_shape(result.artifact)
    assert isinstance(compat_result, TrainResult)
    assert isinstance(compat_artifact, LearnedArtifact)
    assert compat_result.compat_materialized is True
    assert compat_result.converged is True
    assert compat_result.precision == pytest.approx(1.0)
    assert compat_result.recall == pytest.approx(1.0)
    assert compat_result.rule_frequency >= 0.5
    assert compat_result.discovered_rule is not None
    assert "edge" in compat_result.discovered_rule
    assert compat_artifact.compat_materialized is True
    assert compat_artifact.discovered_rule is not None
    assert "edge" in compat_artifact.discovered_rule
    assert len(compat_artifact.logits) > 0
    assert len(compat_artifact.soft_probs) > 0
    assert len(compat_artifact.selected_hard) > 0


def test_train_only_strict_gpu_native_nonconverged_reports_partial_recall():
    pos_mixed = [("reach", [1, 3]), ("reach", [99, 99])]
    result = train_only(
        source=REACH_SOURCE,
        mask_name="W_reach",
        positives=pos_mixed,
        negatives=REACH_NEG,
        config=TrainConfig(
            step_budget_per_attempt=30,
            max_attempts=1,
            seed=42,
            strict_gpu_native=True,
        ),
    )

    compat_result = result.export_compat_result()
    compat_artifact = result.artifact.export_compat_artifact()

    _assert_strict_result_shape(result)
    _assert_strict_artifact_shape(result.artifact)
    assert isinstance(compat_result, TrainResult)
    assert compat_result.converged is False
    assert compat_result.precision == pytest.approx(0.0)
    assert compat_result.recall > 0.0
    assert compat_result.discovered_rule is not None
    assert isinstance(compat_artifact, LearnedArtifact)
    assert compat_artifact.discovered_rule == compat_result.discovered_rule
