# python/tests/test_ilp_trainer.py
"""Integration tests for train_only()."""
import inspect
import math

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig, TrainResult, IlpConfigError
from pyxlog.ilp import trainer as trainer_mod

REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
REACH_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
REACH_NEG = []


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


def test_train_only_converges_on_reach():
    config = TrainConfig(
        step_budget_per_attempt=100, max_attempts=5,
        tau_start=2.0, tau_floor=0.05, seed=42,
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
