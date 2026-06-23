import pytest
import torch

from pyxlog.demos.plasticity import (
    CAND_POSTPRE_NEURAL,
    CAND_PREPOST_NEURAL,
    CAND_PREPOST_REL,
    make_demo_data,
    run_demo,
)
from pyxlog.ilp.neurosymbolic import NeuroSymbolicTrainingConfig

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for neuro-symbolic training"
)

_CFG = NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1)


@requires_cuda
def test_demo_recovers_the_planted_rule() -> None:
    """The neural pre-post candidate (relational eligibility AND a learned saliency
    gate) is the only one that separates strong from weak coincidences; the mixture
    selects it over the relational-only over-firer and the wrong-timing distractor."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, _CFG)

    assert report.selected_rule_id == CAND_PREPOST_NEURAL
    # generalizes on held-out positives, beating the relational-only candidate
    assert report.heldout_coverage.get(CAND_PREPOST_NEURAL, 0.0) > report.heldout_coverage.get(
        CAND_PREPOST_REL, 0.0
    )
    # the relational-only and wrong-timing candidates lose the guard competition
    assert report.symbolic_rule_weights[CAND_PREPOST_NEURAL] > 0.5
    assert report.symbolic_rule_weights[CAND_PREPOST_REL] < 0.5
    assert report.symbolic_rule_weights[CAND_POSTPRE_NEURAL] < 0.5
    # train fit matches the planted rule: strong-LTP edges (0,1) high, all others low
    p = report.train_query_probabilities
    assert min(p[0], p[1]) > 0.6
    assert max(p[2], p[3], p[4], p[5], p[6], p[7]) < 0.4


@requires_cuda
def test_demo_heldout_generalizes_and_keeps_vigilance() -> None:
    """The admitted winner fires on a NEW strong pre-post edge (generalize) and does
    NOT fire on a new weak pre-post edge, a new post-pre edge, or an unrelated edge."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, _CFG)

    adm = report.heldout_admission
    labels = report.heldout_labels
    assert labels[0] is True
    assert adm[0] > 0.6  # new strong LTP fires (generalize)
    assert max(adm[1], adm[2], adm[3]) < 0.4  # weak / wrong-timing / unrelated stay low (vigilance)


@requires_cuda
def test_demo_training_is_zero_host() -> None:
    """The neural-bodied joint training loop performs no tracked device<->host transfers."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=50, learning_rate=0.1))
    stats = report.training_host_transfer_stats
    assert stats["dtoh_calls"] == 0 and stats["htod_calls"] == 0


@requires_cuda
def test_demo_rule_inventory_and_proof_trace_present() -> None:
    """The induced clause is selected and the rule/proof-trace surface is exposed."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, _CFG)
    assert report.proof_trace_map is not None
    assert report.rule_inventory is not None
    assert report.symbolic_rule_weights[report.selected_rule_id] >= 0.5
