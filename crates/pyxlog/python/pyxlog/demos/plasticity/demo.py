"""Path-Y plasticity demo driver: leakage guard -> train -> select winner by
held-out coverage -> admit winner -> report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    evaluate_joint_mixture,
    train_neurosymbolic_program,
)

from .generator import Split
from .leakage import assert_no_leakage
from .program import build_neural_bodies, build_source

_GROUND_TRUTH = {
    "strengthens": "strengthens(E) :- edge_pre_post(E), saliency(E) >= 0.5",
    "weakens": "weakens(E) :- edge_post_pre(E), saliency(E) >= 0.5",
}
GROUND_TRUTH_RULE = _GROUND_TRUTH["strengthens"]


@dataclass
class DemoReport:
    selected_rule_id: str
    symbolic_rule_weights: dict[str, float]
    train_query_probabilities: list[float]
    heldout_coverage: dict[str, float]
    heldout_admission: list[float]
    heldout_labels: list[bool]
    rule_inventory: Any
    proof_trace_map: Any
    training_host_transfer_stats: Any
    ground_truth_rule: str = GROUND_TRUTH_RULE


def _examples(split: Split, outcome: str) -> list[dict[str, Any]]:
    import torch

    n = split.num_queries()
    return [
        {
            "inputs": torch.zeros((n, 1), dtype=torch.float32),
            "targets": torch.tensor(
                [1.0 if t else 0.0 for t in split.labels_for(outcome)], dtype=torch.float32
            ),
        }
    ]


def _select_winner(
    held_out: Split,
    train_weights: dict[str, float],
    neural_state: dict[str, Any],
    outcome: str,
) -> tuple[str, dict[str, float]]:
    """SELECT among train-covering candidates by guard-free held-out coverage over
    held-out positives (per evaluate_joint_mixture's SELECT-vs-ADMIT contract):
    rank each train-covering candidate by the mean of its single-candidate held-out
    probability at weight 1.0 over the held-out positive bindings."""
    import torch

    source = build_source(held_out, outcome)
    held_phi = held_out.phi()
    positives = [i for i, t in enumerate(held_out.labels_for(outcome)) if t]
    train_covering = [c for c, w in train_weights.items() if w >= 0.5]

    coverage: dict[str, float] = {}
    for cand in train_covering:
        neural_heldout = None
        if cand in neural_state:
            neural_heldout = {cand: (neural_state[cand], held_phi)}
        probs = evaluate_joint_mixture(
            source,
            rule_weights={cand: 1.0},  # weight 1.0 -> noisy-OR == this candidate's eligibility
            num_queries=held_out.num_queries(),
            neural_heldout=neural_heldout,
        )
        coverage[cand] = (
            float(torch.tensor([probs[i] for i in positives]).mean().item())
            if positives
            else 0.0
        )

    winner = max(coverage, key=coverage.get)
    return winner, coverage


def run_demo(
    train: Split,
    held_out: Split,
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(
        steps=400, learning_rate=0.1
    ),
    outcome: str = "strengthens",
) -> DemoReport:
    # 1. module boundary: refuse to proceed on held-out leakage.
    assert_no_leakage(train, held_out)

    # 2. train the joint mixture over the three candidates.
    result = train_neurosymbolic_program(
        build_source(train, outcome),
        networks={},
        examples=_examples(train, outcome),
        config=config,
        neural_bodies=build_neural_bodies(train, outcome),
    )
    neural_state = result.neural_body_state or {}

    # 3. SELECT the winner by held-out coverage (guard-free).
    winner, coverage = _select_winner(
        held_out, result.symbolic_rule_weights, neural_state, outcome
    )

    # 4. ADMIT: the faithful held-out read with ONLY the winner's trained guard.
    held_source = build_source(held_out, outcome)
    neural_heldout = (
        {winner: (neural_state[winner], held_out.phi())}
        if winner in neural_state
        else None
    )
    admission = evaluate_joint_mixture(
        held_source,
        rule_weights={winner: result.symbolic_rule_weights[winner]},
        num_queries=held_out.num_queries(),
        neural_heldout=neural_heldout,
    )

    return DemoReport(
        selected_rule_id=winner,
        symbolic_rule_weights=result.symbolic_rule_weights,
        train_query_probabilities=result.query_probabilities,
        heldout_coverage=coverage,
        heldout_admission=list(admission),
        heldout_labels=held_out.labels_for(outcome),
        rule_inventory=result.learned_rule_inventory,
        proof_trace_map=result.proof_trace_map,
        training_host_transfer_stats=result.training_host_transfer_stats,
        ground_truth_rule=_GROUND_TRUTH[outcome],
    )
