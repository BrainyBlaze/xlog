"""Exact-small solver for tiny candidate surfaces.

When the candidate predicate count is small (≤ threshold), exhaustively
enumerate all (bL, bR) assignments per topology mask, score each by
positive coverage minus negative hits, and return the best rule.

This is deterministic, fast, and produces the optimal rule for the
given positive/negative set — no gradient noise, no local minima.

Used by external consumer for the common 1-3 predicate induction regime where
gradient descent is overkill.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)


@dataclass
class ExactSmallResult:
    """Result from exact-small scoring."""
    discovered_rules: list[tuple[str, float]] = field(default_factory=list)
    discovered_rule: str | None = None
    total_steps: int = 0  # enumeration count, not gradient steps
    attempt_count: int = 1
    bootstrap_negative_count: int = 0
    bootstrap_source_predicates: tuple = ()
    exact_small: bool = True
    candidates_scored: int = 0
    ambiguity_margin: float = 0.0


def score_candidates_exact(
    prog: object,
    mask_names: list[str],
    target_name: str,
    candidate_names: list[str],
    positives: dict[str, list],
    negatives: dict[str, list],
) -> ExactSmallResult:
    """Exhaustively score all (bL, bR) candidates across topology masks.

    Uses the compiled program's actual schema size and relation ordering.
    For each mask × (bL, bR) combination:
      1. Set one-hot rule mask using program's N
      2. Evaluate (forward inference)
      3. Score = positives_derived - negatives_derived

    Args:
        prog: Compiled ILP program with facts already uploaded.
        mask_names: List of mask names (e.g., W_chain_p_10006, ...).
        target_name: Target relation name (e.g., p_10006).
        candidate_names: Candidate relation names (e.g., [p_10006, p_10033, p_10040]).
        positives: {target_name: [arg0_tensor, arg1_tensor]}.
        negatives: {target_name: [arg0_tensor, arg1_tensor]}.
    """
    # Get actual schema from compiled program
    N = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    name_to_idx = {name: i for i, name in enumerate(rel_names)}

    # Only score candidate predicates (not bL, bR, or target)
    candidate_indices = [name_to_idx[c] for c in candidate_names if c in name_to_idx]

    best_score = -1
    best_rule = None
    second_best_score = -1
    total_scored = 0
    all_winners: list[tuple[str, float]] = []

    pos_count = positives[target_name][0].shape[0] if target_name in positives else 0
    neg_count = negatives[target_name][0].shape[0] if target_name in negatives else 0

    for mask_name in mask_names:
        mask_best_score = -1
        mask_best_rule = None

        for bL_idx in candidate_indices:
            for bR_idx in candidate_indices:
                # Build one-hot mask in program's N-space
                flat = [0.0] * (N * N * N)
                # xlog mask: flat index = i * N * N + j * N + k
                # i = bL relation index, j = bR relation index, k = head (0 for single)
                idx = bL_idx * N * N + bR_idx * N  # k=0
                if idx < len(flat):
                    flat[idx] = 1.0

                try:
                    prog.set_rule_mask(mask_name, flat, flat, N)
                    prog.evaluate()
                except Exception:
                    continue

                pos_derived = 0
                neg_derived = 0

                if pos_count > 0:
                    try:
                        membership = torch.from_dlpack(
                            prog.batch_fact_membership_device(
                                target_name, positives[target_name],
                            )
                        )
                        pos_derived = int(membership.sum().item())
                    except Exception:
                        pass

                if neg_count > 0:
                    try:
                        membership = torch.from_dlpack(
                            prog.batch_fact_membership_device(
                                target_name, negatives[target_name],
                            )
                        )
                        neg_derived = int(membership.sum().item())
                    except Exception:
                        pass

                score = pos_derived - neg_derived
                total_scored += 1

                if score > mask_best_score:
                    mask_best_score = score
                    bL_name = rel_names[bL_idx]
                    bR_name = rel_names[bR_idx]
                    if "chain" in mask_name:
                        rule_str = f"{target_name}(X, Y) :- {bL_name}(X, Z), {bR_name}(Z, Y)."
                    elif "star" in mask_name:
                        rule_str = f"{target_name}(X, Y) :- {bL_name}(X, Y), {bR_name}(X, Y)."
                    elif "fanout" in mask_name:
                        rule_str = f"{target_name}(X, Y) :- {bL_name}(X, Z), {bR_name}(X, Y)."
                    elif "fanin" in mask_name:
                        rule_str = f"{target_name}(X, Y) :- {bL_name}(X, Y), {bR_name}(Z, Y)."
                    else:
                        rule_str = f"{target_name}(X, Y) :- {bL_name}(X, Z), {bR_name}(Z, Y)."
                    mask_best_rule = (rule_str, float(score))

        if mask_best_rule is not None and mask_best_score > 0:
            all_winners.append(mask_best_rule)
            if mask_best_score > best_score:
                second_best_score = best_score
                best_score = mask_best_score
                best_rule = mask_best_rule[0]
            elif mask_best_score > second_best_score:
                second_best_score = mask_best_score

    margin = (best_score - second_best_score) / max(pos_count, 1) if best_score > 0 else 0.0

    logger.info(
        "Exact-small: scored %d candidates across %d masks, %d winners, margin=%.3f",
        total_scored, len(mask_names), len(all_winners), margin,
    )

    return ExactSmallResult(
        discovered_rules=all_winners,
        discovered_rule=best_rule,
        total_steps=total_scored,
        candidates_scored=total_scored,
        ambiguity_margin=margin,
    )
