"""dILP promotion pipeline — train_and_promote() entry point.

Wraps train_only(), compiles trial program with discovered rule committed,
runs promotion gates, returns PromotionResult.
"""
from __future__ import annotations

import pyxlog
from pyxlog.ilp.trainer import train_only
from pyxlog.ilp.types import (
    GateResult,
    LearnedArtifact,
    PromotionResult,
    PromotionStatus,
    TrainConfig,
)


def train_and_promote(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    config: TrainConfig = TrainConfig(),
    holdout_positives: list[tuple[str, list[int]]] | None = None,
    holdout_negatives: list[tuple[str, list[int]]] | None = None,
) -> PromotionResult:
    """Train and optionally promote a learned rule.

    Steps:
    1. Call train_only() to find best rule.
    2. If not converged: NOT_CONVERGED.
    3. If no holdout provided: MANUAL_REVIEW_REQUIRED.
    4. Compile trial program with discovered rule committed.
    5. Run promotion gates against trial.
    6. All pass: PROMOTED. Any fail: MANUAL_REVIEW_REQUIRED.
    """
    train_result = train_only(source, mask_name, positives, negatives, config)

    if not train_result.converged:
        return PromotionResult(
            status=PromotionStatus.NOT_CONVERGED,
            artifact=train_result.artifact,
        )

    gates: list[GateResult] = []
    discovered = train_result.discovered_rule

    # Compile trial program with discovered rule committed
    trial_source = _commit_rule(source, mask_name, discovered)
    try:
        trial = pyxlog.IlpProgramFactory.compile(
            trial_source, device=config.device, memory_mb=config.memory_mb,
        )
    except Exception as e:
        return PromotionResult(
            status=PromotionStatus.COMMIT_FAILED,
            gates=[GateResult(name="commit", passed=False, detail=str(e))],
            artifact=train_result.artifact,
        )

    # Run the committed program (no mask needed)
    trial.evaluate()

    # Gate: training_positive
    tp_count = sum(1 for r, v in positives if trial.fact_exists(r, v))
    tp_ok = tp_count == len(positives)
    gates.append(GateResult(
        name="training_positive",
        passed=tp_ok,
        detail=f"{tp_count}/{len(positives)} derived",
    ))

    # Gate: training_negative
    tn_count = sum(1 for r, v in negatives if trial.fact_exists(r, v))
    tn_ok = tn_count == 0
    gates.append(GateResult(
        name="training_negative",
        passed=tn_ok,
        detail=f"{tn_count}/{len(negatives)} derived (want 0)",
    ))

    # Gate: novel_fact_audit
    head_rel = discovered.split("(")[0].strip()
    all_derived = trial.relation_facts(head_rel)
    known_set = {(r, tuple(v)) for r, v in positives}
    if holdout_positives:
        known_set |= {(r, tuple(v)) for r, v in holdout_positives}
    novel = [f for f in all_derived if (head_rel, tuple(f)) not in known_set]
    novel_count = len(novel)
    total_derived = len(all_derived) if all_derived else 1
    novel_rate = novel_count / total_derived if total_derived > 0 else 0.0
    novel_ok = novel_rate <= config.max_novel_rate
    gates.append(GateResult(
        name="novel_fact_audit",
        passed=novel_ok,
        detail=f"novel_rate={novel_rate:.3f} ({novel_count}/{total_derived}), threshold={config.max_novel_rate}",
    ))

    # Gate: regression_check
    regression_ok = True
    regression_detail_parts = []
    if config.protected_relations:
        original_prog = pyxlog.IlpProgramFactory.compile(
            source, device=config.device, memory_mb=config.memory_mb,
        )
        original_prog.evaluate()
        for rel in config.protected_relations:
            orig_facts = set(map(tuple, original_prog.relation_facts(rel)))
            trial_facts = set(map(tuple, trial.relation_facts(rel)))
            lost = orig_facts - trial_facts
            if lost:
                regression_ok = False
                regression_detail_parts.append(f"{rel}: lost {len(lost)} facts")
    if not regression_detail_parts:
        regression_detail_parts.append("all protected relations preserved")
    gates.append(GateResult(
        name="regression_check",
        passed=regression_ok,
        detail="; ".join(regression_detail_parts),
    ))

    # Holdout gates
    if not holdout_positives and not holdout_negatives:
        return PromotionResult(
            status=PromotionStatus.MANUAL_REVIEW_REQUIRED,
            gates=gates,
            novel_count=novel_count,
            novel_rate=novel_rate,
            novel_examples=[str(f) for f in novel[:10]],
            artifact=train_result.artifact,
        )

    hp = holdout_positives or []
    hn = holdout_negatives or []

    if hp:
        hp_derived = sum(1 for r, v in hp if trial.fact_exists(r, v))
        hp_ok = hp_derived >= len(hp) * 0.95
        gates.append(GateResult(
            name="holdout_positive",
            passed=hp_ok,
            detail=f"{hp_derived}/{len(hp)} derived (threshold 95%)",
        ))
    if hn:
        hn_derived = sum(1 for r, v in hn if trial.fact_exists(r, v))
        hn_ok = hn_derived == 0
        gates.append(GateResult(
            name="holdout_negative",
            passed=hn_ok,
            detail=f"{hn_derived}/{len(hn)} derived (want 0)",
        ))

    # Ambiguity scan (informational, not gating)
    ambiguous_alts: list[str] | None = None
    if config.check_ambiguity:
        top_m = 256 if not config.exhaustive_ambiguity else 10_000
        ambiguous_alts = _scan_ambiguity(
            source, mask_name, positives, negatives,
            discovered, train_result.artifact, config,
            top_m=top_m,
        )

    all_pass = all(g.passed for g in gates)
    status = PromotionStatus.PROMOTED if all_pass else PromotionStatus.MANUAL_REVIEW_REQUIRED

    return PromotionResult(
        status=status,
        gates=gates,
        novel_count=novel_count,
        novel_rate=novel_rate,
        novel_examples=[str(f) for f in novel[:10]],
        committed_source=trial_source if all_pass else None,
        ambiguous_alternatives=ambiguous_alts,
        artifact=train_result.artifact,
    )


def _scan_ambiguity(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    winning_rule: str,
    artifact: LearnedArtifact,
    config: TrainConfig,
    top_m: int = 256,
) -> list[str]:
    """Check top-M candidates by final soft probability for alternative rules.

    Per design doc Section 4.2: scan top-M by final soft probability, not
    by candidate position. M = min(top_m, C).
    """
    prog = pyxlog.IlpProgramFactory.compile(
        source, device=config.device, memory_mb=config.memory_mb,
    )
    candidates = prog.valid_candidates(mask_name, config.allow_recursive_candidates)

    # Sort candidates by final soft probability (descending) from artifact
    soft_probs = artifact.soft_probs if artifact.soft_probs else []
    if soft_probs and len(soft_probs) == len(candidates):
        indexed = sorted(enumerate(candidates), key=lambda x: -soft_probs[x[0]])
        sorted_candidates = [c for _, c in indexed]
    else:
        sorted_candidates = candidates

    alternatives: list[str] = []
    scan_count = min(top_m, len(sorted_candidates))
    for c in sorted_candidates[:scan_count]:
        rule_str = f"{c['head_name']}(X, Y) :- {c['left_name']}(X, Z), {c['right_name']}(Z, Y)."
        if rule_str == winning_rule:
            continue

        trial_source = _commit_rule(source, mask_name, rule_str)
        try:
            trial = pyxlog.IlpProgramFactory.compile(
                trial_source, device=config.device, memory_mb=config.memory_mb,
            )
            trial.evaluate()

            all_pos = all(trial.fact_exists(r, v) for r, v in positives)
            no_neg = not any(trial.fact_exists(r, v) for r, v in negatives)

            if all_pos and no_neg:
                alternatives.append(rule_str)
        except Exception:
            continue

    return alternatives


def _commit_rule(source: str, mask_name: str, rule: str) -> str:
    """Replace the learnable declaration with the discovered rule."""
    lines = source.splitlines()
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(f"learnable({mask_name})"):
            result.append(f"    {rule}")
        else:
            result.append(line)
    return "\n".join(result)
