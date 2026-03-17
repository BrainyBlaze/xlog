"""dILP promotion pipeline — train_and_promote() entry point.

Wraps train_only(), compiles trial program with discovered rule committed,
runs promotion gates, returns PromotionResult.
"""
from __future__ import annotations

import pyxlog
import re
from pyxlog.ilp.exceptions import IlpConfigError
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
    if config.strict_gpu_native:
        raise IlpConfigError(
            "strict_gpu_native is incompatible with train_and_promote; "
            "disable strict_gpu_native for compatibility promotion gates"
        )

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

    # Gate: holdout F1 (LOO/k-fold on training positives only).
    holdout_threshold = config.holdout_threshold
    holdout_f1 = train_result.holdout_f1
    if holdout_f1 is None:
        gates.append(GateResult(
            name="holdout_f1",
            passed=False,
            detail="holdout_f1 unavailable (insufficient positives for validation)",
        ))
    else:
        gates.append(GateResult(
            name="holdout_f1",
            passed=holdout_f1 >= holdout_threshold,
            detail=(
                f"holdout_f1={holdout_f1:.3f}, threshold={holdout_threshold}, "
                f"variance={train_result.holdout_variance:.6f}"
            ),
        ))

    # Gate: typed schema availability (GA blocker).
    gates.append(_typed_schema_gate(
        trial=trial,
        discovered_rule=discovered or "",
        artifact=train_result.artifact,
        config=config,
    ))

    # Ambiguity scan (informational, not gating) — runs before holdout check
    ambiguous_alts: list[str] | None = None
    if config.check_ambiguity:
        top_m: int | None = 256 if not config.exhaustive_ambiguity else None
        ambiguous_alts = _scan_ambiguity(
            source, mask_name, positives, negatives,
            discovered, train_result.artifact, config,
            top_m=top_m,
        )

    # Holdout gates
    if not holdout_positives and not holdout_negatives:
        return PromotionResult(
            status=PromotionStatus.MANUAL_REVIEW_REQUIRED,
            gates=gates,
            novel_count=novel_count,
            novel_rate=novel_rate,
            novel_examples=[str(f) for f in novel[:10]],
            ambiguous_alternatives=ambiguous_alts,
            artifact=train_result.artifact,
        )

    hp = holdout_positives or []
    hn = holdout_negatives or []

    if hp:
        hp_derived = sum(1 for r, v in hp if trial.fact_exists(r, v))
        hp_ok = hp_derived >= len(hp) * holdout_threshold
        gates.append(GateResult(
            name="holdout_positive",
            passed=hp_ok,
            detail=f"{hp_derived}/{len(hp)} derived (threshold {holdout_threshold:.2f})",
        ))
    if hn:
        hn_derived = sum(1 for r, v in hn if trial.fact_exists(r, v))
        hn_ok = hn_derived == 0
        gates.append(GateResult(
            name="holdout_negative",
            passed=hn_ok,
            detail=f"{hn_derived}/{len(hn)} derived (want 0)",
        ))

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


def _typed_schema_gate(
    *,
    trial,
    discovered_rule: str,
    artifact: LearnedArtifact,
    config: TrainConfig,
) -> GateResult:
    if not (config.typed_schema_required or config.waiver_untyped):
        return GateResult(
            name="typed_schema",
            passed=True,
            detail="typed schema enforcement disabled",
        )

    if not discovered_rule:
        return GateResult(
            name="typed_schema",
            passed=False,
            detail="typed schema check skipped: discovered_rule missing",
        )

    rel_names = _extract_rule_relations(artifact, discovered_rule)
    if not rel_names:
        if config.waiver_untyped:
            return GateResult(
                name="typed_schema",
                passed=False,
                detail=(
                    "typed schema check unavailable: unable to map discovered rule to candidate metadata; "
                    "manual_review_required"
                ),
            )
        return GateResult(
            name="typed_schema",
            passed=False,
            detail="typed schema check unavailable for discovered rule",
        )

    head_name, left_name, right_name = rel_names

    try:
        annotations = trial.relation_type_annotations()
    except Exception as exc:
        if config.waiver_untyped:
            return GateResult(
                name="typed_schema",
                passed=False,
                detail=(
                    f"typed schema metadata unavailable at runtime ({exc}); manual review required"
                ),
            )
        return GateResult(
            name="typed_schema",
            passed=False,
            detail=f"typed schema metadata unavailable at runtime: {exc}",
        )

    annotations_by_name = {name: tuple(types) for name, types in annotations}
    missing: list[str] = []
    for rel in (head_name, left_name, right_name):
        rel_types = annotations_by_name.get(rel)
        if not rel_types or len(rel_types) < 2:
            missing.append(rel)

    if missing:
        unique = ", ".join(sorted(set(missing)))
        if config.waiver_untyped:
            return GateResult(
                name="typed_schema",
                passed=False,
                detail=(
                    f"typed schema missing for: {unique}; "
                    "waiver_untyped=True -> manual_review_required"
                ),
            )
        return GateResult(
            name="typed_schema",
            passed=False,
            detail=(
                f"typed schema missing for: {unique}; "
                "set waiver_untyped=True for manual_review fallback"
            ),
        )

    return GateResult(
        name="typed_schema",
        passed=True,
        detail=(
            "typed schema available for head/left/right relations: "
            f"{head_name}/{left_name}/{right_name}"
        ),
    )


def _extract_rule_relations(
    artifact: LearnedArtifact,
    discovered_rule: str,
) -> tuple[str, str, str] | None:
    for entry in artifact.candidate_map:
        candidate_rule = (
            f"{entry.head_name}(X, Y) :- {entry.left_name}(X, Z), {entry.right_name}(Z, Y)."
        )
        if discovered_rule == candidate_rule:
            return entry.head_name, entry.left_name, entry.right_name

    # Fallback to parsing only for defense in case formats drift.
    match = re.match(
        r"^\s*([^(\s]+)\s*\([^)]*\)\s*:-\s*([^,(\s]+)\([^)]*\)\s*,\s*([^,(\s]+)\([^)]*\)\s*\.\s*$",
        discovered_rule,
    )
    if match:
        return match.group(1), match.group(2), match.group(3)
    return None


def _scan_ambiguity(
    source: str,
    mask_name: str,
    positives: list[tuple[str, list[int]]],
    negatives: list[tuple[str, list[int]]],
    winning_rule: str,
    artifact: LearnedArtifact,
    config: TrainConfig,
    top_m: int | None = 256,
) -> list[str]:
    """Check top-M candidates by final soft probability for alternative rules.

    Per design doc Section 4.2: scan top-M by final soft probability, not
    by candidate position. M = min(top_m, C). Pass None for exhaustive scan.
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
    scan_count = len(sorted_candidates) if top_m is None else min(top_m, len(sorted_candidates))
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
