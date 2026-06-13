"""Exact induction scorer — top-K per topology with structured metadata.

Python prototype of `crates/xlog-induce`. Uses the compiled ILP program's
set_rule_mask / evaluate / batch_fact_membership_device API to exhaustively
score all (left, right) candidate pairs across the four canonical topologies,
returning the top-K per topology with full candidate metadata.

Phase 0 throwaway — Phase 1 replaces this with a Rust/CUDA crate that does
the same scoring in a single batched GPU pass with one D2H transfer.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import torch

try:
    from pyxlog.ilp.exceptions import IlpConfigError
except Exception:  # pragma: no cover - supports direct module loading in tests.
    class IlpConfigError(ValueError):
        pass

logger = logging.getLogger(__name__)

_ALLOW_PYTHON_REFERENCE_ENV = "XLOG_ALLOW_PYTHON_ILP_REFERENCE"

# ── Topology definitions ─────────────────────────────────────────────────

TOPOLOGIES = ("chain", "star", "fanout", "fanin")

TOPOLOGY_TEMPLATES = {
    "chain":  "{h}(X, Y) :- {bL}(X, Z), {bR}(Z, Y).",
    "star":   "{h}(X, Y) :- {bL}(X, Y), {bR}(X, Y).",
    "fanout": "{h}(X, Y) :- {bL}(X, Z), {bR}(X, Y).",
    "fanin":  "{h}(X, Y) :- {bL}(X, Y), {bR}(Z, Y).",
}


# ── Result types (mirrors the Rust spec) ─────────────────────────────────

@dataclass(frozen=True)
class ScoredCandidate:
    """One scored (left, right) candidate for a single topology."""
    topology: str               # "chain" | "star" | "fanout" | "fanin"
    head_relation: str
    left_relation: str
    right_relation: str
    positives_covered: int
    negatives_covered: int
    local_rank: int             # 0-indexed within topology
    # Diagnostics — filled after ranking
    next_positives_covered: int = 0
    next_negatives_covered: int = 0
    tie_class_size: int = 1

    @property
    def rule_text(self) -> str:
        tmpl = TOPOLOGY_TEMPLATES[self.topology]
        return tmpl.format(
            h=self.head_relation, bL=self.left_relation, bR=self.right_relation,
        )


@dataclass
class ExactInductionResult:
    """Combined result from induce_exact()."""
    candidates: list[ScoredCandidate] = field(default_factory=list)
    total_scored: int = 0
    candidate_count: int = 0
    positive_count: int = 0
    negative_count: int = 0


# ── Core scorer ──────────────────────────────────────────────────────────

def induce_exact(
    prog,
    *,
    head_relation: str,
    candidate_relations: list[str],
    positive_arg0: torch.Tensor,
    positive_arg1: torch.Tensor,
    negative_arg0: torch.Tensor | None = None,
    negative_arg1: torch.Tensor | None = None,
    k_per_topology: int = 2,
    deterministic: bool = True,
    backend: str = "native",
    strict_per_topology: bool = False,
) -> ExactInductionResult:
    """Exhaustively score all (left, right) pairs across 4 topologies.

    Note:
        ``candidate_relations`` may include the head relation — the scorer
        will happily score self-referential rules (e.g. p(X,Y) :- p(X,Z), p(Z,Y)).
        The caller is responsible for excluding the head if self-reference is
        undesired. external consumer production code excludes it; the learned-rules data shows
        self-referential rules ARE frequently the top scorer.

    Args:
        prog: Compiled ILP program with facts already uploaded.
        head_relation: Target relation name (e.g. "p_10006").
        candidate_relations: Body candidate relation names.
        positive_arg0, positive_arg1: 1-D device tensors of positive pairs.
        negative_arg0, negative_arg1: Optional 1-D device tensors of negative pairs.
        k_per_topology: How many top candidates to keep per topology.
        deterministic: Unused in Python prototype (determinism is inherent).
        backend: ``"native"`` dispatches to the Phase 1 ``xlog-induce`` engine
            and is the default production path. ``"python"`` is a gated
            reference prototype implementation (the
            host-orchestrated ``set_rule_mask``/``evaluate``/``batch_fact_membership_device``
            loop).
        strict_per_topology: (``backend="python"`` only.) When ``True``, zero
            out the other topology masks before scoring each topology, so each
            ``(topology, L, R)`` pair's coverage is computed in isolation —
            semantically correct. When ``False`` (default), preserves the
            original prototype behavior in which stale masks from prior
            outer-loop iterations contaminate later topologies' coverage
            numbers. Kept default-False for backward compatibility with
            historical external consumer Phase 0 measurements (e.g. Phase 0d's 449/449
            liveness baseline was measured against this behavior).

            The ``"native"`` backend implements the strict-per-topology
            semantics by construction — each ``(topology, L, R)`` triple is
            scored in isolation in a single batched kernel launch. Use
            ``strict_per_topology=True`` here to get a semantically matching
            Python reference for parity testing.

    Returns:
        ExactInductionResult with up to k_per_topology × 4 candidates.
    """
    if backend == "python":
        if os.environ.get(_ALLOW_PYTHON_REFERENCE_ENV) != "1":
            raise IlpConfigError(
                "induce_exact backend='python' is a host-orchestrated reference "
                "scorer; set XLOG_ALLOW_PYTHON_ILP_REFERENCE=1 only for explicit "
                "parity or compatibility validation"
            )
        pass  # fall through to the reference implementation below
    elif backend == "native":
        return _induce_exact_native(
            prog,
            head_relation=head_relation,
            candidate_relations=candidate_relations,
            positive_arg0=positive_arg0,
            positive_arg1=positive_arg1,
            negative_arg0=negative_arg0,
            negative_arg1=negative_arg1,
            k_per_topology=k_per_topology,
            deterministic=deterministic,
        )
    else:
        raise ValueError(
            f"induce_exact: unknown backend {backend!r}; expected 'python' or 'native'",
        )

    N = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    name_to_idx = {n: i for i, n in enumerate(rel_names)}

    # Validate inputs
    if head_relation not in name_to_idx:
        logger.warning("Head relation %s not in program schema", head_relation)
        return ExactInductionResult()

    body_indices = []
    for cname in candidate_relations:
        if cname in name_to_idx:
            body_indices.append(name_to_idx[cname])

    if not body_indices:
        return ExactInductionResult()

    pos_count = positive_arg0.shape[0]
    neg_count = negative_arg0.shape[0] if negative_arg0 is not None else 0

    if pos_count == 0:
        return ExactInductionResult(positive_count=0, negative_count=neg_count)

    # Build fact lists for membership checks (small D2H — typically 20-50 rows)
    pos_facts = [[int(positive_arg0[i].item()), int(positive_arg1[i].item())]
                 for i in range(pos_count)]
    neg_facts = ([[int(negative_arg0[i].item()), int(negative_arg1[i].item())]
                  for i in range(neg_count)]
                 if neg_count > 0 else [])

    device = positive_arg0.device
    mask_names = [f"W_{topo}_{head_relation}" for topo in TOPOLOGIES]

    # Pre-allocate a zero mask. Reused to clear "other" topology masks
    # between outer-loop iterations so each topology's inner-loop scoring is
    # isolated. Without this, a topology's final one-hot mask bleeds into
    # the next topology's coverage counts via stale state inside evaluate(),
    # which was a real bug caught by Phase 1 parity testing against the
    # native backend.
    flat_zero = torch.zeros(N * N * N, dtype=torch.float32, device=device)

    all_candidates: list[ScoredCandidate] = []
    total_scored = 0

    for topo, mask_name in zip(TOPOLOGIES, mask_names):
        # Opt-in: zero out all other topology masks so only this topology's
        # rule can contribute to p_A derivations during evaluate(). See
        # comment on `flat_zero` above. Default-off to preserve historical
        # external consumer Phase 0 measurements that were calibrated against the original
        # prototype's cross-topology contamination behavior.
        if strict_per_topology:
            for other_topo, other_mask in zip(TOPOLOGIES, mask_names):
                if other_topo != topo:
                    try:
                        prog.set_rule_mask(other_mask, flat_zero, flat_zero, N)
                    except Exception:
                        # Best-effort: if the program doesn't declare one of
                        # the other-topology masks (e.g. a head with only a
                        # subset of the four topology rules), just skip.
                        continue

        # Collect all (bL, bR) scores for this topology
        topo_scores: list[tuple[int, int, int, int]] = []  # (pos_cov, neg_cov, bL_idx, bR_idx)

        for bL_idx in body_indices:
            for bR_idx in body_indices:
                flat = torch.zeros(N * N * N, dtype=torch.float32, device=device)
                flat[bL_idx * N * N + bR_idx * N] = 1.0

                try:
                    prog.set_rule_mask(mask_name, flat, flat, N)
                    prog.evaluate()
                except Exception:
                    total_scored += 1
                    continue

                pos_cov = 0
                if pos_count > 0:
                    try:
                        mem = torch.from_dlpack(
                            prog.batch_fact_membership_device(head_relation, pos_facts),
                        )
                        pos_cov = int(mem.sum().item())
                    except Exception:
                        pass

                neg_cov = 0
                if neg_facts:
                    try:
                        mem = torch.from_dlpack(
                            prog.batch_fact_membership_device(head_relation, neg_facts),
                        )
                        neg_cov = int(mem.sum().item())
                    except Exception:
                        pass

                topo_scores.append((pos_cov, neg_cov, bL_idx, bR_idx))
                total_scored += 1

        # Lexicographic sort: max pos_covered, min neg_covered, min left_idx, min right_idx
        topo_scores.sort(key=lambda t: (-t[0], t[1], t[2], t[3]))

        # Keep top-K, skip zero-coverage candidates
        kept = [s for s in topo_scores if s[0] > 0][:k_per_topology]

        for rank, (pos_cov, neg_cov, bL_idx, bR_idx) in enumerate(kept):
            # Diagnostics: next candidate's scores (rank+1 if exists)
            next_pos = 0
            next_neg = 0
            if rank + 1 < len([s for s in topo_scores if s[0] > 0]):
                nxt = [s for s in topo_scores if s[0] > 0][rank + 1]
                next_pos, next_neg = nxt[0], nxt[1]

            # Tie class: how many candidates share the same (pos_cov, neg_cov)
            tie_count = sum(
                1 for s in topo_scores if s[0] == pos_cov and s[1] == neg_cov
            )

            all_candidates.append(ScoredCandidate(
                topology=topo,
                head_relation=head_relation,
                left_relation=rel_names[bL_idx],
                right_relation=rel_names[bR_idx],
                positives_covered=pos_cov,
                negatives_covered=neg_cov,
                local_rank=rank,
                next_positives_covered=next_pos,
                next_negatives_covered=next_neg,
                tie_class_size=tie_count,
            ))

    logger.info(
        "induce_exact: head=%s, scored %d pairs, %d candidates (K=%d)",
        head_relation, total_scored, len(all_candidates), k_per_topology,
    )

    return ExactInductionResult(
        candidates=all_candidates,
        total_scored=total_scored,
        candidate_count=len(body_indices),
        positive_count=pos_count,
        negative_count=neg_count,
    )


# ── Native backend delegation ────────────────────────────────────────────

def _induce_exact_native(
    prog,
    *,
    head_relation: str,
    candidate_relations: list[str],
    positive_arg0: torch.Tensor,
    positive_arg1: torch.Tensor,
    negative_arg0: torch.Tensor | None,
    negative_arg1: torch.Tensor | None,
    k_per_topology: int,
    deterministic: bool,
) -> ExactInductionResult:
    """Dispatch to the Rust ``induce_exact_native`` on CompiledIlpProgram.

    The Rust method returns a plain ``dict`` keyed by the ``ScoredCandidate`` /
    ``ExactInductionResult`` field names. Repackaging into the dataclass shapes
    lives here so the Rust side doesn't have to import the Python module.
    """
    raw = prog.induce_exact_native(
        head_relation=head_relation,
        candidate_relations=candidate_relations,
        positive_arg0=positive_arg0,
        positive_arg1=positive_arg1,
        negative_arg0=negative_arg0,
        negative_arg1=negative_arg1,
        k_per_topology=k_per_topology,
        deterministic=deterministic,
    )
    candidates = [ScoredCandidate(**c) for c in raw["candidates"]]
    return ExactInductionResult(
        candidates=candidates,
        total_scored=raw["total_scored"],
        candidate_count=raw["candidate_count"],
        positive_count=raw["positive_count"],
        negative_count=raw["negative_count"],
    )
