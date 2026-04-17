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
from dataclasses import dataclass, field

import torch

logger = logging.getLogger(__name__)

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
    backend: str = "python",
) -> ExactInductionResult:
    """Exhaustively score all (left, right) pairs across 4 topologies.

    Note:
        ``candidate_relations`` may include the head relation — the scorer
        will happily score self-referential rules (e.g. p(X,Y) :- p(X,Z), p(Z,Y)).
        The caller is responsible for excluding the head if self-reference is
        undesired. DTS production code excludes it; the learned-rules data shows
        self-referential rules ARE frequently the top scorer.

    Args:
        prog: Compiled ILP program with facts already uploaded.
        head_relation: Target relation name (e.g. "p_10006").
        candidate_relations: Body candidate relation names.
        positive_arg0, positive_arg1: 1-D device tensors of positive pairs.
        negative_arg0, negative_arg1: Optional 1-D device tensors of negative pairs.
        k_per_topology: How many top candidates to keep per topology.
        deterministic: Unused in Python prototype (determinism is inherent).
        backend: ``"python"`` uses the reference prototype implementation (the
            host-orchestrated ``set_rule_mask``/``evaluate``/``batch_fact_membership_device``
            loop). ``"native"`` dispatches to the Phase 1 ``xlog-induce`` engine
            when available; raises ``NotImplementedError`` until that engine lands.

    Returns:
        ExactInductionResult with up to k_per_topology × 4 candidates.
    """
    if backend == "python":
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

    all_candidates: list[ScoredCandidate] = []
    total_scored = 0

    for topo, mask_name in zip(TOPOLOGIES, mask_names):
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
    """Dispatch to the Rust `induce_exact_native` method on CompiledIlpProgram.

    Until M8 Phase 1 Task 3 lands, the Rust method raises
    ``PyNotImplementedError``; this function surfaces that unchanged so the
    parity tests fail at the native seam (Rust) rather than the Python
    dispatcher.
    """
    return prog.induce_exact_native(
        head_relation=head_relation,
        candidate_relations=candidate_relations,
        positive_arg0=positive_arg0,
        positive_arg1=positive_arg1,
        negative_arg0=negative_arg0,
        negative_arg1=negative_arg1,
        k_per_topology=k_per_topology,
        deterministic=deterministic,
    )
