"""Mask backend abstraction for trainer.

Two backends:
- DenseMaskBackend: N^3 tensor approach (alpha-compatible)
- SparseMaskBackend: candidate-indexed soft-probs (beta default)

Both provide the same interface:
- init_weights(C, n, device) -> learnable parameters
- apply_mask(prog, mask_name, W, tau, budget, candidates, n) -> candidate_soft_probs
- decode_argmax(W, candidates, n) -> candidate_index
"""
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn.functional as F


class MaskBackend(Protocol):
    """Protocol for mask backends."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        """Return learnable parameter tensor."""
        ...

    def apply_mask(
        self,
        prog,
        mask_name: str,
        W: torch.Tensor,
        tau: float,
        budget: int,
        candidates: list[dict],
        n: int,
        allow_recursive: bool = False,
    ) -> tuple[torch.Tensor, list[int]]:
        """Apply mask to program.

        Returns:
            candidate_soft_probs: Tensor of shape (C,) and selected hard candidate IDs.
        """
        ...

    def decode_argmax(self, W: torch.Tensor, candidates: list[dict], n: int) -> int:
        """Return the candidate index of the argmax."""
        ...


class DenseMaskBackend:
    """Dense N^3 mask -- same as alpha."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn((n, n, n), requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
    ):
        M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
        flat = M_soft.view(-1)
        # Decode hard candidates at the candidate level for telemetry and
        # gate bookkeeping, but keep dense-mask materialization behavior for
        # execution by applying hard updates on the flattened tensor.
        k = min(budget, flat.numel())
        _, topk_idx = flat.topk(k)

        # Map dense flatten indices back to candidate indices for selected_hard.
        # Candidates are defined in the same sorted order as valid_candidates(), so
        # this remains deterministic across implementations.
        flat_to_candidate: dict[int, int] = {}
        for ci, c in enumerate(candidates):
            flat_idx = int(c["i"]) * n * n + int(c["j"]) * n + int(c["k"])
            flat_to_candidate[flat_idx] = ci

        selected_hard = []
        for idx in topk_idx.detach().cpu().tolist():
            ci = flat_to_candidate.get(int(idx))
            if ci is not None:
                selected_hard.append(ci)

        M_hard_flat = torch.zeros_like(flat)
        M_hard_flat[topk_idx] = 1.0
        M_hard = M_hard_flat.view_as(M_soft)

        prog.set_rule_mask(
            mask_name,
            M_hard.detach().contiguous().view(-1),
            M_soft.detach().contiguous().view(-1),
            n,
        )

        # Extract candidate-level soft probs
        cand_probs = torch.stack(
            [M_soft[c["i"], c["j"], c["k"]] for c in candidates]
        )
        selected_hard.sort()
        return cand_probs, selected_hard

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            flat = W.view(-1)
            idx = flat.argmax().item()
            i = idx // (n * n)
            j = (idx % (n * n)) // n
            k = idx % n
        for ci, c in enumerate(candidates):
            if c["i"] == i and c["j"] == j and c["k"] == k:
                return ci
        return 0


class SparseMaskBackend:
    """Sparse candidate-indexed mask -- beta default.

    Uses C-dimensional logits (one per candidate) instead of N^3.
    This gives a smaller, more focused parameter space that converges
    faster on the correct candidate.

    Passes DLPack-wrapped soft-probs directly to Rust via
    set_rule_mask_sparse — no Python-side N^3 materialization.
    Rust owns top-k ranking and sparse mask construction.
    """

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn(C, requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
    ):
        C = len(candidates)

        # Gumbel-softmax over C-dimensional logits
        cand_probs = F.gumbel_softmax(W, tau=tau, hard=False, dim=0)

        # Rust validates candidate count against current store state.
        # After evaluate(), derived relations may gain tuples, changing
        # the expected count.  Re-query the live set and pad with zeros
        # for any new candidates that appeared since compile time.
        live_cands = prog.valid_candidates(mask_name, allow_recursive)
        C_live = len(live_cands)

        # Build (i,j,k) -> original index lookup
        orig_ijk = {
            (c["i"], c["j"], c["k"]): idx for idx, c in enumerate(candidates)
        }
        live_to_orig: list[int | None] = []
        for lc in live_cands:
            key = (lc["i"], lc["j"], lc["k"])
            live_to_orig.append(orig_ijk.get(key))

        if C_live == C:
            soft = cand_probs.detach().contiguous().double()
        else:
            soft = torch.zeros(C_live, device=W.device, dtype=torch.float64)
            for li, lc in enumerate(live_cands):
                key = (lc["i"], lc["j"], lc["k"])
                if key in orig_ijk:
                    soft[li] = cand_probs[orig_ijk[key]].detach().double()

        effective_budget = min(budget, soft.numel())

        candidate_ids = list(range(C_live))
        prog.set_rule_mask_sparse(
            mask_name, candidate_ids, soft.contiguous(), effective_budget,
            allow_recursive,
        )
        selected = torch.topk(
            soft,
            k=min(effective_budget, soft.numel()),
            largest=True,
            sorted=True,
        ).indices
        selected_hard = []
        for i in selected.detach().cpu().tolist():
            idx = int(i)
            orig_idx = live_to_orig[idx] if 0 <= idx < len(live_to_orig) else None
            if orig_idx is not None:
                selected_hard.append(orig_idx)

        selected_hard.sort()
        return cand_probs, selected_hard

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            return W.argmax().item()
