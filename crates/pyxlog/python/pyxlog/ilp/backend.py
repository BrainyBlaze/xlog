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
        allow_recursive: bool = False,  # TODO(task-8): forward to set_rule_mask_sparse
    ) -> torch.Tensor:
        """Apply mask to program. Returns candidate_soft_probs shape (C,)."""
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
        k = min(budget, flat.numel())
        _, topk_idx = flat.topk(k)
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
        return cand_probs

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

    Internally materializes the candidate probs into the dense N^3 mask
    format via set_rule_mask for store-state-independent evaluation.
    """

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn(C, requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
    ):
        # Gumbel-softmax over C-dimensional logits
        cand_probs = F.gumbel_softmax(W, tau=tau, hard=False, dim=0)

        # Materialize into dense N^3 format for set_rule_mask.
        # TODO(RC): switch to prog.set_rule_mask_sparse() once store-state
        # validation issue is resolved — eliminates Python-side N^3 tensor.
        M_soft = torch.zeros((n, n, n), device=W.device)
        for ci, c in enumerate(candidates):
            M_soft[c["i"], c["j"], c["k"]] = cand_probs[ci]

        # Top-k hard mask over C candidates (not N^3 cells).
        #
        # Key insight: in the dense backend, topk(budget) selects budget cells
        # out of N^3 total, so only a fraction of the C valid candidate positions
        # end up active (typically 2-5 for budget=32, N=6). In the sparse backend,
        # ALL C cells are non-zero after materialization, so topk(budget) with
        # budget >= C would activate ALL candidates -- leaking distractor rules
        # into the hard mask and causing Gate 2 to fail.
        #
        # Fix: cap the hard mask at half of C (matching the ~15% activation
        # ratio of the dense backend) while ensuring at least 1 candidate.
        C = len(candidates)
        k = max(1, min(budget, C) // 2)
        _, topk_idx = cand_probs.topk(k)
        # Build hard mask: only activate top-k candidates in the N^3 tensor
        M_hard = torch.zeros((n, n, n), device=W.device)
        for idx in topk_idx:
            c = candidates[idx]
            M_hard[c["i"], c["j"], c["k"]] = 1.0

        prog.set_rule_mask(
            mask_name,
            M_hard.detach().contiguous().view(-1),
            M_soft.detach().contiguous().view(-1),
            n,
        )

        return cand_probs

    def decode_argmax(self, W, candidates, n):
        with torch.no_grad():
            return W.argmax().item()
