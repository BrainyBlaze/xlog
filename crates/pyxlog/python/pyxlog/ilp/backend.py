"""Mask backend abstraction for trainer.

Two backends:
- DenseMaskBackend: N^3 tensor approach (alpha-compatible)
- SparseMaskBackend: candidate-indexed soft-probs (beta default)

Both provide the same interface:
- init_weights(C, n, device) -> learnable parameters
- apply_mask(prog, mask_name, W, tau, budget, candidates, n) -> candidate_soft_probs
- decode_argmax_compat(W, candidates, n) -> candidate_index
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
        strict_gpu_native: bool = False,
    ) -> tuple[torch.Tensor, list[int]]:
        """Apply mask to program.

        Returns:
            candidate_soft_probs: Tensor of shape (C,) and selected hard candidate IDs.
        """
        ...

    def decode_argmax_compat(self, W: torch.Tensor, candidates: list[dict], n: int) -> int:
        """Return the candidate index of the argmax for explicit compatibility export."""
        ...

    def decode_argmax(self, W: torch.Tensor, candidates: list[dict], n: int) -> int:
        return self.decode_argmax_compat(W, candidates, n)


class DenseMaskBackend:
    """Dense N^3 mask -- same as alpha."""

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn((n, n, n), requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
        strict_gpu_native=False,
    ):
        if strict_gpu_native:
            raise RuntimeError("DenseMaskBackend is incompatible with strict_gpu_native")
        M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
        flat = M_soft.view(-1)
        # Restrict hard-mask activation to valid candidate cells only.
        # This keeps dense fallback semantically aligned with valid_candidates().
        candidate_flat_indices = [
            int(c["i"]) * n * n + int(c["j"]) * n + int(c["k"])
            for c in candidates
        ]

        if candidate_flat_indices:
            candidate_flat_indices_tensor = torch.tensor(
                candidate_flat_indices, device=flat.device, dtype=torch.long,
            )
            candidate_scores = flat[candidate_flat_indices_tensor]
            k = min(budget, candidate_scores.numel())
            if k > 0:
                _, topk_rel_idx = candidate_scores.topk(k)
                topk_idx = candidate_flat_indices_tensor[topk_rel_idx]
                selected_hard = topk_rel_idx.detach().cpu().tolist()
            else:
                topk_idx = torch.empty(0, dtype=torch.long, device=flat.device)
                selected_hard = []
        else:
            topk_idx = torch.empty(0, dtype=torch.long, device=flat.device)
            selected_hard = []

        M_hard_flat = torch.zeros_like(flat)
        if topk_idx.numel() > 0:
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

    def decode_argmax_compat(self, W, candidates, n):
        if not candidates:
            return 0
        with torch.no_grad():
            candidate_vals = torch.stack(
                [W[c["i"], c["j"], c["k"]] for c in candidates]
            )
            return int(candidate_vals.argmax().item())

    def decode_argmax(self, W, candidates, n):
        return self.decode_argmax_compat(W, candidates, n)


class SparseMaskBackend:
    """Sparse candidate-indexed mask -- beta default.

    Uses C-dimensional logits (one per candidate) instead of N^3.
    This gives a smaller, more focused parameter space that converges
    faster on the correct candidate.

    Passes only the selected candidate subset to Rust via
    set_rule_mask_sparse_selected — no Python-side N^3 materialization
    and no Rust-side download of the full soft-probability vector.
    """

    def init_weights(self, C: int, n: int, device: str) -> torch.Tensor:
        return torch.randn(C, requires_grad=True, device=device)

    def apply_mask(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
        strict_gpu_native=False,
    ):
        if strict_gpu_native:
            return self._apply_mask_strict(
                prog,
                mask_name,
                W,
                tau,
                budget,
                candidates,
                n,
                allow_recursive=allow_recursive,
            )

        return self._apply_mask_compat(
            prog,
            mask_name,
            W,
            tau,
            budget,
            candidates,
            n,
            allow_recursive=allow_recursive,
        )

    def _apply_mask_strict(
        self, prog, mask_name, W, tau, budget, candidates, n,
        allow_recursive=False,
    ):
        raise RuntimeError(
            "SparseMaskBackend.apply_mask(..., strict_gpu_native=True) is not a supported "
            "public entry point; use train_only(..., strict_gpu_native=True)"
        )

    def _apply_mask_compat(
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
        selected = torch.argsort(
            soft,
            descending=True,
            stable=True,
        )[:effective_budget]
        selected_soft = soft.index_select(0, selected).contiguous()
        selected_ids = [int(i) for i in selected.detach().cpu().tolist()]

        prog.set_rule_mask_sparse_selected(
            mask_name,
            selected_ids,
            selected_soft,
            allow_recursive,
        )
        selected_hard = []
        for idx in selected_ids:
            orig_idx = live_to_orig[idx] if 0 <= idx < len(live_to_orig) else None
            if orig_idx is not None:
                selected_hard.append(orig_idx)

        selected_hard.sort()
        return cand_probs, selected_hard

    def decode_argmax_compat(self, W, candidates, n):
        with torch.no_grad():
            return W.argmax().item()

    def decode_argmax(self, W, candidates, n):
        return self.decode_argmax_compat(W, candidates, n)
