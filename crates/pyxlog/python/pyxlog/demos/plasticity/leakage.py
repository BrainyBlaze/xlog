"""Held-out leakage / module-boundary diagnostic.

The joint-mixture held-out read (``evaluate_joint_mixture``) keys bindings by
position. If a training entity reappears in the held-out split, the held-out
read measures memorization, not generalization. This guard fails closed on any
entity-id overlap between splits, the demo's module boundary."""

from __future__ import annotations

from .generator import Split


class LeakageError(RuntimeError):
    """Raised when train and held-out splits share an entity (held-out leakage)."""


def assert_no_leakage(train: Split, held_out: Split) -> None:
    overlap = train.entity_ids() & held_out.entity_ids()
    if overlap:
        raise LeakageError(
            f"held-out leakage: {len(overlap)} entity id(s) overlap between train "
            f"and held-out splits: {sorted(overlap)[:5]}"
        )
