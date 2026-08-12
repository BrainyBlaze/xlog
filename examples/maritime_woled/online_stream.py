"""Chronological window stream for the pre-registered maritime online
weights column (`docs/experiments/maritime/PREREG_ONLINE.md`, section 2):
one pass over a fold's rows in ascending global `pt_time`, all pairs
interleaved — as in a real stream — cut into mini-batch windows.

The converter lays pt rows out pair-major (time is monotone only inside
a pair's episodes), so the global chronological order is a PERMUTATION of
the row sequence: a stable sort by `(time, row_index)` — the index
tie-break makes the stream deterministic without any RNG. `reverse` is
the exact reversal of the chrono sequence (the H-O2 order-sensitivity
diagnostic), not an independent sort.

This module owns ONLY the ordering and the windowing; it knows nothing
about folds, covers or training (the CV runner owns those)."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

STREAM_WINDOW = 1000


def stream_windows(
    rows_idx: Sequence[int],
    pt_times: Sequence[int],
    *,
    window: int = STREAM_WINDOW,
    order: str = "chrono",
) -> Iterator["torch.Tensor"]:
    """Yield long tensors of row ids from ``rows_idx`` — consecutive
    windows of ``window`` rows (the last window is the remainder) in
    globally non-decreasing ``pt_times`` order (ties: ascending row id);
    ``order="reverse"`` yields the exact reversal of that sequence.
    ``pt_times[k]`` is the time of row ``rows_idx[k]`` (same length)."""
    import torch

    if len(rows_idx) != len(pt_times):
        raise ValueError(
            f"rows_idx has {len(rows_idx)} entries but pt_times has "
            f"{len(pt_times)}; they must align one-to-one."
        )
    if window < 1:
        raise ValueError(f"window must be >= 1 (got {window!r}).")
    if order not in ("chrono", "reverse"):
        raise ValueError(f"order must be 'chrono' or 'reverse' (got {order!r}).")

    ranked = sorted(zip(pt_times, rows_idx))
    if order == "reverse":
        ranked.reverse()
    stream = [row for _, row in ranked]
    for lo in range(0, len(stream), window):
        yield torch.tensor(stream[lo:lo + window], dtype=torch.long)
