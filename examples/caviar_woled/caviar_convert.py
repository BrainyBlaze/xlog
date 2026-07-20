"""CAVIAR-to-relations converter for the WOLED star probe (task S2).

CAVIAR (`caviar_folds.pkl`) stores one (p1, p2) PAIR-WINDOW per datapoint: 24
timesteps of two people's simple activities (walking/active/inactive/running)
and coordinates, labeled with a per-timestep COMPLEX interaction label
(no_interaction/meeting/moving). This module flattens that into the same
PAIR-TIME entity space the star-topology engine mode (`enumerate_specs`,
``topology="star"`` -- see `pyxlog.ilp.neural_credit`) already consumes:
one row per (datapoint, timestep), a binary fact per row, and named ground
relations true at that row -- the same shape as the ingest bench's
``has_event``/``sal`` columns, just built from CAVIAR instead of synthetic
data.

PAIR-TIME ENCODING. Every datapoint is assumed to share ONE window length T
(read from ``complex_labels``, never hardcoded to 24 -- the real pkl happens
to use 24, but nothing here assumes it). The pair-time id is

    pt = dp_index * T + t                      (t in 0..T-1)

which only round-trips (``divmod(pt, T)``) when every datapoint has the SAME
T; `convert_split` raises if a later datapoint disagrees with the first
one's T rather than silently picking a stride that would collide ids.

``is_positive`` IS THE DIRECT (holdsAt-style) TARGET, honestly: it is `True`
exactly where the datapoint's own per-timestep complex label equals
"meeting" at that t. This is a v1 probe label -- CAVIAR's actual meeting
semantics are a temporal INITIATION/INERTIA process (a meeting interval
starts and then persists), and collapsing that to a per-timestep flag
throws away the interval structure. Refining `is_positive` into an
initiation-triggered, inertia-carried signal is explicitly OUT OF SCOPE
here and left for a later task.

FIXED DATASET VOCABULARY. `convert_split` decodes ``p1_labels``/``p2_labels``
and the per-timestep complex label through a small hardcoded id->name table
(`SIMPLE_LABEL_NAMES`, `COMPLEX_MEETING_ID`) rather than re-deriving it from
``simple_label_encoder``/``complex_label_encoder`` on every call, because
`convert_split` takes a plain list of datapoints (no encoder object) --
those tables were read from the real `caviar_folds.pkl` (top-level
``simple_label_encoder`` = {'active': 0, 'inactive': 1, 'running': 2,
'walking': 3}, ``complex_label_encoder`` = {'no_interaction': 0,
'meeting': 1, 'moving': 2}) and are shared across all three folds in that
file, so they are treated as fixed dataset vocabulary. Both are overridable
by keyword for a caller whose encoder disagrees.

COORDS AND MISSING DATA. Coordinates are parsed straight out of the
``atoms`` string with the SAME regex family the caviar loader itself uses:
``r"coords\\(p1,(\\d+),(\\d+),(\\d+)\\)"`` (X, Y, T) and the p2 twin. In the
real pkl every timestep of every datapoint has both p1 and p2 coords (0 /
12888 rows missing, checked across fold1's full train split), but the
converter does not assume that: a pair-time whose regex match is missing
for either person still gets its `facts` row (the entity always exists) and
is flagged in the ``"coords_missing"`` relation; it is OMITTED from both
``"close"`` and ``"far"`` (neither can be computed honestly without both
coordinates) and its `features` row is `(0.0, 0.0)` rather than a distance
computed from a missing point.

FEATURES. `features` is `[num_pt, 2]` float32, `(dx, dy) = (p1_x - p2_x,
p1_y - p2_y)` at that timestep, scaled by `1 / FEATURE_SCALE` (100.0,
documented) -- the raw perception input a future learned-"close" detector
would train against, not itself a relation.
"""

from __future__ import annotations

import pickle
import re
from typing import Any

import torch

# Read from caviar_folds.pkl's own top-level ``simple_label_encoder`` /
# ``complex_label_encoder`` -- see the module docstring for why these are
# hardcoded rather than threaded through as a parameter to every call.
SIMPLE_LABEL_NAMES: dict[int, str] = {
    0: "active",
    1: "inactive",
    2: "running",
    3: "walking",
}
COMPLEX_MEETING_ID = 1  # complex_label_encoder['meeting']

FEATURE_SCALE = 100.0

_COORDS_RE = {
    "p1": re.compile(r"coords\(p1,(\d+),(\d+),(\d+)\)"),
    "p2": re.compile(r"coords\(p2,(\d+),(\d+),(\d+)\)"),
}


def _to_int_list(x: Any) -> list[int]:
    """Flatten a per-timestep label tensor/list to a plain ``list[int]``.

    Accepts a torch tensor of shape ``[T, 1]`` or ``[T]`` (the real pkl's
    ``p1_labels``/``p2_labels``/``complex_labels`` shape), or a plain Python
    list of ints / length-1 lists (what a hand-built synthetic test
    datapoint can use without needing torch at all for its label fields)."""
    if hasattr(x, "flatten"):
        return [int(v) for v in x.flatten().tolist()]
    out = []
    for v in x:
        if isinstance(v, (list, tuple)):
            if len(v) != 1:
                raise ValueError(
                    f"expected a length-1 row per timestep, got {v!r} with "
                    f"{len(v)} entries."
                )
            out.append(int(v[0]))
        else:
            out.append(int(v))
    return out


def window_length(datapoint: dict) -> int:
    """The datapoint's own T, read from ``complex_labels`` -- never a
    hardcoded 24. Works whether ``complex_labels`` is a torch tensor
    ``[T, 1]`` or a plain list."""
    cl = datapoint["complex_labels"]
    if hasattr(cl, "shape"):
        return int(cl.shape[0])
    return len(cl)


def _parse_coords(atoms: str) -> tuple[dict[int, tuple[int, int]], dict[int, tuple[int, int]]]:
    """Parse ``coords(p1,X,Y,T)`` / ``coords(p2,X,Y,T)`` out of the atoms
    string into ``{t: (x, y)}`` maps, one per person. A timestep absent from
    either map is a genuine "coords missing" row, not a bug -- see the
    module docstring."""
    out = {}
    for person, pattern in _COORDS_RE.items():
        m: dict[int, tuple[int, int]] = {}
        for x_s, y_s, t_s in pattern.findall(atoms):
            m[int(t_s)] = (int(x_s), int(y_s))
        out[person] = m
    return out["p1"], out["p2"]


def convert_split(
    datapoints: list[dict],
    close_threshold: float = 25.0,
    simple_label_names: dict[int, str] | None = None,
    complex_meeting_id: int = COMPLEX_MEETING_ID,
) -> dict:
    """Flatten a CAVIAR fold split (a list of datapoint dicts) into the
    pair-time relation space described in the module docstring.

    Returns a dict:
      * ``"facts"``: ``list[(pt, 1)]``, one per pair-time, in the engine's
        star-mode fact convention (entity=pt, constant label column 1,
        n_labels=2).
      * ``"is_positive"``: ``list[bool]``, aligned with ``"facts"``; True
        iff the complex label at that pair-time is "meeting" (see the
        module docstring for the honest v1-probe caveat).
      * ``"relations"``: ``dict[str, list[(pt, 1)]]`` -- ground relations
        true at that pair-time: ``"both_active"``, ``"both_walking"``,
        ``"both_inactive"``, ``"mixed_active_walking"`` (decoded simple
        labels), ``"close"`` / ``"far"`` (euclidean p1-p2 distance vs.
        ``close_threshold``; ``close`` iff ``dist <= close_threshold``,
        so a distance exactly at the threshold is close, as in OLED's
        precomputed ``close/2``; omitted where coords are missing), and
        ``"coords_missing"`` (flags exactly those omitted rows).
      * ``"features"``: ``torch.FloatTensor[num_pt, 2]``, per-pair-time
        ``(dx, dy)`` scaled by ``1 / FEATURE_SCALE``; ``(0.0, 0.0)`` where
        coords are missing.
      * ``"num_pt"``: ``int``, ``len(datapoints) * T``.
      * ``"n_coords_missing"``: ``int``, count of pair-times flagged
        ``"coords_missing"``.
    """
    if not datapoints:
        raise ValueError("convert_split needs at least one datapoint.")
    simple_names = simple_label_names or SIMPLE_LABEL_NAMES

    T = window_length(datapoints[0])
    if T <= 0:
        raise ValueError(f"datapoint 0 has window length {T}; a window needs >= 1 timestep.")
    for i, dp in enumerate(datapoints):
        t_i = window_length(dp)
        if t_i != T:
            raise ValueError(
                f"datapoint {i} has window length {t_i}, but datapoint 0 has "
                f"{T}. The pair-time id `pt = dp_index * T + t` uses a SINGLE "
                "global stride T; a differing window length would make pt "
                "collide across datapoints, so this is refused rather than "
                "silently using the wrong stride."
            )

    num_pt = len(datapoints) * T
    facts: list[tuple[int, int]] = []
    is_positive: list[bool] = []
    relations: dict[str, list[tuple[int, int]]] = {
        "both_active": [],
        "both_walking": [],
        "both_inactive": [],
        "mixed_active_walking": [],
        "close": [],
        "far": [],
        "coords_missing": [],
    }
    feature_rows: list[tuple[float, float]] = []
    n_coords_missing = 0

    for dp_index, dp in enumerate(datapoints):
        p1_labels = _to_int_list(dp["p1_labels"])
        p2_labels = _to_int_list(dp["p2_labels"])
        complex_labels = _to_int_list(dp["complex_labels"])
        p1_coords, p2_coords = _parse_coords(dp["atoms"])

        for t in range(T):
            pt = dp_index * T + t
            facts.append((pt, 1))
            is_positive.append(complex_labels[t] == complex_meeting_id)

            p1_name = simple_names[p1_labels[t]]
            p2_name = simple_names[p2_labels[t]]
            if p1_name == "active" and p2_name == "active":
                relations["both_active"].append((pt, 1))
            if p1_name == "walking" and p2_name == "walking":
                relations["both_walking"].append((pt, 1))
            if p1_name == "inactive" and p2_name == "inactive":
                relations["both_inactive"].append((pt, 1))
            if (p1_name == "active" and p2_name == "walking") or (
                p1_name == "walking" and p2_name == "active"
            ):
                relations["mixed_active_walking"].append((pt, 1))

            c1 = p1_coords.get(t)
            c2 = p2_coords.get(t)
            if c1 is None or c2 is None:
                n_coords_missing += 1
                relations["coords_missing"].append((pt, 1))
                feature_rows.append((0.0, 0.0))
            else:
                dx = c1[0] - c2[0]
                dy = c1[1] - c2[1]
                feature_rows.append((dx / FEATURE_SCALE, dy / FEATURE_SCALE))
                dist = (dx * dx + dy * dy) ** 0.5
                # Tie rule: dist == close_threshold counts as close (<=), matching
                # OLED's close/2 which is defined as distance <= threshold.
                if dist <= close_threshold:
                    relations["close"].append((pt, 1))
                else:
                    relations["far"].append((pt, 1))

    features = torch.tensor(feature_rows, dtype=torch.float32).reshape(num_pt, 2)
    return {
        "facts": facts,
        "is_positive": is_positive,
        "relations": relations,
        "features": features,
        "num_pt": num_pt,
        "n_coords_missing": n_coords_missing,
    }


def load_folds(path: str) -> dict:
    """Thin pickle loader for ``caviar_folds.pkl`` -- kept separate from
    `convert_split` so the conversion core is unit-testable without the
    file on disk."""
    with open(path, "rb") as f:
        return pickle.load(f)


def build_star_schema_source(relation_names: list[str]) -> str:
    """The schema-only source for the star-topology program: one seed row
    per relation (`put_relation` REPLACES the relation wholesale, so the
    seed row's values never reach scoring -- it exists only so the compiler
    sees a 2-column relation, mirroring `bench_ingest.py`'s ``SCHEMA_ONLY``)
    plus the star dILP template rule over the auto-enumerated ``bL``/``bR``
    candidate slots."""
    lines = [f"{name}(0, 0)." for name in relation_names]
    lines.append("learnable(W) :: init_meeting(X, Y) :- bL(X, Y), bR(X, Y).")
    return "\n".join(lines) + "\n"


def put_caviar_relations(prog, converted: dict, n_labels: int = 2) -> str:
    """Ingest ``converted["relations"]`` into a compiled program via
    `put_relation`, as two uint32 CUDA tensor columns each (the same
    ``[pt...], [1...]`` pattern `bench_ingest.py`'s ``cols`` helper uses),
    and return the schema-only source string those relations should have
    been compiled from (`build_star_schema_source`).

    CUDA-ONLY: every relation ends up on ``device="cuda"`` because
    `put_relation`'s DLPack columns feed the engine's own CUDA program.
    ``torch`` itself imports fine on CPU (this function is only called, not
    imported, on a CPU-only machine), so the guard here is a clear refusal
    at CALL TIME rather than a cryptic backend error partway through the
    first `put_relation`.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            "put_caviar_relations builds CUDA tensor columns for "
            "put_relation (torch.utils.dlpack.to_dlpack over a "
            "device='cuda' tensor) and needs a CUDA device -- none is "
            "available here. Convert on CPU with convert_split, then call "
            "this only where CUDA is present."
        )
    import torch.utils.dlpack as dlpack

    def cols(pairs: list[tuple[int, int]]):
        a = torch.tensor([p[0] for p in pairs], dtype=torch.uint32, device="cuda")
        b = torch.tensor([p[1] for p in pairs], dtype=torch.uint32, device="cuda")
        return [dlpack.to_dlpack(a), dlpack.to_dlpack(b)]

    for name, pairs in converted["relations"].items():
        for _, y in pairs:
            if not (0 <= y < n_labels):
                raise ValueError(
                    f"relation '{name}' has a row with label column {y}, but "
                    f"n_labels={n_labels} (valid columns 0..{n_labels - 1}). "
                    "Refused rather than silently ingested out of range."
                )
        prog.put_relation(name, cols(pairs))

    return build_star_schema_source(sorted(converted["relations"]))
