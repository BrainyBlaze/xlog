"""Loader/converter for the ORIGINAL continuous Event-Calculus CAVIAR dump
(`caviar-train.json` / `caviar-test.json`, the OLED/WOLED line-JSON MongoDB
export) -- the exact data the OLED system trained on, and the source
`caviar_convert.py`'s `caviar_folds.pkl` was itself windowed out of.

WHY A SEPARATE MODULE, NOT `caviar_convert.convert_split` REUSE.
`convert_split` REFUSES datapoints whose window length T disagrees (a single
global `pt = dp_index * T + t` stride only round-trips when every datapoint
shares one T -- see `caviar_convert._shared_window_length`). This dataset has
no fixed T at all: it is 28 CAVIAR videos concatenated on one continuous 40ms
timeline, with 21 (train) / 5 (test) video segments of WILDLY different
lengths recovered from non-40ms timestamp jumps (train segments range from a
handful of frames to thousands; see `load_continuous`). Re-chunking this into
fixed-T windows to fit `convert_split`'s contract would silently re-introduce
the exact artifact this dataset exists to avoid: the pkl's own 24-frame
windows treat "already holding at window start" as an initiation because the
window has no earlier frame to show otherwise (`caviar_convert.
derive_ec_targets`'s own documented choice) -- a re-windowing artifact, not a
real event. The continuous timeline has NO such gaps except at genuine video
boundaries, so it can report REAL initiations and REAL terminations only.
This module therefore builds a PARALLEL pair (`convert_continuous`,
`derive_ec_targets_continuous`) that emits the SAME output contract keys
`convert_split`/`derive_ec_targets` do (so `run_caviar_theory.py`'s induction
and scoring code, which only reads those keys, needs no change to consume
either data source) plus one addition, `segment_of_pt`, so nothing at any
call site can accidentally let a pair-time row's neighbor comparison cross a
video boundary.

DATA SHAPE. One JSON document per line; keys `_id`, `time`, `narrative`
(list of atom strings), `annotation` (list of atom strings) -- `_id`/`time`
are read by nothing here (every meaningful value already lives in the atom
strings' own trailing time argument). Narrative atoms of interest:
`happensAt(ACTIVITY(id),T)` (ACTIVITY in walking/active/inactive/running/
abrupt/appear/disappear) and `coords(id,X,Y,T)`. Annotation atoms: `holdsAt(
meeting(P1,P2),T)`, symmetric (both `(p1,p2)` and `(p2,p1)` logged with
identical timestamps).

DEDUP. `caviar-train.json` has every document twice (byte-identical
narrative/annotation) and BOTH files have consecutive chunk-documents sharing
their boundary frame's atoms. Both quirks are handled the same way: every
atom string across every document in the file is folded into one global SET
before any parsing happens, so an atom repeated across duplicate docs or
shared boundary frames is only ever counted once, with no per-document
bookkeeping needed at all.

VIDEO SEGMENTATION. The 40ms grid is reconstructed from every narrative
atom's own trailing time argument (not `happensAt`/`coords` alone -- every
narrative atom's timestamp counts, so segmentation sees the full narrative
timeline), sorted, and split wherever a consecutive gap is not exactly 40ms
-- each such jump is a splice point between two originally-distinct CAVIAR
videos concatenated onto one timeline.

THE ANNOTATION OFFSET -- VERIFIED, NOT ASSUMED. `holdsAt(meeting(p1,p2),T)`'s
own `T` is used AS-IS to key the gold label at narrative timestep `T` --
no shift is applied anywhere in this module. This was checked two ways
against the real train/test files (see the module's own exploration, not
reproduced here as code, only as evidence):
  (1) Every annotated timestamp already sits on the SAME 40ms narrative grid
      (0 of 1,344 train / 489 test annotation timestamps are absent from the
      narrative timestamp set), so no shift is needed just to make lookups
      resolve.
  (2) DECISIVE test: counting real 0->1 / 1->0 transitions of the
      `holdsAt(meeting,T)` signal directly on `T` (no shift) reproduces the
      known per-ordered-pair breakdown exactly -- train 22 initiations / 22
      terminations over the 5 unordered pairs (id0,id1):3/3,
      (id1,id2):5/5, (id1,id3):1/1, (id2,id6):1/1, (id5,id6):1/1 (a
      single-frame meeting) -- and test 6 initiations / 4 terminations,
      with the one KNOWN truncated case ((id1,id2), a single annotated frame
      adjacent to a video boundary: initiation counted, termination clipped
      by the boundary). Shifting the annotation back by one frame (T -> T -
      40) instead gives test 6/6 -- it manufactures a termination for the
      boundary-truncated case that the source data does not actually show
      happening before the video ends. Only the no-shift reading reproduces
      the known, boundary-aware counts, so that is the one this module uses.
  The "+40ms" language some notes attach to this dataset describes a
  SEMANTIC convention (the classic EC reading: an initiating event observed
  at T causes the fluent to hold starting at T+40, one frame later -- visible
  above as e.g. id1/id3's distance already having crossed the "close" range
  one frame before their first annotated meeting frame) -- not a numeric
  correction this module needs to apply: `holdsAt(meeting(p1,p2),T)` already
  names the timestep the fluent holds AT, on the real grid, directly.
"""

from __future__ import annotations

import json
import re
from typing import Any

import torch

from caviar_convert import FEATURE_SCALE

ACTIVITY_NAMES = frozenset({"walking", "active", "inactive", "running", "abrupt"})

_HAPPENS_RE = re.compile(r"happensAt\((\w+)\((id\d+)\),(\d+)\)")
_COORDS_RE = re.compile(r"coords\((id\d+),(-?\d+),(-?\d+),(\d+)\)")
_MEETING_RE = re.compile(r"holdsAt\(meeting\((id\d+),(id\d+)\),(\d+)\)")
_TRAILING_TIME_RE = re.compile(r",(\d+)\)$")


def _person_num(pid: str) -> int:
    """`idN` -> `N`, so pairs/persons sort numerically even if a future dump
    ever has 2+ digit ids (this dump's own ids are single-digit id0..id9,
    where string and numeric order already agree, but nothing here assumes
    that)."""
    return int(pid[2:])


def _canonical_pair(p1: str, p2: str) -> tuple[str, str]:
    """The pair's fixed, orderless identity: the lower-numbered id first,
    always -- so `(id2, id6)` and `(id6, id2)` (both logged, symmetrically,
    in the annotation) collapse onto the SAME canonical key everywhere in
    this module."""
    return tuple(sorted((p1, p2), key=_person_num))


def load_continuous(
    path: str,
    *,
    expected_num_timepoints: int | None = None,
    expected_num_segments: int | None = None,
) -> list[dict]:
    """Parse a `caviar-train.json`/`caviar-test.json` line-JSON file into a
    list of per-video SEGMENT dicts, each:

      * ``"timestamps"``: ``list[int]``, this segment's own sorted 40ms grid.
      * ``"persons"``: ``list[str]``, every person id (``idN``) with at least
        one activity event somewhere in this segment, sorted by
        `_person_num`.
      * ``"activity"``: ``dict[(person_id, t), str]`` -- decoded
        `happensAt(ACTIVITY(id),T)` events, restricted to
        `ACTIVITY_NAMES` (appear/disappear are events too, but carry no
        simple-activity label, so they are not stored here).
      * ``"coords"``: ``dict[(person_id, t), (x, y)]``.
      * ``"meeting"``: ``set[(min_id, max_id, t)]`` -- the annotation,
        canonical-paired and deduplicated (both symmetric ordered atoms
        collapse onto one entry).

    Every atom string across every document in the file is folded into one
    global set BEFORE parsing (see the module docstring's DEDUP paragraph),
    so `caviar-train.json`'s exact doc duplication and both files' shared
    chunk-boundary frames are handled with no per-document bookkeeping.

    REFUSES LOUDLY (``ValueError``) if ``expected_num_timepoints`` or
    ``expected_num_segments`` is given and the parsed file disagrees -- this
    dataset's exact timepoint/segment counts were verified once against the
    real files (train: 22366 timepoints / 21 segments; test: 3248 / 5; see
    the report this module shipped with) and a silent drift from those counts
    (a different download, a corrupted file, an upstream re-export) would
    invalidate every count derived downstream, so callers loading the REAL
    files should always pass the expected counts; synthetic test fixtures
    pass neither and skip the check.
    """
    narrative_atoms: set[str] = set()
    annotation_atoms: set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            narrative_atoms.update(doc["narrative"])
            annotation_atoms.update(doc["annotation"])

    activity: dict[tuple[str, int], str] = {}
    coords: dict[tuple[str, int], tuple[int, int]] = {}
    all_ts: set[int] = set()
    for atom in narrative_atoms:
        m = _HAPPENS_RE.match(atom)
        if m:
            act, pid, t_s = m.groups()
            t = int(t_s)
            all_ts.add(t)
            if act in ACTIVITY_NAMES:
                activity[(pid, t)] = act
            continue
        m = _COORDS_RE.match(atom)
        if m:
            pid, x_s, y_s, t_s = m.groups()
            t = int(t_s)
            all_ts.add(t)
            coords[(pid, t)] = (int(x_s), int(y_s))
            continue
        # orientation(...)/holdsAt(visible|occluded,...) atoms: not read for
        # anything but their OWN timestamp -- still folded into all_ts so
        # segmentation sees every narrative timepoint, not just the subset
        # happensAt/coords touch (in practice a strict superset of it here).
        m = _TRAILING_TIME_RE.search(atom)
        if m:
            all_ts.add(int(m.group(1)))

    meeting: set[tuple[str, str, int]] = set()
    for atom in annotation_atoms:
        m = _MEETING_RE.match(atom)
        if m:
            p1, p2, t_s = m.groups()
            meeting.add((*_canonical_pair(p1, p2), int(t_s)))

    if not all_ts:
        raise ValueError(f"{path}: no narrative timestamps found -- refusing to segment an empty timeline.")

    ts_sorted = sorted(all_ts)
    if expected_num_timepoints is not None and len(ts_sorted) != expected_num_timepoints:
        raise ValueError(
            f"{path}: found {len(ts_sorted)} unique narrative timepoints, "
            f"expected {expected_num_timepoints} -- refusing rather than "
            "silently proceeding on a dataset that has drifted from the "
            "counts this converter was built and verified against."
        )

    segments_ts: list[list[int]] = []
    cur = [ts_sorted[0]]
    for prev, nxt in zip(ts_sorted, ts_sorted[1:]):
        if nxt - prev == 40:
            cur.append(nxt)
        else:
            segments_ts.append(cur)
            cur = [nxt]
    segments_ts.append(cur)

    if expected_num_segments is not None and len(segments_ts) != expected_num_segments:
        raise ValueError(
            f"{path}: found {len(segments_ts)} video segments (split on "
            f"non-40ms narrative timestamp jumps), expected "
            f"{expected_num_segments} -- refusing for the same reason as "
            "the timepoint count check above."
        )

    t_to_seg: dict[int, int] = {}
    for seg_idx, seg_ts in enumerate(segments_ts):
        for t in seg_ts:
            t_to_seg[t] = seg_idx

    seg_persons: list[set] = [set() for _ in segments_ts]
    seg_activity: list[dict] = [dict() for _ in segments_ts]
    seg_coords: list[dict] = [dict() for _ in segments_ts]
    seg_meeting: list[set] = [set() for _ in segments_ts]

    for (pid, t), act in activity.items():
        seg_idx = t_to_seg[t]
        seg_activity[seg_idx][(pid, t)] = act
        seg_persons[seg_idx].add(pid)
    for (pid, t), xy in coords.items():
        seg_coords[t_to_seg[t]][(pid, t)] = xy
    for p1, p2, t in meeting:
        seg_meeting[t_to_seg[t]].add((p1, p2, t))

    return [
        {
            "timestamps": seg_ts,
            "persons": sorted(seg_persons[i], key=_person_num),
            "activity": seg_activity[i],
            "coords": seg_coords[i],
            "meeting": seg_meeting[i],
        }
        for i, seg_ts in enumerate(segments_ts)
    ]


def _iter_pair_rows(segments: list[dict]):
    """Yield ``(segment_index, t, p1, p2)`` for every unordered pair of
    persons CO-VISIBLE (both have an activity event) at a timestep, one row
    per pair per timestep -- the SAME deterministic order `convert_continuous`
    and `_group_pts_by_pair` both rely on: segment index ascending, then
    timestep ascending within the segment, then pair ascending (canonical,
    since ``segment["persons"]`` is already sorted by `_person_num` and this
    walks it as a fixed ordered list)."""
    for seg_idx, seg in enumerate(segments):
        for t in seg["timestamps"]:
            present = [pid for pid in seg["persons"] if (pid, t) in seg["activity"]]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    yield seg_idx, t, present[i], present[j]


def convert_continuous(segments: list[dict], close_threshold: float = 25.0) -> dict:
    """Flatten continuous per-video segments into the SAME pair-time
    relation space `caviar_convert.convert_split` builds for the windowed
    pkl -- see that module's docstring for the relation/close-far/features
    semantics, reproduced identically here (activity names are read directly
    off the narrative, not decoded through `SIMPLE_LABEL_NAMES`, since this
    dataset already spells them out as strings).

    Returns the SAME keys `convert_split` does (``"facts"``, ``"is_positive"``,
    ``"relations"``, ``"features"``, ``"num_pt"``, ``"n_coords_missing"``)
    PLUS ``"segment_of_pt"``: ``list[int]``, the video segment each pair-time
    row belongs to, aligned with every other per-pt list -- so any caller
    comparing neighboring pair-times can refuse to cross a video boundary by
    checking this list, exactly as `derive_ec_targets_continuous` and
    `reconstruct_holds_continuous` do internally via `_group_pts_by_pair`.

    ``pt`` is a plain sequential row index (0..num_pt-1) in `_iter_pair_rows`'s
    own order -- NOT `caviar_convert`'s ``dp_index * T + t`` formula, which
    assumes one fixed-T datapoint's worth of pt's are the same, single, pair;
    here potentially MANY pairs are co-visible at one timestep, so there is
    no single stride to derive it from.
    """
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
    segment_of_pt: list[int] = []
    n_coords_missing = 0

    for pt, (seg_idx, t, p1, p2) in enumerate(_iter_pair_rows(segments)):
        seg = segments[seg_idx]
        facts.append((pt, 1))
        segment_of_pt.append(seg_idx)
        is_positive.append((p1, p2, t) in seg["meeting"])

        a1 = seg["activity"][(p1, t)]
        a2 = seg["activity"][(p2, t)]
        if a1 == "active" and a2 == "active":
            relations["both_active"].append((pt, 1))
        if a1 == "walking" and a2 == "walking":
            relations["both_walking"].append((pt, 1))
        if a1 == "inactive" and a2 == "inactive":
            relations["both_inactive"].append((pt, 1))
        if (a1 == "active" and a2 == "walking") or (a1 == "walking" and a2 == "active"):
            relations["mixed_active_walking"].append((pt, 1))

        c1 = seg["coords"].get((p1, t))
        c2 = seg["coords"].get((p2, t))
        if c1 is None or c2 is None:
            n_coords_missing += 1
            relations["coords_missing"].append((pt, 1))
            feature_rows.append((0.0, 0.0))
        else:
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            feature_rows.append((dx / FEATURE_SCALE, dy / FEATURE_SCALE))
            dist = (dx * dx + dy * dy) ** 0.5
            # Tie rule matches convert_split's own: dist == threshold is close.
            if dist <= close_threshold:
                relations["close"].append((pt, 1))
            else:
                relations["far"].append((pt, 1))

    num_pt = len(facts)
    features = torch.tensor(feature_rows, dtype=torch.float32).reshape(num_pt, 2)
    return {
        "facts": facts,
        "is_positive": is_positive,
        "relations": relations,
        "features": features,
        "num_pt": num_pt,
        "n_coords_missing": n_coords_missing,
        "segment_of_pt": segment_of_pt,
    }


def _group_pts_by_pair(segments: list[dict], num_pt: int) -> dict[tuple, list[int]]:
    """Re-walk `_iter_pair_rows` in lockstep with a `converted` dict's own
    ``pt`` numbering (0..num_pt-1, in the SAME order `convert_continuous`
    built it in) and group pt's by ``(segment_index, p1, p2)``, each group's
    list already in ascending time order -- the shared grouping both
    `derive_ec_targets_continuous` and `reconstruct_holds_continuous` need to
    walk one pair's own observed-co-visible timeline within one video
    segment. Refuses (``ValueError``) if `segments` does not reproduce
    exactly ``num_pt`` rows -- the only way that can happen is calling this
    (indirectly, via either of the two functions above) with a DIFFERENT
    `segments` list than the one that built `converted`.
    """
    groups: dict[tuple, list[int]] = {}
    it = _iter_pair_rows(segments)
    for pt in range(num_pt):
        try:
            seg_idx, _t, p1, p2 = next(it)
        except StopIteration:
            raise ValueError(
                f"segments/converted mismatch: iterating segments produced "
                f"fewer pair-time rows than converted's own num_pt={num_pt} "
                f"(stopped at pt={pt}) -- pass the SAME segments list that "
                "built `converted` via convert_continuous."
            ) from None
        groups.setdefault((seg_idx, p1, p2), []).append(pt)
    if next(it, None) is not None:
        raise ValueError(
            f"segments/converted mismatch: iterating segments produced MORE "
            f"pair-time rows than converted's own num_pt={num_pt} -- pass "
            "the SAME segments list that built `converted` via "
            "convert_continuous."
        )
    return groups


def derive_ec_targets_continuous(
    segments: list[dict],
    converted: dict,
    *,
    treat_first_observed_as_init: bool = False,
) -> dict:
    """Event-Calculus (initiatedAt/terminatedAt) targets on the CONTINUOUS
    timeline: one independent transition sequence PER (video segment,
    canonical pair), walked over only that pair's own observed co-visible
    timesteps within that segment (never across a segment boundary, and
    never across a gap where the pair was not co-visible at all -- see
    `_group_pts_by_pair`).

    DEFAULT (``treat_first_observed_as_init=False``): counts ONLY REAL
    transitions. ``is_init`` is True at a pair-time iff the meeting label is
    True there AND False at that pair's immediately preceding OBSERVED
    co-visible pair-time in the SAME segment -- a pair's very FIRST observed
    co-visible pair-time in a segment is NEVER marked init even if the label
    is already True there, because "already holding when first observed" is
    not a transition this dataset shows happening. That first-observed
    pair-time is not always the segment's OWN first timestamp: it can just as
    well be a pair becoming co-visible partway through a segment (one person
    absent/untracked before then) -- verified on the real train file, where
    exactly this happens once ((id0, id1) in the segment spanning
    507080-570760: they are co-visible for only 412 of that segment's 1593
    timesteps, and the meeting label is already True the first time they are
    both tracked). Either way, "no earlier co-visible frame to compare
    against" is treated the same as `caviar_convert.derive_ec_targets`'s
    windowed ``t == 0`` case -- except here it is an honest "not observed",
    not a re-windowing artifact, because a REAL earlier frame exists in the
    timeline; this dataset's pair-time rows just do not cover it (no row
    exists for a non-co-visible pair -- see `convert_continuous`).
    ``is_term`` is True iff the label is False AND the immediately preceding
    observed co-visible pair-time was True -- never inferred past a pair's
    LAST observed co-visible pair-time in a segment: a meeting that is still
    True there, or that the source annotation resolves to False only AFTER
    the pair stops being co-visible (e.g. one person disappears while still
    "meeting"), produces no terminating pair-time at all -- nothing is
    extrapolated beyond what an existing row could show. Verified on the real
    test file: (id4, id5)'s meeting ends this way (both persons drop out of
    the narrative, mid-"meeting", before any co-visible False frame is
    observed), so it contributes an init but no term here.

    Both variants' totals are always returned, so a caller can see the
    difference `treat_first_observed_as_init` would make without a second
    call: ``"n_init_real_transitions_only"`` and
    ``"n_init_including_first_observed_holding"`` (the latter additionally
    counts a pair's first-observed-and-already-holding pair-time as an
    init -- the reading closest to `caviar_convert.derive_ec_targets`'s own
    windowed ``t == 0`` convention). ``"n_init"``/``"is_init"`` reflect
    whichever the ``treat_first_observed_as_init`` flag selects; ``"is_term"``/
    ``"n_term"`` do not depend on the flag at all (a termination is never
    ambiguous the way a first-observed holding is).

    RECONCILED AGAINST THE known per-ordered-pair annotation transition count
    (22 initiations / 22 terminations, train; 6 / 4, test -- counted directly
    off the raw `holdsAt(meeting,T)` signal without regard to co-visibility,
    treating a gap in either person's own tracking as an implicit "not
    meeting"): on the real files this function's DEFAULT (real-transitions-
    only, co-visibility-scoped) gives train 10 inits / 11 terms and test 3
    inits / 1 term over 11 (train) / 3 (test) canonical meeting intervals --
    fewer than the raw-annotation reading specifically where an
    initiation or termination falls in a gap where the pair was not mutually
    co-visible (the (id0,id1)/(id4,id5) cases above). ``treat_first_observed_
    as_init=True`` recovers the raw reading's init count exactly (train 11,
    test 3); there is no equivalent term-side toggle, because unlike an
    already-holding first frame, a termination that happens off-camera has no
    pair-time row to honestly attach it to at all.

    Returns ``{"is_init", "is_term", "n_init", "n_term",
    "n_init_real_transitions_only", "n_init_including_first_observed_holding",
    "num_pt"}``, all aligned with ``converted``'s own ``pt`` numbering.
    """
    num_pt = converted["num_pt"]
    is_positive = converted["is_positive"]
    if len(is_positive) != num_pt:
        raise ValueError(
            f"converted['is_positive'] has {len(is_positive)} entries but "
            f"converted['num_pt']={num_pt} -- pass convert_continuous's own "
            "unmodified output."
        )

    groups = _group_pts_by_pair(segments, num_pt)

    is_init_real = [False] * num_pt
    is_init_incl_start = [False] * num_pt
    is_term = [False] * num_pt

    for pts in groups.values():
        prev_pos: bool | None = None
        for idx, pt in enumerate(pts):
            pos = is_positive[pt]
            if idx == 0:
                if pos:
                    is_init_incl_start[pt] = True
            else:
                if pos and not prev_pos:
                    is_init_real[pt] = True
                    is_init_incl_start[pt] = True
                if (not pos) and prev_pos:
                    is_term[pt] = True
            prev_pos = pos

    chosen_init = is_init_incl_start if treat_first_observed_as_init else is_init_real
    return {
        "is_init": chosen_init,
        "is_term": is_term,
        "n_init": sum(chosen_init),
        "n_term": sum(is_term),
        "n_init_real_transitions_only": sum(is_init_real),
        "n_init_including_first_observed_holding": sum(is_init_incl_start),
        "num_pt": num_pt,
    }


def reconstruct_holds_continuous(
    init_pred: list[bool], term_pred: list[bool], segments: list[dict], converted: dict,
) -> list[bool]:
    """`ec_scorer.reconstruct_holds`'s inertia closure, generalized from
    "one state machine per fixed-length window" to "one state machine per
    (video segment, canonical pair) observed co-visible run" -- the
    continuous timeline's segments/pairs have no fixed length, so
    `ec_scorer.reconstruct_holds`'s ``num_windows * T`` indexing does not
    apply here; grouping is delegated to the SAME `_group_pts_by_pair`
    `derive_ec_targets_continuous` uses, so the two stay consistent by
    construction. Same SIMULTANEOUS INIT+TERM RULE as `ec_scorer.
    reconstruct_holds` (term clears first, then init re-sets in the same
    step -- see that module's docstring for the justification, unchanged
    here) and the same per-group state reset (a pair's state in one segment
    never carries into another segment, or into a different pair).
    """
    num_pt = converted["num_pt"]
    if len(init_pred) != num_pt:
        raise ValueError(
            f"reconstruct_holds_continuous: init_pred has {len(init_pred)} "
            f"entries, expected converted['num_pt']={num_pt}."
        )
    if len(term_pred) != num_pt:
        raise ValueError(
            f"reconstruct_holds_continuous: term_pred has {len(term_pred)} "
            f"entries, expected converted['num_pt']={num_pt}."
        )

    groups = _group_pts_by_pair(segments, num_pt)
    holds = [False] * num_pt
    for pts in groups.values():
        state = False
        for pt in pts:
            if term_pred[pt]:
                state = False
            if init_pred[pt]:
                state = True
            holds[pt] = state
    return holds
