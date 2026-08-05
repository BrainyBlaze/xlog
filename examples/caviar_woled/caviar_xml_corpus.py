"""Loader/converter for the ORIGINAL CAVIAR ground-truth XML corpus (the 30
per-video files CAVIAR itself ships, `<dataset name="..."><frame
number="N"><objectlist>/<grouplist>...`) -- as opposed to
`caviar_continuous.py`'s OLED/WOLED line-JSON MongoDB dump
(`caviar-train.json`/`caviar-test.json`), which is itself DERIVED from this
same ground truth by OLED's own `ParseCAVIAR.scala`. This module builds an
equivalent corpus directly from the XML, for a CHOSEN target fluent
(``"meeting"`` or ``"moving"``), with REAL per-video boundaries -- no
video-splicing ambiguity of any kind, because every XML file already IS one
complete, un-spliced video.

WHY THIS EXISTS, NOT JUST `caviar_continuous.load_continuous` REUSE. The
dump's own "video segment" boundaries are RECOVERED after the fact, by
looking for non-40ms jumps in one continuous concatenated timeline -- and
that reconstruction is only ever approximate: OLED's own video-concatenation
convention (`ParseCAVIAR`'s `pastTime = lastTime + 40`) glues each next
video's local time axis onto the previous video's last timestamp plus
EXACTLY one ordinary frame step, so a genuine splice is numerically
indistinguishable from an ordinary consecutive frame, and a dump "segment"
can silently contain two or three different source videos end to end. The
ground-truth XML has no such ambiguity: each of the 30 files is one real
video, frame-numbered from its own zero, so `load_xml_corpus`'s own
per-video boundaries are exact by construction, never reconstructed.

XML SCHEMA (verified empirically across all 30 files: exactly one
`<hypothesis>` per object/group entry, always; the same `<hypothesis>`
carries movement/role/context for objects and movement/role/context/
situation for groups):

    <frame number="N">
      <objectlist>
        <object id="K">
          <orientation>DEG</orientation>
          <box xc="X" yc="Y" w="W" h="H"/>
          <appearance>appear|disappear|visible|occluded</appearance>
          <hypothesislist><hypothesis ...>
            <movement evaluation="1.0">walking|active|inactive|running</movement>
            <situation evaluation="1.0">moving|inactive|browsing</situation>
          </hypothesis></hypothesislist>
        </object>
      </objectlist>
      <grouplist>
        <group id="K">
          <members>0,1</members>   <!-- comma-separated OBJECT ids, own local space -->
          <hypothesislist><hypothesis ...>
            <situation evaluation="1.0">moving|interacting|fighting|joining|
                                         split up|leaving object|leaving victim</situation>
          </hypothesis></hypothesislist>
        </group>
      </grouplist>
    </frame>

TIME BASE. ``T = frame_number * frame_ms`` (``frame_ms`` defaults to 40, the
CAVIAR camera's own frame period) for every atom in a video -- individual
narrative events AND group-derived fluent atoms alike use exactly this same
formula, with no separate shift between the two: a group's own `situation`
label at frame N names a fluent holding at the SAME timestamp N*frame_ms an
individual's own `movement` label at frame N does. ``time_offset`` (default
0) shifts a WHOLE video's own timeline by one constant, so one video's local
clock can be placed onto an external clock (e.g. the OLED dump's own
timeline) when needed -- it is never used to glue two different videos onto
one timeline the way OLED's own dump construction does; `load_xml_corpus`
always keeps one video == one independent segment.

LLE (individual) MAPPING. An object's own `<movement>` label
(walking/active/inactive/running) is that object's narrative activity event
at its own ``(id, T)``. `<appearance>` (appear/disappear/visible/occluded)
and `<orientation>` (degrees) are recorded too. `coords` = the object's own
`<box xc="..." yc="...">` attributes taken directly as the box's own center
pixel (CAVIAR's own box already reports its center, not a corner) -- the
same `(X, Y)` convention the OLED dump's own `coords(id,X,Y,T)` atoms use.

HLE (group) MAPPING -> fluent name, verbatim from OLED's own
`ParseCAVIAR.scala` (`nkatzz/OLED`,
`src/main/scala/experiments/datautils/caviar_data/ParseCAVIAR.scala`):
``moving -> moving``, ``fighting -> fighting``,
``"leaving object" -> leavingObject``, ``interacting -> meeting``. The
group's own `situation` string ``"leaving object"`` (a space) is XML's own
spelling; ParseCAVIAR's own mapping key uses an underscore because it reads
the same content from an intermediate Prolog-atom transcription, not the
XML directly -- this loader normalizes over that spelling difference, not
the semantics. `joining`, `"split up"`, and `"leaving victim"` situations
have NO entry in that mapping and are SILENTLY DROPPED here too, exactly as
OLED's own parser combinator would fail to recognize and drop them (a
converter meaning to reproduce OLED's own target set must drop these three,
not merely happen to omit them).

PAIR GENERATION, verbatim from the same `ParseCAVIAR.scala` source
(`AnnotationAtom.atoms`): ``leavingObject`` uses ONLY the group's first two
listed members, emitting ONE ordered pair (`persons.head`, `persons(1)`);
every OTHER mapped HLE (``moving``, ``fighting``, ``interacting``/meeting)
emits ALL 2-subsets of the group's member set, in BOTH orders -- the same
symmetric convention the OLED dump's own annotation uses (both
`holdsAt(meeting(p1,p2),T)` and `holdsAt(meeting(p2,p1),T)` logged, same
timestamp). Only ``moving`` groups are ever seen with more than 2 members in
the real ground truth (3 or 4-member groups occur only there); every other
mapped situation always has exactly 2 members, so its own "all 2-subsets,
both orders" reduces to the ordinary symmetric pair, same as
``leavingObject``'s own single pair reduces to one direction only.

CO-VISIBILITY. Two objects are co-visible at a frame iff BOTH have a tracked
`<object id=...>` ENTRY there, in ANY appearance state (appear, visible,
occluded, or disappear all count) -- a broader test than "both have a
`<movement>` label", even though the two happen to coincide across the real
ground truth (every single object entry recorded there does carry a
`<hypothesislist>`/`<movement>`, occluded or not). This loader tracks
presence and activity as two SEPARATE structures on purpose, so a synthetic
or future video missing a movement label on a tracked-but-occluded object
still counts as co-visible, exactly as CAVIAR's own domain concept intends.

API. `load_xml_corpus` parses every ground-truth XML file (`*.xml` AND
`*.xml_` -- two of the 30 real files carry the trailing-underscore
extension) in a directory into one per-video dict each (see `parse_video`'s
own docstring for the dict's exact shape); `convert_xml_corpus` flattens a
list of such per-video dicts into the SAME pair-time relation-space contract
`caviar_continuous.convert_continuous` builds (``"facts"``, ``"is_positive"``,
``"relations"``, ``"features"``, ``"num_pt"``, ``"n_coords_missing"``,
``"segment_of_pt"``, ``"transition_relations"``), parameterized by
``fluent`` (``"meeting"`` or ``"moving"``) instead of that function's fixed
meeting target -- ``segment_of_pt`` here always names a REAL video index
(``videos[segment_of_pt[pt]]``), never a reconstructed, possibly-spliced
dump segment. `count_fluent_transitions` gives the corpus-level
interval/initiation/termination/pair-frame counts for any mapped fluent
(``meeting``/``moving``/``fighting``/``leavingObject``), walked over each
unordered pair's own co-visible frames within one video (never across
videos) -- a pair's very first co-visible frame is never counted as an
initiation even if the fluent already holds there (no earlier frame exists
to show a transition happening), the same convention
`derive_ec_targets_continuous` uses on the dump's own continuous timeline.

This module does not reimplement any downstream induction/scoring logic --
only the data structures those callers already consume.
"""

from __future__ import annotations

import glob
import os
import xml.etree.ElementTree as ET
from collections import defaultdict

FRAME_MS_DEFAULT = 40

# Matches `caviar_convert.FEATURE_SCALE` / `caviar_continuous.py`'s own
# feature scaling, so a `(dx, dy)` feature row means the same physical
# quantity regardless of which loader produced it.
FEATURE_SCALE = 100.0

# Object-level `<movement>` labels that become narrative activity events.
# CAVIAR's ground truth never records `abrupt` at the object level (the
# OLED dump's own narrative vocabulary includes it, sourced from elsewhere).
LLE_ACTIVITY_NAMES = frozenset({"walking", "active", "inactive", "running"})

# Group-level `<situation>` -> fluent-predicate name, verbatim from OLED's
# own ParseCAVIAR.scala `hleMapping` (see module docstring). `joining`,
# "split up", and "leaving victim" have no entry here and are dropped.
HLE_MAPPING: dict[str, str] = {
    "moving": "moving",
    "fighting": "fighting",
    "leaving object": "leavingObject",
    "interacting": "meeting",
}

# Situations whose pair generation is the single-ordered-pair special case
# (first two members only) rather than "all 2-subsets, both orders".
_SINGLE_PAIR_SITUATIONS = frozenset({"leaving object"})

TRANSITION_RELATION_NAMES: tuple[str, ...] = (
    "any_became_active", "any_became_inactive", "any_became_walking", "any_stopped_walking",
    "became_far", "distance_increasing",
)


def _person_num(pid: str) -> int:
    """``idN`` -> ``N``, so persons/pairs sort numerically regardless of
    digit count (``id10`` sorts after ``id9``, not before it as a plain
    string compare would)."""
    return int(pid[2:])


def _canonical_pair(p1: str, p2: str) -> tuple[str, str]:
    """The pair's orderless identity: lower-numbered id first -- collapses
    the ground truth's own symmetric ``(p1, p2)``/``(p2, p1)`` pair
    generation onto one key wherever a caller needs unordered counting."""
    return tuple(sorted((p1, p2), key=_person_num))


def parse_video(path: str, *, frame_ms: int = FRAME_MS_DEFAULT, time_offset: int = 0) -> dict:
    """Parse ONE CAVIAR ground-truth XML file into a per-video dict:

      * ``"video"``: the file's own base name (e.g. ``"wk2gt.xml"``).
      * ``"frame_ms"`` / ``"time_offset"``: the time-base parameters this
        call used (recorded so a caller can recover ``T -> frame_number``
        without re-deriving the arithmetic).
      * ``"timestamps"``: ``list[int]``, this video's own sorted, complete
        frame grid (``frame_number * frame_ms + time_offset`` for EVERY
        frame in the file, whether or not any object is tracked there).
      * ``"persons"``: ``list[str]`` (``idK``), every object id tracked at
        least once, sorted by `_person_num`.
      * ``"tracked"``: ``set[(person_id, t)]`` -- present iff that object
        has ANY `<object id=...>` entry at that frame (appear/visible/
        occluded/disappear all count). This, not `"activity"`, is the
        CO-VISIBILITY basis (see module docstring).
      * ``"activity"``: ``dict[(person_id, t), str]`` -- the object's own
        `<movement>` label, restricted to `LLE_ACTIVITY_NAMES`, only where
        a `<hypothesislist>`/`<movement>` is actually present.
      * ``"appearance"``: ``dict[(person_id, t), str]`` -- the object's own
        `<appearance>` text.
      * ``"orientation"``: ``dict[(person_id, t), float]`` -- the object's
        own `<orientation>` degrees.
      * ``"coords"``: ``dict[(person_id, t), (int, int)]`` -- the object's
        own `<box xc, yc>`, taken directly as the box's center pixel.
      * ``"holds"``: ``dict[fluent_name, set[(p1, p2, t)]]`` for every
        fluent name in `HLE_MAPPING`'s values -- ordered pairs, BOTH
        directions for every mapped situation except ``leavingObject``
        (one ordered pair only) -- see module docstring's PAIR GENERATION
        section. A fluent with no holding frame anywhere in this video is
        simply absent from this dict (never an empty-set placeholder).

    ``.xml_``-suffixed files (two of the 30 real ground-truth files carry
    this trailing-underscore extension) parse identically to ``.xml``
    files: the extension carries no meaning to the XML content itself.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    video_name = os.path.basename(path)

    timestamps: list[int] = []
    persons: set[str] = set()
    tracked: set[tuple[str, int]] = set()
    activity: dict[tuple[str, int], str] = {}
    appearance: dict[tuple[str, int], str] = {}
    orientation: dict[tuple[str, int], float] = {}
    coords: dict[tuple[str, int], tuple[int, int]] = {}
    holds: dict[str, set[tuple[str, str, int]]] = defaultdict(set)

    for frame_el in root.findall("frame"):
        fnum = int(frame_el.get("number"))
        t = fnum * frame_ms + time_offset
        timestamps.append(t)

        objectlist = frame_el.find("objectlist")
        if objectlist is not None:
            for obj_el in objectlist.findall("object"):
                objid = obj_el.get("id")
                if objid is None:
                    raise ValueError(
                        f"<object> entry with no 'id' attribute in {path!r} "
                        f"at frame {fnum}: an unidentifiable tracked object "
                        "cannot be assigned to a person, and silently "
                        "dropping it would understate co-visibility."
                    )
                # `.strip()` matches the group-<members> parsing below: a
                # whitespace-padded id attribute must still name the SAME
                # person a group's member list names, or one person silently
                # splits into two identities and every group annotation
                # naming them is orphaned.
                pid = f"id{objid.strip()}"
                persons.add(pid)
                tracked.add((pid, t))

                app_el = obj_el.find("appearance")
                if app_el is not None and app_el.text:
                    appearance[(pid, t)] = app_el.text.strip()

                ori_el = obj_el.find("orientation")
                if ori_el is not None and ori_el.text:
                    try:
                        orientation[(pid, t)] = float(ori_el.text.strip())
                    except ValueError:
                        # A non-numeric orientation text drops THIS reading
                        # only (no entry for this (person, frame)) rather than
                        # inventing a value; orientation is auxiliary here --
                        # no relation or target is derived from it, so a
                        # missing entry cannot silently skew the corpus.
                        pass

                box_el = obj_el.find("box")
                if box_el is not None and "xc" in box_el.attrib and "yc" in box_el.attrib:
                    coords[(pid, t)] = (
                        int(float(box_el.attrib["xc"])),
                        int(float(box_el.attrib["yc"])),
                    )

                hyp_list = obj_el.find("hypothesislist")
                hyp = hyp_list.find("hypothesis") if hyp_list is not None else None
                if hyp is not None:
                    mv_el = hyp.find("movement")
                    if mv_el is not None and mv_el.text:
                        mv = mv_el.text.strip()
                        if mv in LLE_ACTIVITY_NAMES:
                            activity[(pid, t)] = mv

        grouplist = frame_el.find("grouplist")
        if grouplist is not None:
            for grp_el in grouplist.findall("group"):
                members_el = grp_el.find("members")
                if members_el is None or not members_el.text:
                    continue
                members = [f"id{m.strip()}" for m in members_el.text.split(",") if m.strip() != ""]
                if len(members) < 2:
                    continue

                hyp_list = grp_el.find("hypothesislist")
                hyp = hyp_list.find("hypothesis") if hyp_list is not None else None
                if hyp is None:
                    continue
                sit_el = hyp.find("situation")
                if sit_el is None or not sit_el.text:
                    continue
                situation = sit_el.text.strip()

                pred = HLE_MAPPING.get(situation)
                if pred is None:
                    # joining / split up / leaving victim: no entry in
                    # OLED's own hleMapping -- silently dropped, matching
                    # ParseCAVIAR.scala's own parser combinator behavior.
                    continue

                if situation in _SINGLE_PAIR_SITUATIONS:
                    holds[pred].add((members[0], members[1], t))
                else:
                    n = len(members)
                    for i in range(n):
                        for j in range(n):
                            if i != j:
                                holds[pred].add((members[i], members[j], t))

    return {
        "video": video_name,
        "frame_ms": frame_ms,
        "time_offset": time_offset,
        "timestamps": sorted(set(timestamps)),
        "persons": sorted(persons, key=_person_num),
        "tracked": tracked,
        "activity": activity,
        "appearance": appearance,
        "orientation": orientation,
        "coords": coords,
        "holds": dict(holds),
    }


def load_xml_corpus(xml_dir: str, *, frame_ms: int = FRAME_MS_DEFAULT) -> list[dict]:
    """Parse every CAVIAR ground-truth file (``*.xml`` AND ``*.xml_``) in
    ``xml_dir`` into a list of per-video dicts (see `parse_video`'s own
    docstring), sorted by file name, each with ``time_offset=0`` -- one
    file IS one independent video here, so there is no external clock to
    align onto and no splicing of any kind between files.

    Raises ``ValueError`` if no matching file is found (an empty corpus is
    never silently accepted -- almost always a wrong path)."""
    paths = sorted(
        set(glob.glob(os.path.join(xml_dir, "*.xml"))) | set(glob.glob(os.path.join(xml_dir, "*.xml_")))
    )
    if not paths:
        raise ValueError(
            f"no CAVIAR ground-truth files (*.xml or *.xml_) found under {xml_dir!r}."
        )
    return [parse_video(p, frame_ms=frame_ms, time_offset=0) for p in paths]


def _iter_pair_rows(videos: list[dict]):
    """Yield ``(video_index, t, p1, p2)`` for every unordered pair of
    persons CO-VISIBLE (both have a tracked entry -- see module docstring)
    at a timestep, one row per pair per timestep -- video index ascending,
    then timestep ascending within the video, then pair ascending
    (``video["persons"]`` is already sorted by `_person_num`), the same
    deterministic row order `caviar_continuous._iter_pair_rows` uses."""
    for v_idx, video in enumerate(videos):
        for t in video["timestamps"]:
            present = [pid for pid in video["persons"] if (pid, t) in video["tracked"]]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    yield v_idx, t, present[i], present[j]


def _pair_transition_counts(co_visible_ts: list[int], holds_ts: set) -> tuple[int, int, int, int]:
    """``(n_intervals, n_init, n_term, n_pairframes)`` for one unordered
    pair, walked over its own ``co_visible_ts`` (sorted ascending) within
    ONE video. A held frame with no earlier co-visible frame to compare
    against counts toward ``n_intervals`` but never ``n_init`` (there is no
    earlier frame to show a transition happening) -- the same convention
    `caviar_continuous.derive_ec_targets_continuous` uses for a pair's
    first observed co-visible pair-time."""
    n_intervals = n_init = n_term = n_pairframes = 0
    prev: bool | None = None
    for t in co_visible_ts:
        pos = t in holds_ts
        if pos:
            n_pairframes += 1
        if prev is None:
            if pos:
                n_intervals += 1
        else:
            if pos and not prev:
                n_init += 1
                n_intervals += 1
            elif (not pos) and prev:
                n_term += 1
        prev = pos
    return n_intervals, n_init, n_term, n_pairframes


def count_fluent_transitions(videos: list[dict], fluent: str) -> dict:
    """Corpus-level ``{"n_intervals", "n_init", "n_term", "n_pairframes"}``
    for ``fluent`` (any of `HLE_MAPPING`'s values: ``meeting``, ``moving``,
    ``fighting``, ``leavingObject``), summed over every video and every
    unordered pair of persons ever co-observed within that SAME video
    (never across videos). Mirrors `_pair_transition_counts`'s own
    first-observed convention for what counts as an initiation.

    Raises ``ValueError`` for an unmapped ``fluent`` name."""
    if fluent not in HLE_MAPPING.values():
        raise ValueError(
            f"fluent must be one of {sorted(set(HLE_MAPPING.values()))} (got {fluent!r})."
        )
    totals = {"n_intervals": 0, "n_init": 0, "n_term": 0, "n_pairframes": 0}
    for video in videos:
        tracked_by_pid: dict[str, list[int]] = defaultdict(list)
        for pid, t in video["tracked"]:
            tracked_by_pid[pid].append(t)
        persons = sorted(tracked_by_pid, key=_person_num)

        holds_by_pair: dict[tuple[str, str], set[int]] = defaultdict(set)
        for p1, p2, t in video["holds"].get(fluent, ()):
            holds_by_pair[_canonical_pair(p1, p2)].add(t)

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                co_visible = sorted(set(tracked_by_pid[a]) & set(tracked_by_pid[b]))
                if not co_visible:
                    continue
                holds_ts = holds_by_pair.get((a, b), set())
                n_int, n_init, n_term, n_pf = _pair_transition_counts(co_visible, holds_ts)
                totals["n_intervals"] += n_int
                totals["n_init"] += n_init
                totals["n_term"] += n_term
                totals["n_pairframes"] += n_pf
    return totals


def convert_xml_corpus(videos: list[dict], fluent: str = "meeting", close_threshold: float = 25.0) -> dict:
    """Flatten a list of `load_xml_corpus` per-video dicts into the SAME
    pair-time relation-space contract `caviar_continuous.convert_continuous`
    builds over the OLED dump's own continuous timeline -- same keys
    (``"facts"``, ``"is_positive"``, ``"relations"``, ``"features"``,
    ``"num_pt"``, ``"n_coords_missing"``, ``"segment_of_pt"``,
    ``"transition_relations"``), same relation/transition-relation
    semantics (see that function's own docstring), computed identically
    here except for two differences this module's own data demands:

      * ``fluent`` (``"meeting"`` or ``"moving"``) selects which of a
        video's own `"holds"` sets becomes ``is_positive``, instead of that
        function's fixed ``"meeting"`` target.
      * ``"segment_of_pt"`` here indexes ``videos`` directly -- ALWAYS a
        REAL, un-spliced video, never a dump segment that might silently
        splice more than one source video together.

    Raises ``ValueError`` for a ``fluent`` outside ``{"meeting",
    "moving"}`` (the two targets this project benchmarks; `fluent` values
    covering ``fighting``/``leavingObject`` corpus statistics are available
    via `count_fluent_transitions` without needing this pair-time
    flattening at all)."""
    if fluent not in ("meeting", "moving"):
        raise ValueError(f"fluent must be 'meeting' or 'moving' (got {fluent!r}).")

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
    transition_relations: dict[str, list[tuple[int, int]]] = {
        name: [] for name in TRANSITION_RELATION_NAMES
    }
    # (video_index, p1, p2) -> that pair's most recently OBSERVED
    # co-visible (a1, a2) activity reading -- see
    # `caviar_continuous.convert_continuous`'s identical bookkeeping for
    # why a single forward pass over `_iter_pair_rows`'s own row order is
    # enough (no second pass needed).
    prev_activity: dict[tuple[int, str, str], tuple[str | None, str | None]] = {}
    prev_dist: dict[tuple[int, str, str], tuple[float, bool] | None] = {}
    feature_rows: list[tuple[float, float]] = []
    segment_of_pt: list[int] = []
    n_coords_missing = 0

    for pt, (v_idx, t, p1, p2) in enumerate(_iter_pair_rows(videos)):
        video = videos[v_idx]
        facts.append((pt, 1))
        segment_of_pt.append(v_idx)
        holds = video["holds"].get(fluent, ())
        is_positive.append((p1, p2, t) in holds)

        a1 = video["activity"].get((p1, t))
        a2 = video["activity"].get((p2, t))
        if a1 == "active" and a2 == "active":
            relations["both_active"].append((pt, 1))
        if a1 == "walking" and a2 == "walking":
            relations["both_walking"].append((pt, 1))
        if a1 == "inactive" and a2 == "inactive":
            relations["both_inactive"].append((pt, 1))
        if (a1 == "active" and a2 == "walking") or (a1 == "walking" and a2 == "active"):
            relations["mixed_active_walking"].append((pt, 1))

        prev = prev_activity.get((v_idx, p1, p2))
        if prev is not None:
            pa1, pa2 = prev
            if (a1 == "active" and pa1 != "active") or (a2 == "active" and pa2 != "active"):
                transition_relations["any_became_active"].append((pt, 1))
            if (a1 == "inactive" and pa1 != "inactive") or (a2 == "inactive" and pa2 != "inactive"):
                transition_relations["any_became_inactive"].append((pt, 1))
            if (a1 == "walking" and pa1 != "walking") or (a2 == "walking" and pa2 != "walking"):
                transition_relations["any_became_walking"].append((pt, 1))
            if (pa1 == "walking" and a1 != "walking") or (pa2 == "walking" and a2 != "walking"):
                transition_relations["any_stopped_walking"].append((pt, 1))
        prev_activity[(v_idx, p1, p2)] = (a1, a2)

        pair_seg_key = (v_idx, p1, p2)
        c1 = video["coords"].get((p1, t))
        c2 = video["coords"].get((p2, t))
        if c1 is None or c2 is None:
            n_coords_missing += 1
            relations["coords_missing"].append((pt, 1))
            feature_rows.append((0.0, 0.0))
            prev_dist[pair_seg_key] = None
        else:
            dx = c1[0] - c2[0]
            dy = c1[1] - c2[1]
            feature_rows.append((dx / FEATURE_SCALE, dy / FEATURE_SCALE))
            dist = (dx * dx + dy * dy) ** 0.5
            is_close = dist <= close_threshold
            if is_close:
                relations["close"].append((pt, 1))
            else:
                relations["far"].append((pt, 1))

            if pair_seg_key in prev_dist and prev_dist[pair_seg_key] is not None:
                prev_dist_val, prev_was_close = prev_dist[pair_seg_key]
                if prev_was_close and not is_close:
                    transition_relations["became_far"].append((pt, 1))
                if dist > prev_dist_val:
                    transition_relations["distance_increasing"].append((pt, 1))
            prev_dist[pair_seg_key] = (dist, is_close)

    num_pt = len(facts)
    # Deferred import, matching `run_caviar_cv.py`'s own CPU-only,
    # torch-free-at-import discipline: every function above this point
    # (parsing, corpus-level transition counts) never needs torch at all;
    # only the returned feature TENSOR does, and only here.
    import torch

    features = torch.tensor(feature_rows, dtype=torch.float32).reshape(num_pt, 2)
    return {
        "facts": facts,
        "is_positive": is_positive,
        "relations": relations,
        "features": features,
        "num_pt": num_pt,
        "n_coords_missing": n_coords_missing,
        "segment_of_pt": segment_of_pt,
        "transition_relations": transition_relations,
    }
