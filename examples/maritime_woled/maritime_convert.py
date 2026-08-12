"""Brest AIS -> pair-time relational format converter for the WOLED
rendezvous probe.

DATA SOURCES. Archive A (the HLE/composite-event export, a `.tar.gz` whose
member of interest is `Maritime Composite Events/CEs/recognised_CEs.csv`) is
a census of fluent intervals: each line is
`fluent|entity1|entity2_or_value|value|st|et` (pipe-separated, half-open
`[st, et)` in unix seconds). `fluent` is one of the pair fluents in
`HLE_PAIR_FLUENTS` (`entity1`/`entity2` are the two vessels; canonicalized to
a sorted `(mmsi, mmsi)` tuple), `withinArea` (`entity1` is a vessel, the
third column is an area-type tag, e.g. "nearPorts"/"nearCoast" -- the entity
key is `(mmsi, area_type)`, never folded into the fluent name), or a plain
single-vessel fluent (`lowSpeed`, `stopped`, ...; entity key `(mmsi,)`). The
fluent key itself follows the census convention: the bare fluent name when
the value column reads "true", else `f"{fluent}={value}"` (e.g.
`"stopped=farFromPorts"`) -- `withinArea` rows always carry `value == "true"`
so their fluent key is always the bare `"withinArea"`, with the area
distinction living entirely in the entity key.

Archive B (the LLE/critical-events export, a `.zip` holding
`brest_critical.csv`) is a much larger stream of raw low-level detections;
this converter only needs its `proximity` rows,
`proximity|<recognition_time>|st|et|true|mmsi1|mmsi2` (the second column is a
detector bookkeeping timestamp, not part of the interval, and is ignored
here). Every other `<kind>|...` row (e.g. `coord|...`) is skipped by a cheap
`str.startswith("proximity|")` prefix check *before* the line is split --
archive B is a ~1GB csv in the real dataset, so paying for a full split on
every one of its (mostly non-proximity) rows is not acceptable.

Both parsers stream their member line-by-line (`tarfile.extractfile(...)`
iterated directly, `zipfile.open(...)` wrapped in `io.TextIOWrapper`) --
never `.read()` the whole member -- for the same reason.

PAIR SCOPE (`select_pairs`). This is a pre-registered, deterministic
scope, not a random sample: `positive_pairs` are every canonical pair with
at least one `rendezVous` interval; `negative_pairs` are a uniform-stride
subsample (no RNG at all) of the `negative_pool` (pairs seen in proximity
but never in rendezVous), capped at `N_NEG_PAIRS`.

PAIR-TIME TIMELINE (`build_pair_timeline`). For one pair, the timepoint
universe is the sorted union of: the pair's own proximity/rendezVous
interval boundaries (always kept), and the boundaries of both vessels'
`lowSpeed` / `stopped=farFromPorts` / `stopped=nearPorts` /
`withinArea=nearPorts` / `withinArea=nearCoast` intervals -- these are kept
only when they fall inside `[st - PAD_S, et + PAD_S)` of some proximity or
rendezVous interval of the pair, i.e. only near an actual encounter, so a
vessel's activity on the far side of the planet decades away from any
encounter never pollutes this pair's timeline. The resulting sorted
timepoints are then cut into episodes wherever a gap exceeds
`EPISODE_GAP_S` -- this bounds every "previous pt" relation (`became_far`,
`became_proximate`, `any_slow_ended`, and the EC targets below) to never
compare across a long silence.

RELATION VOCABULARY -- name -> EXACT HLE fluent set. The vocabulary is
pre-registered by NAME, and several names under-describe their semantics;
this table, not the names, is normative:

    proximity             archive B `proximity` covers t (merged intervals)
    far                   NOT proximity
    both_lowspeed         `lowSpeed` covers t for BOTH vessels
    both_stopped_far      `stopped=farFromPorts` covers t for BOTH vessels
    both_low_or_stopped   (`lowSpeed` UNION `stopped=farFromPorts`) covers t
                          for BOTH vessels
    either_low_or_stopped same union, for AT LEAST ONE vessel
    any_near_ports        (`withinArea`@nearPorts UNION `withinArea`@nearCoast)
                          covers t for at least one vessel -- NOTE: includes
                          nearCoast, NOT just ports, despite the name
    both_open_sea         NOT any_near_ports -- i.e. `withinArea` of ANY OTHER
                          area type (anchorages, fishing/protected areas, ...)
                          still counts as OPEN SEA here; only
                          nearPorts/nearCoast negate it
    became_far            far at t AND proximity at the previous pt (same episode)
    became_proximate      proximity at t AND NOT proximity at the previous pt
    any_slow_ended        some (`lowSpeed` UNION `stopped=farFromPorts`)
                          interval of either vessel ends in (prev_t, t]

INVERTED INTERVALS (et <= st) are skipped WITH a count in both parsers
(`bad_lines["inverted_interval"]` for archive A,
`inverted_proximity_lines` for archive B; both surface in `convert`'s
"counts" and hence in the verifier report): an empty half-open [st, et)
can never cover a timepoint, but its boundaries would otherwise leak into
pair timelines.

EC TARGET SEMANTICS -- MIRRORS `examples/caviar_woled/caviar_continuous.py`
EXACTLY, not the windowed `caviar_convert.derive_ec_targets`. That file's
`derive_ec_targets_continuous(treat_first_observed_as_init=True)` plus
`derive_ec_masks_continuous(treat_first_observed_as_init=True)` is the
semantics this module's `ec` targets copy, one state machine per episode
(the maritime analogue of one state machine per (video segment, pair) run):
`init_labels[i]` is True where `holds[i] and not holds[prev]`, or where
`holds[i]` at the first pt of an episode (an already-holding first frame is
treated as an observed initiation, since a window/episode boundary is not a
real absence of the fluent before it -- the same "holds-from-start" reading
CAVIAR's own `treat_first_observed_as_init=True` uses); don't-care
(`init_dontcare`) where `holds[i] and holds[prev]` (re-initiation is a
harmless no-op under inertia). `term_labels[i]` is True where
`not holds[i] and holds[prev]`; False at the first pt of an episode and
everywhere else `init_labels`/`init_dontcare` do not claim. `term_dontcare`
is True where `not holds[i] and not holds[prev]` (terminating a fluent that
was never holding is also a harmless no-op under inertia) -- including at
the first pt of an episode when it does not hold. Both of these
never-was-holding cells (mid-episode and first-pt) are don't-care, not plain
False: a `terminatedAt` firing there changes nothing an inertia-based
reconstruction would show differently, so scoring it as an ordinary error
would be dishonest, exactly as `derive_ec_masks_continuous` documents for
its own `term_dontcare`. Note this makes `term_dontcare` markedly less
symmetric with `init_dontcare` than a first guess at the vocabulary might
suggest -- `derive_ec_masks_continuous` is the reference for the exact
per-cell truth table this module reproduces.
"""

from __future__ import annotations

import io
import math
import tarfile
import zipfile
from bisect import bisect_right
from collections import Counter

HLE_PAIR_FLUENTS = {"rendezVous", "tugging", "proximity", "pilotBoarding"}
EPISODE_GAP_S = 3600
PAD_S = 1800
N_NEG_PAIRS = 604

# The gold generator's rendezvousTime constant (docs/experiments/maritime/
# PREREG_SOFT.md section (c)): LANGUAGE parity with RTEC, not a data peek.
SUSTAINED_MIN_S = 240

# Opt-in relations `convert` can add beyond the pre-registered baseline
# vocabulary. Anything else in `extra_relations` is refused with a typed
# error, never silently ignored.
_KNOWN_EXTRA_RELATIONS = ("sustained_240",)

_HLE_MEMBER_SUFFIX = "recognised_CEs.csv"
_LLE_MEMBER_SUFFIX = "brest_critical.csv"

# The six single-vessel fluent keys `build_pair_timeline` reads for each
# vessel of a pair, in the (fluent_key, entity_extra) shape needed to look
# them up in `hle["intervals"]`: plain fluents key by `(mmsi,)`; the two
# `withinArea` variants key by `(mmsi, area_type)`.
_VESSEL_FLUENT_LOOKUPS = (
    ("lowSpeed", None),
    ("stopped=farFromPorts", None),
    ("stopped=nearPorts", None),
    ("withinArea", "nearPorts"),
    ("withinArea", "nearCoast"),
)


def _canonical_pair(a: str, b: str) -> tuple[str, str]:
    """The pair's fixed, orderless identity: lexicographically sorted, so
    `("B", "A")` and `("A", "B")` collapse onto the same key everywhere."""
    return tuple(sorted((a, b)))


def _find_member(names: list[str], suffix: str) -> str:
    for name in names:
        if name.endswith(suffix):
            return name
    raise ValueError(f"no archive member ending in {suffix!r} found among {names!r}")


def parse_hle_archive(tar_path: str) -> dict:
    """Stream archive A. Returns {"intervals": {fluent_key: {entity: [(st, et), ...]}},
    "bad_lines": Counter, "n_lines": int}. fluent_key follows the census convention:
    plain fluent name when Value is "true", else f"{fluent}={value}" (e.g.
    "stopped=farFromPorts"). entity is a canonical pair tuple for pair fluents,
    (mmsi, area_type) for withinArea, else (mmsi,). Interval lists sorted by st.
    Fail-closed: CRLF content and non-integer interval bounds RAISE (on the
    md5-pinned archives both mean corruption); only wrong-field-count rows are
    counted in bad_lines and skipped."""
    intervals: dict[str, dict[tuple, list[tuple[int, int]]]] = {}
    bad_lines: Counter = Counter()
    n_lines = 0

    with tarfile.open(tar_path, "r:*") as tf:
        member = _find_member(tf.getnames(), _HLE_MEMBER_SUFFIX)
        fh = tf.extractfile(member)
        if fh is None:
            raise ValueError(f"{tar_path}: member {member!r} is not a regular file")
        text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
        for raw_line in text:
            # Fail-closed EOL check: the pinned archives are LF-only, so a
            # CR anywhere means the bytes are not the pinned bytes.
            if "\r" in raw_line:
                raise ValueError(
                    f"{tar_path}: CRLF line ending in {member!r} near line "
                    f"{n_lines + 1}: the pinned archives are LF-only; refusing "
                    "to parse EOL-mangled content"
                )
            line = raw_line.rstrip("\n")
            if not line:
                continue
            n_lines += 1
            fields = line.split("|")
            if len(fields) != 6:
                bad_lines["field_count"] += 1
                continue
            fluent, arg1, arg2, value, st_s, et_s = fields
            try:
                st, et = int(st_s), int(et_s)
            except ValueError:
                # Fail closed: a full-width row whose interval bounds do not
                # parse is corruption, not archive cruft — skipping it would
                # silently convert a wrong corpus.
                raise ValueError(
                    f"{tar_path}: non-integer interval bounds "
                    f"{st_s!r}/{et_s!r} in {member!r} (line {n_lines}): "
                    "refusing to skip a corrupt row"
                ) from None

            if fluent == "withinArea":
                entity = (arg1, arg2)
            elif fluent in HLE_PAIR_FLUENTS:
                entity = _canonical_pair(arg1, arg2)
            else:
                entity = (arg1,)

            if et <= st:
                # Inverted/zero-length: half-open [st, et) is empty, so the
                # row can never cover a timepoint, but its boundaries would
                # still leak into pair timelines — skip WITH a count.
                bad_lines["inverted_interval"] += 1
                continue

            fluent_key = fluent if value == "true" else f"{fluent}={value}"
            intervals.setdefault(fluent_key, {}).setdefault(entity, []).append((st, et))

    for by_entity in intervals.values():
        for entity, ivs in by_entity.items():
            ivs.sort()

    return {"intervals": intervals, "bad_lines": bad_lines, "n_lines": n_lines}


def _merge_sorted_intervals(ivs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Coalesce overlapping/adjacent half-open `[st, et)` intervals (already
    sorted by `st`) into their minimal covering set."""
    merged: list[tuple[int, int]] = []
    for st, et in ivs:
        if merged and st <= merged[-1][1]:
            prev_st, prev_et = merged[-1]
            if et > prev_et:
                merged[-1] = (prev_st, et)
        else:
            merged.append((st, et))
    return merged


def parse_lle_proximity(zip_path: str) -> dict:
    """Stream archive B; keep ONLY proximity rows. Returns
    {"proximity": {pair: [(st, et), ...]}, "bad_proximity_lines": int,
    "n_lines": int, "n_proximity_rows": int}. Pairs canonical, lists sorted+merged
    (overlapping/adjacent [st,et) intervals coalesced). Fail-closed: CRLF
    content and non-integer bounds on a proximity row RAISE; only
    wrong-field-count proximity rows are counted and skipped."""
    raw: dict[tuple[str, str], list[tuple[int, int]]] = {}
    bad_proximity_lines = 0
    inverted_proximity_lines = 0
    n_lines = 0
    n_proximity_rows = 0

    with zipfile.ZipFile(zip_path) as zf:
        member = _find_member(zf.namelist(), _LLE_MEMBER_SUFFIX)
        with zf.open(member) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8", newline="")
            for raw_line in text:
                # Same fail-closed EOL rule as parse_hle_archive: the pinned
                # archive is LF-only, a CR means these are not the pinned bytes.
                if "\r" in raw_line:
                    raise ValueError(
                        f"{zip_path}: CRLF line ending in {member!r} near line "
                        f"{n_lines + 1}: the pinned archives are LF-only; "
                        "refusing to parse EOL-mangled content"
                    )
                line = raw_line.rstrip("\n")
                if not line:
                    continue
                n_lines += 1
                if not line.startswith("proximity|"):
                    continue
                n_proximity_rows += 1
                fields = line.split("|")
                if len(fields) != 7:
                    bad_proximity_lines += 1
                    continue
                _kind, _rec_t, st_s, et_s, _value, e1, e2 = fields
                try:
                    st, et = int(st_s), int(et_s)
                except ValueError:
                    # Fail closed, as in parse_hle_archive: unparseable bounds
                    # on a full-width proximity row are corruption.
                    raise ValueError(
                        f"{zip_path}: non-integer interval bounds "
                        f"{st_s!r}/{et_s!r} in {member!r} (line {n_lines}): "
                        "refusing to skip a corrupt row"
                    ) from None
                if et <= st:
                    # Same skip-with-count rule as parse_hle_archive's
                    # inverted_interval counter: an empty [st, et) covers
                    # nothing but would leak boundary timepoints.
                    inverted_proximity_lines += 1
                    continue
                pair = _canonical_pair(e1, e2)
                raw.setdefault(pair, []).append((st, et))

    proximity: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for pair, ivs in raw.items():
        proximity[pair] = _merge_sorted_intervals(sorted(ivs))

    return {
        "proximity": proximity,
        "bad_proximity_lines": bad_proximity_lines,
        "inverted_proximity_lines": inverted_proximity_lines,
        "n_lines": n_lines,
        "n_proximity_rows": n_proximity_rows,
    }


def select_pairs(hle: dict, prox: dict) -> dict:
    """Pre-registered pair scope. positive_pairs = pairs with >=1 rendezVous interval.
    negative_pool = pairs with >=1 proximity interval and 0 rendezVous intervals.
    negative_pairs = uniform stride subsample of the lexicographically sorted pool
    down to N_NEG_PAIRS (stride = ceil(len(pool)/N_NEG_PAIRS), take every stride-th
    starting at 0; if pool <= N_NEG_PAIRS take all). Returns {"positive_pairs": [...],
    "negative_pairs": [...], "n_negative_pool": int} (lists sorted)."""
    rendez = hle.get("intervals", {}).get("rendezVous", {})
    positive_pairs = sorted(pair for pair, ivs in rendez.items() if ivs)

    positive_set = set(positive_pairs)
    pool = sorted(
        pair for pair, ivs in prox.get("proximity", {}).items()
        if ivs and pair not in positive_set
    )

    n_pool = len(pool)
    if n_pool <= N_NEG_PAIRS:
        negative_pairs = list(pool)
    else:
        stride = math.ceil(n_pool / N_NEG_PAIRS)
        negative_pairs = pool[::stride]

    return {
        "positive_pairs": positive_pairs,
        "negative_pairs": sorted(negative_pairs),
        "n_negative_pool": n_pool,
    }


def _covers(sorted_ivs: list[tuple[int, int]], t: int) -> bool:
    """Does some half-open `[st, et)` interval in `sorted_ivs` (sorted by
    `st`, non-overlapping) cover `t`? `bisect_right` over the start times
    finds the rightmost interval whose `st <= t`; that is the only candidate
    that could cover `t` when intervals do not overlap."""
    if not sorted_ivs:
        return False
    starts = [iv[0] for iv in sorted_ivs]
    idx = bisect_right(starts, t) - 1
    if idx < 0:
        return False
    st, et = sorted_ivs[idx]
    return st <= t < et


def _covers_closed(sorted_ivs: list[tuple[int, int]], t: int) -> bool:
    """`_covers` with a CLOSED right end: does some `[st, et]` component in
    `sorted_ivs` (sorted, non-overlapping) contain `t`? Used only by
    `sustained_240`, whose pre-registered rule grants the relation to every
    pt inside `[st, et]` of a long-enough intersection component."""
    if not sorted_ivs:
        return False
    starts = [iv[0] for iv in sorted_ivs]
    idx = bisect_right(starts, t) - 1
    if idx < 0:
        return False
    st, et = sorted_ivs[idx]
    return st <= t <= et


def _intersect_sorted(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Intersection of two MERGED sorted half-open interval lists — the
    classic two-pointer sweep; output is again merged and sorted."""
    out: list[tuple[int, int]] = []
    i = j = 0
    while i < len(a) and j < len(b):
        st = max(a[i][0], b[j][0])
        et = min(a[i][1], b[j][1])
        if st < et:
            out.append((st, et))
        if a[i][1] <= b[j][1]:
            i += 1
        else:
            j += 1
    return out


def _subtract_sorted(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`a` minus `b` over MERGED sorted half-open interval lists: every part
    of an `a` interval not covered by some `b` interval survives."""
    out: list[tuple[int, int]] = []
    j = 0
    for st, et in a:
        cur = st
        while j < len(b) and b[j][1] <= cur:
            j += 1
        k = j
        while k < len(b) and b[k][0] < et:
            bst, bet = b[k]
            if bst > cur:
                out.append((cur, bst))
            cur = max(cur, bet)
            if bet >= et:
                break
            k += 1
        if cur < et:
            out.append((cur, et))
    return out


def _in_pad_window(t: int, anchor_ivs: list[tuple[int, int]]) -> bool:
    for st, et in anchor_ivs:
        if st - PAD_S <= t < et + PAD_S:
            return True
    return False


def build_pair_timeline(pair, hle, prox) -> dict:
    """Timepoint universe for one pair: the sorted union of all [st, et) boundary
    timestamps of: the pair's proximity intervals, the pair's rendezVous intervals,
    and both vessels' lowSpeed / stopped=farFromPorts / stopped=nearPorts /
    withinArea=nearPorts / withinArea=nearCoast intervals — KEPT only if the
    timestamp falls inside [st - PAD_S, et + PAD_S) of some proximity or rendezVous
    interval of the pair. Then split into episodes: a new episode starts when the
    gap to the previous kept timepoint exceeds EPISODE_GAP_S. Returns
    {"timepoints": [int, ...], "episodes": [(lo_idx, hi_idx), ...]} with hi
    exclusive, plus {"n_dropped_outside_pad": int}."""
    prox_ivs = prox.get("proximity", {}).get(pair, [])
    rendez_ivs = hle.get("intervals", {}).get("rendezVous", {}).get(pair, [])
    anchor_ivs = sorted(prox_ivs) + sorted(rendez_ivs)

    timepoints: set[int] = set()
    for st, et in anchor_ivs:
        timepoints.add(st)
        timepoints.add(et)

    n_dropped_outside_pad = 0
    v1, v2 = pair
    for vessel in (v1, v2):
        for fluent_key, area in _VESSEL_FLUENT_LOOKUPS:
            entity = (vessel, area) if area is not None else (vessel,)
            ivs = hle.get("intervals", {}).get(fluent_key, {}).get(entity, [])
            for st, et in ivs:
                for ts in (st, et):
                    if _in_pad_window(ts, anchor_ivs):
                        timepoints.add(ts)
                    else:
                        n_dropped_outside_pad += 1

    sorted_tp = sorted(timepoints)

    episodes: list[tuple[int, int]] = []
    if sorted_tp:
        lo = 0
        for i in range(1, len(sorted_tp)):
            if sorted_tp[i] - sorted_tp[i - 1] > EPISODE_GAP_S:
                episodes.append((lo, i))
                lo = i
        episodes.append((lo, len(sorted_tp)))

    return {
        "timepoints": sorted_tp,
        "episodes": episodes,
        "n_dropped_outside_pad": n_dropped_outside_pad,
    }


def _vessel_low_or_stopped_intervals(hle: dict, vessel: str) -> list[tuple[int, int]]:
    """Merged `lowSpeed` OR `stopped=farFromPorts` intervals for one vessel,
    used both by the `*_low_or_stopped` relations (cover check) and by
    `any_slow_ended` (endpoint check)."""
    low = hle.get("intervals", {}).get("lowSpeed", {}).get((vessel,), [])
    stopped_far = hle.get("intervals", {}).get("stopped=farFromPorts", {}).get((vessel,), [])
    combined = sorted(list(low) + list(stopped_far))
    return _merge_sorted_intervals(combined)


def _vessel_near_ports_intervals(hle: dict, vessel: str) -> list[tuple[int, int]]:
    near_ports = hle.get("intervals", {}).get("withinArea", {}).get((vessel, "nearPorts"), [])
    near_coast = hle.get("intervals", {}).get("withinArea", {}).get((vessel, "nearCoast"), [])
    combined = sorted(list(near_ports) + list(near_coast))
    return _merge_sorted_intervals(combined)


def convert(tar_path: str, zip_path: str, extra_relations: tuple[str, ...] = ()) -> dict:
    """Full conversion. `extra_relations` (default `()` — byte-identical to
    the pre-`extra_relations` behavior) opts additional relations into the
    output vocabulary; the only known name is `"sustained_240"`
    (PREREG_SOFT.md section (c)): per pair, the intersection of the
    CONTINUOUS merged interval lists `proximity ∩ both_low_or_stopped ∩
    both_open_sea` is computed by interval algebra (never from the sparse
    pt grid — a single-pt gold run inside a long intersection must still
    count), and every pt lying in `[st, et]` (closed; the `== 240` tie
    included) of a component with `et - st >= SUSTAINED_MIN_S` receives
    the relation. An unknown name raises `ValueError`.

    Returns dict with keys:
    - "pairs": [pair, ...] in fixed order (positives sorted, then negatives sorted)
    - "pt_pair_index": [int, ...]   # for each pt row, index into "pairs"
    - "pt_time": [int, ...]         # unix second of the pt row
    - "segments": [(lo, hi), ...]   # global pt index ranges, one per episode,
                                    #   never spanning two pairs
    - "relations": {name: sorted [pt_index, ...]}  # vocabulary of section below
    - "is_positive": [bool, ...]    # holdsAt(rendezVous) at that pt ([st,et) cover)
    - "ec": {"init_labels": [bool], "init_dontcare": [bool],
             "term_labels": [bool], "term_dontcare": [bool]}
    - "counts": every skip/drop/bad-line counter from the parsers + row totals
    """
    unknown = [n for n in extra_relations if n not in _KNOWN_EXTRA_RELATIONS]
    if unknown:
        raise ValueError(
            f"unknown extra_relations {unknown!r}: the only names this "
            f"converter can add are {_KNOWN_EXTRA_RELATIONS!r}."
        )
    if len(set(extra_relations)) != len(extra_relations):
        raise ValueError(f"extra_relations contains duplicates: {extra_relations!r}")
    with_sustained = "sustained_240" in extra_relations

    hle = parse_hle_archive(tar_path)
    prox = parse_lle_proximity(zip_path)
    sel = select_pairs(hle, prox)

    pairs = list(sel["positive_pairs"]) + list(sel["negative_pairs"])

    pt_pair_index: list[int] = []
    pt_time: list[int] = []
    segments: list[tuple[int, int]] = []
    is_positive: list[bool] = []

    relation_names = (
        "proximity", "far", "both_lowspeed", "both_stopped_far",
        "both_low_or_stopped", "either_low_or_stopped", "any_near_ports",
        "both_open_sea", "became_far", "became_proximate", "any_slow_ended",
    ) + tuple(extra_relations)
    relations: dict[str, list[int]] = {name: [] for name in relation_names}

    init_labels: list[bool] = []
    init_dontcare: list[bool] = []
    term_labels: list[bool] = []
    term_dontcare: list[bool] = []

    n_dropped_outside_pad = 0

    for pair_idx, pair in enumerate(pairs):
        v1, v2 = pair
        timeline = build_pair_timeline(pair, hle, prox)
        n_dropped_outside_pad += timeline["n_dropped_outside_pad"]

        # EVERY interval list handed to `_covers` must be merged, not just
        # sorted: `_covers` inspects only the rightmost interval with
        # st <= t, so a strictly NESTED same-key interval (e.g.
        # [(0, 100), (10, 20)] at t = 50) would otherwise falsify coverage
        # inside the covering interval. parse_lle_proximity pre-merges
        # proximity; the HLE lists are merged here.
        prox_ivs = sorted(prox.get("proximity", {}).get(pair, []))
        rendez_ivs = _merge_sorted_intervals(
            sorted(hle.get("intervals", {}).get("rendezVous", {}).get(pair, []))
        )

        v1_low_stop = _vessel_low_or_stopped_intervals(hle, v1)
        v2_low_stop = _vessel_low_or_stopped_intervals(hle, v2)
        v1_low = _merge_sorted_intervals(
            sorted(hle.get("intervals", {}).get("lowSpeed", {}).get((v1,), []))
        )
        v2_low = _merge_sorted_intervals(
            sorted(hle.get("intervals", {}).get("lowSpeed", {}).get((v2,), []))
        )
        v1_stop_far = _merge_sorted_intervals(
            sorted(hle.get("intervals", {}).get("stopped=farFromPorts", {}).get((v1,), []))
        )
        v2_stop_far = _merge_sorted_intervals(
            sorted(hle.get("intervals", {}).get("stopped=farFromPorts", {}).get((v2,), []))
        )
        v1_near_ports = _vessel_near_ports_intervals(hle, v1)
        v2_near_ports = _vessel_near_ports_intervals(hle, v2)

        if with_sustained:
            # The CONTINUOUS intersection proximity ∩ both_low_or_stopped ∩
            # both_open_sea over the pair's already-merged interval lists;
            # both_open_sea is the complement of either vessel's
            # nearPorts/nearCoast union, so it is a subtraction here.
            both_ls_ivs = _intersect_sorted(v1_low_stop, v2_low_stop)
            inter = _intersect_sorted(prox_ivs, both_ls_ivs)
            near_union = _merge_sorted_intervals(sorted(v1_near_ports + v2_near_ports))
            sustained_ivs = [
                (st, et) for st, et in _subtract_sorted(inter, near_union)
                if et - st >= SUSTAINED_MIN_S
            ]
        else:
            sustained_ivs = []

        local_times = timeline["timepoints"]
        base = len(pt_time)

        for lo, hi in timeline["episodes"]:
            global_lo = base + lo
            global_hi = base + hi
            segments.append((global_lo, global_hi))

        for lo, hi in timeline["episodes"]:
            prev_t: int | None = None
            for local_i in range(lo, hi):
                t = local_times[local_i]
                pt_pair_index.append(pair_idx)
                pt_time.append(t)

                proximate = _covers(prox_ivs, t)
                far = not proximate
                pos = _covers(rendez_ivs, t)
                is_positive.append(pos)

                both_lowspeed = _covers(v1_low, t) and _covers(v2_low, t)
                both_stopped_far = _covers(v1_stop_far, t) and _covers(v2_stop_far, t)
                v1_ls = _covers(v1_low_stop, t)
                v2_ls = _covers(v2_low_stop, t)
                both_low_or_stopped = v1_ls and v2_ls
                either_low_or_stopped = v1_ls or v2_ls
                any_near_ports = _covers(v1_near_ports, t) or _covers(v2_near_ports, t)
                both_open_sea = not any_near_ports

                pt_idx = len(pt_time) - 1
                if proximate:
                    relations["proximity"].append(pt_idx)
                if far:
                    relations["far"].append(pt_idx)
                if both_lowspeed:
                    relations["both_lowspeed"].append(pt_idx)
                if both_stopped_far:
                    relations["both_stopped_far"].append(pt_idx)
                if both_low_or_stopped:
                    relations["both_low_or_stopped"].append(pt_idx)
                if either_low_or_stopped:
                    relations["either_low_or_stopped"].append(pt_idx)
                if any_near_ports:
                    relations["any_near_ports"].append(pt_idx)
                if both_open_sea:
                    relations["both_open_sea"].append(pt_idx)
                if with_sustained and _covers_closed(sustained_ivs, t):
                    relations["sustained_240"].append(pt_idx)

                if prev_t is None:
                    became_far = False
                    became_proximate = False
                    any_slow_ended = False
                else:
                    prev_proximate = _covers(prox_ivs, prev_t)
                    became_far = far and prev_proximate
                    became_proximate = proximate and not prev_proximate
                    any_slow_ended = False
                    for ivs in (v1_low_stop, v2_low_stop):
                        for _st, et in ivs:
                            if prev_t < et <= t:
                                any_slow_ended = True
                                break
                        if any_slow_ended:
                            break

                if became_far:
                    relations["became_far"].append(pt_idx)
                if became_proximate:
                    relations["became_proximate"].append(pt_idx)
                if any_slow_ended:
                    relations["any_slow_ended"].append(pt_idx)

                # EC targets: one state machine per episode, mirroring
                # `caviar_continuous.derive_ec_targets_continuous(
                # treat_first_observed_as_init=True)` plus
                # `derive_ec_masks_continuous(treat_first_observed_as_init=True)`
                # -- see the module docstring.
                if prev_t is None:
                    if pos:
                        init_labels.append(True)
                        init_dontcare.append(False)
                    else:
                        init_labels.append(False)
                        init_dontcare.append(False)
                    term_labels.append(False)
                    term_dontcare.append(not pos)
                else:
                    prev_pos = is_positive[pt_idx - 1]
                    if pos and not prev_pos:
                        init_labels.append(True)
                        init_dontcare.append(False)
                    elif pos and prev_pos:
                        init_labels.append(False)
                        init_dontcare.append(True)
                    else:
                        init_labels.append(False)
                        init_dontcare.append(False)

                    if (not pos) and prev_pos:
                        term_labels.append(True)
                        term_dontcare.append(False)
                    elif (not pos) and (not prev_pos):
                        term_labels.append(False)
                        term_dontcare.append(True)
                    else:
                        term_labels.append(False)
                        term_dontcare.append(False)

                prev_t = t

    counts = {
        "n_lines_hle": hle["n_lines"],
        "bad_lines_hle": dict(hle["bad_lines"]),
        "n_lines_lle": prox["n_lines"],
        "n_proximity_rows": prox["n_proximity_rows"],
        "bad_proximity_lines": prox["bad_proximity_lines"],
        "inverted_proximity_lines": prox["inverted_proximity_lines"],
        "n_negative_pool": sel["n_negative_pool"],
        "n_positive_pairs": len(sel["positive_pairs"]),
        "n_negative_pairs": len(sel["negative_pairs"]),
        "n_pairs": len(pairs),
        "n_pt": len(pt_time),
        "n_segments": len(segments),
        "n_dropped_outside_pad": n_dropped_outside_pad,
    }

    return {
        "pairs": pairs,
        "pt_pair_index": pt_pair_index,
        "pt_time": pt_time,
        "segments": segments,
        "relations": relations,
        "is_positive": is_positive,
        "ec": {
            "init_labels": init_labels,
            "init_dontcare": init_dontcare,
            "term_labels": term_labels,
            "term_dontcare": term_dontcare,
        },
        "counts": counts,
    }
