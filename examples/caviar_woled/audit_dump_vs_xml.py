"""Data-integrity audit of the OLED dump's meeting transition events
against the 30 CAVIAR ground-truth XML files -- the reproducible form of
README section F's classification (21 real / 3 duplicates / 1 splice
phantom), which previously existed only as a one-off analysis with no
shipped tool or artifact.

WHAT IT ESTABLISHES. The OLED dump (`caviar-train.json`/`caviar-test.json`)
is DERIVED from the 30 ground-truth XML videos by OLED's own
`ParseCAVIAR.scala`, which (a) concatenates consecutive videos onto one
timeline with an exactly-40ms glue step (a real video boundary becomes
numerically indistinguishable from an ordinary frame), and (b) ships two
videos in BOTH its train and test files. This script cross-matches every
meeting transition event the dump's own reading produces against the XML
ground truth, classifying each as:

  * ``real`` -- maps 1:1 onto an annotation change in the matched source
    video's own frame sequence (matched by exact local frame number);
  * ``duplicate`` -- a second dump event mapping onto an XML event already
    claimed by an earlier dump event (the same real event counted twice
    because its source video appears in more than one dump segment --
    the train/test duplication);
  * ``splice_artifact`` -- no such event exists in ANY single source
    video's own ground truth: the transition is created by the dump's
    segment-joining rule bridging two different videos as if they were
    one contiguous recording.

METHOD (independent of any assumed mapping). Dump segments are matched to
XML videos by exact pixel-coordinate trajectory agreement, per person,
with iterative peeling (`_multi_match_person`): a vote is cast for a
``(video, xml_person, offset)`` hypothesis at every dump timestamp whose
``(x, y)`` equals that XML person's box center at ``(t - offset) / 40``;
a hypothesis is accepted only on >= ``min_points`` exact (t, x, y)
agreements at ONE constant offset, its explained points are removed, and
the search repeats -- so one dump person id straddling a genuine splice
resolves into MULTIPLE ``(video, offset)`` clusters, one per source
video, which is exactly what exposes the splice phantom. Transition
events on both sides are derived by the SAME walk
`caviar_continuous.derive_ec_targets_continuous` uses (first observed
co-visible frame is never an initiation; a positive->negative flip
between consecutive co-visible frames is a termination).

CPU-only, deterministic, no CUDA and no induction machinery -- this is a
data audit, not a learning run.

Usage:
    python examples/caviar_woled/audit_dump_vs_xml.py \
      --train-json caviar-train.json --test-json caviar-test.json \
      --xml-dir <dir with the 30 CAVIAR ground-truth XML files> \
      --out AUDIT.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

FRAME_MS = 40

# Minimum exact (t, x, y) agreements, at one constant offset, for a
# (video, xml_person, offset) hypothesis to be accepted. Exact pixel
# agreement across 10+ distinct timestamps at a single offset is not a
# coincidence; fractional score thresholds are deliberately NOT used,
# because a segment that splices two videos makes every per-video
# fraction low while the absolute evidence stays overwhelming.
MATCH_MIN_POINTS = 10


def pair_transition_events(co_visible_ts: list[int], holds_ts: set) -> list[tuple[int, str]]:
    """``(t, "init"|"term")`` events for ONE unordered pair walked over its
    own sorted co-visible timestamps -- the same convention
    `caviar_continuous.derive_ec_targets_continuous` and
    `caviar_xml_corpus._pair_transition_counts` use: the first observed
    co-visible frame is never an event (no earlier frame exists to show a
    transition happening), an off->on flip is an initiation at the ON
    frame, an on->off flip is a termination at the OFF frame."""
    events = []
    prev: bool | None = None
    for t in co_visible_ts:
        pos = t in holds_ts
        if prev is not None:
            if pos and not prev:
                events.append((t, "init"))
            elif (not pos) and prev:
                events.append((t, "term"))
        prev = pos
    return events


def dump_meeting_events(segments: list[dict], split_name: str) -> list[dict]:
    """Every meeting transition event the DUMP's own reading produces, per
    `caviar_continuous.load_continuous` segment: co-visibility is "both
    persons have an activity entry at t" (`_iter_pair_rows`'s own test),
    holds come from the segment's canonical-paired ``"meeting"`` set."""
    from caviar_continuous import _person_num

    events = []
    for seg_idx, seg in enumerate(segments):
        pair_ts: dict[tuple[str, str], list[int]] = defaultdict(list)
        for t in seg["timestamps"]:
            present = [p for p in seg["persons"] if (p, t) in seg["activity"]]
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    a, b = sorted((present[i], present[j]), key=_person_num)
                    pair_ts[(a, b)].append(t)
        for (p1, p2), ts_list in sorted(pair_ts.items()):
            holds = {t for t in ts_list if (p1, p2, t) in seg["meeting"]}
            for t, kind in pair_transition_events(sorted(ts_list), holds):
                events.append({
                    "seg_label": f"{split_name}#{seg_idx}",
                    "split": split_name, "seg_idx": seg_idx,
                    "pair": [p1, p2], "t": t, "kind": kind,
                })
    return events


def xml_meeting_events(videos: list[dict]) -> list[dict]:
    """Every meeting transition event in the XML ground truth's OWN frame
    sequences -- one segment per real video by construction, so no splice
    is possible on this side. Pairs are canonicalized; co-visibility is
    `caviar_xml_corpus`'s own "both tracked at the frame" test."""
    from caviar_xml_corpus import _canonical_pair, _person_num

    events = []
    for v_idx, video in enumerate(videos):
        frame_ms = video["frame_ms"]
        offset = video["time_offset"]
        tracked_by_pid: dict[str, list[int]] = defaultdict(list)
        for pid, t in video["tracked"]:
            tracked_by_pid[pid].append(t)
        persons = sorted(tracked_by_pid, key=_person_num)
        holds_by_pair: dict[tuple[str, str], set] = defaultdict(set)
        for p1, p2, t in video["holds"].get("meeting", ()):
            holds_by_pair[_canonical_pair(p1, p2)].add(t)
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                a, b = persons[i], persons[j]
                co_visible = sorted(set(tracked_by_pid[a]) & set(tracked_by_pid[b]))
                if not co_visible:
                    continue
                cp = _canonical_pair(a, b)
                for t, kind in pair_transition_events(co_visible, holds_by_pair.get(cp, set())):
                    events.append({
                        "video": video["video"], "video_idx": v_idx,
                        "pair": list(cp), "frame": (t - offset) // frame_ms,
                        "kind": kind,
                    })
    return events


def build_xml_xy_index(videos: list[dict]):
    """``traj[(v_idx, pid)] = {frame: (x, y)}`` plus the inverted
    ``index[(x, y)] -> [(v_idx, pid, frame), ...]`` the trajectory voting
    walks -- built once over the whole XML corpus."""
    traj: dict[tuple[int, str], dict[int, tuple[int, int]]] = defaultdict(dict)
    index: dict[tuple[int, int], list[tuple[int, str, int]]] = defaultdict(list)
    for v_idx, video in enumerate(videos):
        frame_ms = video["frame_ms"]
        offset = video["time_offset"]
        for (pid, t), xy in video["coords"].items():
            frame = (t - offset) // frame_ms
            traj[(v_idx, pid)][frame] = xy
            index[xy].append((v_idx, pid, frame))
    return dict(traj), dict(index)


def _multi_match_person(
    coords: dict, traj: dict, index: dict, min_points: int = MATCH_MIN_POINTS,
    max_rounds: int = 5,
):
    """Iterative-peeling trajectory match for ONE dump person's own
    ``{t: (x, y)}`` within one segment (see the module docstring's METHOD).
    Returns a list of accepted hypotheses, each
    ``{"video_idx", "xml_pid", "offset", "ts": set[int]}`` -- ``ts`` being
    exactly the dump timestamps that hypothesis explains by per-point
    re-check against the XML trajectory (never just "was the argmax")."""
    remaining = dict(coords)
    results = []
    for _ in range(max_rounds):
        if not remaining:
            break
        votes: Counter = Counter()
        for t, xy in remaining.items():
            for (v_idx, pid, frame) in index.get(xy, ()):
                votes[(v_idx, pid, t - frame * FRAME_MS)] += 1
        if not votes:
            break
        (v_idx, xml_pid, offset), n = votes.most_common(1)[0]
        if n < min_points:
            break
        obj_traj = traj[(v_idx, xml_pid)]
        matched_ts = set()
        for t, xy in remaining.items():
            rem = t - offset
            if rem % FRAME_MS != 0:
                continue
            if obj_traj.get(rem // FRAME_MS) == xy:
                matched_ts.add(t)
        if len(matched_ts) < min_points:
            break
        results.append({"video_idx": v_idx, "xml_pid": xml_pid, "offset": offset, "ts": matched_ts})
        for t in matched_ts:
            del remaining[t]
    return results


def match_segments(
    segments: list[dict], videos: list[dict], traj: dict, index: dict,
    min_points: int = MATCH_MIN_POINTS,
) -> list[dict]:
    """Per dump segment: every person's accepted trajectory hypotheses,
    grouped into ``(video_idx, offset)`` CLUSTERS -- ``clusters[(v, o)] =
    {dump_pid: {"xml_pid": ..., "ts": set}}``. A segment that splices two
    source videos yields (at least) two clusters."""
    out = []
    for seg in segments:
        coords_by_pid: dict[str, dict[int, tuple[int, int]]] = defaultdict(dict)
        for (pid, t), xy in seg["coords"].items():
            coords_by_pid[pid][t] = xy
        clusters: dict[tuple[int, int], dict] = defaultdict(dict)
        for pid, coords in coords_by_pid.items():
            for r in _multi_match_person(coords, traj, index, min_points=min_points):
                clusters[(r["video_idx"], r["offset"])][pid] = {
                    "xml_pid": r["xml_pid"], "ts": r["ts"],
                }
        out.append({"clusters": dict(clusters)})
    return out


def _map_event_to_xml(event: dict, seg_match: dict, videos: list[dict]):
    """Map one dump event onto ``(video_name, canonical xml pair, local
    frame, kind)`` via the segment's clusters: prefer the cluster whose
    accepted point sets contain the event's own timestamp for BOTH
    persons; fall back to the cluster explaining the most of the pair's
    points. Returns ``None`` if no cluster matched both persons."""
    from caviar_xml_corpus import _canonical_pair

    p1, p2 = event["pair"]
    best = None
    for (v_idx, offset), members in seg_match["clusters"].items():
        if p1 not in members or p2 not in members:
            continue
        covers_t = event["t"] in members[p1]["ts"] and event["t"] in members[p2]["ts"]
        weight = len(members[p1]["ts"]) + len(members[p2]["ts"])
        key = (1 if covers_t else 0, weight)
        if best is None or key > best[0]:
            best = (key, v_idx, offset, members)
    if best is None:
        return None
    _, v_idx, offset, members = best
    video = videos[v_idx]
    frame = (event["t"] - offset - video["time_offset"]) // video["frame_ms"]
    xml_pair = _canonical_pair(members[p1]["xml_pid"], members[p2]["xml_pid"])
    return {
        "video": video["video"], "video_idx": v_idx, "offset": offset,
        "xml_pair": list(xml_pair), "frame": frame,
        "t_pixel_matched": best[0][0] == 1,
    }


def classify_events(
    dump_events: list[dict], xml_events: list[dict], seg_matches_by_split: dict,
    videos: list[dict],
) -> dict:
    """The audit's core: classify every dump event as ``real`` /
    ``duplicate`` / ``splice_artifact`` (module docstring's definitions).
    Dump events are visited in deterministic order (train before test,
    then segment index, then timestamp), so WHICH copy of a
    double-counted event is labeled the duplicate is a fixed convention
    (the later one), not an accident of iteration order. Also reports
    every XML event with the dump events that claimed it -- an XML event
    claimed by zero dump events would mean the dump MISSED a real
    transition (none occur in the real corpus, but the audit checks
    rather than assumes)."""
    xml_key_set = {
        (e["video"], tuple(e["pair"]), e["frame"], e["kind"]) for e in xml_events
    }
    claimed: dict[tuple, list[dict]] = defaultdict(list)
    classified = []
    ordered = sorted(
        dump_events,
        key=lambda e: (0 if e["split"] == "train" else 1, e["seg_idx"], e["t"],
                       e["kind"], tuple(e["pair"])),
    )
    for event in ordered:
        seg_match = seg_matches_by_split[event["split"]][event["seg_idx"]]
        mapped = _map_event_to_xml(event, seg_match, videos)
        record = dict(event)
        record["mapped"] = mapped
        if mapped is None:
            record["classification"] = "unmatched"
        else:
            key = (mapped["video"], tuple(mapped["xml_pair"]), mapped["frame"], event["kind"])
            if key not in xml_key_set:
                record["classification"] = "splice_artifact"
            elif claimed[key]:
                record["classification"] = "duplicate"
            else:
                record["classification"] = "real"
            if key in xml_key_set:
                claimed[key].append(
                    {"seg_label": event["seg_label"], "t": event["t"]}
                )
        classified.append(record)

    xml_side = []
    for e in xml_events:
        key = (e["video"], tuple(e["pair"]), e["frame"], e["kind"])
        xml_side.append({**e, "claimed_by": claimed.get(key, [])})

    counts = Counter(r["classification"] for r in classified)
    return {
        "dump_events": classified,
        "xml_events": xml_side,
        "classification_counts": dict(counts),
        "n_dump_events": {
            "init": sum(1 for e in dump_events if e["kind"] == "init"),
            "term": sum(1 for e in dump_events if e["kind"] == "term"),
        },
        "n_xml_events": {
            "init": sum(1 for e in xml_events if e["kind"] == "init"),
            "term": sum(1 for e in xml_events if e["kind"] == "term"),
        },
        "xml_events_unclaimed": [
            {k: e[k] for k in ("video", "pair", "frame", "kind")}
            for e in xml_side if not e["claimed_by"]
        ],
    }


def duplicated_gold_pairframes(videos: list[dict], seg_matches_by_split: dict) -> dict:
    """For every XML video matched by dump segments in BOTH the train and
    the test file (the train/test duplication defect itself): that
    video's own gold meeting pair-frame count (unordered pairs, counted
    over the video's own co-visible frames -- the same accounting
    `caviar_xml_corpus.count_fluent_transitions` uses), i.e. how much
    positive mass the duplication double-counts."""
    from caviar_xml_corpus import _canonical_pair

    videos_by_split: dict[str, set[int]] = defaultdict(set)
    for split, matches in seg_matches_by_split.items():
        for m in matches:
            for (v_idx, _offset) in m["clusters"]:
                videos_by_split[split].add(v_idx)
    both = videos_by_split.get("train", set()) & videos_by_split.get("test", set())

    out = {}
    for v_idx in sorted(both):
        video = videos[v_idx]
        tracked = video["tracked"]
        n = len({
            (*_canonical_pair(p1, p2), t)
            for p1, p2, t in video["holds"].get("meeting", ())
            if (p1, t) in tracked and (p2, t) in tracked
        })
        out[video["video"]] = n
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-match the OLED dump's meeting transition events "
        "against the 30 CAVIAR ground-truth XML videos and classify each "
        "as real / duplicate / splice_artifact (README section F). "
        "CPU-only, deterministic.",
    )
    p.add_argument("--train-json", required=True, help="path to caviar-train.json")
    p.add_argument("--test-json", required=True, help="path to caviar-test.json")
    p.add_argument("--xml-dir", required=True, help="directory of the 30 CAVIAR ground-truth XML files")
    p.add_argument("--out", required=True, help="path to write the audit JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from caviar_continuous import load_continuous
    from caviar_xml_corpus import load_xml_corpus

    train_segments = load_continuous(args.train_json)
    test_segments = load_continuous(args.test_json)
    videos = load_xml_corpus(args.xml_dir)

    traj, index = build_xml_xy_index(videos)
    seg_matches_by_split = {
        "train": match_segments(train_segments, videos, traj, index),
        "test": match_segments(test_segments, videos, traj, index),
    }
    dump_events = (
        dump_meeting_events(train_segments, "train")
        + dump_meeting_events(test_segments, "test")
    )
    xml_events = xml_meeting_events(videos)
    audit = classify_events(dump_events, xml_events, seg_matches_by_split, videos)
    audit["duplicated_gold_pairframes"] = duplicated_gold_pairframes(
        videos, seg_matches_by_split,
    )
    audit["segment_video_map"] = {
        split: [
            sorted(
                (
                    {
                        "video": videos[v_idx]["video"], "offset": offset,
                        "n_matched_points": sum(len(m["ts"]) for m in members.values()),
                        "persons": {pid: members[pid]["xml_pid"] for pid in sorted(members)},
                    }
                    for (v_idx, offset), members in match["clusters"].items()
                ),
                key=lambda c: (c["video"], c["offset"]),
            )
            for match in matches
        ]
        for split, matches in seg_matches_by_split.items()
    }
    result = {
        "train_json": args.train_json,
        "test_json": args.test_json,
        "xml_dir": args.xml_dir,
        "match_min_points": MATCH_MIN_POINTS,
        **audit,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    print(f"dump events: {audit['n_dump_events']}  xml events: {audit['n_xml_events']}")
    print(f"classification: {audit['classification_counts']}")
    print(f"xml events unclaimed by any dump event: {len(audit['xml_events_unclaimed'])}")
    print(f"duplicated gold pair-frames: {audit['duplicated_gold_pairframes']}")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
