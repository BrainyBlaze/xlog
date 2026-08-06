"""Section G's diagnostic census, as a shipped tool: the numbers README
section G quotes for WHY the clean-corpus meeting zero is a data property
(two disjoint annotation regimes) rather than a harness failure — the
direct-protocol clause applied to the held-out wk-scene fold directly,
and the per-segment coverage census over meeting positives — previously
existed only as an unrecorded diagnostic run with no artifact.

WHAT IT COMPUTES (all deterministic, CPU-only, no induction):

1. The scene-family fold assignment, via `run_caviar_cv`'s OWN
   `_xml_family_fold_assignment` (same table, same seed semantics as the
   shipped `f_xml_scene_cv` CV artifacts — never a re-implementation),
   and which fold holds the wk scene family.
2. The fixed reference clause `both_inactive & close` (the clause the
   direct-protocol search selects on 9 of 10 folds in the shipped
   meeting CV artifact) applied DIRECTLY to the wk fold's held-out
   videos: frame-level P/R/F1 with tp/fp/fn — the "machinery is sound"
   number (README: frame F1 0.890, tp=1060, fp=0).
3. The same clause's coverage on the wk fold's TRAIN side (every
   non-wk-fold video): how many meeting positives and how many negatives
   it covers — the "cannot learn it at all without wk scenes" number.
4. A per-video census over every meeting-bearing video: gold meeting
   pair-frames, how many of them the clause covers, and how many of them
   `close` alone covers (the lb1gt/rffgt "meeting annotated at distances
   above the close threshold" observation falls out of the last column).

Usage:
    python examples/caviar_woled/xml_meeting_census.py \
      --xml-dir <dir with the 30 CAVIAR ground-truth XML files> \
      --out CENSUS.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

# The clause under census: the direct-protocol reference selection on 9 of
# 10 folds in results/f_xml_scene_cv/caviar-f-xml-meeting-cv10.json. Fixed
# here, not searched — this tool measures a KNOWN clause's coverage
# structure, it does not induce anything.
CLAUSE: tuple[str, str] = ("both_inactive", "close")
FLUENT = "meeting"


def _clause_rows(converted: dict, clause: tuple[str, str]) -> set[int]:
    """The pair-time rows (pt ids) where every literal of ``clause`` holds
    -- a star-rule body's cover is a plain set intersection of its
    relations' own row sets (`relational_search.body_cover`'s reading,
    inlined here over the converted dict's ``relations`` lists)."""
    sets = []
    for name in clause:
        sets.append({pt for pt, _ in converted["relations"][name]})
    rows = sets[0]
    for s in sets[1:]:
        rows = rows & s
    return rows


def _prf1(pred_rows: set[int], is_positive: list[bool]) -> dict:
    tp = sum(1 for pt in pred_rows if is_positive[pt])
    fp = len(pred_rows) - tp
    fn = sum(is_positive) - tp
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def census_by_video(videos: list[dict], clause: tuple[str, str] = CLAUSE) -> list[dict]:
    """Per video: gold meeting pair-frame count, the clause's coverage of
    those positives, and ``close``'s own coverage of them (each video
    converted on its own, so row ids never cross video boundaries)."""
    from caviar_xml_corpus import convert_xml_corpus

    out = []
    for video in videos:
        converted = convert_xml_corpus([video], fluent=FLUENT)
        positives = {pt for pt, pos in enumerate(converted["is_positive"]) if pos}
        clause_rows = _clause_rows(converted, clause)
        close_rows = {pt for pt, _ in converted["relations"]["close"]}
        out.append({
            "video": video["video"],
            "n_meeting_positive_pairframes": len(positives),
            "n_covered_by_clause": len(positives & clause_rows),
            "n_close_on_positives": len(positives & close_rows),
        })
    return out


def fold_scoring(
    videos: list[dict], fold_of_video: list[int], fold_index: int,
    clause: tuple[str, str] = CLAUSE,
) -> dict:
    """The clause applied directly to fold ``fold_index``'s held-out
    videos (frame-level P/R/F1) plus its coverage on that fold's TRAIN
    side — the two numbers README section G's transfer argument rests
    on. Conversion is per side, exactly like `run_caviar_cv.run_fold`'s
    own fold-local conversion."""
    from caviar_xml_corpus import convert_xml_corpus

    test_videos = [v for v, f in zip(videos, fold_of_video) if f == fold_index]
    train_videos = [v for v, f in zip(videos, fold_of_video) if f != fold_index]

    test = convert_xml_corpus(test_videos, fluent=FLUENT)
    test_scoring = _prf1(_clause_rows(test, clause), test["is_positive"])

    train = convert_xml_corpus(train_videos, fluent=FLUENT)
    train_rows = _clause_rows(train, clause)
    train_pos_covered = sum(1 for pt in train_rows if train["is_positive"][pt])
    return {
        "fold": fold_index,
        "test_videos": [v["video"] for v in test_videos],
        "clause_applied_to_test_fold": test_scoring,
        "train_side": {
            "n_positives_total": int(sum(train["is_positive"])),
            "n_positives_covered_by_clause": train_pos_covered,
            "n_negatives_covered_by_clause": len(train_rows) - train_pos_covered,
        },
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Per-segment meeting-coverage census + the "
        "both_inactive & close clause applied directly to the held-out "
        "wk-scene fold (README section G's diagnostic numbers). "
        "CPU-only, deterministic.",
    )
    p.add_argument("--xml-dir", required=True, help="directory of the 30 CAVIAR ground-truth XML files")
    p.add_argument("--folds", type=int, default=10, help="number of CV folds (default: 10, as in section G)")
    p.add_argument("--seed", type=int, default=7, help="fold-assignment seed (default: 7, as in section G)")
    p.add_argument("--out", required=True, help="path to write the census JSON")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from caviar_xml_corpus import load_xml_corpus
    from run_caviar_cv import _xml_family_fold_assignment, _xml_stem

    videos = load_xml_corpus(args.xml_dir)
    fold_of_video = _xml_family_fold_assignment(videos, FLUENT, args.folds, args.seed)

    # The wk fold is wherever the wk scene family was dealt -- read off
    # the assignment itself, never hardcoded.
    wk_fold = next(
        f for v, f in zip(videos, fold_of_video) if _xml_stem(v["video"]) == "wk1gt"
    )

    census = census_by_video(videos)
    wk = fold_scoring(videos, fold_of_video, wk_fold)

    result = {
        "xml_dir": args.xml_dir,
        "folds": args.folds,
        "seed": args.seed,
        "fluent": FLUENT,
        "clause": list(CLAUSE),
        "wk_fold": wk_fold,
        "fold_of_video": {
            v["video"]: f for v, f in zip(videos, fold_of_video)
        },
        "wk_fold_scoring": wk,
        "census_by_video": census,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))

    s = wk["clause_applied_to_test_fold"]
    print(f"wk fold = {wk_fold}; clause {CLAUSE[0]} & {CLAUSE[1]} on its held-out videos: "
          f"P={s['precision']:.4f} R={s['recall']:.4f} F1={s['f1']:.4f} "
          f"(tp={s['tp']} fp={s['fp']} fn={s['fn']})")
    t = wk["train_side"]
    print(f"train side: clause covers {t['n_positives_covered_by_clause']}/"
          f"{t['n_positives_total']} positives and {t['n_negatives_covered_by_clause']} negatives")
    for row in census:
        if row["n_meeting_positive_pairframes"]:
            print(f"  {row['video']}: clause {row['n_covered_by_clause']}/"
                  f"{row['n_meeting_positive_pairframes']} "
                  f"(close alone: {row['n_close_on_positives']})")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
