"""Manual verification for `caviar_convert.convert_split` against the REAL
`caviar_folds.pkl` (task S2).

Loads the pkl, converts one fold/split, then samples a fixed number of
pair-times (seeded, so the sample is reproducible) and prints everything a
human needs to eyeball-verify the conversion by hand: the datapoint tag,
timestep, decoded p1/p2 simple labels, the coords parsed straight from the
``atoms`` string, the euclidean distance computed from those coords, which
relations the converter asserted for that pair-time, and the complex label.

Usage:
    python verify_conversion.py <path-to-caviar_folds.pkl> [fold] [split] [seed] [n]

Defaults: fold=fold1, split=train, seed=0, n=10.

This script does NOT commit or read any output file -- it is meant to be
run once and the printed output pasted into the task report by hand.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from caviar_convert import (  # noqa: E402
    COMPLEX_MEETING_ID,
    SIMPLE_LABEL_NAMES,
    _parse_coords,
    convert_split,
    load_folds,
    window_length,
)

COMPLEX_LABEL_NAMES = {0: "no_interaction", 1: "meeting", 2: "moving"}


def main() -> None:
    argv = sys.argv[1:]
    if not argv:
        print(__doc__)
        raise SystemExit(1)
    path = argv[0]
    fold = argv[1] if len(argv) > 1 else "fold1"
    split = argv[2] if len(argv) > 2 else "train"
    seed = int(argv[3]) if len(argv) > 3 else 0
    n = int(argv[4]) if len(argv) > 4 else 10

    print(f"Loading {path} ...")
    folds = load_folds(path)
    datapoints = folds[fold][split]
    print(f"fold={fold} split={split}: {len(datapoints)} datapoints")

    converted = convert_split(datapoints)
    T = window_length(datapoints[0])
    num_pt = converted["num_pt"]
    print(f"T (window length) = {T}, num_pt = {num_pt}, "
          f"n_coords_missing = {converted['n_coords_missing']}")

    relation_sets = {
        name: set(pairs) for name, pairs in converted["relations"].items()
    }

    rng = random.Random(seed)
    sample_pts = sorted(rng.sample(range(num_pt), min(n, num_pt)))

    print(f"\nSampling {len(sample_pts)} pair-times (seed={seed}):\n")
    for pt in sample_pts:
        dp_index, t = divmod(pt, T)
        dp = datapoints[dp_index]
        tag = dp["tag"]

        p1_labels = dp["p1_labels"].flatten().tolist()
        p2_labels = dp["p2_labels"].flatten().tolist()
        complex_labels = dp["complex_labels"].flatten().tolist()
        p1_name = SIMPLE_LABEL_NAMES[p1_labels[t]]
        p2_name = SIMPLE_LABEL_NAMES[p2_labels[t]]
        complex_id = complex_labels[t]
        complex_name = COMPLEX_LABEL_NAMES.get(complex_id, f"?{complex_id}")

        p1_coords, p2_coords = _parse_coords(dp["atoms"])
        c1 = p1_coords.get(t)
        c2 = p2_coords.get(t)
        if c1 is not None and c2 is not None:
            dx, dy = c1[0] - c2[0], c1[1] - c2[1]
            dist = (dx * dx + dy * dy) ** 0.5
            dist_str = f"{dist:.2f}"
        else:
            dist_str = "N/A (coords missing)"

        asserted = sorted(
            name for name, s in relation_sets.items() if (pt, 1) in s
        )
        is_pos = converted["is_positive"][pt]
        feat = converted["features"][pt].tolist()

        print(f"pt={pt} (dp={dp_index} tag={tag} t={t})")
        print(f"  p1_label={p1_name} p2_label={p2_name}")
        print(f"  p1_coords={c1} p2_coords={c2} distance={dist_str}")
        print(f"  features(dx,dy)={feat}")
        print(f"  relations_asserted={asserted}")
        print(f"  complex_label={complex_name} (id={complex_id}) "
              f"is_positive(meeting)={is_pos} "
              f"[COMPLEX_MEETING_ID={COMPLEX_MEETING_ID}]")
        print()


if __name__ == "__main__":
    main()
