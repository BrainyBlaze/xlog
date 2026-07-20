"""CAVIAR star-search pod entrypoint (task S3a).

Runs the first real WOLED/CAVIAR star-topology candidate search: load a fold
split of `caviar_folds.pkl`, convert it to the pair-time relation space
(`caviar_convert.convert_split`), compile a schema-only star program and
`put_relation` the TRAIN split's ground relations into it, then
`kfold_select` the star body (`head(X, Y) :- bL(X, Y), bR(X, Y)`) over that
candidate pool by k-fold holdout. If a rule is selected, it is ALSO scored on
the held-out TEST split -- by exact set intersection in plain Python
(`scorer.rule_predictions`), not a second engine pass; that is an honest
reading of the star body (no existential to lose by skipping the engine),
documented in `scorer.py`.

CUDA-ONLY AT RUNTIME: `IlpProgramFactory.compile`/`put_relation` need a real
CUDA device (mirrors `caviar_convert.put_caviar_relations`'s own guard), so
this refuses fast with a clear message rather than failing deep inside the
engine. Argument parsing itself (``--help``) does NOT need CUDA or even
`pyxlog`/`torch` to succeed -- see `main()`, which defers those imports past
`parse_args`.

COORDS_MISSING IS EXCLUDED FROM THE CANDIDATE VOCABULARY. `coords_missing`
(from `convert_split`) flags pair-times where a person's coordinates were
absent from the atoms string -- a data-quality marker, not activity/proximity
evidence a star rule should be allowed to explain meetings with. It is
therefore never declared in the compiled schema and never `put_relation`'d
into the engine here, so `valid_candidates` can never enumerate it as a
bL/bR candidate. It remains available in Python -- `converted["relations"]
["coords_missing"]` and `converted["n_coords_missing"]` -- for a FUTURE
witness_mask (contract #155) built from it; wiring that mask up is out of
scope for this task and left for later.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

from caviar_convert import (  # noqa: E402
    build_star_schema_source,
    convert_split,
    load_folds,
    put_caviar_relations,
)
from scorer import baseline_report, prf1, rule_predictions  # noqa: E402

CLOSE_THRESHOLD = 25.0
MASK_NAME = "W"  # the learnable weight name in build_star_schema_source's template
N_LABELS = 2
MEMORY_MB = 2048


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAVIAR star-topology candidate search over a fold split "
        "(task S3a). Needs CUDA at run time; --help does not."
    )
    p.add_argument("--pkl", required=True, help="path to caviar_folds.pkl")
    p.add_argument("--fold", default="fold1", help="fold key, e.g. fold1 (default: fold1)")
    p.add_argument("--k", type=int, default=4, help="k-fold holdout folds for kfold_select (default: 4)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed, covers the whole run (default: 7)")
    p.add_argument("--steps", type=int, required=True, help="training steps per fold")
    p.add_argument("--out", required=True, help="path to write RESULT.json")
    return p.parse_args(argv)


def _require_cuda() -> None:
    """Fail fast, before touching pyxlog, if there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "run_caviar_star.py needs a CUDA device: IlpProgramFactory.compile "
            "and put_relation (DLPack over device='cuda' tensors) only run on "
            "CUDA -- the same guard caviar_convert.put_caviar_relations "
            "enforces at call time. Run this on the RunPod A40 target, not "
            "locally (this environment has torch.cuda.is_available() == "
            "False)."
        )


def _filtered_relation_names(converted: dict) -> list[str]:
    """Every relation name from `convert_split`'s output EXCEPT
    'coords_missing' -- see the module docstring for why that one is
    excluded from the candidate vocabulary."""
    return sorted(name for name in converted["relations"] if name != "coords_missing")


def _compile_and_ingest(pyxlog, converted: dict, n_labels: int = N_LABELS):
    """Schema-only compile + `put_relation` of every TRAIN relation except
    'coords_missing', reusing `caviar_convert`'s own helpers (never
    duplicating their schema-source or column-building logic)."""
    relation_names = _filtered_relation_names(converted)
    schema_src = build_star_schema_source(relation_names)
    prog = pyxlog.IlpProgramFactory.compile(schema_src, device=0, memory_mb=MEMORY_MB)

    ingest_converted = dict(
        converted,
        relations={
            name: pairs
            for name, pairs in converted["relations"].items()
            if name != "coords_missing"
        },
    )
    returned_schema = put_caviar_relations(prog, ingest_converted, n_labels=n_labels)
    if returned_schema != schema_src:
        raise RuntimeError(
            "put_caviar_relations's derived schema does not match the schema "
            "the program was actually compiled with -- the ingested "
            "relations and the compiled declarations have drifted apart:\n"
            f"compiled:\n{schema_src}\nderived:\n{returned_schema}"
        )
    return prog


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _require_cuda()

    import torch
    import pyxlog
    from pyxlog.ilp.neural_credit import kfold_select

    wall: dict[str, float] = {}

    t0 = time.perf_counter()
    folds = load_folds(args.pkl)
    if args.fold not in folds:
        raise KeyError(
            f"fold {args.fold!r} not found in {args.pkl!r} (have: "
            f"{sorted(folds)})."
        )
    split = folds[args.fold]
    train = convert_split(split["train"], close_threshold=CLOSE_THRESHOLD)
    test = convert_split(split["test"], close_threshold=CLOSE_THRESHOLD)
    wall["convert"] = time.perf_counter() - t0

    def prog_factory():
        return _compile_and_ingest(pyxlog, train)

    device = torch.device("cuda")
    features = train["features"].to(device)

    def make_network():
        return torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], N_LABELS), torch.nn.Softmax(dim=-1)
        ).to(device)

    t1 = time.perf_counter()
    selection = kfold_select(
        prog_factory,
        MASK_NAME,
        train["facts"],
        train["is_positive"],
        make_network,
        features,
        neural_relations={},  # v1 probe: relational star candidates only, no detector yet
        folds=args.k,
        seed=args.seed,
        steps=args.steps,
        topology="star",
    )
    wall["kfold_select"] = time.perf_counter() - t1

    t2 = time.perf_counter()
    train_baselines = baseline_report(train["relations"], train["is_positive"], train["num_pt"])
    test_baselines = baseline_report(test["relations"], test["is_positive"], test["num_pt"])

    selected_prf1 = None
    if selection.rule is not None:
        left, right = selection.rule
        train_pred = rule_predictions(left, right, train["relations"], train["num_pt"])
        test_pred = rule_predictions(left, right, test["relations"], test["num_pt"])
        selected_prf1 = {
            "train": prf1(train_pred, train["is_positive"]),
            "test": prf1(test_pred, test["is_positive"]),
        }
    wall["test_scoring"] = time.perf_counter() - t2
    wall["total"] = time.perf_counter() - t0

    result = {
        "pkl": args.pkl,
        "fold": args.fold,
        "close_threshold": CLOSE_THRESHOLD,
        "k": args.k,
        "seed": args.seed,
        "steps": args.steps,
        "num_pt": {"train": train["num_pt"], "test": test["num_pt"]},
        "n_pos": {
            "train": int(sum(train["is_positive"])),
            "test": int(sum(test["is_positive"])),
        },
        "n_coords_missing": {
            "train": train["n_coords_missing"],
            "test": test["n_coords_missing"],
        },
        "selection": {
            "rule": list(selection.rule) if selection.rule is not None else None,
            "tied": [list(t) for t in selection.tied],
            "margin": selection.margin,
            "top_weight": selection.top_weight,
            "reason": selection.reason,
            "coverage": {f"{l}|{r}": v for (l, r), v in selection.coverage.items()},
        },
        "selected_rule_prf1": selected_prf1,
        "baselines": {"train": train_baselines, "test": test_baselines},
        "wall_clock_s": wall,
    }

    Path(args.out).write_text(json.dumps(result, indent=2))

    print(
        f"CAVIAR star-search: pkl={args.pkl} fold={args.fold} k={args.k} "
        f"seed={args.seed} steps={args.steps}"
    )
    print(
        f"  train: num_pt={train['num_pt']} n_pos={result['n_pos']['train']} "
        f"n_coords_missing={train['n_coords_missing']}"
    )
    print(
        f"  test:  num_pt={test['num_pt']} n_pos={result['n_pos']['test']} "
        f"n_coords_missing={test['n_coords_missing']}"
    )
    print(f"  selection: rule={selection.rule} reason={selection.reason}")
    print(f"             margin={selection.margin:.4f} top_weight={selection.top_weight:.4f}")
    if selected_prf1 is not None:
        print(f"  selected rule train prf1: {selected_prf1['train']}")
        print(f"  selected rule test  prf1: {selected_prf1['test']}")
    print("  baselines (test):")
    for name, scores in test_baselines.items():
        print(
            f"    {name}: precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} f1={scores['f1']:.3f} "
            f"degenerate={scores['degenerate']}"
        )
    print(
        f"  wall clock: convert={wall['convert']:.2f}s "
        f"kfold_select={wall['kfold_select']:.2f}s "
        f"test_scoring={wall['test_scoring']:.2f}s total={wall['total']:.2f}s"
    )
    print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
