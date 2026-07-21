"""CAVIAR neural-close DIFFERENTIATOR pod entrypoint (task S4a).

WHAT THIS SCRIPT IS FOR. `run_caviar_star.py` (task S3a) searched the star
body over a candidate pool that INCLUDES the precomputed `close`/`far`
relations (euclidean distance vs. a fixed threshold, computed once in
`caviar_convert.convert_split` and handed to the engine as ground truth).
This script removes that crutch: `close`/`far` (and `coords_missing`) are
NEVER declared in the compiled program, NEVER `put_relation`'d, and NEVER
visible to `kfold_select`'s candidate pool at all -- the search only ever
sees the four activity relations (`both_active`, `both_walking`,
`both_inactive`, `mixed_active_walking`) as RELATIONAL candidates, plus one
NEURAL relation, `close_nn`, trained end to end through the same engine
credit as everything else in `pyxlog.ilp.neural_credit`, over the raw
per-pair-time `(dx, dy)` coordinates `convert_split` already exposes as
`"features"`. If a rule built on `close_nn` is ever competitive with a rule
built on the precomputed `close`, that is the differentiator's positive
result: a network trained purely through dILP-style engine credit, with no
distance/threshold supervision anywhere, has learned to approximate the
same geometric predicate a hand-coded `close/2` provides.

LOUD STATEMENT OF THE CANDIDATE VOCABULARY (read this twice):

    RELATIONAL candidates (compiled + `put_relation`'d, real ground truth):
        both_active, both_walking, both_inactive, mixed_active_walking
    NEURAL candidate (declared with a seed row so `valid_candidates`
    cross-products it, but NEVER `put_relation`'d -- see `_compile_and_ingest`):
        close_nn
    EXCLUDED FROM THE PROGRAM ENTIRELY -- not declared, not ingested, not
    enumerable by `valid_candidates`, not passed to `kfold_select` in any
    form:
        close, far, coords_missing

`convert_split` still computes `close`/`far`/`coords_missing` internally (it
always has, since S2) -- this script reads `train["relations"]["close"]` /
`test["relations"]["close"]` ONLY for the DETECTOR PROBE below, which
compares the trained network's output against that ground truth AFTER
training has finished. `close`/`far` never reach `kfold_select`,
`train_engine_mode`, or the network's loss in any form -- grep this file (and
`caviar_convert.py`, unmodified) yourself before trusting that claim.

`close_nn`'S ENGINE REGISTRATION. Star mode's neural-tail semantics
(`pyxlog.ilp.neural_credit.enumerate_specs`, `topology="star"`) are:
"a neural B's witness set for fact (x, y) is the SINGLE flat row
`x * n_labels + y` ... but ONLY when A covers (x, y)" -- a cover-gated
single-witness noisy-OR that reduces to exactly the network's own
probability at that row. A candidate (`A`, `close_nn`) therefore needs
`close_nn` to appear in the engine's `valid_candidates` cross product, which
requires it to be a DECLARED relation in the compiled schema (arity 2, one
seed row, exactly like every other relation `build_star_schema_source`
emits) -- but `enumerate_specs` never calls `prog.relation_facts("close_nn")`
for a name registered in `neural_relations` (the `if rn not in
neural_relations: ... _readable(rn) ...` check is skipped outright for a
neural right), so the seed row's actual content is inert: exactly the same
principle the engine already relies on for the `bL`/`bR` dILP template
placeholders, which have NO ground extension at all and are still
cross-producted by `valid_candidates` before being pool-filtered downstream.
`close_nn` is therefore declared but never `put_relation`'d -- see
`_compile_and_ingest`'s docstring for the exact mechanics. `neural_credit.py`
is NOT modified by this task; this registration uses only its existing,
tested API (mirrors `test_star_neural_tail_is_cover_gated_single_witness` in
`test_neural_credit.py`).

NETWORK SHAPE (the exact convention a pod runner should sanity-check
against): `close_nn`'s MLP is `Linear(2, hidden) -> ReLU -> Linear(hidden,
hidden) -> ReLU -> Linear(hidden, N_LABELS=2) -> Softmax(dim=-1)`, applied to
`features` (`[num_pt, 2]`, the `(dx, dy)/100` tensor `convert_split`
returns) to produce `[num_pt, 2]` -- the same 2-D `[num_events, num_labels]`
output shape `train_engine_mode`'s `_validated_output` requires (mirrors
`run_caviar_star.py`'s `make_network`, just with two hidden layers instead
of a bare `Linear`). `--hidden` (default 16) is the hidden width.

CLOSE-CALL THRESHOLD (documented choice): the star engine's cover-gated
held-out scoring uses `s_c(f) >= 0.5` (non-strict); this script's TEST-time
prediction for a selected `close_nn` rule uses `network output at label 1
> 0.5` (strict). The strict/non-strict difference matters only at exactly
0.5 -- measure-zero for a float softmax output -- so the threshold VALUE is
shared with the engine's semantics and not re-tuned here, but the
comparisons are not literally identical operators.

`--steps` STAYS UN-CLAMPED, UNLIKE S3a (documented there as a cost-knob
guard when `neural_relations={}`): here `neural_relations` always contains
`close_nn`, so every held-out score for an (`A`, `close_nn`) candidate DOES
depend on the trained network (`spec.is_neural` is taken), and the
star-canonicalized relational-relational candidates still exist alongside it
-- training steps are a real result knob. The clamp code is kept (mirroring
S3a's structure so a future all-relational configuration stays protected by
the same guard) but is provably dead code in THIS script, since
`neural_relations` is never empty here.

DETECTOR PROBE (the differentiator's actual evidence -- see
`detector_probe.py`): after training, the network's output is compared
against the GROUND-TRUTH `close` relation (train AND test), never fed to
training, in three ways -- accuracy/PRF1 of `net > 0.5` vs. `close`; a
per-distance-bin mean-score table (bins in `detector_probe.DIST_BIN_EDGES`,
recomputed as `dist = 100 * sqrt(dx^2 + dy^2)` from the SAME `features`
tensor used for training, never a separate distance channel); and the same
two readings for an UNTRAINED CONTROL network sharing the trained network's
OWN INIT SEED (`torch.manual_seed(args.seed)` immediately before each
`make_network()` call, so their initial weights are IDENTICAL) -- so any
difference between the trained and control probes is attributable to
training, not to a different random init.

CUDA-ONLY AT RUNTIME (mirrors `run_caviar_star.py`): `IlpProgramFactory.
compile`/`put_relation`/`kfold_select` need a real CUDA device; `--help`
needs neither CUDA nor `pyxlog`/`torch` (every such import is deferred past
`parse_args`, exactly like S3a after its F3 fix).
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
if str(EXAMPLE_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLE_DIR))

# Both `scorer` and `detector_probe` are pure Python/duck-typed (no torch or
# engine import at module level -- see their own docstrings), so importing
# them here does not compromise --help's dependency-free claim, unlike
# `caviar_convert` (imports torch at module level), which stays deferred into
# the functions that actually need it.
from detector_probe import monotone_decay_report, probe_detector  # noqa: E402
from scorer import baseline_report, prf1, rule_predictions  # noqa: E402

CLOSE_THRESHOLD = 25.0  # ground-truth-probe use only; see module docstring
MASK_NAME = "W"
N_LABELS = 2
MEMORY_MB = 2048

# The ONLY relational candidate vocabulary this script compiles/ingests --
# see the module docstring's "LOUD STATEMENT" paragraph.
ACTIVITY_RELATIONS: tuple[str, ...] = (
    "both_active", "both_inactive", "both_walking", "mixed_active_walking",
)
CLOSE_NN_NAME = "close_nn"

# S3a's cost-knob guard, kept for structural parity (see module docstring's
# "--steps STAYS UN-CLAMPED" paragraph) -- unreachable in this script, since
# `neural_relations` always contains `close_nn` here.
EMPTY_NEURAL_POOL_STEP_CAP = 25

# Activities-only baseline pairs (close/far are absent from the vocabulary,
# so `scorer.DEFAULT_BASELINE_PAIRS`, which pairs activities against `close`,
# does not apply here) -- every unordered pair among the four relations.
ACTIVITY_BASELINE_PAIRS: list[tuple[str, str]] = list(
    itertools.combinations(sorted(ACTIVITY_RELATIONS), 2)
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAVIAR neural-close differentiator: star-topology "
        "candidate search with a trained close_nn detector replacing the "
        "precomputed close/far relations (task S4a). Needs CUDA at run "
        "time; --help does not."
    )
    p.add_argument("--pkl", required=True, help="path to caviar_folds.pkl")
    p.add_argument("--fold", default="fold1", help="fold key, e.g. fold1 (default: fold1)")
    p.add_argument("--k", type=int, default=4, help="k-fold holdout folds for kfold_select (default: 4)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed, covers the whole run (default: 7)")
    p.add_argument(
        "--steps", type=int, required=True,
        help="training steps per fold, AND for the final full-train pass. "
        "Unlike run_caviar_star.py, this is a real result knob here: "
        "close_nn is always a registered neural candidate.",
    )
    p.add_argument("--hidden", type=int, default=16, help="close_nn MLP hidden width (default: 16)")
    p.add_argument("--out", required=True, help="path to write RESULT.json")
    return p.parse_args(argv)


def _require_cuda() -> None:
    """Fail fast, before touching pyxlog, if there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "run_caviar_neural.py needs a CUDA device: IlpProgramFactory."
            "compile and put_relation (DLPack over device='cuda' tensors) "
            "only run on CUDA -- the same guard run_caviar_star.py and "
            "caviar_convert.put_caviar_relations enforce at call time. Run "
            "this on the RunPod A40 target, not locally (this environment "
            "has torch.cuda.is_available() == False)."
        )


def _prepare_out_path(out: str) -> Path:
    """Create --out's parent directory and write a tiny probe file BEFORE
    any expensive work starts (mirrors S3a's F4 fix): a missing parent
    directory or a permissions problem must surface in the first second,
    not after a paid GPU run. Overwritten by the real RESULT.json at the
    end of a successful run."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("started\n")
    return out_path


def _activity_only_relations(converted: dict) -> dict[str, list[tuple[int, int]]]:
    """`converted["relations"]` filtered to the four activity relations only
    -- excludes `close`, `far`, and `coords_missing` (see the module
    docstring's "LOUD STATEMENT" paragraph). Raises if an expected activity
    relation is missing (a `convert_split` contract change should surface
    here, not as a silently smaller candidate pool)."""
    missing = [name for name in ACTIVITY_RELATIONS if name not in converted["relations"]]
    if missing:
        raise KeyError(
            f"convert_split's output is missing expected activity "
            f"relation(s) {missing}; have: {sorted(converted['relations'])}."
        )
    return {name: converted["relations"][name] for name in ACTIVITY_RELATIONS}


def _compile_and_ingest(pyxlog, converted: dict, n_labels: int = N_LABELS):
    """Compile a schema declaring the four activity relations PLUS
    `close_nn` (a seed row only -- see the module docstring's "`close_nn`'S
    ENGINE REGISTRATION" paragraph), then `put_relation` ONLY the four
    activity relations' real ground-truth rows. `close_nn` is deliberately
    never `put_relation`'d: it has no table to ingest, its score comes
    entirely from the network at held-out-scoring time.

    The parity check below only compares the ACTIVITY-relation subset of the
    compiled schema against what `put_caviar_relations` derives from the
    ingested dict -- `close_nn` is excluded from that check by construction
    (it is never in the ingested dict), not because the check is weaker,
    but because there is nothing on the ingest side to compare it against.
    """
    from caviar_convert import build_star_schema_source, put_caviar_relations

    activity_names = sorted(ACTIVITY_RELATIONS)
    schema_src = build_star_schema_source(activity_names + [CLOSE_NN_NAME])
    prog = pyxlog.IlpProgramFactory.compile(schema_src, device=0, memory_mb=MEMORY_MB)

    ingest_converted = dict(converted, relations=_activity_only_relations(converted))
    returned_schema = put_caviar_relations(prog, ingest_converted, n_labels=n_labels)
    expected_ingest_schema = build_star_schema_source(activity_names)
    if returned_schema != expected_ingest_schema:
        raise RuntimeError(
            "put_caviar_relations's derived schema (over the activity-only "
            "ingest dict) does not match the expected activity-only schema "
            "-- the ingested relations and the compiled activity "
            "declarations have drifted apart:\n"
            f"expected:\n{expected_ingest_schema}\nderived:\n{returned_schema}"
        )
    return prog


def _build_mlp(hidden: int, device):
    """`close_nn`'s network: `Linear(2, hidden) -> ReLU -> Linear(hidden,
    hidden) -> ReLU -> Linear(hidden, N_LABELS) -> Softmax(dim=-1)` -- 2-D
    `[num_events, num_labels]` output, the exact shape
    `train_engine_mode`'s `_validated_output` requires. See the module
    docstring's "NETWORK SHAPE" paragraph."""
    import torch

    return torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_LABELS),
        torch.nn.Softmax(dim=-1),
    ).to(device)


def _pair_dists(features) -> list[float]:
    """Ground-truth euclidean distance per pair-time, recovered from the
    SAME `(dx, dy)/100` `features` tensor training used -- `dist = 100 *
    sqrt(dx^2 + dy^2)` undoes `caviar_convert.FEATURE_SCALE`. Used ONLY by
    the detector probe, never by training."""
    return [100.0 * ((dx * dx + dy * dy) ** 0.5) for dx, dy in features.tolist()]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Fail fast on a bad --out BEFORE any CUDA check or expensive work.
    out_path = _prepare_out_path(args.out)

    _require_cuda()

    import torch
    import pyxlog
    from pyxlog.ilp.neural_credit import NeuralRelationSpec, kfold_select, train_engine_mode

    from caviar_convert import convert_split, load_folds

    wall: dict[str, float] = {}

    t0 = time.perf_counter()
    folds = load_folds(args.pkl)
    if args.fold not in folds:
        available = sorted(
            k for k, v in folds.items()
            if isinstance(v, dict) and "train" in v and "test" in v
        )
        raise KeyError(f"fold {args.fold!r} not found in {args.pkl!r} (have: {available}).")
    split = folds[args.fold]
    train = convert_split(split["train"], close_threshold=CLOSE_THRESHOLD)
    test = convert_split(split["test"], close_threshold=CLOSE_THRESHOLD)
    wall["convert"] = time.perf_counter() - t0

    def prog_factory():
        return _compile_and_ingest(pyxlog, train)

    device = torch.device("cuda")
    features_train = train["features"].to(device)
    features_test = test["features"].to(device)

    def make_network():
        return _build_mlp(args.hidden, device)

    # The ONE neural candidate: close_nn, over the train split's pair-time
    # domain (star mode's witness row IS the fact's own key -- see
    # enumerate_specs' star docstring). Test scoring never re-enters the
    # engine (see below), so no test-side registration is needed.
    neural_relations = {CLOSE_NN_NAME: NeuralRelationSpec(num_rows=train["num_pt"], arity=2)}

    # See module docstring's "--steps STAYS UN-CLAMPED" paragraph: this
    # branch is provably dead in this script (neural_relations is never
    # empty here), kept only for structural parity with run_caviar_star.py.
    steps_requested = args.steps
    if not neural_relations:
        steps_effective = min(args.steps, EMPTY_NEURAL_POOL_STEP_CAP)
        steps_clamped = steps_effective != steps_requested
        print(
            "WARNING: neural_relations is empty -- see run_caviar_star.py's "
            "cost-knob guard. Clamping training steps per fold from "
            f"--steps={steps_requested} to {steps_effective}."
        )
    else:
        steps_effective = steps_requested
        steps_clamped = False

    t1 = time.perf_counter()
    selection = kfold_select(
        prog_factory,
        MASK_NAME,
        train["facts"],
        train["is_positive"],
        make_network,
        features_train,
        neural_relations=neural_relations,
        folds=args.k,
        seed=args.seed,
        steps=steps_effective,
        topology="star",
    )
    wall["kfold_select"] = time.perf_counter() - t1

    # Final full-train network (for TEST scoring + the detector probe) and
    # an UNTRAINED control network sharing the SAME init seed -- see the
    # module docstring's "DETECTOR PROBE" paragraph. kfold_select's own
    # per-fold networks are discarded after selection; this is a fresh
    # training pass over ALL of train, matching the deployed-artifact
    # convention (k-fold decides the rule, a full-data pass produces the
    # artifact actually scored on test).
    t2 = time.perf_counter()
    torch.manual_seed(args.seed)
    control_net = make_network()
    control_net.eval()

    torch.manual_seed(args.seed)  # identical init to control_net, by construction
    net_to_train = make_network()
    final_prog = prog_factory()
    train_result = train_engine_mode(
        final_prog, MASK_NAME, train["facts"], train["is_positive"],
        net_to_train, features_train, neural_relations=neural_relations,
        steps=steps_effective, seed=args.seed, topology="star",
    )
    trained_net = train_result.network
    trained_net.eval()
    wall["final_train"] = time.perf_counter() - t2

    t3 = time.perf_counter()
    with torch.no_grad():
        train_scores = trained_net(features_train)[:, 1].tolist()
        test_scores = trained_net(features_test)[:, 1].tolist()
        control_scores = control_net(features_test)[:, 1].tolist()

    train_dists = _pair_dists(train["features"])
    test_dists = _pair_dists(test["features"])
    train_close_rows = {pt for pt, _ in train["relations"]["close"]}
    test_close_rows = {pt for pt, _ in test["relations"]["close"]}
    train_missing = {pt for pt, _ in train["relations"]["coords_missing"]}
    test_missing = {pt for pt, _ in test["relations"]["coords_missing"]}

    probe_train = probe_detector(train_scores, train_close_rows, train_dists, exclude_rows=train_missing)
    probe_test = probe_detector(test_scores, test_close_rows, test_dists, exclude_rows=test_missing)
    # Control net scored on the SAME split (test) as the trained-net "test"
    # probe, so the two are directly comparable -- the only thing that
    # differs between them is whether training happened.
    probe_control = probe_detector(control_scores, test_close_rows, test_dists, exclude_rows=test_missing)

    detector_probe_result = {
        "train": probe_train,
        "test": probe_test,
        "control": probe_control,
        "monotone_decay": {
            "train": monotone_decay_report(probe_train["bins"]),
            "test": monotone_decay_report(probe_test["bins"]),
            "control": monotone_decay_report(probe_control["bins"]),
        },
        "note": (
            "close/far were never fed to training in any form; 'control' is "
            f"an UNTRAINED network sharing the trained network's own init "
            f"seed (seed={args.seed}, torch.manual_seed(seed) immediately "
            "before each make_network() call), so any difference between "
            "'test' and 'control' is attributable to training alone."
        ),
    }
    wall["detector_probe"] = time.perf_counter() - t3

    t4 = time.perf_counter()
    train_baselines = baseline_report(
        train["relations"], train["is_positive"], train["num_pt"], pairs=ACTIVITY_BASELINE_PAIRS
    )
    test_baselines = baseline_report(
        test["relations"], test["is_positive"], test["num_pt"], pairs=ACTIVITY_BASELINE_PAIRS
    )

    selected_prf1 = None
    if selection.rule is not None:
        left, right = selection.rule
        if right == CLOSE_NN_NAME:
            left_train_set = set(train["relations"][left])
            left_test_set = set(test["relations"][left])
            train_pred = [
                (pt, 1) in left_train_set and train_scores[pt] > 0.5
                for pt in range(train["num_pt"])
            ]
            test_pred = [
                (pt, 1) in left_test_set and test_scores[pt] > 0.5
                for pt in range(test["num_pt"])
            ]
        else:
            train_pred = rule_predictions(left, right, train["relations"], train["num_pt"])
            test_pred = rule_predictions(left, right, test["relations"], test["num_pt"])
        selected_prf1 = {
            "train": prf1(train_pred, train["is_positive"]),
            "test": prf1(test_pred, test["is_positive"]),
        }
    wall["scoring"] = time.perf_counter() - t4
    wall["total"] = time.perf_counter() - t0

    result = {
        "pkl": args.pkl,
        "fold": args.fold,
        "close_threshold": CLOSE_THRESHOLD,
        "k": args.k,
        "seed": args.seed,
        "hidden": args.hidden,
        "steps_requested": steps_requested,
        "steps_effective": steps_effective,
        "steps_clamped": steps_clamped,
        "candidate_vocabulary": {
            "relational": sorted(ACTIVITY_RELATIONS),
            "neural": [CLOSE_NN_NAME],
            "excluded": ["close", "far", "coords_missing"],
        },
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
        "detector_probe": detector_probe_result,
        "baselines": {"train": train_baselines, "test": test_baselines},
        "wall_clock_s": wall,
    }

    out_path.write_text(json.dumps(result, indent=2))

    print(
        f"CAVIAR neural-close differentiator: pkl={args.pkl} fold={args.fold} "
        f"k={args.k} seed={args.seed} hidden={args.hidden} "
        f"steps_requested={steps_requested} steps_effective={steps_effective} "
        f"(clamped={steps_clamped})"
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
    print("  detector probe (net > 0.5 vs. ground-truth close, never trained on):")
    for split_name, probe in (("train", probe_train), ("test", probe_test), ("control", probe_control)):
        print(
            f"    {split_name}: accuracy={probe['accuracy']:.4f} "
            f"f1={probe['prf1']['f1']:.4f} rows={probe['num_rows']} "
            f"excluded={probe['num_excluded']}"
        )
    for split_name in ("train", "test", "control"):
        m = detector_probe_result["monotone_decay"][split_name]
        print(
            f"    {split_name} monotone_non_increasing={m['monotone_non_increasing']} "
            f"knee={m['knee_label']} drop={m['knee_drop']:.4f}"
        )
    print("  baselines (test, activities-only -- close/far excluded):")
    for name, scores in test_baselines.items():
        print(
            f"    {name}: precision={scores['precision']:.3f} "
            f"recall={scores['recall']:.3f} f1={scores['f1']:.3f} "
            f"degenerate={scores['degenerate']}"
        )
    print(
        f"  wall clock: convert={wall['convert']:.2f}s "
        f"kfold_select={wall['kfold_select']:.2f}s "
        f"final_train={wall['final_train']:.2f}s "
        f"detector_probe={wall['detector_probe']:.2f}s "
        f"scoring={wall['scoring']:.2f}s total={wall['total']:.2f}s"
    )
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
