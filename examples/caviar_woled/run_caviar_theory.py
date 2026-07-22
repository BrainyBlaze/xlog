"""CAVIAR multi-clause THEORY LOOP pod entrypoint.

WHAT THIS SCRIPT IS FOR. `run_caviar_star.py` and `run_caviar_neural.py`
each search for a SINGLE star clause. A single clause is capped by CAVIAR
fold1's own composition: ``both_inactive`` covers
77.2% of TRAIN positives but only 21.5% of TEST positives, so whichever
single body the arbiter picks on train is structurally recall-capped on
test. A TWO-clause theory that also covers the OTHER composition mode
(``both_active``) lifts test F1 from 0.354/0.078 to 0.921. This
script wraps `theory_loop.induce_theory` (pure sequential-covering control
logic, engine-agnostic, unit-tested on its own) around the real engine to
build such a multi-clause theory for real, in two vocabularies:

* ``--mode relational``: the same vocabulary as `run_caviar_star.py` (4
  activities + the PRECOMPUTED
  ``close``/``far`` ground-truth relations) -- geometry is given, not
  learned. `select_once` wraps `kfold_select(topology="star")` over
  whatever residual facts/labels the theory loop hands it; the compiled
  PROGRAM and its ingested relations never change between iterations, only
  the fact/label lists searched over shrink.
* ``--mode neural``: the same vocabulary as `run_caviar_neural.py` (4
  activities only, plus a trained
  ``close_nn`` detector -- no precomputed geometry reaches the candidate
  pool at all) BUT with SYMMETRIZATION (see
  `_build_symmetric_mlp` below): the network is wrapped so
  ``forward(x) == forward(-x)`` EXACTLY, by construction, addressing the
  fact that a person-pair's (dx, dy) sign is an arbitrary numbering
  choice an unconstrained network can still lean on (an unconstrained net
  trained this way has measured pair-swap asymmetry up to ~0.364 on real
  data; the wrapper makes it exactly 0). Each theory-loop iteration RE-TRAINS a
  fresh network via `kfold_select`, and -- once a clause is accepted -- a
  SEPARATE full-train pass over that iteration's own residual produces the
  network actually deployed for THAT clause; the theory therefore ends up
  with one independently-trained ``close_nn`` network PER close_nn-tailed
  clause, not one shared network (documented honestly -- see
  `_run_neural_theory`'s docstring).

SYMMETRIZATION IS SCOPED TO THIS SCRIPT ONLY. `run_caviar_neural.py`'s
entrypoint semantics stay byte-equivalent --
its own network construction (`_build_mlp`) is untouched, and this script
does not import or call it. `caviar_convert.py`, `scorer.py`,
`detector_probe.py`, and `ec_scorer.py` are shared, unmodified-behavior
helpers (new functions were added to `caviar_convert.py`, `scorer.py`, and
`detector_probe.py`; `ec_scorer.py` is new; every pre-existing function's
behavior is unchanged -- see their own module docstrings).

CLOSE/FAR NEVER REACH NEURAL TRAINING (same guarantee as `run_caviar_neural.py`): in
``--mode neural``, `close`/`far`/`coords_missing` are never declared in the
compiled schema and never `put_relation`'d; they are read ONLY after all
training has finished, purely for the detector-probe/polar-spread/
pair-swap-asymmetry evidence below.

TWO EVALUATION PROTOCOLS. ``--protocol direct`` (default, unchanged) induces
ONE theory against the per-timestep holdsAt-style label, exactly as
described above. ``--protocol ec`` instead induces TWO theories -- one
against `caviar_convert.derive_ec_targets`'s ``is_init``, one against its
``is_term`` -- in the SAME candidate vocabulary and with the SAME theory-
loop control logic (relational mode: both reuse the one compiled program
and its set-intersection cover; neural mode: each gets its own per-clause
``close_nn`` networks, exactly as the direct protocol's single theory
already does). The two predicted event sequences are then reconstructed
into a holdsAt sequence by the classic Event-Calculus inertia rule
(`ec_scorer.reconstruct_holds`) and scored frame-by-frame against the same
gold the direct protocol scores against (`ec_scorer.frame_f1`); the direct
protocol's own theory is run once more, in full, on this SAME fold and
reported alongside for context.

CUDA-ONLY AT RUNTIME (mirrors `run_caviar_star.py`/`run_caviar_neural.py`): `IlpProgramFactory.compile`/
`put_relation`/`kfold_select`/`train_engine_mode` need a real CUDA device;
`--help` needs neither CUDA nor `pyxlog`/`torch` -- every such import is
deferred past `parse_args`.
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

# `theory_loop`, `scorer`, and `detector_probe` are all pure Python (no
# torch/engine import at module level -- see their own docstrings; the two
# new `detector_probe` functions used here import torch LAZILY, inside
# themselves), so importing them here does not compromise --help's
# dependency-free claim, unlike `caviar_convert` (imports torch at module
# level), which stays deferred into the functions that actually need it.
from detector_probe import (  # noqa: E402
    monotone_decay_report,
    pair_swap_asymmetry,
    polar_spread,
    probe_detector,
)
from ec_scorer import frame_f1, reconstruct_holds  # noqa: E402
from scorer import baseline_report, pr_curve, prf1, theory_predictions  # noqa: E402
from theory_loop import induce_theory  # noqa: E402

CLOSE_THRESHOLD = 25.0  # ground-truth-probe use only in neural mode; see module docstring
MASK_NAME = "W"
N_LABELS = 2
MEMORY_MB = 2048
CLOSE_NN_NAME = "close_nn"

ACTIVITY_RELATIONS: tuple[str, ...] = (
    "both_active", "both_inactive", "both_walking", "mixed_active_walking",
)

# The theory loop's own coverage-acceptance floor (default value;
# not exposed on the CLI).
MIN_NEW_COVERED = 10

# Cost-knob guard: relational mode's `neural_relations` is always
# empty (geometry is precomputed `close`/`far`, not a trained detector), so
# every held-out score is a fixed relational cover and the trained
# placeholder network/candidate weights cannot affect the result -- see
# `run_caviar_star.py`'s own identical guard.
EMPTY_NEURAL_POOL_STEP_CAP = 25


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CAVIAR multi-clause theory loop: sequential "
        "covering over a star-topology candidate pool, relational or "
        "neural vocabulary. Needs CUDA at run time; --help does not.",
    )
    p.add_argument("--mode", required=True, choices=("relational", "neural"))
    p.add_argument(
        "--protocol", default="direct", choices=("direct", "ec"),
        help="'direct': the existing per-timestep holdsAt-style target "
        "(default, unchanged behavior). 'ec': induce a SEPARATE "
        "initiatedAt/terminatedAt theory pair, reconstruct holdsAt by "
        "inertia, and report frame-level F1 alongside the direct "
        "protocol's own theory F1 on the same fold for context.",
    )
    p.add_argument("--pkl", required=True, help="path to caviar_folds.pkl")
    p.add_argument("--fold", default="fold1", help="fold key, e.g. fold1 (default: fold1)")
    p.add_argument("--k", type=int, default=4, help="k-fold holdout folds per select_once call (default: 4)")
    p.add_argument("--seed", type=int, default=7, help="RNG seed, covers the whole run (default: 7)")
    p.add_argument(
        "--steps", type=int, required=True,
        help="training steps per fold, AND per per-clause final-train pass "
        "(neural mode). Relational mode clamps this the same way "
        "run_caviar_star.py does (neural_relations is always empty there).",
    )
    p.add_argument("--hidden", type=int, default=16, help="close_nn MLP hidden width, neural mode only (default: 16)")
    p.add_argument("--max-clauses", type=int, default=4, help="theory_loop.induce_theory's max_clauses (default: 4)")
    p.add_argument("--out", required=True, help="path to write RESULT.json")
    return p.parse_args(argv)


def _require_cuda() -> None:
    """Fail fast, before touching pyxlog, if there is no CUDA device."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "run_caviar_theory.py needs a CUDA device: IlpProgramFactory."
            "compile/put_relation/kfold_select/train_engine_mode (DLPack "
            "over device='cuda' tensors) only run on CUDA -- the same guard "
            "run_caviar_star.py and run_caviar_neural.py enforce. Run this "
            "on the RunPod A40 target, not locally."
        )


def _prepare_out_path(out: str) -> Path:
    """Create --out's parent directory and write a tiny probe file BEFORE
    any expensive work starts (mirrors `run_caviar_star.py`'s/
    `run_caviar_neural.py`'s fail-fast fix)."""
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("started\n")
    return out_path


def _pair_dists(features) -> list[float]:
    """Ground-truth euclidean distance per pair-time, recovered from the
    SAME ``(dx, dy) / FEATURE_SCALE`` features tensor training used --
    undoes `caviar_convert.FEATURE_SCALE`. Used ONLY for the detector probe
    (neural mode), never by training."""
    return [100.0 * ((dx * dx + dy * dy) ** 0.5) for dx, dy in features.tolist()]


# ---------------------------------------------------------------------------
# Relational mode: vocabulary = 4 activities + close + far.
# ---------------------------------------------------------------------------


def _filtered_relation_names(converted: dict) -> list[str]:
    """Every relation name except 'coords_missing' -- mirrors
    `run_caviar_star.py`'s identically-named helper (not imported from
    there: that script's own entrypoint semantics stay untouched,
    so its helpers are re-derived here rather than shared, to preserve
    byte-for-byte equivalent behavior)."""
    return sorted(name for name in converted["relations"] if name != "coords_missing")


def _compile_and_ingest_relational(pyxlog, converted: dict, n_labels: int = N_LABELS):
    """Schema-only compile + `put_relation` of every TRAIN relation except
    'coords_missing' (activities AND close/far) -- called ONCE for the whole
    run; every theory-loop iteration's `select_once` reuses the SAME
    compiled program object (only the fact/label lists shrink between
    iterations, never the relations ingested into it)."""
    from caviar_convert import build_star_schema_source, put_caviar_relations

    relation_names = _filtered_relation_names(converted)
    schema_src = build_star_schema_source(relation_names)
    prog = pyxlog.IlpProgramFactory.compile(schema_src, device=0, memory_mb=MEMORY_MB)

    ingest_converted = dict(
        converted,
        relations={n: p for n, p in converted["relations"].items() if n != "coords_missing"},
    )
    returned_schema = put_caviar_relations(prog, ingest_converted, n_labels=n_labels)
    if returned_schema != schema_src:
        raise RuntimeError(
            "put_caviar_relations's derived schema does not match the "
            "schema the program was compiled with:\n"
            f"compiled:\n{schema_src}\nderived:\n{returned_schema}"
        )
    return prog


def _predict_clause_relational(relations: dict):
    """A pure ``predict_clause(rule, fact) -> bool`` closure: exact
    set-intersection membership over ``relations`` -- the same reading
    `scorer.rule_predictions` gives for a whole split at once, done here
    per-fact so it fits `theory_loop.induce_theory`'s calling convention.
    Precomputes one `set` per relation name up front (not per call)."""
    sets = {name: set(rows) for name, rows in relations.items()}

    def predict(rule, fact):
        left, right = rule
        return fact in sets[left] and fact in sets[right]

    return predict


def _run_relational_theory(pyxlog, torch, kfold_select, args, train, test, wall):
    prog = _compile_and_ingest_relational(pyxlog, train)

    device = torch.device("cuda")
    features = train["features"].to(device)

    def make_network():
        return torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], N_LABELS), torch.nn.Softmax(dim=-1)
        ).to(device)

    # See the module docstring / EMPTY_NEURAL_POOL_STEP_CAP: neural_relations
    # is always empty here, so training steps beyond a small sanity budget
    # cannot change any held-out score.
    steps_requested = args.steps
    steps_effective = min(args.steps, EMPTY_NEURAL_POOL_STEP_CAP)
    steps_clamped = steps_effective != steps_requested
    if steps_clamped:
        print(
            "WARNING: relational mode has no neural candidate -- clamping "
            f"training steps per fold from --steps={steps_requested} to "
            f"{steps_effective} (see run_caviar_star.py's identical guard)."
        )

    iteration_wall: list[float] = []

    def select_once(residual_facts, residual_is_positive):
        t = time.perf_counter()
        sel = kfold_select(
            lambda: prog, MASK_NAME, residual_facts, residual_is_positive,
            make_network, features, neural_relations={}, folds=args.k,
            seed=args.seed, steps=steps_effective, topology="star",
        )
        iteration_wall.append(time.perf_counter() - t)
        return sel

    predict_clause_train = _predict_clause_relational(train["relations"])

    t0 = time.perf_counter()
    theory = induce_theory(
        select_once, predict_clause_train, train["facts"], train["is_positive"],
        max_clauses=args.max_clauses, min_new_covered=MIN_NEW_COVERED,
    )
    wall["theory_loop"] = time.perf_counter() - t0
    wall["theory_loop_per_iteration"] = iteration_wall

    predict_clause_test = _predict_clause_relational(test["relations"])
    scoring = _score_theory(
        theory["clauses"], predict_clause_train, predict_clause_test,
        train["num_pt"], train["is_positive"], test["num_pt"], test["is_positive"],
    )

    baselines = {
        "train": baseline_report(train["relations"], train["is_positive"], train["num_pt"]),
        "test": baseline_report(test["relations"], test["is_positive"], test["num_pt"]),
    }

    return {
        "candidate_vocabulary": {
            "relational": sorted(set(train["relations"]) - {"coords_missing"}),
            "neural": [],
            "excluded": ["coords_missing"],
        },
        "steps_requested": steps_requested,
        "steps_effective": steps_effective,
        "steps_clamped": steps_clamped,
        "theory": _theory_json(theory),
        "scoring": scoring,
        "baselines": baselines,
        "detector_probe": None,
    }


# ---------------------------------------------------------------------------
# Neural mode: vocabulary = 4 activities + close_nn, symmetrized.
# ---------------------------------------------------------------------------


def _compile_and_ingest_neural(pyxlog, converted: dict, n_labels: int = N_LABELS):
    """Schema-only compile + `put_relation` of the 4 activity relations PLUS
    a `close_nn` seed row (never `put_relation`'d -- it has no ground table,
    see `run_caviar_neural.py`'s identically-reasoned helper, which this one
    mirrors rather than imports, for the same byte-equivalence reason as
    `_compile_and_ingest_relational` above)."""
    from caviar_convert import build_star_schema_source, put_caviar_relations

    activity_names = sorted(ACTIVITY_RELATIONS)
    schema_src = build_star_schema_source(activity_names + [CLOSE_NN_NAME])
    prog = pyxlog.IlpProgramFactory.compile(schema_src, device=0, memory_mb=MEMORY_MB)

    missing = [n for n in ACTIVITY_RELATIONS if n not in converted["relations"]]
    if missing:
        raise KeyError(f"convert_split's output is missing {missing}; have {sorted(converted['relations'])}.")
    ingest_converted = dict(
        converted, relations={n: converted["relations"][n] for n in ACTIVITY_RELATIONS},
    )
    returned_schema = put_caviar_relations(prog, ingest_converted, n_labels=n_labels)
    expected = build_star_schema_source(activity_names)
    if returned_schema != expected:
        raise RuntimeError(
            "put_caviar_relations's derived schema does not match the "
            f"activity-only schema:\nexpected:\n{expected}\nderived:\n{returned_schema}"
        )
    return prog


def _build_symmetric_mlp(hidden: int, device):
    """`close_nn`'s network, WRAPPED for exact pair-order invariance:
    ``forward(x) = (base(x) + base(-x)) / 2``.

    WHY. A CAVIAR pair-time's ``(dx, dy)`` input sign depends on which
    person is arbitrarily labeled p1 vs p2 -- an artifact of enumeration
    order, not evidence about the world. An unsymmetrized network trained
    this way has measured pair-swap asymmetry
    (``mean |s(dx,dy) - s(-dx,-dy)|``) as high as 0.364 on real test data:
    over a third of the score's own [0, 1] range moved for no reason but
    which person happened to be listed first. Wrapping the base network this way makes
    ``forward(x) == forward(-x)`` EXACTLY, for every input, by construction
    (algebraic identity: ``base(-(-x)) == base(x)``, so swapping the wrapped
    network's own input just swaps the two summands) -- not an
    approximation trained away, a guarantee that holds before a single
    gradient step. Same output SHAPE as the unwrapped MLP
    (``[num_events, num_labels]``), so it is a drop-in replacement anywhere
    `train_engine_mode`'s `_validated_output` is checked; still fully
    differentiable (both summands are, and addition/division are).

    THIS WRAPPER IS SCOPED TO THIS SCRIPT ONLY -- see the module docstring's
    "SYMMETRIZATION IS SCOPED TO THIS SCRIPT ONLY" paragraph.
    """
    import torch

    base = torch.nn.Sequential(
        torch.nn.Linear(2, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, N_LABELS),
        torch.nn.Softmax(dim=-1),
    ).to(device)

    class _Symmetrized(torch.nn.Module):
        def __init__(self, base_net):
            super().__init__()
            self.base = base_net

        def forward(self, x):
            return (self.base(x) + self.base(-x)) / 2

    return _Symmetrized(base).to(device)


def _induce_neural_theory_for_target(
    torch, kfold_select, train_engine_mode, prog, make_network, features_train,
    neural_relations, activity_sets_train, args, facts, target_labels, wall, wall_key,
):
    """Neural-mode theory induction for ONE target label sequence aligned
    with ``facts`` -- the exact per-clause-retrain mechanism `_run_neural_
    theory` uses for the direct (meeting) target, generalized so `--protocol
    ec` can call it again for ``is_init`` and again for ``is_term``: each
    committed clause still gets its OWN independently trained ``close_nn``
    network (never a network shared across clauses, or across target
    calls -- see the module docstring's "each theory gets its own nets"
    note).

    Returns ``(theory, nets_by_clause_idx)``: ``theory`` is `theory_loop.
    induce_theory`'s own result dict; ``nets_by_clause_idx`` maps each
    COMMITTED clause's position to its trained network, recovered
    positionally from a per-call ``call_log`` (never keyed by the rule's
    VALUE, since a theory may legally commit the same rule twice, each with
    its own net)."""
    call_log: list[tuple] = []
    # In-loop-only mutable slot: `theory_loop.induce_theory` only ever calls
    # `predict_clause` about the rule `select_once` JUST returned THIS
    # iteration, before the next `select_once` call overwrites this slot --
    # so a single current-net cache is correct here, and precomputing the
    # score over ALL of `features_train` once (batched) per iteration is far
    # cheaper than a per-fact forward pass.
    current = {"rule": None, "scores": None}
    iteration_wall: list[float] = []

    def select_once(residual_facts, residual_is_positive):
        t = time.perf_counter()
        selection = kfold_select(
            lambda: prog, MASK_NAME, residual_facts, residual_is_positive,
            make_network, features_train, neural_relations=neural_relations,
            folds=args.k, seed=args.seed, steps=args.steps, topology="star",
        )
        net = None
        if selection.rule is not None:
            # Same init seed every clause (documented choice): each clause's
            # own trained network AND its own later "control" (see below)
            # both start from torch.manual_seed(args.seed) immediately
            # before construction, so the two are directly comparable per
            # clause -- mirrors run_caviar_neural.py's trained-vs-control
            # convention, generalized to "per clause" instead of "the one net".
            torch.manual_seed(args.seed)
            net_to_train = make_network()
            train_result = train_engine_mode(
                prog, MASK_NAME, residual_facts, residual_is_positive,
                net_to_train, features_train, neural_relations=neural_relations,
                steps=args.steps, seed=args.seed, topology="star",
            )
            net = train_result.network
            net.eval()
            with torch.no_grad():
                current["scores"] = net(features_train)[:, 1].tolist()
            current["rule"] = selection.rule
        else:
            current["rule"] = None
            current["scores"] = None
        call_log.append((selection.rule, net))
        iteration_wall.append(time.perf_counter() - t)
        return selection

    def loop_predict_clause(rule, fact):
        left, right = rule
        if fact not in activity_sets_train[left]:
            return False
        if right == CLOSE_NN_NAME:
            pt = fact[0]
            return current["scores"][pt] > 0.5
        return fact in activity_sets_train[right]

    t0 = time.perf_counter()
    theory = induce_theory(
        select_once, loop_predict_clause, facts, target_labels,
        max_clauses=args.max_clauses, min_new_covered=MIN_NEW_COVERED,
    )
    wall[wall_key] = time.perf_counter() - t0
    wall[f"{wall_key}_per_iteration"] = iteration_wall

    # Recover each COMMITTED clause's own net, positionally (see this
    # function's own docstring) -- `clauses[j]` and `nets_per_clause[j]` are
    # THE SAME commit, by construction (both built by iterating
    # `iterations` and `call_log` in lockstep and keeping only "committed"
    # entries).
    nets_per_clause = [
        net for it, (_, net) in zip(theory["iterations"], call_log)
        if it["reason"] == "committed"
    ]
    clauses = theory["clauses"]
    if len(nets_per_clause) != len(clauses):
        raise RuntimeError(
            f"per-clause net recovery drifted for {wall_key}: "
            f"{len(nets_per_clause)} nets for {len(clauses)} committed "
            "clauses -- the call_log/iterations lockstep invariant is broken."
        )
    # Keyed POSITIONALLY (clause index), never by rule value: a theory may
    # legally commit the same rule twice (each with its own net), and a
    # value-keyed dict would silently collapse them.
    nets_by_clause_idx = dict(enumerate(nets_per_clause))
    return theory, nets_by_clause_idx


def _make_final_predict_clause(clauses, nets_by_clause_idx, torch, relations, features):
    """A `predict_clause(rule, fact) -> bool` closure for the FINAL,
    committed theory (as opposed to `_induce_neural_theory_for_target`'s own
    in-loop ``loop_predict_clause``, which only ever needs the CURRENT
    iteration's clause): reads each clause's own trained network's score
    over the given ``features`` tensor (train or test), batched once up
    front rather than per fact."""
    sets = {n: set(relations[n]) for n in ACTIVITY_RELATIONS}
    scores_by_clause_idx = {}
    for idx, net in nets_by_clause_idx.items():
        with torch.no_grad():
            scores_by_clause_idx[idx] = net(features)[:, 1].tolist()

    def predict(rule, fact, _clause_idx=None):
        left, right = rule
        if fact not in sets[left]:
            return False
        if right == CLOSE_NN_NAME:
            idx = (_clause_idx if _clause_idx is not None
                   else clauses.index(rule))
            return scores_by_clause_idx[idx][fact[0]] > 0.5
        return fact in sets[right]

    return predict, scores_by_clause_idx


def _run_neural_theory(pyxlog, torch, kfold_select, args, train, test, wall):
    from pyxlog.ilp.neural_credit import NeuralRelationSpec, train_engine_mode

    prog = _compile_and_ingest_neural(pyxlog, train)

    device = torch.device("cuda")
    features_train = train["features"].to(device)
    features_test = test["features"].to(device)

    def make_network():
        return _build_symmetric_mlp(args.hidden, device)

    neural_relations = {CLOSE_NN_NAME: NeuralRelationSpec(num_rows=train["num_pt"], arity=2)}
    activity_sets_train = {n: set(train["relations"][n]) for n in ACTIVITY_RELATIONS}

    theory, nets_by_clause_idx = _induce_neural_theory_for_target(
        torch, kfold_select, train_engine_mode, prog, make_network, features_train,
        neural_relations, activity_sets_train, args, train["facts"], train["is_positive"],
        wall, "theory_loop",
    )
    clauses = theory["clauses"]

    predict_clause_train, train_scores_by_idx = _make_final_predict_clause(
        clauses, nets_by_clause_idx, torch, train["relations"], features_train
    )
    predict_clause_test, test_scores_by_idx = _make_final_predict_clause(
        clauses, nets_by_clause_idx, torch, test["relations"], features_test
    )

    scoring = _score_theory(
        clauses, predict_clause_train, predict_clause_test,
        train["num_pt"], train["is_positive"], test["num_pt"], test["is_positive"],
    )

    # Soft PR curves: per neural clause, the
    # COVER-GATED score (the network's own P(label=1) where the clause's
    # left literal covers the row, 0.0 elsewhere -- the same cover-gating
    # `enumerate_specs`' star mode already applies), swept over thresholds;
    # and the whole theory's soft union (max over clauses' own gated/crisp
    # scores -- a soft OR, documented below).
    pr_curves = {"clauses": {}, "theory": None}
    activity_sets_test = {n: set(test["relations"][n]) for n in ACTIVITY_RELATIONS}
    for idx, rule in enumerate(clauses):
        left, right = rule
        if right == CLOSE_NN_NAME:
            gated = [
                test_scores_by_idx[idx][pt] if (pt, 1) in activity_sets_test[left] else 0.0
                for pt in range(test["num_pt"])
            ]
        else:
            gated = [
                1.0 if predict_clause_test(rule, (pt, 1)) else 0.0
                for pt in range(test["num_pt"])
            ]
        pr_curves["clauses"][f"clause{idx}:{left}|{right}"] = pr_curve(gated, test["is_positive"])

    if clauses:
        # Soft union: max over each clause's own gated score at that row --
        # a soft OR (documented choice: a hard union would just be
        # theory_predictions itself; this is the SOFT reading that
        # generalizes "one clause's PR curve" to "the whole theory's").
        per_clause_gated = []
        for idx, rule in enumerate(clauses):
            left, right = rule
            if right == CLOSE_NN_NAME:
                per_clause_gated.append([
                    test_scores_by_idx[idx][pt] if (pt, 1) in activity_sets_test[left] else 0.0
                    for pt in range(test["num_pt"])
                ])
            else:
                per_clause_gated.append([
                    1.0 if predict_clause_test(rule, (pt, 1)) else 0.0
                    for pt in range(test["num_pt"])
                ])
        theory_soft = [max(vals) for vals in zip(*per_clause_gated)]
        pr_curves["theory"] = pr_curve(theory_soft, test["is_positive"])

    # Detector probe + polar_spread + pair_swap_asymmetry, per close_nn
    # clause, vs an UNTRAINED control sharing that clause's own init seed
    # (see select_once's docstring comment on seeding).
    train_dists = _pair_dists(train["features"])
    test_dists = _pair_dists(test["features"])
    train_close_rows = {pt for pt, _ in train["relations"]["close"]}
    test_close_rows = {pt for pt, _ in test["relations"]["close"]}
    train_missing = {pt for pt, _ in train["relations"]["coords_missing"]}
    test_missing = {pt for pt, _ in test["relations"]["coords_missing"]}

    clause_detector_evidence = {}
    for clause_idx, net in nets_by_clause_idx.items():
        rule = clauses[clause_idx]
        left, right = rule
        if right != CLOSE_NN_NAME:
            continue
        torch.manual_seed(args.seed)  # SAME init as this clause's own trained net -- see select_once
        control_net = make_network()
        control_net.eval()
        with torch.no_grad():
            train_scores = net(features_train)[:, 1].tolist()
            test_scores = net(features_test)[:, 1].tolist()
            control_scores = control_net(features_test)[:, 1].tolist()

        probe_train = probe_detector(train_scores, train_close_rows, train_dists, exclude_rows=train_missing)
        probe_test = probe_detector(test_scores, test_close_rows, test_dists, exclude_rows=test_missing)
        probe_control = probe_detector(control_scores, test_close_rows, test_dists, exclude_rows=test_missing)

        # polar_spread builds its grid on CPU; the nets live on the CUDA
        # device -- move the probe input to the net's device and the scores
        # back, so both probes work regardless of where the net sits.
        net_device = next(net.parameters()).device

        def score_fn(x, net=net):
            with torch.no_grad():
                return net(x.to(net_device))[:, 1].cpu()

        def control_score_fn(x, control_net=control_net):
            with torch.no_grad():
                return control_net(x.to(net_device))[:, 1].cpu()

        clause_detector_evidence[f"clause{clause_idx}:{left}|{right}"] = {
            "probe": {
                "train": probe_train, "test": probe_test, "control": probe_control,
                "monotone_decay": {
                    "train": monotone_decay_report(probe_train["bins"]),
                    "test": monotone_decay_report(probe_test["bins"]),
                    "control": monotone_decay_report(probe_control["bins"]),
                },
            },
            "polar_spread": {
                "trained": {str(r): v for r, v in polar_spread(score_fn).items()},
                "control": {str(r): v for r, v in polar_spread(control_score_fn).items()},
            },
            "pair_swap_asymmetry": {
                "trained": pair_swap_asymmetry(score_fn, features_train),
                "control": pair_swap_asymmetry(control_score_fn, features_train),
            },
        }

    baselines = {
        "train": baseline_report(
            train["relations"], train["is_positive"], train["num_pt"],
            pairs=list(itertools.combinations(sorted(ACTIVITY_RELATIONS), 2)),
        ),
        "test": baseline_report(
            test["relations"], test["is_positive"], test["num_pt"],
            pairs=list(itertools.combinations(sorted(ACTIVITY_RELATIONS), 2)),
        ),
    }

    return {
        "candidate_vocabulary": {
            "relational": sorted(ACTIVITY_RELATIONS),
            "neural": [CLOSE_NN_NAME],
            "excluded": ["close", "far", "coords_missing"],
        },
        "steps_requested": args.steps,
        "steps_effective": args.steps,
        "steps_clamped": False,
        "theory": _theory_json(theory),
        "scoring": scoring,
        "pr_curves": pr_curves,
        "baselines": baselines,
        "detector_probe": clause_detector_evidence,
        "note": (
            "close/far were never fed to any close_nn training in any form; "
            "each close_nn-tailed clause carries its OWN independently "
            "trained network (see _run_neural_theory's docstring); every "
            "such network is wrapped for exact pair-order invariance "
            "(_build_symmetric_mlp), scoped to this script only."
        ),
    }


# ---------------------------------------------------------------------------
# Shared scoring: theory F1 (train/test) + per-clause marginal contribution.
# ---------------------------------------------------------------------------


def _theory_json(theory: dict) -> dict:
    """`induce_theory`'s result, JSON-safe (rule tuples -> lists)."""
    return {
        "clauses": [list(c) for c in theory["clauses"]],
        "iterations": [
            {**it, "rule": (list(it["rule"]) if it["rule"] is not None else None)}
            for it in theory["iterations"]
        ],
        "stop_reason": theory["stop_reason"],
    }


def _score_theory(
    clauses, predict_clause_train, predict_clause_test,
    num_pt_train, gold_train, num_pt_test, gold_test,
) -> dict:
    """Theory F1 (train + test, union of clauses) plus each clause's
    marginal contribution to TEST F1 (theory F1 minus the F1 of the theory
    with that one clause removed) -- a NEGATIVE marginal is possible and
    reported honestly (a clause can, in principle, hurt precision more than
    it helps recall once combined with the others).

    ``gold_train``/``gold_test`` are passed explicitly (rather than read off
    a ``train``/``test`` dict's own ``"is_positive"``) so this same scoring
    logic is reusable against ANY aligned target -- the direct per-timestep
    label, or the Event-Calculus ``is_init``/``is_term`` targets `run_caviar_
    theory.py`'s ``--protocol ec`` induces separate theories for."""
    train_pred = theory_predictions(clauses, predict_clause_train, num_pt_train)
    test_pred = theory_predictions(clauses, predict_clause_test, num_pt_test)
    theory_prf1 = {
        "train": prf1(train_pred, gold_train),
        "test": prf1(test_pred, gold_test),
    }

    marginal = []
    full_test_f1 = theory_prf1["test"]["f1"]
    for i, clause in enumerate(clauses):
        without = clauses[:i] + clauses[i + 1:]
        pred_without = theory_predictions(without, predict_clause_test, num_pt_test)
        f1_without = prf1(pred_without, gold_test)["f1"]
        marginal.append({
            "clause": list(clause),
            "test_f1_without": f1_without,
            "marginal_test_f1": full_test_f1 - f1_without,
        })

    return {"theory_prf1": theory_prf1, "marginal_contribution": marginal}


# ---------------------------------------------------------------------------
# Event-Calculus protocol: induce a SEPARATE is_init/is_term theory pair,
# reconstruct holdsAt by inertia, score against the same per-timestep gold
# the direct protocol uses -- alongside the direct protocol's own theory F1
# on the SAME fold, for context.
# ---------------------------------------------------------------------------


def _run_relational_ec(pyxlog, torch, kfold_select, args, train, test, ec_train, ec_test, wall):
    """Relational-vocabulary Event-Calculus protocol. Both the init search
    and the term search reuse the SAME compiled program and the SAME
    set-intersection `predict_clause` closures `_run_relational_theory`
    itself would build -- a star rule's cover is a fixed relation
    membership test that does not depend on which target label is being
    explained, so nothing about the candidate pool changes between the
    direct target and either event target.

    The direct protocol's own theory is run in full, once, on this SAME
    fold, and returned under ``"direct_context"`` so the two protocols'
    numbers sit side by side without a second invocation of this script."""
    direct_wall: dict = {}
    t_direct = time.perf_counter()
    direct_result = _run_relational_theory(pyxlog, torch, kfold_select, args, train, test, direct_wall)
    wall["direct_context"] = time.perf_counter() - t_direct
    wall["direct_context_wall_clock_s"] = direct_wall

    prog = _compile_and_ingest_relational(pyxlog, train)

    device = torch.device("cuda")
    features = train["features"].to(device)

    def make_network():
        return torch.nn.Sequential(
            torch.nn.Linear(features.shape[1], N_LABELS), torch.nn.Softmax(dim=-1)
        ).to(device)

    # See EMPTY_NEURAL_POOL_STEP_CAP / _run_relational_theory's identical guard.
    steps_requested = args.steps
    steps_effective = min(args.steps, EMPTY_NEURAL_POOL_STEP_CAP)
    steps_clamped = steps_effective != steps_requested

    predict_clause_train = _predict_clause_relational(train["relations"])
    predict_clause_test = _predict_clause_relational(test["relations"])

    def induce_for(target_train_labels, wall_key):
        iteration_wall: list[float] = []

        def select_once(residual_facts, residual_is_positive):
            t = time.perf_counter()
            sel = kfold_select(
                lambda: prog, MASK_NAME, residual_facts, residual_is_positive,
                make_network, features, neural_relations={}, folds=args.k,
                seed=args.seed, steps=steps_effective, topology="star",
            )
            iteration_wall.append(time.perf_counter() - t)
            return sel

        t0 = time.perf_counter()
        theory = induce_theory(
            select_once, predict_clause_train, train["facts"], target_train_labels,
            max_clauses=args.max_clauses, min_new_covered=MIN_NEW_COVERED,
        )
        wall[wall_key] = time.perf_counter() - t0
        wall[f"{wall_key}_per_iteration"] = iteration_wall
        return theory

    init_theory = induce_for(ec_train["is_init"], "theory_loop_init")
    term_theory = induce_for(ec_train["is_term"], "theory_loop_term")

    init_scoring = _score_theory(
        init_theory["clauses"], predict_clause_train, predict_clause_test,
        train["num_pt"], ec_train["is_init"], test["num_pt"], ec_test["is_init"],
    )
    term_scoring = _score_theory(
        term_theory["clauses"], predict_clause_train, predict_clause_test,
        train["num_pt"], ec_train["is_term"], test["num_pt"], ec_test["is_term"],
    )

    init_pred_test = theory_predictions(init_theory["clauses"], predict_clause_test, test["num_pt"])
    term_pred_test = theory_predictions(term_theory["clauses"], predict_clause_test, test["num_pt"])
    num_windows = test["num_pt"] // ec_test["T"]
    holds_pred_test = reconstruct_holds(init_pred_test, term_pred_test, num_windows, ec_test["T"])
    frame_scoring = frame_f1(holds_pred_test, test["is_positive"])

    return {
        "candidate_vocabulary": {
            "relational": sorted(set(train["relations"]) - {"coords_missing"}),
            "neural": [],
            "excluded": ["coords_missing"],
        },
        "steps_requested": steps_requested,
        "steps_effective": steps_effective,
        "steps_clamped": steps_clamped,
        "ec": {
            "n_init": {"train": ec_train["n_init"], "test": ec_test["n_init"]},
            "n_term": {"train": ec_train["n_term"], "test": ec_test["n_term"]},
            "init_theory": _theory_json(init_theory),
            "term_theory": _theory_json(term_theory),
            "init_scoring": init_scoring,
            "term_scoring": term_scoring,
            "frame_f1": frame_scoring,
        },
        "direct_context": direct_result,
        "detector_probe": None,
    }


def _run_neural_ec(pyxlog, torch, kfold_select, args, train, test, ec_train, ec_test, wall):
    """Neural-vocabulary Event-Calculus protocol. The init search and the
    term search each call `_induce_neural_theory_for_target` independently,
    so EACH theory ends up with its OWN per-clause `close_nn` networks --
    exactly like the direct protocol's single theory does (see that
    function's docstring); nothing is shared between the two searches
    except the compiled program and the candidate vocabulary.

    The direct protocol's own theory is run in full, once, on this SAME
    fold, and returned under ``"direct_context"``, mirroring
    `_run_relational_ec`."""
    from pyxlog.ilp.neural_credit import NeuralRelationSpec, train_engine_mode

    direct_wall: dict = {}
    t_direct = time.perf_counter()
    direct_result = _run_neural_theory(pyxlog, torch, kfold_select, args, train, test, direct_wall)
    wall["direct_context"] = time.perf_counter() - t_direct
    wall["direct_context_wall_clock_s"] = direct_wall

    prog = _compile_and_ingest_neural(pyxlog, train)

    device = torch.device("cuda")
    features_train = train["features"].to(device)
    features_test = test["features"].to(device)

    def make_network():
        return _build_symmetric_mlp(args.hidden, device)

    neural_relations = {CLOSE_NN_NAME: NeuralRelationSpec(num_rows=train["num_pt"], arity=2)}
    activity_sets_train = {n: set(train["relations"][n]) for n in ACTIVITY_RELATIONS}

    init_theory, init_nets = _induce_neural_theory_for_target(
        torch, kfold_select, train_engine_mode, prog, make_network, features_train,
        neural_relations, activity_sets_train, args, train["facts"], ec_train["is_init"],
        wall, "theory_loop_init",
    )
    term_theory, term_nets = _induce_neural_theory_for_target(
        torch, kfold_select, train_engine_mode, prog, make_network, features_train,
        neural_relations, activity_sets_train, args, train["facts"], ec_train["is_term"],
        wall, "theory_loop_term",
    )

    init_predict_train, _ = _make_final_predict_clause(
        init_theory["clauses"], init_nets, torch, train["relations"], features_train
    )
    init_predict_test, _ = _make_final_predict_clause(
        init_theory["clauses"], init_nets, torch, test["relations"], features_test
    )
    term_predict_train, _ = _make_final_predict_clause(
        term_theory["clauses"], term_nets, torch, train["relations"], features_train
    )
    term_predict_test, _ = _make_final_predict_clause(
        term_theory["clauses"], term_nets, torch, test["relations"], features_test
    )

    init_scoring = _score_theory(
        init_theory["clauses"], init_predict_train, init_predict_test,
        train["num_pt"], ec_train["is_init"], test["num_pt"], ec_test["is_init"],
    )
    term_scoring = _score_theory(
        term_theory["clauses"], term_predict_train, term_predict_test,
        train["num_pt"], ec_train["is_term"], test["num_pt"], ec_test["is_term"],
    )

    init_pred_test = theory_predictions(init_theory["clauses"], init_predict_test, test["num_pt"])
    term_pred_test = theory_predictions(term_theory["clauses"], term_predict_test, test["num_pt"])
    num_windows = test["num_pt"] // ec_test["T"]
    holds_pred_test = reconstruct_holds(init_pred_test, term_pred_test, num_windows, ec_test["T"])
    frame_scoring = frame_f1(holds_pred_test, test["is_positive"])

    return {
        "candidate_vocabulary": {
            "relational": sorted(ACTIVITY_RELATIONS),
            "neural": [CLOSE_NN_NAME],
            "excluded": ["close", "far", "coords_missing"],
        },
        "steps_requested": args.steps,
        "steps_effective": args.steps,
        "steps_clamped": False,
        "ec": {
            "n_init": {"train": ec_train["n_init"], "test": ec_test["n_init"]},
            "n_term": {"train": ec_train["n_term"], "test": ec_test["n_term"]},
            "init_theory": _theory_json(init_theory),
            "term_theory": _theory_json(term_theory),
            "init_scoring": init_scoring,
            "term_scoring": term_scoring,
            "frame_f1": frame_scoring,
        },
        "direct_context": direct_result,
        "detector_probe": None,
        "note": (
            "init/term theories each carry their OWN independently trained "
            "close_nn networks per clause (see "
            "_induce_neural_theory_for_target); close/far were never fed to "
            "any close_nn training in any form, in either theory."
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    out_path = _prepare_out_path(args.out)

    _require_cuda()

    import torch
    import pyxlog
    from pyxlog.ilp.neural_credit import kfold_select

    from caviar_convert import convert_split, load_folds

    wall: dict = {}
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

    # `--protocol ec` only: initiatedAt/terminatedAt targets, in the SAME
    # pt indexing `train`/`test` already use (both derived from the SAME
    # `split["train"]`/`split["test"]` datapoints). `--protocol direct`
    # never derives these -- untouched, so its own run is unaffected.
    ec_train = ec_test = None
    if args.protocol == "ec":
        from caviar_convert import derive_ec_targets

        ec_train = derive_ec_targets(split["train"])
        ec_test = derive_ec_targets(split["test"])

    t1 = time.perf_counter()
    if args.mode == "relational":
        if args.protocol == "direct":
            mode_result = _run_relational_theory(pyxlog, torch, kfold_select, args, train, test, wall)
        else:
            mode_result = _run_relational_ec(
                pyxlog, torch, kfold_select, args, train, test, ec_train, ec_test, wall
            )
    else:
        if args.protocol == "direct":
            mode_result = _run_neural_theory(pyxlog, torch, kfold_select, args, train, test, wall)
        else:
            mode_result = _run_neural_ec(
                pyxlog, torch, kfold_select, args, train, test, ec_train, ec_test, wall
            )
    wall["mode_total"] = time.perf_counter() - t1
    wall["total"] = time.perf_counter() - t0

    result = {
        "pkl": args.pkl,
        "fold": args.fold,
        "mode": args.mode,
        "protocol": args.protocol,
        "close_threshold": CLOSE_THRESHOLD,
        "k": args.k,
        "seed": args.seed,
        "hidden": args.hidden,
        "max_clauses": args.max_clauses,
        "min_new_covered": MIN_NEW_COVERED,
        "num_pt": {"train": train["num_pt"], "test": test["num_pt"]},
        "n_pos": {
            "train": int(sum(train["is_positive"])),
            "test": int(sum(test["is_positive"])),
        },
        "wall_clock_s": wall,
        **mode_result,
    }

    out_path.write_text(json.dumps(result, indent=2))

    print(
        f"CAVIAR theory loop: mode={args.mode} pkl={args.pkl} fold={args.fold} "
        f"k={args.k} seed={args.seed} max_clauses={args.max_clauses}"
    )
    if args.protocol == "direct":
        print(f"  theory clauses: {result['theory']['clauses']}")
        print(f"  stop_reason: {result['theory']['stop_reason']}")
        print(f"  train prf1: {result['scoring']['theory_prf1']['train']}")
        print(f"  test  prf1: {result['scoring']['theory_prf1']['test']}")
        for m in result["scoring"]["marginal_contribution"]:
            print(f"    clause {m['clause']}: marginal test F1 = {m['marginal_test_f1']:.4f}")
        print(f"  wall clock total: {wall['total']:.2f}s (theory_loop: {wall['theory_loop']:.2f}s)")
    else:
        ec = result["ec"]
        print(f"  init theory clauses: {ec['init_theory']['clauses']}")
        print(f"  term theory clauses: {ec['term_theory']['clauses']}")
        print(f"  n_init: {ec['n_init']}  n_term: {ec['n_term']}")
        print(f"  frame F1 (holdsAt reconstruction, test): {ec['frame_f1']}")
        print(
            "  direct-protocol theory F1 on this fold (context, test): "
            f"{result['direct_context']['scoring']['theory_prf1']['test']}"
        )
        print(f"  wall clock total: {wall['total']:.2f}s")
    print(f"  wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
