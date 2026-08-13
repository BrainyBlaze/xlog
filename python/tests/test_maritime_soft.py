"""Unit tests for the pre-registered maritime soft-credit column
(`docs/experiments/maritime/PREREG_SOFT.md`): body enumeration + coverage
matrix, the noisy-OR weight trainer, the `sustained_240` relation, the
duration-vocabulary ceiling probe and the CV-runner columns. CPU only, no
real archives — synthetic fixtures throughout, following
`test_maritime_convert.py` / `test_maritime_cv.py`."""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "maritime_woled"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "examples", "caviar_woled"))


# ---------------------------------------------------------------------------
# Task 2: enumerate_bodies + coverage_matrix (PREREG_SOFT.md section (b):
# the pool is every conjunction of 1..3 literals over the vocabulary)
# ---------------------------------------------------------------------------

BASELINE_VOCABULARY = [
    "proximity", "far", "both_lowspeed", "both_stopped_far",
    "both_low_or_stopped", "either_low_or_stopped", "any_near_ports",
    "both_open_sea", "became_far", "became_proximate", "any_slow_ended",
]


def test_enumerate_bodies_counts_11_choose_1_2_3():
    from enumerate_bodies import enumerate_bodies

    bodies = enumerate_bodies(BASELINE_VOCABULARY, max_literals=3)
    # C(11,1) + C(11,2) + C(11,3) = 11 + 55 + 165 = 231
    assert len(bodies) == 231
    assert sum(1 for b in bodies if len(b) == 1) == 11
    assert sum(1 for b in bodies if len(b) == 2) == 55
    assert sum(1 for b in bodies if len(b) == 3) == 165
    # no duplicates
    assert len(set(bodies)) == 231


def test_enumerate_bodies_combinations_not_permutations():
    from enumerate_bodies import enumerate_bodies

    bodies = enumerate_bodies(["b", "a"], max_literals=2)
    assert ("a", "b") in bodies
    assert ("b", "a") not in bodies
    # every body is a sorted tuple, no relation repeated inside a body
    for b in bodies:
        assert tuple(sorted(b)) == b
        assert len(set(b)) == len(b)


def test_coverage_matrix_rows_are_intersections():
    torch = pytest.importorskip("torch")
    from enumerate_bodies import coverage_matrix, enumerate_bodies

    # hand fixture: 3 relations x 6 pt
    relations = {"a": {0, 1, 2}, "b": {1, 2, 5}, "c": {2, 3}}
    bodies = enumerate_bodies(["a", "b", "c"], max_literals=3)
    assert bodies == [
        ("a",), ("b",), ("c",),
        ("a", "b"), ("a", "c"), ("b", "c"),
        ("a", "b", "c"),
    ]
    m = coverage_matrix(bodies, relations, n_pt=6)
    assert m.dtype == torch.bool
    assert m.shape == (7, 6)
    # hand-computed rows (True where pt is in EVERY relation of the body)
    assert m[0].tolist() == [True, True, True, False, False, False]     # a
    assert m[1].tolist() == [False, True, True, False, False, True]     # b
    assert m[2].tolist() == [False, False, True, True, False, False]    # c
    assert m[3].tolist() == [False, True, True, False, False, False]    # a&b
    assert m[4].tolist() == [False, False, True, False, False, False]   # a&c
    assert m[5].tolist() == [False, False, True, False, False, False]   # b&c
    assert m[6].tolist() == [False, False, True, False, False, False]   # a&b&c


def test_coverage_matrix_empty_intersection_is_a_zero_row():
    pytest.importorskip("torch")
    from enumerate_bodies import coverage_matrix

    # 'a' and 'b' never co-fire: the (a, b) row is all-False, legally
    m = coverage_matrix([("a", "b")], {"a": {0, 1}, "b": {2}}, n_pt=4)
    assert m.shape == (1, 4)
    assert m.sum().item() == 0


# ---------------------------------------------------------------------------
# Task 3: soft_weights — the noisy-OR BCE trainer (PREREG_SOFT.md section
# (b): score(pt) = 1 - PROD_c (1 - sigmoid(w_c) * cover_c(pt)), BCE, Adam,
# steps=300, lr=0.05, seed=7, deterministic CPU, init w = -2.0)
# ---------------------------------------------------------------------------


def test_soft_scores_hand_computed_noisy_or():
    torch = pytest.importorskip("torch")
    import math

    from soft_weights import soft_scores

    cover = torch.tensor([[True, True, False], [False, True, False]])
    # weights are LOGITS: pick them so sigmoid(w) = [0.6, 0.4] exactly
    weights = torch.tensor([math.log(0.6 / 0.4), math.log(0.4 / 0.6)])
    scores = soft_scores(cover, weights)
    # pt0: 1 - (1 - 0.6)            = 0.6
    # pt1: 1 - (1 - 0.6)(1 - 0.4)   = 1 - 0.4*0.6 = 0.76
    # pt2: nothing covers it        = 0.0
    assert scores.tolist() == pytest.approx([0.6, 0.76, 0.0])


def _planted_cover_and_labels(torch, n_pt=40):
    """Body 0 fires exactly on the positives (t % 2 == 0); body 1 is noise
    (t % 5 == 0: covers some positives AND some negatives)."""
    cover = torch.zeros((2, n_pt), dtype=torch.bool)
    y = torch.zeros(n_pt, dtype=torch.bool)
    for t in range(n_pt):
        if t % 2 == 0:
            cover[0, t] = True
            y[t] = True
        if t % 5 == 0:
            cover[1, t] = True
    return cover, y


def test_train_soft_weights_recovers_planted_body_and_mutes_noise():
    torch = pytest.importorskip("torch")
    from soft_weights import soft_scores, train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    weights = train_soft_weights(cover, y)
    sig = torch.sigmoid(weights)
    assert sig[0].item() > 0.9, "the planted body must be turned on"
    assert sig[1].item() < 0.1, "the noise body must be turned off"
    # the trained scores separate the classes at the 0.5 threshold
    scores = soft_scores(cover, weights)
    assert ((scores > 0.5) == y).all()


def test_train_soft_weights_bce_falls_with_training():
    # The credit_nll parity pin, measured on the shared semantics: the BCE
    # of the noisy-OR scores against the labels falls as training proceeds
    # (checkpoints at steps 0 < 10 < 300; deterministic restarts make the
    # 10-step run a prefix of the 300-step run).
    torch = pytest.importorskip("torch")
    from soft_weights import soft_scores, train_soft_weights

    cover, y = _planted_cover_and_labels(torch)

    def bce(weights):
        scores = soft_scores(cover, weights).clamp(1e-7, 1 - 1e-7)
        return torch.nn.functional.binary_cross_entropy(scores, y.float()).item()

    bce_init = bce(torch.full((2,), -2.0))
    bce_10 = bce(train_soft_weights(cover, y, steps=10))
    bce_300 = bce(train_soft_weights(cover, y, steps=300))
    assert bce_300 < bce_10 < bce_init


def test_train_soft_weights_is_bitwise_deterministic():
    torch = pytest.importorskip("torch")
    from soft_weights import train_soft_weights

    cover, y = _planted_cover_and_labels(torch)
    w1 = train_soft_weights(cover, y, steps=50, seed=7)
    w2 = train_soft_weights(cover, y, steps=50, seed=7)
    assert torch.equal(w1, w2), "same seed must reproduce bitwise-equal weights"


# ---------------------------------------------------------------------------
# sustained_240 is computed on continuous interval intersections
# (PREREG_SOFT.md section (c)), never on the sparse point grid.
# ---------------------------------------------------------------------------

import io
import tarfile
import zipfile


def _archives(tmp_path, hle_lines, lle_lines, stem="soft"):
    tar_p = tmp_path / f"{stem}.tar.gz"
    data = "\n".join(hle_lines).encode()
    with tarfile.open(tar_p, "w:gz") as tf:
        info = tarfile.TarInfo("Maritime Composite Events/CEs/recognised_CEs.csv")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    zip_p = tmp_path / f"{stem}.zip"
    with zipfile.ZipFile(zip_p, "w") as z:
        z.writestr("brest_critical.csv", "\n".join(lle_lines))
    return str(tar_p), str(zip_p)


def test_sustained_240_long_component_uses_half_open_membership(tmp_path):
    from maritime_convert import convert

    tar_p, zip_p = _archives(tmp_path, [
        "lowSpeed|A| |true|900|2200",
        "lowSpeed|B| |true|900|2200",
    ], [
        "proximity|1|1000|1300|true|A|B",   # intersection component: 300 s
        "proximity|1|2000|2100|true|A|B",   # intersection component: 100 s
    ])
    conv = convert(tar_p, zip_p, extra_relations=("sustained_240",))
    times = conv["pt_time"]
    assert times == [900, 1000, 1300, 2000, 2100, 2200]
    # Only pts inside the 300 s half-open component [1000, 1300) receive
    # the relation; its right endpoint and the 100 s component do not.
    assert conv["relations"]["sustained_240"] == [times.index(1000)]


def test_sustained_240_exact_duration_component_uses_half_open_membership(tmp_path):
    from maritime_convert import convert

    tar_p, zip_p = _archives(tmp_path, [
        "lowSpeed|A| |true|900|1300",
        "lowSpeed|B| |true|900|1300",
    ], [
        "proximity|1|1000|1240|true|A|B",   # et - st == 240 exactly: the tie
    ])
    conv = convert(tar_p, zip_p, extra_relations=("sustained_240",))
    times = conv["pt_time"]
    sustained = set(conv["relations"]["sustained_240"])
    assert sustained == {times.index(1000)}
    for defining_relation in ("proximity", "both_low_or_stopped", "both_open_sea"):
        assert sustained <= set(conv["relations"][defining_relation])


def test_sustained_240_single_pt_inside_long_intersection_gets_it(tmp_path):
    # THE pt-grid trap the interval semantics exists for: the grid holds a
    # single body-covered pt (10000; 20000 is excluded by the half-open
    # cover), so any run-length reading over the pt grid sees a 0 s run and
    # kills recall — while the CONTINUOUS intersection is 10,000 s long.
    from maritime_convert import convert

    tar_p, zip_p = _archives(tmp_path, [
        "lowSpeed|A| |true|9000|21000",
        "lowSpeed|B| |true|9000|21000",
    ], [
        "proximity|1|10000|20000|true|A|B",
    ])
    conv = convert(tar_p, zip_p, extra_relations=("sustained_240",))
    times = conv["pt_time"]
    assert times == [9000, 10000, 20000, 21000]
    body_pts = (
        set(conv["relations"]["proximity"])
        & set(conv["relations"]["both_low_or_stopped"])
        & set(conv["relations"]["both_open_sea"])
    )
    assert body_pts == {times.index(10000)}, "the grid must hold exactly one covered pt"
    assert times.index(10000) in conv["relations"]["sustained_240"]


def test_convert_default_has_no_sustained_key_and_matches_main_snapshot(tmp_path):
    # Default-path compatibility: convert() without the flag returns the
    # selected snapshot values from the pre-change (main 332a6837) code on the
    # test_maritime_convert fixture — snapshot pinned below by value — and
    # deep-equals an explicit extra_relations=().
    from maritime_convert import convert

    tar_p, zip_p = _archives(tmp_path, [
        "rendezVous|B|A|true|1000|2000",
        "lowSpeed|A| |true|900|2100",
        "lowSpeed|B| |true|900|1500",
        "stopped|B| |farFromPorts|1500|2100",
        "withinArea|A|nearPorts|true|5000|6000",
        "garbage line without pipes",
    ], [
        "proximity|2200|900|2200|true|B|A",
        "proximity|2500|2200|2500|true|A|B",
        "coord|900|900|A|-4.3|48.1",
        "proximity|9|900|2200|true|A",
    ])
    conv = convert(tar_p, zip_p)
    assert conv == convert(tar_p, zip_p, extra_relations=())
    assert "sustained_240" not in conv["relations"]
    # main-code snapshot, captured on 332a6837 before this feature existed:
    assert conv["pt_time"] == [900, 1000, 1500, 2000, 2100, 2500]
    assert conv["segments"] == [(0, 6)]
    assert conv["is_positive"] == [False, True, True, False, False, False]
    assert conv["relations"] == {
        "proximity": [0, 1, 2, 3, 4],
        "far": [5],
        "both_lowspeed": [0, 1],
        "both_stopped_far": [],
        "both_low_or_stopped": [0, 1, 2, 3],
        "either_low_or_stopped": [0, 1, 2, 3],
        "any_near_ports": [],
        "both_open_sea": [0, 1, 2, 3, 4, 5],
        "became_far": [5],
        "became_proximate": [],
        "any_slow_ended": [4],
    }


def test_convert_rejects_unknown_extra_relation(tmp_path):
    from maritime_convert import convert

    tar_p, zip_p = _archives(tmp_path, ["lowSpeed|A| |true|0|10"], [
        "proximity|1|0|10|true|A|B",
    ])
    with pytest.raises(ValueError):
        convert(tar_p, zip_p, extra_relations=("bogus",))


# ---------------------------------------------------------------------------
# ceiling_probe --vocab duration derives the pre-registered
# duration-vocabulary ceiling before any CV run.
# ---------------------------------------------------------------------------


def _duration_probe_archives(tmp_path):
    """Hand-computed fixture for the duration relation plus one gold interval.
    pts: 900/1000/1300/2000/2100/2200. The base definitional body covers
    {1000, 2000} (half-open); sustained_240 covers {1000} (the 300 s
    component is also half-open); gold rendezVous [1000, 1300) covers {1000}. So:
    base body tp=1 fp=1 fn=0; base ∧ sustained_240 tp=1 fp=0 fn=0."""
    return _archives(tmp_path, [
        "rendezVous|A|B|true|1000|1300",
        "lowSpeed|A| |true|900|2200",
        "lowSpeed|B| |true|900|2200",
    ], [
        "proximity|1|1000|1300|true|A|B",
        "proximity|1|2000|2100|true|A|B",
    ], stem="probe")


def test_ceiling_probe_duration_vocab_adds_pointwise_duration_block(tmp_path):
    import json

    import ceiling_probe

    tar_p, zip_p = _duration_probe_archives(tmp_path)
    out_path = tmp_path / "probe_duration.json"
    rc = ceiling_probe.main([
        "--tar", tar_p, "--zip", zip_p, "--out", str(out_path),
        "--vocab", "duration",
    ])
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    base = report["pointwise"]
    assert (base["tp"], base["fp"], base["fn"]) == (1, 1, 0)
    dur = report["pointwise_duration"]
    assert dur["body"] == [
        "proximity", "both_low_or_stopped", "both_open_sea", "sustained_240",
    ]
    assert (dur["tp"], dur["fp"], dur["fn"]) == (1, 0, 0)
    assert dur["precision"] == 1.0
    assert dur["recall"] == 1.0
    assert dur["f1"] == 1.0


def test_ceiling_probe_default_vocab_has_no_duration_block(tmp_path):
    import json

    import ceiling_probe

    tar_p, zip_p = _duration_probe_archives(tmp_path)
    out_path = tmp_path / "probe_base.json"
    rc = ceiling_probe.main(["--tar", tar_p, "--zip", zip_p, "--out", str(out_path)])
    assert rc == 0
    report = json.loads(out_path.read_text(encoding="utf-8"))
    assert "pointwise_duration" not in report
    assert (report["pointwise"]["tp"], report["pointwise"]["fp"]) == (1, 1)


# ---------------------------------------------------------------------------
# The CV runner's pre-registered columns use --column {hard,soft} and
# --vocab {base,duration}; hard/base preserves the selected baseline contract.
# ---------------------------------------------------------------------------


def _cv_archives(tmp_path, n_pos=6, n_neg=6):
    """Planted-rule mini-corpus: per positive pair the pts are
    900/1000/2000/2100/2200, gold rendezVous [1000, 2000) covers exactly
    {1000}, and `both_stopped_far` (stopped=farFromPorts 1000-2000 for both
    vessels) covers exactly {1000} too — a perfect 1-literal soft body.
    Negative pairs carry proximity only."""
    hle, lle = [], []
    for i in range(n_pos):
        a, b = f"P{i:02d}a", f"P{i:02d}b"
        hle += [
            f"rendezVous|{a}|{b}|true|1000|2000",
            f"lowSpeed|{a}| |true|900|2100",
            f"lowSpeed|{b}| |true|900|2100",
            f"stopped|{a}| |farFromPorts|1000|2000",
            f"stopped|{b}| |farFromPorts|1000|2000",
        ]
        lle.append(f"proximity|2200|900|2200|true|{a}|{b}")
    for j in range(n_neg):
        a, b = f"N{j:02d}a", f"N{j:02d}b"
        lle.append(f"proximity|2200|900|2200|true|{a}|{b}")
    return _archives(tmp_path, hle, lle, stem="cv")


CV_ARGS = ["--smoke", "--skip-verify", "--folds", "3"]


def _run_cv(tmp_path, extra_args, out_name="out.json"):
    import json

    import run_maritime_cv

    tar_p, zip_p = _cv_archives(tmp_path)
    out = tmp_path / out_name
    rc = run_maritime_cv.main(
        ["--tar", tar_p, "--zip", zip_p, "--out", str(out)] + CV_ARGS + extra_args
    )
    assert rc == 0
    return json.loads(out.read_text(encoding="utf-8"))


def test_cv_hard_base_defaults_preserve_selected_baseline_contract(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(tmp_path, [])
    # snapshot of the pre-change (main 332a6837) runner on this fixture:
    assert result["micro"]["point"] == {
        "tp": 6, "fp": 0, "fn": 0, "precision": 1.0, "recall": 1.0, "f1": 1.0,
    }
    assert [f["clauses"] for f in result["folds"]] == [
        [["both_low_or_stopped", "both_stopped_far"]],
        [["both_low_or_stopped", "both_stopped_far"]],
        [["both_low_or_stopped", "both_stopped_far"]],
    ]
    assert result["per_fold_point_f1"]["values"] == [1.0, 1.0, 1.0]
    assert result["fold_of_pair"] == [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
    assert result["candidate_vocabulary"] == sorted(BASELINE_VOCABULARY)
    assert "unverified or synthetic invocation" in (
        result["vocabulary_ceiling_note"]
    )
    assert "0.6599" not in result["vocabulary_ceiling_note"]
    # The output contract is additive rather than byte-identical: the selected
    # baseline fields above are pinned, and the result records these params.
    assert result["params"]["column"] == "hard"
    assert result["params"]["vocab"] == "base"


def test_cv_soft_column_recovers_planted_rule(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(tmp_path, ["--column", "soft"], out_name="soft.json")
    assert result["params"]["column"] == "soft"
    assert result["micro"]["point"]["f1"] == pytest.approx(1.0)
    for record in result["folds"]:
        assert record["column"] == "soft"
        assert record["vocab"] == "base"
        # C(11,1) + C(11,2) + C(11,3) over the 11-relation vocabulary
        assert record["n_bodies_pool"] == 231
        assert record["n_bodies_gated"] >= 1
        assert record["soft_params"] == {
            "steps": 300, "lr": 0.05, "seed": 7, "threshold": 0.5,
        }
        assert record["scoring"]["point"]["f1"] == pytest.approx(1.0)
        # the top-weighted body is (a conjunction containing) the planted
        # one. Its weight is far below 1.0 BY DESIGN: every conjunction of
        # both_stopped_far with a superset relation has the identical cover,
        # so noisy-OR shares the credit evenly across those bodies (their
        # gradients are identical) — the COMBINED score is what crosses 0.5.
        top_body, top_weight = next(iter(record["weights_top10"].items()))
        assert "both_stopped_far" in top_body.split("&")
        assert top_weight > 0.3
        # every top-10 body carries the planted literal (nothing else
        # separates the classes on this fixture)
        assert all("both_stopped_far" in b.split("&") for b in record["weights_top10"])
        # provenance: the gate is the same permutation-null construction
        assert record["min_fit"] == record["null_summary"]["threshold"]


def test_cv_duration_vocab_reaches_the_converter(tmp_path):
    pytest.importorskip("torch")
    result = _run_cv(tmp_path, ["--vocab", "duration"], out_name="dur.json")
    assert result["params"]["vocab"] == "duration"
    assert "sustained_240" in result["candidate_vocabulary"]
    assert len(result["candidate_vocabulary"]) == 12
    assert "unverified or synthetic invocation" in (
        result["vocabulary_ceiling_note"]
    )
    assert "0.9969" not in result["vocabulary_ceiling_note"]


def test_cv_verified_vocabulary_notes_name_the_pinned_corpus_reference():
    from run_maritime_cv import _vocabulary_ceiling_note

    base = _vocabulary_ceiling_note("base", pinned_corpus_verified=True)
    assert "base-vocabulary definitional-body operating point F1 0.6599" in base
    duration = _vocabulary_ceiling_note("duration", pinned_corpus_verified=True)
    assert "duration-vocabulary definitional-body canon F1 0.9969" in duration


def test_cv_soft_column_two_runs_identical(tmp_path):
    pytest.importorskip("torch")

    def strip_walls(result):
        result.pop("convert_wall_s", None)
        for record in result["folds"]:
            record.pop("wall_s", None)
        return result

    a = strip_walls(_run_cv(tmp_path, ["--column", "soft"], out_name="a.json"))
    b = strip_walls(_run_cv(tmp_path, ["--column", "soft"], out_name="b.json"))
    assert a == b
