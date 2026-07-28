"""Independent, from-scratch cross-check of the annotation-alignment claims
`caviar_continuous.py`'s own module docstring makes ("THE ANNOTATION OFFSET
-- VERIFIED, NOT ASSUMED" section): recomputes every number quoted there
(annotation-timestamp grid coverage, per-pair init/term transition counts,
and the T-40 falsification) from the RAW `caviar-train.json`/
`caviar-test.json` files using its OWN regex parse, OWN segmentation, and
OWN transition-counting logic below -- this module never imports
`load_continuous`/`convert_continuous`/`derive_ec_targets_continuous` from
`caviar_continuous.py`, so a bug shared between the two implementations
cannot hide a real data-alignment mistake behind agreement between them.

See `docs/experiments/caviar/alignment_evidence.json` for the recorded
counts alongside the data md5s and a one-line provenance note (referenced
from `docs/experiments/caviar/README.md`'s loader section).

DATA-GATED: every test below is skipped when the real files are not present
locally -- they are not shipped in this repo (see `caviar_continuous.py`'s
own docstring and `docs/experiments/caviar/README.md`'s "Data provenance"
table for where to get them and their md5s). Located via the
``CAVIAR_CONTINUOUS_DIR`` environment variable; if unset, this module
defaults to ``../caviar_data`` relative to the CURRENT WORKING DIRECTORY the
test suite is invoked from (a documented, honest guess -- NOT this
repository's root and NOT this test file's own directory: this dataset is
never checked into the repo, so there is no in-repo path to default to; a
real run sets ``CAVIAR_CONTINUOUS_DIR`` explicitly to wherever the two files
actually live).
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

DEFAULT_DATA_DIR = "../caviar_data"
DATA_DIR = Path(os.environ.get("CAVIAR_CONTINUOUS_DIR", DEFAULT_DATA_DIR))
TRAIN_PATH = DATA_DIR / "caviar-train.json"
TEST_PATH = DATA_DIR / "caviar-test.json"

pytestmark = pytest.mark.skipif(
    not (TRAIN_PATH.is_file() and TEST_PATH.is_file()),
    reason=(
        "real caviar-train.json/caviar-test.json not found under "
        f"{DATA_DIR!s} -- set CAVIAR_CONTINUOUS_DIR to the directory "
        "holding both real files (see caviar_continuous.py's own "
        "data-provenance note and docs/experiments/caviar/README.md's "
        "'Data provenance' table for the download location and md5s); "
        "these files are not shipped in this repo."
    ),
)

# Own, independent regex pass -- deliberately NOT imported from
# caviar_continuous.py (see module docstring above): a trailing
# ",<digits>)" at the end of any narrative/annotation atom string is that
# atom's own trailing time argument, regardless of which predicate it names.
_TRAILING_TIME_RE = re.compile(r",(\d+)\)$")
_MEETING_RE = re.compile(r"holdsAt\(meeting\((id\d+),(id\d+)\),(\d+)\)")


def _person_num(pid: str) -> int:
    return int(pid[2:])


def _canonical_pair(p1: str, p2: str) -> tuple[str, str]:
    """The pair's orderless identity: lower-numbered id first -- collapses
    the annotation's own symmetric `(p1,p2)`/`(p2,p1)` atom pairs onto one
    key, independently of how `caviar_continuous._canonical_pair` does it."""
    return tuple(sorted((p1, p2), key=_person_num))


def _load_raw(path: Path) -> tuple[set[str], set[str]]:
    """Fold every narrative/annotation atom string, across every line of the
    file, into two global sets -- own re-derivation of the dedup strategy
    `caviar_continuous.load_continuous`'s docstring documents (exact
    document duplication, shared chunk-boundary frames): both quirks are
    handled for free by using a set at all, with no per-document
    bookkeeping."""
    narrative_atoms: set[str] = set()
    annotation_atoms: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            narrative_atoms.update(doc["narrative"])
            annotation_atoms.update(doc["annotation"])
    return narrative_atoms, annotation_atoms


def _trailing_times(atoms: set[str]) -> set[int]:
    out: set[int] = set()
    for atom in atoms:
        m = _TRAILING_TIME_RE.search(atom)
        if m:
            out.add(int(m.group(1)))
    return out


def _meeting_set(annotation_atoms: set[str]) -> set[tuple[str, str, int]]:
    out: set[tuple[str, str, int]] = set()
    for atom in annotation_atoms:
        m = _MEETING_RE.match(atom)
        if m:
            p1, p2, t = m.groups()
            out.add((*_canonical_pair(p1, p2), int(t)))
    return out


def _segments(ts_sorted: list[int]) -> list[list[int]]:
    """Split a sorted narrative-timestamp list wherever a consecutive gap is
    not exactly 40ms -- own re-derivation of the same video-splice rule
    `caviar_continuous.load_continuous`'s docstring documents (a jump other
    than 40ms is a splice point between two originally-distinct CAVIAR
    videos concatenated onto one timeline)."""
    segs: list[list[int]] = [[ts_sorted[0]]]
    for prev, nxt in zip(ts_sorted, ts_sorted[1:]):
        if nxt - prev == 40:
            segs[-1].append(nxt)
        else:
            segs.append([nxt])
    return segs


def _per_pair_transitions(
    meeting: set[tuple[str, str, int]], segments: list[list[int]], shift: int = 0,
) -> tuple[dict[tuple[str, str], tuple[int, int]], int, int]:
    """Per (canonical, UNORDERED) pair, per-video-segment init/term
    transition counts of the (optionally time-shifted) meeting signal, over
    the segment's OWN FULL 40ms grid -- every narrative frame in the
    segment, not just the frames where the pair happens to be co-visible
    (the "without regard to co-visibility" raw reading `caviar_continuous`'s
    module docstring describes and reconciles its own co-visibility-scoped
    ``derive_ec_targets_continuous`` counts against).

    A segment's very first frame is compared against an IMPLICIT "not
    meeting" predecessor (``prev`` starts ``False``, never skipped) -- this
    is what a pair already meeting at the first frame of a NEW video
    (verified on the real test file: (id2,id6)'s segment begins exactly one
    frame before their first annotated meeting frame) needs to be counted as
    a real initiation under this raw reading, and is what makes this
    function's totals reproduce the known counts below exactly.

    ``shift``: query ``(t + shift)`` in the pair's own true-timestamp set
    when evaluating frame ``t``. ``shift=0`` is the as-is reading
    `caviar_continuous.py` actually uses (``holdsAt(meeting(p1,p2),T)``'s own
    ``T`` taken literally, no offset). ``shift=40`` is the "T -> T-40"
    falsification probe the module docstring's decisive test describes:
    reinterpreting each annotated frame as describing the PRIOR frame
    instead.

    Returns ``(per_pair, total_init, total_term)``, both totals summed over
    every UNORDERED pair -- the published "per-ordered-pair" totals this
    dataset's annotation logs (both `(p1,p2)` and `(p2,p1)` atoms, always
    with identical timestamps, since the source data states meetings
    symmetrically) are exactly DOUBLE these unordered sums."""
    pairs = sorted({(p1, p2) for (p1, p2, _t) in meeting})
    per_pair: dict[tuple[str, str], tuple[int, int]] = {}
    total_init = 0
    total_term = 0
    for pair in pairs:
        true_ts = {t for (p1, p2, t) in meeting if (p1, p2) == pair}
        n_init = 0
        n_term = 0
        for seg in segments:
            prev = False
            for t in seg:
                pos = (t + shift) in true_ts
                if pos and not prev:
                    n_init += 1
                if (not pos) and prev:
                    n_term += 1
                prev = pos
        per_pair[pair] = (n_init, n_term)
        total_init += n_init
        total_term += n_term
    return per_pair, total_init, total_term


@pytest.fixture(scope="module")
def train_raw():
    return _load_raw(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_raw():
    return _load_raw(TEST_PATH)


# ---------------------------------------------------------------------------
# Annotation-timestamp grid coverage: every annotated timestamp already sits
# on the SAME 40ms narrative grid -- 0 of 1344 (train) / 489 (test) absent.
# ---------------------------------------------------------------------------


def test_train_annotation_timestamps_all_on_narrative_grid(train_raw):
    narrative_atoms, annotation_atoms = train_raw
    narrative_ts = _trailing_times(narrative_atoms)
    annotation_ts = _trailing_times(annotation_atoms)
    assert len(annotation_ts) == 1344
    assert annotation_ts - narrative_ts == set()


def test_test_annotation_timestamps_all_on_narrative_grid(test_raw):
    narrative_atoms, annotation_atoms = test_raw
    narrative_ts = _trailing_times(narrative_atoms)
    annotation_ts = _trailing_times(annotation_atoms)
    assert len(annotation_ts) == 489
    assert annotation_ts - narrative_ts == set()


# ---------------------------------------------------------------------------
# Per-pair init/term transition breakdown, no shift -- the "T taken as-is"
# reading, reconciled against the known per-ordered-pair counts.
# ---------------------------------------------------------------------------


def test_train_per_pair_transition_breakdown_no_shift(train_raw):
    narrative_atoms, annotation_atoms = train_raw
    narrative_ts = sorted(_trailing_times(narrative_atoms))
    segs = _segments(narrative_ts)
    meeting = _meeting_set(annotation_atoms)

    per_pair, total_init, total_term = _per_pair_transitions(meeting, segs, shift=0)

    assert per_pair == {
        ("id0", "id1"): (3, 3),
        ("id1", "id2"): (5, 5),
        ("id1", "id3"): (1, 1),
        ("id2", "id6"): (1, 1),
        ("id5", "id6"): (1, 1),
    }
    # Per-ORDERED-pair totals (both symmetric annotation directions) --
    # exactly double the unordered sums above; no boundary truncation on
    # train (every meeting interval both starts and ends inside its video).
    assert 2 * total_init == 22
    assert 2 * total_term == 22


def test_test_per_pair_transition_breakdown_no_shift_boundary_truncated(test_raw):
    narrative_atoms, annotation_atoms = test_raw
    narrative_ts = sorted(_trailing_times(narrative_atoms))
    segs = _segments(narrative_ts)
    meeting = _meeting_set(annotation_atoms)

    per_pair, total_init, total_term = _per_pair_transitions(meeting, segs, shift=0)

    assert per_pair == {
        ("id1", "id2"): (1, 0),  # boundary-truncated: init counted, term clipped
        ("id2", "id6"): (1, 1),
        ("id4", "id5"): (1, 1),
    }
    assert 2 * total_init == 6
    assert 2 * total_term == 4


# ---------------------------------------------------------------------------
# The T-40 reading, falsified: shifting the annotation back one frame
# manufactures the (id1,id2) termination the boundary genuinely clips.
# ---------------------------------------------------------------------------


def test_t_minus_40_reading_is_falsified_by_the_boundary_case(test_raw):
    """The "+40ms" convention some notes attach to this dataset would, if
    applied as a literal T -> T-40 shift on the annotation, manufacture a
    termination for the (id1,id2) boundary-truncated case that the source
    data does not actually show happening before the video ends -- this is
    `caviar_continuous.py`'s own falsification argument for why no shift is
    applied when reading ``holdsAt(meeting(p1,p2),T)``'s own ``T``.
    Recomputed independently here: under the shifted reading, test goes
    from 6 initiations / 4 terminations (the genuine, boundary-aware count)
    to 6/6 -- purely by inventing the missing termination, not by finding a
    real one; the initiation count is UNCHANGED (6), confirming the shift
    only ever fabricates the one specific termination -- not a wash across
    unrelated pairs."""
    narrative_atoms, annotation_atoms = test_raw
    narrative_ts = sorted(_trailing_times(narrative_atoms))
    segs = _segments(narrative_ts)
    meeting = _meeting_set(annotation_atoms)

    per_pair, total_init, total_term = _per_pair_transitions(meeting, segs, shift=40)

    assert per_pair[("id1", "id2")] == (1, 1)  # the manufactured termination
    assert per_pair[("id2", "id6")] == (1, 1)  # unaffected
    assert per_pair[("id4", "id5")] == (1, 1)  # unaffected
    assert 2 * total_init == 6  # unchanged from the no-shift reading
    assert 2 * total_term == 6  # == init count now -- the shift is refuted
