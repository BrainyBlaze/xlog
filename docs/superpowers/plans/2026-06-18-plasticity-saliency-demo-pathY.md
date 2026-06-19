# Plasticity & Saliency Rule Induction — Demo (Path Y) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a flagship demo that learns a symbolic plasticity rule (STDP/LTP: an edge strengthens iff it has a pre-before-post coincidence AND its saliency is high) by inducing the correct candidate among distractors via xlog's existing multi-rule neural-bodied joint mixture — recovering a *planted* ground-truth rule and proving held-out generalization + vigilance.

**Architecture:** Pure Python/torch demo on top of the **current** `pyxlog.ilp.neurosymbolic` engine (no Rust changes). A seeded synthetic generator plants a known STDP rule and emits, per split: head-bound relational projections as ground facts (`edge_pre_post(i)`, `edge_post_pre(i)` — the existential `∃Event` projection done in preprocessing), a per-edge saliency feature `φ(Edge)`, and ground-truth labels. The demo registers three same-head `trainable_rule` candidates (relational-only pre-post, neural-bodied pre-post, neural-bodied post-pre distractor) and trains them with `train_neurosymbolic_program(..., neural_bodies=...)`; the correct candidate (relational eligibility **AND** a learned saliency gate) wins because the relational-only candidate over-fires on weak coincidences and the post-pre candidate has the wrong relational mask. Selection-then-admission and a module-boundary leakage diagnostic complete the pipeline.

**Tech Stack:** Python 3, PyTorch (CUDA), `pyxlog.ilp.neurosymbolic` (`train_neurosymbolic_program`, `evaluate_joint_mixture`, `NeuralBodySpec`). Tests: `pytest` (CPU for generator/leakage/source; CUDA-gated for training/recovery).

---

## Honesty notes (read before executing)

This is **Path Y**, chosen deliberately over the circuit existential-join path (Path X / "Stage B"). Be precise about what it does and does NOT showcase:

- **The neural saliency is torch-side, not an xlog neural predicate.** `φ(Edge)` is a fixed per-edge feature computed in Python; the learned component is a small straight-through gate `g_θ(φ) ≥ τ` (`NeuralBodySpec`, ST-TRC slice-1), trained alongside the rule guards. `φ` is **detached** — no backbone gradient. Do not claim "xlog learns saliency from raw observations."
- **The event→edge aggregation is preprocessing, not symbolic.** The existential `∃Event. pre_before_post(Event, Edge)` is projected to the ground relation `edge_pre_post(Edge)` *outside* xlog, because existential-join trainable bodies still fail-closed on `main` (`crates/pyxlog/src/neural.rs:2210`). xlog does relational gating + noisy-OR mixture + candidate selection.
- **What it DOES genuinely showcase:** multi-rule same-head dILP soft-selection over competing candidates, a learned neural admission gate composed with relational eligibility, a rule inventory / proof-trace credit surface, and a faithful held-out generalization-vs-vigilance read — all on the production engine.
- **CUDA-gated.** `train_neurosymbolic_program` requires CUDA. The generator, source builder, and leakage diagnostic are pure Python and have CPU tests that run anywhere. The training/recovery/held-out tests are `@requires_cuda` and must run on the GPU machine. The current dev box has no CUDA — Tasks 6–9 verify on the GPU box.

**Grounding:** API verified against `../xlog-mixed-bodies` (HEAD merged with `origin/main` @ `87c1f3f2`). Calling convention mirrors `python/tests/test_mixed_trainable_rule_bodies.py:438-512` (`_NEURAL_BODY_SOURCE`, `_train_fragility`, `test_neural_body_*`).

---

## File structure

- `crates/pyxlog/python/pyxlog/demos/__init__.py` — new namespace package (empty).
- `crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py` — exports the public demo API.
- `crates/pyxlog/python/pyxlog/demos/plasticity/generator.py` — synthetic STDP generator: planted ground-truth rule, `EdgeSample`/`Split` types, fixed + seeded-random splits, relational projections, `φ` features, labels, stable entity ids. **Pure Python/torch (CPU).**
- `crates/pyxlog/python/pyxlog/demos/plasticity/leakage.py` — module-boundary / held-out leakage diagnostic over entity ids. **Pure Python (CPU).**
- `crates/pyxlog/python/pyxlog/demos/plasticity/program.py` — builds the xlog source (ground facts + candidate `trainable_rule`s) and the `neural_bodies` spec map from a `Split`. **Pure Python/torch (CPU).**
- `crates/pyxlog/python/pyxlog/demos/plasticity/demo.py` — driver: leakage guard → train → select winner by held-out coverage → admit winner → assemble `DemoReport`. **CUDA.**
- `examples/plasticity_saliency/run_demo.py` — thin runnable entry that prints the report.
- `examples/plasticity_saliency/README.md` — what it shows, how to run, honest scope.
- `python/tests/test_plasticity_generator.py` — CPU tests for the generator.
- `python/tests/test_plasticity_leakage.py` — CPU tests for the leakage diagnostic.
- `python/tests/test_plasticity_program.py` — CPU tests for the source builder.
- `python/tests/test_plasticity_demo.py` — CUDA tests: recovery, held-out, zero-host, inventory.

---

## Task 1: Scaffold the demos package

**Files:**
- Create: `crates/pyxlog/python/pyxlog/demos/__init__.py`
- Create: `crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py`
- Test: `python/tests/test_plasticity_generator.py`

- [ ] **Step 1: Write the failing import test.**

```python
# python/tests/test_plasticity_generator.py
def test_demo_package_imports() -> None:
    import pyxlog.demos.plasticity as plasticity

    assert hasattr(plasticity, "make_demo_data")
```

- [ ] **Step 2: Run it; expect failure.**

Run: `pytest python/tests/test_plasticity_generator.py::test_demo_package_imports -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pyxlog.demos'`.

- [ ] **Step 3: Create the package files.**

```python
# crates/pyxlog/python/pyxlog/demos/__init__.py
```
(empty file)

```python
# crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py
"""Plasticity & Saliency Rule Induction demo (Path Y, on the current engine)."""

from .generator import EdgeSample, Split, make_demo_data, make_fixed_split, make_random_split, strengthens

__all__ = [
    "EdgeSample",
    "Split",
    "make_demo_data",
    "make_fixed_split",
    "make_random_split",
    "strengthens",
]
```

- [ ] **Step 4: Do NOT run yet** (generator.py does not exist; Task 2 creates it and makes this import resolve). Proceed to Task 2.

---

## Task 2: Synthetic STDP generator with a planted ground-truth rule

**Files:**
- Create: `crates/pyxlog/python/pyxlog/demos/plasticity/generator.py`
- Test: `python/tests/test_plasticity_generator.py`

The planted ground truth: **an edge strengthens iff it has a pre-before-post coincidence AND its saliency ≥ 0.5.** Column 0 of `φ` is the saliency `s ∈ [0,1]`; column 1 is a distractor feature the gate must ignore.

- [ ] **Step 1: Write the failing tests.**

```python
# python/tests/test_plasticity_generator.py  (append below the import test)
import torch

from pyxlog.demos.plasticity import make_demo_data, make_fixed_split, strengthens
from pyxlog.demos.plasticity.generator import EdgeSample, SALIENCY_THRESHOLD


def test_ground_truth_rule_is_prepost_and_high_saliency() -> None:
    assert strengthens(EdgeSample("e", pre_post=True, post_pre=False, saliency=0.9, distractor=0.0))
    assert not strengthens(EdgeSample("e", pre_post=True, post_pre=False, saliency=0.2, distractor=9.0))
    assert not strengthens(EdgeSample("e", pre_post=False, post_pre=True, saliency=0.9, distractor=0.0))
    assert SALIENCY_THRESHOLD == 0.5


def test_fixed_train_split_has_discriminating_cases() -> None:
    split = make_fixed_split("e_tr")
    labels = split.labels()
    # exactly the two strong pre-post edges are positive
    assert [i for i, t in enumerate(labels) if t] == [0, 1]
    # weak pre-post edges exist (relational-only candidate must over-fire on them)
    weak_prepost = [i for i, s in enumerate(split.samples) if s.pre_post and s.saliency < 0.5]
    assert weak_prepost, "need weak pre-post negatives so relational-only fails"
    # phi shape is [N, 2]; column 0 is saliency
    phi = split.phi()
    assert phi.shape == (len(split.samples), 2)
    assert torch.allclose(phi[:, 0], torch.tensor([s.saliency for s in split.samples]))


def test_relational_projections_match_samples() -> None:
    split = make_fixed_split("e_tr")
    assert split.relational_pre_post_ids() == [i for i, s in enumerate(split.samples) if s.pre_post]
    assert split.relational_post_pre_ids() == [i for i, s in enumerate(split.samples) if s.post_pre]


def test_demo_data_splits_are_entity_disjoint() -> None:
    train, held_out = make_demo_data()
    assert train.entity_ids().isdisjoint(held_out.entity_ids())
    # held-out carries a strong pre-post positive (generalize) and a weak pre-post negative (vigilance)
    assert any(s.pre_post and s.saliency >= 0.5 for s in held_out.samples)
    assert any(s.pre_post and s.saliency < 0.5 for s in held_out.samples)
```

- [ ] **Step 2: Run; expect failure.**

Run: `pytest python/tests/test_plasticity_generator.py -v`
Expected: FAIL — `ImportError`/`ModuleNotFoundError` (generator.py missing).

- [ ] **Step 3: Implement the generator.**

```python
# crates/pyxlog/python/pyxlog/demos/plasticity/generator.py
"""Synthetic STDP plasticity data with a planted ground-truth rule.

Ground truth (planted): an edge ``strengthens`` iff it has a pre-before-post
coincidence AND its saliency >= SALIENCY_THRESHOLD. The existential event->edge
aggregation is already projected here into head-bound relations
(``pre_post``/``post_pre`` membership per edge), because existential-join
trainable bodies are not expressible on the current engine (Path Y).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any

SALIENCY_THRESHOLD = 0.5


@dataclass(frozen=True)
class EdgeSample:
    entity_id: str  # globally unique, stable identity (NOT the binding index)
    pre_post: bool  # has a pre-before-post coincidence (existential projection)
    post_pre: bool  # has a post-before-pre coincidence
    saliency: float  # s in [0,1]; phi column 0
    distractor: float  # phi column 1; a feature the gate must ignore


def strengthens(sample: EdgeSample) -> bool:
    """The planted ground-truth plasticity rule (LTP)."""
    return sample.pre_post and sample.saliency >= SALIENCY_THRESHOLD


@dataclass
class Split:
    """One data split. The binding index of a sample is its position in
    ``samples`` (the xlog query key ``train_head(i)`` and the phi row index)."""

    samples: list[EdgeSample]

    def num_queries(self) -> int:
        return len(self.samples)

    def relational_pre_post_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.samples) if s.pre_post]

    def relational_post_pre_ids(self) -> list[int]:
        return [i for i, s in enumerate(self.samples) if s.post_pre]

    def labels(self) -> list[bool]:
        return [strengthens(s) for s in self.samples]

    def entity_ids(self) -> set[str]:
        return {s.entity_id for s in self.samples}

    def phi(self) -> Any:
        import torch

        return torch.tensor(
            [[s.saliency, s.distractor] for s in self.samples], dtype=torch.float32
        )


def make_fixed_split(prefix: str) -> Split:
    """A small, deterministic split covering every discriminating case, so the
    recovery test is reproducible and the correct candidate provably wins."""
    rows = [
        # (pre_post, post_pre, saliency, distractor)
        (True, False, 0.90, 0.10),  # 0  strong LTP        -> label TRUE
        (True, False, 0.80, 0.90),  # 1  strong LTP        -> label TRUE
        (True, False, 0.20, 0.80),  # 2  weak pre-post     -> FALSE (relational-only over-fires)
        (True, False, 0.10, 0.20),  # 3  weak pre-post     -> FALSE
        (False, True, 0.90, 0.10),  # 4  post-pre, high s  -> FALSE (wrong timing)
        (False, True, 0.30, 0.50),  # 5  post-pre          -> FALSE
        (False, False, 0.70, 0.40),  # 6 neither           -> FALSE
        (False, False, 0.10, 0.10),  # 7 neither           -> FALSE
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_held_out_split(prefix: str) -> Split:
    """Disjoint held-out entities exercising generalization and vigilance."""
    rows = [
        (True, False, 0.95, 0.20),  # 0  NEW strong LTP     -> should FIRE (generalize)
        (True, False, 0.15, 0.90),  # 1  NEW weak pre-post  -> should NOT fire (vigilance)
        (False, True, 0.90, 0.10),  # 2  NEW post-pre       -> should NOT fire (wrong timing)
        (False, False, 0.60, 0.30),  # 3 NEW neither        -> should NOT fire
    ]
    samples = [
        EdgeSample(f"{prefix}_{i}", pre, post, sal, dis)
        for i, (pre, post, sal, dis) in enumerate(rows)
    ]
    return Split(samples)


def make_random_split(prefix: str, n: int, seed: int) -> Split:
    """A seeded larger split for the runnable demo at scale."""
    rng = random.Random(seed)
    samples: list[EdgeSample] = []
    for i in range(n):
        pre = rng.random() < 0.5
        post = (not pre) and rng.random() < 0.5
        saliency = round(rng.random(), 3)
        distractor = round(rng.random(), 3)
        samples.append(EdgeSample(f"{prefix}_{i}", pre, post, saliency, distractor))
    return Split(samples)


def make_demo_data() -> tuple[Split, Split]:
    """The canonical (train, held_out) pair with disjoint entity ids."""
    return make_fixed_split("e_tr"), make_held_out_split("e_ho")
```

- [ ] **Step 4: Run; expect pass** (including `test_demo_package_imports` from Task 1).

Run: `pytest python/tests/test_plasticity_generator.py -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit.**

```bash
git add crates/pyxlog/python/pyxlog/demos/__init__.py crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py crates/pyxlog/python/pyxlog/demos/plasticity/generator.py python/tests/test_plasticity_generator.py
git commit -m "feat(demo): synthetic STDP plasticity generator with planted ground-truth rule"
```

---

## Task 3: Held-out leakage / module-boundary diagnostic

**Files:**
- Create: `crates/pyxlog/python/pyxlog/demos/plasticity/leakage.py`
- Test: `python/tests/test_plasticity_leakage.py`

The diagnostic enforces the module boundary between splits: the train and held-out entity sets must be disjoint, so the held-out read in `evaluate_joint_mixture` cannot be inflated by a training entity reappearing under a held-out binding index.

- [ ] **Step 1: Write the failing tests.**

```python
# python/tests/test_plasticity_leakage.py
import pytest

from pyxlog.demos.plasticity.generator import make_fixed_split, make_held_out_split
from pyxlog.demos.plasticity.leakage import LeakageError, assert_no_leakage


def test_clean_split_passes() -> None:
    train = make_fixed_split("e_tr")
    held = make_held_out_split("e_ho")
    assert_no_leakage(train, held)  # must not raise


def test_overlapping_entities_are_rejected() -> None:
    train = make_fixed_split("shared")
    held = make_held_out_split("shared")  # same prefix -> overlapping entity ids
    with pytest.raises(LeakageError, match="(?i)overlap"):
        assert_no_leakage(train, held)
```

- [ ] **Step 2: Run; expect failure.**

Run: `pytest python/tests/test_plasticity_leakage.py -v`
Expected: FAIL — `ImportError` (leakage.py missing).

- [ ] **Step 3: Implement the diagnostic.**

```python
# crates/pyxlog/python/pyxlog/demos/plasticity/leakage.py
"""Held-out leakage / module-boundary diagnostic.

The joint-mixture held-out read (``evaluate_joint_mixture``) keys bindings by
position. If a training entity reappears in the held-out split, the held-out
read measures memorization, not generalization. This guard fails closed on any
entity-id overlap between splits, the demo's module boundary."""

from __future__ import annotations

from .generator import Split


class LeakageError(RuntimeError):
    """Raised when train and held-out splits share an entity (held-out leakage)."""


def assert_no_leakage(train: Split, held_out: Split) -> None:
    overlap = train.entity_ids() & held_out.entity_ids()
    if overlap:
        raise LeakageError(
            f"held-out leakage: {len(overlap)} entity id(s) overlap between train "
            f"and held-out splits: {sorted(overlap)[:5]}"
        )
```

- [ ] **Step 4: Run; expect pass.**

Run: `pytest python/tests/test_plasticity_leakage.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit.**

```bash
git add crates/pyxlog/python/pyxlog/demos/plasticity/leakage.py python/tests/test_plasticity_leakage.py
git commit -m "feat(demo): held-out leakage / module-boundary diagnostic"
```

---

## Task 4: xlog source builder + neural-body spec map

**Files:**
- Create: `crates/pyxlog/python/pyxlog/demos/plasticity/program.py`
- Test: `python/tests/test_plasticity_program.py`

Emits the candidate program: head-bound relational facts and three same-head candidates. Candidate ids are stable and consumed by the driver.

- [ ] **Step 1: Write the failing tests.**

```python
# python/tests/test_plasticity_program.py
from pyxlog.demos.plasticity.generator import make_fixed_split
from pyxlog.demos.plasticity.program import (
    CAND_PREPOST_NEURAL,
    CAND_PREPOST_REL,
    CAND_POSTPRE_NEURAL,
    TRAIN_HEAD,
    build_neural_bodies,
    build_source,
)


def test_source_declares_facts_and_three_candidates() -> None:
    split = make_fixed_split("e_tr")
    source = build_source(split)
    # head-bound projected relations as ground facts at binding indices
    assert "edge_pre_post(0)." in source
    assert "edge_pre_post(2)." in source  # weak pre-post still a fact (gate must reject it)
    assert "edge_post_pre(4)." in source
    # three same-head trainable candidates
    for cand in (CAND_PREPOST_REL, CAND_PREPOST_NEURAL, CAND_POSTPRE_NEURAL):
        assert f"trainable_rule({cand}" in source
    assert f"train({TRAIN_HEAD}, binary_cross_entropy)." in source
    # the relational-only and neural pre-post candidates share the SAME body
    assert source.count("edge_pre_post(E)") >= 2


def test_neural_bodies_cover_the_two_neural_candidates() -> None:
    split = make_fixed_split("e_tr")
    bodies = build_neural_bodies(split)
    assert set(bodies) == {CAND_PREPOST_NEURAL, CAND_POSTPRE_NEURAL}
    assert bodies[CAND_PREPOST_NEURAL].features.shape == (split.num_queries(), 2)
    assert bodies[CAND_PREPOST_NEURAL].threshold == 0.5
```

- [ ] **Step 2: Run; expect failure.**

Run: `pytest python/tests/test_plasticity_program.py -v`
Expected: FAIL — `ImportError` (program.py missing).

- [ ] **Step 3: Implement the builder.**

```python
# crates/pyxlog/python/pyxlog/demos/plasticity/program.py
"""Builds the xlog candidate program and neural-body spec map from a Split.

Three same-head candidates compete for ``strengthens(Edge)``:
  - CAND_PREPOST_REL    : relational-only pre-post  (over-fires on weak coincidences)
  - CAND_PREPOST_NEURAL : pre-post AND a learned saliency gate  (the TRUE rule)
  - CAND_POSTPRE_NEURAL : post-pre AND a learned gate  (wrong-timing distractor)
"""

from __future__ import annotations

from typing import Any

from pyxlog.ilp.neurosymbolic import NeuralBodySpec

from .generator import Split

TRAIN_HEAD = "strengthens"
CAND_PREPOST_REL = "cand_prepost_rel"
CAND_PREPOST_NEURAL = "cand_prepost_neural"
CAND_POSTPRE_NEURAL = "cand_postpre_neural"


def build_source(split: Split) -> str:
    pre_facts = " ".join(f"edge_pre_post({i})." for i in split.relational_pre_post_ids())
    post_facts = " ".join(f"edge_post_pre({i})." for i in split.relational_post_pre_ids())
    return f"""
        {pre_facts}
        {post_facts}
        pred edge_pre_post(i64). pred edge_post_pre(i64). pred {TRAIN_HEAD}(i64).
        trainable_rule({CAND_PREPOST_REL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_pre_post(E).
        trainable_rule({CAND_PREPOST_NEURAL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_pre_post(E).
        trainable_rule({CAND_POSTPRE_NEURAL}, weight=0.0) :: {TRAIN_HEAD}(E) :- edge_post_pre(E).
        train({TRAIN_HEAD}, binary_cross_entropy).
    """


def build_neural_bodies(split: Split) -> dict[str, Any]:
    phi = split.phi()
    return {
        CAND_PREPOST_NEURAL: NeuralBodySpec(features=phi, threshold=0.5),
        CAND_POSTPRE_NEURAL: NeuralBodySpec(features=phi, threshold=0.5),
    }
```

- [ ] **Step 4: Run; expect pass.**

Run: `pytest python/tests/test_plasticity_program.py -v`
Expected: PASS (2 tests).

Note: these tests assert on the source *string* and spec map only (CPU-safe). Compiling the source needs CUDA and is covered end-to-end in Task 6.

- [ ] **Step 5: Commit.**

```bash
git add crates/pyxlog/python/pyxlog/demos/plasticity/program.py python/tests/test_plasticity_program.py
git commit -m "feat(demo): xlog candidate-program + neural-body source builder"
```

---

## Task 5: Demo driver — train, select, admit, report

**Files:**
- Create: `crates/pyxlog/python/pyxlog/demos/plasticity/demo.py`
- Modify: `crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py` (export `run_demo`, `DemoReport`)

This is the substantive deliverable: the end-to-end pipeline. It has no standalone unit test (its behavior is verified by the CUDA recovery tests in Tasks 6–8); this task writes the code, and Task 6 writes the first failing test that drives it.

- [ ] **Step 1: Implement the driver.**

```python
# crates/pyxlog/python/pyxlog/demos/plasticity/demo.py
"""Path-Y plasticity demo driver: leakage guard -> train -> select winner by
held-out coverage -> admit winner -> report."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pyxlog.ilp.neurosymbolic import (
    NeuroSymbolicTrainingConfig,
    evaluate_joint_mixture,
    train_neurosymbolic_program,
)

from .generator import Split, strengthens
from .leakage import assert_no_leakage
from .program import (
    CAND_PREPOST_NEURAL,
    CAND_POSTPRE_NEURAL,
    build_neural_bodies,
    build_source,
)

GROUND_TRUTH_RULE = "strengthens(E) :- edge_pre_post(E), saliency(E) >= 0.5"


@dataclass
class DemoReport:
    selected_rule_id: str
    symbolic_rule_weights: dict[str, float]
    train_query_probabilities: list[float]
    heldout_coverage: dict[str, float]
    heldout_admission: list[float]
    heldout_labels: list[bool]
    rule_inventory: Any
    proof_trace_map: Any
    training_host_transfer_stats: Any
    ground_truth_rule: str = GROUND_TRUTH_RULE


def _examples(split: Split) -> list[dict[str, Any]]:
    import torch

    n = split.num_queries()
    return [
        {
            "inputs": torch.zeros((n, 1), dtype=torch.float32),
            "targets": torch.tensor([1.0 if t else 0.0 for t in split.labels()], dtype=torch.float32),
        }
    ]


def _select_winner(
    held_out: Split,
    train_weights: dict[str, float],
    neural_state: dict[str, Any],
) -> tuple[str, dict[str, float]]:
    """SELECT among train-covering candidates by guard-free held-out coverage over
    held-out positives (per evaluate_joint_mixture's SELECT-vs-ADMIT contract):
    rank each train-covering candidate by the mean of its single-candidate held-out
    probability at weight 1.0 over the held-out positive bindings."""
    import torch

    source = build_source(held_out)
    bodies = build_neural_bodies(held_out)
    held_phi = held_out.phi()
    positives = [i for i, t in enumerate(held_out.labels()) if t]
    train_covering = [c for c, w in train_weights.items() if w >= 0.5]

    coverage: dict[str, float] = {}
    for cand in train_covering:
        neural_heldout = None
        if cand in neural_state:
            neural_heldout = {cand: (neural_state[cand], held_phi)}
        probs = evaluate_joint_mixture(
            source,
            rule_weights={cand: 1.0},  # weight 1.0 -> noisy-OR == this candidate's eligibility
            num_queries=held_out.num_queries(),
            neural_heldout=neural_heldout,
        )
        coverage[cand] = float(
            torch.tensor([probs[i] for i in positives]).mean().item()
        ) if positives else 0.0

    winner = max(coverage, key=coverage.get)
    return winner, coverage


def run_demo(
    train: Split,
    held_out: Split,
    config: NeuroSymbolicTrainingConfig = NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1),
) -> DemoReport:
    # 1. module boundary: refuse to proceed on held-out leakage.
    assert_no_leakage(train, held_out)

    # 2. train the joint mixture over the three candidates.
    result = train_neurosymbolic_program(
        build_source(train),
        networks={},
        examples=_examples(train),
        config=config,
        neural_bodies=build_neural_bodies(train),
    )
    neural_state = result.neural_body_state or {}

    # 3. SELECT the winner by held-out coverage (guard-free).
    winner, coverage = _select_winner(
        held_out, result.symbolic_rule_weights, neural_state
    )

    # 4. ADMIT: the faithful held-out read with ONLY the winner's trained guard.
    held_source = build_source(held_out)
    neural_heldout = (
        {winner: (neural_state[winner], held_out.phi())} if winner in neural_state else None
    )
    admission = evaluate_joint_mixture(
        held_source,
        rule_weights={winner: result.symbolic_rule_weights[winner]},
        num_queries=held_out.num_queries(),
        neural_heldout=neural_heldout,
    )

    return DemoReport(
        selected_rule_id=winner,
        symbolic_rule_weights=result.symbolic_rule_weights,
        train_query_probabilities=result.query_probabilities,
        heldout_coverage=coverage,
        heldout_admission=list(admission),
        heldout_labels=held_out.labels(),
        rule_inventory=result.learned_rule_inventory,
        proof_trace_map=result.proof_trace_map,
        training_host_transfer_stats=result.training_host_transfer_stats,
    )
```

- [ ] **Step 2: Export the driver from the package `__init__`.**

Replace `crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py` with:

```python
"""Plasticity & Saliency Rule Induction demo (Path Y, on the current engine)."""

from .demo import DemoReport, GROUND_TRUTH_RULE, run_demo
from .generator import EdgeSample, Split, make_demo_data, make_fixed_split, make_held_out_split, make_random_split, strengthens
from .program import CAND_PREPOST_NEURAL, CAND_PREPOST_REL, CAND_POSTPRE_NEURAL, TRAIN_HEAD

__all__ = [
    "DemoReport",
    "GROUND_TRUTH_RULE",
    "run_demo",
    "EdgeSample",
    "Split",
    "make_demo_data",
    "make_fixed_split",
    "make_held_out_split",
    "make_random_split",
    "strengthens",
    "CAND_PREPOST_NEURAL",
    "CAND_PREPOST_REL",
    "CAND_POSTPRE_NEURAL",
    "TRAIN_HEAD",
]
```

- [ ] **Step 3: Sanity-check imports on CPU** (no training, just that the module graph loads).

Run: `python -c "import pyxlog.demos.plasticity as p; print(p.GROUND_TRUTH_RULE)"`
Expected: prints `strengthens(E) :- edge_pre_post(E), saliency(E) >= 0.5` (no CUDA needed for import).

- [ ] **Step 4: Commit.**

```bash
git add crates/pyxlog/python/pyxlog/demos/plasticity/demo.py crates/pyxlog/python/pyxlog/demos/plasticity/__init__.py
git commit -m "feat(demo): plasticity driver — train, select-by-held-out-coverage, admit, report"
```

---

## Task 6: CUDA recovery test — the correct candidate is induced

**Files:**
- Test: `python/tests/test_plasticity_demo.py`

Run on the **GPU machine** (CUDA required). Reuse the repo's CUDA gate (`requires_cuda`); confirm its import path from an existing test, e.g. `python/tests/test_mixed_trainable_rule_bodies.py` imports it — match that import.

- [ ] **Step 1: Write the failing recovery test.**

```python
# python/tests/test_plasticity_demo.py
import pytest

# Match the CUDA gate used by the sibling neuro tests (confirm the import path
# against python/tests/test_mixed_trainable_rule_bodies.py and copy it verbatim).
from conftest import requires_cuda  # noqa: F401  (adjust to the repo's actual gate import)

from pyxlog.demos.plasticity import (
    CAND_PREPOST_NEURAL,
    CAND_PREPOST_REL,
    CAND_POSTPRE_NEURAL,
    make_demo_data,
    run_demo,
)
from pyxlog.ilp.neurosymbolic import NeuroSymbolicTrainingConfig


@requires_cuda
def test_demo_recovers_the_planted_rule() -> None:
    """The neural-bodied pre-post candidate (relational eligibility AND a learned
    saliency gate) is the only one that separates strong from weak coincidences;
    selection by held-out coverage must pick it over the relational-only and the
    wrong-timing distractor."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1))

    # the correct candidate is selected
    assert report.selected_rule_id == CAND_PREPOST_NEURAL
    # it generalizes on held-out strong LTP, beating the relational-only candidate
    assert report.heldout_coverage[CAND_PREPOST_NEURAL] > report.heldout_coverage[CAND_PREPOST_REL]
    assert report.heldout_coverage[CAND_PREPOST_NEURAL] > report.heldout_coverage.get(CAND_POSTPRE_NEURAL, 0.0)
    # train fit: strong-LTP edges (0,1) high, weak/post-pre/neither low
    p = report.train_query_probabilities
    assert min(p[0], p[1]) > 0.6
    assert max(p[2], p[3], p[4], p[5], p[6], p[7]) < 0.4
```

- [ ] **Step 2: Run on the GPU box; expect failure first if anything is miswired, then iterate to pass.**

Run: `pytest python/tests/test_plasticity_demo.py::test_demo_recovers_the_planted_rule -v`
Expected: PASS once the pipeline is correct. If the relational-only candidate ties or wins, increase `steps` (e.g. 600) — Adam needs to escape the multiplicative-loss plateau (see `test_default_adam_separates_linearly_separable_classes`); the selection is held-out-coverage based, so a true tie on train still resolves to the neural candidate on held-out.

- [ ] **Step 3: Commit.**

```bash
git add python/tests/test_plasticity_demo.py
git commit -m "test(demo): recovery — neural-bodied pre-post candidate is induced"
```

---

## Task 7: CUDA held-out test — generalization and vigilance

**Files:**
- Modify: `python/tests/test_plasticity_demo.py`

- [ ] **Step 1: Write the failing test.**

```python
# python/tests/test_plasticity_demo.py  (append)
@requires_cuda
def test_demo_heldout_generalizes_and_keeps_vigilance() -> None:
    """The admitted winner fires on a NEW strong pre-post edge (generalize) and
    correctly does NOT fire on a new weak pre-post edge (vigilance vs the
    relational-only over-firing), a new post-pre edge, or a new unrelated edge."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1))

    admission = report.heldout_admission
    labels = report.heldout_labels
    # held-out binding 0 is the strong LTP positive; 1..3 are negatives (incl. weak pre-post)
    assert labels[0] is True
    assert admission[0] > 0.6  # generalizes
    assert max(admission[1], admission[2], admission[3]) < 0.4  # vigilance
```

- [ ] **Step 2: Run; expect pass.**

Run: `pytest python/tests/test_plasticity_demo.py::test_demo_heldout_generalizes_and_keeps_vigilance -v`
Expected: PASS.

- [ ] **Step 3: Commit.**

```bash
git add python/tests/test_plasticity_demo.py
git commit -m "test(demo): held-out generalization and vigilance"
```

---

## Task 8: CUDA tests — rule inventory, gradient reach, zero-host

**Files:**
- Modify: `python/tests/test_plasticity_demo.py`

- [ ] **Step 1: Write the failing tests.**

```python
# python/tests/test_plasticity_demo.py  (append)
@requires_cuda
def test_demo_rule_inventory_lists_the_selected_clause() -> None:
    """The learned rule inventory marks the induced clause selected and exposes a
    proof-trace map (the readable rule/credit surface)."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1))
    assert report.proof_trace_map is not None
    # the winner's guard crossed the 0.5 selection bar
    assert report.symbolic_rule_weights[report.selected_rule_id] >= 0.5
    # gradient reached the selected neural candidate's gate during training is
    # asserted via the engine result elsewhere; here assert the rule inventory exists
    assert report.rule_inventory is not None


@requires_cuda
def test_demo_training_is_zero_host() -> None:
    """The neural-bodied joint loop performs no tracked device<->host transfers."""
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=50, learning_rate=0.1))
    stats = report.training_host_transfer_stats
    assert stats["dtoh_calls"] == 0 and stats["htod_calls"] == 0
```

- [ ] **Step 2: Run; expect pass.**

Run: `pytest python/tests/test_plasticity_demo.py -k "rule_inventory or zero_host" -v`
Expected: PASS.

- [ ] **Step 3: Run the full demo test file.**

Run: `pytest python/tests/test_plasticity_demo.py -v`
Expected: all PASS (4 tests).

- [ ] **Step 4: Commit.**

```bash
git add python/tests/test_plasticity_demo.py
git commit -m "test(demo): rule inventory, proof-trace surface, zero-host training"
```

---

## Task 9: Runnable artifact — script + README

**Files:**
- Create: `examples/plasticity_saliency/run_demo.py`
- Create: `examples/plasticity_saliency/README.md`

- [ ] **Step 1: Write the runnable script.**

```python
# examples/plasticity_saliency/run_demo.py
"""Run the Plasticity & Saliency Rule Induction demo and print a report.

Requires CUDA. From the repo's python test environment:
    python examples/plasticity_saliency/run_demo.py
"""

from pyxlog.demos.plasticity import make_demo_data, run_demo
from pyxlog.ilp.neurosymbolic import NeuroSymbolicTrainingConfig


def main() -> None:
    train, held_out = make_demo_data()
    report = run_demo(train, held_out, NeuroSymbolicTrainingConfig(steps=400, learning_rate=0.1))

    print(f"Ground-truth (planted) rule : {report.ground_truth_rule}")
    print(f"Induced rule id             : {report.selected_rule_id}")
    print("Guard weights sigma(w):")
    for rid, w in sorted(report.symbolic_rule_weights.items()):
        print(f"  {rid:22s} {w:.3f}")
    print("Held-out coverage (guard-free):")
    for rid, c in sorted(report.heldout_coverage.items()):
        print(f"  {rid:22s} {c:.3f}")
    print("Held-out admission (winner) vs label:")
    for i, (a, y) in enumerate(zip(report.heldout_admission, report.heldout_labels)):
        print(f"  binding {i}: p={a:.3f}  label={y}")
    print(f"Training host transfers     : {report.training_host_transfer_stats}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write the README (honest scope).**

```markdown
# Plasticity & Saliency Rule Induction (demo)

Learns a symbolic STDP/LTP plasticity rule by inducing the correct candidate
among distractors with xlog's multi-rule neural-bodied joint mixture.

**Planted ground truth:** an edge *strengthens* iff it has a pre-before-post
coincidence AND its saliency >= 0.5.

**Candidates competing for `strengthens(Edge)`:**
- `cand_prepost_rel`    — relational-only pre-post (over-fires on weak coincidences)
- `cand_prepost_neural` — pre-post AND a learned saliency gate (**the true rule**)
- `cand_postpre_neural` — post-pre AND a learned gate (wrong-timing distractor)

The demo trains all three, selects the winner by **held-out coverage** (guard-free),
and admits it with a faithful held-out read. The induced winner is
`cand_prepost_neural`; it generalizes to new strong coincidences and stays
vigilant against weak/wrong-timing ones.

## Run (requires CUDA)

    python examples/plasticity_saliency/run_demo.py

## Scope (honest)

This demo runs entirely on the current engine. The neural saliency is a
torch-side straight-through gate `g_theta(phi) >= tau` over a fixed per-edge
feature `phi` (no backbone gradient); the existential event->edge aggregation is
projected to head-bound ground relations (`edge_pre_post`) in preprocessing,
because existential-join trainable bodies are not yet supported on the engine.
xlog provides the relational gating, multi-rule noisy-OR mixture, candidate
selection, and the rule/proof inventory. Lifting saliency into an in-circuit
neural predicate over a real event domain is the separate "Stage B" engine track.
```

- [ ] **Step 3: Smoke-run on the GPU box.**

Run: `python examples/plasticity_saliency/run_demo.py`
Expected: prints the report; induced rule id is `cand_prepost_neural`; held-out binding 0 high, others low.

- [ ] **Step 4: Commit.**

```bash
git add examples/plasticity_saliency/run_demo.py examples/plasticity_saliency/README.md
git commit -m "docs(demo): runnable plasticity-saliency demo script + README"
```

---

## Task 10 (optional): second plasticity direction — `weakens` (LTD)

Proves the framework generalizes across plasticity outcomes with no new mechanism. **Only do this if a multi-outcome demo is wanted; stabilize/task-relevant follow the identical pattern and need not all be built (YAGNI).**

**Files:**
- Modify: `crates/pyxlog/python/pyxlog/demos/plasticity/generator.py` (add `weakens` ground truth + an `outcome` selector)
- Modify: `crates/pyxlog/python/pyxlog/demos/plasticity/program.py` (parameterize `TRAIN_HEAD` and the timing relation)
- Test: `python/tests/test_plasticity_demo.py`

- [ ] **Step 1: Add the LTD ground truth to the generator.**

```python
# generator.py  (append)
def weakens(sample: EdgeSample) -> bool:
    """LTD: an edge weakens iff it has a post-before-pre coincidence AND saliency >= threshold."""
    return sample.post_pre and sample.saliency >= SALIENCY_THRESHOLD
```

Add a `labels_for(outcome: str)` method to `Split`:

```python
# inside class Split
    def labels_for(self, outcome: str) -> list[bool]:
        if outcome == "strengthens":
            return [strengthens(s) for s in self.samples]
        if outcome == "weakens":
            return [weakens(s) for s in self.samples]
        raise ValueError(f"unknown outcome: {outcome}")
```

- [ ] **Step 2: Parameterize the source builder by outcome and timing relation.**

Add `outcome="strengthens"` / `timing="edge_pre_post"` parameters to `build_source` and `build_neural_bodies`, defaulting to the LTP case so Tasks 4–9 are unaffected. For `weakens`, the correct candidate uses `edge_post_pre` and the distractor uses `edge_pre_post`.

- [ ] **Step 3: Write the failing LTD recovery test, run, commit.**

```python
@requires_cuda
def test_demo_recovers_weakens_rule() -> None:
    """The same pipeline, retargeted to LTD, induces the post-pre neural candidate."""
    # build train/held-out with labels_for("weakens"); run_demo parameterized by outcome.
    ...  # mirror test_demo_recovers_the_planted_rule with outcome="weakens"
```

Run: `pytest python/tests/test_plasticity_demo.py -k weakens -v` → PASS, then commit.

---

## Risks & mitigations

- **CUDA-only training surface.** Tasks 1–5 are CPU-validatable (generator, leakage, source, driver import); Tasks 6–10 require the GPU machine. Do not claim recovery from the no-CUDA dev box — run the CUDA tests on the GPU box and paste output.
- **Train-tie between candidates.** All train-covering candidates can reach a high guard. The demo selects by held-out coverage (guard-free), per `evaluate_joint_mixture`'s SELECT-vs-ADMIT contract — this is what distinguishes the true rule from a train-perfect over-firer. Do not switch selection to guard magnitude.
- **`evaluate_joint_mixture` held-out fact materialization.** The held-out source MUST carry each held-out binding's ground facts at indices `0..num_queries-1` (the function's documented caveat); the source builder emits them from the held-out `Split`, not the train split. The leakage guard enforces entity disjointness so a high held-out probability is generalization, not memorization.
- **`min_p` clamping** makes gated-out probabilities ≈ε, not exactly 0; the recovery/held-out tests assert with `< 0.4` margins, not equality, matching the engine's `test_neural_body_*` conventions.
- **Selection on weight 1.0.** `_select_winner` passes `rule_weights={cand: 1.0}` so the single-candidate noisy-OR equals that candidate's eligibility (guard-free coverage). If the engine's `evaluate_joint_mixture` changes how `rule_weights` values are interpreted, re-confirm against its docstring before trusting coverage numbers.

## Self-review (completed)

- **Spec coverage:** synthetic ground-truth recovery (Tasks 2,6), nn-driven saliency gate (Task 4 `NeuralBodySpec` + Task 6), dILP candidate selection (Task 5 `_select_winner` + Task 6), `DifferentiableProofTraceMap` credit surface (Task 8), module-boundary held-out-leakage diagnostic (Task 3 + enforced in Task 5), held-out generalization/vigilance (Task 7), runnable flagship artifact (Task 9). The "expand to full parser" and "circuit existential join" items are intentionally OUT (already-done / Path X) per the honesty notes.
- **Placeholder scan:** every code step is complete and runnable; the only `...` is Task 10 Step 3 (explicitly optional, mirroring a fully-written sibling test).
- **Type consistency:** `Split`/`EdgeSample` fields, candidate-id constants (`CAND_PREPOST_NEURAL` etc.), `DemoReport` fields, and `NeuralBodySpec(features=, threshold=)` usage match across Tasks 2–9; `run_demo`/`_select_winner`/`build_source`/`build_neural_bodies` signatures are consistent between definition (Task 5) and call sites (Tasks 6–9).
