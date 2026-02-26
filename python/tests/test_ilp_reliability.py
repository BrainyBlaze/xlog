# python/tests/test_ilp_reliability.py
"""Alpha reliability gate: 5/5 consecutive passes on showcase suite.

See design doc Section 6.1. Validates train_only() converges on all 4 stages
with seeds 0..4.
"""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig

# --- Stage definitions (matching ilp_showcase.py domains) ---

STAGE_1_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5). edge(5, 6).
    learnable(W_reach) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_1_POS = [("reach", [1, 3]), ("reach", [2, 4]), ("reach", [3, 5]), ("reach", [4, 6])]
STAGE_1_NEG = []

STAGE_2_SOURCE = """
    parent(1, 2). parent(2, 3). parent(2, 4). parent(3, 5). parent(4, 6).
    gender(1, 0). gender(2, 1). gender(3, 1). gender(4, 0).
    sibling(2, 7). sibling(7, 2).
    learnable(W_gp) :: grandparent(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_2_POS = [("grandparent", [1, 3]), ("grandparent", [1, 4]),
               ("grandparent", [2, 5]), ("grandparent", [2, 6])]
STAGE_2_NEG = [("grandparent", [1, 2]), ("grandparent", [3, 1])]

STAGE_3_SOURCE = """
    worksAt(1, 100). worksAt(7, 100). worksAt(2, 200). worksAt(4, 200).
    livesIn(1, 201). livesIn(2, 201). livesIn(7, 202). livesIn(4, 202).
    learnable(W_col) :: colleague(X, Y) :- bL(X, Z), bR(Y, Z).
"""
STAGE_3_POS = [("colleague", [1, 7]), ("colleague", [7, 1]),
               ("colleague", [2, 4]), ("colleague", [4, 2])]
STAGE_3_NEG = [("colleague", [1, 2]), ("colleague", [3, 4])]

STAGE_4_SOURCE = """
    succ(0, 1). succ(1, 2). succ(2, 3). succ(3, 4). succ(4, 5).
    pred(1, 0). pred(2, 1). pred(3, 2). pred(4, 3). pred(5, 4).
    learnable(W_p2) :: plus2(X, Y) :- bL(X, Z), bR(Z, Y).
"""
STAGE_4_POS = [("plus2", [0, 2]), ("plus2", [1, 3]), ("plus2", [2, 4]), ("plus2", [3, 5])]
STAGE_4_NEG = [("plus2", [0, 1]), ("plus2", [5, 0])]

STAGES = [
    ("reach", STAGE_1_SOURCE, STAGE_1_POS, STAGE_1_NEG, "W_reach"),
    ("grandparent", STAGE_2_SOURCE, STAGE_2_POS, STAGE_2_NEG, "W_gp"),
    ("colleague", STAGE_3_SOURCE, STAGE_3_POS, STAGE_3_NEG, "W_col"),
    ("plus2", STAGE_4_SOURCE, STAGE_4_POS, STAGE_4_NEG, "W_p2"),
]


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize(
    "stage_name,source,positives,negatives,mask_name",
    STAGES,
    ids=["reach", "grandparent", "colleague", "plus2"],
)
def test_alpha_reliability(stage_name, source, positives, negatives, mask_name, seed):
    """Each stage converges with each seed (alpha gate: 5/5)."""
    config = TrainConfig(
        step_budget_per_attempt=150,
        max_attempts=7,
        tau_start=2.0,
        tau_floor=0.05,
        seed=seed,
        device=0,
        memory_mb=512,
    )
    result = train_only(
        source=source,
        mask_name=mask_name,
        positives=positives,
        negatives=negatives,
        config=config,
    )
    assert result.converged, (
        f"Stage {stage_name} failed with seed={seed}: "
        f"attempts={result.attempt_count}, steps={result.total_steps}, "
        f"best_rule={result.discovered_rule}"
    )
