"""GA reliability gate: 50/50 with Clopper-Pearson lower-bound check."""
import os

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")
try:
    from scipy.stats import beta as scipy_beta_dist
except Exception:  # pragma: no cover - exercised when SciPy optional extra is absent
    scipy_beta_dist = None

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()

from pyxlog.ilp import TrainConfig, train_only
from test_ilp_reliability import STAGES


def _clopper_p_lower(success: int, runs: int, alpha: float = 0.025) -> float:
    if success <= 0:
        return 0.0
    if scipy_beta_dist is None:
        pytest.skip("scipy is required for exact Clopper-Pearson interval computation")
    return float(scipy_beta_dist.ppf(alpha, success, runs - success + 1))


@pytest.mark.slow
def test_ga_reliability_50():
    """Run 50 seeds across the 4-stage showcase and enforce CI lower bound."""
    seed_count = int(os.getenv("GA_RELIABILITY_SEEDS", "50"))
    seed_count = max(1, seed_count)
    runs = 0
    success = 0
    failures = []
    stage_stats = {
        "reach": {"success": 0, "total": 0},
        "grandparent": {"success": 0, "total": 0},
        "colleague": {"success": 0, "total": 0},
        "plus2": {"success": 0, "total": 0},
    }

    for seed in range(seed_count):
        for stage_name, source, positives, negatives, mask_name in STAGES:
            config = TrainConfig(
                step_budget_per_attempt=150,
                max_attempts=7,
                tau_start=2.0,
                tau_floor=0.05,
                device=0,
                memory_mb=512,
                debug_dense_mask=False,
                seed=seed,
            )
            runs += 1
            stage_stats[stage_name]["total"] += 1

            result = train_only(
                source=source,
                mask_name=mask_name,
                positives=positives,
                negatives=negatives,
                config=config,
                _compute_holdout=False,
            )

            if result.converged:
                success += 1
                stage_stats[stage_name]["success"] += 1
            else:
                failures.append(
                    f"{stage_name}:seed={seed}: rule={result.discovered_rule!r}:"
                    f" attempts={result.attempt_count}, steps={result.total_steps}"
                )

    ci_low = _clopper_p_lower(success, runs)
    rate = success / runs if runs else 0.0

    print("GA reliability summary:")
    print(f"  runs: {runs}")
    print(f"  success: {success}/{runs} ({rate:.4f})")
    print(f"  clopper_pearson_95_lower: {ci_low:.6f}")
    for stage_name in sorted(stage_stats):
        stat = stage_stats[stage_name]
        print(
            f"  stage={stage_name}: "
            f"{stat['success']}/{stat['total']} "
            f"({stat['success']/stat['total'] if stat['total'] else 0:.4f})"
        )
    if failures:
        print("FAILURES:\n" + "\n".join(failures[:10]))

    assert ci_low >= 0.929, (
        f"GA reliability check failed: lower95={ci_low:.6f}, "
        f"success={success}/{runs}, failures={len(failures)}"
    )
