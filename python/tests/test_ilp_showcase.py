"""Integration test: run the dILP showcase end-to-end.

This test runs the showcase script as a subprocess and verifies
all 4 stages converge. It's an integration smoke test, not a unit test.

The dILP optimiser is stochastic, so the test allows up to 2 full
attempts before reporting failure.
"""

import subprocess
import sys
import os

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP showcase", allow_module_level=True)

MAX_ATTEMPTS = 2


def _run_showcase() -> subprocess.CompletedProcess:
    """Run the showcase script once."""
    return subprocess.run(
        [sys.executable, "python/examples/ilp_showcase.py"],
        capture_output=True,
        text=True,
        timeout=300,
        cwd="/home/dev/projects/xlog",
    )


@pytest.mark.slow
def test_ilp_showcase_all_stages_converge():
    """Run the showcase script and verify exit code 0 (all stages converged).

    Slow: up to 2 attempts × 300s subprocess timeout = 600s worst case.
    """
    last_result = None
    for attempt in range(1, MAX_ATTEMPTS + 1):
        result = _run_showcase()
        last_result = result

        if result.returncode == 0 and "All stages converged" in result.stdout:
            return  # success

        if attempt < MAX_ATTEMPTS:
            print(f"Attempt {attempt} did not converge; retrying...")

    # Final attempt also failed -- report diagnostics
    print("STDOUT:", last_result.stdout[-2000:] if len(last_result.stdout) > 2000 else last_result.stdout)
    print("STDERR:", last_result.stderr[-2000:] if len(last_result.stderr) > 2000 else last_result.stderr)

    assert last_result.returncode == 0, (
        f"Showcase exited with code {last_result.returncode} after {MAX_ATTEMPTS} attempts.\n"
        f"Last output: {last_result.stdout[-500:]}"
    )
    assert "All stages converged" in last_result.stdout, (
        f"Expected convergence message not found after {MAX_ATTEMPTS} attempts.\n"
        f"Last output: {last_result.stdout[-500:]}"
    )
