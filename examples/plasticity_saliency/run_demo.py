"""Run the Plasticity & Saliency Rule Induction demo and print a report.

Requires CUDA + an installed pyxlog wheel. From the repo's python environment:
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
