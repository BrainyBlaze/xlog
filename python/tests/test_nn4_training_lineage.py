"""Runtime coverage for neural-network training lineage and influence records."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


def test_registration_metadata_and_influence_records_survive_runtime_reads() -> None:
    program = pyxlog.Program.compile(
        "nn(classifier, [Input], Label, [negative, positive]) :: label(Input, Label)."
    )
    network = torch.nn.Linear(1, 2)
    optimizer = torch.optim.SGD(network.parameters(), lr=0.1)

    program.register_network(
        "classifier",
        network,
        optimizer,
        checkpoint_hash="sha256:checkpoint",
        split_hashes={"train": "sha256:train", "validation": "sha256:validation"},
        calibration_metrics={"ece": 0.03},
        cuda_device=0,
        influence_audit={"calibration_set": "heldout-a"},
    )
    record = program.record_nn4_influence(
        "classifier",
        query="label(7, positive)",
        changed_acceptance=True,
        before=False,
        after=True,
        evidence={"threshold": 0.8},
    )
    record["evidence"]["threshold"] = 0.0

    lineage = program.nn4_lineage("classifier")
    assert lineage["checkpoint_hash"] == "sha256:checkpoint"
    assert lineage["split_hashes"] == {
        "train": "sha256:train",
        "validation": "sha256:validation",
    }
    assert lineage["calibration_metrics"] == {"ece": 0.03}
    assert lineage["cuda_device"] == 0
    assert lineage["influence_audit"] == {
        "registration": {"calibration_set": "heldout-a"},
        "records": [
            {
                "query": "label(7, positive)",
                "changed_acceptance": True,
                "before": False,
                "after": True,
                "evidence": {"threshold": 0.8},
            }
        ],
    }
    assert program.nn4_lineage()["classifier"] == lineage
    assert program.neural_hot_loop_diagnostics()["nn4_lineage"]["classifier"] == lineage

    lineage["split_hashes"]["train"] = "caller mutation"
    lineage["calibration_metrics"]["ece"] = 1.0
    lineage["influence_audit"]["registration"]["calibration_set"] = "caller mutation"
    fresh_lineage = program.nn4_lineage("classifier")
    assert fresh_lineage["split_hashes"]["train"] == "sha256:train"
    assert fresh_lineage["calibration_metrics"]["ece"] == 0.03
    assert fresh_lineage["influence_audit"]["registration"] == {
        "calibration_set": "heldout-a"
    }


def test_influence_record_rejects_an_unregistered_network() -> None:
    program = pyxlog.Program.compile(
        "nn(classifier, [Input], Label, [negative, positive]) :: label(Input, Label)."
    )

    with pytest.raises(ValueError, match="no registered nn/4 lineage"):
        program.nn4_lineage("classifier")
    with pytest.raises(ValueError, match="no registered nn/4 lineage"):
        program.record_nn4_influence(
            "classifier",
            query="label(7, positive)",
            changed_acceptance=False,
        )
