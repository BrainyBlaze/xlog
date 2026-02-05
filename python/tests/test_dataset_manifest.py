import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.neural_datasets import DatasetManifest, DatasetSplit


def test_manifest_parse_and_roundtrip(tmp_path: Path):
    sample = tmp_path / "dataset.json"
    sample.write_text(
        json.dumps(
            {
                "name": "mnist",
                "provider": "torchvision",
                "version": "1.0",
                "splits": {
                    "train": {"count": 60000},
                    "test": {"count": 10000},
                },
                "ci_subset": {
                    "train": 128,
                    "test": 64,
                },
                "artifacts": [],
                "metrics": {"min_accuracy": 0.80},
            }
        )
    )

    manifest = DatasetManifest.load(sample)
    assert manifest.name == "mnist"
    assert manifest.provider == "torchvision"
    assert manifest.splits["train"].count == 60000
    assert manifest.ci_subset["train"] == 128

    out = tmp_path / "roundtrip.json"
    manifest.save(out)
    loaded = DatasetManifest.load(out)
    assert loaded.name == "mnist"
