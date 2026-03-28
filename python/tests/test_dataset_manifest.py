import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.neural_datasets import DatasetManifest, download_file, verify_sha256


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


def test_verify_sha256(tmp_path: Path):
    f = tmp_path / "blob.txt"
    f.write_bytes(b"xlog-datasets")
    digest = hashlib.sha256(b"xlog-datasets").hexdigest()
    assert verify_sha256(f, digest) is True
    assert verify_sha256(f, "deadbeef") is False


def test_download_file_local(tmp_path: Path):
    src = tmp_path / "src.bin"
    src.write_bytes(b"xlog-download")
    dst = tmp_path / "dst.bin"
    download_file(f"file://{src}", dst)
    assert dst.read_bytes() == b"xlog-download"
