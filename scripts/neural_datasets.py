from __future__ import annotations

import hashlib
import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class DatasetSplit:
    name: str
    count: int


@dataclass(frozen=True)
class DatasetArtifact:
    name: str
    url: str
    sha256: str


@dataclass
class DatasetManifest:
    name: str
    provider: str
    version: str
    splits: Dict[str, DatasetSplit]
    ci_subset: Dict[str, int]
    artifacts: List[DatasetArtifact]
    metrics: Dict[str, float]

    @staticmethod
    def load(path: Path) -> "DatasetManifest":
        data = json.loads(Path(path).read_text())
        splits = {
            k: DatasetSplit(name=k, count=int(v["count"]))
            for k, v in data.get("splits", {}).items()
        }
        artifacts = [DatasetArtifact(**a) for a in data.get("artifacts", [])]
        return DatasetManifest(
            name=data["name"],
            provider=data["provider"],
            version=data.get("version", ""),
            splits=splits,
            ci_subset={k: int(v) for k, v in data.get("ci_subset", {}).items()},
            artifacts=artifacts,
            metrics={k: float(v) for k, v in data.get("metrics", {}).items()},
        )

    def save(self, path: Path) -> None:
        payload = {
            "name": self.name,
            "provider": self.provider,
            "version": self.version,
            "splits": {k: {"count": v.count} for k, v in self.splits.items()},
            "ci_subset": self.ci_subset,
            "artifacts": [a.__dict__ for a in self.artifacts],
            "metrics": self.metrics,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n")


def verify_sha256(path: Path, expected_hex: str) -> bool:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest() == expected_hex


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as r, open(dest, "wb") as f:
        f.write(r.read())
