"""Validation evidence staging helpers.

Canonical evidence is only replaced after a complete PASS result. Interrupted
or failing runs leave their files in a staging directory with an event log.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any


class ValidationStagingRun:
    """Stage validation outputs and promote them only after PASS."""

    def __init__(self, canonical_dir: Path) -> None:
        self.canonical_dir = Path(canonical_dir)
        self.canonical_dir.mkdir(parents=True, exist_ok=True)
        self.staging_dir = self.canonical_dir.parent / (
            f".{self.canonical_dir.name}.staging-{int(time.time() * 1_000_000)}"
        )
        self.staging_dir.mkdir(parents=True, exist_ok=False)
        self._canceled = False
        self._event("started", {})

    def write_json(self, relative_path: str, payload: dict[str, Any]) -> Path:
        """Write a JSON artifact under the staging directory."""

        output_path = self.staging_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        )
        self._event("wrote_json", {"path": relative_path})
        return output_path

    def cancel(self, reason: str) -> None:
        """Record cancellation and keep canonical evidence untouched."""

        self._canceled = True
        self._event("canceled", {"reason": reason})

    def promote_if_pass(self, summary: dict[str, Any]) -> bool:
        """Promote staged files to canonical only when summary status is PASS."""

        status = summary.get("status")
        self._event("completed", {"status": status})
        if self._canceled or status != "PASS":
            return False
        for path in self.staging_dir.rglob("*"):
            if not path.is_file() or path.name == "validation_events.jsonl":
                continue
            target = self.canonical_dir / path.relative_to(self.staging_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        self._event("promoted", {"canonical_dir": str(self.canonical_dir)})
        return True

    def _event(self, event: str, payload: dict[str, Any]) -> None:
        event_path = self.staging_dir / "validation_events.jsonl"
        record = {"event": event, **payload}
        with event_path.open("a") as handle:
            handle.write(json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n")
