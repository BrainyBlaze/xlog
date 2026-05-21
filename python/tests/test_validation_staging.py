from pathlib import Path

from scripts.validation_staging import ValidationStagingRun


def test_xlog_repro_006_validation_staging_promotes_only_pass(tmp_path: Path) -> None:
    canonical = tmp_path / "outputs"
    canonical.mkdir()
    (canonical / "summary.json").write_text('{"status":"PASS","version":"old"}\n')

    failed = ValidationStagingRun(canonical)
    failed.write_json("summary.json", {"status": "FAIL", "version": "new"})
    assert failed.promote_if_pass({"status": "FAIL"}) is False
    assert '"version":"old"' in (canonical / "summary.json").read_text()
    assert "FAIL" in (failed.staging_dir / "validation_events.jsonl").read_text()

    canceled = ValidationStagingRun(canonical)
    canceled.write_json("summary.json", {"status": "PASS", "version": "canceled"})
    canceled.cancel("operator stopped soak")
    assert '"version":"old"' in (canonical / "summary.json").read_text()
    assert "operator stopped soak" in (
        canceled.staging_dir / "validation_events.jsonl"
    ).read_text()

    passing = ValidationStagingRun(canonical)
    passing.write_json("summary.json", {"status": "PASS", "version": "new"})
    assert passing.promote_if_pass({"status": "PASS"}) is True
    assert '"version":"new"' in (canonical / "summary.json").read_text()
