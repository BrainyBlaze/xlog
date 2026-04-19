import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts import xlog_doctor as doctor

ROOT = Path(__file__).resolve().parents[2]


def _ok_check(message: str = "ok"):
    return doctor.CheckResult("ok", "OK", message, None)


def _patch_supported_env(monkeypatch):
    monkeypatch.setattr(doctor, "_check_platform", lambda: _ok_check("Linux x86_64"))
    monkeypatch.setattr(doctor, "_check_nvidia_smi", lambda: _ok_check("nvidia-smi visible"))
    monkeypatch.setattr(doctor, "_check_nvcc", lambda: _ok_check("nvcc visible"))
    monkeypatch.setattr(doctor, "_check_rust", lambda: _ok_check("rustc/cargo visible"))
    monkeypatch.setattr(doctor, "_check_python", lambda: _ok_check("Python supported"))
    monkeypatch.setattr(doctor, "_check_cuda_loader", lambda: _ok_check("CUDA loader ready"))


def test_help_works(capsys):
    with pytest.raises(SystemExit) as exc:
        doctor.main(["--help"])

    assert exc.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()
    assert "--workflow" in out
    assert "--json" in out


def test_unsupported_platform_emits_unsupported(monkeypatch, capsys):
    monkeypatch.setattr(
        doctor,
        "_check_platform",
        lambda: doctor.CheckResult(
            "unsupported",
            "UNSUPPORTED",
            "xlog public release supports Linux x86_64 only",
            None,
        ),
    )

    exit_code = doctor.main([])
    out = capsys.readouterr().out

    assert exit_code == doctor.EXIT_UNSUPPORTED
    assert "UNSUPPORTED" in out
    assert "Linux x86_64" in out


@pytest.mark.parametrize(
    "probe_name, failure_text",
    [
        ("_check_nvcc", "nvcc --version"),
        ("_check_nvidia_smi", "nvidia-smi"),
    ],
)
def test_missing_nvcc_or_gpu_emits_actionable_fail(
    monkeypatch, capsys, probe_name, failure_text
):
    _patch_supported_env(monkeypatch)
    monkeypatch.setattr(
        doctor,
        probe_name,
        lambda: doctor.CheckResult(
            "fail",
            "FAIL",
            f"Missing {failure_text}",
            f"Install CUDA Toolkit and make sure {failure_text} works",
        ),
    )

    exit_code = doctor.main([])
    out = capsys.readouterr().out

    assert exit_code == doctor.EXIT_FAIL
    assert "FAIL" in out
    assert failure_text in out
    assert "Install CUDA Toolkit" in out


def test_smoke_path_exits_zero_on_supported_env(monkeypatch, capsys):
    _patch_supported_env(monkeypatch)

    exit_code = doctor.main([])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "SUPPORTED" in out
    assert "Linux x86_64" in out


def test_prob_cli_mentions_host_io_requirement(monkeypatch, capsys):
    _patch_supported_env(monkeypatch)

    exit_code = doctor.main(["--workflow", "prob-cli"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert "prob-cli" in out
    assert "host-io" in out


def test_json_cli_invocation_round_trip():
    with tempfile.TemporaryDirectory() as tmpdir:
        result = subprocess.run(
            [sys.executable, "scripts/xlog_doctor.py", "--workflow", "prob-cli", "--json"],
            cwd=ROOT,
            env={"PATH": tmpdir},
            capture_output=True,
            text=True,
            check=False,
        )

    assert result.returncode in (doctor.EXIT_FAIL, doctor.EXIT_UNSUPPORTED)
    payload = json.loads(result.stdout)
    assert payload["exit_code"] == result.returncode
    assert payload["workflow"] == "prob-cli"
    assert payload["overall_status"] in {"FAIL", "UNSUPPORTED"}
    if payload["overall_status"] == "FAIL":
        failing = {check["slug"] for check in payload["checks"] if check["status"] == "FAIL"}
        assert failing & {"nvidia-smi", "nvcc", "rust"}
        assert any(check["slug"] == "workflow" for check in payload["checks"])
    else:
        assert payload["checks"][0]["slug"] == "platform"
        assert payload["checks"][0]["status"] == "UNSUPPORTED"


def test_import_does_not_create_runtime_shim() -> None:
    shim_dir = Path("/tmp/xlog-cuda-shim")
    shutil.rmtree(shim_dir, ignore_errors=True)

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import scripts.xlog_doctor; "
            "print(Path('/tmp/xlog-cuda-shim').exists())",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "False"
