from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import yaml

from scripts.cuda_ci import changes_are_relevant, evaluate_python_wheel_gate


ROOT = Path(__file__).resolve().parents[2]


def load_workflow(name: str) -> dict[str, object]:
    return yaml.load(
        (ROOT / ".github" / "workflows" / name).read_text(encoding="utf-8"),
        Loader=yaml.BaseLoader,
    )


def job_commands(job: dict[str, object]) -> str:
    steps = job.get("steps", [])
    assert isinstance(steps, list)
    return "\n".join(
        step["run"]
        for step in steps
        if isinstance(step, dict) and isinstance(step.get("run"), str)
    )


def assert_unfiltered_required_workflow(workflow: dict[str, object]) -> None:
    triggers = workflow["on"]
    assert isinstance(triggers, dict)
    for event in ("pull_request", "push"):
        config = triggers[event]
        if config in (None, ""):
            continue
        assert isinstance(config, dict)
        assert "paths" not in config
        assert "paths-ignore" not in config


def test_required_contexts_are_stable_and_report_for_every_change() -> None:
    ci = load_workflow("ci.yml")
    cuda = load_workflow("cuda-ci.yml")
    assert_unfiltered_required_workflow(ci)
    assert_unfiltered_required_workflow(cuda)

    expected = {
        "clippy": (ci, "clippy"),
        "workspace-tests": (ci, "workspace-tests"),
        "python-wheel": (cuda, "python-wheel"),
    }
    for context, (workflow, job_id) in expected.items():
        jobs = workflow["jobs"]
        assert isinstance(jobs, dict)
        job = jobs[job_id]
        assert isinstance(job, dict)
        assert job.get("name") == context


def test_workspace_validation_runs_cpu_tests_and_compiles_every_target() -> None:
    workflow = load_workflow("ci.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)

    workspace_tests = jobs["workspace-tests"]
    assert isinstance(workspace_tests, dict)
    assert workspace_tests["container"] == {
        "image": "nvidia/cuda:13.1.1-devel-ubuntu22.04"
    }
    workspace_env = workspace_tests["env"]
    assert isinstance(workspace_env, dict)
    assert workspace_env["CARGO_BUILD_JOBS"] == "1"
    assert workspace_env["CARGO_INCREMENTAL"] == "0"
    assert workspace_env["CARGO_PROFILE_TEST_DEBUG"] == "0"
    workspace_commands = job_commands(workspace_tests)
    complete_compile = "cargo test --workspace --all-targets --locked --no-run"
    assert complete_compile in workspace_commands
    assert workspace_commands.count("cargo test --workspace --all-targets --locked") == 1
    for crate in (
        "xlog-core",
        "xlog-ir",
        "xlog-logic",
        "xlog-stats",
        "xlog-solve",
        "xlog-induce",
    ):
        assert f"-p {crate}" in workspace_commands

    clippy = jobs["clippy"]
    assert isinstance(clippy, dict)
    command = job_commands(clippy)
    assert (
        "cargo clippy --workspace --all-targets --all-features --locked -- -D warnings"
        in command
    )
    assert "--no-deps" not in command
    assert "-A clippy::" not in command


def test_cuda_workflow_separates_classification_gpu_work_and_aggregate() -> None:
    workflow = load_workflow("cuda-ci.yml")
    jobs = workflow["jobs"]
    assert isinstance(jobs, dict)

    assert {"cuda-changes", "python-wheel-gpu", "python-wheel"} <= jobs.keys()
    classifier = jobs["cuda-changes"]
    gpu_job = jobs["python-wheel-gpu"]
    aggregate = jobs["python-wheel"]
    rust_tests = jobs["rust-tests"]
    assert isinstance(classifier, dict)
    assert isinstance(gpu_job, dict)
    assert isinstance(aggregate, dict)
    assert isinstance(rust_tests, dict)
    classifier_command = job_commands(classifier)
    assert "git diff --no-renames --name-only -z" in classifier_command
    assert "scripts/cuda_ci.py classify --null" in classifier_command
    assert gpu_job["needs"] == "cuda-changes"
    assert "needs.cuda-changes.outputs.relevant == 'true'" in gpu_job["if"]
    assert "head.repo.full_name == github.repository" in gpu_job["if"]
    assert aggregate["needs"] == ["cuda-changes", "python-wheel-gpu"]
    assert "always()" in aggregate["if"]
    assert "scripts/cuda_ci.py aggregate" in job_commands(aggregate)
    rust_test_commands = job_commands(rust_tests)
    assert (
        "RUST_TEST_THREADS=1 cargo test --workspace --all-targets --release"
        in rust_test_commands
    )
    assert "-- --test-threads=1" not in rust_test_commands

    gpu_test_paths = set(
        re.findall(r"python/tests/[A-Za-z0-9_./-]+\.py", job_commands(gpu_job))
    )
    assert gpu_test_paths
    assert all(changes_are_relevant([path]) for path in gpu_test_paths)


def test_cuda_change_classification_is_complete_and_deterministic() -> None:
    for path in (
        ".github/workflows/cuda-ci.yml",
        "Cargo.lock",
        "crates/xlog-prob/src/mc/resident.rs",
        "python/tests/conftest.py",
        "python/tests/test_pyxlog_conditioned_reuse.py",
        "scripts/cuda_ci.py",
        "scripts/validate_release_gpu.sh",
    ):
        assert changes_are_relevant([path]), path

    assert not changes_are_relevant(
        ["README.md", "docs/index.mdx", ".github/workflows/docs.yml"]
    )


def test_cuda_change_classifier_cli_consumes_null_delimited_git_paths() -> None:
    classifier = ROOT / "scripts" / "cuda_ci.py"
    relevant = subprocess.run(
        [sys.executable, classifier, "classify", "--null"],
        input=b"README.md\0crates/xlog-core/src/lib.rs\0",
        capture_output=True,
        check=False,
    )
    assert relevant.returncode == 0
    assert relevant.stdout == b"true\n"

    irrelevant = subprocess.run(
        [sys.executable, classifier, "classify", "--null"],
        input=b"README.md\0docs/index.mdx\0",
        capture_output=True,
        check=False,
    )
    assert irrelevant.returncode == 0
    assert irrelevant.stdout == b"false\n"


def test_python_wheel_gate_handles_irrelevant_forks_and_relevant_trusted_runs() -> None:
    irrelevant_fork = evaluate_python_wheel_gate(
        event_name="pull_request",
        repository="owner/xlog",
        head_repository="contributor/xlog",
        relevant=False,
        classification_result="success",
        gpu_result="skipped",
    )
    assert irrelevant_fork.passed

    relevant_fork = evaluate_python_wheel_gate(
        event_name="pull_request",
        repository="owner/xlog",
        head_repository="contributor/xlog",
        relevant=True,
        classification_result="success",
        gpu_result="skipped",
    )
    assert not relevant_fork.passed
    assert "trusted same-repository branch" in relevant_fork.message

    trusted_success = evaluate_python_wheel_gate(
        event_name="pull_request",
        repository="owner/xlog",
        head_repository="owner/xlog",
        relevant=True,
        classification_result="success",
        gpu_result="success",
    )
    assert trusted_success.passed


def test_python_wheel_gate_fails_closed_on_incomplete_prerequisites() -> None:
    classification_failure = evaluate_python_wheel_gate(
        event_name="push",
        repository="owner/xlog",
        head_repository="",
        relevant=True,
        classification_result="failure",
        gpu_result="skipped",
    )
    assert not classification_failure.passed
    assert "classification" in classification_failure.message

    missing_classification_output = evaluate_python_wheel_gate(
        event_name="push",
        repository="owner/xlog",
        head_repository="",
        relevant=None,
        classification_result="success",
        gpu_result="skipped",
    )
    assert not missing_classification_output.passed
    assert "produced no result" in missing_classification_output.message

    gpu_failure = evaluate_python_wheel_gate(
        event_name="push",
        repository="owner/xlog",
        head_repository="",
        relevant=True,
        classification_result="success",
        gpu_result="failure",
    )
    assert not gpu_failure.passed
    assert "GPU wheel job" in gpu_failure.message
