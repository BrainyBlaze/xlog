from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_pyxlog_stubs_expose_v080_runtime_session_api() -> None:
    stub = (ROOT / "crates/pyxlog/python/pyxlog/_native.pyi").read_text()
    init_stub = (ROOT / "crates/pyxlog/python/pyxlog/__init__.pyi").read_text()

    for needle in [
        "class AsyncEvaluation:",
        "class LogicQueryChunk:",
        "def evaluate_async(",
        "def progress_stats(",
        "def memory_stats(",
        "memory_mb: Optional[int] = None",
        "def iter_chunks(",
        "def iter_query_chunks(",
    ]:
        assert needle in stub or needle in init_stub


def test_python_package_installs_v080_async_streaming_wrappers() -> None:
    init_py = (ROOT / "crates/pyxlog/python/pyxlog/__init__.py").read_text()

    for needle in [
        "class AsyncEvaluation",
        "class LogicQueryChunk",
        "ThreadPoolExecutor",
        "evaluate_async",
        "iter_chunks",
        "iter_query_chunks",
        "progress_stats",
        "_install_v080_runtime_api()",
    ]:
        assert needle in init_py


def test_rust_bindings_enforce_per_call_memory_and_expose_diagnostics() -> None:
    logic_rs = (ROOT / "crates/pyxlog/src/logic.rs").read_text()
    program_rs = (ROOT / "crates/pyxlog/src/program.rs").read_text()

    for source in [logic_rs, program_rs]:
        assert "memory_mb: Option<u64>" in source
        assert "enforce_call_memory_limit" in source
        assert "memory_stats" in source
        assert "allocated_bytes" in source
        assert "memory_limit_bytes" in source


def test_pyxlog_runtime_probe_reproducer_is_committed() -> None:
    probe = (ROOT / "scripts/v080_pyxlog_runtime_probe.py").read_text()

    for needle in [
        "--probe",
        "--output",
        "run_pyapi_probe",
        "evaluate_async",
        "iter_query_chunks",
        "memory_mb",
        "progress_stats",
    ]:
        assert needle in probe
