# ruff: noqa: E402,F405
from pyxlog._kernel_paths import configure_kernel_search_path

configure_kernel_search_path()

# Re-export everything from the native Rust module
import asyncio
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Iterator

try:
    import pyxlog._native as _native
    from pyxlog._native import *  # noqa: F401,F403
    _NATIVE_AVAILABLE = True
except ModuleNotFoundError as exc:
    if exc.name != "pyxlog._native":
        raise
    _native = None
    _NATIVE_AVAILABLE = False
    __all__: list[str] = []

__doc__ = _native.__doc__ if _native is not None else "pyxlog pure-Python helpers"
if _native is not None and hasattr(_native, "__all__"):
    __all__ = _native.__all__
elif _native is None:
    class _NativeUnavailableIlpProgramFactory:
        @staticmethod
        def compile(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("pyxlog._native is not available")

    IlpProgramFactory = _NativeUnavailableIlpProgramFactory


class AsyncEvaluation:
    """Small awaitable wrapper around a background pyxlog evaluation."""

    def __init__(self, future: Future):
        self._future = future

    def done(self) -> bool:
        return self._future.done()

    def cancel(self) -> bool:
        return self._future.cancel()

    def cancelled(self) -> bool:
        return self._future.cancelled()

    def exception(self, timeout: float | None = None) -> BaseException | None:
        return self._future.exception(timeout=timeout)

    def result(self, timeout: float | None = None) -> Any:
        return self._future.result(timeout=timeout)

    def __await__(self):
        return asyncio.wrap_future(self._future).__await__()


class LogicQueryChunk:
    """Chunk of one LogicQueryResult with DLPack-compatible tensor columns."""

    def __init__(
        self,
        *,
        relation_name: str,
        columns: list[str],
        sort_labels: list[str],
        tensors: list[Any],
        offset: int,
        num_rows: int,
        is_true: bool,
    ):
        self.relation_name = relation_name
        self.columns = columns
        self.sort_labels = sort_labels
        self.tensors = tensors
        self.offset = offset
        self.num_rows = num_rows
        self.is_true = is_true


_V080_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pyxlog-v080")
_V080_ORIGINALS: dict[tuple[type, str], Any] = {}
_V080_PROGRESS: dict[int, dict[str, Any]] = {}


def _progress_for(obj: Any) -> dict[str, Any]:
    key = id(obj)
    if key not in _V080_PROGRESS:
        _V080_PROGRESS[key] = {
            "evaluations_started": 0,
            "evaluations_completed": 0,
            "evaluations_failed": 0,
            "last_rows": 0,
            "last_error": None,
        }
    return _V080_PROGRESS[key]


def _count_logic_rows(result: Any) -> int:
    return sum(int(query.num_rows) for query in getattr(result, "queries", []))


def _recorded_call(obj: Any, original: Any, *args: Any, **kwargs: Any) -> Any:
    progress = _progress_for(obj)
    progress["evaluations_started"] += 1
    try:
        result = original(obj, *args, **kwargs)
    except Exception as exc:
        progress["evaluations_failed"] += 1
        progress["last_error"] = str(exc)
        raise
    progress["evaluations_completed"] += 1
    progress["last_error"] = None
    if hasattr(result, "queries"):
        progress["last_rows"] = _count_logic_rows(result)
    return result


def _evaluate_async(self: Any, *args: Any, **kwargs: Any) -> AsyncEvaluation:
    return AsyncEvaluation(_V080_EXECUTOR.submit(self.evaluate, *args, **kwargs))


def _progress_stats(self: Any) -> dict[str, Any]:
    return dict(_progress_for(self))


def _logic_query_iter_chunks(self: Any, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]:
    if chunk_rows <= 0:
        raise ValueError("chunk_rows must be > 0")
    if not self.tensors:
        yield LogicQueryChunk(
            relation_name=self.relation_name,
            columns=list(self.columns),
            sort_labels=list(self.sort_labels),
            tensors=[],
            offset=0,
            num_rows=int(self.num_rows),
            is_true=bool(self.is_true),
        )
        return

    import torch

    columns = [torch.from_dlpack(tensor) for tensor in self.tensors]
    total = int(self.num_rows)
    for offset in range(0, total, chunk_rows):
        size = min(chunk_rows, total - offset)
        yield LogicQueryChunk(
            relation_name=self.relation_name,
            columns=list(self.columns),
            sort_labels=list(self.sort_labels),
            tensors=[column.narrow(0, offset, size) for column in columns],
            offset=offset,
            num_rows=size,
            is_true=bool(self.is_true),
        )


def _logic_eval_iter_query_chunks(
    self: Any, chunk_rows: int = 1024
) -> Iterator[LogicQueryChunk]:
    for query in self.queries:
        yield from query.iter_chunks(chunk_rows=chunk_rows)


def _logic_program_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _V080_ORIGINALS[(CompiledLogicProgram, "evaluate")], *args, **kwargs
    )


def _logic_program_evaluate_stream(
    self: Any, *args: Any, chunk_rows: int = 1024, **kwargs: Any
) -> Iterator[LogicQueryChunk]:
    return self.evaluate(*args, **kwargs).iter_query_chunks(chunk_rows=chunk_rows)


def _logic_session_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _V080_ORIGINALS[(LogicRelationSession, "evaluate")], *args, **kwargs
    )


def _logic_session_evaluate_stream(
    self: Any, *args: Any, chunk_rows: int = 1024, **kwargs: Any
) -> Iterator[LogicQueryChunk]:
    return self.evaluate(*args, **kwargs).iter_query_chunks(chunk_rows=chunk_rows)


def _compiled_program_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _V080_ORIGINALS[(CompiledProgram, "evaluate")], *args, **kwargs
    )


def _install_v080_runtime_api() -> None:
    if getattr(CompiledLogicProgram, "_v080_runtime_api_installed", False):
        return

    for cls, name, replacement in [
        (CompiledLogicProgram, "evaluate", _logic_program_evaluate),
        (LogicRelationSession, "evaluate", _logic_session_evaluate),
        (CompiledProgram, "evaluate", _compiled_program_evaluate),
    ]:
        _V080_ORIGINALS[(cls, name)] = getattr(cls, name)
        setattr(cls, name, replacement)

    for cls in [CompiledLogicProgram, LogicRelationSession, CompiledProgram]:
        setattr(cls, "evaluate_async", _evaluate_async)
        setattr(cls, "progress_stats", _progress_stats)

    setattr(CompiledLogicProgram, "evaluate_stream", _logic_program_evaluate_stream)
    setattr(LogicRelationSession, "evaluate_stream", _logic_session_evaluate_stream)
    setattr(LogicQueryResult, "iter_chunks", _logic_query_iter_chunks)
    setattr(LogicEvalResult, "iter_query_chunks", _logic_eval_iter_query_chunks)
    setattr(CompiledLogicProgram, "_v080_runtime_api_installed", True)


if _NATIVE_AVAILABLE:
    _install_v080_runtime_api()

try:
    __all__ = list(__all__) + ["AsyncEvaluation", "LogicQueryChunk"]
except NameError:
    pass
