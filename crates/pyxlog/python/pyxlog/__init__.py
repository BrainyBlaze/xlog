# ruff: noqa: E402,F405
from pyxlog._kernel_paths import configure_kernel_search_path

configure_kernel_search_path()

# Re-export everything from the native Rust module
import asyncio
import ctypes
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
    CompiledLogicProgram = _native.CompiledLogicProgram
    LogicRelationSession = _native.LogicRelationSession
    CompiledProgram = _native.CompiledProgram
    LogicQueryResult = _native.LogicQueryResult
    LogicEvalResult = _native.LogicEvalResult
    DifferentiableProofTraceMap = _native.DifferentiableProofTraceMap
elif _native is None:
    class RelationMetadataError(ValueError):
        """Relation metadata error type retained for source-only imports."""

    class RelationEvidence:
        """Opaque native evidence type retained for source-only imports."""

        def __new__(cls, *_args: Any, **_kwargs: Any) -> "RelationEvidence":
            raise RuntimeError("pyxlog._native is not available")

    class _NativeUnavailableIlpProgramFactory:
        @staticmethod
        def compile(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("pyxlog._native is not available")

    IlpProgramFactory = _NativeUnavailableIlpProgramFactory
    __all__.extend(["RelationEvidence", "RelationMetadataError"])

if _native is not None:
    RelationEvidence = _native.RelationEvidence
    RelationMetadataError = _native.RelationMetadataError


_DLPACK_CAPSULE_NAME = b"dltensor"
_K_DLCUDA = 2


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int), ("device_id", ctypes.c_int)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    _fields_ = [
        ("dl_tensor", _DLTensor),
        ("manager_ctx", ctypes.c_void_p),
        ("deleter", ctypes.c_void_p),
    ]


_pycapsule_is_valid = ctypes.pythonapi.PyCapsule_IsValid
_pycapsule_is_valid.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pycapsule_is_valid.restype = ctypes.c_int
_pycapsule_get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
_pycapsule_get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
_pycapsule_get_pointer.restype = ctypes.c_void_p


def _dlpack_is_cuda_fallback(capsule: Any) -> bool:
    if _pycapsule_is_valid(capsule, _DLPACK_CAPSULE_NAME) == 0:
        raise ValueError("Expected a DLPack capsule (dltensor)")
    ptr = _pycapsule_get_pointer(capsule, _DLPACK_CAPSULE_NAME)
    if ptr is None:
        raise RuntimeError("Failed to get DLPack pointer")
    managed = ctypes.cast(ptr, ctypes.POINTER(_DLManagedTensor)).contents
    return int(managed.dl_tensor.device.device_type) == _K_DLCUDA


if "dlpack_is_cuda" not in globals():
    dlpack_is_cuda = _dlpack_is_cuda_fallback
    try:
        __all__ = list(__all__) + ["dlpack_is_cuda"]
    except NameError:
        __all__ = ["dlpack_is_cuda"]


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


_RUNTIME_API_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="pyxlog-runtime-api")
_RUNTIME_API_ORIGINALS: dict[tuple[type, str], Any] = {}
_RUNTIME_API_PROGRESS: dict[int, dict[str, Any]] = {}
_TEMPORAL_PROVENANCE: dict[int, dict[str, dict[str, Any]]] = {}
_NN4_LINEAGE: dict[int, dict[str, dict[str, Any]]] = {}
_NN4_INFLUENCE: dict[int, dict[str, list[dict[str, Any]]]] = {}


def _progress_for(obj: Any) -> dict[str, Any]:
    key = id(obj)
    if key not in _RUNTIME_API_PROGRESS:
        _RUNTIME_API_PROGRESS[key] = {
            "evaluations_started": 0,
            "evaluations_completed": 0,
            "evaluations_failed": 0,
            "last_rows": 0,
            "last_error": None,
        }
    return _RUNTIME_API_PROGRESS[key]


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
    return AsyncEvaluation(_RUNTIME_API_EXECUTOR.submit(self.evaluate, *args, **kwargs))


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


def _logic_session_put_temporal_relation(
    self: Any,
    name: str,
    dlpack_columns: Any,
    *,
    timestamp_column: str,
    dataset_id: str | None = None,
    row_hashes: list[str] | None = None,
    field_hashes: dict[str, list[str]] | None = None,
    uncertainty: Any = None,
    stream_id: str | None = None,
    order_column: str | None = None,
    source: str | None = None,
    process_boundary: str | None = None,
    temporal_order: Any = None,
) -> dict[str, Any]:
    self.put_relation(name, dlpack_columns)
    metadata = {
        "status": "ok",
        "relation": name,
        "timestamp_column": timestamp_column,
        "dataset_id": dataset_id,
        "row_hashes": list(row_hashes or []),
        "field_hashes": dict(field_hashes or {}),
        "uncertainty": uncertainty,
        "stream_id": stream_id,
        "order_column": order_column,
        "source": source,
        "process_boundary": process_boundary,
        "temporal_order": temporal_order,
    }
    _TEMPORAL_PROVENANCE.setdefault(id(self), {})[name] = metadata
    return dict(metadata)


def _logic_session_temporal_provenance(
    self: Any, name: str | None = None
) -> dict[str, Any]:
    records = _TEMPORAL_PROVENANCE.get(id(self), {})
    if name is None:
        return {relation: dict(metadata) for relation, metadata in records.items()}
    return dict(records.get(name, {}))


def put_temporal_relation(
    session: Any,
    name: str,
    dlpack_columns: Any,
    *,
    timestamp_column: str,
    dataset_id: str | None = None,
    row_hashes: Any = None,
    field_hashes: Any = None,
    uncertainty: Any = None,
    stream_id: str | None = None,
    order_column: str | None = None,
    source: str | None = None,
    process_boundary: str | None = None,
    temporal_order: Any = None,
) -> dict[str, Any]:
    """Upload a temporal relation through a session and record provenance."""

    return session.put_temporal_relation(
        name,
        dlpack_columns,
        timestamp_column=timestamp_column,
        dataset_id=dataset_id,
        row_hashes=row_hashes,
        field_hashes=field_hashes,
        uncertainty=uncertainty,
        stream_id=stream_id,
        order_column=order_column,
        source=source,
        process_boundary=process_boundary,
        temporal_order=temporal_order,
    )


def temporal_provenance(session: Any, name: str) -> dict[str, Any]:
    """Return temporal provenance for one session relation."""

    metadata = session.temporal_provenance(name)
    if not metadata:
        return {
            "status": "unavailable",
            "relation": name,
            "reason": "no temporal provenance recorded for relation",
        }
    return dict(metadata)


def _logic_program_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _RUNTIME_API_ORIGINALS[(CompiledLogicProgram, "evaluate")], *args, **kwargs
    )


def _logic_program_evaluate_stream(
    self: Any, *args: Any, chunk_rows: int = 1024, **kwargs: Any
) -> Iterator[LogicQueryChunk]:
    return self.evaluate(*args, **kwargs).iter_query_chunks(chunk_rows=chunk_rows)


def _logic_session_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _RUNTIME_API_ORIGINALS[(LogicRelationSession, "evaluate")], *args, **kwargs
    )


def _logic_session_evaluate_stream(
    self: Any, *args: Any, chunk_rows: int = 1024, **kwargs: Any
) -> Iterator[LogicQueryChunk]:
    return self.evaluate(*args, **kwargs).iter_query_chunks(chunk_rows=chunk_rows)


def _compiled_program_evaluate(self: Any, *args: Any, **kwargs: Any) -> Any:
    return _recorded_call(
        self, _RUNTIME_API_ORIGINALS[(CompiledProgram, "evaluate")], *args, **kwargs
    )


def _compiled_program_register_network_with_lineage(
    self: Any,
    name: str,
    module: Any,
    optimizer: Any,
    scheduler: Any = None,
    batching: bool = True,
    k: int | None = None,
    det: bool = False,
    cache: bool = True,
    cache_size: int = 10000,
    *,
    arity: int | None = None,
    arg_sorts: Any = None,
    artifact_hash: str | None = None,
    checkpoint_hash: str | None = None,
    split_hashes: dict[str, str] | None = None,
    calibration_metrics: dict[str, float] | None = None,
    cuda_device: str | int | None = None,
    influence_audit: dict[str, Any] | None = None,
) -> Any:
    result = _RUNTIME_API_ORIGINALS[(CompiledProgram, "register_network")](
        self, name, module, optimizer, scheduler, batching, k, det, cache, cache_size,
        arity=arity, arg_sorts=arg_sorts, artifact_hash=artifact_hash,
    )
    lineage = {
        "network": name,
        "checkpoint_hash": checkpoint_hash,
        "split_hashes": dict(split_hashes or {}),
        "calibration_metrics": dict(calibration_metrics or {}),
        "cuda_device": cuda_device,
        "influence_audit": dict(influence_audit or {}),
    }
    _NN4_LINEAGE.setdefault(id(self), {})[name] = lineage
    return result


def _compiled_program_nn4_lineage(self: Any, name: str | None = None) -> dict[str, Any]:
    records = _NN4_LINEAGE.get(id(self), {})
    influence = _NN4_INFLUENCE.get(id(self), {})
    if name is not None:
        lineage = dict(records.get(name, {}))
        lineage["influence_audit"] = influence.get(name, [])
        return lineage
    return {
        network: {**dict(lineage), "influence_audit": influence.get(network, [])}
        for network, lineage in records.items()
    }


def _compiled_program_record_nn4_influence(
    self: Any,
    name: str,
    *,
    query: str,
    changed_acceptance: bool,
    before: Any = None,
    after: Any = None,
    evidence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record = {
        "query": query,
        "changed_acceptance": bool(changed_acceptance),
        "before": before,
        "after": after,
        "evidence": dict(evidence or {}),
    }
    _NN4_INFLUENCE.setdefault(id(self), {}).setdefault(name, []).append(record)
    return dict(record)


def _compiled_program_neural_hot_loop_diagnostics(self: Any) -> dict[str, Any]:
    original = _RUNTIME_API_ORIGINALS.get((CompiledProgram, "neural_hot_loop_diagnostics"))
    diagnostics = dict(original(self)) if original is not None else {}
    diagnostics["nn4_lineage"] = _compiled_program_nn4_lineage(self)
    return diagnostics


def _install_runtime_api_extensions() -> None:
    if getattr(CompiledLogicProgram, "_runtime_api_extensions_installed", False):
        return

    for cls, name, replacement in [
        (CompiledLogicProgram, "evaluate", _logic_program_evaluate),
        (LogicRelationSession, "evaluate", _logic_session_evaluate),
        (CompiledProgram, "evaluate", _compiled_program_evaluate),
        (CompiledProgram, "register_network", _compiled_program_register_network_with_lineage),
    ]:
        _RUNTIME_API_ORIGINALS[(cls, name)] = getattr(cls, name)
        setattr(cls, name, replacement)

    if hasattr(CompiledProgram, "neural_hot_loop_diagnostics"):
        _RUNTIME_API_ORIGINALS[(CompiledProgram, "neural_hot_loop_diagnostics")] = getattr(
            CompiledProgram, "neural_hot_loop_diagnostics"
        )
        setattr(
            CompiledProgram,
            "neural_hot_loop_diagnostics",
            _compiled_program_neural_hot_loop_diagnostics,
        )

    for cls in [CompiledLogicProgram, LogicRelationSession, CompiledProgram]:
        setattr(cls, "evaluate_async", _evaluate_async)
        setattr(cls, "progress_stats", _progress_stats)

    setattr(CompiledLogicProgram, "evaluate_stream", _logic_program_evaluate_stream)
    setattr(LogicRelationSession, "evaluate_stream", _logic_session_evaluate_stream)
    setattr(LogicRelationSession, "put_temporal_relation", _logic_session_put_temporal_relation)
    setattr(LogicRelationSession, "temporal_provenance", _logic_session_temporal_provenance)
    setattr(CompiledProgram, "nn4_lineage", _compiled_program_nn4_lineage)
    setattr(CompiledProgram, "record_nn4_influence", _compiled_program_record_nn4_influence)
    setattr(LogicQueryResult, "iter_chunks", _logic_query_iter_chunks)
    setattr(LogicEvalResult, "iter_query_chunks", _logic_eval_iter_query_chunks)
    setattr(CompiledLogicProgram, "_runtime_api_extensions_installed", True)


if _NATIVE_AVAILABLE:
    _install_runtime_api_extensions()

try:
    __all__ = list(__all__) + [
        "AsyncEvaluation",
        "LogicQueryChunk",
        "put_temporal_relation",
        "temporal_provenance",
    ]
except NameError:
    pass
