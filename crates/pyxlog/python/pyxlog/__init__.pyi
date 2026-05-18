"""Type stubs for the top-level ``pyxlog`` package.

Everything is re-exported from :mod:`pyxlog._native`; this file exists so that
``import pyxlog`` surfaces the same names as ``from pyxlog._native import *``.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

# Re-export the full public surface of the native module.
from pyxlog._native import (
    # Module constant
    __version__ as __version__,
    # Logic (pure Datalog)
    LogicProgram as LogicProgram,
    CompiledLogicProgram as CompiledLogicProgram,
    LogicRelationSession as LogicRelationSession,
    LogicQueryResult as LogicQueryResult,
    LogicEvalResult as LogicEvalResult,
    # Probabilistic / neural-symbolic
    Program as Program,
    CompiledProgram as CompiledProgram,
    EvalResult as EvalResult,
    McDeviceEvalResult as McDeviceEvalResult,
    # Training
    EpochStats as EpochStats,
    TrainingHistory as TrainingHistory,
    train_model as train_model,
    train_model_tensor as train_model_tensor,
    # ILP
    IlpProgramFactory as IlpProgramFactory,
    CompiledIlpProgram as CompiledIlpProgram,
    IlpTaggedCreditDeviceResult as IlpTaggedCreditDeviceResult,
    # DLPack / Arrow utilities
    dlpack_roundtrip as dlpack_roundtrip,
)

# Arrow imports are feature-gated; expose them for type checkers but they may
# be absent at runtime when pyxlog is built without ``arrow-device-import``.
try:
    from pyxlog._native import (
        export_arrow_device as export_arrow_device,
        import_arrow_device as import_arrow_device,
    )
except ImportError:
    pass


class AsyncEvaluation:
    def done(self) -> bool: ...
    def cancel(self) -> bool: ...
    def cancelled(self) -> bool: ...
    def exception(self, timeout: Optional[float] = None) -> BaseException | None: ...
    def result(self, timeout: Optional[float] = None) -> Any: ...
    def __await__(self) -> Any: ...


class LogicQueryChunk:
    relation_name: str
    columns: list[str]
    sort_labels: list[str]
    tensors: list[Any]
    offset: int
    num_rows: int
    is_true: bool


class _V080RuntimeApiMixin:
    def evaluate_async(self, *args: Any, **kwargs: Any) -> AsyncEvaluation: ...
    def progress_stats(self) -> dict[str, Any]: ...
    def memory_stats(self) -> dict[str, Any]: ...
    def insert_relation(self, name: str, dlpack_columns: Any) -> dict[str, Any]: ...
    def delete_relation(self, name: str, dlpack_columns: Any) -> dict[str, Any]: ...
    def apply_relation_delta(
        self,
        name: str,
        insert_columns: Optional[Any] = None,
        delete_columns: Optional[Any] = None,
    ) -> dict[str, Any]: ...
    def delta_stats(self) -> dict[str, Any]: ...


class _V080StreamingMixin:
    def iter_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...
    def iter_query_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...


# v0.8.0 monkey-patched method signatures exposed on the imported native classes:
# def evaluate_async(self, *args: Any, **kwargs: Any) -> AsyncEvaluation: ...
# def progress_stats(self) -> dict[str, Any]: ...
# def memory_stats(self) -> dict[str, Any]: ...
# def insert_relation(self, name: str, dlpack_columns: Any) -> dict[str, Any]: ...
# def delete_relation(self, name: str, dlpack_columns: Any) -> dict[str, Any]: ...
# def apply_relation_delta(self, name: str, insert_columns: Optional[Any] = None, delete_columns: Optional[Any] = None) -> dict[str, Any]: ...
# def delta_stats(self) -> dict[str, Any]: ...
# def iter_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...
# def iter_query_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...
# memory_mb: Optional[int] = None
