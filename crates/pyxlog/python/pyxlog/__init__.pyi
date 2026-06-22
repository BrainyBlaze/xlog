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
    DifferentiableProofTraceMap as DifferentiableProofTraceMap,
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
    dlpack_is_cuda as dlpack_is_cuda,
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
    def put_temporal_relation(
        self,
        name: str,
        dlpack_columns: Any,
        *,
        timestamp_column: str,
        dataset_id: Optional[str] = None,
        row_hashes: Optional[list[str]] = None,
        field_hashes: Optional[dict[str, list[str]]] = None,
        uncertainty: Any = None,
        stream_id: Optional[str] = None,
        order_column: Optional[str] = None,
        source: Optional[str] = None,
        process_boundary: Optional[str] = None,
        temporal_order: Any = None,
    ) -> dict[str, Any]: ...
    def temporal_provenance(self, name: Optional[str] = None) -> dict[str, Any]: ...


def put_temporal_relation(
    session: Any,
    name: str,
    dlpack_columns: Any,
    *,
    timestamp_column: str,
    dataset_id: Optional[str] = None,
    row_hashes: Any = None,
    field_hashes: Any = None,
    uncertainty: Any = None,
    stream_id: Optional[str] = None,
    order_column: Optional[str] = None,
    source: Optional[str] = None,
    process_boundary: Optional[str] = None,
    temporal_order: Any = None,
) -> dict[str, Any]: ...


def temporal_provenance(session: Any, name: str) -> dict[str, Any]: ...


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
# def put_temporal_relation(self, name: str, dlpack_columns: Any, *, timestamp_column: str, dataset_id: Optional[str] = None, row_hashes: Optional[list[str]] = None, field_hashes: Optional[dict[str, list[str]]] = None, uncertainty: Any = None, stream_id: Optional[str] = None, order_column: Optional[str] = None, source: Optional[str] = None, process_boundary: Optional[str] = None, temporal_order: Any = None) -> dict[str, Any]: ...
# def temporal_provenance(self, name: Optional[str] = None) -> dict[str, Any]: ...
# def iter_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...
# def iter_query_chunks(self, chunk_rows: int = 1024) -> Iterator[LogicQueryChunk]: ...
# memory_mb: Optional[int] = None


class RelationEvidence:
    def provenance(self) -> dict[str, Any]: ...


# v0.8.7 evidence/provenance and nn/4 lineage monkey-patched APIs:
# def put_relation_with_provenance(self, name: str, dlpack_columns: Any, *, relation_schema: Optional[list[str]] = None, source_path: Optional[str] = None, source_hash: Optional[str] = None, row_hashes: Optional[list[str]] = None, field_hashes: Optional[dict[str, list[str]]] = None, accepted_count: Optional[int] = None, rejected_count: Optional[int] = None, output_path: Optional[str] = None, output_hash: Optional[str] = None, decision_counts: Optional[dict[str, int]] = None) -> dict[str, Any]: ...
# def evidence(self, name: Optional[str] = None) -> dict[str, Any]: ...
# def relation(self, name: str) -> RelationEvidence: ...
# class RelationEvidence: def provenance(self) -> dict[str, Any]: ...
# evidence payload keys: program_hash, relation_schema, source_hash, row_hashes, accepted_count, rejected_count, output_hash, decision_counts
# def nn4_lineage(self, name: Optional[str] = None) -> dict[str, Any]: ...
# def record_nn4_influence(self, name: str, *, query: str, changed_acceptance: bool, before: Any = None, after: Any = None, evidence: Optional[dict[str, Any]] = None) -> dict[str, Any]: ...
# nn/4 lineage keys: checkpoint_hash, split_hashes, calibration_metrics, cuda_device, influence_audit
