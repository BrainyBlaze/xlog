import inspect

import pytest


def test_runtime_api_extensions_are_installed_on_native_types() -> None:
    pyxlog = pytest.importorskip("pyxlog")

    assert callable(pyxlog.AsyncEvaluation)
    assert callable(pyxlog.LogicQueryChunk)

    for class_name in ("CompiledLogicProgram", "LogicRelationSession", "CompiledProgram"):
        cls = getattr(pyxlog, class_name)
        assert callable(getattr(cls, "evaluate_async"))
        assert callable(getattr(cls, "progress_stats"))
        assert callable(getattr(cls, "memory_stats"))
        assert callable(getattr(cls, "rule_provenance"))
        assert callable(getattr(cls, "proof_traces"))

    assert callable(pyxlog.CompiledLogicProgram.evaluate_stream)
    assert callable(pyxlog.LogicRelationSession.evaluate_stream)
    assert callable(pyxlog.LogicRelationSession.put_temporal_relation)
    assert callable(pyxlog.LogicRelationSession.temporal_provenance)
    assert callable(pyxlog.CompiledProgram.nn4_lineage)
    assert callable(pyxlog.CompiledProgram.record_nn4_influence)
    assert callable(pyxlog.CompiledProgram.neural_hot_loop_diagnostics)
    assert callable(pyxlog.LogicQueryResult.iter_chunks)
    assert callable(pyxlog.LogicEvalResult.iter_query_chunks)
    assert callable(pyxlog.JointConstraintCarrier.note_producer_stream)
    assert callable(pyxlog.JointConstraintCarrier.note_consumer_stream)

    for method_name in (
        "deterministic_topk",
        "neural_cache_stats",
        "belnap_loss",
        "semantic_loss_tensor",
        "mse_loss_tensor",
        "infoloss_tensor",
    ):
        assert callable(getattr(pyxlog.CompiledProgram, method_name))

    for method_name in (
        "insert_relation",
        "delete_relation",
        "apply_relation_delta",
        "apply_relation_delta_batch",
        "apply_relation_delta_debug",
        "delta_stats",
        "join_index_cache_stats",
        "register_relation_callback",
        "unregister_relation_callback",
    ):
        assert callable(getattr(pyxlog.LogicRelationSession, method_name))


def test_runtime_api_signatures_keep_public_controls() -> None:
    pyxlog = pytest.importorskip("pyxlog")

    async_parameters = inspect.signature(
        pyxlog.CompiledLogicProgram.evaluate_async
    ).parameters
    assert "args" in async_parameters
    assert "kwargs" in async_parameters

    stream_parameters = inspect.signature(
        pyxlog.CompiledLogicProgram.evaluate_stream
    ).parameters
    assert stream_parameters["chunk_rows"].default == 1024
