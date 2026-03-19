"""Guard tests: sparse trainer path must not fall back to dense APIs."""

import inspect
from pathlib import Path
import re

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()

from pyxlog.ilp import train_only, TrainConfig
import pyxlog.ilp.backend as backend_mod
from pyxlog.ilp import trainer as trainer_mod


SOURCE = """
edge(1, 2). edge(2, 3). edge(3, 4). edge(4, 5).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""
POS = [("reach", [1, 3]), ("reach", [2, 4])]
NEG = [("reach", [1, 1])]
SESSION_SOURCE = """
pred edge(u32, u32).
pred reach(u32, u32).
learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
"""


def _u32_columns(left: list[int], right: list[int]) -> list[torch.Tensor]:
    return [
        torch.tensor(left, device="cuda", dtype=torch.int32),
        torch.tensor(right, device="cuda", dtype=torch.int32),
    ]


def test_sparse_backend_prefers_selected_sparse_api():
    """Sparse backend apply_mask must use the selected sparse API, never dense.

    Two-pronged verification:
    1. Static: inspect SparseMaskBackend.apply_mask source for bare
       set_rule_mask( calls.
    2. Runtime: run a short sparse training to confirm the code path
       executes without error.
    """
    # --- Prong 1: static source inspection ---
    compat_src = inspect.getsource(backend_mod.SparseMaskBackend._apply_mask_compat)
    strict_src = inspect.getsource(backend_mod.SparseMaskBackend._apply_mask_strict)
    src = compat_src + "\n" + strict_src

    # Find all set_rule_mask calls; filter out set_rule_mask_sparse.
    # Pattern: .set_rule_mask( NOT followed by _sparse
    bare_calls = re.findall(r'\.set_rule_mask\b(?!_sparse)', src)
    assert len(bare_calls) == 0, (
        f"SparseMaskBackend.apply_mask contains {len(bare_calls)} call(s) "
        f"to set_rule_mask (dense API); expected only sparse APIs"
    )

    legacy_sparse_calls = re.findall(r'\.set_rule_mask_sparse\b(?!_selected)', src)
    assert len(legacy_sparse_calls) == 0, (
        "SparseMaskBackend.apply_mask still calls legacy set_rule_mask_sparse"
    )

    selected_sparse_calls = re.findall(r'\.set_rule_mask_sparse_selected\b', src)
    assert len(selected_sparse_calls) > 0, (
        "SparseMaskBackend.apply_mask does not call set_rule_mask_sparse_selected"
    )

    # --- Prong 2: runtime smoke test ---
    config = TrainConfig(
        step_budget_per_attempt=10,
        max_attempts=1,
        seed=42,
        debug_dense_mask=False,  # sparse backend
    )
    result = train_only(SOURCE, "W", POS, NEG, config)
    assert result.attempt_count >= 1


def test_train_on_compiled_relations_wrapper_does_not_recompile_or_fallback_to_train_only():
    train_from_compiled = getattr(trainer_mod, "train_on_compiled_relations", None)
    assert train_from_compiled is not None, (
        "trainer.train_on_compiled_relations must exist for relation-native strict training"
    )
    src = inspect.getsource(train_from_compiled)
    assert "train_only(" not in src
    assert "IlpProgramFactory.compile(" not in src


def test_train_on_compiled_relations_strict_loop_applies_selected_device_mask_before_loss_grad():
    src = inspect.getsource(trainer_mod._run_single_attempt_strict_relations)
    assert "set_rule_mask_sparse_selected_device(" in src
    assert "compute_ilp_loss_grad_gpu_relations(" in src


def test_relation_native_strict_result_builder_has_no_compat_exporter():
    src = inspect.getsource(trainer_mod._build_relation_native_strict_train_result)
    assert "_compat_exporter=None" in src
    assert "_export_compat_result" not in src


def test_compiled_ilp_program_put_relation_persists_across_reset_runtime():
    prog = pyxlog.IlpProgramFactory.compile(SESSION_SOURCE, device=0, memory_mb=512)
    assert hasattr(prog, "put_relation"), (
        "CompiledIlpProgram.put_relation must exist for relation-native strict training"
    )
    prog.put_relation("edge", _u32_columns([1, 2, 3], [2, 3, 4]))
    prog.reset_runtime()

    mask_dl = prog.batch_fact_membership_device("edge", [[1, 2], [2, 3], [9, 9]])
    mask = torch.from_dlpack(mask_dl)
    assert mask.device.type == "cuda"
    assert mask.cpu().tolist() == [True, True, False]


def test_train_on_compiled_relations_is_strict_only():
    train_from_compiled = getattr(trainer_mod, "train_on_compiled_relations", None)
    assert train_from_compiled is not None, (
        "trainer.train_on_compiled_relations must exist for relation-native strict training"
    )

    prog = pyxlog.IlpProgramFactory.compile(SESSION_SOURCE, device=0, memory_mb=512)
    prog.put_relation("edge", _u32_columns([1, 2, 3], [2, 3, 4]))
    positives = {"reach": _u32_columns([1], [3])}
    negatives = {}

    with pytest.raises(Exception, match="strict_gpu_native"):
        train_from_compiled(
            prog,
            "W",
            positives,
            negatives,
            TrainConfig(strict_gpu_native=False),
        )


def test_sparse_backend_internal_strict_apply_mask_is_hard_gated():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    backend = backend_mod.SparseMaskBackend()
    candidates = prog.valid_candidates("W", False)
    W = torch.zeros(len(candidates), device="cuda", requires_grad=True)

    with pytest.raises(RuntimeError, match="train_only"):
        backend.apply_mask(
            prog=prog,
            mask_name="W",
            W=W,
            tau=1.0,
            budget=2,
            candidates=candidates,
            n=prog.ilp_schema_size(),
            allow_recursive=False,
            strict_gpu_native=True,
        )


def test_sparse_backend_apply_mask_selected_path_is_zero_dtoh():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    backend = backend_mod.SparseMaskBackend()
    candidates = prog.valid_candidates("W", False)
    W = torch.zeros(len(candidates), device="cuda", requires_grad=True)

    prog.reset_host_transfer_stats()
    backend.apply_mask(
        prog=prog,
        mask_name="W",
        W=W,
        tau=1.0,
        budget=2,
        candidates=candidates,
        n=prog.ilp_schema_size(),
        allow_recursive=False,
    )
    after = prog.host_transfer_stats()

    assert after["dtoh_calls"] == 0
    assert after["dtoh_bytes"] == 0


def test_sparse_d2h_counter_clean_after_mask_setup():
    """set_rule_mask_sparse must not increment the D2H transfer counter.

    The Rust implementation uses download_f64_untracked for the DLPack
    soft-probs import, so the D2H counter should remain at zero.
    """
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    cands = prog.valid_candidates("W", False)
    c = len(cands)

    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.reset_d2h_transfer_count()
    prog.set_rule_mask_sparse("W", list(range(c)), soft, 32)

    assert prog.d2h_transfer_count() == 0, (
        f"set_rule_mask_sparse incremented D2H counter to {prog.d2h_transfer_count()}"
    )


def test_legacy_sparse_api_rejected_in_strict_zero_dtoh_mode():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    cands = prog.valid_candidates("W", False)
    c = len(cands)

    soft = torch.tensor([1.0 / c] * c, device="cuda", dtype=torch.float64)
    prog.set_strict_zero_dtoh(True)

    with pytest.raises(RuntimeError, match="strict_zero_dtoh"):
        prog.set_rule_mask_sparse("W", list(range(c)), soft, 32)


def test_sparse_backend_strict_helper_is_hard_gated():
    src = inspect.getsource(backend_mod.SparseMaskBackend._apply_mask_strict)
    assert "raise RuntimeError" in src
    assert "train_only(..., strict_gpu_native=True)" in src
    assert "set_rule_mask_sparse_selected" not in src
    assert "set_rule_mask_sparse_selected_device" not in src


def test_strict_selected_device_mask_is_stored_as_runtime_sparse_device_variant():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    prog.set_candidate_map([(i_edge, i_edge, k_reach)])
    prog.set_strict_zero_dtoh(True)
    prog.set_rule_mask_sparse_selected_device(
        "W",
        torch.tensor([0], device="cuda", dtype=torch.int32),
        torch.tensor([1.0], device="cuda", dtype=torch.float64),
    )

    assert prog.debug_ilp_mask_kind("W") == "sparse_device"


def test_strict_selected_device_public_api_rejects_out_of_range_ids():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    prog.set_candidate_map([(i_edge, i_edge, k_reach)])
    prog.set_strict_zero_dtoh(True)

    with pytest.raises(ValueError, match="out of range"):
        prog.set_rule_mask_sparse_selected_device(
            "W",
            torch.tensor([7], device="cuda", dtype=torch.int64),
            torch.tensor([1.0], device="cuda", dtype=torch.float64),
        )


def test_strict_selected_device_public_api_rejects_duplicate_ids():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    prog.set_candidate_map([(i_edge, i_edge, k_reach)])
    prog.set_strict_zero_dtoh(True)

    with pytest.raises(ValueError, match="duplicate"):
        prog.set_rule_mask_sparse_selected_device(
            "W",
            torch.tensor([0, 0], device="cuda", dtype=torch.int64),
            torch.tensor([1.0, 1.0], device="cuda", dtype=torch.float64),
        )


def test_strict_selected_device_evaluate_is_hard_gated_in_strict_mode():
    prog = pyxlog.IlpProgramFactory.compile(SOURCE, device=0, memory_mb=512)
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")

    prog.set_candidate_map([(i_edge, i_edge, k_reach)])
    prog.set_strict_zero_dtoh(True)
    prog.set_rule_mask_sparse_selected_device(
        "W",
        torch.tensor([0], device="cuda", dtype=torch.int64),
        torch.tensor([1.0], device="cuda", dtype=torch.float64),
    )

    with pytest.raises(RuntimeError, match="SparseDevice.*strict_zero_dtoh"):
        prog.evaluate()


def test_strict_selected_device_setter_contains_no_download_helpers():
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "crates/pyxlog/src/ilp.rs").read_text()
    match = re.search(
        r"pub fn set_rule_mask_sparse_selected_device\([\s\S]*?\n    }\n\n    pub fn evaluate",
        src,
    )
    assert match is not None, "could not isolate set_rule_mask_sparse_selected_device body"
    setter_src = match.group(0)
    assert "download_" not in setter_src


def test_strict_selected_device_provider_helper_contains_no_raw_dtoh():
    repo_root = Path(__file__).resolve().parents[2]
    src = (repo_root / "crates/xlog-cuda/src/provider/ilp.rs").read_text()
    match = re.search(
        r"pub fn build_selected_id_mask\([\s\S]*?\n    }\n\n    pub fn filter_buffer_by_candidate_flag",
        src,
    )
    assert match is not None, "could not isolate build_selected_id_mask body"
    helper_src = match.group(0)
    assert "download_" not in helper_src
    assert "dtoh_sync_copy_into" not in helper_src
