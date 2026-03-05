"""Tests for GPU-resident ILP credit/loss path (compute_ilp_loss_grad_gpu)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()


def _compile_reach():
    """Compile a minimal ILP program with an edge relation and learnable reach rule."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()
    return prog


def test_set_candidate_map_basic():
    """set_candidate_map succeeds with valid triples."""
    prog = _compile_reach()
    candidates = [(0, 0, 0), (0, 1, 0), (1, 0, 0)]
    prog.set_candidate_map(candidates)


def test_set_candidate_map_length():
    """candidate_map_len returns correct count after set_candidate_map."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0), (1, 1, 0)])
    assert prog.candidate_map_len() == 2


def test_compute_ilp_loss_grad_gpu_basic():
    """GPU loss/grad returns correct shapes, dtypes, and device."""
    prog = _compile_reach()
    candidates = [(0, 0, 0)]
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32)

    positives = [("edge", [1, 2])]
    negatives = []

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    assert loss.device.type == "cuda"
    assert loss.numel() == 1
    assert loss.dtype == torch.float32
    assert grad.shape == cand_probs.shape
    assert grad.dtype == torch.float32
    assert grad.device.type == "cuda"


def test_compute_ilp_loss_grad_gpu_empty_facts():
    """Empty positive + negative lists should return zero loss and zero grad."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu([], [], cand_probs)
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    assert loss.item() == 0.0
    assert torch.all(grad == 0.0)


def test_compute_ilp_loss_grad_gpu_no_candidate_map_error():
    """Should error if candidate_map not set."""
    prog = _compile_reach()
    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)

    with pytest.raises(Exception, match="candidate_map"):
        prog.compute_ilp_loss_grad_gpu([("edge", [1, 2])], [], cand_probs)


def test_compute_ilp_loss_grad_gpu_loss_positive():
    """Positive loss should be > 0 when credit < 1."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.3], device=device, dtype=torch.float32)

    # edge(1,2) is a base fact that may or may not be in the tagged entries
    positives = [("edge", [1, 2])]
    negatives = []

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    # Loss should be finite and non-negative
    assert loss.item() >= 0.0
    assert torch.isfinite(loss).item()
    # Grad should be finite
    assert torch.all(torch.isfinite(grad)).item()


def test_compute_ilp_loss_grad_gpu_f64():
    """F64 dtype is correctly handled for cand_probs."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float64)

    positives = [("edge", [1, 2])]
    negatives = []

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    assert loss.dtype == torch.float64
    assert grad.dtype == torch.float64
    assert loss.device.type == "cuda"
    assert grad.device.type == "cuda"


def test_compute_ilp_loss_grad_gpu_cand_size_mismatch_error():
    """Should error if cand_probs length != candidate_map length."""
    prog = _compile_reach()
    prog.set_candidate_map([(0, 0, 0), (1, 0, 0)])  # 2 candidates

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)  # 1 element

    with pytest.raises(Exception, match="length"):
        prog.compute_ilp_loss_grad_gpu([("edge", [1, 2])], [], cand_probs)
