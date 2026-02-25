"""Integration test for tensorized ILP via pyxlog."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for ILP tests", allow_module_level=True)


def test_ilp_compile_and_schema():
    """Test basic ILP compilation returns correct schema."""
    source = """
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    assert n > 0
    names = prog.ilp_relation_names()
    assert "edge" in names
    assert "reach" in names


def test_ilp_set_mask_and_evaluate():
    """Test mask injection and evaluation."""
    source = """
        edge(1, 2).
        edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()

    # Create mask: all zeros (no active rules)
    W = torch.zeros((n, n, n), device='cuda')
    M_hard = W.contiguous().view(-1)
    M_soft = W.contiguous().view(-1)

    prog.set_rule_mask("W", M_hard, M_soft, n)
    prog.evaluate()

    results = prog.get_tagged_results()
    assert len(results) == 0  # No active rules, no results


def test_ilp_gradient_flow():
    """Test that gradients flow through the ILP mask (RFC T4.1).

    Uses the RFC's per-fact surrogate credit architecture:
    - 3D Gumbel-Softmax with dim=-1 (not flattened)
    - Per-fact credit via tagged_entries_containing_fact (RD-24)
    - Differentiable missed-positive penalty (RD-21)
    """
    import torch.nn.functional as F

    source = """
        edge(1, 2).
        edge(2, 3).
        edge(3, 4).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')

    # RFC: Per-(i,j) Gumbel-Softmax with dim=-1 on 3D tensor
    M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
    index = M_soft.max(dim=-1, keepdim=True)[1]
    M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)
    M = (M_hard - M_soft).detach() + M_soft  # Straight-through

    # RD-16: Flatten to 1D for DLPack ndim==1 compliance
    # DLPack requires .detach() since it cannot export grad-tracking tensors.
    # Gradient flow happens entirely on the Python side through M_soft indexing.
    prog.set_rule_mask(
        "W_mask",
        M_hard.detach().contiguous().view(-1),
        M_soft.detach().contiguous().view(-1),
        n,
    )
    prog.evaluate()

    # Per-fact surrogate credit assignment
    positive_examples = [("reach", [1, 3])]  # edge(1,2) join edge(2,3) should derive this
    loss = torch.tensor(0.0, device='cuda')

    for rel_name, values in positive_examples:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        if contributing:
            # Per-fact credit: sum M_soft for all (i,j,k) that derived this fact
            credit = sum(M_soft[i, j, k] for (i, j, k) in contributing)
            loss = loss + (-torch.log(credit.clamp(min=1e-8)))
        else:
            # RD-21: Differentiable missed-positive penalty
            k_idx = rel_names.index(rel_name)
            penalty = -M_soft[:, :, k_idx].sum() / (n * n)
            loss = loss + penalty

    loss.backward()
    assert W.grad is not None, "Gradients must flow through ST-Gumbel-Softmax"
    assert W.grad.abs().sum().item() > 0, "Non-zero gradient expected (T4.1)"
