"""Integration test for tensorized ILP via pyxlog."""

import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")
F = pytest.importorskip("torch.nn.functional")

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


def test_ilp_missed_positive_penalty():
    """T4.5: Missed-positive penalty produces non-zero gradient (RD-21)."""
    source = """
        edge(1, 2).
        edge(2, 3).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')

    M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
    index = M_soft.max(dim=-1, keepdim=True)[1]
    M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)

    prog.set_rule_mask("W_mask",
                        M_hard.detach().contiguous().view(-1),
                        M_soft.detach().contiguous().view(-1), n)
    prog.evaluate()

    # Ask for a fact that almost certainly won't be derived
    contributing = prog.tagged_entries_containing_fact("reach", [99, 99])
    assert len(contributing) == 0, "Sanity: this fact shouldn't be derived"

    # RD-21: Differentiable missed-positive penalty
    k_idx = rel_names.index("reach")
    penalty = -M_soft[:, :, k_idx].sum() / (n * n)
    penalty.backward()
    assert W.grad is not None, "Missed-positive penalty must produce gradients"
    assert W.grad.abs().sum().item() > 0, "Non-zero gradient from penalty (T4.5)"


def test_ilp_temperature_annealing():
    """T4.4: Temperature annealing produces increasingly discrete M."""
    n = 4
    W = torch.randn((n, n, n), device='cuda')

    discreteness = []
    for tau in [2.0, 1.0, 0.5, 0.1]:
        M_soft = F.gumbel_softmax(W, tau=tau, hard=False, dim=-1)
        max_vals = M_soft.max(dim=-1)[0]
        discreteness.append(max_vals.mean().item())

    for i in range(len(discreteness) - 1):
        assert discreteness[i] <= discreteness[i + 1] + 0.05, \
            f"Temperature annealing: tau decrease should increase discreteness"


def test_ilp_predecessor_benchmark_smoke():
    """T5.1 smoke: Predecessor benchmark setup compiles and runs 5 steps."""
    source = """
        edge(0, 1). edge(1, 2). edge(2, 3). edge(3, 4).
        base(0, 1). base(1, 2). base(2, 3).
        learnable(W_mask) :: pred(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()

    W = torch.randn((n, n, n), requires_grad=True, device='cuda')
    optimizer = torch.optim.Adam([W], lr=0.1)

    for step in range(5):
        optimizer.zero_grad()
        M_soft = F.gumbel_softmax(W, tau=0.5, hard=False, dim=-1)
        index = M_soft.max(dim=-1, keepdim=True)[1]
        M_hard = torch.zeros_like(M_soft).scatter_(-1, index, 1.0)

        prog.set_rule_mask("W_mask",
                            M_hard.detach().contiguous().view(-1),
                            M_soft.detach().contiguous().view(-1), n)
        prog.evaluate()

        results = prog.get_tagged_results()
        loss = torch.tensor(0.0, device='cuda')
        for (i, j, k, nr) in results:
            if nr > 0:
                loss = loss + M_soft[i, j, k]
        if loss.requires_grad:
            loss.backward()
            optimizer.step()

    assert True  # Smoke: completed without crash


def test_ilp_commit_rule():
    """T5.5 + T5.6: Rule commit removes learnable, post-commit matches."""
    source = """
        edge(1, 2). edge(2, 3).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)

    # commit_induced_rule recompiles with a concrete rule (using actual
    # predicates like 'edge', not the learnable template names b1/b2).
    prog.commit_induced_rule("reach(X, Y) :- edge(X, Z), edge(Z, Y).")

    prog.evaluate()

    assert prog.fact_exists("reach", [1, 3]), "reach(1,3) should be derived post-commit"


def test_ilp_schema_size_mismatch_rejected():
    """schema_size passed to set_rule_mask must match compiled schema_size."""
    source = """
        edge(1, 2). edge(2, 3).
        learnable(W) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()

    W = torch.zeros((n, n, n), device='cuda')
    flat = W.contiguous().view(-1)

    # Correct size should work
    prog.set_rule_mask("W", flat, flat, n)

    # Wrong size should raise
    with pytest.raises(ValueError, match="schema_size mismatch"):
        prog.set_rule_mask("W", flat, flat, n + 1)


def test_ilp_projected_credit_path():
    """tagged_entries_containing_fact must use head_projection on join output."""
    source = """
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W_mask) :: reach(X, Y) :- body1(X, Z), body2(Z, Y).
    """
    prog = pyxlog.IlpProgramFactory.compile(source, device=0, memory_mb=512)
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()

    # Build a mask that maps body1→edge, body2→edge (should derive reach(1,3) etc.)
    edge_idx = rel_names.index("edge")
    M_hard = torch.zeros((n, n, n), device='cuda')
    M_soft = torch.zeros((n, n, n), device='cuda')
    M_hard[edge_idx, edge_idx, rel_names.index("reach")] = 1.0
    M_soft[edge_idx, edge_idx, rel_names.index("reach")] = 1.0

    prog.set_rule_mask("W_mask",
                        M_hard.contiguous().view(-1),
                        M_soft.contiguous().view(-1), n)
    prog.evaluate()

    # The credit path should find (1,3) via projected columns, not raw join
    contributing = prog.tagged_entries_containing_fact("reach", [1, 3])
    assert len(contributing) > 0, \
        "Projected credit path should find reach(1,3) from edge⋈edge"
