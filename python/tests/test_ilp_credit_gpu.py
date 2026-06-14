"""Tests for GPU-resident ILP credit/loss path (compute_ilp_loss_grad_gpu)."""
import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda
skip_unless_pyxlog_cuda()


def _compile_reach():
    """Compile a minimal ILP program with an edge relation and learnable reach rule.

    The correct candidate for bL=edge, bR=edge -> reach is derived from
    ilp_relation_names() because meta relations may be inserted ahead of reach.
    Must set a rule mask + re-evaluate to populate tagged results.
    """
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    # Activate bL=edge, bR=edge -> reach via the current relation index map.
    N = prog.ilp_schema_size()
    device = torch.device("cuda:0")
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    mask = torch.zeros(N ** 3, device=device, dtype=torch.float32)
    mask[edge_idx * N * N + edge_idx * N + reach_idx] = 1.0
    prog.set_rule_mask("W", mask, mask, N)
    prog.evaluate()
    return prog


def _edge_edge_reach_candidate(prog):
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    reach_idx = names.index("reach")
    return edge_idx, reach_idx


def _compile_reach_with_uploaded_relation():
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32).
        pred reach(u32, u32).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    assert hasattr(prog, "put_relation"), (
        "CompiledIlpProgram.put_relation must exist for relation-native strict training"
    )
    prog.put_relation(
        "edge",
        [
            torch.tensor([1, 2, 3], device="cuda", dtype=torch.int32),
            torch.tensor([2, 3, 4], device="cuda", dtype=torch.int32),
        ],
    )
    prog.evaluate()

    n = prog.ilp_schema_size()
    device = torch.device("cuda:0")
    mask = torch.zeros(n ** 3, device=device, dtype=torch.float32)
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    reach_idx = names.index("reach")
    mask[edge_idx * n * n + edge_idx * n + reach_idx] = 1.0
    prog.set_rule_mask("W", mask, mask, n)
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
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    candidates = [(edge_idx, edge_idx, reach_idx)]
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32)

    # reach(1,3) is derivable via edge(1,2)+edge(2,3)
    positives = [("reach", [1, 3])]
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


def test_compute_ilp_loss_grad_gpu_relations_matches_host_fact_path():
    prog = _compile_reach_with_uploaded_relation()
    compute_rel = getattr(prog, "compute_ilp_loss_grad_gpu_relations", None)
    assert compute_rel is not None, (
        "CompiledIlpProgram.compute_ilp_loss_grad_gpu_relations must exist for relation-native training"
    )
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    reach_idx = names.index("reach")
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]
    positive_relations = {
        "reach": [
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([3], device="cuda", dtype=torch.int32),
        ],
    }
    negative_relations = {
        "reach": [
            torch.tensor([1], device="cuda", dtype=torch.int32),
            torch.tensor([4], device="cuda", dtype=torch.int32),
        ],
    }

    host_loss_dl, host_grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives,
        negatives,
        cand_probs,
    )
    relation_loss_dl, relation_grad_dl = compute_rel(
        positive_relations,
        negative_relations,
        cand_probs,
    )

    host_loss = torch.from_dlpack(host_loss_dl)
    host_grad = torch.from_dlpack(host_grad_dl)
    relation_loss = torch.from_dlpack(relation_loss_dl)
    relation_grad = torch.from_dlpack(relation_grad_dl)

    assert torch.allclose(relation_loss, host_loss, atol=1e-6)
    assert torch.allclose(relation_grad, host_grad, atol=1e-5)


def test_compute_ilp_loss_grad_gpu_empty_facts():
    """Empty positive + negative lists should return zero loss and zero grad."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

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
        prog.compute_ilp_loss_grad_gpu([("reach", [1, 3])], [], cand_probs)


def test_compute_ilp_loss_grad_gpu_loss_positive():
    """Positive loss should be > 0 and grad should be non-zero for derivable facts."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.3], device=device, dtype=torch.float32)

    # reach(1,3) is derivable: loss = -log(0.3) > 0
    positives = [("reach", [1, 3])]
    negatives = []

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    # Loss should be positive (since -log(0.3) > 0) and finite
    assert loss.item() > 0.0
    assert torch.isfinite(loss).item()
    # Grad should be non-zero and finite
    assert torch.all(torch.isfinite(grad)).item()
    assert not torch.all(grad == 0.0).item()


def test_compute_ilp_loss_grad_gpu_f64():
    """F64 dtype is correctly handled for cand_probs."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float64)

    positives = [("reach", [1, 3])]
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
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([
        (edge_idx, edge_idx, reach_idx),
        (reach_idx, edge_idx, reach_idx),
    ])  # 2 candidates

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)  # 1 element

    with pytest.raises(Exception, match="length"):
        prog.compute_ilp_loss_grad_gpu([("reach", [1, 3])], [], cand_probs)


def test_compute_ilp_loss_grad_gpu_negative_facts():
    """Negative facts should contribute loss = -log(1 - credit)."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.8], device=device, dtype=torch.float32)

    # reach(1,4) — not derivable in one hop, so credit = 0 → loss = -log(1-0) = 0
    # reach(1,3) — derivable, credit = 0.8 as negative → loss = -log(1-0.8) = -log(0.2)
    positives = []
    negatives = [("reach", [1, 3])]

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    # loss = -log(1 - 0.8) = -log(0.2) ≈ 1.609
    assert loss.item() > 1.0
    assert torch.isfinite(loss).item()
    # Grad for negative: +1/(1-sum) = 1/0.2 = 5.0
    assert grad[0].item() > 0.0


# ---------------------------------------------------------------------------
# Gradient parity tests: GPU versus Python reference
# ---------------------------------------------------------------------------

def _reference_loss_grad(prog, cand_probs, ijk_to_cidx, positives, negatives):
    """Pure-PyTorch reference for ILP loss + gradient.

    Uses ``tagged_entries_containing_fact`` to discover which (i,j,k) triples
    contribute to each fact, then accumulates the standard surrogate loss:

        positive: -log(clamp(credit, 1e-8))
        negative: -log(clamp(1 - credit, 1e-8))

    Returns (loss_scalar, grad_tensor) both on CPU as plain floats/tensors
    so comparisons are easy.
    """
    # Clone with grad so we can .backward()
    p = cand_probs.detach().clone().requires_grad_(True)
    loss = torch.tensor(0.0, device=p.device, dtype=p.dtype)

    for rel_name, values in positives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        credit = sum(
            (p[ijk_to_cidx[tuple(ijk)]] for ijk in contributing if tuple(ijk) in ijk_to_cidx),
            torch.tensor(0.0, device=p.device, dtype=p.dtype),
        )
        loss = loss + (-torch.log(credit.clamp(min=1e-8)))

    for rel_name, values in negatives:
        contributing = prog.tagged_entries_containing_fact(rel_name, values)
        credit = sum(
            (p[ijk_to_cidx[tuple(ijk)]] for ijk in contributing if tuple(ijk) in ijk_to_cidx),
            torch.tensor(0.0, device=p.device, dtype=p.dtype),
        )
        loss = loss + (-torch.log((1.0 - credit).clamp(min=1e-8)))

    loss.backward()
    return loss.detach(), p.grad.detach()


def test_loss_parity_f32_reach():
    """GPU loss/grad matches Python reference (f32, single candidate, pos+neg)."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    candidates = [(edge_idx, edge_idx, reach_idx)]
    ijk_to_cidx = {candidates[0]: 0}
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # --- reference ---
    ref_loss, ref_grad = _reference_loss_grad(
        prog, cand_probs, ijk_to_cidx, positives, negatives,
    )

    # --- GPU kernel ---
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs,
    )
    gpu_loss = torch.from_dlpack(loss_dl)
    gpu_grad = torch.from_dlpack(grad_dl)

    assert torch.allclose(gpu_loss, ref_loss, atol=1e-6), (
        f"loss mismatch: gpu={gpu_loss.item()}, ref={ref_loss.item()}"
    )
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-5), (
        f"grad mismatch: gpu={gpu_grad}, ref={ref_grad}"
    )


def test_loss_parity_f64_reach():
    """GPU loss/grad matches Python reference (f64, tighter tolerance)."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    candidates = [(edge_idx, edge_idx, reach_idx)]
    ijk_to_cidx = {candidates[0]: 0}
    prog.set_candidate_map(candidates)

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float64)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # --- reference ---
    ref_loss, ref_grad = _reference_loss_grad(
        prog, cand_probs, ijk_to_cidx, positives, negatives,
    )

    # --- GPU kernel ---
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs,
    )
    gpu_loss = torch.from_dlpack(loss_dl)
    gpu_grad = torch.from_dlpack(grad_dl)

    assert torch.allclose(gpu_loss, ref_loss, atol=1e-12), (
        f"loss mismatch: gpu={gpu_loss.item()}, ref={ref_loss.item()}"
    )
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-10), (
        f"grad mismatch: gpu={gpu_grad}, ref={ref_grad}"
    )


def test_loss_parity_multi_candidate():
    """GPU loss/grad matches Python reference with multiple candidates."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred link(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4). edge(1, 4).
        link(1, 2). link(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    # Schema: ['edge'(0), 'link'(1), 'reach'(2), 'bL'(3), 'bR'(4)]  (N=5)
    N = prog.ilp_schema_size()
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    link_idx = names.index("link")
    reach_idx = names.index("reach")

    # Two candidates that both derive reach(1,3):
    #   edge+edge→reach and link+edge→reach
    cand_a = (edge_idx, edge_idx, reach_idx)
    cand_b = (link_idx, edge_idx, reach_idx)

    # Set mask enabling both candidates, then re-evaluate
    device = torch.device("cuda:0")
    mask = torch.zeros(N ** 3, device=device, dtype=torch.float32)
    for i, j, k in [cand_a, cand_b]:
        mask[i * N * N + j * N + k] = 1.0
    prog.set_rule_mask("W", mask, mask, N)
    prog.evaluate()

    candidates = [cand_a, cand_b]
    ijk_to_cidx = {cand_a: 0, cand_b: 1}
    prog.set_candidate_map(candidates)

    cand_probs = torch.tensor([0.5, 0.3], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = []

    # --- reference ---
    ref_loss, ref_grad = _reference_loss_grad(
        prog, cand_probs, ijk_to_cidx, positives, negatives,
    )

    # --- GPU kernel ---
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs,
    )
    gpu_loss = torch.from_dlpack(loss_dl)
    gpu_grad = torch.from_dlpack(grad_dl)

    assert torch.allclose(gpu_loss, ref_loss, atol=1e-6), (
        f"loss mismatch: gpu={gpu_loss.item()}, ref={ref_loss.item()}"
    )
    assert torch.allclose(gpu_grad, ref_grad, atol=1e-5), (
        f"grad mismatch: gpu={gpu_grad}, ref={ref_grad}"
    )


# ---------------------------------------------------------------------------
# Device-to-host accounting test
# ---------------------------------------------------------------------------

def test_zero_device_to_host_transfers():
    """compute_ilp_loss_grad_gpu must not cause additional device-to-host column transfers."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Column-level gate (coarse)
    prog.reset_d2h_transfer_count()
    # Byte-level gate (strict)
    prog.reset_host_transfer_stats()

    prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)

    # Coarse gate
    assert prog.d2h_transfer_count() == 0, (
        f"compute_ilp_loss_grad_gpu caused {prog.d2h_transfer_count()} device-to-host column transfers; expected 0"
    )
    # Strict gate
    stats = prog.host_transfer_stats()
    assert stats['dtoh_calls'] == 0, (
        f"compute_ilp_loss_grad_gpu caused {stats['dtoh_calls']} device-to-host calls; expected 0"
    )
    assert stats['dtoh_bytes'] == 0, (
        f"compute_ilp_loss_grad_gpu transferred {stats['dtoh_bytes']} device-to-host bytes; expected 0"
    )


def test_zero_device_to_host_strict():
    """compute_ilp_loss_grad_gpu must cause zero device-to-host transfers."""
    prog = _compile_reach()
    edge_idx, reach_idx = _edge_edge_reach_candidate(prog)
    prog.set_candidate_map([(edge_idx, edge_idx, reach_idx)])

    device = torch.device("cuda:0")
    cand_probs = torch.tensor([0.7], device=device, dtype=torch.float32)

    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Reset byte-level transfer stats, then call GPU loss/grad path
    prog.reset_host_transfer_stats()
    prog.compute_ilp_loss_grad_gpu(positives, negatives, cand_probs)
    stats = prog.host_transfer_stats()

    assert stats['dtoh_calls'] == 0, (
        f"compute_ilp_loss_grad_gpu caused {stats['dtoh_calls']} device-to-host calls; expected 0"
    )
    assert stats['dtoh_bytes'] == 0, (
        f"compute_ilp_loss_grad_gpu transferred {stats['dtoh_bytes']} device-to-host bytes; expected 0"
    )


# ---------------------------------------------------------------------------
# COO memory cap and chunked fallback
# ---------------------------------------------------------------------------

def test_compute_ilp_loss_grad_gpu_memory_cap():
    """Force chunked COO assembly by setting a tiny memory cap.

    Uses the multi-candidate setup from test_loss_parity_multi_candidate
    so that facts are covered by multiple candidates. With a 1-byte cap,
    each task goes into its own chunk. The chunked path must merge all
    chunk COO entries before building a single CSR and running
    forward/backward, otherwise -log(a+b) != -log(a) + -log(b).
    """
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred link(u32, u32).
        edge(1, 2). edge(2, 3). edge(3, 4). edge(1, 4).
        link(1, 2). link(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    N = prog.ilp_schema_size()
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    link_idx = names.index("link")
    reach_idx = names.index("reach")

    cand_a = (edge_idx, edge_idx, reach_idx)
    cand_b = (link_idx, edge_idx, reach_idx)

    device = torch.device("cuda:0")
    mask = torch.zeros(N ** 3, device=device, dtype=torch.float32)
    for i, j, k in [cand_a, cand_b]:
        mask[i * N * N + j * N + k] = 1.0
    prog.set_rule_mask("W", mask, mask, N)
    prog.evaluate()

    prog.set_candidate_map([cand_a, cand_b])
    # Use probs that don't sum to 1 so NLL > 0 (both cands derive reach(1,3),
    # so coverage = 0.3+0.4 = 0.7, NLL = -log(0.7) > 0).
    cand_probs = torch.tensor([0.3, 0.4], device=device, dtype=torch.float32)
    positives = [("reach", [1, 3])]
    negatives = [("reach", [1, 4])]

    # Reference: non-chunked path (default 16 MB cap)
    loss_ref_dl, grad_ref_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss_ref = torch.from_dlpack(loss_ref_dl).clone()
    grad_ref = torch.from_dlpack(grad_ref_dl).clone()

    # Force chunking: 1-byte cap means each task is its own chunk
    prog.set_coo_chunk_budget(1)

    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        positives, negatives, cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    grad = torch.from_dlpack(grad_dl)

    assert loss.item() > 0.0
    assert torch.isfinite(loss).item()
    assert torch.all(torch.isfinite(grad)).item()
    assert torch.allclose(loss, loss_ref, atol=1e-5), (
        f"chunked loss={loss.item()} != ref={loss_ref.item()}"
    )
    assert torch.allclose(grad, grad_ref, atol=1e-5), (
        f"chunked grad={grad} != ref={grad_ref}"
    )


def test_strict_zero_device_to_host_chunked_passes():
    """Chunked path now stays GPU-only under strict zero device-to-host mode."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred reach(u32, u32).
        edge(1, 2). edge(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    prog.evaluate()

    N = prog.ilp_schema_size()
    names = prog.ilp_relation_names()
    edge_idx = names.index("edge")
    reach_idx = names.index("reach")

    cand = (edge_idx, edge_idx, reach_idx)
    device = torch.device("cuda:0")
    mask = torch.zeros(N ** 3, device=device, dtype=torch.float32)
    mask[edge_idx * N * N + edge_idx * N + reach_idx] = 1.0
    prog.set_rule_mask("W", mask, mask, N)
    prog.evaluate()

    prog.set_candidate_map([cand])
    cand_probs = torch.tensor([0.5], device=device, dtype=torch.float32)

    # Enable strict mode + tiny budget to force chunking
    prog.set_strict_zero_dtoh(True)
    prog.set_coo_chunk_budget(1)  # Force chunking

    # Reset stats, run loss/grad, verify zero tracked device-to-host transfers.
    prog.reset_host_transfer_stats()
    loss_dl, grad_dl = prog.compute_ilp_loss_grad_gpu(
        [("reach", [1, 3])], [("reach", [1, 2])], cand_probs
    )
    loss = torch.from_dlpack(loss_dl)
    assert torch.isfinite(loss).item(), f"Loss not finite: {loss.item()}"

    stats = prog.host_transfer_stats()
    assert stats['dtoh_calls'] == 0, (
        f"Chunked path caused {stats['dtoh_calls']} tracked device-to-host calls; expected 0"
    )
    assert stats['dtoh_bytes'] == 0, (
        f"Chunked path transferred {stats['dtoh_bytes']} tracked device-to-host bytes; expected 0"
    )


def test_deprecated_set_coo_memory_cap():
    """Deprecated alias still works."""
    prog = pyxlog.IlpProgramFactory.compile("""
        pred edge(u32, u32). pred reach(u32, u32).
        edge(1, 2). edge(2, 3).
        learnable(W) :: reach(X, Y) :- bL(X, Z), bR(Z, Y).
    """, device=0, memory_mb=64)
    # Should not raise — alias forwards to set_coo_chunk_budget
    prog.set_coo_memory_cap(1024)
