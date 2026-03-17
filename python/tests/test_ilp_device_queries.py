import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

from conftest import skip_unless_pyxlog_cuda

skip_unless_pyxlog_cuda()


REACH_SOURCE = """
    edge(1, 2). edge(2, 3). edge(3, 4).
    learnable(W_reach) :: reach(X, Y) :- b1(X, Z), b2(Z, Y).
"""


def _compile_reach_prog():
    prog = pyxlog.IlpProgramFactory.compile(
        REACH_SOURCE, device=0, memory_mb=64,
    )
    n = prog.ilp_schema_size()
    rel_names = prog.ilp_relation_names()
    k_reach = rel_names.index("reach")
    i_edge = rel_names.index("edge")
    mask_hard = torch.zeros(n, n, n, device="cuda")
    mask_soft = torch.zeros(n, n, n, device="cuda")
    mask_hard[i_edge, i_edge, k_reach] = 1.0
    mask_soft[i_edge, i_edge, k_reach] = 1.0
    prog.set_rule_mask("W_reach", mask_hard.view(-1), mask_soft.view(-1), n)
    prog.evaluate()
    return prog


def test_batch_fact_membership_device_returns_gpu_mask():
    prog = _compile_reach_prog()

    prog.reset_host_transfer_stats()
    mask_dl = prog.batch_fact_membership_device("edge", [[1, 2], [9, 9], [2, 3]])
    stats = prog.host_transfer_stats()

    mask = torch.from_dlpack(mask_dl)
    assert mask.device.type == "cuda"
    assert mask.dtype == torch.bool
    assert mask.cpu().tolist() == [True, False, True]
    assert stats["dtoh_calls"] == 0
    assert stats["dtoh_bytes"] == 0


def test_batch_tagged_credit_device_matches_host_credit():
    prog = _compile_reach_prog()
    facts = [[1, 3], [2, 4], [99, 99]]
    expected = prog.batch_tagged_credit("reach", facts)

    prog.reset_host_transfer_stats()
    credit = prog.batch_tagged_credit_device("reach", facts)
    stats = prog.host_transfer_stats()

    row_offsets = torch.from_dlpack(credit.fact_row_offsets).cpu().tolist()
    entry_indices = torch.from_dlpack(credit.entry_indices).cpu().tolist()
    entry_i = torch.from_dlpack(credit.entry_i).cpu().tolist()
    entry_j = torch.from_dlpack(credit.entry_j).cpu().tolist()
    entry_k = torch.from_dlpack(credit.entry_k).cpu().tolist()

    nnz = row_offsets[-1]
    actual = []
    for start, end in zip(row_offsets[:-1], row_offsets[1:]):
        actual.append(
            [
                (entry_i[idx], entry_j[idx], entry_k[idx])
                for idx in entry_indices[start:end]
            ]
        )

    assert actual == expected
    assert stats["dtoh_calls"] == 0
    assert stats["dtoh_bytes"] == 0
