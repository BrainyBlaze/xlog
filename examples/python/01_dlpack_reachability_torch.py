import torch

from xlog_gpu import LogicProgram


def main() -> None:
    source = """
pred edge(i64, i64).
pred reach(i64, i64).

reach(X, Y) :- edge(X, Y).
reach(X, Z) :- reach(X, Y), edge(Y, Z).

?- reach(X, Y).
""".strip()

    prog = LogicProgram.compile(source, device=0, memory_mb=1024)

    # Input relation: edge(src, dst)
    edge_src = torch.tensor([1, 2], device="cuda", dtype=torch.int64)
    edge_dst = torch.tensor([2, 3], device="cuda", dtype=torch.int64)

    result = prog.evaluate(
        {
            "edge": [edge_src, edge_dst],
        }
    )

    q0 = result.queries[0]
    out_src = torch.utils.dlpack.from_dlpack(q0.tensors[0])
    out_dst = torch.utils.dlpack.from_dlpack(q0.tensors[1])

    pairs = torch.stack([out_src, out_dst], dim=1).cpu().tolist()
    pairs.sort()

    print("reach(X, Y) rows:", pairs)


if __name__ == "__main__":
    main()

