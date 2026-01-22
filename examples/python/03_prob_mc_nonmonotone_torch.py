import torch

from pyxlog import Program


def main() -> None:
    source = r"""
    #pragma prob_engine = mc

    0.5::flip().

    p() :- flip().
    q() :- not p().
    p() :- not q().

    query(p()).
    query(flip()).
    """.strip()

    prog = Program.compile(source, device=0, memory_mb=1024)
    result = prog.evaluate(
        return_grads=False,
        samples=100_000,
        seed=123,
        confidence=0.95,
        max_nonmonotone_iterations=256,
    )

    assert result.approx
    assert result.stderr is not None
    assert result.ci_low is not None
    assert result.ci_high is not None

    probs = torch.utils.dlpack.from_dlpack(result.prob).cpu().tolist()
    log_probs = torch.utils.dlpack.from_dlpack(result.log_prob).cpu().tolist()
    stderrs = torch.utils.dlpack.from_dlpack(result.stderr).cpu().tolist()
    ci_lows = torch.utils.dlpack.from_dlpack(result.ci_low).cpu().tolist()
    ci_highs = torch.utils.dlpack.from_dlpack(result.ci_high).cpu().tolist()

    print(
        f"samples={result.samples} evidence_samples={result.evidence_samples} seed={result.seed} confidence={result.confidence}"
    )
    print("nonmonotone_semantics:", result.nonmonotone_semantics)

    for atom, p, lp, se, lo, hi in zip(
        result.atoms, probs, log_probs, stderrs, ci_lows, ci_highs
    ):
        print(
            f"{atom}: prob={p:.6f} log_prob={lp:.6f} stderr={se:.6f} CI[{lo:.6f}, {hi:.6f}]"
        )


if __name__ == "__main__":
    main()

