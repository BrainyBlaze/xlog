import torch

from pyxlog import Program


def main() -> None:
    source = """
0.7::rain().
0.2::sprinkler().
wet() :- rain().
wet() :- sprinkler().
evidence(wet(), true).
query(rain()).
query(sprinkler()).
""".strip()

    prog = Program.compile(source, device=0, memory_mb=1024)
    result = prog.evaluate(return_grads=False)

    probs = torch.utils.dlpack.from_dlpack(result.prob).cpu().tolist()
    log_probs = torch.utils.dlpack.from_dlpack(result.log_prob).cpu().tolist()

    for atom, p, lp in zip(result.atoms, probs, log_probs):
        print(f"{atom}: prob={p:.8f} log_prob={lp:.8f}")


if __name__ == "__main__":
    main()

