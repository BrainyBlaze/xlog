from pathlib import Path

from pyxlog import Program


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    for path in sorted((root / "examples" / "prob").glob("*.xlog")):
        print(f"[PROB] {path}")
        source = path.read_text()
        prog = Program.compile(source, device=0, memory_mb=1024)
        result = prog.evaluate(return_grads=False)
        print(f"atoms={len(result.atoms)} approx={result.approx}")


if __name__ == "__main__":
    main()
