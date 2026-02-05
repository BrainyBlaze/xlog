import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd, cwd=None):
    print("+", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=cwd)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    args = parser.parse_args()

    # Deterministic/probabilistic XLOG examples
    run_cmd(["bash", "scripts/run_xlog_examples.sh"], cwd=ROOT)

    # Probabilistic .xlog examples
    for xlog in (ROOT / "examples/prob").glob("*.xlog"):
        run_cmd(["cargo", "run", "-q", "-p", "xlog-cli", "--", "prob", str(xlog)])

    # Python examples (DLPack, etc.)
    for py in (ROOT / "examples/python").glob("*.py"):
        run_cmd([sys.executable, str(py)])

    # Neural examples (defer to per-example train.py)
    for example in (ROOT / "examples/neural").iterdir():
        train = example / "train.py"
        if train.exists():
            run_cmd([sys.executable, str(train), "--mode", args.mode], cwd=example)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
