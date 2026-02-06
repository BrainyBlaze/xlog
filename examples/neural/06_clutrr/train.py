import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "clutrr"
MANIFEST = ROOT / "dataset.json"
LABELS = ["parent", "child", "spouse", "sibling"]


def build_vocab(texts, max_tokens=10000):
    vocab = {"<pad>": 0}
    for text in texts:
        for token in text.lower().split():
            if token not in vocab and len(vocab) < max_tokens:
                vocab[token] = len(vocab)
    return vocab


def encode(text, vocab, max_len=64):
    tokens = [vocab.get(t, 0) for t in text.lower().split()[:max_len]]
    return tokens + [0] * (max_len - len(tokens))


def load_clutrr(mode: str):
    data_file = DATA / "train.jsonl"
    if not data_file.exists():
        raise SystemExit(
            "CLUTRR dataset missing. Place train.jsonl under examples/neural/06_clutrr/data/clutrr"
        )

    manifest = DatasetManifest.load(MANIFEST)
    limit = manifest.ci_subset.get("train", 64) if mode == "ci" else 2048

    rows = []
    with data_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            story = item.get("story") or item.get("text") or ""
            label = item.get("target_relation") or item.get("label") or ""
            if label not in LABELS:
                continue
            if not story:
                continue
            rows.append((story, label))
            if len(rows) >= limit:
                break

    if not rows:
        raise SystemExit("CLUTRR dataset missing: no rows with supported labels found")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    rows = load_clutrr(args.mode)
    import pyxlog
    import torch

    from model import RelNet

    texts = [x[0] for x in rows]
    labels = [x[1] for x in rows]

    vocab = build_vocab(texts)
    inputs = torch.tensor([encode(t, vocab) for t in texts], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = inputs.to(device)

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    net = RelNet(len(vocab), len(LABELS)).to(device)
    program.register_network("rel_net", net, torch.optim.Adam(net.parameters(), lr=1e-3))
    program.add_tensor_source("train", inputs)

    queries = [f"rel({i}, {labels[i]})" for i in range(len(labels))]
    for _ in range(args.epochs):
        program.train_epoch(queries, batch_size=min(16, len(queries)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
