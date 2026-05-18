import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest
from scripts.neural_training import (
    classification_accuracy,
    neural_fixture_smoke_enabled,
    report_and_enforce_metric,
    resolve_epochs,
    resolve_min_accuracy,
    set_seed,
    split_indices,
    write_frozen_metrics,
)

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


def _fixture_clutrr(mode: str):
    manifest = DatasetManifest.load(MANIFEST)
    if mode == "ci":
        n = manifest.ci_subset.get("train", 64)
    else:
        n = 64
    rows = []
    for i in range(n):
        label = LABELS[i % len(LABELS)]
        story = f"{label} relation fixture person{i} token_{label} token_{label}"
        rows.append((story, label))
    return manifest, rows


def load_clutrr(mode: str):
    data_file = DATA / "train.jsonl"
    if not data_file.exists():
        if neural_fixture_smoke_enabled():
            print("INFO: using synthetic CLUTRR fixture; dataset is unavailable")
            return _fixture_clutrr(mode)
        raise SystemExit(
            "CLUTRR dataset missing. Place train.jsonl under examples/neural/06_clutrr/data/clutrr"
        )

    manifest = DatasetManifest.load(MANIFEST)
    if mode == "ci":
        limit = manifest.ci_subset.get("train", 64)
    elif mode == "dev":
        limit = 2048
    else:
        limit = None

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
            if limit is not None and len(rows) >= limit:
                break

    if not rows:
        raise SystemExit("CLUTRR dataset missing: no rows with supported labels found")

    return manifest, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    manifest, rows = load_clutrr(args.mode)
    epochs = resolve_epochs(args.mode, args.epochs, ci=2, dev=10, release=20)

    import pyxlog
    import torch

    from model import RelNet

    train_idx, eval_idx = split_indices(len(rows), args.eval_ratio, seed=args.seed)
    train_rows = [rows[int(i)] for i in train_idx.tolist()]
    eval_rows = [rows[int(i)] for i in eval_idx.tolist()]
    if not eval_rows:
        eval_rows = train_rows

    train_texts = [x[0] for x in train_rows]
    train_labels = [x[1] for x in train_rows]
    eval_texts = [x[0] for x in eval_rows]
    eval_labels = [x[1] for x in eval_rows]

    vocab = build_vocab(train_texts)
    train_inputs = torch.tensor([encode(t, vocab) for t in train_texts], dtype=torch.long)
    eval_inputs = torch.tensor([encode(t, vocab) for t in eval_texts], dtype=torch.long)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_inputs = train_inputs.to(device)
    eval_inputs = eval_inputs.to(device)

    import time as _time

    t0 = _time.monotonic()
    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    compile_api_sec = _time.monotonic() - t0

    net = RelNet(len(vocab), len(LABELS)).to(device)
    program.register_network("rel_net", net, torch.optim.Adam(net.parameters(), lr=args.lr))
    program.add_tensor_source("train", train_inputs)

    queries = [f"rel({i}, {train_labels[i]})" for i in range(len(train_labels))]
    epoch_times = []
    train_start = _time.monotonic()
    for _ in range(epochs):
        ep_start = _time.monotonic()
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))
        epoch_times.append(_time.monotonic() - ep_start)
    total_train_sec = _time.monotonic() - train_start

    write_frozen_metrics(args.metrics_path, compile_api_sec, epoch_times, total_train_sec, len(queries))

    label_to_idx = {name: idx for idx, name in enumerate(LABELS)}
    train_targets = torch.tensor(
        [label_to_idx[name] for name in train_labels], dtype=torch.long, device=device
    )
    eval_targets = torch.tensor(
        [label_to_idx[name] for name in eval_labels], dtype=torch.long, device=device
    )
    train_acc = classification_accuracy(net, train_inputs, train_targets)
    eval_acc = classification_accuracy(net, eval_inputs, eval_targets)
    print(f"train_acc={train_acc:.4f} eval_acc={eval_acc:.4f} epochs={epochs}")

    threshold = resolve_min_accuracy(args.mode, manifest, args.min_accuracy)
    report_and_enforce_metric("eval_acc", eval_acc, threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
