import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest
from scripts.neural_training import (
    classification_accuracy,
    report_and_enforce_metric,
    resolve_epochs,
    resolve_min_accuracy,
    set_seed,
    write_frozen_metrics,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST = ROOT / "dataset.json"


def load_dataset(mode: str):
    manifest = DatasetManifest.load(MANIFEST)
    train_dir = DATA / "coins" / "train"
    test_dir = DATA / "coins" / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(
            "Dataset missing: examples/neural/02_coins/data/coins with train/ and test/"
        )

    try:
        from torchvision import datasets, transforms
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    test_ds = datasets.ImageFolder(test_dir, transform=transform)

    if mode == "ci":
        train_n = manifest.ci_subset.get("train", len(train_ds))
        test_n = manifest.ci_subset.get("test", len(test_ds))
        train_ds.samples = train_ds.samples[:train_n]
        train_ds.targets = train_ds.targets[:train_n]
        test_ds.samples = test_ds.samples[:test_n]
        test_ds.targets = test_ds.targets[:test_n]

    if len(train_ds) == 0:
        raise SystemExit("Dataset missing: coins train split is empty")

    return manifest, train_ds, test_ds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    args = parser.parse_args()

    manifest, train_ds, test_ds = load_dataset(args.mode)
    epochs = resolve_epochs(args.mode, args.epochs, ci=2, dev=12, release=40)
    set_seed(args.seed)

    import pyxlog
    import torch

    from model import CoinNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    import time as _time

    t0 = _time.monotonic()
    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    compile_api_sec = _time.monotonic() - t0

    net = CoinNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)
    program.register_network("coin_net", net, opt)

    train_images = torch.stack([x for x, _ in train_ds]).to(device)
    train_idx_labels = torch.tensor(train_ds.targets, dtype=torch.long, device=device)
    train_atom_labels = [train_ds.classes[int(y)] for y in train_ds.targets]
    test_images = torch.stack([x for x, _ in test_ds]).to(device)
    test_idx_labels = torch.tensor(test_ds.targets, dtype=torch.long, device=device)
    if int(test_idx_labels.numel()) == 0:
        raise SystemExit("Dataset missing: coins test split is empty")
    program.add_tensor_source("train", train_images)

    queries = [f"coin({i}, {train_atom_labels[i]})" for i in range(len(train_atom_labels))]
    epoch_times = []
    train_start = _time.monotonic()
    for _ in range(epochs):
        ep_start = _time.monotonic()
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))
        epoch_times.append(_time.monotonic() - ep_start)
    total_train_sec = _time.monotonic() - train_start

    write_frozen_metrics(args.metrics_path, compile_api_sec, epoch_times, total_train_sec, len(queries))

    train_acc = classification_accuracy(net, train_images, train_idx_labels)
    test_acc = classification_accuracy(net, test_images, test_idx_labels)
    print(f"train_acc={train_acc:.4f} test_acc={test_acc:.4f} epochs={epochs}")

    threshold = resolve_min_accuracy(args.mode, manifest, args.min_accuracy)
    report_and_enforce_metric("test_acc", test_acc, threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
