import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
MANIFEST = ROOT / "dataset.json"


def load_dataset(mode: str):
    from torchvision import datasets, transforms

    manifest = DatasetManifest.load(MANIFEST)
    train_dir = DATA / "coins" / "train"
    test_dir = DATA / "coins" / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise SystemExit(
            "Dataset missing: examples/neural/02_coins/data/coins with train/ and test/"
        )

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

    return train_ds, test_ds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train_ds, _ = load_dataset(args.mode)

    import pyxlog
    import torch

    from model import CoinNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())

    net = CoinNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    program.register_network("coin_net", net, opt)

    images = torch.stack([x for x, _ in train_ds]).to(device)
    labels = [train_ds.classes[y] for _, y in train_ds.samples]
    program.add_tensor_source("train", images)

    queries = [f"coin({i}, {labels[i]})" for i in range(len(labels))]
    for _ in range(args.epochs):
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
