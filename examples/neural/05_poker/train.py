import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "cards"
MANIFEST = ROOT / "dataset.json"

RANKS = ["2", "3", "4", "5", "6", "7", "8", "9", "10", "J", "Q", "K", "A"]
SUITS = ["C", "D", "H", "S"]
RANK_ATOMS = {
    "2": "r2",
    "3": "r3",
    "4": "r4",
    "5": "r5",
    "6": "r6",
    "7": "r7",
    "8": "r8",
    "9": "r9",
    "10": "r10",
    "J": "rj",
    "Q": "rq",
    "K": "rk",
    "A": "ra",
}
SUIT_ATOMS = {"C": "c", "D": "d", "H": "h", "S": "s"}


def load_cards(mode: str):
    if not DATA.exists():
        raise SystemExit(
            "Card dataset missing. Place under examples/neural/05_poker/data/cards"
        )

    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")
    import torch

    manifest = DatasetManifest.load(MANIFEST)
    limit = manifest.ci_subset.get("train", 64) if mode == "ci" else 2048

    images = []
    labels = []
    for img in sorted(DATA.glob("**/*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        name = img.stem.upper()
        rank = name[:-1]
        suit = name[-1]
        if rank not in RANKS or suit not in SUITS:
            continue
        images.append(Image.open(img).convert("RGB"))
        labels.append((RANK_ATOMS[rank], SUIT_ATOMS[suit]))
        if len(images) >= limit:
            break

    if not images:
        raise SystemExit("Card dataset missing: no parsable card images found")

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    tensor = torch.stack([transform(im) for im in images])
    return tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    images, labels = load_cards(args.mode)
    import pyxlog
    import torch

    from model import CardNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    rank_net = CardNet(len(RANKS)).to(device)
    suit_net = CardNet(len(SUITS)).to(device)
    program.register_network(
        "rank_net", rank_net, torch.optim.Adam(rank_net.parameters(), lr=1e-3)
    )
    program.register_network(
        "suit_net", suit_net, torch.optim.Adam(suit_net.parameters(), lr=1e-3)
    )

    program.add_tensor_source("train", images)
    queries = []
    for i, (rank, suit) in enumerate(labels):
        queries.append(f"rank({i}, {rank})")
        queries.append(f"suit({i}, {suit})")

    for _ in range(args.epochs):
        program.train_epoch(queries, batch_size=min(16, len(queries)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
