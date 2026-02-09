import argparse
import random
import re
import sys
from collections import defaultdict
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
    split_indices,
)

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


def paired_split_indices(labels, seed: int):
    import torch

    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        groups[label].append(idx)

    rng = random.Random(seed)
    train_idx = []
    eval_idx = []
    for _, indices in groups.items():
        rng.shuffle(indices)
        if len(indices) >= 2:
            eval_idx.append(indices[0])
            train_idx.extend(indices[1:])
        else:
            train_idx.extend(indices)

    if not eval_idx:
        return split_indices(len(labels), eval_ratio=0.1, seed=seed)

    rng.shuffle(train_idx)
    rng.shuffle(eval_idx)
    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(eval_idx, dtype=torch.long),
    )


def parse_card_atoms(stem: str):
    token = stem.upper()
    token = token.split("_", 1)[0]
    token = token.split("-", 1)[0]
    token = re.sub(r"[^A-Z0-9]", "", token)
    if len(token) < 2:
        return None
    rank = token[:-1]
    suit = token[-1]
    if rank not in RANKS or suit not in SUITS:
        return None
    return RANK_ATOMS[rank], SUIT_ATOMS[suit]


def pick_epoch_indices(n_items: int, max_items: int, seed: int, epoch: int):
    if n_items <= 0:
        return []
    cap = max(1, int(max_items))
    if n_items <= cap:
        return list(range(n_items))
    rng = random.Random(int(seed) + 1009 * int(epoch))
    indices = list(range(n_items))
    rng.shuffle(indices)
    return indices[:cap]


def build_training_queries(train_labels, picked_indices, rank_weight: int):
    queries = []
    rank_repeat = max(1, int(rank_weight))
    for idx in picked_indices:
        rank, suit = train_labels[idx]
        for _ in range(rank_repeat):
            queries.append(f"rank({idx}, {rank})")
        queries.append(f"suit({idx}, {suit})")
    return queries


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

    transform = transforms.Compose(
        [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    tensors = []
    labels = []
    for img in sorted(DATA.glob("**/*")):
        if img.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        parsed = parse_card_atoms(img.stem)
        if parsed is None:
            continue
        with Image.open(img) as image:
            tensors.append(transform(image.convert("RGB")))
        labels.append(parsed)
        if len(tensors) >= limit:
            break

    if not tensors:
        raise SystemExit("Card dataset missing: no parsable card images found")

    tensor = torch.stack(tensors)
    return manifest, tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--queries-per-epoch", type=int, default=None)
    parser.add_argument("--rank-query-weight", type=int, default=1)
    parser.add_argument("--min-accuracy", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    manifest, images, labels = load_cards(args.mode)
    epochs = resolve_epochs(args.mode, args.epochs, ci=2, dev=20, release=60)

    import pyxlog
    import torch

    from model import CardNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    if args.mode == "release":
        train_idx, eval_idx = paired_split_indices(labels, seed=args.seed)
    else:
        train_idx, eval_idx = split_indices(len(labels), args.eval_ratio, seed=args.seed)
    train_images = images[train_idx]
    eval_images = images[eval_idx]
    train_labels = [labels[int(i)] for i in train_idx.tolist()]
    eval_labels = [labels[int(i)] for i in eval_idx.tolist()]
    if not eval_labels:
        eval_images = train_images
        eval_labels = train_labels

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    rank_net = CardNet(len(RANKS), focus="rank").to(device)
    suit_net = CardNet(len(SUITS), focus="suit").to(device)
    program.register_network(
        "rank_net", rank_net, torch.optim.Adam(rank_net.parameters(), lr=args.lr)
    )
    program.register_network(
        "suit_net", suit_net, torch.optim.Adam(suit_net.parameters(), lr=args.lr)
    )

    program.add_tensor_source("train", train_images)
    if args.queries_per_epoch is not None:
        query_examples = max(1, int(args.queries_per_epoch))
    elif args.mode == "ci":
        query_examples = 64
    elif args.mode == "dev":
        query_examples = 256
    else:
        query_examples = 512

    for epoch in range(epochs):
        picked = pick_epoch_indices(
            len(train_labels), query_examples, seed=args.seed, epoch=epoch
        )
        queries = build_training_queries(
            train_labels, picked, rank_weight=args.rank_query_weight
        )
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))

    rank_order = ["r2", "r3", "r4", "r5", "r6", "r7", "r8", "r9", "r10", "rj", "rq", "rk", "ra"]
    suit_order = ["c", "d", "h", "s"]
    rank_to_idx = {name: idx for idx, name in enumerate(rank_order)}
    suit_to_idx = {name: idx for idx, name in enumerate(suit_order)}

    train_rank_targets = torch.tensor(
        [rank_to_idx[r] for r, _ in train_labels], dtype=torch.long, device=device
    )
    train_suit_targets = torch.tensor(
        [suit_to_idx[s] for _, s in train_labels], dtype=torch.long, device=device
    )
    eval_rank_targets = torch.tensor(
        [rank_to_idx[r] for r, _ in eval_labels], dtype=torch.long, device=device
    )
    eval_suit_targets = torch.tensor(
        [suit_to_idx[s] for _, s in eval_labels], dtype=torch.long, device=device
    )

    train_rank_acc = classification_accuracy(rank_net, train_images, train_rank_targets)
    train_suit_acc = classification_accuracy(suit_net, train_images, train_suit_targets)
    eval_rank_acc = classification_accuracy(rank_net, eval_images, eval_rank_targets)
    eval_suit_acc = classification_accuracy(suit_net, eval_images, eval_suit_targets)
    train_joint_proxy = 0.5 * (train_rank_acc + train_suit_acc)
    eval_joint_proxy = 0.5 * (eval_rank_acc + eval_suit_acc)
    print(
        "train_rank_acc={:.4f} train_suit_acc={:.4f} eval_rank_acc={:.4f} "
        "eval_suit_acc={:.4f} train_joint_proxy={:.4f} eval_joint_proxy={:.4f} epochs={}".format(
            train_rank_acc,
            train_suit_acc,
            eval_rank_acc,
            eval_suit_acc,
            train_joint_proxy,
            eval_joint_proxy,
            epochs,
        )
    )

    threshold = resolve_min_accuracy(args.mode, manifest, args.min_accuracy)
    report_and_enforce_metric("eval_joint_proxy", eval_joint_proxy, threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
