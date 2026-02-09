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
    split_indices,
)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "svhn"
MANIFEST = ROOT / "dataset.json"


def _read_h5_scalar(h5, obj):
    if hasattr(obj, "shape") and obj.shape == ():
        obj = obj[()]
    if isinstance(obj, (int, float)):
        return float(obj)
    if hasattr(obj, "shape") and obj.shape == (1, 1):
        return float(obj[0][0])
    return float(obj)


def _read_bbox_field_values(h5, bbox_group, field: str):
    ds = bbox_group[field]
    if ds.dtype == object:
        if ds.shape == (1, 1):
            return [_read_h5_scalar(h5, h5[ds[0][0]])]
        return [_read_h5_scalar(h5, h5[ds[i][0]]) for i in range(ds.shape[0])]
    if ds.shape == (1, 1):
        return [float(ds[0][0])]
    return [float(ds[i][0]) for i in range(ds.shape[0])]


def load_svhn_two_digit(mode: str):
    mat_path = DATA / "train" / "digitStruct.mat"
    if not mat_path.exists():
        raise SystemExit(
            "SVHN data missing. Place SVHN train/ with digitStruct.mat under examples/neural/03_mnist_multidigit/data/svhn"
        )

    try:
        import h5py
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")
    import torch

    manifest = DatasetManifest.load(MANIFEST)
    limit = manifest.ci_subset.get("train", 128) if mode == "ci" else 2048

    with h5py.File(mat_path, "r") as f:
        names = f["digitStruct"]["name"]
        bbox = f["digitStruct"]["bbox"]

        def get_name(idx: int) -> str:
            name_ref = names[idx][0]
            chars = f[name_ref][()]
            return "".join(chr(c[0]) for c in chars)

        left_digit_crops = []
        right_digit_crops = []
        labels = []
        for i in range(len(names)):
            img_path = DATA / "train" / get_name(i)
            if not img_path.exists():
                continue

            bbox_ref = bbox[i][0]
            bbox_group = f[bbox_ref]
            label_vals = [int(v) for v in _read_bbox_field_values(f, bbox_group, "label")]
            left_vals = _read_bbox_field_values(f, bbox_group, "left")
            top_vals = _read_bbox_field_values(f, bbox_group, "top")
            width_vals = _read_bbox_field_values(f, bbox_group, "width")
            height_vals = _read_bbox_field_values(f, bbox_group, "height")

            if not (
                len(label_vals)
                == len(left_vals)
                == len(top_vals)
                == len(width_vals)
                == len(height_vals)
            ):
                continue
            if len(label_vals) != 2:
                continue

            entries = []
            for label, left, top, width, height in zip(
                label_vals, left_vals, top_vals, width_vals, height_vals
            ):
                if label == 10:
                    label = 0
                entries.append((float(left), float(top), float(width), float(height), int(label)))
            entries.sort(key=lambda e: e[0])

            image = Image.open(img_path).convert("RGB")
            img_w, img_h = image.size

            def crop_digit(entry):
                left, top, width, height, label = entry
                x1 = max(0, int(round(left)) - 2)
                y1 = max(0, int(round(top)) - 2)
                x2 = min(img_w, int(round(left + width)) + 2)
                y2 = min(img_h, int(round(top + height)) + 2)
                if x2 <= x1 or y2 <= y1:
                    return None, None
                return image.crop((x1, y1, x2, y2)), label

            left_crop, left_label = crop_digit(entries[0])
            right_crop, right_label = crop_digit(entries[1])
            if left_crop is None or right_crop is None:
                continue

            left_digit_crops.append(left_crop)
            right_digit_crops.append(right_crop)
            labels.append((left_label, right_label))
            if len(labels) >= limit:
                break

    if not labels:
        raise SystemExit("SVHN data missing: no 2-digit samples found")

    transform = transforms.Compose(
        [
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ]
    )
    left_tensor = torch.stack([transform(im) for im in left_digit_crops])
    right_tensor = torch.stack([transform(im) for im in right_digit_crops])
    tensor = torch.cat([left_tensor, right_tensor], dim=1)
    return manifest, tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--min-accuracy", type=float, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    manifest, images, labels = load_svhn_two_digit(args.mode)
    epochs = resolve_epochs(args.mode, args.epochs, ci=2, dev=12, release=30)

    import pyxlog
    import torch

    from model import DigitNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)
    train_idx, eval_idx = split_indices(len(labels), args.eval_ratio, seed=args.seed)

    train_images = images[train_idx]
    eval_images = images[eval_idx]
    train_labels = [labels[int(i)] for i in train_idx.tolist()]
    eval_labels = [labels[int(i)] for i in eval_idx.tolist()]
    if not eval_labels:
        eval_images = train_images
        eval_labels = train_labels

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    left_net = DigitNet(side="left").to(device)
    right_net = DigitNet(side="right").to(device)
    program.register_network(
        "left_net", left_net, torch.optim.Adam(left_net.parameters(), lr=args.lr)
    )
    program.register_network(
        "right_net", right_net, torch.optim.Adam(right_net.parameters(), lr=args.lr)
    )

    program.add_tensor_source("train", train_images)
    queries = []
    for i, (left, right) in enumerate(train_labels):
        queries.append(f"digit_left({i}, {left})")
        queries.append(f"digit_right({i}, {right})")

    for _ in range(epochs):
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))

    train_left_targets = torch.tensor([l for l, _ in train_labels], dtype=torch.long, device=device)
    train_right_targets = torch.tensor([r for _, r in train_labels], dtype=torch.long, device=device)
    eval_left_targets = torch.tensor([l for l, _ in eval_labels], dtype=torch.long, device=device)
    eval_right_targets = torch.tensor([r for _, r in eval_labels], dtype=torch.long, device=device)

    train_left_acc = classification_accuracy(left_net, train_images, train_left_targets)
    train_right_acc = classification_accuracy(right_net, train_images, train_right_targets)
    eval_left_acc = classification_accuracy(left_net, eval_images, eval_left_targets)
    eval_right_acc = classification_accuracy(right_net, eval_images, eval_right_targets)
    eval_joint_proxy = 0.5 * (eval_left_acc + eval_right_acc)
    print(
        "train_left_acc={:.4f} train_right_acc={:.4f} eval_left_acc={:.4f} "
        "eval_right_acc={:.4f} eval_joint_proxy={:.4f} epochs={}".format(
            train_left_acc,
            train_right_acc,
            eval_left_acc,
            eval_right_acc,
            eval_joint_proxy,
            epochs,
        )
    )

    threshold = resolve_min_accuracy(args.mode, manifest, args.min_accuracy)
    report_and_enforce_metric("eval_joint_proxy", eval_joint_proxy, threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
