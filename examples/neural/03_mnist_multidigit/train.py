import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "svhn"
MANIFEST = ROOT / "dataset.json"


def _read_h5_scalar(h5, obj):
    if hasattr(obj, "shape") and obj.shape == ():
        obj = obj[()]
    if isinstance(obj, (int, float)):
        return int(obj)
    if hasattr(obj, "shape") and obj.shape == (1, 1):
        return int(obj[0][0])
    return int(obj)


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

        def get_labels(idx: int):
            bbox_ref = bbox[idx][0]
            label_ds = f[bbox_ref]["label"]
            if label_ds.shape == (1, 1):
                value = _read_h5_scalar(f, f[label_ds[0][0]])
                if value == 10:
                    value = 0
                return [value]
            out = []
            for i in range(label_ds.shape[0]):
                value = _read_h5_scalar(f, f[label_ds[i][0]])
                if value == 10:
                    value = 0
                out.append(value)
            return out

        images = []
        labels = []
        for i in range(len(names)):
            digits = get_labels(i)
            if len(digits) != 2:
                continue
            img_path = DATA / "train" / get_name(i)
            if not img_path.exists():
                continue
            images.append(Image.open(img_path).convert("RGB"))
            labels.append((digits[0], digits[1]))
            if len(images) >= limit:
                break

    if not images:
        raise SystemExit("SVHN data missing: no 2-digit samples found")

    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    tensor = torch.stack([transform(im) for im in images])
    return tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    images, labels = load_svhn_two_digit(args.mode)
    import pyxlog
    import torch

    from model import DigitNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    left_net = DigitNet().to(device)
    right_net = DigitNet().to(device)
    program.register_network(
        "left_net", left_net, torch.optim.Adam(left_net.parameters(), lr=1e-3)
    )
    program.register_network(
        "right_net", right_net, torch.optim.Adam(right_net.parameters(), lr=1e-3)
    )

    program.add_tensor_source("train", images)
    queries = []
    for i, (left, right) in enumerate(labels):
        queries.append(f"digit_left({i}, {left})")
        queries.append(f"digit_right({i}, {right})")

    for _ in range(args.epochs):
        program.train_epoch(queries, batch_size=min(16, len(queries)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
