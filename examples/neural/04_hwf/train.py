import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest
from scripts.neural_training import (
    class_pattern_tensor,
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
DATA = ROOT / "data" / "crohme"
MANIFEST = ROOT / "dataset.json"
LABEL_ORDER = ["add", "sub", "mul", "eq"]


def inkml_to_image(path: Path, size: int = 128):
    from PIL import Image, ImageDraw

    tree = ET.parse(path)
    root = tree.getroot()
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)

    for trace in root.findall(".//{*}trace"):
        if trace.text is None:
            continue
        points = []
        for raw_pt in trace.text.strip().split(","):
            coords = raw_pt.strip().split()
            if len(coords) < 2:
                continue
            try:
                x = float(coords[0])
                y = float(coords[1])
            except ValueError:
                continue
            points.append((x, y))
        if len(points) >= 2:
            draw.line(points, fill=0, width=2)

    return image


def label_from_latex(latex: str) -> str:
    if "=" in latex:
        return "eq"
    if "+" in latex:
        return "add"
    if "-" in latex:
        return "sub"
    return "mul"


def _fixture_crohme(mode: str):
    manifest = DatasetManifest.load(MANIFEST)
    if mode == "ci":
        n = manifest.ci_subset.get("train", 64)
    else:
        n = 64
    label_idx = [i % len(LABEL_ORDER) for i in range(n)]
    labels = [LABEL_ORDER[i] for i in label_idx]
    return manifest, class_pattern_tensor(label_idx, len(LABEL_ORDER), 1, 128, 128), labels


def load_crohme(mode: str):
    if not DATA.exists():
        if neural_fixture_smoke_enabled():
            print("INFO: using synthetic CROHME fixture; dataset is unavailable")
            return _fixture_crohme(mode)
        raise SystemExit(
            "CROHME dataset missing. Place under examples/neural/04_hwf/data/crohme"
        )

    try:
        from torchvision import transforms
    except ImportError as exc:
        if neural_fixture_smoke_enabled():
            print("INFO: using synthetic CROHME fixture; torchvision is unavailable")
            return _fixture_crohme(mode)
        raise SystemExit(f"Missing dependency: {exc}")
    import torch

    manifest = DatasetManifest.load(MANIFEST)
    if mode == "ci":
        limit = manifest.ci_subset.get("train", 64)
    elif mode == "dev":
        limit = 1024
    else:
        limit = None

    inkml_files = sorted(DATA.glob("**/*.inkml"))
    images = []
    labels = []
    for inkml_file in inkml_files:
        txt_label = inkml_file.with_suffix(".txt")
        if not txt_label.exists():
            continue
        latex = txt_label.read_text().strip()
        image = inkml_to_image(inkml_file)
        images.append(image)
        labels.append(label_from_latex(latex))
        if limit is not None and len(images) >= limit:
            break

    if not images:
        raise SystemExit("CROHME dataset missing: no labeled InkML samples found")

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    tensor = torch.stack([transform(im) for im in images])
    return manifest, tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-ratio", type=float, default=0.2)
    parser.add_argument("--min-accuracy", type=float, default=None)
    parser.add_argument("--metrics-path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    manifest, images, labels = load_crohme(args.mode)
    epochs = resolve_epochs(args.mode, args.epochs, ci=2, dev=12, release=30)

    import pyxlog
    import torch

    from model import HWFNet

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

    import time as _time

    t0 = _time.monotonic()
    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    compile_api_sec = _time.monotonic() - t0

    net = HWFNet().to(device)
    program.register_network("hw_net", net, torch.optim.Adam(net.parameters(), lr=args.lr))
    program.add_tensor_source("train", train_images)

    queries = [f"expr_type({i}, {train_labels[i]})" for i in range(len(train_labels))]
    epoch_times = []
    train_start = _time.monotonic()
    for _ in range(epochs):
        ep_start = _time.monotonic()
        program.train_epoch(queries, batch_size=min(args.batch_size, len(queries)))
        epoch_times.append(_time.monotonic() - ep_start)
    total_train_sec = _time.monotonic() - train_start

    write_frozen_metrics(args.metrics_path, compile_api_sec, epoch_times, total_train_sec, len(queries))

    label_to_idx = {name: idx for idx, name in enumerate(LABEL_ORDER)}
    train_targets = torch.tensor(
        [label_to_idx[name] for name in train_labels], dtype=torch.long, device=device
    )
    eval_targets = torch.tensor(
        [label_to_idx[name] for name in eval_labels], dtype=torch.long, device=device
    )
    train_acc = classification_accuracy(net, train_images, train_targets)
    eval_acc = classification_accuracy(net, eval_images, eval_targets)
    print(f"train_acc={train_acc:.4f} eval_acc={eval_acc:.4f} epochs={epochs}")

    threshold = resolve_min_accuracy(args.mode, manifest, args.min_accuracy)
    report_and_enforce_metric("eval_acc", eval_acc, threshold)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
