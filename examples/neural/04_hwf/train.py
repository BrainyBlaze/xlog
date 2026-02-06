import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.neural_datasets import DatasetManifest

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "crohme"
MANIFEST = ROOT / "dataset.json"


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


def load_crohme(mode: str):
    if not DATA.exists():
        raise SystemExit(
            "CROHME dataset missing. Place under examples/neural/04_hwf/data/crohme"
        )

    try:
        from torchvision import transforms
    except ImportError as exc:
        raise SystemExit(f"Missing dependency: {exc}")
    import torch

    manifest = DatasetManifest.load(MANIFEST)
    limit = manifest.ci_subset.get("train", 64) if mode == "ci" else 1024

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
        if len(images) >= limit:
            break

    if not images:
        raise SystemExit("CROHME dataset missing: no labeled InkML samples found")

    transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])
    tensor = torch.stack([transform(im) for im in images])
    return tensor, labels


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ci", "dev", "release"], required=True)
    parser.add_argument("--epochs", type=int, default=2)
    args = parser.parse_args()

    images, labels = load_crohme(args.mode)
    import pyxlog
    import torch

    from model import HWFNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images = images.to(device)

    program = pyxlog.Program.compile((ROOT / "program.xlog").read_text())
    net = HWFNet().to(device)
    program.register_network("hw_net", net, torch.optim.Adam(net.parameters(), lr=1e-3))
    program.add_tensor_source("train", images)

    queries = [f"expr_type({i}, {labels[i]})" for i in range(len(labels))]
    for _ in range(args.epochs):
        program.train_epoch(queries, batch_size=min(8, len(queries)))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
