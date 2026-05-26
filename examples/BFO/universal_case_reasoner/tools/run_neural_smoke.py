#!/usr/bin/env python3
"""Run the strict CUDA `nn/4` smoke for the BFO universal case reasoner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyxlog
import torch


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["low", "high"]


class CaseNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([0.0, 2.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.logits.to(device=x.device), dim=0)
        return probs.expand(x.shape[0], -1)


def _tensor_float(value: Any) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def _top_label(net: CaseNet, device: torch.device) -> str:
    with torch.no_grad():
        scores = net(torch.zeros(1, 4, device=device))[0]
        return LABELS[int(torch.argmax(scores).detach().cpu())]


def run(output: Path) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BFO neural smoke")

    source_path = ROOT / "programs" / "neural_ranker.xlog"
    source = source_path.read_text(encoding="utf-8")
    if "nn(case_net, [X], Y, [low, high]) :: neural_failure_signal(X, Y)." not in source:
        raise RuntimeError("neural_ranker.xlog does not declare the expected nn/4 predicate")

    program = pyxlog.Program.compile(source, device=0, memory_mb=64)
    device = torch.device("cuda")
    net = CaseNet().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    program.register_network("case_net", net, optimizer, k=2, det=True, cache=True, cache_size=16)
    program.add_tensor_source("case_features", torch.zeros(2, 4, device=device))

    initial_top = _top_label(net, device)
    program.zero_grad()
    loss = program.forward_backward_tensor("root_cause_rank(0, high)")
    if not bool(loss.is_cuda):
        raise AssertionError("forward_backward_tensor did not return a CUDA loss tensor")
    if net.logits.grad is None or not bool(torch.isfinite(net.logits.grad).all()):
        raise AssertionError("case_net gradients were not finite after XLOG nn/4 invocation")

    with torch.no_grad():
        net.logits.copy_(torch.tensor([3.0, -1.0], device=device))
    flipped_top = _top_label(net, device)

    payload = {
        "status": "PASS",
        "program": str(source_path.relative_to(ROOT)),
        "program_declares_nn4": True,
        "loss_is_cuda": bool(loss.is_cuda),
        "loss": _tensor_float(loss),
        "gradient_finite": True,
        "initial_top_label": initial_top,
        "flipped_top_label": flipped_top,
        "ranking_changed": initial_top != flipped_top,
        "registered_network": "case_net",
        "tensor_source": "case_features",
        "torch_version": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "pyxlog_version": getattr(pyxlog, "__version__", "unknown"),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "evidence" / "neural_smoke.json",
        help="Path for neural smoke evidence JSON.",
    )
    args = parser.parse_args()
    payload = run(args.output)
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
