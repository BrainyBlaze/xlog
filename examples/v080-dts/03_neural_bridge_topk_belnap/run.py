#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import torch
import pyxlog


LABELS = ["accept", "review", "reject"]


class BridgeNet(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.logits = torch.nn.Parameter(torch.tensor([3.0, 2.0, 1.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(self.logits.to(device=x.device), dim=0)
        return probs.expand(x.shape[0], -1)


def _tensor_float(value: object) -> float:
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for v0.8.0 DTS examples")

    source = Path(__file__).with_name("program.xlog").read_text(encoding="utf-8")
    program = pyxlog.Program.compile(source, device=0, memory_mb=512)

    device = torch.device("cuda")
    net = BridgeNet().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    program.register_network("bridge_net", net, optimizer, k=2, det=True, cache=True, cache_size=32)
    program.add_tensor_source("bridge_features", torch.zeros(2, 4, device=device))

    program.zero_grad()
    loss = program.forward_backward_tensor("label_for(0, accept)")
    if not bool(loss.is_cuda):
        raise AssertionError("forward_backward_tensor did not return a CUDA loss tensor")
    if net.logits.grad is None or not bool(torch.isfinite(net.logits.grad).all()):
        raise AssertionError("registered deterministic network mode did not preserve finite gradients")

    # Repeat the same template once to exercise the circuit cache path.
    program.zero_grad()
    second_loss = program.forward_backward_tensor("label_for(0, accept)")

    scores = torch.tensor([0.5, 0.5, 0.7], device=device)
    topk = program.deterministic_topk(scores, k=2)
    selected_indices = [int(v) for v in topk["indices"].detach().cpu().tolist()]
    selected_labels = [LABELS[idx] for idx in selected_indices]

    belnap = program.belnap_loss(
        pro=torch.tensor([0.8], device=device),
        contra=torch.tensor([0.1], device=device),
        quarantine=torch.tensor([0.2], device=device),
        pro_reward=1.0,
        contra_penalty=2.0,
        quarantine_penalty=0.5,
    )
    cache_stats = program.neural_cache_stats()

    summary = {
        "example": "03_neural_bridge_topk_belnap",
        "status": "PASS",
        "cuda_tensor_checks": {
            "loss_is_cuda": bool(loss.is_cuda),
            "second_loss_is_cuda": bool(second_loss.is_cuda),
        },
        "deterministic_topk_selected_labels": selected_labels,
        "deterministic_topk_indices": selected_indices,
        "registered_network_modes": cache_stats["networks"]["bridge_net"],
        "cache_stats": cache_stats,
        "belnap": {
            "loss": _tensor_float(belnap["loss"]),
            "formula": belnap["formula"],
            "cfr_regret_proxy": _tensor_float(belnap["cfr_regret_proxy"]),
        },
        "forward_backward_loss": _tensor_float(loss),
        "second_forward_backward_loss": _tensor_float(second_loss),
        "gradient_finite": True,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
