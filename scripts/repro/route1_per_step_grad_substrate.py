#!/usr/bin/env python3
"""Route-1 compiled-once-grad-substrate: per-step confidence refresh + grads via Path-B.

glm route-decision (2026-07-01): REUSE the existing nn-predicate Path-B infrastructure
for the trainable coupling, instead of building a new set-weights API on the p::atom
path (Route-2, deferred).

Demonstrates the per-diffusion-step contract on the PRODUCTION neural path:
  1. compile the program ONCE (nn-predicate leaf + rule);
  2. each step, swap the tensor source with the step's fresh confidence (the value the
     diffusion boundary produces) — NO recompile;
  3. the query marginal tracks the fresh confidence through the logic;
  4. gradients flow to the trainable coupling parameter (guard) each step, so the
     coupling can LEARN end-to-end (the compiled-once-grad-substrate requirement).

The coupling module is a passthrough-with-guard: p = sigmoid(w) * x, where x is the
per-step confidence (refresh axis) and w is the trainable guard (learning axis).

Requires torch + pyxlog with CUDA (integration env: system python3).
"""
import math

import torch
import pyxlog

PROGRAM = """
nn(conf, [X], Y, [yes, no]) :: fact(X, Y).
h() :- fact(X, yes).
"""


class GuardedConfidence(torch.nn.Module):
    """[p, 1-p] where p = sigmoid(guard) * x — x per-step confidence, guard trainable."""

    def __init__(self) -> None:
        super().__init__()
        self.guard = torch.nn.Parameter(torch.tensor([2.0]))  # sigmoid(2.0) ≈ 0.881

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(self.guard) * x.squeeze(-1)
        return torch.stack([p, 1.0 - p], dim=-1)


def main() -> None:
    program = pyxlog.Program.compile(PROGRAM)
    net = GuardedConfidence().cuda()
    program.register_network("conf", net, torch.optim.SGD(net.parameters(), lr=0.05))

    guard = torch.sigmoid(net.guard).item()
    print(f"compiled ONCE; trainable guard sigmoid(w)={guard:.4f}")

    # per-step loop: fresh confidence each step, same compiled program
    step_confidences = [0.542, 0.90, 0.20]  # e.g. per-diffusion-step boundary values
    for step, conf in enumerate(step_confidences, 1):
        x = torch.tensor([[conf]], device="cuda")
        program.add_tensor_source("data", x)      # refresh: this step's confidence
        program.zero_grad()
        loss = program.forward_backward("h()", expected=True)
        p = math.exp(-loss)
        expected = torch.sigmoid(net.guard).item() * conf
        grad = net.guard.grad
        grad_ok = grad is not None and not torch.all(grad == 0)
        print(f"step {step}: conf={conf:.3f} -> P(h)={p:.4f} "
              f"(expect guard*conf={expected:.4f}, match={abs(p - expected) < 1e-3}) "
              f"grad_on_guard={'NONZERO' if grad_ok else 'MISSING'}")
        assert abs(p - expected) < 1e-3, (p, expected)
        assert grad_ok, "no gradient on trainable coupling parameter"

    # learning axis: a few optimizer steps move the guard (trainable end-to-end)
    before = torch.sigmoid(net.guard).item()
    for _ in range(5):
        program.add_tensor_source("data", torch.tensor([[0.542]], device="cuda"))
        program.zero_grad()
        program.forward_backward("h()", expected=True)
        program.optimizer_step()
    after = torch.sigmoid(net.guard).item()
    print(f"training moves guard: {before:.4f} -> {after:.4f} (toward 1.0 for expected=True)")
    assert after > before

    print("\nROUTE-1 SUBSTRATE WORKS: compile-once + per-step confidence refresh "
          "(marginal tracks fresh values, no recompile) + grads to trainable coupling.")


if __name__ == "__main__":
    main()
