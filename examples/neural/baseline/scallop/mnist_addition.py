#!/usr/bin/env python3
"""Scallop MNIST Addition Baseline

Mirrors the XLOG 01_minimal example: trains digit classification from
addition supervision only, using Scallop's differentiable reasoning.

Same network (MNISTNet), same data, same hyperparameters.
Outputs frozen-schema metrics + held-out accuracy for comparison.
"""
import argparse
import json
import random
import time
import sys
from pathlib import Path

import torch
import torch.nn as nn
import scallopy

PROJECT_ROOT = Path(__file__).resolve().parents[4]
MNIST_DATA = PROJECT_ROOT / "examples" / "neural" / "01_minimal" / "data" / "mnist"


class MNISTNet(nn.Module):
    """Same architecture as 01_minimal/train.py."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return torch.softmax(self.fc3(x), dim=-1)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mnist(train=True, limit=None):
    from torchvision import datasets, transforms

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    dataset = datasets.MNIST(str(MNIST_DATA), train=train, download=True, transform=transform)
    images = torch.stack([img for img, _ in dataset])
    labels = [label for _, label in dataset]
    if limit is not None:
        images = images[:limit]
        labels = labels[:limit]
    return images, labels


def addition_sum_distribution(probs_a, probs_b):
    """P(sum=s) for s in [0, 18] from two digit distributions."""
    batch = probs_a.shape[0]
    sum_probs = torch.zeros(batch, 19, device=probs_a.device, dtype=probs_a.dtype)
    for d in range(10):
        sum_probs[:, d:d + 10] += probs_a[:, d:d + 1] * probs_b
    return sum_probs


@torch.no_grad()
def compute_addition_accuracy(model, images, labels, device, batch_size=256):
    """Held-out addition accuracy on adjacent pairs."""
    model.eval()
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.to(device)
    n_pairs = labels.numel() // 2
    if n_pairs == 0:
        return 0.0, 0, 0
    correct = total = 0
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        pair_ids = torch.arange(start, end, device=device)
        left_idx = 2 * pair_ids
        right_idx = left_idx + 1
        probs_a = model(images[left_idx])
        probs_b = model(images[right_idx])
        sum_probs = addition_sum_distribution(probs_a, probs_b)
        pred_sum = sum_probs.argmax(dim=1)
        true_sum = labels[left_idx] + labels[right_idx]
        correct += int((pred_sum == true_sum).sum().item())
        total += int(true_sum.numel())
    return correct / total, correct, total


def build_scallop_forward(provenance="difftopbottomkclauses", k=3):
    """Build a Scallop forward function for MNIST addition."""
    ctx = scallopy.ScallopContext(provenance=provenance, k=k)
    ctx.add_relation("digit_1", int, input_mapping=list(range(10)))
    ctx.add_relation("digit_2", int, input_mapping=list(range(10)))
    ctx.add_rule("sum_2(a + b) = digit_1(a) and digit_2(b)")
    return ctx.forward_function("sum_2", list(range(19)), jit=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-limit", type=int, default=512)
    parser.add_argument("--provenance", type=str, default="difftopbottomkclauses")
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--metrics-path", type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading MNIST data from {MNIST_DATA}...")
    train_images, train_labels = load_mnist(train=True, limit=args.train_limit)
    train_images = train_images.to(device)

    net = MNISTNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Compile Scallop program
    print("Compiling Scallop forward function...")
    t0 = time.monotonic()
    scallop_fwd = build_scallop_forward(provenance=args.provenance, k=args.k)
    compile_sec = time.monotonic() - t0
    print(f"  compile: {compile_sec:.3f}s")

    # Build training pairs (same as 01_minimal: adjacent pairs)
    n_pairs = len(train_labels) // 2
    pair_left = list(range(0, 2 * n_pairs, 2))
    pair_right = list(range(1, 2 * n_pairs, 2))
    pair_sums = [train_labels[pair_left[i]] + train_labels[pair_right[i]] for i in range(n_pairs)]

    print(f"\nTraining {args.epochs} epochs, {n_pairs} pairs, batch_size={args.batch_size}")
    print(f"  provenance: {args.provenance}, k={args.k}")
    print(f"  device: {device}")

    epoch_times = []
    train_start = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.monotonic()
        net.train()

        # Shuffle pairs each epoch
        indices = list(range(n_pairs))
        random.shuffle(indices)

        total_loss = 0.0
        total_steps = 0
        for batch_start in range(0, n_pairs, args.batch_size):
            batch_end = min(batch_start + args.batch_size, n_pairs)
            batch_idx = indices[batch_start:batch_end]

            left = torch.tensor([pair_left[i] for i in batch_idx], dtype=torch.long)
            right = torch.tensor([pair_right[i] for i in batch_idx], dtype=torch.long)
            targets = torch.tensor([pair_sums[i] for i in batch_idx], dtype=torch.long, device=device)

            # Forward through network
            probs_a = net(train_images[left])  # [bs, 10]
            probs_b = net(train_images[right])  # [bs, 10]

            # Forward through Scallop reasoning (returns CPU tensors)
            result = scallop_fwd(digit_1=probs_a.cpu(), digit_2=probs_b.cpu())  # [bs, 19]

            # NLL loss (keep on CPU since Scallop output is CPU)
            log_probs = torch.log(result.clamp_min(1e-12))
            loss = torch.nn.functional.nll_loss(log_probs, targets.cpu())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_steps += 1

        ep_time = time.monotonic() - ep_start
        epoch_times.append(ep_time)
        avg_loss = total_loss / max(total_steps, 1)
        print(f"  epoch {epoch}/{args.epochs}: loss={avg_loss:.6f} ({ep_time:.2f}s)")

    total_train_sec = time.monotonic() - train_start

    # Held-out evaluation
    test_images, test_labels = load_mnist(train=False, limit=10000)
    test_images = test_images.to(device)
    acc, correct, total = compute_addition_accuracy(net, test_images, test_labels, device)
    print(f"\nHeld-out addition accuracy: {acc:.4f} ({correct}/{total})")
    print(f"FINAL_METRIC: heldout_addition_acc={acc:.4f}, threshold=none")

    # Write frozen-schema metrics
    if args.metrics_path is not None:
        first_epoch_sec = epoch_times[0] if epoch_times else 0.0
        steady = epoch_times[1:] if len(epoch_times) > 1 else epoch_times
        steady_mean = sum(steady) / len(steady) if steady else first_epoch_sec
        warmup = max(0.0, first_epoch_sec - steady_mean)
        n_epochs = len(epoch_times)
        per_query_ms = total_train_sec / (n_pairs * max(n_epochs, 1)) * 1000

        data = {
            "compile_api_sec": round(compile_sec, 3),
            "epoch_sec": [round(t, 3) for t in epoch_times],
            "first_epoch_sec": round(first_epoch_sec, 3),
            "steady_epoch_sec_mean": round(steady_mean, 3),
            "warmup_sec": round(warmup, 3),
            "total_train_sec": round(total_train_sec, 3),
            "per_query_ms": round(per_query_ms, 3),
            "framework": "scallop",
            "provenance": args.provenance,
            "k": args.k,
        }
        Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.metrics_path).write_text(json.dumps(data, indent=2) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
