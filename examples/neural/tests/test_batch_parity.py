"""Verify batched and sequential training produce equivalent results.

Runs 2 epochs of training on a small MNIST addition program with both
batch_queries=True and batch_queries=False, and asserts that per-epoch
losses and final network parameters are within floating-point tolerance.
"""
import copy
import sys
import torch
import torch.nn as nn

import pyxlog


class SmallNet(nn.Module):
    """Minimal CNN for the parity test (fast, deterministic)."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(4 * 12 * 12, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.softmax(self.fc2(x), dim=-1)


def make_program():
    return pyxlog.Program.compile("""
        nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        addition(X, Y, Z) :-
            digit(X, LeftDigit),
            digit(Y, RightDigit),
            Z is LeftDigit + RightDigit.
    """)


def train_run(program, net, optimizer, images, queries, *, batch_queries):
    """Run 2 epochs and return (epoch_losses, state_dict copy)."""
    program.register_network("mnist_net", net, optimizer)
    program.add_tensor_source("train", images)
    program.set_batch_queries(batch_queries)

    history = pyxlog.train_model_tensor(
        program,
        queries,
        epochs=2,
        batch_size=8,
        log_iter=999999,  # suppress per-batch logging
        shuffle=False,     # deterministic ordering
    )
    return history.epoch_losses, copy.deepcopy(net.state_dict())


def main():
    seed = 42
    if not torch.cuda.is_available():
        print("SKIP: test_batch_parity requires CUDA")
        sys.exit(0)
    device = "cuda"

    # Verify CUDA driver is actually usable (torch may report available
    # even when cudarc cannot load the shared library).
    try:
        prog_probe = make_program()
        del prog_probe
    except RuntimeError as e:
        if "CUDA" in str(e) or "cuda" in str(e):
            print(f"SKIP: CUDA driver not usable: {e}")
            sys.exit(0)
        raise

    # Deterministic random data: 32 images, 16 pairs.
    torch.manual_seed(seed)
    images = torch.randn(32, 1, 28, 28, device=device)
    labels = torch.randint(0, 10, (32,)).tolist()
    queries = []
    for i in range(16):
        idx1, idx2 = 2 * i, 2 * i + 1
        s = labels[idx1] + labels[idx2]
        queries.append(f"addition({idx1}, {idx2}, {s})")

    # Save initial weights for resetting.
    torch.manual_seed(seed)
    init_net = SmallNet().to(device)
    init_state = copy.deepcopy(init_net.state_dict())

    # ── Run 1: sequential (batch_queries=False) ─────────────────────
    torch.manual_seed(seed)
    net_seq = SmallNet().to(device)
    net_seq.load_state_dict(copy.deepcopy(init_state))
    opt_seq = torch.optim.Adam(net_seq.parameters(), lr=1e-3)
    prog_seq = make_program()

    losses_seq, params_seq = train_run(
        prog_seq, net_seq, opt_seq, images, queries, batch_queries=False
    )

    # ── Run 2: batched (batch_queries=True) ──────────────────────────
    torch.manual_seed(seed)
    net_bat = SmallNet().to(device)
    net_bat.load_state_dict(copy.deepcopy(init_state))
    opt_bat = torch.optim.Adam(net_bat.parameters(), lr=1e-3)
    prog_bat = make_program()

    losses_bat, params_bat = train_run(
        prog_bat, net_bat, opt_bat, images, queries, batch_queries=True
    )

    # ── Assertions ───────────────────────────────────────────────────
    tol = 1e-4
    ok = True

    print(f"Sequential losses: {losses_seq}")
    print(f"Batched losses:    {losses_bat}")

    for epoch, (ls, lb) in enumerate(zip(losses_seq, losses_bat)):
        diff = abs(ls - lb)
        status = "PASS" if diff < tol else "FAIL"
        print(f"  Epoch {epoch}: seq={ls:.6f} bat={lb:.6f} diff={diff:.2e} [{status}]")
        if diff >= tol:
            ok = False

    # Parameter delta: L2 norm of (params_seq - params_bat) for each layer.
    for name in params_seq:
        delta = (params_seq[name].float() - params_bat[name].float()).norm().item()
        status = "PASS" if delta < tol else "FAIL"
        print(f"  Param {name}: delta={delta:.2e} [{status}]")
        if delta >= tol:
            ok = False

    if ok:
        print("\ntest_batch_parity: PASSED")
    else:
        print("\ntest_batch_parity: FAILED")
        sys.exit(1)


if __name__ == "__main__":
    main()
