"""
Minimal MNIST Addition Example

Trains a neural network to classify MNIST digits using only addition labels.
The network never sees individual digit labels during training - it learns
to classify digits purely from the supervision signal on sums.

This demonstrates neural-symbolic integration:
- Neural network outputs become probabilistic facts
- Probabilistic logic computes query probabilities
- Gradients flow from loss through logic back to networks

Usage:
    python train.py --epochs 50 --batch-size 32

Example:
    # The program defines:
    # nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
    # addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
    #
    # Given images at indices 0 and 1, the query:
    #   addition(0, 1, 7)
    # asks: what is the probability that image[0] + image[1] = 7?
    #
    # Training minimizes -log P(addition(i, j, correct_sum))
"""
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_improvement_percent(epoch_losses):
    """Return percent loss improvement or None when it is not well-defined."""
    if len(epoch_losses) < 2:
        return None
    initial = epoch_losses[0]
    if initial == 0:
        return None
    final = epoch_losses[-1]
    return (1 - final / initial) * 100


def set_seed(seed):
    """Set deterministic seeds for reproducible runs."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sample_addition_pairs(n_items, n_pairs, generator=None):
    """Sample random index pairs for addition supervision."""
    if n_items <= 0:
        raise ValueError("n_items must be > 0")
    if n_pairs <= 0:
        raise ValueError("n_pairs must be > 0")
    left = torch.randint(0, n_items, (n_pairs,), generator=generator, dtype=torch.long)
    right = torch.randint(0, n_items, (n_pairs,), generator=generator, dtype=torch.long)
    return left, right


def addition_sum_distribution(probs_a, probs_b):
    """Compute P(sum=s) for s in [0, 18] from two digit distributions."""
    if probs_a.ndim != 2 or probs_b.ndim != 2:
        raise ValueError("Expected [batch, 10] tensors")
    if probs_a.shape != probs_b.shape:
        raise ValueError("probs_a and probs_b must have identical shape")
    if probs_a.shape[1] != 10:
        raise ValueError("Expected 10 digit classes")
    batch = probs_a.shape[0]
    sum_probs = torch.zeros(batch, 19, device=probs_a.device, dtype=probs_a.dtype)
    # For each first-digit class d, shift-add d * P(second digit).
    for d in range(10):
        sum_probs[:, d:d + 10] += probs_a[:, d:d + 1] * probs_b
    return sum_probs


def addition_nll_loss(probs_a, probs_b, target_sums, eps=1e-12):
    """NLL loss for supervision on the sum of two digits."""
    sum_probs = addition_sum_distribution(probs_a, probs_b)
    return F.nll_loss(torch.log(sum_probs.clamp_min(eps)), target_sums)


@torch.no_grad()
def compute_digit_accuracy(model, images, labels, device, batch_size=256):
    """Compute digit classification accuracy for a tensor/label set."""
    model.eval()
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.to(device)
    correct = 0
    total = labels.numel()
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        logits = model(images[start:end].to(device))
        preds = logits.argmax(dim=1)
        correct += int((preds == labels[start:end]).sum().item())
    return correct / total if total else 0.0


@torch.no_grad()
def compute_addition_accuracy(model, images, labels, device, batch_size=256):
    """Evaluate addition held-out accuracy on adjacent image pairs."""
    model.eval()
    if isinstance(labels, list):
        labels = torch.tensor(labels, dtype=torch.long)
    labels = labels.to(device)

    n_items = int(labels.numel())
    n_pairs = n_items // 2
    if n_pairs == 0:
        return 0.0, 0, 0

    correct = 0
    total = 0
    for start in range(0, n_pairs, batch_size):
        end = min(start + batch_size, n_pairs)
        pair_ids = torch.arange(start, end, device=device, dtype=torch.long)
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


class MNISTNet(nn.Module):
    """CNN for MNIST digit classification.

    Architecture follows LeNet-5 style:
    - 2 convolutional layers with max pooling
    - 3 fully connected layers
    - Softmax output for 10 digit classes

    The network outputs a probability distribution over digits 0-9.
    These probabilities become the weights of probabilistic facts in XLOG.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # x: [batch, 1, 28, 28]
        x = self.pool(torch.relu(self.conv1(x)))  # [batch, 6, 12, 12]
        x = self.pool(torch.relu(self.conv2(x)))  # [batch, 16, 4, 4]
        x = x.view(-1, 16 * 4 * 4)                # [batch, 256]
        x = torch.relu(self.fc1(x))               # [batch, 120]
        x = torch.relu(self.fc2(x))               # [batch, 84]
        return torch.softmax(self.fc3(x), dim=-1) # [batch, 10]


def create_program():
    """Compile the XLOG program for MNIST addition.

    The program defines:
    1. A neural predicate `digit/2` backed by `mnist_net`
    2. An addition rule that sums two classified digits

    Returns:
        CompiledProgram ready for network registration
    """
    import pyxlog

    return pyxlog.Program.compile("""
        // Neural predicate: classify digit images
        // nn(network, [inputs], output, [labels]) :: predicate(args).
        nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).

        // Addition rule: sum of two classified digits
        // Given digit(img1, d1) and digit(img2, d2), compute d1 + d2
        addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
    """)


def generate_queries(n_pairs, labels=None):
    """Generate addition queries for training.

    Each query is of the form: addition(img1_idx, img2_idx, expected_sum)

    Args:
        n_pairs: Number of image pairs to generate queries for
        labels: Optional list of digit labels for each image index.
                If provided, queries use correct sums from labels.
                If None, uses (index % 10) as mock label.

    Returns:
        List of query strings for training

    Example:
        >>> generate_queries(2, labels=[3, 5, 7, 2])
        ['addition(0, 1, 8)', 'addition(2, 3, 9)']
    """
    queries = []
    for i in range(n_pairs):
        img1_idx = 2 * i
        img2_idx = 2 * i + 1
        if labels:
            expected_sum = labels[img1_idx] + labels[img2_idx]
        else:
            # For testing without real data, use index mod 10 as "label"
            expected_sum = (img1_idx % 10) + (img2_idx % 10)
        queries.append(f"addition({img1_idx}, {img2_idx}, {expected_sum})")
    return queries


def load_mnist(data_path):
    """Load MNIST dataset.

    Args:
        data_path: Path to MNIST data directory

    Returns:
        Tuple of (images, labels) where:
        - images: torch.Tensor of shape [N, 1, 28, 28]
        - labels: list of int labels 0-9

    """
    return _load_mnist_split(data_path, train=True, limit=None)


def load_mnist_test(data_path, limit=None):
    """Load MNIST test split."""
    return _load_mnist_split(data_path, train=False, limit=limit)


def _load_mnist_split(data_path, train, limit=None):
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        dataset = datasets.MNIST(
            data_path,
            train=train,
            download=True,
            transform=transform
        )

        images = torch.stack([img for img, _ in dataset])
        labels = [label for _, label in dataset]

        env_limit = os.environ.get("XLOG_PY_EXAMPLE_MNIST_LIMIT")
        if limit is None and env_limit:
            limit = max(1, int(env_limit))
        if limit is not None:
            n = max(1, int(limit))
            images = images[:n]
            labels = labels[:n]

        return images, labels

    except ImportError as exc:
        raise RuntimeError(
            "torchvision is required to load MNIST datasets for this example."
        ) from exc


def train_torch_addition(
    net,
    optimizer,
    train_images,
    train_labels,
    device,
    epochs,
    batch_size,
    pairs_per_epoch,
    eval_images=None,
    eval_labels=None,
    target_train_acc=None,
):
    """Train with a direct addition objective equivalent to the logic rule."""
    labels_t = torch.tensor(train_labels, dtype=torch.long, device=device)
    pair_gen = torch.Generator().manual_seed(0)
    epoch_losses = []

    for epoch in range(1, epochs + 1):
        net.train()
        left, right = sample_addition_pairs(len(train_labels), pairs_per_epoch, generator=pair_gen)
        total_loss = 0.0
        total_steps = 0
        for start in range(0, pairs_per_epoch, batch_size):
            end = min(start + batch_size, pairs_per_epoch)
            ia = left[start:end].to(device)
            ib = right[start:end].to(device)
            probs_a = net(train_images[ia])
            probs_b = net(train_images[ib])
            targets = labels_t[ia] + labels_t[ib]
            loss = addition_nll_loss(probs_a, probs_b, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item())
            total_steps += 1

        avg_loss = total_loss / max(total_steps, 1)
        epoch_losses.append(avg_loss)

        train_acc = compute_digit_accuracy(net, train_images, train_labels, device)
        eval_acc = None
        if eval_images is not None and eval_labels is not None:
            eval_acc = compute_digit_accuracy(net, eval_images, eval_labels, device)

        if eval_acc is None:
            print(f"Epoch {epoch}/{epochs}: avg_loss={avg_loss:.6f}, train_acc={train_acc:.4f}")
        else:
            print(
                f"Epoch {epoch}/{epochs}: avg_loss={avg_loss:.6f}, "
                f"train_acc={train_acc:.4f}, eval_acc={eval_acc:.4f}"
            )

        if target_train_acc is not None and train_acc >= target_train_acc:
            print(f"Reached target train accuracy {target_train_acc:.4f} at epoch {epoch}.")
            break

    return epoch_losses


def main():
    parser = argparse.ArgumentParser(description='Train MNIST addition model')
    parser.add_argument('--engine', choices=['xlog', 'torch'], default='xlog',
                        help='Training engine: full XLOG path or fast direct-addition path')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--data-path', type=str, default='./data/mnist',
                        help='Path to MNIST data')
    parser.add_argument('--save-path', type=str, default='mnist_net.pt',
                        help='Path to save trained model')
    parser.add_argument('--log-iter', type=int, default=10,
                        help='Log every N batches')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed for reproducible training')
    parser.add_argument('--train-limit', type=int, default=None,
                        help='Optional cap on number of train samples')
    parser.add_argument('--eval-limit', type=int, default=512,
                        help='Number of test samples for evaluation in torch engine')
    parser.add_argument('--pairs-per-epoch', type=int, default=4096,
                        help='Number of random addition pairs per epoch in torch engine')
    parser.add_argument('--target-train-acc', type=float, default=None,
                        help='Optional early-stop target for train accuracy in torch engine')
    args = parser.parse_args()

    set_seed(args.seed)

    # Setup
    print("Initializing network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MNISTNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    # Load data
    print(f"Loading MNIST data from {args.data_path}...")
    train_images, train_labels = load_mnist(args.data_path)
    if args.train_limit is not None:
        n = max(1, int(args.train_limit))
        train_images = train_images[:n]
        train_labels = train_labels[:n]
    if device == 'cuda':
        train_images = train_images.cuda()

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print(f"  Engine: {args.engine}")
    print()

    if args.engine == 'xlog':
        print("Compiling XLOG program...")
        import pyxlog

        program = create_program()
        program.register_network("mnist_net", net, optimizer)
        program.add_tensor_source("train", train_images)

        n_pairs = len(train_labels) // 2
        queries = generate_queries(n_pairs, train_labels)
        print(f"Generated {len(queries)} training queries")

        history = pyxlog.train_model_tensor(
            program,
            queries,
            epochs=args.epochs,
            batch_size=args.batch_size,
            log_iter=args.log_iter,
        )
        epoch_losses = history.epoch_losses
    else:
        eval_images = None
        eval_labels = None
        if args.eval_limit and args.eval_limit > 0:
            eval_images, eval_labels = load_mnist_test(args.data_path, limit=args.eval_limit)
            if device == 'cuda':
                eval_images = eval_images.cuda()
        epoch_losses = train_torch_addition(
            net,
            optimizer,
            train_images,
            train_labels,
            device=device,
            epochs=args.epochs,
            batch_size=args.batch_size,
            pairs_per_epoch=max(args.pairs_per_epoch, args.batch_size),
            eval_images=eval_images,
            eval_labels=eval_labels,
            target_train_acc=args.target_train_acc,
        )

    # Summary
    print(f"\nTraining complete!")
    print(f"  Initial loss: {epoch_losses[0]:.4f}")
    print(f"  Final loss: {epoch_losses[-1]:.4f}")
    improvement = compute_improvement_percent(epoch_losses)
    if improvement is None:
        print("  Improvement: N/A (requires >=2 epochs and non-zero initial loss)")
    else:
        print(f"  Improvement: {improvement:.1f}%")
    final_train_acc = compute_digit_accuracy(net, train_images, train_labels, device)
    print(f"  Final train accuracy: {final_train_acc:.4f}")

    test_images, test_labels = load_mnist_test(args.data_path, limit=10000)
    if device == "cuda":
        test_images = test_images.cuda()
    heldout_digit_acc = compute_digit_accuracy(net, test_images, test_labels, device)
    heldout_add_acc, heldout_add_correct, heldout_add_total = compute_addition_accuracy(
        net, test_images, test_labels, device
    )
    print(f"  Held-out digit accuracy: {heldout_digit_acc:.4f}")
    print(f"Held-out Accuracy {heldout_add_acc:.4f}")
    print(f"Held-out Correct/Total {heldout_add_correct} {heldout_add_total}")
    print(
        f"FINAL_METRIC: heldout_addition_acc={heldout_add_acc:.4f}, threshold=none"
    )

    # Save model
    torch.save(net.state_dict(), args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()
