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
import torch
import torch.nn as nn
import xlog_gpu


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
    return xlog_gpu.Program.compile("""
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

    Note:
        This is a placeholder. In production, use torchvision.datasets.MNIST
        or load from pre-processed files.
    """
    try:
        from torchvision import datasets, transforms

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST(
            data_path,
            train=True,
            download=True,
            transform=transform
        )

        # Convert to tensor format expected by XLOG
        images = torch.stack([img for img, _ in train_dataset])
        labels = [label for _, label in train_dataset]

        return images, labels

    except ImportError:
        print("Warning: torchvision not available, using random data")
        # Fallback to random data for testing
        n_samples = 1000
        images = torch.randn(n_samples, 1, 28, 28)
        labels = torch.randint(0, 10, (n_samples,)).tolist()
        return images, labels


def main():
    parser = argparse.ArgumentParser(description='Train MNIST addition model')
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
    args = parser.parse_args()

    # Setup
    print("Compiling XLOG program...")
    program = create_program()

    print("Initializing network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = MNISTNet().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    program.register_network("mnist_net", net, optimizer)

    # Load data
    print(f"Loading MNIST data from {args.data_path}...")
    train_images, train_labels = load_mnist(args.data_path)
    if device == 'cuda':
        train_images = train_images.cuda()

    program.add_tensor_source("train", train_images)

    # Generate training queries
    n_pairs = len(train_labels) // 2
    queries = generate_queries(n_pairs, train_labels)
    print(f"Generated {len(queries)} training queries")

    # Train
    print(f"\nTraining for {args.epochs} epochs...")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Device: {device}")
    print()

    history = xlog_gpu.train_model(
        program,
        queries,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_iter=args.log_iter,
    )

    # Summary
    print(f"\nTraining complete!")
    print(f"  Initial loss: {history.epoch_losses[0]:.4f}")
    print(f"  Final loss: {history.epoch_losses[-1]:.4f}")
    print(f"  Improvement: {(1 - history.epoch_losses[-1]/history.epoch_losses[0])*100:.1f}%")

    # Save model
    torch.save(net.state_dict(), args.save_path)
    print(f"\nModel saved to {args.save_path}")


if __name__ == "__main__":
    main()
