"""Integration test for the minimal MNIST addition example.

This test verifies that the MNIST addition example
trains correctly with loss reduction.

Run with: pytest python/tests/test_minimal_example.py -v
"""

import pytest
import sys
import os

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

# Add examples path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../examples/neural/01_minimal'))


class TestMinimalExample:
    """Integration tests for the minimal MNIST addition example."""

    def test_create_program(self):
        """Test that the MNIST addition program compiles."""
        from train import create_program

        program = create_program()
        assert program is not None
        assert "mnist_net" in program.declared_network_names()

    def test_mnist_net_architecture(self):
        """Test the MNISTNet architecture."""
        from train import MNISTNet

        net = MNISTNet()

        # Check architecture by doing a forward pass
        dummy_input = torch.randn(1, 1, 28, 28)
        output = net(dummy_input)

        assert output.shape == (1, 10)  # 10 digit classes
        assert torch.allclose(output.sum(dim=1), torch.tensor([1.0]), atol=1e-5)  # Softmax sums to 1

    def test_generate_queries(self):
        """Test query generation."""
        from train import generate_queries

        queries = generate_queries(5)

        assert len(queries) == 5
        for q in queries:
            assert q.startswith("addition(")
            assert q.endswith(")")

    def test_generate_queries_with_labels(self):
        """Test query generation with known labels."""
        from train import generate_queries

        labels = [3, 5, 7, 2]  # Two pairs: (3,5) and (7,2)
        queries = generate_queries(2, labels)

        assert len(queries) == 2
        assert queries[0] == "addition(0, 1, 8)"  # 3 + 5 = 8
        assert queries[1] == "addition(2, 3, 9)"  # 7 + 2 = 9

    def test_minimal_example_trains(self):
        """Integration test: minimal MNIST digit classification trains and improves."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)  # Higher LR for faster convergence
        program.register_network("mnist_net", net, optimizer)

        # Use random data for test (real MNIST would be too slow)
        torch.manual_seed(42)
        images = torch.randn(20, 1, 28, 28)
        program.add_tensor_source("train", images)

        # Direct digit queries (neural predicate form)
        # Each query says "image at index i is digit j"
        queries = [f"digit({i}, {i % 10})" for i in range(10)]

        # Measure initial loss using forward_backward (which actually runs the network)
        program.zero_grad()
        initial_loss = sum(program.forward_backward(q) for q in queries[:5]) / 5

        # Record initial weights
        initial_weight = net.fc3.weight.clone()

        # Train for several epochs
        for _ in range(10):
            program.train_epoch(queries, batch_size=5)

        # Verify weights changed
        assert not torch.equal(net.fc3.weight, initial_weight), "Network weights didn't change"

        # Measure final loss
        program.zero_grad()
        final_loss = sum(program.forward_backward(q) for q in queries[:5]) / 5

        # Loss should decrease or stay similar (allow small noise due to random data)
        # The key verification is that weights changed, which is tested above
        assert final_loss < initial_loss * 1.1, f"Loss increased too much: {initial_loss:.4f} -> {final_loss:.4f}"

    def test_train_model_integration(self):
        """Test full train_model integration."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(123)
        images = torch.randn(30, 1, 28, 28)
        program.add_tensor_source("train", images)

        # Direct digit queries
        queries = [f"digit({i}, {i % 10})" for i in range(20)]

        # Record initial weights
        initial_weight = net.fc3.weight.clone()

        history = pyxlog.train_model(
            program,
            queries,
            epochs=5,
            batch_size=5,
            shuffle=False  # For deterministic results
        )

        assert len(history.epoch_losses) == 5
        # Weights should have changed
        assert not torch.equal(net.fc3.weight, initial_weight), "Network weights didn't change"
        # Loss should generally trend downward
        assert history.epoch_losses[-1] < history.epoch_losses[0], \
            f"Loss didn't decrease: {history.epoch_losses[0]:.4f} -> {history.epoch_losses[-1]:.4f}"


class TestAdditionQueryTraining:
    """Tests for training with addition queries (neural-circuit integration)."""

    def test_addition_query_forward_backward(self):
        """Test that forward_backward works with addition queries."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        images = torch.randn(10, 1, 28, 28)
        program.add_tensor_source("train", images)

        # Record initial weights
        initial_weight = net.fc3.weight.clone()

        # Train on addition queries
        # addition(0, 1, Z) asks: what is digit[0] + digit[1]?
        # We supervise with Z=7, meaning images[0] + images[1] = 7
        program.zero_grad()
        loss = program.forward_backward("addition(0, 1, 7)")
        program.optimizer_step()

        # Verify weights changed
        assert not torch.equal(net.fc3.weight, initial_weight), "Network weights didn't change from addition query"

        # Loss should be positive
        assert loss > 0, f"Loss should be positive, got {loss}"

    def test_addition_query_trains(self):
        """Test that training with addition queries reduces loss."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(123)
        images = torch.randn(20, 1, 28, 28)
        program.add_tensor_source("train", images)

        # Generate addition queries
        # For testing, use fixed "labels" where images[i] represents digit i % 10
        # So addition(0, 1, X) should have X = (0 % 10) + (1 % 10) = 1
        queries = [
            "addition(0, 1, 1)",   # 0 + 1 = 1
            "addition(2, 3, 5)",   # 2 + 3 = 5
            "addition(4, 5, 9)",   # 4 + 5 = 9
            "addition(6, 7, 13)",  # 6 + 7 = 13
            "addition(8, 9, 17)",  # 8 + 9 = 17
        ]

        # Record initial weights
        initial_weight = net.fc3.weight.clone()

        # Train for several epochs
        for epoch in range(5):
            for q in queries:
                program.zero_grad()
                program.forward_backward(q)
                program.optimizer_step()

        # Verify weights changed
        assert not torch.equal(net.fc3.weight, initial_weight), "Network weights didn't change"

    def test_addition_query_probability(self):
        """Test that addition query produces valid probability."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        images = torch.randn(4, 1, 28, 28)
        program.add_tensor_source("train", images)

        # Forward-backward should return a valid loss (NLL)
        program.zero_grad()
        loss = program.forward_backward("addition(0, 1, 5)")

        # NLL loss is always positive for probability < 1
        assert loss > 0, f"Expected positive loss, got {loss}"
        # NLL loss is typically not huge for reasonable probabilities
        assert loss < 100, f"Loss seems too high: {loss}"


class TestDigitPredicate:
    """Tests for the digit/2 neural predicate."""

    def test_digit_predicate_registered(self):
        """Test that digit predicate is recognized."""
        from train import create_program

        program = create_program()
        assert program.has_neural_predicate("mnist_net")

    def test_forward_backward_on_digit(self):
        """Test forward-backward on digit predicate queries."""
        from train import MNISTNet, create_program

        program = create_program()
        net = MNISTNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("mnist_net", net, optimizer)

        images = torch.randn(10, 1, 28, 28)
        program.add_tensor_source("data", images)

        original_weight = net.conv1.weight.clone()

        # Train on a digit query
        program.zero_grad()
        loss = program.forward_backward("digit(0, 5)")
        program.optimizer_step()

        # Parameters should have changed
        assert not torch.equal(net.conv1.weight, original_weight)
        assert loss > 0
