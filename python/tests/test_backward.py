"""Tests for backward pass (gradient flow) through neural-symbolic programs.

The backward pass enables training by flowing gradients from the loss
through the probabilistic circuit back to the neural network parameters.

Run with: pytest python/tests/test_backward.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class SimpleNet(torch.nn.Module):
    """Simple neural network for testing gradient flow."""

    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class TestZeroGrad:
    """Tests for zeroing gradients."""

    def test_zero_grad_clears_gradients(self):
        """Test that zero_grad clears network parameter gradients."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        # Manually set some gradients
        for param in net.parameters():
            param.grad = torch.ones_like(param)

        # Zero them
        program.zero_grad()

        # Check they're cleared
        for param in net.parameters():
            assert param.grad is None or torch.all(param.grad == 0)

    def test_zero_grad_multiple_networks(self):
        """Test zero_grad works with multiple networks."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [0, 1]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1, 2]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=5, output_dim=2)
        net_b = SimpleNet(input_dim=5, output_dim=3)

        program.register_network("net_a", net_a, torch.optim.SGD(net_a.parameters(), lr=0.01))
        program.register_network("net_b", net_b, torch.optim.SGD(net_b.parameters(), lr=0.01))

        # Set gradients
        for net in [net_a, net_b]:
            for param in net.parameters():
                param.grad = torch.ones_like(param)

        program.zero_grad()

        # All should be cleared
        for net in [net_a, net_b]:
            for param in net.parameters():
                assert param.grad is None or torch.all(param.grad == 0)


class TestOptimizerStep:
    """Tests for optimizer step."""

    def test_optimizer_step_updates_parameters(self):
        """Test that optimizer_step updates network parameters."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)  # Large LR for visible change
        program.register_network("test_net", net, optimizer)

        # Save original parameters
        original_weight = net.fc.weight.clone()
        original_bias = net.fc.bias.clone()

        # Manually set gradients (simulating backward pass)
        net.fc.weight.grad = torch.ones_like(net.fc.weight)
        net.fc.bias.grad = torch.ones_like(net.fc.bias)

        # Step
        program.optimizer_step()

        # Parameters should have changed (SGD: param = param - lr * grad)
        assert not torch.equal(net.fc.weight, original_weight)
        assert not torch.equal(net.fc.bias, original_bias)

        # With lr=1.0 and grad=1, new = old - 1
        assert torch.allclose(net.fc.weight, original_weight - 1.0)
        assert torch.allclose(net.fc.bias, original_bias - 1.0)

    def test_optimizer_step_multiple_networks(self):
        """Test optimizer_step updates all networks."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [0, 1]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1, 2]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=5, output_dim=2)
        net_b = SimpleNet(input_dim=5, output_dim=3)

        program.register_network("net_a", net_a, torch.optim.SGD(net_a.parameters(), lr=1.0))
        program.register_network("net_b", net_b, torch.optim.SGD(net_b.parameters(), lr=1.0))

        # Save and set gradients
        orig_a = net_a.fc.weight.clone()
        orig_b = net_b.fc.weight.clone()
        net_a.fc.weight.grad = torch.ones_like(net_a.fc.weight)
        net_a.fc.bias.grad = torch.ones_like(net_a.fc.bias)
        net_b.fc.weight.grad = torch.ones_like(net_b.fc.weight)
        net_b.fc.bias.grad = torch.ones_like(net_b.fc.bias)

        program.optimizer_step()

        # Both should have changed
        assert not torch.equal(net_a.fc.weight, orig_a)
        assert not torch.equal(net_b.fc.weight, orig_b)


class TestForwardBackward:
    """Tests for forward-backward pass."""

    def test_forward_backward_returns_loss(self):
        """Test that forward_backward returns NLL loss value."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        # Add tensor source
        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Forward-backward should return loss
        loss = program.forward_backward("pred(0, a)")

        # Loss should be positive (NLL is always positive for p < 1)
        assert isinstance(loss, float)
        assert loss >= 0

    def test_forward_backward_produces_gradients(self):
        """Test that forward_backward produces gradients in network parameters."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Clear any existing gradients
        program.zero_grad()

        # Forward-backward
        loss = program.forward_backward("pred(0, a)")

        # Check gradients exist and are non-zero
        for name, param in net.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    def test_forward_backward_different_labels(self):
        """Test forward_backward with different target labels."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Query for each label should work
        for label in ['a', 'b', 'c']:
            program.zero_grad()
            loss = program.forward_backward(f"pred(0, {label})")
            assert loss >= 0


class TestTrainingStep:
    """Integration tests for complete training steps."""

    def test_complete_training_step(self):
        """Test a complete training iteration: zero_grad -> forward_backward -> step."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Save original parameters
        original_weight = net.fc.weight.clone()

        # Training step
        program.zero_grad()
        loss = program.forward_backward("pred(0, a)")
        program.optimizer_step()

        # Parameters should have changed
        assert not torch.equal(net.fc.weight, original_weight), "Parameters unchanged after training step"

    def test_multiple_training_steps_reduce_loss(self):
        """Test that multiple training steps reduce the loss."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        # Fixed input for reproducibility
        torch.manual_seed(42)
        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Train for several steps and track loss
        losses = []
        for _ in range(10):
            program.zero_grad()
            loss = program.forward_backward("pred(0, a)")
            program.optimizer_step()
            losses.append(loss)

        # Loss should generally decrease (not strictly, but trend should be down)
        assert losses[-1] < losses[0], f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


class TestSchedulerStep:
    """Tests for learning rate scheduler."""

    def test_scheduler_step(self):
        """Test that scheduler_step updates learning rate."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        program.register_network("test_net", net, optimizer, scheduler)

        # Initial LR
        assert optimizer.param_groups[0]['lr'] == 1.0

        # Step scheduler
        program.scheduler_step()

        # LR should be halved
        assert optimizer.param_groups[0]['lr'] == 0.5

        # Another step
        program.scheduler_step()
        assert optimizer.param_groups[0]['lr'] == 0.25
