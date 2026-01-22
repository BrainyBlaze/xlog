"""Tests for neural network registration API.

These tests require:
1. pyxlog Python module built and installed (maturin develop)
2. PyTorch installed

Run with: pytest python/tests/test_network_registry.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class SimpleNet(torch.nn.Module):
    """Simple network for testing - classifies 784-dim input to 10 classes."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10)

    def forward(self, x):
        return torch.softmax(self.fc(x.view(-1, 784)), dim=-1)


class EmbeddingNet(torch.nn.Module):
    """Embedding network for testing - produces 128-dim embeddings."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 128)

    def forward(self, x):
        return self.fc(x.view(-1, 784))


class TestNetworkRegistration:
    """Test suite for register_network() API."""

    def test_register_network_basic(self):
        """Test basic network registration."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

        program.register_network("test_net", net, optimizer)

        assert "test_net" in program.network_names()

    def test_register_network_with_scheduler(self):
        """Test network registration with learning rate scheduler."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        program.register_network(
            "test_net",
            net,
            optimizer,
            scheduler=scheduler,
            batching=True,
            k=5,
            det=False,
        )

        assert "test_net" in program.network_names()

    def test_register_network_deterministic(self):
        """Test network registration in deterministic mode."""
        program = pyxlog.Program.compile("""
            nn(det_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        program.register_network(
            "det_net", net, optimizer, det=True, cache=False
        )

        assert "det_net" in program.network_names()

    def test_register_network_undeclared_fails(self):
        """Test that registering an undeclared network fails."""
        program = pyxlog.Program.compile("""
            nn(declared_net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError, match="not declared"):
            program.register_network("undeclared_net", net, optimizer)

    def test_register_multiple_networks(self):
        """Test registering multiple networks."""
        program = pyxlog.Program.compile("""
            nn(net1, [X], Y1, [0,1,2,3,4,5,6,7,8,9]) :: digit1(X, Y1).
            nn(net2, [X], Y2, [0,1,2,3,4,5,6,7,8,9]) :: digit2(X, Y2).
            addition(X, Y, Z) :- digit1(X, D1), digit2(Y, D2), Z is D1 + D2.
        """)

        net1 = SimpleNet()
        net2 = SimpleNet()
        opt1 = torch.optim.Adam(net1.parameters())
        opt2 = torch.optim.Adam(net2.parameters())

        program.register_network("net1", net1, opt1)
        program.register_network("net2", net2, opt2)

        names = program.network_names()
        assert "net1" in names
        assert "net2" in names
        assert len(names) == 2

    def test_declared_network_names(self):
        """Test getting declared network names from nn() declarations."""
        program = pyxlog.Program.compile("""
            nn(alpha, [X], Y, [0,1]) :: pred_a(X, Y).
            nn(beta, [X], Y, [0,1]) :: pred_b(X, Y).
            nn(gamma, [X], Y, [0,1]) :: pred_c(X, Y).
        """)

        declared = program.declared_network_names()
        assert len(declared) == 3
        assert "alpha" in declared
        assert "beta" in declared
        assert "gamma" in declared

    def test_has_neural_predicate(self):
        """Test checking if a network is declared."""
        program = pyxlog.Program.compile("""
            nn(exists, [X], Y, [0,1]) :: pred(X, Y).
        """)

        assert program.has_neural_predicate("exists")
        assert not program.has_neural_predicate("not_exists")

    def test_set_train_mode(self):
        """Test setting training mode for all networks."""
        program = pyxlog.Program.compile("""
            nn(train_net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())
        program.register_network("train_net", net, optimizer)

        # Set to training mode
        program.set_train_mode(True)

        # Set to eval mode
        program.set_train_mode(False)

    def test_embedding_network(self):
        """Test registering an embedding network (no labels)."""
        program = pyxlog.Program.compile("""
            nn(encoder, [X], Embedding) :: encode(X, Embedding).
        """)

        net = EmbeddingNet()
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network("encoder", net, optimizer)

        assert "encoder" in program.network_names()


class TestNetworkRegistrationEdgeCases:
    """Edge case tests for network registration."""

    def test_empty_program_no_networks(self):
        """Test that a program without nn() has no declared networks."""
        program = pyxlog.Program.compile("""
            edge(1, 2).
            reach(X, Y) :- edge(X, Y).
        """)

        assert len(program.declared_network_names()) == 0
        assert len(program.network_names()) == 0

    def test_network_with_symbol_labels(self):
        """Test network with symbol labels."""
        program = pyxlog.Program.compile("""
            nn(coin_net, [X], Y, [heads, tails]) :: coin(X, Y).
        """)

        net = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network("coin_net", net, optimizer)

        assert "coin_net" in program.network_names()

    def test_network_with_multiple_inputs(self):
        """Test network with multiple input variables."""
        program = pyxlog.Program.compile("""
            nn(multi_net, [X, Y, Z], Out, [0,1,2]) :: classify(X, Y, Z, Out).
        """)

        # Network that takes 3 inputs
        class MultiInputNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(30, 3)

            def forward(self, x, y, z):
                combined = torch.cat([x, y, z], dim=-1)
                return torch.softmax(self.fc(combined), dim=-1)

        net = MultiInputNet()
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network("multi_net", net, optimizer)

        assert "multi_net" in program.network_names()
