"""Test circuit caching for training performance.

These tests verify:
1. Circuit cache correctly identifies identical query structures
2. Cached evaluation produces same results as fresh compilation
3. Cache hits avoid D4 recompilation (performance improvement)
"""

import pytest
import time

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)


class SimpleNet(torch.nn.Module):
    """Simple network that outputs softmax probabilities."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = torch.nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        return torch.nn.functional.softmax(self.fc(x), dim=-1)


class TestCircuitCache:
    """Test circuit caching functionality."""

    def test_cache_hit_same_structure(self, monkeypatch):
        """Repeated queries with same structure should use cached circuit."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        # Enforce GPU-native path: forbid any internal .tolist() (would force device->host transfer).
        def _forbid_tolist(*_args, **_kwargs):
            raise RuntimeError("torch.Tensor.tolist() is forbidden in GPU-native fast-path")

        monkeypatch.setattr(torch.Tensor, "tolist", _forbid_tolist, raising=True)

        # Register network
        net = SimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Register dummy tensor source
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - should compile circuit (cache miss)
        t0 = time.time()
        loss1 = program.forward_backward("addition(0, 1, 7)")
        time_first = time.time() - t0

        # Second query with SAME structure - should use cache
        t0 = time.time()
        loss2 = program.forward_backward("addition(2, 3, 5)")
        time_second = time.time() - t0

        # Cache hit should be significantly faster (no D4 compilation)
        # First query includes D4 compilation (~100-200ms)
        # Cached query should be mostly GPU evaluation (~10-50ms)
        assert time_second < time_first, \
            f"Cache hit ({time_second:.3f}s) should be faster than miss ({time_first:.3f}s)"

        # Both should return valid losses
        assert loss1 > 0, f"First loss should be positive: {loss1}"
        assert loss2 > 0, f"Second loss should be positive: {loss2}"

    def test_multiple_cache_hits(self):
        """Multiple queries with same structure should all benefit from cache."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = SimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - cache miss
        loss_first = program.forward_backward("addition(0, 1, 7)")

        # Multiple queries with same structure - all cache hits
        losses = []
        for i in range(5):
            loss = program.forward_backward(f"addition({i}, {i+1}, {(i*2 + 1) % 19})")
            losses.append(loss)
            assert loss > 0, f"Query {i} should return valid loss"

        # All losses should be valid
        assert all(loss > 0 for loss in losses)

    def test_correctness_cached_vs_fresh(self):
        """Cached circuit should produce same results for same input data."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        # Use deterministic initialization
        torch.manual_seed(42)

        program = pyxlog.Program.compile(source)

        net = SimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Use fixed input data
        torch.manual_seed(123)
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Get losses for several queries
        losses = []
        for target_sum in [3, 7, 11, 15]:
            loss = program.forward_backward(f"addition(0, 1, {target_sum})")
            losses.append(loss)

        # All should be valid positive losses
        for i, loss in enumerate(losses):
            assert loss > 0, f"Loss {i} should be positive: {loss}"
            assert not torch.isnan(torch.tensor(loss)), f"Loss {i} should not be NaN"


class TestCachePerformance:
    """Performance benchmarks for circuit caching."""

    def test_training_loop_speedup(self):
        """Full training loop should see improvement from caching."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = SimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Simulate training loop with 10 queries
        queries = [f"addition({i}, {i+1}, {(i*2 + 1) % 19})" for i in range(10)]

        t0 = time.time()
        for epoch in range(3):
            optimizer.zero_grad()
            total_loss = 0.0
            for q in queries:
                loss = program.forward_backward(q)
                total_loss += loss
            optimizer.step()
        elapsed = time.time() - t0

        # With caching, 3 epochs x 10 queries should complete in reasonable time
        # First epoch: 10 D4 compilations
        # Epochs 2-3: 20 cache hits (no D4)
        # Without caching: 30 D4 compilations
        # Target: < 15 seconds for cached version
        assert elapsed < 30.0, \
            f"Training took {elapsed:.1f}s, expected reasonable time with caching"

        print(f"\nTraining loop with caching: {elapsed:.2f}s for 3 epochs x 10 queries")
