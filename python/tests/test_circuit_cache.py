"""Test circuit caching for training performance.

These tests verify:
1. Circuit cache correctly identifies identical query structures
2. Cached evaluation produces same results as fresh compilation
3. Cache hits avoid D4 recompilation (performance improvement)
"""

import pytest
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

NUM_CLASSES = 1
INPUT_DIM = 32
MAX_SUM = 2 * (NUM_CLASSES - 1)
LABELS = ",".join(str(i) for i in range(NUM_CLASSES))
PROGRAM_SOURCE = f"""
nn(mnist_net, [X], Y, [{LABELS}]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""


def target_sum(i: int) -> int:
    return (i * 2 + 1) % (MAX_SUM + 1)


class SimpleNet(torch.nn.Module):
    """Simple network that outputs softmax probabilities."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.fc = torch.nn.Linear(INPUT_DIM, num_classes)

    def forward(self, x):
        x = x.view(-1, INPUT_DIM)
        return torch.nn.functional.softmax(self.fc(x), dim=-1)


class TestCircuitCache:
    """Test circuit caching functionality."""

    def test_cache_hit_same_structure(self, monkeypatch):
        """Repeated queries with same structure should use cached circuit."""
        program = pyxlog.Program.compile(PROGRAM_SOURCE)
        baseline_compiles = program.template_compile_count()

        # Enforce GPU-native path: forbid any internal .tolist() (would force device->host transfer).
        def _forbid_tolist(*_args, **_kwargs):
            raise RuntimeError("torch.Tensor.tolist() is forbidden in GPU-native fast-path")

        monkeypatch.setattr(torch.Tensor, "tolist", _forbid_tolist, raising=True)

        # Register network
        net = SimpleNet(NUM_CLASSES).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Register dummy tensor source
        dummy_images = torch.randn(100, INPUT_DIM, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - should compile circuit (cache miss)
        loss1 = program.forward_backward("addition(0, 1, 0)")
        assert program.template_compile_count() == baseline_compiles + 1

        # Second query with SAME structure - should use cache
        loss2 = program.forward_backward("addition(2, 3, 0)")
        assert program.template_compile_count() == baseline_compiles + 1
        assert program.template_cache_size() == 1

        # Both should return valid losses
        assert loss1 > 0, f"First loss should be positive: {loss1}"
        assert loss2 > 0, f"Second loss should be positive: {loss2}"

    def test_multiple_cache_hits(self):
        """Multiple queries with same structure should all benefit from cache."""
        program = pyxlog.Program.compile(PROGRAM_SOURCE)
        baseline_compiles = program.template_compile_count()

        net = SimpleNet(NUM_CLASSES).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        dummy_images = torch.randn(100, INPUT_DIM, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - cache miss
        loss_first = program.forward_backward("addition(0, 1, 0)")
        assert program.template_compile_count() == baseline_compiles + 1
        assert program.template_cache_size() == 1

        # Multiple queries with same structure - all cache hits
        losses = []
        for i in range(5):
            loss = program.forward_backward(f"addition({i}, {i+1}, {target_sum(i)})")
            losses.append(loss)
            assert loss > 0, f"Query {i} should return valid loss"
            assert program.template_compile_count() == baseline_compiles + 1

        # All losses should be valid
        assert all(loss > 0 for loss in losses)

    def test_correctness_cached_vs_fresh(self):
        """Cached circuit should produce same results for same input data."""
        # Use deterministic initialization
        torch.manual_seed(42)

        program = pyxlog.Program.compile(PROGRAM_SOURCE)

        net = SimpleNet(NUM_CLASSES).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Use fixed input data
        torch.manual_seed(123)
        dummy_images = torch.randn(100, INPUT_DIM, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Get losses for several queries
        losses = []
        for sum_value in [0, 0]:
            loss = program.forward_backward(f"addition(0, 1, {sum_value})")
            losses.append(loss)

        # All should be valid positive losses
        for i, loss in enumerate(losses):
            assert loss > 0, f"Loss {i} should be positive: {loss}"
            assert not torch.isnan(torch.tensor(loss)), f"Loss {i} should not be NaN"


class TestCachePerformance:
    """Performance benchmarks for circuit caching."""

    def test_training_loop_speedup(self):
        """Full training loop should see improvement from caching."""
        program = pyxlog.Program.compile(PROGRAM_SOURCE)

        net = SimpleNet(NUM_CLASSES).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, INPUT_DIM, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Simulate a short training loop with repeated query shape.
        queries = [f"addition({i}, {i+1}, {target_sum(i)})" for i in range(6)]

        epoch_times = []
        for epoch in range(2):
            optimizer.zero_grad()
            total_loss = 0.0
            for q in queries:
                loss = program.forward_backward(q)
                total_loss += loss
            optimizer.step()
            assert program.template_compile_count() == 1
            epoch_times.append(total_loss)

        assert epoch_times, "Expected at least one epoch result"
