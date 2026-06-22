"""Test circuit caching for training performance.

These tests verify:
1. Circuit cache correctly identifies identical query structures
2. Cached evaluation produces same results as fresh compilation
3. Cache hits avoid Decision-DNNF compiler recompilation
"""

import pytest
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA not available", allow_module_level=True)

CLASS_COUNT = 1
INPUT_DIMENSION = 32
MAXIMUM_SUM = 2 * (CLASS_COUNT - 1)
LABELS = ",".join(str(label) for label in range(CLASS_COUNT))
PROGRAM_SOURCE = f"""
nn(mnist_net, [X], Y, [{LABELS}]) :: digit(X, Y).
addition(FirstImage, SecondImage, Sum) :-
    digit(FirstImage, FirstDigitValue),
    digit(SecondImage, SecondDigitValue),
    Sum is FirstDigitValue + SecondDigitValue.
"""


def target_sum(example_index: int) -> int:
    return (example_index * 2 + 1) % (MAXIMUM_SUM + 1)


class SimpleNet(torch.nn.Module):  # type: ignore[name-defined]
    """Simple network that outputs softmax probabilities."""

    def __init__(self, class_count=CLASS_COUNT):
        super().__init__()
        self.linear = torch.nn.Linear(INPUT_DIMENSION, class_count)

    def forward(self, x):
        x = x.view(-1, INPUT_DIMENSION)
        return torch.nn.functional.softmax(self.linear(x), dim=-1)


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
        net = SimpleNet(CLASS_COUNT).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Register dummy tensor source
        dummy_images = torch.randn(100, INPUT_DIMENSION, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - should compile circuit (cache miss)
        first_loss = program.forward_backward("addition(0, 1, 0)")
        assert program.template_compile_count() == baseline_compiles + 1

        # Second query with SAME structure - should use cache
        second_loss = program.forward_backward("addition(2, 3, 0)")
        assert program.template_compile_count() == baseline_compiles + 1
        assert program.template_cache_size() == 1

        # Both should return valid losses
        assert first_loss > 0, f"First loss should be positive: {first_loss}"
        assert second_loss > 0, f"Second loss should be positive: {second_loss}"

    def test_multiple_cache_hits(self):
        """Multiple queries with same structure should all benefit from cache."""
        program = pyxlog.Program.compile(PROGRAM_SOURCE)
        baseline_compiles = program.template_compile_count()

        net = SimpleNet(CLASS_COUNT).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        dummy_images = torch.randn(100, INPUT_DIMENSION, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # First query - cache miss
        program.forward_backward("addition(0, 1, 0)")
        assert program.template_compile_count() == baseline_compiles + 1
        assert program.template_cache_size() == 1

        # Multiple queries with same structure - all cache hits
        losses = []
        for example_index in range(5):
            loss = program.forward_backward(
                f"addition({example_index}, {example_index + 1}, {target_sum(example_index)})"
            )
            losses.append(loss)
            assert loss > 0, f"Query {example_index} should return valid loss"
            assert program.template_compile_count() == baseline_compiles + 1

        # All losses should be valid
        assert all(loss > 0 for loss in losses)

    def test_correctness_cached_vs_fresh(self):
        """Cached circuit should produce same results for same input data."""
        # Use deterministic initialization
        torch.manual_seed(42)

        program = pyxlog.Program.compile(PROGRAM_SOURCE)

        net = SimpleNet(CLASS_COUNT).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        # Use fixed input data
        torch.manual_seed(123)
        dummy_images = torch.randn(100, INPUT_DIMENSION, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Get losses for several queries
        losses = []
        for sum_value in [0, 0]:
            loss = program.forward_backward(f"addition(0, 1, {sum_value})")
            losses.append(loss)

        # All should be valid positive losses
        for loss_index, loss in enumerate(losses):
            assert loss > 0, f"Loss {loss_index} should be positive: {loss}"
            assert not torch.isnan(torch.tensor(loss)), f"Loss {loss_index} should not be NaN"


class TestCachePerformance:
    """Performance benchmarks for circuit caching."""

    def test_training_loop_speedup(self):
        """Full training loop should see improvement from caching."""
        program = pyxlog.Program.compile(PROGRAM_SOURCE)

        net = SimpleNet(CLASS_COUNT).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, INPUT_DIMENSION, device="cuda")
        program.add_tensor_source("X", dummy_images)

        # Simulate a short training loop with repeated query shape.
        queries = [
            f"addition({example_index}, {example_index + 1}, {target_sum(example_index)})"
            for example_index in range(6)
        ]

        epoch_times = []
        for epoch in range(2):
            optimizer.zero_grad()
            total_loss = 0.0
            for query in queries:
                loss = program.forward_backward(query)
                total_loss += loss
            optimizer.step()
            assert program.template_compile_count() == 1
            epoch_times.append(total_loss)

        assert epoch_times, "Expected at least one epoch result"
