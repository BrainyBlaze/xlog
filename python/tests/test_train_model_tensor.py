"""Tests for GPU-native tensor training path.

Tests verify:
1. Parity between scalar and tensor training paths
2. Host sync reduction (.item() call count)
3. Circuit cache reuse during tensor training

Run with: pytest python/tests/test_train_model_tensor.py -v
"""

import pytest
import time

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class SimpleNet(torch.nn.Module):
    """Simple neural network for testing training loop."""

    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class MNISTSimpleNet(torch.nn.Module):
    """Simple network that outputs softmax probabilities over 10 digits."""

    def __init__(self, num_classes=10):
        super().__init__()
        self.fc = torch.nn.Linear(784, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        return torch.nn.functional.softmax(self.fc(x), dim=-1)


# ---------------------------------------------------------------------------
# 4a. Parity test -- same losses from scalar and tensor paths (tolerance 1e-5)
# ---------------------------------------------------------------------------


class TestTrainModelTensorParity:
    """Test scalar/tensor path produce identical results."""

    def test_epoch_losses_match(self):
        """train_model and train_model_tensor produce the same epoch losses."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        # --- program 1 (scalar path) ---
        torch.manual_seed(42)
        net1 = SimpleNet()
        program1 = pyxlog.Program.compile(source)
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
        program1.register_network("test_net", net1, optimizer1)

        torch.manual_seed(99)
        inputs1 = torch.randn(20, 10)
        program1.add_tensor_source("data", inputs1)

        queries = [f"pred({i}, a)" for i in range(8)]

        history1 = pyxlog.train_model(
            program1,
            queries,
            epochs=3,
            batch_size=4,
            log_iter=100,
            shuffle=False,
        )

        # --- program 2 (tensor path) ---
        torch.manual_seed(42)
        net2 = SimpleNet()
        program2 = pyxlog.Program.compile(source)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        program2.register_network("test_net", net2, optimizer2)

        torch.manual_seed(99)
        inputs2 = torch.randn(20, 10)
        program2.add_tensor_source("data", inputs2)

        history2 = pyxlog.train_model_tensor(
            program2,
            queries,
            epochs=3,
            batch_size=4,
            log_iter=100,
            shuffle=False,
        )

        # Epoch losses must match within tolerance
        assert len(history1.epoch_losses) == len(history2.epoch_losses), (
            f"Epoch count mismatch: {len(history1.epoch_losses)} vs {len(history2.epoch_losses)}"
        )
        for i, (s, t) in enumerate(zip(history1.epoch_losses, history2.epoch_losses)):
            assert abs(s - t) < 1e-5, (
                f"Epoch {i} loss mismatch: scalar={s:.8f} tensor={t:.8f}"
            )

    def test_batch_losses_match(self):
        """train_model and train_model_tensor produce the same batch losses."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        # --- program 1 (scalar path) ---
        torch.manual_seed(42)
        net1 = SimpleNet()
        program1 = pyxlog.Program.compile(source)
        optimizer1 = torch.optim.SGD(net1.parameters(), lr=0.01)
        program1.register_network("test_net", net1, optimizer1)

        torch.manual_seed(99)
        inputs1 = torch.randn(20, 10)
        program1.add_tensor_source("data", inputs1)

        queries = [f"pred({i}, a)" for i in range(8)]

        history1 = pyxlog.train_model(
            program1,
            queries,
            epochs=3,
            batch_size=4,
            log_iter=100,
            shuffle=False,
        )

        # --- program 2 (tensor path) ---
        torch.manual_seed(42)
        net2 = SimpleNet()
        program2 = pyxlog.Program.compile(source)
        optimizer2 = torch.optim.SGD(net2.parameters(), lr=0.01)
        program2.register_network("test_net", net2, optimizer2)

        torch.manual_seed(99)
        inputs2 = torch.randn(20, 10)
        program2.add_tensor_source("data", inputs2)

        history2 = pyxlog.train_model_tensor(
            program2,
            queries,
            epochs=3,
            batch_size=4,
            log_iter=100,
            shuffle=False,
        )

        # Batch losses must match within tolerance
        assert len(history1.batch_losses) == len(history2.batch_losses), (
            f"Batch count mismatch: {len(history1.batch_losses)} vs {len(history2.batch_losses)}"
        )
        for i, (s, t) in enumerate(zip(history1.batch_losses, history2.batch_losses)):
            assert abs(s - t) < 1e-5, (
                f"Batch {i} loss mismatch: scalar={s:.8f} tensor={t:.8f}"
            )

    def test_tensor_path_reduces_loss(self):
        """train_model_tensor should reduce loss over epochs, same as scalar path."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        torch.manual_seed(42)
        net = SimpleNet()
        program = pyxlog.Program.compile(source)
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        torch.manual_seed(99)
        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        history = pyxlog.train_model_tensor(
            program,
            queries,
            epochs=10,
            batch_size=5,
            log_iter=100,
            shuffle=False,
        )

        assert history.epoch_losses[-1] < history.epoch_losses[0], (
            f"Loss did not decrease: {history.epoch_losses[0]:.4f} -> {history.epoch_losses[-1]:.4f}"
        )


# ---------------------------------------------------------------------------
# 4b. .item() call count guard -- tensor path calls .item() once per batch
# ---------------------------------------------------------------------------


class TestItemCallCount:
    """Test .item() is only called once per batch, not once per query.

    Uses direct neural predicate queries (fast, no circuit compilation)
    with SGD optimizer (no internal .item() calls) to isolate our code's
    .item() usage.
    """

    def test_tensor_path_fewer_item_calls_than_scalar(self, monkeypatch):
        """Tensor path eliminates per-query .item() — only 1 per batch."""
        source = "nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y)."

        num_queries = 8
        batch_size = 4
        epochs = 2
        num_batches = num_queries // batch_size  # 2

        original_item = torch.Tensor.item

        def make_program():
            torch.manual_seed(42)
            net = SimpleNet()
            program = pyxlog.Program.compile(source)
            opt = torch.optim.SGD(net.parameters(), lr=0.01)
            program.register_network("test_net", net, opt)
            torch.manual_seed(99)
            inputs = torch.randn(20, 10)
            program.add_tensor_source("data", inputs)
            return program

        queries = [f"pred({i}, a)" for i in range(num_queries)]

        # --- Scalar path ---
        program1 = make_program()
        scalar_count = 0
        def scalar_item(tensor_self):
            nonlocal scalar_count
            scalar_count += 1
            return original_item(tensor_self)

        monkeypatch.setattr(torch.Tensor, "item", scalar_item)
        pyxlog.train_model(program1, queries, epochs=epochs, batch_size=batch_size, log_iter=100, shuffle=False)
        monkeypatch.undo()

        # --- Tensor path ---
        program2 = make_program()
        tensor_count = 0
        def tensor_item(tensor_self):
            nonlocal tensor_count
            tensor_count += 1
            return original_item(tensor_self)

        monkeypatch.setattr(torch.Tensor, "item", tensor_item)
        pyxlog.train_model_tensor(program2, queries, epochs=epochs, batch_size=batch_size, log_iter=100, shuffle=False)
        monkeypatch.undo()

        # Scalar: 1 .item() per query = num_queries * epochs
        assert scalar_count == num_queries * epochs, (
            f"Scalar path: expected {num_queries * epochs} .item() calls, got {scalar_count}"
        )
        # Tensor: 1 .item() per batch = num_batches * epochs
        assert tensor_count == num_batches * epochs, (
            f"Tensor path: expected {num_batches * epochs} .item() calls, got {tensor_count}"
        )


# ---------------------------------------------------------------------------
# 4c. Cache reuse across tensor training
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestTensorTrainingCacheReuse:
    """Test circuit cache works with tensor training on complex queries."""

    def test_training_completes_with_addition_queries(self):
        """Tensor training with addition queries should complete and produce valid losses."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = MNISTSimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        queries = [
            f"addition({i}, {i+1}, {(i * 2 + 1) % 19})" for i in range(8)
        ]

        history = pyxlog.train_model_tensor(
            program,
            queries,
            epochs=2,
            batch_size=4,
            log_iter=100,
            shuffle=False,
        )

        # Training must produce the expected number of epoch/batch losses
        assert len(history.epoch_losses) == 2
        assert len(history.batch_losses) == 4  # 2 batches/epoch * 2 epochs

        # All losses must be valid positive numbers
        for i, loss in enumerate(history.epoch_losses):
            assert loss > 0, f"Epoch {i} loss should be positive: {loss}"
        for i, loss in enumerate(history.batch_losses):
            assert loss > 0, f"Batch {i} loss should be positive: {loss}"

    def test_second_epoch_faster_than_first(self):
        """Second epoch should be faster due to circuit cache hits."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = MNISTSimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        queries = [
            f"addition({i}, {i+1}, {(i * 2 + 1) % 19})" for i in range(8)
        ]

        # Epoch 1: includes D4 circuit compilation (cache misses)
        t0 = time.time()
        stats1 = program.train_epoch_tensor(queries, batch_size=4)
        time_epoch1 = time.time() - t0

        # Epoch 2: all cache hits (no D4 compilation)
        t0 = time.time()
        stats2 = program.train_epoch_tensor(queries, batch_size=4)
        time_epoch2 = time.time() - t0

        # Cache should make the second epoch faster
        assert time_epoch2 < time_epoch1, (
            f"Epoch 2 ({time_epoch2:.3f}s) should be faster than epoch 1 ({time_epoch1:.3f}s) "
            f"due to circuit cache"
        )

        # Both epochs should produce valid stats
        assert stats1.avg_loss > 0
        assert stats2.avg_loss > 0

    def test_cache_cleared_forces_recompilation(self):
        """Clearing the circuit cache should force recompilation on next epoch."""
        source = """
nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
"""
        program = pyxlog.Program.compile(source)

        net = MNISTSimpleNet(10).cuda()
        optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
        program.register_network("mnist_net", net, optimizer)

        torch.manual_seed(42)
        dummy_images = torch.randn(100, 1, 28, 28, device="cuda")
        program.add_tensor_source("X", dummy_images)

        queries = [
            f"addition({i}, {i+1}, {(i * 2 + 1) % 19})" for i in range(8)
        ]

        # Warm up the cache
        program.train_epoch_tensor(queries, batch_size=4)

        # Cached epoch should be fast
        t0 = time.time()
        program.train_epoch_tensor(queries, batch_size=4)
        time_cached = time.time() - t0

        # Clear the cache and force recompilation
        program.clear_circuit_cache()

        t0 = time.time()
        stats = program.train_epoch_tensor(queries, batch_size=4)
        time_after_clear = time.time() - t0

        # After cache clear, the epoch should be slower (recompilation needed)
        assert time_after_clear > time_cached, (
            f"After cache clear ({time_after_clear:.3f}s) should be slower "
            f"than cached ({time_cached:.3f}s)"
        )

        # Training should still produce valid results
        assert stats.avg_loss > 0


class TestTensorTrainingGradClipping:
    """Test gradient clipping works through train_model_tensor entrypoint."""

    def test_tensor_grad_clipping_limits_param_delta(self):
        """Tight max_grad_norm via tensor path produces smaller weight changes."""
        source = """
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """

        def make_program_and_net():
            torch.manual_seed(42)
            n = SimpleNet()
            prog = pyxlog.Program.compile(source)
            opt = torch.optim.SGD(n.parameters(), lr=1.0)
            prog.register_network("test_net", n, opt)
            torch.manual_seed(99)
            inputs = torch.randn(20, 10)
            prog.add_tensor_source("data", inputs)
            return prog, n

        queries = [f"pred({i}, a)" for i in range(10)]

        # Run WITHOUT clipping
        prog_no_clip, net_no_clip = make_program_and_net()
        w_before = net_no_clip.fc.weight.clone()
        pyxlog.train_model_tensor(prog_no_clip, queries, epochs=1,
                                  batch_size=10, shuffle=False)
        delta_no_clip = (net_no_clip.fc.weight - w_before).norm().item()

        # Run WITH tight clipping
        prog_clip, net_clip = make_program_and_net()
        w_before = net_clip.fc.weight.clone()
        pyxlog.train_model_tensor(prog_clip, queries, epochs=1,
                                  batch_size=10, shuffle=False,
                                  max_grad_norm=0.001)
        delta_clip = (net_clip.fc.weight - w_before).norm().item()

        assert delta_clip < delta_no_clip, \
            f"Clipped delta {delta_clip:.6f} not smaller than unclipped {delta_no_clip:.6f}"
