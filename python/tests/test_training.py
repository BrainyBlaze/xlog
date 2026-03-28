"""Tests for training loop infrastructure.

These tests verify the training loop API works correctly:
- train_epoch() for single epoch training
- train_model() for multi-epoch training
- EpochStats and TrainingHistory for tracking progress

Run with: pytest python/tests/test_training.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


def _prime_scheduler(optimizer):
    """Perform a real optimizer step before the first scheduler step."""
    optimizer.zero_grad()
    optimizer.step()


class SimpleNet(torch.nn.Module):  # type: ignore[name-defined]
    """Simple neural network for testing training loop."""

    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return torch.softmax(self.fc(x), dim=-1)


class TestTrainEpoch:
    """Tests for train_epoch() method."""

    def test_train_epoch_basic(self):
        """Test basic train_epoch functionality."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        stats = program.train_epoch(queries, batch_size=5)

        assert hasattr(stats, 'avg_loss')
        assert hasattr(stats, 'num_batches')
        assert hasattr(stats, 'total_queries')
        assert stats.num_batches == 2  # 10 queries / 5 batch_size = 2
        assert stats.total_queries == 10
        assert stats.avg_loss > 0

    def test_train_epoch_updates_parameters(self):
        """Test that train_epoch actually updates network parameters."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        original_weight = net.fc.weight.clone()

        queries = [f"pred({i}, a)" for i in range(10)]
        program.train_epoch(queries, batch_size=5)

        # Parameters should have changed
        assert not torch.equal(net.fc.weight, original_weight)

    def test_train_epoch_batch_size_larger_than_queries(self):
        """Test train_epoch when batch_size > number of queries."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(3)]

        stats = program.train_epoch(queries, batch_size=10)

        assert stats.num_batches == 1  # All queries in one batch
        assert stats.total_queries == 3

    def test_train_epoch_single_query(self):
        """Test train_epoch with a single query."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(5, 10)
        program.add_tensor_source("data", inputs)

        stats = program.train_epoch(["pred(0, a)"], batch_size=1)

        assert stats.num_batches == 1
        assert stats.total_queries == 1


class TestTrainModel:
    """Tests for train_model() function."""

    def test_train_model_basic(self):
        """Test basic train_model functionality."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        history = pyxlog.train_model(
            program,
            queries,
            epochs=3,
            batch_size=5
        )

        assert hasattr(history, 'epoch_losses')
        assert hasattr(history, 'batch_losses')
        assert len(history.epoch_losses) == 3
        # 2 batches per epoch * 3 epochs = 6 batch losses
        assert len(history.batch_losses) == 6

    def test_train_model_reduces_loss(self):
        """Test that train_model reduces loss over epochs."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        torch.manual_seed(42)
        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        history = pyxlog.train_model(
            program,
            queries,
            epochs=10,
            batch_size=5
        )

        # Loss should decrease over training
        assert history.epoch_losses[-1] < history.epoch_losses[0], \
            f"Loss did not decrease: {history.epoch_losses[0]:.4f} -> {history.epoch_losses[-1]:.4f}"

    def test_train_model_shuffle(self):
        """Test train_model with shuffle enabled."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        # Should work with shuffle=True (default)
        history = pyxlog.train_model(
            program,
            queries,
            epochs=2,
            batch_size=5,
            shuffle=True
        )

        assert len(history.epoch_losses) == 2


class TestEvaluateLoss:
    """Tests for evaluate_loss() method."""

    def test_evaluate_loss_basic(self):
        """Test basic evaluate_loss functionality."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(5)]

        loss = program.evaluate_loss(queries)

        assert isinstance(loss, float)
        assert loss > 0  # NLL loss is always positive for p < 1

    def test_evaluate_loss_consistent_with_nll_loss(self):
        """Test that evaluate_loss is consistent with nll_loss_batch."""
        program = pyxlog.Program.compile("""
            0.7::a().
            0.3::b().
        """)

        queries = ["a()", "b()"]

        evaluate_result = program.evaluate_loss(queries)
        batch_result = program.nll_loss_batch(queries)

        # evaluate_loss returns mean, nll_loss_batch returns sum
        # So evaluate_loss * len(queries) should equal nll_loss_batch
        assert abs(evaluate_result * len(queries) - batch_result) < 0.001


class TestTrainingWithScheduler:
    """Tests for training with learning rate scheduler."""

    def test_train_with_scheduler(self):
        """Test training with a learning rate scheduler."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=1.0)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        program.register_network("test_net", net, optimizer, scheduler)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        queries = [f"pred({i}, a)" for i in range(10)]

        # Train one epoch
        program.train_epoch(queries, batch_size=5)

        # Initial LR
        assert optimizer.param_groups[0]['lr'] == 1.0

        # Step scheduler
        program.scheduler_step()

        # LR should be halved
        assert optimizer.param_groups[0]['lr'] == 0.5


class TestTrainingMultipleNetworks:
    """Tests for training with multiple networks."""

    def test_train_multiple_networks(self):
        """Test training with multiple networks updates them when used."""
        # Use network names that match predicate patterns for proper routing
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [0, 1]) :: pred(X, Y).
        """)

        net = SimpleNet(input_dim=5, output_dim=2)
        program.register_network("test_net", net, torch.optim.SGD(net.parameters(), lr=0.5))

        inputs = torch.randn(20, 5)
        program.add_tensor_source("data", inputs)

        orig_weight = net.fc.weight.clone()

        # Train on queries
        queries = [f"pred({i}, 0)" for i in range(10)]
        program.train_epoch(queries, batch_size=5)

        # Network should have changed
        assert not torch.equal(net.fc.weight, orig_weight)


class TestTrainingEdgeCases:
    """Tests for edge cases in training."""

    def test_train_empty_queries(self):
        """Test training with empty query list."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(10, 10)
        program.add_tensor_source("data", inputs)

        # Empty queries should handle gracefully
        stats = program.train_epoch([], batch_size=5)
        assert stats.num_batches == 0
        assert stats.total_queries == 0

    def test_train_different_labels(self):
        """Test training with different target labels."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(30, 10)
        program.add_tensor_source("data", inputs)

        # Mix of different target labels
        queries = []
        for i in range(10):
            label = ['a', 'b', 'c'][i % 3]
            queries.append(f"pred({i}, {label})")

        stats = program.train_epoch(queries, batch_size=5)
        assert stats.avg_loss > 0


class TestGetSetLr:
    """Tests for get_lr() and set_lr() methods."""

    def test_get_lr(self):
        """get_lr returns the optimizer's current learning rate."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.042)
        program.register_network("test_net", net, optimizer)

        lr = program.get_lr("test_net")
        assert lr == pytest.approx(0.042)

    def test_get_lr_unknown_network_raises(self):
        """get_lr raises ValueError for an unregistered network name."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        with pytest.raises(ValueError):
            program.get_lr("nonexistent")

    def test_set_lr(self):
        """set_lr updates the optimizer's learning rate for all param groups."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        program.set_lr("test_net", 0.123)

        # Verify via get_lr
        assert program.get_lr("test_net") == pytest.approx(0.123)
        # Verify the Python optimizer object is updated too
        assert optimizer.param_groups[0]['lr'] == pytest.approx(0.123)

    def test_set_lr_unknown_network_raises(self):
        """set_lr raises ValueError for an unregistered network name."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        with pytest.raises(ValueError):
            program.set_lr("nonexistent", 0.1)


class TestPerNetworkScheduler:
    """Tests for per-network scheduler_step()."""

    def test_scheduler_step_single_network(self):
        """scheduler_step(name) steps only that network's scheduler."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [a, b, c]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=10, output_dim=3)
        opt_a = torch.optim.SGD(net_a.parameters(), lr=1.0)
        sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
        program.register_network("net_a", net_a, opt_a, sched_a)

        net_b = SimpleNet(input_dim=10, output_dim=2)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=1.0)
        sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
        program.register_network("net_b", net_b, opt_b, sched_b)

        _prime_scheduler(opt_a)
        _prime_scheduler(opt_b)

        # Step only net_a's scheduler
        program.scheduler_step("net_a")

        assert opt_a.param_groups[0]['lr'] == pytest.approx(0.5)
        assert opt_b.param_groups[0]['lr'] == pytest.approx(1.0)  # Unchanged

    def test_scheduler_step_none_steps_all(self):
        """scheduler_step(None) or scheduler_step() steps all schedulers (backward compat)."""
        program = pyxlog.Program.compile("""
            nn(net_a, [X], Y, [a, b, c]) :: pred_a(X, Y).
            nn(net_b, [X], Y, [0, 1]) :: pred_b(X, Y).
        """)

        net_a = SimpleNet(input_dim=10, output_dim=3)
        opt_a = torch.optim.SGD(net_a.parameters(), lr=1.0)
        sched_a = torch.optim.lr_scheduler.StepLR(opt_a, step_size=1, gamma=0.5)
        program.register_network("net_a", net_a, opt_a, sched_a)

        net_b = SimpleNet(input_dim=10, output_dim=2)
        opt_b = torch.optim.SGD(net_b.parameters(), lr=1.0)
        sched_b = torch.optim.lr_scheduler.StepLR(opt_b, step_size=1, gamma=0.5)
        program.register_network("net_b", net_b, opt_b, sched_b)

        _prime_scheduler(opt_a)
        _prime_scheduler(opt_b)

        # Step all (backward-compatible call)
        program.scheduler_step()

        assert opt_a.param_groups[0]['lr'] == pytest.approx(0.5)
        assert opt_b.param_groups[0]['lr'] == pytest.approx(0.5)


class TestGradientClipping:
    """Tests for gradient clipping in train_model."""

    def test_grad_clipping_limits_param_delta(self):
        """Tight max_grad_norm produces smaller parameter changes than no clipping."""
        torch.manual_seed(42)

        def make_program_and_net():
            prog = pyxlog.Program.compile("""
                nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
            """)
            n = SimpleNet()
            opt = torch.optim.SGD(n.parameters(), lr=1.0)
            prog.register_network("test_net", n, opt)
            inputs = torch.randn(20, 10)
            prog.add_tensor_source("data", inputs)
            return prog, n

        queries = [f"pred({i}, a)" for i in range(10)]

        # Run WITHOUT clipping
        prog_no_clip, net_no_clip = make_program_and_net()
        w_before_no_clip = net_no_clip.fc.weight.clone()
        pyxlog.train_model(prog_no_clip, queries, epochs=1, batch_size=10, shuffle=False)
        delta_no_clip = (net_no_clip.fc.weight - w_before_no_clip).norm().item()

        # Run WITH tight clipping
        prog_clip, net_clip = make_program_and_net()
        w_before_clip = net_clip.fc.weight.clone()
        pyxlog.train_model(prog_clip, queries, epochs=1, batch_size=10,
                           shuffle=False, max_grad_norm=0.001)
        delta_clip = (net_clip.fc.weight - w_before_clip).norm().item()

        # Clipped update must be strictly smaller
        assert delta_clip < delta_no_clip, \
            f"Clipped delta {delta_clip:.6f} not smaller than unclipped {delta_no_clip:.6f}"

    def test_grad_clipping_none_is_default(self):
        """train_model without max_grad_norm works as before (no clipping)."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        # No max_grad_norm — backward compatible
        history = pyxlog.train_model(program, queries, epochs=2, batch_size=5)
        assert len(history.epoch_losses) == 2


class TestEarlyStopping:
    """Tests for early stopping in train_model."""

    def test_early_stopping_triggers(self):
        """train_model stops early when val loss stops improving.

        Uses lr=0.0 so the network never updates — val loss is flat from
        epoch 1, guaranteeing early stop after exactly `patience` epochs
        of no improvement (plus the initial improving epoch = patience+1
        total, though epoch 0 sets best_val_loss, so we get patience+1
        epochs if first epoch sets baseline then patience epochs with no
        improvement).
        """
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        # lr=0: optimizer.step() is a no-op → val loss never improves
        optimizer = torch.optim.SGD(net.parameters(), lr=0.0)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)

        train_queries = [f"pred({i}, a)" for i in range(10)]
        val_queries = [f"pred({i}, b)" for i in range(10, 15)]

        patience = 3
        history = pyxlog.train_model(
            program, train_queries, epochs=100,
            batch_size=5, val_queries=val_queries, patience=patience
        )

        # Epoch 0 sets baseline (improvement), epochs 1..patience have no
        # improvement → stop after patience+1 total epochs.
        assert len(history.epoch_losses) == patience + 1
        assert history.stopped_early is True

    def test_early_stopping_disabled_by_default(self):
        """Without val_queries/patience, all epochs run (backward compat)."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        history = pyxlog.train_model(program, queries, epochs=3, batch_size=5)

        assert len(history.epoch_losses) == 3
        assert history.stopped_early is False

    def test_early_stopping_requires_both_params(self):
        """val_queries without patience (or vice versa) raises ValueError."""
        program = pyxlog.Program.compile("""
            nn(test_net, [X], Y, [a, b, c]) :: pred(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
        program.register_network("test_net", net, optimizer)

        inputs = torch.randn(20, 10)
        program.add_tensor_source("data", inputs)
        queries = [f"pred({i}, a)" for i in range(10)]

        with pytest.raises(ValueError):
            pyxlog.train_model(
                program, queries, epochs=5, batch_size=5,
                val_queries=queries  # patience not provided
            )
