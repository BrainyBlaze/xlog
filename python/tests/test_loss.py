"""Tests for NLL (negative log-likelihood) loss function.

NLL loss is the fundamental training objective for neural-symbolic programs:
loss = -log(P(evidence | neural_params))

Run with: pytest python/tests/test_loss.py -v
"""

import pytest
import math

# Skip all tests if xlog_gpu or torch not available
torch = pytest.importorskip("torch")
xlog_gpu = pytest.importorskip("xlog_gpu")


class TestNLLLossBasic:
    """Basic NLL loss computation tests."""

    def test_nll_loss_single_query(self):
        """Test NLL loss: -log(P(query))"""
        program = xlog_gpu.Program.compile("""
            0.7::a().
        """)

        loss = program.nll_loss("a()")

        # -log(0.7) ≈ 0.357
        expected = -math.log(0.7)
        assert abs(loss - expected) < 0.001

    def test_nll_loss_certain_fact(self):
        """Test NLL loss for certain facts (P=1.0, loss=0)."""
        program = xlog_gpu.Program.compile("""
            fact(x).
        """)

        loss = program.nll_loss("fact(x)")

        # -log(1.0) = 0
        assert abs(loss) < 0.001

    def test_nll_loss_low_probability(self):
        """Test NLL loss for low probability (high loss)."""
        program = xlog_gpu.Program.compile("""
            0.1::rare().
        """)

        loss = program.nll_loss("rare()")

        # -log(0.1) ≈ 2.303
        expected = -math.log(0.1)
        assert abs(loss - expected) < 0.001

    def test_nll_loss_impossible_query(self):
        """Test NLL loss for impossible query (should be clamped)."""
        program = xlog_gpu.Program.compile("""
            0.5::a().
            b() :- a(), c().
        """)

        # Loss should be very high but not infinite
        loss = program.nll_loss("b()")
        assert loss > 10  # High loss
        assert not math.isinf(loss)  # But not infinite


class TestNLLLossBatch:
    """Batched NLL loss computation tests."""

    def test_nll_loss_batch_sum(self):
        """Test that batch loss is sum of individual losses."""
        program = xlog_gpu.Program.compile("""
            0.7::a().
            0.3::b().
        """)

        queries = ["a()", "b()"]
        batch_loss = program.nll_loss_batch(queries)

        # Should be sum of individual NLL losses
        loss_a = -math.log(0.7)
        loss_b = -math.log(0.3)
        expected = loss_a + loss_b

        assert abs(batch_loss - expected) < 0.01

    def test_nll_loss_batch_path_queries(self):
        """Test NLL loss over path queries."""
        program = xlog_gpu.Program.compile("""
            0.8::edge(1,2).
            0.6::edge(2,3).
            path(X,Y) :- edge(X,Y).
            path(X,Y) :- edge(X,Z), path(Z,Y).
        """)

        queries = ["path(1,2)", "path(2,3)"]
        loss = program.nll_loss_batch(queries)

        # Both should have positive loss
        assert loss > 0

    def test_nll_loss_batch_empty(self):
        """Test NLL loss with empty batch returns 0."""
        program = xlog_gpu.Program.compile("""
            0.5::a().
        """)

        loss = program.nll_loss_batch([])
        assert loss == 0.0

    def test_nll_loss_batch_single(self):
        """Test batch with single query equals single query loss."""
        program = xlog_gpu.Program.compile("""
            0.6::fact().
        """)

        single_loss = program.nll_loss("fact()")
        batch_loss = program.nll_loss_batch(["fact()"])

        assert abs(single_loss - batch_loss) < 0.001


class TestNLLLossWithRules:
    """NLL loss with complex rule structures."""

    def test_nll_loss_conjunction(self):
        """Test NLL loss for conjunction of probabilistic facts."""
        program = xlog_gpu.Program.compile("""
            0.8::a().
            0.9::b().
            both() :- a(), b().
        """)

        loss = program.nll_loss("both()")

        # P(both) = 0.8 * 0.9 = 0.72
        expected = -math.log(0.72)
        assert abs(loss - expected) < 0.01

    def test_nll_loss_disjunction(self):
        """Test NLL loss for disjunction via multiple rules."""
        program = xlog_gpu.Program.compile("""
            0.5::a().
            0.5::b().
            either() :- a().
            either() :- b().
        """)

        loss = program.nll_loss("either()")

        # P(either) = 1 - (1-0.5)*(1-0.5) = 0.75
        expected = -math.log(0.75)
        assert abs(loss - expected) < 0.01

    def test_nll_loss_negation(self):
        """Test NLL loss with negation."""
        program = xlog_gpu.Program.compile("""
            0.3::rain().
            dry() :- not rain().
        """)

        loss = program.nll_loss("dry()")

        # P(dry) = P(not rain) = 0.7
        expected = -math.log(0.7)
        assert abs(loss - expected) < 0.01


class TestNLLLossNeuralIntegration:
    """NLL loss integration with neural predicates."""

    def test_nll_loss_with_neural_predicate_declared(self):
        """Test program with neural predicate can compute loss."""
        program = xlog_gpu.Program.compile("""
            nn(digit_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
            0.9::correct(0).
        """)

        # Loss on non-neural query should work
        loss = program.nll_loss("correct(0)")
        expected = -math.log(0.9)
        assert abs(loss - expected) < 0.01


class TestNLLLossMean:
    """Mean NLL loss (average over batch)."""

    def test_nll_loss_mean(self):
        """Test mean NLL loss computation."""
        program = xlog_gpu.Program.compile("""
            0.7::a().
            0.3::b().
            0.5::c().
        """)

        queries = ["a()", "b()", "c()"]
        mean_loss = program.nll_loss_mean(queries)

        # Should be average of individual NLL losses
        loss_a = -math.log(0.7)
        loss_b = -math.log(0.3)
        loss_c = -math.log(0.5)
        expected = (loss_a + loss_b + loss_c) / 3

        assert abs(mean_loss - expected) < 0.01

    def test_nll_loss_mean_empty_raises(self):
        """Test that mean NLL loss with empty batch raises error."""
        program = xlog_gpu.Program.compile("""
            0.5::a().
        """)

        with pytest.raises(ValueError, match="empty"):
            program.nll_loss_mean([])


class TestNLLLossGradient:
    """Tests for gradient computation through NLL loss."""

    def test_nll_loss_tensor_output(self):
        """Test NLL loss can return PyTorch tensor for gradient computation."""
        program = xlog_gpu.Program.compile("""
            0.7::a().
        """)

        loss_tensor = program.nll_loss_tensor("a()")

        # Should be a torch tensor
        assert isinstance(loss_tensor, torch.Tensor)
        assert loss_tensor.ndim == 0  # Scalar
        assert abs(loss_tensor.item() - (-math.log(0.7))) < 0.001

    def test_nll_loss_batch_tensor(self):
        """Test batch NLL loss returns tensor."""
        program = xlog_gpu.Program.compile("""
            0.7::a().
            0.3::b().
        """)

        loss_tensor = program.nll_loss_batch_tensor(["a()", "b()"])

        assert isinstance(loss_tensor, torch.Tensor)
        expected = -math.log(0.7) + (-math.log(0.3))
        assert abs(loss_tensor.item() - expected) < 0.01
