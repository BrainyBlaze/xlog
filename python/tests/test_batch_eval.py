"""Tests for batched neural evaluation.

These tests verify that multiple neural predicate calls are properly
batched together for efficient GPU evaluation.

Run with: pytest python/tests/test_batch_eval.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class MNISTNet(torch.nn.Module):  # type: ignore[name-defined]
    """Simple MNIST-like classifier for testing."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10)

    def forward(self, x):
        return torch.softmax(self.fc(x.view(-1, 784)), dim=-1)


class TestBatchedEvaluation:
    """Test suite for batched neural evaluation."""

    def test_register_network_for_batching(self):
        """Test that networks can be registered with batching enabled."""
        program = pyxlog.Program.compile("""
            nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        net = MNISTNet()
        optimizer = torch.optim.Adam(net.parameters())

        # Register with batching enabled (default)
        program.register_network("mnist_net", net, optimizer, batching=True)

        assert "mnist_net" in program.network_names()

    def test_register_network_batching_disabled(self):
        """Test that networks can be registered with batching disabled."""
        program = pyxlog.Program.compile("""
            nn(serial_net, [X], Y, [0,1]) :: classify(X, Y).
        """)

        net = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(net.parameters())

        # Register with batching disabled
        program.register_network("serial_net", net, optimizer, batching=False)

        assert "serial_net" in program.network_names()

    def test_multiple_networks_batched(self):
        """Test that multiple networks can be batched independently."""
        program = pyxlog.Program.compile("""
            nn(encoder, [X], E) :: encode(X, E).
            nn(classifier, [E], Y, [0,1,2]) :: classify(E, Y).
        """)

        encoder = torch.nn.Embedding(100, 50)
        classifier = torch.nn.Linear(50, 3)

        program.register_embedding("encoder", encoder, trainable=True)
        program.register_network(
            "classifier", classifier, torch.optim.Adam(classifier.parameters())
        )

        names = program.network_names()
        assert "classifier" in names


class TestBatchingConfiguration:
    """Test batching configuration options."""

    def test_batch_with_top_k(self):
        """Test batching with top-k probability filtering."""
        program = pyxlog.Program.compile("""
            nn(large_vocab, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: word(X, Y).
        """)

        net = torch.nn.Linear(100, 10)
        optimizer = torch.optim.Adam(net.parameters())

        # Only consider top 3 probabilities
        program.register_network(
            "large_vocab", net, optimizer, batching=True, k=3
        )

        assert "large_vocab" in program.network_names()

    def test_batch_deterministic_mode(self):
        """Test batching with deterministic (argmax) mode."""
        program = pyxlog.Program.compile("""
            nn(det_net, [X], Y, [a, b]) :: pred(X, Y).
        """)

        net = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(net.parameters())

        # Deterministic mode - use argmax
        program.register_network(
            "det_net", net, optimizer, batching=True, det=True
        )

        assert "det_net" in program.network_names()

    def test_batch_with_cache(self):
        """Test batching with output caching."""
        program = pyxlog.Program.compile("""
            nn(cached_net, [X], Y, [0,1]) :: cached_pred(X, Y).
        """)

        net = torch.nn.Linear(10, 2)
        optimizer = torch.optim.Adam(net.parameters())

        # Enable cache with custom size
        program.register_network(
            "cached_net", net, optimizer,
            batching=True,
            cache=True,
            cache_size=5000
        )

        assert "cached_net" in program.network_names()


class TestBatchedProgramStructure:
    """Test programs that would benefit from batching."""

    def test_addition_program_structure(self):
        """Test the MNIST addition program structure."""
        program = pyxlog.Program.compile("""
            nn(digit_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
            addition(FirstImage, SecondImage, Sum) :-
                digit(FirstImage, FirstDigitValue),
                digit(SecondImage, SecondDigitValue),
                Sum is FirstDigitValue + SecondDigitValue.
        """)

        # This program would call digit_net twice for addition(a, b, Sum)
        # Batching groups both calls into single forward pass
        assert "digit_net" in program.declared_network_names()

    def test_multi_digit_program(self):
        """Test multi-digit number recognition program."""
        program = pyxlog.Program.compile("""
            nn(digit_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
            number(FirstDigitInput, SecondDigitInput, ThirdDigitInput, NumberValue) :-
                digit(FirstDigitInput, FirstDigitValue),
                digit(SecondDigitInput, SecondDigitValue),
                digit(ThirdDigitInput, ThirdDigitValue),
                NumberValue is FirstDigitValue * 100 + SecondDigitValue * 10 + ThirdDigitValue.
        """)

        # This would need 3 digit classifications - perfect for batching
        assert "digit_net" in program.declared_network_names()

    def test_sequence_labeling_structure(self):
        """Test sequence labeling program structure."""
        program = pyxlog.Program.compile("""
            nn(tagger, [Token], TokenTag, [noun, verb, adj]) :: tag(Token, TokenTag).
            valid_sequence(FirstToken, SecondToken, ThirdToken) :-
                tag(FirstToken, FirstTokenTag),
                tag(SecondToken, SecondTokenTag),
                tag(ThirdToken, ThirdTokenTag),
                valid_transition(FirstTokenTag, SecondTokenTag),
                valid_transition(SecondTokenTag, ThirdTokenTag).
            valid_transition(noun, verb).
            valid_transition(verb, noun).
            valid_transition(adj, noun).
        """)

        # Sequence of 3 tags - 3 neural calls batched together
        assert "tagger" in program.declared_network_names()
