"""Tests for tensor source registry.

These tests verify that tensor sources can be registered and managed
for neural predicate evaluation.

Run with: pytest python/tests/test_tensor_source.py -v
"""

import pytest

# Skip all tests if pyxlog or torch not available
torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")


class TestTensorSourceBasics:
    """Basic tensor source registration tests."""

    def test_add_tensor_source(self):
        """Test adding a tensor source."""
        program = pyxlog.Program.compile("""
            nn(digit_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        # Create a fake MNIST-like tensor
        images = torch.randn(1000, 1, 28, 28)
        program.add_tensor_source("train", images)

        assert "train" in program.tensor_source_names()

    def test_first_source_becomes_active(self):
        """Test that the first source added becomes active automatically."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [a, b]) :: pred(X, Y).
        """)

        data = torch.randn(100, 10)
        program.add_tensor_source("data", data)

        assert program.active_tensor_source() == "data"

    def test_multiple_tensor_sources(self):
        """Test adding multiple tensor sources."""
        program = pyxlog.Program.compile("""
            nn(classifier, [X], Y, [0,1,2]) :: classify(X, Y).
        """)

        train = torch.randn(60000, 784)
        val = torch.randn(10000, 784)
        test = torch.randn(10000, 784)

        program.add_tensor_source("train", train)
        program.add_tensor_source("val", val)
        program.add_tensor_source("test", test)

        names = program.tensor_source_names()
        assert len(names) == 3
        assert "train" in names
        assert "val" in names
        assert "test" in names

    def test_set_active_tensor_source(self):
        """Test switching the active tensor source."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        program.add_tensor_source("train", torch.randn(1000, 10))
        program.add_tensor_source("test", torch.randn(200, 10))

        # First one is active by default
        assert program.active_tensor_source() == "train"

        # Switch to test
        program.set_active_tensor_source("test")
        assert program.active_tensor_source() == "test"

        # Switch back to train
        program.set_active_tensor_source("train")
        assert program.active_tensor_source() == "train"

    def test_set_invalid_active_source_raises(self):
        """Test that setting an invalid active source raises an error."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        with pytest.raises(ValueError, match="not found"):
            program.set_active_tensor_source("nonexistent")


class TestTensorSourceMetadata:
    """Tests for tensor source metadata extraction."""

    def test_active_tensor_source_size(self):
        """Test getting the size of the active source."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        program.add_tensor_source("data", torch.randn(500, 10))
        assert program.active_tensor_source_size() == 500

    def test_tensor_source_size_with_switch(self):
        """Test size changes when switching active source."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        program.add_tensor_source("small", torch.randn(100, 10))
        program.add_tensor_source("large", torch.randn(10000, 10))

        assert program.active_tensor_source_size() == 100

        program.set_active_tensor_source("large")
        assert program.active_tensor_source_size() == 10000

    def test_has_tensor_source(self):
        """Test checking if a tensor source exists."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        assert not program.has_tensor_source("data")

        program.add_tensor_source("data", torch.randn(100, 10))

        assert program.has_tensor_source("data")
        assert not program.has_tensor_source("other")


class TestTensorSourceShapes:
    """Tests for various tensor shapes."""

    def test_1d_tensor(self):
        """Test 1D tensor (vector per sample)."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        # 500 samples, each a 10-element vector
        program.add_tensor_source("embeddings", torch.randn(500, 10))
        assert program.active_tensor_source_size() == 500

    def test_2d_tensor_grayscale(self):
        """Test 2D tensor (grayscale images)."""
        program = pyxlog.Program.compile("""
            nn(mnist_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
        """)

        # 1000 grayscale MNIST images: [batch, channel, height, width]
        program.add_tensor_source("mnist", torch.randn(1000, 1, 28, 28))
        assert program.active_tensor_source_size() == 1000

    def test_3d_tensor_rgb(self):
        """Test 3D tensor (RGB images)."""
        program = pyxlog.Program.compile("""
            nn(imagenet, [X], Y, [cat, dog, bird]) :: classify(X, Y).
        """)

        # 500 RGB images: [batch, channel, height, width]
        program.add_tensor_source("images", torch.randn(500, 3, 224, 224))
        assert program.active_tensor_source_size() == 500


class TestTensorSourceDtypes:
    """Tests for various tensor dtypes."""

    def test_float32_tensor(self):
        """Test float32 tensor (default)."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        data = torch.randn(100, 10, dtype=torch.float32)
        program.add_tensor_source("data", data)
        assert program.has_tensor_source("data")

    def test_float64_tensor(self):
        """Test float64 tensor."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        data = torch.randn(100, 10, dtype=torch.float64)
        program.add_tensor_source("data", data)
        assert program.has_tensor_source("data")

    def test_int_tensor(self):
        """Test integer tensor (for label data)."""
        program = pyxlog.Program.compile("""
            nn(net, [X], Y, [0,1]) :: pred(X, Y).
        """)

        # Labels stored as int64
        labels = torch.randint(0, 10, (100,), dtype=torch.int64)
        program.add_tensor_source("labels", labels)
        assert program.has_tensor_source("labels")


class TestTensorSourceIntegration:
    """Integration tests for tensor sources with neural predicates."""

    def test_mnist_addition_setup(self):
        """Test setting up tensor sources for MNIST addition."""
        program = pyxlog.Program.compile("""
            nn(digit_net, [X], Y, [0,1,2,3,4,5,6,7,8,9]) :: digit(X, Y).
            addition(FirstImage, SecondImage, Sum) :-
                digit(FirstImage, FirstDigitValue),
                digit(SecondImage, SecondDigitValue),
                Sum is FirstDigitValue + SecondDigitValue.
        """)

        # Training data
        train_images = torch.randn(60000, 1, 28, 28)
        train_labels = torch.randint(0, 10, (60000,))

        # Test data
        test_images = torch.randn(10000, 1, 28, 28)
        test_labels = torch.randint(0, 10, (10000,))

        program.add_tensor_source("train_images", train_images)
        program.add_tensor_source("train_labels", train_labels)
        program.add_tensor_source("test_images", test_images)
        program.add_tensor_source("test_labels", test_labels)

        # Set images as active for inference
        program.set_active_tensor_source("train_images")
        assert program.active_tensor_source() == "train_images"
        assert program.active_tensor_source_size() == 60000

    def test_sequence_labeling_setup(self):
        """Test setting up tensor sources for sequence labeling."""
        program = pyxlog.Program.compile("""
            nn(tagger, [Word], WordTag, [noun, verb, adj, det]) :: pos_tag(Word, WordTag).
            valid_phrase(FirstWord, SecondWord, ThirdWord) :-
                pos_tag(FirstWord, det),
                pos_tag(SecondWord, adj),
                pos_tag(ThirdWord, noun).
        """)

        # Word embeddings: 10000 words, 300-dim embeddings
        embeddings = torch.randn(10000, 300)
        program.add_tensor_source("embeddings", embeddings)

        assert program.active_tensor_source() == "embeddings"
        assert program.active_tensor_source_size() == 10000

    def test_multi_network_different_sources(self):
        """Test multiple networks with different tensor sources."""
        program = pyxlog.Program.compile("""
            nn(image_encoder, [ImageInput], ImageEmbedding) :: encode_image(ImageInput, ImageEmbedding).
            nn(text_encoder, [TextInput], TextEmbedding) :: encode_text(TextInput, TextEmbedding).
            nn(classifier, [Embedding], MatchLabel, [match, no_match]) :: classify(Embedding, MatchLabel).
        """)

        # Different tensor sources for different input types
        images = torch.randn(5000, 3, 224, 224)
        texts = torch.randn(5000, 768)  # Text embeddings

        program.add_tensor_source("images", images)
        program.add_tensor_source("texts", texts)

        assert len(program.tensor_source_names()) == 2
        assert program.has_tensor_source("images")
        assert program.has_tensor_source("texts")
