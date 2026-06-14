"""Tests for term embedding registration and forward_embedding API."""

import pytest
import torch
import pyxlog


EMBEDDING_SOURCE = """
    nn(entity_embed, [Entity], EmbeddingVector) :: embed(Entity, EmbeddingVector).
"""

CLASSIFICATION_SOURCE = """
    nn(classifier, [Item], ClassLabel, [0, 1, 2]) :: classify(Item, ClassLabel).
"""


class TestRegisterEmbeddingNnEmbedding:
    """Test 1: register_embedding with nn.Embedding produces correct vectors."""

    def test_forward_embedding_shape_and_values(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        vocabulary_size, embedding_dimension = 10, 8
        embedding = torch.nn.Embedding(vocabulary_size, embedding_dimension)

        program.register_embedding("entity_embed", embedding, trainable=True)

        # Look up 3 entities
        result = program.forward_embedding("entity_embed", [0, 3, 7])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, embedding_dimension)

        # Verify values match direct nn.Embedding call
        expected = embedding(torch.tensor([0, 3, 7]))
        assert torch.allclose(result, expected)


class TestRegisterEmbeddingFrozenTensor:
    """Test 2: register_embedding with frozen torch.Tensor."""

    def test_frozen_tensor_correct_values(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        program.register_embedding("entity_embed", weights, trainable=False)

        result = program.forward_embedding("entity_embed", [2, 5])

        assert isinstance(result, torch.Tensor)
        assert result.shape == (2, 8)
        assert torch.allclose(result, weights[[2, 5]])

    def test_trainable_true_with_tensor_rejected(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        with pytest.raises(ValueError, match="trainable=True requires nn.Embedding"):
            program.register_embedding("entity_embed", weights, trainable=True)


class TestCrossRegistrationErrors:
    """Test 3: cross-registration errors in both directions."""

    def test_embedding_declaration_reject_register_network(self):
        """Embedding declaration + register_network -> error."""
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        embedding_model = torch.nn.Embedding(10, 8)
        optimizer = torch.optim.Adam(embedding_model.parameters())

        with pytest.raises(ValueError, match="is an embedding.*register_embedding"):
            program.register_network("entity_embed", embedding_model, optimizer)

    def test_classification_declaration_reject_register_embedding(self):
        """Classification declaration + register_embedding -> error."""
        program = pyxlog.Program.compile(CLASSIFICATION_SOURCE)

        embedding = torch.nn.Embedding(10, 8)

        with pytest.raises(ValueError, match="is a classification.*register_network"):
            program.register_embedding("classifier", embedding, trainable=True)


class TestGradientFlow:
    """Test 4: gradient flow through forward_embedding."""

    def test_backward_updates_embedding_weights(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        embedding = torch.nn.Embedding(10, 8)
        optimizer = torch.optim.SGD(embedding.parameters(), lr=0.1)

        program.register_embedding("entity_embed", embedding, trainable=True)

        # Save original weights
        original_weights = embedding.weight.data.clone()

        # Forward
        result = program.forward_embedding("entity_embed", [0, 1])

        # External loss
        target = torch.randn(2, 8)
        loss = torch.nn.functional.mse_loss(result, target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Weights must have changed
        assert not torch.equal(embedding.weight.data, original_weights)


class TestFrozenOutputNonTrainable:
    """Test 5: frozen embedding output has no gradient path."""

    def test_frozen_tensor_requires_grad_false(self):
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8)
        program.register_embedding("entity_embed", weights, trainable=False)

        result = program.forward_embedding("entity_embed", [0, 1])
        assert not result.requires_grad

    def test_requires_grad_tensor_detached_on_register(self):
        """Raw tensor with requires_grad=True is detached; output has no gradient."""
        program = pyxlog.Program.compile(EMBEDDING_SOURCE)

        weights = torch.randn(10, 8, requires_grad=True)
        program.register_embedding("entity_embed", weights, trainable=False)

        result = program.forward_embedding("entity_embed", [0, 1])
        assert not result.requires_grad


class TestMixedFormRejection:
    """Test compile-time rejection of mixed-form network names."""

    def test_same_name_both_forms_rejected(self):
        """Same network name as both embedding and classification -> compile error."""
        with pytest.raises(ValueError, match="declared as both classification and embedding"):
            pyxlog.Program.compile("""
                nn(shared, [Entity], EmbeddingVector) :: embed(Entity, EmbeddingVector).
                nn(shared, [Item], ClassLabel, [0, 1]) :: classify(Item, ClassLabel).
            """)
