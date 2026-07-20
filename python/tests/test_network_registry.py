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


class SimpleNet(torch.nn.Module):  # type: ignore[name-defined]
    """Simple network for testing - classifies 784-dim input to 10 classes."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(784, 10)

    def forward(self, x):
        return torch.softmax(self.fc(x.view(-1, 784)), dim=-1)


class EmbeddingNet(torch.nn.Module):  # type: ignore[name-defined]
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
            nn(first_digit_net, [FirstInput], FirstDigitLabel, [0,1,2,3,4,5,6,7,8,9]) :: first_digit(FirstInput, FirstDigitLabel).
            nn(second_digit_net, [SecondInput], SecondDigitLabel, [0,1,2,3,4,5,6,7,8,9]) :: second_digit(SecondInput, SecondDigitLabel).
            addition(FirstInput, SecondInput, Sum) :-
                first_digit(FirstInput, FirstDigitValue),
                second_digit(SecondInput, SecondDigitValue),
                Sum is FirstDigitValue + SecondDigitValue.
        """)

        first_network = SimpleNet()
        second_network = SimpleNet()
        first_optimizer = torch.optim.Adam(first_network.parameters())
        second_optimizer = torch.optim.Adam(second_network.parameters())

        program.register_network("first_digit_net", first_network, first_optimizer)
        program.register_network("second_digit_net", second_network, second_optimizer)

        names = program.network_names()
        assert "first_digit_net" in names
        assert "second_digit_net" in names
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

        embedding = torch.nn.Embedding(100, 128)
        program.register_embedding("encoder", embedding, trainable=True)

        # Verify it is registered (not in network_names, which is classification only)
        result = program.forward_embedding("encoder", [0])
        assert result.shape == (1, 128)


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
            nn(multi_net, [FirstInput, SecondInput, ThirdInput], OutputLabel, [0,1,2]) :: classify(FirstInput, SecondInput, ThirdInput, OutputLabel).
        """)

        # Network that takes 3 inputs
        class MultiInputNet(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = torch.nn.Linear(30, 3)

            def forward(self, first_input, second_input, third_input):
                combined = torch.cat(
                    [first_input, second_input, third_input],
                    dim=-1,
                )
                return torch.softmax(self.fc(combined), dim=-1)

        net = MultiInputNet()
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network("multi_net", net, optimizer)

        assert "multi_net" in program.network_names()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Program.compile requires CUDA (device=0)")
class TestRegistrationMetadata:
    """Test suite for the arity/arg_sorts/artifact_hash registration surface
    (register_network's new kwargs) and its network_metadata(name) getter."""

    def test_registration_metadata_roundtrips(self):
        """arity matching the declaration, arg_sorts of that length, and an
        artifact_hash all round-trip through network_metadata exactly, plus
        the declared nn/4 entry is reported with the right shape."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network(
            "evt_net",
            net,
            optimizer,
            arity=2,
            arg_sorts=(0, 1),
            artifact_hash="sha256:test",
        )

        meta = program.network_metadata("evt_net")
        assert meta["arity"] == 2
        assert meta["arg_sorts"] == [0, 1]
        # arg_sorts are catalog sort ids (ints); bool is a subclass of int in
        # Python, so this pins the roundtrip to actual ints, not bools.
        assert all(type(sort_id) is int for sort_id in meta["arg_sorts"])
        assert meta["artifact_hash"] == "sha256:test"
        assert meta["declared"] == [
            {
                "predicate": "event_label",
                "predicate_arity": 2,
                "input_arity": 1,
                "labels": ["0", "1"],
            }
        ]

    def test_reregistration_is_retrain_last_write_wins(self):
        """Re-registering the same name with a fresh module/optimizer and a
        new artifact_hash overwrites the prior registration -- the
        documented retrain semantics are last-write-wins, not accumulate or
        refuse."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net_a = SimpleNet()
        optimizer_a = torch.optim.Adam(net_a.parameters())
        program.register_network(
            "evt_net", net_a, optimizer_a, artifact_hash="hash-A",
        )
        assert program.network_metadata("evt_net")["artifact_hash"] == "hash-A"

        net_b = SimpleNet()
        optimizer_b = torch.optim.Adam(net_b.parameters())
        program.register_network(
            "evt_net", net_b, optimizer_b, artifact_hash="hash-B",
        )

        meta = program.network_metadata("evt_net")
        assert meta["artifact_hash"] == "hash-B"

    def test_register_network_without_new_kwargs_is_byte_compatible(self):
        """Registering WITHOUT arity/arg_sorts/artifact_hash still succeeds
        exactly as before, and network_metadata reports None for all three
        registration-metadata fields -- the legacy call path is untouched."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        program.register_network("evt_net", net, optimizer)

        assert "evt_net" in program.network_names()

        meta = program.network_metadata("evt_net")
        assert meta["arity"] is None
        assert meta["arg_sorts"] is None
        assert meta["artifact_hash"] is None

    def test_arity_contradicting_the_declaration_is_refused(self):
        """A caller-supplied arity is validated against the program's own
        nn/4 declaration, not trusted -- a mismatch names the predicate and
        both arities."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError) as excinfo:
            program.register_network("evt_net", net, optimizer, arity=3)

        message = str(excinfo.value)
        assert "event_label" in message
        assert "declaration" in message
        assert "2" in message  # the declared arity
        assert "3" in message  # the passed arity

    def test_arg_sorts_without_arity_is_refused(self):
        """arg_sorts names the arguments, so it requires an arity to name
        them against."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError, match="arg_sorts"):
            program.register_network(
                "evt_net", net, optimizer, arg_sorts=(0, 1)
            )

    def test_arg_sorts_length_mismatched_with_arity_is_refused(self):
        """len(arg_sorts) != arity is refused, naming both lengths."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError) as excinfo:
            program.register_network(
                "evt_net", net, optimizer, arity=2, arg_sorts=(0,)
            )

        message = str(excinfo.value)
        assert "1" in message  # len(arg_sorts)
        assert "2" in message  # arity

    def test_arg_sorts_bool_element_is_refused(self):
        """bool is a subclass of int in Python (isinstance(True, int) holds),
        but a bool is never a valid catalog sort id, so it is refused
        explicitly -- pyo3 would otherwise extract it into i64 silently."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError) as excinfo:
            program.register_network(
                "evt_net", net, optimizer, arity=2, arg_sorts=(0, True)
            )

        message = str(excinfo.value)
        assert "bool" in message
        assert "1" in message  # the offending index

    def test_arity_bool_is_refused_regardless_of_declared_arity(self):
        """arity is a plain int; bool is a subclass of int in Python
        (isinstance(True, int) holds), so arity=True is refused explicitly,
        mirroring the arg_sorts bool trap. The refusal precedes the
        declared-arity comparison, so it fires regardless of what this
        fixture's declared arity actually is (event_label/2, declared arity
        2 here) -- arity=True is never treated as arity=1."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError, match="bool"):
            program.register_network("evt_net", net, optimizer, arity=True)

    def test_arg_sorts_string_element_is_refused(self):
        """arg_sorts are catalog sort ids (ints), not sort names -- a string
        element is refused."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(ValueError) as excinfo:
            program.register_network(
                "evt_net", net, optimizer, arity=2, arg_sorts=(0, "label")
            )

        message = str(excinfo.value)
        assert "int" in message
        assert "1" in message  # the offending index

    def test_registration_metadata_kwargs_are_keyword_only(self):
        """arity/arg_sorts/artifact_hash must be passed as keywords, matching
        the consumer's registration signature `(..., cache_size, *, arity,
        arg_sorts, artifact_hash)` -- passing them positionally raises
        TypeError."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        net = SimpleNet()
        optimizer = torch.optim.Adam(net.parameters())

        with pytest.raises(TypeError):
            program.register_network(
                "evt_net", net, optimizer, None, True, None, False, True, 10000, 2,
            )

    def test_network_metadata_undeclared_name_is_refused(self):
        """The same 'not declared' wording as register_network's existing
        undeclared-name refusal."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        with pytest.raises(ValueError, match="not declared"):
            program.network_metadata("undeclared_net")

    def test_network_metadata_declared_but_unregistered_is_refused(self):
        """A declared classification network that has not been registered
        yet must be told to call register_network() first."""
        program = pyxlog.Program.compile("""
            nn(evt_net, [X], Y, [0,1]) :: event_label(X, Y).
        """)

        with pytest.raises(ValueError, match="register_network"):
            program.network_metadata("evt_net")

    def test_network_metadata_on_embedding_declared_network_is_refused(self):
        """network_metadata covers classification networks only -- an
        embedding-declared name is refused with the honest boundary,
        naming register_embedding() as the actual registration path."""
        program = pyxlog.Program.compile("""
            nn(encoder, [X], Embedding) :: encode(X, Embedding).
        """)

        embedding = torch.nn.Embedding(100, 128)
        program.register_embedding("encoder", embedding, trainable=True)

        with pytest.raises(ValueError, match="register_embedding"):
            program.network_metadata("encoder")
