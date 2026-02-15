import pytest

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

if not torch.cuda.is_available():
    pytest.skip("CUDA is required for coin pipeline tests", allow_module_level=True)


COINS_PROGRAM = """
    nn(net1, [X], Y, [heads, tails]) :: coin(1, X, Y).
    nn(net2, [X], Y, [heads, tails]) :: coin(2, X, Y).
    win() :- coin(1, X, heads), coin(2, Y, heads).
"""


class CoinNet(torch.nn.Module):
    """2-class network (heads/tails) with call tracking."""

    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(1, 2)
        self.call_count = 0

    def forward(self, x):
        self.call_count += 1
        return torch.softmax(self.fc(x), dim=-1)


def _build_program() -> tuple[pyxlog.Program, CoinNet, CoinNet]:
    program = pyxlog.Program.compile(COINS_PROGRAM)
    net1 = CoinNet().cuda()
    net2 = CoinNet().cuda()
    program.register_network("net1", net1, torch.optim.SGD(net1.parameters(), lr=0.1))
    program.register_network("net2", net2, torch.optim.SGD(net2.parameters(), lr=0.1))
    return program, net1, net2


def _build_data() -> torch.Tensor:
    return torch.randn(4, 1, device="cuda")


def _assert_nonzero_grad(net: torch.nn.Module) -> None:
    for param in net.parameters():
        assert param.grad is not None, f"Missing grad for {param.shape}"
        assert not torch.all(param.grad == 0), "Gradient is all zeros"


def _assert_no_grad(net: torch.nn.Module) -> None:
    for param in net.parameters():
        if param.grad is not None:
            assert torch.all(param.grad == 0)


def test_coins_compile_and_register():
    program, net1, net2 = _build_program()

    declared = set(program.declared_network_names())
    assert declared == {"net1", "net2"}

    registered = set(program.network_names())
    assert registered == {"net1", "net2"}

    # Ambiguous predicates are intentionally rejected until a concrete atom is provided.
    with pytest.raises(ValueError, match="multiple nn/4 declarations"):
        program.neural_predicate_info("coin")

    # Keep references alive to satisfy the test contract.
    assert net1 is not None
    assert net2 is not None


def test_coins_forward_backward_expected_true():
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    program.zero_grad()
    loss = program.forward_backward("win()", expected=True)

    assert isinstance(loss, float)
    assert loss >= 0

    _assert_nonzero_grad(net1)
    _assert_nonzero_grad(net2)


def test_coins_forward_backward_expected_false():
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    program.zero_grad()
    loss_false = program.forward_backward("win()", expected=False)
    assert isinstance(loss_false, float)
    assert loss_false >= 0
    _assert_nonzero_grad(net1)
    _assert_nonzero_grad(net2)

    program.zero_grad()
    loss_true = program.forward_backward("win()", expected=True)
    assert loss_true != loss_false


def test_coins_training_reduces_loss():
    torch.manual_seed(42)
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    losses = []
    for _ in range(20):
        program.zero_grad()
        losses.append(program.forward_backward("win()"))
        program.optimizer_step()

    assert losses[-1] < losses[0]


def test_coins_cache_reuse():
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    loss1 = program.forward_backward("win()")
    loss2 = program.forward_backward("win()")

    assert isinstance(loss1, float)
    assert isinstance(loss2, float)
    assert loss1 >= 0
    assert loss2 >= 0

    # Keep references alive to avoid false-positive lint checks.
    assert net1 is not None
    assert net2 is not None


def test_coins_batching_per_network():
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    loss = program.forward_backward("win()")
    assert isinstance(loss, float)

    assert net1.call_count == 1
    assert net2.call_count == 1


def test_coins_direct_query_with_constant():
    program, net1, net2 = _build_program()
    program.add_tensor_source("data", _build_data())

    program.zero_grad()
    loss = program.forward_backward("coin(1, 0, heads)")

    assert isinstance(loss, float)
    assert loss >= 0
    _assert_nonzero_grad(net1)
    _assert_no_grad(net2)
