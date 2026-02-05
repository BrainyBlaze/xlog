import pytest
import torch

pyxlog = pytest.importorskip("pyxlog")


class CountingNet(torch.nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.calls = 0
        self.fc = torch.nn.Linear(1, out_dim)

    def forward(self, x):
        self.calls += 1
        return torch.softmax(self.fc(x), dim=-1)


def test_complex_query_batches_forward_calls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for neural fast-path")

    device = torch.device("cuda")
    program = pyxlog.Program.compile(
        """
        nn(net, [X], Y, [0,1,2,3]) :: digit(X, Y).
        addition(X, Y, Z) :- digit(X, D1), digit(Y, D2), Z is D1 + D2.
        """
    )

    net = CountingNet(4)
    net.to(device)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    program.register_network("net", net, opt, batching=True)

    program.add_tensor_source("data", torch.randn(10, 1, device=device))
    program.forward_backward("addition(0, 1, 2)")

    # Two neural calls, but should be batched -> one forward
    assert net.calls == 1
