import pytest
import torch

pyxlog = pytest.importorskip("pyxlog")


def test_direct_query_uses_correct_network():
    program = pyxlog.Program.compile(
        """
        nn(net1, [X], Y, [a, b]) :: pred1(X, Y).
        nn(net2, [X], Y, [0,1,2]) :: pred2(X, Y).
        """
    )

    class NetA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 2)

        def forward(self, x):
            return torch.softmax(self.fc(x), dim=-1)

    class NetB(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(1, 3)

        def forward(self, x):
            return torch.softmax(self.fc(x), dim=-1)

    net_a = NetA()
    net_b = NetB()
    program.register_network("net1", net_a, torch.optim.SGD(net_a.parameters(), lr=0.1))
    program.register_network("net2", net_b, torch.optim.SGD(net_b.parameters(), lr=0.1))

    program.add_tensor_source("data", torch.randn(4, 1))

    loss1 = program.forward_backward("pred1(0, a)")
    loss2 = program.forward_backward("pred2(0, 2)")
    assert loss1 > 0
    assert loss2 > 0
