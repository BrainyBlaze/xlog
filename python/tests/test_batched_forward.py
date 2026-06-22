import pytest
import torch

pyxlog = pytest.importorskip("pyxlog")


class CountingNet(torch.nn.Module):
    def __init__(self, output_dimension):
        super().__init__()
        self.output_dimension = output_dimension
        self.calls = 0
        self.linear = torch.nn.Linear(1, output_dimension)

    def forward(self, x):
        self.calls += 1
        return torch.softmax(self.linear(x), dim=-1)


def test_complex_query_batches_forward_calls():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required for neural fast-path")

    device = torch.device("cuda")
    program = pyxlog.Program.compile(
        """
        nn(net, [X], Y, [0,1,2,3]) :: digit(X, Y).
        addition(FirstImage, SecondImage, Sum) :-
            digit(FirstImage, FirstDigitValue),
            digit(SecondImage, SecondDigitValue),
            Sum is FirstDigitValue + SecondDigitValue.
        """
    )

    net = CountingNet(4)
    net.to(device)
    learning_rate = 0.1
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)
    program.register_network("net", net, optimizer, batching=True)

    program.add_tensor_source("data", torch.randn(10, 1, device=device))
    program.forward_backward("addition(0, 1, 2)")

    # Two neural calls, but should be batched -> one forward
    assert net.calls == 1
