import pytest
import os
import sys

torch = pytest.importorskip("torch")
pyxlog = pytest.importorskip("pyxlog")

# Add examples path for imports (matches other integration tests).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../examples/neural/01_minimal"))


def test_forward_backward_addition_returns_cuda_tensor_loss(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    # Guardrail: forward_backward must not use CPU extraction helpers internally.
    monkeypatch.setattr(torch.Tensor, "tolist", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("tolist forbidden")))
    monkeypatch.setattr(torch.Tensor, "item", lambda self, *a, **k: (_ for _ in ()).throw(RuntimeError("item forbidden")))

    from train import MNISTNet, create_program

    program = create_program()
    net = MNISTNet().cuda()
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)
    program.register_network("mnist_net", net, optimizer)

    torch.manual_seed(0)
    images = torch.randn(4, 1, 28, 28, device="cuda")
    program.add_tensor_source("train", images)

    program.zero_grad()
    loss = program.forward_backward_tensor("addition(0, 1, 7)")

    assert isinstance(loss, torch.Tensor)
    assert loss.is_cuda
    assert loss.numel() == 1
