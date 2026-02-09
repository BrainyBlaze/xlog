import importlib.util
from pathlib import Path

import torch


def _load_model_module():
    path = Path("examples/neural/05_poker/model.py").resolve()
    spec = importlib.util.spec_from_file_location("example_05_poker_model", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_card_net_supports_focus_mode_and_output_shape():
    module = _load_model_module()
    net = module.CardNet(13, pretrained=False, focus="rank")
    x = torch.rand(2, 3, 64, 64)
    y = net(x)
    assert y.shape == (2, 13)
