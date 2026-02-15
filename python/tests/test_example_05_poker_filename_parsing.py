import importlib.util
from pathlib import Path
import pytest


def _load_train_module():
    path = Path("examples/neural/05_poker/train.py").resolve()
    if not path.exists():
        pytest.skip(f"Missing example file: {path}", allow_module_level=True)
    spec = importlib.util.spec_from_file_location("example_05_poker_train", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_card_atoms_accepts_plain_and_suffixed_names():
    module = _load_train_module()

    assert module.parse_card_atoms("AS") == ("ra", "s")
    assert module.parse_card_atoms("10H_0001") == ("r10", "h")
    assert module.parse_card_atoms("qc-extra") == ("rq", "c")
    assert module.parse_card_atoms("badname") is None


def test_pick_epoch_indices_caps_and_is_deterministic():
    module = _load_train_module()

    first = module.pick_epoch_indices(1000, 256, seed=7, epoch=3)
    second = module.pick_epoch_indices(1000, 256, seed=7, epoch=3)
    assert first == second
    assert len(first) == 256

    no_cap = module.pick_epoch_indices(128, 512, seed=1, epoch=0)
    assert len(no_cap) == 128


def test_build_training_queries_can_upweight_rank_signal():
    module = _load_train_module()
    labels = [("ra", "s"), ("rk", "h")]
    queries = module.build_training_queries(labels, [0, 1], rank_weight=3)

    assert queries.count("rank(0, ra)") == 3
    assert queries.count("rank(1, rk)") == 3
    assert queries.count("suit(0, s)") == 1
    assert queries.count("suit(1, h)") == 1
