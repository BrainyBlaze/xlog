import pytest

from pyxlog.demos.plasticity.generator import make_fixed_split, make_held_out_split
from pyxlog.demos.plasticity.leakage import LeakageError, assert_no_leakage


def test_clean_split_passes() -> None:
    train = make_fixed_split("e_tr")
    held = make_held_out_split("e_ho")
    assert_no_leakage(train, held)  # must not raise


def test_overlapping_entities_are_rejected() -> None:
    train = make_fixed_split("shared")
    held = make_held_out_split("shared")  # same prefix -> overlapping entity ids
    with pytest.raises(LeakageError, match="(?i)overlap"):
        assert_no_leakage(train, held)
