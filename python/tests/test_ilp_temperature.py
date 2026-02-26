# python/tests/test_ilp_temperature.py
"""Tests for adaptive temperature controller."""
from pyxlog.ilp.temperature import AdaptiveTempController, TempMode


def test_initial_mode_is_cooling():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    assert controller.mode == TempMode.COOLING
    assert controller.tau == 2.0


def test_cooling_decreases_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    tau0 = controller.tau
    controller.step(loss=5.0, disc=0.3, witness_coverage=0.0)
    assert controller.tau < tau0


def test_tau_never_below_floor():
    controller = AdaptiveTempController(
        tau_start=0.1, tau_floor=0.05, plateau_window=5,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=10,
    )
    for _ in range(20):
        controller.step(loss=0.1, disc=0.3, witness_coverage=1.0)
    assert controller.tau >= 0.05


def test_plateau_holds_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=3,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=10,
        total_steps=100,
    )
    for _ in range(10):
        controller.step(loss=1.0, disc=0.5, witness_coverage=0.5)
    tau_at_plateau = controller.tau
    controller.step(loss=1.0, disc=0.5, witness_coverage=0.5)
    assert controller.mode == TempMode.PLATEAU
    assert controller.tau == tau_at_plateau


def test_trap_warms_up_tau():
    controller = AdaptiveTempController(
        tau_start=2.0, tau_floor=0.05, plateau_window=3,
        plateau_threshold=0.01, warmup_increment=0.3,
        trap_disc_threshold=0.85, trap_progress_window=3,
        total_steps=100,
    )
    for _ in range(5):
        controller.step(loss=0.5, disc=0.95, witness_coverage=0.0)
    assert controller.mode == TempMode.WARMUP
    assert controller.tau > 0.05
