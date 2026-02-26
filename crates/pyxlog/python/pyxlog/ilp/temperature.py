"""Adaptive temperature controller for dILP training.

Three-mode FSM: COOLING → PLATEAU → WARMUP.
See docs/plans/2026-02-26-dilp-hardening-design.md §2.1.
"""
from __future__ import annotations

from collections import deque
from enum import Enum, auto


class TempMode(Enum):
    COOLING = auto()
    PLATEAU = auto()
    WARMUP = auto()


class AdaptiveTempController:
    """Controls the Gumbel-Softmax temperature τ during training.

    Parameters
    ----------
    tau_start : float
        Initial temperature.
    tau_floor : float
        Minimum temperature (never go below this).
    plateau_window : int
        Number of steps over which EMA loss must be flat to trigger PLATEAU.
    plateau_threshold : float
        Maximum absolute EMA-loss change to count as "flat".
    warmup_increment : float
        Amount to increase τ when a trap is detected.
    trap_disc_threshold : float
        Discreteness above this triggers trap detection.
    trap_progress_window : int
        Number of steps without witness-coverage progress to confirm a trap.
    total_steps : int
        Total step budget for this attempt (used for cooling rate).
    """

    def __init__(
        self,
        tau_start: float,
        tau_floor: float,
        plateau_window: int,
        plateau_threshold: float,
        warmup_increment: float,
        trap_disc_threshold: float,
        trap_progress_window: int,
        total_steps: int,
    ) -> None:
        self.tau = tau_start
        self._tau_start = tau_start
        self._tau_floor = tau_floor
        self._plateau_window = plateau_window
        self._plateau_threshold = plateau_threshold
        self._warmup_increment = warmup_increment
        self._trap_disc_threshold = trap_disc_threshold
        self._trap_progress_window = trap_progress_window
        self._total_steps = max(total_steps, 1)

        self.mode = TempMode.COOLING

        # EMA loss tracking for plateau detection
        self._ema_alpha = 0.1
        self._ema_loss: float | None = None
        self._ema_history: deque[float] = deque(maxlen=plateau_window)

        # Witness coverage tracking for trap detection
        self._coverage_history: deque[float] = deque(maxlen=trap_progress_window)

        # Linear cooling: decrease per step
        self._cooling_rate = (tau_start - tau_floor) / self._total_steps

        self._step_count = 0

    def step(self, loss: float, disc: float, witness_coverage: float) -> None:
        """Advance the controller by one training step.

        Parameters
        ----------
        loss : float
            Current task loss value.
        disc : float
            Current discreteness (max soft_prob value, 0..1).
        witness_coverage : float
            Fraction of positive examples with witness derivations (0..1).
        """
        self._step_count += 1

        # Update EMA loss
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
        self._ema_history.append(self._ema_loss)

        # Track witness coverage
        self._coverage_history.append(witness_coverage)

        # Check for trap: high discreteness + no coverage progress
        is_trapped = self._detect_trap(disc)

        if is_trapped:
            self.mode = TempMode.WARMUP
            self.tau = min(self.tau + self._warmup_increment, self._tau_start)
        elif self._detect_plateau():
            self.mode = TempMode.PLATEAU
            # Hold tau constant — no change
        else:
            self.mode = TempMode.COOLING
            self.tau = max(self.tau - self._cooling_rate, self._tau_floor)

    def _detect_plateau(self) -> bool:
        """EMA loss is flat over the plateau window."""
        if len(self._ema_history) < self._plateau_window:
            return False
        min_ema = min(self._ema_history)
        max_ema = max(self._ema_history)
        return (max_ema - min_ema) < self._plateau_threshold

    def _detect_trap(self, disc: float) -> bool:
        """High discreteness with stalled witness coverage."""
        if disc < self._trap_disc_threshold:
            return False
        if len(self._coverage_history) < self._trap_progress_window:
            return False
        # Check if coverage made any progress over the window
        cov_min = min(self._coverage_history)
        cov_max = max(self._coverage_history)
        return (cov_max - cov_min) < 1e-6
