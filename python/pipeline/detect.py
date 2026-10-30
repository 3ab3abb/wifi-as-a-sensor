"""Step 9 — Decide: threshold + hysteresis + presence hold.

1. Calibrate on an EMPTY room: threshold = mean + 3*std of its activity (or a percentile).
   With Gaussian noise, mean + 3 sigma would be exceeded only ~0.1% of the time, so this
   sets the false-alarm rate. Higher threshold -> fewer false alarms, more misses.
2. Hysteresis: don't flip on a single window. Switch to MOTION only after `n_on`
   consecutive windows above the threshold, and back only after `n_off` below.
3. Presence: "someone is here" if there was motion in the last `hold_s` seconds
   (a person sitting still between movements is still present).
4. Windows marked invalid (they contain a packet gap) don't change the state.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def calibrate_threshold(empty_activity: np.ndarray, method: str = "sigma",
                        k: float = 3.0, percentile: float = 99.0) -> float:
    a = np.asarray(empty_activity)
    if method == "sigma":
        return float(a.mean() + k * a.std())
    if method == "percentile":
        return float(np.percentile(a, percentile))
    raise ValueError(f"unknown method {method!r}")


class Detector:
    """Stateful decision logic; feed one activity value per hop (works live and offline)."""

    def __init__(self, threshold: float, hop_s: float, n_on: int = 2, n_off: int = 4,
                 hold_s: float = 30.0):
        self.threshold, self.hop_s = threshold, hop_s
        self.n_on, self.n_off, self.hold_s = n_on, n_off, hold_s
        self.motion = False
        self._run = 0                       # consecutive windows disagreeing with state
        self._since_motion = np.inf         # seconds since last MOTION window

    def update(self, activity: float, valid: bool = True) -> tuple[bool, bool]:
        """Returns (motion, present)."""
        above = activity > self.threshold
        if not valid:
            pass                            # untrusted window: keep the current state
        elif above != self.motion:
            self._run += 1
            if self._run >= (self.n_on if above else self.n_off):
                self.motion, self._run = above, 0
        else:
            self._run = 0
        self._since_motion = 0.0 if self.motion else self._since_motion + self.hop_s
        return self.motion, self._since_motion <= self.hold_s


@dataclass
class Decisions:
    raw: np.ndarray          # activity > threshold (no hysteresis)
    motion: np.ndarray       # after hysteresis
    present: np.ndarray      # motion within the last hold_s seconds


def decide(activity: np.ndarray, threshold: float, hop_s: float, n_on: int = 2,
           n_off: int = 4, hold_s: float = 30.0, valid: np.ndarray | None = None) -> Decisions:
    det = Detector(threshold, hop_s, n_on, n_off, hold_s)
    valid = np.ones(activity.size, bool) if valid is None else valid
    out = np.array([det.update(a, v) for a, v in zip(activity, valid)], dtype=bool).reshape(-1, 2)
    return Decisions(raw=activity > threshold, motion=out[:, 0], present=out[:, 1])
