"""Step 4b — Normalize each packet, then resample onto a uniform time grid.

Why normalize: the receiver's AGC (automatic gain control) changes its amplifier gain
from time to time. That multiplies ALL subcarriers of a packet by the same factor, which
looks like a sudden jump. Dividing each packet by its own mean amplitude removes it.

Why resample: in router/ping mode packets do not arrive exactly every 10 ms (jitter,
lost packets). Filters and the FFT assume uniform sampling, so we interpolate every
subcarrier onto a regular grid at fs = 100 Hz.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def normalize_packets(amp: np.ndarray) -> np.ndarray:
    """Divide each packet (row) by its mean amplitude across subcarriers."""
    scale = amp.mean(axis=1, keepdims=True)
    return amp / np.where(scale > 0, scale, 1.0)


@dataclass
class TimingStats:
    """Checkpoint 1: how regular is our 'sampling'?"""
    n_packets: int
    duration_s: float
    mean_rate_hz: float
    interval_mean_ms: float
    interval_std_ms: float       # jitter
    interval_p99_ms: float
    longest_gap_ms: float


def timing_stats(t: np.ndarray) -> TimingStats:
    dt = np.diff(t) * 1e3
    return TimingStats(
        n_packets=len(t),
        duration_s=float(t[-1] - t[0]),
        mean_rate_hz=float((len(t) - 1) / (t[-1] - t[0])),
        interval_mean_ms=float(dt.mean()),
        interval_std_ms=float(dt.std()),
        interval_p99_ms=float(np.percentile(dt, 99)),
        longest_gap_ms=float(dt.max()),
    )


def resample_uniform(t: np.ndarray, x: np.ndarray, fs: float, t0: float | None = None,
                     max_gap_s: float = 0.05):
    """Linear interpolation of each column of x (T x K) onto a grid t0, t0+1/fs, ...

    Returns (grid, x_uniform, gap_mask). gap_mask marks grid points that fall inside a
    hole of more than max_gap_s with no packets — interpolated values there are made up
    (a straight line between two noisy packets), so windows containing them are not
    trusted later on.
    """
    t0 = t[0] if t0 is None else t0
    grid = np.arange(t0, t[-1] + 1e-9, 1.0 / fs)
    xu = np.empty((grid.size, x.shape[1]))
    for k in range(x.shape[1]):
        xu[:, k] = np.interp(grid, t, x[:, k])
    i = np.clip(np.searchsorted(t, grid), 1, len(t) - 1)
    gap_mask = (t[i] - t[i - 1]) > max_gap_s
    return grid, xu, gap_mask


class StreamingResampler:
    """Live version: feed packets as they arrive, get uniform samples out.

    A grid point is emitted once a packet *after* it has arrived, so the output lags
    the input by at most one packet interval. Each sample comes with an `in_gap` flag
    (same meaning as resample_uniform's gap_mask)."""

    def __init__(self, fs: float, max_gap_s: float = 0.05):
        self.dt = 1.0 / fs
        self.max_gap_s = max_gap_s
        self.next_t: float | None = None
        self.prev: tuple[float, np.ndarray] | None = None

    def push(self, t: float, x: np.ndarray) -> list[tuple[float, np.ndarray, bool]]:
        out = []
        if self.prev is None:
            self.prev, self.next_t = (t, x), t
            return out
        t_prev, x_prev = self.prev
        if t <= t_prev:
            return out  # duplicate / out-of-order packet
        in_gap = (t - t_prev) > self.max_gap_s
        while self.next_t <= t:
            w = (self.next_t - t_prev) / (t - t_prev)
            out.append((self.next_t, (1 - w) * x_prev + w * x, in_gap))
            self.next_t += self.dt
        self.prev = (t, x)
        return out
