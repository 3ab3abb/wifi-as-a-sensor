"""Step 6 — Smooth: Butterworth low-pass filter (IIR = infinite impulse response).

A Butterworth filter has the flattest possible pass-band ("maximally flat"): no ripple,
then a roll-off of 20*N dB per decade above the cutoff fc (N = order). We design it with
scipy and store it as second-order sections (SOS): a cascade of small 2nd-order filters,
which is numerically safer than one big polynomial for higher orders.

Two ways to apply it:
* zero-phase (offline): `sosfiltfilt` runs the filter forward, then backward. The phase
  shifts cancel, so features are not delayed — but it needs the whole signal (future
  samples) and the magnitude response is applied twice: |H|^2, i.e. -6 dB at fc instead
  of -3 dB, with effectively double the order.
* causal (live): `sosfilt` only uses past samples. Works in real time, but delays the
  signal (frequency-dependent group delay).
"""
from __future__ import annotations

import numpy as np
from scipy import signal


def butter_lowpass(order: int, cutoff_hz: float, fs: float) -> np.ndarray:
    return signal.butter(order, cutoff_hz, btype="low", fs=fs, output="sos")


def butter_bandpass(order: int, low_hz: float, high_hz: float, fs: float) -> np.ndarray:
    return signal.butter(order, [low_hz, high_hz], btype="band", fs=fs, output="sos")


def apply_zero_phase(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    return signal.sosfiltfilt(sos, x, axis=0)


def apply_causal(sos: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Causal filtering, started in steady state at the first sample (avoids a big
    start-up transient from assuming the signal was 0 before t = 0)."""
    return StreamingFilter(sos).push(x)


class StreamingFilter:
    """Causal filter that keeps its internal state between chunks (live mode)."""

    def __init__(self, sos: np.ndarray):
        self.sos = sos
        self.zi: np.ndarray | None = None

    def push(self, x_new: np.ndarray) -> np.ndarray:
        if x_new.shape[0] == 0:
            return x_new
        if self.zi is None:
            self.zi = signal.sosfilt_zi(self.sos)[:, :, None] * x_new[0][None, None, :]
        y, self.zi = signal.sosfilt(self.sos, x_new, axis=0, zi=self.zi)
        return y


def frequency_response(sos: np.ndarray, fs: float, n: int = 2048):
    """(f_hz, |H| in dB, group delay in samples) — for the checkpoint plot."""
    f, h = signal.sosfreqz(sos, worN=n, fs=fs)
    mag_db = 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    phase = np.unwrap(np.angle(h))
    gd = -np.gradient(phase, 2 * np.pi * f / fs)
    return f, mag_db, gd
