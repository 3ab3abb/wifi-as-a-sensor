"""Step 5 — Clean: Hampel filter (a non-linear, median-based outlier remover).

For each sample, look at a window of W samples centred on it:
    med   = median of the window
    sigma = 1.4826 * MAD     (MAD = median absolute deviation; 1.4826 makes it equal to
                              the standard deviation if the noise were Gaussian)
If |x - med| > n_sigmas * sigma, the sample is an outlier and is replaced by med.

Unlike a linear (e.g. Butterworth) filter, a single huge spike does not get smeared
over its neighbours — it is simply removed, and normal samples pass through unchanged.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

MAD_TO_SIGMA = 1.4826


def _hampel_valid(x: np.ndarray, window: int, n_sigmas: float):
    """Core: x is (T + window - 1) x K; returns cleaned centres (T x K) and outlier mask."""
    half = window // 2
    win = sliding_window_view(x, window, axis=0)          # T x K x W
    med = np.median(win, axis=-1)
    sigma = MAD_TO_SIGMA * np.median(np.abs(win - med[..., None]), axis=-1)
    centre = x[half:half + med.shape[0]]
    outlier = np.abs(centre - med) > n_sigmas * sigma
    return np.where(outlier, med, centre), outlier


def hampel(x: np.ndarray, window: int = 11, n_sigmas: float = 3.0):
    """Offline Hampel filter along time (axis 0) of a T x K array (or 1-D array).
    Returns (cleaned, outlier_mask). Edges are handled by reflecting the signal."""
    if window % 2 == 0:
        raise ValueError("window must be odd")
    one_d = x.ndim == 1
    x2 = x[:, None] if one_d else x
    half = window // 2
    padded = np.pad(x2, ((half, half), (0, 0)), mode="reflect")
    y, mask = _hampel_valid(padded, window, n_sigmas)
    return (y[:, 0], mask[:, 0]) if one_d else (y, mask)


class StreamingHampel:
    """Live version. A centred window needs `half` future samples, so output is delayed
    by window//2 samples (50 ms at 100 Hz for W = 11) — small, and it keeps the result
    identical to the offline filter (except at the very start)."""

    def __init__(self, window: int = 11, n_sigmas: float = 3.0):
        self.window, self.n_sigmas = window, n_sigmas
        self.history: np.ndarray | None = None

    def push(self, x_new: np.ndarray) -> np.ndarray:
        """x_new: n x K new samples. Returns cleaned samples (possibly 0 rows at start)."""
        buf = x_new if self.history is None else np.vstack([self.history, x_new])
        self.history = buf[-(self.window - 1):]
        if buf.shape[0] < self.window:
            return np.empty((0, x_new.shape[1]))
        y, _ = _hampel_valid(buf, self.window, self.n_sigmas)
        return y
