"""Step 8 — Measure: moving variance of the principal components = "activity level".

Every `hop` seconds we look at the last `window` seconds (default 1 s = 100 samples):
1. Fit PCA on a longer trailing history (`pca_window`, default 10 s) — only past data,
   so exactly the same computation works live and offline.
2. Project the last 1 s onto the kept components.
3. Activity = sum of the variances of those projections.
An empty room gives a small, steady activity (noise only); motion makes it jump up.
"""
from __future__ import annotations

import numpy as np

from .combine import remove_background, select_components


def window_activity(history: np.ndarray, window_n: int, n_keep: int = 3,
                    drop_first: bool = False) -> float:
    """history: trailing samples (N x K, N >= window_n). Uses its last window_n rows."""
    xc = remove_background(history)
    _, _, vt = np.linalg.svd(xc, full_matrices=False)
    comps = vt[select_components(n_keep, drop_first)]
    proj = history[-window_n:] @ comps.T          # window_n x n_keep
    return float(proj.var(axis=0, ddof=1).sum())


def activity_series(y: np.ndarray, fs: float, window_s: float = 1.0, hop_s: float = 0.5,
                    pca_window_s: float = 10.0, n_keep: int = 3, drop_first: bool = False,
                    gap_mask: np.ndarray | None = None):
    """Offline: activity at every hop. Returns (t_end_s, activity, valid).
    The first output waits for a full PCA history (warm-up), like the live version.
    valid[i] is False if window i contains samples interpolated across a packet gap."""
    win_n, hop_n, pca_n = int(round(window_s * fs)), int(round(hop_s * fs)), int(round(pca_window_s * fs))
    ends = np.arange(max(win_n, pca_n), y.shape[0] + 1, hop_n)
    act = np.array([window_activity(y[max(0, e - pca_n):e], win_n, n_keep, drop_first) for e in ends])
    if gap_mask is None:
        valid = np.ones(ends.size, bool)
    else:
        valid = np.array([not gap_mask[e - win_n:e].any() for e in ends])
    return ends / fs, act, valid
