"""Step 7 — Combine: background removal + PCA (principal component analysis).

Background removal: when nothing moves, each subcarrier sits at its own constant level
(the static room). Subtracting each subcarrier's mean (= removing its DC component)
leaves only the *changes* caused by motion (and noise).

PCA: a moving person changes all 52 subcarriers in a correlated way, while noise is
mostly independent per subcarrier. PCA finds the directions (weighted combinations of
subcarriers) along which the data varies most. Keeping the top few components gathers
the motion "signal" from all subcarriers and leaves most of the noise behind.
We compute PCA ourselves with the SVD (singular value decomposition):
    Xc = U S V^T   ->   components = rows of V^T, scores = U S, variance = S^2/(T-1).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def remove_background(x: np.ndarray) -> np.ndarray:
    return x - x.mean(axis=0, keepdims=True)


@dataclass
class PCAResult:
    components: np.ndarray        # n_components x K (unit vectors over subcarriers)
    explained_variance: np.ndarray
    explained_ratio: np.ndarray
    scores: np.ndarray            # T x n_components (the combined time signals)


def pca(x: np.ndarray, n_components: int | None = None) -> PCAResult:
    xc = remove_background(x)
    u, s, vt = np.linalg.svd(xc, full_matrices=False)
    var = s**2 / max(x.shape[0] - 1, 1)
    n = len(s) if n_components is None else n_components
    return PCAResult(
        components=vt[:n],
        explained_variance=var[:n],
        explained_ratio=var[:n] / var.sum(),
        scores=(u[:, :n] * s[:n]),
    )


def select_components(n_keep: int, drop_first: bool) -> slice:
    """Which principal components feed the activity measure. The 1st one sometimes is
    slow hardware drift rather than motion — `drop_first` lets us test that."""
    return slice(1, n_keep + 1) if drop_first else slice(0, n_keep)
