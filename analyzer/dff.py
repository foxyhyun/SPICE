from __future__ import annotations
import numpy as np

def compute_dff(
    mat: np.ndarray,            # (T,64,64) float32, NaN allowed
    valid_grid: np.ndarray,     # (64,64) bool
    f0_percentile: float = 20.0,
    eps: float = 1e-6,
):
    """
    returns:
      dff: (T,64,64) float32 (invalid -> NaN)
      f0 : (64,64) float32 (invalid -> NaN)
    """
    if mat.ndim != 3:
        raise ValueError("mat must be (T,G,G)")
    if valid_grid.shape != mat.shape[1:]:
        raise ValueError("valid_grid shape mismatch")

    T, G1, G2 = mat.shape
    dff = np.full_like(mat, np.nan, dtype=np.float32)
    f0_map = np.full((G1, G2), np.nan, dtype=np.float32)

    ys, xs = np.where(valid_grid)
    for y, x in zip(ys, xs):
        trace = mat[:, y, x].astype(np.float32)
        f0 = np.nanpercentile(trace, f0_percentile)
        f0_map[y, x] = f0
        dff[:, y, x] = (trace - f0) / (f0 + eps)

    return dff, f0_map
