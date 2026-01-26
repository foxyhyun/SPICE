from __future__ import annotations
import numpy as np

def grid_to_64_with_valid(
    frames: np.ndarray,
    mask: np.ndarray,
    grid: int = 64,
    min_coverage: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    frames: (T,H,W) float32
    mask: (H,W) bool

    returns:
      mat: (T,grid,grid) float32  (invalid bin -> NaN)
      valid_grid: (grid,grid) bool
    """
    if frames.ndim != 3:
        raise ValueError("frames must be (T,H,W)")
    if mask.ndim != 2:
        raise ValueError("mask must be (H,W)")
    T, H, W = frames.shape
    if mask.shape != (H, W):
        raise ValueError("mask shape mismatch")

    f = frames.copy()
    f[:, ~mask] = np.nan

    ys = np.linspace(0, H, grid + 1, dtype=int)
    xs = np.linspace(0, W, grid + 1, dtype=int)

    mat = np.full((T, grid, grid), np.nan, dtype=np.float32)
    valid_grid = np.zeros((grid, grid), dtype=bool)

    for gy in range(grid):
        y0, y1 = ys[gy], ys[gy + 1]
        for gx in range(grid):
            x0, x1 = xs[gx], xs[gx + 1]

            # coverage from mask only (time-invariant, stable)
            mblock = mask[y0:y1, x0:x1].reshape(-1)
            coverage = float(mblock.mean()) if mblock.size > 0 else 0.0
            if coverage <= min_coverage:
                continue

            block = f[:, y0:y1, x0:x1].reshape(T, -1)  # (T,N)
            valid = np.isfinite(block)                  # (T,N)

            if not valid.any():
                continue

            count = valid.sum(axis=1).astype(np.float32)  # (T,)
            s = np.where(valid, block, 0.0).sum(axis=1).astype(np.float32)

            v = np.full((T,), np.nan, dtype=np.float32)
            np.divide(s, count, out=v, where=(count > 0))

            mat[:, gy, gx] = v
            valid_grid[gy, gx] = True

    return mat, valid_grid

def grid_to_64(
    frames: np.ndarray,
    mask: np.ndarray,
    grid: int = 64,
    fill_value: float = 0.0,
    min_coverage: float = 0.0,
) -> np.ndarray:
    """
    Backward-compatible wrapper.
    Returns matrix only. Invalid bins -> fill_value (default 0.0).
    """
    mat, _ = grid_to_64_with_valid(frames, mask, grid=grid, min_coverage=min_coverage)
    if fill_value is None:
        return mat
    return np.nan_to_num(mat, nan=float(fill_value)).astype(np.float32)
