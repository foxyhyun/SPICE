from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


def gaussian_smooth_masked(
    frames: np.ndarray,
    valid: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Spatial Gaussian smoothing that respects invalid bins.
    - frames: (T,H,W)
    - valid : (H,W) bool

    Invalid bins are excluded using weighted smoothing:
      smooth(X*W) / smooth(W)
    """
    frames = np.asarray(frames, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)

    if frames.ndim != 3:
        raise ValueError("frames must be (T,H,W)")
    if v.ndim != 2:
        raise ValueError("valid must be (H,W)")
    if frames.shape[1:] != v.shape:
        raise ValueError(f"shape mismatch: frames {frames.shape} vs valid {v.shape}")

    out = frames.copy()
    out[:, ~v] = np.nan

    if sigma <= 0:
        return out

    x = frames.copy()
    x[~np.isfinite(x)] = 0.0
    x[:, ~v] = 0.0

    w = v.astype(np.float32)

    num = gaussian_filter(
        x,
        sigma=(0.0, float(sigma), float(sigma)),
        mode="nearest",
    )
    den = gaussian_filter(
        w,
        sigma=float(sigma),
        mode="nearest",
    )

    out = num / np.maximum(den[None, :, :], 1e-6)
    out[:, ~v] = np.nan
    return out.astype(np.float32)


def _shift_with_fill(a: np.ndarray, dy: int, dx: int, fill: float = 0.0) -> np.ndarray:
    """
    Shift array with constant fill.
    a: (T,H,W) or (H,W)
    """
    if a.ndim == 2:
        a2 = a[None, ...]
        squeeze = True
    elif a.ndim == 3:
        a2 = a
        squeeze = False
    else:
        raise ValueError("a must be 2D or 3D")

    t, h, w = a2.shape
    out = np.full_like(a2, fill, dtype=a2.dtype)

    y0_src = max(0, -dy)
    y1_src = min(h, h - dy)
    x0_src = max(0, -dx)
    x1_src = min(w, w - dx)

    y0_dst = max(0, dy)
    y1_dst = min(h, h + dy)
    x0_dst = max(0, dx)
    x1_dst = min(w, w + dx)

    out[:, y0_dst:y1_dst, x0_dst:x1_dst] = a2[:, y0_src:y1_src, x0_src:x1_src]
    return out[0] if squeeze else out


def masked_laplacian_4n(frames: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    4-neighbor Laplacian with neighbor-count correction.
    This is safer than filling invalid bins with zero.

    ΔX ≈ (sum of valid neighbors) - (n_valid_neighbors * X_center)
    """
    frames = np.asarray(frames, dtype=np.float32)
    v = np.asarray(valid, dtype=bool)

    if frames.ndim != 3:
        raise ValueError("frames must be (T,H,W)")
    if v.ndim != 2:
        raise ValueError("valid must be (H,W)")
    if frames.shape[1:] != v.shape:
        raise ValueError(f"shape mismatch: frames {frames.shape} vs valid {v.shape}")

    x = frames.copy()
    x[~np.isfinite(x)] = 0.0
    x[:, ~v] = 0.0

    v_f = v.astype(np.float32)

    v_up = _shift_with_fill(v_f, -1, 0, fill=0.0)
    v_dn = _shift_with_fill(v_f, 1, 0, fill=0.0)
    v_lt = _shift_with_fill(v_f, 0, -1, fill=0.0)
    v_rt = _shift_with_fill(v_f, 0, 1, fill=0.0)

    n = v_up + v_dn + v_lt + v_rt  # (H,W)

    up = _shift_with_fill(x, -1, 0, fill=0.0)
    dn = _shift_with_fill(x, 1, 0, fill=0.0)
    lt = _shift_with_fill(x, 0, -1, fill=0.0)
    rt = _shift_with_fill(x, 0, 1, fill=0.0)

    neigh_sum = up * v_up + dn * v_dn + lt * v_lt + rt * v_rt
    lap = neigh_sum - (n[None, :, :] * x)

    lap[:, ~v] = np.nan
    return lap.astype(np.float32)


def compute_pseudo_csd(
    dff: np.ndarray,
    valid: np.ndarray,
    *,
    sigma: float = 1.0,
    dx: float = 1.0,
    smooth_sigma: float = 1.0,
) -> np.ndarray:
    """
    Pseudo-CSD(t) = -sigma * ∇²(dff)(t) / dx^2

    Reflected from the reference code:
      - optional Gaussian smoothing before Laplacian
      - scaling by dx^2
    """
    if dx <= 0:
        raise ValueError("dx must be > 0")

    smoothed = gaussian_smooth_masked(dff, valid, sigma=float(smooth_sigma))
    lap = masked_laplacian_4n(smoothed, valid)
    csd = -(float(sigma) / (float(dx) ** 2)) * lap
    csd[:, ~valid.astype(bool)] = np.nan
    return csd.astype(np.float32)


def compute_activity_map(
    csd: np.ndarray,
    valid: np.ndarray,
    *,
    mode: str = "mean_abs",
) -> np.ndarray:
    """
    Aggregate CSD over time to a single map.
    """
    v = valid.astype(bool)

    if mode == "mean_abs":
        m = np.nanmean(np.abs(csd), axis=0)
    elif mode == "rms":
        m = np.sqrt(np.nanmean(csd * csd, axis=0))
    elif mode == "std":
        m = np.nanstd(csd, axis=0)
    else:
        raise ValueError("mode must be one of: mean_abs, rms, std")

    m = m.astype(np.float32)
    m[~v] = np.nan
    return m