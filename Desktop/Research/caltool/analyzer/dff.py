from __future__ import annotations
import numpy as np



# 모든 T에 대해 (64,64)픽셀들의 delta F구하기
def compute_dff(
    mat: np.ndarray,          
    valid_grid: np.ndarray, 
    f0_percentile: float = 20.0,  
    eps: float = 1e-6,    
) -> tuple[np.ndarray, np.ndarray]:

    if mat.ndim != 3:
        raise ValueError("mat must be (T,G,G)")
    if valid_grid.shape != mat.shape[1:]:
        raise ValueError("valid_grid shape mismatch")

    T, G1, G2 = mat.shape

    dff = np.full((T, G1, G2), np.nan, dtype=np.float32)
    f0_map = np.full((G1, G2), np.nan, dtype=np.float32)

    ys, xs = np.where(valid_grid)

    for y, x in zip(ys, xs):
        # : 시간축 전부
        trace = mat[:, y, x].astype(np.float32, copy=False)

        if not np.isfinite(trace).any():
            continue


        # F0는 "신호가 낮은 구간(비활성/기저 상태)"을 대표해야 한다는 가정.
        f0 = np.nanpercentile(trace, float(f0_percentile))
        f0_map[y, x] = f0

        if not np.isfinite(f0):
            continue

        # ΔF(t) = F(t) - F0
        # dF/F0(t) = ΔF(t) / F0
        dff[:, y, x] = (trace - f0) / (f0 + eps)

    return dff, f0_map
