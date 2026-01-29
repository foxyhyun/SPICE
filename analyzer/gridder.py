from __future__ import annotations
import numpy as np

def grid_to_64_with_valid(
    frames: np.ndarray, 
    mask: np.ndarray, # bool -> True/False
    grid: int = 64,
    min_coverage: float = 0.0, # 이 값 설정
) -> tuple[np.ndarray, np.ndarray]:
    
    if frames.ndim != 3:
        raise ValueError("frames must be (T,H,W)")
    if mask.ndim != 2:
        raise ValueError("mask must be (H,W)")
    T, H, W = frames.shape
    if mask.shape != (H, W):
        raise ValueError("mask shape mismatch")

    # ROI 밖은 전부 NaN
    f = frames.copy()
    f[:, ~mask] = np.nan

    ys = np.linspace(0, H, grid + 1, dtype=int)
    xs = np.linspace(0, W, grid + 1, dtype=int)

    # mat : 전부 nan으로 채우기 -> 신호값
    # valid_grid : 전부 0으로 채우기 -> binary
    mat = np.full((T, grid, grid), np.nan, dtype=np.float32)
    valid_grid = np.zeros((grid, grid), dtype=bool)

    for gy in range(grid):
        y0, y1 = ys[gy], ys[gy + 1]
        for gx in range(grid):
            x0, x1 = xs[gx], xs[gx + 1]

            # ROI 안의 신호 평균값이 낮으면 패스
            mblock = mask[y0:y1, x0:x1].reshape(-1) # to calcluate mean
            coverage = float(mblock.mean()) if mblock.size > 0 else 0.0
            if coverage <= min_coverage:
                continue

            # block 내에서 NaN은 전부 0으로 처리    
            block = f[:, y0:y1, x0:x1].reshape(T, -1)
            valid = np.isfinite(block) # NaN 아닌 픽셀만 True
            if not valid.any():
                continue

            # 시간별 평균 계산
            count = valid.sum(axis=1).astype(np.float32)  # (T,)
            s = np.where(valid, block, 0.0).sum(axis=1).astype(np.float32)
            v = np.full((T,), np.nan, dtype=np.float32)
            np.divide(s, count, out=v, where=(count > 0)) # count > 0인 곳은 v = s/count로 채움

            mat[:, gy, gx] = v # 각 블록 내의 평균 intensity
            valid_grid[gy, gx] = True # 계산 대상으로 써도 되는지에 대한 bool 값

    return mat, valid_grid


# I don't recommend use this function. 
def grid_to_64(
    frames: np.ndarray,
    mask: np.ndarray,
    grid: int = 64,
    fill_value: float = 0.0,
    min_coverage: float = 0.0,
) -> np.ndarray:
    
    mat, _ = grid_to_64_with_valid(frames, mask, grid=grid, min_coverage=min_coverage)
    if fill_value is None:
        return mat
    return np.nan_to_num(mat, nan=float(fill_value)).astype(np.float32)
    