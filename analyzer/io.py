from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile

# TIFF를 항상 (T,H,W) float32로 표준화해 이후 파이프라인(shape 가정)이 깨지지 않게 한다
def load_tiff_stack(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(str(p))

    arr = tifffile.imread(str(p))

    # (H,W) -> (1,H,W)
    if arr.ndim == 2:
        arr = arr[None, ...]

    # only accept (T,H,W)
    if arr.ndim != 3:
        raise ValueError(
            f"Unsupported TIFF shape: {arr.shape}. Expected (T,H,W) or (H,W)."
        )

    return np.asarray(arr, dtype=np.float32)
