from __future__ import annotations
from pathlib import Path
import numpy as np
import tifffile

def load_tiff_stack(path: str | Path) -> np.ndarray:
    p = Path(path)
    arr = tifffile.imread(str(p))
    if arr.ndim == 2:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError(f"Unsupported TIFF shape: {arr.shape}. Expected (T,H,W) or (H,W).")
    return arr.astype(np.float32)
