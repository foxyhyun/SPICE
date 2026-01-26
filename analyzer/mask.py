from __future__ import annotations
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk

def _preprocess_ref(ref: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    x = ref.astype(np.float32)
    if blur_ksize and blur_ksize >= 3:
        x = cv2.GaussianBlur(x, (blur_ksize, blur_ksize), 0)
    return x

def build_mask_from_ref(
    ref: np.ndarray,
    blur_ksize: int = 5,
    closing_radius: int = 3,
    min_object_size: int = 500,
) -> np.ndarray:
    """
    ref: (H,W) float32
    """
    x = _preprocess_ref(ref, blur_ksize)
    thr = threshold_otsu(x)
    m = x > thr

    if closing_radius and closing_radius > 0:
        m = binary_closing(m, footprint=disk(closing_radius))
    if min_object_size and min_object_size > 0:
        m = remove_small_objects(m.astype(bool), min_size=min_object_size)
    return m.astype(bool)

def projection_ref(frames: np.ndarray, mode: str = "mean", percentile: float = 98.0) -> np.ndarray:
    """
    frames: (T,H,W)
    mode: "mean" | "max" | "percentile"
    """
    if mode == "mean":
        return frames.mean(axis=0)
    if mode == "max":
        return frames.max(axis=0)
    if mode == "percentile":
        return np.percentile(frames, percentile, axis=0)
    raise ValueError(f"Unknown mode: {mode}")
