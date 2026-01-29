from __future__ import annotations
import numpy as np
import cv2
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_objects, binary_closing, disk

# 대표 이미지 1장 뽑기
def projection_ref(frames: np.ndarray, mode: str = "mean", percentile: float = 98.0) -> np.ndarray:
    """
    mode:
    - "mean": 전체 프레임 평균 (노이즈 감소, 안정적)
    - "max": 전체 프레임 최대값 (밝은 구조 강조)
    - "percentile": 상위 분위수 (max보다 outlier에 덜 민감, 구조 강조)
    """
    if mode == "mean":
        return frames.mean(axis=0)
    if mode == "max":
        return frames.max(axis=0)
    if mode == "percentile":
        return np.percentile(frames, percentile, axis=0)
    raise ValueError(f"Unknown mode: {mode}")

# === Denoising --> Gaussian Blur ===
def _preprocess_ref(ref: np.ndarray, blur_ksize: int = 5) -> np.ndarray:
    x = ref.astype(np.float32) 
    if blur_ksize and blur_ksize >= 3:
        x = cv2.GaussianBlur(x, (blur_ksize, blur_ksize), 0)
    return x

# === 대표 이미지(ref)를 입력 받아 Binary Mask 생성 ===
def build_mask_from_ref(
    ref: np.ndarray,
    blur_ksize: int = 5,
    closing_radius: int = 3,
    min_object_size: int = 500,
) -> np.ndarray:
    """
    입력:
      ref: (H,W) float32 reference image (예: mean/max/percentile projection)
    출력:
      mask: (H,W) bool, ROI=True / background=False

    로직:
      1) preprocess ref로 denoising
      2) Otsu로 자동 임계값(threshold) 계산
      3) ref > threshold 로 foreground 후보 생성
      4) closing으로 작은 구멍을 메워 ROI가 끊기지 않게 보정
      5) 작은 객체 제거로 background 잡영 제거
    """
    x = _preprocess_ref(ref, blur_ksize)

    thr = threshold_otsu(x)
    m = x > thr # m : True(ROI)

    # 구멍 채우기
    if closing_radius and closing_radius > 0:
        m = binary_closing(m, footprint=disk(closing_radius))

    # 작은 점 제거
    if min_object_size and min_object_size > 0:
        m = remove_small_objects(m.astype(bool), min_size=min_object_size)

    return m.astype(bool)


