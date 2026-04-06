from __future__ import annotations
import numpy as np
import cv2

def _to_u8(img: np.ndarray) -> np.ndarray:
    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def rigid_motion_correct_ecc(
    frames: np.ndarray,
    max_iters: int = 50,
    eps: float = 1e-5,
    mode: int = cv2.MOTION_TRANSLATION,
) -> tuple[np.ndarray, dict]:
    """
    frames: (T,H,W) float32
    mode: cv2.MOTION_TRANSLATION (권장 시작) or cv2.MOTION_EUCLIDEAN
    returns:
      stabilized: (T,H,W)
      info: dict with fail_count, warp_mats (T,2,3)
    """
    if frames.ndim != 3:
        raise ValueError("frames must be (T,H,W)")

    T, H, W = frames.shape
    ref = frames[0]
    ref8 = _to_u8(ref)

    stabilized = np.empty_like(frames)
    stabilized[0] = ref

    warp_mats = np.zeros((T, 2, 3), dtype=np.float32)
    warp_mats[0] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, max_iters, eps)

    fail = 0
    for t in range(1, T):
        img = frames[t]
        img8 = _to_u8(img)

        warp = np.eye(2, 3, dtype=np.float32)
        try:
            cv2.findTransformECC(ref8, img8, warp, mode, criteria)
            warped = cv2.warpAffine(
                img, warp, (W, H),
                flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT
            )
            stabilized[t] = warped
            warp_mats[t] = warp
        except cv2.error:
            stabilized[t] = img
            warp_mats[t] = np.array([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
            fail += 1

    info = {"fail_count": fail, "warp_mats": warp_mats}
    return stabilized, info
