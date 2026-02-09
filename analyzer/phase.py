from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from scipy.signal import butter, sosfiltfilt, hilbert

@dataclass
class PhaseResult:
    phase: np.ndarray     
    eff_fs: float        
    meta: dict

def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    idx = np.arange(x.size)
    good = np.isfinite(x)

    if not good.any():
        return x

    x_filled = x.copy()
    x_filled[~good] = np.interp(idx[~good], idx[good], x[good]).astype(np.float32)
    return x_filled


def compute_hilbert_phase(
    dff: np.ndarray,       
    valid_grid: np.ndarray,   
    fs: float,             
    f_lo: float = 0.1,        
    f_hi: float = 3.0,
    filter_order: int = 3,
    time_decimate: int = 1,         
    fill_nan: str = "interp",       # "interp" | "zero" | "none"
) -> PhaseResult:
    """
      StepB의 dFF (T,G,G)에서 각 valid bin의 time-trace에 대해
      bandpass -> Hilbert -> angle 로 instantaneous phase를 구한다.
    """
    if dff.ndim != 3:
        raise ValueError("dff must be (T,G,G)")
    if valid_grid.shape != dff.shape[1:]:
        raise ValueError("valid_grid shape mismatch")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    T, G1, G2 = dff.shape # Time, Grid
    dec = max(1, int(time_decimate)) # 시간 다운 샘플링 비율, 1 : 그대로, n : n프레임에 1프레임만 사용
    eff_fs = float(fs) / dec # 시간 다운샘플링 후의 실제 샘플링 주파수

    nyq = 0.5 * float(fs) # 이것보다 높은 주파수는 무시
    lo_hz = max(0.0, float(f_lo))
    hi_hz = float(f_hi)

    # hi는 Nyquist보다 작아야 함
    hi_hz = min(hi_hz, nyq * 0.95)
    if hi_hz <= 0:
        raise ValueError(f"Invalid f_hi after Nyquist clamp: f_hi={f_hi}, nyq={nyq}")
    if lo_hz >= hi_hz:
        raise ValueError(f"Need f_lo < f_hi. Got f_lo={lo_hz}, f_hi={hi_hz}, fs={fs}")

    # normalized
    lo = lo_hz / nyq
    hi = hi_hz / nyq
    sos = butter(filter_order, [lo, hi], btype="band", output="sos")

    T2 = (T + dec - 1) // dec
    phase = np.full((T2, G1, G2), np.nan, dtype=np.float32)

    ys, xs = np.where(valid_grid)

    # 64 x 64 grid의 각 칸을 하나의 time series로 보고 그 신호에서 pahse 계산       
    for y, x in zip(ys, xs):
        trace = dff[:, y, x].astype(np.float32, copy=False) # 신호 칸 하나를 불러와서 delta F

        if not np.isfinite(trace).any():
            continue

        if fill_nan == "interp":
            trace2 = _interp_nan_1d(trace)
        elif fill_nan == "zero":
            trace2 = np.nan_to_num(trace, nan=0.0).astype(np.float32, copy=False)
        elif fill_nan == "none":
            trace2 = trace
        else:
            raise ValueError("fill_nan must be one of: 'interp', 'zero', 'none'")

        # 느린 drift 제거, 빠른 노이즈 제거
        x_f = sosfiltfilt(sos, trace2).astype(np.float32, copy=False)

        # 시간 다운샘플링
        if dec > 1:
            x_f = x_f[::dec]

        # Hilbert transform -> phase
        z = hilbert(x_f)
        ph = np.angle(z).astype(np.float32, copy=False) 

        # 위상 시계열 저장 
        phase[:, y, x] = ph

    meta = {
        "fs": float(fs),
        "eff_fs": float(eff_fs),
        "time_decimate": int(dec),
        "f_lo": float(lo_hz),
        "f_hi": float(hi_hz),
        "filter_order": int(filter_order),
        "fill_nan": str(fill_nan),
        "phase_nan_ratio": float(np.isnan(phase).mean()),
    }

    return PhaseResult(phase=phase, eff_fs=eff_fs, meta=meta)
