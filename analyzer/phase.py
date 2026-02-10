from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from scipy.signal import butter, sosfiltfilt, hilbert


@dataclass
class PhaseResult:
    phase: np.ndarray     # (T2,G,G) float32, invalid bin -> NaN
    eff_fs: float         # effective sampling rate after decimation
    meta: dict


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    1D NaN/inf를 선형 보간으로 채움.
    - 모두 invalid면 그대로 반환
    """
    x = np.asarray(x, dtype=np.float32)
    idx = np.arange(x.size)
    good = np.isfinite(x)
    if not good.any():
        return x

    y = x.copy()
    y[~good] = np.interp(idx[~good], idx[good], x[good]).astype(np.float32)
    return y


def compute_hilbert_phase(
    dff: np.ndarray,          # (T,G,G)
    valid_grid: np.ndarray,   # (G,G)
    fs: float,
    f_lo: float = 0.1,
    f_hi: float = 3.0,
    filter_order: int = 3,
    time_decimate: int = 1,
    fill_nan: str = "interp",       # "interp" | "zero" | "none"
) -> PhaseResult:
    """
    dFF (T,G,G)에서 각 valid bin의 time-trace에 대해
    bandpass -> Hilbert -> angle 로 instantaneous phase를 구한다.

    구현 중심 리팩토링 포인트:
    - NaN 처리 유틸 통일
    - 필터 파라미터 clamp/검증 명확화
    - sosfiltfilt/hilbert dtype 안정화
    """
    if dff.ndim != 3:
        raise ValueError("dff must be (T,G,G)")
    if valid_grid.shape != dff.shape[1:]:
        raise ValueError("valid_grid shape mismatch")
    if fs <= 0:
        raise ValueError("fs must be > 0")

    T, G1, G2 = dff.shape

    dec = max(1, int(time_decimate))
    eff_fs = float(fs) / dec

    # ---- bandpass design (based on original fs) ----
    nyq = 0.5 * float(fs)
    lo_hz = max(0.0, float(f_lo))
    hi_hz = float(f_hi)

    # clamp hi below Nyquist
    hi_hz = min(hi_hz, nyq * 0.95)

    if hi_hz <= 0:
        raise ValueError(f"Invalid f_hi after Nyquist clamp: f_hi={f_hi}, nyq={nyq}")
    if lo_hz >= hi_hz:
        raise ValueError(f"Need f_lo < f_hi. Got f_lo={lo_hz}, f_hi={hi_hz}, fs={fs}")

    lo = lo_hz / nyq
    hi = hi_hz / nyq
    sos = butter(int(filter_order), [lo, hi], btype="band", output="sos")

    # output length after decimation
    T2 = (T + dec - 1) // dec
    phase = np.full((T2, G1, G2), np.nan, dtype=np.float32)

    ys, xs = np.where(valid_grid)

    for y, x in zip(ys, xs):
        trace = dff[:, y, x].astype(np.float32, copy=False)

        # 완전 invalid면 스킵
        if not np.isfinite(trace).any():
            continue

        # ---- fill nan policy ----
        if fill_nan == "interp":
            trace2 = _interp_nan_1d(trace)
        elif fill_nan == "zero":
            trace2 = np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
        elif fill_nan == "none":
            trace2 = trace
        else:
            raise ValueError("fill_nan must be one of: 'interp', 'zero', 'none'")

        # ---- bandpass (zero-phase) ----
        # sosfiltfilt는 float64로 돌 수도 있으니 float32로 다시 캐스팅
        x_f = sosfiltfilt(sos, trace2).astype(np.float32, copy=False)

        # ---- decimate (simple stride) ----
        if dec > 1:
            x_f = x_f[::dec]

        # ---- Hilbert -> phase ----
        # hilbert는 complex128를 줄 수 있어서 angle 후 float32로 다운캐스팅
        z = hilbert(x_f)
        ph = np.angle(z).astype(np.float32, copy=False)

        # save
        phase[: ph.size, y, x] = ph  # 혹시라도 길이 차이 방어

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
