from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class KuramotoResult:
    R: np.ndarray
    Phi: np.ndarray
    Phi_vel: np.ndarray
    local_coh: np.ndarray | None
    meta: dict


def _interp_nan_1d(x: np.ndarray) -> np.ndarray:
    """
    1D NaN/inf 구간을 선형 보간으로 메꾼다.
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


def _kuramoto_R_Phi(P: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    입력 P: (N_bins, T) phase (rad)
    출력: R(t), Phi(t)  (각각 (T,))
    """
    P = np.asarray(P, dtype=np.float32)
    valid = np.isfinite(P)

    # exp(j*theta) for valid entries, 0 for invalid
    E = np.zeros(P.shape, dtype=np.complex64)
    E[valid] = np.exp(1j * P[valid]).astype(np.complex64)

    cnt = valid.sum(axis=0).astype(np.float32)  # (T,)
    sum_vec = E.sum(axis=0)                     # (T,) complex

    # mean_vec = sum_vec / cnt, but if cnt==0 -> NaN
    mean_vec = np.full(sum_vec.shape, np.nan + 1j * np.nan, dtype=np.complex64)
    ok = cnt > 0
    mean_vec[ok] = (sum_vec[ok] / cnt[ok]).astype(np.complex64)

    R = np.abs(mean_vec).astype(np.float32)
    Phi = np.angle(mean_vec).astype(np.float32)
    return R, Phi


def _phi_velocity(Phi: np.ndarray, fs_eff: float) -> np.ndarray:
    """
    Phi(t)에서 phase velocity dPhi/dt 계산.
    구현 안정성을 위해:
    - Phi에 NaN이 있으면 1D 보간 후 unwrap + gradient
    """
    if fs_eff <= 0:
        raise ValueError("fs_eff must be > 0")

    Phi = np.asarray(Phi, dtype=np.float32).reshape(-1)
    if not np.isfinite(Phi).any():
        return np.full_like(Phi, np.nan, dtype=np.float32)

    Phi_f = _interp_nan_1d(Phi)
    Phi_u = np.unwrap(Phi_f).astype(np.float32)
    return (np.gradient(Phi_u) * float(fs_eff)).astype(np.float32)


def _local_coherence_per_bin(P: np.ndarray, Phi: np.ndarray) -> np.ndarray:
    """
    각 bin i에 대해:
      coh_i = | < exp(j( theta_i(t) - Phi(t) )) >_t |
    입력:
      P   : (N_bins, T)
      Phi : (T,)
    출력:
      coh : (N_bins,)
    """
    P = np.asarray(P, dtype=np.float32)
    Phi = np.asarray(Phi, dtype=np.float32).reshape(-1)

    vt = np.isfinite(Phi)
    if vt.sum() == 0:
        return np.full((P.shape[0],), np.nan, dtype=np.float32)

    P_sub = P[:, vt]           # (N, T_v)
    Phi_sub = Phi[vt][None, :] # (1, T_v)

    V = np.isfinite(P_sub)
    diff = P_sub - Phi_sub

    # valid만 exp, invalid는 0
    Z = np.zeros_like(diff, dtype=np.complex64)
    Z[V] = np.exp(1j * diff[V]).astype(np.complex64)

    K = V.sum(axis=1).astype(np.float32)  # (N,)
    out = np.full((P.shape[0],), np.nan, dtype=np.float32)
    ok = K > 0
    out[ok] = np.abs(Z.sum(axis=1)[ok] / K[ok]).astype(np.float32)
    return out


def compute_kuramoto_metrics(
    phase: np.ndarray,
    valid_grid: np.ndarray,
    fs_eff: float,
    *,
    compute_local_coherence: bool = True,
) -> KuramotoResult:
    if phase.ndim != 3:
        raise ValueError("phase must be (T,G,G)")
    if valid_grid.shape != phase.shape[1:]:
        raise ValueError("valid_grid shape mismatch")
    if fs_eff <= 0:
        raise ValueError("fs_eff must be > 0")

    T2, G1, G2 = phase.shape

    ys, xs = np.where(valid_grid)
    N = int(len(ys))
    if N == 0:
        R = np.full((T2,), np.nan, dtype=np.float32)
        Phi = np.full((T2,), np.nan, dtype=np.float32)
        Phi_vel = np.full((T2,), np.nan, dtype=np.float32)
        return KuramotoResult(
            R=R,
            Phi=Phi,
            Phi_vel=Phi_vel,
            local_coh=None,
            meta={"N_valid_bins": 0, "R_nan_ratio": float(np.isnan(R).mean())},
        )

    # P: (N_bins, T)
    P = phase[:, ys, xs].T.astype(np.float32, copy=False)

    # === Kuramoto global ===
    R, Phi = _kuramoto_R_Phi(P)
    Phi_vel = _phi_velocity(Phi, fs_eff=float(fs_eff))
    # =======================

    local_map = None
    if compute_local_coherence:
        coh = _local_coherence_per_bin(P, Phi)  # (N,)
        local_map = np.full((G1, G2), np.nan, dtype=np.float32)
        local_map[ys, xs] = coh

    meta = {
        "N_valid_bins": int(N),
        "fs_eff": float(fs_eff),
        "R_mean": float(np.nanmean(R)),
        "R_nan_ratio": float(np.isnan(R).mean()),
        "Phi_nan_ratio": float(np.isnan(Phi).mean()),
        "vel_nan_ratio": float(np.isnan(Phi_vel).mean()),
    }
    return KuramotoResult(R=R, Phi=Phi, Phi_vel=Phi_vel, local_coh=local_map, meta=meta)
