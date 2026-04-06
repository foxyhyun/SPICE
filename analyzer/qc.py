from __future__ import annotations
from pathlib import Path
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas


def _save_fig(fig: Figure, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas = FigureCanvas(fig)
    canvas.print_figure(str(path), dpi=160, bbox_inches="tight")


def save_mask_overlay(ref: np.ndarray, mask: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    ref = np.asarray(ref)
    mask = np.asarray(mask)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(ref, cmap="gray")
    ax.imshow(mask.astype(np.float32), alpha=0.35)
    ax.set_title("Mask overlay on reference")
    ax.axis("off")
    _save_fig(fig, out_path)


def save_valid_grid(valid64: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    valid64 = np.asarray(valid64)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(valid64.astype(np.float32), interpolation="nearest")
    ax.set_title("Valid grid (64x64)")
    ax.axis("off")
    _save_fig(fig, out_path)


def save_heatmap(
    map2d: np.ndarray,
    title: str,
    out_path: str | Path,
    *,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str | None = None,
):
    out_path = Path(out_path)
    map2d = np.asarray(map2d)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(map2d, interpolation="nearest", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_path)


def save_dff_snapshot(dff: np.ndarray, t_index: int, out_path: str | Path):
    dff = np.asarray(dff)
    t_index = int(t_index)
    if t_index < 0 or t_index >= dff.shape[0]:
        t_index = dff.shape[0] // 2
    save_heatmap(dff[t_index], f"dFF snapshot (t={t_index})", out_path)


def save_random_traces(
    dff: np.ndarray,
    valid64: np.ndarray,
    k: int,
    out_path: str | Path,
    seed: int = 0,
):
    dff = np.asarray(dff)
    valid64 = np.asarray(valid64)

    ys, xs = np.where(valid64)
    if len(ys) == 0:
        return

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=min(int(k), len(ys)), replace=False)

    out_path = Path(out_path)
    fig = Figure(figsize=(8, 4))
    ax = fig.add_subplot(1, 1, 1)

    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        ax.plot(dff[:, y, x], linewidth=1.0, label=f"({y},{x})")

    ax.set_title(f"dFF traces (random {min(int(k), len(ys))} valid bins)")
    ax.set_xlabel("t")
    ax.set_ylabel("dF/F0")
    ax.legend(fontsize=7, ncol=2)
    _save_fig(fig, out_path)


def save_phase_snapshot(phase: np.ndarray, t_index: int, out_path: str | Path):
    phase = np.asarray(phase)
    t_index = int(t_index)
    if t_index < 0 or t_index >= phase.shape[0]:
        t_index = phase.shape[0] // 2

    snap = phase[t_index]
    out_path = Path(out_path)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(snap, interpolation="nearest", vmin=-np.pi, vmax=np.pi)
    ax.set_title(f"Phase snapshot (t={t_index})")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save_fig(fig, out_path)


def _moving_average(x: np.ndarray, win: int) -> np.ndarray:
    """
    아주 가벼운 QC용 smoothing.
    - win<=1이면 그대로 반환
    - NaN은 일단 무시하지 않고 그대로 전파될 수 있음(대부분 NaN 없을 거라 가정)
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    w = int(win)
    if w <= 1 or x.size == 0:
        return x
    if w > x.size:
        w = x.size
    kernel = np.ones((w,), dtype=np.float32) / float(w)
    # 'same'으로 길이 유지
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def save_kuramoto_timeseries(
    R: np.ndarray,
    Phi: np.ndarray,
    Phi_vel: np.ndarray,
    out_path: str | Path,
    *,
    unwrap_phi: bool = True,
    vel_smooth_win: int = 11,   # QC용: 11프레임 이동평균
):
    """
    개선사항:
    - Phi는 angle 특성상 [-pi,pi] 점프가 많아서 plot이 난잡해짐 → QC에서는 unwrap 권장
    - Phi_vel은 gradient 특성상 노이즈가 커 보임 → QC에서는 가벼운 smoothing 권장
    """
    R = np.asarray(R, dtype=np.float32).reshape(-1)
    Phi = np.asarray(Phi, dtype=np.float32).reshape(-1)
    Phi_vel = np.asarray(Phi_vel, dtype=np.float32).reshape(-1)

    out_path = Path(out_path)
    fig = Figure(figsize=(10, 6))

    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    ax1.plot(R)
    ax1.set_title("Kuramoto R(t)")
    ax1.set_ylabel("R")

    if unwrap_phi:
        phi_plot = np.unwrap(Phi.astype(np.float64)).astype(np.float32)
        phi_plot = phi_plot - phi_plot[0]
        ax2.plot(phi_plot)
        ax2.set_title("Global phase Phi(t) [unwrapped]")
    else:
        ax2.plot(Phi)
        ax2.set_title("Global phase Phi(t)")
    ax2.set_ylabel("rad")

    vel_plot = _moving_average(Phi_vel, vel_smooth_win)
    ax3.plot(vel_plot)
    ax3.set_title(f"Phase velocity dPhi/dt [smoothed win={int(vel_smooth_win)}]")
    ax3.set_ylabel("rad/s")
    ax3.set_xlabel("t")

    fig.tight_layout()
    _save_fig(fig, out_path)


def save_local_coherence_map(local_coh: np.ndarray, out_path: str | Path):
    local_coh = np.asarray(local_coh, dtype=np.float32)
    out_path = Path(out_path)

    fig = Figure()
    ax = fig.add_subplot(1, 1, 1)

    im = ax.imshow(local_coh, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title("Local coherence per bin (0~1)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save_fig(fig, out_path)
