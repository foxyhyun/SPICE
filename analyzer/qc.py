from __future__ import annotations
from pathlib import Path
import numpy as np

import matplotlib
matplotlib.use("Agg")   # ✅ pyplot import 이전에!

import matplotlib.pyplot as plt


def _save_fig(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_mask_overlay(ref: np.ndarray, mask: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(ref, cmap="gray")
    ax.imshow(mask.astype(np.float32), alpha=0.35)
    ax.set_title("Mask overlay on reference")
    ax.axis("off")
    _save_fig(fig, out_path)


def save_valid_grid(valid64: np.ndarray, out_path: str | Path):
    out_path = Path(out_path)
    fig = plt.figure()
    ax = plt.gca()
    ax.imshow(valid64.astype(np.float32), interpolation="nearest")
    ax.set_title("Valid grid (64x64)")
    ax.axis("off")
    _save_fig(fig, out_path)


def save_heatmap(map2d: np.ndarray, title: str, out_path: str | Path):
    out_path = Path(out_path)
    fig = plt.figure()
    ax = plt.gca()
    im = ax.imshow(map2d, interpolation="nearest")
    ax.set_title(title)
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    _save_fig(fig, out_path)


def save_dff_snapshot(dff: np.ndarray, t_index: int, out_path: str | Path):
    t_index = int(t_index)
    if t_index < 0 or t_index >= dff.shape[0]:
        t_index = dff.shape[0] // 2
    save_heatmap(dff[t_index], f"dFF snapshot (t={t_index})", out_path)


def save_random_traces(dff: np.ndarray, valid64: np.ndarray, k: int, out_path: str | Path, seed: int = 0):
    ys, xs = np.where(valid64)
    if len(ys) == 0:
        return
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(ys), size=min(k, len(ys)), replace=False)

    fig = plt.figure()
    ax = plt.gca()
    for i in idx:
        y, x = ys[i], xs[i]
        ax.plot(dff[:, y, x], linewidth=1.0, label=f"({y},{x})")
    ax.set_title(f"dFF traces (random {min(k, len(ys))} valid bins)")
    ax.set_xlabel("t")
    ax.set_ylabel("dF/F0")
    ax.legend(fontsize=7, ncol=2)
    _save_fig(fig, Path(out_path))


def save_phase_snapshot(phase: np.ndarray, t_index: int, out_path: str | Path):
    t_index = int(t_index)
    if t_index < 0 or t_index >= phase.shape[0]:
        t_index = phase.shape[0] // 2

    snap = phase[t_index]

    out_path = Path(out_path)
    fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(snap, interpolation="nearest", vmin=-np.pi, vmax=np.pi)
    ax.set_title(f"Phase snapshot (t={t_index})")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    _save_fig(fig, out_path)
