from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .io import load_tiff_stack
from .motion import rigid_motion_correct_ecc
from .mask import projection_ref, build_mask_from_ref
from .gridder import grid_to_64_with_valid
from .dff import compute_dff
from .phase import compute_hilbert_phase
from .kuramoto import compute_kuramoto_metrics
from .csd import compute_pseudo_csd, compute_activity_map
from .report import make_spice_report

from .qc import (
    save_mask_overlay,
    save_valid_grid,
    save_heatmap,
    save_dff_snapshot,
    save_random_traces,
    save_phase_snapshot,
    save_kuramoto_timeseries,
    save_local_coherence_map,
)


@dataclass
class StepAResult:
    matrix: np.ndarray      # (T,64,64) float32, invalid bin -> NaN
    valid_grid: np.ndarray  # (64,64) bool
    mask: np.ndarray        # (H,W) bool
    meta: dict


def run_stepA(
    tiff_path: str | Path,
    out_dir: str | Path,
    *,
    grid: int = 64,
    do_motion: bool = False,          # default OFF
    mask_mode: str = "mean",          # mean / max / percentile
    mask_percentile: float = 98.0,
    min_coverage: float = 0.0,
) -> StepAResult:
    tiff_path = Path(tiff_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = load_tiff_stack(tiff_path)  # (T,H,W)

    if do_motion:
        frames, info = rigid_motion_correct_ecc(frames, mode=0)  # translation
    else:
        info = {"fail_count": 0}

    ref = projection_ref(frames, mode=mask_mode, percentile=mask_percentile)
    mask = build_mask_from_ref(ref, blur_ksize=5, closing_radius=3, min_object_size=500)

    matrix, valid_grid = grid_to_64_with_valid(frames, mask, grid=grid, min_coverage=min_coverage)

    meta = {
        "input": str(tiff_path),
        "shape_THW": list(frames.shape),
        "grid": int(grid),
        "do_motion": bool(do_motion),
        "motion_fail_count": int(info.get("fail_count", 0)),
        "mask_mode": str(mask_mode),
        "mask_percentile": float(mask_percentile),
        "mask_ratio": float(mask.mean()),
        "valid_grid_ratio": float(valid_grid.mean()),
        "nan_ratio": float(np.isnan(matrix).mean()),
        "min_coverage": float(min_coverage),
    }

    stem = tiff_path.stem

    np.save(out_dir / f"{stem}.stepA.matrix64.npy", matrix)
    np.save(out_dir / f"{stem}.stepA.valid64.npy", valid_grid.astype(np.uint8))
    np.save(out_dir / f"{stem}.stepA.mask.npy", mask.astype(np.uint8))

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    save_mask_overlay(ref, mask, qc_dir / f"{stem}.stepA_mask_overlay.png")
    save_valid_grid(valid_grid, qc_dir / f"{stem}.stepA_valid64.png")
    save_heatmap(ref, "Reference (projection)", qc_dir / f"{stem}.stepA_reference.png")

    meta["qc_stepA_overlay"] = str(qc_dir / f"{stem}.stepA_mask_overlay.png")
    meta["qc_stepA_valid64"] = str(qc_dir / f"{stem}.stepA_valid64.png")
    meta["qc_stepA_reference"] = str(qc_dir / f"{stem}.stepA_reference.png")

    return StepAResult(matrix=matrix, valid_grid=valid_grid, mask=mask, meta=meta)


def run_stepB_dff(
    matrix_path: str | Path,
    valid_path: str | Path,
    out_dir: str | Path,
    f0_percentile: float = 20.0,
) -> dict:
    matrix_path = Path(matrix_path)
    valid_path = Path(valid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = np.load(str(matrix_path)).astype(np.float32)
    valid = (np.load(str(valid_path)).astype(np.uint8) > 0)

    dff, f0 = compute_dff(mat, valid, f0_percentile=float(f0_percentile))

    name = matrix_path.name
    stem = name.replace(".stepA.matrix64.npy", "").replace(".matrix64.npy", "")

    dff_path = out_dir / f"{stem}.stepB.dff64.npy"
    f0_path = out_dir / f"{stem}.stepB.f0_64.npy"
    np.save(dff_path, dff)
    np.save(f0_path, f0)

    meta = {
        "saved_dff": str(dff_path),
        "saved_f0": str(f0_path),
        "f0_percentile": float(f0_percentile),
        "dff_nan_ratio": float(np.isnan(dff).mean()),
        "dff_min": float(np.nanmin(dff)),
        "dff_max": float(np.nanmax(dff)),
        "dff_mean": float(np.nanmean(dff)),
    }

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    save_heatmap(f0, "F0 map (64x64)", qc_dir / f"{stem}.stepB_f0_map.png")
    save_dff_snapshot(dff, t_index=dff.shape[0] // 2, out_path=qc_dir / f"{stem}.stepB_dff_snapshot.png")
    save_random_traces(dff, valid, k=6, out_path=qc_dir / f"{stem}.stepB_dff_traces.png", seed=42)

    meta["qc_stepB_f0"] = str(qc_dir / f"{stem}.stepB_f0_map.png")
    meta["qc_stepB_snap"] = str(qc_dir / f"{stem}.stepB_dff_snapshot.png")
    meta["qc_stepB_traces"] = str(qc_dir / f"{stem}.stepB_dff_traces.png")

    return meta


def run_stepC_phase(
    dff_path: str | Path,
    valid_path: str | Path,
    out_dir: str | Path,
    *,
    fs: float,
    f_lo: float = 0.1,
    f_hi: float = 3.0,
    filter_order: int = 3,
    time_decimate: int = 1,
    fill_nan: str = "interp",
    snapshot_t: int | None = None,
) -> dict:
    dff_path = Path(dff_path)
    valid_path = Path(valid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dff = np.load(str(dff_path)).astype(np.float32)
    valid = (np.load(str(valid_path)).astype(np.uint8) > 0)

    res = compute_hilbert_phase(
        dff,
        valid,
        fs=float(fs),
        f_lo=float(f_lo),
        f_hi=float(f_hi),
        filter_order=int(filter_order),
        time_decimate=int(time_decimate),
        fill_nan=str(fill_nan),
    )

    name = dff_path.name
    stem = name.replace(".stepB.dff64.npy", "").replace(".dff64.npy", "")

    phase_path = out_dir / f"{stem}.stepC.phase64.npy"
    np.save(phase_path, res.phase)

    meta = {
        "saved_phase": str(phase_path),
        **res.meta,
        "phase_shape": list(res.phase.shape),
    }

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    t_index = (res.phase.shape[0] // 2) if (snapshot_t is None) else int(snapshot_t)
    snap_path = qc_dir / f"{stem}.stepC_phase_snapshot.png"
    save_phase_snapshot(res.phase, t_index=t_index, out_path=snap_path)
    meta["qc_stepC_snap"] = str(snap_path)

    return meta


def run_stepD_kuramoto(
    phase_path: str | Path,
    valid_path: str | Path,
    out_dir: str | Path,
    *,
    fs_eff: float,
    compute_local_coherence: bool = True,
) -> dict:
    phase_path = Path(phase_path)
    valid_path = Path(valid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    phase = np.load(str(phase_path)).astype(np.float32)
    valid = (np.load(str(valid_path)).astype(np.uint8) > 0)

    res = compute_kuramoto_metrics(
        phase=phase,
        valid_grid=valid,
        fs_eff=float(fs_eff),
        compute_local_coherence=bool(compute_local_coherence),
    )

    name = phase_path.name
    stem = name.replace(".stepC.phase64.npy", "").replace(".phase64.npy", "")

    r_path = out_dir / f"{stem}.stepD.R.npy"
    phi_path = out_dir / f"{stem}.stepD.Phi.npy"
    vel_path = out_dir / f"{stem}.stepD.Phi_vel.npy"
    np.save(r_path, res.R)
    np.save(phi_path, res.Phi)
    np.save(vel_path, res.Phi_vel)

    local_path = None
    if res.local_coh is not None:
        local_path = out_dir / f"{stem}.stepD.local_coh64.npy"
        np.save(local_path, res.local_coh)

    meta = {
        "saved_R": str(r_path),
        "saved_Phi": str(phi_path),
        "saved_Phi_vel": str(vel_path),
        "saved_local_coh": (str(local_path) if local_path is not None else ""),
        **res.meta,
    }

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    ts_png = qc_dir / f"{stem}.stepD_kuramoto_timeseries.png"
    save_kuramoto_timeseries(res.R, res.Phi, res.Phi_vel, ts_png)
    meta["qc_stepD_ts"] = str(ts_png)

    if res.local_coh is not None:
        lc_png = qc_dir / f"{stem}.stepD_local_coh_map.png"
        save_local_coherence_map(res.local_coh, lc_png)
        meta["qc_stepD_local"] = str(lc_png)

    return meta


def run_stepE_csd(
    dff_path: str | Path,
    valid_path: str | Path,
    out_dir: str | Path,
    *,
    sigma: float = 1.0,
    dx: float = 1.0,
    smooth_sigma: float = 1.0,
    activity_mode: str = "mean_abs",
    snapshot_t: int | None = None,
) -> dict:
    """
    Step E (Path B):
      Pseudo-CSD(t) = -sigma * ∇²(dF/F0)(t) / dx^2

    Reflects the reference code:
      - Gaussian smoothing before Laplacian
      - dx scaling
    But uses masked Laplacian for valid64 holes/edges.
    """
    dff_path = Path(dff_path)
    valid_path = Path(valid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dff = np.load(str(dff_path)).astype(np.float32)
    valid = (np.load(str(valid_path)).astype(np.uint8) > 0)

    csd = compute_pseudo_csd(
        dff,
        valid,
        sigma=float(sigma),
        dx=float(dx),
        smooth_sigma=float(smooth_sigma),
    )
    activity = compute_activity_map(csd, valid, mode=str(activity_mode))

    name = dff_path.name
    stem = name.replace(".stepB.dff64.npy", "").replace(".dff64.npy", "")

    csd_path = out_dir / f"{stem}.stepE.csd64.npy"
    act_path = out_dir / f"{stem}.stepE.activity64.npy"
    np.save(csd_path, csd)
    np.save(act_path, activity)

    def _finite_stats(x: np.ndarray) -> tuple[float, float, float]:
        xf = x[np.isfinite(x)]
        if xf.size == 0:
            return (float("nan"), float("nan"), float("nan"))
        return (float(np.min(xf)), float(np.max(xf)), float(np.mean(xf)))

    csd_min, csd_max, csd_mean = _finite_stats(csd)
    act_min, act_max, act_mean = _finite_stats(activity)

    meta = {
        "saved_csd": str(csd_path),
        "saved_activity": str(act_path),
        "sigma": float(sigma),
        "dx": float(dx),
        "smooth_sigma": float(smooth_sigma),
        "activity_mode": str(activity_mode),
        "csd_shape": list(csd.shape),
        "csd_nan_ratio": float(np.isnan(csd).mean()),
        "csd_min": csd_min,
        "csd_max": csd_max,
        "csd_mean": csd_mean,
        "activity_min": act_min,
        "activity_max": act_max,
        "activity_mean": act_mean,
    }

    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    t_total = int(csd.shape[0])
    if t_total <= 0:
        raise ValueError("CSD output has zero frames. Check StepB dff shape.")

    if snapshot_t is None:
        t_index = t_total // 2
    else:
        t_index = int(snapshot_t)
        t_index = max(0, min(t_total - 1, t_index))

    csd_png = qc_dir / f"{stem}.stepE_csd_snapshot.png"
    act_png = qc_dir / f"{stem}.stepE_activity_map.png"

    save_heatmap(csd[t_index], f"Pseudo-CSD snapshot (t={t_index})", csd_png)
    save_heatmap(activity, f"Activity map ({activity_mode})", act_png)

    meta["qc_stepE_csd"] = str(csd_png)
    meta["qc_stepE_activity"] = str(act_png)

    return meta

def run_stepF_report(
    out_dir: str | Path,
    stem: str,
) -> dict:
    """
    Step F (Final Output):
      Collect StepA-E QC images and generate:
        - dashboard PNG
        - PDF report
        - summary JSON
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = str(stem).strip()
    if not stem:
        raise ValueError("StepF requires a non-empty stem")

    res = make_spice_report(out_dir=out_dir, stem=stem)

    meta = {
        "saved_pdf": str(res.pdf_path),
        "saved_dashboard": str(res.dashboard_png),
        "saved_summary_json": str(res.summary_json),
        "report_dir": str(res.pdf_path.parent),
        "n_found_images": int(len(res.found_images)),
        "n_missing_images": int(len(res.missing_images)),
        "included_images": [lab for lab, _ in res.found_images],
        "missing_images": list(res.missing_images),
    }

    for k, v in res.metrics.items():
        meta[k] = v

    return meta