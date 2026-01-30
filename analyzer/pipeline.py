from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .io import load_tiff_stack
from .motion import rigid_motion_correct_ecc
from .mask import projection_ref, build_mask_from_ref
from .gridder import grid_to_64_with_valid
from .dff import compute_dff

# QC
from .qc import (
    save_mask_overlay,
    save_valid_grid,
    save_heatmap,
    save_dff_snapshot,
    save_random_traces,
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

    # Mask reference
    ref = projection_ref(frames, mode=mask_mode, percentile=mask_percentile)
    mask = build_mask_from_ref(ref, blur_ksize=5, closing_radius=3, min_object_size=500)

    # Grid
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

    # Save outputs
    np.save(out_dir / f"{stem}.stepA.matrix64.npy", matrix)
    np.save(out_dir / f"{stem}.stepA.valid64.npy", valid_grid.astype(np.uint8))
    np.save(out_dir / f"{stem}.stepA.mask.npy", mask.astype(np.uint8))

    # ---- QC outputs (saved as PNG) ----
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
):
    matrix_path = Path(matrix_path)
    valid_path = Path(valid_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mat = np.load(str(matrix_path)).astype(np.float32)     
    valid = (np.load(str(valid_path)).astype(np.uint8) > 0)      

    # ===delta F 분석 및 저장 ===
    dff, f0 = compute_dff(mat, valid, f0_percentile=float(f0_percentile))

    name = matrix_path.name
    stem = name.replace(".stepA.matrix64.npy", "").replace(".matrix64.npy", "")

    dff_path = out_dir / f"{stem}.stepB.dff64.npy"
    f0_path  = out_dir / f"{stem}.stepB.f0_64.npy"

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

    # =====================

    # ---- QC outputs ----
    qc_dir = out_dir / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)

    save_heatmap(f0, "F0 map (64x64)", qc_dir / f"{stem}.stepB_f0_map.png")
    save_dff_snapshot(dff, t_index=dff.shape[0] // 2, out_path=qc_dir / f"{stem}.stepB_dff_snapshot.png")
    save_random_traces(dff, valid, k=6, out_path=qc_dir / f"{stem}.stepB_dff_traces.png", seed=42)

    meta["qc_stepB_f0"] = str(qc_dir / f"{stem}.stepB_f0_map.png")
    meta["qc_stepB_snap"] = str(qc_dir / f"{stem}.stepB_dff_snapshot.png")
    meta["qc_stepB_traces"] = str(qc_dir / f"{stem}.stepB_dff_traces.png")

    return meta
