from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .io import load_tiff_stack
from .mask import otsu_mask
from .gridder import grid_to_64

# optional
from .motion import rigid_motion_correct_ecc

@dataclass
class StepAResult:
    matrix: np.ndarray  # (T,64,64)
    mask: np.ndarray    # (H,W)

class SPICE_Analyzer:
    def __init__(self, grid: int = 64, do_motion: bool = False):
        self.grid = grid
        self.do_motion = do_motion

    def video_to_matrix(self, tiff_path: str | Path) -> StepAResult:
        frames = load_tiff_stack(tiff_path)  # (T,H,W)

        if self.do_motion:
            frames, info = rigid_motion_correct_ecc(frames, mode=0)  # translation
            # info is available if needed later

        m = otsu_mask(frames[0])
        mat = grid_to_64(frames, m, grid=self.grid, fill_value=0.0)
        return StepAResult(matrix=mat, mask=m)

    def save_matrix(self, matrix: np.ndarray, out_path: str | Path) -> Path:
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out), matrix)
        return out
