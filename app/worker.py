from __future__ import annotations
from PySide6.QtCore import QObject, Signal, Slot
from analyzer.pipeline import run_stepA, run_stepB_dff

class StepAWorker(QObject):
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, tiff_path: str, out_dir: str, do_motion: bool, mask_mode: str, mask_p: float):
        super().__init__()
        self.tiff_path = tiff_path
        self.out_dir = out_dir
        self.do_motion = do_motion
        self.mask_mode = mask_mode
        self.mask_p = mask_p

    @Slot()
    def run(self):
        try:
            self.log.emit("Running Step A: load -> (opt) motion -> mask -> grid64 -> save")
            res = run_stepA(
                self.tiff_path,
                self.out_dir,
                do_motion=self.do_motion,
                mask_mode=self.mask_mode,
                mask_percentile=self.mask_p,
            )
            self.done.emit(res.meta)
        except Exception as e:
            self.failed.emit(str(e))

class StepBWorker(QObject):
    log = Signal(str)
    done = Signal(dict)
    failed = Signal(str)

    def __init__(self, matrix_path: str, valid_path: str, out_dir: str, f0_percentile: float):
        super().__init__()
        self.matrix_path = matrix_path
        self.valid_path = valid_path
        self.out_dir = out_dir
        self.f0_percentile = f0_percentile

    @Slot()
    def run(self):
        try:
            self.log.emit("Running Step B: ΔF/F0 (valid bins only) -> save")
            meta = run_stepB_dff(
                self.matrix_path,
                self.valid_path,
                self.out_dir,
                f0_percentile=self.f0_percentile,
            )
            self.done.emit(meta)
        except Exception as e:
            self.failed.emit(str(e))