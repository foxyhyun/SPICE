from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QThread
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QCheckBox, QComboBox,
    QDoubleSpinBox, QSpinBox, QLineEdit, QGroupBox, QStackedWidget, QMessageBox
)

from .worker import StepAWorker, StepBWorker, StepCWorker


# ----------------------------
# Small UI helpers
# ----------------------------
class ImagePreview(QLabel):
    def __init__(self, title: str):
        super().__init__()
        self._title = title
        self._img_path: str | None = None

        self.setAlignment(Qt.AlignCenter)
        self.setMinimumHeight(240)
        self.setStyleSheet("border: 1px solid #999; background: #111; color: #ddd;")
        self.setText(f"{self._title}\n(no image)")

    def set_image(self, path: str | None):
        self._img_path = path
        if not path or not Path(path).exists():
            self.setText(f"{self._title}\n(no image)")
            self.setPixmap(QPixmap())
            return
        pix = QPixmap(path)
        if pix.isNull():
            self.setText(f"{self._title}\n(failed to load)")
            self.setPixmap(QPixmap())
            return
        self._render_scaled(pix)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._img_path and Path(self._img_path).exists():
            pix = QPixmap(self._img_path)
            if not pix.isNull():
                self._render_scaled(pix)

    def _render_scaled(self, pix: QPixmap):
        scaled = pix.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.setPixmap(scaled)
        self.setText("")


# ----------------------------
# Wizard Pages
# ----------------------------
class StepAPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        box = QGroupBox("Step 1: TIFF → (opt) Motion → Mask → Grid64 → Save")
        b = QVBoxLayout(box)

        r1 = QHBoxLayout()
        self.in_tiff = QLineEdit()
        btn_pick = QPushButton("Select TIFF")
        btn_pick.clicked.connect(self.pick_tiff)
        r1.addWidget(QLabel("Input TIFF:"))
        r1.addWidget(self.in_tiff)
        r1.addWidget(btn_pick)
        b.addLayout(r1)

        r2 = QHBoxLayout()
        self.out_dir = QLineEdit()
        btn_out = QPushButton("Select Output Folder")
        btn_out.clicked.connect(self.pick_outdir)
        r2.addWidget(QLabel("Output folder:"))
        r2.addWidget(self.out_dir)
        r2.addWidget(btn_out)
        b.addLayout(r2)

        r3 = QHBoxLayout()
        self.chk_motion = QCheckBox("Rigid motion correction (ECC) [default OFF]")
        self.chk_motion.setChecked(False)

        self.cmb_mask = QComboBox()
        self.cmb_mask.addItems(["mean", "percentile", "max"])

        self.sp_maskp = QDoubleSpinBox()
        self.sp_maskp.setRange(50.0, 99.9)
        self.sp_maskp.setSingleStep(1.0)
        self.sp_maskp.setValue(98.0)

        r3.addWidget(self.chk_motion)
        r3.addSpacing(20)
        r3.addWidget(QLabel("Mask mode:"))
        r3.addWidget(self.cmb_mask)
        r3.addWidget(QLabel("p:"))
        r3.addWidget(self.sp_maskp)
        r3.addStretch(1)
        b.addLayout(r3)

        r4 = QHBoxLayout()
        self.btn_run = QPushButton("Run Step 1")
        r4.addWidget(self.btn_run)
        r4.addStretch(1)

        self.btn_open_qc = QPushButton("Open QC folder")
        self.btn_open_qc.setEnabled(False)
        r4.addWidget(self.btn_open_qc)
        b.addLayout(r4)

        lay.addWidget(box)

        pbox = QGroupBox("Step 1 Results (QC Preview)")
        p = QHBoxLayout(pbox)
        self.prev_overlay = ImagePreview("Mask Overlay")
        self.prev_valid = ImagePreview("Valid Grid (64×64)")
        p.addWidget(self.prev_overlay)
        p.addWidget(self.prev_valid)
        lay.addWidget(pbox)

        lay.addStretch(1)

        self.btn_open_qc.clicked.connect(self.open_qc)

    def pick_tiff(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select TIFF", "", "TIFF Files (*.tif *.tiff)")
        if p:
            self.in_tiff.setText(p)
            self.out_dir.setText(str(Path(p).parent))

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir.setText(d)

    def open_qc(self):
        out = self.out_dir.text().strip()
        if not out:
            return
        qc_dir = Path(out) / "qc"
        if qc_dir.exists():
            os.startfile(str(qc_dir))


class StepBPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        box = QGroupBox("Step 2: ΔF/F0 (valid bins only) → Save")
        b = QVBoxLayout(box)

        r1 = QHBoxLayout()
        self.in_matrix = QLineEdit()
        btn_m = QPushButton("Select StepA matrix64.npy")
        btn_m.clicked.connect(self.pick_matrix)
        r1.addWidget(QLabel("StepA matrix:"))
        r1.addWidget(self.in_matrix)
        r1.addWidget(btn_m)
        b.addLayout(r1)

        r2 = QHBoxLayout()
        self.in_valid = QLineEdit()
        btn_v = QPushButton("Select StepA valid64.npy")
        btn_v.clicked.connect(self.pick_valid)
        r2.addWidget(QLabel("StepA valid:"))
        r2.addWidget(self.in_valid)
        r2.addWidget(btn_v)
        b.addLayout(r2)

        r3 = QHBoxLayout()
        self.out_dir = QLineEdit()
        btn_out = QPushButton("Select Output Folder")
        btn_out.clicked.connect(self.pick_outdir)
        r3.addWidget(QLabel("Output folder:"))
        r3.addWidget(self.out_dir)
        r3.addWidget(btn_out)
        b.addLayout(r3)

        r4 = QHBoxLayout()
        self.sp_f0 = QDoubleSpinBox()
        self.sp_f0.setRange(1.0, 50.0)
        self.sp_f0.setSingleStep(1.0)
        self.sp_f0.setValue(20.0)
        r4.addWidget(QLabel("F0 percentile:"))
        r4.addWidget(self.sp_f0)
        r4.addStretch(1)
        b.addLayout(r4)

        r5 = QHBoxLayout()
        self.btn_run = QPushButton("Run Step 2")
        r5.addWidget(self.btn_run)
        r5.addStretch(1)

        self.btn_open_qc = QPushButton("Open QC folder")
        self.btn_open_qc.setEnabled(False)
        r5.addWidget(self.btn_open_qc)
        b.addLayout(r5)

        lay.addWidget(box)

        pbox = QGroupBox("Step 2 Results (QC Preview)")
        p = QHBoxLayout(pbox)
        self.prev_f0 = ImagePreview("F0 Map (64×64)")
        self.prev_snap = ImagePreview("dFF Snapshot")
        p.addWidget(self.prev_f0)
        p.addWidget(self.prev_snap)
        lay.addWidget(pbox)

        lay.addStretch(1)

        self.btn_open_qc.clicked.connect(self.open_qc)

    def pick_matrix(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select StepA matrix", "", "NPY Files (*.npy)")
        if p:
            self.in_matrix.setText(p)

    def pick_valid(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select StepA valid", "", "NPY Files (*.npy)")
        if p:
            self.in_valid.setText(p)

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir.setText(d)

    def open_qc(self):
        out = self.out_dir.text().strip()
        if not out:
            return
        qc_dir = Path(out) / "qc"
        if qc_dir.exists():
            os.startfile(str(qc_dir))


class StepCPage(QWidget):
    def __init__(self):
        super().__init__()
        lay = QVBoxLayout(self)

        box = QGroupBox("Step 3: Phase (bandpass + Hilbert) → Save")
        b = QVBoxLayout(box)

        r1 = QHBoxLayout()
        self.in_dff = QLineEdit()
        btn_d = QPushButton("Select StepB dff64.npy")
        btn_d.clicked.connect(self.pick_dff)
        r1.addWidget(QLabel("StepB dFF:"))
        r1.addWidget(self.in_dff)
        r1.addWidget(btn_d)
        b.addLayout(r1)

        r2 = QHBoxLayout()
        self.in_valid = QLineEdit()
        btn_v = QPushButton("Select StepA valid64.npy")
        btn_v.clicked.connect(self.pick_valid)
        r2.addWidget(QLabel("StepA valid:"))
        r2.addWidget(self.in_valid)
        r2.addWidget(btn_v)
        b.addLayout(r2)

        r3 = QHBoxLayout()
        self.out_dir = QLineEdit()
        btn_out = QPushButton("Select Output Folder")
        btn_out.clicked.connect(self.pick_outdir)
        r3.addWidget(QLabel("Output folder:"))
        r3.addWidget(self.out_dir)
        r3.addWidget(btn_out)
        b.addLayout(r3)

        r4 = QHBoxLayout()
        self.sp_fs = QDoubleSpinBox()
        self.sp_fs.setRange(0.01, 1_000_000.0)
        self.sp_fs.setValue(10.0)

        self.sp_flo = QDoubleSpinBox()
        self.sp_flo.setRange(0.0, 10_000.0)
        self.sp_flo.setValue(0.1)

        self.sp_fhi = QDoubleSpinBox()
        self.sp_fhi.setRange(0.01, 10_000.0)
        self.sp_fhi.setValue(3.0)

        r4.addWidget(QLabel("fs:"))
        r4.addWidget(self.sp_fs)
        r4.addSpacing(10)
        r4.addWidget(QLabel("f_lo:"))
        r4.addWidget(self.sp_flo)
        r4.addSpacing(10)
        r4.addWidget(QLabel("f_hi:"))
        r4.addWidget(self.sp_fhi)
        r4.addStretch(1)
        b.addLayout(r4)

        r5 = QHBoxLayout()
        self.sp_order = QSpinBox()
        self.sp_order.setRange(1, 10)
        self.sp_order.setValue(3)

        self.sp_dec = QSpinBox()
        self.sp_dec.setRange(1, 1000)
        self.sp_dec.setValue(1)

        self.cmb_fill = QComboBox()
        self.cmb_fill.addItems(["interp", "zero", "none"])

        self.sp_snap_t = QSpinBox()
        self.sp_snap_t.setRange(0, 10_000_000)
        self.sp_snap_t.setValue(0)

        r5.addWidget(QLabel("order:"))
        r5.addWidget(self.sp_order)
        r5.addSpacing(10)
        r5.addWidget(QLabel("time_decimate:"))
        r5.addWidget(self.sp_dec)
        r5.addSpacing(10)
        r5.addWidget(QLabel("fill_nan:"))
        r5.addWidget(self.cmb_fill)
        r5.addSpacing(10)
        r5.addWidget(QLabel("snapshot t:"))
        r5.addWidget(self.sp_snap_t)
        r5.addStretch(1)
        b.addLayout(r5)

        r6 = QHBoxLayout()
        self.btn_run = QPushButton("Run Step 3")
        r6.addWidget(self.btn_run)
        r6.addStretch(1)
        self.btn_open_qc = QPushButton("Open QC folder")
        self.btn_open_qc.setEnabled(False)
        r6.addWidget(self.btn_open_qc)
        b.addLayout(r6)

        lay.addWidget(box)

        pbox = QGroupBox("Step 3 Results (QC Preview)")
        p = QHBoxLayout(pbox)
        self.prev_phase = ImagePreview("Phase Snapshot (-π~π)")
        p.addWidget(self.prev_phase)
        lay.addWidget(pbox)

        lay.addStretch(1)

        self.btn_open_qc.clicked.connect(self.open_qc)

    def pick_dff(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select StepB dff", "", "NPY Files (*.npy)")
        if p:
            self.in_dff.setText(p)

    def pick_valid(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select StepA valid", "", "NPY Files (*.npy)")
        if p:
            self.in_valid.setText(p)

    def pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Select output folder")
        if d:
            self.out_dir.setText(d)

    def open_qc(self):
        out = self.out_dir.text().strip()
        if not out:
            return
        qc_dir = Path(out) / "qc"
        if qc_dir.exists():
            os.startfile(str(qc_dir))


# ----------------------------
# Main Wizard Window
# ----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CALTOOL Wizard (Continue-based) - Step-by-step")
        self.resize(1100, 820)

        self.thread: QThread | None = None
        self.worker = None

        self.input_tiff: Path | None = None
        self.out_dir: Path | None = None
        self.stem: str | None = None

        self.stepA_paths: dict = {}
        self.stepB_paths: dict = {}
        self.stepC_paths: dict = {}

        root = QWidget()
        self.setCentralWidget(root)
        lay = QVBoxLayout(root)

        self.lbl_progress = QLabel("Step 1 / 3")
        self.lbl_progress.setStyleSheet("font-size: 18px; font-weight: 600;")
        lay.addWidget(self.lbl_progress)

        self.stack = QStackedWidget()
        self.pageA = StepAPage()
        self.pageB = StepBPage()
        self.pageC = StepCPage()
        self.stack.addWidget(self.pageA)
        self.stack.addWidget(self.pageB)
        self.stack.addWidget(self.pageC)
        lay.addWidget(self.stack, stretch=1)

        lay.addWidget(QLabel("Log / Meta:"))
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        lay.addWidget(self.log, stretch=1)

        nav = QHBoxLayout()
        self.btn_back = QPushButton("Back")
        self.btn_continue = QPushButton("Continue")
        self.btn_back.clicked.connect(self.go_back)
        self.btn_continue.clicked.connect(self.go_continue)
        nav.addWidget(self.btn_back)
        nav.addStretch(1)
        nav.addWidget(self.btn_continue)
        lay.addLayout(nav)

        self.btn_back.setEnabled(False)
        self.btn_continue.setEnabled(False)

        self.pageA.btn_run.clicked.connect(self.run_stepA)
        self.pageB.btn_run.clicked.connect(self.run_stepB)
        self.pageC.btn_run.clicked.connect(self.run_stepC)

        self._update_ui()

    def append(self, s: str):
        self.log.append(s)

    def _set_busy(self, busy: bool):
        self.pageA.btn_run.setEnabled(not busy)
        self.pageB.btn_run.setEnabled(not busy)
        self.pageC.btn_run.setEnabled(not busy)
        self.btn_back.setEnabled((not busy) and self.stack.currentIndex() > 0)

    def _update_ui(self):
        idx = self.stack.currentIndex()
        self.lbl_progress.setText(f"Step {idx+1} / 3")
        self.btn_back.setEnabled(idx > 0)

        if idx == 0:
            self.btn_continue.setEnabled(bool(self.stepA_paths))
        elif idx == 1:
            self.btn_continue.setEnabled(bool(self.stepB_paths))
        else:
            self.btn_continue.setEnabled(bool(self.stepC_paths))

        self.pageA.btn_open_qc.setEnabled(bool(self.stepA_paths))
        self.pageB.btn_open_qc.setEnabled(bool(self.stepB_paths))
        self.pageC.btn_open_qc.setEnabled(bool(self.stepC_paths))

    def _start_thread(self, worker):
        if self.thread is not None and self.thread.isRunning():
            self.append("ERROR: Previous thread still running.")
            return

        self.thread = QThread(self)
        self.worker = worker
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.log.connect(self.append, Qt.QueuedConnection)

        def finish_ok(meta: dict):
            self.append("\n--- DONE ---")
            for k, v in meta.items():
                self.append(f"{k}: {v}")
            self._set_busy(False)
            self._update_ui()
            if self.thread and self.thread.isRunning():
                self.thread.quit()

        def finish_fail(err: str):
            self.append("\n--- FAILED ---")
            self.append(err)
            self._set_busy(False)
            self._update_ui()
            if self.thread and self.thread.isRunning():
                self.thread.quit()

        self.worker.done.connect(finish_ok, Qt.QueuedConnection)
        self.worker.failed.connect(finish_fail, Qt.QueuedConnection)

        def cleanup():
            try:
                if self.worker is not None:
                    self.worker.deleteLater()
            finally:
                self.worker = None
            if self.thread is not None:
                self.thread.deleteLater()
                self.thread = None

        self.thread.finished.connect(cleanup, Qt.QueuedConnection)

        self._set_busy(True)
        self.thread.start()

    def go_back(self):
        idx = self.stack.currentIndex()
        if idx > 0:
            self.stack.setCurrentIndex(idx - 1)
        self._update_ui()

    def go_continue(self):
        idx = self.stack.currentIndex()
        if idx == 0:
            if not self.stepA_paths:
                QMessageBox.information(self, "Not ready", "Run Step 1 and confirm results first.")
                return
            self.stack.setCurrentIndex(1)

        elif idx == 1:
            if not self.stepB_paths:
                QMessageBox.information(self, "Not ready", "Run Step 2 and confirm results first.")
                return
            self.stack.setCurrentIndex(2)

        else:
            if not self.stepC_paths:
                QMessageBox.information(self, "Not ready", "Run Step 3 and confirm results first.")
                return
            QMessageBox.information(self, "Completed", "Step 1~3 completed.\nNext: Kuramoto / CSD.")
        self._update_ui()

    # ----------------------------
    # Step 1
    # ----------------------------
    def run_stepA(self):
        inp = self.pageA.in_tiff.text().strip()
        out = self.pageA.out_dir.text().strip()

        if not inp or not Path(inp).exists():
            self.append("ERROR: Input TIFF missing.")
            return
        if not out:
            self.append("ERROR: Output folder missing.")
            return

        self.log.clear()
        self.append("Starting Step 1...")
        self.input_tiff = Path(inp)
        self.out_dir = Path(out)
        self.stem = self.input_tiff.stem

        self.stepA_paths = {}
        self.stepB_paths = {}
        self.stepC_paths = {}

        self.pageA.prev_overlay.set_image(None)
        self.pageA.prev_valid.set_image(None)
        self.pageB.prev_f0.set_image(None)
        self.pageB.prev_snap.set_image(None)
        self.pageC.prev_phase.set_image(None)

        w = StepAWorker(
            inp, out,
            do_motion=self.pageA.chk_motion.isChecked(),
            mask_mode=self.pageA.cmb_mask.currentText(),
            mask_p=float(self.pageA.sp_maskp.value()),
        )

        def on_done(meta: dict):
            stem = self.stem or Path(inp).stem
            out_dir = Path(out)

            self.stepA_paths = {
                "matrix": str(out_dir / f"{stem}.stepA.matrix64.npy"),
                "valid": str(out_dir / f"{stem}.stepA.valid64.npy"),
                "mask": str(out_dir / f"{stem}.stepA.mask.npy"),
                "qc_overlay": meta.get("qc_stepA_overlay"),
                "qc_valid64": meta.get("qc_stepA_valid64"),
            }

            self.pageA.prev_overlay.set_image(self.stepA_paths.get("qc_overlay"))
            self.pageA.prev_valid.set_image(self.stepA_paths.get("qc_valid64"))

            self.pageB.in_matrix.setText(self.stepA_paths["matrix"])
            self.pageB.in_valid.setText(self.stepA_paths["valid"])
            self.pageB.out_dir.setText(str(out_dir))

            self._update_ui()

        w.done.connect(on_done, Qt.QueuedConnection)
        self._start_thread(w)

    # ----------------------------
    # Step 2
    # ----------------------------
    def run_stepB(self):
        m = self.pageB.in_matrix.text().strip()
        v = self.pageB.in_valid.text().strip()
        out = self.pageB.out_dir.text().strip()

        if not m or not Path(m).exists():
            self.append("ERROR: StepA matrix file missing.")
            return
        if not v or not Path(v).exists():
            self.append("ERROR: StepA valid file missing.")
            return
        if not out:
            self.append("ERROR: Output folder missing.")
            return

        self.append("\nStarting Step 2...")

        self.stepB_paths = {}
        self.stepC_paths = {}
        self.pageB.prev_f0.set_image(None)
        self.pageB.prev_snap.set_image(None)
        self.pageC.prev_phase.set_image(None)

        w = StepBWorker(m, v, out, f0_percentile=float(self.pageB.sp_f0.value()))

        def on_done(meta: dict):
            self.stepB_paths = {
                "dff": meta.get("saved_dff"),
                "f0": meta.get("saved_f0"),
                "qc_f0": meta.get("qc_stepB_f0"),
                "qc_snap": meta.get("qc_stepB_snap"),
            }
            self.pageB.prev_f0.set_image(self.stepB_paths.get("qc_f0"))
            self.pageB.prev_snap.set_image(self.stepB_paths.get("qc_snap"))

            # StepC 자동 입력
            self.pageC.in_dff.setText(self.stepB_paths.get("dff") or "")
            self.pageC.in_valid.setText(self.stepA_paths.get("valid") or "")
            self.pageC.out_dir.setText(out)

            self._update_ui()

        w.done.connect(on_done, Qt.QueuedConnection)
        self._start_thread(w)

    # ----------------------------
    # Step 3
    # ----------------------------
    def run_stepC(self):
        dff = self.pageC.in_dff.text().strip()
        valid = self.pageC.in_valid.text().strip()
        out = self.pageC.out_dir.text().strip()

        if not dff or not Path(dff).exists():
            self.append("ERROR: StepB dff file missing.")
            return
        if not valid or not Path(valid).exists():
            self.append("ERROR: StepA valid file missing.")
            return
        if not out:
            self.append("ERROR: Output folder missing.")
            return

        self.append("\nStarting Step 3...")

        self.stepC_paths = {}
        self.pageC.prev_phase.set_image(None)

        w = StepCWorker(
            dff_path=dff,
            valid_path=valid,
            out_dir=out,
            fs=float(self.pageC.sp_fs.value()),
            f_lo=float(self.pageC.sp_flo.value()),
            f_hi=float(self.pageC.sp_fhi.value()),
            filter_order=int(self.pageC.sp_order.value()),
            time_decimate=int(self.pageC.sp_dec.value()),
            fill_nan=str(self.pageC.cmb_fill.currentText()),
            snapshot_t=int(self.pageC.sp_snap_t.value()),
        )

        def on_done(meta: dict):
            self.stepC_paths = {
                "phase": meta.get("saved_phase"),
                "qc_snap": meta.get("qc_stepC_snap"),
            }
            self.pageC.prev_phase.set_image(self.stepC_paths.get("qc_snap"))
            self._update_ui()

        w.done.connect(on_done, Qt.QueuedConnection)
        self._start_thread(w)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
