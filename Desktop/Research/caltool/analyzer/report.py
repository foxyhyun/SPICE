from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

try:
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen import canvas
    _REPORTLAB_OK = True
    _REPORTLAB_ERR = None
except Exception as e:
    letter = None
    ImageReader = None
    canvas = None
    _REPORTLAB_OK = False
    _REPORTLAB_ERR = e


@dataclass
class ReportResult:
    pdf_path: Path
    dashboard_png: Path
    summary_json: Path
    found_images: List[Tuple[str, Path]]
    missing_images: List[str]
    metrics: Dict[str, Any]


def _safe_read_img(p: Path):
    try:
        return plt.imread(str(p))
    except Exception:
        return None


def _finite_stats(x: np.ndarray) -> Dict[str, float | None]:
    x = np.asarray(x)
    xf = x[np.isfinite(x)]
    if xf.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(np.min(xf)),
        "max": float(np.max(xf)),
        "mean": float(np.mean(xf)),
        "std": float(np.std(xf)),
    }


def _safe_np_load(path: Path):
    if not path.exists():
        return None
    try:
        return np.load(str(path))
    except Exception:
        return None


def collect_report_metrics(out_dir: str | Path, stem: str) -> Dict[str, Any]:
    out_dir = Path(out_dir)

    metrics: Dict[str, Any] = {
        "stem": stem,
        "output_dir": str(out_dir),
    }

    # StepD
    r = _safe_np_load(out_dir / f"{stem}.stepD.R.npy")
    if r is not None:
        st = _finite_stats(r)
        metrics["kuramoto_R_mean"] = st["mean"]
        metrics["kuramoto_R_std"] = st["std"]
        metrics["kuramoto_R_min"] = st["min"]
        metrics["kuramoto_R_max"] = st["max"]

    phi_vel = _safe_np_load(out_dir / f"{stem}.stepD.Phi_vel.npy")
    if phi_vel is not None:
        xv = np.asarray(phi_vel)
        xf = xv[np.isfinite(xv)]
        if xf.size > 0:
            metrics["phi_vel_mean"] = float(np.mean(xf))
            metrics["phi_vel_std"] = float(np.std(xf))
            metrics["phi_vel_mean_abs"] = float(np.mean(np.abs(xf)))

    local = _safe_np_load(out_dir / f"{stem}.stepD.local_coh64.npy")
    if local is not None:
        st = _finite_stats(local)
        metrics["local_coh_mean"] = st["mean"]
        metrics["local_coh_std"] = st["std"]
        metrics["local_coh_min"] = st["min"]
        metrics["local_coh_max"] = st["max"]

    # StepE
    csd = _safe_np_load(out_dir / f"{stem}.stepE.csd64.npy")
    if csd is not None:
        xf = np.asarray(csd)
        ff = xf[np.isfinite(xf)]
        if ff.size > 0:
            metrics["csd_mean"] = float(np.mean(ff))
            metrics["csd_std"] = float(np.std(ff))
            metrics["csd_mean_abs"] = float(np.mean(np.abs(ff)))
            metrics["csd_max_abs"] = float(np.max(np.abs(ff)))

    activity = _safe_np_load(out_dir / f"{stem}.stepE.activity64.npy")
    if activity is not None:
        st = _finite_stats(activity)
        metrics["activity_mean"] = st["mean"]
        metrics["activity_std"] = st["std"]
        metrics["activity_min"] = st["min"]
        metrics["activity_max"] = st["max"]

    # StepB
    dff = _safe_np_load(out_dir / f"{stem}.stepB.dff64.npy")
    if dff is not None:
        xf = np.asarray(dff)
        ff = xf[np.isfinite(xf)]
        if ff.size > 0:
            metrics["dff_mean"] = float(np.mean(ff))
            metrics["dff_std"] = float(np.std(ff))
            metrics["dff_min"] = float(np.min(ff))
            metrics["dff_max"] = float(np.max(ff))

    return metrics


def build_dashboard_png(
    title: str,
    items: List[Tuple[str, Path]],
    out_png: Path,
) -> Path:
    out_png.parent.mkdir(parents=True, exist_ok=True)

    imgs = []
    labels = []
    for lab, p in items:
        im = _safe_read_img(p)
        if im is None:
            continue
        imgs.append(im)
        labels.append(lab)

    if not imgs:
        plt.figure(figsize=(10, 6), dpi=200)
        plt.text(0.5, 0.5, "No QC images found.", ha="center", va="center", fontsize=14)
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(out_png, dpi=200)
        plt.close()
        return out_png

    n = len(imgs)
    ncols = 2 if n >= 2 else 1
    nrows = int(np.ceil(n / ncols))

    plt.figure(figsize=(13, 3.8 * nrows), dpi=200)
    plt.suptitle(title, fontsize=15, y=0.995)

    for i, (im, lab) in enumerate(zip(imgs, labels), start=1):
        ax = plt.subplot(nrows, ncols, i)
        ax.imshow(im)
        ax.set_title(lab, fontsize=10)
        ax.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.985])
    plt.savefig(out_png, dpi=200)
    plt.close()
    return out_png


def build_pdf_report(
    title: str,
    items: List[Tuple[str, Path]],
    out_pdf: Path,
    summary_lines: List[str] | None = None,
) -> Path:
    if not _REPORTLAB_OK:
        raise ImportError(
            f"reportlab is required to create the PDF report. Original error: {_REPORTLAB_ERR}"
        )

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    c = canvas.Canvas(str(out_pdf), pagesize=letter)
    page_w, page_h = letter

    # Cover page
    x0 = 54
    y = page_h - 60

    c.setFont("Helvetica-Bold", 18)
    c.drawString(x0, y, title)
    y -= 28

    c.setFont("Helvetica", 10)
    if summary_lines:
        for line in summary_lines:
            if y < 60:
                c.showPage()
                y = page_h - 60
                c.setFont("Helvetica", 10)
            c.drawString(x0, y, line[:140])
            y -= 13

    c.showPage()

    # Image pages: 2 per page
    margin = 48
    gap = 18
    label_h = 14
    slot_h = (page_h - 2 * margin - gap) / 2
    box_w = page_w - 2 * margin
    img_box_h = slot_h - label_h - 8

    def _draw_one(img_path: Path, label: str, top_y: float):
        c.setFont("Helvetica-Bold", 11)
        c.drawString(margin, top_y - 2, label[:120])

        reader = ImageReader(str(img_path))
        iw, ih = reader.getSize()
        scale = min(box_w / iw, img_box_h / ih)

        draw_w = iw * scale
        draw_h = ih * scale

        draw_x = margin + (box_w - draw_w) / 2
        draw_y = top_y - label_h - 6 - draw_h - (img_box_h - draw_h) / 2

        c.drawImage(
            str(img_path),
            draw_x,
            draw_y,
            width=draw_w,
            height=draw_h,
            preserveAspectRatio=True,
            mask='auto',
        )

    page_items: List[Tuple[str, Path]] = []
    for it in items:
        page_items.append(it)
        if len(page_items) == 2:
            (lab1, p1), (lab2, p2) = page_items
            _draw_one(p1, lab1, page_h - margin)
            _draw_one(p2, lab2, page_h - margin - slot_h - gap)
            c.showPage()
            page_items = []

    if page_items:
        lab1, p1 = page_items[0]
        _draw_one(p1, lab1, page_h - margin)
        c.showPage()

    c.save()
    return out_pdf


def make_spice_report(out_dir: str | Path, stem: str) -> ReportResult:
    out_dir = Path(out_dir)
    qc_dir = out_dir / "qc"
    rep_dir = out_dir / "reports"
    rep_dir.mkdir(parents=True, exist_ok=True)

    candidates = [
        ("StepA mask overlay", qc_dir / f"{stem}.stepA_mask_overlay.png"),
        ("StepA valid grid", qc_dir / f"{stem}.stepA_valid64.png"),
        ("StepB F0 map", qc_dir / f"{stem}.stepB_f0_map.png"),
        ("StepB dFF snapshot", qc_dir / f"{stem}.stepB_dff_snapshot.png"),
        ("StepB dFF traces", qc_dir / f"{stem}.stepB_dff_traces.png"),
        ("StepC phase snapshot", qc_dir / f"{stem}.stepC_phase_snapshot.png"),
        ("StepD Kuramoto timeseries", qc_dir / f"{stem}.stepD_kuramoto_timeseries.png"),
        ("StepD local coherence", qc_dir / f"{stem}.stepD_local_coh_map.png"),
        ("StepE pseudo-CSD snapshot", qc_dir / f"{stem}.stepE_csd_snapshot.png"),
        ("StepE activity map", qc_dir / f"{stem}.stepE_activity_map.png"),
    ]

    found: List[Tuple[str, Path]] = []
    missing: List[str] = []

    for lab, p in candidates:
        if p.exists():
            found.append((lab, p))
        else:
            missing.append(f"{lab}: {p.name}")

    metrics = collect_report_metrics(out_dir, stem)

    dashboard_png = rep_dir / f"{stem}.SPICE_dashboard.png"
    pdf_path = rep_dir / f"{stem}.SPICE_report.pdf"
    summary_json = rep_dir / f"{stem}.SPICE_summary.json"

    title = f"SPICE Integrated Report - {stem}"

    summary_lines = [
        f"Stem: {stem}",
        f"Output folder: {str(out_dir)}",
        f"QC folder: {str(qc_dir)}",
        f"Number of included figures: {len(found)}",
        f"Number of missing figures: {len(missing)}",
        "",
        "Key metrics:",
    ]

    metric_order = [
        "kuramoto_R_mean", "kuramoto_R_std",
        "phi_vel_mean_abs",
        "local_coh_mean",
        "activity_mean", "activity_max",
        "csd_mean_abs", "csd_max_abs",
        "dff_mean", "dff_std",
    ]
    for k in metric_order:
        if k in metrics and metrics[k] is not None:
            summary_lines.append(f"- {k}: {metrics[k]:.6g}")

    summary_lines.append("")
    summary_lines.append("Included figures:")
    for lab, p in found:
        summary_lines.append(f"- {lab}: {p.name}")

    if missing:
        summary_lines.append("")
        summary_lines.append("Missing figures:")
        for m in missing:
            summary_lines.append(f"- {m}")

    build_dashboard_png(title=title, items=found, out_png=dashboard_png)
    build_pdf_report(title=title, items=found, out_pdf=pdf_path, summary_lines=summary_lines)

    summary_obj = {
        "stem": stem,
        "output_dir": str(out_dir),
        "qc_dir": str(qc_dir),
        "report_dir": str(rep_dir),
        "dashboard_png": str(dashboard_png),
        "pdf_path": str(pdf_path),
        "included_images": [{"label": lab, "path": str(p)} for lab, p in found],
        "missing_images": missing,
        "metrics": metrics,
    }
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, ensure_ascii=False, indent=2)

    return ReportResult(
        pdf_path=pdf_path,
        dashboard_png=dashboard_png,
        summary_json=summary_json,
        found_images=found,
        missing_images=missing,
        metrics=metrics,
    )