from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyzer.pipeline import (
    run_stepA,
    run_stepB_dff,
    run_stepC_phase,
    run_stepD_kuramoto,
    run_stepE_csd,
    run_stepF_report,
)


def _print_meta(meta: dict):
    print("__META__" + json.dumps(meta, ensure_ascii=False), flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=["A", "B", "C", "D", "E", "F"])
    ap.add_argument("--out_dir", required=True)

    # StepA
    ap.add_argument("--tiff", default=None)
    ap.add_argument("--do_motion", action="store_true")
    ap.add_argument("--mask_mode", default="mean", choices=["mean", "percentile", "max"])
    ap.add_argument("--mask_p", type=float, default=98.0)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--min_coverage", type=float, default=0.0)

    # StepB
    ap.add_argument("--matrix", default=None)
    ap.add_argument("--valid", default=None)
    ap.add_argument("--f0_percentile", type=float, default=20.0)

    # StepC
    ap.add_argument("--dff", default=None)
    ap.add_argument("--fs", type=float, default=10.0)
    ap.add_argument("--f_lo", type=float, default=0.1)
    ap.add_argument("--f_hi", type=float, default=3.0)
    ap.add_argument("--filter_order", type=int, default=3)
    ap.add_argument("--time_decimate", type=int, default=1)
    ap.add_argument("--fill_nan", default="interp", choices=["interp", "zero", "none"])
    ap.add_argument("--snapshot_t", type=int, default=-1)

    # StepD
    ap.add_argument("--phase", default=None)
    ap.add_argument("--fs_eff", type=float, default=None)
    ap.add_argument("--local_coh", action="store_true")

    # StepE
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--dx", type=float, default=1.0)
    ap.add_argument("--smooth_sigma", type=float, default=1.0)
    ap.add_argument("--activity_mode", default="mean_abs", choices=["mean_abs", "rms", "std"])
    ap.add_argument("--csd_snapshot_t", type=int, default=-1)

    # StepF
    ap.add_argument("--stem", default=None)

    args = ap.parse_args()
    out_dir = Path(args.out_dir)

    try:
        if args.step == "A":
            if not args.tiff:
                raise ValueError("StepA requires --tiff")

            res = run_stepA(
                args.tiff,
                out_dir,
                grid=int(args.grid),
                do_motion=bool(args.do_motion),
                mask_mode=str(args.mask_mode),
                mask_percentile=float(args.mask_p),
                min_coverage=float(args.min_coverage),
            )
            meta = res.meta

        elif args.step == "B":
            if not args.matrix or not args.valid:
                raise ValueError("StepB requires --matrix and --valid")

            meta = run_stepB_dff(
                args.matrix,
                args.valid,
                out_dir,
                f0_percentile=float(args.f0_percentile),
            )

        elif args.step == "C":
            if not args.dff or not args.valid:
                raise ValueError("StepC requires --dff and --valid")

            snap = None if args.snapshot_t < 0 else int(args.snapshot_t)

            meta = run_stepC_phase(
                args.dff,
                args.valid,
                out_dir,
                fs=float(args.fs),
                f_lo=float(args.f_lo),
                f_hi=float(args.f_hi),
                filter_order=int(args.filter_order),
                time_decimate=int(args.time_decimate),
                fill_nan=str(args.fill_nan),
                snapshot_t=snap,
            )

        elif args.step == "D":
            if not args.phase or not args.valid:
                raise ValueError("StepD requires --phase and --valid")

            fs_eff = args.fs_eff
            if fs_eff is None:
                fs_eff = float(args.fs) / max(1, int(args.time_decimate))

            meta = run_stepD_kuramoto(
                args.phase,
                args.valid,
                out_dir,
                fs_eff=float(fs_eff),
                compute_local_coherence=bool(args.local_coh),
            )

        elif args.step == "E":
            if not args.dff or not args.valid:
                raise ValueError("StepE requires --dff and --valid")

            snap = None if args.csd_snapshot_t < 0 else int(args.csd_snapshot_t)

            meta = run_stepE_csd(
                args.dff,
                args.valid,
                out_dir,
                sigma=float(args.sigma),
                dx=float(args.dx),
                smooth_sigma=float(args.smooth_sigma),
                activity_mode=str(args.activity_mode),
                snapshot_t=snap,
            )

        else:  # Step F
            if not args.stem:
                raise ValueError("StepF requires --stem")

            meta = run_stepF_report(
                out_dir=out_dir,
                stem=str(args.stem),
            )

        _print_meta(meta)
        return 0

    except Exception as e:
        _print_meta({"error": str(e)})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())