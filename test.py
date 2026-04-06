import numpy as np
from analyzer.io import load_tiff_stack
from analyzer.mask import projection_ref, build_mask_from_ref
from analyzer.gridder import grid_to_64

TIFF_PATH = r"D:\caltool\data\3_stack.tif"
WIN = 600

def iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter / (union + 1e-8))

def eval_window(frames_win: np.ndarray, mode: str, p: float = 98.0):
    ref = projection_ref(frames_win, mode=mode, percentile=p)
    m = build_mask_from_ref(ref, blur_ksize=5, closing_radius=3, min_object_size=500)
    mat = grid_to_64(frames_win, m, grid=64, fill_value=0.0)

    return {
        "mask": m,
        "mask_ratio": float(m.mean()),
        "zero_ratio": float((mat == 0).mean()),
        "mean": float(mat.mean()),
        "max": float(mat.max()),
    }

def main():
    frames = load_tiff_stack(TIFF_PATH)
    T = frames.shape[0]
    print("Total frames:", frames.shape)

    # 대표 구간 선택: start / middle / end + 2 random
    rng = np.random.default_rng(42)
    starts = [0, max(0, T//2 - WIN//2), max(0, T - WIN)]
    if T > WIN:
        starts += list(rng.integers(0, T - WIN, size=2))
    starts = [int(s) for s in starts]
    print("Window starts:", starts, "WIN=", WIN)

    configs = [("mean", None), ("percentile", 95.0), ("percentile", 98.0), ("percentile", 99.0), ("max", None)]

    for mode, p in configs:
        print("\n=== MODE:", mode, ("p="+str(p) if p is not None else ""), "===")
        masks = []
        mr_list, zr_list = [], []

        for s in starts:
            win = frames[s:s+WIN] if (s+WIN <= T) else frames[s:T]
            out = eval_window(win, mode, p if p is not None else 98.0)
            masks.append(out["mask"])
            mr_list.append(out["mask_ratio"])
            zr_list.append(out["zero_ratio"])
            print(f"  window[{s}:{s+len(win)}] mask_ratio={out['mask_ratio']:.3f}  zero_ratio={out['zero_ratio']:.3f}  mean={out['mean']:.1f}")

        # 마스크 일관성(IoU): 첫 구간 기준으로 비교
        base = masks[0]
        ious = [iou(base, m) for m in masks[1:]]
        print("  mask_ratio avg/std:", float(np.mean(mr_list)), float(np.std(mr_list)))
        print("  zero_ratio avg/std:", float(np.mean(zr_list)), float(np.std(zr_list)))
        if ious:
            print("  mask IoU vs first window avg/min:", float(np.mean(ious)), float(np.min(ious)))

if __name__ == "__main__":
    main()
