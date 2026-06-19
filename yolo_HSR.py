import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from ultralytics import YOLO


# Supported image extensions
IMG_EXT = {".ppm", ".png", ".jpg", ".jpeg", ".PPM", ".PNG", ".JPG", ".JPEG"}


def list_frames(Pdir: Path):
    files = [p for p in Pdir.rglob("*") if p.suffix in IMG_EXT]

    def natkey(p):
        s = p.stem
        digits = "".join(c for c in s if c.isdigit())
        return (p.parent.as_posix(), len(digits), digits.zfill(16), s)

    files.sort(key=natkey)
    return files


def assign_HSR(dets):
    best = {
        0: {"conf": -1, "pt": None},  # H
        1: {"conf": -1, "pt": None},  # R
        2: {"conf": -1, "pt": None},  # S
    }

    for cls, conf, (x1, y1, x2, y2) in dets:
        if cls not in best:
            continue

        if conf > best[cls]["conf"]:
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            best[cls] = {"conf": conf, "pt": (cx, cy)}

    H = best[0]["pt"]
    R = best[1]["pt"]
    S = best[2]["pt"]

    return H, S, R


def process_folder(Pdir, model, out_csv_dir, out_overlay_dir,
                     conf_thr=0.25, overlay_every=2000):
    print(f"\nProcessing folder: {Pdir.name}")
    frames = list_frames(Pdir)

    out_csv_dir.mkdir(parents=True, exist_ok=True)
    out_overlay_dir.mkdir(parents=True, exist_ok=True)

    out_csv = out_csv_dir / f"{Pdir.name}_HSR_rawPX.csv"
    overlay_sub = out_overlay_dir / Pdir.name
    overlay_sub.mkdir(parents=True, exist_ok=True)

    rows = []

    # Previous valid detections (used if current frame misses detection)
    prev_H = prev_S = prev_R = (None, None)

    for i, fp in enumerate(tqdm(frames, desc=f"{Pdir.name}")):
        img = cv2.imread(str(fp))
        if img is None:
            rows.append([i, None, None, None, None, None, None])
            continue

        # Run YOLO inference
        r = model.predict(source=img, conf=conf_thr, verbose=False)[0]

        dets = []
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            classes = r.boxes.cls.cpu().numpy().astype(int)

            for (x1, y1, x2, y2), cf, cl in zip(xyxy, confs, classes):
                dets.append((cl, float(cf), (x1, y1, x2, y2)))

        H, S, R = assign_HSR(dets)

        # Temporal persistence:
        # If a marker is not detected in the current frame,
        # reuse its last valid position.
        if H is None: H = prev_H
        if S is None: S = prev_S
        if R is None: R = prev_R

        rows.append([i,
                     H[0] if H else None, H[1] if H else None,
                     S[0] if S else None, S[1] if S else None,
                     R[0] if R else None, R[1] if R else None])

        prev_H, prev_S, prev_R = H, S, R

        # Save overlay image every N frames for qualitative inspection
        if i % overlay_every == 0:
            vis = img.copy()
            for (pt, col) in zip([H, S, R], [(0,0,255),(0,255,0),(255,0,0)]):
                if pt is not None:
                    cv2.circle(vis, (int(pt[0]), int(pt[1])), 6, col, 2)
            cv2.imwrite(str(overlay_sub / f"{i:06d}.png"), vis)

    # Save CSV
    df = pd.DataFrame(rows, columns=[
        "frame", "H_x_px", "H_y_px",
        "S_x_px", "S_y_px",
        "R_x_px", "R_y_px"
    ])
    df.to_csv(out_csv, index=False)
    print(f"[OK] CSV saved: {out_csv}")


def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="Input image folder")
    ap.add_argument("--model", required=True, help="YOLO model weights path")
    ap.add_argument("--out_csv", required=True, help="Output CSV directory")
    ap.add_argument("--out_overlay", required=True, help="Overlay image directory")
    ap.add_argument("--exclude", nargs="*", default=[], help="Folders to skip")
    ap.add_argument("--conf", type=float, default=0.25,
                    help="Confidence threshold for detection")
    args = ap.parse_args()

    root = Path(args.root)
    model = YOLO(args.model)

    folders = [root]

    for P in folders:
        if P.name in args.exclude:
            print(f"Skipping {P.name}")
            continue

        process_cartella(P, model,
                         Path(args.out_csv),
                         Path(args.out_overlay),
                         conf_thr=args.conf)

    print("\nCompleted!")


if __name__ == "__main__":
    main()
