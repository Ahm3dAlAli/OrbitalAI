#!/usr/bin/env python3
"""Generate a multi-sensor SAMPLE-DETECTIONS contact sheet for the docs/proposal.

For each test sequence it picks the best-IoU windows, crops a zoomed region around
the detected object (so the RSO is clearly visible, not a dot in a 1280x720 frame),
overlays the predicted box (yellow + confidence) and GT box (green), and arranges
one row per sensor into a single labeled contact sheet.

Reuses the rendering primitives in scripts/visualize.py.

    python3 scripts/make_samples.py \
        --data-dir OrbitSight_Dataset/Testing_sets \
        --pred-dir predictions/testing_router2 \
        --gt-dir  OrbitSight_Dataset/Testing_sets \
        --out docs/figures/sample_detections.png
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from visualize import read_boxes, iou, event_image           # reuse primitives

TEST = [
    ("2025_12_23_20_53_46_EVK4_mag7.3", "EVK4 · mag7.3"),
    ("DAVIS_SAOCOM1B_46265_2024-12-04-18-21-37", "DAVIS · SAOCOM1B"),
    ("DVX_Filtered_Stars3_2025-01-20-20-22-53", "DVX · Stars3"),
    ("DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43", "DVX · Thuraya3"),
]
CELL = (300, 210)          # contact-sheet cell size (px)
LEFT = 96                  # left label column width
CYAN, GT_G, PRED_Y = (46, 196, 182), (0, 255, 0), (255, 255, 0)


def sample_cell(ev, win, W, H, pred, gt, cell=CELL, pad=2.4, min_half=55):
    """Zoomed, box-overlaid crop around the object for one window -> PIL cell."""
    lo, hi = win.lo, win.hi
    img = event_image(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], W, H)   # HxW uint8
    p, g = pred.get(win.start_us), gt.get(win.start_us)
    ref = p or g
    if ref is None:
        return None, 0.0
    cx, cy, bw, bh = ref[:4]
    half = max(max(bw, bh) * pad, min_half)
    x0 = int(np.clip(cx - half, 0, W - 1)); x1 = int(np.clip(cx + half, 1, W))
    y0 = int(np.clip(cy - half, 0, H - 1)); y1 = int(np.clip(cy + half, 1, H))
    crop = img[y0:y1, x0:x1]
    if crop.size == 0:
        return None, 0.0
    ch, cw = crop.shape[:2]
    s = min(cell[0] / cw, cell[1] / ch)
    im = Image.fromarray(crop).resize((max(int(cw * s), 1), max(int(ch * s), 1)),
                                      Image.NEAREST)
    cell_im = Image.new("RGB", cell, (6, 10, 20))
    ox, oy = (cell[0] - im.width) // 2, (cell[1] - im.height) // 2
    cell_im.paste(im, (ox, oy))
    d = ImageDraw.Draw(cell_im)

    def box(bx, color):
        if bx is None:
            return
        X, Y, Wd, Hd = bx[:4]
        rx0 = (X - Wd / 2 - x0) * s + ox; ry0 = (Y - Hd / 2 - y0) * s + oy
        rx1 = (X + Wd / 2 - x0) * s + ox; ry1 = (Y + Hd / 2 - y0) * s + oy
        d.rectangle([rx0, ry0, rx1, ry1], outline=color, width=2)

    box(g, GT_G)
    box(p, PRED_Y)
    ov = iou(p[:4] if p else None, g[:4] if g else None)
    conf = f"  p={p[4]:.2f}" if p else ""
    d.text((4, 3), f"IoU {ov:.2f}{conf}", fill=(255, 255, 255))
    return cell_im, ov


def build(data_dir, pred_dir, gt_dir, sequences, per, out):
    cfg = DEFAULT_CONFIG
    rows = []
    for seq, label in sequences:
        p = D.find_event_file(data_dir, seq)
        if not p:
            print(f"[skip] no events for {seq}"); continue
        ev = D.Events.from_npy(p)
        sn = sensor_for_sequence(seq)
        wins = D.make_window_grid(ev.t, cfg.window_us)
        pred = read_boxes(os.path.join(pred_dir, seq + D.GT_SUFFIX))
        gt = read_boxes(os.path.join(gt_dir, seq + D.GT_SUFFIX))
        # rank windows that have BOTH a pred and GT by IoU (true detections)
        scored = []
        for w in wins:
            if w.start_us in pred and w.start_us in gt:
                scored.append((iou(pred[w.start_us][:4], gt[w.start_us][:4]), w))
        scored.sort(key=lambda t: t[0], reverse=True)
        cells = []
        for _, w in scored[:per]:
            c, ov = sample_cell(ev, w, sn.width, sn.height, pred, gt)
            if c is not None:
                cells.append(c)
        if cells:
            rows.append((label, cells))
        del ev
        print(f"[row] {label}: {len(cells)} samples")

    if not rows:
        print("[ERR] no rows built"); return
    cols = max(len(c) for _, c in rows)
    cw, chh = CELL
    W = LEFT + cols * cw
    H = len(rows) * chh
    sheet = Image.new("RGB", (W, H), (12, 18, 30))
    d = ImageDraw.Draw(sheet)
    for r, (label, cells) in enumerate(rows):
        y = r * chh
        d.rectangle([0, y, LEFT - 1, y + chh - 1], fill=(13, 27, 42))
        # vertical-ish label (short lines)
        for i, part in enumerate(label.split(" · ")):
            d.text((8, y + 14 + i * 16), part, fill=CYAN)
        for cnum, cell in enumerate(cells):
            sheet.paste(cell, (LEFT + cnum * cw, y))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    sheet.save(out)
    print(f"[samples] {out}  {W}x{H}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--per", type=int, default=4, help="samples per sensor row")
    ap.add_argument("--out", default="docs/figures/sample_detections.png")
    args = ap.parse_args()
    build(args.data_dir, args.pred_dir, args.gt_dir, TEST, args.per, args.out)


if __name__ == "__main__":
    main()
