#!/usr/bin/env python3
"""FR-9 visualization tool (model-agnostic).

Renders detection results for a sequence.  It reads boxes from a *prediction
directory* (any model — classical, CenterNet, ensemble, or the final router
output — writes the same tab-separated format), so it visualizes whatever our
best pipeline produced without re-running it.

Outputs:
  * animation (--out .gif / .mp4): per-window event image (positive polarity
    cyan, negative magenta) with the predicted box (yellow + confidence) and,
    if a GT dir is given, the ground-truth box (green) and per-window IoU;
  * failure gallery (--gallery): a PNG montage of the WORST windows (missed GT,
    false positives, low IoU) — the "failure cases" the docs criterion rewards;
  * (x, y, t) scatter (--xyt): RSO vs background 3-D coherence plot (Plotly
    HTML) — the Hypothesis-H1 validation figure.

Dependencies: numpy + Pillow.  imageio optional (MP4); plotly optional (--xyt).

    # animate the final router output with GT overlay + IoU:
    python3 scripts/visualize.py --data-dir OrbitSight_Dataset/Testing_sets \
        --seq DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43 \
        --pred-dir predictions/router_ctta --gt-dir OrbitSight_Dataset/Testing_sets \
        --out docs/vis/thuraya3.gif

    # failure gallery (worst 12 windows):
    python3 scripts/visualize.py --data-dir ... --seq ... --pred-dir ... \
        --gt-dir ... --gallery --out docs/vis/thuraya3_failures.png
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence

try:
    from PIL import Image, ImageDraw
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# --------------------------------------------------------------------------- #
#  Box I/O + geometry
# --------------------------------------------------------------------------- #
def read_boxes(path):
    """Read a prediction/GT file -> {window_start_us: (cx, cy, w, h, conf)}."""
    out = {}
    if not path or not os.path.exists(path):
        return out
    with open(path, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            ws = int(r["window_start_timestamp_us"])
            out[ws] = (float(r["center_x"]), float(r["center_y"]),
                       float(r["width"]), float(r["height"]),
                       float(r.get("confidence", 1.0)))
    return out


def iou(a, b):
    if a is None or b is None:
        return 0.0
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    ua = (ax1-ax0)*(ay1-ay0) + (bx1-bx0)*(by1-by0) - inter
    return inter / ua if ua > 0 else 0.0


# --------------------------------------------------------------------------- #
#  Rendering
# --------------------------------------------------------------------------- #
def event_image(x, y, pol, W, H):
    """HxW RGB, log-scaled: positive polarity cyan, negative magenta on black."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    if x.size:
        xi = np.clip(x.astype(np.int32), 0, W - 1)
        yi = np.clip(y.astype(np.int32), 0, H - 1)
        pos = pol >= 0.5
        cp = np.zeros((H, W), np.float32); np.add.at(cp, (yi[pos], xi[pos]), 1.0)
        cn = np.zeros((H, W), np.float32); np.add.at(cn, (yi[~pos], xi[~pos]), 1.0)
        cp, cn = np.log1p(cp), np.log1p(cn)
        m = max(cp.max(), cn.max(), 1e-6)
        cp, cn = cp / m, cn / m
        img[..., 1] += cp; img[..., 2] += cp            # cyan   (G+B)
        img[..., 0] += cn; img[..., 2] += cn            # magenta(R+B)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _box_xyxy(box, scale):
    cx, cy, w, h = box[0]*scale, box[1]*scale, box[2]*scale, box[3]*scale
    return [cx - w/2, cy - h/2, cx + w/2, cy + h/2]


def render_frame(ev, win, W, H, pred, gt, scale):
    base = event_image(ev.x[win.lo:win.hi], ev.y[win.lo:win.hi],
                        ev.pol[win.lo:win.hi], W, H)
    im = Image.fromarray(base)
    if scale != 1.0:
        im = im.resize((int(W*scale), int(H*scale)), Image.NEAREST)
    d = ImageDraw.Draw(im)
    g = gt.get(win.start_us)
    p = pred.get(win.start_us)
    if g:
        d.rectangle(_box_xyxy(g, scale), outline=(0, 255, 0), width=2)
    if p:
        d.rectangle(_box_xyxy(p, scale), outline=(255, 255, 0), width=2)
        d.text((_box_xyxy(p, scale)[0], _box_xyxy(p, scale)[1] - 11),
               f"{p[4]:.2f}", fill=(255, 255, 0))
    ov = iou(p[:4] if p else None, g[:4] if g else None)
    tag = f"win {win.index}  t={win.start_us/1e6:.2f}s  n={win.hi-win.lo}"
    if g or p:
        tag += f"  IoU={ov:.2f}"
    d.text((4, 4), tag, fill=(255, 255, 255))
    return im, ov, (p is not None), (g is not None)


def save_animation(frames, out, fps):
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    if out.lower().endswith(".mp4"):
        try:
            import imageio.v2 as imageio
            imageio.mimsave(out, [np.asarray(f) for f in frames], fps=fps)
            return out
        except Exception as e:
            out = out[:-4] + ".gif"
            print(f"[WARN] mp4 failed ({e}); wrote GIF instead")
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000/max(fps, 1)), loop=0, optimize=True)
    return out


def save_gallery(items, out, cols=4):
    if not items:
        print("[WARN] nothing to put in gallery"); return None
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    tw = max(im.width for im, _ in items)
    th = max(im.height for im, _ in items)
    rows = (len(items) + cols - 1) // cols
    canvas = Image.new("RGB", (cols*tw, rows*th), (20, 20, 20))
    d = ImageDraw.Draw(canvas)
    for i, (im, cap) in enumerate(items):
        r, cc = divmod(i, cols)
        canvas.paste(im, (cc*tw, r*th))
        d.text((cc*tw + 4, r*th + th - 12), cap, fill=(255, 220, 0))
    canvas.save(out)
    return out


def render_xyt(ev, seq, out, max_pts=40000):
    """(x, y, t) scatter: RSO (label==1) vs background — the H1 figure."""
    try:
        import plotly.graph_objects as go
    except Exception:
        print("[viz] plotly unavailable, skipping --xyt"); return
    t = (ev.t - ev.t[0]) / 1e6
    rng = np.random.default_rng(0)
    def sub(mask):
        idx = np.nonzero(mask)[0]
        return rng.choice(idx, max_pts, replace=False) if idx.size > max_pts else idx
    bg, rso = sub(ev.label != 1), sub(ev.label == 1)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(x=ev.x[bg], y=ev.y[bg], z=t[bg], mode="markers",
        marker=dict(size=1, color="lightgray", opacity=0.3), name="background"))
    fig.add_trace(go.Scatter3d(x=ev.x[rso], y=ev.y[rso], z=t[rso], mode="markers",
        marker=dict(size=2, color="crimson"), name="RSO (label=1)"))
    fig.update_layout(title=f"{seq} — (x, y, t) coherence [H1]",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="t (s)"))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    fig.write_html(out)
    print(f"[viz] (x,y,t) plot -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--pred-dir", default=None)
    ap.add_argument("--gt-dir", default=None)
    ap.add_argument("--out", required=True, help=".gif/.mp4 (anim), .png (gallery), .html (xyt)")
    ap.add_argument("--scale", type=float, default=1.0, help="resize factor (EVK4: use 0.5)")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--max-frames", type=int, default=300)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--gallery", action="store_true",
                    help="montage of the WORST windows (FN/FP/low-IoU)")
    ap.add_argument("--best", action="store_true",
                    help="with --gallery: montage of the BEST windows (high IoU)")
    ap.add_argument("--gallery-n", type=int, default=12)
    ap.add_argument("--xyt", action="store_true", help="also write the (x,y,t) H1 plot")
    args = ap.parse_args()

    if not _HAS_PIL:
        sys.exit("[viz] Pillow required: pip install pillow")
    sn = sensor_for_sequence(args.seq)
    p = D.find_event_file(args.data_dir, args.seq)
    if not p:
        sys.exit(f"[ERR] no event file for {args.seq} in {args.data_dir}")
    ev = D.Events.from_npy(p)
    W, H = sn.width, sn.height
    wins = D.make_window_grid(ev.t, DEFAULT_CONFIG.window_us)
    pred = read_boxes(os.path.join(args.pred_dir, args.seq + D.GT_SUFFIX)) if args.pred_dir else {}
    gt = read_boxes(os.path.join(args.gt_dir, args.seq + D.GT_SUFFIX)) if args.gt_dir else {}
    print(f"[INFO] {args.seq}  {sn.name} {W}x{H}  windows={len(wins)} "
          f"pred={len(pred)} gt={len(gt)}")

    if args.xyt:
        render_xyt(ev, args.seq, args.out if args.out.endswith(".html")
                   else os.path.splitext(args.out)[0] + "_xyt.html")
        if args.out.endswith(".html"):
            return

    if args.gallery:
        scored = []
        for w in wins:
            g, pr = gt.get(w.start_us), pred.get(w.start_us)
            if g is None and pr is None:
                continue
            scored.append((iou(pr[:4] if pr else None, g[:4] if g else None), w))
        # --best -> highest IoU first (success cases); default -> worst first
        scored.sort(key=lambda t: t[0], reverse=args.best)
        items = []
        for ov, w in scored[:args.gallery_n]:
            im, o, hp, hg = render_frame(ev, w, W, H, pred, gt, args.scale)
            if args.best:
                cap = f"win {w.index}  IoU {o:.2f}"
            else:
                cap = ("FN(missed)" if hg and not hp else
                       ("FP" if hp and not hg else f"IoU {o:.2f}"))
                cap = f"win {w.index}  {cap}"
            items.append((im, cap))
        out = save_gallery(items, args.out, cols=4)
        kind = "detection examples" if args.best else "failure gallery"
        print(f"[viz] {kind} ({len(items)}) -> {out}")
        return

    frames = [render_frame(ev, w, W, H, pred, gt, args.scale)[0]
              for w in wins[args.start:args.start + args.max_frames]]
    out = save_animation(frames, args.out, args.fps)
    print(f"[viz] animation ({len(frames)} frames @ {args.fps}fps) -> {out}")


if __name__ == "__main__":
    main()
