#!/usr/bin/env python3
"""GHOST separability diagnostic for CenterNet detections.

Tests whether false-positive detections are separable from true positives in the
encoder's feature space via a Gaussian z-score (the core of GHOST, adapted from
open-set *classification* to single-class event *detection*).

For each decoded peak we sample the pre-head feature map `u[:, :, cy, cx]` as the
detection embedding, match the box to GT (TP if IoU>=0.5, else FP), fit a diagonal
Gaussian (mu, sigma) over TP embeddings, and compute the GHOST L1 z-score
s = sum_d |phi_d - mu_d| / sigma_d for every detection. If FP z-scores are clearly
higher than TP z-scores, a GHOST-style post-hoc reject is a real lever; if they
overlap, phantoms are feature-indistinguishable from real RSOs (frontier confirmed).

    python3 scripts/ghost_diag.py --model models/g192_ctx.pt \
        --data-dir OrbitSight_Dataset/Testing_sets \
        --seq DVX_Filtered_Thuraya3_32404_2025-01-20-20-02-43
"""
from __future__ import annotations

import argparse, csv, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.evt_model import voxelize
from orbitsight.evt_centernet import EventCenterNet, decode_peaks


def load(path, device):
    b = torch.load(path, map_location="cpu", weights_only=False); c = b["cfg"]
    m = EventCenterNet(grid=c["grid"], patch=c["patch"], tbins=c["tbins"], dim=c["dim"],
                       hm_div=c["hm_div"], enc_layers=c.get("enc_layers", 3), variant=c["variant"])
    m.load_state_dict(b["state_dict"]); m.eval(); m.to(device)
    return m, c


def load_gt(path):
    d = {}
    if os.path.exists(path):
        for r in csv.DictReader(open(path), delimiter="\t"):
            d.setdefault(int(r["window_start_timestamp_us"]), []).append(
                (float(r["center_x"]), float(r["center_y"]), float(r["width"]), float(r["height"])))
    return d


def iou(a, b):
    ax0, ay0, ax1, ay1 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx0, by0, bx1, by1 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    iw = max(0, min(ax1, bx1)-max(ax0, bx0)); ih = max(0, min(ay1, by1)-max(ay0, by0))
    inter = iw*ih; ua = (ax1-ax0)*(ay1-ay0)+(bx1-bx0)*(by1-by0)-inter
    return inter/ua if ua > 0 else 0


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--seq", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--topk", type=int, default=3)
    args = ap.parse_args()
    dev = args.device
    m, c = load(args.model, dev)
    ctx = c.get("context", 0); grid, tb = c["grid"], c["tbins"]
    sn = sensor_for_sequence(args.seq)
    ev = D.Events.from_npy(D.find_event_file(args.data_dir, args.seq))
    gt = load_gt(os.path.join(args.data_dir, args.seq + D.GT_SUFFIX))
    wins = D.make_window_grid(ev.t, DEFAULT_CONFIG.window_us)

    feat_box = {}
    h = m.up.register_forward_hook(lambda mod, i, o: feat_box.__setitem__("u", o.detach()))

    embs, istp, confs = [], [], []
    for w in wins:
        lo, hi, ws, we = w.lo, w.hi, w.start_us, w.end_us
        if ctx > 0:
            ws, we = ws-ctx*DEFAULT_CONFIG.window_us, we+ctx*DEFAULT_CONFIG.window_us
            lo = int(np.searchsorted(ev.t, ws, "left")); hi = int(np.searchsorted(ev.t, we, "left"))
        vox = voxelize(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], ev.t[lo:hi], ws, we,
                       sn.width, sn.height, grid, tb)
        x = torch.from_numpy(vox[None]).float().to(dev)
        hm, wh, off = m(x)
        u = feat_box["u"][0]                                   # (ch, H, W)
        H, W = u.shape[1], u.shape[2]
        dec = decode_peaks(hm.cpu(), wh.cpu(), off.cpu(), topk=args.topk)[0]
        gts = gt.get(w.start_us, [])
        for s, cx, cy, bw, bh in dec:
            if s < args.thresh:
                continue
            ci = min(int(cx*W), W-1); cj = min(int(cy*H), H-1)
            emb = u[:, cj, ci].numpy()
            box = (cx*sn.width, cy*sn.height, bw*sn.width, bh*sn.height)
            gbox = [(g[0], g[1], g[2], g[3]) for g in gts]
            tp = max((iou(box, g) for g in gbox), default=0) >= 0.5
            embs.append(emb); istp.append(tp); confs.append(float(s))
    h.remove()

    embs = np.array(embs); istp = np.array(istp); confs = np.array(confs)
    tp_e = embs[istp]; fp_e = embs[~istp]
    print(f"detections: {len(embs)}  (TP={istp.sum()}, FP={(~istp).sum()})")
    if istp.sum() < 10 or (~istp).sum() < 10:
        print("too few of one class to fit"); return
    mu = tp_e.mean(0); sig = tp_e.std(0) + 1e-6                 # GHOST Gaussian on TPs
    z = np.abs(embs - mu) / sig                                # (N, D)
    s = z.sum(1)                                               # GHOST L1 z-score
    s_tp, s_fp = s[istp], s[~istp]
    print(f"\nGHOST z-score (sum |phi-mu|/sigma over {embs.shape[1]} dims):")
    print(f"  TP : median={np.median(s_tp):8.1f}  mean={s_tp.mean():8.1f}")
    print(f"  FP : median={np.median(s_fp):8.1f}  mean={s_fp.mean():8.1f}")
    sep = (np.median(s_fp) - np.median(s_tp)) / (np.median(s_tp) + 1e-6)
    # AUROC of z-score separating FP (positive) from TP
    from bisect import bisect_left
    order = np.argsort(s); ranks = np.empty_like(order, float); ranks[order] = np.arange(len(s))
    nfp, ntp = (~istp).sum(), istp.sum()
    auc = (ranks[~istp].sum() - nfp*(nfp-1)/2) / (nfp*ntp)     # FP should have HIGH z
    print(f"\n  FP-vs-TP z-score AUROC = {auc:.3f}  (0.5=no separation, >0.7=useful)")
    print(f"  median FP z is {100*sep:+.0f}% vs TP")
    if auc > 0.65:
        print("\n=> SEPARABLE: a GHOST-style z-score reject is a real lever -> build it.")
    else:
        print("\n=> NOT separable: phantoms are feature-indistinguishable from RSOs")
        print("   -> GHOST cannot help; frontier confirmed at the feature level.")


if __name__ == "__main__":
    main()
