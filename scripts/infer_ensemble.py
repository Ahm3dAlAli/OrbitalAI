#!/usr/bin/env python3
"""Ensemble + TTA inference for CenterNet event detectors.

Averages the heatmap/size/offset predictions of several checkpoints (must share
the same grid/hm) and, with --tta, also averages horizontal+vertical flips.
Ensembling + TTA are the standard "free" accuracy levers we hadn't used yet;
each typically adds a few points of mAP.

    python3 scripts/infer_ensemble.py --device cuda \
        --data-dir OrbitSight_Dataset/Testing_sets --out-dir predictions/ens --tta \
        --models models/g192_s1.pt models/g192_s2.pt models/g192_s3.pt
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.evt_model import voxelize
from orbitsight.evt_centernet import EventCenterNet, decode, decode_peaks


def load(path, device):
    b = torch.load(path, map_location="cpu", weights_only=False)
    c = b["cfg"]
    m = EventCenterNet(grid=c["grid"], patch=c["patch"], tbins=c["tbins"],
                       dim=c["dim"], hm_div=c["hm_div"],
                       enc_layers=c.get("enc_layers", 3), variant=c["variant"])
    m.load_state_dict(b["state_dict"]); m.eval(); m.to(device)
    return m, c


@torch.no_grad()
def _predict(model, x, tta):
    """Return averaged sigmoid-heatmap, wh, off for a voxel batch x."""
    hm, wh, off = model(x)
    hm = torch.sigmoid(hm)
    n = 1
    if tta:
        # horizontal flip: un-flip heatmap/size; offset x mirrors (1 - ox)
        xf = torch.flip(x, dims=[3])
        h2, w2, o2 = model(xf)
        h2 = torch.flip(torch.sigmoid(h2), dims=[3])
        w2 = torch.flip(w2, dims=[3])
        o2 = torch.flip(o2, dims=[3]); o2[:, 0] = 1.0 - o2[:, 0]
        hm = hm + h2; wh = wh + w2; off = off + o2; n += 1
        # vertical flip
        xf = torch.flip(x, dims=[2])
        h3, w3, o3 = model(xf)
        h3 = torch.flip(torch.sigmoid(h3), dims=[2])
        w3 = torch.flip(w3, dims=[2])
        o3 = torch.flip(o3, dims=[2]); o3[:, 1] = 1.0 - o3[:, 1]
        hm = hm + h3; wh = wh + w3; off = off + o3; n += 1
    return hm / n, wh / n, off / n


@torch.no_grad()
def run(ev, seq, models, cfgs, cfg, thresh, tta, device, batch=128, topk=1):
    """Ensemble over models that may have DIFFERENT grids/tbins.  Each model is
    fed a voxel at its own grid; heatmap/size/offset maps are resized to a common
    resolution (the finest member's heatmap) before averaging — so scale-diverse
    members (grid-128/192/256) can be combined."""
    sn = sensor_for_sequence(seq)
    H0 = max(c["grid"] // c["hm_div"] for c in cfgs)   # common heatmap side
    wins = D.make_window_grid(ev.t, cfg.window_us)
    dets = []
    wbuf, meta = [], []

    def _vox(w, grid, tbins, ctx):
        lo, hi, ws, we = w.lo, w.hi, w.start_us, w.end_us
        if ctx > 0:                                  # temporal context: widen
            ws, we = ws - ctx * cfg.window_us, we + ctx * cfg.window_us
            lo = int(np.searchsorted(ev.t, ws, "left"))
            hi = int(np.searchsorted(ev.t, we, "left"))
        return voxelize(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], ev.t[lo:hi],
                        ws, we, sn.width, sn.height, grid, tbins)

    def flush():
        if not wbuf:
            return
        # voxelize once per unique (grid, tbins, context) config
        vcache = {}
        for c in cfgs:
            key = (c["grid"], c["tbins"], c.get("context", 0))
            if key not in vcache:
                arr = np.stack([_vox(w, key[0], key[1], key[2]) for w in wbuf])
                vcache[key] = torch.from_numpy(arr).float().to(device)
        hm_sum = wh_sum = off_sum = None
        for m, c in zip(models, cfgs):
            hm, wh, off = _predict(m, vcache[(c["grid"], c["tbins"], c.get("context", 0))], tta)
            if hm.shape[-1] != H0:                      # resize to common res
                hm = F.interpolate(hm, size=(H0, H0), mode="bilinear", align_corners=False)
                wh = F.interpolate(wh, size=(H0, H0), mode="bilinear", align_corners=False)
                off = F.interpolate(off, size=(H0, H0), mode="bilinear", align_corners=False)
            hm_sum = hm if hm_sum is None else hm_sum + hm
            wh_sum = wh if wh_sum is None else wh_sum + wh
            off_sum = off if off_sum is None else off_sum + off
        k = len(models)
        p = (hm_sum / k).clamp(1e-6, 1 - 1e-6)
        logit = torch.log(p / (1 - p))
        if topk > 1:      # multi-object: local-maxima NMS peaks (e.g. Stars3 field)
            dec = decode_peaks(logit.cpu(), (wh_sum / k).cpu(), (off_sum / k).cpu(), topk=topk)
        else:
            dec = decode(logit.cpu(), (wh_sum / k).cpu(), (off_sum / k).cpu(), topk=1)
        for j, w in enumerate(wbuf):
            for s, cx, cy, bw, bh in dec[j]:
                if s >= thresh:
                    dets.append(D.Detection(w.start_us, w.end_us,
                        int(round(np.clip(cx * sn.width, 0, sn.width - 1))),
                        int(round(np.clip(cy * sn.height, 0, sn.height - 1))),
                        max(int(round(bw * sn.width)), 1), max(int(round(bh * sn.height)), 1),
                        float(s)))
        wbuf.clear()

    for win in wins:
        wbuf.append(win)
        if len(wbuf) >= batch:
            flush()
    flush()
    return dets, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--topk", type=int, default=1,
                    help=">1 emits multiple boxes/window (multi-object, e.g. Stars3)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    loaded = [load(p, device) for p in args.models]
    models = [m for m, _ in loaded]; cfgs = [c for _, c in loaded]
    grids = sorted({c["grid"] for c in cfgs})
    print(f"[INFO] ensemble of {len(models)} models grids={grids} "
          f"tta={args.tta} [{device}]"
          + ("  (cross-grid: heatmaps resized to common res)" if len(grids) > 1 else ""))
    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, seq in enumerate(seqs, 1):
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        t0 = time.perf_counter()
        dets, nw = run(ev, seq, models, cfgs, cfg, args.thresh, args.tta, device, topk=args.topk)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"[{i:2d}/{len(seqs)}] {seq[:36]:36s} win={nw:5d} det={len(dets):5d} "
              f"{1000*(time.perf_counter()-t0)/max(nw,1):.2f} ms/win")
        del ev


if __name__ == "__main__":
    main()
