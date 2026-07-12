#!/usr/bin/env python3
"""Inference for the SNN / PointNet ablation backbones -> prediction files."""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, sensor_for_sequence
from orbitsight.evt_model import voxelize
from orbitsight.baselines import SpikingDetector, PointNetDetector


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    if c["arch"] == "snn":
        m = SpikingDetector(grid=c["grid"], tbins=c["tbins"])
    else:
        m = PointNetDetector()
    m.load_state_dict(blob["state_dict"]); m.eval()
    return m, c


def _points(ev, lo, hi, ws, we, sn, M):
    out = np.zeros((M, 4), dtype=np.float32); mask = np.zeros(M, dtype=bool)
    n = hi - lo
    if n > 0:
        idx = np.arange(lo, hi)
        if n > M:
            idx = np.random.default_rng(lo).choice(idx, M, replace=False)
        k = len(idx)
        out[:k, 0] = ev.x[idx] / sn.width; out[:k, 1] = ev.y[idx] / sn.height
        out[:k, 2] = (ev.t[idx] - ws) / max(int(we - ws), 1)
        out[:k, 3] = (ev.pol[idx] > 0).astype(np.float32); mask[:k] = True
    return out, mask


@torch.no_grad()
def run(ev, seq, model, c, cfg, thresh, batch=256):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    dets = []
    buf, meta = [], []
    mbuf = []

    def flush():
        if not buf:
            return
        if c["arch"] == "pointnet":
            pts = torch.from_numpy(np.stack(buf)).float()
            msk = torch.from_numpy(np.stack(mbuf))
            obj, box = model(pts, msk)
        else:
            obj, box = model(torch.from_numpy(np.stack(buf)).float())
        p = torch.sigmoid(obj).numpy(); b = box.numpy()
        for k, (ws, we) in enumerate(meta):
            if p[k] < thresh:
                continue
            cx, cy, w, h = b[k]
            dets.append(D.Detection(ws, we,
                int(round(np.clip(cx * sn.width, 0, sn.width - 1))),
                int(round(np.clip(cy * sn.height, 0, sn.height - 1))),
                max(int(round(w * sn.width)), 1), max(int(round(h * sn.height)), 1),
                float(p[k])))
        buf.clear(); meta.clear(); mbuf.clear()

    for win in wins:
        if c["arch"] == "pointnet":
            pts, mask = _points(ev, win.lo, win.hi, win.start_us, win.end_us, sn, c["npts"])
            buf.append(pts); mbuf.append(mask)
        else:
            buf.append(voxelize(ev.x[win.lo:win.hi], ev.y[win.lo:win.hi],
                       ev.pol[win.lo:win.hi], ev.t[win.lo:win.hi],
                       win.start_us, win.end_us, sn.width, sn.height,
                       c["grid"], c["tbins"]))
        meta.append((win.start_us, win.end_us))
        if len(buf) >= batch:
            flush()
    flush()
    return dets, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG
    model, c = load_model(args.model)
    print(f"[INFO] {c['arch'].upper()} backbone")
    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, seq in enumerate(seqs, 1):
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        t0 = time.perf_counter()
        dets, nw = run(ev, seq, model, c, cfg, args.thresh)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"[{i:2d}/{len(seqs)}] {seq[:38]:38s} win={nw:5d} det={len(dets):5d} "
              f"{1000*(time.perf_counter()-t0)/max(nw,1):.2f} ms/win")
        del ev


if __name__ == "__main__":
    main()
