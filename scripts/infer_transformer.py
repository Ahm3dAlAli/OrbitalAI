#!/usr/bin/env python3
"""Inference with the Sparse Event Transformer (EvT-SSA / LinaEvT).

Voxelizes each 40 ms window, runs the transformer, and emits the highest-
objectness query as that window's box (one box per window, matching the GT
structure).  Writes the standard prediction format so the frozen evaluator and
the rest of the toolchain work unchanged.

Usage:
    python3 scripts/infer_transformer.py \
        --data-dir OrbitSight_Dataset/Testing_sets \
        --model models/evt_ssa.pt --out-dir predictions/testing_evt
"""
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
from orbitsight.evt_model import EventTransformer, voxelize


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    m = EventTransformer(grid=c["grid"], patch=c.get("patch", 8), tbins=c["tbins"],
                         dim=c["dim"], queries=c["queries"], variant=c["variant"])
    m.load_state_dict(blob["state_dict"])
    m.eval()
    return m, c


def discover(data_dir):
    return sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
                  for p in glob.glob(os.path.join(data_dir, "*" + D.EV_SUFFIX)))


@torch.no_grad()
def run_sequence(ev, seq, model, c, cfg, thresh, batch=256):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    dets = []
    buf_vox, buf_meta = [], []

    def flush():
        if not buf_vox:
            return
        x = torch.from_numpy(np.stack(buf_vox)).float()
        obj, box = model(x)
        prob = torch.sigmoid(obj)
        best = prob.argmax(dim=1)
        ar = torch.arange(len(best))
        p = prob[ar, best].numpy()
        b = box[ar, best].numpy()
        for k, (ws, we) in enumerate(buf_meta):
            if p[k] < thresh:
                continue
            cx, cy, w, h = b[k]
            dets.append(D.Detection(
                ws, we,
                int(round(np.clip(cx * sn.width, 0, sn.width - 1))),
                int(round(np.clip(cy * sn.height, 0, sn.height - 1))),
                max(int(round(w * sn.width)), 1), max(int(round(h * sn.height)), 1),
                float(p[k])))
        buf_vox.clear(); buf_meta.clear()

    for win in wins:
        v = voxelize(ev.x[win.lo:win.hi], ev.y[win.lo:win.hi],
                     ev.pol[win.lo:win.hi], ev.t[win.lo:win.hi],
                     win.start_us, win.end_us, sn.width, sn.height,
                     c["grid"], c["tbins"])
        buf_vox.append(v); buf_meta.append((win.start_us, win.end_us))
        if len(buf_vox) >= batch:
            flush()
    flush()
    return dets, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="models/evt_ssa.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--thresh", type=float, default=0.5)
    args = ap.parse_args()

    cfg = DEFAULT_CONFIG
    model, c = load_model(args.model)
    print(f"[INFO] {c['variant'].upper()} transformer  grid={c['grid']} dim={c['dim']}")
    seqs = args.sequences or discover(args.data_dir)
    os.makedirs(args.out_dir, exist_ok=True)
    for i, seq in enumerate(seqs, 1):
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        t0 = time.perf_counter()
        dets, nw = run_sequence(ev, seq, model, c, cfg, args.thresh)
        dt = time.perf_counter() - t0
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"[{i:2d}/{len(seqs)}] {seq[:40]:40s} win={nw:5d} det={len(dets):5d} "
              f"{1000*dt/max(nw,1):.2f} ms/win")
        del ev


if __name__ == "__main__":
    main()
