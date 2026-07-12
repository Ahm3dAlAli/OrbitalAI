#!/usr/bin/env python3
"""Inference for the CenterNet event transformer -> prediction files."""
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
from orbitsight.evt_centernet import EventCenterNet, decode


def load_model(path, device="cpu"):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    m = EventCenterNet(grid=c["grid"], patch=c["patch"], tbins=c["tbins"],
                       dim=c["dim"], hm_div=c["hm_div"], variant=c["variant"])
    m.load_state_dict(blob["state_dict"]); m.eval()
    m.to(device)
    return m, c


@torch.no_grad()
def run(ev, seq, model, c, cfg, thresh, batch=128, device="cpu"):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    dets = []
    buf, meta = [], []

    def flush():
        if not buf:
            return
        x = torch.from_numpy(np.stack(buf)).float().to(device)
        hm, wh, off = model(x)
        hm, wh, off = hm.cpu(), wh.cpu(), off.cpu()
        decoded = decode(hm, wh, off, topk=1)
        for k, (ws, we) in enumerate(meta):
            s, cx, cy, w, h = decoded[k][0]
            if s < thresh:
                continue
            dets.append(D.Detection(ws, we,
                int(round(np.clip(cx * sn.width, 0, sn.width - 1))),
                int(round(np.clip(cy * sn.height, 0, sn.height - 1))),
                max(int(round(w * sn.width)), 1), max(int(round(h * sn.height)), 1),
                float(s)))
        buf.clear(); meta.clear()

    ctx = c.get("context", 0)
    for win in wins:
        lo, hi, ws, we = win.lo, win.hi, win.start_us, win.end_us
        if ctx > 0:                                  # widen to +/- ctx windows
            ws, we = ws - ctx * cfg.window_us, we + ctx * cfg.window_us
            lo = int(np.searchsorted(ev.t, ws, "left"))
            hi = int(np.searchsorted(ev.t, we, "left"))
        buf.append(voxelize(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], ev.t[lo:hi],
                   ws, we, sn.width, sn.height, c["grid"], c["tbins"]))
        meta.append((win.start_us, win.end_us))
        if len(buf) >= batch:
            flush()
    flush()
    return dets, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="models/evt_centernet.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--thresh", type=float, default=0.3)
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG
    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    model, c = load_model(args.model, device)
    print(f"[INFO] CenterNet-{c['variant'].upper()} grid={c['grid']} hm={model.hm} [{device}]")
    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, seq in enumerate(seqs, 1):
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        t0 = time.perf_counter()
        dets, nw = run(ev, seq, model, c, cfg, args.thresh, device=device)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"[{i:2d}/{len(seqs)}] {seq[:38]:38s} win={nw:5d} det={len(dets):5d} "
              f"{1000*(time.perf_counter()-t0)/max(nw,1):.2f} ms/win")
        del ev


if __name__ == "__main__":
    main()
