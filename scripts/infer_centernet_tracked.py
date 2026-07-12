#!/usr/bin/env python3
"""CenterNet peaks -> existing motion-gated tracker (the unification bridge).

Instead of argmax-decoding one box per window, decode the top-k heatmap peaks as
per-window *candidates* and run them through the classical tracker
(`track_candidates` + `tracks_to_detections`): constant-velocity gating,
motion/smoothness rejection, top-K selection, and bidirectional track extension.
This gives the deep detector the temporal robustness the classical side has —
the cleanest path to lifting CenterNet on the DVX sequences.
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
from orbitsight.evt_model import voxelize
from orbitsight.evt_centernet import EventCenterNet, decode_peaks
from orbitsight.detect import Candidate, track_candidates, tracks_to_detections


def load_model(path):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    c = blob["cfg"]
    m = EventCenterNet(grid=c["grid"], patch=c["patch"], tbins=c["tbins"],
                       dim=c["dim"], hm_div=c["hm_div"], variant=c["variant"])
    m.load_state_dict(blob["state_dict"]); m.eval()
    return m, c


@torch.no_grad()
def run(ev, seq, model, c, cfg, topk, peak_thresh, batch=128):
    sn = sensor_for_sequence(seq)
    wins = D.make_window_grid(ev.t, cfg.window_us)
    per_window = []
    buf, idxs = [], []

    def flush():
        if not buf:
            return
        hm, wh, off = model(torch.from_numpy(np.stack(buf)).float())
        dec = decode_peaks(hm, wh, off, topk=topk)
        for k, wi in enumerate(idxs):
            win = wins[wi]
            cands = []
            for s, cx, cy, w, h in dec[k]:
                if s < peak_thresh:
                    continue
                pcx, pcy = cx * sn.width, cy * sn.height
                pw, ph = max(w * sn.width, 1), max(h * sn.height, 1)
                cands.append(Candidate(
                    win.index, win.start_us, win.end_us, pcx, pcy,
                    pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2,
                    n=max(int(round(s * 50)), 1), score=s))
            cands.sort(key=lambda cc: cc.n * cc.score, reverse=True)
            per_window[wi] = cands[: cfg.max_candidates]
        buf.clear(); idxs.clear()

    per_window = [[] for _ in wins]
    for wi, win in enumerate(wins):
        buf.append(voxelize(ev.x[win.lo:win.hi], ev.y[win.lo:win.hi],
                   ev.pol[win.lo:win.hi], ev.t[win.lo:win.hi],
                   win.start_us, win.end_us, sn.width, sn.height,
                   c["grid"], c["tbins"]))
        idxs.append(wi)
        if len(buf) >= batch:
            flush()
    flush()

    tracks = track_candidates(per_window, sn, cfg)
    dets = tracks_to_detections(tracks, sn, cfg, n_windows=len(wins))
    return dets, len(wins)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--model", default="models/evt_centernet.pt")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sequences", nargs="*", default=None)
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--peak-thresh", type=float, default=0.2)
    args = ap.parse_args()
    cfg = DEFAULT_CONFIG
    model, c = load_model(args.model)
    print(f"[INFO] CenterNet->tracker  topk={args.topk} peak_thresh={args.peak_thresh}")
    seqs = args.sequences or sorted(os.path.basename(p)[:-len(D.EV_SUFFIX)]
        for p in glob.glob(os.path.join(args.data_dir, "*" + D.EV_SUFFIX)))
    os.makedirs(args.out_dir, exist_ok=True)
    for i, seq in enumerate(seqs, 1):
        p = D.find_event_file(args.data_dir, seq)
        if not p:
            continue
        ev = D.Events.from_npy(p)
        t0 = time.perf_counter()
        dets, nw = run(ev, seq, model, c, cfg, args.topk, args.peak_thresh)
        D.write_predictions(os.path.join(args.out_dir, seq + D.GT_SUFFIX), dets)
        print(f"[{i:2d}/{len(seqs)}] {seq[:38]:38s} win={nw:5d} det={len(dets):5d} "
              f"{1000*(time.perf_counter()-t0)/max(nw,1):.2f} ms/win")
        del ev


if __name__ == "__main__":
    main()
