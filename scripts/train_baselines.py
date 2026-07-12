#!/usr/bin/env python3
"""Train alternative-model ablation backbones: SNN or PointNet/graph-NN.

Usage:
    python3 scripts/train_baselines.py --arch snn      --epochs 12 --out models/snn.pt
    python3 scripts/train_baselines.py --arch pointnet --epochs 12 --out models/pointnet.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, TRAIN_SEQUENCES, sensor_for_sequence
from orbitsight.evt_model import voxelize
from orbitsight.baselines import SpikingDetector, PointNetDetector, single_box_loss


class WindowSet(Dataset):
    def __init__(self, data_dir, sequences, cfg, arch, grid=64, tbins=5,
                 npts=256, neg_per_pos=1.5, seed=0):
        self.arch, self.grid, self.tbins, self.npts = arch, grid, tbins, npts
        self.cfg = cfg
        self.events, self.sensors, items = {}, {}, []
        rng = np.random.default_rng(seed)
        pos, neg = [], []
        for seq in sequences:
            p = D.find_event_file(data_dir, seq)
            if not p:
                continue
            ev = D.Events.from_npy(p)
            sn = sensor_for_sequence(seq)
            self.events[seq], self.sensors[seq] = ev, sn
            gt = {}
            gp = os.path.join(data_dir, seq + D.GT_SUFFIX)
            if os.path.exists(gp):
                for ws, we, cx, cy, w, h in D.load_gt_boxes(gp):
                    gt[ws] = (cx / sn.width, cy / sn.height, w / sn.width, h / sn.height)
            for win in D.make_window_grid(ev.t, cfg.window_us):
                box = gt.get(win.start_us)
                rec = (seq, win.lo, win.hi, win.start_us, win.end_us, box)
                (pos if box is not None else neg).append(rec)
        rng.shuffle(neg)
        self.items = pos + neg[:min(len(neg), int(len(pos) * neg_per_pos))]
        rng.shuffle(self.items)
        print(f"[data] {len(pos)} pos + {len(self.items)-len(pos)} neg = {len(self.items)}")

    def __len__(self):
        return len(self.items)

    def _points(self, ev, lo, hi, ws, we, sn):
        n = hi - lo
        M = self.npts
        out = np.zeros((M, 4), dtype=np.float32)
        mask = np.zeros(M, dtype=bool)
        if n > 0:
            idx = np.arange(lo, hi)
            if n > M:
                idx = np.random.default_rng(lo).choice(idx, M, replace=False)
            k = len(idx)
            out[:k, 0] = ev.x[idx] / sn.width
            out[:k, 1] = ev.y[idx] / sn.height
            out[:k, 2] = (ev.t[idx] - ws) / max(int(we - ws), 1)
            out[:k, 3] = (ev.pol[idx] > 0).astype(np.float32)
            mask[:k] = True
        return out, mask

    def __getitem__(self, i):
        seq, lo, hi, ws, we, box = self.items[i]
        ev, sn = self.events[seq], self.sensors[seq]
        has = torch.tensor(float(box is not None))
        b = torch.tensor(np.array(box if box else (0, 0, 0, 0), dtype=np.float32))
        if self.arch == "pointnet":
            pts, mask = self._points(ev, lo, hi, ws, we, sn)
            return torch.from_numpy(pts), torch.from_numpy(mask), has, b
        vox = voxelize(ev.x[lo:hi], ev.y[lo:hi], ev.pol[lo:hi], ev.t[lo:hi],
                       ws, we, sn.width, sn.height, self.grid, self.tbins)
        return torch.from_numpy(vox), has, b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["snn", "pointnet"], required=True)
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Training_sets")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--tbins", type=int, default=5)
    ap.add_argument("--npts", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    torch.manual_seed(DEFAULT_CONFIG.random_seed)
    ds = WindowSet(args.data_dir, TRAIN_SEQUENCES, DEFAULT_CONFIG, args.arch,
                   args.grid, args.tbins, args.npts)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0,
                    drop_last=True)
    if args.arch == "snn":
        model = SpikingDetector(grid=args.grid, tbins=args.tbins)
    else:
        model = PointNetDetector()
    print(f"[model] {args.arch.upper()} {sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for ep in range(args.epochs):
        model.train(); t0 = time.perf_counter(); tot = ol = 0.0; nb = 0
        for batch in dl:
            opt.zero_grad()
            if args.arch == "pointnet":
                pts, mask, has, gt = batch
                obj, box = model(pts, mask)
            else:
                vox, has, gt = batch
                obj, box = model(vox)
            loss, o = single_box_loss(obj, box, has, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); ol += o; nb += 1
        sched.step()
        print(f"[ep {ep+1:2d}/{args.epochs}] loss={tot/nb:.4f} obj={ol/nb:.4f} "
              f"({time.perf_counter()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"arch": args.arch, "grid": args.grid,
                        "tbins": args.tbins, "npts": args.npts}}, args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
