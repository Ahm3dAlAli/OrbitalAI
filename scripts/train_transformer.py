#!/usr/bin/env python3
"""Train the Sparse Event Transformer (EvT-SSA / LinaEvT) on the OrbitSight
training sequences.  CPU-only PyTorch.

Builds a per-window dataset (sparse voxel grid -> one normalized GT box, or
none), and trains a DETR-style detector with objectness + L1 + GIoU losses and
1-GT Hungarian matching.

Usage:
    python3 scripts/train_transformer.py --variant evt --epochs 12 \
        --out models/evt_ssa.pt
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight import data as D
from orbitsight.config import DEFAULT_CONFIG, TRAIN_SEQUENCES, sensor_for_sequence
from orbitsight.evt_model import EventTransformer, voxelize, box_giou
from orbitsight.augment import augment as augment_events


# --------------------------------------------------------------------------- #
#  Dataset
# --------------------------------------------------------------------------- #
class WindowSet(Dataset):
    def __init__(self, data_dir, sequences, cfg, grid, tbins,
                 neg_per_pos=1.5, seed=0, augment=False, context=0,
                 _items=None, _events=None, _sensors=None):
        self.grid, self.tbins, self.cfg = grid, tbins, cfg
        self.augment = augment
        self.context = context          # +/- windows of temporal context
        self._rng = np.random.default_rng(seed + 7)
        # Shared-split constructor: reuse another set's loaded events + a
        # pre-selected item list (used to make a leakage-safe val split without
        # re-loading the multi-GB event arrays).
        if _events is not None:
            self.events, self.sensors, self.items = _events, _sensors, _items
            self.n_pos = sum(1 for it in _items if it[5] is not None)
            return
        self.events = {}            # seq -> Events
        self.sensors = {}
        self.items = []             # (seq, lo, hi, ws, we, box[4] or None)
        rng = np.random.default_rng(seed)
        pos, neg = [], []
        for seq in sequences:
            p = D.find_event_file(data_dir, seq)
            if not p:
                continue
            ev = D.Events.from_npy(p)
            sn = sensor_for_sequence(seq)
            self.events[seq] = ev
            self.sensors[seq] = sn
            gt = {}
            gp = os.path.join(data_dir, seq + D.GT_SUFFIX)
            if os.path.exists(gp):
                for ws, we, cx, cy, w, h in D.load_gt_boxes(gp):
                    gt[ws] = (cx / sn.width, cy / sn.height, w / sn.width, h / sn.height)
            wins = D.make_window_grid(ev.t, cfg.window_us)
            for win in wins:
                box = gt.get(win.start_us)
                rec = (seq, win.lo, win.hi, win.start_us, win.end_us, box)
                (pos if box is not None else neg).append(rec)
        rng.shuffle(neg)
        keep_neg = min(len(neg), int(len(pos) * neg_per_pos))
        self.items = pos + neg[:keep_neg]
        rng.shuffle(self.items)
        self.n_pos = len(pos)
        print(f"[data] {len(sequences)} seqs -> {len(pos)} pos + {keep_neg} neg "
              f"= {len(self.items)} windows")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        seq, lo, hi, ws, we, box = self.items[i]
        ev = self.events[seq]
        sn = self.sensors[seq]
        # temporal context: widen the voxel to +/- `context` windows so the
        # model sees the object's TRACK (many windows of evidence), not one
        # 40 ms slice.  The GT box target stays the CENTER window.
        if self.context > 0:
            wu = self.cfg.window_us
            ws, we = ws - self.context * wu, we + self.context * wu
            lo = int(np.searchsorted(ev.t, ws, "left"))
            hi = int(np.searchsorted(ev.t, we, "left"))
        # normalized coords so augmentation is sensor-agnostic; voxelize with
        # width=height=1 (xn in [0,1]) is identical to the pixel path.
        xn = ev.x[lo:hi].astype(np.float64) / sn.width
        yn = ev.y[lo:hi].astype(np.float64) / sn.height
        pol = ev.pol[lo:hi]
        t = ev.t[lo:hi]
        if self.augment:
            xn, yn, pol, t, box = augment_events(xn, yn, pol, t, box, ws, we, self._rng)
        vox = voxelize(xn, yn, pol, t, ws, we, 1.0, 1.0, self.grid, self.tbins)
        has = box is not None
        b = np.array(box if has else (0, 0, 0, 0), dtype=np.float32)
        return torch.from_numpy(vox), torch.tensor(float(has)), torch.from_numpy(b)


# --------------------------------------------------------------------------- #
#  Loss (1-GT matching)
# --------------------------------------------------------------------------- #
def detr_loss(obj, box, has, gt, w_obj=1.0, w_l1=5.0, w_giou=2.0):
    """obj (B,Q) logits, box (B,Q,4), has (B,), gt (B,4)."""
    B, Q = obj.shape
    tgt_obj = torch.zeros_like(obj)
    pos = has > 0.5
    box_l1 = box.new_zeros(())
    box_gi = box.new_zeros(())
    n_pos = int(pos.sum())
    if n_pos > 0:
        bp = box[pos]                              # (P,Q,4)
        gp = gt[pos].unsqueeze(1)                  # (P,1,4)
        cost = (F.l1_loss(bp, gp.expand_as(bp), reduction="none").sum(-1)
                + (1 - box_giou(bp, gp.expand_as(bp))))      # (P,Q)
        match = cost.argmin(dim=1)                 # (P,)
        idxP = torch.arange(n_pos)
        tgt_obj[pos.nonzero(as_tuple=True)[0], match] = 1.0
        mb = bp[idxP, match]                       # (P,4)
        gb = gt[pos]
        box_l1 = F.l1_loss(mb, gb)
        box_gi = (1 - box_giou(mb, gb)).mean()
    # focal-ish BCE for heavy negative imbalance
    obj_loss = F.binary_cross_entropy_with_logits(
        obj, tgt_obj, pos_weight=torch.tensor(float(Q)))
    return w_obj * obj_loss + w_l1 * box_l1 + w_giou * box_gi, obj_loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Training_sets")
    ap.add_argument("--out", default="models/evt_ssa.pt")
    ap.add_argument("--variant", choices=["evt", "lina"], default="evt")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--grid", type=int, default=64)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--tbins", type=int, default=3)
    ap.add_argument("--dim", type=int, default=96)
    ap.add_argument("--queries", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    torch.manual_seed(DEFAULT_CONFIG.random_seed)
    cfg = DEFAULT_CONFIG
    ds = WindowSet(args.data_dir, TRAIN_SEQUENCES, cfg, args.grid, args.tbins)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, drop_last=True)

    model = EventTransformer(grid=args.grid, patch=args.patch, tbins=args.tbins,
                             dim=args.dim, queries=args.queries, variant=args.variant)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] {args.variant.upper()}  {n_params/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    for ep in range(args.epochs):
        model.train()
        t0 = time.perf_counter()
        tot, tot_obj, nb = 0.0, 0.0, 0
        for vox, has, gt in dl:
            opt.zero_grad()
            obj, box = model(vox)
            loss, ol = detr_loss(obj, box, has, gt)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item(); tot_obj += ol; nb += 1
        sched.step()
        print(f"[ep {ep+1:2d}/{args.epochs}] loss={tot/nb:.4f} obj={tot_obj/nb:.4f} "
              f"({time.perf_counter()-t0:.0f}s)")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"grid": args.grid, "patch": args.patch, "tbins": args.tbins,
                        "dim": args.dim, "queries": args.queries,
                        "variant": args.variant}},
               args.out)
    print(f"[done] saved -> {args.out}")


if __name__ == "__main__":
    main()
