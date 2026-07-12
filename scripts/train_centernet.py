#!/usr/bin/env python3
"""Train the CenterNet-style event transformer (fair localization head).

GPU-ready with a leakage-safe validation split + early stopping, so you can set
a high epoch cap and let it stop itself (recommended on rolf: --epochs 40
--patience 6 --augment --device cuda).

Examples
--------
CPU (laptop):
    python3 scripts/train_centernet.py --grid 128 --augment --epochs 22

rolf (GPU):
    python3 scripts/train_centernet.py --device cuda --grid 192 --augment \
        --epochs 40 --patience 6 --batch 128 --workers 8 --out models/evt_g192.pt
"""
from __future__ import annotations

import argparse
import copy
import os
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from orbitsight.config import DEFAULT_CONFIG, TRAIN_SEQUENCES, sensor_for_sequence
from orbitsight.augment import AugCfg
from orbitsight.evt_centernet import EventCenterNet, build_targets, centernet_loss
from scripts.train_transformer import WindowSet


def _split_items(items, val_frac, seed):
    """Stratified train/val split on windows (keep pos/neg ratio in both)."""
    rng = np.random.default_rng(seed)
    pos = [it for it in items if it[5] is not None]
    neg = [it for it in items if it[5] is None]
    rng.shuffle(pos); rng.shuffle(neg)
    npv, nnv = int(len(pos) * val_frac), int(len(neg) * val_frac)
    val = pos[:npv] + neg[:nnv]
    train = pos[npv:] + neg[nnv:]
    rng.shuffle(train); rng.shuffle(val)
    return train, val


def _run_epoch(model, dl, hm_size, device, opt=None):
    train = opt is not None
    model.train(train)
    tot = lh = 0.0; nb = 0
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for vox, has, gt in dl:
            boxes = [(float(has[i]), *gt[i].tolist()) for i in range(len(has))]
            tgt = [t.to(device) for t in build_targets(boxes, hm_size)]
            vox = vox.to(device, non_blocking=True)
            if train:
                opt.zero_grad()
            hm, wh, off = model(vox)
            loss, l_hm = centernet_loss(hm, wh, off, tgt)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            tot += loss.item(); lh += l_hm; nb += 1
    return tot / max(nb, 1), lh / max(nb, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="OrbitSight_Dataset/Training_sets")
    ap.add_argument("--out", default="models/evt_centernet.pt")
    ap.add_argument("--variant", choices=["evt", "lina"], default="evt")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=6,
                    help="early-stop after this many epochs w/o val improvement")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=48)
    ap.add_argument("--grid", type=int, default=128)
    ap.add_argument("--patch", type=int, default=8)
    ap.add_argument("--tbins", type=int, default=3)
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--hm-div", type=int, default=2)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--augment", action="store_true")
    ap.add_argument("--device", default="auto", help="auto|cpu|cuda")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=None,
                    help="override seed (train diverse ensemble members)")
    ap.add_argument("--enc-layers", type=int, default=3,
                    help="encoder depth (raise for capacity)")
    ap.add_argument("--context", type=int, default=0,
                    help="+/- windows of temporal context (0=single window). "
                         "Set --tbins to span it, e.g. --context 3 --tbins 7")
    # --- DVX-focused reweighting (multi-object / dim-object specialization) ---
    ap.add_argument("--dvx-weight", type=float, default=1.0,
                    help="oversample DVX windows by this factor (e.g. 3.0)")
    ap.add_argument("--evk4-weight", type=float, default=1.0,
                    help="down/upweight EVK4 windows (e.g. 0.5 so the bright, "
                         "dense sequence doesn't dominate)")
    ap.add_argument("--davis-weight", type=float, default=1.0)
    ap.add_argument("--dim-weight", type=float, default=0.0,
                    help="exponent for inverse-event-count weighting; sparse/dim "
                         "windows are sampled more (0=off, 0.5=moderate, 1=strong)")
    ap.add_argument("--dim-aug", action="store_true",
                    help="aggressive dim-object augmentation: event-drop down to "
                         "~10%% of events (vs 30%%) so the model trains on 2-5 event "
                         "objects like Stars3/Thuraya3. Pair with --augment.")
    args = ap.parse_args()

    device = (("cuda" if torch.cuda.is_available() else "cpu")
              if args.device == "auto" else args.device)
    seed = args.seed if args.seed is not None else DEFAULT_CONFIG.random_seed
    torch.manual_seed(seed)
    print(f"[device] {device}"
          + (f"  ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    # load once, split into aug-train / clean-val (shared events, no reload)
    full = WindowSet(args.data_dir, TRAIN_SEQUENCES, DEFAULT_CONFIG,
                     args.grid, args.tbins)
    tr_items, va_items = _split_items(full.items, args.val_frac, seed)
    # aggressive dim-object augmentation: drop to as low as 10% of events (default
    # 30%), applied more often, to synthesize the 2-5 event/window dim regime.
    dim_cfg = AugCfg(drop_p=0.85, drop_min=0.10, noise=0.5) if args.dim_aug else None
    train_ds = WindowSet(None, None, DEFAULT_CONFIG, args.grid, args.tbins,
                         augment=args.augment, context=args.context, _items=tr_items,
                         _events=full.events, _sensors=full.sensors, aug_cfg=dim_cfg)
    val_ds = WindowSet(None, None, DEFAULT_CONFIG, args.grid, args.tbins,
                       augment=False, context=args.context, _items=va_items,
                       _events=full.events, _sensors=full.sensors)
    print(f"[data] train={len(train_ds)} val={len(val_ds)}  aug={'ON' if args.augment else 'off'}")
    pin = device == "cuda"
    reweight = (args.dvx_weight != 1.0 or args.evk4_weight != 1.0
                or args.davis_weight != 1.0 or args.dim_weight > 0)
    if reweight:
        sw = {"DVX": args.dvx_weight, "EVK4": args.evk4_weight, "DAVIS": args.davis_weight}
        weights = []
        for it in tr_items:
            w = sw.get(sensor_for_sequence(it[0]).name, 1.0)
            if args.dim_weight > 0:                     # sparse windows -> higher weight
                n_ev = max(int(it[2]) - int(it[1]), 1)
                w *= (1000.0 / n_ev) ** args.dim_weight
            weights.append(max(w, 1e-6))
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        dl = DataLoader(train_ds, batch_size=args.batch, sampler=sampler,
                        num_workers=args.workers, drop_last=True, pin_memory=pin)
        print(f"[reweight] DVX×{args.dvx_weight} EVK4×{args.evk4_weight} "
              f"DAVIS×{args.davis_weight} dim^{args.dim_weight}  "
              f"(effective sampling; expected DVX share up ~{args.dvx_weight:.1f}x)")
    else:
        dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                        num_workers=args.workers, drop_last=True, pin_memory=pin)
    vdl = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                     num_workers=args.workers, pin_memory=pin)

    model = EventCenterNet(grid=args.grid, patch=args.patch, tbins=args.tbins,
                           dim=args.dim, hm_div=args.hm_div, enc_layers=args.enc_layers,
                           variant=args.variant).to(device)
    hm_size = model.hm
    print(f"[model] CenterNet-{args.variant.upper()} hm={hm_size} "
          f"{sum(p.numel() for p in model.parameters())/1e6:.2f}M params")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)

    best_val = float("inf"); best_state = None; bad = 0
    cfg_blob = {"grid": args.grid, "patch": args.patch, "tbins": args.tbins,
                "dim": args.dim, "hm_div": args.hm_div, "variant": args.variant,
                "enc_layers": args.enc_layers, "context": args.context,
                "reweight": {"dvx": args.dvx_weight, "evk4": args.evk4_weight,
                             "davis": args.davis_weight, "dim": args.dim_weight,
                             "dim_aug": bool(args.dim_aug)}}
    if args.dim_aug:
        print("[dim-aug] aggressive event-drop (keep 10-100%), noise 0.5 — "
              "synthesizing the 2-5 event dim regime")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    for ep in range(args.epochs):
        t0 = time.perf_counter()
        tr_loss, tr_hm = _run_epoch(model, dl, hm_size, device, opt)
        va_loss, va_hm = _run_epoch(model, vdl, hm_size, device, opt=None)
        sched.step()
        improved = va_loss < best_val - 1e-4
        if improved:
            best_val = va_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
            torch.save({"state_dict": best_state, "cfg": cfg_blob}, args.out)
        else:
            bad += 1
        print(f"[ep {ep+1:2d}/{args.epochs}] train={tr_loss:.4f} val={va_loss:.4f} "
              f"(hm {va_hm:.4f}) {'*best*' if improved else f'no-improve {bad}/{args.patience}'} "
              f"({time.perf_counter()-t0:.0f}s)")
        if bad >= args.patience:
            print(f"[early-stop] no val improvement for {args.patience} epochs")
            break

    # ensure the best checkpoint is what's on disk
    if best_state is not None:
        torch.save({"state_dict": best_state, "cfg": cfg_blob}, args.out)
    print(f"[done] best val={best_val:.4f} -> {args.out}")


if __name__ == "__main__":
    main()
