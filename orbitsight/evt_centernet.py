"""CenterNet-style event detector — a *fair* localization head for the event
transformer ablation.

The query/global-regression head (evt_model) trains a good objectness signal but
cannot localize a ~50 px box from coarse patch features (centers land >65 px
off).  CenterNet replaces global box regression with a **center heatmap** at
cell resolution + sub-cell offset + size regression, which localizes far more
precisely — giving the deep detector a fair shot at IoU>=0.5.

Encoder is the same sparse-masked transformer as EvT-SSA; only the head differs.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evt_model import MaskedSelfAttention, LinearAttention, FFN


class SSMMixer(nn.Module):
    """Diagonal state-space (S4D-style) token mixer — a linear-time, bidirectional
    alternative to self-attention (the 'ssm' variant of the encoder).

    Each channel c is a real one-state SSM  h_t = a_c h_{t-1} + b_c x_t,  y_t = h_t,
    with a learned decay a_c in (0,1).  Its impulse response is the geometric kernel
    K_l = b_c a_c^l, applied to the token sequence by FFT causal convolution
    (O(L log L)); forward + backward scans give global context in linear time.
    This is the recurrent/state-space temporal integrator, cast as a token mixer so
    it slots into the existing CenterNet encoder and ablation."""

    def __init__(self, dim, heads=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        # per-channel decay logits spread over fast->slow time-scales
        self.a_logit = nn.Parameter(torch.linspace(0.5, 4.0, dim).unsqueeze(0))
        self.b_f = nn.Parameter(torch.randn(1, dim) * 0.5)
        self.b_b = nn.Parameter(torch.randn(1, dim) * 0.5)
        self.D = nn.Parameter(torch.ones(dim))
        self.out = nn.Linear(dim, dim)

    def _causal_conv(self, x, b):
        B, L, C = x.shape
        a = torch.sigmoid(self.a_logit).clamp(1e-4, 1 - 1e-4)     # (1,C) in (0,1)
        l = torch.arange(L, device=x.device).float().unsqueeze(1)  # (L,1)
        K = b * torch.exp(l * torch.log(a))                        # (L,C) = b*a^l
        n = 1
        while n < 2 * L:
            n *= 2
        Xf = torch.fft.rfft(x, n=n, dim=1)
        Kf = torch.fft.rfft(K.unsqueeze(0), n=n, dim=1)
        return torch.fft.irfft(Xf * Kf, n=n, dim=1)[:, :L]

    def forward(self, x, key_padding_mask=None):
        r = x
        z = self.norm(x)
        y = self._causal_conv(z, self.b_f)                         # forward scan
        y = y + self._causal_conv(z.flip(1), self.b_b).flip(1)     # backward scan
        y = y + self.D * z
        return r + self.out(y)


class EventCenterNet(nn.Module):
    def __init__(self, grid=128, patch=8, tbins=3, dim=128, heads=4,
                 enc_layers=3, hm_div=2, variant="evt"):
        super().__init__()
        self.grid, self.patch, self.tbins = grid, patch, tbins
        self.gp = grid // patch                       # token grid side
        self.hm = grid // hm_div                       # heatmap side
        C = tbins * 2
        self.embed = nn.Conv2d(C, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.gp * self.gp, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        Attn = {"lina": LinearAttention, "ssm": SSMMixer}.get(variant, MaskedSelfAttention)
        self.enc = nn.ModuleList([nn.ModuleList([Attn(dim, heads), FFN(dim)])
                                  for _ in range(enc_layers)])
        # upsample token map (gp x gp) -> heatmap (hm x hm)
        ups = []
        ch = dim
        size = self.gp
        while size < self.hm:
            ups += [nn.ConvTranspose2d(ch, ch // 2, 4, 2, 1),
                    nn.GroupNorm(8, ch // 2), nn.GELU()]
            ch //= 2
            size *= 2
        self.up = nn.Sequential(*ups) if ups else nn.Identity()
        self.hm_head = nn.Conv2d(ch, 1, 1)
        self.wh_head = nn.Conv2d(ch, 2, 1)
        self.off_head = nn.Conv2d(ch, 2, 1)
        self.hm_head.bias.data.fill_(-2.19)            # focal-loss prior

    def forward(self, vox):
        B = vox.shape[0]
        feat = self.embed(vox)
        tok = feat.flatten(2).transpose(1, 2)
        with torch.no_grad():
            occ = F.max_pool2d((vox.abs().sum(1, keepdim=True) > 0).float(),
                               self.patch).flatten(2).squeeze(1)
            key_pad = occ <= 0
            key_pad[(~key_pad).sum(1) == 0, 0] = False
        x = tok + self.pos
        for attn, ffn in self.enc:
            x = ffn(attn(x, key_pad))
        fmap = x.transpose(1, 2).reshape(B, -1, self.gp, self.gp)
        u = self.up(fmap)
        return self.hm_head(u), self.wh_head(u), self.off_head(u)


# --------------------------------------------------------------------------- #
#  Target generation + loss (CenterNet)
# --------------------------------------------------------------------------- #
def _gaussian2d(shape, sigma):
    m, n = [(s - 1) / 2 for s in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_gaussian(hm, cx, cy, radius):
    d = 2 * radius + 1
    g = _gaussian2d((d, d), sigma=d / 6.0)
    H, W = hm.shape
    l, r = min(cx, radius), min(W - cx, radius + 1)
    t, b = min(cy, radius), min(H - cy, radius + 1)
    if r <= -l or b <= -t:
        return
    masked = hm[cy - t:cy + b, cx - l:cx + r]
    gm = g[radius - t:radius + b, radius - l:radius + r]
    np.maximum(masked, gm, out=masked)


def build_targets(boxes_has, hm_size, min_radius=1):
    """boxes_has: list of (has, cx, cy, w, h) normalized.  Returns batched
    heatmap, wh, offset, ind(center flat idx), reg_mask.  min_radius floors the
    Gaussian splat radius: the 0.3*max(w,h) formula collapses to ~1 cell for a
    ~10 px object, giving almost no positive signal; a floor of 2-3 restores it."""
    B = len(boxes_has)
    hm = np.zeros((B, 1, hm_size, hm_size), np.float32)
    wh = np.zeros((B, 2), np.float32)
    off = np.zeros((B, 2), np.float32)
    ind = np.zeros(B, np.int64)
    mask = np.zeros(B, np.float32)
    for i, (has, cx, cy, w, h) in enumerate(boxes_has):
        if has < 0.5:
            continue
        fcx, fcy = cx * hm_size, cy * hm_size
        icx, icy = int(min(fcx, hm_size - 1)), int(min(fcy, hm_size - 1))
        radius = max(min_radius, int(0.3 * max(w, h) * hm_size))
        draw_gaussian(hm[i, 0], icx, icy, radius)
        wh[i] = (w, h)
        off[i] = (fcx - icx, fcy - icy)
        ind[i] = icy * hm_size + icx
        mask[i] = 1.0
    return (torch.from_numpy(hm), torch.from_numpy(wh), torch.from_numpy(off),
            torch.from_numpy(ind), torch.from_numpy(mask))


def focal_hm_loss(pred, gt, hard_neg_w=0.0, hard_neg_frac=0.1):
    pred = pred.clamp(1e-4, 1 - 1e-4)
    pos = gt.eq(1).float()
    neg = gt.lt(1).float()
    neg_w = torch.pow(1 - gt, 4)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_w * neg
    if hard_neg_w > 0:
        # Online hard-negative mining: upweight the highest-confidence STRICT
        # background cells (gt < 0.01). This targets exactly the false positives
        # the model is currently confident about (e.g. Stars3 background stars).
        # Because we use strict background, the Gaussian skirt (0 < gt < 1) around
        # real objects is excluded -> we never penalize near-misses / real signal.
        B = pred.shape[0]
        bg = gt.lt(0.01).float()
        flat = (pred * bg).reshape(B, -1)
        k = max(int(hard_neg_frac * flat.shape[1]), 1)
        thr = flat.topk(k, dim=1).values[:, -1].reshape(B, *([1] * (pred.dim() - 1)))
        hard = ((pred >= thr) & bg.bool()).float()
        neg_loss = neg_loss * (1.0 + hard_neg_w * hard)
    npos = pos.sum()
    if npos == 0:
        return -neg_loss.sum()
    return -(pos_loss.sum() + neg_loss.sum()) / npos


def _gather(feat, ind):
    # feat (B,2,H,W) -> (B,2) at flat ind
    B, C, H, W = feat.shape
    feat = feat.view(B, C, H * W)
    ind = ind.view(B, 1, 1).expand(B, C, 1)
    return feat.gather(2, ind).squeeze(-1)


def _diou_hinge_size_loss(pwh, poff, twh, toff, mask, hm_size,
                          tau=0.5, margin=0.15, lam=2.0, base_w=0.5):
    """Scale-free box loss for the size head. Assembles the predicted and GT box
    at the (shared) true center cell -- center = offset/hm_size (the cell index
    cancels), size = (w,h) in normalized units -- and returns

        base_w * (1 - DIoU) + lam * hinge(tau + margin - IoU)^2

    averaged over the object cells. The DIoU base gives non-vanishing gradient for
    disjoint boxes; the hinge is zero once IoU clears tau+margin (=0.65) and grows
    quadratically as a box approaches the IoU>=0.5 scoring cliff, so gradient
    concentrates on the near-miss population -- the named failure mode. Unlike L1
    on (w,h), this is scale-invariant: 2 px of error on a 10 px object is penalized
    like 2 px is fatal, not like it is free."""
    eps = 1e-7
    # center (normalized) -- shared cell index cancels between pred and GT.
    pcx, pcy = poff[:, 0] / hm_size, poff[:, 1] / hm_size
    gcx, gcy = toff[:, 0] / hm_size, toff[:, 1] / hm_size
    pw, ph = pwh[:, 0].clamp_min(eps), pwh[:, 1].clamp_min(eps)
    gw, gh = twh[:, 0].clamp_min(eps), twh[:, 1].clamp_min(eps)
    px1, py1, px2, py2 = pcx - pw / 2, pcy - ph / 2, pcx + pw / 2, pcy + ph / 2
    gx1, gy1, gx2, gy2 = gcx - gw / 2, gcy - gh / 2, gcx + gw / 2, gcy + gh / 2
    iw = (torch.min(px2, gx2) - torch.max(px1, gx1)).clamp_min(0)
    ih = (torch.min(py2, gy2) - torch.max(py1, gy1)).clamp_min(0)
    inter = iw * ih
    union = pw * ph + gw * gh - inter + eps
    iou = inter / union
    # smallest enclosing box diagonal (DIoU penalty) + center distance.
    cw = torch.max(px2, gx2) - torch.min(px1, gx1)
    ch = torch.max(py2, gy2) - torch.min(py1, gy1)
    c2 = cw * cw + ch * ch + eps
    rho2 = (pcx - gcx) ** 2 + (pcy - gcy) ** 2
    diou = iou - rho2 / c2
    hinge = torch.clamp(tau + margin - iou, min=0) ** 2
    per = base_w * (1 - diou) + lam * hinge
    return (per * mask).sum() / mask.sum().clamp_min(1)


def centernet_loss(hm, wh, off, tgt, w_hm=1.0, w_wh=1.0, w_off=1.0, hard_neg_w=0.0,
                   iou_size=False, w_iou=1.0, iou_tau=0.5, iou_margin=0.15,
                   iou_lambda=2.0, iou_base=0.5):
    thm, twh, toff, ind, mask = tgt
    lhm = focal_hm_loss(torch.sigmoid(hm), thm, hard_neg_w=hard_neg_w)
    m = mask.unsqueeze(1)
    pwh = _gather(wh, ind)
    poff = _gather(off, ind)
    n = mask.sum().clamp_min(1)
    loff = (F.l1_loss(poff * m, toff * m, reduction="sum") / n)
    if iou_size:
        # Scale-free DIoU + hinge on the size head (A/B against L1 via --iou-size).
        hm_size = wh.shape[-1]
        lbox = _diou_hinge_size_loss(pwh, poff, twh, toff, mask, hm_size,
                                     tau=iou_tau, margin=iou_margin,
                                     lam=iou_lambda, base_w=iou_base)
        return w_hm * lhm + w_iou * lbox + w_off * loff, lhm.item()
    lwh = (F.l1_loss(pwh * m, twh * m, reduction="sum") / n)
    return w_hm * lhm + w_wh * lwh + w_off * loff, lhm.item()


@torch.no_grad()
def decode_peaks(hm, wh, off, topk=5, kernel=3):
    """Decode the top-k *local-maxima* heatmap peaks per sample (NMS via
    max-pool), returning per batch a list of (score, cx, cy, w, h) normalized.
    Local-maxima NMS prevents one blob from spawning several adjacent peaks —
    needed when peaks feed a tracker as candidates."""
    B, _, H, W = hm.shape
    prob = torch.sigmoid(hm)
    pad = (kernel - 1) // 2
    pooled = F.max_pool2d(prob, kernel, stride=1, padding=pad)
    keep = (pooled == prob).float()
    prob = (prob * keep).view(B, -1)
    scores, idx = prob.topk(topk, dim=1)
    whf = wh.view(B, 2, -1)
    offf = off.view(B, 2, -1)
    out = []
    for b in range(B):
        dets = []
        for k in range(topk):
            i = int(idx[b, k])
            s = float(scores[b, k])
            cy, cx = divmod(i, W)
            ox, oy = float(offf[b, 0, i]), float(offf[b, 1, i])
            w, h = float(whf[b, 0, i]), float(whf[b, 1, i])
            dets.append((s, (cx + ox) / W, (cy + oy) / H, w, h))
        out.append(dets)
    return out


@torch.no_grad()
def decode(hm, wh, off, topk=1):
    """Return list per batch of (score, cx, cy, w, h) normalized."""
    B, _, H, W = hm.shape
    prob = torch.sigmoid(hm).view(B, -1)
    scores, idx = prob.topk(topk, dim=1)
    out = []
    whf = wh.view(B, 2, -1)
    offf = off.view(B, 2, -1)
    for b in range(B):
        dets = []
        for k in range(topk):
            i = int(idx[b, k])
            cy, cx = divmod(i, W)
            ox, oy = float(offf[b, 0, i]), float(offf[b, 1, i])
            w, h = float(whf[b, 0, i]), float(whf[b, 1, i])
            dets.append((float(scores[b, k]),
                         (cx + ox) / W, (cy + oy) / H, w, h))
        out.append(dets)
    return out
