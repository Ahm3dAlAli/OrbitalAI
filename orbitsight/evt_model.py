"""Sparse Event Transformer for RSO detection (proposal implementation).

Implements the core of the OrbitSight transformer proposal, scoped to what is
trainable on the available data and runnable on CPU within the challenge's
constraints:

  * Sparse voxel-grid event representation (§3.2), resolution-agnostic via
    per-sensor normalization to a fixed grid.
  * Patch tokenization with **sparsity-aware attention masking** (§3.3.1):
    empty patches are masked out of attention so compute focuses on non-empty
    regions.
  * EvT-SSA encoder (masked multi-head self-attention) OR LinaEvT linear
    attention (§3.5, kernel feature map -> O(n) attention).
  * DETR-style object queries + detection head (§3.6): each query predicts an
    objectness score and a normalized box; one box per 40 ms window matches the
    dataset's GT structure.

CPU-only PyTorch; small by design (16 training sequences is a tiny regime for
transformers, so capacity is deliberately limited to curb overfitting).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Event representation — sparse voxel grid (resolution-agnostic)
# --------------------------------------------------------------------------- #
def voxelize(x, y, pol, t, ws, we, width, height, grid=64, tbins=3):
    """Accumulate a window's events into a (tbins*2, grid, grid) voxel tensor.

    Coordinates are normalized by the sensor size before gridding, so the same
    grid (and the same model) serves DAVIS / DVX / EVK4 (resolution-agnostic,
    PRD NFR-3).  Channels interleave (time-bin, polarity).
    """
    C = tbins * 2
    vox = np.zeros((C, grid, grid), dtype=np.float32)
    if x.size == 0:
        return vox
    gx = np.clip((x.astype(np.float64) / width * grid).astype(np.int64), 0, grid - 1)
    gy = np.clip((y.astype(np.float64) / height * grid).astype(np.int64), 0, grid - 1)
    dur = max(int(we - ws), 1)
    tb = np.clip(((t - ws).astype(np.float64) / dur * tbins).astype(np.int64), 0, tbins - 1)
    p = (pol > 0).astype(np.int64)
    ch = tb * 2 + p
    np.add.at(vox, (ch, gy, gx), 1.0)
    # log compression tames the dense-EVK4 vs sparse-DVX count gap
    np.log1p(vox, out=vox)
    return vox


# --------------------------------------------------------------------------- #
#  Attention blocks
# --------------------------------------------------------------------------- #
class MaskedSelfAttention(nn.Module):
    """Standard multi-head self-attention with a key-padding mask so empty
    (sparse) patches receive no attention — the EvT-SSA sparse-attention core."""

    def __init__(self, dim, heads):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, key_padding_mask):
        h = self.norm(x)
        out, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask,
                           need_weights=False)
        return x + out


class LinearAttention(nn.Module):
    """Kernel linear attention (LinaEvT, §3.5): phi(Q)(phi(K)^T V) gives O(n)
    complexity.  Uses the elu+1 feature map (Katharopoulos et al.)."""

    def __init__(self, dim, heads):
        super().__init__()
        self.h = heads
        self.dh = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x, key_padding_mask):
        B, N, D = x.shape
        h = self.norm(x)
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        q = self._phi(q).view(B, N, self.h, self.dh)
        k = self._phi(k).view(B, N, self.h, self.dh)
        v = v.view(B, N, self.h, self.dh)
        if key_padding_mask is not None:                  # zero out empty tokens
            m = (~key_padding_mask).float().unsqueeze(-1).unsqueeze(-1)
            k = k * m
            v = v * m
        # (B,h,dh,dh) = sum_n k_n outer v_n ;  (B,h,dh) = sum_n k_n
        kv = torch.einsum("bnhd,bnhe->bhde", k, v)
        z = k.sum(dim=1)                                  # (B,h,dh)
        num = torch.einsum("bnhd,bhde->bnhe", q, kv)
        den = torch.einsum("bnhd,bhd->bnh", q, z).clamp_min(1e-6).unsqueeze(-1)
        out = (num / den).reshape(B, N, D)
        return x + self.proj(out)


class FFN(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(nn.Linear(dim, dim * mult), nn.GELU(),
                                 nn.Linear(dim * mult, dim))

    def forward(self, x):
        return x + self.net(self.norm(x))


# --------------------------------------------------------------------------- #
#  The detector
# --------------------------------------------------------------------------- #
class EventTransformer(nn.Module):
    def __init__(self, grid=64, patch=8, tbins=3, dim=96, heads=4,
                 enc_layers=3, dec_layers=2, queries=8, variant="evt"):
        super().__init__()
        self.grid = grid
        self.patch = patch
        self.tbins = tbins
        C = tbins * 2
        self.npatch = (grid // patch) ** 2
        self.embed = nn.Conv2d(C, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.zeros(1, self.npatch, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

        Attn = LinearAttention if variant == "lina" else MaskedSelfAttention
        self.enc = nn.ModuleList()
        for _ in range(enc_layers):
            self.enc.append(nn.ModuleList([Attn(dim, heads), FFN(dim)]))

        # object queries + DETR-style decoder
        self.queries = nn.Parameter(torch.zeros(1, queries, dim))
        nn.init.trunc_normal_(self.queries, std=0.02)
        self.dec = nn.ModuleList()
        for _ in range(dec_layers):
            self.dec.append(nn.ModuleList([
                nn.MultiheadAttention(dim, heads, batch_first=True),   # self
                nn.MultiheadAttention(dim, heads, batch_first=True),   # cross
                nn.LayerNorm(dim), nn.LayerNorm(dim), FFN(dim)]))
        self.obj_head = nn.Linear(dim, 1)
        self.box_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(),
                                      nn.Linear(dim, 4))

    def forward(self, vox):
        """vox: (B, C, grid, grid) -> objectness (B,Q), boxes (B,Q,4) in [0,1]."""
        B = vox.shape[0]
        feat = self.embed(vox)                            # (B, dim, g/p, g/p)
        tok = feat.flatten(2).transpose(1, 2)             # (B, N, dim)
        # sparsity mask: patches with no events -> True (padded/ignored)
        with torch.no_grad():
            occ = F.max_pool2d((vox.abs().sum(1, keepdim=True) > 0).float(),
                               self.patch).flatten(2).squeeze(1)   # (B, N)
            key_pad = occ <= 0
            key_pad[(~key_pad).sum(1) == 0, 0] = False    # keep >=1 token/sample
        x = tok + self.pos
        for attn, ffn in self.enc:
            x = ffn(attn(x, key_pad))

        q = self.queries.expand(B, -1, -1)
        for self_attn, cross_attn, n1, n2, ffn in self.dec:
            h = n1(q)
            q = q + self_attn(h, h, h, need_weights=False)[0]
            h = n2(q)
            q = q + cross_attn(h, x, x, key_padding_mask=key_pad,
                               need_weights=False)[0]
            q = ffn(q)
        obj = self.obj_head(q).squeeze(-1)                # (B, Q)
        box = self.box_head(q).sigmoid()                  # (B, Q, 4) cx,cy,w,h
        return obj, box


# --------------------------------------------------------------------------- #
#  Box utilities (normalized cx,cy,w,h)
# --------------------------------------------------------------------------- #
def box_giou(a, b, eps=1e-7):
    """GIoU between two sets of boxes (..., 4) in cx,cy,w,h."""
    ax1, ay1 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2
    ax2, ay2 = a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx1, by1 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    bx2, by2 = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    ix1, iy1 = torch.max(ax1, bx1), torch.max(ay1, by1)
    ix2, iy2 = torch.min(ax2, bx2), torch.min(ay2, by2)
    iw, ih = (ix2 - ix1).clamp_min(0), (iy2 - iy1).clamp_min(0)
    inter = iw * ih
    area_a = (ax2 - ax1).clamp_min(0) * (ay2 - ay1).clamp_min(0)
    area_b = (bx2 - bx1).clamp_min(0) * (by2 - by1).clamp_min(0)
    union = area_a + area_b - inter + eps
    iou = inter / union
    cx1, cy1 = torch.min(ax1, bx1), torch.min(ay1, by1)
    cx2, cy2 = torch.max(ax2, bx2), torch.max(ay2, by2)
    carea = (cx2 - cx1) * (cy2 - cy1) + eps
    return iou - (carea - union) / carea
