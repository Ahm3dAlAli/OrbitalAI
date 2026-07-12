"""Alternative-model ablation backbones (innovation-criterion baselines):

  * SpikingDetector  — a Spiking Neural Network (LIF neurons + surrogate
    gradient) over the temporal voxel bins.  The natural event-native, low-power
    framing the brief highlights.
  * PointNetDetector — a permutation-invariant point-set network over raw event
    clouds (the PointNet / graph-NN family the brief and proposal cite).

Both share a single-box detection head (one box per 40 ms window, matching the
GT structure) so the comparison isolates the *backbone*.  CPU-only PyTorch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
#  Shared single-box head + loss
# --------------------------------------------------------------------------- #
class BoxHead(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.obj = nn.Linear(dim, 1)
        self.box = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 4))

    def forward(self, feat):
        return self.obj(feat).squeeze(-1), self.box(feat).sigmoid()


def box_giou(a, b, eps=1e-7):
    ax1, ay1 = a[..., 0] - a[..., 2] / 2, a[..., 1] - a[..., 3] / 2
    ax2, ay2 = a[..., 0] + a[..., 2] / 2, a[..., 1] + a[..., 3] / 2
    bx1, by1 = b[..., 0] - b[..., 2] / 2, b[..., 1] - b[..., 3] / 2
    bx2, by2 = b[..., 0] + b[..., 2] / 2, b[..., 1] + b[..., 3] / 2
    ix1, iy1 = torch.max(ax1, bx1), torch.max(ay1, by1)
    ix2, iy2 = torch.min(ax2, bx2), torch.min(ay2, by2)
    iw, ih = (ix2 - ix1).clamp_min(0), (iy2 - iy1).clamp_min(0)
    inter = iw * ih
    ua = (ax2 - ax1).clamp_min(0) * (ay2 - ay1).clamp_min(0)
    ub = (bx2 - bx1).clamp_min(0) * (by2 - by1).clamp_min(0)
    union = ua + ub - inter + eps
    iou = inter / union
    cx1, cy1 = torch.min(ax1, bx1), torch.min(ay1, by1)
    cx2, cy2 = torch.max(ax2, bx2), torch.max(ay2, by2)
    carea = (cx2 - cx1) * (cy2 - cy1) + eps
    return iou - (carea - union) / carea


def single_box_loss(obj, box, has, gt, w_l1=5.0, w_giou=2.0):
    obj_loss = F.binary_cross_entropy_with_logits(
        obj, has, pos_weight=torch.tensor(3.0))
    pos = has > 0.5
    if pos.any():
        b, g = box[pos], gt[pos]
        bl = F.l1_loss(b, g)
        gi = (1 - box_giou(b, g)).mean()
    else:
        bl = box.new_zeros(())
        gi = box.new_zeros(())
    return obj_loss + w_l1 * bl + w_giou * gi, obj_loss.item()


# --------------------------------------------------------------------------- #
#  SNN — LIF neuron with surrogate gradient
# --------------------------------------------------------------------------- #
class _Spike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return (x > 0).float()

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        sg = 1.0 / (1.0 + 10.0 * x.abs()) ** 2     # fast-sigmoid surrogate
        return g * sg


spike = _Spike.apply


class LIFConv(nn.Module):
    """Convolution followed by a Leaky-Integrate-and-Fire activation.  Membrane
    potential is carried across time steps by the caller via ``state``."""

    def __init__(self, cin, cout, stride=1, beta=0.9, thresh=1.0):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, 3, stride=stride, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.beta, self.thresh = beta, thresh

    def forward(self, x, mem):
        cur = self.bn(self.conv(x))
        mem = self.beta * mem + cur if mem is not None else cur
        s = spike(mem - self.thresh)
        mem = mem - s * self.thresh                # soft reset
        return s, mem


class SpikingDetector(nn.Module):
    """Spiking CNN over the temporal voxel bins (T time steps, 2 polarity
    channels each).  Accumulated output spikes feed the single-box head."""

    def __init__(self, grid=64, tbins=5, dim=96):
        super().__init__()
        self.T = tbins
        self.l1 = LIFConv(2, 32, stride=2)
        self.l2 = LIFConv(32, 64, stride=2)
        self.l3 = LIFConv(64, dim, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = BoxHead(dim)

    def forward(self, vox):
        # vox: (B, T*2, G, G) -> (B, T, 2, G, G)
        B = vox.shape[0]
        x = vox.view(B, self.T, 2, vox.shape[-2], vox.shape[-1])
        m1 = m2 = m3 = None
        acc = 0.0
        for t in range(self.T):
            s, m1 = self.l1(x[:, t], m1)
            s, m2 = self.l2(s, m2)
            s, m3 = self.l3(s, m3)
            acc = acc + s
        feat = self.pool(acc / self.T).flatten(1)
        return self.head(feat)


# --------------------------------------------------------------------------- #
#  PointNet / graph-NN — permutation-invariant set network over event clouds
# --------------------------------------------------------------------------- #
class PointNetDetector(nn.Module):
    """Shared per-point MLP + symmetric max-pool (PointNet).  Operates directly
    on the sparse event cloud, respecting sparsity without voxelization — the
    point/graph family baseline."""

    def __init__(self, dim=256, in_ch=4):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_ch, 64, 1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, dim, 1), nn.BatchNorm1d(dim), nn.ReLU())
        self.post = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.head = BoxHead(dim)

    def forward(self, pts, mask):
        # pts: (B, M, 4) normalized; mask: (B, M) valid-point mask
        x = pts.transpose(1, 2)                    # (B,4,M)
        f = self.mlp(x)                            # (B,dim,M)
        f = f.masked_fill(~mask.unsqueeze(1), -1e4)
        g = f.max(dim=2).values                    # (B,dim) symmetric pool
        g = self.post(g)
        return self.head(g)
