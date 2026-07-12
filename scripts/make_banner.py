#!/usr/bin/env python3
"""Generate the 920x400 submission banner for the ChallengeON form.

A space-tech hero image for the OrbitSight project: event-camera dot field, an
RSO track with a detection box, the project name, tagline, and the headline
results (mAP, real-time latency, multi-sensor). Output is exactly 920x400 px.
"""
from __future__ import annotations

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "docs", "figures", "orbitsight_banner_920x400.png")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

W, H, DPI = 920, 400, 100
NAVY, DEEP = "#0a1430", "#050a1c"
CYAN, MAG, AMBER, WHITE = "#2ec4b6", "#e05a9b", "#f4a261", "#f2f6ff"

fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI)
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.axis("off")

# --- vertical gradient background ---
grad = np.linspace(0, 1, 256).reshape(-1, 1)
ax.imshow(grad, extent=[0, W, 0, H], aspect="auto", cmap=
          matplotlib.colors.LinearSegmentedColormap.from_list("bg", [DEEP, NAVY]),
          origin="lower", zorder=0)

rng = np.random.default_rng(7)
# --- event-camera dot field (cyan = +polarity, magenta = -polarity) ---
n = 520
ex, ey = rng.uniform(0, W, n), rng.uniform(0, H, n)
pol = rng.random(n) > 0.5
sz = rng.uniform(1, 9, n)
ax.scatter(ex[pol], ey[pol], s=sz[pol], c=CYAN, alpha=0.55, zorder=1, linewidths=0)
ax.scatter(ex[~pol], ey[~pol], s=sz[~pol], c=MAG, alpha=0.45, zorder=1, linewidths=0)

# --- an RSO track: a curved streak of bright events + a detection box ---
tt = np.linspace(0, 1, 60)
tx = 250 + 470 * tt
ty = 250 + 60 * np.sin(tt * 2.2) + 30 * tt
ax.plot(tx, ty, color=WHITE, lw=1.2, alpha=0.35, zorder=2)
ax.scatter(tx, ty, s=rng.uniform(4, 16, 60), c=AMBER, alpha=0.9, zorder=3, linewidths=0)
# detection box around the head of the track
hx, hy = tx[-1], ty[-1]
ax.add_patch(Rectangle((hx - 26, hy - 22), 52, 44, fill=False,
                       edgecolor=CYAN, lw=2.0, zorder=4))
ax.text(hx + 30, hy + 14, "RSO  0.98", color=CYAN, fontsize=9,
        family="DejaVu Sans", zorder=4)

# --- title + tagline ---
ax.text(48, 300, "OrbitSight", color=WHITE, fontsize=54, fontweight="bold",
        family="DejaVu Sans", zorder=5)
ax.text(52, 262, "Neuromorphic-Vision RSO Detection", color=CYAN, fontsize=17,
        family="DejaVu Sans", zorder=5)
ax.text(52, 238, "Event-native AI  ·  real-time  ·  low-light Space Situational Awareness",
        color="#9fb3d1", fontsize=11, family="DejaVu Sans", zorder=5)

# --- result stat chips ---
chips = [("mAP  0.675", CYAN), ("< 20 ms real-time", AMBER),
         ("3 sensors, 1 pipeline", MAG)]
x0 = 52
for label, col in chips:
    w = 30 + 9.6 * len(label)
    ax.add_patch(FancyBboxPatch((x0, 40), w, 42, boxstyle="round,pad=2,rounding_size=11",
                                linewidth=1.5, edgecolor=col, facecolor="#0e1c3a",
                                zorder=5))
    ax.text(x0 + w / 2, 61, label, color=col, fontsize=12.5, ha="center",
            va="center", fontweight="bold", family="DejaVu Sans", zorder=6)
    x0 += w + 22

# --- corner tag (bottom-right, clear of the track) ---
ax.text(W - 24, 62, "TII OrbitSight Challenge", color="#8fa6c8", fontsize=10.5,
        ha="right", family="DejaVu Sans", zorder=5)
ax.text(W - 24, 44, "Space Situational Awareness", color="#5f7392", fontsize=9,
        ha="right", family="DejaVu Sans", zorder=5)

fig.savefig(OUT, dpi=DPI)
plt.close(fig)

# guarantee exact 920x400 (matplotlib can be off by a pixel)
try:
    from PIL import Image
    im = Image.open(OUT)
    if im.size != (W, H):
        im.resize((W, H), Image.LANCZOS).save(OUT)
    print(f"[banner] {OUT}  {Image.open(OUT).size[0]}x{Image.open(OUT).size[1]}")
except Exception:
    print(f"[banner] {OUT}")
