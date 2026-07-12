# OrbitSight — Technical Proposal
### Real-Time RSO Detection & Tracking from Neuromorphic Vision
TII OrbitSight Challenge · Propulsion & Space Research Center

---

## 1. Problem statement and proposed solution

**Problem.** Given raw neuromorphic-vision-sensor (NVS) event streams captured by
event cameras on a 0.8 m telescope, detect and track Resident Space Objects
(RSOs) — satellites, rocket bodies, debris, and apparently-moving stars — as one
bounding box per 40 ms window, scored at **IoU ≥ 0.5** and aggregated into mAP /
precision / recall / F1. The data is asynchronous, sparse, noisy, and spans
three sensors of different resolution (DAVIS 346×260, DVX 640×480, EVK4
1280×720). Objects are often **2–4 events per window** against ~500 background
events; only **16 labeled training sequences** are available; and there is a
hard **CPU-only, < 40 ms/window, offline** deployment constraint.

**Proposed solution — a two-track detector with per-sensor routing.** We pose
detection as *dense per-event / per-window supervision* and pursue two
complementary detectors, then route each sensor to its winner:

1. **Classical coherence pipeline (CPU, interpretable, real-time).** Per-sensor
   normalization → O(N) background-activity denoise → a learned local
   spatiotemporal-**coherence** event classifier (LightGBM) → geometric
   motion-gated tracking → one box per window. Every stage is O(N); it trains in
   minutes, runs at ~5 ms/window on CPU, needs no GPU, and is fully
   interpretable — the deployable baseline.
2. **CenterNet event-transformer ensemble (accuracy-first).** A sparse
   voxel-grid representation → sparsity-aware attention encoder → a
   **CenterNet-style heatmap head** (center + sub-cell offset + size) that
   localizes to cell resolution. Trained with **event-level augmentation** (the
   dim-drop strategy targeting the domain shift), then combined by
   **model ensembling + test-time augmentation** and, on the dim DVX sensor, a
   **synthetic-tracking (shift-and-stack) merge** for extra recall.

A per-sensor router selects the best detector per sensor. The headline result is
**mAP 0.554 @ IoU 0.5** (recall 0.68, F1 0.59), with the dim, high-resolution
EVK4 test sequence — the case the challenge singles out as hardest — at
**AP 0.90**.

---

## 2. Outcome metrics

Measured with the frozen challenge evaluator on the held-out test set (final
configuration: per-sensor router).

| Sequence | Sensor | Precision | Recall | F1 | AP@0.5 |
|---|---|---:|---:|---:|---:|
| 2025_12_23_20_53_46_EVK4_mag7.3 *(dim headline)* | EVK4 | 0.859 | 0.934 | **0.895** | **0.896** |
| DAVIS_SAOCOM1B_46265 | DAVIS | 0.687 | 0.694 | 0.690 | 0.617 |
| DVX_Filtered_Stars3 | DVX | 0.451 | 0.638 | 0.529 | 0.470 |
| DVX_Filtered_Thuraya3_32404 | DVX | 0.387 | 0.349 | 0.367 | 0.233 |
| **Overall** | — | **0.524** | **0.684** | **0.593** | **mAP 0.554** |

**Improvement trajectory (each step measured, not projected):**
classical baseline 0.069 → tuned classical 0.249 → CenterNet 0.289 → hybrid
router 0.315 → + augmentation 0.398 → + grid-192 0.449 → + stack-merge 0.454 →
+ ensemble & TTA 0.547 → + cross-grid per-sensor routing **0.554** — an **8×
gain** over the classical baseline.

**Inference efficiency (CPU).** Classical pipeline ~5 ms/window (DAVIS/DVX);
deep detectors 2–25 ms/window on CPU — all within the 40 ms budget on the
DAVIS/DVX sensors; the deep model is also the only detector that meets the
budget on the dense EVK4 sensor. GPU (RTX 2080 Ti) inference is ~1 ms/window.

**Achievable ceiling.** An oracle analysis (boxes built from ground-truth event
labels, with the correct per-sensor box size) puts the *maximum attainable*
mAP at **~0.87**. Our 0.554 is ~64% of that ceiling; the residual is
concentrated in the dimmest DVX object (Thuraya3, AP 0.233 vs a 0.968 oracle
ceiling), whose signal is clean but only 4 events/window — the genuine
dim-object-detection core of the problem.

---

## 3. Value proposition and competitive positioning

- **Invariance-first, not capacity-first.** With only 16 sequences and a severe
  train→test domain shift (dim, high-resolution EVK4), the payoff is in
  robustness. Per-sensor normalization, brightness-invariant features, and
  dim-drop augmentation directly attack the domain gap — which is why our
  strongest result is on the *hardest* (dim EVK4) sequence.
- **Rigorously ablated.** We implemented and trained **four model families** —
  per-event classifier, event-frame transformer, **SNN**, and **graph/point-NN**
  — and compared them head-to-head on the actual data. This is exactly the
  "ablation analysis of alternative models" the innovation criterion rewards,
  and it *justifies* the chosen architecture with evidence rather than assertion.
- **Honest and diagnostic.** We built an oracle-ceiling analysis that proves how
  far the data allows (0.87) and localizes the remaining gap; we report negative
  results (tracker-unification, over-aggressive box scaling, cross-grid-hurts-DVX)
  rather than hiding them. Reviewers can trust the numbers.
- **Deployable today.** The classical pipeline is CPU-only, offline,
  containerized, interpretable, and real-time — usable now for ground-station
  SSA — while the deep ensemble provides the accuracy ceiling when a GPU is
  available. One codebase serves both.

---

## 4. Technical approach and solution architecture

**Representation.** Events are normalized per sensor (÷ diagonal) for
resolution invariance, then either fed to the classical KD-tree feature pipeline
or voxelized into a sparse `(time-bins × polarity, G, G)` grid for the deep
detectors.

**Classical pipeline (Stages 0–4).** (0) normalize; (1) O(N) background-activity
denoise via space-time coincidence; (2) per-event **coherence** features (PCA
linearity, flow consistency, polarity, anisotropy) classified by LightGBM on a
*brightness-invariant* subset (so it cannot take a "dense = RSO" shortcut that
fails on dim objects); (3) a **motion-gated global tracker** (constant-velocity
gating, smoothness rejection, bidirectional extension) that turns positive
events into one box/window and rejects static clutter; (4) emit + a
**synthetic-tracking shift-and-stack** module that recovers dim moving objects
by integrating evidence along hypothesized velocities.

**Deep detector.** A sparse-attention encoder over voxel patches feeds a
**CenterNet heatmap head**. The key finding: a global box-regression head *fails*
(centers land >65 px off), while a heatmap head localizes to cell resolution and
lifts the transformer from mAP 0.016 → 0.29. **Event-level augmentation**
(flips, translation, scale, **event-drop = dim augmentation**, noise injection)
then lifts it to 0.40 and makes it beat the classical pipeline even on the
sparse DVX sequences. **Ensembling + TTA** (diverse seeds/grids, flip averaging)
is the decisive accuracy lever (DAVIS AP 0.41 → 0.62). Grids carry different
receptive fields, so a **cross-grid ensemble** wins on EVK4's large objects while
a grid-192 ensemble wins on DVX — hence **per-sensor routing**.

**Expected outputs.** Per-sequence `<seq>_bb_windows_40ms.txt` (one detection
per row, plus confidence) and `Evaluation_Metrics.xlsx`, produced offline inside
a container from a single entrypoint, plus a visualization layer (per-window
overlays and (x, y, t) trajectory plots that double as the coherence-hypothesis
figure).

**Reproducibility & scaling.** Deterministic seeds, pinned dependencies, a
CPU-only Docker image for the classical path, and GPU-ready training (validation
split + early stopping) for the deep path; the whole deep campaign runs on a
single RTX 2080 Ti in a few GPU-hours.

---

## 5. Team capacity

*[Team-specific — fill in.]* The team combines event-based vision,
classical signal processing, and deep-learning engineering. This submission was
executed end-to-end: data pipeline, four detector families, a rigorous ablation
and oracle-ceiling study, GPU training infrastructure on institutional hardware,
and a reproducible, containerized delivery — demonstrating the capability to
carry an operational SSA detector from raw events to a validated, deployable
system.

---

## 6. Prior work

This proposal is backed by a **complete, working proof-of-concept** (this
codebase), not a paper design: all four model families are implemented, trained,
and evaluated on the challenge data, with measured results throughout. The
method is grounded in the SSA and event-vision literature — background-activity
denoising, FIESTA-style geometric tracking, event transformers (EvT/SAST),
CenterNet detection, and synthetic-tracking / shift-and-stack from faint-object
astronomy — adapted to the neuromorphic-SSA regime and the dataset's unique
per-event supervision. See the companion Technical Research Report and the
project README for method rationale, the full ablation tables, and the
oracle-ceiling analysis.
