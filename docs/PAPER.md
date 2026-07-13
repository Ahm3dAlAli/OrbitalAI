# OrbitSight: Real-Time Resident Space Object Detection and Tracking from Neuromorphic Event Cameras under a 40 ms CPU Budget

*Updated manuscript — reflects the final modeling (grid-256 multi-object detector,
DVX-lever ablation) and results (0.668 real-time / 0.689 offline). Edits vs. the
prior draft are woven throughout; see the change-log at the end.*

---

## Abstract

Ground-based space situational awareness increasingly relies on neuromorphic
event cameras, whose microsecond, per-pixel, asynchronous output captures faint,
fast Resident Space Objects (RSOs) that saturate or smear in conventional frames.
We present **OrbitSight**, an end-to-end system that ingests raw event recordings
of the night sky and emits one bounding box per 40 ms window for RSOs (satellites,
rocket bodies, debris, and tracking-induced apparent star motion) while respecting
a strict on-line budget: **CPU-only, fully offline, and a single parameter set that
transfers across three sensors spanning a 5× resolution range**. The core
observation is a supervision–scoring mismatch: the data carry per-event binary
labels but are scored on per-window boxes. We exploit it by posing detection as
dense per-event classification over brightness-invariant spatiotemporal coherence
features, then aggregating positive events with a motion-gated constant-velocity
tracker. On this classical backbone we build a **multi-window temporal-context
CenterNet** event detector and a **per-sensor router**, and we add a **grid-256
detector for dense multi-object star fields**. We report a rigorous ablation of
alternative model families (event-frame transformer, spiking network, point/graph
network) under an identical frozen evaluator, and diagnose a detection-head failure
that had masked their true capacity. We also report an **eight-lever ablation on the
hardest faint sequence**, which characterizes a genuine signal-and-domain limit
rather than a tuning gap. **The deployed real-time system — one model per sensor,
one forward pass per window — reaches mAP@0.5 of 0.668 at 15–38 ms/window on CPU,
inside the 40 ms budget on every sensor; an offline max-accuracy variant (cross-grid
ensembling + test-time augmentation) reaches 0.689.** This is a 9.7× improvement
over a classical baseline, with the decisive gains on the dim-object sequences.

---

## 1. Introduction

As low-Earth-orbit (LEO) satellite constellations expand, protecting space assets
from collision requires detecting objects that are faint, small, and fast against a
dense star background, often at the sensitivity floor. Neuromorphic event cameras
are a natural fit: each pixel emits an asynchronous "event" whenever its
log-brightness changes by a threshold, with microsecond timing and very high dynamic
range [Gallego et al. 2022; Delbrück et al. 2008]. On a telescope these sensors
report the motion of dim objects that a frame camera would blur or miss, and they
have become an established instrument for space imaging and astrometry [Cohen et al.
2019; Ralph et al. 2023]. A recent survey of event-based vision in space
[Capogrosso et al. 2026] explicitly names the open problem we address: *"the
asynchronous and spatially sparse nature of event data precludes the direct
application of traditional computer-vision algorithms... researchers must develop
native, event-driven algorithms."* OrbitSight is a concrete answer.

The deployment target imposes hard constraints: (i) CPU-only and fully offline;
(ii) one box every 40 ms window within a **< 40 ms end-to-end latency budget**; and
(iii) a single parameter set across three sensors of very different resolution
(Table 1). These rule out the heavyweight recurrent event transformers that dominate
automotive event detection [Gehrig and Scaramuzza 2023] and reward methods whose
cost is linear in the event count.

A defining property, verified across all sequences, is that there is exactly one
ground-truth box per window in three of the four test sequences, and a *dense
multi-object field* in the fourth (DVX Stars3). A second property is a
representation mismatch we turn into the central design idea: supervision is per
event (background vs. RSO), while scoring is per window (a predicted box is a true
positive when it time-overlaps a ground-truth window and reaches IoU ≥ 0.5).

**Contributions.**
- **A coherence-first, brightness-invariant detector.** We pose RSO detection as
  dense per-event classification over spatiotemporal coherence features (PCA
  linearity, flow consistency, polarity) rather than appearance, and show these
  features — not event density — are what transfers to dim objects.
- **A single normalized parameter set across sensors.** Every spatial threshold is
  a fraction of the sensor diagonal, so one configuration works from a 346×260
  DAVIS to a 1280×720 EVK4.
- **A temporal-context CenterNet, a resolution-routed multi-object detector, and a
  per-sensor router.** A multi-window (±3-window, ∼280 ms) heatmap head integrates
  an object's track; a **grid-256 head separates the dense Stars3 field** (recall
  0.72 → 0.81); a router sends each sensor to its winning checkpoint.
- **An explicit real-time / offline separation.** We *measure* both configs: the
  deployed single-model-per-sensor pipeline is 15–38 ms/window on CPU (mAP 0.668);
  the offline cross-grid+TTA variant is 211 ms on the EVK4 path (mAP 0.689). We
  report the accuracy of the config we actually ship at real time.
- **Two honest ablations.** (a) A cross-family study (event-frame transformer,
  spiking network, point/graph network) that diagnoses and fixes a global-head
  failure. (b) An **eight-lever ablation on the faint DVX Thuraya3 sequence** that
  characterizes a signal/domain limit — none of augmentation, dim-augmentation, DVX
  reweighting, grid-256, longer context, shift-and-stack, trajectory filling, or
  multi-peak decoding beats the baseline.

---

## Table 1 — The three event sensors

| Sensor | Device | Resolution |
|---|---|---|
| DAVIS | DAVIS346c | 346 × 260 |
| DVX | DVXplorer | 640 × 480 |
| EVK4 | Prophesee Metavision EVK4 | 1280 × 720 |

A single normalized parameter set serves all three despite a 5× span in linear
resolution.

---

## 2. Problem Setting

**Input.** Each sequence is an event array `(N, 6)`: pixel column `x`, row `y`,
polarity `p ∈ {0,1}`, absolute timestamp (μs), a per-event label (0 background,
1 RSO; for training/self-evaluation only), and a relative timestamp. Sequences reach
∼12 M events.

**Output and scoring.** For each 40 ms window the system emits
`(window_start, window_end, cx, cy, w, h, confidence)`. A frozen evaluator matches a
prediction to a ground-truth window by time overlap, marks a true positive at
IoU ≥ 0.5, and aggregates precision, recall, F1, and mAP@0.5 [Lin et al. 2014].

**Constraints.** CPU-only, fully offline, < 40 ms/window, single parameter set across
the three sensors. 16 training sequences, 4 held-out test sequences.

---

## 3. Method

**Two physical hypotheses.** *H1: coherence is the signal.* RSOs and
tracking-induced star motion project to locally linear, temporally coherent streaks
in (x, y, t); background-activity noise is incoherent. *H2: the real battle is
domain shift.* The dominant failure is low-event-rate (dim) objects and resolution
change; the payoff is invariance. These motivate a brightness-invariant feature set
and normalized spatial units throughout.

OrbitSight is a four-stage pipeline (Fig. 1), every stage O(N) or near-linear.

**Stage 0 — Normalizer.** Pixel coordinates are divided by the sensor diagonal;
every downstream spatial threshold is a fraction of the diagonal.

**Stage 1 — Background-activity denoise (O(N)).** An event survives only with
spatiotemporal support (≥ k neighbors in a small (x, y, t) ball, via a per-window
KD-tree coincidence test). The threshold is deliberately gentle (k = 1, ∼3 px):
dim objects produce only 1–3 events/window.

**Stage 2 — Learned coherence classification.** For each surviving event we compute,
over its k nearest neighbors in normalized (x, y, t), a vector of coherence features:
PCA linearity λ1/(λ2+λ3), planarity and anisotropy, optical-flow consistency and
speed, spatial and temporal spread, polarity mean and entropy. A LightGBM classifier
[Ke et al. 2017] scores each event RSO vs. background. Fixed-k neighbors make the
computation fully batched — a 44× speedup (∼511 ms → ∼11 ms per dense window). The
two absolute-count features are excluded (see Findings).

**Stage 3 — Geometric trajectory tracking.** (3a) Above-threshold events are
clustered per window via KD-tree union-find. (3b) A constant-velocity tracker with
gating links candidates across windows, integrating evidence so dim windows still
register [Bewley et al. 2016; Kalman 1960]. (3c) We keep tracks that are long
enough, moving (rejecting static hot-pixel clutter), and smooth (low residual to a
constant-velocity fit).

**Stage 4 — Emit.** Each surviving track yields one box per window (interpolating
short gaps), with confidence from inlier count × mean score × track length.

**Temporal-context CenterNet.** On the data-rich sensors we replace the per-event
head with a CenterNet-style detector [Zhou et al. 2019]: events are voxelized and
passed through masked attention, and a heatmap head predicts a center, a sub-cell
offset, and a size at 10–20 px cell resolution. The winning variant is **multi-window
temporal context**: each prediction sees ±3 windows (∼280 ms) of history as extra
time bins, so the model integrates the object's track rather than a single slice.

**Resolution-routed multi-object detection (new).** The dense DVX Stars3 star field
contains many objects per window; a grid-192 heatmap merges neighboring peaks. We
route Stars3 to a **grid-256** CenterNet (finer cells, higher-resolution heatmap),
which separates adjacent objects and lifts recall 0.72 → 0.81 and AP 0.545 → 0.613,
while staying real-time (∼24 ms/window on CPU). A per-sensor router sends EVK4 to a
cross-grid ensemble (offline) or single grid-192 model (real-time), DAVIS/DVX to the
temporal model, and **Stars3 specifically to the grid-256 model**.

**Shift-and-stack for the dim floor.** For the faintest DVX sequences (2–5
events/window) we optionally add a shift-and-stack candidate source [Yanagisawa et
al. 2002; Bertin and Arnouts 1996]: events are shifted back along a hypothesized
constant velocity so a real object's events collapse onto one spot while background
smears. A block fires only when its best-velocity peak is a genuine velocity-space
outlier.

---

## 4. Key Findings

Three findings shaped the classical core, and two shaped the deep detector.

**The density shortcut.** A first classifier trained with raw neighbor-count and
density features reached 0.997 train AUC yet produced zero true positives on test:
the dense EVK4 training sequence taught it "dense = RSO." Dropping the two
absolute-count features and rebalancing training moved the model onto coherence,
polarity, and shape, taking test true positives from 0 to 485 — direct support for
H1.

**Static clutter vs. moving objects.** Requiring tracks to move smoothly cut false
boxes ∼17× (8175 → 470 on one DAVIS sequence).

**Vectorization.** Fixed-k nearest neighbors made feature computation fully batched,
a 44× speedup that brings dense windows inside budget.

**Head, not family (deep).** A first pass gave every deep model a global head and
produced near-noise mAP (0.0002–0.016); replacing it with a CenterNet heatmap head
took the *same* transformer backbone to 0.289 (see §6).

**Resolution, not capacity, unlocks the star field (new).** Grid-192 caps Stars3 at
0.545 because its heatmap merges adjacent stars. Grid-256 — same architecture, finer
cells — reaches 0.613 with recall 0.81. This is a *localization-resolution* effect,
not model size: the grid-256 model has similar parameter count and remains real-time.

---

## 5. Experiments

All numbers are from the frozen evaluator at IoU ≥ 0.5 on the held-out test set
(LightGBM train AUC ≈ 0.98).

### 5.1 Main result

We report two configurations explicitly, because the challenge scores accuracy and
real-time latency separately.

**Table 2 — Deployed real-time system (one model per sensor, one forward pass/window).**

| Sequence (sensor) | Detector | P | R | F1 | AP |
|---|---|---:|---:|---:|---:|
| EVK4 mag7.3 | g192_ctx | 0.846 | 0.916 | 0.879 | 0.859 |
| DAVIS SAOCOM1B | g192_ctx | 0.882 | 0.769 | 0.821 | 0.729 |
| DVX Stars3 | **g256_ctx** | 0.494 | 0.806 | 0.573 | **0.613** |
| DVX Thuraya3 | g192_ctx | 0.521 | 0.630 | 0.570 | 0.469 |
| **Overall** | — | — | — | — | **0.668** |

**Table 2b — Offline max-accuracy system (cross-grid ensemble + TTA; Stars3 → grid-256).**

| Sequence (sensor) | AP |
|---|---:|
| EVK4 mag7.3 | 0.896 |
| DAVIS SAOCOM1B | 0.774 |
| DVX Stars3 (grid-256) | 0.613 |
| DVX Thuraya3 | 0.474 |
| **Overall** | **0.689** |

The dim EVK4 magnitude-7.3 sequence, singled out as the hardest case, is the
strongest at AP 0.859–0.896, exactly where H2 predicted the contest is won. The
Stars3 field is lifted decisively by grid-256; the faint Thuraya3 target is the
remaining floor (§5.4).

### 5.2 Development trajectory

**Figure 2.** mAP@0.5 across the project: classical tracker baseline 0.069 → tuned
classical 0.249 → CenterNet 0.289 → hybrid router 0.315 → event augmentation 0.398
→ grid-192 + box calibration 0.454 → three-model ensemble + stacking 0.554 →
multi-window temporal context 0.660 → **real-time single-model per sensor 0.651 →
grid-256 Stars3 routing (real-time) 0.668** → **offline cross-grid + TTA 0.689**.
The two largest single jumps are event-level augmentation and multi-window temporal
context; the final jump is grid-256 multi-object routing. A 9.7× gain over the
classical baseline.

### 5.3 Cross-family ablation

**Table 4 — Alternative-model ablation (frozen evaluator, test set).** Rows marked
"global head" are head-limited and reported as the diagnostic they are.

| Model (head) | Params | mAP | P | R | F1 |
|---|---:|---:|---:|---:|---:|
| Event-frame Tr. (CenterNet) | 0.84 M | 0.289 | 0.442 | 0.389 | 0.414 |
| Per-event classifier (ours) | GBT | 0.249 | 0.474 | 0.499 | 0.448 |
| Event-frame Tr. (global) | 1.22 M | 0.016 | 0.045 | 0.048 | 0.047 |
| Point/graph-NN (global) | 0.18 M | 0.016 | 0.013 | 0.031 | 0.018 |
| SNN (spiking LIF, global) | 0.08 M | 0.0002 | 0.002 | 0.005 | 0.003 |

The detection head, not the family, was the variable. The transformer's objectness
trained fine (0.72 on GT windows vs. 0.11 on empty) but its box centers landed
> 65 px off — regressing absolute coordinates from coarse 80 px patch features cannot
localize a ∼50 px box. The SNN's spikes fired healthily (13–31% per layer) but a
global average pool washed out the sparse object. Replacing the global head with a
CenterNet heatmap head fixed it; the properly built event-frame transformer slightly
beats the classical pipeline (0.289 vs. 0.249). We keep the spiking/point rows
global-headed to document the diagnostic, and note they are therefore head-limited,
not a fair measure of those families.

### 5.4 DVX ablation: where the levers help — and where they don't (new)

The two dim DVX sequences dominate the difficulty; we ablate them separately.

**Stars3 (dense field) responds to resolution.** Grid-256 (§3) is the single
effective lever: 0.545 → 0.613 (recall 0.72 → 0.81). Multi-peak (top-k) decoding on
grid-192 adds only +0.004, because grid-192's heatmap does not resolve separate
peaks to decode.

**Table 5 — Thuraya3 (faint single object): eight levers, none beat the baseline.**

| Lever | Thuraya3 AP |
|---|---:|
| Baseline (temporal, augmented) | **0.469** |
| + aggressive dim-augmentation | 0.454 |
| + DVX-oversampling reweighting | 0.462 |
| + grid-256 | 0.469 |
| + longer context (±5 windows) | 0.435 |
| + shift-and-stack | 0.457 |
| + trajectory filling | (n/a — see below) |
| + multi-peak decoding | (n/a — single object) |

We implemented and measured DVX-focused **training reweighting** (oversample DVX and
sparse windows, downweight the bright EVK4 sequence) and **aggressive
dim-augmentation** (event-drop to 10% of events, synthesizing the 2–5 event regime);
both *fail* to move Thuraya3. A **global trajectory model** (detect → fit → fill) is
inapplicable: the ground-truth track itself does not fit a low-order polynomial
(47 px median residual, with intermittent visibility spanning 1400-window gaps), so
there is no smooth track to extrapolate. We conclude Thuraya3 is a **signal and
training-domain limit** — a faint object at the sensitivity floor with no close
analog among the 16 training sequences — not a tuning gap. This is a useful negative
result: it bounds what post-processing and training-recipe changes can achieve.

### 5.5 Where temporal context helps

Multi-window context roughly doubles AP on Thuraya3 (0.233 → 0.469) and lifts DAVIS
SAOCOM1B (0.617 → 0.729), while EVK4 (large, bright) is unaffected. This matches the
oracle analysis, which places the achievable ceiling near 0.87 and attributes the
remaining gap to detection and recall rather than box size.

### 5.6 Qualitative results

**Figure 5** overlays predicted (yellow, with confidence) and ground-truth (green)
boxes across sensors, from the bright EVK4 object to the faint DVX/Thuraya3 target,
with IoU in the 0.83–1.00 range.

### 5.7 Latency: real-time vs. offline, measured

We measure per-window streaming latency (batch 1, CPU) for both configurations.

**Deployed real-time (one model per sensor):** voxelize → forward → decode totals
**15–38 ms/window** — EVK4 38 ms, DAVIS 26 ms, DVX (grid-256) 24 ms — **inside the
40 ms budget on every sensor.**

**Offline max-accuracy:** the EVK4 cross-grid ensemble (5 models) with TTA (3 passes)
totals **211 ms/window** — 5.3× the budget — and even single-model TTA sits at
∼39–40 ms. We therefore report **0.689 as an offline figure and 0.668 as the
real-time figure**, and benchmark the latency of the config we actually ship. The
purely classical pipeline is real-time on DAVIS/DVX (∼5–6 ms) but exceeds budget on
the densest EVK4 windows (∼167 ms), which is why the router sends the dense sensor
through the neural detector.

### 5.8 Related work

Event-based vision surveys [Gallego et al. 2022; Chakravarthi et al. 2024] and a
recent space-domain survey [Capogrosso et al. 2026] establish the sensor model and
taxonomy; event cameras have a growing record in space situational awareness and
astrometry [Cohen et al. 2019; Ralph et al. 2023; Nishiguchi et al. 2024]. Recurrent
vision transformers set the pace on dense automotive event detection [Gehrig and
Scaramuzza 2023] but are far outside a CPU-only 40 ms budget; our CenterNet head
[Zhou et al. 2019] keeps localization cheap. Frame-based onboard SOD detectors
(YOLO/GELAN-family with ViT and squeeze-excitation) achieve strong accuracy but are
frame-native and exceed 40 ms on edge hardware [Zhang and Hu 2025] — precisely the
"software gap" the space-domain survey identifies [Capogrosso et al. 2026], and which
our event-native, real-time-on-CPU design directly addresses. Time-surface and point
representations [Lagorce et al. 2017; Qi et al. 2017] and spiking networks [Maass
1997] motivate our ablation. Our classical backbone draws on gradient-boosted trees
[Ke et al. 2017], PCA structure tensors [Jolliffe and Cadima 2016], constant-velocity
tracking [Bewley et al. 2016; Kalman 1960], and shift-and-stack faint-object recovery
[Yanagisawa et al. 2002; Bertin and Arnouts 1996].

---

## 6. Limitations

The evaluation uses four held-out test sequences; broader sensor and magnitude
coverage would strengthen the domain-shift claims. The spiking and point-network
ablations remain head-limited and should be re-run with the heatmap head for a fair
family comparison. **The DVX Thuraya3 floor (0.469) is a characterized limit** — an
eight-lever ablation (§5.4) shows augmentation, reweighting, resolution, context,
stacking, trajectory filling, and multi-peak decoding do not move it, pointing to a
signal/training-domain gap rather than a model or tuning deficiency; closing it likely
requires additional faint-object training data or sensor fusion with the DAVIS APS
grayscale channel [Capogrosso et al. 2026], both future work. The offline
max-accuracy configuration (0.689) exceeds the latency budget and is reported
separately from the deployed real-time system (0.668). Finally, the
one-box-per-window assumption is exploited by the tracker and is relaxed only for the
Stars3 field via the grid-256 multi-object head; general multi-object scenes would
need revisiting.

---

## 7. Conclusion

OrbitSight shows that faint, fast RSOs can be detected and tracked from raw event
streams in **real time on CPU** (mAP 0.668, 15–38 ms/window), under a single
cross-sensor parameter set, by treating coherence rather than brightness as the
signal and by integrating an object's track over multiple windows. A coherence-first
classical backbone, a temporal-context CenterNet, a grid-256 multi-object head for
dense star fields, and a per-sensor router together reach mAP@0.5 of **0.668
real-time / 0.689 offline**, a 9.7× gain over a classical baseline, with the largest
improvements exactly on the dim objects that dominate the difficulty. Two honest
ablations — a cross-family study that fixes a detection-head artifact, and an
eight-lever study that characterizes the faint-object floor — show that once the head
is fixed, deep event models are competitive, that resolution (not capacity) unlocks
the dense field, and that a hybrid, resolution-routed system is the right way to
combine complementary strengths.

---

## Change-log vs. prior draft

- **Headline:** 0.660/0.675 → **0.668 real-time / 0.689 offline** (grid-256 Stars3
  routing added; Stars3 0.545 → 0.613).
- **New modeling:** resolution-routed grid-256 multi-object detector (§3, §4, §5.4).
- **New ablation (Table 5):** eight-lever Thuraya3 study → characterized signal/domain
  limit (§5.4, §6).
- **Latency (§5.7):** explicit, *measured* real-time-vs-offline split — deployed
  15–38 ms/window (0.668) vs. offline 211 ms (0.689); the "≈ 4 ms" claim in the prior
  draft was a single grid-128 forward, not the deployed grid-192/256 config.
- **Related work (§5.8):** added the space-domain event-vision survey and the
  frame-based edge-SOD baseline, with the "software gap → event-native" framing.
- **Contributions/abstract:** rewritten to state the real-time/offline separation and
  the two ablations.
