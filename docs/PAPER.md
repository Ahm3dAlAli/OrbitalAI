# OrbitSight: Real-Time Resident Space Object Detection and Tracking from Neuromorphic Event Cameras under a 40 ms CPU Budget

*Updated manuscript — reflects the final modeling (grid-256 multi-object detector,
DVX-lever ablation) and results (0.692 real-time / 0.709 offline). Edits vs. the
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
detector with online hard-negative mining** for dense multi-object star fields and a
**coasting Kalman tracker** for faint intermittently-visible objects. We report a
rigorous ablation of alternative model families (event-frame transformer, spiking
network, point/graph network) under an identical frozen evaluator, and diagnose a
detection-head failure that had masked their true capacity. We further ablate the
DVX levers: grid-256 + hard-negative mining raise the star-field precision, and a
local coasting filter recovers the faint object's recall (0.63 → 0.72) where a
global-fit tracker cannot. **The deployed real-time system — one model per sensor,
one forward pass per window — reaches mAP@0.5 of 0.692 at 15–38 ms/window on CPU,
inside the 40 ms budget on every sensor; an offline max-accuracy variant (cross-grid
ensembling + test-time augmentation) reaches 0.709.** This is a 10× improvement over
a classical baseline, with the decisive gains on the dim-object sequences.

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
  deployed single-model-per-sensor pipeline is 15–38 ms/window on CPU (mAP 0.692);
  the offline cross-grid+TTA variant is 211 ms on the EVK4 path (mAP 0.709). We
  report the accuracy of the config we actually ship at real time.
- **Hard-negative mining and coasting for the DVX floor.** Online hard-negative
  mining in the CenterNet focal loss suppresses background-star false positives
  (Stars3 0.613 → 0.651, DAVIS 0.729 → 0.753); a **coasting Kalman tracker** recovers
  the faint Thuraya3 object's recall (0.63 → 0.72, AP 0.469 → 0.506) by bridging
  short sub-threshold gaps — succeeding precisely where a *global*-fit tracker fails.
- **A rigorous lever ablation.** A cross-family study diagnoses and fixes a
  global-head failure; a DVX-lever study shows which levers help (grid-256,
  hard-neg, local coasting) and which do not (dim-augmentation, DVX reweighting,
  longer context, center-only Kalman smoothing, global trajectory filling), and why.

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
| DAVIS SAOCOM1B | g256 hard-neg | 0.829 | 0.801 | 0.815 | 0.753 |
| DVX Stars3 | **g256 hard-neg** | 0.492 | 0.819 | 0.615 | **0.651** |
| DVX Thuraya3 | g192_ctx **+ coast** | 0.414 | 0.723 | 0.526 | **0.506** |
| **Overall** | — | — | — | — | **0.692** |

**Table 2b — Offline max-accuracy system (cross-grid ensemble + TTA; Stars3 → grid-256 hard-neg; Thuraya3 coasted).**

| Sequence (sensor) | AP |
|---|---:|
| EVK4 mag7.3 | 0.896 |
| DAVIS SAOCOM1B | 0.774 |
| DVX Stars3 (grid-256 hard-neg) | 0.651 |
| DVX Thuraya3 (coasted) | 0.515 |
| **Overall** | **0.709** |

The dim EVK4 magnitude-7.3 sequence, singled out as the hardest case, is the
strongest at AP 0.859–0.896, exactly where H2 predicted the contest is won. The
Stars3 field is lifted decisively by grid-256; the faint Thuraya3 target is the
remaining floor (§5.4).

### 5.2 Development trajectory

**Figure 2.** mAP@0.5 across the project: classical tracker baseline 0.069 → tuned
classical 0.249 → CenterNet 0.289 → hybrid router 0.315 → event augmentation 0.398
→ grid-192 + box calibration 0.454 → three-model ensemble + stacking 0.554 →
multi-window temporal context 0.660 → real-time single-model per sensor 0.651 →
grid-256 Stars3 routing 0.668 → **hard-negative mining + coasting Kalman (real-time)
0.692** → **offline cross-grid + TTA 0.709**. The two largest single jumps are
event-level augmentation and multi-window temporal context; the final DVX gains come
from grid-256 resolution, hard-negative mining, and recall-recovery coasting. A 10×
gain over the classical baseline.

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

**Attention vs. a state-space backbone.** With the head fixed, we compare the
encoder token mixer itself: the masked-attention encoder against a **diagonal
state-space (S4D-style) mixer** — a real one-state SSM per channel,
`h_t = a_c h_{t-1} + b_c x_t`, applied bidirectionally by FFT causal convolution
(linear-time). Both are trained at the deployed recipe (grid-192, ±3-window
context, identical schedule).

**Table 4b — Attention vs. state-space encoder (grid-192, ctx=3, single model).**

| Encoder | mAP | EVK4 | DAVIS | Stars3 | Thuraya3 | s/epoch |
|---|---:|---:|---:|---:|---:|---:|
| Masked attention | **0.651** | 0.859 | 0.729 | 0.545 | 0.469 | ~160 |
| Diagonal SSM (S4D) | 0.562 | 0.768 | 0.619 | 0.441 | 0.421 | **~94** |

The SSM **matches the transformer's validation loss (0.78) and trains ≈ 1.8×
faster** (linear-time recurrence vs. quadratic attention), yet reaches **lower
detection AP (0.562 vs. 0.651)** — a clean illustration that validation loss is not
detection AP. The gap is uniform across sensors and interpretable: attention's
**content-based** mixing routes each sparse object cluster to the relevant tokens,
whereas the SSM's **position-ordered** scan over raster-flattened tokens has no such
content routing, so it localizes sparse events less precisely. Attention remains the
accuracy choice for sparse-event detection; the state-space encoder is a viable,
markedly cheaper alternative when compute — not accuracy — is the binding constraint.

![Attention vs state-space backbone](figures/ssm_vs_attn.png)

### 5.4 DVX ablation: where the levers help — and where they don't (new)

The two dim DVX sequences dominate the difficulty; we ablate them separately.

**Stars3 (dense field): resolution + hard-negative mining.** Grid-256 first
separates adjacent stars (0.545 → 0.613, recall 0.72 → 0.81). We then add **online
hard-negative mining** to the CenterNet focal loss: at each step we upweight the
highest-confidence *strict*-background cells (target heatmap < 0.01), which are
exactly the background-star false positives. Because we use strict background, the
Gaussian skirt around real objects is excluded, so near-misses / real signal are
never penalized. This lifts Stars3 to **0.651** (precision 0.44 → 0.49, recall held
at 0.82 — the model suppresses false stars without losing true ones) and, as a
bonus, DAVIS to **0.753**. Multi-peak (top-k) decoding on grid-192 adds only +0.004.

*Heatmap resolution has a hard latency wall.* Since grid-256 helped by resolving
adjacent peaks, we tested a full-resolution heatmap (`hm_div=1`, 256×256 vs. the
deployed 128×128). It **improves accuracy — DAVIS 0.753 → 0.815, Stars3 → 0.655** —
but the extra upsampling stage over a 256×256 feature map costs **~2900 ms/window on
CPU** (≈ 70× the budget; the forward pass, not decode), versus ~24 ms at `hm_div=2`.
Heatmap resolution is thus a genuine accuracy lever with a **hard real-time wall**:
`hm_div=2` is the deployable optimum, and the `hm_div=1` gain is GPU-offline only.

![DVX-lever ablation](figures/dvx_levers.png)

**Table 5 — Thuraya3 (faint single object): which levers move recall.**

| Lever | Thuraya3 AP |
|---|---:|
| Baseline (temporal, augmented) | 0.469 |
| + aggressive dim-augmentation | 0.454 |
| + DVX-oversampling reweighting | 0.462 |
| + grid-256 | 0.469 |
| + longer context (±5 windows) | 0.435 |
| + shift-and-stack | 0.457 |
| + center-only Kalman smoothing | 0.462 |
| + global trajectory filling | (skipped — 47 px global-fit residual) |
| **+ coasting Kalman (local)** | **0.506** |

Thuraya3's binding constraint is **recall** (0.63), not precision: the object is
present but drops below threshold for a few windows and the track is lost.
Training-recipe changes (dim-augmentation, DVX reweighting) and larger context do
not help — the object is at the sensitivity floor with no close analog among the 16
training sequences. Crucially, the *global* trajectory approaches fail for a subtle
reason: the ground-truth track does not fit one low-order polynomial (47 px median
residual, intermittent visibility with 1400-window gaps), and a center-only Kalman
*smoother* even slightly hurts (−0.012) because the learned localization already
tracks to ∼1 px. **But the motion is smooth window-to-window**, so a *local* coasting
Kalman — which predicts the object forward through short sub-threshold gaps and emits
a decayed-confidence box, bounded so it never extrapolates into the long absences —
recovers recall 0.63 → 0.72 and lifts AP to **0.506** (+0.037). The distinction is
the finding: *global* fits fail on this non-polynomial track, but *local* coasting
succeeds. This bounds the faint-object problem to a recall-recovery task and shows
which class of tracker solves it.

**The residual Thuraya3 precision (0.41) is a proven frontier, not an untried
lever.** We diagnosed the 285 false positives by *coast-age* — windows since the
track last had above-threshold (non-coasted) evidence. They do **not** pile up in
over-coasted tails: only 8 % sit at age ≥ 6, and the coasted FPs decay smoothly with
age (69, 31, 14, 10, 0 across ages 1–2, 3–5, 6–10, 11–20, 21+). **56 % (161) are
age-0 *real detections*** — the detector firing above threshold, not a coasting
artifact. Two levers shaped like this problem come up empty. (i) A *coast cap*
(terminate a track after N evidence-free windows) has **no knee**: sweeping it is the
max-coast sweep, where AP rises monotonically to the cap-50 optimum (0.480 → 0.506),
so tightening only sheds recall. (ii) *Confidence discounting is already applied* —
coasted boxes emit at median confidence 0.40 vs 0.48 for evidence-backed ones, so
they already sort below true detections on the PR curve. Decomposing the 161 age-0
FPs by center distance, **83 % (133) are phantoms in windows with no ground-truth
object at all** — the detector confidently fires where the GT is empty — and only
17 % are resizable near-misses (a box-size sweep confirms +0.006 at 12 px, then
IoU-loss flips tight true positives to FPs beyond 13 px). The precision floor is
therefore fully explained: it is *confident detection in windows the ground truth
considers empty*, which no gating, capping, resizing, or reordering can touch — only
a stronger detector or more faint-object training data would. This is the frontier of
the current signal, not an unexplored lever.

![Thuraya3 FP coast-age histogram](figures/coast_age_hist.png)

**A feature-space probe (GHOST-adapted) confirms the floor is intrinsic.** As a final
check we adapt the Gaussian-hypothesis z-score of GHOST [Rabinowitz et al. 2025] — a
hyperparameter-free open-set-*classification* method — into a *diagnostic* for
detection false positives: we sample the CenterNet encoder embedding at each decoded
peak, fit a diagonal Gaussian over true-positive embeddings, and score every detection
by its L1 z-score s = Σ_d |φ_d − μ_d| / σ_d. If phantoms deviated in feature space, a
GHOST-style post-hoc reject would remove them. They do not: on both dim sequences the
z-score fails to separate false positives from true positives (FP-vs-TP AUROC **0.455**
on Thuraya3, **0.486** on Stars3 — at chance, with phantoms if anything slightly
*closer* to the true-positive mean). The detector fires confidently on empty-window
patterns whose learned embeddings are statistically identical to real RSOs. This is
the deepest layer of the frontier: the precision floor is the detector's *decision
boundary itself*, not a post-hoc-correctable artifact — no gating, capping, resizing,
reordering, *or feature-space rejection* touches it. We report this as a negative
result and, notably, as a reusable *frontier diagnostic*: adapting the GHOST z-score
from classification to single-class event detection gives a cheap, hyperparameter-free
test of whether a detector's false positives are separable in feature space at all.

### 5.5 Where temporal context helps

Multi-window context roughly doubles AP on Thuraya3 (0.233 → 0.469) and lifts DAVIS
SAOCOM1B (0.617 → 0.729), while EVK4 (large, bright) is unaffected. This matches the
oracle analysis, which places the achievable ceiling near 0.87 and attributes the
remaining gap to detection and recall rather than box size.

**The context length is unimodal, peaking at ±3 windows.** We sweep the temporal
context (single grid-192 model, all else fixed):

**Table 6 — Temporal-context sweep (single model, all four sequences).**

| Context | overall mAP | Thuraya3 AP | s/epoch |
|---|---:|---:|---:|
| ±2 (5 windows) | 0.641 | 0.422 | ~78 |
| **±3 (7 windows)** | **0.651** | **0.469** | ~130 |
| ±5 (11 windows) | (DVX ↓) | 0.435 | ~170 |

±2 trains ≈ 1.7× faster but loses 0.010 mAP; ±5 *degrades* the faint object
(Thuraya3 0.469 → 0.435) — too much history dilutes the sparse signal. **±3 is the
sweet spot**, confirming the context is a genuine optimum rather than a monotonic
"more is better." A per-sensor nuance appears in the sweep — DAVIS/Stars3 are
marginally better at ±2 while EVK4/Thuraya3 prefer ±3 — but the per-sensor router
already selects each sensor's best checkpoint, so ±3 remains the shared default.

### 5.6 Qualitative results

**Figure 5** overlays predicted (yellow, with confidence) and ground-truth (green)
boxes across sensors, from the bright EVK4 object to the faint DVX/Thuraya3 target,
with IoU in the 0.83–1.00 range. Our model-agnostic visualization tool additionally
renders per-window detection animations, a failure gallery (missed / false-positive /
low-IoU windows), an (x, y, t) coherence view, and an **analyst diagnostics panel**
(event-rate spatial heatmap, per-window event rate as an SNR proxy, and
confidence/matched-IoU distributions over time) for operational interpretation.

### 5.7 Latency: real-time vs. offline, measured

We measure per-window streaming latency (batch 1, CPU) for both configurations.

**Deployed real-time (one model per sensor):** voxelize → forward → decode totals
**15–38 ms/window** — EVK4 38 ms, DAVIS 26 ms, DVX (grid-256) 24 ms — **inside the
40 ms budget on every sensor.**

**Offline max-accuracy:** the EVK4 cross-grid ensemble (5 models) with TTA (3 passes)
totals **211 ms/window** — 5.3× the budget — and even single-model TTA sits at
∼39–40 ms. We therefore report **0.709 as an offline figure and 0.692 as the
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
family comparison. **The DVX Thuraya3 faint object is partly recovered and partly a
signal limit.** Training-recipe levers (augmentation, reweighting, longer context)
and *global*-fit trackers do not move it (§5.4), but a *local* coasting Kalman
recovers recall 0.63 → 0.72 (AP 0.469 → 0.506); the residual gap to the oracle
ceiling reflects windows where the object is genuinely below the sensitivity floor,
which would need additional faint-object training data or sensor fusion with the
DAVIS APS grayscale channel [Capogrosso et al. 2026]. The offline max-accuracy
configuration (0.709) exceeds the latency budget and is reported separately from the
deployed real-time system (0.692).

**A tested distinction and one deployment direction.** (i) *Center-refinement vs.
recall-recovery tracking.* We separately tested a constant-velocity Kalman + RTS
smoother that *refines existing* detection centers; it **slightly degrades AP**
(−0.012 overall) because the CenterNet's learned sub-cell offset already localizes to
~1 px (residual 0.8–1.2 px), tighter than a constant-velocity prior. The lesson is
that the localization head, not a motion model, is the accurate component — so we do
*not* smooth centers, but we *do* use the same motion model for **coasting** (adding
boxes through sub-threshold gaps), which recovers recall (§5.4). Refinement and
recall-recovery are different uses of the filter with opposite outcomes here.
(ii) *Deployment quantization.* The deployed pipeline is already CPU real-time in
float32;
**INT8/FP16 quantization via ONNX Runtime / OpenVINO** (with quantization-aware
training) would roughly halve the forward-pass cost, leaving headroom to promote more
sensors to the grid-256 head or add temporal context on the cheaper sensors while
staying inside the 40 ms budget. Both are integration-level extensions of the shipped
system, not new research.

Finally, the one-box-per-window assumption is exploited by the tracker and is relaxed
only for the Stars3 field via the grid-256 multi-object head; general multi-object
scenes would need revisiting.

---

## 7. Conclusion

OrbitSight shows that faint, fast RSOs can be detected and tracked from raw event
streams in **real time on CPU** (mAP 0.692, 15–38 ms/window), under a single
cross-sensor parameter set, by treating coherence rather than brightness as the
signal and by integrating an object's track over multiple windows. A coherence-first
classical backbone, a temporal-context CenterNet, a grid-256 head with hard-negative
mining for dense star fields, a coasting Kalman tracker for the faint object, and a
per-sensor router together reach mAP@0.5 of **0.692 real-time / 0.709 offline** (the
offline system crosses 0.70), a 10× gain over a classical baseline, with the largest
improvements exactly on the dim objects that dominate the difficulty. Honest
ablations — a cross-family study that fixes a detection-head artifact, and a DVX-lever
study — show that once the head is fixed deep event models are competitive, that
resolution (not capacity) unlocks the dense field, that hard-negative mining removes
background-star false positives, and that a *local* coasting filter recovers faint-
object recall where a *global* fit cannot. A hybrid, per-sensor-routed system is the
right way to combine these complementary strengths.

---

## Change-log vs. prior draft

- **Headline:** 0.660/0.675 → **0.692 real-time / 0.709 offline** (offline crosses
  0.70). DVX gains: Stars3 0.545 → 0.651, DAVIS 0.729 → 0.753, Thuraya3 0.469 → 0.506.
- **New modeling:** resolution-routed grid-256 detector (§3), **online hard-negative
  mining** in the CenterNet focal loss (§5.4), and a **coasting Kalman tracker** for
  faint-object recall recovery (§5.4).
- **New ablation (Table 5):** DVX-lever study — which levers help (grid-256, hard-neg,
  local coasting) and which do not (dim-aug, reweighting, ctx±5, center-only smoothing,
  global fill), with the *local-coasting-succeeds-where-global-fit-fails* finding.
- **Temporal-context sweep (Table 6):** ctx ±2/±3/±5 — unimodal, peaking at ±3
  (0.641/0.651/degrades); ±2 is ~1.7× faster but −0.010 mAP, ±5 dilutes the faint
  object. Confirms ±3 as a genuine optimum, not "more is better."
- **State-space backbone ablation (Table 4b):** a diagonal S4D-style SSM encoder vs.
  masked attention at the deployed recipe — SSM matches val loss and trains ~1.8×
  faster (linear-time) but reaches lower AP (0.562 vs. 0.651); content-based attention
  localizes sparse events better than a position-ordered state-space scan.
- **Frontier proof (§5.4):** four-layer analysis of the Thuraya3 precision floor —
  box-size decomposition, coast-age histogram, coast-cap sweep + confidence-discount,
  and a **GHOST-adapted feature-space probe** (z-score FP-vs-TP AUROC 0.455 / 0.486,
  at chance) — proving the floor is the detector's decision boundary, not a
  post-hoc-correctable artifact. The probe is a reusable, hyperparameter-free
  diagnostic (GHOST re-purposed from OSR classification to detection FP separability).
- **Latency (§5.7):** explicit, *measured* real-time-vs-offline split — deployed
  15–38 ms/window (0.692) vs. offline 211 ms (0.709); the "≈ 4 ms" claim in the prior
  draft was a single grid-128 forward, not the deployed grid-192/256 config.
- **Related work (§5.8):** added the space-domain event-vision survey and the
  frame-based edge-SOD baseline, with the "software gap → event-native" framing.
- **Contributions/abstract:** rewritten to state the real-time/offline separation and
  the two ablations.
