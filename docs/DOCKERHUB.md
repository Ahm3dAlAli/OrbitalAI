# OrbitSight — Real-Time RSO Detection from Neuromorphic Vision

Event-native AI pipeline that detects and tracks **Resident Space Objects** (satellites,
rocket bodies, debris, apparently-moving stars) directly from raw neuromorphic
(event-camera) telescope recordings — **CPU-only, real-time (< 40 ms/window)**, across
three sensor resolutions with one parameter set. Built for the **TII OrbitSight Challenge**.

- **Real-time mAP@0.5 = 0.721** (single model per sensor, all sensors < 40 ms/window on CPU)
- **10× over the classical baseline**; decisive gains on the dim-object sequences
- Source & full write-up: **https://github.com/Ahm3dAlAli/OrbitalAI**

---

## Quick start

```sh
docker run --rm \
  -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
  -v /path/to/output:/work \
  ahm3dalali/orbitsight:latest
```

- Mount the dataset read-only at **`/OrbitSight_dataset`** (a folder of
  `*_labeled_events.npy`, optionally split into `Testing_sets/` and `Training_sets/`).
- Mount a writable **`/work`**; results are written to **`/work/OrbitalAI/DDMMYYYY`**.

## Entrypoint (automatic, non-interactive)

```dockerfile
CMD ["sh", "run_infer.sh"]
```

No shell, no arguments needed — the container **runs the full pipeline to completion and
exits**. For every sequence it: voxelizes events → runs the per-sensor detector → decodes
one box per 40 ms window → applies the coasting tracker → writes predictions and the
scoring sheet, all into the output folder.

## Outputs (written to `/work/OrbitalAI/DDMMYYYY`)

- **`<sequencename>.txt`** — one detection per row, header + 8 fields:
  `sequence_id, window_start_timestamp_us, window_end_timestamp_us, x_centre, y_centre, w, h, class_id, confidence`
- **`Evaluation_Metrics.xlsx`** — precision / recall / F1 / mAP@0.5 (when ground truth is present)

## The deployed pipeline (per-sensor router)

| Sensor | Detector | AP@0.5 |
|--------|----------|-------:|
| EVK4 (1280×720) | grid-192 CenterNet + **min-radius** heatmap floor | **0.891** |
| DAVIS (346×260) | grid-256 CenterNet + hard-neg + **DIoU** size loss | 0.783 |
| DVX Stars3 (640×480) | grid-256 CenterNet + hard-neg + **DIoU** | 0.677 |
| DVX Thuraya3 (640×480) | grid-192 CenterNet + **coasting Kalman** | 0.534 |
| **Overall** | per-sensor router, single forward pass/window | **0.721** |

Event-native sparse-attention CenterNet with a multi-window (±3, ~280 ms) temporal
context, routed per sensor to its winning checkpoint. CPU-only by design (no CUDA /
nvidia-docker); measured **15–38 ms/window** end-to-end.

## Configuration (environment variables)

| Variable | Default | Purpose |
|----------|---------|---------|
| `ORBITSIGHT_DATASET` | `/OrbitSight_dataset` | input dataset mount |
| `ORBITSIGHT_TEAM` | `OrbitalAI` | output folder `/work/<team>/DDMMYYYY` |
| `ORBITSIGHT_CLASS_ID` | `0` | class id written to every detection row |
| `ORBITSIGHT_DEVICE` | `cpu` | `cpu` (portable) or `cuda` |
| `ORBITSIGHT_TTA` | `--tta` | set to `""` to disable test-time augmentation (faster) |

Example — score only a test split and label detections as class 1:
```sh
docker run --rm \
  -e ORBITSIGHT_DATASET=/OrbitSight_dataset/Testing_sets \
  -e ORBITSIGHT_CLASS_ID=1 \
  -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
  -v /path/to/output:/work \
  ahm3dalali/orbitsight:latest
```

## Tags

- `ahm3dalali/orbitsight:0.721` — pinned to the 0.721 real-time pipeline
- `ahm3dalali/orbitsight:latest`

---

*TII OrbitSight Challenge · Propulsion & Space Research Center. CPU-only, offline,
single parameter set across DAVIS / DVX / EVK4.*
