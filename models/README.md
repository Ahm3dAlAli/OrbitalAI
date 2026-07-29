# Deployed checkpoints (real-time mAP 0.721)

The three checkpoints the per-sensor router uses. Each is a sparse-attention CenterNet
(event-native); together they reproduce the deployed real-time result on the 4 test
sequences.

| Checkpoint | Sensor(s) | Recipe | AP@0.5 |
|---|---|---|---:|
| `g192_ctx_r3.pt` | EVK4 | grid-192, ±3 temporal context, **min-radius** heatmap floor | 0.891 |
| `g256_hn_iou.pt` | DAVIS, DVX Stars3 | grid-256, hard-negative mining, **DIoU** size loss | 0.783 / 0.677 |
| `g192_ctx_v2.pt` | DVX Thuraya3 (base) | grid-192, ±3 temporal context (+ coasting Kalman at inference) | 0.534 |
| **Overall** | per-sensor router | one forward pass/window, CPU < 40 ms | **0.721** |

The router wiring lives in `run_infer.sh`; the Docker image bakes these in via
`COPY models/`. Other `*.pt` files from the ablation studies are intentionally not
committed (kept the repo lean) — they are reproducible from `scripts/train_centernet.py`
with the flags documented in `docs/PAPER.md`.
