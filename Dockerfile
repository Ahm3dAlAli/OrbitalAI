# OrbitSight — reproducible inference container for the deployed real-time pipeline.
#
# Deployed per-sensor pipeline (mAP 0.721 real-time, all sensors < 40 ms CPU):
#   EVK4          -> g192_ctx_r3   (min-radius heatmap floor; AP 0.891)
#   DAVIS, Stars3 -> g256_hn_iou   (grid-256 hard-neg + DIoU size loss; 0.783 / 0.677)
#   Thuraya3      -> g192_ctx_v2 + coasting Kalman recall recovery (0.534)
# The router sends each sensor to its winning checkpoint; each has a fallback if its
# specific checkpoint is absent (g192_ctx_r3->g192_ctx_v2->g192_ctx; g256_hn_iou->
# g256_hn). CPU-only by design: measured 15-38 ms/window end-to-end, so the image runs
# anywhere with no CUDA/nvidia-docker dependency.  (Offline max-accuracy variant uses
# cross-grid ensembling + TTA at ~211 ms — not the real-time path.)
#
# Build:
#   docker build -t orbitsight .
#
# Run (challenge mounts):
#   docker run --rm \
#     -v /path/to/OrbitSight_dataset:/OrbitSight_dataset:ro \
#     -v /path/to/output:/work \
#     orbitsight
#
# The Stars3/DAVIS grid-256 model and Thuraya3 coasting are on by default; override
# via ORBITSIGHT_G256_MODEL / ORBITSIGHT_COAST if needed. Classical LightGBM path:
#   docker run --rm ... orbitsight sh run.sh
FROM python:3.11-slim

# System libs: libgomp1 for lightgbm/torch OpenMP.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Pinned base deps (numpy/scipy/lightgbm/pillow/openpyxl/...) — layer-cached.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2) CPU build of PyTorch (training env requires >=2.1; no CUDA).
#    Pin to a specific version once the exact training wheel is confirmed;
#    the environment.yml floor is >=2.1 so any current stable build works.
RUN pip install --no-cache-dir "torch>=2.1" \
        --index-url https://download.pytorch.org/whl/cpu

# 3) Application: package, scripts, trained checkpoints, entrypoints.
COPY orbitsight/ ./orbitsight/
COPY scripts/    ./scripts/
COPY models/     ./models/
COPY run.sh run_infer.sh ./
RUN chmod +x run.sh run_infer.sh

# Guard: the deployed checkpoints must be baked in. A grid-192 temporal checkpoint is
# required (DVX/DAVIS + Thuraya3 base) — g192_ctx_v2 preferred, g192_ctx accepted as
# fallback. The others give the full 0.721 result; the pipeline degrades gracefully.
RUN test -f models/g192_ctx_v2.pt -o -f models/g192_ctx.pt || \
    { echo "ERROR: no temporal checkpoint (g192_ctx_v2.pt or g192_ctx.pt) in models/."; exit 1; }
RUN test -f models/g192_ctx_r3.pt || \
    echo "WARNING: models/g192_ctx_r3.pt absent — EVK4 falls back to g192_ctx_v2 (AP ~0.88, not 0.891)."
RUN test -f models/g256_hn_iou.pt || \
    echo "WARNING: models/g256_hn_iou.pt absent — DAVIS/Stars3 fall back to g256_hn/temporal (0.75/0.65, not 0.783/0.677 -> lower mAP)."
RUN test -f models/g192_ctx_v2.pt || \
    echo "WARNING: models/g192_ctx_v2.pt absent — using g192_ctx (lower Thuraya3/base accuracy)."

ENV KMP_DUPLICATE_LIB_OK=TRUE \
    PYTHONUNBUFFERED=1 \
    ORBITSIGHT_DATASET=/OrbitSight_dataset \
    ORBITSIGHT_DEVICE=cpu \
    ORBITSIGHT_TEAM=OrbitalAI \
    ORBITSIGHT_CLASS_ID=0

# Automatic, non-interactive entrypoint: the deployed pipeline runs to completion
# with no shell interaction. Writes into the portal-collected folder
# /work/OrbitalAI/DDMMYYYY:
#   - <sequencename>.txt        one detection/row: sequence_id, timestamp_us, x, y,
#                               w, h, class_id, confidence
#   - Evaluation_Metrics.xlsx   scoring sheet (when GT is present)
CMD ["sh", "run_infer.sh"]

# ---------------------------------------------------------------------------
# GPU variant (faster eval hardware): swap the base image and the torch install:
#   FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04
#   ... install python3.11 ...
#   RUN pip install torch --index-url https://download.pytorch.org/whl/cu121
#   ENV ORBITSIGHT_DEVICE=cuda
#   run with:  docker run --gpus all ...
# ---------------------------------------------------------------------------
